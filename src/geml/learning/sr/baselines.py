"""Matched-budget PySR/GP and compact transformer-SR baselines (issue 9-2).

Three baselines, one result interface. Every one of them emits the
:class:`geml.learning.sr.guided_search.SRMethodResult` schema that the two controlled arms
emit, is matched to the same :class:`SearchBudget`, and routes exact-recovery questions
through the same verifier boundary. None of them is allowed to invent a recovery claim from
numeric agreement.

Backend honesty rules enforced here:

* PySR is **version pinned**. A missing package, an unimportable package, or a version other
  than the pin produces an explicit ``dependency_unavailable`` row that records what was
  actually found. It never silently substitutes a different package or version.
* The in-repository genetic-programming fallback is a *different method*
  (``SRMethod.GP_FALLBACK``) with its own backend name. It can never be labelled PySR:
  :func:`run_pysr_baseline` refuses to emit a PySR-labelled row from GP output, and the
  method enum keeps the two apart in every downstream aggregate.
* The compact transformer baseline reuses Workstream 2's prefix backbone through an injected
  protocol. This module implements no second transformer stack.
"""

import importlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from random import Random
from typing import Protocol, runtime_checkable

import sympy
from pydantic import BaseModel, ConfigDict, Field

from geml.data.sr.benchmark import (
    EquivalenceOutcome,
    EquivalenceVerifier,
    NumericFitStatus,
    ObservationSet,
    SRTask,
    UnavailableEquivalenceVerifier,
    check_grammar,
    evaluate_numeric_fit,
    verify_exact_recovery,
)
from geml.learning.sr.guided_search import (
    SHARED_BINARY_MOVES,
    SHARED_CONSTANTS,
    SHARED_UNARY_MOVES,
    CandidateStatus,
    ResourceTelemetry,
    RunStatus,
    SearchBudget,
    SemanticCandidate,
    SRCandidate,
    SRMethod,
    SRMethodResult,
    SRRepresentation,
    TerminationReason,
    derive_candidate_id,
    generate_moves,
    materialize,
    pareto_front,
    symbols_for_task,
)

PRODUCTION_OUTPUT_ROOT = "outputs/final/goal9/baselines"

#: The single supported PySR release. Verified against the official PyPI project page on
#: 2026-07-26. Any other installed version is a dependency mismatch, not a substitution.
PYSR_PINNED_VERSION = "1.5.10"

#: PySR maps the shared inventory onto Julia operator strings. Only operators that exist in
#: the shared v1 inventory are ever offered, so PySR cannot search a richer space than the
#: controlled arms.
PYSR_BINARY_OPERATORS: tuple[str, ...] = ("+", "-", "*", "/")
PYSR_UNARY_OPERATORS: tuple[str, ...] = ("exp", "log", "sin", "cos", "tanh")

_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class BaselineError(ValueError):
    """A baseline configuration or invocation was invalid."""


class BaselineBackend(StrEnum):
    """Which concrete backend produced a row. Never inferred, always recorded."""

    PYSR = "pysr"
    GP_FALLBACK = "geml_gp_fallback"
    TRANSFORMER_SR = "transformer_sr"


class BackendStatus(StrEnum):
    """Backend availability and execution outcome."""

    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    VERSION_MISMATCH = "version_mismatch"
    IMPORT_FAILED = "import_failed"
    TIMEOUT = "timeout"


class BackendProbe(BaseModel):
    """The result of asking whether a backend may be used, and at what exact version."""

    model_config = _FROZEN

    backend: BaselineBackend
    status: BackendStatus
    requested_version: str = ""
    found_version: str = ""
    detail: str = ""

    @property
    def usable(self) -> bool:
        """Return whether the backend may be invoked."""

        return self.status is BackendStatus.AVAILABLE


class BaselineConfig(BaseModel):
    """Baseline-side configuration. The budget itself comes from issue 9-1."""

    model_config = _FROZEN

    schema_version: str = "geml-goal9-baselines-config-v1"
    output_root: str = PRODUCTION_OUTPUT_ROOT
    pysr_pinned_version: str = PYSR_PINNED_VERSION
    pysr_populations: int = Field(default=15, ge=1)
    pysr_population_size: int = Field(default=33, ge=2)
    pysr_parallelism: str = "serial"
    pysr_deterministic: bool = True
    gp_fallback_enabled: bool = True
    gp_population_size: int = Field(default=32, ge=2)
    gp_generations: int = Field(default=8, ge=1)
    gp_tournament_size: int = Field(default=3, ge=2)
    transformer_samples: int = Field(default=32, ge=1)

    def pysr_native_budget(self, budget: SearchBudget) -> dict[str, str]:
        """Return the exact PySR constructor arguments derived from the shared budget.

        PySR counts *iterations over populations*, not node expansions, so the mapping is
        approximate by construction. The mapping and the residual mismatch are both recorded
        on every emitted row rather than being quietly absorbed.
        """

        iterations = max(1, budget.max_expansions // max(1, self.pysr_populations))
        return {
            "niterations": str(iterations),
            "populations": str(self.pysr_populations),
            "population_size": str(self.pysr_population_size),
            "maxsize": str(budget.max_complexity),
            "maxdepth": str(budget.max_depth),
            "timeout_in_seconds": str(int(budget.wall_seconds)),
            "binary_operators": ",".join(PYSR_BINARY_OPERATORS),
            "unary_operators": ",".join(PYSR_UNARY_OPERATORS),
            "parallelism": self.pysr_parallelism,
            "deterministic": str(self.pysr_deterministic),
        }


PYSR_BUDGET_MISMATCH = (
    "PySR's native budget is (niterations x populations x population_size) evolutionary "
    "evaluations with a wall-clock timeout; the controlled arms are budgeted in node "
    "expansions. Wall time, maximum complexity, and maximum depth are matched exactly; the "
    "evaluation count is matched only approximately and both budgets are reported."
)


# --------------------------------------------------------------------------------------
# Backend probing
# --------------------------------------------------------------------------------------


def probe_pysr(*, pinned_version: str = PYSR_PINNED_VERSION) -> BackendProbe:
    """Report whether the pinned PySR release is importable, without importing it eagerly.

    Import is lazy so that a core installation, CI, and every test in this repository stay
    free of the PySR dependency and its Julia runtime.
    """

    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError as error:  # pragma: no cover - stdlib is always present
        return BackendProbe(
            backend=BaselineBackend.PYSR,
            status=BackendStatus.IMPORT_FAILED,
            requested_version=pinned_version,
            detail=str(error),
        )

    try:
        found = version("pysr")
    except PackageNotFoundError:
        return BackendProbe(
            backend=BaselineBackend.PYSR,
            status=BackendStatus.NOT_INSTALLED,
            requested_version=pinned_version,
            detail="pysr is not installed in this environment",
        )
    except Exception as error:
        return BackendProbe(
            backend=BaselineBackend.PYSR,
            status=BackendStatus.IMPORT_FAILED,
            requested_version=pinned_version,
            found_version="",
            detail=f"{type(error).__name__}: {error}",
        )

    if found != pinned_version:
        return BackendProbe(
            backend=BaselineBackend.PYSR,
            status=BackendStatus.VERSION_MISMATCH,
            requested_version=pinned_version,
            found_version=found,
            detail=(
                f"installed pysr {found} does not match the pinned {pinned_version}; "
                "refusing to substitute a different version"
            ),
        )

    if importlib.util.find_spec("pysr") is None:  # pragma: no cover - metadata without module
        return BackendProbe(
            backend=BaselineBackend.PYSR,
            status=BackendStatus.IMPORT_FAILED,
            requested_version=pinned_version,
            found_version=found,
            detail="pysr metadata is present but the module cannot be located",
        )
    return BackendProbe(
        backend=BaselineBackend.PYSR,
        status=BackendStatus.AVAILABLE,
        requested_version=pinned_version,
        found_version=found,
    )


# --------------------------------------------------------------------------------------
# Injected adapters
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BaselineProposal:
    """One expression proposed by a baseline backend, before any GEML-side scoring."""

    srepr: str
    backend_complexity: int | None = None
    backend_loss: float | None = None


@runtime_checkable
class PySRAdapter(Protocol):
    """Minimal seam over ``pysr.PySRRegressor``.

    Keeping the seam this small means the mocked tests exercise the real control flow —
    probing, budget mapping, labelling, verification, failure capture — without needing PySR
    or Julia.
    """

    def backend_version(self) -> str:
        """Return the exact backend version string that was used."""
        ...

    def fit(
        self,
        *,
        task: SRTask,
        observations: ObservationSet,
        native_budget: Mapping[str, str],
        seed: int,
    ) -> Sequence[BaselineProposal]:
        """Return the discovered Pareto set as canonical ``srepr`` proposals."""
        ...


@runtime_checkable
class TransformerProposer(Protocol):
    """Workstream 2's compact prefix transformer, injected rather than reimplemented."""

    def descriptor(self) -> "TransformerDescriptor":
        """Return model, checkpoint, and training-data identity."""
        ...

    def propose(
        self, *, task: SRTask, observations: ObservationSet, seed: int, samples: int
    ) -> Sequence[BaselineProposal]:
        """Return sampled candidate expressions for one task."""
        ...


class TransformerDescriptor(BaseModel):
    """Identity and training-data policy of an injected transformer proposer."""

    model_config = _FROZEN

    model_id: str
    model_hash: str = ""
    checkpoint_hash: str = ""
    config_hash: str = ""
    training_split: str = "goal1_train"
    trained_on_benchmark_test_tasks: bool = False
    notes: str = ""


class TransformerTrainingLeakError(BaselineError):
    """An injected transformer declared that it saw benchmark test targets."""


# --------------------------------------------------------------------------------------
# Shared scoring of backend proposals
# --------------------------------------------------------------------------------------


def _score_proposals(
    proposals: Sequence[BaselineProposal],
    *,
    task: SRTask,
    method: SRMethod,
    representation: SRRepresentation,
    seed: int,
    fit_observations: ObservationSet,
    budget: SearchBudget,
    backend: BaselineBackend,
    verifier: EquivalenceVerifier,
) -> tuple[tuple[SRCandidate, ...], int, int, int]:
    """Score backend proposals through the same pipeline the controlled arms use.

    Returns the retained rows plus ``(invalid, verifier_timeouts, verifier_errors)``.
    """

    rows: list[SRCandidate] = []
    invalid = 0
    timeouts = 0
    errors = 0
    seen: set[str] = set()

    for step, proposal in enumerate(proposals):
        if proposal.srepr in seen:
            continue
        seen.add(proposal.srepr)
        candidate_id = derive_candidate_id(
            task_id=task.task_id, method=method, seed=seed, srepr=proposal.srepr, step=step
        )
        base = {
            "candidate_id": candidate_id,
            "task_id": task.task_id,
            "method": method,
            "representation": representation,
            "seed": seed,
            "expression_srepr": proposal.srepr,
            "expression_display": proposal.srepr,
            "structural_signature": None,
            "complexity": None,
            "representation_node_count": None,
            "fit": None,
            "evaluation_fit": None,
            "equivalence": None,
            "proposal_scorer_id": backend.value,
            "proposal_step": step,
            "proposal_score": proposal.backend_loss,
        }

        check = check_grammar(proposal.srepr)
        if not check.in_grammar:
            invalid += 1
            rows.append(
                SRCandidate(**base, status=CandidateStatus.OUT_OF_GRAMMAR, detail=check.detail)
            )
            continue

        candidate = SemanticCandidate(
            expression=sympy.sympify(proposal.srepr), srepr=proposal.srepr, depth=0
        )
        view = materialize(candidate, representation)
        if view.status is not CandidateStatus.EVALUATED:
            invalid += 1
            rows.append(
                SRCandidate(
                    **base,
                    status=CandidateStatus.MATERIALIZATION_FAILED,
                    detail=view.signature,
                )
            )
            continue

        from geml.ast.builder import build_ast_from_parsed
        from geml.ast.statistics import structural_signature
        from geml.data.sr.benchmark import measure_complexity
        from geml.parsing.srepr import parse_srepr

        tree = build_ast_from_parsed(parse_srepr(proposal.srepr), expression_id="0" * 64)
        base["structural_signature"] = structural_signature(tree)
        base["complexity"] = measure_complexity(tree)
        base["representation_node_count"] = view.node_count
        base["expression_display"] = sympy.sstr(candidate.expression, order="none")

        if base["complexity"].ast_node_count > budget.max_complexity:
            invalid += 1
            rows.append(
                SRCandidate(
                    **base,
                    status=CandidateStatus.COMPLEXITY_EXCEEDED,
                    detail=(f"{base['complexity'].ast_node_count} > {budget.max_complexity}"),
                )
            )
            continue

        fit = evaluate_numeric_fit(proposal.srepr, fit_observations)
        base["fit"] = fit
        if fit.status in {NumericFitStatus.INVALID_DOMAIN, NumericFitStatus.NONFINITE}:
            invalid += 1
            rows.append(
                SRCandidate(**base, status=CandidateStatus.DOMAIN_INVALID, detail=fit.detail)
            )
            continue
        if fit.status in {NumericFitStatus.PARSE_ERROR, NumericFitStatus.ERROR}:
            invalid += 1
            rows.append(
                SRCandidate(**base, status=CandidateStatus.EVALUATION_ERROR, detail=fit.detail)
            )
            continue

        equivalence = verify_exact_recovery(
            verifier,
            target_srepr=task.target_srepr,
            candidate_srepr=proposal.srepr,
            domain_mode=task.domain_mode,
            used_operators=task.used_operators,
        )
        if equivalence.outcome is EquivalenceOutcome.TIMEOUT:
            timeouts += 1
        elif equivalence.outcome is EquivalenceOutcome.ERROR:
            errors += 1
        base["equivalence"] = equivalence
        rows.append(SRCandidate(**base, status=CandidateStatus.EVALUATED))

    return tuple(rows), invalid, timeouts, errors


def _assemble(
    *,
    task: SRTask,
    method: SRMethod,
    representation: SRRepresentation,
    seed: int,
    budget: SearchBudget,
    budget_digest: str,
    backend: BaselineBackend,
    backend_version: str,
    verifier: EquivalenceVerifier,
    rows: Sequence[SRCandidate],
    evaluation_observations: ObservationSet | None,
    telemetry: ResourceTelemetry,
    status: RunStatus,
    termination: TerminationReason,
    native_budget: Mapping[str, str],
    budget_mismatch: str,
    detail: str = "",
) -> SRMethodResult:
    """Build the shared result row from scored proposals."""

    capability = verifier.capability()
    scored = [
        row
        for row in rows
        if row.status is CandidateStatus.EVALUATED
        and row.fit is not None
        and row.fit.root_mean_squared_error is not None
    ]
    best = (
        min(
            scored,
            key=lambda row: (
                row.fit.root_mean_squared_error,
                row.complexity.ast_node_count if row.complexity else 1 << 30,
                row.candidate_id,
            ),
        )
        if scored
        else None
    )

    final_rows = list(rows)
    evaluation_rmse: float | None = None
    if best is not None and evaluation_observations is not None:
        held_out = evaluate_numeric_fit(best.expression_srepr, evaluation_observations)
        evaluation_rmse = held_out.root_mean_squared_error
        final_rows = [
            row.model_copy(update={"evaluation_fit": held_out})
            if row.candidate_id == best.candidate_id
            else row
            for row in final_rows
        ]

    verified = next(
        (
            row
            for row in final_rows
            if row.equivalence is not None
            and row.equivalence.outcome is EquivalenceOutcome.VERIFIED
        ),
        None,
    )
    outcomes = {row.equivalence.outcome for row in final_rows if row.equivalence is not None}
    if verified is not None:
        recovery = EquivalenceOutcome.VERIFIED
    else:
        recovery = next(
            (
                outcome
                for outcome in (
                    EquivalenceOutcome.UNKNOWN,
                    EquivalenceOutcome.TIMEOUT,
                    EquivalenceOutcome.ERROR,
                    EquivalenceOutcome.NOT_EQUIVALENT,
                )
                if outcome in outcomes
            ),
            EquivalenceOutcome.UNSUPPORTED,
        )

    return SRMethodResult(
        task_id=task.task_id,
        task_set=task.task_set.value,
        family=task.family,
        method=method,
        representation=representation,
        seed=seed,
        status=status,
        termination_reason=termination,
        budget=budget,
        budget_digest=budget_digest,
        native_budget=dict(native_budget),
        budget_mismatch=budget_mismatch,
        backend_name=backend.value,
        backend_version=backend_version,
        scorer_id=backend.value,
        verifier_id=capability.verifier_id,
        verifier_version=capability.verifier_version,
        telemetry=telemetry,
        candidates=tuple(final_rows),
        best_fit_candidate_id=best.candidate_id if best else None,
        best_evaluation_rmse=evaluation_rmse,
        exact_recovery_outcome=recovery,
        exact_recovery_candidate_id=verified.candidate_id if verified else None,
        pareto_candidate_ids=pareto_front(final_rows),
        detail=detail,
    )


def _empty_result(
    *,
    task: SRTask,
    method: SRMethod,
    seed: int,
    budget: SearchBudget,
    budget_digest: str,
    backend: BaselineBackend,
    backend_version: str,
    status: RunStatus,
    termination: TerminationReason,
    detail: str,
    native_budget: Mapping[str, str] | None = None,
    budget_mismatch: str = "",
) -> SRMethodResult:
    """Return an explicit failure row so a missing backend is never an absent task."""

    return SRMethodResult(
        task_id=task.task_id,
        task_set=task.task_set.value,
        family=task.family,
        method=method,
        representation=SRRepresentation.NONE,
        seed=seed,
        status=status,
        termination_reason=termination,
        budget=budget,
        budget_digest=budget_digest,
        native_budget=dict(native_budget or {}),
        budget_mismatch=budget_mismatch,
        backend_name=backend.value,
        backend_version=backend_version,
        scorer_id=backend.value,
        telemetry=ResourceTelemetry(
            wall_seconds=0.0,
            expansions=0,
            candidates_generated=0,
            candidates_retained=0,
            duplicates_skipped=0,
            invalid_candidates=0,
            verifier_calls=0,
            verifier_timeouts=0,
            verifier_errors=0,
            scorer_batches=0,
            peak_frontier=0,
        ),
        exact_recovery_outcome=EquivalenceOutcome.UNSUPPORTED,
        detail=detail,
    )


# --------------------------------------------------------------------------------------
# PySR
# --------------------------------------------------------------------------------------


def run_pysr_baseline(
    *,
    task: SRTask,
    fit_observations: ObservationSet,
    evaluation_observations: ObservationSet | None,
    budget: SearchBudget,
    budget_digest: str,
    seed: int,
    config: BaselineConfig | None = None,
    adapter: PySRAdapter | None = None,
    probe: BackendProbe | None = None,
    verifier: EquivalenceVerifier | None = None,
) -> SRMethodResult:
    """Run the version-pinned PySR baseline, or emit an explicit unavailability row.

    ``adapter`` is the only seam. When PySR is genuinely unavailable this returns a
    ``dependency_unavailable`` row naming the exact version found; it never quietly becomes
    the genetic-programming fallback. Callers that want the fallback must ask for it by name
    through :func:`run_gp_fallback_baseline`, which emits ``SRMethod.GP_FALLBACK``.
    """

    settings = config or BaselineConfig()
    native = settings.pysr_native_budget(budget)
    resolved_probe = probe or probe_pysr(pinned_version=settings.pysr_pinned_version)

    if adapter is None or not resolved_probe.usable:
        return _empty_result(
            task=task,
            method=SRMethod.PYSR,
            seed=seed,
            budget=budget,
            budget_digest=budget_digest,
            backend=BaselineBackend.PYSR,
            backend_version=resolved_probe.found_version,
            status=RunStatus.DEPENDENCY_UNAVAILABLE,
            termination=TerminationReason.ERROR,
            detail=resolved_probe.detail or "no PySR adapter was supplied",
            native_budget=native,
            budget_mismatch=PYSR_BUDGET_MISMATCH,
        )

    started = time.perf_counter()
    active_verifier = verifier or UnavailableEquivalenceVerifier()
    try:
        proposals = adapter.fit(
            task=task, observations=fit_observations, native_budget=native, seed=seed
        )
        backend_version = adapter.backend_version()
    except TimeoutError as error:
        return _empty_result(
            task=task,
            method=SRMethod.PYSR,
            seed=seed,
            budget=budget,
            budget_digest=budget_digest,
            backend=BaselineBackend.PYSR,
            backend_version=resolved_probe.found_version,
            status=RunStatus.TIMEOUT,
            termination=TerminationReason.WALL_TIME_BUDGET,
            detail=f"{type(error).__name__}: {error}",
            native_budget=native,
            budget_mismatch=PYSR_BUDGET_MISMATCH,
        )
    except Exception as error:
        return _empty_result(
            task=task,
            method=SRMethod.PYSR,
            seed=seed,
            budget=budget,
            budget_digest=budget_digest,
            backend=BaselineBackend.PYSR,
            backend_version=resolved_probe.found_version,
            status=RunStatus.FAILED,
            termination=TerminationReason.ERROR,
            detail=f"{type(error).__name__}: {error}",
            native_budget=native,
            budget_mismatch=PYSR_BUDGET_MISMATCH,
        )

    if backend_version != settings.pysr_pinned_version:
        return _empty_result(
            task=task,
            method=SRMethod.PYSR,
            seed=seed,
            budget=budget,
            budget_digest=budget_digest,
            backend=BaselineBackend.PYSR,
            backend_version=backend_version,
            status=RunStatus.DEPENDENCY_UNAVAILABLE,
            termination=TerminationReason.ERROR,
            detail=(
                f"adapter reported backend version {backend_version!r}, which is not the "
                f"pinned {settings.pysr_pinned_version!r}"
            ),
            native_budget=native,
            budget_mismatch=PYSR_BUDGET_MISMATCH,
        )

    rows, invalid, timeouts, errors = _score_proposals(
        proposals,
        task=task,
        method=SRMethod.PYSR,
        representation=SRRepresentation.NONE,
        seed=seed,
        fit_observations=fit_observations,
        budget=budget,
        backend=BaselineBackend.PYSR,
        verifier=active_verifier,
    )
    telemetry = ResourceTelemetry(
        wall_seconds=time.perf_counter() - started,
        expansions=0,
        candidates_generated=len(proposals),
        candidates_retained=len(rows),
        duplicates_skipped=max(0, len(proposals) - len(rows)),
        invalid_candidates=invalid,
        verifier_calls=len(rows) - invalid,
        verifier_timeouts=timeouts,
        verifier_errors=errors,
        scorer_batches=0,
        peak_frontier=0,
    )
    return _assemble(
        task=task,
        method=SRMethod.PYSR,
        representation=SRRepresentation.NONE,
        seed=seed,
        budget=budget,
        budget_digest=budget_digest,
        backend=BaselineBackend.PYSR,
        backend_version=backend_version,
        verifier=active_verifier,
        rows=rows,
        evaluation_observations=evaluation_observations,
        telemetry=telemetry,
        status=RunStatus.COMPLETE,
        termination=TerminationReason.CANDIDATE_BUDGET,
        native_budget=native,
        budget_mismatch=PYSR_BUDGET_MISMATCH,
    )


# --------------------------------------------------------------------------------------
# Explicitly labelled in-repository genetic-programming fallback
# --------------------------------------------------------------------------------------


GP_BUDGET_MISMATCH = (
    "The in-repository genetic-programming fallback is budgeted in generations x population "
    "size, mapped from the shared expansion budget. It shares the controlled arms' grammar, "
    "primitive inventory, depth ceiling, complexity ceiling, wall-clock limit, observations, "
    "seeds, and task set. It is not PySR and is reported under its own method label."
)


def run_gp_fallback_baseline(
    *,
    task: SRTask,
    fit_observations: ObservationSet,
    evaluation_observations: ObservationSet | None,
    budget: SearchBudget,
    budget_digest: str,
    seed: int,
    config: BaselineConfig | None = None,
    verifier: EquivalenceVerifier | None = None,
) -> SRMethodResult:
    """Run the deterministic in-repository GP fallback, labelled as itself.

    The issue permits an explicitly labelled fallback when PySR cannot be used. This
    function always emits ``SRMethod.GP_FALLBACK`` and ``BaselineBackend.GP_FALLBACK``; no
    code path can relabel its output as PySR.
    """

    settings = config or BaselineConfig()
    if not settings.gp_fallback_enabled:
        return _empty_result(
            task=task,
            method=SRMethod.GP_FALLBACK,
            seed=seed,
            budget=budget,
            budget_digest=budget_digest,
            backend=BaselineBackend.GP_FALLBACK,
            backend_version="disabled",
            status=RunStatus.UNSUPPORTED,
            termination=TerminationReason.ERROR,
            detail="gp_fallback_enabled is false",
            budget_mismatch=GP_BUDGET_MISMATCH,
        )

    started = time.perf_counter()
    active_verifier = verifier or UnavailableEquivalenceVerifier()
    rng = Random(seed)
    symbols = symbols_for_task(task)
    population = [
        SemanticCandidate(expression=symbol, srepr=sympy.srepr(symbol, order="none"), depth=0)
        for symbol in symbols
    ]
    proposals: dict[str, BaselineProposal] = {}
    generations = 0
    generated = 0

    while generations < settings.gp_generations:
        if time.perf_counter() - started >= budget.wall_seconds:
            break
        if len(proposals) >= budget.max_candidates:
            break
        generations += 1
        offspring: list[SemanticCandidate] = []
        for parent in population:
            children = generate_moves(
                parent,
                symbols,
                rng,
                max_depth=budget.max_depth,
                fan_out=max(2, settings.gp_population_size // max(1, len(population))),
            )
            generated += len(children)
            offspring.extend(children)
        if not offspring:
            break
        scored: list[tuple[float, SemanticCandidate]] = []
        for child in offspring:
            fit = evaluate_numeric_fit(child.srepr, fit_observations)
            error = (
                fit.root_mean_squared_error
                if fit.root_mean_squared_error is not None
                else float("inf")
            )
            scored.append((error, child))
            proposals.setdefault(
                child.srepr, BaselineProposal(srepr=child.srepr, backend_loss=error)
            )
        scored.sort(key=lambda item: (item[0], item[1].srepr))
        population = [child for _error, child in scored[: settings.gp_tournament_size]]

    ordered = [proposals[key] for key in sorted(proposals)]
    rows, invalid, timeouts, errors = _score_proposals(
        ordered,
        task=task,
        method=SRMethod.GP_FALLBACK,
        representation=SRRepresentation.NONE,
        seed=seed,
        fit_observations=fit_observations,
        budget=budget,
        backend=BaselineBackend.GP_FALLBACK,
        verifier=active_verifier,
    )
    telemetry = ResourceTelemetry(
        wall_seconds=time.perf_counter() - started,
        expansions=generations,
        candidates_generated=generated,
        candidates_retained=len(rows),
        duplicates_skipped=max(0, generated - len(ordered)),
        invalid_candidates=invalid,
        verifier_calls=len(rows) - invalid,
        verifier_timeouts=timeouts,
        verifier_errors=errors,
        scorer_batches=0,
        peak_frontier=len(population),
    )
    return _assemble(
        task=task,
        method=SRMethod.GP_FALLBACK,
        representation=SRRepresentation.NONE,
        seed=seed,
        budget=budget,
        budget_digest=budget_digest,
        backend=BaselineBackend.GP_FALLBACK,
        backend_version=f"geml-gp-fallback-v1;generations={settings.gp_generations}",
        verifier=active_verifier,
        rows=rows,
        evaluation_observations=evaluation_observations,
        telemetry=telemetry,
        status=RunStatus.COMPLETE,
        termination=TerminationReason.EXPANSION_BUDGET,
        native_budget={
            "generations": str(settings.gp_generations),
            "population_size": str(settings.gp_population_size),
            "tournament_size": str(settings.gp_tournament_size),
            "max_depth": str(budget.max_depth),
            "max_complexity": str(budget.max_complexity),
            "wall_seconds": str(budget.wall_seconds),
        },
        budget_mismatch=GP_BUDGET_MISMATCH,
    )


# --------------------------------------------------------------------------------------
# Compact transformer-SR
# --------------------------------------------------------------------------------------

TRANSFORMER_BUDGET_MISMATCH = (
    "The compact transformer proposes a fixed number of samples per task rather than "
    "expanding nodes. Wall time, depth, complexity, observations, seeds, and task set are "
    "matched; the sample count is reported as the native budget."
)


def run_transformer_baseline(
    *,
    task: SRTask,
    fit_observations: ObservationSet,
    evaluation_observations: ObservationSet | None,
    budget: SearchBudget,
    budget_digest: str,
    seed: int,
    proposer: TransformerProposer | None = None,
    config: BaselineConfig | None = None,
    verifier: EquivalenceVerifier | None = None,
) -> SRMethodResult:
    """Run the compact transformer-SR baseline using Workstream 2's injected backbone.

    Raises :class:`TransformerTrainingLeakError` if the injected proposer declares that it
    was trained on benchmark test targets. Refusing loudly is the point: a leaked baseline is
    worse than a missing one.
    """

    settings = config or BaselineConfig()
    if proposer is None:
        return _empty_result(
            task=task,
            method=SRMethod.TRANSFORMER_SR,
            seed=seed,
            budget=budget,
            budget_digest=budget_digest,
            backend=BaselineBackend.TRANSFORMER_SR,
            backend_version="",
            status=RunStatus.DEPENDENCY_UNAVAILABLE,
            termination=TerminationReason.ERROR,
            detail=(
                "no transformer proposer was injected; Workstream 2 owns the compact "
                "prefix backbone and this module implements no second transformer stack"
            ),
            native_budget={"samples": str(settings.transformer_samples)},
            budget_mismatch=TRANSFORMER_BUDGET_MISMATCH,
        )

    descriptor = proposer.descriptor()
    if descriptor.trained_on_benchmark_test_tasks:
        raise TransformerTrainingLeakError(
            f"transformer {descriptor.model_id!r} declares training on benchmark test "
            "targets; it may not be evaluated on this benchmark"
        )

    started = time.perf_counter()
    active_verifier = verifier or UnavailableEquivalenceVerifier()
    try:
        proposals = proposer.propose(
            task=task,
            observations=fit_observations,
            seed=seed,
            samples=settings.transformer_samples,
        )
    except TimeoutError as error:
        return _empty_result(
            task=task,
            method=SRMethod.TRANSFORMER_SR,
            seed=seed,
            budget=budget,
            budget_digest=budget_digest,
            backend=BaselineBackend.TRANSFORMER_SR,
            backend_version=descriptor.model_id,
            status=RunStatus.TIMEOUT,
            termination=TerminationReason.WALL_TIME_BUDGET,
            detail=f"{type(error).__name__}: {error}",
            native_budget={"samples": str(settings.transformer_samples)},
            budget_mismatch=TRANSFORMER_BUDGET_MISMATCH,
        )
    except Exception as error:
        return _empty_result(
            task=task,
            method=SRMethod.TRANSFORMER_SR,
            seed=seed,
            budget=budget,
            budget_digest=budget_digest,
            backend=BaselineBackend.TRANSFORMER_SR,
            backend_version=descriptor.model_id,
            status=RunStatus.FAILED,
            termination=TerminationReason.ERROR,
            detail=f"{type(error).__name__}: {error}",
            native_budget={"samples": str(settings.transformer_samples)},
            budget_mismatch=TRANSFORMER_BUDGET_MISMATCH,
        )

    rows, invalid, timeouts, errors = _score_proposals(
        proposals,
        task=task,
        method=SRMethod.TRANSFORMER_SR,
        representation=SRRepresentation.NONE,
        seed=seed,
        fit_observations=fit_observations,
        budget=budget,
        backend=BaselineBackend.TRANSFORMER_SR,
        verifier=active_verifier,
    )
    telemetry = ResourceTelemetry(
        wall_seconds=time.perf_counter() - started,
        expansions=0,
        candidates_generated=len(proposals),
        candidates_retained=len(rows),
        duplicates_skipped=max(0, len(proposals) - len(rows)),
        invalid_candidates=invalid,
        verifier_calls=len(rows) - invalid,
        verifier_timeouts=timeouts,
        verifier_errors=errors,
        scorer_batches=1,
        peak_frontier=0,
    )
    return _assemble(
        task=task,
        method=SRMethod.TRANSFORMER_SR,
        representation=SRRepresentation.NONE,
        seed=seed,
        budget=budget,
        budget_digest=budget_digest,
        backend=BaselineBackend.TRANSFORMER_SR,
        backend_version=descriptor.model_id,
        verifier=active_verifier,
        rows=rows,
        evaluation_observations=evaluation_observations,
        telemetry=telemetry,
        status=RunStatus.COMPLETE,
        termination=TerminationReason.CANDIDATE_BUDGET,
        native_budget={
            "samples": str(settings.transformer_samples),
            "model_id": descriptor.model_id,
            "training_split": descriptor.training_split,
            "checkpoint_hash": descriptor.checkpoint_hash,
        },
        budget_mismatch=TRANSFORMER_BUDGET_MISMATCH,
    )


def shared_primitive_inventory() -> dict[str, tuple[str, ...]]:
    """Return the primitive inventory every Goal 9 method is limited to.

    Exposed so that an audit can assert the PySR operator lists are a faithful projection of
    the controlled arms' move set rather than a richer or poorer search space.
    """

    return {
        "binary_moves": SHARED_BINARY_MOVES,
        "unary_moves": SHARED_UNARY_MOVES,
        "constants": tuple(str(constant) for constant in SHARED_CONSTANTS),
        "pysr_binary_operators": PYSR_BINARY_OPERATORS,
        "pysr_unary_operators": PYSR_UNARY_OPERATORS,
    }
