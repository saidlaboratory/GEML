"""Matched-budget EML-guided versus AST-guided symbolic-regression search (issue 9-1).

The scientific point of Goal 9 is a *representation* contrast, so the two guided arms must
differ only in the materialised representation and the learned guidance. Everything else is
frozen and shared:

* one canonical source-expression grammar (the enabled v1 vocabulary from issue 9-0);
* one primitive, operator, and constant inventory;
* one representation-independent candidate-generation move set;
* the same task ids and the same fit observations;
* identical wall-time, expansion, candidate, depth, and complexity budgets;
* one complexity objective and one stopping rule;
* the same seeds and the same checkpoint cadence.

A candidate is generated once as a :class:`SemanticCandidate` over the shared grammar and is
then *materialised* into an AST view or a pure-EML view for the respective guide. Generating
natively in EML against a different AST grammar would confound representation with search
space, so it is not done. Neither arm receives a hidden primitive or a precomputed target.

This module also owns the shared candidate and per-method result schemas that issue 9-2
reuses, so the two tracks cannot drift into duplicate incompatible records.
"""

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from random import Random
from typing import Protocol, runtime_checkable

import sympy
from pydantic import BaseModel, ConfigDict, Field, model_validator

from geml.ast.builder import build_ast_from_parsed
from geml.ast.statistics import structural_signature
from geml.data.sr.benchmark import (
    ComplexityMeasure,
    EGraphFragmentEquivalenceVerifier,
    EquivalenceOutcome,
    EquivalenceResult,
    EquivalenceVerifier,
    NumericFit,
    NumericFitStatus,
    ObservationSet,
    SRTask,
    check_grammar,
    evaluate_numeric_fit,
    measure_complexity,
    verify_exact_recovery,
)
from geml.parsing.srepr import SreprParseError, parse_srepr

SR_CANDIDATE_SCHEMA_VERSION = "geml-sr-candidate-v1"
SR_METHOD_RESULT_SCHEMA_VERSION = "geml-sr-method-result-v1"

_CANDIDATE_ID_PREFIX = "geml-sr-candidate-v1"

PRODUCTION_OUTPUT_ROOT = "outputs/final/goal9/guided_search"

_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class GuidedSearchError(ValueError):
    """A search contract, budget, or configuration input was invalid."""


class SRMethod(StrEnum):
    """Every Goal 9 method that can emit a :class:`SRMethodResult`."""

    EML_GUIDED = "eml_guided"
    AST_GUIDED = "ast_guided"
    PYSR = "pysr"
    GP_FALLBACK = "gp_fallback"
    TRANSFORMER_SR = "transformer_sr"


class SRRepresentation(StrEnum):
    """The materialised view a guide consumes.

    ``NONE`` marks a method whose guidance is not representation-conditioned at all, such as
    an external genetic-programming or sequence baseline; it is never used to disguise a
    fallback as one of the two controlled arms.
    """

    PURE_EML = "pure_eml"
    AST = "ast"
    NONE = "none"


class CandidateStatus(StrEnum):
    """Per-candidate outcome. Invalid candidates are retained, never discarded."""

    EVALUATED = "evaluated"
    OUT_OF_GRAMMAR = "out_of_grammar"
    DOMAIN_INVALID = "domain_invalid"
    COMPLEXITY_EXCEEDED = "complexity_exceeded"
    DEPTH_EXCEEDED = "depth_exceeded"
    MATERIALIZATION_FAILED = "materialization_failed"
    EVALUATION_ERROR = "evaluation_error"


class TerminationReason(StrEnum):
    """Why a search run stopped. Both arms use the same stopping rules."""

    EXACT_RECOVERY = "exact_recovery"
    WALL_TIME_BUDGET = "wall_time_budget"
    EXPANSION_BUDGET = "expansion_budget"
    CANDIDATE_BUDGET = "candidate_budget"
    FRONTIER_EXHAUSTED = "frontier_exhausted"
    ERROR = "error"


class RunStatus(StrEnum):
    """Whether a (task, method, seed) cell produced a usable result row."""

    COMPLETE = "complete"
    TIMEOUT = "timeout"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


# --------------------------------------------------------------------------------------
# Frozen shared budget
# --------------------------------------------------------------------------------------


class SearchBudget(BaseModel):
    """The budget every controlled Goal 9 method must receive identically."""

    model_config = _FROZEN

    wall_seconds: float = Field(gt=0.0)
    max_expansions: int = Field(ge=1)
    max_candidates: int = Field(ge=1)
    max_depth: int = Field(ge=1)
    max_complexity: int = Field(ge=1)
    beam_width: int = Field(ge=1)
    checkpoint_every_candidates: int = Field(default=64, ge=1)

    def digest(self) -> str:
        """Return a stable digest of the budget for equality checks and provenance."""

        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_matched_budgets(budgets: Mapping[str, SearchBudget]) -> str:
    """Raise unless every named method received a byte-identical budget.

    Returns the shared budget digest so it can be recorded on every emitted row.
    """

    if not budgets:
        raise GuidedSearchError("no budgets were supplied")
    digests = {name: budget.digest() for name, budget in budgets.items()}
    unique = set(digests.values())
    if len(unique) != 1:
        detail = ", ".join(f"{name}={digest[:12]}" for name, digest in sorted(digests.items()))
        raise GuidedSearchError(f"methods did not receive identical budgets: {detail}")
    return unique.pop()


# --------------------------------------------------------------------------------------
# Representation-independent candidate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    """One expression in the shared source grammar, before any representation choice."""

    expression: sympy.Expr
    srepr: str
    depth: int

    @property
    def key(self) -> str:
        """Return the deterministic structural deduplication key."""

        return self.srepr


class MaterializedCandidate(BaseModel):
    """A candidate rendered into one representation for one guide."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    representation: SRRepresentation
    node_count: int = Field(ge=0)
    depth: int = Field(ge=0)
    signature: str
    status: CandidateStatus


class SRCandidate(BaseModel):
    """One retained candidate row, shared by every Goal 9 method."""

    model_config = _FROZEN

    schema_version: str = SR_CANDIDATE_SCHEMA_VERSION
    candidate_id: str
    task_id: str
    method: SRMethod
    representation: SRRepresentation
    seed: int
    expression_srepr: str
    expression_display: str
    structural_signature: str | None
    complexity: ComplexityMeasure | None
    representation_node_count: int | None
    fit: NumericFit | None
    evaluation_fit: NumericFit | None
    equivalence: EquivalenceResult | None
    status: CandidateStatus
    detail: str = ""
    proposal_scorer_id: str = ""
    proposal_score: float | None = None
    proposal_step: int = Field(default=0, ge=0)
    parent_candidate_id: str | None = None


class ResourceTelemetry(BaseModel):
    """Complete resource accounting for one (task, method, seed) cell."""

    model_config = _FROZEN

    wall_seconds: float = Field(ge=0.0)
    expansions: int = Field(ge=0)
    candidates_generated: int = Field(ge=0)
    candidates_retained: int = Field(ge=0)
    duplicates_skipped: int = Field(ge=0)
    invalid_candidates: int = Field(ge=0)
    verifier_calls: int = Field(ge=0)
    verifier_timeouts: int = Field(ge=0)
    verifier_errors: int = Field(ge=0)
    scorer_batches: int = Field(ge=0)
    peak_frontier: int = Field(ge=0)


class SRMethodResult(BaseModel):
    """The single result schema every Goal 9 method emits."""

    model_config = _FROZEN

    schema_version: str = SR_METHOD_RESULT_SCHEMA_VERSION
    task_id: str
    task_set: str
    family: str
    method: SRMethod
    representation: SRRepresentation
    seed: int
    status: RunStatus
    termination_reason: TerminationReason
    budget: SearchBudget
    budget_digest: str
    native_budget: dict[str, str] = Field(default_factory=dict)
    budget_mismatch: str = ""
    backend_name: str = ""
    backend_version: str = ""
    scorer_id: str = ""
    verifier_id: str = ""
    verifier_version: str = ""
    telemetry: ResourceTelemetry
    candidates: tuple[SRCandidate, ...] = ()
    best_fit_candidate_id: str | None = None
    best_evaluation_rmse: float | None = None
    exact_recovery_outcome: EquivalenceOutcome = EquivalenceOutcome.UNSUPPORTED
    exact_recovery_candidate_id: str | None = None
    pareto_candidate_ids: tuple[str, ...] = ()
    detail: str = ""

    @model_validator(mode="after")
    def _no_recovery_without_evidence(self) -> "SRMethodResult":
        if (
            self.exact_recovery_outcome is EquivalenceOutcome.VERIFIED
            and self.exact_recovery_candidate_id is None
        ):
            raise ValueError("a verified exact recovery must name the candidate that achieved it")
        if self.budget.digest() != self.budget_digest:
            raise ValueError("budget_digest does not match the recorded budget")
        return self


# --------------------------------------------------------------------------------------
# Guidance interface
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoringRequest:
    """One scoring question: a materialised candidate in the context of one task."""

    task_id: str
    representation: SRRepresentation
    candidate: SemanticCandidate
    materialized: MaterializedCandidate
    fit: NumericFit | None


@runtime_checkable
class CandidateScorer(Protocol):
    """Guidance model interface.

    Workstream 2 owns the compact GNN and prefix transformer. During Phase A this protocol
    is satisfied by injected deterministic fixture scorers; after that branch merges, the
    shared backbone is wrapped behind the same two methods with no change here.
    """

    def descriptor(self) -> "ScorerDescriptor":
        """Return the identity of the model, checkpoint, and configuration."""
        ...

    def score_batch(self, requests: Sequence[ScoringRequest]) -> tuple[float, ...]:
        """Return one score per request, higher meaning more promising."""
        ...


class ScorerDescriptor(BaseModel):
    """Identity of a guidance model, recorded on every row it influenced."""

    model_config = _FROZEN

    scorer_id: str
    representation: SRRepresentation
    model_hash: str = ""
    checkpoint_hash: str = ""
    config_hash: str = ""
    training_data_policy: str = "not_trained"
    notes: str = ""


#: Every fixture scorer id carries this prefix so downstream guards can recognise rows
#: that were never produced by a trained model.
FIXTURE_SCORER_ID_PREFIX = "fixture-"


class ErrorPriorityScorer:
    """Deterministic fixture scorer used by Phase-A smokes.

    It ranks purely on fit error and a complexity penalty, so it is identical across
    representations by construction. It exists to exercise the search machinery without
    pretending to be a learned model: ``training_data_policy`` says ``not_trained``.
    """

    def __init__(self, representation: SRRepresentation, *, complexity_weight: float = 0.01):
        self._representation = representation
        self._complexity_weight = complexity_weight

    def descriptor(self) -> ScorerDescriptor:
        """Return the fixture scorer identity."""

        return ScorerDescriptor(
            scorer_id=f"{FIXTURE_SCORER_ID_PREFIX}error-priority:{self._representation.value}",
            representation=self._representation,
            training_data_policy="not_trained",
            notes="Phase-A fixture scorer; replaced by the shared compact model after merge",
        )

    def score_batch(self, requests: Sequence[ScoringRequest]) -> tuple[float, ...]:
        """Score every request deterministically from its numeric fit and size."""

        scores: list[float] = []
        for request in requests:
            fit = request.fit
            if fit is None or fit.root_mean_squared_error is None:
                scores.append(-1e18)
                continue
            penalty = self._complexity_weight * request.materialized.node_count
            scores.append(-(fit.root_mean_squared_error + penalty))
        return tuple(scores)


# --------------------------------------------------------------------------------------
# Shared candidate-generation moves
# --------------------------------------------------------------------------------------

#: The frozen verifier-supported primitive inventory. Both arms receive exactly this.
SHARED_BINARY_MOVES: tuple[str, ...] = ("add", "subtract", "multiply", "divide")
SHARED_UNARY_MOVES: tuple[str, ...] = ("negate", "exp", "log")
SHARED_CONSTANTS: tuple[sympy.Expr, ...] = (
    sympy.Integer(1),
    sympy.Integer(2),
    sympy.Integer(-1),
    sympy.Rational(1, 2),
)
SHARED_POWERS: tuple[sympy.Expr, ...] = (sympy.Integer(2), sympy.Integer(3))


def _apply_binary(name: str, left: sympy.Expr, right: sympy.Expr) -> sympy.Expr | None:
    if name == "add":
        return left + right
    if name == "subtract":
        return left - right
    if name == "multiply":
        return left * right
    if right.is_zero:
        return None
    return left / right


def _apply_unary(name: str, argument: sympy.Expr) -> sympy.Expr:
    if name == "negate":
        return -argument
    return getattr(sympy, name)(argument)


def _expression_depth(expression: sympy.Expr) -> int:
    if not expression.args:
        return 0
    return 1 + max(_expression_depth(argument) for argument in expression.args)


def generate_moves(
    parent: SemanticCandidate,
    symbols: Sequence[sympy.Symbol],
    rng: Random,
    *,
    max_depth: int,
    fan_out: int,
) -> tuple[SemanticCandidate, ...]:
    """Expand one candidate using the shared, representation-independent move set.

    The move set is a pure function of ``parent``, the declared variables, and the seeded
    stream. It never consults the target expression, and it is the same object for both
    guided arms.
    """

    leaves: list[sympy.Expr] = [*symbols, *SHARED_CONSTANTS]
    produced: dict[str, SemanticCandidate] = {}
    attempts = fan_out * 4
    for _ in range(attempts):
        if len(produced) >= fan_out:
            break
        roll = rng.random()
        if roll < 0.45:
            name = SHARED_BINARY_MOVES[rng.randrange(len(SHARED_BINARY_MOVES))]
            other = leaves[rng.randrange(len(leaves))]
            candidate = (
                _apply_binary(name, parent.expression, other)
                if rng.random() < 0.5
                else _apply_binary(name, other, parent.expression)
            )
        elif roll < 0.75:
            name = SHARED_UNARY_MOVES[rng.randrange(len(SHARED_UNARY_MOVES))]
            candidate = _apply_unary(name, parent.expression)
        elif roll < 0.9:
            candidate = sympy.Pow(
                parent.expression, SHARED_POWERS[rng.randrange(len(SHARED_POWERS))]
            )
        else:
            candidate = leaves[rng.randrange(len(leaves))]

        if candidate is None or not isinstance(candidate, sympy.Expr):
            continue
        depth = _expression_depth(candidate)
        if depth > max_depth:
            continue
        srepr = sympy.srepr(candidate, order="none")
        if srepr == parent.srepr or srepr in produced:
            continue
        produced[srepr] = SemanticCandidate(expression=candidate, srepr=srepr, depth=depth)
    return tuple(produced[key] for key in sorted(produced))


def materialize(
    candidate: SemanticCandidate, representation: SRRepresentation
) -> MaterializedCandidate:
    """Render one shared candidate into the requested representation view.

    ``AST`` uses the validated binary source AST. ``PURE_EML`` uses the frozen expanded
    pure-EML DAG cost boundary. Neither path is allowed to fall back to the other: a failure
    is a typed ``materialization_failed`` row.
    """

    try:
        parsed = parse_srepr(candidate.srepr)
        tree = build_ast_from_parsed(parsed, expression_id="0" * 64)
    except (SreprParseError, ValueError) as error:
        return MaterializedCandidate(
            representation=representation,
            node_count=0,
            depth=0,
            signature=f"invalid:{type(error).__name__}",
            status=CandidateStatus.MATERIALIZATION_FAILED,
        )

    if representation is SRRepresentation.AST:
        return MaterializedCandidate(
            representation=representation,
            node_count=tree.statistics.node_count,
            depth=tree.statistics.depth,
            signature=structural_signature(tree),
            status=CandidateStatus.EVALUATED,
        )

    if representation is SRRepresentation.NONE:
        return MaterializedCandidate(
            representation=representation,
            node_count=tree.statistics.node_count,
            depth=tree.statistics.depth,
            signature=structural_signature(tree),
            status=CandidateStatus.EVALUATED,
        )

    from geml.interfaces.eml_dag_cost import EMLDagCostStatus, compute_eml_dag_cost

    cost = compute_eml_dag_cost(tree)
    if cost.status is not EMLDagCostStatus.SUCCESS or cost.eml_dag_node_count is None:
        return MaterializedCandidate(
            representation=representation,
            node_count=0,
            depth=0,
            signature=f"eml_cost:{cost.status}",
            status=CandidateStatus.MATERIALIZATION_FAILED,
        )
    return MaterializedCandidate(
        representation=representation,
        node_count=cost.eml_dag_node_count,
        depth=cost.eml_dag_depth or 0,
        signature=cost.root_signature or structural_signature(tree),
        status=CandidateStatus.EVALUATED,
    )


def derive_candidate_id(*, task_id: str, method: SRMethod, seed: int, srepr: str, step: int) -> str:
    """Return the version-tagged identity of one retained candidate row."""

    hasher = hashlib.sha256()
    hasher.update(_CANDIDATE_ID_PREFIX.encode("utf-8"))
    for field in (task_id, method.value, str(seed), srepr, str(step)):
        encoded = field.encode("utf-8")
        hasher.update(f"{len(encoded)}:".encode("ascii"))
        hasher.update(encoded)
        hasher.update(b"\0")
    return hasher.hexdigest()


# --------------------------------------------------------------------------------------
# Accuracy-complexity Pareto front
# --------------------------------------------------------------------------------------


def pareto_front(candidates: Sequence[SRCandidate]) -> tuple[str, ...]:
    """Return the accuracy-complexity Pareto-optimal candidate ids.

    Accuracy is the fit root-mean-squared error and complexity is the shared source-AST node
    count, so the front is comparable across representations. Ties are broken by candidate
    id so the front is deterministic.
    """

    points: list[tuple[float, int, str]] = []
    for candidate in candidates:
        if candidate.status is not CandidateStatus.EVALUATED:
            continue
        if candidate.fit is None or candidate.fit.root_mean_squared_error is None:
            continue
        if candidate.complexity is None:
            continue
        points.append(
            (
                candidate.fit.root_mean_squared_error,
                candidate.complexity.ast_node_count,
                candidate.candidate_id,
            )
        )
    points.sort(key=lambda item: (item[1], item[0], item[2]))

    front: list[str] = []
    best_error = float("inf")
    for error, _complexity, candidate_id in points:
        if error < best_error:
            best_error = error
            front.append(candidate_id)
    return tuple(front)


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------


def checkpoint_path(root: str | Path, task_id: str, method: SRMethod, seed: int) -> Path:
    """Return the atomic per-task, per-method, per-seed checkpoint path."""

    return Path(root) / method.value / f"{task_id}.seed{seed}.json"


def write_result(result: SRMethodResult, root: str | Path) -> Path:
    """Write one cell atomically. Existing evidence is replaced only by a complete file."""

    path = checkpoint_path(root, result.task_id, result.method, result.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def read_result(path: str | Path) -> SRMethodResult:
    """Load one persisted cell."""

    return SRMethodResult.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_results(root: str | Path) -> tuple[SRMethodResult, ...]:
    """Load every persisted cell under ``root`` in a stable order."""

    paths = sorted(Path(root).rglob("*.json"))
    return tuple(read_result(path) for path in paths)


# --------------------------------------------------------------------------------------
# The search itself
# --------------------------------------------------------------------------------------


def symbols_for_task(task: SRTask) -> tuple[sympy.Symbol, ...]:
    """Return the declared variables as SymPy symbols carrying their domain assumptions.

    Shared with the baselines so that both tracks build candidates over literally the same
    symbols, which is a precondition for the structural deduplication keys to be comparable.
    """

    return _symbols_for(task)


def _symbols_for(task: SRTask) -> tuple[sympy.Symbol, ...]:
    symbols: list[sympy.Symbol] = []
    for variable in task.variables:
        if variable.domain_mode == "positive_real":
            symbols.append(sympy.Symbol(variable.name, positive=True))
        elif variable.domain_mode == "nonzero_real":
            symbols.append(sympy.Symbol(variable.name, nonzero=True, real=True))
        else:
            symbols.append(sympy.Symbol(variable.name, real=True))
    return tuple(symbols)


def run_guided_search(
    *,
    task: SRTask,
    fit_observations: ObservationSet,
    evaluation_observations: ObservationSet | None,
    method: SRMethod,
    representation: SRRepresentation,
    scorer: CandidateScorer,
    budget: SearchBudget,
    budget_digest: str,
    seed: int,
    verifier: EquivalenceVerifier | None = None,
    clock: "object | None" = None,
) -> SRMethodResult:
    """Run one (task, method, seed) cell under the shared budget.

    The traversal is a deterministic beam over the shared move set. Structural
    deduplication is exact and representation-independent, scoring is batched, and every
    invalid, duplicated, or failed candidate is retained as a typed row. There is no
    implicit fallback between the EML and AST arms: a materialisation failure stays a
    failure.
    """

    if representation is SRRepresentation.NONE and method in {
        SRMethod.EML_GUIDED,
        SRMethod.AST_GUIDED,
    }:
        raise GuidedSearchError(
            f"{method.value} must declare a concrete representation, not 'none'"
        )
    if budget.digest() != budget_digest:
        raise GuidedSearchError("budget_digest does not match the supplied budget")

    now = time.perf_counter if clock is None else clock
    started = now()
    active_verifier = verifier or EGraphFragmentEquivalenceVerifier()
    capability = active_verifier.capability()
    descriptor = scorer.descriptor()

    symbols = _symbols_for(task)
    rng = Random(seed)
    root_candidates = [
        SemanticCandidate(expression=symbol, srepr=sympy.srepr(symbol, order="none"), depth=0)
        for symbol in symbols
    ]
    frontier: list[tuple[float, SemanticCandidate, str | None]] = [
        (0.0, candidate, None) for candidate in root_candidates
    ]

    seen: set[str] = {candidate.key for candidate in root_candidates}
    rows: list[SRCandidate] = []
    expansions = 0
    generated = 0
    duplicates = 0
    invalid = 0
    verifier_calls = 0
    verifier_timeouts = 0
    verifier_errors = 0
    scorer_batches = 0
    peak_frontier = len(frontier)
    recovery_outcome = EquivalenceOutcome.UNSUPPORTED
    recovery_candidate_id: str | None = None
    termination = TerminationReason.FRONTIER_EXHAUSTED
    status = RunStatus.COMPLETE
    detail = ""

    try:
        while frontier:
            if now() - started >= budget.wall_seconds:
                termination = TerminationReason.WALL_TIME_BUDGET
                break
            if expansions >= budget.max_expansions:
                termination = TerminationReason.EXPANSION_BUDGET
                break
            if len(rows) >= budget.max_candidates:
                termination = TerminationReason.CANDIDATE_BUDGET
                break

            frontier.sort(key=lambda item: (-item[0], item[1].srepr))
            del frontier[budget.beam_width :]
            _score, parent, parent_id = frontier.pop(0)
            expansions += 1

            children = generate_moves(
                parent,
                symbols,
                rng,
                max_depth=budget.max_depth,
                fan_out=budget.beam_width,
            )
            requests: list[ScoringRequest] = []
            request_rows: list[SRCandidate] = []

            for child in children:
                generated += 1
                if child.key in seen:
                    duplicates += 1
                    continue
                seen.add(child.key)
                row, request = _evaluate_candidate(
                    child,
                    task=task,
                    method=method,
                    representation=representation,
                    seed=seed,
                    step=expansions,
                    parent_candidate_id=parent_id,
                    fit_observations=fit_observations,
                    budget=budget,
                    scorer_id=descriptor.scorer_id,
                )
                if row.status is not CandidateStatus.EVALUATED:
                    invalid += 1
                    rows.append(row)
                    continue
                request_rows.append(row)
                requests.append(request)

            if requests:
                scores = scorer.score_batch(requests)
                scorer_batches += 1
                if len(scores) != len(requests):
                    raise GuidedSearchError(
                        "scorer returned a different number of scores than requests"
                    )
            else:
                scores = ()

            for row, request, score in zip(request_rows, requests, scores, strict=True):
                scored_row = row.model_copy(update={"proposal_score": float(score)})
                equivalence = verify_exact_recovery(
                    active_verifier,
                    target_srepr=task.target_srepr,
                    candidate_srepr=scored_row.expression_srepr,
                    domain_mode=task.domain_mode,
                    used_operators=task.used_operators,
                )
                verifier_calls += 1
                if equivalence.outcome is EquivalenceOutcome.TIMEOUT:
                    verifier_timeouts += 1
                elif equivalence.outcome is EquivalenceOutcome.ERROR:
                    verifier_errors += 1
                scored_row = scored_row.model_copy(update={"equivalence": equivalence})
                rows.append(scored_row)
                frontier.append((float(score), request.candidate, scored_row.candidate_id))

                if equivalence.outcome is EquivalenceOutcome.VERIFIED:
                    recovery_outcome = EquivalenceOutcome.VERIFIED
                    recovery_candidate_id = scored_row.candidate_id
                    termination = TerminationReason.EXACT_RECOVERY
                    frontier.clear()
                    break

            peak_frontier = max(peak_frontier, len(frontier))
    except GuidedSearchError:
        raise
    except Exception as error:  # a search fault must stay a typed row
        status = RunStatus.FAILED
        termination = TerminationReason.ERROR
        detail = f"{type(error).__name__}: {error}"

    if recovery_outcome is not EquivalenceOutcome.VERIFIED:
        recovery_outcome = _summarise_outcomes(rows)

    best_row = _best_by_fit(rows)
    evaluation_rmse: float | None = None
    if best_row is not None and evaluation_observations is not None:
        held_out = evaluate_numeric_fit(best_row.expression_srepr, evaluation_observations)
        evaluation_rmse = held_out.root_mean_squared_error
        rows = [
            row.model_copy(update={"evaluation_fit": held_out})
            if row.candidate_id == best_row.candidate_id
            else row
            for row in rows
        ]

    telemetry = ResourceTelemetry(
        wall_seconds=max(0.0, now() - started),
        expansions=expansions,
        candidates_generated=generated,
        candidates_retained=len(rows),
        duplicates_skipped=duplicates,
        invalid_candidates=invalid,
        verifier_calls=verifier_calls,
        verifier_timeouts=verifier_timeouts,
        verifier_errors=verifier_errors,
        scorer_batches=scorer_batches,
        peak_frontier=peak_frontier,
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
        scorer_id=descriptor.scorer_id,
        verifier_id=capability.verifier_id,
        verifier_version=capability.verifier_version,
        telemetry=telemetry,
        candidates=tuple(rows),
        best_fit_candidate_id=best_row.candidate_id if best_row else None,
        best_evaluation_rmse=evaluation_rmse,
        exact_recovery_outcome=recovery_outcome,
        exact_recovery_candidate_id=recovery_candidate_id,
        pareto_candidate_ids=pareto_front(rows),
        detail=detail,
    )


def _evaluate_candidate(
    candidate: SemanticCandidate,
    *,
    task: SRTask,
    method: SRMethod,
    representation: SRRepresentation,
    seed: int,
    step: int,
    parent_candidate_id: str | None,
    fit_observations: ObservationSet,
    budget: SearchBudget,
    scorer_id: str,
) -> tuple[SRCandidate, ScoringRequest]:
    """Build one candidate row plus its scoring request, without ever raising."""

    candidate_id = derive_candidate_id(
        task_id=task.task_id, method=method, seed=seed, srepr=candidate.srepr, step=step
    )
    base = {
        "candidate_id": candidate_id,
        "task_id": task.task_id,
        "method": method,
        "representation": representation,
        "seed": seed,
        "expression_srepr": candidate.srepr,
        "expression_display": sympy.sstr(candidate.expression, order="none"),
        "structural_signature": None,
        "complexity": None,
        "representation_node_count": None,
        "fit": None,
        "evaluation_fit": None,
        "equivalence": None,
        "proposal_scorer_id": scorer_id,
        "proposal_step": step,
        "parent_candidate_id": parent_candidate_id,
    }
    materialized = MaterializedCandidate(
        representation=representation,
        node_count=0,
        depth=candidate.depth,
        signature="",
        status=CandidateStatus.MATERIALIZATION_FAILED,
    )

    check = check_grammar(candidate.srepr)
    if not check.in_grammar:
        row = SRCandidate(**base, status=CandidateStatus.OUT_OF_GRAMMAR, detail=check.detail)
        return row, ScoringRequest(
            task_id=task.task_id,
            representation=representation,
            candidate=candidate,
            materialized=materialized,
            fit=None,
        )
    if not set(check.used_operators).issubset(task.allowed_operators):
        row = SRCandidate(
            **base,
            status=CandidateStatus.OUT_OF_GRAMMAR,
            detail="candidate used an operator outside this task's frozen verifier fragment",
        )
        return row, ScoringRequest(
            task_id=task.task_id,
            representation=representation,
            candidate=candidate,
            materialized=materialized,
            fit=None,
        )

    materialized = materialize(candidate, representation)
    if materialized.status is not CandidateStatus.EVALUATED:
        row = SRCandidate(
            **base,
            status=CandidateStatus.MATERIALIZATION_FAILED,
            detail=materialized.signature,
        )
        return row, ScoringRequest(
            task_id=task.task_id,
            representation=representation,
            candidate=candidate,
            materialized=materialized,
            fit=None,
        )

    try:
        parsed = parse_srepr(candidate.srepr)
        tree = build_ast_from_parsed(parsed, expression_id="0" * 64)
        complexity = measure_complexity(tree)
        signature = structural_signature(tree)
    except (SreprParseError, ValueError) as error:
        row = SRCandidate(
            **base,
            status=CandidateStatus.MATERIALIZATION_FAILED,
            detail=f"{type(error).__name__}: {error}",
        )
        return row, ScoringRequest(
            task_id=task.task_id,
            representation=representation,
            candidate=candidate,
            materialized=materialized,
            fit=None,
        )

    base["structural_signature"] = signature
    base["complexity"] = complexity
    base["representation_node_count"] = materialized.node_count

    if complexity.ast_node_count > budget.max_complexity:
        return (
            SRCandidate(
                **base,
                status=CandidateStatus.COMPLEXITY_EXCEEDED,
                detail=f"{complexity.ast_node_count} > {budget.max_complexity}",
            ),
            ScoringRequest(
                task_id=task.task_id,
                representation=representation,
                candidate=candidate,
                materialized=materialized,
                fit=None,
            ),
        )
    if complexity.ast_depth > budget.max_depth:
        return (
            SRCandidate(
                **base,
                status=CandidateStatus.DEPTH_EXCEEDED,
                detail=f"{complexity.ast_depth} > {budget.max_depth}",
            ),
            ScoringRequest(
                task_id=task.task_id,
                representation=representation,
                candidate=candidate,
                materialized=materialized,
                fit=None,
            ),
        )

    fit = evaluate_numeric_fit(candidate.srepr, fit_observations)
    if fit.status in {NumericFitStatus.INVALID_DOMAIN, NumericFitStatus.NONFINITE}:
        return (
            SRCandidate(**base, fit=fit, status=CandidateStatus.DOMAIN_INVALID, detail=fit.detail),
            ScoringRequest(
                task_id=task.task_id,
                representation=representation,
                candidate=candidate,
                materialized=materialized,
                fit=fit,
            ),
        )
    if fit.status in {NumericFitStatus.PARSE_ERROR, NumericFitStatus.ERROR}:
        return (
            SRCandidate(
                **base, fit=fit, status=CandidateStatus.EVALUATION_ERROR, detail=fit.detail
            ),
            ScoringRequest(
                task_id=task.task_id,
                representation=representation,
                candidate=candidate,
                materialized=materialized,
                fit=fit,
            ),
        )

    base["fit"] = fit
    return (
        SRCandidate(**base, status=CandidateStatus.EVALUATED),
        ScoringRequest(
            task_id=task.task_id,
            representation=representation,
            candidate=candidate,
            materialized=materialized,
            fit=fit,
        ),
    )


def _best_by_fit(rows: Sequence[SRCandidate]) -> SRCandidate | None:
    scored = [
        row
        for row in rows
        if row.status is CandidateStatus.EVALUATED
        and row.fit is not None
        and row.fit.root_mean_squared_error is not None
    ]
    if not scored:
        return None
    return min(
        scored,
        key=lambda row: (
            row.fit.root_mean_squared_error,
            row.complexity.ast_node_count if row.complexity else 1 << 30,
            row.candidate_id,
        ),
    )


def _summarise_outcomes(rows: Sequence[SRCandidate]) -> EquivalenceOutcome:
    """Summarise per-candidate equivalence outcomes conservatively.

    ``verified`` requires an actual verified row. Otherwise the most informative observed
    outcome wins, and the absence of any verifier evidence stays ``unsupported``.
    """

    outcomes = {row.equivalence.outcome for row in rows if row.equivalence is not None}
    for outcome in (
        EquivalenceOutcome.VERIFIED,
        EquivalenceOutcome.UNKNOWN,
        EquivalenceOutcome.TIMEOUT,
        EquivalenceOutcome.ERROR,
        EquivalenceOutcome.NOT_EQUIVALENT,
    ):
        if outcome in outcomes:
            return outcome
    return EquivalenceOutcome.UNSUPPORTED
