"""Goal 9 analysis and Gate G9 (issue 9-3).

Every aggregate here rebuilds from per-task rows and carries explicit denominators. The
module is deliberately conservative in four ways:

1. **Exact recovery never comes from numbers.** A recovery is counted only when a row
   carries a ``verified`` equivalence outcome. Numeric error is reported separately and can
   never promote a candidate to a recovery.
2. **Denominators are attempted-first.** A method with many invalid, unsupported, or
   timed-out rows cannot look better by quietly using only its successes as the denominator.
   Both the attempted and the verifier-supported denominators are reported for every cell.
3. **Comparisons are paired and clustered.** Methods are contrasted task by task, and
   uncertainty comes from resampling tasks (the cluster), not candidate rows. Raw seed rows
   are always published. With three seeds no asymptotic significance claim is made.
4. **Nothing is invented.** Missing or fixture-only inputs produce an explicit incomplete
   report and ``insufficient_evidence``; they never produce plausible numerical content.
"""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from enum import StrEnum
from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from geml.data.sr.benchmark import (
    SR_MANIFEST_SCHEMA_VERSION,
    BenchmarkManifest,
    EquivalenceOutcome,
    ManifestStatus,
)
from geml.learning.sr.guided_search import (
    FIXTURE_SCORER_ID_PREFIX,
    SR_METHOD_RESULT_SCHEMA_VERSION,
    CandidateStatus,
    RunStatus,
    SRCandidate,
    SRMethod,
    SRMethodResult,
    load_results,
)

GOAL9_SUMMARY_SCHEMA_VERSION = "geml-goal9-summary-v1"
GATE_G9_SCHEMA_VERSION = "geml-gate-g9-v1"

#: The controlled comparison. LLM rows are external context and never enter this set.
CONTROLLED_METHODS: frozenset[SRMethod] = frozenset(
    {
        SRMethod.EML_GUIDED,
        SRMethod.AST_GUIDED,
        SRMethod.PYSR,
        SRMethod.GP_FALLBACK,
        SRMethod.TRANSFORMER_SR,
    }
)

#: The reference the gate is decided against.
GATE_REFERENCE_METHODS: tuple[SRMethod, ...] = (SRMethod.PYSR, SRMethod.GP_FALLBACK)

#: The representation arms whose contrast Goal 9 exists to measure.
GATE_TREATMENT_METHOD = SRMethod.EML_GUIDED
GATE_CONTROL_METHOD = SRMethod.AST_GUIDED

_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Goal9AnalysisError(ValueError):
    """A manifest, result row, or aggregation input was invalid."""


class GateState(StrEnum):
    """The three predeclared Gate G9 verdicts."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ReportCompleteness(StrEnum):
    """Whether the report is built from a complete frozen production population."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FIXTURE_ONLY = "fixture_only"


class ExactRatio(BaseModel):
    """An exact rational rate with both of its denominators visible."""

    model_config = _FROZEN

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)

    @classmethod
    def of(cls, numerator: int, denominator: int) -> "ExactRatio":
        """Return a ratio, tolerating a zero denominator rather than raising."""

        return cls(numerator=numerator, denominator=denominator)

    @property
    def value(self) -> float | None:
        """Return the rate, or ``None`` when the denominator is zero."""

        if self.denominator == 0:
            return None
        return float(Fraction(self.numerator, self.denominator))


class ValidationIssue(BaseModel):
    """One reason a report cannot be treated as complete."""

    model_config = _FROZEN

    code: str
    detail: str


class MethodSummary(BaseModel):
    """Aggregates for one (task set, method) cell, rebuilt from per-task rows."""

    model_config = _FROZEN

    task_set: str
    method: SRMethod
    seeds: tuple[int, ...]
    cells_attempted: int = Field(ge=0)
    cells_complete: int = Field(ge=0)
    cells_failed: int = Field(ge=0)
    cells_timeout: int = Field(ge=0)
    cells_dependency_unavailable: int = Field(ge=0)
    cells_unsupported: int = Field(ge=0)

    exact_recovery_attempted: ExactRatio
    exact_recovery_verifier_supported: ExactRatio
    outcome_counts: dict[str, int]

    numeric_fit_reported: int = Field(ge=0)
    median_fit_rmse: float | None = None
    median_evaluation_rmse: float | None = None

    median_best_complexity: float | None = None
    median_wall_seconds: float | None = None
    median_expansions: float | None = None
    total_candidates: int = Field(ge=0)
    invalid_candidate_rate: ExactRatio

    budget_digests: tuple[str, ...]
    native_budgets: tuple[str, ...]
    budget_mismatches: tuple[str, ...]
    backends: tuple[str, ...]


class PairedContrast(BaseModel):
    """A paired, task-level contrast between two methods."""

    model_config = _FROZEN

    task_set: str
    treatment: SRMethod
    control: SRMethod
    metric: str
    paired_tasks: int = Field(ge=0)
    treatment_better: int = Field(ge=0)
    control_better: int = Field(ge=0)
    ties: int = Field(ge=0)
    mean_difference: float | None = None
    bootstrap_low: float | None = None
    bootstrap_high: float | None = None
    resamples: int = Field(ge=0)
    note: str = ""


class ParetoPoint(BaseModel):
    """One point on the accuracy-complexity front for one method."""

    model_config = _FROZEN

    method: SRMethod
    complexity: int = Field(ge=0)
    best_rmse: float
    task_id: str
    candidate_id: str


class SeedRow(BaseModel):
    """A raw per-seed row. Always published, never collapsed into a mean alone."""

    model_config = _FROZEN

    task_set: str
    method: SRMethod
    seed: int
    tasks: int = Field(ge=0)
    verified_recoveries: int = Field(ge=0)
    median_fit_rmse: float | None = None
    median_wall_seconds: float | None = None


class GateG9(BaseModel):
    """The predeclared Gate G9 verdict."""

    model_config = _FROZEN

    schema_version: str = GATE_G9_SCHEMA_VERSION
    state: GateState
    rationale: str
    verifier_coverage_known: bool
    verifier_supported_tasks: int = Field(ge=0)
    benchmark_tasks: int = Field(ge=0)
    treatment: SRMethod = GATE_TREATMENT_METHOD
    control: SRMethod = GATE_CONTROL_METHOD
    reference_methods: tuple[SRMethod, ...] = GATE_REFERENCE_METHODS
    treatment_verified_recoveries: int = Field(default=0, ge=0)
    reference_verified_recoveries: int = Field(default=0, ge=0)
    external_llm_rows_considered: bool = False
    caveats: tuple[str, ...] = ()


class Goal9Summary(BaseModel):
    """The complete Goal 9 report."""

    model_config = _FROZEN

    schema_version: str = GOAL9_SUMMARY_SCHEMA_VERSION
    completeness: ReportCompleteness
    validation_issues: tuple[ValidationIssue, ...]
    manifest_ids: tuple[str, ...]
    manifest_statuses: tuple[str, ...]
    task_sets: tuple[str, ...]
    methods: tuple[SRMethod, ...]
    seeds: tuple[int, ...]
    method_summaries: tuple[MethodSummary, ...]
    seed_rows: tuple[SeedRow, ...]
    contrasts: tuple[PairedContrast, ...]
    pareto_points: tuple[ParetoPoint, ...]
    budget_mismatch_notes: tuple[str, ...]
    external_reference_rows: int = Field(default=0, ge=0)
    gate: GateG9
    rows_digest: str
    bounded_benchmark_caveats: tuple[str, ...] = (
        "The benchmark is fixed-scale: 256 synthetic tasks and a frozen restricted "
        "Feynman-style subset. No conclusion extends to corpus-size scaling or to general "
        "symbolic-regression superiority beyond this benchmark.",
        "Three seeds are enough to publish raw variation and not enough for an asymptotic "
        "significance claim.",
        "Exact-recovery coverage is bounded by the assigned verifier's declared capability; "
        "tasks outside it are reported as unsupported, not as failures to recover.",
    )


# --------------------------------------------------------------------------------------
# Manifest and row validation
# --------------------------------------------------------------------------------------


def is_fixture_row(result: SRMethodResult) -> bool:
    """Return whether a row was produced by a fixture scorer rather than a trained model."""

    if result.scorer_id.startswith(FIXTURE_SCORER_ID_PREFIX):
        return True
    return any(
        candidate.proposal_scorer_id.startswith(FIXTURE_SCORER_ID_PREFIX)
        for candidate in result.candidates
    )


def validate_inputs(
    manifests: Sequence[BenchmarkManifest], results: Sequence[SRMethodResult]
) -> tuple[ValidationIssue, ...]:
    """Validate manifests and result rows before any aggregation happens.

    Fixture-scorer rows are detected here, from the rows themselves: the fixture-only rule
    never rests on a CLI flag alone.
    """

    issues: list[ValidationIssue] = []
    if not manifests:
        issues.append(
            ValidationIssue(code="no_manifest", detail="no benchmark manifest was supplied")
        )
    if not results:
        issues.append(
            ValidationIssue(code="no_results", detail="no method result rows were supplied")
        )

    known_task_ids: set[str] = set()
    for manifest in manifests:
        if manifest.schema_version != SR_MANIFEST_SCHEMA_VERSION:
            issues.append(
                ValidationIssue(
                    code="manifest_schema_version",
                    detail=(
                        f"manifest {manifest.benchmark_id[:12]} has schema "
                        f"{manifest.schema_version!r}"
                    ),
                )
            )
        if manifest.status is not ManifestStatus.COMPLETE:
            issues.append(
                ValidationIssue(
                    code="manifest_not_frozen",
                    detail=(
                        f"manifest {manifest.benchmark_id[:12]} status is "
                        f"{manifest.status.value}: {manifest.status_detail}"
                    ),
                )
            )
        known_task_ids.update(manifest.task_ids)

    seen_budget_digests: set[str] = set()
    for result in results:
        if result.schema_version != SR_METHOD_RESULT_SCHEMA_VERSION:
            issues.append(
                ValidationIssue(
                    code="result_schema_version",
                    detail=f"row for {result.task_id[:12]} has {result.schema_version!r}",
                )
            )
        if known_task_ids and result.task_id not in known_task_ids:
            issues.append(
                ValidationIssue(
                    code="task_not_in_manifest",
                    detail=f"result task {result.task_id[:12]} is not in any manifest",
                )
            )
        if result.method in {GATE_TREATMENT_METHOD, GATE_CONTROL_METHOD}:
            seen_budget_digests.add(result.budget_digest)

    fixture_rows = [result for result in results if is_fixture_row(result)]
    if fixture_rows:
        first = fixture_rows[0]
        example = first.scorer_id or next(
            candidate.proposal_scorer_id
            for candidate in first.candidates
            if candidate.proposal_scorer_id.startswith(FIXTURE_SCORER_ID_PREFIX)
        )
        issues.append(
            ValidationIssue(
                code="fixture_scorer_rows",
                detail=(
                    f"{len(fixture_rows)} of {len(results)} rows carry a fixture scorer id "
                    f"(for example {example!r}); fixture rows can only feed a fixture-only "
                    "report, never a production verdict"
                ),
            )
        )

    if len(seen_budget_digests) > 1:
        issues.append(
            ValidationIssue(
                code="unmatched_budgets",
                detail=(
                    "the two controlled representation arms did not share one budget "
                    f"digest: {sorted(digest[:12] for digest in seen_budget_digests)}"
                ),
            )
        )
    return tuple(issues)


def rows_digest(results: Sequence[SRMethodResult]) -> str:
    """Return a stable digest over the exact result rows an aggregate was built from."""

    hasher = hashlib.sha256()
    hasher.update(b"geml-goal9-rows-v1\0")
    for result in sorted(results, key=lambda item: (item.task_id, item.method.value, item.seed)):
        payload = json.dumps(
            result.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        hasher.update(f"{len(payload)}:".encode("ascii"))
        hasher.update(payload)
    return hasher.hexdigest()


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def _median(values: Iterable[float]) -> float | None:
    collected = [value for value in values if value is not None and math.isfinite(value)]
    if not collected:
        return None
    return float(statistics.median(collected))


def _best_candidate(result: SRMethodResult) -> SRCandidate | None:
    if result.best_fit_candidate_id is None:
        return None
    return next(
        (
            candidate
            for candidate in result.candidates
            if candidate.candidate_id == result.best_fit_candidate_id
        ),
        None,
    )


def summarize_method(
    task_set: str, method: SRMethod, results: Sequence[SRMethodResult]
) -> MethodSummary:
    """Aggregate one (task set, method) cell from its raw rows."""

    cells = [
        result for result in results if result.task_set == task_set and result.method is method
    ]
    outcome_counts: dict[str, int] = defaultdict(int)
    verified = 0
    supported_denominator = 0
    fit_values: list[float] = []
    evaluation_values: list[float] = []
    complexities: list[float] = []
    wall: list[float] = []
    expansions: list[float] = []
    total_candidates = 0
    invalid_candidates = 0

    for result in cells:
        outcome_counts[result.exact_recovery_outcome.value] += 1
        if result.exact_recovery_outcome is EquivalenceOutcome.VERIFIED:
            verified += 1
        if result.exact_recovery_outcome is not EquivalenceOutcome.UNSUPPORTED:
            supported_denominator += 1
        best = _best_candidate(result)
        if best is not None and best.fit is not None:
            if best.fit.root_mean_squared_error is not None:
                fit_values.append(best.fit.root_mean_squared_error)
            if best.complexity is not None:
                complexities.append(float(best.complexity.ast_node_count))
        if result.best_evaluation_rmse is not None:
            evaluation_values.append(result.best_evaluation_rmse)
        wall.append(result.telemetry.wall_seconds)
        expansions.append(float(result.telemetry.expansions))
        total_candidates += len(result.candidates)
        invalid_candidates += sum(
            1
            for candidate in result.candidates
            if candidate.status is not CandidateStatus.EVALUATED
        )

    return MethodSummary(
        task_set=task_set,
        method=method,
        seeds=tuple(sorted({result.seed for result in cells})),
        cells_attempted=len(cells),
        cells_complete=sum(1 for r in cells if r.status is RunStatus.COMPLETE),
        cells_failed=sum(1 for r in cells if r.status is RunStatus.FAILED),
        cells_timeout=sum(1 for r in cells if r.status is RunStatus.TIMEOUT),
        cells_dependency_unavailable=sum(
            1 for r in cells if r.status is RunStatus.DEPENDENCY_UNAVAILABLE
        ),
        cells_unsupported=sum(1 for r in cells if r.status is RunStatus.UNSUPPORTED),
        exact_recovery_attempted=ExactRatio.of(verified, len(cells)),
        exact_recovery_verifier_supported=ExactRatio.of(verified, supported_denominator),
        outcome_counts=dict(sorted(outcome_counts.items())),
        numeric_fit_reported=len(fit_values),
        median_fit_rmse=_median(fit_values),
        median_evaluation_rmse=_median(evaluation_values),
        median_best_complexity=_median(complexities),
        median_wall_seconds=_median(wall),
        median_expansions=_median(expansions),
        total_candidates=total_candidates,
        invalid_candidate_rate=ExactRatio.of(invalid_candidates, total_candidates),
        budget_digests=tuple(sorted({result.budget_digest for result in cells})),
        native_budgets=tuple(
            sorted(
                {
                    json.dumps(result.native_budget, sort_keys=True)
                    for result in cells
                    if result.native_budget
                }
            )
        ),
        budget_mismatches=tuple(
            sorted({result.budget_mismatch for result in cells if result.budget_mismatch})
        ),
        backends=tuple(sorted({result.backend_name for result in cells if result.backend_name})),
    )


def seed_rows(results: Sequence[SRMethodResult]) -> tuple[SeedRow, ...]:
    """Return the raw per-(task set, method, seed) rows."""

    grouped: dict[tuple[str, SRMethod, int], list[SRMethodResult]] = defaultdict(list)
    for result in results:
        grouped[(result.task_set, result.method, result.seed)].append(result)

    rows: list[SeedRow] = []
    for (task_set, method, seed), cells in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1].value, item[0][2])
    ):
        errors = [
            candidate.fit.root_mean_squared_error
            for result in cells
            for candidate in [_best_candidate(result)]
            if candidate is not None
            and candidate.fit is not None
            and candidate.fit.root_mean_squared_error is not None
        ]
        rows.append(
            SeedRow(
                task_set=task_set,
                method=method,
                seed=seed,
                tasks=len({result.task_id for result in cells}),
                verified_recoveries=sum(
                    1
                    for result in cells
                    if result.exact_recovery_outcome is EquivalenceOutcome.VERIFIED
                ),
                median_fit_rmse=_median(errors),
                median_wall_seconds=_median([result.telemetry.wall_seconds for result in cells]),
            )
        )
    return tuple(rows)


def _task_level_metric(
    results: Sequence[SRMethodResult], task_set: str, method: SRMethod, metric: str
) -> dict[str, float]:
    """Reduce a method to one number per task by taking the median across seeds."""

    per_task: dict[str, list[float]] = defaultdict(list)
    for result in results:
        if result.task_set != task_set or result.method is not method:
            continue
        if metric == "verified_recovery":
            per_task[result.task_id].append(
                1.0 if result.exact_recovery_outcome is EquivalenceOutcome.VERIFIED else 0.0
            )
        elif metric == "fit_rmse":
            best = _best_candidate(result)
            if (
                best is not None
                and best.fit is not None
                and best.fit.root_mean_squared_error is not None
            ):
                per_task[result.task_id].append(best.fit.root_mean_squared_error)
        elif metric == "wall_seconds":
            per_task[result.task_id].append(result.telemetry.wall_seconds)
        else:  # pragma: no cover - guarded by callers
            raise Goal9AnalysisError(f"unknown metric {metric!r}")
    return {
        task_id: float(statistics.median(values)) for task_id, values in per_task.items() if values
    }


def paired_contrast(
    results: Sequence[SRMethodResult],
    *,
    task_set: str,
    treatment: SRMethod,
    control: SRMethod,
    metric: str = "fit_rmse",
    resamples: int = 2000,
    seed: int = 20260726,
) -> PairedContrast:
    """Contrast two methods task by task with a cluster (task-level) bootstrap.

    Only tasks both methods attempted enter the contrast, and the resampling unit is the
    task, never the candidate row. Lower is better for ``fit_rmse`` and ``wall_seconds``;
    higher is better for ``verified_recovery``.
    """

    treatment_values = _task_level_metric(results, task_set, treatment, metric)
    control_values = _task_level_metric(results, task_set, control, metric)
    shared = sorted(set(treatment_values) & set(control_values))
    lower_is_better = metric in {"fit_rmse", "wall_seconds"}

    if not shared:
        return PairedContrast(
            task_set=task_set,
            treatment=treatment,
            control=control,
            metric=metric,
            paired_tasks=0,
            treatment_better=0,
            control_better=0,
            ties=0,
            resamples=0,
            note="no task was attempted by both methods",
        )

    differences = [treatment_values[task] - control_values[task] for task in shared]
    if lower_is_better:
        treatment_better = sum(1 for difference in differences if difference < 0)
        control_better = sum(1 for difference in differences if difference > 0)
    else:
        treatment_better = sum(1 for difference in differences if difference > 0)
        control_better = sum(1 for difference in differences if difference < 0)
    ties = len(differences) - treatment_better - control_better

    low: float | None = None
    high: float | None = None
    if len(differences) >= 2 and resamples > 0:
        from random import Random

        rng = Random(seed)
        means: list[float] = []
        count = len(differences)
        for _ in range(resamples):
            sample = [differences[rng.randrange(count)] for _ in range(count)]
            means.append(sum(sample) / count)
        means.sort()
        low = means[int(0.025 * (len(means) - 1))]
        high = means[int(0.975 * (len(means) - 1))]

    return PairedContrast(
        task_set=task_set,
        treatment=treatment,
        control=control,
        metric=metric,
        paired_tasks=len(shared),
        treatment_better=treatment_better,
        control_better=control_better,
        ties=ties,
        mean_difference=sum(differences) / len(differences),
        bootstrap_low=low,
        bootstrap_high=high,
        resamples=resamples if low is not None else 0,
        note=(
            "paired task-level contrast with a task-clustered bootstrap; with three seeds "
            "this interval describes task variation, not an asymptotic significance test"
        ),
    )


def pareto_points(results: Sequence[SRMethodResult]) -> tuple[ParetoPoint, ...]:
    """Return the accuracy-complexity front across methods, one point per front member."""

    points: list[ParetoPoint] = []
    for result in results:
        by_id = {candidate.candidate_id: candidate for candidate in result.candidates}
        for candidate_id in result.pareto_candidate_ids:
            candidate = by_id.get(candidate_id)
            if candidate is None or candidate.complexity is None or candidate.fit is None:
                continue
            if candidate.fit.root_mean_squared_error is None:
                continue
            points.append(
                ParetoPoint(
                    method=result.method,
                    complexity=candidate.complexity.ast_node_count,
                    best_rmse=candidate.fit.root_mean_squared_error,
                    task_id=result.task_id,
                    candidate_id=candidate_id,
                )
            )
    points.sort(key=lambda point: (point.method.value, point.complexity, point.best_rmse))
    return tuple(points)


# --------------------------------------------------------------------------------------
# Gate G9
# --------------------------------------------------------------------------------------


def decide_gate(
    *,
    manifests: Sequence[BenchmarkManifest],
    summaries: Sequence[MethodSummary],
    issues: Sequence[ValidationIssue],
    completeness: ReportCompleteness,
) -> GateG9:
    """Decide Gate G9 against the predeclared rule.

    The gate compares verifier-confirmed exact recovery of the EML-guided arm against the
    pinned PySR (or explicitly labelled GP) reference under the matched budget, or records
    an explicit negative result. It **cannot pass** while exact-verification coverage is
    unknown, and LLM rows never enter the decision.
    """

    caveats: list[str] = []
    benchmark_tasks = sum(manifest.task_count for manifest in manifests)
    verifier_supported = sum(manifest.verifier_supported_count for manifest in manifests)

    coverage_known = bool(manifests) and all(
        manifest.status is ManifestStatus.COMPLETE for manifest in manifests
    )
    if not coverage_known:
        caveats.append(
            "at least one benchmark manifest is not frozen, so exact-verification coverage "
            "is unknown"
        )

    treatment = next((s for s in summaries if s.method is GATE_TREATMENT_METHOD), None)
    references = [s for s in summaries if s.method in GATE_REFERENCE_METHODS]
    treatment_recoveries = treatment.exact_recovery_attempted.numerator if treatment else 0
    reference_recoveries = sum(s.exact_recovery_attempted.numerator for s in references)

    blocking = [issue for issue in issues if issue.code != "task_not_in_manifest"]
    if completeness is not ReportCompleteness.COMPLETE or blocking or not coverage_known:
        return GateG9(
            state=GateState.INSUFFICIENT_EVIDENCE,
            rationale=(
                "Gate G9 cannot be decided: "
                + (
                    "; ".join(issue.code for issue in blocking)
                    if blocking
                    else f"report completeness is {completeness.value}"
                )
                + ". A gate cannot pass while exact verification coverage is unknown."
            ),
            verifier_coverage_known=coverage_known,
            verifier_supported_tasks=verifier_supported,
            benchmark_tasks=benchmark_tasks,
            treatment_verified_recoveries=treatment_recoveries,
            reference_verified_recoveries=reference_recoveries,
            caveats=tuple(caveats),
        )

    if treatment is None or not references:
        return GateG9(
            state=GateState.INSUFFICIENT_EVIDENCE,
            rationale=(
                "the intended comparison requires both the EML-guided arm and a pinned "
                "PySR or explicitly labelled GP reference; one of them produced no rows"
            ),
            verifier_coverage_known=coverage_known,
            verifier_supported_tasks=verifier_supported,
            benchmark_tasks=benchmark_tasks,
            treatment_verified_recoveries=treatment_recoveries,
            reference_verified_recoveries=reference_recoveries,
            caveats=tuple(caveats),
        )

    if treatment_recoveries > reference_recoveries:
        state = GateState.PASS
        rationale = (
            f"under the matched budget the EML-guided arm achieved "
            f"{treatment_recoveries} verifier-confirmed exact recoveries against "
            f"{reference_recoveries} for the pinned reference"
        )
    else:
        state = GateState.FAIL
        rationale = (
            f"explicit negative result: the EML-guided arm achieved "
            f"{treatment_recoveries} verifier-confirmed exact recoveries against "
            f"{reference_recoveries} for the pinned reference under the matched budget"
        )
    return GateG9(
        state=state,
        rationale=rationale,
        verifier_coverage_known=coverage_known,
        verifier_supported_tasks=verifier_supported,
        benchmark_tasks=benchmark_tasks,
        treatment_verified_recoveries=treatment_recoveries,
        reference_verified_recoveries=reference_recoveries,
        caveats=tuple(caveats),
    )


# --------------------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------------------


def summarize(
    *,
    manifests: Sequence[BenchmarkManifest],
    results: Sequence[SRMethodResult],
    external_reference_rows: int = 0,
    fixture_only: bool = False,
) -> Goal9Summary:
    """Build the complete Goal 9 report, or an explicit incomplete one.

    ``external_reference_rows`` counts issue 11-3 LLM rows that were ingested purely as
    external context. They are counted and labelled; they never enter a summary, a contrast,
    or the gate.

    ``fixture_only`` can only widen the fixture-only classification: rows produced by a
    fixture scorer force it regardless of the flag, so forgetting the flag can never turn a
    fixture sweep into a production report.
    """

    issues = validate_inputs(manifests, results)
    controlled = [result for result in results if result.method in CONTROLLED_METHODS]
    task_sets = tuple(sorted({result.task_set for result in controlled}))
    methods = tuple(sorted({result.method for result in controlled}, key=lambda item: item.value))

    if fixture_only or any(issue.code == "fixture_scorer_rows" for issue in issues):
        completeness = ReportCompleteness.FIXTURE_ONLY
    elif issues:
        completeness = ReportCompleteness.INCOMPLETE
    else:
        completeness = ReportCompleteness.COMPLETE

    summaries = tuple(
        summarize_method(task_set, method, controlled)
        for task_set in task_sets
        for method in methods
        if any(result.task_set == task_set and result.method is method for result in controlled)
    )

    contrasts: list[PairedContrast] = []
    for task_set in task_sets:
        available = {summary.method for summary in summaries if summary.task_set == task_set}
        if {GATE_TREATMENT_METHOD, GATE_CONTROL_METHOD} <= available:
            for metric in ("fit_rmse", "verified_recovery", "wall_seconds"):
                contrasts.append(
                    paired_contrast(
                        controlled,
                        task_set=task_set,
                        treatment=GATE_TREATMENT_METHOD,
                        control=GATE_CONTROL_METHOD,
                        metric=metric,
                    )
                )
        for reference in GATE_REFERENCE_METHODS:
            if {GATE_TREATMENT_METHOD, reference} <= available:
                contrasts.append(
                    paired_contrast(
                        controlled,
                        task_set=task_set,
                        treatment=GATE_TREATMENT_METHOD,
                        control=reference,
                        metric="fit_rmse",
                    )
                )

    gate = decide_gate(
        manifests=manifests,
        summaries=summaries,
        issues=issues,
        completeness=completeness,
    )
    return Goal9Summary(
        completeness=completeness,
        validation_issues=issues,
        manifest_ids=tuple(manifest.benchmark_id for manifest in manifests),
        manifest_statuses=tuple(manifest.status.value for manifest in manifests),
        task_sets=task_sets,
        methods=methods,
        seeds=tuple(sorted({result.seed for result in controlled})),
        method_summaries=summaries,
        seed_rows=seed_rows(controlled),
        contrasts=tuple(contrasts),
        pareto_points=pareto_points(controlled),
        budget_mismatch_notes=tuple(
            sorted({note for summary in summaries for note in summary.budget_mismatches})
        ),
        external_reference_rows=external_reference_rows,
        gate=gate,
        rows_digest=rows_digest(controlled),
    )


def render_markdown(summary: Goal9Summary) -> str:
    """Render the report as Markdown, refusing to invent content when it is incomplete."""

    lines: list[str] = ["# Goal 9 results", ""]
    lines.append(f"- Report completeness: `{summary.completeness.value}`")
    lines.append(f"- Rows digest: `{summary.rows_digest}`")
    lines.append(f"- Gate G9: **`{summary.gate.state.value}`**")
    lines.append("")

    if summary.completeness is not ReportCompleteness.COMPLETE:
        lines.append("## Incomplete report")
        lines.append("")
        lines.append(
            "This report was not built from a complete frozen production population, so no "
            "result tables are published. The blocking issues are:"
        )
        lines.append("")
        if summary.validation_issues:
            lines.extend(
                f"- `{issue.code}` — {issue.detail}" for issue in summary.validation_issues
            )
        else:
            lines.append("- the report was explicitly built from fixture rows only")
        lines.append("")
        lines.append(f"Gate G9 rationale: {summary.gate.rationale}")
        lines.append("")
        return "\n".join(lines)

    lines.append("## Method summaries")
    lines.append("")
    lines.append(
        "| Task set | Method | Backend | Cells | Verified/attempted | "
        "Verified/verifier-supported | Median fit RMSE | Median complexity | Median wall (s) |"
    )
    lines.append("|---|---|---|---:|---|---|---:|---:|---:|")
    for item in summary.method_summaries:
        lines.append(
            f"| {item.task_set} | `{item.method.value}` | "
            f"{', '.join(item.backends) or '-'} | {item.cells_attempted} | "
            f"{item.exact_recovery_attempted.numerator}/"
            f"{item.exact_recovery_attempted.denominator} | "
            f"{item.exact_recovery_verifier_supported.numerator}/"
            f"{item.exact_recovery_verifier_supported.denominator} | "
            f"{_fmt(item.median_fit_rmse)} | {_fmt(item.median_best_complexity)} | "
            f"{_fmt(item.median_wall_seconds)} |"
        )
    lines.append("")

    lines.append("## Paired task-level contrasts")
    lines.append("")
    lines.append(
        "| Task set | Treatment | Control | Metric | Paired tasks | Treatment better | "
        "Control better | Ties | Mean difference | 95% task bootstrap |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|")
    for contrast in summary.contrasts:
        interval = (
            f"[{_fmt(contrast.bootstrap_low)}, {_fmt(contrast.bootstrap_high)}]"
            if contrast.bootstrap_low is not None
            else "-"
        )
        lines.append(
            f"| {contrast.task_set} | `{contrast.treatment.value}` | "
            f"`{contrast.control.value}` | {contrast.metric} | {contrast.paired_tasks} | "
            f"{contrast.treatment_better} | {contrast.control_better} | {contrast.ties} | "
            f"{_fmt(contrast.mean_difference)} | {interval} |"
        )
    lines.append("")

    lines.append("## Raw seed rows")
    lines.append("")
    lines.append("| Task set | Method | Seed | Tasks | Verified recoveries | Median fit RMSE |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in summary.seed_rows:
        lines.append(
            f"| {row.task_set} | `{row.method.value}` | {row.seed} | {row.tasks} | "
            f"{row.verified_recoveries} | {_fmt(row.median_fit_rmse)} |"
        )
    lines.append("")

    if summary.budget_mismatch_notes:
        lines.append("## Budget mismatches")
        lines.append("")
        lines.extend(f"- {note}" for note in summary.budget_mismatch_notes)
        lines.append("")

    lines.append("## Gate G9")
    lines.append("")
    lines.append(f"- State: **`{summary.gate.state.value}`**")
    lines.append(f"- Rationale: {summary.gate.rationale}")
    lines.append(
        f"- Verifier coverage known: `{str(summary.gate.verifier_coverage_known).lower()}`"
    )
    lines.append(
        f"- Verifier-supported tasks: {summary.gate.verifier_supported_tasks} of "
        f"{summary.gate.benchmark_tasks}"
    )
    lines.append(
        f"- External LLM rows ingested as context: {summary.external_reference_rows} "
        "(excluded from the gate)"
    )
    lines.append("")
    lines.append("## Bounded-benchmark caveats")
    lines.append("")
    lines.extend(f"- {caveat}" for caveat in summary.bounded_benchmark_caveats)
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def load_manifests(paths: Sequence[str | Path]) -> tuple[BenchmarkManifest, ...]:
    """Load benchmark manifests from disk."""

    return tuple(
        BenchmarkManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
        for path in paths
    )


def count_external_reference_rows(path: str | Path | None) -> int:
    """Count issue 11-3 SR reference rows without letting them influence any aggregate."""

    if path is None:
        return 0
    target = Path(path)
    if not target.exists():
        return 0
    return sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip())


def write_analysis_artifacts(summary: Goal9Summary, output_dir: str | Path) -> tuple[Path, ...]:
    """Write the JSON summary, the Markdown report, and the gate record."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "goal9.summary.json"
    markdown_path = root / "GOAL9_SUMMARY.md"
    gate_path = root / "GATE_G9.json"

    summary_path.write_text(
        json.dumps(summary.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    gate_path.write_text(
        json.dumps(summary.gate.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return (summary_path, markdown_path, gate_path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI shell
    """Rebuild the Goal 9 tables, plots, and gate from frozen manifests and rows."""

    parser = argparse.ArgumentParser(description="Analyse Goal 9 results")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--results-root", action="append", required=True)
    parser.add_argument("--llm-reference-rows", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fixture-only", action="store_true")
    arguments = parser.parse_args(argv)

    manifests = load_manifests(arguments.manifest)
    results: list[SRMethodResult] = []
    for root in arguments.results_root:
        results.extend(load_results(root))

    summary = summarize(
        manifests=manifests,
        results=results,
        external_reference_rows=count_external_reference_rows(arguments.llm_reference_rows),
        fixture_only=arguments.fixture_only,
    )
    write_analysis_artifacts(summary, arguments.output_dir)
    print(
        f"completeness={summary.completeness.value} gate={summary.gate.state.value} "
        f"rows={len(results)} issues={len(summary.validation_issues)}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module executable entry point
    raise SystemExit(main())
