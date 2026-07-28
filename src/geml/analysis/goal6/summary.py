"""Issue 6-6: denominator-complete Goal 6 analysis and the Gate G6 state machine.

What this module refuses to do
------------------------------
* emit a scientific verdict from fixture or incomplete rows;
* represent a missing seed row as zero;
* aggregate metrics whose denominators cannot be reconstructed;
* pool structural metrics from different representations into one "alpha" column;
* treat correlated pair rows or three seed observations as independent samples;
* write a plausible-looking numerical placeholder in place of a missing value.

Every one of those is a way to make an unfinished experiment look finished, so each is an explicit
error or an explicit machine-readable missing state rather than a silent default.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

GOAL6_ANALYSIS_SCHEMA_VERSION = "geml-goal6-analysis-v1"
GATE_G6_SCHEMA_VERSION = "geml-goal6-gate-v1"

#: The fixed-scale caveat that must accompany every Goal 6 claim.
FIXED_SCALE_CAVEATS: tuple[str, ...] = (
    "all results are at the fixed 50,000 training-pair scale over the frozen 250k-v1 corpus",
    "no claim extrapolates beyond this scale; no scaling law is fitted",
    "three seeds support raw-seed reporting and effect sizes, not strong asymptotic significance",
    "structural metrics from different representations are not mutually comparable",
)


class AnalysisError(ValueError):
    """The result rows cannot support the requested aggregate."""


class GateState(StrEnum):
    """The only three Gate G6 outcomes."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class MissingReason(StrEnum):
    """Machine-readable reasons a value is absent.  Never replaced by a number."""

    CELL_MISSING = "cell_missing"
    CELL_FAILED = "cell_failed"
    VIEW_UNSUPPORTED = "view_unsupported"
    METRIC_NOT_REPORTED = "metric_not_reported"
    DENOMINATOR_UNRECONSTRUCTABLE = "denominator_unreconstructable"


REQUIRED_DENOMINATOR_KEYS: frozenset[str] = frozenset(
    {"attempted", "valid", "failed", "unsupported", "timed_out"}
)


@dataclass(frozen=True)
class MissingValue:
    """An explicit absence carrying its reason."""

    reason: MissingReason
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"missing": True, "reason": str(self.reason), "detail": self.detail}


@dataclass(frozen=True)
class SeedAggregate:
    """Three-seed aggregation that always keeps the raw rows."""

    metric: str
    view: str
    arm_id: str
    raw_by_seed: dict[int, float]
    expected_seeds: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return set(self.raw_by_seed) == set(self.expected_seeds)

    @property
    def values(self) -> list[float]:
        return [self.raw_by_seed[seed] for seed in sorted(self.raw_by_seed)]

    @property
    def mean(self) -> float | MissingValue:
        if not self.complete:
            return MissingValue(
                MissingReason.CELL_MISSING,
                f"have seeds {sorted(self.raw_by_seed)}, expected {list(self.expected_seeds)}",
            )
        return sum(self.values) / len(self.values)

    @property
    def sample_standard_deviation(self) -> float | MissingValue:
        if not self.complete:
            return MissingValue(MissingReason.CELL_MISSING, "incomplete seed set")
        values = self.values
        if len(values) < 2:
            return MissingValue(MissingReason.METRIC_NOT_REPORTED, "fewer than two seeds")
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))

    @property
    def spread(self) -> float | MissingValue:
        """Min-to-max range: with three seeds this is more honest than a confidence interval."""

        if not self.complete:
            return MissingValue(MissingReason.CELL_MISSING, "incomplete seed set")
        return max(self.values) - min(self.values)

    def as_dict(self) -> dict[str, object]:
        def render(value: float | MissingValue) -> object:
            return value.as_dict() if isinstance(value, MissingValue) else value

        return {
            "arm_id": self.arm_id,
            "view": self.view,
            "metric": self.metric,
            "raw_by_seed": {str(seed): value for seed, value in sorted(self.raw_by_seed.items())},
            "complete": self.complete,
            "mean": render(self.mean),
            "sample_standard_deviation": render(self.sample_standard_deviation),
            "spread_min_to_max": render(self.spread),
            "seed_count": len(self.raw_by_seed),
            "expected_seed_count": len(self.expected_seeds),
        }


def validate_rows(
    rows: Sequence[Mapping[str, object]],
    expected_seeds: Sequence[int],
    *,
    require_single_commit: bool = True,
) -> None:
    """Fail closed on every manifest inconsistency issue 6-6 enumerates."""

    if not rows:
        raise AnalysisError("no result rows were supplied")

    identities = Counter((str(row["arm_id"]), int(row["seed"])) for row in rows)
    duplicates = sorted(identity for identity, count in identities.items() if count > 1)
    if duplicates:
        raise AnalysisError(f"duplicate cell identities are not aggregatable: {duplicates}")

    commits = {str(row.get("commit", "")) for row in rows}
    if require_single_commit and len(commits) > 1:
        raise AnalysisError(
            f"rows span multiple commits {sorted(commits)}; pass require_single_commit=False only "
            "with an explicit grouping decision"
        )

    schemas = {str(row.get("schema_version", "")) for row in rows}
    if len(schemas) > 1:
        raise AnalysisError(f"rows span multiple row schemas {sorted(schemas)}")

    for row in rows:
        if int(row["seed"]) not in set(expected_seeds):
            raise AnalysisError(
                f"row for arm {row['arm_id']!r} carries unexpected seed {row['seed']}"
            )
        for view, denominators in dict(row.get("denominators_by_view", {})).items():  # type: ignore[arg-type]
            missing = REQUIRED_DENOMINATOR_KEYS - set(denominators)
            if missing:
                raise AnalysisError(
                    f"arm {row['arm_id']!r} view {view!r} cannot reconstruct its denominator; "
                    f"missing {sorted(missing)}"
                )
            accounted = sum(
                int(denominators[key]) for key in ("valid", "failed", "unsupported", "timed_out")
            )
            if accounted != int(denominators["attempted"]):
                raise AnalysisError(
                    f"arm {row['arm_id']!r} view {view!r} denominators do not sum to attempted "
                    f"({accounted} vs {denominators['attempted']})"
                )


def validate_ood_membership(rows: Sequence[Mapping[str, object]], view: str) -> None:
    """Refuse to compare arms that disagree about what an OOD view contains."""

    memberships = {
        str(row["arm_id"]): tuple(sorted(dict(row.get("ood_membership", {})).get(view, ())))  # type: ignore[arg-type]
        for row in rows
    }
    distinct = set(memberships.values())
    if len(distinct) > 1:
        raise AnalysisError(
            f"arms disagree about the membership of view {view!r}; comparing them would score "
            "different subsets under one name"
        )


def aggregate(
    rows: Sequence[Mapping[str, object]],
    arm_id: str,
    view: str,
    metric: str,
    expected_seeds: Sequence[int],
) -> SeedAggregate:
    """Collect the raw per-seed values for one (arm, view, metric).

    A failed or missing cell contributes no value at all.  It is never coerced to zero, which would
    silently drag an arm's mean toward a number nobody measured.
    """

    raw: dict[int, float] = {}
    for row in rows:
        if str(row["arm_id"]) != arm_id:
            continue
        if str(row["status"]) != "complete":
            continue
        metrics = dict(row.get("metrics_by_view", {})).get(view)  # type: ignore[arg-type]
        if not metrics or metric not in metrics:
            continue
        raw[int(row["seed"])] = float(metrics[metric])
    return SeedAggregate(
        metric=metric,
        view=view,
        arm_id=arm_id,
        raw_by_seed=raw,
        expected_seeds=tuple(expected_seeds),
    )


@dataclass(frozen=True)
class PairedContrast:
    """A predeclared paired contrast between two arms over shared seeds."""

    left_arm: str
    right_arm: str
    view: str
    metric: str
    per_seed_difference: dict[int, float]
    left_mean: float | None
    right_mean: float | None

    @property
    def complete(self) -> bool:
        return bool(self.per_seed_difference)

    @property
    def mean_difference(self) -> float | MissingValue:
        if not self.complete:
            return MissingValue(MissingReason.CELL_MISSING, "no shared complete seeds")
        return sum(self.per_seed_difference.values()) / len(self.per_seed_difference)

    @property
    def effect_size(self) -> float | MissingValue:
        """Standardized paired mean difference (Cohen's d for paired samples).

        Reported as a practical effect size, deliberately alongside the raw per-seed differences.
        With three seeds this is descriptive, not an asymptotic significance claim.
        """

        if len(self.per_seed_difference) < 2:
            return MissingValue(MissingReason.METRIC_NOT_REPORTED, "fewer than two shared seeds")
        differences = list(self.per_seed_difference.values())
        mean = sum(differences) / len(differences)
        variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
        if variance == 0.0:
            return MissingValue(
                MissingReason.METRIC_NOT_REPORTED, "zero paired variance; report raw differences"
            )
        return mean / math.sqrt(variance)

    def as_dict(self) -> dict[str, object]:
        def render(value: object) -> object:
            return value.as_dict() if isinstance(value, MissingValue) else value

        return {
            "left_arm": self.left_arm,
            "right_arm": self.right_arm,
            "view": self.view,
            "metric": self.metric,
            "per_seed_difference": {
                str(seed): value for seed, value in sorted(self.per_seed_difference.items())
            },
            "left_mean": self.left_mean,
            "right_mean": self.right_mean,
            "mean_difference": render(self.mean_difference),
            "paired_effect_size": render(self.effect_size),
            "interpretation_note": (
                "paired over identical seeds and identical pair identities; with three seeds this "
                "is a descriptive effect size, not an asymptotic significance test"
            ),
        }


def paired_contrast(
    rows: Sequence[Mapping[str, object]],
    left_arm: str,
    right_arm: str,
    view: str,
    metric: str,
    expected_seeds: Sequence[int],
) -> PairedContrast:
    """Contrast two arms seed-by-seed, never by comparing independently pooled means."""

    left = aggregate(rows, left_arm, view, metric, expected_seeds)
    right = aggregate(rows, right_arm, view, metric, expected_seeds)
    shared = sorted(set(left.raw_by_seed) & set(right.raw_by_seed))
    differences = {seed: left.raw_by_seed[seed] - right.raw_by_seed[seed] for seed in shared}
    return PairedContrast(
        left_arm=left_arm,
        right_arm=right_arm,
        view=view,
        metric=metric,
        per_seed_difference=differences,
        left_mean=(sum(left.values) / len(left.values)) if left.raw_by_seed else None,
        right_mean=(sum(right.values) / len(right.values)) if right.raw_by_seed else None,
    )


def cluster_bootstrap_interval(
    values_by_group: Mapping[str, Sequence[float]],
    seed: int,
    iterations: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float] | MissingValue:
    """Resample whole groups, not individual rows.

    Pair rows that share a source expression, e-class, or trace group are correlated.  Resampling
    individual rows would understate uncertainty; the unit of resampling is therefore the group.
    """

    groups = [group for group, values in values_by_group.items() if values]
    if len(groups) < 2:
        return MissingValue(
            MissingReason.METRIC_NOT_REPORTED, "cluster resampling needs at least two groups"
        )
    if not 0.0 < confidence < 1.0:
        raise AnalysisError("confidence must lie strictly between 0 and 1")

    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        drawn = [generator.choice(groups) for _ in groups]
        pooled = [value for group in drawn for value in values_by_group[group]]
        if pooled:
            means.append(sum(pooled) / len(pooled))
    if not means:
        return MissingValue(MissingReason.METRIC_NOT_REPORTED, "no resampled means")
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low = means[max(0, math.floor(tail * len(means)))]
    high = means[min(len(means) - 1, math.ceil((1.0 - tail) * len(means)) - 1)]
    return (low, high)


def structural_metric_table(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return per-channel structural metrics, each flagged for comparability.

    There is deliberately no pooled column.  Pure-EML alpha, ordinary node/edge size, macro size,
    and dictionary-inclusive motif MDL measure different things.
    """

    table: list[dict[str, object]] = []
    for row in rows:
        for metric in list(row.get("structural_metrics", ())):  # type: ignore[arg-type]
            entry = dict(metric)
            entry["arm_id"] = row["arm_id"]
            entry["channel_name"] = row.get("channel_name")
            entry["representation_mode"] = row.get("representation_mode")
            table.append(entry)
    return table


def assert_not_pooled_across_representations(table: Sequence[Mapping[str, object]]) -> None:
    """Refuse to pool a metric name that spans representations and is flagged incomparable."""

    by_name: dict[str, set[str]] = defaultdict(set)
    for entry in table:
        if not bool(entry.get("comparable_across_channels", False)):
            by_name[str(entry["name"])].add(str(entry.get("representation_mode")))
    offenders = sorted(name for name, modes in by_name.items() if len(modes) > 1)
    if offenders:
        raise AnalysisError(
            f"structural metrics {offenders} are flagged incomparable yet span multiple "
            "representation modes; they may not be plotted or averaged as one quantity"
        )


@dataclass
class GateVerdict:
    """A Gate G6 verdict with its exact numerical support and caveats."""

    state: GateState
    rationale: str
    supporting_contrasts: list[dict[str, object]] = field(default_factory=list)
    unmet_requirements: list[str] = field(default_factory=list)
    caveats: tuple[str, ...] = FIXED_SCALE_CAVEATS
    is_fixture: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": GATE_G6_SCHEMA_VERSION,
            "state": str(self.state),
            "rationale": self.rationale,
            "supporting_contrasts": self.supporting_contrasts,
            "unmet_requirements": list(self.unmet_requirements),
            "caveats": list(self.caveats),
            "is_fixture": self.is_fixture,
        }


def evaluate_gate_g6(
    rows: Sequence[Mapping[str, object]],
    expected_seeds: Sequence[int],
    expected_arms: Sequence[str],
    predeclared_contrasts: Sequence[tuple[str, str, str, str]],
    *,
    is_fixture: bool = False,
    minimum_effect_size: float = 0.5,
) -> GateVerdict:
    """Apply the predeclared Gate G6 rules.

    The rules are fixed before production results exist, and a fixture report can never produce a
    scientific verdict: it returns ``insufficient_evidence`` and says why.
    """

    if is_fixture:
        return GateVerdict(
            state=GateState.INSUFFICIENT_EVIDENCE,
            rationale=(
                "this report was generated from fixture rows and cannot express a scientific "
                "verdict about Goal 6"
            ),
            unmet_requirements=["production result rows"],
            is_fixture=True,
        )

    unmet: list[str] = []
    present_arms = {str(row["arm_id"]) for row in rows}
    missing_arms = sorted(set(expected_arms) - present_arms)
    if missing_arms:
        unmet.append(f"missing arms: {missing_arms}")

    for arm in expected_arms:
        seeds = {int(row["seed"]) for row in rows if str(row["arm_id"]) == arm}
        if not set(expected_seeds).issubset(seeds):
            unmet.append(f"arm {arm} is missing seeds {sorted(set(expected_seeds) - seeds)}")

    failed = [f"{row['arm_id']}@{row['seed']}" for row in rows if str(row["status"]) != "complete"]
    if failed:
        unmet.append(f"incomplete or failed cells: {sorted(failed)}")

    contrasts = [
        paired_contrast(rows, left, right, view, metric, expected_seeds).as_dict()
        for left, right, view, metric in predeclared_contrasts
    ]

    if unmet:
        return GateVerdict(
            state=GateState.INSUFFICIENT_EVIDENCE,
            rationale="the predeclared evidence requirements are not met",
            supporting_contrasts=contrasts,
            unmet_requirements=unmet,
        )

    decisive = [
        contrast
        for contrast in contrasts
        if isinstance(contrast["paired_effect_size"], float)
        and abs(float(contrast["paired_effect_size"])) >= minimum_effect_size
        and isinstance(contrast["mean_difference"], float)
        and float(contrast["mean_difference"]) > 0.0
    ]
    if decisive:
        return GateVerdict(
            state=GateState.PASS,
            rationale=(
                f"{len(decisive)} of {len(contrasts)} predeclared contrasts show a positive paired "
                f"difference with |effect size| >= {minimum_effect_size}"
            ),
            supporting_contrasts=contrasts,
        )
    return GateVerdict(
        state=GateState.FAIL,
        rationale=(
            "no predeclared contrast reaches the preregistered effect-size threshold; this is a "
            "reportable null result and is not to be resolved by rerunning or by reselecting arms"
        ),
        supporting_contrasts=contrasts,
    )


def build_summary(
    rows: Sequence[Mapping[str, object]],
    expected_seeds: Sequence[int],
    expected_arms: Sequence[str],
    views: Sequence[str],
    metrics: Sequence[str],
    *,
    is_fixture: bool = False,
) -> dict[str, object]:
    """Assemble the complete, denominator-explicit Goal 6 summary payload."""

    validate_rows(rows, expected_seeds)
    table = structural_metric_table(rows)
    assert_not_pooled_across_representations(table)

    aggregates = [
        aggregate(rows, arm, view, metric, expected_seeds).as_dict()
        for arm in expected_arms
        for view in views
        for metric in metrics
    ]
    missing_cells = [
        {"arm_id": arm, "seed": seed, "reason": str(MissingReason.CELL_MISSING)}
        for arm in expected_arms
        for seed in expected_seeds
        if not any(str(row["arm_id"]) == arm and int(row["seed"]) == seed for row in rows)
    ]
    failed_cells = [
        {
            "arm_id": str(row["arm_id"]),
            "seed": int(row["seed"]),
            "status": str(row["status"]),
            "failure_reason": row.get("failure_reason"),
        }
        for row in rows
        if str(row["status"]) != "complete"
    ]
    return {
        "schema_version": GOAL6_ANALYSIS_SCHEMA_VERSION,
        "is_fixture": is_fixture,
        "aggregates": aggregates,
        "structural_metrics": table,
        "missing_cells": missing_cells,
        "failed_cells": failed_cells,
        "caveats": list(FIXED_SCALE_CAVEATS),
        "row_count": len(rows),
        "expected_cell_count": len(expected_arms) * len(expected_seeds),
    }
