"""Compute-normalized analysis at one fixed corpus scale.

No function in this module fits a data-scaling curve or extrapolates to a
different dataset size.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import defaultdict
from enum import StrEnum
from statistics import fmean
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

FIXED_SCALE_INPUT_SCHEMA_VERSION = "geml-goal11-fixed-scale-input-v1"
FIXED_SCALE_RESULT_SCHEMA_VERSION = "geml-goal11-fixed-scale-result-v1"

_NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_SafeId = Annotated[
    str,
    StringConstraints(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
_NonNegativeFloat = Annotated[StrictFloat, Field(ge=0.0)]


class FixedScaleAnalysisError(ValueError):
    """Fixed-scale observations violate the comparison contract."""


class Track(StrEnum):
    EQUIVALENCE = "equivalence"
    REWRITE_POLICY = "rewrite_policy"
    PROOF = "proof"
    SIMPLIFICATION = "simplification"
    SYMBOLIC_REGRESSION = "symbolic_regression"


class MetricDirection(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class MetricAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class ObservationStatus(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"
    INVALID = "invalid"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"


class MeasurementMethod(StrEnum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    DECLARED = "declared"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class QualityObservation(_FrozenModel):
    metric_id: _NonBlank
    unit: _NonBlank
    direction: MetricDirection
    availability: MetricAvailability
    value: StrictFloat | None = None
    attempted_count: _NonNegativeInt
    valid_count: _NonNegativeInt
    failed_count: _NonNegativeInt
    invalid_count: _NonNegativeInt
    unsupported_count: _NonNegativeInt
    timeout_count: _NonNegativeInt
    reason: _NonBlank | None = None

    @model_validator(mode="after")
    def validate_metric(self) -> Self:
        accounted = (
            self.valid_count
            + self.failed_count
            + self.invalid_count
            + self.unsupported_count
            + self.timeout_count
        )
        if accounted != self.attempted_count:
            raise ValueError("quality denominators must account for every attempt")
        if self.availability is MetricAvailability.AVAILABLE:
            if self.value is None:
                raise ValueError("available quality requires a value")
            if self.valid_count == 0:
                raise ValueError("available quality requires at least one valid row")
            if self.reason is not None:
                raise ValueError("available quality cannot carry an unavailable reason")
        elif self.value is not None or self.reason is None:
            raise ValueError("unavailable quality requires a reason and no value")
        return self


class ResourceObservation(_FrozenModel):
    metric_id: _NonBlank
    unit: _NonBlank
    availability: MetricAvailability
    value: _NonNegativeFloat | None = None
    method: MeasurementMethod | None = None
    reason: _NonBlank | None = None

    @model_validator(mode="after")
    def validate_resource(self) -> Self:
        if self.availability is MetricAvailability.AVAILABLE:
            if self.value is None or self.method is None:
                raise ValueError("available resources require value and measurement method")
            if self.reason is not None:
                raise ValueError("available resources cannot carry an unavailable reason")
        elif self.value is not None or self.method is not None or self.reason is None:
            raise ValueError("unavailable resources require a reason and no value/method")
        return self


class OutcomeCounts(_FrozenModel):
    attempted_count: _NonNegativeInt
    valid_count: _NonNegativeInt
    failed_count: _NonNegativeInt
    invalid_count: _NonNegativeInt
    unsupported_count: _NonNegativeInt
    timeout_count: _NonNegativeInt

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        accounted = (
            self.valid_count
            + self.failed_count
            + self.invalid_count
            + self.unsupported_count
            + self.timeout_count
        )
        if accounted != self.attempted_count:
            raise ValueError("observation outcomes must account for every attempt")
        return self


class EfficiencyObservation(_FrozenModel):
    """One method/seed/source-group observation with authenticated provenance."""

    schema_version: Literal["geml-goal11-fixed-scale-input-v1"] = FIXED_SCALE_INPUT_SCHEMA_VERSION
    row_id: _NonBlank
    source_artifact_id: _NonBlank
    source_sha256: _Sha256
    source_locator: _NonBlank
    track: Track
    task_view: _NonBlank
    method_id: _NonBlank
    representation_id: _NonBlank
    seed: StrictInt
    group_id: _NonBlank
    cohort_digest: _Sha256
    comparison_protocol_digest: _Sha256
    config_digest: _Sha256
    hardware_digest: _Sha256
    precision: _NonBlank
    status: ObservationStatus
    outcomes: OutcomeCounts
    quality: QualityObservation | None = None
    resources: tuple[ResourceObservation, ...] = ()
    failure_reason: _NonBlank | None = None

    @field_validator("resources")
    @classmethod
    def validate_resource_ids(
        cls,
        value: tuple[ResourceObservation, ...],
    ) -> tuple[ResourceObservation, ...]:
        ids = tuple(item.metric_id for item in value)
        if len(set(ids)) != len(ids):
            raise ValueError("resource metric IDs must be unique within a row")
        return value

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status is ObservationStatus.COMPLETE:
            if self.quality is None:
                raise ValueError("complete observations require quality")
            if self.failure_reason is not None:
                raise ValueError("complete observations cannot carry failure_reason")
            quality_counts = (
                self.quality.attempted_count,
                self.quality.valid_count,
                self.quality.failed_count,
                self.quality.invalid_count,
                self.quality.unsupported_count,
                self.quality.timeout_count,
            )
            outcome_counts = (
                self.outcomes.attempted_count,
                self.outcomes.valid_count,
                self.outcomes.failed_count,
                self.outcomes.invalid_count,
                self.outcomes.unsupported_count,
                self.outcomes.timeout_count,
            )
            if quality_counts != outcome_counts:
                raise ValueError("complete quality and observation outcomes disagree")
        else:
            if self.failure_reason is None:
                raise ValueError("non-complete observations require failure_reason")
            if self.quality is not None:
                raise ValueError("non-complete observations cannot carry quality")
            status_count = {
                ObservationStatus.FAILED: self.outcomes.failed_count,
                ObservationStatus.INVALID: self.outcomes.invalid_count,
                ObservationStatus.UNSUPPORTED: self.outcomes.unsupported_count,
                ObservationStatus.TIMEOUT: self.outcomes.timeout_count,
                ObservationStatus.MISSING: 0,
            }[self.status]
            if self.status is ObservationStatus.MISSING:
                if self.outcomes.attempted_count != 0:
                    raise ValueError("missing observations cannot claim attempted rows")
            elif status_count == 0:
                raise ValueError("non-complete status must appear in its outcome counts")
        return self

    def resource(self, metric_id: str) -> ResourceObservation | None:
        return next((item for item in self.resources if item.metric_id == metric_id), None)

    def evidence_projection(self) -> dict[str, object]:
        """Scientific fields that must exactly match the cited frozen source row."""

        return {
            "row_id": self.row_id,
            "track": self.track.value,
            "task_view": self.task_view,
            "method_id": self.method_id,
            "representation_id": self.representation_id,
            "seed": self.seed,
            "group_id": self.group_id,
            "cohort_digest": self.cohort_digest,
            "comparison_protocol_digest": self.comparison_protocol_digest,
            "config_digest": self.config_digest,
            "hardware_digest": self.hardware_digest,
            "precision": self.precision,
            "status": self.status.value,
            "outcomes": self.outcomes.model_dump(mode="json"),
            "quality": (None if self.quality is None else self.quality.model_dump(mode="json")),
            "resources": [item.model_dump(mode="json") for item in self.resources],
            "failure_reason": self.failure_reason,
        }


class ComparisonSpec(_FrozenModel):
    panel_id: _SafeId
    track: Track
    task_view: _NonBlank
    quality_metric_id: _NonBlank
    resource_metric_ids: tuple[_SafeId, ...] = Field(min_length=1)
    require_same_hardware: bool = False
    require_same_precision: bool = False

    @field_validator("track", mode="before")
    @classmethod
    def normalize_yaml_track(cls, value: object) -> object:
        return Track(value) if isinstance(value, str) else value

    @field_validator("resource_metric_ids", mode="before")
    @classmethod
    def normalize_yaml_resources(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("resource_metric_ids")
    @classmethod
    def validate_resource_axes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("resource_metric_ids must be unique")
        parameter_only = {"parameter_count", "parameters"}
        if set(value).issubset(parameter_only):
            raise ValueError("parameter count alone is not a compute-efficiency analysis")
        return value


class FixedScaleAnalysisConfig(_FrozenModel):
    schema_version: Literal["geml-goal11-fixed-scale-config-v1"] = (
        "geml-goal11-fixed-scale-config-v1"
    )
    expected_seeds: tuple[StrictInt, StrictInt, StrictInt]
    bootstrap_seed: StrictInt
    bootstrap_replicates: Annotated[StrictInt, Field(ge=100)]
    comparisons: tuple[ComparisonSpec, ...]

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if len(set(self.expected_seeds)) != 3:
            raise ValueError("expected_seeds must contain exactly three distinct seeds")
        ids = tuple(item.panel_id for item in self.comparisons)
        if len(set(ids)) != len(ids):
            raise ValueError("comparison panel IDs must be unique")
        return self


class AggregateMetric(_FrozenModel):
    metric_id: _NonBlank
    unit: _NonBlank
    availability: MetricAvailability
    mean: StrictFloat | None = None
    method: MeasurementMethod | None = None
    reason: _NonBlank | None = None

    @model_validator(mode="after")
    def validate_aggregate(self) -> Self:
        if self.availability is MetricAvailability.AVAILABLE:
            if self.mean is None or self.method is None or self.reason is not None:
                raise ValueError("available aggregate requires mean/method and no reason")
        elif self.mean is not None or self.method is not None or self.reason is None:
            raise ValueError("unavailable aggregate requires a reason and no mean/method")
        return self


class SeedPoint(_FrozenModel):
    method_id: _NonBlank
    representation_id: _NonBlank
    seed: StrictInt
    group_count: Annotated[StrictInt, Field(ge=1)]
    quality_mean: StrictFloat
    resources: tuple[AggregateMetric, ...]
    source_row_ids: tuple[_NonBlank, ...] = Field(min_length=1)


class ParetoPoint(_FrozenModel):
    method_id: _NonBlank
    representation_id: _NonBlank
    quality_mean: StrictFloat
    resources: tuple[AggregateMetric, ...]
    eligible: bool
    dominated_by: tuple[_NonBlank, ...] = ()
    exclusion_reason: _NonBlank | None = None
    source_row_ids: tuple[_NonBlank, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_eligibility(self) -> Self:
        if self.eligible == (self.exclusion_reason is not None):
            raise ValueError("eligible points have no exclusion reason; ineligible points do")
        if not self.eligible and self.dominated_by:
            raise ValueError("ineligible points cannot be classified as dominated")
        return self


class PairedContrast(_FrozenModel):
    left_method: _NonBlank
    right_method: _NonBlank
    estimate: StrictFloat | None = None
    ci_low: StrictFloat | None = None
    ci_high: StrictFloat | None = None
    cluster_count: _NonNegativeInt
    availability: MetricAvailability
    reason: _NonBlank | None = None

    @model_validator(mode="after")
    def validate_contrast(self) -> Self:
        values = (self.estimate, self.ci_low, self.ci_high)
        if self.availability is MetricAvailability.AVAILABLE:
            if any(value is None for value in values) or self.reason is not None:
                raise ValueError("available contrast requires estimate and interval")
        elif any(value is not None for value in values) or self.reason is None:
            raise ValueError("unavailable contrast requires a reason and no estimates")
        return self


class SourceRowLocator(_FrozenModel):
    row_id: _NonBlank
    source_locator: _NonBlank


class SourceBinding(_FrozenModel):
    source_artifact_id: _NonBlank
    source_sha256: _Sha256
    source_rows: tuple[SourceRowLocator, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rows(self) -> Self:
        row_ids = tuple(item.row_id for item in self.source_rows)
        if len(set(row_ids)) != len(row_ids) or row_ids != tuple(sorted(row_ids)):
            raise ValueError("source rows must have unique sorted row IDs")
        return self


class MethodSeedCoverage(_FrozenModel):
    method_id: _NonBlank
    representation_id: _NonBlank
    observed_seeds: tuple[StrictInt, ...]
    missing_seeds: tuple[StrictInt, ...]


class RetainedObservation(_FrozenModel):
    row_id: _NonBlank
    source_artifact_id: _NonBlank
    source_sha256: _Sha256
    source_locator: _NonBlank
    track: Track
    task_view: _NonBlank
    method_id: _NonBlank
    representation_id: _NonBlank
    seed: StrictInt
    group_id: _NonBlank
    status: ObservationStatus
    outcomes: OutcomeCounts
    failure_reason: _NonBlank


class FixedScalePanel(_FrozenModel):
    panel_id: _NonBlank
    comparison_key: _Sha256
    track: Track
    task_view: _NonBlank
    quality_metric_id: _NonBlank
    quality_unit: _NonBlank
    quality_direction: MetricDirection
    resource_metric_ids: tuple[_NonBlank, ...]
    seed_points: tuple[SeedPoint, ...]
    pareto_points: tuple[ParetoPoint, ...]
    contrasts: tuple[PairedContrast, ...]
    retained_noncomplete_row_ids: tuple[_NonBlank, ...]
    retained_noncomplete: tuple[RetainedObservation, ...]
    observed_seeds: tuple[StrictInt, ...]
    missing_seeds: tuple[StrictInt, ...]
    method_seed_coverage: tuple[MethodSeedCoverage, ...]
    source_bindings: tuple[SourceBinding, ...] = Field(min_length=1)


class FixedScaleResult(_FrozenModel):
    schema_version: Literal["geml-goal11-fixed-scale-result-v1"] = FIXED_SCALE_RESULT_SCHEMA_VERSION
    analysis_config_sha256: _Sha256
    observations_sha256: _Sha256
    observations_file_sha256: _Sha256 | None = None
    run_config_sha256: _Sha256 | None = None
    manifest_sha256: _Sha256 | None = None
    manifest_audit_sha256: _Sha256 | None = None
    bootstrap_seed: StrictInt
    bootstrap_replicates: Annotated[StrictInt, Field(ge=100)]
    implementation_id: Literal["geml-goal11-fixed-scale-v1"] = "geml-goal11-fixed-scale-v1"
    panels: tuple[FixedScalePanel, ...]
    unassigned_row_ids: tuple[_NonBlank, ...]
    retained_noncomplete_row_ids: tuple[_NonBlank, ...] = ()
    retained_noncomplete: tuple[RetainedObservation, ...] = ()
    boundaries: tuple[str, ...] = (
        "All comparisons use the frozen 250k-v1 evidence boundary.",
        "No 10-100x experiment, scaling exponent, or extrapolation is reported.",
        "Raw seeds are retained; repeated rows and seeds are not treated as IID.",
    )


def _method_key(row: EfficiencyObservation) -> str:
    return f"{row.method_id}::{row.representation_id}"


def _compatibility_key(row: EfficiencyObservation, spec: ComparisonSpec) -> tuple[str, ...]:
    quality = row.quality
    if quality is None:
        raise FixedScaleAnalysisError("complete row is missing quality")
    parts = [
        row.track.value,
        row.task_view,
        quality.metric_id,
        quality.unit,
        quality.direction.value,
        row.cohort_digest,
        row.comparison_protocol_digest,
        row.config_digest,
    ]
    for metric_id in spec.resource_metric_ids:
        resource = row.resource(metric_id)
        if resource is None:
            parts.extend((metric_id, "missing"))
        else:
            parts.extend(
                (
                    metric_id,
                    resource.unit,
                    resource.availability.value,
                    "" if resource.method is None else resource.method.value,
                )
            )
    if spec.require_same_hardware:
        parts.append(row.hardware_digest)
    if spec.require_same_precision:
        parts.append(row.precision)
    return tuple(parts)


def _key_digest(parts: tuple[str, ...]) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _model_payload_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_observations(observations: tuple[EfficiencyObservation, ...]) -> None:
    """Reject duplicate scientific cells instead of silently averaging reruns."""

    row_ids = tuple(row.row_id for row in observations)
    if len(set(row_ids)) != len(row_ids):
        raise FixedScaleAnalysisError("row_id values must be unique")
    cell_keys = [
        (
            row.track,
            row.task_view,
            row.method_id,
            row.representation_id,
            row.seed,
            row.group_id,
            row.comparison_protocol_digest,
        )
        for row in observations
    ]
    if len(set(cell_keys)) != len(cell_keys):
        raise FixedScaleAnalysisError("duplicate method/seed/group cells are not allowed")


def _aggregate_resource(
    rows: list[EfficiencyObservation],
    metric_id: str,
) -> AggregateMetric:
    observations = [row.resource(metric_id) for row in rows]
    if any(item is None for item in observations):
        return AggregateMetric(
            metric_id=metric_id,
            unit="unavailable",
            availability=MetricAvailability.UNAVAILABLE,
            reason="one or more rows do not publish this resource metric",
        )
    resources = [item for item in observations if item is not None]
    units = {item.unit for item in resources}
    if len(units) != 1:
        return AggregateMetric(
            metric_id=metric_id,
            unit="incompatible",
            availability=MetricAvailability.UNAVAILABLE,
            reason="resource units are incompatible",
        )
    if any(item.availability is not MetricAvailability.AVAILABLE for item in resources):
        return AggregateMetric(
            metric_id=metric_id,
            unit=next(iter(units)),
            availability=MetricAvailability.UNAVAILABLE,
            reason="one or more rows have unavailable resource telemetry",
        )
    methods = {item.method for item in resources}
    if len(methods) != 1:
        return AggregateMetric(
            metric_id=metric_id,
            unit=next(iter(units)),
            availability=MetricAvailability.UNAVAILABLE,
            reason="resource measurement methods are incompatible",
        )
    method = next(iter(methods))
    if method is None:
        raise FixedScaleAnalysisError("available resource is missing its measurement method")
    return AggregateMetric(
        metric_id=metric_id,
        unit=next(iter(units)),
        availability=MetricAvailability.AVAILABLE,
        mean=float(fmean(item.value for item in resources if item.value is not None)),
        method=method,
    )


def _seed_points(
    rows: list[EfficiencyObservation],
    spec: ComparisonSpec,
) -> tuple[SeedPoint, ...]:
    grouped: dict[tuple[str, str, int], list[EfficiencyObservation]] = defaultdict(list)
    for row in rows:
        grouped[(row.method_id, row.representation_id, row.seed)].append(row)
    points = []
    for (method_id, representation_id, seed), group_rows in sorted(grouped.items()):
        quality_values = [
            row.quality.value
            for row in group_rows
            if row.quality is not None
            and row.quality.availability is MetricAvailability.AVAILABLE
            and row.quality.value is not None
        ]
        if len(quality_values) != len(group_rows):
            continue
        points.append(
            SeedPoint(
                method_id=method_id,
                representation_id=representation_id,
                seed=seed,
                group_count=len({row.group_id for row in group_rows}),
                quality_mean=float(fmean(quality_values)),
                resources=tuple(
                    _aggregate_resource(group_rows, metric_id)
                    for metric_id in spec.resource_metric_ids
                ),
                source_row_ids=tuple(sorted(row.row_id for row in group_rows)),
            )
        )
    return tuple(points)


def _percentile(sorted_values: list[float], probability: float) -> float:
    index = round((len(sorted_values) - 1) * probability)
    return sorted_values[index]


def paired_group_contrast(
    rows: list[EfficiencyObservation],
    left_method: str,
    right_method: str,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> PairedContrast:
    """Pair by seed/group, then resample source groups rather than repeated rows."""

    by_method_cell: dict[tuple[str, int, str], float] = {}
    for row in rows:
        if row.quality is None or row.quality.value is None:
            continue
        by_method_cell[(_method_key(row), row.seed, row.group_id)] = row.quality.value
    left_cells = {
        (seed, group_id) for method, seed, group_id in by_method_cell if method == left_method
    }
    right_cells = {
        (seed, group_id) for method, seed, group_id in by_method_cell if method == right_method
    }
    if left_cells != right_cells:
        paired_groups = {group for _, group in left_cells & right_cells}
        return PairedContrast(
            left_method=left_method,
            right_method=right_method,
            cluster_count=len(paired_groups),
            availability=MetricAvailability.UNAVAILABLE,
            reason="method cohorts differ; no favorable-subset contrast is computed",
        )
    groups = sorted(
        {
            group_id
            for method, _, group_id in by_method_cell
            if method in {left_method, right_method}
        }
    )
    group_contrasts: list[float] = []
    for group_id in groups:
        seed_differences = [
            by_method_cell[(left_method, seed, group_id)]
            - by_method_cell[(right_method, seed, group_id)]
            for seed in sorted({row.seed for row in rows})
            if (left_method, seed, group_id) in by_method_cell
            and (right_method, seed, group_id) in by_method_cell
        ]
        if seed_differences:
            group_contrasts.append(float(fmean(seed_differences)))
    if len(group_contrasts) < 2:
        return PairedContrast(
            left_method=left_method,
            right_method=right_method,
            cluster_count=len(group_contrasts),
            availability=MetricAvailability.UNAVAILABLE,
            reason="fewer than two paired source/task groups",
        )
    estimate = float(fmean(group_contrasts))
    generator = random.Random(bootstrap_seed)
    bootstrap = sorted(
        float(
            fmean(
                group_contrasts[generator.randrange(len(group_contrasts))] for _ in group_contrasts
            )
        )
        for _ in range(bootstrap_replicates)
    )
    return PairedContrast(
        left_method=left_method,
        right_method=right_method,
        estimate=estimate,
        ci_low=_percentile(bootstrap, 0.025),
        ci_high=_percentile(bootstrap, 0.975),
        cluster_count=len(group_contrasts),
        availability=MetricAvailability.AVAILABLE,
    )


def _mean_points(
    rows: list[EfficiencyObservation],
    spec: ComparisonSpec,
    expected_seeds: tuple[int, int, int],
) -> list[ParetoPoint]:
    grouped: dict[tuple[str, str], list[EfficiencyObservation]] = defaultdict(list)
    for row in rows:
        grouped[(row.method_id, row.representation_id)].append(row)
    cell_sets = {
        method: {(row.seed, row.group_id) for row in method_rows}
        for method, method_rows in grouped.items()
    }
    matched_cohorts = len({frozenset(cells) for cells in cell_sets.values()}) <= 1
    points = []
    for (method_id, representation_id), method_rows in sorted(grouped.items()):
        quality = float(
            fmean(
                row.quality.value
                for row in method_rows
                if row.quality is not None and row.quality.value is not None
            )
        )
        resources = tuple(
            _aggregate_resource(method_rows, metric_id) for metric_id in spec.resource_metric_ids
        )
        missing = [
            metric.metric_id
            for metric in resources
            if metric.availability is not MetricAvailability.AVAILABLE
        ]
        observed_seeds = {row.seed for row in method_rows}
        absent_seeds = [seed for seed in expected_seeds if seed not in observed_seeds]
        exclusions = []
        if missing:
            exclusions.append(f"missing compatible telemetry for {', '.join(missing)}")
        if absent_seeds:
            exclusions.append(f"missing frozen seeds {absent_seeds}")
        if not matched_cohorts:
            exclusions.append("method seed/group cohorts differ")
        points.append(
            ParetoPoint(
                method_id=method_id,
                representation_id=representation_id,
                quality_mean=quality,
                resources=resources,
                eligible=not exclusions,
                exclusion_reason=(None if not exclusions else "; ".join(exclusions)),
                source_row_ids=tuple(sorted(row.row_id for row in method_rows)),
            )
        )
    return points


def _method_seed_coverage(
    rows: list[EfficiencyObservation],
    expected_seeds: tuple[int, int, int],
) -> tuple[MethodSeedCoverage, ...]:
    grouped: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        grouped[(row.method_id, row.representation_id)].add(row.seed)
    return tuple(
        MethodSeedCoverage(
            method_id=method_id,
            representation_id=representation_id,
            observed_seeds=tuple(sorted(seeds)),
            missing_seeds=tuple(seed for seed in expected_seeds if seed not in seeds),
        )
        for (method_id, representation_id), seeds in sorted(grouped.items())
    )


def _dominates(
    left: ParetoPoint,
    right: ParetoPoint,
    direction: MetricDirection,
) -> bool:
    left_quality = left.quality_mean
    right_quality = right.quality_mean
    quality_no_worse = (
        left_quality >= right_quality
        if direction is MetricDirection.HIGHER_IS_BETTER
        else left_quality <= right_quality
    )
    quality_strict = left_quality != right_quality
    left_resources = [item.mean for item in left.resources]
    right_resources = [item.mean for item in right.resources]
    resource_no_worse = all(
        left_value is not None and right_value is not None and left_value <= right_value
        for left_value, right_value in zip(left_resources, right_resources, strict=True)
    )
    resource_strict = any(
        left_value is not None and right_value is not None and left_value < right_value
        for left_value, right_value in zip(left_resources, right_resources, strict=True)
    )
    return quality_no_worse and resource_no_worse and (quality_strict or resource_strict)


def pareto_front(
    points: list[ParetoPoint],
    direction: MetricDirection,
) -> tuple[ParetoPoint, ...]:
    """Classify strict non-domination without imputing unavailable axes."""

    output = []
    for point in points:
        if not point.eligible:
            output.append(point)
            continue
        dominators = tuple(
            f"{other.method_id}::{other.representation_id}"
            for other in points
            if other.eligible and other is not point and _dominates(other, point, direction)
        )
        output.append(point.model_copy(update={"dominated_by": dominators}))
    return tuple(output)


def _retained_observation(row: EfficiencyObservation) -> RetainedObservation:
    if row.status is ObservationStatus.COMPLETE or row.failure_reason is None:
        raise FixedScaleAnalysisError("retained failure ledger received a complete row")
    return RetainedObservation(
        row_id=row.row_id,
        source_artifact_id=row.source_artifact_id,
        source_sha256=row.source_sha256,
        source_locator=row.source_locator,
        track=row.track,
        task_view=row.task_view,
        method_id=row.method_id,
        representation_id=row.representation_id,
        seed=row.seed,
        group_id=row.group_id,
        status=row.status,
        outcomes=row.outcomes,
        failure_reason=row.failure_reason,
    )


def _panel_for_rows(
    spec: ComparisonSpec,
    key: tuple[str, ...],
    rows: list[EfficiencyObservation],
    retained_noncomplete: tuple[EfficiencyObservation, ...],
    config: FixedScaleAnalysisConfig,
) -> FixedScalePanel:
    quality = rows[0].quality
    if quality is None:
        raise FixedScaleAnalysisError("panel row is missing quality")
    observed_seeds = tuple(sorted({row.seed for row in rows}))
    missing_seeds = tuple(seed for seed in config.expected_seeds if seed not in observed_seeds)
    methods = sorted({_method_key(row) for row in rows})
    source_rows: dict[tuple[str, str], list[SourceRowLocator]] = defaultdict(list)
    for row in rows:
        source_rows[(row.source_artifact_id, row.source_sha256)].append(
            SourceRowLocator(row_id=row.row_id, source_locator=row.source_locator)
        )
    contrasts = tuple(
        paired_group_contrast(
            rows,
            left,
            right,
            bootstrap_seed=config.bootstrap_seed,
            bootstrap_replicates=config.bootstrap_replicates,
        )
        for left, right in itertools.combinations(methods, 2)
    )
    return FixedScalePanel(
        panel_id=spec.panel_id,
        comparison_key=_key_digest(key),
        track=spec.track,
        task_view=spec.task_view,
        quality_metric_id=quality.metric_id,
        quality_unit=quality.unit,
        quality_direction=quality.direction,
        resource_metric_ids=spec.resource_metric_ids,
        seed_points=_seed_points(rows, spec),
        pareto_points=pareto_front(
            _mean_points(rows, spec, config.expected_seeds),
            quality.direction,
        ),
        contrasts=contrasts,
        retained_noncomplete_row_ids=tuple(row.row_id for row in retained_noncomplete),
        retained_noncomplete=tuple(_retained_observation(row) for row in retained_noncomplete),
        observed_seeds=observed_seeds,
        missing_seeds=missing_seeds,
        method_seed_coverage=_method_seed_coverage(rows, config.expected_seeds),
        source_bindings=tuple(
            SourceBinding(
                source_artifact_id=artifact_id,
                source_sha256=checksum,
                source_rows=tuple(sorted(row_ids, key=lambda item: item.row_id)),
            )
            for (artifact_id, checksum), row_ids in sorted(source_rows.items())
        ),
    )


def summarize_fixed_scale(
    observations: tuple[EfficiencyObservation, ...],
    config: FixedScaleAnalysisConfig,
) -> FixedScaleResult:
    """Build only comparable fixed-scale panels and retain all excluded rows."""

    validate_observations(observations)
    assigned: set[str] = set()
    panels: list[FixedScalePanel] = []
    retained_noncomplete_rows = tuple(
        sorted(
            (row for row in observations if row.status is not ObservationStatus.COMPLETE),
            key=lambda row: row.row_id,
        )
    )
    for spec in config.comparisons:
        candidate = [
            row
            for row in observations
            if row.track is spec.track and row.task_view == spec.task_view
        ]
        noncomplete = tuple(
            sorted(
                (row for row in candidate if row.status is not ObservationStatus.COMPLETE),
                key=lambda row: row.row_id,
            )
        )
        compatible: dict[tuple[str, ...], list[EfficiencyObservation]] = defaultdict(list)
        for row in candidate:
            if row.status is not ObservationStatus.COMPLETE:
                assigned.add(row.row_id)
                continue
            if (
                row.quality is None
                or row.quality.metric_id != spec.quality_metric_id
                or row.quality.availability is not MetricAvailability.AVAILABLE
            ):
                continue
            compatible[_compatibility_key(row, spec)].append(row)
            assigned.add(row.row_id)
        for key, rows in sorted(compatible.items()):
            panels.append(_panel_for_rows(spec, key, rows, noncomplete, config))
    return FixedScaleResult(
        analysis_config_sha256=_model_payload_sha256(config.model_dump(mode="json")),
        observations_sha256=_model_payload_sha256(
            [
                row.model_dump(mode="json")
                for row in sorted(observations, key=lambda item: item.row_id)
            ]
        ),
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_replicates=config.bootstrap_replicates,
        panels=tuple(panels),
        unassigned_row_ids=tuple(
            sorted(row.row_id for row in observations if row.row_id not in assigned)
        ),
        retained_noncomplete_row_ids=tuple(row.row_id for row in retained_noncomplete_rows),
        retained_noncomplete=tuple(_retained_observation(row) for row in retained_noncomplete_rows),
    )


def _aggregate_text(metric: AggregateMetric) -> str:
    if metric.availability is MetricAvailability.AVAILABLE:
        return f"{metric.mean} {metric.unit} ({metric.method.value})"
    return f"{metric.availability.value}: {metric.reason}"


def render_fixed_scale_markdown(result: FixedScaleResult) -> str:
    """Render traceable fixed-scale tables without a cross-task scalar score."""

    lines = [
        "# Goal 11 fixed-scale efficiency results",
        "",
        *(f"- {boundary}" for boundary in result.boundaries),
        "",
    ]
    if not result.panels:
        lines.extend(["No compatible complete panel is available.", ""])
    for panel in result.panels:
        lines.extend(
            [
                f"## {panel.panel_id}",
                "",
                f"Comparison key: `{panel.comparison_key}`",
                "",
                f"Observed seeds: `{list(panel.observed_seeds)}`; "
                f"missing seeds: `{list(panel.missing_seeds)}`",
                "",
                "Method seed coverage: "
                + "; ".join(
                    f"`{item.method_id}/{item.representation_id}` "
                    f"observed={list(item.observed_seeds)} missing={list(item.missing_seeds)}"
                    for item in panel.method_seed_coverage
                ),
                "",
                "| Method | Representation | Quality | Resources | Pareto state |",
                "|---|---|---:|---|---|",
            ]
        )
        for point in panel.pareto_points:
            resources = "; ".join(_aggregate_text(item) for item in point.resources)
            if not point.eligible:
                pareto = f"ineligible: {point.exclusion_reason}"
            elif point.dominated_by:
                pareto = "dominated by " + ", ".join(point.dominated_by)
            else:
                pareto = "non-dominated"
            lines.append(
                f"| `{point.method_id}` | `{point.representation_id}` | "
                f"{point.quality_mean} {panel.quality_unit} | {resources} | {pareto} |"
            )
        lines.extend(
            [
                "",
                "### Raw seed points",
                "",
                "| Method | Representation | Seed | Groups | Quality | Source rows |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for point in panel.seed_points:
            lines.append(
                f"| `{point.method_id}` | `{point.representation_id}` | {point.seed} | "
                f"{point.group_count} | {point.quality_mean} | "
                f"{', '.join(f'`{item}`' for item in point.source_row_ids)} |"
            )
        lines.extend(["", "### Group-paired contrasts", ""])
        if not panel.contrasts:
            lines.append("No method pair is available.")
        for contrast in panel.contrasts:
            if contrast.availability is MetricAvailability.AVAILABLE:
                detail = (
                    f"estimate={contrast.estimate}; interval=[{contrast.ci_low}, "
                    f"{contrast.ci_high}]; clusters={contrast.cluster_count}"
                )
            else:
                detail = f"{contrast.availability.value}: {contrast.reason}"
            lines.append(f"- `{contrast.left_method}` minus `{contrast.right_method}`: {detail}")
        if panel.retained_noncomplete:
            lines.extend(
                [
                    "",
                    "### Retained non-complete rows",
                    "",
                ]
            )
            lines.extend(
                f"- `{row.row_id}`: `{row.status.value}`; "
                f"attempted={row.outcomes.attempted_count}, "
                f"valid={row.outcomes.valid_count}, failed={row.outcomes.failed_count}, "
                f"invalid={row.outcomes.invalid_count}, "
                f"unsupported={row.outcomes.unsupported_count}, "
                f"timeout={row.outcomes.timeout_count} — {row.failure_reason}"
                for row in panel.retained_noncomplete
            )
        lines.append("")
    if result.retained_noncomplete:
        lines.extend(
            [
                "## All retained non-complete rows",
                "",
            ]
        )
        lines.extend(
            f"- `{row.row_id}` / `{row.status.value}` / "
            f"`{row.source_artifact_id}` `{row.source_locator}` / "
            f"attempted={row.outcomes.attempted_count}, "
            f"valid={row.outcomes.valid_count}, failed={row.outcomes.failed_count}, "
            f"invalid={row.outcomes.invalid_count}, "
            f"unsupported={row.outcomes.unsupported_count}, "
            f"timeout={row.outcomes.timeout_count}: {row.failure_reason}"
            for row in result.retained_noncomplete
        )
        lines.append("")
    if result.unassigned_row_ids:
        lines.extend(
            [
                "## Unassigned/incomparable rows",
                "",
                ", ".join(f"`{row_id}`" for row_id in result.unassigned_row_ids),
                "",
            ]
        )
    return "\n".join(lines)
