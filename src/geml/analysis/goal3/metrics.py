"""Reproducible Goal 3 analysis over validated saved DAG-run artifacts.

The analysis is deliberately structural.  It reports exact-sharing ratios,
reuse accounting, runtime, and resource measurements without making claims
about transformations that the Goal 3 experiment did not perform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Final

from geml.experiments.goal3.run import (
    iter_metric_tables,
    iter_shard_telemetry,
)
from geml.experiments.goal3.runtime import (
    Goal3ArtifactError,
    atomic_write_json,
    canonical_json,
    load_json_mapping,
    sha256_file,
)

if TYPE_CHECKING:
    from geml.analysis.goal3.failures import OutcomeReport
    from geml.plots.goal3 import StabilityPoint

ANALYSIS_SCHEMA_VERSION: Final = "geml-goal3-analysis-v1"
ANALYSIS_ARTIFACT_SCHEMA_VERSION: Final = "geml-goal3-analysis-artifacts-v1"
UNAVAILABLE_STRATUM: Final = "<unavailable>"
RATIO_NAMES: Final = (
    "raw_tree_alpha",
    "dag_alpha_vs_ast_tree",
    "dag_alpha_vs_ast_dag",
    "ast_compression",
    "eml_compression",
)


class AnalysisArtifactError(RuntimeError):
    """A validated Goal 3 run cannot be analyzed or safely exported."""


class StratificationAxis(StrEnum):
    """Independent grouping dimensions required by the Goal 3 analysis."""

    FAMILY = "family"
    OPERATOR_SIGNATURE = "operator_signature"
    ACTUAL_AST_SIZE = "actual_ast_size"
    ACTUAL_AST_DEPTH = "actual_ast_depth"
    SPLIT = "split"
    DOMAIN = "domain"
    REUSE_PATTERN = "reuse_pattern"


class ReusePattern(StrEnum):
    """Which exact-sharing paths contain at least one reused node."""

    NONE = "none"
    AST_ONLY = "ast_only"
    EML_ONLY = "eml_only"
    BOTH = "both"


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnalysisArtifactError(f"metric row has invalid {name}")
    return value


def _optional_text(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name=name)


def _required_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnalysisArtifactError(f"metric row has invalid {name}")
    return value


def _optional_integer(value: object, *, name: str, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _required_integer(value, name=name, minimum=minimum)


def _fraction_from_columns(row: Mapping[str, object], prefix: str) -> Fraction:
    try:
        value = Fraction(
            int(_required_text(row.get(f"{prefix}_numerator"), name=f"{prefix} numerator")),
            int(_required_text(row.get(f"{prefix}_denominator"), name=f"{prefix} denominator")),
        )
    except (ValueError, ZeroDivisionError) as error:
        raise AnalysisArtifactError(f"metric row has invalid exact {prefix}") from error
    expected = f"{value.numerator}/{value.denominator}"
    if row.get(f"{prefix}_exact") != expected:
        raise AnalysisArtifactError(f"metric row has noncanonical exact {prefix}")
    approximate = row.get(f"{prefix}_value")
    if not isinstance(approximate, float) or approximate != float(value):
        raise AnalysisArtifactError(f"metric row has inconsistent float {prefix}")
    return value


@dataclass(frozen=True, slots=True)
class ExactMean:
    """An exact arithmetic mean with its explicit valid-only denominator."""

    numerator: int
    denominator: int
    sample_count: int

    def __post_init__(self) -> None:
        if self.denominator <= 0 or self.sample_count <= 0:
            raise ValueError("exact means require positive denominators and sample counts")
        canonical = Fraction(self.numerator, self.denominator)
        if (canonical.numerator, canonical.denominator) != (
            self.numerator,
            self.denominator,
        ):
            raise ValueError("exact means must be stored in canonical form")

    @classmethod
    def from_total(cls, total: Fraction | int, sample_count: int) -> ExactMean:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        mean = Fraction(total, sample_count)
        return cls(
            numerator=mean.numerator,
            denominator=mean.denominator,
            sample_count=sample_count,
        )

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    @property
    def exact(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    @property
    def value(self) -> float:
        return float(self.fraction)

    def as_dict(self) -> dict[str, object]:
        return {
            "numerator": str(self.numerator),
            "denominator": str(self.denominator),
            "exact": self.exact,
            "value": self.value,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class RowReuse:
    """Validated per-expression reuse measurements for one representation."""

    reused_node_count: int
    reused_reference_count: int
    excess_reference_count: int
    child_reference_overhead: int
    max_reuse_indegree: int
    reuse_depth_sum: int
    reuse_depth_count: int
    sharing_concentration: Fraction

    @classmethod
    def from_mapping(cls, row: Mapping[str, object], prefix: str) -> RowReuse:
        reused_nodes = _required_integer(
            row.get(f"{prefix}_reused_node_count"),
            name=f"{prefix} reused-node count",
        )
        reused_references = _required_integer(
            row.get(f"{prefix}_reused_reference_count"),
            name=f"{prefix} reused-reference count",
        )
        child_overhead = _required_integer(
            row.get(f"{prefix}_child_reference_overhead"),
            name=f"{prefix} child-reference overhead",
        )
        max_reuse = _required_integer(
            row.get(f"{prefix}_max_reuse_count"),
            name=f"{prefix} max-reuse indegree",
        )
        depth_sum = _required_integer(
            row.get(f"{prefix}_reuse_depth_sum"),
            name=f"{prefix} reuse-depth sum",
        )
        depth_count = _required_integer(
            row.get(f"{prefix}_reuse_depth_count"),
            name=f"{prefix} reuse-depth count",
        )
        excess = reused_references - reused_nodes
        if excess < 0 or excess != child_overhead:
            raise AnalysisArtifactError(
                f"{prefix} reused-reference and child-overhead accounting disagree"
            )
        if depth_count != reused_nodes:
            raise AnalysisArtifactError(f"{prefix} reuse-depth denominator is inconsistent")
        if reused_nodes == 0:
            if any((reused_references, child_overhead, max_reuse, depth_sum, depth_count)):
                raise AnalysisArtifactError(f"{prefix} no-reuse row has nonzero reuse values")
        elif max_reuse < 2:
            raise AnalysisArtifactError(f"{prefix} reused nodes require indegree at least two")
        return cls(
            reused_node_count=reused_nodes,
            reused_reference_count=reused_references,
            excess_reference_count=excess,
            child_reference_overhead=child_overhead,
            max_reuse_indegree=max_reuse,
            reuse_depth_sum=depth_sum,
            reuse_depth_count=depth_count,
            sharing_concentration=_fraction_from_columns(
                row,
                f"{prefix}_sharing_concentration",
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidMetrics:
    """All exact ratios and reuse measurements present on a successful row."""

    ratios: tuple[tuple[str, Fraction], ...]
    ast_reuse: RowReuse
    eml_reuse: RowReuse

    def ratio(self, name: str) -> Fraction:
        for candidate, value in self.ratios:
            if candidate == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class AnalysisRow:
    """Typed analysis view of one validated Goal 3 metric row."""

    expression_id: str
    status: str
    family: str
    split: str
    domain: str
    operator_signature: str | None
    actual_ast_size: int | None
    actual_ast_depth: int | None
    input_shard_id: str
    input_shard_path: str
    input_row_index: int
    metrics: ValidMetrics | None
    error_stage: str | None
    error_type: str | None
    error_message: str | None

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> AnalysisRow:
        status = _required_text(row.get("status"), name="status")
        if status not in {"success", "failure"}:
            raise AnalysisArtifactError(f"unknown metric-row status {status!r}")
        metrics: ValidMetrics | None = None
        if status == "success":
            metrics = ValidMetrics(
                ratios=tuple((name, _fraction_from_columns(row, name)) for name in RATIO_NAMES),
                ast_reuse=RowReuse.from_mapping(row, "ast_dag"),
                eml_reuse=RowReuse.from_mapping(row, "eml_dag"),
            )
        result = cls(
            expression_id=_required_text(row.get("expression_id"), name="expression ID"),
            status=status,
            family=_required_text(row.get("operator_family"), name="family"),
            split=_required_text(row.get("split"), name="split"),
            domain=_required_text(row.get("domain_mode"), name="domain"),
            operator_signature=_optional_text(
                row.get("operator_signature"),
                name="operator signature",
            ),
            actual_ast_size=_optional_integer(
                row.get("ast_tree_node_count"),
                name="actual AST size",
                minimum=1,
            ),
            actual_ast_depth=_optional_integer(
                row.get("ast_tree_depth"),
                name="actual AST depth",
            ),
            input_shard_id=_required_text(
                row.get("input_shard_id"),
                name="input shard ID",
            ),
            input_shard_path=_required_text(
                row.get("input_shard_path"),
                name="input shard path",
            ),
            input_row_index=_required_integer(
                row.get("input_row_index"),
                name="input row index",
            ),
            metrics=metrics,
            error_stage=_optional_text(row.get("error_stage"), name="error stage"),
            error_type=_optional_text(row.get("error_type"), name="error type"),
            error_message=_optional_text(row.get("error_message"), name="error message"),
        )
        if status == "success":
            if result.operator_signature is None:
                raise AnalysisArtifactError("successful row has no operator signature")
            if result.actual_ast_size is None or result.actual_ast_depth is None:
                raise AnalysisArtifactError("successful row has no actual AST size/depth")
            if any(
                value is not None
                for value in (
                    result.error_stage,
                    result.error_type,
                    result.error_message,
                )
            ):
                raise AnalysisArtifactError("successful row contains failure diagnostics")
        elif not all(
            value is not None
            for value in (
                result.error_stage,
                result.error_type,
                result.error_message,
            )
        ):
            raise AnalysisArtifactError("failure row lacks detailed diagnostics")
        return result

    @property
    def valid(self) -> bool:
        return self.status == "success"

    @property
    def reuse_pattern(self) -> ReusePattern | None:
        if self.metrics is None:
            return None
        ast = self.metrics.ast_reuse.reused_node_count > 0
        eml = self.metrics.eml_reuse.reused_node_count > 0
        if ast and eml:
            return ReusePattern.BOTH
        if ast:
            return ReusePattern.AST_ONLY
        if eml:
            return ReusePattern.EML_ONLY
        return ReusePattern.NONE

    def stratum(self, axis: StratificationAxis) -> str:
        values: dict[StratificationAxis, object | None] = {
            StratificationAxis.FAMILY: self.family,
            StratificationAxis.OPERATOR_SIGNATURE: self.operator_signature,
            StratificationAxis.ACTUAL_AST_SIZE: self.actual_ast_size,
            StratificationAxis.ACTUAL_AST_DEPTH: self.actual_ast_depth,
            StratificationAxis.SPLIT: self.split,
            StratificationAxis.DOMAIN: self.domain,
            StratificationAxis.REUSE_PATTERN: (
                self.reuse_pattern.value if self.reuse_pattern is not None else None
            ),
        }
        value = values[axis]
        return UNAVAILABLE_STRATUM if value is None else str(value)


@dataclass(frozen=True, slots=True)
class ReuseAggregate:
    """Valid-only aggregate of one representation's exact-sharing behavior."""

    total_reused_node_count: int
    total_reused_reference_count: int
    total_excess_reference_count: int
    total_child_reference_overhead: int
    max_reuse_indegree: int
    mean_reused_node_count: ExactMean
    mean_reused_reference_count: ExactMean
    mean_excess_reference_count: ExactMean
    mean_child_reference_overhead: ExactMean
    mean_reuse_depth: ExactMean | None
    mean_sharing_concentration: ExactMean

    def as_dict(self) -> dict[str, object]:
        return {
            "total_reused_node_count": self.total_reused_node_count,
            "total_reused_reference_count": self.total_reused_reference_count,
            "total_excess_reference_count": self.total_excess_reference_count,
            "total_child_reference_overhead": self.total_child_reference_overhead,
            "max_reuse_indegree": self.max_reuse_indegree,
            "mean_reused_node_count": self.mean_reused_node_count.as_dict(),
            "mean_reused_reference_count": self.mean_reused_reference_count.as_dict(),
            "mean_excess_reference_count": self.mean_excess_reference_count.as_dict(),
            "mean_child_reference_overhead": self.mean_child_reference_overhead.as_dict(),
            "mean_reuse_depth": (
                self.mean_reuse_depth.as_dict() if self.mean_reuse_depth is not None else None
            ),
            "mean_sharing_concentration": self.mean_sharing_concentration.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class GroupStatistics:
    """One stratum with honest all-processed and valid-only denominators."""

    key: str
    all_processed_count: int
    valid_count: int
    failure_count: int
    ratio_means: tuple[tuple[str, ExactMean], ...]
    ast_reuse: ReuseAggregate | None
    eml_reuse: ReuseAggregate | None

    def __post_init__(self) -> None:
        if self.all_processed_count != self.valid_count + self.failure_count:
            raise ValueError("group denominators do not account for every processed row")

    def ratio_mean(self, name: str) -> ExactMean | None:
        for candidate, value in self.ratio_means:
            if candidate == name:
                return value
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "all_processed_count": self.all_processed_count,
            "valid_count": self.valid_count,
            "failure_count": self.failure_count,
            "ratio_means": {name: value.as_dict() for name, value in self.ratio_means},
            "ast_reuse": self.ast_reuse.as_dict() if self.ast_reuse else None,
            "eml_reuse": self.eml_reuse.as_dict() if self.eml_reuse else None,
        }


@dataclass(frozen=True, slots=True)
class StratifiedTable:
    """Deterministically ordered groups for one independent axis."""

    axis: StratificationAxis
    groups: tuple[GroupStatistics, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis.value,
            "groups": [group.as_dict() for group in self.groups],
        }


@dataclass(frozen=True, slots=True)
class CheckpointMetrics:
    """Exact cumulative metric state at one authoritative corpus prefix."""

    processed_count: int
    valid_count: int
    failure_count: int
    ratio_means: tuple[tuple[str, ExactMean], ...]

    def ratio_mean(self, name: str) -> ExactMean | None:
        for candidate, value in self.ratio_means:
            if candidate == name:
                return value
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "processed_count": self.processed_count,
            "valid_count": self.valid_count,
            "failure_count": self.failure_count,
            "ratio_means": {name: value.as_dict() for name, value in self.ratio_means},
        }


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Complete structural, failure, and scale analysis of one saved run."""

    schema_version: str
    source_manifest_path: str
    source_manifest_sha256: str
    source_science_fingerprint: str
    source_stage: str
    overall: GroupStatistics
    strata: tuple[StratifiedTable, ...]
    checkpoints: tuple[CheckpointMetrics, ...]
    stability_curve: tuple[StabilityPoint, ...]
    outcomes: OutcomeReport
    missing_standard_checkpoints: tuple[int, ...]
    fingerprint: str

    def metrics_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": {
                "manifest_path": self.source_manifest_path,
                "manifest_sha256": self.source_manifest_sha256,
                "science_fingerprint": self.source_science_fingerprint,
                "stage": self.source_stage,
            },
            "metric_definitions": {
                "ratio_means": "arithmetic mean of exact per-expression ratios",
                "reuse_depth": (
                    "reused-node-weighted mean of minimum root-to-node child-reference distance"
                ),
                "reused_reference_count": (
                    "sum of indegrees over nodes whose indegree is greater than one"
                ),
                "excess_reference_count": "sum of indegree minus one over reused nodes",
                "sharing_concentration": (
                    "per-expression max excess indegree divided by total excess indegree, "
                    "then averaged over valid expressions"
                ),
                "child_reference_overhead": (
                    "child-reference count beyond a rooted tree; equal to excess references"
                ),
            },
            "overall": self.overall.as_dict(),
            "strata": [table.as_dict() for table in self.strata],
            "checkpoints": [checkpoint.as_dict() for checkpoint in self.checkpoints],
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self.metrics_payload(),
            "stability_curve": [point.as_dict() for point in self.stability_curve],
            "outcomes": self.outcomes.as_dict(),
            "missing_standard_checkpoints": list(self.missing_standard_checkpoints),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class SavedAnalysis:
    """Paths for deterministic analysis products written outside run artifacts."""

    output_directory: Path
    manifest_path: Path
    metrics_path: Path
    outcomes_path: Path
    plot_data_path: Path
    analysis_fingerprint: str


@dataclass(slots=True)
class _ReuseAccumulator:
    valid_count: int = 0
    reused_nodes: int = 0
    reused_references: int = 0
    excess_references: int = 0
    child_overhead: int = 0
    max_reuse_indegree: int = 0
    reuse_depth_sum: int = 0
    reuse_depth_count: int = 0
    concentration_sum: Fraction = field(default_factory=Fraction)

    def add(self, reuse: RowReuse) -> None:
        self.valid_count += 1
        self.reused_nodes += reuse.reused_node_count
        self.reused_references += reuse.reused_reference_count
        self.excess_references += reuse.excess_reference_count
        self.child_overhead += reuse.child_reference_overhead
        self.max_reuse_indegree = max(
            self.max_reuse_indegree,
            reuse.max_reuse_indegree,
        )
        self.reuse_depth_sum += reuse.reuse_depth_sum
        self.reuse_depth_count += reuse.reuse_depth_count
        self.concentration_sum += reuse.sharing_concentration

    def finish(self) -> ReuseAggregate:
        if self.valid_count <= 0:
            raise ValueError("cannot finish an empty reuse aggregate")
        return ReuseAggregate(
            total_reused_node_count=self.reused_nodes,
            total_reused_reference_count=self.reused_references,
            total_excess_reference_count=self.excess_references,
            total_child_reference_overhead=self.child_overhead,
            max_reuse_indegree=self.max_reuse_indegree,
            mean_reused_node_count=ExactMean.from_total(
                self.reused_nodes,
                self.valid_count,
            ),
            mean_reused_reference_count=ExactMean.from_total(
                self.reused_references,
                self.valid_count,
            ),
            mean_excess_reference_count=ExactMean.from_total(
                self.excess_references,
                self.valid_count,
            ),
            mean_child_reference_overhead=ExactMean.from_total(
                self.child_overhead,
                self.valid_count,
            ),
            mean_reuse_depth=(
                ExactMean.from_total(self.reuse_depth_sum, self.reuse_depth_count)
                if self.reuse_depth_count
                else None
            ),
            mean_sharing_concentration=ExactMean.from_total(
                self.concentration_sum,
                self.valid_count,
            ),
        )


@dataclass(slots=True)
class _GroupAccumulator:
    all_processed_count: int = 0
    valid_count: int = 0
    ratio_totals: dict[str, Fraction] = field(
        default_factory=lambda: {name: Fraction() for name in RATIO_NAMES}
    )
    ast_reuse: _ReuseAccumulator = field(default_factory=_ReuseAccumulator)
    eml_reuse: _ReuseAccumulator = field(default_factory=_ReuseAccumulator)

    def add(self, row: AnalysisRow) -> None:
        self.all_processed_count += 1
        if row.metrics is None:
            return
        self.valid_count += 1
        for name in RATIO_NAMES:
            self.ratio_totals[name] += row.metrics.ratio(name)
        self.ast_reuse.add(row.metrics.ast_reuse)
        self.eml_reuse.add(row.metrics.eml_reuse)

    def finish(self, key: str) -> GroupStatistics:
        ratio_means = (
            tuple(
                (
                    name,
                    ExactMean.from_total(self.ratio_totals[name], self.valid_count),
                )
                for name in RATIO_NAMES
            )
            if self.valid_count
            else ()
        )
        return GroupStatistics(
            key=key,
            all_processed_count=self.all_processed_count,
            valid_count=self.valid_count,
            failure_count=self.all_processed_count - self.valid_count,
            ratio_means=ratio_means,
            ast_reuse=self.ast_reuse.finish() if self.valid_count else None,
            eml_reuse=self.eml_reuse.finish() if self.valid_count else None,
        )


def _checkpoint_snapshot(
    processed_count: int,
    accumulator: _GroupAccumulator,
) -> CheckpointMetrics:
    group = accumulator.finish("all")
    return CheckpointMetrics(
        processed_count=processed_count,
        valid_count=group.valid_count,
        failure_count=group.failure_count,
        ratio_means=group.ratio_means,
    )


def analyze_goal3_artifacts(
    manifest_path: str | Path,
    *,
    scale_checkpoints: Sequence[int] | None = None,
    ranking_limit: int = 20,
) -> AnalysisReport:
    """Analyze a validated saved Goal 3 run without modifying its artifacts."""

    from geml.analysis.goal3.failures import OutcomeMiner
    from geml.plots.goal3 import (
        STANDARD_SCALE_CHECKPOINTS,
        build_runtime_curve,
        build_stability_curve,
    )

    source = Path(manifest_path).resolve()
    source_manifest_sha256 = sha256_file(source)
    manifest = load_json_mapping(source, label="Goal 3 completion manifest")
    processed_expected = _required_integer(
        manifest.get("processed_count"),
        name="manifest processed count",
        minimum=1,
    )
    success_expected = _required_integer(
        manifest.get("success_count"),
        name="manifest success count",
    )
    failure_expected = _required_integer(
        manifest.get("failure_count"),
        name="manifest failure count",
    )
    if processed_expected != success_expected + failure_expected:
        raise AnalysisArtifactError("Goal 3 summary denominators are inconsistent")

    requested = (
        tuple(scale_checkpoints) if scale_checkpoints is not None else STANDARD_SCALE_CHECKPOINTS
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in requested
    ) or requested != tuple(sorted(set(requested))):
        raise ValueError("scale checkpoints must be positive, sorted, and unique")
    active_checkpoints = tuple(value for value in requested if value <= processed_expected)

    overall = _GroupAccumulator()
    by_axis: dict[StratificationAxis, dict[str, _GroupAccumulator]] = {
        axis: defaultdict(_GroupAccumulator) for axis in StratificationAxis
    }
    checkpoints: list[CheckpointMetrics] = []
    checkpoint_set = set(active_checkpoints)
    outcomes = OutcomeMiner(limit=ranking_limit)
    processed = 0
    for table in iter_metric_tables(source):
        for batch in table.to_batches(max_chunksize=2_048):
            for mapping in batch.to_pylist():
                row = AnalysisRow.from_mapping(mapping)
                processed += 1
                overall.add(row)
                for axis, groups in by_axis.items():
                    groups[row.stratum(axis)].add(row)
                outcomes.add(row)
                if processed in checkpoint_set:
                    checkpoints.append(_checkpoint_snapshot(processed, overall))

    overall_finished = overall.finish("all")
    if (
        processed != processed_expected
        or overall_finished.valid_count != success_expected
        or overall_finished.failure_count != failure_expected
    ):
        raise AnalysisArtifactError("analysis denominators differ from the validated summary")
    missing_metric_checkpoints = sorted(
        checkpoint_set - {checkpoint.processed_count for checkpoint in checkpoints}
    )
    if missing_metric_checkpoints:
        raise AnalysisArtifactError(
            f"metric rows do not reach checkpoints {missing_metric_checkpoints}"
        )

    strata = tuple(
        StratifiedTable(
            axis=axis,
            groups=tuple(groups[key].finish(key) for key in sorted(groups)),
        )
        for axis, groups in by_axis.items()
    )
    runtime_curve = build_runtime_curve(iter_shard_telemetry(source))
    if sha256_file(source) != source_manifest_sha256:
        raise AnalysisArtifactError("Goal 3 manifest changed while it was being analyzed")
    stability_curve = build_stability_curve(
        checkpoints,
        runtime_curve,
        required_checkpoints=active_checkpoints,
    )
    standard_present = {point.processed_count for point in stability_curve}
    missing_standard = tuple(
        checkpoint
        for checkpoint in STANDARD_SCALE_CHECKPOINTS
        if checkpoint <= processed_expected and checkpoint not in standard_present
    )
    summary_descriptor = manifest.get("summary")
    if not isinstance(summary_descriptor, dict):
        raise AnalysisArtifactError("validated manifest has no summary descriptor")
    summary_payload = load_json_mapping(
        source.parent / str(summary_descriptor.get("path", "")),
        label="validated Goal 3 summary",
    )
    source_science_fingerprint = _required_text(
        summary_payload.get("science_fingerprint"),
        name="summary science fingerprint",
    )
    provisional = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "source_manifest_path": source.as_posix(),
        "source_manifest_sha256": source_manifest_sha256,
        "source_science_fingerprint": source_science_fingerprint,
        "source_stage": _required_text(manifest.get("stage"), name="source stage"),
        "overall": overall_finished,
        "strata": strata,
        "checkpoints": tuple(checkpoints),
        "stability_curve": stability_curve,
        "outcomes": outcomes.finish(),
        "missing_standard_checkpoints": missing_standard,
    }
    report_without_fingerprint = AnalysisReport(
        **provisional,
        fingerprint="",
    )
    fingerprint_payload = report_without_fingerprint.as_dict()
    fingerprint_payload.pop("fingerprint")
    fingerprint_payload["source"].pop("manifest_path")
    fingerprint = hashlib.sha256(canonical_json(fingerprint_payload).encode("utf-8")).hexdigest()
    return AnalysisReport(**provisional, fingerprint=fingerprint)


def _assert_external_output(source_manifest: Path, output_directory: Path) -> None:
    source_root = source_manifest.resolve().parent
    destination = output_directory.resolve()
    if destination == source_root or source_root in destination.parents:
        raise AnalysisArtifactError(
            "analysis output must be outside the read-only Goal 3 run artifact tree"
        )


def save_analysis(
    report: AnalysisReport,
    output_directory: str | Path,
) -> SavedAnalysis:
    """Write deterministic tables and plot-ready data outside source artifacts."""

    from geml.plots.goal3 import plot_data_payload

    source = Path(report.source_manifest_path)
    destination = Path(output_directory).resolve()
    _assert_external_output(source, destination)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.table.json"
    outcomes_path = destination / "outcomes.table.json"
    plot_data_path = destination / "stability.plot-data.json"
    manifest_path = destination / "analysis.manifest.json"

    atomic_write_json(metrics_path, report.metrics_payload(), resume_identical=True)
    atomic_write_json(
        outcomes_path,
        report.outcomes.as_dict(),
        resume_identical=True,
    )
    atomic_write_json(
        plot_data_path,
        plot_data_payload(report.stability_curve),
        resume_identical=True,
    )
    products = tuple(
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
        }
        for path in (metrics_path, outcomes_path, plot_data_path)
    )
    atomic_write_json(
        manifest_path,
        {
            "schema_version": ANALYSIS_ARTIFACT_SCHEMA_VERSION,
            "analysis_fingerprint": report.fingerprint,
            "source_manifest_path": report.source_manifest_path,
            "source_manifest_sha256": report.source_manifest_sha256,
            "source_science_fingerprint": report.source_science_fingerprint,
            "products": list(products),
            "reproduction_command": (
                "python -m geml.analysis.goal3.metrics "
                f"--manifest {report.source_manifest_path} "
                f"--output-dir {destination.as_posix()}"
            ),
        },
        resume_identical=True,
    )
    return SavedAnalysis(
        output_directory=destination,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        outcomes_path=outcomes_path,
        plot_data_path=plot_data_path,
        analysis_fingerprint=report.fingerprint,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ranking-limit", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run reproducible analysis and print its deterministic product manifest."""

    arguments = _parser().parse_args(argv)
    try:
        report = analyze_goal3_artifacts(
            arguments.manifest,
            ranking_limit=arguments.ranking_limit,
        )
        saved = save_analysis(report, arguments.output_dir)
    except (AnalysisArtifactError, Goal3ArtifactError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                indent=2,
                sort_keys=True,
            ),
            file=os.sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "passed": True,
                "analysis_fingerprint": saved.analysis_fingerprint,
                "manifest_path": saved.manifest_path.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
