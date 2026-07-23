"""Deterministic plot-ready Goal 3 scale and metric-stability data."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import localcontext
from itertools import pairwise
from typing import Final

from geml.analysis.goal3.metrics import (
    AGGREGATE_MEAN_METHOD,
    AGGREGATE_REPORTED_PRECISION_DIGITS,
    AGGREGATE_ROUNDING,
    AGGREGATE_WORKING_PRECISION_DIGITS,
    RATIO_NAMES,
    CheckpointMetrics,
    DecimalMean,
    aggregate_decimal_context,
    aggregate_mean_policy,
)

STANDARD_SCALE_CHECKPOINTS: Final = (10_000, 50_000, 100_000, 250_000)
PEAK_MEMORY_SCOPE: Final = (
    "maximum sampled simultaneous RSS across the runner and live worker descendants"
)


class PlotDataError(ValueError):
    """Saved telemetry cannot support a requested exact scale checkpoint."""


def _finite_nonnegative(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise PlotDataError(f"telemetry has invalid {name}")
    return float(value)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PlotDataError(f"telemetry has invalid {name}")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlotDataError(f"telemetry has invalid {name}")
    return value


@dataclass(frozen=True, slots=True)
class RuntimePoint:
    """Cumulative runtime, throughput, and process-tree RSS at one prefix."""

    processed_count: int
    processing_wall_seconds: float
    throughput_rows_per_second: float | None
    peak_resident_memory_bytes: int
    processing_time_scope: str
    peak_memory_scope: str = PEAK_MEMORY_SCOPE

    def as_dict(self) -> dict[str, object]:
        return {
            "processed_count": self.processed_count,
            "processing_wall_seconds": self.processing_wall_seconds,
            "throughput_rows_per_second": self.throughput_rows_per_second,
            "peak_resident_memory_bytes": self.peak_resident_memory_bytes,
            "processing_time_scope": self.processing_time_scope,
            "peak_memory_scope": self.peak_memory_scope,
        }


@dataclass(frozen=True, slots=True)
class StabilityPoint:
    """Runtime and bounded high-precision means joined at an exact prefix."""

    processed_count: int
    valid_count: int
    failure_count: int
    processing_wall_seconds: float
    throughput_rows_per_second: float | None
    peak_resident_memory_bytes: int
    processing_time_scope: str
    peak_memory_scope: str
    ratio_means: tuple[tuple[str, DecimalMean], ...]

    def ratio_mean(self, name: str) -> DecimalMean | None:
        for candidate, mean in self.ratio_means:
            if candidate == name:
                return mean
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "processed_count": self.processed_count,
            "valid_count": self.valid_count,
            "failure_count": self.failure_count,
            "processing_wall_seconds": self.processing_wall_seconds,
            "throughput_rows_per_second": self.throughput_rows_per_second,
            "peak_resident_memory_bytes": self.peak_resident_memory_bytes,
            "processing_time_scope": self.processing_time_scope,
            "peak_memory_scope": self.peak_memory_scope,
            "ratio_means": {name: mean.as_dict() for name, mean in self.ratio_means},
        }


@dataclass(frozen=True, slots=True)
class StabilityDelta:
    """Approximate change between two reported high-precision means."""

    processed_count: int
    decimal: str
    value: float
    approximate: bool = True
    method: str = "difference_of_reported_decimal_means"

    def as_dict(self) -> dict[str, object]:
        return {
            "processed_count": self.processed_count,
            "decimal": self.decimal,
            "value": self.value,
            "approximate": self.approximate,
            "method": self.method,
        }


def _stability_delta(
    processed_count: int,
    previous: DecimalMean,
    current: DecimalMean,
) -> StabilityDelta:
    context = aggregate_decimal_context(precision=AGGREGATE_REPORTED_PRECISION_DIGITS)
    with localcontext(context):
        difference = +(current.decimal_value - previous.decimal_value)
        decimal_text = str(difference)
    return StabilityDelta(
        processed_count=processed_count,
        decimal=decimal_text,
        value=float(difference),
    )


def build_runtime_curve(
    telemetry_payloads: Iterable[Mapping[str, object]],
) -> tuple[RuntimePoint, ...]:
    """Rebuild cumulative telemetry from validated per-shard sidecars."""

    points: dict[int, RuntimePoint] = {}
    prior_seconds = 0.0
    prior_peak = 0
    processing_time_scope: str | None = None
    for payload in telemetry_payloads:
        payload_scope = payload.get("telemetry_scope")
        if not isinstance(payload_scope, str) or not payload_scope:
            raise PlotDataError("telemetry sidecar has no processing-time scope")
        if processing_time_scope is None:
            processing_time_scope = payload_scope
        elif payload_scope != processing_time_scope:
            raise PlotDataError("telemetry sidecars disagree about processing-time scope")
        samples = payload.get("progress_samples")
        if not isinstance(samples, list) or not samples:
            raise PlotDataError("telemetry sidecar has no progress samples")
        shard_seconds = _finite_nonnegative(
            payload.get("processing_wall_seconds"),
            name="shard processing time",
        )
        shard_peak = _nonnegative_integer(
            payload.get("peak_resident_memory_bytes"),
            name="shard peak memory",
        )
        for sample in samples:
            if not isinstance(sample, dict):
                raise PlotDataError("telemetry progress sample is not an object")
            processed = _positive_integer(
                sample.get("global_processed_count"),
                name="processed count",
            )
            elapsed = prior_seconds + _finite_nonnegative(
                sample.get("processing_wall_seconds"),
                name="progress processing time",
            )
            peak = max(
                prior_peak,
                _nonnegative_integer(
                    sample.get("peak_resident_memory_bytes"),
                    name="progress peak memory",
                ),
            )
            points[processed] = RuntimePoint(
                processed_count=processed,
                processing_wall_seconds=elapsed,
                throughput_rows_per_second=processed / elapsed if elapsed else None,
                peak_resident_memory_bytes=peak,
                processing_time_scope=payload_scope,
            )
        prior_seconds += shard_seconds
        prior_peak = max(prior_peak, shard_peak)

    ordered = tuple(points[count] for count in sorted(points))
    if any(
        later.processing_wall_seconds < earlier.processing_wall_seconds
        or later.peak_resident_memory_bytes < earlier.peak_resident_memory_bytes
        for earlier, later in pairwise(ordered)
    ):
        raise PlotDataError("cumulative runtime or memory telemetry decreases")
    return ordered


def missing_checkpoints(
    points: Iterable[RuntimePoint | StabilityPoint],
    required: Sequence[int] = STANDARD_SCALE_CHECKPOINTS,
) -> tuple[int, ...]:
    present = {point.processed_count for point in points}
    return tuple(checkpoint for checkpoint in required if checkpoint not in present)


def build_stability_curve(
    checkpoints: Iterable[CheckpointMetrics],
    runtime_curve: Iterable[RuntimePoint],
    *,
    required_checkpoints: Sequence[int],
) -> tuple[StabilityPoint, ...]:
    """Join exact metric prefixes to runtime points at the requested counts."""

    required = tuple(required_checkpoints)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in required
    ) or required != tuple(sorted(set(required))):
        raise ValueError("required checkpoints must be positive, sorted, and unique")
    metric_by_count = {point.processed_count: point for point in checkpoints}
    runtime_by_count = {point.processed_count: point for point in runtime_curve}
    missing_metrics = tuple(value for value in required if value not in metric_by_count)
    missing_runtime = tuple(value for value in required if value not in runtime_by_count)
    if missing_metrics or missing_runtime:
        raise PlotDataError(
            "missing exact scale checkpoints: "
            f"metrics={list(missing_metrics)}, runtime={list(missing_runtime)}"
        )

    points: list[StabilityPoint] = []
    for count in required:
        metric = metric_by_count[count]
        runtime = runtime_by_count[count]
        points.append(
            StabilityPoint(
                processed_count=count,
                valid_count=metric.valid_count,
                failure_count=metric.failure_count,
                processing_wall_seconds=runtime.processing_wall_seconds,
                throughput_rows_per_second=runtime.throughput_rows_per_second,
                peak_resident_memory_bytes=runtime.peak_resident_memory_bytes,
                processing_time_scope=runtime.processing_time_scope,
                peak_memory_scope=runtime.peak_memory_scope,
                ratio_means=metric.ratio_means,
            )
        )
    return tuple(points)


def stability_deltas(
    points: Sequence[StabilityPoint],
    metric: str,
) -> tuple[StabilityDelta, ...]:
    """Return bounded high-precision changes between reported prefix means."""

    if metric not in RATIO_NAMES:
        raise ValueError(f"unknown stability metric {metric!r}")
    ordered = tuple(sorted(points, key=lambda point: point.processed_count))
    deltas: list[StabilityDelta] = []
    for previous, current in pairwise(ordered):
        previous_mean = previous.ratio_mean(metric)
        current_mean = current.ratio_mean(metric)
        if previous_mean is None or current_mean is None:
            raise PlotDataError(f"{metric} has no valid-only denominator at a requested checkpoint")
        deltas.append(
            _stability_delta(
                current.processed_count,
                previous_mean,
                current_mean,
            )
        )
    return tuple(deltas)


def plot_data_payload(points: Sequence[StabilityPoint]) -> dict[str, object]:
    """Return deterministic series ready for a plotting backend."""

    ordered = tuple(sorted(points, key=lambda point: point.processed_count))
    metric_means = {
        name: tuple(point.ratio_mean(name) for point in ordered) for name in RATIO_NAMES
    }
    metric_deltas: dict[str, list[dict[str, object]]] = {}
    for name, means in metric_means.items():
        deltas: list[dict[str, object]] = []
        for index, (previous, current) in enumerate(pairwise(means), start=1):
            processed_count = ordered[index].processed_count
            if previous is None or current is None:
                deltas.append(
                    {
                        "processed_count": processed_count,
                        "decimal": None,
                        "value": None,
                        "approximate": True,
                        "method": "difference_of_reported_decimal_means",
                    }
                )
                continue
            deltas.append(
                _stability_delta(
                    processed_count,
                    previous,
                    current,
                ).as_dict()
            )
        metric_deltas[name] = deltas
    return {
        "x_axis": {
            "name": "processed_count",
            "values": [point.processed_count for point in ordered],
        },
        "denominators": {
            "all_processed_count": [point.processed_count for point in ordered],
            "valid_count": [point.valid_count for point in ordered],
            "failure_count": [point.failure_count for point in ordered],
        },
        "runtime": {
            "processing_time_scope": (ordered[0].processing_time_scope if ordered else None),
            "peak_memory_scope": ordered[0].peak_memory_scope if ordered else PEAK_MEMORY_SCOPE,
            "processing_wall_seconds": [point.processing_wall_seconds for point in ordered],
            "throughput_rows_per_second": [point.throughput_rows_per_second for point in ordered],
            "peak_resident_memory_bytes": [point.peak_resident_memory_bytes for point in ordered],
        },
        "aggregate_mean_policy": aggregate_mean_policy(),
        "metric_stability": {
            name: {
                "decimal": [
                    mean.decimal if mean is not None else None for mean in metric_means[name]
                ],
                "value": [mean.value if mean is not None else None for mean in metric_means[name]],
                "approximate": True,
                "method": AGGREGATE_MEAN_METHOD,
                "working_precision_digits": AGGREGATE_WORKING_PRECISION_DIGITS,
                "reported_precision_digits": AGGREGATE_REPORTED_PRECISION_DIGITS,
                "rounding": AGGREGATE_ROUNDING,
                "valid_denominator": [
                    mean.sample_count if mean is not None else 0 for mean in metric_means[name]
                ],
                "difference_from_previous": metric_deltas[name],
            }
            for name in RATIO_NAMES
        },
    }
