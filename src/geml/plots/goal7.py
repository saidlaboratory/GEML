"""Deterministic plots from an authenticated Goal 7 summary."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from geml.analysis.goal7.summary import Goal7SummaryV1

PLOT_DATA_SCHEMA_VERSION = "geml-goal7-plot-data-v1"


class PlotDependencyError(RuntimeError):
    """Plot rendering was requested without matplotlib."""


@dataclass(frozen=True, slots=True)
class Goal7PlotPoint:
    arm_id: str
    seed: int
    status: str
    exact_action_top1: float | None
    exact_successor_top1: float | None
    verifier_valid_top1: float | None
    parameter_count: int | None
    estimated_flops: float | None
    wall_time_seconds: float
    peak_host_memory_bytes: int | None
    peak_device_memory_bytes: int | None


@dataclass(frozen=True, slots=True)
class Goal7PlotData:
    source_summary_digest: str
    points: tuple[Goal7PlotPoint, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": PLOT_DATA_SCHEMA_VERSION,
            "source_summary_digest": self.source_summary_digest,
            "points": [
                {
                    "arm_id": point.arm_id,
                    "seed": point.seed,
                    "status": point.status,
                    "exact_action_top1": point.exact_action_top1,
                    "exact_successor_top1": point.exact_successor_top1,
                    "verifier_valid_top1": point.verifier_valid_top1,
                    "parameter_count": point.parameter_count,
                    "estimated_flops": point.estimated_flops,
                    "wall_time_seconds": point.wall_time_seconds,
                    "peak_host_memory_bytes": point.peak_host_memory_bytes,
                    "peak_device_memory_bytes": point.peak_device_memory_bytes,
                }
                for point in self.points
            ],
        }


def build_plot_data(summary: Goal7SummaryV1) -> Goal7PlotData:
    """Extract exact seed-level rates without averaging precomputed rates."""

    if not isinstance(summary, Goal7SummaryV1):
        raise TypeError("summary must be Goal7SummaryV1")
    payload = summary.as_dict()
    raw_rows = payload.get("raw_seed_rows")
    if not isinstance(raw_rows, list):
        raise ValueError("Goal 7 summary has no raw seed rows")
    points: list[Goal7PlotPoint] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("raw seed row must be an object")
        top1 = _top1(raw.get("aggregate"))
        host = _optional_integer(raw.get("peak_host_memory_bytes"))
        device = _optional_integer(raw.get("peak_device_memory_bytes"))
        points.append(
            Goal7PlotPoint(
                arm_id=_string(raw, "arm_id"),
                seed=_integer(raw, "seed"),
                status=_string(raw, "status"),
                exact_action_top1=_optional_rate(
                    None if top1 is None else top1.get("demonstration_action_match_rate_all")
                ),
                exact_successor_top1=_optional_rate(
                    None if top1 is None else top1.get("exact_successor_structure_match_rate_all")
                ),
                verifier_valid_top1=_optional_rate(
                    None if top1 is None else top1.get("verifier_valid_success_rate_all")
                ),
                parameter_count=_optional_integer(raw.get("parameter_count")),
                estimated_flops=_optional_number(raw.get("estimated_flops")),
                wall_time_seconds=_number(raw, "wall_time_seconds"),
                peak_host_memory_bytes=host,
                peak_device_memory_bytes=device,
            )
        )
    digest = payload.get("content_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Goal 7 summary lacks its content digest")
    return Goal7PlotData(
        source_summary_digest=digest,
        points=tuple(points),
    )


def render_goal7_plots(
    data: Goal7PlotData,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Render metric, compute, and retained-status panels."""

    if not isinstance(data, Goal7PlotData):
        raise TypeError("data must be Goal7PlotData")
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:  # pragma: no cover - optional environment
        raise PlotDependencyError("matplotlib is required to render Goal 7 plots") from error

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    arms = sorted({point.arm_id for point in data.points})
    colors = {
        "exact action": "#3366cc",
        "exact successor": "#dc3912",
        "verifier valid": "#109618",
    }

    metric_path = output / "goal7_top1_by_seed.png"
    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    offsets = {"exact action": -0.18, "exact successor": 0.0, "verifier valid": 0.18}
    seed_offsets = _seed_offsets(data.points)
    for label, attribute in (
        ("exact action", "exact_action_top1"),
        ("exact successor", "exact_successor_top1"),
        ("verifier valid", "verifier_valid_top1"),
    ):
        first = True
        for point in data.points:
            value = getattr(point, attribute)
            if value is None:
                continue
            x = arms.index(point.arm_id) + offsets[label] + seed_offsets[point.seed]
            axis.scatter(
                x,
                value,
                color=colors[label],
                label=label if first else None,
                marker="o",
                s=34,
            )
            first = False
    axis.set_xticks(range(len(arms)), arms, rotation=25, ha="right")
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("rate over all frozen step rows")
    axis.set_title("Goal 7 top-1 metrics (one point per seed)")
    if axis.has_data():
        axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(metric_path, dpi=160)
    plt.close(figure)

    compute_path = output / "goal7_compute_by_seed.png"
    figure, axes_grid = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = tuple(axes_grid.flat)
    panels = (
        ("wall_time_seconds", "wall time (seconds)", "linear"),
        ("parameter_count", "parameters", "log"),
        ("estimated_flops", "estimated FLOPs", "log"),
        ("peak_host_memory_bytes", "peak host memory (bytes)", "log"),
        ("peak_device_memory_bytes", "peak device memory (bytes)", "log"),
    )
    for axis, (attribute, title, scale) in zip(
        axes[: len(panels)],
        panels,
        strict=True,
    ):
        for point in data.points:
            value = getattr(point, attribute)
            if value is None or (scale == "log" and value <= 0):
                continue
            x = arms.index(point.arm_id) + seed_offsets[point.seed]
            axis.scatter(x, value, color="#3366cc", s=30)
        axis.set_xticks(range(len(arms)), arms, rotation=30, ha="right")
        axis.set_title(title)
        axis.set_yscale(scale)
        axis.grid(axis="y", alpha=0.25)
    for axis in axes[len(panels) :]:
        axis.remove()
    figure.savefig(compute_path, dpi=160)
    plt.close(figure)

    status_path = output / "goal7_cell_status.png"
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    statuses = sorted({point.status for point in data.points})
    bottoms = [0] * len(arms)
    palette = ("#109618", "#dc3912", "#ff9900", "#990099", "#0099c6", "#777777")
    for color, status in zip(palette, statuses, strict=False):
        counts = Counter(point.arm_id for point in data.points if point.status == status)
        values = [counts[arm] for arm in arms]
        axis.bar(arms, values, bottom=bottoms, label=status, color=color)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    axis.set_ylabel("retained cells")
    axis.set_title("Goal 7 cell outcomes (failures are retained)")
    axis.tick_params(axis="x", rotation=25)
    if statuses:
        axis.legend()
    figure.savefig(status_path, dpi=160)
    plt.close(figure)
    receipt_path = output / "goal7_plot_receipt.json"
    plotted = (metric_path, compute_path, status_path)
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": PLOT_DATA_SCHEMA_VERSION,
                "source_summary_digest": data.source_summary_digest,
                "files": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in plotted
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return *plotted, receipt_path


def _top1(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("cell aggregate must be an object or null")
    rows = value.get("top_k")
    if not isinstance(rows, list):
        raise ValueError("cell aggregate top_k must be a list")
    for row in rows:
        if isinstance(row, Mapping) and row.get("k") == 1:
            return row
    raise ValueError("complete Goal 7 aggregate lacks top-1 metrics")


def _seed_offsets(points: tuple[Goal7PlotPoint, ...]) -> dict[int, float]:
    seeds = sorted({point.seed for point in points})
    midpoint = (len(seeds) - 1) / 2
    return {seed: (index - midpoint) * 0.04 for index, seed in enumerate(seeds)}


def _string(value: Mapping[str, object], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{field} must be a nonempty string")
    return result


def _integer(value: Mapping[str, object], field: str) -> int:
    result = value.get(field)
    if type(result) is not int:
        raise ValueError(f"{field} must be an integer")
    return result


def _number(value: Mapping[str, object], field: str) -> float:
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int | float):
        raise ValueError(f"{field} must be numeric")
    return float(result)


def _optional_integer(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("optional integer is invalid")
    return value


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("optional number is invalid")
    return float(value)


def _optional_rate(value: object) -> float | None:
    result = _optional_number(value)
    if result is not None and not 0 <= result <= 1:
        raise ValueError("rate must be in [0, 1]")
    return result


__all__ = [
    "PLOT_DATA_SCHEMA_VERSION",
    "Goal7PlotData",
    "Goal7PlotPoint",
    "PlotDependencyError",
    "build_plot_data",
    "render_goal7_plots",
]
