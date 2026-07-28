"""Issue 6-6: Goal 6 figures rendered only from saved result rows.

Every plotted value must be reproducible from the rows on disk, so these functions take the
already-validated analysis payload rather than recomputing anything of their own.

Two refusals are enforced at render time:

* a metric flagged ``comparable_across_channels = false`` cannot be drawn on a shared axis with a
  metric from a different representation mode;
* a missing or failed cell is drawn as a visible gap with an annotation, never as a zero bar.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from geml.analysis.goal6.summary import (
    AnalysisError,
    assert_not_pooled_across_representations,
)

FIGURE_DPI = 150


def _raw_seed_points(aggregate: Mapping[str, object]) -> list[float]:
    return [float(value) for value in dict(aggregate["raw_by_seed"]).values()]  # type: ignore[arg-type]


def plot_arm_metric_by_seed(
    aggregates: Sequence[Mapping[str, object]],
    view: str,
    metric: str,
    output_path: Path,
    title: str | None = None,
) -> Path:
    """Plot every raw seed value per arm, with incomplete arms annotated rather than filled in."""

    selected = [
        item for item in aggregates if str(item["view"]) == view and str(item["metric"]) == metric
    ]
    if not selected:
        raise AnalysisError(f"no aggregates for view {view!r} and metric {metric!r}")

    figure, axis = plt.subplots(figsize=(max(6.0, 1.4 * len(selected)), 4.0))
    for position, item in enumerate(selected):
        points = _raw_seed_points(item)
        if points:
            axis.scatter([position] * len(points), points, marker="o", zorder=3)
        if bool(item["complete"]) and isinstance(item["mean"], int | float):
            axis.scatter([position], [float(item["mean"])], marker="_", s=600, zorder=4)
        else:
            axis.annotate(
                "incomplete",
                (position, 0.0),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                rotation=90,
            )

    axis.set_xticks(range(len(selected)))
    axis.set_xticklabels([str(item["arm_id"]) for item in selected], rotation=30, ha="right")
    axis.set_ylabel(metric)
    axis.set_title(title or f"{metric} on {view} (raw seed values; long dash = mean)")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)
    return output_path


def plot_quality_against_cost(
    rows: Sequence[Mapping[str, object]],
    view: str,
    metric: str,
    cost_field: str,
    output_path: Path,
) -> Path:
    """Plot quality jointly with a compute cost, so neither is read in isolation."""

    points: list[tuple[float, float, str]] = []
    for row in rows:
        if str(row["status"]) != "complete":
            continue
        metrics = dict(row.get("metrics_by_view", {})).get(view)  # type: ignore[arg-type]
        cost = row.get(cost_field)
        if not metrics or metric not in metrics or cost is None:
            continue
        points.append((float(cost), float(metrics[metric]), str(row["arm_id"])))
    if not points:
        raise AnalysisError(f"no complete rows carry both {metric!r} and {cost_field!r}")

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for cost, quality, arm in points:
        axis.scatter([cost], [quality], zorder=3)
        axis.annotate(arm, (cost, quality), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axis.set_xlabel(cost_field)
    axis.set_ylabel(metric)
    axis.set_title(f"{metric} against {cost_field} on {view}")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)
    return output_path


def plot_structural_metrics(
    table: Sequence[Mapping[str, object]],
    output_path: Path,
) -> Path:
    """Render structural metrics in separate panels, one per metric name.

    Refuses to draw incomparable metrics on a shared axis, which is exactly the mistake that would
    produce a single misleading cross-channel "alpha" figure.
    """

    assert_not_pooled_across_representations(table)
    names = sorted({str(entry["name"]) for entry in table})
    if not names:
        raise AnalysisError("no structural metrics to plot")

    figure, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 4.0), squeeze=False)
    for index, name in enumerate(names):
        axis = axes[0][index]
        entries = [entry for entry in table if str(entry["name"]) == name]
        labels = [str(entry.get("channel_name") or entry["arm_id"]) for entry in entries]
        axis.bar(range(len(entries)), [float(entry["value"]) for entry in entries])
        axis.set_xticks(range(len(entries)))
        axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        comparable = all(bool(entry.get("comparable_across_channels")) for entry in entries)
        verdict = "comparable" if comparable else "NOT comparable across channels"
        axis.set_title(f"{name}\n{verdict}")
        axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)
    return output_path
