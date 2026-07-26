"""Plot-data preparation for fixed-scale Goal 11 efficiency panels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geml.analysis.goal11.scaling import FixedScaleResult, MetricAvailability


@dataclass(frozen=True)
class FixedScalePlotPoint:
    panel_id: str
    comparison_key: str
    method_id: str
    representation_id: str
    resource_metric_id: str
    quality: float
    resource: float | None
    eligible: bool


def build_plot_data(result: FixedScaleResult) -> tuple[FixedScalePlotPoint, ...]:
    """Keep missing telemetry as ``None`` rather than plotting a false zero."""

    points = []
    for panel in result.panels:
        for point in panel.pareto_points:
            for resource in point.resources:
                points.append(
                    FixedScalePlotPoint(
                        panel_id=panel.panel_id,
                        comparison_key=panel.comparison_key,
                        method_id=point.method_id,
                        representation_id=point.representation_id,
                        resource_metric_id=resource.metric_id,
                        quality=point.quality_mean,
                        resource=(
                            resource.mean
                            if resource.availability is MetricAvailability.AVAILABLE
                            else None
                        ),
                        eligible=point.eligible,
                    )
                )
    return tuple(points)


def render_plots(result: FixedScaleResult, output_dir: str | Path) -> tuple[Path, ...]:
    """Render one quality/resource plot per compatible panel and metric."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    data = build_plot_data(result)
    outputs = []
    keys = sorted({(item.panel_id, item.comparison_key, item.resource_metric_id) for item in data})
    for panel_id, comparison_key, metric_id in keys:
        rows = [
            item
            for item in data
            if item.panel_id == panel_id
            and item.comparison_key == comparison_key
            and item.resource_metric_id == metric_id
            and item.resource is not None
        ]
        figure, axes = plt.subplots(figsize=(6.4, 4.2))
        if rows:
            for row in rows:
                axes.scatter(row.resource, row.quality)
                axes.annotate(
                    f"{row.method_id}/{row.representation_id}",
                    (row.resource, row.quality),
                    fontsize=8,
                )
        else:
            axes.text(0.5, 0.5, "No compatible telemetry", ha="center", va="center")
        axes.set_xlabel(metric_id)
        axes.set_ylabel("track-specific quality")
        axes.set_title(f"{panel_id} [{comparison_key[:12]}]: fixed-scale only")
        figure.tight_layout()
        path = (root / f"{panel_id}-{comparison_key[:12]}-{metric_id}.png").resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("fixed-scale plot path escapes output directory") from error
        figure.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(path)
    return tuple(outputs)
