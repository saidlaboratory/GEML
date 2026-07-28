"""Plot-data preparation for the Goal 11 cross-track synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geml.analysis.goal11.final_eval import Goal11Synthesis


@dataclass(frozen=True)
class Goal11MetricPlotPoint:
    track: str
    metric_id: str
    value: float
    external: bool


def build_plot_data(synthesis: Goal11Synthesis) -> tuple[Goal11MetricPlotPoint, ...]:
    """Return controlled points only; external rows have no commensurate scalar metric."""

    return tuple(
        Goal11MetricPlotPoint(
            track=track.track.value,
            metric_id=metric.metric_id,
            value=metric.estimate,
            external=False,
        )
        for track in synthesis.tracks
        for metric in track.metrics
    )


def render_plots(synthesis: Goal11Synthesis, output_dir: str | Path) -> tuple[Path, ...]:
    """Render separate track panels without a cross-task aggregate score."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs = []
    data = build_plot_data(synthesis)
    for track in sorted({item.track for item in data}):
        rows = [item for item in data if item.track == track]
        figure, axes = plt.subplots(figsize=(6.4, 4.2))
        axes.bar([item.metric_id for item in rows], [item.value for item in rows])
        axes.set_title(f"{track}: track-specific metrics")
        axes.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        path = root / f"{track}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(path)
    return tuple(outputs)
