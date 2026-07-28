"""Plot payloads and optional rendering for the denominator-complete Goal 6 grid."""

from __future__ import annotations

from pathlib import Path

from geml.analysis.goal6.summary import Goal6Analysis
from geml.experiments.goal6.run_grid import EvaluationView

PLOT_DATA_SCHEMA_VERSION = "geml-goal6-plot-data-v1"


def build_plot_data(analysis: Goal6Analysis) -> dict[str, object]:
    """Extract only saved aggregate values; unavailable cells remain explicit nulls."""

    rows = []
    for summary in analysis.arm_view_summaries:
        if summary.view is not EvaluationView.TEST_IID:
            continue
        rows.append(
            {
                "accuracy": summary.accuracy.as_dict(),
                "arm_id": summary.arm_id,
                "flop_estimate": summary.flop_estimate.as_dict(),
                "parameter_count": summary.parameter_count.as_dict(),
                "status_counts": dict(summary.status_counts),
                "valid": summary.valid,
                "attempted": summary.attempted,
            }
        )
    return {
        "rows": rows,
        "schema_version": PLOT_DATA_SCHEMA_VERSION,
        "verdict": analysis.verdict.value,
    }


def render_iid_accuracy_plot(analysis: Goal6Analysis, output_path: Path) -> Path:
    """Render a labeled IID chart, including a visible no-complete-results state."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - project dependency normally supplies it.
        raise RuntimeError("matplotlib is required to render Goal 6 plots") from error

    data = build_plot_data(analysis)["rows"]
    labels = [str(item["arm_id"]) for item in data]
    values = [item["accuracy"]["mean"] for item in data]
    figure, axis = plt.subplots(figsize=(10, 4))
    numeric = [value if isinstance(value, float) else 0.0 for value in values]
    bars = axis.bar(labels, numeric, color="#4c78a8")
    for bar, value in zip(bars, values, strict=True):
        if value is None:
            bar.set_hatch("//")
    axis.set_ylim(0, 1)
    axis.set_ylabel("mean IID accuracy across completed seeds")
    axis.set_title(f"Goal 6 IID accuracy — {analysis.verdict.value.replace('_', ' ')}")
    axis.tick_params(axis="x", rotation=30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
