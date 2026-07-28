"""Goal 7 plot payloads retain incomplete cells rather than suppressing them."""

from __future__ import annotations

from geml.analysis.goal7.summary import Goal7Summary


def build_plot_data(summary: Goal7Summary) -> dict[str, object]:
    return {
        "schema_version": "geml-goal7-plot-data-v1",
        "status_counts": dict(summary.status_counts),
        "verdict": summary.verdict.value,
    }
