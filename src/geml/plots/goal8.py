"""Goal 8 plot payloads keep absent evidence explicit."""

from __future__ import annotations

from dataclasses import asdict

from geml.analysis.goal8.summary import Goal8SummaryV1


def build_plot_data(summary: Goal8SummaryV1) -> dict[str, object]:
    return {"schema_version": "geml-goal8-plot-data-v1", **asdict(summary)}
