"""Goal 9 plot payload uses separately retained verifier-exact and numeric-only counts."""

from __future__ import annotations

from dataclasses import asdict

from geml.analysis.goal9.summary import Goal9SummaryV1


def build_plot_data(summary: Goal9SummaryV1) -> dict[str, object]:
    return {"schema_version": "geml-goal9-plot-data-v1", **asdict(summary)}
