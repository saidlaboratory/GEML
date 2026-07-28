"""Plot-ready Goal 11 gate status with external references kept out of controlled verdicts."""

from __future__ import annotations

from geml.analysis.goal11.final_eval import GateSynthesisV1


def build_plot_data(summary: GateSynthesisV1) -> dict[str, object]:
    return {
        "controlled_gate_statuses": dict(summary.gate_statuses),
        "external_llm_reference_excluded": True,
        "schema_version": "geml-goal11-final-plot-data-v1",
        "verdict": summary.verdict,
    }
