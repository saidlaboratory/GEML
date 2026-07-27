"""Analysis tests use a temporary grid fixture, never a production output directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from geml.analysis.goal6.summary import (
    GateVerdict,
    Goal6AnalysisError,
    render_summary_markdown,
    summarize_manifest,
)
from geml.experiments.goal6.run_grid import build_grid_manifest, load_grid_config
from geml.plots.goal6 import build_plot_data


def _phase_a_manifest() -> dict[str, object]:
    config = load_grid_config(Path("configs/goal6_grid.yaml"))
    return build_grid_manifest(config)


def test_phase_a_manifest_remains_insufficient_and_retains_all_denominators() -> None:
    analysis = summarize_manifest(_phase_a_manifest())
    rendered = render_summary_markdown(analysis)

    assert analysis.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
    assert dict(analysis.total_cell_status_counts) == {"pending": 15, "unsupported": 3}
    assert len(analysis.arm_view_summaries) == 30
    assert "No representation-quality conclusion" in rendered


def test_analysis_rejects_manifest_with_unbound_configuration_hash() -> None:
    manifest = _phase_a_manifest()
    manifest["config_hash"] = "sha256:wrong"

    with pytest.raises(Goal6AnalysisError, match="configuration hash"):
        summarize_manifest(manifest)


def test_iid_plot_data_keeps_unavailable_accuracy_as_null() -> None:
    plot_data = build_plot_data(summarize_manifest(_phase_a_manifest()))

    assert plot_data["verdict"] == GateVerdict.INSUFFICIENT_EVIDENCE.value
    assert len(plot_data["rows"]) == 6
    assert all(row["accuracy"]["mean"] is None for row in plot_data["rows"])
