"""Count-25 fixture planning checks for Goal 7."""

from __future__ import annotations

from pathlib import Path

from geml.analysis.goal7.summary import GateVerdict, summarize_manifest
from geml.experiments.goal7.run_grid import build_grid_manifest, fixed_cells, load_grid_config
from geml.plots.goal7 import build_plot_data


def test_goal7_plan_keeps_three_seeds_and_blocked_control() -> None:
    manifest = build_grid_manifest(load_grid_config(Path("configs/goal7_grid.yaml")))
    assert len(fixed_cells()) == 18
    assert [row["status"] for row in manifest["cells"]].count("unsupported") == 3
    assert [row["status"] for row in manifest["cells"]].count("pending") == 15


def test_goal7_phase_a_gate_is_conservative() -> None:
    config = load_grid_config(Path("configs/goal7_grid.yaml"))
    summary = summarize_manifest(build_grid_manifest(config))
    assert summary.verdict is GateVerdict.INSUFFICIENT_EVIDENCE
    assert build_plot_data(summary)["status_counts"] == {"pending": 15, "unsupported": 3}
