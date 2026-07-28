"""Tiny-fixture planning tests for the fixed Goal 6 grid."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from geml.experiments.goal6.run_grid import (
    FIXED_ARMS,
    FIXTURE_PAIR_COUNT,
    ArmFamily,
    CellStatus,
    EvaluationMetrics,
    EvaluationView,
    Goal6GridError,
    GridCellResult,
    build_grid_manifest,
    fixed_grid_cells,
    load_grid_config,
    write_grid_manifest,
)


def test_fixed_grid_has_six_arms_three_seeds_and_explicit_blocked_control() -> None:
    cells = fixed_grid_cells()

    assert len(FIXED_ARMS) == 6
    assert len(cells) == 18
    blocked = [cell for cell in cells if cell.arm.availability is CellStatus.UNSUPPORTED]
    assert len(blocked) == 3
    assert all("not a fair motif-AST substitute" in cell.arm.unavailable_reason for cell in blocked)


def test_phase_a_manifest_has_15_pending_and_3_unsupported_rows(tmp_path: Path) -> None:
    config = load_grid_config(Path("configs/goal6_grid.yaml"))
    output_path = write_grid_manifest(config, output_path=tmp_path / "goal6.grid.json")
    manifest = output_path.read_text(encoding="utf-8")

    assert "phase_a_planning" in manifest
    assert manifest.count('"status": "pending"') == 15
    assert manifest.count('"status": "unsupported"') == 3


def test_count_25_fixture_executor_retains_all_views_for_available_cells(tmp_path: Path) -> None:
    config = replace(
        load_grid_config(Path("configs/goal6_grid.yaml")),
        output_directory=str(tmp_path),
        train_pair_count=FIXTURE_PAIR_COUNT,
        validation_pair_count=FIXTURE_PAIR_COUNT,
        test_pair_count=FIXTURE_PAIR_COUNT,
    )

    def executor(cell, _config):
        assert cell.arm.family in {ArmFamily.GINE, ArmFamily.PREFIX_TRANSFORMER, ArmFamily.TRIVIAL}
        return GridCellResult(
            cell=cell,
            status=CellStatus.COMPLETE,
            evaluations=tuple(
                EvaluationMetrics(
                    view=view,
                    attempted=FIXTURE_PAIR_COUNT,
                    valid=FIXTURE_PAIR_COUNT,
                    correct=FIXTURE_PAIR_COUNT - 1,
                    macro_f1=0.9,
                    calibration_error=0.1,
                )
                for view in EvaluationView
            ),
            parameter_count=123,
            flop_estimate=456,
            wall_seconds=1.0,
            peak_host_memory_bytes=789,
            peak_gpu_memory_bytes=None,
        )

    manifest = build_grid_manifest(config, executor=executor)
    statuses = [row["status"] for row in manifest["cells"]]

    assert statuses.count(CellStatus.COMPLETE.value) == 15
    assert statuses.count(CellStatus.UNSUPPORTED.value) == 3


def test_complete_cell_rejects_missing_ood_denominators() -> None:
    cell = fixed_grid_cells()[0]
    with pytest.raises(Goal6GridError, match="every train/validation/IID/OOD view"):
        GridCellResult(
            cell=cell,
            status=CellStatus.COMPLETE,
            evaluations=(),
            parameter_count=1,
            flop_estimate=1,
            wall_seconds=0.0,
            peak_host_memory_bytes=0,
            peak_gpu_memory_bytes=0,
        )
