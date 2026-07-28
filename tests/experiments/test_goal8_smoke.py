"""Fixture-only ATP grid tests; no checkpoint or production benchmark is required."""

from __future__ import annotations

import pytest

from geml.experiments.goal8.run_atp import ATPBudgetV1, ATPMethod, fixed_cells, planning_rows


def test_fixed_atp_grid_retains_identical_budgets_for_all_methods() -> None:
    budget = ATPBudgetV1(
        beam_width=2,
        expansion_budget=4,
        proof_depth_budget=3,
        wall_seconds_budget=1.0,
    )
    problem_ids = tuple(f"problem-{index:03d}" for index in range(256))
    cells = fixed_cells(problem_ids, budget)

    assert len(cells) == 256 * 10
    assert {cell.budget for cell in cells} == {budget}
    assert sum(cell.method is ATPMethod.UNIFORM for cell in cells) == 256
    assert all(row.status.value == "pending" for row in planning_rows(problem_ids, budget))
    with pytest.raises(ValueError, match="256"):
        fixed_cells(problem_ids[:-1], budget)
