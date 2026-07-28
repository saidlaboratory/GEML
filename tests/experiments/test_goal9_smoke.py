"""Small fixed-cell schema smoke for Goal 9 guided SR."""

from __future__ import annotations

from geml.experiments.goal9.run_sr import planned_cells
from geml.learning.sr.guided_search import SRBudgetV1


def test_guided_sr_plan_has_matched_ast_eml_budgets() -> None:
    budget = SRBudgetV1(1.0, 2, 3, 4)
    cells = planned_cells(tuple(f"task-{index}" for index in range(256)), budget)
    assert len(cells) == 256 * 2 * 3
    assert {cell[-1] for cell in cells} == {budget}
