"""Fixed task/method/seed plan for bounded Goal 9 guided SR execution."""

from __future__ import annotations

from geml.learning.sr.guided_search import SRBudgetV1, SRGuidance


def planned_cells(
    task_ids: tuple[str, ...],
    budget: SRBudgetV1,
) -> tuple[tuple[str, SRGuidance, int, SRBudgetV1], ...]:
    if len(task_ids) != 256 or len(set(task_ids)) != 256:
        raise ValueError(
            "production guided SR requires exactly 256 unique frozen synthetic task IDs"
        )
    return tuple(
        (task_id, guidance, seed, budget)
        for task_id in sorted(task_ids)
        for guidance in SRGuidance
        for seed in (20260726, 20260727, 20260728)
    )
