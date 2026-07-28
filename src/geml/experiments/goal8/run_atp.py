"""Fixed-budget, per-problem ATP experiment planning and result retention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class ATPMethod(StrEnum):
    UNIFORM = "uniform_legal"
    GNN_POLICY = "gnn_policy"
    GNN_POLICY_VALUE = "gnn_policy_value"
    PREFIX = "prefix_transformer"


class ATPStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ATPBudgetV1:
    beam_width: int
    expansion_budget: int
    proof_depth_budget: int
    wall_seconds_budget: float

    def __post_init__(self) -> None:
        if min(self.beam_width, self.expansion_budget, self.proof_depth_budget) < 1:
            raise ValueError("all ATP search budgets must be positive")
        if self.wall_seconds_budget <= 0:
            raise ValueError("ATP wall-time budget must be positive")


@dataclass(frozen=True, slots=True)
class ATPCellV1:
    problem_id: str
    method: ATPMethod
    seed: int | None
    budget: ATPBudgetV1


@dataclass(frozen=True, slots=True)
class ATPResultV1:
    cell: ATPCellV1
    status: ATPStatus
    solved: bool
    verifier_replayed: bool
    proof_length: int | None
    nodes_expanded: int
    frontier_peak: int
    invalid_actions: int
    verifier_timeouts: int
    error: str | None = None

    def __post_init__(self) -> None:
        if self.solved and not self.verifier_replayed:
            raise ValueError("a claimed ATP solution requires a replayed verifier proof")
        if self.status is ATPStatus.COMPLETE and self.solved != self.verifier_replayed:
            raise ValueError("complete solved status must agree with verifier proof evidence")
        if self.status in {ATPStatus.FAILED, ATPStatus.UNSUPPORTED} and not self.error:
            raise ValueError("failed/unsupported ATP cells require retained detail")


def fixed_cells(problem_ids: tuple[str, ...], budget: ATPBudgetV1) -> tuple[ATPCellV1, ...]:
    """Build every method/problem/seed cell; uniform is deterministic and has one seed."""

    if len(problem_ids) != 256 or len(set(problem_ids)) != 256:
        raise ValueError("production ATP requires exactly 256 unique frozen problem IDs")
    seeds = (20260726, 20260727, 20260728)
    return tuple(
        ATPCellV1(problem_id, method, None if method is ATPMethod.UNIFORM else seed, budget)
        for problem_id in sorted(problem_ids)
        for method in ATPMethod
        for seed in ((None,) if method is ATPMethod.UNIFORM else seeds)
    )


def planning_rows(problem_ids: tuple[str, ...], budget: ATPBudgetV1) -> tuple[ATPResultV1, ...]:
    return tuple(
        ATPResultV1(cell, ATPStatus.PENDING, False, False, None, 0, 0, 0, 0)
        for cell in fixed_cells(problem_ids, budget)
    )


def result_dict(result: ATPResultV1) -> dict[str, object]:
    return {
        "budget": asdict(result.cell.budget),
        "error": result.error,
        "frontier_peak": result.frontier_peak,
        "invalid_actions": result.invalid_actions,
        "method": result.cell.method.value,
        "nodes_expanded": result.nodes_expanded,
        "problem_id": result.cell.problem_id,
        "proof_length": result.proof_length,
        "seed": result.cell.seed,
        "solved": result.solved,
        "status": result.status.value,
        "verifier_replayed": result.verifier_replayed,
        "verifier_timeouts": result.verifier_timeouts,
    }
