"""Typed, representation-honest result contract for bounded SR guided search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from geml.data.sr.benchmark import ExactRecoveryStatus, exact_recovery_claim


class SRGuidance(StrEnum):
    AST = "ast_guided"
    PURE_EML = "pure_eml_guided"


@dataclass(frozen=True, slots=True)
class SRBudgetV1:
    wall_seconds: float
    expansion_budget: int
    depth_budget: int
    complexity_budget: int

    def __post_init__(self) -> None:
        count_budgets = (self.expansion_budget, self.depth_budget, self.complexity_budget)
        if self.wall_seconds <= 0 or min(count_budgets) < 1:
            raise ValueError("every SR budget must be positive")


@dataclass(frozen=True, slots=True)
class SRResultV1:
    task_id: str
    guidance: SRGuidance
    seed: int
    budget: SRBudgetV1
    exact_recovery_status: ExactRecoveryStatus
    numeric_error: float | None
    candidate_complexity: int | None
    expansions: int
    timeout: bool
    error: str | None = None

    @property
    def exact_recovery(self) -> bool:
        return exact_recovery_claim(self.exact_recovery_status, numeric_error=self.numeric_error)

    def __post_init__(self) -> None:
        if not self.task_id or self.expansions < 0:
            raise ValueError("SR result task identity and expansion count are invalid")
        if self.numeric_error is not None and self.numeric_error < 0:
            raise ValueError("numeric error must be nonnegative")
        if self.candidate_complexity is not None and self.candidate_complexity < 0:
            raise ValueError("candidate complexity must be nonnegative")
        if self.exact_recovery and self.timeout:
            raise ValueError("timed-out SR result cannot claim exact recovery")
