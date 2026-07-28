"""Deterministic priority frontiers shared by uniform, policy, and value search modes."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FrontierItem:
    structural_signature: str
    depth: int
    insertion_index: int
    policy_score: float = 0.0
    value_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.structural_signature or self.depth < 0 or self.insertion_index < 0:
            raise ValueError("frontier item identity/depth/index is invalid")


@dataclass(slots=True)
class DeterministicFrontier:
    """Stable heap priority with insertion order as the final traversal tie-breaker."""

    mode: str
    _items: list[tuple[tuple[float, float, int], FrontierItem]] = field(default_factory=list)

    def push(self, item: FrontierItem) -> None:
        if self.mode == "uniform":
            priority = (0.0, float(item.depth), item.insertion_index)
        elif self.mode == "policy":
            priority = (-item.policy_score, float(item.depth), item.insertion_index)
        elif self.mode == "value":
            priority = (item.value_score, -item.policy_score, item.insertion_index)
        else:
            raise ValueError(f"unknown search mode {self.mode!r}")
        heapq.heappush(self._items, (priority, item))

    def pop(self) -> FrontierItem:
        return heapq.heappop(self._items)[1]

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)
