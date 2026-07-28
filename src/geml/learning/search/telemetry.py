"""Complete search outcome telemetry, including invalid and timeout transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SearchTermination(StrEnum):
    SOLVED = "solved"
    EXHAUSTED = "exhausted"
    EXPANSION_BUDGET = "expansion_budget"
    DEPTH_BUDGET = "depth_budget"
    TIMEOUT = "timeout"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class SearchTelemetryV1:
    expansions: int
    generated_states: int
    valid_states: int
    duplicate_states: int
    invalid_actions: int
    verifier_timeouts: int
    verifier_failures: int
    frontier_peak: int
    wall_seconds: float
    termination: SearchTermination

    def __post_init__(self) -> None:
        if (
            min(
                self.expansions,
                self.generated_states,
                self.valid_states,
                self.duplicate_states,
                self.invalid_actions,
                self.verifier_timeouts,
                self.verifier_failures,
                self.frontier_peak,
            )
            < 0
            or self.wall_seconds < 0
        ):
            raise ValueError("search telemetry must be nonnegative")
