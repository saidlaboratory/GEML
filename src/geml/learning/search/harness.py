"""Verifier-gated fixed-budget search over explicit structural state identities."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from geml.learning.search.frontier import DeterministicFrontier, FrontierItem
from geml.learning.search.telemetry import SearchTelemetryV1, SearchTermination


class SearchMode(StrEnum):
    UNIFORM = "uniform"
    POLICY = "policy"
    VALUE = "value"


@dataclass(frozen=True, slots=True)
class SearchConfigV1:
    beam_width: int
    expansion_budget: int
    proof_depth_budget: int
    wall_seconds_budget: float

    def __post_init__(self) -> None:
        if min(self.beam_width, self.expansion_budget, self.proof_depth_budget) < 1:
            raise ValueError("beam, expansion, and depth budgets must be positive")
        if self.wall_seconds_budget <= 0:
            raise ValueError("wall budget must be positive")


@dataclass(frozen=True, slots=True)
class SearchTransition:
    action_id: str
    successor_signature: str
    policy_score: float = 0.0
    value_score: float = 0.0


TransitionProvider = Callable[[str], Iterable[SearchTransition]]
TransitionVerifier = Callable[[str, SearchTransition], bool]


@dataclass(frozen=True, slots=True)
class SearchResultV1:
    solved: bool
    proof_action_ids: tuple[str, ...]
    proof_state_signatures: tuple[str, ...]
    telemetry: SearchTelemetryV1


def search(
    *,
    source_signature: str,
    target_signature: str,
    config: SearchConfigV1,
    mode: SearchMode,
    transitions: TransitionProvider,
    verifier: TransitionVerifier,
) -> SearchResultV1:
    """Run one deterministic bounded traversal; a success needs a verified transition chain."""

    if not source_signature or not target_signature:
        raise ValueError("source and target signatures must be nonblank")
    started = time.monotonic()
    frontier = DeterministicFrontier(mode.value)
    frontier.push(FrontierItem(source_signature, 0, 0))
    parents: dict[str, tuple[str, str] | None] = {source_signature: None}
    expansions = generated = valid = duplicates = invalid = 0
    verifier_timeouts = verifier_failures = 0
    peak = 1
    termination = SearchTermination.EXHAUSTED
    solved_signature: str | None = (
        source_signature if source_signature == target_signature else None
    )

    def telemetry() -> SearchTelemetryV1:
        return SearchTelemetryV1(
            expansions,
            generated,
            valid,
            duplicates,
            invalid,
            verifier_timeouts,
            verifier_failures,
            peak,
            time.monotonic() - started,
            termination,
        )

    while frontier and solved_signature is None:
        if time.monotonic() - started > config.wall_seconds_budget:
            termination = SearchTermination.TIMEOUT
            break
        if expansions >= config.expansion_budget:
            termination = SearchTermination.EXPANSION_BUDGET
            break
        item = frontier.pop()
        if item.depth >= config.proof_depth_budget:
            termination = SearchTermination.DEPTH_BUDGET
            continue
        expansions += 1
        for transition in transitions(item.structural_signature):
            generated += 1
            try:
                accepted = verifier(item.structural_signature, transition)
            except TimeoutError:
                verifier_timeouts += 1
                continue
            except Exception:
                verifier_failures += 1
                continue
            if not accepted:
                invalid += 1
                continue
            valid += 1
            if transition.successor_signature in parents:
                duplicates += 1
                continue
            parents[transition.successor_signature] = (
                item.structural_signature,
                transition.action_id,
            )
            if transition.successor_signature == target_signature:
                solved_signature = target_signature
                termination = SearchTermination.SOLVED
                break
            frontier.push(
                FrontierItem(
                    transition.successor_signature,
                    item.depth + 1,
                    len(parents),
                    transition.policy_score,
                    transition.value_score,
                )
            )
        frontier.trim_to_beam(config.beam_width)
        peak = max(peak, len(frontier))
    if solved_signature is None:
        return SearchResultV1(False, (), (), telemetry())
    states = []
    actions = []
    current = solved_signature
    while parents[current] is not None:
        parent, action = parents[current]
        states.append(current)
        actions.append(action)
        current = parent
    states.append(source_signature)
    return SearchResultV1(True, tuple(reversed(actions)), tuple(reversed(states)), telemetry())
