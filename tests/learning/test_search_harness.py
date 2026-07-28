"""Deterministic verifier-gated search fixtures for all fixed search modes."""

from __future__ import annotations

from geml.learning.search.harness import (
    SearchConfigV1,
    SearchMode,
    SearchTermination,
    SearchTransition,
    search,
)


def test_all_modes_share_budget_and_only_verified_path_solves() -> None:
    config = SearchConfigV1(
        beam_width=4,
        expansion_budget=3,
        proof_depth_budget=2,
        wall_seconds_budget=1.0,
    )

    def transitions(state: str):
        return (
            (
                SearchTransition("invalid", "bad", policy_score=100.0, value_score=-10.0),
                SearchTransition("valid", "target", policy_score=0.0, value_score=0.0),
            )
            if state == "source"
            else ()
        )

    for mode in SearchMode:
        result = search(
            source_signature="source",
            target_signature="target",
            config=config,
            mode=mode,
            transitions=transitions,
            verifier=lambda _source, transition: transition.action_id == "valid",
        )
        assert result.solved
        assert result.proof_action_ids == ("valid",)
        assert result.telemetry.invalid_actions == 1
        assert result.telemetry.termination is SearchTermination.SOLVED


def test_exhausted_search_retains_unsolved_outcome() -> None:
    result = search(
        source_signature="source",
        target_signature="target",
        config=SearchConfigV1(2, 1, 1, 1.0),
        mode=SearchMode.UNIFORM,
        transitions=lambda _state: (),
        verifier=lambda _source, _transition: False,
    )
    assert not result.solved
    assert result.telemetry.termination is SearchTermination.EXHAUSTED


def test_beam_pruning_retains_highest_priority_successor() -> None:
    config = SearchConfigV1(
        beam_width=1,
        expansion_budget=2,
        proof_depth_budget=2,
        wall_seconds_budget=1.0,
    )

    def transitions(state: str):
        if state == "source":
            return (
                SearchTransition("worse", "worse", policy_score=0.0),
                SearchTransition("better", "better", policy_score=1.0),
            )
        if state == "better":
            return (SearchTransition("finish", "target", policy_score=0.0),)
        return ()

    result = search(
        source_signature="source",
        target_signature="target",
        config=config,
        mode=SearchMode.POLICY,
        transitions=transitions,
        verifier=lambda _source, _transition: True,
    )

    assert result.solved
    assert result.proof_action_ids == ("better", "finish")
