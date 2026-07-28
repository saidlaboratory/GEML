"""Fixture-only manifest-first tests for Goal 8 simplification selection."""

from __future__ import annotations

from geml.experiments.goal8.run_simplify import SimplifySampleV1, freeze_simplification_sample


def test_simplification_sample_is_deterministic_before_method_outputs() -> None:
    candidates = tuple(
        SimplifySampleV1(f"id-{index}", "family", "d", "s", "safe_real", "test_iid")
        for index in range(3)
    )
    reversed_selection = freeze_simplification_sample(reversed(candidates), required_count=2)
    selection = freeze_simplification_sample(candidates, required_count=2)
    assert reversed_selection == selection
