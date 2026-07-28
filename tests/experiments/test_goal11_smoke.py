"""Goal 11 final synthesis blocks claims when any controlled track is incomplete."""

from __future__ import annotations

from geml.experiments.goal11.run_final_eval import summarize_gates


def test_final_gate_synthesis_keeps_incomplete_tracks_visible() -> None:
    synthesis = summarize_gates((("equivalence", "pass"), ("proof", "pass"), ("sr", "pending")))
    assert synthesis.verdict == "insufficient_evidence"


def test_final_gate_synthesis_requires_all_controlled_tracks() -> None:
    assert summarize_gates((("proof", "pass"),)).verdict == "insufficient_evidence"
