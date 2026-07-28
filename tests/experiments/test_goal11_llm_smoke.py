"""Fully mocked LLM reference tests require neither network access nor API keys."""

from __future__ import annotations

from geml.experiments.goal11.llm_reference import (
    AttemptStatus,
    LLMModelPinV1,
    Provider,
    freeze_panel_ids,
    mocked_attempt,
    prompt_hash,
)


def test_mocked_panel_preserves_raw_and_verifier_outcomes() -> None:
    model = LLMModelPinV1(
        Provider.OPENAI,
        "fixture-model",
        "2026-07-27",
        prompt_hash("s", "u"),
    )
    attempt = mocked_attempt(
        task_id="proof-1",
        track="proof",
        model=model,
        call=lambda: "x",
        verifier=lambda raw: raw == "x",
    )
    assert attempt.status is AttemptStatus.COMPLETE
    assert attempt.verifier_confirmed is True
    proof, sr = freeze_panel_ids(
        (f"p{index}" for index in range(100)),
        (f"s{index}" for index in range(100)),
    )
    assert len(proof) == len(sr) == 100
