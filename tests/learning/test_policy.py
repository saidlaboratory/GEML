"""Core legal-action masking tests; torch forwards are optional-ML tests elsewhere."""

from __future__ import annotations

import pytest

from geml.data.pairs.generate import RewriteActionV1
from geml.learning.policy.head import (
    LegalActionSetV1,
    PolicyContractError,
    ProposalStatus,
    masked_top_k,
)


def _action(rule_id: str, path: tuple[int, ...]) -> RewriteActionV1:
    return RewriteActionV1.create(
        rule_id=rule_id,
        direction="forward",
        occurrence_path=path,
        bindings=(),
        source_structural_signature="state-signature",
        successor_structural_signature="next-signature",
        rule_assumptions=(),
        domain_mode="safe_real",
    )


def test_shared_mask_emits_only_registered_legal_actions() -> None:
    first = _action("SAFE-ADD-COMM", (0,))
    second = _action("SAFE-ADD-ZERO", (1,))
    action_set = LegalActionSetV1(
        state_structural_signature="state-signature",
        registered_rule_ids=("SAFE-ADD-COMM", "SAFE-ADD-ZERO"),
        legal_actions=(first, second),
    )
    unknown_digest = "sha256:" + "f" * 64
    output = masked_top_k(
        action_set,
        candidate_scores={
            first.semantic_digest: 0.0,
            second.semantic_digest: 2.0,
            unknown_digest: 100.0,
        },
        top_k=2,
    )

    assert output.status is ProposalStatus.PROPOSED
    assert [item.action.rule_id for item in output.proposals] == ["SAFE-ADD-ZERO", "SAFE-ADD-COMM"]
    assert output.masked_action_digests == (unknown_digest,)
    assert sum(item.probability for item in output.proposals) == pytest.approx(1.0)


def test_empty_and_unseen_rule_states_are_explicit() -> None:
    empty = LegalActionSetV1(
        state_structural_signature="state-signature",
        registered_rule_ids=("SAFE-ADD-COMM",),
        legal_actions=(),
    )
    output = masked_top_k(empty, candidate_scores={"sha256:" + "a" * 64: 1.0}, top_k=1)

    assert output.status is ProposalStatus.NO_LEGAL_ACTION
    assert output.proposals == ()
    with pytest.raises(PolicyContractError, match="unregistered"):
        LegalActionSetV1(
            state_structural_signature="state-signature",
            registered_rule_ids=("SAFE-ADD-COMM",),
            legal_actions=(_action("HELD-OUT-RULE", (0,)),),
        )


def test_missing_legal_scores_fail_closed() -> None:
    action = _action("SAFE-ADD-COMM", (0,))
    action_set = LegalActionSetV1(
        state_structural_signature="state-signature",
        registered_rule_ids=("SAFE-ADD-COMM",),
        legal_actions=(action,),
    )
    with pytest.raises(PolicyContractError, match="did not score"):
        masked_top_k(action_set, candidate_scores={}, top_k=1)
