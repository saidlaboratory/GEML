"""Hand-written verifier-gated metric tests with no production proof artifact."""

from __future__ import annotations

from geml.contracts.corpus import CorpusSplit
from geml.data.pairs.generate import (
    ReplayStatus,
    RewriteActionV1,
    TransitionVerificationV1,
    sha256_digest,
)
from geml.data.steps.extract import RewriteStepRecordV1
from geml.learning.eval.step_metrics import (
    ProposalOutcomeStatus,
    evaluate_step,
    summarize_step_outcomes,
)
from geml.learning.policy.head import LegalActionSetV1, ProposalStatus, masked_top_k


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


def _record() -> RewriteStepRecordV1:
    action = _action("SAFE-ADD-ZERO", (1, 0))
    return RewriteStepRecordV1.create(
        pair_id="sha256:" + "a" * 64,
        trace_digest="sha256:" + "b" * 64,
        step_index=0,
        state_expression_id="left",
        state_structural_signature="state-signature",
        state_graph_id="graph-left",
        action=action,
        next_state_expression_id="right",
        next_state_structural_signature="next-signature",
        remaining_step_distance=0,
        source_split=CorpusSplit.TEST_OOD,
        group_closure=("group-a",),
        operator_family="unseen_trigonometric",
    )


def _verification(record: RewriteStepRecordV1, action: RewriteActionV1, status: ReplayStatus):
    return TransitionVerificationV1(
        action_digest=action.semantic_digest,
        source_structural_signature=record.state_structural_signature,
        successor_structural_signature=record.next_state_structural_signature,
        verifier_version="fixture-verifier-v1",
        status=status,
        evidence_digest=sha256_digest(action.semantic_digest.encode("ascii")),
    )


def test_alternate_valid_action_counts_as_verifier_valid_but_not_structural_exact() -> None:
    record = _record()
    alternate = _action("SAFE-ADD-COMM", (1, 1))
    action_set = LegalActionSetV1(
        state_structural_signature=record.state_structural_signature,
        registered_rule_ids=("SAFE-ADD-COMM", "SAFE-ADD-ZERO"),
        legal_actions=(record.action, alternate),
    )
    output = masked_top_k(
        action_set,
        candidate_scores={record.action.semantic_digest: 0.0, alternate.semantic_digest: 1.0},
        top_k=1,
    )
    outcome = evaluate_step(
        record,
        output,
        verifier=lambda source, action: _verification(source, action, ReplayStatus.PASSED),
    )

    assert output.status is ProposalStatus.PROPOSED
    assert not outcome.exact_action_top_k
    assert outcome.verifier_valid_top_k
    assert outcome.outcome_statuses == (ProposalOutcomeStatus.VALID,)


def test_invalid_timeout_and_no_action_rows_remain_in_denominators() -> None:
    record = _record()
    action_set = LegalActionSetV1(
        state_structural_signature=record.state_structural_signature,
        registered_rule_ids=("SAFE-ADD-ZERO",),
        legal_actions=(record.action,),
    )
    output = masked_top_k(
        action_set,
        candidate_scores={record.action.semantic_digest: 0.0},
        top_k=1,
    )
    invalid = evaluate_step(
        record,
        output,
        verifier=lambda source, action: _verification(source, action, ReplayStatus.FAILED),
    )

    def timeout_verifier(_source, _action):
        raise TimeoutError

    timeout = evaluate_step(record, output, verifier=timeout_verifier)
    no_action = evaluate_step(
        record,
        masked_top_k(
            LegalActionSetV1(
                state_structural_signature=record.state_structural_signature,
                registered_rule_ids=("SAFE-ADD-ZERO",),
                legal_actions=(),
            ),
            candidate_scores={},
            top_k=1,
        ),
        verifier=lambda source, action: _verification(source, action, ReplayStatus.PASSED),
    )

    summary = summarize_step_outcomes(
        (invalid, timeout, no_action),
        registered_rule_ids=("SAFE-ADD-ZERO", "SAFE-MUL-ONE"),
        unseen_families=("unseen_trigonometric",),
    )
    assert summary.attempted_examples == 3
    assert summary.invalid_proposal_count == 1
    assert summary.timeout_count == 1
    assert summary.no_action_count == 1
    assert summary.per_rule[1][1]["attempted"] == 0
