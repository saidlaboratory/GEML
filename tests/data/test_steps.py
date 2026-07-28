"""Temporary-fixture tests for verifier-replayable Goal 7 step extraction."""

from __future__ import annotations

import pytest

from geml.contracts.corpus import CorpusSplit
from geml.data.pairs.generate import (
    ExpressionReferenceV1,
    GroupLineageV1,
    PairRecordV1,
    PairStatus,
    ReplayStatus,
    RewriteActionV1,
    RewriteTraceV1,
    TraceStateV1,
    TransitionVerificationV1,
    VerificationTier,
    make_transition_verifier,
    sha256_digest,
)
from geml.data.steps.extract import (
    StepFailureStatus,
    extract_steps,
    write_fixture_steps,
)
from geml.data.steps.stratify import StepSplitError, rule_coverage, validate_step_group_isolation


def _record() -> PairRecordV1:
    source = TraceStateV1(
        expression_id="left",
        sympy_srepr="Add(Symbol('x'), Integer(0))",
        structural_signature="left-signature",
    )
    goal = TraceStateV1(
        expression_id="right",
        sympy_srepr="Symbol('x')",
        structural_signature="right-signature",
    )
    action = RewriteActionV1.create(
        rule_id="SAFE-ADD-ZERO",
        direction="forward",
        occurrence_path=(0,),
        bindings=(),
        source_structural_signature=source.structural_signature,
        successor_structural_signature=goal.structural_signature,
        rule_assumptions=(),
        domain_mode="safe_real",
    )
    trace = RewriteTraceV1(
        source=source,
        goal=goal,
        states=(source, goal),
        actions=(action,),
        transitions=(
            TransitionVerificationV1(
                action_digest=action.semantic_digest,
                source_structural_signature=source.structural_signature,
                successor_structural_signature=goal.structural_signature,
                verifier_version="fixture-verifier-v1",
                status=ReplayStatus.PASSED,
                evidence_digest=sha256_digest(b"fixture"),
            ),
        ),
        verified_step_count=1,
        rule_set_digest=sha256_digest(b"rules"),
        policy_digest=sha256_digest(b"policy"),
        domain_mode="safe_real",
        generation_seed=20260726,
        replay_status=ReplayStatus.PASSED,
    )
    group = GroupLineageV1(group_id="group-a", source_split=CorpusSplit.TRAIN)

    def endpoint(state: TraceStateV1) -> ExpressionReferenceV1:
        return ExpressionReferenceV1(
            expression_id=state.expression_id,
            sympy_srepr=state.sympy_srepr,
            structural_signature=state.structural_signature,
            domain_mode="safe_real",
            operator_family="algebraic_core",
            source_split=CorpusSplit.TRAIN,
            group=group,
        )

    return PairRecordV1.create(
        left=endpoint(source),
        right=endpoint(goal),
        label=True,
        pair_group_set=("group-a",),
        source_split=CorpusSplit.TRAIN,
        evaluation_views=("iid",),
        verification_tier=VerificationTier.REPLAYED_RULE_ENGINE,
        trace=trace,
        non_equivalence_evidence=None,
        status=PairStatus.ACCEPTED,
        outcome_type=None,
        outcome_detail=None,
    )


def _verifier(record: PairRecordV1):
    assert record.trace is not None

    def apply(state, action, _domain_mode):
        if state == record.trace.source and action == record.trace.actions[0]:
            return record.trace.goal
        return None

    return make_transition_verifier(
        apply,
        verifier_version="fixture-verifier-v1",
    )


def test_step_extraction_replays_trace_and_is_byte_stable(tmp_path) -> None:
    record = _record()
    steps, failures = extract_steps(
        (record,),
        state_graph_ids={"left": "graph-left", "right": "graph-right"},
        verifier=_verifier(record),
    )

    assert not failures
    assert steps[0].action.occurrence_path == (0,)
    assert steps[0].remaining_step_distance == 0
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_manifest = write_fixture_steps(steps, failures, first, seed=20260726)
    second_manifest = write_fixture_steps(
        reversed(steps),
        reversed(failures),
        second,
        seed=20260726,
    )
    assert first.read_bytes() == second.read_bytes()
    assert first_manifest.content_digest == second_manifest.content_digest
    assert rule_coverage(steps, registered_rule_ids=("SAFE-ADD-ZERO", "SAFE-MUL-ONE")) == {
        "SAFE-ADD-ZERO": 1,
        "SAFE-MUL-ONE": 0,
    }


def test_missing_state_graph_and_group_leakage_remain_explicit() -> None:
    record = _record()
    steps, failures = extract_steps(
        (record,),
        state_graph_ids={"left": "graph-left"},
        verifier=_verifier(record),
    )

    assert not steps
    assert failures[0].status is StepFailureStatus.STATE_GRAPH_MISSING
    accepted, _ = extract_steps(
        (record,),
        state_graph_ids={"left": "graph-left", "right": "graph-right"},
        verifier=_verifier(record),
    )
    leaked = accepted[0].model_copy(update={"source_split": CorpusSplit.VALIDATION})
    with pytest.raises(StepSplitError, match="group-a"):
        validate_step_group_isolation((accepted[0], leaked))
