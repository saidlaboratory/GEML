"""Fixture-only tests for Goal 6 pair, trace, negative, and split contracts."""

from __future__ import annotations

import json

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
    canonical_json_bytes,
    make_transition_verifier,
    replay_trace,
    sha256_digest,
    write_fixture_pairs,
)
from geml.data.pairs.negatives import (
    NegativeCandidate,
    formal_counterexample_evidence,
    numerical_disagreement_evidence,
    reject_structural_match,
    within_size_tolerance,
)
from geml.data.pairs.splits import PairSplitError, validate_group_isolation


def _state(name: str, signature: str) -> TraceStateV1:
    return TraceStateV1(
        expression_id=name,
        sympy_srepr=f"Symbol('{name}', real=True)",
        structural_signature=signature,
    )


def _action() -> RewriteActionV1:
    return RewriteActionV1.create(
        rule_id="SAFE-ADD-COMM",
        direction="forward",
        occurrence_path=(1, 0),
        bindings=(),
        source_structural_signature="left-signature",
        successor_structural_signature="right-signature",
        rule_assumptions=(),
        domain_mode="safe_real",
    )


def _trace() -> RewriteTraceV1:
    source = _state("left", "left-signature")
    goal = _state("right", "right-signature")
    action = _action()
    verification = TransitionVerificationV1(
        action_digest=action.semantic_digest,
        source_structural_signature=source.structural_signature,
        successor_structural_signature=goal.structural_signature,
        verifier_version="fixture-verifier-v1",
        status=ReplayStatus.PASSED,
        evidence_digest=sha256_digest(b"fixture-transition"),
        detail="fixture replay",
    )
    return RewriteTraceV1(
        source=source,
        goal=goal,
        states=(source, goal),
        actions=(action,),
        transitions=(verification,),
        verified_step_count=1,
        rule_set_digest=sha256_digest(b"rules"),
        policy_digest=sha256_digest(b"policy"),
        domain_mode="safe_real",
        generation_seed=20260726,
        replay_status=ReplayStatus.PASSED,
    )


def _endpoint(
    name: str,
    group: str,
    *,
    split: CorpusSplit = CorpusSplit.TRAIN,
) -> ExpressionReferenceV1:
    return ExpressionReferenceV1(
        expression_id=name,
        sympy_srepr=f"Symbol('{name}', real=True)",
        structural_signature=f"{name}-signature",
        domain_mode="safe_real",
        operator_family="algebraic_core",
        source_split=split,
        group=GroupLineageV1(group_id=group, source_split=split),
    )


def _positive_record() -> PairRecordV1:
    trace = _trace()
    left = _endpoint("left", "group-a")
    right = _endpoint("right", "group-a")
    return PairRecordV1.create(
        left=left,
        right=right,
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


def test_positive_pair_binds_concrete_trace() -> None:
    record = _positive_record()

    assert record.pair_id.startswith("sha256:")
    assert record.trace is not None
    assert record.trace.states[0].structural_signature == "left-signature"
    assert record.trace.actions[0].occurrence_path == (1, 0)
    assert "eclass" not in json.dumps(record.model_dump(mode="json"))


def test_numeric_disagreement_cannot_become_an_accepted_negative() -> None:
    evidence = numerical_disagreement_evidence(
        method="mpmath-200",
        detail="values disagree at a sampled safe point",
        samples={"x": "1.0"},
    )
    with pytest.raises(ValueError, match="numeric disagreement alone"):
        PairRecordV1.create(
            left=_endpoint("left", "group-left"),
            right=_endpoint("right", "group-right"),
            label=False,
            pair_group_set=("group-left", "group-right"),
            source_split=CorpusSplit.TRAIN,
            evaluation_views=("iid",),
            verification_tier=VerificationTier.NUMERIC_COUNTEREXAMPLE,
            trace=None,
            non_equivalence_evidence=evidence,
            status=PairStatus.ACCEPTED,
            outcome_type=None,
            outcome_detail=None,
        )


def test_formal_negative_and_fixture_writer_are_deterministic(tmp_path) -> None:
    evidence = formal_counterexample_evidence(
        method="interval-proof-v1",
        detail="disjoint intervals at x=1",
        witness={"x": ["1", "1"], "left": ["0", "0"], "right": ["1", "1"]},
    )
    negative = PairRecordV1.create(
        left=_endpoint("left", "group-left"),
        right=_endpoint("right", "group-right"),
        label=False,
        pair_group_set=("group-left", "group-right"),
        source_split=CorpusSplit.TRAIN,
        evaluation_views=("iid",),
        verification_tier=VerificationTier.FORMAL_COUNTEREXAMPLE,
        trace=None,
        non_equivalence_evidence=evidence,
        status=PairStatus.ACCEPTED,
        outcome_type=None,
        outcome_detail=None,
    )

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_manifest = write_fixture_pairs((_positive_record(), negative), first, seed=20260726)
    second_manifest = write_fixture_pairs((negative, _positive_record()), second, seed=20260726)

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest.content_digest == second_manifest.content_digest
    assert first_manifest.row_count == 2


def test_trace_replay_uses_injected_concrete_verifier() -> None:
    trace = _trace()
    verifier = make_transition_verifier(
        lambda state, action, domain_mode: (
            trace.goal
            if (state == trace.source and action == trace.actions[0] and domain_mode == "safe_real")
            else None
        ),
        verifier_version="fixture-verifier-v1",
    )

    replayed = replay_trace(trace, verifier)
    assert replayed.replay_status is ReplayStatus.PASSED
    assert replayed.transitions[0].status is ReplayStatus.PASSED


def test_group_relatives_cannot_cross_partitions() -> None:
    record = _positive_record()
    leaked_left = _endpoint("other", "group-a", split=CorpusSplit.VALIDATION)
    leaked = PairRecordV1.create(
        left=leaked_left,
        right=leaked_left,
        label=None,
        pair_group_set=("group-a",),
        source_split=CorpusSplit.VALIDATION,
        evaluation_views=(),
        verification_tier=VerificationTier.FAILED,
        trace=None,
        non_equivalence_evidence=None,
        status=PairStatus.FAILED,
        outcome_type="fixture_failure",
        outcome_detail="retained",
    )
    with pytest.raises(PairSplitError, match="group-a"):
        validate_group_isolation((record, leaked))


def test_near_miss_helpers_do_not_accept_structural_matches() -> None:
    near_miss = NegativeCandidate(
        left_signature="a",
        right_signature="b",
        left_size=10,
        right_size=12,
        operator_family="algebraic_core",
    )
    duplicate = NegativeCandidate(
        left_signature="a",
        right_signature="a",
        left_size=10,
        right_size=10,
        operator_family="algebraic_core",
    )

    assert within_size_tolerance(near_miss, absolute_tolerance=2)
    assert not reject_structural_match(near_miss)
    assert reject_structural_match(duplicate)
    assert canonical_json_bytes({"x": 1}) == b'{"x":1}'
