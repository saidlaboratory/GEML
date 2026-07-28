"""Fixture-only tests for Goal 6 pair, trace, negative, and split contracts."""

from __future__ import annotations

import json

import pytest

from geml.contracts.corpus import CorpusSplit
from geml.data.pairs.generate import (
    ExpressionReferenceV1,
    GroupLineageV1,
    PairContractError,
    PairRecordV1,
    PairStatus,
    ReplayStatus,
    RewriteActionV1,
    RewriteTraceV1,
    TraceStateV1,
    TrajectoryGenerationOutcomeV1,
    TrajectoryOutcomeStatus,
    TrajectoryPolicyV1,
    TransitionVerificationV1,
    VerificationTier,
    canonical_json_bytes,
    derive_trajectory_seed,
    generate_concrete_trajectory_attempts,
    make_transition_verifier,
    replay_trace,
    sha256_digest,
    write_fixture_pairs,
    write_pair_shard,
)
from geml.data.pairs.negatives import (
    NegativeCandidate,
    NegativeDisposition,
    assess_hard_negative,
    formal_counterexample_evidence,
    numerical_disagreement_evidence,
    reject_structural_match,
    within_size_tolerance,
)
from geml.data.pairs.splits import (
    PairSplitError,
    require_family_ood_view,
    require_ood_stress_view,
    require_strict_depth_ood_view,
    validate_group_isolation,
)


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
    operator_family: str = "algebraic_core",
) -> ExpressionReferenceV1:
    return ExpressionReferenceV1(
        expression_id=name,
        sympy_srepr=f"Symbol('{name}', real=True)",
        structural_signature=f"{name}-signature",
        domain_mode="safe_real",
        operator_family=operator_family,
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


def _ood_positive_record(view: str, *, operator_family: str = "exp_log") -> PairRecordV1:
    trace = _trace()
    left = _endpoint(
        "left",
        "group-a",
        split=CorpusSplit.TEST_OOD,
        operator_family=operator_family,
    )
    right = _endpoint(
        "right",
        "group-a",
        split=CorpusSplit.TEST_OOD,
        operator_family=operator_family,
    )
    return PairRecordV1.create(
        left=left,
        right=right,
        label=True,
        pair_group_set=("group-a",),
        source_split=CorpusSplit.TEST_OOD,
        evaluation_views=(view,),
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


def test_ood_views_refuse_misleading_depth_or_family_labels() -> None:
    stress = _ood_positive_record("ood_stress")
    strict = _ood_positive_record("strict_depth_ood")
    family = _ood_positive_record("family_ood", operator_family="exp_log")

    assert require_ood_stress_view((stress,)) == (stress,)
    assert require_strict_depth_ood_view(
        (strict,), training_depths=(1, 2), evaluation_depth_by_pair_id={strict.pair_id: 8}
    ) == (strict,)
    with pytest.raises(PairSplitError, match="overlap"):
        require_strict_depth_ood_view(
            (strict,), training_depths=(8,), evaluation_depth_by_pair_id={strict.pair_id: 8}
        )
    assert require_family_ood_view((family,), training_families=("algebraic_core",)) == (family,)
    with pytest.raises(PairSplitError, match="observed in training"):
        require_family_ood_view((family,), training_families=("exp_log",))


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


def test_hard_negative_admission_retains_unresolved_numeric_evidence() -> None:
    candidate = NegativeCandidate(
        left_signature="left",
        right_signature="right",
        left_size=10,
        right_size=11,
        operator_family="algebraic_core",
    )
    numeric = numerical_disagreement_evidence(
        method="mpmath-200",
        detail="sampled disagreement",
        samples={"x": "1"},
    )
    formal = formal_counterexample_evidence(
        method="interval-proof-v1",
        detail="disjoint intervals",
        witness={"x": ["1", "1"]},
    )

    assert (
        assess_hard_negative(candidate, absolute_size_tolerance=2, evidence=numeric).disposition
        is NegativeDisposition.UNRESOLVED
    )
    assert (
        assess_hard_negative(candidate, absolute_size_tolerance=2, evidence=formal).disposition
        is NegativeDisposition.ACCEPTED
    )


def _action_between(
    source_signature: str,
    successor_signature: str,
    *,
    direction: str,
) -> RewriteActionV1:
    return RewriteActionV1.create(
        rule_id="fixture-rule",
        direction=direction,
        occurrence_path=(0,),
        bindings=(),
        source_structural_signature=source_signature,
        successor_structural_signature=successor_signature,
        rule_assumptions=(),
        domain_mode="safe_real",
    )


def test_bounded_trajectory_generation_is_seeded_and_retain_shortfalls() -> None:
    source = _state("source", "s0")
    successor = _state("successor", "s1")
    forward = _action_between("s0", "s1", direction="forward")
    backward = _action_between("s1", "s0", direction="backward")
    policy = TrajectoryPolicyV1(min_trace_length=2, max_trace_length=2, max_attempts_per_source=2)

    outcomes = generate_concrete_trajectory_attempts(
        source,
        run_seed=20260726,
        domain_mode="safe_real",
        rule_set_digest=sha256_digest(b"rules"),
        policy_digest=sha256_digest(b"policy"),
        policy=policy,
        enumerate_actions=lambda state, _seed, _domain: (forward if state == source else backward,),
        apply_action=lambda state, action, _domain: successor if action is forward else source,
        verifier_version="fixture-verifier-v1",
    )

    assert tuple(outcome.status for outcome in outcomes) == (
        TrajectoryOutcomeStatus.SHORTFALL,
        TrajectoryOutcomeStatus.SHORTFALL,
    )
    assert all(outcome.trace is None for outcome in outcomes)
    assert outcomes[0].derived_seed == derive_trajectory_seed(
        component="goal6.pairs.trajectory",
        run_seed=20260726,
        source_expression_id="source",
        attempt_index=0,
    )
    assert outcomes[0].derived_seed == 14158669760612414972
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        TrajectoryPolicyV1(min_trace_length=0, max_trace_length=1, max_attempts_per_source=1)


def test_bounded_trajectory_generation_returns_concrete_replayable_trace() -> None:
    source = _state("source", "s0")
    successor = _state("successor", "s1")
    forward = _action_between("s0", "s1", direction="forward")
    outcomes: tuple[TrajectoryGenerationOutcomeV1, ...] = generate_concrete_trajectory_attempts(
        source,
        run_seed=20260726,
        domain_mode="safe_real",
        rule_set_digest=sha256_digest(b"rules"),
        policy_digest=sha256_digest(b"policy"),
        policy=TrajectoryPolicyV1(
            min_trace_length=1,
            max_trace_length=1,
            max_attempts_per_source=1,
        ),
        enumerate_actions=lambda _state, _seed, _domain: (forward,),
        apply_action=lambda _state, _action, _domain: successor,
        verifier_version="fixture-verifier-v1",
    )

    assert len(outcomes) == 1
    assert outcomes[0].status is TrajectoryOutcomeStatus.ACCEPTED
    assert outcomes[0].trace is not None
    assert outcomes[0].trace.states == (source, successor)
    assert outcomes[0].trace.actions == (forward,)


def test_pair_shard_safe_resume_refuses_mismatched_bytes(tmp_path) -> None:
    output = tmp_path / "pairs.jsonl"
    record = _positive_record()
    arguments = {
        "shard_id": "train-0000",
        "config_digest": sha256_digest(b"config"),
        "input_digest": sha256_digest(b"inputs"),
        "rule_set_digest": sha256_digest(b"rules"),
        "depth_by_pair_id": {record.pair_id: 2},
    }
    first = write_pair_shard((record,), output, **arguments)
    resumed = write_pair_shard((record,), output, **arguments)

    assert resumed == first
    assert first.record_count == 1
    assert first.accepted_count == 1
    with pytest.raises(PairContractError, match="differs"):
        write_pair_shard((), output, **arguments)
