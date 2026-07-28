from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import yaml

from geml.data.steps.extract import (
    ActionDirectionV1,
    CanonicalJson,
    NormalizedActionV1,
    NormalizedStateV1,
    NormalizedTraceV1,
    ReplayResultV1,
    ReplayStatusV1,
    ResumeMismatchError,
    SplitLeakageError,
    SplitV1,
    StepDatasetConfigV1,
    StepDatasetProtocolError,
    StepFailureCodeV1,
    StepFailureV1,
    StepRecordV1,
    StepRuntimeIdentityV1,
    TraceInputError,
    VerificationResultV1,
    VerificationStatusV1,
    dataset_tree_digest,
    extract_step_dataset,
    extract_trace_steps,
    load_step_dataset_manifest,
    load_step_rows,
    write_step_dataset,
)
from geml.data.steps.stratify import (
    RuleRegistryEntryV1,
    RuleRegistrySnapshotV1,
    build_stratification_report,
)

RULE_DIGEST = hashlib.sha256(b"fixture-rule-set").hexdigest()
VERIFIER_DIGEST = hashlib.sha256(b"fixture-verifier").hexdigest()


def _digest(value: object) -> str:
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _refresh_manifest_dataset_digest(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {
        "accepted_count": manifest["accepted_count"],
        "failure_count": manifest["failure_count"],
        "input_digest": manifest["input_digest"],
        "receipts": manifest["shard_receipts"],
        "rule_set_digest": manifest["rule_set_digest"],
        "sidecar_digests": manifest["sidecar_digests"],
        "verifier_digest": manifest["verifier_digest"],
    }
    manifest["dataset_digest"] = hashlib.sha256(
        b"geml-step-dataset-content-v1\0" + _canonical_bytes(payload)
    ).hexdigest()
    _write_canonical_json(manifest_path, manifest)
    return manifest


def _rewrite_sidecar_and_rehash(
    root: Path,
    name: str,
    value: object,
) -> None:
    sidecar_path = root / name
    _write_canonical_json(sidecar_path, value)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sidecar_digests"][name] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    _write_canonical_json(manifest_path, manifest)
    _refresh_manifest_dataset_digest(root)


def _state(name: str, family: str = "algebraic_core") -> NormalizedStateV1:
    payload = {"fixture_expression": name}
    return NormalizedStateV1(
        state=CanonicalJson.from_value(payload),
        structural_signature=_digest({"state": payload}),
        family=family,
    )


def _action(
    source: NormalizedStateV1,
    successor: NormalizedStateV1,
    *,
    rule_id: str = "R-ADD-ZERO",
    direction: ActionDirectionV1 = ActionDirectionV1.FORWARD,
    path: tuple[int, ...] = (),
    arguments: tuple[object, ...] = (),
    replay_status: ReplayStatusV1 = ReplayStatusV1.APPLIED,
    verification_status: VerificationStatusV1 = VerificationStatusV1.ACCEPTED,
    stored_verification_status: VerificationStatusV1 = VerificationStatusV1.ACCEPTED,
    source_signature: str | None = None,
    successor_signature: str | None = None,
    stored_verifier_digest: str = VERIFIER_DIGEST,
) -> NormalizedActionV1:
    payload = {
        "direction": direction.value,
        "fixture_replay_status": replay_status.value,
        "fixture_successor": successor.state.to_value(),
        "fixture_successor_signature": successor.structural_signature,
        "fixture_verification_status": verification_status.value,
        "occurrence_path": list(path),
        "ordered_arguments": list(arguments),
        "rule_id": rule_id,
    }
    return NormalizedActionV1(
        action=CanonicalJson.from_value(payload),
        action_digest=_digest({"fixture-action": payload}),
        rule_id=rule_id,
        direction=direction,
        occurrence_path=path,
        ordered_arguments=tuple(CanonicalJson.from_value(value) for value in arguments),
        source_signature=source.structural_signature
        if source_signature is None
        else source_signature,
        successor_signature=successor.structural_signature
        if successor_signature is None
        else successor_signature,
        assumptions=("real",),
        domain_mode="safe_real",
        stored_verification_status=stored_verification_status,
        stored_verifier_digest=stored_verifier_digest,
        stored_verification_evidence_digest=_digest({"stored-verification": payload}),
    )


@dataclass(frozen=True)
class _FixtureInput:
    source_bytes: bytes
    normalized: NormalizedTraceV1 | TraceInputError


class _FixtureAdapter:
    def input_record_digest(self, trace: object) -> str:
        assert isinstance(trace, _FixtureInput)
        return hashlib.sha256(trace.source_bytes).hexdigest()

    def normalize_and_authenticate(self, trace: object) -> NormalizedTraceV1:
        assert isinstance(trace, _FixtureInput)
        if isinstance(trace.normalized, TraceInputError):
            raise trace.normalized
        return trace.normalized

    def structural_signature(self, state: CanonicalJson) -> str:
        return _digest({"state": state.to_value()})


class _FixtureReplayer:
    def apply(
        self,
        current_state: CanonicalJson,
        action: NormalizedActionV1,
    ) -> ReplayResultV1:
        del current_state
        payload = action.action.to_value()
        assert isinstance(payload, dict)
        status = ReplayStatusV1(payload["fixture_replay_status"])
        if status is not ReplayStatusV1.APPLIED:
            return ReplayResultV1(status=status, reason=f"fixture {status.value}")
        return ReplayResultV1(
            status=status,
            reason="fixture action applied",
            successor_state=CanonicalJson.from_value(payload["fixture_successor"]),
            successor_signature=payload["fixture_successor_signature"],
        )


class _ClaimingLyingReplayer:
    def apply(
        self,
        current_state: CanonicalJson,
        action: NormalizedActionV1,
    ) -> ReplayResultV1:
        del current_state
        payload = action.action.to_value()
        assert isinstance(payload, dict)
        return ReplayResultV1(
            status=ReplayStatusV1.APPLIED,
            reason="claims the stored signature for a different successor state",
            successor_state=CanonicalJson.from_value({"fixture_expression": "wrong"}),
            successor_signature=payload["fixture_successor_signature"],
        )


class _FixtureVerifier:
    def __init__(
        self,
        *,
        configured_digest: str = VERIFIER_DIGEST,
        result_digest: str | None = None,
    ) -> None:
        self._configured_digest = configured_digest
        self._result_digest = result_digest or configured_digest

    @property
    def verifier_digest(self) -> str:
        return self._configured_digest

    def verify(
        self,
        current_state: CanonicalJson,
        successor_state: CanonicalJson,
        action: NormalizedActionV1,
        *,
        assumptions: tuple[str, ...],
        domain_mode: str,
    ) -> VerificationResultV1:
        del current_state, successor_state
        assert assumptions == action.assumptions
        assert domain_mode == action.domain_mode
        payload = action.action.to_value()
        assert isinstance(payload, dict)
        status = VerificationStatusV1(payload["fixture_verification_status"])
        if status is VerificationStatusV1.TIMEOUT:
            raise TimeoutError("fixture timeout")
        if status is VerificationStatusV1.ERROR:
            raise RuntimeError("fixture verifier error")
        return VerificationResultV1(
            status=status,
            reason=f"fixture verifier {status.value}",
            verifier_digest=self._result_digest,
            evidence_digest=_digest(
                {
                    "action_digest": action.action_digest,
                    "verification": status.value,
                }
            ),
        )


def _trace(
    trace_id: str,
    states: tuple[NormalizedStateV1, ...],
    actions: tuple[NormalizedActionV1, ...],
    *,
    split: SplitV1 = SplitV1.TRAIN,
    source_group: str | None = None,
    lineage_groups: tuple[str, ...] | None = None,
    pair_id: str | None = None,
    source_id: str | None = None,
    stored_replay_verified: bool = True,
) -> _FixtureInput:
    source_group = source_group or f"group-{trace_id}"
    lineage_groups = lineage_groups or (source_group,)
    pair_id = pair_id or f"pair-{trace_id}"
    source_id = source_id or f"source-{trace_id}"
    source_bytes = (
        json.dumps(
            {
                "actions": [action.action.to_value() for action in actions],
                "pair_id": pair_id,
                "split": split.value,
                "states": [state.state.to_value() for state in states],
                "trace_id": trace_id,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    input_digest = hashlib.sha256(source_bytes).hexdigest()
    normalized = NormalizedTraceV1(
        trace_schema_version="fixture-rewrite-trace-v1",
        trace_id=trace_id,
        trace_digest=_digest({"authenticated-trace": source_bytes.decode()}),
        input_record_digest=input_digest,
        pair_id=pair_id,
        source_id=source_id,
        source_group=source_group,
        lineage_group_ids=tuple(sorted(lineage_groups)),
        authoritative_split=split,
        evaluation_views=("fixture",),
        source_family=states[0].family if states else "algebraic_core",
        domain_mode="safe_real",
        rewrite_mode="safe_real",
        rule_set_digest=RULE_DIGEST,
        authentication_evidence_digest=_digest({"authenticated": input_digest}),
        stored_replay_verified=stored_replay_verified,
        states=states,
        actions=actions,
    )
    return _FixtureInput(source_bytes=source_bytes, normalized=normalized)


def _bad_trace(reason: str = "bad schema") -> _FixtureInput:
    return _FixtureInput(
        source_bytes=b'{"malformed":true}\n',
        normalized=TraceInputError(
            StepFailureCodeV1.TRACE_AUTHENTICATION_FAILED,
            reason,
        ),
    )


def _extract(inputs: tuple[_FixtureInput, ...]):
    return extract_step_dataset(
        inputs,
        adapter=_FixtureAdapter(),
        replayer=_FixtureReplayer(),
        verifier=_FixtureVerifier(),
        expected_verifier_digest=VERIFIER_DIGEST,
        expected_rule_set_digest=RULE_DIGEST,
    )


def _registry() -> RuleRegistrySnapshotV1:
    entries = tuple(
        sorted(
            (
                RuleRegistryEntryV1(
                    rule_id="R-ADD-ZERO",
                    direction="backward",
                    supported=True,
                ),
                RuleRegistryEntryV1(
                    rule_id="R-ADD-ZERO",
                    direction="forward",
                    supported=True,
                ),
                RuleRegistryEntryV1(
                    rule_id="R-NEVER-SEEN",
                    direction="forward",
                    supported=True,
                ),
                RuleRegistryEntryV1(
                    rule_id="R-UNSUPPORTED",
                    direction="forward",
                    supported=False,
                ),
            ),
            key=lambda entry: (entry.rule_id, entry.direction),
        )
    )
    return RuleRegistrySnapshotV1(
        authoritative_rule_set_digest=RULE_DIGEST,
        source_schema_version="fixture-rule-registry-v1",
        source_content_digest=_digest([entry.as_dict() for entry in entries]),
        entries=entries,
    )


def _config(result) -> StepDatasetConfigV1:
    return StepDatasetConfigV1(
        seed=20260726,
        shard_size=2,
        expected_input_digest=result.input_digest,
        expected_rule_set_digest=result.rule_set_digest,
        expected_verifier_digest=result.verifier_digest,
        exact_command=("python -m geml.data.steps.extract --config configs/goal7_steps.yaml"),
        runtime=StepRuntimeIdentityV1(
            git_commit="998a139b09d232db5ed4ef4222d1e0dc778d3542",
            python_version="3.12.fixture",
            hardware="fixture-cpu",
            package_versions=(("geml", "0.1.0"),),
            deterministic_settings=("fixture_no_randomness",),
        ),
    )


def test_goal_conditioning_changes_identity_and_preserves_roles():
    current = _state("x + 0")
    goal_left = _state("x")
    goal_right = _state("0 + x")
    left = _trace(
        "left-goal",
        (current, goal_left),
        (_action(current, goal_left, path=()),),
        source_id="shared-source",
    )
    right = _trace(
        "right-goal",
        (current, goal_right),
        (_action(current, goal_right, path=(1,)),),
        source_id="shared-source",
    )

    result = _extract((right, left))

    assert not result.failures
    assert len(result.accepted) == 2
    by_goal = {row.goal_signature: row for row in result.accepted}
    assert by_goal[goal_left.structural_signature].occurrence_path == ()
    assert by_goal[goal_right.structural_signature].occurrence_path == (1,)
    assert len({row.record_id for row in result.accepted}) == 2
    assert all(row.current_signature == current.structural_signature for row in result.accepted)
    assert all(row.remaining_witness_steps == 1 for row in result.accepted)
    assert all(StepRecordV1.from_dict(row.as_dict()) == row for row in result.accepted)


def test_forward_backward_and_repeated_occurrences_remain_distinct():
    source = _state("(x + 0) + (x + 0)")
    successor = _state("(x + 0) + x")
    backward_successor = _state("((x + 0) + x) + 0")
    left_action = _action(source, successor, path=(0,))
    right_action = _action(source, successor, path=(1,))
    backward = _action(
        successor,
        backward_successor,
        direction=ActionDirectionV1.BACKWARD,
        path=(),
    )
    traces = (
        _trace("left-occurrence", (source, successor), (left_action,)),
        _trace("right-occurrence", (source, successor), (right_action,)),
        _trace("backward", (successor, backward_successor), (backward,)),
    )

    result = _extract(traces)

    assert not result.failures
    identities = {
        (row.direction, row.occurrence_path, row.action_digest) for row in result.accepted
    }
    assert len(identities) == 3
    assert {row.occurrence_depth for row in result.accepted} == {0, 1}
    assert any(row.direction is ActionDirectionV1.BACKWARD for row in result.accepted)


def test_alternative_actions_with_same_successor_keep_action_identity():
    current = _state("x + 0")
    goal = _state("x")
    first = _action(current, goal, rule_id="R-ADD-ZERO")
    second = _action(
        current,
        goal,
        rule_id="R-ALT",
        arguments=({"binding": "x"},),
    )
    result = _extract(
        (
            _trace("alternative-a", (current, goal), (first,)),
            _trace("alternative-b", (current, goal), (second,)),
        )
    )

    assert len(result.accepted) == 2
    assert len({row.next_signature for row in result.accepted}) == 1
    assert len({row.action_digest for row in result.accepted}) == 2
    assert len({row.record_id for row in result.accepted}) == 2


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ReplayStatusV1.AMBIGUOUS_SITE, StepFailureCodeV1.AMBIGUOUS_SITE),
        (ReplayStatusV1.INVALID_SITE, StepFailureCodeV1.INVALID_SITE),
        (ReplayStatusV1.INVALID_ARGUMENTS, StepFailureCodeV1.INVALID_ARGUMENTS),
        (ReplayStatusV1.MISSING_RULE, StepFailureCodeV1.MISSING_RULE),
        (ReplayStatusV1.MISSING_DIRECTION, StepFailureCodeV1.MISSING_DIRECTION),
        (
            ReplayStatusV1.UNSUPPORTED_OPERATOR,
            StepFailureCodeV1.UNSUPPORTED_OPERATOR,
        ),
        (ReplayStatusV1.UNSUPPORTED_DOMAIN, StepFailureCodeV1.UNSUPPORTED_DOMAIN),
    ],
)
def test_replay_failures_are_typed(status, expected):
    current = _state("source")
    goal = _state("goal")
    action = _action(current, goal, replay_status=status)

    result = _extract((_trace(f"failure-{status.value}", (current, goal), (action,)),))

    assert not result.accepted
    assert [row.failure_code for row in result.failures] == [expected]


def test_one_failed_transition_invalidates_the_whole_witness_suffix():
    first = _state("a")
    second = _state("b")
    goal = _state("c")
    actions = (
        _action(first, second),
        _action(second, goal, replay_status=ReplayStatusV1.AMBIGUOUS_SITE),
    )

    result = _extract((_trace("partial", (first, second, goal), actions),))

    assert not result.accepted
    assert [row.failure_code for row in result.failures] == [
        StepFailureCodeV1.INCOMPLETE_TRACE,
        StepFailureCodeV1.AMBIGUOUS_SITE,
    ]
    retained_fresh_result = result.failures[0]
    assert retained_fresh_result.verification_status is VerificationStatusV1.ACCEPTED
    assert retained_fresh_result.verification_evidence_digest is not None
    assert retained_fresh_result.verifier_digest == VERIFIER_DIGEST
    failed_before_fresh_verification = result.failures[1]
    assert failed_before_fresh_verification.verification_status is None
    assert failed_before_fresh_verification.verification_evidence_digest is None
    assert failed_before_fresh_verification.verifier_digest is None
    assert result.replay_audit.attempted_step_count == 2


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (VerificationStatusV1.REJECTED, StepFailureCodeV1.VERIFIER_REJECTED),
        (VerificationStatusV1.UNSUPPORTED, StepFailureCodeV1.VERIFIER_UNSUPPORTED),
        (VerificationStatusV1.TIMEOUT, StepFailureCodeV1.VERIFIER_TIMEOUT),
        (VerificationStatusV1.ERROR, StepFailureCodeV1.VERIFIER_ERROR),
    ],
)
def test_verifier_failures_and_exceptions_are_retained(status, expected):
    current = _state("source")
    goal = _state("goal")
    action = _action(current, goal, verification_status=status)

    result = _extract((_trace(f"verification-{status.value}", (current, goal), (action,)),))

    assert not result.accepted
    assert result.failures[0].failure_code is expected


def test_verifier_identity_is_pinned_and_evidence_channels_remain_distinct():
    current = _state("source")
    goal = _state("goal")
    other_verifier = _digest("other-verifier")
    producer_mismatch = _trace(
        "producer-verifier-mismatch",
        (current, goal),
        (
            _action(
                current,
                goal,
                stored_verifier_digest=other_verifier,
            ),
        ),
    )
    producer_result = _extract((producer_mismatch,))

    assert not producer_result.accepted
    producer_failure = producer_result.failures[0]
    assert producer_failure.failure_code is StepFailureCodeV1.VERIFIER_IDENTITY_MISMATCH
    assert producer_failure.producer_verifier_digest == other_verifier
    assert producer_failure.verifier_digest is None
    assert producer_failure.producer_verification_evidence_digest is not None

    fresh_mismatch = _trace(
        "fresh-verifier-mismatch",
        (current, goal),
        (_action(current, goal),),
    )
    result = extract_step_dataset(
        (fresh_mismatch,),
        adapter=_FixtureAdapter(),
        replayer=_FixtureReplayer(),
        verifier=_FixtureVerifier(result_digest=other_verifier),
        expected_verifier_digest=VERIFIER_DIGEST,
        expected_rule_set_digest=RULE_DIGEST,
    )
    fresh_failure = result.failures[0]
    assert fresh_failure.failure_code is StepFailureCodeV1.VERIFIER_IDENTITY_MISMATCH
    assert fresh_failure.producer_verifier_digest == VERIFIER_DIGEST
    assert fresh_failure.verifier_digest == other_verifier
    assert (
        fresh_failure.producer_verification_evidence_digest
        != fresh_failure.verification_evidence_digest
    )

    with pytest.raises(StepDatasetProtocolError, match="configured verifier digest"):
        extract_step_dataset(
            (),
            adapter=_FixtureAdapter(),
            replayer=_FixtureReplayer(),
            verifier=_FixtureVerifier(configured_digest=other_verifier),
            expected_verifier_digest=VERIFIER_DIGEST,
            expected_rule_set_digest=RULE_DIGEST,
        )


def test_digest_mismatch_failures_use_adapter_computed_source_digest():
    current = _state("source")
    goal = _state("goal")
    valid = _trace("untrusted-claimed-digest", (current, goal), (_action(current, goal),))
    claimed_digest = _digest("untrusted claimed digest")
    assert isinstance(valid.normalized, NormalizedTraceV1)
    mismatched = replace(
        valid,
        normalized=replace(
            valid.normalized,
            input_record_digest=claimed_digest,
        ),
    )
    exact_source_digest = hashlib.sha256(mismatched.source_bytes).hexdigest()

    _, direct_failures, authenticated = extract_trace_steps(
        mismatched,
        adapter=_FixtureAdapter(),
        replayer=_FixtureReplayer(),
        verifier=_FixtureVerifier(),
        expected_verifier_digest=VERIFIER_DIGEST,
    )
    _, claimed_failures, claimed_authenticated = extract_trace_steps(
        mismatched,
        adapter=_FixtureAdapter(),
        replayer=_FixtureReplayer(),
        verifier=_FixtureVerifier(),
        input_record_digest=claimed_digest,
        expected_verifier_digest=VERIFIER_DIGEST,
    )
    dataset_result = _extract((mismatched,))

    assert authenticated is False
    assert claimed_authenticated is False
    assert direct_failures[0].input_record_digest == exact_source_digest
    assert claimed_failures[0].input_record_digest == exact_source_digest
    assert dataset_result.failures[0].input_record_digest == exact_source_digest
    assert direct_failures[0].input_record_digest != claimed_digest


def test_replay_claimed_signature_cannot_hide_a_different_successor_state():
    current = _state("source")
    goal = _state("goal")
    trace = _trace("lying-replayer", (current, goal), (_action(current, goal),))

    result = extract_step_dataset(
        (trace,),
        adapter=_FixtureAdapter(),
        replayer=_ClaimingLyingReplayer(),
        verifier=_FixtureVerifier(),
        expected_verifier_digest=VERIFIER_DIGEST,
        expected_rule_set_digest=RULE_DIGEST,
    )

    assert not result.accepted
    assert result.failures[0].failure_code is StepFailureCodeV1.SUCCESSOR_SIGNATURE_MISMATCH
    assert result.failures[0].verification_status is None
    assert result.failures[0].verification_evidence_digest is None
    assert result.failures[0].verifier_digest is None


def test_producer_rejection_does_not_impersonate_a_fresh_verifier_result():
    current = _state("source")
    goal = _state("goal")
    action = _action(
        current,
        goal,
        stored_verification_status=VerificationStatusV1.REJECTED,
    )

    result = _extract((_trace("producer-rejected", (current, goal), (action,)),))

    assert not result.accepted
    failure = result.failures[0]
    assert failure.failure_code is StepFailureCodeV1.VERIFIER_REJECTED
    assert failure.producer_verification_status is VerificationStatusV1.REJECTED
    assert failure.producer_verifier_digest == VERIFIER_DIGEST
    assert failure.producer_verification_evidence_digest is not None
    assert failure.verification_status is None
    assert failure.verification_evidence_digest is None
    assert failure.verifier_digest is None
    assert StepFailureV1.from_dict(failure.as_dict()) == failure


def test_signature_mismatch_zero_length_corrupt_and_auth_failure_rows():
    current = _state("source")
    goal = _state("goal")
    mismatch = _action(
        current,
        goal,
        source_signature=_digest("different-source"),
    )
    successor_mismatch = _action(
        current,
        goal,
        successor_signature=_digest("different-successor"),
    )
    zero = _trace("zero", (current,), ())
    corrupt = _trace("corrupt", (current, goal, _state("extra")), (_action(current, goal),))
    mismatch_trace = _trace("mismatch", (current, goal), (mismatch,))
    successor_mismatch_trace = _trace(
        "successor-mismatch",
        (current, goal),
        (successor_mismatch,),
    )

    result = _extract((mismatch_trace, successor_mismatch_trace, zero, corrupt, _bad_trace()))

    assert not result.accepted
    assert {row.failure_code for row in result.failures} == {
        StepFailureCodeV1.SOURCE_SIGNATURE_MISMATCH,
        StepFailureCodeV1.SUCCESSOR_SIGNATURE_MISMATCH,
        StepFailureCodeV1.ZERO_LENGTH_TRACE,
        StepFailureCodeV1.CORRUPT_TRACE,
        StepFailureCodeV1.TRACE_AUTHENTICATION_FAILED,
    }
    assert result.replay_audit.input_trace_count == 5
    assert result.replay_audit.zero_length_trace_count == 1
    assert all(StepFailureV1.from_dict(row.as_dict()) == row for row in result.failures)


def test_multi_action_corrupt_trace_is_retained_without_fake_replay_attempts():
    current = _state("source")
    goal = _state("goal")
    action = _action(current, goal)
    trace = _trace("multi-action-corrupt", (current, goal), (action,))
    assert isinstance(trace.normalized, NormalizedTraceV1)
    corrupt = replace(
        trace,
        normalized=replace(trace.normalized, actions=(action, action, action)),
    )

    result = _extract((corrupt,))

    assert not result.accepted
    assert len(result.failures) == 1
    assert result.failures[0].failure_code is StepFailureCodeV1.CORRUPT_TRACE
    assert result.replay_audit.attempted_step_count == 0
    assert result.replay_audit.failure_row_count == 1


def test_group_trace_source_and_pair_cross_split_leakage_is_rejected():
    source = _state("source")
    goal = _state("goal")
    action = _action(source, goal)
    first = _trace(
        "trace-a",
        (source, goal),
        (action,),
        split=SplitV1.TRAIN,
        source_group="shared-group",
    )
    second = _trace(
        "trace-b",
        (source, goal),
        (action,),
        split=SplitV1.TEST_IID,
        source_group="shared-group",
    )

    result = _extract((first, second))

    assert not result.accepted
    assert len(result.failures) == 2
    assert all(row.failure_code is StepFailureCodeV1.GROUP_LEAKAGE for row in result.failures)
    assert result.replay_audit.attempted_step_count == 0


@pytest.mark.parametrize(
    ("identity_kind", "expected_key"),
    [
        ("pair", "pair:shared-pair"),
        ("source", "source:shared-source"),
        ("trace", "trace:shared-trace"),
        ("eclass", "lineage:eclass-relative:shared"),
        ("derived", "lineage:derived-relative:shared"),
    ],
)
def test_each_partition_identity_is_a_distinct_leakage_barrier(
    identity_kind,
    expected_key,
):
    source = _state("source")
    goal = _state("goal")
    action = _action(source, goal)
    first_trace_id = "identity-a"
    second_trace_id = "identity-b"
    first_pair = "pair-a"
    second_pair = "pair-b"
    first_source = "source-a"
    second_source = "source-b"
    first_group = "group-a"
    second_group = "group-b"
    first_lineage = (first_group,)
    second_lineage = (second_group,)
    if identity_kind == "pair":
        first_pair = second_pair = "shared-pair"
    elif identity_kind == "source":
        first_source = second_source = "shared-source"
    elif identity_kind == "trace":
        first_trace_id = second_trace_id = "shared-trace"
    elif identity_kind == "eclass":
        first_lineage = tuple(sorted((first_group, "eclass-relative:shared")))
        second_lineage = tuple(sorted((second_group, "eclass-relative:shared")))
    else:
        first_lineage = tuple(sorted((first_group, "derived-relative:shared")))
        second_lineage = tuple(sorted((second_group, "derived-relative:shared")))

    first = _trace(
        first_trace_id,
        (source, goal),
        (action,),
        split=SplitV1.TRAIN,
        source_group=first_group,
        lineage_groups=first_lineage,
        pair_id=first_pair,
        source_id=first_source,
    )
    second = _trace(
        second_trace_id,
        (source, goal),
        (action,),
        split=SplitV1.TEST_OOD,
        source_group=second_group,
        lineage_groups=second_lineage,
        pair_id=second_pair,
        source_id=second_source,
    )

    result = _extract((first, second))

    assert not result.accepted
    assert len(result.failures) == 2
    assert all(row.failure_code is StepFailureCodeV1.GROUP_LEAKAGE for row in result.failures)
    assert all(expected_key in row.reason for row in result.failures)


def test_duplicate_trace_identity_is_never_duplicated_into_training():
    source = _state("source")
    goal = _state("goal")
    first = _trace("duplicate", (source, goal), (_action(source, goal),))
    second = _trace("duplicate", (source, goal), (_action(source, goal),))

    result = _extract((first, second))

    assert not result.accepted
    assert len(result.failures) == 2
    assert all(row.failure_code is StepFailureCodeV1.CORRUPT_TRACE for row in result.failures)
    assert {row.input_occurrence_index for row in result.failures} == {0, 1}
    assert len({row.failure_id for row in result.failures}) == 2


def test_stratification_is_descriptive_and_includes_zero_and_unsupported_rules():
    source = _state("source")
    goal = _state("goal", family="exp_log")
    accepted_result = _extract((_trace("accepted", (source, goal), (_action(source, goal),)),))
    invalid = _extract(
        (
            _trace(
                "invalid",
                (source, goal),
                (
                    _action(
                        source,
                        goal,
                        replay_status=ReplayStatusV1.INVALID_SITE,
                    ),
                ),
            ),
        )
    )
    report = build_stratification_report(
        accepted_result.accepted,
        invalid.failures,
        _registry(),
    )

    coverage = {(row.rule_id, row.direction): row for row in report.rule_direction_coverage}
    assert coverage[("R-ADD-ZERO", "forward")].accepted_count == 1
    assert coverage[("R-ADD-ZERO", "forward")].failure_count == 1
    assert coverage[("R-NEVER-SEEN", "forward")].total_count == 0
    assert coverage[("R-UNSUPPORTED", "forward")].supported is False
    assert sum(dict(report.current_family_counts).values()) == 2
    assert report.as_dict()["accepted_count"] == 1


def test_deterministic_resumable_shards_and_identical_two_run_tree_hashes(tmp_path):
    source = _state("source")
    middle = _state("middle")
    goal = _state("goal")
    good = _trace(
        "good",
        (source, middle, goal),
        (_action(source, middle), _action(middle, goal)),
    )
    invalid = _trace(
        "invalid",
        (source, goal),
        (_action(source, goal, replay_status=ReplayStatusV1.INVALID_SITE),),
    )
    result = _extract((invalid, good))
    config = _config(result)
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest_a = write_step_dataset(
        result,
        output_root=first,
        config=config,
        frozen_registry=_registry(),
    )
    manifest_b = write_step_dataset(
        result,
        output_root=second,
        config=config,
        frozen_registry=_registry(),
    )
    resumed = write_step_dataset(
        result,
        output_root=first,
        config=config,
        frozen_registry=_registry(),
        resume=True,
    )

    assert manifest_a == manifest_b == resumed
    assert manifest_a.verifier_digest == VERIFIER_DIGEST
    assert config.scientific_payload()["expected_verifier_digest"] == VERIFIER_DIGEST
    assert (
        replace(config, expected_verifier_digest=_digest("different-verifier")).config_digest
        != config.config_digest
    )
    assert dataset_tree_digest(first) == dataset_tree_digest(second)
    assert {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    loaded_manifest = load_step_dataset_manifest(
        first / "manifest.json",
        expected_config_digest=config.config_digest,
        expected_input_digest=result.input_digest,
        expected_rule_set_digest=RULE_DIGEST,
        expected_verifier_digest=VERIFIER_DIGEST,
    )
    assert loaded_manifest == manifest_a
    with pytest.raises(StepDatasetProtocolError, match="verifier digest"):
        load_step_dataset_manifest(
            first / "manifest.json",
            expected_verifier_digest=_digest("wrong-verifier"),
        )
    loaded_rows = load_step_rows(first / "manifest.json")
    assert sum(isinstance(row, StepRecordV1) for row in loaded_rows) == 2
    assert sum(isinstance(row, StepFailureV1) for row in loaded_rows) == 1


def test_writer_fails_before_publication_on_accepted_split_leakage(tmp_path):
    source = _state("source")
    goal = _state("goal")
    result = _extract(
        (
            _trace("leak-a", (source, goal), (_action(source, goal),)),
            _trace("leak-b", (source, goal), (_action(source, goal),)),
        )
    )
    first, second = result.accepted
    leaking_result = replace(
        result,
        accepted=(
            first,
            replace(
                second,
                source_id=first.source_id,
                authoritative_split=SplitV1.TEST_IID,
                record_id="",
            ),
        ),
    )
    root = tmp_path / "leaking"

    with pytest.raises(SplitLeakageError, match="accepted rows cross"):
        write_step_dataset(
            leaking_result,
            output_root=root,
            config=_config(leaking_result),
            frozen_registry=_registry(),
        )

    assert not root.exists()


@pytest.mark.parametrize(
    ("rule_id", "expected_message"),
    [
        ("R-ABSENT", "absent from the frozen registry"),
        ("R-UNSUPPORTED", "unsupported frozen registry"),
    ],
)
def test_writer_fails_before_publication_on_unavailable_accepted_rule(
    tmp_path,
    rule_id,
    expected_message,
):
    source = _state("source")
    goal = _state("goal")
    result = _extract((_trace("invalid-rule", (source, goal), (_action(source, goal),)),))
    invalid_result = replace(
        result,
        accepted=(
            replace(
                result.accepted[0],
                rule_id=rule_id,
                record_id="",
            ),
        ),
    )
    root = tmp_path / rule_id.lower()

    with pytest.raises(StepDatasetProtocolError, match=expected_message):
        write_step_dataset(
            invalid_result,
            output_root=root,
            config=_config(invalid_result),
            frozen_registry=_registry(),
        )

    assert not root.exists()


def test_resume_removes_only_exact_writer_orphan_temporaries(tmp_path):
    source = _state("source")
    goal = _state("goal")
    result = _extract((_trace("resume-orphan", (source, goal), (_action(source, goal),)),))
    root = tmp_path / "dataset"
    config = _config(result)
    write_step_dataset(
        result,
        output_root=root,
        config=config,
        frozen_registry=_registry(),
    )
    exact_orphan = root / "shards" / ".shard-00000.jsonl.abcdefgh.tmp"
    exact_orphan.write_bytes(b"interrupted immutable write")

    write_step_dataset(
        result,
        output_root=root,
        config=config,
        frozen_registry=_registry(),
    )

    assert not exact_orphan.exists()
    arbitrary_file = root / "shards" / ".shard-00000.jsonl.abcdefghi.tmp"
    arbitrary_file.write_bytes(b"not an exact writer temporary")
    unrelated_file = root / ".unrelated.abcdefgh.tmp"
    unrelated_file.write_bytes(b"unrelated exact-looking temporary")
    with pytest.raises(ResumeMismatchError, match="unexpected files"):
        write_step_dataset(
            result,
            output_root=root,
            config=config,
            frozen_registry=_registry(),
        )
    assert arbitrary_file.is_file()
    assert unrelated_file.is_file()


def test_resume_and_recursive_load_refuse_mutation(tmp_path):
    source = _state("source")
    goal = _state("goal")
    result = _extract((_trace("one", (source, goal), (_action(source, goal),)),))
    root = tmp_path / "dataset"
    write_step_dataset(
        result,
        output_root=root,
        config=_config(result),
        frozen_registry=_registry(),
    )
    shard = next((root / "shards").glob("*.jsonl"))
    shard.write_bytes(shard.read_bytes() + b" ")

    with pytest.raises(StepDatasetProtocolError, match="shard bytes differ"):
        load_step_dataset_manifest(root / "manifest.json")
    with pytest.raises(ResumeMismatchError, match="differs"):
        write_step_dataset(
            result,
            output_root=root,
            config=_config(result),
            frozen_registry=_registry(),
        )


def test_loader_cross_binds_config_sidecar_even_after_self_consistent_rehash(tmp_path):
    source = _state("source")
    goal = _state("goal")
    result = _extract((_trace("config-binding", (source, goal), (_action(source, goal),)),))
    root = tmp_path / "dataset"
    write_step_dataset(
        result,
        output_root=root,
        config=_config(result),
        frozen_registry=_registry(),
    )

    config_path = root / "config.json"
    config_value = json.loads(config_path.read_text(encoding="utf-8"))
    config_value["exact_command"] = "python malicious-or-stale-command.py"
    _write_canonical_json(config_path, config_value)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sidecar_digests"]["config.json"] = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    _write_canonical_json(manifest_path, manifest)
    _refresh_manifest_dataset_digest(root)

    with pytest.raises(StepDatasetProtocolError, match="exact command differs"):
        load_step_dataset_manifest(manifest_path)


@pytest.mark.parametrize(
    "sidecar_name",
    [
        "split-audit.json",
        "replay-audit.json",
        "per-rule-manifest.json",
    ],
)
def test_loader_rejects_self_consistent_scientific_sidecar_forgery(
    tmp_path,
    sidecar_name,
):
    source = _state("source")
    goal = _state("goal")
    result = _extract(
        (_trace(f"sidecar-{sidecar_name}", (source, goal), (_action(source, goal),)),)
    )
    root = tmp_path / sidecar_name.removesuffix(".json")
    write_step_dataset(
        result,
        output_root=root,
        config=_config(result),
        frozen_registry=_registry(),
    )

    sidecar_path = root / sidecar_name
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar_name == "split-audit.json":
        payload["accepted_leakage_count"] = 1
        payload["accepted_leakages"] = {"lineage:forged-cross-split-group": ["test_iid", "train"]}
        payload["status"] = "failed"
    elif sidecar_name == "replay-audit.json":
        payload["accepted_trace_count"] = 0
        payload["failed_trace_count"] = 1
    else:
        payload["current_family_counts"] = {"forged-family": 1}
    _rewrite_sidecar_and_rehash(root, sidecar_name, payload)

    with pytest.raises(
        StepDatasetProtocolError,
        match="does not reconstruct",
    ):
        load_step_dataset_manifest(root / "manifest.json")


def test_loader_typed_parses_rows_even_after_receipt_rehash(tmp_path):
    source = _state("source")
    goal = _state("goal")
    result = _extract((_trace("typed-row", (source, goal), (_action(source, goal),)),))
    root = tmp_path / "dataset"
    write_step_dataset(
        result,
        output_root=root,
        config=_config(result),
        frozen_registry=_registry(),
    )

    shard_path = root / "shards" / "shard-00000.jsonl"
    envelopes = [json.loads(line) for line in shard_path.read_text(encoding="utf-8").splitlines()]
    envelopes[0]["row"]["current_family"] = "tampered-family"
    shard_data = b"".join(_canonical_bytes(envelope) + b"\n" for envelope in envelopes)
    shard_path.write_bytes(shard_data)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = manifest["shard_receipts"][0]
    receipt["byte_count"] = len(shard_data)
    receipt["sha256"] = hashlib.sha256(shard_data).hexdigest()
    _write_canonical_json(manifest_path, manifest)
    _refresh_manifest_dataset_digest(root)

    with pytest.raises(StepDatasetProtocolError, match="record_id"):
        load_step_dataset_manifest(manifest_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "shards/stale.jsonl",
        "shards/.shard-00000.jsonl.abcdefgh.tmp",
    ],
)
def test_loader_rejects_unlisted_stale_files(tmp_path, relative_path):
    source = _state("source")
    goal = _state("goal")
    result = _extract((_trace("stale-file", (source, goal), (_action(source, goal),)),))
    root = tmp_path / "dataset"
    write_step_dataset(
        result,
        output_root=root,
        config=_config(result),
        frozen_registry=_registry(),
    )
    stale = root / relative_path
    stale.write_text("{}\n", encoding="utf-8")

    with pytest.raises(StepDatasetProtocolError, match="unlisted files"):
        load_step_dataset_manifest(root / "manifest.json")


def test_input_order_does_not_change_rows_or_input_digest():
    first_source = _state("a")
    first_goal = _state("b")
    second_source = _state("c")
    second_goal = _state("d")
    first = _trace(
        "first",
        (first_source, first_goal),
        (_action(first_source, first_goal),),
    )
    second = _trace(
        "second",
        (second_source, second_goal),
        (_action(second_source, second_goal),),
    )

    forward = _extract((first, second))
    reverse = _extract((second, first))

    assert forward.input_digest == reverse.input_digest
    assert [row.as_dict() for row in forward.accepted] == [
        row.as_dict() for row in reverse.accepted
    ]


def test_invalid_hashes_config_mismatch_and_noncanonical_json_fail_closed(tmp_path):
    with pytest.raises(StepDatasetProtocolError, match="canonical JSON"):
        CanonicalJson('{"b":1,"a":2}')
    with pytest.raises(StepDatasetProtocolError, match="SHA-256"):
        replace(
            _action(_state("a"), _state("b")),
            action_digest="not-a-hash",
        )

    source = _state("source")
    goal = _state("goal")
    result = _extract((_trace("one", (source, goal), (_action(source, goal),)),))
    wrong = replace(_config(result), expected_input_digest="0" * 64)
    with pytest.raises(StepDatasetProtocolError, match="config/input"):
        write_step_dataset(
            result,
            output_root=tmp_path,
            config=wrong,
            frozen_registry=_registry(),
        )


def test_config_yaml_freezes_phase_a_and_production_contract():
    path = Path("configs/goal7_steps.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "geml-step-dataset-config-v1"
    assert payload["phase"] == "phase_a_implemented_production_pending"
    assert payload["output_root"] == "outputs/final/goal7/steps"
    assert payload["seed"] == 20260726
    assert payload["resume"] is True
    assert payload["expected_input_digest"] is None
    assert payload["expected_rule_set_digest"] is None
    assert payload["expected_verifier_digest"] is None
    assert payload["production_command"] is None
    assert payload["production_command_status"] == "pending_workstream_1_integration"
    assert payload["production_providers"]["trace_adapter"] is None
    assert payload["production_providers"]["action_replayer"] is None
    assert payload["production_providers"]["transition_verifier"] is None
