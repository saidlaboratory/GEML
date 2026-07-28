from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import pytest

from geml.data.steps.extract import (
    ActionDirectionV1,
    CanonicalJson,
    NormalizedActionV1,
    SplitV1,
    StepRecordV1,
    VerificationStatusV1,
)
from geml.learning.eval.step_metrics import (
    FAMILY_PARTITION_EVIDENCE_SCHEMA_VERSION,
    ActionIdentityV1,
    CandidateMetricStatus,
    ExampleMetricStatus,
    FamilyGeneralization,
    FamilyPartitionEvidenceV1,
    LegalityResultV1,
    LegalityStatus,
    ReplayResultV1,
    ReplayStatus,
    StepMetricOutcomeV1,
    StepRecordProtocol,
    VerificationResultV1,
    VerificationStatus,
    aggregate_step_metrics,
    canonical_ordered_arguments,
    evaluate_step,
)
from geml.learning.policy.head import (
    ModelIdentityV1,
    ProposalCandidateV1,
    ProposalStatus,
    ProposalV1,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


RULE_DIGEST = _digest("registry")
LEGAL_MASK_DIGEST = _digest("legal-mask")
RECORD_ID = _digest("record-1")
CURRENT_SIGNATURE = _digest("current")
GOAL_SIGNATURE = _digest("goal")
DEMONSTRATED_SUCCESSOR = _digest("demonstrated-successor")
REGISTERED_RULES = ("add_zero", "commute_add", "factor", "unused_rule")
REGISTERED_RULE_DIRECTIONS = (
    ("add_zero", "forward"),
    ("commute_add", "backward"),
    ("commute_add", "forward"),
    ("factor", "forward"),
    ("unused_rule", "forward"),
)
TOP_KS = (1, 3, 5)
FAMILY_EVIDENCE = FamilyPartitionEvidenceV1(
    schema_version=FAMILY_PARTITION_EVIDENCE_SCHEMA_VERSION,
    step_manifest_digest=_digest("step-manifest"),
    training_family_ids=("algebra",),
)


@dataclass(frozen=True)
class _CanonicalJson:
    text: str

    def to_value(self) -> object:
        return json.loads(self.text)


@dataclass(frozen=True)
class _BrokenCanonicalJson:
    text: str

    def to_value(self) -> object:
        return {"different": True}


@dataclass(frozen=True)
class _Action:
    rule_id: str
    direction: str
    occurrence_path: tuple[int, ...]
    ordered_arguments: tuple[object, ...]
    action_digest: str
    successor_signature: str | None
    legality: LegalityStatus = LegalityStatus.LEGAL
    replay_status: ReplayStatus = ReplayStatus.SUCCEEDED
    verification: VerificationStatus | str = VerificationStatus.ACCEPTED


@dataclass(frozen=True)
class _Record:
    record_id: str = RECORD_ID
    trace_id: str = "trace-1"
    source_group: str = "group-1"
    lineage_group_ids: tuple[str, ...] = ("group-1",)
    authoritative_split: str = "test_iid"
    current_state: object = "current-state"
    current_signature: str = CURRENT_SIGNATURE
    goal_signature: str = GOAL_SIGNATURE
    next_signature: str = DEMONSTRATED_SUCCESSOR
    rule_id: str = "add_zero"
    direction: str = "forward"
    occurrence_path: tuple[int, ...] = (0,)
    ordered_arguments: tuple[object, ...] = (_CanonicalJson('{"binding":"x"}'),)
    action_digest: str = _digest("demonstrated-action")
    current_family: str = "algebra"
    goal_family: str = "algebra"
    evaluation_views: tuple[str, ...] = ("iid",)
    remaining_witness_steps: int = 2
    trace_length: int = 3
    rule_set_digest: str = RULE_DIGEST
    supported: bool = True


@dataclass(frozen=True)
class _Candidate:
    rank: int
    action: object


@dataclass(frozen=True)
class _Proposal:
    current_signature: str
    goal_signature: str
    candidates: tuple[_Candidate, ...]
    legal_action_count: int
    requested_top_k: int = 5
    legal_mask_digest: str = LEGAL_MASK_DIGEST
    rule_registry_digest: str = RULE_DIGEST
    status: str = "success"


class _Adapter:
    def __init__(self) -> None:
        self.replayed: list[str] = []
        self.verified: list[str] = []

    def action_identity(self, action: object) -> ActionIdentityV1:
        if not isinstance(action, _Action):
            raise TypeError("fixture action has the wrong schema")
        return ActionIdentityV1(
            rule_id=action.rule_id,
            direction=action.direction,
            occurrence_path=action.occurrence_path,
            ordered_arguments_json=canonical_ordered_arguments(action.ordered_arguments),
            action_digest=action.action_digest,
        )

    def classify_legality(
        self,
        record: _Record,
        action: object,
    ) -> LegalityResultV1:
        del record
        assert isinstance(action, _Action)
        return LegalityResultV1(action.legality, f"legality={action.legality.value}")

    def replay(self, record: _Record, action: object) -> ReplayResultV1:
        del record
        assert isinstance(action, _Action)
        self.replayed.append(action.action_digest)
        if action.replay_status is ReplayStatus.SUCCEEDED:
            return ReplayResultV1(
                status=ReplayStatus.SUCCEEDED,
                successor_signature=action.successor_signature,
                successor_state={"signature": action.successor_signature},
                detail="fixture replay succeeded",
            )
        return ReplayResultV1(
            status=action.replay_status,
            successor_signature=None,
            successor_state=None,
            detail=f"replay={action.replay_status.value}",
        )

    def structural_signature(self, successor_state: object) -> str:
        if not isinstance(successor_state, dict):
            raise TypeError("fixture successor state must be a dictionary")
        signature = successor_state.get("signature")
        if not isinstance(signature, str):
            raise TypeError("fixture successor state lacks its structural signature")
        return signature

    def verify(
        self,
        record: _Record,
        action: object,
        replay: ReplayResultV1,
    ) -> VerificationResultV1:
        del record, replay
        assert isinstance(action, _Action)
        self.verified.append(action.action_digest)
        if action.verification == "raise_timeout":
            raise TimeoutError("fixture verifier deadline")
        if action.verification == "raise_error":
            raise RuntimeError("fixture verifier failure")
        assert isinstance(action.verification, VerificationStatus)
        return VerificationResultV1(
            action.verification,
            f"verification={action.verification.value}",
        )


class _ConcreteAdapter:
    def action_identity(self, action: object) -> ActionIdentityV1:
        assert isinstance(action, NormalizedActionV1)
        return ActionIdentityV1(
            rule_id=action.rule_id,
            direction=action.direction.value,
            occurrence_path=action.occurrence_path,
            ordered_arguments_json=canonical_ordered_arguments(action.ordered_arguments),
            action_digest=action.action_digest,
        )

    def classify_legality(
        self,
        record: StepRecordV1,
        action: object,
    ) -> LegalityResultV1:
        del record
        assert isinstance(action, NormalizedActionV1)
        return LegalityResultV1(LegalityStatus.LEGAL, "fixture registry accepted")

    def replay(
        self,
        record: StepRecordV1,
        action: object,
    ) -> ReplayResultV1:
        assert isinstance(action, NormalizedActionV1)
        return ReplayResultV1(
            status=ReplayStatus.SUCCEEDED,
            successor_signature=action.successor_signature,
            successor_state=record.next_state,
            detail="fixture replayer applied the normalized action",
        )

    def structural_signature(self, successor_state: object) -> str:
        if not isinstance(successor_state, CanonicalJson):
            raise TypeError("fixture successor state must be CanonicalJson")
        if successor_state.to_value() != {"expr": "x"}:
            raise ValueError("fixture successor state is not the expected exact structure")
        return DEMONSTRATED_SUCCESSOR

    def verify(
        self,
        record: StepRecordV1,
        action: object,
        replay: ReplayResultV1,
    ) -> VerificationResultV1:
        del record, replay
        assert isinstance(action, NormalizedActionV1)
        return VerificationResultV1(
            VerificationStatus.ACCEPTED,
            "fixture verifier accepted",
        )


def _action(
    rule_id: str,
    *,
    direction: str = "forward",
    path: tuple[int, ...] = (0,),
    arguments: tuple[object, ...] | None = None,
    digest: str | None = None,
    successor: str | None = DEMONSTRATED_SUCCESSOR,
    legality: LegalityStatus = LegalityStatus.LEGAL,
    replay_status: ReplayStatus = ReplayStatus.SUCCEEDED,
    verification: VerificationStatus | str = VerificationStatus.ACCEPTED,
) -> _Action:
    return _Action(
        rule_id=rule_id,
        direction=direction,
        occurrence_path=path,
        ordered_arguments=(_CanonicalJson('{"binding":"x"}'),) if arguments is None else arguments,
        action_digest=digest or _digest(f"{rule_id}-{direction}-{path}"),
        successor_signature=successor,
        legality=legality,
        replay_status=replay_status,
        verification=verification,
    )


def _proposal(*actions: _Action, legal_action_count: int | None = None) -> _Proposal:
    return _Proposal(
        current_signature=CURRENT_SIGNATURE,
        goal_signature=GOAL_SIGNATURE,
        candidates=tuple(
            _Candidate(rank=index, action=action) for index, action in enumerate(actions, start=1)
        ),
        legal_action_count=len(actions) if legal_action_count is None else legal_action_count,
    )


def _evaluate(
    record: _Record | StepRecordV1,
    proposal: _Proposal | None,
    adapter: _Adapter | None = None,
    *,
    family_partition_evidence: FamilyPartitionEvidenceV1 | None = FAMILY_EVIDENCE,
) -> StepMetricOutcomeV1:
    return evaluate_step(
        record,
        proposal,
        adapter=adapter or _Adapter(),
        registered_rule_ids=REGISTERED_RULES,
        registered_rule_directions=REGISTERED_RULE_DIRECTIONS,
        rule_registry_digest=RULE_DIGEST,
        top_ks=TOP_KS,
        family_partition_evidence=family_partition_evidence,
    )


def test_concrete_issue_61_step_record_satisfies_the_metrics_protocol() -> None:
    record = StepRecordV1(
        trace_id="trace-concrete",
        trace_digest=_digest("trace"),
        pair_id="pair-concrete",
        source_id="source-concrete",
        step_index=1,
        trace_length=3,
        current_state=CanonicalJson.from_value({"expr": "x+0"}),
        goal_state=CanonicalJson.from_value({"expr": "x"}),
        action=CanonicalJson.from_value({"rule": "add_zero"}),
        next_state=CanonicalJson.from_value({"expr": "x"}),
        current_signature=CURRENT_SIGNATURE,
        goal_signature=GOAL_SIGNATURE,
        action_source_signature=CURRENT_SIGNATURE,
        action_successor_signature=DEMONSTRATED_SUCCESSOR,
        next_signature=DEMONSTRATED_SUCCESSOR,
        remaining_witness_steps=2,
        source_group="group-concrete",
        lineage_group_ids=("group-concrete",),
        authoritative_split=SplitV1.VALIDATION,
        evaluation_views=("iid",),
        source_family="algebra",
        current_family="algebra",
        goal_family="algebra",
        domain_mode="safe_real",
        rewrite_mode="safe",
        rule_set_digest=RULE_DIGEST,
        action_digest=_digest("concrete-action"),
        rule_id="add_zero",
        direction=ActionDirectionV1.FORWARD,
        occurrence_path=(0,),
        ordered_arguments=(CanonicalJson.from_value({"binding": "x"}),),
        assumptions=(),
        verification_evidence_digest=_digest("verification-evidence"),
        verifier_digest=_digest("verifier"),
    )
    assert isinstance(record, StepRecordProtocol)
    normalized_action = NormalizedActionV1(
        action=CanonicalJson.from_value({"rule": "add_zero"}),
        action_digest=record.action_digest,
        rule_id=record.rule_id,
        direction=record.direction,
        occurrence_path=record.occurrence_path,
        ordered_arguments=record.ordered_arguments,
        source_signature=record.current_signature,
        successor_signature=record.next_signature,
        assumptions=record.assumptions,
        domain_mode=record.domain_mode,
        stored_verification_status=VerificationStatusV1.ACCEPTED,
        stored_verifier_digest=record.verifier_digest,
        stored_verification_evidence_digest=record.verification_evidence_digest,
    )
    proposal = ProposalV1(
        current_signature=record.current_signature,
        goal_signature=record.goal_signature,
        candidates=(
            ProposalCandidateV1(
                rank=1,
                action=normalized_action,
                logit=1.0,
                probability=1.0,
            ),
        ),
        legal_action_count=1,
        requested_top_k=5,
        legal_mask_digest=LEGAL_MASK_DIGEST,
        rule_registry_digest=RULE_DIGEST,
        model_identity=ModelIdentityV1(
            model_family="gnn",
            model_id="fixture-policy",
            checkpoint_digest=_digest("checkpoint"),
            config_digest=_digest("config"),
        ),
        probability_temperature=1.0,
        status=ProposalStatus.SUCCESS,
        detail="concrete compatibility fixture",
    )
    row = evaluate_step(
        record,
        proposal,
        adapter=_ConcreteAdapter(),
        registered_rule_ids=REGISTERED_RULES,
        registered_rule_directions=REGISTERED_RULE_DIRECTIONS,
        rule_registry_digest=RULE_DIGEST,
        top_ks=TOP_KS,
    )
    assert row.status is ExampleMetricStatus.EVALUATED
    assert row.demonstration_action is not None
    assert row.demonstration_action.direction == "forward"
    assert row.candidates[0].exact_demonstration_action


def test_canonical_arguments_preserve_order_and_canonical_json_wrappers() -> None:
    arguments = (
        _CanonicalJson('{"a":1,"b":[2,3]}'),
        "x",
        [1, 2],
    )
    assert canonical_ordered_arguments(arguments) == (
        '{"a":1,"b":[2,3]}',
        '"x"',
        "[1,2]",
    )
    assert canonical_ordered_arguments((_CanonicalJson('{"b":2, "a":1}'),)) == ('{"a":1,"b":2}',)
    assert canonical_ordered_arguments((_CanonicalJson('"π"'), "π")) == (
        '"\\u03c0"',
        '"\\u03c0"',
    )
    with pytest.raises(ValueError, match="finite JSON"):
        canonical_ordered_arguments((_BrokenCanonicalJson('{"a":1}'),))
    with pytest.raises(ValueError, match="canonical JSON"):
        canonical_ordered_arguments((float("nan"),))


def test_imitation_exact_structure_and_verifier_safety_do_not_collapse() -> None:
    record = _Record()
    different_valid_successor = _action(
        "commute_add",
        path=(),
        digest=_digest("different-successor-action"),
        successor=_digest("equivalent-but-not-demonstrated"),
    )
    same_successor_different_action = _action(
        "factor",
        digest=_digest("same-successor-action"),
    )
    exact_demonstration = _action(
        "add_zero",
        digest=_digest("exact-demonstration"),
    )

    row = _evaluate(
        record,
        _proposal(
            different_valid_successor,
            same_successor_different_action,
            exact_demonstration,
        ),
    )

    assert row.status is ExampleMetricStatus.EVALUATED
    first, second, third = row.candidates
    assert (
        first.exact_demonstration_action,
        first.exact_successor_structure,
        first.verifier_confirmed_valid,
    ) == (False, False, True)
    assert (
        second.exact_demonstration_action,
        second.exact_successor_structure,
        second.verifier_confirmed_valid,
    ) == (False, True, True)
    assert (
        third.exact_demonstration_action,
        third.exact_successor_structure,
        third.verifier_confirmed_valid,
    ) == (True, True, True)
    assert row.at_k(1).verifier_valid_success
    assert not row.at_k(1).exact_successor_structure_match
    assert not row.at_k(2).demonstration_action_match
    assert row.at_k(2).exact_successor_structure_match
    assert row.at_k(3).demonstration_action_match


def test_two_distinct_actions_may_share_the_exact_successor() -> None:
    row = _evaluate(
        _Record(),
        _proposal(
            _action("commute_add", path=(), digest=_digest("action-a")),
            _action("factor", path=(1,), digest=_digest("action-b")),
        ),
    )
    assert [candidate.exact_successor_structure for candidate in row.candidates] == [
        True,
        True,
    ]
    assert not any(candidate.exact_demonstration_action for candidate in row.candidates)


def test_direction_path_and_arguments_are_exact_action_identity() -> None:
    wrong_direction = _action(
        "add_zero",
        direction="backward",
        digest=_digest("wrong-direction"),
    )
    wrong_path = _action("add_zero", path=(1,), digest=_digest("wrong-path"))
    wrong_arguments = _action(
        "add_zero",
        arguments=(_CanonicalJson('{"binding":"y"}'),),
        digest=_digest("wrong-arguments"),
    )
    exact = _action("add_zero", digest=_digest("exact"))
    row = _evaluate(
        _Record(),
        _proposal(wrong_direction, wrong_path, wrong_arguments, exact),
    )
    assert [candidate.exact_demonstration_action for candidate in row.candidates] == [
        False,
        False,
        False,
        True,
    ]


@pytest.mark.parametrize(
    ("legality", "expected_status"),
    [
        (LegalityStatus.INVALID_ACTION, CandidateMetricStatus.INVALID_ACTION),
        (LegalityStatus.INVALID_SITE, CandidateMetricStatus.INVALID_SITE),
        (LegalityStatus.INVALID_ARGUMENTS, CandidateMetricStatus.INVALID_ARGUMENTS),
        (LegalityStatus.UNSUPPORTED, CandidateMetricStatus.UNSUPPORTED),
    ],
)
def test_illegal_candidates_are_typed_and_never_replayed_or_verified(
    legality: LegalityStatus,
    expected_status: CandidateMetricStatus,
) -> None:
    adapter = _Adapter()
    action = _action("add_zero", legality=legality)
    row = _evaluate(_Record(), _proposal(action), adapter)
    candidate = row.candidates[0]
    assert candidate.status is expected_status
    assert not candidate.verifier_confirmed_valid
    assert candidate.replay_status is None
    assert candidate.verifier_status is None
    assert adapter.replayed == []
    assert adapter.verified == []


@pytest.mark.parametrize(
    ("verification", "status", "verifier_status"),
    [
        (
            "raise_timeout",
            CandidateMetricStatus.VERIFIER_TIMEOUT,
            VerificationStatus.TIMEOUT,
        ),
        (
            "raise_error",
            CandidateMetricStatus.VERIFIER_ERROR,
            VerificationStatus.ERROR,
        ),
        (
            VerificationStatus.REJECTED,
            CandidateMetricStatus.VERIFIER_REJECTED,
            VerificationStatus.REJECTED,
        ),
    ],
)
def test_verifier_timeout_error_and_rejection_are_retained(
    verification: VerificationStatus | str,
    status: CandidateMetricStatus,
    verifier_status: VerificationStatus,
) -> None:
    row = _evaluate(
        _Record(),
        _proposal(_action("add_zero", verification=verification)),
    )
    candidate = row.candidates[0]
    assert candidate.status is status
    assert candidate.verifier_status is verifier_status
    assert candidate.exact_demonstration_action
    assert candidate.exact_successor_structure
    assert not candidate.verifier_confirmed_valid
    aggregate = aggregate_step_metrics((row,)).as_dict()
    assert aggregate["candidate_status_counts"][status.value] == 1
    assert aggregate["candidate_status_rates"][status.value] == 1


def test_replay_failure_is_retained_and_verifier_is_not_called() -> None:
    adapter = _Adapter()
    row = _evaluate(
        _Record(),
        _proposal(_action("add_zero", replay_status=ReplayStatus.INVALID_SITE)),
        adapter,
    )
    candidate = row.candidates[0]
    assert candidate.status is CandidateMetricStatus.INVALID_SITE
    assert candidate.replay_status is ReplayStatus.INVALID_SITE
    assert candidate.verifier_status is None
    assert adapter.verified == []


def test_replay_signature_must_match_the_replayed_state() -> None:
    class _MismatchedSignatureAdapter(_Adapter):
        def structural_signature(self, successor_state: object) -> str:
            del successor_state
            return _digest("different-replayed-state")

    adapter = _MismatchedSignatureAdapter()
    row = _evaluate(
        _Record(),
        _proposal(_action("add_zero", successor=DEMONSTRATED_SUCCESSOR)),
        adapter,
    )
    candidate = row.candidates[0]
    assert candidate.status is CandidateMetricStatus.REPLAY_ERROR
    assert candidate.replay_status is ReplayStatus.ERROR
    assert candidate.successor_signature is None
    assert not candidate.exact_successor_structure
    assert not candidate.verifier_confirmed_valid
    assert adapter.verified == []
    assert candidate.replay_detail is not None
    assert "independently derived" in candidate.replay_detail


def test_empty_legal_inventory_and_missing_proposal_have_distinct_statuses() -> None:
    no_legal = _Proposal(
        current_signature=CURRENT_SIGNATURE,
        goal_signature=GOAL_SIGNATURE,
        candidates=(),
        legal_action_count=0,
        status="no_legal_action",
    )
    no_legal_row = _evaluate(_Record(), no_legal)
    missing_row = _evaluate(replace(_Record(), record_id=_digest("record-2")), None)
    assert no_legal_row.status is ExampleMetricStatus.NO_LEGAL_ACTION
    assert no_legal_row.legal_action_count == 0
    assert no_legal_row.legal_mask_digest == LEGAL_MASK_DIGEST
    assert missing_row.status is ExampleMetricStatus.NO_PROPOSAL
    assert missing_row.legal_action_count is None
    assert missing_row.legal_mask_digest is None


@pytest.mark.parametrize("status", ["no_legal_action", "success"])
def test_internally_inconsistent_empty_proposal_is_invalid(status: str) -> None:
    row = _evaluate(
        _Record(),
        _Proposal(
            current_signature=CURRENT_SIGNATURE,
            goal_signature=GOAL_SIGNATURE,
            candidates=(),
            legal_action_count=1,
            status=status,
        ),
    )
    assert row.status is ExampleMetricStatus.INVALID_PROPOSAL
    assert row.legal_action_count == 1


def test_unsupported_and_malformed_proposals_are_not_dropped() -> None:
    unsupported = _Proposal(
        current_signature=CURRENT_SIGNATURE,
        goal_signature=GOAL_SIGNATURE,
        candidates=(),
        legal_action_count=2,
        status="unsupported",
    )
    unsupported_row = _evaluate(_Record(), unsupported)
    wrong_identity_row = _evaluate(
        replace(_Record(), record_id=_digest("record-2")),
        replace(
            unsupported,
            status="success",
            current_signature=_digest("wrong"),
        ),
    )
    wrong_cutoff_row = _evaluate(
        replace(_Record(), record_id=_digest("record-3")),
        replace(unsupported, status="success", requested_top_k=3),
    )
    malformed_mask_row = _evaluate(
        replace(_Record(), record_id=_digest("record-4")),
        replace(unsupported, status="success", legal_mask_digest="not-a-digest"),
    )
    assert unsupported_row.status is ExampleMetricStatus.UNSUPPORTED
    assert wrong_identity_row.status is ExampleMetricStatus.PARSE_SCHEMA_ERROR
    assert wrong_cutoff_row.status is ExampleMetricStatus.PARSE_SCHEMA_ERROR
    assert malformed_mask_row.status is ExampleMetricStatus.PARSE_SCHEMA_ERROR
    assert malformed_mask_row.legal_mask_digest is None


def test_malformed_candidate_is_retained_as_a_ranked_parse_failure() -> None:
    row = _evaluate(
        _Record(),
        _Proposal(
            current_signature=CURRENT_SIGNATURE,
            goal_signature=GOAL_SIGNATURE,
            candidates=(_Candidate(rank=1, action=object()),),
            legal_action_count=1,
        ),
    )
    candidate = row.candidates[0]
    assert candidate.status is CandidateMetricStatus.PARSE_SCHEMA_ERROR
    assert candidate.parse_error is not None


def test_unregistered_candidate_is_unsupported_without_adapter_calls() -> None:
    adapter = _Adapter()
    row = _evaluate(
        _Record(),
        _proposal(_action("not_registered")),
        adapter,
    )
    candidate = row.candidates[0]
    assert candidate.status is CandidateMetricStatus.UNSUPPORTED
    assert candidate.legality_status is LegalityStatus.UNSUPPORTED
    assert adapter.replayed == []
    assert adapter.verified == []
    coverage = aggregate_step_metrics((row,)).rule_coverage
    assert coverage.unregistered_proposed_rule_ids == ("not_registered",)
    assert coverage.proposed_rule_ids == ()
    assert coverage.as_dict()["proposed_rule_rate"] == 0


def test_unregistered_direction_is_explicitly_unsupported_and_zero_covered() -> None:
    row = _evaluate(
        _Record(),
        _proposal(_action("add_zero", direction="backward")),
    )
    candidate = row.candidates[0]
    assert candidate.status is CandidateMetricStatus.UNSUPPORTED
    coverage = aggregate_step_metrics((row,)).rule_coverage
    assert coverage.unregistered_proposed_rule_directions == (("add_zero", "backward"),)
    assert ("commute_add", "backward") in coverage.zero_proposal_rule_directions


def test_registered_but_illegal_candidate_does_not_count_as_rule_coverage() -> None:
    row = _evaluate(
        replace(
            _Record(),
            rule_id="factor",
            action_digest=_digest("factor-demonstration"),
        ),
        _proposal(_action("factor", legality=LegalityStatus.INVALID_SITE)),
    )
    coverage = aggregate_step_metrics((row,)).rule_coverage
    assert "factor" in coverage.demonstrated_rule_ids
    assert "factor" not in coverage.proposed_rule_ids
    assert "factor" in coverage.zero_proposal_rule_ids
    assert ("factor", "forward") in coverage.zero_proposal_rule_directions


def test_unregistered_demonstration_direction_rejects_the_metric_row() -> None:
    row = _evaluate(
        replace(_Record(), direction="backward"),
        _proposal(_action("add_zero", direction="backward")),
    )
    assert row.status is ExampleMetricStatus.PARSE_SCHEMA_ERROR
    assert row.candidates == ()
    assert "demonstrated rule direction" in row.detail
    coverage = aggregate_step_metrics((row,)).rule_coverage
    assert coverage.demonstrated_rule_ids == ()
    assert coverage.unregistered_demonstrated_rule_ids == ()
    assert coverage.unregistered_demonstrated_rule_directions == (("add_zero", "backward"),)
    assert "add_zero" in coverage.zero_demonstration_rule_ids
    assert coverage.as_dict()["demonstrated_rule_rate"] == 0


def test_unregistered_demonstration_rule_is_separate_from_registry_coverage() -> None:
    row = _evaluate(
        replace(
            _Record(),
            rule_id="not_registered",
            action_digest=_digest("not-registered-demonstration"),
        ),
        None,
    )
    assert row.status is ExampleMetricStatus.PARSE_SCHEMA_ERROR
    coverage = aggregate_step_metrics((row,)).rule_coverage
    assert coverage.demonstrated_rule_ids == ()
    assert coverage.unregistered_demonstrated_rule_ids == ("not_registered",)
    assert coverage.unregistered_demonstrated_rule_directions == (("not_registered", "forward"),)
    assert coverage.as_dict()["demonstrated_rule_rate"] == 0


def test_successful_truncated_ranking_is_invalid_not_silently_scored() -> None:
    row = _evaluate(
        _Record(),
        _proposal(_action("add_zero"), legal_action_count=5),
    )
    assert row.status is ExampleMetricStatus.INVALID_PROPOSAL
    assert row.candidates == ()
    assert "truncated" in row.detail


def test_duplicate_action_is_a_parse_failure_not_a_second_success() -> None:
    action = _action("add_zero")
    row = _evaluate(
        _Record(),
        _Proposal(
            current_signature=CURRENT_SIGNATURE,
            goal_signature=GOAL_SIGNATURE,
            candidates=(
                _Candidate(rank=1, action=action),
                _Candidate(
                    rank=2,
                    action=replace(action, action_digest=_digest("other-evidence")),
                ),
            ),
            legal_action_count=2,
        ),
    )
    assert row.candidates[0].status is CandidateMetricStatus.VERIFIED_VALID
    assert row.candidates[1].status is CandidateMetricStatus.PARSE_SCHEMA_ERROR


def test_strict_per_example_json_round_trip_preserves_aggregate_evidence() -> None:
    row = _evaluate(
        _Record(goal_family="held_out_family"),
        _proposal(
            _action("commute_add", path=(), successor=_digest("other")),
            _action("add_zero"),
        ),
    )
    payload = json.loads(json.dumps(row.as_dict(), sort_keys=True))
    restored = StepMetricOutcomeV1.from_dict(payload)
    assert restored == row
    assert restored.legal_mask_digest == LEGAL_MASK_DIGEST
    assert restored.at_k(1) == row.at_k(1)

    payload["proposal_candidate_count"] = 99
    with pytest.raises(ValueError, match="proposal_candidate_count"):
        StepMetricOutcomeV1.from_dict(payload)

    inconsistent = row.as_dict()
    inconsistent["candidates"][0]["status"] = "invalid_site"
    with pytest.raises(ValueError, match="disagrees"):
        StepMetricOutcomeV1.from_dict(inconsistent)

    fabricated_action = row.as_dict()
    fabricated_action["candidates"][0]["exact_demonstration_action"] = True
    with pytest.raises(ValueError, match="exact-action"):
        StepMetricOutcomeV1.from_dict(fabricated_action)

    fabricated_successor = row.as_dict()
    fabricated_successor["candidates"][0]["exact_successor_structure"] = True
    with pytest.raises(ValueError, match="exact-successor"):
        StepMetricOutcomeV1.from_dict(fabricated_successor)

    truncated = row.as_dict()
    truncated["candidates"].pop()
    truncated["proposal_candidate_count"] = 1
    with pytest.raises(ValueError, match="complete ranking"):
        StepMetricOutcomeV1.from_dict(truncated)

    duplicate = row.as_dict()
    duplicate["candidates"][1]["action"] = duplicate["candidates"][0]["action"]
    duplicate["candidates"][1]["exact_demonstration_action"] = False
    with pytest.raises(ValueError, match="duplicate concrete action"):
        StepMetricOutcomeV1.from_dict(duplicate)


def test_family_generalization_is_derived_from_manifest_bound_inventory() -> None:
    seen = _evaluate(_Record(), _proposal(_action("add_zero")))
    held_out = _evaluate(
        _Record(goal_family="unseen_family"),
        _proposal(_action("add_zero")),
    )
    unknown = _evaluate(
        _Record(),
        _proposal(_action("add_zero")),
        family_partition_evidence=None,
    )

    assert seen.family_generalization is FamilyGeneralization.SEEN
    assert seen.unseen_family_roles == ()
    assert seen.family_evidence_manifest_digest == FAMILY_EVIDENCE.step_manifest_digest
    assert held_out.family_generalization is FamilyGeneralization.HELD_OUT
    assert held_out.unseen_family_roles == ("goal",)
    assert held_out.training_family_inventory_digest == FAMILY_EVIDENCE.inventory_digest
    assert unknown.family_generalization is FamilyGeneralization.UNKNOWN
    assert unknown.family_evidence_manifest_digest is None
    assert unknown.training_family_inventory_digest is None
    assert unknown.unseen_family_roles == ()


def test_legal_mask_digest_is_retained_for_cross_method_audits() -> None:
    record = _Record()
    action = _action("add_zero")
    first = _evaluate(record, _proposal(action))
    second_mask = _digest("different-legal-mask")
    second = _evaluate(
        record,
        replace(_proposal(action), legal_mask_digest=second_mask),
    )
    assert first.legal_mask_digest == LEGAL_MASK_DIGEST
    assert second.legal_mask_digest == second_mask
    assert first.as_dict()["legal_mask_digest"] != second.as_dict()["legal_mask_digest"]


def test_aggregate_is_reconstructible_and_includes_zero_rule_coverage() -> None:
    accepted = _evaluate(
        _Record(record_id=_digest("accepted"), evaluation_views=("iid",)),
        _proposal(_action("add_zero")),
    )
    invalid = _evaluate(
        _Record(
            record_id=_digest("invalid"),
            trace_id="trace-2",
            source_group="group-2",
            lineage_group_ids=("group-2",),
            rule_id="factor",
            action_digest=_digest("factor-demonstration"),
            current_family="power",
            goal_family="algebra",
            evaluation_views=("held_out_family", "ood"),
            remaining_witness_steps=1,
        ),
        _proposal(
            _action(
                "factor",
                legality=LegalityStatus.INVALID_SITE,
                digest=_digest("invalid-site"),
            )
        ),
    )
    no_legal = _evaluate(
        _Record(
            record_id=_digest("no-legal"),
            trace_id="trace-3",
            source_group="group-3",
            lineage_group_ids=("group-3",),
            rule_id="commute_add",
            action_digest=_digest("commute-demonstration"),
        ),
        _Proposal(
            current_signature=CURRENT_SIGNATURE,
            goal_signature=GOAL_SIGNATURE,
            candidates=(),
            legal_action_count=0,
            status="no_legal_action",
        ),
    )
    restored_rows = tuple(
        StepMetricOutcomeV1.from_dict(json.loads(json.dumps(row.as_dict(), sort_keys=True)))
        for row in (accepted, invalid, no_legal)
    )
    aggregate = aggregate_step_metrics(restored_rows)
    payload = aggregate.as_dict()

    assert aggregate.total_examples == 3
    assert dict(aggregate.example_status_counts) == {
        "evaluated": 2,
        "no_legal_action": 1,
        "no_proposal": 0,
        "unsupported": 0,
        "invalid_proposal": 0,
        "parse_schema_error": 0,
    }
    assert dict(aggregate.candidate_status_counts)["verified_valid"] == 1
    assert dict(aggregate.candidate_status_counts)["invalid_site"] == 1
    assert aggregate.rule_coverage.zero_demonstration_rule_ids == ("unused_rule",)
    assert aggregate.rule_coverage.zero_proposal_rule_ids == (
        "commute_add",
        "factor",
        "unused_rule",
    )
    assert ("commute_add", "backward") in (
        aggregate.rule_coverage.zero_demonstration_rule_directions
    )
    assert ("commute_add", "backward") in aggregate.rule_coverage.zero_proposal_rule_directions
    unused = next(rule for rule in aggregate.per_rule if rule.rule_id == "unused_rule")
    assert unused.example_count == 0
    assert all(metric.example_denominator == 0 for metric in unused.top_k)
    top_1 = aggregate.top_k[0]
    assert top_1.example_denominator == 3
    assert top_1.attempted_example_denominator == 2
    assert top_1.candidate_attempt_denominator == 2
    assert top_1.verifier_attempt_denominator == 1
    assert top_1.verifier_resolved_candidate_denominator == 1
    assert top_1.verifier_valid_candidate_count == 1
    assert payload["top_k"][0]["verifier_valid_success_rate_all"] == pytest.approx(1 / 3)
    assert payload["example_status_rates"]["no_legal_action"] == pytest.approx(1 / 3)
    assert payload["candidate_status_rates"]["invalid_site"] == pytest.approx(1 / 2)
    assert {(row["dimension"], row["value"]) for row in payload["breakdowns"]} >= {
        ("family_generalization", "held_out"),
        ("family_generalization", "seen"),
        ("authoritative_split", "test_iid"),
        ("unseen_family_role", "current"),
        ("remaining_witness_steps", "1"),
        ("remaining_witness_steps", "2"),
        ("trace_length", "3"),
        ("evaluation_view", "held_out_family"),
    }


def test_aggregate_rejects_duplicates_and_incompatible_registry_contracts() -> None:
    row = _evaluate(_Record(), _proposal(_action("add_zero")))
    with pytest.raises(ValueError, match="duplicate record"):
        aggregate_step_metrics((row, row))
    incompatible = replace(row, rule_registry_digest=_digest("different"))
    with pytest.raises(ValueError, match="incompatible registry"):
        aggregate_step_metrics((row, replace(incompatible, record_id=_digest("record-2"))))


def test_no_verifier_confirmation_never_counts_as_safety_success() -> None:
    row = _evaluate(
        _Record(),
        _proposal(
            _action(
                "add_zero",
                verification=VerificationStatus.REJECTED,
            )
        ),
    )
    metric = row.at_k(1)
    assert metric.demonstration_action_match
    assert metric.exact_successor_structure_match
    assert not metric.verifier_valid_success
    aggregate = aggregate_step_metrics((row,)).as_dict()["top_k"][0]
    assert aggregate["verifier_valid_success_rate_all"] == 0
    assert aggregate["verifier_acceptance_rate_resolved"] == 0
