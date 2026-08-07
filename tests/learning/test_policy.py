"""Issue 7-1 tests use only tiny injected action and encoder fixtures."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import JsonValue

from geml.egraph.rules_domain import domain_rules
from geml.egraph.rules_safe import SAFE_RULES
from geml.learning.policy.head import (
    ActionInventoryStatus,
    ActionInventoryV1,
    ActionScoringViewV1,
    GoalConditionedPolicyHead,
    InvalidPolicyStateError,
    LegalActionEnumerationV1,
    MissingMLDependencyError,
    ModelIdentityV1,
    ProposalStatus,
    ProposalV1,
    ResolvedActionFeaturesV1,
    RuleKeyV1,
    RuleVocabularyV1,
    UnsupportedPolicyStateError,
    action_scoring_view,
    build_action_inventory,
    compute_legal_mask_digest,
    torch,
)
from geml.learning.policy.proposer_transformer import (
    PrefixTransformerProposer,
    RoleSeparatedPrefixV1,
    RoleTokenIdsV1,
)

_CURRENT_SIGNATURE = "1" * 64
_GOAL_A_SIGNATURE = "2" * 64
_GOAL_B_SIGNATURE = "3" * 64
_REGISTRY_DIGEST = hashlib.sha256(b"issue-7-1-fixture-registry").hexdigest()
_CONFIG_DIGEST = hashlib.sha256(b"issue-7-1-fixture-config").hexdigest()
_CHECKPOINT_DIGEST = hashlib.sha256(b"issue-7-1-fixture-checkpoint").hexdigest()


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FixtureAction:
    """The narrow Phase-A adapter for the upstream normalized action."""

    action: JsonValue
    action_digest: str
    rule_id: str
    direction: str
    occurrence_path: tuple[int, ...]
    ordered_arguments: tuple[JsonValue, ...]
    source_signature: str
    successor_signature: str

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "action_digest": self.action_digest,
            "direction": self.direction,
            "occurrence_path": list(self.occurrence_path),
            "ordered_arguments": list(self.ordered_arguments),
            "rule_id": self.rule_id,
            "source_signature": self.source_signature,
            "successor_signature": self.successor_signature,
        }

    @classmethod
    def create(
        cls,
        *,
        rule_id: str = "RULE-A",
        direction: str = "forward",
        occurrence_path: tuple[int, ...] = (0,),
        ordered_arguments: tuple[JsonValue, ...] = (),
        source_signature: str = _CURRENT_SIGNATURE,
        successor_signature: str | None = None,
        salt: str = "",
    ) -> FixtureAction:
        successor = successor_signature or _digest(
            {
                "arguments": ordered_arguments,
                "direction": direction,
                "path": occurrence_path,
                "rule": rule_id,
                "salt": salt,
            }
        )
        canonical_action: JsonValue = {
            "bindings": list(ordered_arguments),
            "direction": direction,
            "occurrence_path": list(occurrence_path),
            "rule_id": rule_id,
            "source_signature": source_signature,
            "successor_signature": successor,
        }
        return cls(
            action=canonical_action,
            action_digest=_digest(
                {
                    "action": canonical_action,
                    "version": "issue-7-1-fixture-action-v1",
                }
            ),
            rule_id=rule_id,
            direction=direction,
            occurrence_path=occurrence_path,
            ordered_arguments=ordered_arguments,
            source_signature=source_signature,
            successor_signature=successor,
        )

    @classmethod
    def from_proposal_payload(cls, payload: dict[str, object]) -> FixtureAction:
        expected = {
            "action",
            "action_digest",
            "direction",
            "occurrence_path",
            "ordered_arguments",
            "rule_id",
            "source_signature",
            "successor_signature",
        }
        if set(payload) != expected:
            raise ValueError("fixture action payload fields differ")
        path = payload["occurrence_path"]
        arguments = payload["ordered_arguments"]
        if not isinstance(path, list) or not isinstance(arguments, list):
            raise TypeError("fixture action path/arguments must be JSON arrays")
        return cls(
            action=payload["action"],  # type: ignore[arg-type]
            action_digest=payload["action_digest"],  # type: ignore[arg-type]
            rule_id=payload["rule_id"],  # type: ignore[arg-type]
            direction=payload["direction"],  # type: ignore[arg-type]
            occurrence_path=tuple(path),
            ordered_arguments=tuple(arguments),
            source_signature=payload["source_signature"],  # type: ignore[arg-type]
            successor_signature=payload["successor_signature"],  # type: ignore[arg-type]
        )


def _vocabulary() -> RuleVocabularyV1:
    return RuleVocabularyV1.from_registry(
        (
            ("RULE-A", "forward"),
            ("RULE-A", "backward"),
            ("RULE-B", "forward"),
        ),
        registry_digest=_REGISTRY_DIGEST,
    )


def _inventory(
    actions: tuple[FixtureAction, ...],
    mask: tuple[bool, ...],
    *,
    goal_signature: str = _GOAL_A_SIGNATURE,
    status: ActionInventoryStatus = ActionInventoryStatus.READY,
    detail: str = "registry enumeration complete",
) -> ActionInventoryV1:
    return ActionInventoryV1(
        current_signature=_CURRENT_SIGNATURE,
        goal_signature=goal_signature,
        vocabulary=_vocabulary(),
        actions=actions,
        legal_mask=mask,
        status=status,
        detail=detail,
    )


def _identity(family: str) -> ModelIdentityV1:
    return ModelIdentityV1(
        model_family=family,
        model_id=f"fixture-{family}",
        checkpoint_digest=_CHECKPOINT_DIGEST,
        config_digest=_CONFIG_DIGEST,
    )


def test_rule_vocabulary_is_registry_derived_and_canonical() -> None:
    vocabulary = _vocabulary()
    assert tuple((entry.rule_id, entry.direction) for entry in vocabulary.entries) == (
        ("RULE-A", "backward"),
        ("RULE-A", "forward"),
        ("RULE-B", "forward"),
    )
    with pytest.raises(ValueError, match="canonical sorted"):
        RuleVocabularyV1(
            registry_digest=_REGISTRY_DIGEST,
            entries=tuple(reversed(vocabulary.entries)),
        )


def test_goal4_registry_feeds_vocabulary_without_a_second_rule_table() -> None:
    rules = SAFE_RULES.merged_with(domain_rules(include_optional=True))
    vocabulary = RuleVocabularyV1.from_registry(
        tuple((rule.rule_id, rule.direction.value) for rule in rules),
        registry_digest=_REGISTRY_DIGEST,
    )
    assert set(vocabulary.entries) == {
        RuleKeyV1(rule.rule_id, rule.direction.value) for rule in rules
    }
    assert vocabulary.rule_count == len({rule.rule_id for rule in rules})


def test_inventory_binds_mask_order_and_rejects_duplicate_actions() -> None:
    first = FixtureAction.create(occurrence_path=(0,))
    second = FixtureAction.create(occurrence_path=(1,))
    inventory = _inventory((first, second), (True, False))
    assert inventory.legal_action_count == 1
    assert len(inventory.legal_mask_digest) == 64
    assert inventory.legal_mask_digest == compute_legal_mask_digest(
        action_digests=(first.action_digest, second.action_digest),
        legal_mask=(True, False),
        current_signature=inventory.current_signature,
        goal_signature=inventory.goal_signature,
        registry_digest=inventory.vocabulary.registry_digest,
        status=ActionInventoryStatus.READY,
    )
    reversed_mask = _inventory((first, second), (False, True))
    assert reversed_mask.legal_mask_digest != inventory.legal_mask_digest
    reordered = compute_legal_mask_digest(
        action_digests=(second.action_digest, first.action_digest),
        legal_mask=(False, True),
        current_signature=inventory.current_signature,
        goal_signature=inventory.goal_signature,
        registry_digest=inventory.vocabulary.registry_digest,
        status=ActionInventoryStatus.READY,
    )
    assert reordered != inventory.legal_mask_digest
    with pytest.raises(ValueError, match="duplicate action"):
        _inventory((first, first), (True, True))


def test_action_snapshot_rejects_mirrored_payload_drift() -> None:
    base = FixtureAction.create(ordered_arguments=(1,), salt="mirror")

    class InconsistentAction:
        action = base.action
        action_digest = base.action_digest
        rule_id = base.rule_id
        direction = base.direction
        occurrence_path = base.occurrence_path
        ordered_arguments = base.ordered_arguments
        source_signature = base.source_signature
        successor_signature = base.successor_signature

        def as_dict(self) -> dict[str, object]:
            payload = base.as_dict()
            payload["ordered_arguments"] = [999]
            return payload

    with pytest.raises(ValueError, match="serialized action fields disagree"):
        _inventory((InconsistentAction(),), (True,))


def test_current_only_legal_provider_receives_domain_context_and_builds_shared_inventory() -> None:
    action = FixtureAction.create(salt="provider")

    class Provider:
        def __init__(self) -> None:
            self.call: dict[str, object] | None = None

        def enumerate_actions(
            self,
            *,
            current_state: object,
            current_signature: str,
            assumptions: tuple[str, ...],
            domain_mode: str,
            vocabulary: RuleVocabularyV1,
        ) -> LegalActionEnumerationV1:
            self.call = {
                "assumptions": assumptions,
                "current_signature": current_signature,
                "current_state": current_state,
                "domain_mode": domain_mode,
                "vocabulary": vocabulary,
            }
            return LegalActionEnumerationV1(actions=(action,), legal_mask=(True,))

    provider = Provider()
    vocabulary = _vocabulary()
    inventory = build_action_inventory(
        current_state={"expr": "x"},
        current_signature=_CURRENT_SIGNATURE,
        goal_signature=_GOAL_A_SIGNATURE,
        assumptions=("real",),
        domain_mode="real",
        vocabulary=vocabulary,
        provider=provider,
    )

    assert inventory.goal_signature == _GOAL_A_SIGNATURE
    assert inventory.actions == (action,)
    assert provider.call == {
        "assumptions": ("real",),
        "current_signature": _CURRENT_SIGNATURE,
        "current_state": {"expr": "x"},
        "domain_mode": "real",
        "vocabulary": vocabulary,
    }


def test_inventory_rejects_source_signature_drift_and_bad_mask() -> None:
    wrong_source = FixtureAction.create(source_signature="9" * 64)
    with pytest.raises(InvalidPolicyStateError, match="source signature"):
        _inventory((wrong_source,), (True,))
    with pytest.raises(TypeError, match="legal_mask"):
        _inventory((FixtureAction.create(),), ())
    with pytest.raises(ValueError, match="cannot mark an action legal"):
        _inventory(
            (FixtureAction.create(),),
            (True,),
            status=ActionInventoryStatus.UNSUPPORTED,
            detail="unsupported source operator",
        )


def test_scoring_view_excludes_successor_digest_and_verifier_data() -> None:
    action = FixtureAction.create(ordered_arguments=(0, 1))
    view = action_scoring_view(action)
    assert view == ActionScoringViewV1(
        rule_id="RULE-A",
        direction="forward",
        occurrence_path=(0,),
        ordered_arguments=(0, 1),
    )
    assert not hasattr(view, "successor_signature")
    assert not hasattr(view, "action_digest")
    assert not hasattr(view, "verification_status")
    assert not hasattr(view, "remaining_witness_steps")


def test_role_separated_prefix_is_directional_and_padding_is_suffix() -> None:
    roles = RoleTokenIdsV1(current=90, separator=91, goal=92, padding=0)
    pair = RoleSeparatedPrefixV1.build(
        (11, 12),
        (21,),
        role_tokens=roles,
        pad_to_length=8,
    )
    assert pair.token_ids == (90, 11, 12, 91, 92, 21, 0, 0)
    assert pair.attention_mask == (True, True, True, True, True, True, False, False)
    assert pair.current_positions == (1, 2)
    assert pair.goal_positions == (5,)
    swapped = RoleSeparatedPrefixV1.build((21,), (11, 12), role_tokens=roles)
    assert swapped.token_ids != pair.token_ids[: len(swapped.token_ids)]
    with pytest.raises(InvalidPolicyStateError, match="reserved"):
        RoleSeparatedPrefixV1.build((90,), (21,), role_tokens=roles)


def test_core_import_reports_missing_ml_extra_in_clean_subprocess(tmp_path: Path) -> None:
    """The public contracts import without torch and neural construction fails clearly."""

    blocker = tmp_path / "torch.py"
    blocker.write_text("raise ModuleNotFoundError('blocked fixture torch')\n", encoding="utf-8")
    script = (
        f"import sys; sys.path[:0] = [{str(tmp_path)!r}, {str(Path.cwd() / 'src')!r}]\n"
        "from geml.learning.policy.head import MissingMLDependencyError, "
        "GoalConditionedPolicyHead\n"
        "try:\n"
        "    GoalConditionedPolicyHead(encoder=None, action_resolver=None, "
        "vocabulary=None, model_identity=None, hidden_width=2)\n"
        "except MissingMLDependencyError:\n"
        "    print('clean failure')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "clean failure"


pytestmark_ml = pytest.mark.skipif(torch is None, reason="optional ml extra is not installed")


if torch is not None:

    @dataclass(frozen=True, slots=True)
    class FixtureEncoding:
        graph_embedding: object
        node_embeddings: object

    class FixtureGraphEncoder(torch.nn.Module):
        """Shared fixture encoder with one trainable scale."""

        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, graph_input: tuple[object, object]) -> FixtureEncoding:
            graph, nodes = graph_input
            return FixtureEncoding(
                graph_embedding=graph * self.scale,
                node_embeddings=nodes * self.scale,
            )

    class FixtureResolver:
        def __init__(self) -> None:
            self.views: list[ActionScoringViewV1] = []

        def resolve(
            self,
            current_encoding: FixtureEncoding,
            action: ActionScoringViewV1,
        ) -> ResolvedActionFeaturesV1:
            self.views.append(action)
            site_index = 0 if not action.occurrence_path else action.occurrence_path[-1]
            site = current_encoding.node_embeddings[site_index]
            argument_indices: list[int] = []
            for argument in action.ordered_arguments:
                if isinstance(argument, bool) or not isinstance(argument, int):
                    raise InvalidPolicyStateError("fixture arguments must be node indices")
                argument_indices.append(argument)
            arguments = (
                current_encoding.node_embeddings[argument_indices]
                if argument_indices
                else current_encoding.node_embeddings.new_empty(
                    (0, current_encoding.node_embeddings.shape[1])
                )
            )
            return ResolvedActionFeaturesV1(
                site_embedding=site,
                ordered_argument_embeddings=arguments,
            )

    @dataclass(frozen=True, slots=True)
    class FixturePrefixEncoding:
        current_embedding: object
        goal_embedding: object
        current_token_embeddings: object

    class FixturePrefixEncoder(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(128, width)

        def forward(self, pair: RoleSeparatedPrefixV1) -> FixturePrefixEncoding:
            token_ids = torch.tensor(pair.token_ids, dtype=torch.long)
            embeddings = self.embedding(token_ids)
            mask = torch.tensor(pair.attention_mask, dtype=torch.bool)
            assert not embeddings[~mask].requires_grad or not mask[~mask].any()
            current = embeddings[list(pair.current_positions)]
            goal = embeddings[list(pair.goal_positions)]
            return FixturePrefixEncoding(
                current_embedding=current.sum(dim=0),
                goal_embedding=goal.sum(dim=0),
                current_token_embeddings=current,
            )


def _graph_model(
    *,
    width: int = 2,
    probability_temperature: float = 1.0,
) -> tuple[GoalConditionedPolicyHead, object]:
    if torch is None:  # pragma: no cover - guarded by marks on callers.
        raise MissingMLDependencyError
    torch.manual_seed(7101)
    resolver = FixtureResolver()
    model = GoalConditionedPolicyHead(
        encoder=FixtureGraphEncoder(),
        action_resolver=resolver,
        vocabulary=_vocabulary(),
        model_identity=_identity("gnn"),
        hidden_width=width,
        max_arguments=4,
        probability_temperature=probability_temperature,
    )
    return model, resolver


def _prefix_model(
    *,
    width: int = 2,
) -> tuple[PrefixTransformerProposer, object]:
    if torch is None:  # pragma: no cover - guarded by marks on callers.
        raise MissingMLDependencyError
    torch.manual_seed(7102)
    resolver = FixtureResolver()
    model = PrefixTransformerProposer(
        encoder=FixturePrefixEncoder(width),
        action_resolver=resolver,
        vocabulary=_vocabulary(),
        model_identity=_identity("prefix_transformer"),
        role_tokens=RoleTokenIdsV1(current=90, separator=91, goal=92, padding=0),
        hidden_width=width,
        max_arguments=4,
    )
    return model, resolver


def _configure_directional_ranking(model: object) -> None:
    """Make goal sign interact with occurrence-site sign exactly."""

    if torch is None:  # pragma: no cover
        raise MissingMLDependencyError
    scorer = model.scorer
    with torch.no_grad():
        for parameter in scorer.parameters():
            parameter.zero_()
        scorer.query_projection[0].weight[0, 2] = 1.0
        scorer.action_projection[0].weight[0, 0] = 1.0


@pytestmark_ml
def test_goal_change_changes_gnn_ranking_for_same_current_state() -> None:
    model, _ = _graph_model()
    _configure_directional_ranking(model)
    left = FixtureAction.create(occurrence_path=(0,), salt="left")
    right = FixtureAction.create(occurrence_path=(1,), salt="right")
    current = (torch.tensor([0.0, 0.0]), torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    goal_positive = (torch.tensor([1.0, 0.0]), torch.zeros((1, 2)))
    goal_negative = (torch.tensor([-1.0, 0.0]), torch.zeros((1, 2)))
    positive = model.propose(
        current,
        goal_positive,
        _inventory((left, right), (True, True), goal_signature=_GOAL_A_SIGNATURE),
        top_k=2,
    )
    negative = model.propose(
        current,
        goal_negative,
        _inventory((left, right), (True, True), goal_signature=_GOAL_B_SIGNATURE),
        top_k=2,
    )
    assert positive.candidates[0].action.occurrence_path == (0,)
    assert negative.candidates[0].action.occurrence_path == (1,)


@pytestmark_ml
def test_probability_temperature_is_explicit_and_does_not_change_logits() -> None:
    sharp, _ = _graph_model(probability_temperature=0.5)
    smooth, _ = _graph_model(probability_temperature=2.0)
    _configure_directional_ranking(sharp)
    _configure_directional_ranking(smooth)
    actions = (
        FixtureAction.create(occurrence_path=(0,), salt="left"),
        FixtureAction.create(occurrence_path=(1,), salt="right"),
    )
    inventory = _inventory(actions, (True, True))
    current = (torch.zeros(2), torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    goal = (torch.tensor([1.0, 0.0]), torch.zeros((1, 2)))
    sharp_scores = sharp.score_inventory(current, goal, inventory)
    smooth_scores = smooth.score_inventory(current, goal, inventory)
    assert torch.allclose(sharp_scores.logits, smooth_scores.logits)
    assert sharp_scores.probabilities.max() > smooth_scores.probabilities.max()
    proposal = smooth.propose(current, goal, inventory, top_k=1)
    assert proposal.probability_temperature == 2.0
    with pytest.raises(ValueError, match="finite and positive"):
        _graph_model(probability_temperature=0.0)


@pytestmark_ml
def test_goal_change_changes_transformer_ranking_for_same_current_state() -> None:
    model, _ = _prefix_model()
    _configure_directional_ranking(model)
    with torch.no_grad():
        model.encoder.embedding.weight.zero_()
        model.encoder.embedding.weight[11] = torch.tensor([1.0, 0.0])
        model.encoder.embedding.weight[12] = torch.tensor([-1.0, 0.0])
        model.encoder.embedding.weight[21] = torch.tensor([1.0, 0.0])
        model.encoder.embedding.weight[22] = torch.tensor([-1.0, 0.0])
    left = FixtureAction.create(occurrence_path=(0,), salt="left")
    right = FixtureAction.create(occurrence_path=(1,), salt="right")
    positive = model.propose(
        (11, 12),
        (21,),
        _inventory((left, right), (True, True), goal_signature=_GOAL_A_SIGNATURE),
        top_k=2,
    )
    negative = model.propose(
        (11, 12),
        (22,),
        _inventory((left, right), (True, True), goal_signature=_GOAL_B_SIGNATURE),
        top_k=2,
    )
    assert positive.candidates[0].action.occurrence_path == (0,)
    assert negative.candidates[0].action.occurrence_path == (1,)


@pytestmark_ml
@pytest.mark.parametrize("family", ["gnn", "prefix_transformer"])
def test_shared_legal_mask_assigns_exact_zero_mass_and_hides_illegal_action(
    family: str,
) -> None:
    legal = FixtureAction.create(occurrence_path=(0,), salt="legal")
    illegal = FixtureAction.create(occurrence_path=(1,), salt="illegal")
    inventory = _inventory((legal, illegal), (True, False))
    if family == "gnn":
        model, resolver = _graph_model()
        scores = model.score_inventory(
            (torch.ones(2), torch.eye(2)),
            (torch.ones(2), torch.eye(2)),
            inventory,
        )
        proposal = model.propose(
            (torch.ones(2), torch.eye(2)),
            (torch.ones(2), torch.eye(2)),
            inventory,
            top_k=2,
        )
    else:
        model, resolver = _prefix_model()
        scores = model.score_inventory((11, 12), (21,), inventory)
        proposal = model.propose((11, 12), (21,), inventory, top_k=2)
    assert scores.probabilities.tolist()[1] == 0.0
    assert proposal.status is ProposalStatus.SUCCESS
    assert proposal.legal_action_count == 1
    assert tuple(candidate.action_digest for candidate in proposal.candidates) == (
        legal.action_digest,
    )
    assert proposal.legal_mask_digest == inventory.legal_mask_digest
    assert all(view.occurrence_path == (0,) for view in resolver.views)


@pytestmark_ml
@pytest.mark.parametrize("family", ["gnn", "prefix_transformer"])
def test_empty_legal_set_returns_typed_no_action_without_encoding(family: str) -> None:
    action = FixtureAction.create()
    inventory = _inventory((action,), (False,))
    if family == "gnn":
        model, _ = _graph_model()
        proposal = model.propose(object(), object(), inventory, top_k=5)
    else:
        model, _ = _prefix_model()
        proposal = model.propose((), (), inventory, top_k=5)
    assert proposal.status is ProposalStatus.NO_LEGAL_ACTION
    assert proposal.candidates == ()
    assert proposal.legal_action_count == 0


@pytestmark_ml
@pytest.mark.parametrize(
    ("inventory_status", "proposal_status"),
    [
        (ActionInventoryStatus.UNSUPPORTED, ProposalStatus.UNSUPPORTED),
        (ActionInventoryStatus.INVALID, ProposalStatus.INVALID),
    ],
)
def test_enumerator_failure_status_is_preserved_without_model_execution(
    inventory_status: ActionInventoryStatus,
    proposal_status: ProposalStatus,
) -> None:
    model, _ = _graph_model()
    inventory = _inventory(
        (),
        (),
        status=inventory_status,
        detail=f"fixture {inventory_status.value} reason",
    )
    proposal = model.propose(object(), object(), inventory, top_k=1)
    assert proposal.status is proposal_status
    assert proposal.detail == f"fixture {inventory_status.value} reason"
    assert proposal.candidates == ()


@pytestmark_ml
@pytest.mark.parametrize("family", ["gnn", "prefix_transformer"])
def test_unseen_rule_is_retained_as_unsupported_and_never_emitted(family: str) -> None:
    unknown = FixtureAction.create(rule_id="HELD-OUT", salt="unknown")
    inventory = _inventory((unknown,), (True,))
    if family == "gnn":
        model, _ = _graph_model()
        proposal = model.propose(object(), object(), inventory, top_k=1)
    else:
        model, _ = _prefix_model()
        proposal = model.propose((), (), inventory, top_k=1)
    assert proposal.status is ProposalStatus.UNSUPPORTED
    assert proposal.candidates == ()
    assert "HELD-OUT" in proposal.detail


@pytestmark_ml
@pytest.mark.parametrize("family", ["gnn", "prefix_transformer"])
def test_invalid_occurrence_is_typed_and_retained(family: str) -> None:
    invalid = FixtureAction.create(occurrence_path=(99,))
    inventory = _inventory((invalid,), (True,))
    if family == "gnn":
        model, _ = _graph_model()
        proposal = model.propose(
            (torch.ones(2), torch.eye(2)),
            (torch.ones(2), torch.eye(2)),
            inventory,
            top_k=1,
        )
    else:
        model, _ = _prefix_model()
        proposal = model.propose((11, 12), (21,), inventory, top_k=1)
    assert proposal.status is ProposalStatus.INVALID
    assert proposal.candidates == ()
    assert "occurrence path" in proposal.detail


@pytestmark_ml
@pytest.mark.parametrize("family", ["gnn", "prefix_transformer"])
def test_unsupported_feature_resolution_returns_typed_status(family: str) -> None:
    class UnsupportedResolver:
        def resolve(
            self,
            current_encoding: object,
            action: ActionScoringViewV1,
        ) -> ResolvedActionFeaturesV1:
            raise UnsupportedPolicyStateError("unsupported action feature")

    action = FixtureAction.create(salt="unsupported-resolver")
    inventory = _inventory((action,), (True,))
    if family == "gnn":
        model, _ = _graph_model()
        model.action_resolver = UnsupportedResolver()
        proposal = model.propose(
            (torch.ones(2), torch.eye(2)),
            (torch.zeros(2), torch.eye(2)),
            inventory,
            top_k=1,
        )
    else:
        model, _ = _prefix_model()
        model.action_resolver = UnsupportedResolver()
        proposal = model.propose((11, 12), (21,), inventory, top_k=1)

    assert proposal.status is ProposalStatus.UNSUPPORTED
    assert proposal.candidates == ()
    assert proposal.detail == "unsupported action feature"


@pytestmark_ml
def test_direction_sites_references_and_ordered_arguments_remain_distinct() -> None:
    model, _ = _graph_model()
    forward = FixtureAction.create(
        direction="forward",
        occurrence_path=(0,),
        ordered_arguments=(0, 1),
        salt="forward",
    )
    backward = FixtureAction.create(
        direction="backward",
        occurrence_path=(0,),
        ordered_arguments=(0, 1),
        salt="backward",
    )
    repeated_slot = FixtureAction.create(
        direction="forward",
        occurrence_path=(1,),
        ordered_arguments=(0, 1),
        salt="slot",
    )
    reversed_arguments = FixtureAction.create(
        direction="forward",
        occurrence_path=(0,),
        ordered_arguments=(1, 0),
        salt="arguments",
    )
    actions = (forward, backward, repeated_slot, reversed_arguments)
    inventory = _inventory(actions, (True,) * len(actions))
    with torch.no_grad():
        model.scorer.argument_position.weight[0].fill_(1.0)
        model.scorer.argument_position.weight[1].fill_(2.0)
    scores = model.score_inventory(
        (torch.tensor([0.5, -0.5]), torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        (torch.tensor([-0.5, 0.5]), torch.zeros((1, 2))),
        inventory,
    )
    assert len(set(action.action_digest for action in actions)) == 4
    assert forward.occurrence_path != repeated_slot.occurrence_path
    assert forward.direction != backward.direction
    assert forward.ordered_arguments != reversed_arguments.ordered_arguments
    assert scores.logits.shape == (4,)
    # Position-specific multiplication makes argument order observable.
    assert not torch.allclose(scores.logits[0], scores.logits[3])


@pytestmark_ml
def test_variable_sizes_and_repeated_references_preserve_concrete_paths() -> None:
    model, _ = _graph_model()
    first_reference = FixtureAction.create(occurrence_path=(0,), salt="first-reference")
    repeated_reference = FixtureAction.create(occurrence_path=(1,), salt="repeated-reference")
    inventory = _inventory((first_reference, repeated_reference), (True, True))
    # Both ordered child slots reference the same underlying encoded node value.
    repeated_nodes = torch.tensor([[2.0, -1.0], [2.0, -1.0]])
    proposal = model.propose(
        (torch.ones(2), repeated_nodes),
        (torch.zeros(2), torch.zeros((1, 2))),
        inventory,
        top_k=2,
    )
    assert {candidate.action.occurrence_path for candidate in proposal.candidates} == {
        (0,),
        (1,),
    }
    assert proposal.candidates[0].logit != pytest.approx(proposal.candidates[1].logit)
    # The same injected encoder also accepts a different node count.
    larger = FixtureAction.create(occurrence_path=(2,), salt="larger")
    scores = model.score_inventory(
        (torch.ones(2), torch.ones((3, 2))),
        (torch.zeros(2), torch.zeros((1, 2))),
        _inventory((larger,), (True,)),
    )
    assert scores.logits.shape == (1,)


@pytestmark_ml
def test_successor_and_digest_are_not_model_features() -> None:
    model, resolver = _graph_model()
    first = FixtureAction.create(successor_signature="a" * 64, salt="first")
    second = FixtureAction.create(successor_signature="b" * 64, salt="second")
    graph = (torch.ones(2), torch.eye(2))
    first_scores = model.score_inventory(graph, graph, _inventory((first,), (True,)))
    second_scores = model.score_inventory(graph, graph, _inventory((second,), (True,)))
    assert torch.allclose(first_scores.logits, second_scores.logits)
    assert all(not hasattr(view, "successor_signature") for view in resolver.views)


@pytestmark_ml
def test_deterministic_digest_tie_breaking_and_proposal_json_round_trip() -> None:
    model, _ = _graph_model()
    with torch.no_grad():
        for parameter in model.scorer.parameters():
            parameter.zero_()
    actions = (
        FixtureAction.create(occurrence_path=(0,), salt="z"),
        FixtureAction.create(occurrence_path=(1,), salt="a"),
    )
    inventory = _inventory(actions, (True, True))
    graph = (torch.ones(2), torch.eye(2))
    proposal = model.propose(graph, graph, inventory, top_k=2)
    assert tuple(candidate.action_digest for candidate in proposal.candidates) == tuple(
        sorted(action.action_digest for action in actions)
    )
    encoded = json.loads(json.dumps(proposal.as_dict(), allow_nan=False, sort_keys=True))
    loaded = ProposalV1.from_dict(
        encoded,
        action_loader=lambda payload: FixtureAction.from_proposal_payload(dict(payload)),
    )
    assert loaded.as_dict() == proposal.as_dict()
    assert loaded.returned_probability_mass == pytest.approx(1.0)
    encoded["candidates"].append("not-an-object")
    with pytest.raises(TypeError, match="candidate must be an object"):
        ProposalV1.from_dict(
            encoded,
            action_loader=lambda payload: FixtureAction.from_proposal_payload(dict(payload)),
        )


@pytestmark_ml
@pytest.mark.parametrize("family", ["gnn", "prefix_transformer"])
def test_cpu_forward_backward_and_state_dict_round_trip(family: str) -> None:
    actions = (
        FixtureAction.create(occurrence_path=(0,), salt="zero"),
        FixtureAction.create(occurrence_path=(1,), salt="one"),
    )
    inventory = _inventory(actions, (True, True))
    if family == "gnn":
        model, _ = _graph_model()
        arguments = (
            (torch.tensor([0.5, -0.5]), torch.eye(2)),
            (torch.tensor([-0.5, 0.5]), torch.eye(2)),
            inventory,
        )
        clone, _ = _graph_model()
    else:
        model, _ = _prefix_model()
        arguments = ((11, 12), (21,), inventory)
        clone, _ = _prefix_model()
    scores = model(*arguments)
    loss = -torch.log(scores.probabilities[0].clamp_min(1e-8))
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    clone.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    model.eval()
    clone.eval()
    with torch.no_grad():
        expected = model(*arguments).logits
        observed = clone(*arguments).logits
    assert torch.allclose(expected, observed)


@pytestmark_ml
def test_transformer_masked_padding_does_not_change_scores() -> None:
    model, _ = _prefix_model()
    actions = (
        FixtureAction.create(occurrence_path=(0,), salt="zero"),
        FixtureAction.create(occurrence_path=(1,), salt="one"),
    )
    inventory = _inventory(actions, (True, True))
    unpadded = model.score_inventory((11, 12), (21,), inventory)
    padded = model.score_inventory((11, 12), (21,), inventory, pad_to_length=12)
    assert torch.allclose(unpadded.logits, padded.logits)
    assert torch.allclose(unpadded.probabilities, padded.probabilities)
