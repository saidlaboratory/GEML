"""Goal-conditioned scoring and proposal contracts for concrete rewrite actions.

The persisted rewrite-action schema and graph encoder are owned by upstream
workstreams.  This module deliberately consumes both through narrow protocols:
it does not duplicate either contract.  A single :class:`ActionInventoryV1`
contains the registry-derived candidate order and legal mask used by every
proposer family.

Only current/goal encodings and the rule, direction, site, and ordered argument
views of an action enter the model.  Successor signatures, verifier outcomes,
demonstration labels, and future costs remain output/provenance data and are
never exposed to the scoring head.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from geml.graph.schema import strict_json_snapshot

try:  # The core package remains importable without the optional ``ml`` extra.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised in a core-only subprocess test.
    torch = None
    nn = None

_ModuleBase = object if nn is None else nn.Module

PROPOSAL_SCHEMA_VERSION = "geml-rewrite-proposal-v1"
LEGAL_MASK_VERSION = "geml-legal-action-mask-v1"
_DIRECTIONS = ("backward", "forward")
_MODEL_FAMILIES = ("gnn", "prefix_transformer")


class MissingMLDependencyError(RuntimeError):
    """Raised when a neural proposer is constructed without the optional ML extra."""


class PolicyInputError(ValueError):
    """Base class for a typed, attributable proposal-input failure."""


class InvalidPolicyStateError(PolicyInputError):
    """The current state, goal, action inventory, or feature resolution is invalid."""


class UnsupportedPolicyStateError(PolicyInputError):
    """The inventory contains a rule absent from the frozen scoring vocabulary."""


class ProposalStatus(StrEnum):
    """Outcome of one proposal request."""

    SUCCESS = "success"
    NO_LEGAL_ACTION = "no_legal_action"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


class ActionInventoryStatus(StrEnum):
    """Outcome of registry-backed enumeration before model scoring."""

    READY = "ready"
    NO_LEGAL_ACTION = "no_legal_action"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


class ScoreSemantics(StrEnum):
    """Meaning of persisted candidate scores."""

    LEGAL_SET_SOFTMAX = (
        "raw_logit_and_temperature_scaled_softmax_probability_over_complete_legal_set"
    )


@runtime_checkable
class NormalizedActionProtocol(Protocol):
    """Read-only view of the upstream normalized ``RewriteActionV1``.

    Issue #61 supplies a compatibility record with these attributes while the
    canonical persisted action remains owned by Issue #55.
    """

    action: object
    action_digest: str
    rule_id: str
    direction: str
    occurrence_path: tuple[int, ...]
    ordered_arguments: tuple[object, ...]
    source_signature: str
    successor_signature: str

    def as_dict(self) -> Mapping[str, object]:
        """Return the lossless, strict-JSON normalized action boundary."""


@runtime_checkable
class GraphEncodingProtocol(Protocol):
    """Minimum shared-encoder output consumed by the GNN policy."""

    graph_embedding: object
    node_embeddings: object


@runtime_checkable
class SharedGraphEncoderProtocol(Protocol):
    """Injected Workstream-2 encoder boundary."""

    def forward(self, graph_input: object) -> GraphEncodingProtocol:
        """Encode one variable-sized graph with shared parameters."""


@dataclass(frozen=True, slots=True)
class ActionScoringViewV1:
    """Leakage-safe action fields available to an action-feature resolver."""

    rule_id: str
    direction: str
    occurrence_path: tuple[int, ...]
    ordered_arguments: tuple[JsonValue, ...]


@dataclass(frozen=True, slots=True)
class ResolvedActionFeaturesV1:
    """Current-state features for one concrete rule/site/argument choice.

    ``site_embedding`` has shape ``[hidden_width]`` and
    ``ordered_argument_embeddings`` has shape
    ``[argument_count, hidden_width]``.  The latter stays ordered; the scoring
    head applies position-specific interactions before pooling it.
    """

    site_embedding: object
    ordered_argument_embeddings: object


@dataclass(frozen=True, slots=True)
class LegalActionEnumerationV1:
    """Registry-derived actions and mask before goal provenance is attached."""

    actions: tuple[NormalizedActionProtocol, ...]
    legal_mask: tuple[bool, ...]
    status: ActionInventoryStatus = ActionInventoryStatus.READY
    detail: str = "registry enumeration complete"

    def __post_init__(self) -> None:
        if not isinstance(self.actions, tuple):
            raise TypeError("enumerated actions must be a tuple")
        if (
            not isinstance(self.legal_mask, tuple)
            or len(self.legal_mask) != len(self.actions)
            or any(not isinstance(value, bool) for value in self.legal_mask)
        ):
            raise TypeError("enumerated legal_mask must be a bool tuple aligned with actions")
        if not isinstance(self.status, ActionInventoryStatus):
            raise TypeError("enumeration status must be an ActionInventoryStatus")
        object.__setattr__(self, "detail", _require_nonblank(self.detail, "detail"))
        if self.status is not ActionInventoryStatus.READY and any(self.legal_mask):
            raise ValueError(f"{self.status.value} enumeration cannot mark an action legal")


@runtime_checkable
class ActionFeatureResolverProtocol(Protocol):
    """Maps a source occurrence and ordered bindings into encoder-space features."""

    def resolve(
        self,
        current_encoding: GraphEncodingProtocol,
        action: ActionScoringViewV1,
    ) -> ResolvedActionFeaturesV1:
        """Resolve features without access to successor or verifier evidence."""


@runtime_checkable
class LegalActionProviderProtocol(Protocol):
    """Injected, registry-backed concrete-action enumerator.

    The provider owns legality and may replay/materialize successors internally,
    but the resulting successor and verification data do not enter model
    features.  Legality is a property of the current state under its recorded
    assumptions/domain; the goal is deliberately absent from this boundary so
    the provider cannot prune the mask using target distance or future cost.
    """

    def enumerate_actions(
        self,
        *,
        current_state: object,
        current_signature: str,
        assumptions: tuple[str, ...],
        domain_mode: str,
        vocabulary: RuleVocabularyV1,
    ) -> LegalActionEnumerationV1:
        """Enumerate the current-state action set under the frozen registry."""


def _require_nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-blank string")
    return value


def _require_digest(value: object, name: str) -> str:
    digest = _require_nonblank(value, name)
    if (
        len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 digest")
    return digest


def _require_signature(value: object, name: str) -> str:
    return _require_digest(value, name)


def _direction_value(value: object) -> str:
    direction = value.value if isinstance(value, StrEnum) else value
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}")
    return str(direction)


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _snapshot_action(action: NormalizedActionProtocol) -> dict[str, JsonValue]:
    """Validate and snapshot the complete upstream action identity."""

    if not isinstance(action, NormalizedActionProtocol):
        raise TypeError("action must implement NormalizedActionProtocol")
    raw_payload = action.as_dict()
    if not isinstance(raw_payload, Mapping):
        raise TypeError("action.as_dict() must return a mapping")
    required = {
        "action",
        "action_digest",
        "direction",
        "occurrence_path",
        "ordered_arguments",
        "rule_id",
        "source_signature",
        "successor_signature",
    }
    if not required <= set(raw_payload):
        raise ValueError(
            "normalized action is missing required fields: "
            + ", ".join(sorted(required - set(raw_payload)))
        )
    snapshot = strict_json_snapshot(dict(raw_payload))
    if not isinstance(snapshot, dict):  # pragma: no cover - mapping input guarantees this.
        raise TypeError("normalized action must serialize as a JSON object")

    rule_id = _require_nonblank(snapshot["rule_id"], "action.rule_id")
    direction = _direction_value(snapshot["direction"])
    path = snapshot["occurrence_path"]
    if not isinstance(path, list) or any(
        isinstance(slot, bool) or not isinstance(slot, int) or slot < 0 for slot in path
    ):
        raise TypeError("serialized occurrence_path must be nonnegative integer JSON array")
    arguments = snapshot["ordered_arguments"]
    if not isinstance(arguments, list):
        raise TypeError("serialized ordered_arguments must be a JSON array")
    action_digest = _require_digest(snapshot["action_digest"], "action.action_digest")
    source_signature = _require_signature(snapshot["source_signature"], "action.source_signature")
    successor_signature = _require_signature(
        snapshot["successor_signature"], "action.successor_signature"
    )
    mirrored_action = _snapshot_scientific_json(action.action, "action.action")
    mirrored_arguments = [
        _snapshot_scientific_json(argument, "action.ordered_arguments item")
        for argument in action.ordered_arguments
    ]
    if (
        snapshot["action"] != mirrored_action
        or rule_id != action.rule_id
        or direction != _direction_value(action.direction)
        or tuple(path) != action.occurrence_path
        or arguments != mirrored_arguments
        or action_digest != action.action_digest
        or source_signature != action.source_signature
        or successor_signature != action.successor_signature
    ):
        raise ValueError("serialized action fields disagree with the normalized action view")
    return snapshot


def _snapshot_scientific_json(value: object, name: str) -> JsonValue:
    """Snapshot a raw JSON value or Issue #61 ``CanonicalJson`` wrapper."""

    to_value = getattr(value, "to_value", None)
    if callable(to_value):
        value = to_value()
    try:
        return strict_json_snapshot(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain a strict finite JSON value") from error


def action_scoring_view(action: NormalizedActionProtocol) -> ActionScoringViewV1:
    """Return only fields authorized as proposal features."""

    snapshot = _snapshot_action(action)
    return ActionScoringViewV1(
        rule_id=str(snapshot["rule_id"]),
        direction=str(snapshot["direction"]),
        occurrence_path=tuple(int(slot) for slot in snapshot["occurrence_path"]),  # type: ignore[arg-type]
        ordered_arguments=tuple(snapshot["ordered_arguments"]),  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True, order=True)
class RuleKeyV1:
    """One registry-derived directed rule key."""

    rule_id: str
    direction: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _require_nonblank(self.rule_id, "rule_id"))
        object.__setattr__(self, "direction", _direction_value(self.direction))


@dataclass(frozen=True, slots=True)
class RuleVocabularyV1:
    """Frozen scoring indices derived from, but not replacing, the rule registry."""

    registry_digest: str
    entries: tuple[RuleKeyV1, ...]
    _entry_to_index: Mapping[RuleKeyV1, int] = field(init=False, repr=False, compare=False)
    _rule_to_index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "registry_digest", _require_digest(self.registry_digest, "registry_digest")
        )
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, RuleKeyV1) for entry in self.entries
        ):
            raise TypeError("entries must be a tuple of RuleKeyV1 values")
        canonical_entries = tuple(sorted(self.entries))
        if self.entries != canonical_entries:
            raise ValueError("rule vocabulary entries must be in canonical sorted order")
        if len(set(self.entries)) != len(self.entries):
            raise ValueError("rule vocabulary entries must be unique")
        entry_to_index = {entry: index for index, entry in enumerate(self.entries)}
        rule_ids = tuple(sorted({entry.rule_id for entry in self.entries}))
        rule_to_index = {rule_id: index for index, rule_id in enumerate(rule_ids)}
        object.__setattr__(self, "_entry_to_index", entry_to_index)
        object.__setattr__(self, "_rule_to_index", rule_to_index)

    @classmethod
    def from_registry(
        cls,
        entries: Sequence[tuple[str, str] | RuleKeyV1],
        *,
        registry_digest: str,
    ) -> RuleVocabularyV1:
        """Create deterministic indices from an authoritative registry snapshot."""

        keys = tuple(
            entry if isinstance(entry, RuleKeyV1) else RuleKeyV1(*entry) for entry in entries
        )
        return cls(registry_digest=registry_digest, entries=tuple(sorted(keys)))

    @property
    def rule_count(self) -> int:
        return len(self._rule_to_index)

    def contains(self, action: NormalizedActionProtocol) -> bool:
        return RuleKeyV1(action.rule_id, _direction_value(action.direction)) in self._entry_to_index

    def rule_index(self, rule_id: str) -> int:
        try:
            return self._rule_to_index[rule_id]
        except KeyError as error:
            raise UnsupportedPolicyStateError(f"unregistered rule {rule_id!r}") from error

    @staticmethod
    def direction_index(direction: str) -> int:
        return _DIRECTIONS.index(_direction_value(direction))


def compute_legal_mask_digest(
    *,
    action_digests: tuple[str, ...],
    legal_mask: tuple[bool, ...],
    current_signature: str,
    goal_signature: str,
    registry_digest: str,
    status: ActionInventoryStatus,
) -> str:
    """Hash one complete, ordered registry inventory and its legal mask.

    This is the public audit boundary used by consumers that persist the shared
    inventory separately from :class:`ActionInventoryV1`.  The ordered action
    digests are part of the identity: permuting the inventory without permuting
    the mask cannot preserve the digest.
    """

    if not isinstance(action_digests, tuple):
        raise TypeError("action_digests must be a tuple")
    normalized_digests = tuple(
        _require_digest(digest, f"action_digests[{index}]")
        for index, digest in enumerate(action_digests)
    )
    if len(set(normalized_digests)) != len(normalized_digests):
        raise ValueError("action_digests must be unique")
    if (
        not isinstance(legal_mask, tuple)
        or len(legal_mask) != len(normalized_digests)
        or any(not isinstance(value, bool) for value in legal_mask)
    ):
        raise TypeError("legal_mask must be a bool tuple aligned with action_digests")
    if not isinstance(status, ActionInventoryStatus):
        raise TypeError("status must be an ActionInventoryStatus")
    payload = {
        "actions": [
            {"action_digest": digest, "legal": legal}
            for digest, legal in zip(normalized_digests, legal_mask, strict=True)
        ],
        "current_signature": _require_signature(current_signature, "current_signature"),
        "goal_signature": _require_signature(goal_signature, "goal_signature"),
        "registry_digest": _require_digest(registry_digest, "registry_digest"),
        "status": status.value,
        "version": LEGAL_MASK_VERSION,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class ActionInventoryV1:
    """Concrete candidate actions plus the single authoritative legal mask."""

    current_signature: str
    goal_signature: str
    vocabulary: RuleVocabularyV1
    actions: tuple[NormalizedActionProtocol, ...]
    legal_mask: tuple[bool, ...]
    status: ActionInventoryStatus = ActionInventoryStatus.READY
    detail: str = "registry enumeration complete"
    _action_snapshots: tuple[dict[str, JsonValue], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_signature",
            _require_signature(self.current_signature, "current_signature"),
        )
        object.__setattr__(
            self,
            "goal_signature",
            _require_signature(self.goal_signature, "goal_signature"),
        )
        if not isinstance(self.vocabulary, RuleVocabularyV1):
            raise TypeError("vocabulary must be a RuleVocabularyV1")
        if not isinstance(self.actions, tuple):
            raise TypeError("actions must be a tuple")
        if (
            not isinstance(self.legal_mask, tuple)
            or len(self.legal_mask) != len(self.actions)
            or any(not isinstance(value, bool) for value in self.legal_mask)
        ):
            raise TypeError("legal_mask must be a bool tuple aligned with actions")
        if not isinstance(self.status, ActionInventoryStatus):
            raise TypeError("status must be an ActionInventoryStatus")
        object.__setattr__(self, "detail", _require_nonblank(self.detail, "detail"))
        snapshots = tuple(_snapshot_action(action) for action in self.actions)
        digests = tuple(str(snapshot["action_digest"]) for snapshot in snapshots)
        if len(set(digests)) != len(digests):
            raise ValueError("action inventory contains duplicate action digests")
        for snapshot in snapshots:
            if snapshot["source_signature"] != self.current_signature:
                raise InvalidPolicyStateError(
                    "every action source signature must equal the inventory current signature"
                )
        object.__setattr__(self, "_action_snapshots", snapshots)
        if (
            self.status
            in (
                ActionInventoryStatus.NO_LEGAL_ACTION,
                ActionInventoryStatus.UNSUPPORTED,
                ActionInventoryStatus.INVALID,
            )
            and self.legal_action_count != 0
        ):
            raise ValueError(f"{self.status.value} inventory cannot mark an action legal")

    @property
    def legal_action_count(self) -> int:
        return sum(self.legal_mask)

    @property
    def unknown_rule_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {action.rule_id for action in self.actions if not self.vocabulary.contains(action)}
            )
        )

    @property
    def legal_mask_digest(self) -> str:
        return compute_legal_mask_digest(
            action_digests=tuple(
                str(snapshot["action_digest"]) for snapshot in self._action_snapshots
            ),
            legal_mask=self.legal_mask,
            current_signature=self.current_signature,
            goal_signature=self.goal_signature,
            registry_digest=self.vocabulary.registry_digest,
            status=self.status,
        )


def build_action_inventory(
    *,
    current_state: object,
    current_signature: str,
    goal_signature: str,
    assumptions: tuple[str, ...],
    domain_mode: str,
    vocabulary: RuleVocabularyV1,
    provider: LegalActionProviderProtocol,
) -> ActionInventoryV1:
    """Attach goal provenance to a current-only registry enumeration.

    The returned inventory is the single object callers should share across
    GNN, transformer, and uniform-valid proposal arms.
    """

    current = _require_signature(current_signature, "current_signature")
    goal = _require_signature(goal_signature, "goal_signature")
    if not isinstance(assumptions, tuple) or any(
        not isinstance(assumption, str) or not assumption.strip() for assumption in assumptions
    ):
        raise TypeError("assumptions must be a tuple of non-blank strings")
    if assumptions != tuple(sorted(set(assumptions))):
        raise ValueError("assumptions must be sorted and unique")
    domain = _require_nonblank(domain_mode, "domain_mode")
    if not isinstance(vocabulary, RuleVocabularyV1):
        raise TypeError("vocabulary must be a RuleVocabularyV1")
    if not isinstance(provider, LegalActionProviderProtocol):
        raise TypeError("provider must implement LegalActionProviderProtocol")
    enumeration = provider.enumerate_actions(
        current_state=current_state,
        current_signature=current,
        assumptions=assumptions,
        domain_mode=domain,
        vocabulary=vocabulary,
    )
    if not isinstance(enumeration, LegalActionEnumerationV1):
        raise TypeError("provider must return LegalActionEnumerationV1")
    return ActionInventoryV1(
        current_signature=current,
        goal_signature=goal,
        vocabulary=vocabulary,
        actions=enumeration.actions,
        legal_mask=enumeration.legal_mask,
        status=enumeration.status,
        detail=enumeration.detail,
    )


@dataclass(frozen=True, slots=True)
class ModelIdentityV1:
    """Frozen model/config/checkpoint identity attached to every proposal."""

    model_family: str
    model_id: str
    checkpoint_digest: str
    config_digest: str

    def __post_init__(self) -> None:
        if self.model_family not in _MODEL_FAMILIES:
            raise ValueError(f"model_family must be one of {_MODEL_FAMILIES}")
        object.__setattr__(self, "model_id", _require_nonblank(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "checkpoint_digest",
            _require_digest(self.checkpoint_digest, "checkpoint_digest"),
        )
        object.__setattr__(
            self, "config_digest", _require_digest(self.config_digest, "config_digest")
        )

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "checkpoint_digest": self.checkpoint_digest,
            "config_digest": self.config_digest,
            "model_family": self.model_family,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ModelIdentityV1:
        _require_exact_keys(
            payload,
            {"checkpoint_digest", "config_digest", "model_family", "model_id"},
            "model identity",
        )
        return cls(
            model_family=payload["model_family"],  # type: ignore[arg-type]
            model_id=payload["model_id"],  # type: ignore[arg-type]
            checkpoint_digest=payload["checkpoint_digest"],  # type: ignore[arg-type]
            config_digest=payload["config_digest"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ProposalCandidateV1:
    """One ranked legal action with normalized probability over all legal actions."""

    rank: int
    action: NormalizedActionProtocol
    logit: float
    probability: float
    _action_snapshot: dict[str, JsonValue] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise TypeError("rank must be a positive integer")
        if isinstance(self.logit, bool) or not isinstance(self.logit, int | float):
            raise TypeError("logit must be a real number")
        if isinstance(self.probability, bool) or not isinstance(self.probability, int | float):
            raise TypeError("probability must be a real number")
        if not math.isfinite(float(self.logit)):
            raise ValueError("candidate logits must be finite")
        if not math.isfinite(float(self.probability)) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("candidate probability must be finite and in [0, 1]")
        object.__setattr__(self, "logit", float(self.logit))
        object.__setattr__(self, "probability", float(self.probability))
        object.__setattr__(self, "_action_snapshot", _snapshot_action(self.action))

    @property
    def action_digest(self) -> str:
        return str(self._action_snapshot["action_digest"])

    @property
    def successor_signature(self) -> str:
        return str(self._action_snapshot["successor_signature"])

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "action": self._action_snapshot,
            "logit": self.logit,
            "probability": self.probability,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        action_loader: Callable[[Mapping[str, object]], NormalizedActionProtocol],
    ) -> ProposalCandidateV1:
        _require_exact_keys(payload, {"action", "logit", "probability", "rank"}, "candidate")
        action_payload = payload["action"]
        if not isinstance(action_payload, Mapping):
            raise TypeError("candidate action must be an object")
        action = action_loader(action_payload)
        candidate = cls(
            rank=payload["rank"],  # type: ignore[arg-type]
            action=action,
            logit=payload["logit"],  # type: ignore[arg-type]
            probability=payload["probability"],  # type: ignore[arg-type]
        )
        if candidate._action_snapshot != dict(action_payload):
            raise ValueError("action loader did not reproduce the serialized action")
        return candidate


@dataclass(frozen=True, slots=True)
class ProposalV1:
    """Typed top-k policy output shared by GNN and transformer proposers."""

    current_signature: str
    goal_signature: str
    candidates: tuple[ProposalCandidateV1, ...]
    legal_action_count: int
    requested_top_k: int
    legal_mask_digest: str
    rule_registry_digest: str
    model_identity: ModelIdentityV1
    probability_temperature: float
    status: ProposalStatus
    detail: str
    score_semantics: ScoreSemantics = ScoreSemantics.LEGAL_SET_SOFTMAX
    schema_version: str = PROPOSAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_signature",
            _require_signature(self.current_signature, "current_signature"),
        )
        object.__setattr__(
            self,
            "goal_signature",
            _require_signature(self.goal_signature, "goal_signature"),
        )
        if self.schema_version != PROPOSAL_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {PROPOSAL_SCHEMA_VERSION!r}")
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(candidate, ProposalCandidateV1) for candidate in self.candidates
        ):
            raise TypeError("candidates must be a tuple of ProposalCandidateV1")
        if (
            isinstance(self.legal_action_count, bool)
            or not isinstance(self.legal_action_count, int)
            or self.legal_action_count < 0
        ):
            raise TypeError("legal_action_count must be a nonnegative integer")
        if (
            isinstance(self.requested_top_k, bool)
            or not isinstance(self.requested_top_k, int)
            or self.requested_top_k < 1
        ):
            raise TypeError("requested_top_k must be a positive integer")
        object.__setattr__(
            self,
            "legal_mask_digest",
            _require_digest(self.legal_mask_digest, "legal_mask_digest"),
        )
        object.__setattr__(
            self,
            "rule_registry_digest",
            _require_digest(self.rule_registry_digest, "rule_registry_digest"),
        )
        if not isinstance(self.model_identity, ModelIdentityV1):
            raise TypeError("model_identity must be a ModelIdentityV1")
        if (
            isinstance(self.probability_temperature, bool)
            or not isinstance(self.probability_temperature, int | float)
            or not math.isfinite(float(self.probability_temperature))
            or self.probability_temperature <= 0
        ):
            raise ValueError("probability_temperature must be finite and positive")
        object.__setattr__(self, "probability_temperature", float(self.probability_temperature))
        if not isinstance(self.status, ProposalStatus):
            raise TypeError("status must be a ProposalStatus")
        object.__setattr__(self, "detail", _require_nonblank(self.detail, "detail"))
        if self.score_semantics is not ScoreSemantics.LEGAL_SET_SOFTMAX:
            raise ValueError("unsupported score semantics")
        expected_candidate_count = (
            min(self.requested_top_k, self.legal_action_count)
            if self.status is ProposalStatus.SUCCESS
            else 0
        )
        if len(self.candidates) != expected_candidate_count:
            raise ValueError("candidate count is inconsistent with status/top-k/legal count")
        if self.status is ProposalStatus.SUCCESS and self.legal_action_count == 0:
            raise ValueError("successful proposals require at least one legal action")
        if self.status is ProposalStatus.NO_LEGAL_ACTION and self.legal_action_count != 0:
            raise ValueError("no_legal_action status requires zero legal actions")
        ranks = tuple(candidate.rank for candidate in self.candidates)
        if ranks != tuple(range(1, len(self.candidates) + 1)):
            raise ValueError("candidate ranks must be contiguous and one-based")
        digests = tuple(candidate.action_digest for candidate in self.candidates)
        if len(set(digests)) != len(digests):
            raise ValueError("proposal candidates must have unique action digests")
        if any(
            candidate._action_snapshot["source_signature"] != self.current_signature
            for candidate in self.candidates
        ):
            raise ValueError("every candidate action must originate at current_signature")
        order = tuple(
            sorted(
                self.candidates,
                key=lambda candidate: (-candidate.logit, candidate.action_digest),
            )
        )
        if self.candidates != order:
            raise ValueError("candidates must use descending logit and digest tie-breaking")
        returned_mass = self.returned_probability_mass
        if returned_mass > 1.0 + 1e-6:
            raise ValueError("returned candidate probability mass cannot exceed one")
        if (
            self.status is ProposalStatus.SUCCESS
            and len(self.candidates) == self.legal_action_count
            and not math.isclose(returned_mass, 1.0, rel_tol=1e-6, abs_tol=1e-7)
        ):
            raise ValueError("a complete legal candidate list must carry unit probability mass")

    @property
    def returned_probability_mass(self) -> float:
        return sum(candidate.probability for candidate in self.candidates)

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "current_signature": self.current_signature,
            "detail": self.detail,
            "goal_signature": self.goal_signature,
            "legal_action_count": self.legal_action_count,
            "legal_mask_digest": self.legal_mask_digest,
            "model_identity": self.model_identity.as_dict(),
            "probability_temperature": self.probability_temperature,
            "requested_top_k": self.requested_top_k,
            "rule_registry_digest": self.rule_registry_digest,
            "schema_version": self.schema_version,
            "score_semantics": self.score_semantics.value,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        action_loader: Callable[[Mapping[str, object]], NormalizedActionProtocol],
    ) -> ProposalV1:
        _require_exact_keys(
            payload,
            {
                "candidates",
                "current_signature",
                "detail",
                "goal_signature",
                "legal_action_count",
                "legal_mask_digest",
                "model_identity",
                "probability_temperature",
                "requested_top_k",
                "rule_registry_digest",
                "schema_version",
                "score_semantics",
                "status",
            },
            "proposal",
        )
        candidate_payloads = payload["candidates"]
        if not isinstance(candidate_payloads, list):
            raise TypeError("proposal candidates must be a JSON array")
        if any(not isinstance(item, Mapping) for item in candidate_payloads):
            raise TypeError("every proposal candidate must be an object")
        model_payload = payload["model_identity"]
        if not isinstance(model_payload, Mapping):
            raise TypeError("model_identity must be an object")
        return cls(
            current_signature=payload["current_signature"],  # type: ignore[arg-type]
            goal_signature=payload["goal_signature"],  # type: ignore[arg-type]
            candidates=tuple(
                ProposalCandidateV1.from_dict(item, action_loader=action_loader)
                for item in candidate_payloads
            ),
            legal_action_count=payload["legal_action_count"],  # type: ignore[arg-type]
            requested_top_k=payload["requested_top_k"],  # type: ignore[arg-type]
            legal_mask_digest=payload["legal_mask_digest"],  # type: ignore[arg-type]
            rule_registry_digest=payload["rule_registry_digest"],  # type: ignore[arg-type]
            model_identity=ModelIdentityV1.from_dict(model_payload),
            probability_temperature=payload["probability_temperature"],  # type: ignore[arg-type]
            status=ProposalStatus(payload["status"]),
            detail=payload["detail"],  # type: ignore[arg-type]
            score_semantics=ScoreSemantics(payload["score_semantics"]),
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    name: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )


@dataclass(frozen=True, slots=True)
class PolicyScoreBatchV1:
    """Differentiable full-inventory scores; masked probabilities are exactly zero."""

    logits: object
    probabilities: object
    legal_mask: object


class FactoredActionScorer(_ModuleBase):
    """Compact rule/direction/site/ordered-argument scorer.

    The directional query is ``[current, goal, current-goal, current*goal]``.
    A concrete action key combines its current-state site, ordered argument
    interactions, rule embedding, and direction embedding.
    """

    def __init__(
        self,
        *,
        hidden_width: int,
        vocabulary: RuleVocabularyV1,
        max_arguments: int = 8,
        probability_temperature: float = 1.0,
    ) -> None:
        _require_torch()
        super().__init__()
        if isinstance(hidden_width, bool) or not isinstance(hidden_width, int) or hidden_width < 1:
            raise TypeError("hidden_width must be a positive integer")
        if (
            isinstance(max_arguments, bool)
            or not isinstance(max_arguments, int)
            or max_arguments < 1
        ):
            raise TypeError("max_arguments must be a positive integer")
        if vocabulary.rule_count < 1:
            raise ValueError("the scoring vocabulary must contain at least one rule")
        if (
            isinstance(probability_temperature, bool)
            or not isinstance(probability_temperature, int | float)
            or not math.isfinite(float(probability_temperature))
            or probability_temperature <= 0
        ):
            raise ValueError("probability_temperature must be finite and positive")
        self.hidden_width = hidden_width
        self.max_arguments = max_arguments
        self.probability_temperature = float(probability_temperature)
        self.vocabulary = vocabulary
        self.rule_embedding = nn.Embedding(vocabulary.rule_count, hidden_width)
        self.direction_embedding = nn.Embedding(len(_DIRECTIONS), hidden_width)
        self.argument_position = nn.Embedding(max_arguments, hidden_width)
        self.occurrence_path_encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_width,
            batch_first=True,
        )
        self.query_projection = nn.Sequential(
            nn.Linear(4 * hidden_width, hidden_width),
            nn.Tanh(),
        )
        self.action_projection = nn.Sequential(
            nn.Linear(5 * hidden_width, hidden_width),
            nn.Tanh(),
        )
        self.action_bias = nn.Linear(hidden_width, 1)

    def forward(
        self,
        current_graph_embedding: object,
        goal_graph_embedding: object,
        current_encoding: GraphEncodingProtocol,
        inventory: ActionInventoryV1,
        resolver: ActionFeatureResolverProtocol,
    ) -> PolicyScoreBatchV1:
        _require_torch()
        current = _vector(current_graph_embedding, self.hidden_width, "current graph embedding")
        goal = _vector(goal_graph_embedding, self.hidden_width, "goal graph embedding")
        unknown = inventory.unknown_rule_ids
        if unknown:
            raise UnsupportedPolicyStateError(
                "inventory contains unregistered rules: " + ", ".join(unknown)
            )
        legal_mask = torch.tensor(
            inventory.legal_mask,
            dtype=torch.bool,
            device=current.device,
        )
        if not inventory.actions:
            empty = current.new_empty((0,))
            return PolicyScoreBatchV1(logits=empty, probabilities=empty, legal_mask=legal_mask)
        if inventory.legal_action_count == 0:
            zeros = current.new_zeros((len(inventory.actions),))
            return PolicyScoreBatchV1(logits=zeros, probabilities=zeros, legal_mask=legal_mask)

        query = self.query_projection(
            torch.cat((current, goal, current - goal, current * goal), dim=0)
        )
        logits: list[object] = []
        for action, legal in zip(inventory.actions, inventory.legal_mask, strict=True):
            if not legal:
                logits.append(current.new_zeros(()))
                continue
            view = action_scoring_view(action)
            try:
                resolved = resolver.resolve(current_encoding, view)
            except PolicyInputError:
                raise
            except (IndexError, KeyError) as error:
                raise InvalidPolicyStateError(
                    f"could not resolve occurrence path {view.occurrence_path}"
                ) from error
            if not isinstance(resolved, ResolvedActionFeaturesV1):
                raise TypeError("resolver must return ResolvedActionFeaturesV1")
            site = _vector(resolved.site_embedding, self.hidden_width, "site embedding")
            arguments = _argument_matrix(
                resolved.ordered_argument_embeddings,
                self.hidden_width,
                self.max_arguments,
            )
            if arguments.shape[0]:
                positions = self.argument_position(
                    torch.arange(arguments.shape[0], device=arguments.device)
                )
                argument_summary = (arguments * positions).sum(dim=0) / arguments.shape[0]
            else:
                argument_summary = site.new_zeros((self.hidden_width,))
            path_summary = self._encode_occurrence_path(view.occurrence_path, site)
            rule_index = self.vocabulary.rule_index(view.rule_id)
            direction_index = self.vocabulary.direction_index(view.direction)
            rule = self.rule_embedding(
                torch.tensor(rule_index, dtype=torch.long, device=site.device)
            )
            direction = self.direction_embedding(
                torch.tensor(direction_index, dtype=torch.long, device=site.device)
            )
            action_key = self.action_projection(
                torch.cat((site, argument_summary, rule, direction, path_summary), dim=0)
            )
            logit = (query * action_key).sum() / math.sqrt(self.hidden_width)
            logits.append(logit + self.action_bias(action_key).squeeze(-1))
        stacked = torch.stack(logits)
        probabilities = _masked_softmax(
            stacked / self.probability_temperature,
            legal_mask,
        )
        return PolicyScoreBatchV1(
            logits=stacked,
            probabilities=probabilities,
            legal_mask=legal_mask,
        )

    def _encode_occurrence_path(
        self,
        occurrence_path: tuple[int, ...],
        reference: object,
    ) -> object:
        """Encode ordered child slots without collapsing repeated DAG references."""

        if not occurrence_path:
            return reference.new_zeros((self.hidden_width,))
        path_values = reference.new_tensor(
            [math.log1p(slot + 1) for slot in occurrence_path]
        ).reshape(1, len(occurrence_path), 1)
        _, hidden = self.occurrence_path_encoder(path_values)
        return hidden[0, 0]


class GoalConditionedPolicyHead(_ModuleBase):
    """GNN proposal model reusing one injected shared graph encoder."""

    def __init__(
        self,
        *,
        encoder: SharedGraphEncoderProtocol,
        action_resolver: ActionFeatureResolverProtocol,
        vocabulary: RuleVocabularyV1,
        model_identity: ModelIdentityV1,
        hidden_width: int,
        max_arguments: int = 8,
        probability_temperature: float = 1.0,
    ) -> None:
        _require_torch()
        super().__init__()
        if model_identity.model_family != "gnn":
            raise ValueError("GoalConditionedPolicyHead requires model_family='gnn'")
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be an injected torch.nn.Module")
        self.encoder = encoder
        self.action_resolver = action_resolver
        self.vocabulary = vocabulary
        self.model_identity = model_identity
        self.scorer = FactoredActionScorer(
            hidden_width=hidden_width,
            vocabulary=vocabulary,
            max_arguments=max_arguments,
            probability_temperature=probability_temperature,
        )

    def score_inventory(
        self,
        current_graph: object,
        goal_graph: object,
        inventory: ActionInventoryV1,
    ) -> PolicyScoreBatchV1:
        """Return differentiable scores over the complete candidate inventory."""

        if not isinstance(inventory, ActionInventoryV1):
            raise TypeError("inventory must be an ActionInventoryV1")
        _raise_for_inventory_status(inventory)
        if inventory.unknown_rule_ids:
            raise UnsupportedPolicyStateError(
                "inventory contains unregistered rules: " + ", ".join(inventory.unknown_rule_ids)
            )
        if inventory.legal_action_count == 0:
            empty_or_zero = next(self.parameters()).new_zeros((len(inventory.actions),))
            legal_mask = torch.tensor(
                inventory.legal_mask,
                dtype=torch.bool,
                device=empty_or_zero.device,
            )
            return PolicyScoreBatchV1(
                logits=empty_or_zero,
                probabilities=empty_or_zero,
                legal_mask=legal_mask,
            )
        current_encoding = self.encoder(current_graph)
        goal_encoding = self.encoder(goal_graph)
        _validate_encoding(current_encoding, "current")
        _validate_encoding(goal_encoding, "goal")
        return self.scorer(
            current_encoding.graph_embedding,
            goal_encoding.graph_embedding,
            current_encoding,
            inventory,
            self.action_resolver,
        )

    def forward(
        self,
        current_graph: object,
        goal_graph: object,
        inventory: ActionInventoryV1,
    ) -> PolicyScoreBatchV1:
        return self.score_inventory(current_graph, goal_graph, inventory)

    def propose(
        self,
        current_graph: object,
        goal_graph: object,
        inventory: ActionInventoryV1,
        *,
        top_k: int,
    ) -> ProposalV1:
        """Return deterministic top-k candidates or one typed empty/failure state."""

        return _propose_with_scorer(
            inventory,
            top_k=top_k,
            model_identity=self.model_identity,
            probability_temperature=self.scorer.probability_temperature,
            score=lambda: self.score_inventory(current_graph, goal_graph, inventory),
        )


def _propose_with_scorer(
    inventory: ActionInventoryV1,
    *,
    top_k: int,
    model_identity: ModelIdentityV1,
    probability_temperature: float,
    score: Callable[[], PolicyScoreBatchV1],
) -> ProposalV1:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise TypeError("top_k must be a positive integer")
    if inventory.status is ActionInventoryStatus.UNSUPPORTED:
        return _empty_proposal(
            inventory,
            top_k=top_k,
            model_identity=model_identity,
            probability_temperature=probability_temperature,
            status=ProposalStatus.UNSUPPORTED,
            detail=inventory.detail,
        )
    if inventory.status is ActionInventoryStatus.INVALID:
        return _empty_proposal(
            inventory,
            top_k=top_k,
            model_identity=model_identity,
            probability_temperature=probability_temperature,
            status=ProposalStatus.INVALID,
            detail=inventory.detail,
        )
    if inventory.status is ActionInventoryStatus.NO_LEGAL_ACTION:
        return _empty_proposal(
            inventory,
            top_k=top_k,
            model_identity=model_identity,
            probability_temperature=probability_temperature,
            status=ProposalStatus.NO_LEGAL_ACTION,
            detail=inventory.detail,
        )
    unknown = inventory.unknown_rule_ids
    if unknown:
        return _empty_proposal(
            inventory,
            top_k=top_k,
            model_identity=model_identity,
            probability_temperature=probability_temperature,
            status=ProposalStatus.UNSUPPORTED,
            detail="unregistered rules: " + ", ".join(unknown),
        )
    if inventory.legal_action_count == 0:
        return _empty_proposal(
            inventory,
            top_k=top_k,
            model_identity=model_identity,
            probability_temperature=probability_temperature,
            status=ProposalStatus.NO_LEGAL_ACTION,
            detail="the registry-derived legal mask contains no legal action",
        )
    try:
        scores = score()
    except UnsupportedPolicyStateError as error:
        return _empty_proposal(
            inventory,
            top_k=top_k,
            model_identity=model_identity,
            probability_temperature=probability_temperature,
            status=ProposalStatus.UNSUPPORTED,
            detail=str(error),
        )
    except InvalidPolicyStateError as error:
        return _empty_proposal(
            inventory,
            top_k=top_k,
            model_identity=model_identity,
            probability_temperature=probability_temperature,
            status=ProposalStatus.INVALID,
            detail=str(error),
        )
    logits = _tensor_to_floats(scores.logits, "logits")
    probabilities = _tensor_to_floats(scores.probabilities, "probabilities")
    if len(logits) != len(inventory.actions) or len(probabilities) != len(inventory.actions):
        raise RuntimeError("scorer output is not aligned with the action inventory")
    masked_probability = sum(
        probability
        for probability, legal in zip(probabilities, inventory.legal_mask, strict=True)
        if not legal
    )
    if abs(masked_probability) > 1e-12:
        raise RuntimeError("masked actions received nonzero probability mass")
    legal_indices = [index for index, legal in enumerate(inventory.legal_mask) if legal]
    for index in legal_indices:
        if not math.isfinite(logits[index]) or not math.isfinite(probabilities[index]):
            raise RuntimeError("legal action scores must be finite")
    selected = sorted(
        legal_indices,
        key=lambda index: (-logits[index], inventory.actions[index].action_digest),
    )[:top_k]
    candidates = tuple(
        ProposalCandidateV1(
            rank=rank,
            action=inventory.actions[index],
            logit=logits[index],
            probability=probabilities[index],
        )
        for rank, index in enumerate(selected, start=1)
    )
    return ProposalV1(
        current_signature=inventory.current_signature,
        goal_signature=inventory.goal_signature,
        candidates=candidates,
        legal_action_count=inventory.legal_action_count,
        requested_top_k=top_k,
        legal_mask_digest=inventory.legal_mask_digest,
        rule_registry_digest=inventory.vocabulary.registry_digest,
        model_identity=model_identity,
        probability_temperature=probability_temperature,
        status=ProposalStatus.SUCCESS,
        detail="ranked registered actions under the shared legal mask",
    )


def _empty_proposal(
    inventory: ActionInventoryV1,
    *,
    top_k: int,
    model_identity: ModelIdentityV1,
    probability_temperature: float,
    status: ProposalStatus,
    detail: str,
) -> ProposalV1:
    return ProposalV1(
        current_signature=inventory.current_signature,
        goal_signature=inventory.goal_signature,
        candidates=(),
        legal_action_count=inventory.legal_action_count,
        requested_top_k=top_k,
        legal_mask_digest=inventory.legal_mask_digest,
        rule_registry_digest=inventory.vocabulary.registry_digest,
        model_identity=model_identity,
        probability_temperature=probability_temperature,
        status=status,
        detail=detail,
    )


def _require_torch() -> None:
    if torch is None or nn is None:
        raise MissingMLDependencyError(
            "neural rewrite proposers require the optional 'ml' dependency group"
        )


def _raise_for_inventory_status(inventory: ActionInventoryV1) -> None:
    if inventory.status is ActionInventoryStatus.UNSUPPORTED:
        raise UnsupportedPolicyStateError(inventory.detail)
    if inventory.status is ActionInventoryStatus.INVALID:
        raise InvalidPolicyStateError(inventory.detail)


def _validate_encoding(encoding: object, role: str) -> None:
    if not isinstance(encoding, GraphEncodingProtocol):
        raise TypeError(f"{role} encoder output must implement GraphEncodingProtocol")


def _vector(value: object, width: int, name: str) -> object:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1 or value.shape[0] != width:
        raise InvalidPolicyStateError(f"{name} must have shape [{width}]")
    if not torch.isfinite(value).all():
        raise InvalidPolicyStateError(f"{name} must contain only finite values")
    return value


def _argument_matrix(value: object, width: int, maximum: int) -> object:
    if not isinstance(value, torch.Tensor):
        raise TypeError("ordered argument embeddings must be a torch.Tensor")
    if value.ndim != 2 or value.shape[1] != width:
        raise InvalidPolicyStateError(
            f"ordered argument embeddings must have shape [count, {width}]"
        )
    if value.shape[0] > maximum:
        raise InvalidPolicyStateError(
            f"action has {value.shape[0]} arguments but maximum is {maximum}"
        )
    if not torch.isfinite(value).all():
        raise InvalidPolicyStateError("ordered argument embeddings must contain only finite values")
    return value


def _masked_softmax(logits: object, legal_mask: object) -> object:
    if logits.ndim != 1 or legal_mask.shape != logits.shape:
        raise RuntimeError("logits and legal mask must be aligned vectors")
    if not bool(legal_mask.any()):
        return torch.zeros_like(logits)
    masked_logits = logits.masked_fill(~legal_mask, -torch.inf)
    probabilities = torch.softmax(masked_logits, dim=0)
    return torch.where(legal_mask, probabilities, torch.zeros_like(probabilities))


def _tensor_to_floats(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise RuntimeError(f"{name} must be a one-dimensional torch tensor")
    return tuple(float(item) for item in value.detach().cpu().tolist())
