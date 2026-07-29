"""Verifier-grounded metrics for goal-conditioned rewrite proposals.

This module deliberately does not import the concurrently developed Goal 7
record or proposal implementations.  The small protocols below are the merge
boundary.  Transition semantics are injected through :class:`StepMetricAdapter`
so this module never creates a second rule registry or verifier.

Three questions remain separate throughout evaluation:

* did a proposal imitate the stored demonstration action exactly?
* did replay produce the exact stored successor structure?
* was the proposed transition legal, replayable, and verifier-accepted?

The final question is a safety metric.  A different verifier-valid equality
rewrite is not silently relabelled as demonstration correctness.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

STEP_METRIC_SCHEMA_VERSION = "geml-step-metric-outcome-v1"
STEP_METRIC_AGGREGATE_SCHEMA_VERSION = "geml-step-metric-aggregate-v1"
FAMILY_PARTITION_EVIDENCE_SCHEMA_VERSION = "geml-family-partition-evidence-v1"
DEFAULT_TOP_KS = (1, 3, 5)
_FAMILY_INVENTORY_DOMAIN = b"geml-training-family-inventory-v1\0"


class FamilyGeneralization(StrEnum):
    """Whether absence from training has been established for the example family."""

    SEEN = "seen"
    HELD_OUT = "held_out"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FamilyPartitionEvidenceV1:
    """Authenticated training-family inventory used to derive held-out labels.

    The manifest digest binds this inventory to the frozen step dataset at the
    Goal 7 integration boundary. Callers cannot directly assert ``held_out``:
    the label and unseen input roles are derived from the inventory.
    """

    schema_version: str
    step_manifest_digest: str
    training_family_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != FAMILY_PARTITION_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported family-partition evidence schema")
        _sha256_digest(self.step_manifest_digest, field="step_manifest_digest")
        families = _string_tuple(
            self.training_family_ids,
            field="training_family_ids",
            sorted_unique=True,
        )
        if not families:
            raise ValueError("training_family_ids cannot be empty")

    @property
    def inventory_digest(self) -> str:
        payload = json.dumps(
            list(self.training_family_ids),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(_FAMILY_INVENTORY_DOMAIN + payload).hexdigest()

    def classify(
        self,
        *,
        current_family: str,
        goal_family: str,
    ) -> tuple[FamilyGeneralization, tuple[str, ...]]:
        """Derive generalization status and unseen goal-conditioned roles."""

        training = set(self.training_family_ids)
        unseen_roles = tuple(
            role
            for role, family in (
                ("current", current_family),
                ("goal", goal_family),
            )
            if family not in training
        )
        return (
            FamilyGeneralization.HELD_OUT if unseen_roles else FamilyGeneralization.SEEN,
            unseen_roles,
        )


@dataclass(frozen=True, slots=True)
class _FamilyEvidenceSnapshot:
    generalization: FamilyGeneralization
    manifest_digest: str | None
    inventory_digest: str | None
    unseen_roles: tuple[str, ...]


class LegalityStatus(StrEnum):
    """Registry-derived classification of a concrete action in the current state."""

    LEGAL = "legal"
    INVALID_ACTION = "invalid_action"
    INVALID_SITE = "invalid_site"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNSUPPORTED = "unsupported"


class ReplayStatus(StrEnum):
    """Outcome of applying a concrete action to the current state."""

    SUCCEEDED = "succeeded"
    INVALID_ACTION = "invalid_action"
    INVALID_SITE = "invalid_site"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class VerificationStatus(StrEnum):
    """Outcome returned by the authoritative transition verifier."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNSUPPORTED = "unsupported"


class CandidateMetricStatus(StrEnum):
    """Terminal accounting status for one ranked candidate."""

    VERIFIED_VALID = "verified_valid"
    VERIFIER_REJECTED = "verifier_rejected"
    INVALID_ACTION = "invalid_action"
    INVALID_SITE = "invalid_site"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNSUPPORTED = "unsupported"
    REPLAY_ERROR = "replay_error"
    VERIFIER_TIMEOUT = "verifier_timeout"
    VERIFIER_ERROR = "verifier_error"
    PARSE_SCHEMA_ERROR = "parse_schema_error"


class ExampleMetricStatus(StrEnum):
    """Accounting status for one supervised example."""

    EVALUATED = "evaluated"
    NO_LEGAL_ACTION = "no_legal_action"
    NO_PROPOSAL = "no_proposal"
    UNSUPPORTED = "unsupported"
    INVALID_PROPOSAL = "invalid_proposal"
    PARSE_SCHEMA_ERROR = "parse_schema_error"


@runtime_checkable
class StepRecordProtocol(Protocol):
    """Read-only subset of issue #61's accepted step record."""

    record_id: str
    trace_id: str
    source_group: str
    lineage_group_ids: tuple[str, ...]
    authoritative_split: object
    current_state: object
    current_signature: str
    goal_signature: str
    next_signature: str
    rule_id: str
    direction: str
    occurrence_path: tuple[int, ...]
    ordered_arguments: tuple[object, ...]
    action_digest: str
    current_family: str
    goal_family: str
    evaluation_views: tuple[str, ...]
    remaining_witness_steps: int
    trace_length: int
    rule_set_digest: str
    supported: bool


@runtime_checkable
class ProposalCandidateProtocol(Protocol):
    """Read-only subset of one issue #62 ranked candidate."""

    rank: int
    action: object


@runtime_checkable
class ProposalProtocol(Protocol):
    """Read-only subset of issue #62's typed proposal."""

    current_signature: str
    goal_signature: str
    candidates: tuple[ProposalCandidateProtocol, ...]
    legal_action_count: int
    requested_top_k: int
    legal_mask_digest: str
    rule_registry_digest: str
    status: object


@runtime_checkable
class StepMetricAdapter(Protocol):
    """Injected bridge to action identity, registry replay, and verification."""

    def action_identity(self, action: object) -> ActionIdentityV1:
        """Return the non-lossy scientific identity of a proposed action."""

    def classify_legality(
        self,
        record: StepRecordProtocol,
        action: object,
    ) -> LegalityResultV1:
        """Classify the action against the frozen registry and current state."""

    def replay(
        self,
        record: StepRecordProtocol,
        action: object,
    ) -> ReplayResultV1:
        """Apply the action once without repairing or canonicalizing the result."""

    def structural_signature(self, successor_state: object) -> str:
        """Derive the canonical structural signature from a replayed state."""

    def verify(
        self,
        record: StepRecordProtocol,
        action: object,
        replay: ReplayResultV1,
    ) -> VerificationResultV1:
        """Verify the replayed transition under the record's assumptions."""


def _nonblank(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    return value


def _sha256_digest(value: object, *, field: str) -> str:
    digest = _nonblank(value, field=field)
    if (
        len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256 digest")
    return digest


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    result = _nonnegative_integer(value, field=field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _string_tuple(value: object, *, field: str, sorted_unique: bool = False) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field} must be a tuple")
    result = tuple(_nonblank(item, field=f"{field} item") for item in value)
    if sorted_unique and result != tuple(sorted(set(result))):
        raise ValueError(f"{field} must be sorted and unique")
    return result


def _directed_rule_tuple(
    value: object,
    *,
    field: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field} must be a tuple")
    result: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{field} entries must be (rule_id, direction) tuples")
        rule_id = _nonblank(item[0], field=f"{field} rule_id")
        direction = _nonblank(item[1], field=f"{field} direction")
        if direction not in {"backward", "forward"}:
            raise ValueError(f"{field} direction must be backward or forward")
        result.append((rule_id, direction))
    frozen = tuple(result)
    if frozen != tuple(sorted(set(frozen))) or not frozen:
        raise ValueError(f"{field} must be nonempty, sorted, and unique")
    return frozen


def _canonical_json(value: object) -> str:
    """Return a strict canonical JSON rendering for one ordered argument."""

    def render_json(item: object) -> str:
        return json.dumps(
            item,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    # Issue #61 wraps persisted scientific values in a CanonicalJson object.
    # Normalize its decoded value with the same metric-local rendering used for
    # #62's raw JSON view; this avoids coupling equality to an escape style.
    wrapped_text = getattr(value, "text", None)
    to_value = getattr(value, "to_value", None)
    if isinstance(wrapped_text, str) and callable(to_value):
        try:
            decoded = json.loads(wrapped_text)
            unwrapped = to_value()
            decoded_text = render_json(decoded)
            if render_json(unwrapped) != decoded_text:
                raise ValueError("CanonicalJson.text and to_value() disagree")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("CanonicalJson does not contain a finite JSON value") from error
        return decoded_text
    try:
        rendered = render_json(value)
    except (TypeError, ValueError) as error:
        raise ValueError("ordered arguments must contain canonical JSON values") from error
    return rendered


def canonical_ordered_arguments(value: object) -> tuple[str, ...]:
    """Preserve argument order while canonicalizing each argument without coercion."""

    if not isinstance(value, tuple):
        raise TypeError("ordered_arguments must be a tuple")
    return tuple(_canonical_json(argument) for argument in value)


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


@dataclass(frozen=True, slots=True)
class ActionIdentityV1:
    """Exact demonstration-matching fields plus the upstream action digest."""

    rule_id: str
    direction: str
    occurrence_path: tuple[int, ...]
    ordered_arguments_json: tuple[str, ...]
    action_digest: str

    def __post_init__(self) -> None:
        _nonblank(self.rule_id, field="rule_id")
        _nonblank(self.direction, field="direction")
        _sha256_digest(self.action_digest, field="action_digest")
        if not isinstance(self.occurrence_path, tuple) or any(
            isinstance(slot, bool) or not isinstance(slot, int) or slot < 0
            for slot in self.occurrence_path
        ):
            raise ValueError("occurrence_path must be a tuple of nonnegative child slots")
        _string_tuple(self.ordered_arguments_json, field="ordered_arguments_json")
        for value in self.ordered_arguments_json:
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError("ordered argument is not valid JSON") from error
            if _canonical_json(decoded) != value:
                raise ValueError("ordered argument JSON must be canonical")

    @property
    def demonstration_key(self) -> tuple[str, str, tuple[int, ...], tuple[str, ...]]:
        """Fields defining exact imitation of the stored demonstration."""

        return (
            self.rule_id,
            self.direction,
            self.occurrence_path,
            self.ordered_arguments_json,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "action_digest": self.action_digest,
            "direction": self.direction,
            "occurrence_path": list(self.occurrence_path),
            "ordered_arguments_json": list(self.ordered_arguments_json),
            "rule_id": self.rule_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> ActionIdentityV1:
        if not isinstance(value, dict) or set(value) != {
            "action_digest",
            "direction",
            "occurrence_path",
            "ordered_arguments_json",
            "rule_id",
        }:
            raise ValueError("action identity fields are incompatible")
        path = value["occurrence_path"]
        arguments = value["ordered_arguments_json"]
        if not isinstance(path, list) or not isinstance(arguments, list):
            raise ValueError("action path and arguments must be JSON arrays")
        return cls(
            rule_id=value["rule_id"],
            direction=value["direction"],
            occurrence_path=tuple(path),
            ordered_arguments_json=tuple(arguments),
            action_digest=value["action_digest"],
        )


@dataclass(frozen=True, slots=True)
class LegalityResultV1:
    """Typed registry legality result."""

    status: LegalityStatus
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, LegalityStatus):
            raise TypeError("legality status must be a LegalityStatus")
        _nonblank(self.detail, field="legality detail")


@dataclass(frozen=True, slots=True)
class ReplayResultV1:
    """Typed replay result; the concrete state is transient verifier input."""

    status: ReplayStatus
    successor_signature: str | None
    successor_state: object | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ReplayStatus):
            raise TypeError("replay status must be a ReplayStatus")
        _nonblank(self.detail, field="replay detail")
        if self.status is ReplayStatus.SUCCEEDED:
            _sha256_digest(
                self.successor_signature,
                field="replay successor_signature",
            )
            if self.successor_state is None:
                raise ValueError("successful replay requires the concrete successor state")
        elif self.successor_signature is not None or self.successor_state is not None:
            raise ValueError("failed replay cannot claim a successor")


@dataclass(frozen=True, slots=True)
class VerificationResultV1:
    """Typed result from the authoritative transition verifier."""

    status: VerificationStatus
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, VerificationStatus):
            raise TypeError("verification status must be a VerificationStatus")
        _nonblank(self.detail, field="verification detail")


_LEGALITY_TO_CANDIDATE_STATUS = {
    LegalityStatus.INVALID_ACTION: CandidateMetricStatus.INVALID_ACTION,
    LegalityStatus.INVALID_SITE: CandidateMetricStatus.INVALID_SITE,
    LegalityStatus.INVALID_ARGUMENTS: CandidateMetricStatus.INVALID_ARGUMENTS,
    LegalityStatus.UNSUPPORTED: CandidateMetricStatus.UNSUPPORTED,
}
_REPLAY_TO_CANDIDATE_STATUS = {
    ReplayStatus.INVALID_ACTION: CandidateMetricStatus.INVALID_ACTION,
    ReplayStatus.INVALID_SITE: CandidateMetricStatus.INVALID_SITE,
    ReplayStatus.INVALID_ARGUMENTS: CandidateMetricStatus.INVALID_ARGUMENTS,
    ReplayStatus.UNSUPPORTED: CandidateMetricStatus.UNSUPPORTED,
    ReplayStatus.ERROR: CandidateMetricStatus.REPLAY_ERROR,
}
_VERIFICATION_TO_CANDIDATE_STATUS = {
    VerificationStatus.ACCEPTED: CandidateMetricStatus.VERIFIED_VALID,
    VerificationStatus.REJECTED: CandidateMetricStatus.VERIFIER_REJECTED,
    VerificationStatus.TIMEOUT: CandidateMetricStatus.VERIFIER_TIMEOUT,
    VerificationStatus.ERROR: CandidateMetricStatus.VERIFIER_ERROR,
    VerificationStatus.UNSUPPORTED: CandidateMetricStatus.UNSUPPORTED,
}


@dataclass(frozen=True, slots=True)
class CandidateMetricOutcomeV1:
    """Persisted non-collapsing outcome for one ranked proposal."""

    rank: int
    action: ActionIdentityV1 | None
    status: CandidateMetricStatus
    exact_demonstration_action: bool
    exact_successor_structure: bool
    verifier_confirmed_valid: bool
    successor_signature: str | None
    legality_status: LegalityStatus | None
    replay_status: ReplayStatus | None
    verifier_status: VerificationStatus | None
    legality_detail: str | None
    replay_detail: str | None
    verifier_detail: str | None
    parse_error: str | None

    def __post_init__(self) -> None:
        _positive_integer(self.rank, field="candidate rank")
        if not isinstance(self.status, CandidateMetricStatus):
            raise TypeError("candidate status must be a CandidateMetricStatus")
        for name in (
            "exact_demonstration_action",
            "exact_successor_structure",
            "verifier_confirmed_valid",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if self.action is None:
            if self.status is not CandidateMetricStatus.PARSE_SCHEMA_ERROR:
                raise ValueError("only a parse/schema failure may lack action identity")
            if self.exact_demonstration_action:
                raise ValueError("an unparsed action cannot match the demonstration")
        if self.status is CandidateMetricStatus.PARSE_SCHEMA_ERROR:
            if self.parse_error is None:
                raise ValueError("parse/schema status requires its retained error")
        elif self.parse_error is not None:
            raise ValueError("only parse/schema status may retain a parse error")
        if self.successor_signature is not None:
            _sha256_digest(
                self.successor_signature,
                field="candidate successor_signature",
            )
        if self.exact_successor_structure and self.replay_status is not ReplayStatus.SUCCEEDED:
            raise ValueError("exact successor structure requires successful replay")
        if self.verifier_confirmed_valid != (self.verifier_status is VerificationStatus.ACCEPTED):
            raise ValueError("verifier validity must be exactly the accepted verifier status")
        if (
            self.status is CandidateMetricStatus.VERIFIED_VALID
            and not self.verifier_confirmed_valid
        ):
            raise ValueError("verified-valid status requires verifier acceptance")
        if self.verifier_status is not None and self.replay_status is not ReplayStatus.SUCCEEDED:
            raise ValueError("verification may run only after successful replay")
        if self.replay_status is not None and self.legality_status is not LegalityStatus.LEGAL:
            raise ValueError("replay may run only after a legal classification")
        if self.legality_status is not None and self.legality_detail is None:
            raise ValueError("a legality result requires retained detail")
        if self.replay_status is not None and self.replay_detail is None:
            raise ValueError("a replay result requires retained detail")
        if self.verifier_status is not None and self.verifier_detail is None:
            raise ValueError("a verifier result requires retained detail")
        if self.replay_status is ReplayStatus.SUCCEEDED and self.successor_signature is None:
            raise ValueError("successful replay requires a persisted successor signature")
        if (
            self.replay_status is not ReplayStatus.SUCCEEDED
            and self.successor_signature is not None
        ):
            raise ValueError("only successful replay may persist a successor signature")
        if self.status is not CandidateMetricStatus.PARSE_SCHEMA_ERROR:
            if self.legality_status is None:
                raise ValueError("non-parse candidates require a legality result")
            if self.legality_status is not LegalityStatus.LEGAL:
                expected_status = _LEGALITY_TO_CANDIDATE_STATUS[self.legality_status]
            elif self.replay_status is None:
                raise ValueError("legal candidates require a replay result")
            elif self.replay_status is not ReplayStatus.SUCCEEDED:
                expected_status = _REPLAY_TO_CANDIDATE_STATUS[self.replay_status]
            elif self.verifier_status is None:
                raise ValueError("successfully replayed candidates require verifier evidence")
            else:
                expected_status = _VERIFICATION_TO_CANDIDATE_STATUS[self.verifier_status]
            if self.status is not expected_status:
                raise ValueError("candidate status disagrees with its retained phase evidence")
        if self.parse_error is not None:
            _nonblank(self.parse_error, field="parse_error")
        for name in ("legality_detail", "replay_detail", "verifier_detail"):
            detail = getattr(self, name)
            if detail is not None:
                _nonblank(detail, field=name)

    def as_dict(self) -> dict[str, object]:
        return {
            "action": None if self.action is None else self.action.as_dict(),
            "exact_demonstration_action": self.exact_demonstration_action,
            "exact_successor_structure": self.exact_successor_structure,
            "legality_detail": self.legality_detail,
            "legality_status": None if self.legality_status is None else self.legality_status.value,
            "parse_error": self.parse_error,
            "rank": self.rank,
            "replay_detail": self.replay_detail,
            "replay_status": None if self.replay_status is None else self.replay_status.value,
            "status": self.status.value,
            "successor_signature": self.successor_signature,
            "verifier_confirmed_valid": self.verifier_confirmed_valid,
            "verifier_detail": self.verifier_detail,
            "verifier_status": None if self.verifier_status is None else self.verifier_status.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> CandidateMetricOutcomeV1:
        expected = {
            "action",
            "exact_demonstration_action",
            "exact_successor_structure",
            "legality_detail",
            "legality_status",
            "parse_error",
            "rank",
            "replay_detail",
            "replay_status",
            "status",
            "successor_signature",
            "verifier_confirmed_valid",
            "verifier_detail",
            "verifier_status",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("candidate metric fields are incompatible")
        action = value["action"]
        return cls(
            rank=value["rank"],
            action=None if action is None else ActionIdentityV1.from_dict(action),
            status=CandidateMetricStatus(value["status"]),
            exact_demonstration_action=value["exact_demonstration_action"],
            exact_successor_structure=value["exact_successor_structure"],
            verifier_confirmed_valid=value["verifier_confirmed_valid"],
            successor_signature=value["successor_signature"],
            legality_status=None
            if value["legality_status"] is None
            else LegalityStatus(value["legality_status"]),
            replay_status=None
            if value["replay_status"] is None
            else ReplayStatus(value["replay_status"]),
            verifier_status=None
            if value["verifier_status"] is None
            else VerificationStatus(value["verifier_status"]),
            legality_detail=value["legality_detail"],
            replay_detail=value["replay_detail"],
            verifier_detail=value["verifier_detail"],
            parse_error=value["parse_error"],
        )


@dataclass(frozen=True, slots=True)
class StepTopKOutcomeV1:
    """Counts for one example at one requested cutoff."""

    k: int
    candidate_attempts: int
    verifier_attempts: int
    verifier_resolved: int
    verifier_valid_candidates: int
    demonstration_action_match: bool
    exact_successor_structure_match: bool
    verifier_valid_success: bool

    def __post_init__(self) -> None:
        _positive_integer(self.k, field="k")
        for name in (
            "candidate_attempts",
            "verifier_attempts",
            "verifier_resolved",
            "verifier_valid_candidates",
        ):
            _nonnegative_integer(getattr(self, name), field=name)
        if not (
            self.verifier_valid_candidates
            <= self.verifier_resolved
            <= self.verifier_attempts
            <= self.candidate_attempts
            <= self.k
        ):
            raise ValueError("top-k candidate/verifier denominators are inconsistent")
        for name in (
            "demonstration_action_match",
            "exact_successor_structure_match",
            "verifier_valid_success",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if self.verifier_valid_success != (self.verifier_valid_candidates > 0):
            raise ValueError("verifier-valid success must reflect accepted candidate count")


@dataclass(frozen=True, slots=True)
class StepMetricOutcomeV1:
    """Complete persisted evidence for one accepted step record."""

    schema_version: str
    record_id: str
    trace_id: str
    source_group: str
    lineage_group_ids: tuple[str, ...]
    authoritative_split: str
    current_signature: str
    goal_signature: str
    target_successor_signature: str
    current_family: str
    goal_family: str
    evaluation_views: tuple[str, ...]
    family_generalization: FamilyGeneralization
    family_evidence_manifest_digest: str | None
    training_family_inventory_digest: str | None
    unseen_family_roles: tuple[str, ...]
    remaining_witness_steps: int
    trace_length: int
    demonstration_action: ActionIdentityV1 | None
    registered_rule_ids: tuple[str, ...]
    registered_rule_directions: tuple[tuple[str, str], ...]
    rule_registry_digest: str
    legal_mask_digest: str | None
    requested_top_ks: tuple[int, ...]
    legal_action_count: int | None
    proposal_candidate_count: int
    status: ExampleMetricStatus
    detail: str
    candidates: tuple[CandidateMetricOutcomeV1, ...]

    def __post_init__(self) -> None:
        if self.schema_version != STEP_METRIC_SCHEMA_VERSION:
            raise ValueError("unsupported step metric schema version")
        for name in (
            "trace_id",
            "source_group",
            "authoritative_split",
            "current_family",
            "goal_family",
            "detail",
        ):
            _nonblank(getattr(self, name), field=name)
        for name in (
            "record_id",
            "current_signature",
            "goal_signature",
            "target_successor_signature",
            "rule_registry_digest",
        ):
            _sha256_digest(getattr(self, name), field=name)
        if self.legal_mask_digest is not None:
            _sha256_digest(self.legal_mask_digest, field="legal_mask_digest")
        lineage = _string_tuple(
            self.lineage_group_ids,
            field="lineage_group_ids",
            sorted_unique=True,
        )
        if self.source_group not in lineage:
            raise ValueError("lineage_group_ids must include source_group")
        _string_tuple(self.evaluation_views, field="evaluation_views", sorted_unique=True)
        if not isinstance(self.family_generalization, FamilyGeneralization):
            raise TypeError("family_generalization must be a FamilyGeneralization")
        unseen_roles = _string_tuple(
            self.unseen_family_roles,
            field="unseen_family_roles",
            sorted_unique=True,
        )
        if any(role not in {"current", "goal"} for role in unseen_roles):
            raise ValueError("unseen_family_roles may contain only current and goal")
        family_evidence_digests = (
            self.family_evidence_manifest_digest,
            self.training_family_inventory_digest,
        )
        if self.family_generalization is FamilyGeneralization.UNKNOWN:
            if any(value is not None for value in family_evidence_digests) or unseen_roles:
                raise ValueError("unknown family status cannot claim partition evidence")
        else:
            for name, value in zip(
                (
                    "family_evidence_manifest_digest",
                    "training_family_inventory_digest",
                ),
                family_evidence_digests,
                strict=True,
            ):
                _sha256_digest(value, field=name)
            if (self.family_generalization is FamilyGeneralization.HELD_OUT) != bool(unseen_roles):
                raise ValueError("held-out family status must exactly reflect unseen roles")
        _positive_integer(self.remaining_witness_steps, field="remaining_witness_steps")
        _positive_integer(self.trace_length, field="trace_length")
        if self.remaining_witness_steps > self.trace_length:
            raise ValueError("remaining witness length cannot exceed trace length")
        _string_tuple(self.registered_rule_ids, field="registered_rule_ids", sorted_unique=True)
        if not self.registered_rule_ids:
            raise ValueError("registered_rule_ids cannot be empty")
        directed_rules = _directed_rule_tuple(
            self.registered_rule_directions,
            field="registered_rule_directions",
        )
        if {rule_id for rule_id, _ in directed_rules} != set(self.registered_rule_ids):
            raise ValueError(
                "registered_rule_ids must exactly match the directed registry inventory"
            )
        if (
            not isinstance(self.requested_top_ks, tuple)
            or not self.requested_top_ks
            or self.requested_top_ks != tuple(sorted(set(self.requested_top_ks)))
        ):
            raise ValueError("requested_top_ks must be a nonempty sorted unique tuple")
        for k in self.requested_top_ks:
            _positive_integer(k, field="requested top-k")
        if self.legal_action_count is not None:
            _nonnegative_integer(self.legal_action_count, field="legal_action_count")
            if self.legal_mask_digest is None:
                raise ValueError("a known legal-action count requires legal-mask provenance")
        _nonnegative_integer(self.proposal_candidate_count, field="proposal_candidate_count")
        if self.proposal_candidate_count != len(self.candidates):
            raise ValueError("proposal_candidate_count must equal persisted candidates")
        if tuple(candidate.rank for candidate in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("candidate ranks must be consecutive from one")
        if not isinstance(self.status, ExampleMetricStatus):
            raise TypeError("status must be an ExampleMetricStatus")
        if self.status is ExampleMetricStatus.EVALUATED and not self.candidates:
            raise ValueError("evaluated rows require at least one candidate")
        if self.status is ExampleMetricStatus.EVALUATED and self.legal_mask_digest is None:
            raise ValueError("evaluated rows require legal-mask provenance")
        if self.status is ExampleMetricStatus.EVALUATED:
            if self.legal_action_count is None:
                raise ValueError("evaluated rows require the legal-action denominator")
            expected_candidates = min(max(self.requested_top_ks), self.legal_action_count)
            if len(self.candidates) != expected_candidates:
                raise ValueError(
                    "evaluated rows require a complete ranking through the largest cutoff"
                )
        if self.status is not ExampleMetricStatus.EVALUATED and self.candidates:
            raise ValueError("non-evaluated rows cannot retain ranked candidates")
        if self.status is ExampleMetricStatus.NO_LEGAL_ACTION and self.legal_action_count != 0:
            raise ValueError("no-legal-action rows require a zero legal-action count")
        if (
            self.demonstration_action is None
            and self.status is not ExampleMetricStatus.PARSE_SCHEMA_ERROR
        ):
            raise ValueError("only a parse/schema row may lack demonstration identity")
        seen_actions: set[tuple[str, str, tuple[int, ...], tuple[str, ...]]] = set()
        for candidate in self.candidates:
            expected_action_match = (
                self.demonstration_action is not None
                and candidate.action is not None
                and candidate.action.demonstration_key
                == self.demonstration_action.demonstration_key
            )
            if candidate.exact_demonstration_action != expected_action_match:
                raise ValueError(
                    "candidate exact-action label disagrees with persisted action identity"
                )
            expected_successor_match = (
                candidate.successor_signature is not None
                and candidate.successor_signature == self.target_successor_signature
            )
            if candidate.exact_successor_structure != expected_successor_match:
                raise ValueError(
                    "candidate exact-successor label disagrees with persisted signatures"
                )
            if candidate.action is not None:
                action_key = candidate.action.demonstration_key
                if action_key in seen_actions:
                    raise ValueError("persisted ranking contains a duplicate concrete action")
                seen_actions.add(action_key)

    @property
    def demonstrated_rule_id(self) -> str | None:
        return None if self.demonstration_action is None else self.demonstration_action.rule_id

    def at_k(self, k: int) -> StepTopKOutcomeV1:
        """Reconstruct this row's outcomes at ``k`` from candidate evidence."""

        _positive_integer(k, field="k")
        candidates = self.candidates[:k]
        verifier_attempts = sum(item.verifier_status is not None for item in candidates)
        verifier_resolved = sum(
            item.verifier_status in {VerificationStatus.ACCEPTED, VerificationStatus.REJECTED}
            for item in candidates
        )
        verifier_valid = sum(item.verifier_confirmed_valid for item in candidates)
        return StepTopKOutcomeV1(
            k=k,
            candidate_attempts=len(candidates),
            verifier_attempts=verifier_attempts,
            verifier_resolved=verifier_resolved,
            verifier_valid_candidates=verifier_valid,
            demonstration_action_match=any(item.exact_demonstration_action for item in candidates),
            exact_successor_structure_match=any(
                item.exact_successor_structure for item in candidates
            ),
            verifier_valid_success=verifier_valid > 0,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "authoritative_split": self.authoritative_split,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "current_family": self.current_family,
            "current_signature": self.current_signature,
            "demonstration_action": None
            if self.demonstration_action is None
            else self.demonstration_action.as_dict(),
            "detail": self.detail,
            "evaluation_views": list(self.evaluation_views),
            "family_evidence_manifest_digest": self.family_evidence_manifest_digest,
            "family_generalization": self.family_generalization.value,
            "goal_family": self.goal_family,
            "goal_signature": self.goal_signature,
            "legal_action_count": self.legal_action_count,
            "legal_mask_digest": self.legal_mask_digest,
            "lineage_group_ids": list(self.lineage_group_ids),
            "proposal_candidate_count": self.proposal_candidate_count,
            "record_id": self.record_id,
            "registered_rule_directions": [
                list(value) for value in self.registered_rule_directions
            ],
            "registered_rule_ids": list(self.registered_rule_ids),
            "remaining_witness_steps": self.remaining_witness_steps,
            "requested_top_ks": list(self.requested_top_ks),
            "rule_registry_digest": self.rule_registry_digest,
            "schema_version": self.schema_version,
            "source_group": self.source_group,
            "status": self.status.value,
            "target_successor_signature": self.target_successor_signature,
            "trace_id": self.trace_id,
            "trace_length": self.trace_length,
            "training_family_inventory_digest": self.training_family_inventory_digest,
            "unseen_family_roles": list(self.unseen_family_roles),
        }

    @classmethod
    def from_dict(cls, value: object) -> StepMetricOutcomeV1:
        expected = {
            "authoritative_split",
            "candidates",
            "current_family",
            "current_signature",
            "demonstration_action",
            "detail",
            "evaluation_views",
            "family_evidence_manifest_digest",
            "family_generalization",
            "goal_family",
            "goal_signature",
            "legal_action_count",
            "legal_mask_digest",
            "lineage_group_ids",
            "proposal_candidate_count",
            "record_id",
            "registered_rule_directions",
            "registered_rule_ids",
            "remaining_witness_steps",
            "requested_top_ks",
            "rule_registry_digest",
            "schema_version",
            "source_group",
            "status",
            "target_successor_signature",
            "trace_id",
            "trace_length",
            "training_family_inventory_digest",
            "unseen_family_roles",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("step metric fields are incompatible")
        list_fields = (
            "candidates",
            "evaluation_views",
            "lineage_group_ids",
            "registered_rule_directions",
            "registered_rule_ids",
            "requested_top_ks",
            "unseen_family_roles",
        )
        if any(not isinstance(value[name], list) for name in list_fields):
            raise ValueError("step metric tuple fields must be JSON arrays")
        demonstration = value["demonstration_action"]
        return cls(
            schema_version=value["schema_version"],
            record_id=value["record_id"],
            trace_id=value["trace_id"],
            source_group=value["source_group"],
            lineage_group_ids=tuple(value["lineage_group_ids"]),
            authoritative_split=value["authoritative_split"],
            current_signature=value["current_signature"],
            goal_signature=value["goal_signature"],
            target_successor_signature=value["target_successor_signature"],
            current_family=value["current_family"],
            goal_family=value["goal_family"],
            evaluation_views=tuple(value["evaluation_views"]),
            family_generalization=FamilyGeneralization(value["family_generalization"]),
            family_evidence_manifest_digest=value["family_evidence_manifest_digest"],
            training_family_inventory_digest=value["training_family_inventory_digest"],
            unseen_family_roles=tuple(value["unseen_family_roles"]),
            remaining_witness_steps=value["remaining_witness_steps"],
            trace_length=value["trace_length"],
            demonstration_action=None
            if demonstration is None
            else ActionIdentityV1.from_dict(demonstration),
            registered_rule_directions=tuple(
                tuple(item) for item in value["registered_rule_directions"]
            ),
            registered_rule_ids=tuple(value["registered_rule_ids"]),
            rule_registry_digest=value["rule_registry_digest"],
            legal_mask_digest=value["legal_mask_digest"],
            requested_top_ks=tuple(value["requested_top_ks"]),
            legal_action_count=value["legal_action_count"],
            proposal_candidate_count=value["proposal_candidate_count"],
            status=ExampleMetricStatus(value["status"]),
            detail=value["detail"],
            candidates=tuple(
                CandidateMetricOutcomeV1.from_dict(candidate) for candidate in value["candidates"]
            ),
        )


def _record_action_identity(record: StepRecordProtocol) -> ActionIdentityV1:
    return ActionIdentityV1(
        rule_id=record.rule_id,
        direction=record.direction,
        occurrence_path=record.occurrence_path,
        ordered_arguments_json=canonical_ordered_arguments(record.ordered_arguments),
        action_digest=record.action_digest,
    )


def _record_authoritative_split(record: StepRecordProtocol) -> str:
    raw_split = record.authoritative_split
    value = raw_split.value if isinstance(raw_split, StrEnum) else raw_split
    return _nonblank(value, field="record.authoritative_split")


def _family_evidence_snapshot(
    record: StepRecordProtocol,
    evidence: FamilyPartitionEvidenceV1 | None,
) -> _FamilyEvidenceSnapshot:
    if evidence is None:
        return _FamilyEvidenceSnapshot(
            generalization=FamilyGeneralization.UNKNOWN,
            manifest_digest=None,
            inventory_digest=None,
            unseen_roles=(),
        )
    if not isinstance(evidence, FamilyPartitionEvidenceV1):
        raise TypeError("family_partition_evidence must be FamilyPartitionEvidenceV1 or None")
    generalization, unseen_roles = evidence.classify(
        current_family=record.current_family,
        goal_family=record.goal_family,
    )
    return _FamilyEvidenceSnapshot(
        generalization=generalization,
        manifest_digest=evidence.step_manifest_digest,
        inventory_digest=evidence.inventory_digest,
        unseen_roles=unseen_roles,
    )


def _outcome_without_candidates(
    record: StepRecordProtocol,
    *,
    demonstration_action: ActionIdentityV1 | None,
    registered_rule_ids: tuple[str, ...],
    registered_rule_directions: tuple[tuple[str, str], ...],
    rule_registry_digest: str,
    top_ks: tuple[int, ...],
    family_evidence: _FamilyEvidenceSnapshot,
    status: ExampleMetricStatus,
    detail: str,
    legal_action_count: int | None,
    legal_mask_digest: str | None = None,
) -> StepMetricOutcomeV1:
    return StepMetricOutcomeV1(
        schema_version=STEP_METRIC_SCHEMA_VERSION,
        record_id=record.record_id,
        trace_id=record.trace_id,
        source_group=record.source_group,
        lineage_group_ids=record.lineage_group_ids,
        authoritative_split=_record_authoritative_split(record),
        current_signature=record.current_signature,
        goal_signature=record.goal_signature,
        target_successor_signature=record.next_signature,
        current_family=record.current_family,
        goal_family=record.goal_family,
        evaluation_views=record.evaluation_views,
        family_generalization=family_evidence.generalization,
        family_evidence_manifest_digest=family_evidence.manifest_digest,
        training_family_inventory_digest=family_evidence.inventory_digest,
        unseen_family_roles=family_evidence.unseen_roles,
        remaining_witness_steps=record.remaining_witness_steps,
        trace_length=record.trace_length,
        demonstration_action=demonstration_action,
        registered_rule_ids=registered_rule_ids,
        registered_rule_directions=registered_rule_directions,
        rule_registry_digest=rule_registry_digest,
        legal_mask_digest=legal_mask_digest,
        requested_top_ks=top_ks,
        legal_action_count=legal_action_count,
        proposal_candidate_count=0,
        status=status,
        detail=detail,
        candidates=(),
    )


def _parse_candidate_failure(rank: int, error: Exception) -> CandidateMetricOutcomeV1:
    return CandidateMetricOutcomeV1(
        rank=rank,
        action=None,
        status=CandidateMetricStatus.PARSE_SCHEMA_ERROR,
        exact_demonstration_action=False,
        exact_successor_structure=False,
        verifier_confirmed_valid=False,
        successor_signature=None,
        legality_status=None,
        replay_status=None,
        verifier_status=None,
        legality_detail=None,
        replay_detail=None,
        verifier_detail=None,
        parse_error=f"{type(error).__name__}: {error}",
    )


def _evaluate_candidate(
    *,
    rank: int,
    raw_action: object,
    demonstration: ActionIdentityV1,
    record: StepRecordProtocol,
    adapter: StepMetricAdapter,
    registered_rule_ids: tuple[str, ...],
    registered_rule_directions: tuple[tuple[str, str], ...],
) -> CandidateMetricOutcomeV1:
    try:
        action = adapter.action_identity(raw_action)
        if not isinstance(action, ActionIdentityV1):
            raise TypeError("adapter action_identity must return ActionIdentityV1")
    except Exception as error:  # adapter boundary: retain malformed provider output
        return _parse_candidate_failure(rank, error)

    exact_action = action.demonstration_key == demonstration.demonstration_key
    if (
        action.rule_id not in registered_rule_ids
        or (action.rule_id, action.direction) not in registered_rule_directions
    ):
        return CandidateMetricOutcomeV1(
            rank=rank,
            action=action,
            status=CandidateMetricStatus.UNSUPPORTED,
            exact_demonstration_action=exact_action,
            exact_successor_structure=False,
            verifier_confirmed_valid=False,
            successor_signature=None,
            legality_status=LegalityStatus.UNSUPPORTED,
            replay_status=None,
            verifier_status=None,
            legality_detail="proposed rule is absent from the supplied frozen registry",
            replay_detail=None,
            verifier_detail=None,
            parse_error=None,
        )
    try:
        legality = adapter.classify_legality(record, raw_action)
        if not isinstance(legality, LegalityResultV1):
            raise TypeError("adapter classify_legality must return LegalityResultV1")
    except Exception as error:
        return CandidateMetricOutcomeV1(
            rank=rank,
            action=action,
            status=CandidateMetricStatus.PARSE_SCHEMA_ERROR,
            exact_demonstration_action=exact_action,
            exact_successor_structure=False,
            verifier_confirmed_valid=False,
            successor_signature=None,
            legality_status=None,
            replay_status=None,
            verifier_status=None,
            legality_detail=None,
            replay_detail=None,
            verifier_detail=None,
            parse_error=f"{type(error).__name__}: {error}",
        )

    if legality.status is not LegalityStatus.LEGAL:
        return CandidateMetricOutcomeV1(
            rank=rank,
            action=action,
            status=_LEGALITY_TO_CANDIDATE_STATUS[legality.status],
            exact_demonstration_action=exact_action,
            exact_successor_structure=False,
            verifier_confirmed_valid=False,
            successor_signature=None,
            legality_status=legality.status,
            replay_status=None,
            verifier_status=None,
            legality_detail=legality.detail,
            replay_detail=None,
            verifier_detail=None,
            parse_error=None,
        )

    try:
        replay = adapter.replay(record, raw_action)
        if not isinstance(replay, ReplayResultV1):
            raise TypeError("adapter replay must return ReplayResultV1")
    except Exception as error:
        return CandidateMetricOutcomeV1(
            rank=rank,
            action=action,
            status=CandidateMetricStatus.REPLAY_ERROR,
            exact_demonstration_action=exact_action,
            exact_successor_structure=False,
            verifier_confirmed_valid=False,
            successor_signature=None,
            legality_status=legality.status,
            replay_status=ReplayStatus.ERROR,
            verifier_status=None,
            legality_detail=legality.detail,
            replay_detail=f"{type(error).__name__}: {error}",
            verifier_detail=None,
            parse_error=None,
        )

    if replay.status is not ReplayStatus.SUCCEEDED:
        return CandidateMetricOutcomeV1(
            rank=rank,
            action=action,
            status=_REPLAY_TO_CANDIDATE_STATUS[replay.status],
            exact_demonstration_action=exact_action,
            exact_successor_structure=False,
            verifier_confirmed_valid=False,
            successor_signature=None,
            legality_status=legality.status,
            replay_status=replay.status,
            verifier_status=None,
            legality_detail=legality.detail,
            replay_detail=replay.detail,
            verifier_detail=None,
            parse_error=None,
        )

    try:
        derived_successor_signature = _sha256_digest(
            adapter.structural_signature(replay.successor_state),
            field="derived replay successor signature",
        )
    except Exception as error:
        return CandidateMetricOutcomeV1(
            rank=rank,
            action=action,
            status=CandidateMetricStatus.REPLAY_ERROR,
            exact_demonstration_action=exact_action,
            exact_successor_structure=False,
            verifier_confirmed_valid=False,
            successor_signature=None,
            legality_status=legality.status,
            replay_status=ReplayStatus.ERROR,
            verifier_status=None,
            legality_detail=legality.detail,
            replay_detail=(
                "could not derive the structural signature of the replayed state: "
                f"{type(error).__name__}: {error}"
            ),
            verifier_detail=None,
            parse_error=None,
        )
    if derived_successor_signature != replay.successor_signature:
        return CandidateMetricOutcomeV1(
            rank=rank,
            action=action,
            status=CandidateMetricStatus.REPLAY_ERROR,
            exact_demonstration_action=exact_action,
            exact_successor_structure=False,
            verifier_confirmed_valid=False,
            successor_signature=None,
            legality_status=legality.status,
            replay_status=ReplayStatus.ERROR,
            verifier_status=None,
            legality_detail=legality.detail,
            replay_detail=(
                "replay-reported successor signature disagrees with the independently "
                "derived structural signature"
            ),
            verifier_detail=None,
            parse_error=None,
        )

    exact_successor = derived_successor_signature == record.next_signature
    try:
        verification = adapter.verify(record, raw_action, replay)
        if not isinstance(verification, VerificationResultV1):
            raise TypeError("adapter verify must return VerificationResultV1")
    except TimeoutError as error:
        verification = VerificationResultV1(
            status=VerificationStatus.TIMEOUT,
            detail=f"{type(error).__name__}: {error}",
        )
    except Exception as error:  # verifier exceptions are evidence, not dropped rows
        verification = VerificationResultV1(
            status=VerificationStatus.ERROR,
            detail=f"{type(error).__name__}: {error}",
        )

    return CandidateMetricOutcomeV1(
        rank=rank,
        action=action,
        status=_VERIFICATION_TO_CANDIDATE_STATUS[verification.status],
        exact_demonstration_action=exact_action,
        exact_successor_structure=exact_successor,
        verifier_confirmed_valid=verification.status is VerificationStatus.ACCEPTED,
        successor_signature=derived_successor_signature,
        legality_status=legality.status,
        replay_status=replay.status,
        verifier_status=verification.status,
        legality_detail=legality.detail,
        replay_detail=replay.detail,
        verifier_detail=verification.detail,
        parse_error=None,
    )


def evaluate_step(
    record: StepRecordProtocol,
    proposal: ProposalProtocol | None,
    *,
    adapter: StepMetricAdapter,
    registered_rule_ids: tuple[str, ...],
    registered_rule_directions: tuple[tuple[str, str], ...],
    rule_registry_digest: str,
    top_ks: tuple[int, ...] = DEFAULT_TOP_KS,
    family_partition_evidence: FamilyPartitionEvidenceV1 | None = None,
) -> StepMetricOutcomeV1:
    """Evaluate one proposal without collapsing imitation, structure, and safety.

    Generalization is ``unknown`` unless a manifest-bound training-family
    inventory is supplied. A held-out label is derived, never caller-assigned.
    """

    rules = _string_tuple(
        registered_rule_ids,
        field="registered_rule_ids",
        sorted_unique=True,
    )
    if not rules:
        raise ValueError("registered_rule_ids cannot be empty")
    directed_rules = _directed_rule_tuple(
        registered_rule_directions,
        field="registered_rule_directions",
    )
    if {rule_id for rule_id, _ in directed_rules} != set(rules):
        raise ValueError("registered rule IDs and directed inventory disagree")
    digest = _sha256_digest(rule_registry_digest, field="rule_registry_digest")
    if not isinstance(top_ks, tuple) or not top_ks or top_ks != tuple(sorted(set(top_ks))):
        raise ValueError("top_ks must be a nonempty sorted unique tuple")
    for k in top_ks:
        _positive_integer(k, field="top-k")
    # These accepted-record fields are the minimum stable identity needed to
    # persist a metric row.  A malformed accepted record is an integration
    # error; issue #61 retains raw trace parse failures before this boundary.
    for name in (
        "trace_id",
        "source_group",
        "current_family",
        "goal_family",
    ):
        _nonblank(getattr(record, name), field=f"record.{name}")
    for name in (
        "record_id",
        "current_signature",
        "goal_signature",
        "next_signature",
        "rule_set_digest",
    ):
        _sha256_digest(getattr(record, name), field=f"record.{name}")
    _string_tuple(record.evaluation_views, field="record.evaluation_views", sorted_unique=True)
    lineage = _string_tuple(
        record.lineage_group_ids,
        field="record.lineage_group_ids",
        sorted_unique=True,
    )
    if record.source_group not in lineage:
        raise ValueError("record.lineage_group_ids must include record.source_group")
    _record_authoritative_split(record)
    _positive_integer(record.remaining_witness_steps, field="record.remaining_witness_steps")
    _positive_integer(record.trace_length, field="record.trace_length")
    if type(record.supported) is not bool:
        raise TypeError("record.supported must be a boolean")
    family_evidence = _family_evidence_snapshot(record, family_partition_evidence)

    try:
        demonstration = _record_action_identity(record)
    except (AttributeError, TypeError, ValueError) as error:
        return _outcome_without_candidates(
            record,
            demonstration_action=None,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.PARSE_SCHEMA_ERROR,
            detail=f"demonstration action parse failed: {type(error).__name__}: {error}",
            legal_action_count=None,
        )
    if demonstration.rule_id not in rules:
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.PARSE_SCHEMA_ERROR,
            detail="demonstrated rule is absent from the supplied frozen registry",
            legal_action_count=None,
        )
    if (demonstration.rule_id, demonstration.direction) not in directed_rules:
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.PARSE_SCHEMA_ERROR,
            detail=("demonstrated rule direction is absent from the supplied frozen registry"),
            legal_action_count=None,
        )
    if record.rule_set_digest != digest:
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.PARSE_SCHEMA_ERROR,
            detail="record rule-set digest differs from the metric registry digest",
            legal_action_count=None,
        )
    if not record.supported:
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.UNSUPPORTED,
            detail="record is explicitly unsupported",
            legal_action_count=None,
        )
    if proposal is None:
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.NO_PROPOSAL,
            detail="no proposal was supplied",
            legal_action_count=None,
        )

    try:
        proposal_status = str(proposal.status)
        proposal_current = _sha256_digest(
            proposal.current_signature,
            field="proposal.current_signature",
        )
        proposal_goal = _sha256_digest(
            proposal.goal_signature,
            field="proposal.goal_signature",
        )
        proposal_digest = _sha256_digest(
            proposal.rule_registry_digest,
            field="proposal.rule_registry_digest",
        )
        legal_mask_digest = _sha256_digest(
            proposal.legal_mask_digest,
            field="proposal.legal_mask_digest",
        )
        legal_action_count = _nonnegative_integer(
            proposal.legal_action_count,
            field="proposal.legal_action_count",
        )
        requested_top_k = _positive_integer(
            proposal.requested_top_k,
            field="proposal.requested_top_k",
        )
        if not isinstance(proposal.candidates, tuple):
            raise TypeError("proposal.candidates must be a tuple")
    except (AttributeError, TypeError, ValueError) as error:
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.PARSE_SCHEMA_ERROR,
            detail=f"proposal parse failed: {type(error).__name__}: {error}",
            legal_action_count=None,
        )
    if (
        proposal_current != record.current_signature
        or proposal_goal != record.goal_signature
        or proposal_digest != digest
    ):
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.PARSE_SCHEMA_ERROR,
            detail="proposal current/goal/registry identity does not match the step record",
            legal_action_count=legal_action_count,
            legal_mask_digest=legal_mask_digest,
        )
    if requested_top_k != max(top_ks):
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.PARSE_SCHEMA_ERROR,
            detail="proposal requested_top_k differs from the largest metric cutoff",
            legal_action_count=legal_action_count,
            legal_mask_digest=legal_mask_digest,
        )

    raw_candidates = proposal.candidates
    if legal_action_count == 0:
        if raw_candidates or proposal_status != "no_legal_action":
            status = ExampleMetricStatus.INVALID_PROPOSAL
            detail = "zero legal actions require an empty no-legal-action proposal"
        else:
            status = ExampleMetricStatus.NO_LEGAL_ACTION
            detail = "the frozen registry enumerated no legal action"
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=status,
            detail=detail,
            legal_action_count=legal_action_count,
            legal_mask_digest=legal_mask_digest,
        )
    if proposal_status == "no_legal_action":
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.INVALID_PROPOSAL,
            detail="proposal claimed no legal action despite a nonempty legal inventory",
            legal_action_count=legal_action_count,
            legal_mask_digest=legal_mask_digest,
        )
    if proposal_status == "unsupported":
        status = (
            ExampleMetricStatus.UNSUPPORTED
            if not raw_candidates
            else ExampleMetricStatus.INVALID_PROPOSAL
        )
        detail = (
            "proposal provider marked the example unsupported"
            if not raw_candidates
            else "an unsupported proposal cannot contain candidates"
        )
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=status,
            detail=detail,
            legal_action_count=legal_action_count,
            legal_mask_digest=legal_mask_digest,
        )
    if proposal_status == "invalid":
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.INVALID_PROPOSAL,
            detail="proposal provider marked the input invalid",
            legal_action_count=legal_action_count,
            legal_mask_digest=legal_mask_digest,
        )
    if proposal_status != "success":
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.PARSE_SCHEMA_ERROR,
            detail=f"unknown proposal status {proposal_status!r}",
            legal_action_count=legal_action_count,
            legal_mask_digest=legal_mask_digest,
        )
    if not raw_candidates:
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.INVALID_PROPOSAL,
            detail="successful proposal is empty despite a nonempty legal inventory",
            legal_action_count=legal_action_count,
            legal_mask_digest=legal_mask_digest,
        )
    if len(raw_candidates) > legal_action_count:
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.INVALID_PROPOSAL,
            detail="proposal contains more candidates than the legal inventory",
            legal_action_count=legal_action_count,
            legal_mask_digest=legal_mask_digest,
        )
    if len(raw_candidates) > max(top_ks):
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.INVALID_PROPOSAL,
            detail="proposal exceeds the largest predeclared metric cutoff",
            legal_action_count=legal_action_count,
            legal_mask_digest=legal_mask_digest,
        )
    required_candidate_count = min(requested_top_k, legal_action_count)
    if len(raw_candidates) != required_candidate_count:
        return _outcome_without_candidates(
            record,
            demonstration_action=demonstration,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
            rule_registry_digest=digest,
            top_ks=top_ks,
            family_evidence=family_evidence,
            status=ExampleMetricStatus.INVALID_PROPOSAL,
            detail=(
                "successful proposal is a truncated legal-action ranking: "
                f"expected {required_candidate_count}, observed {len(raw_candidates)}"
            ),
            legal_action_count=legal_action_count,
            legal_mask_digest=legal_mask_digest,
        )

    candidates: list[CandidateMetricOutcomeV1] = []
    seen_actions: set[tuple[str, str, tuple[int, ...], tuple[str, ...]]] = set()
    for expected_rank, candidate in enumerate(raw_candidates, start=1):
        try:
            rank = _positive_integer(candidate.rank, field="proposal candidate rank")
            if rank != expected_rank:
                raise ValueError("proposal candidate ranks must be consecutive from one")
            raw_action = candidate.action
        except (AttributeError, TypeError, ValueError) as error:
            candidates.append(_parse_candidate_failure(expected_rank, error))
            continue
        outcome = _evaluate_candidate(
            rank=rank,
            raw_action=raw_action,
            demonstration=demonstration,
            record=record,
            adapter=adapter,
            registered_rule_ids=rules,
            registered_rule_directions=directed_rules,
        )
        if outcome.action is not None:
            key = outcome.action.demonstration_key
            if key in seen_actions:
                outcome = _parse_candidate_failure(
                    rank,
                    ValueError("duplicate concrete action in ranked proposal"),
                )
            else:
                seen_actions.add(key)
        candidates.append(outcome)

    return StepMetricOutcomeV1(
        schema_version=STEP_METRIC_SCHEMA_VERSION,
        record_id=record.record_id,
        trace_id=record.trace_id,
        source_group=record.source_group,
        lineage_group_ids=record.lineage_group_ids,
        authoritative_split=_record_authoritative_split(record),
        current_signature=record.current_signature,
        goal_signature=record.goal_signature,
        target_successor_signature=record.next_signature,
        current_family=record.current_family,
        goal_family=record.goal_family,
        evaluation_views=record.evaluation_views,
        family_generalization=family_evidence.generalization,
        family_evidence_manifest_digest=family_evidence.manifest_digest,
        training_family_inventory_digest=family_evidence.inventory_digest,
        unseen_family_roles=family_evidence.unseen_roles,
        remaining_witness_steps=record.remaining_witness_steps,
        trace_length=record.trace_length,
        demonstration_action=demonstration,
        registered_rule_ids=rules,
        registered_rule_directions=directed_rules,
        rule_registry_digest=digest,
        legal_mask_digest=legal_mask_digest,
        requested_top_ks=top_ks,
        legal_action_count=legal_action_count,
        proposal_candidate_count=len(candidates),
        status=ExampleMetricStatus.EVALUATED,
        detail="ranked candidates evaluated with independent legality, replay, and verification",
        candidates=tuple(candidates),
    )


@dataclass(frozen=True, slots=True)
class TopKAggregateV1:
    """Micro counts and explicit denominators at one cutoff."""

    k: int
    example_denominator: int
    attempted_example_denominator: int
    verifier_resolved_example_denominator: int
    demonstration_action_match_count: int
    exact_successor_structure_match_count: int
    verifier_valid_example_count: int
    candidate_attempt_denominator: int
    verifier_attempt_denominator: int
    verifier_resolved_candidate_denominator: int
    verifier_valid_candidate_count: int

    def __post_init__(self) -> None:
        _positive_integer(self.k, field="aggregate k")
        for name in (
            "example_denominator",
            "attempted_example_denominator",
            "verifier_resolved_example_denominator",
            "demonstration_action_match_count",
            "exact_successor_structure_match_count",
            "verifier_valid_example_count",
            "candidate_attempt_denominator",
            "verifier_attempt_denominator",
            "verifier_resolved_candidate_denominator",
            "verifier_valid_candidate_count",
        ):
            _nonnegative_integer(getattr(self, name), field=name)
        if not (
            self.verifier_valid_example_count
            <= self.verifier_resolved_example_denominator
            <= self.attempted_example_denominator
            <= self.example_denominator
        ):
            raise ValueError("aggregate example denominators are inconsistent")
        if (
            self.demonstration_action_match_count > self.example_denominator
            or self.exact_successor_structure_match_count > self.example_denominator
        ):
            raise ValueError("aggregate match count exceeds example denominator")
        if not (
            self.verifier_valid_candidate_count
            <= self.verifier_resolved_candidate_denominator
            <= self.verifier_attempt_denominator
            <= self.candidate_attempt_denominator
        ):
            raise ValueError("aggregate candidate denominators are inconsistent")

    def as_dict(self) -> dict[str, object]:
        return {
            "attempted_example_denominator": self.attempted_example_denominator,
            "candidate_attempt_denominator": self.candidate_attempt_denominator,
            "demonstration_action_match_count": self.demonstration_action_match_count,
            "demonstration_action_match_rate_all": _rate(
                self.demonstration_action_match_count,
                self.example_denominator,
            ),
            "demonstration_action_match_rate_attempted": _rate(
                self.demonstration_action_match_count,
                self.attempted_example_denominator,
            ),
            "exact_successor_structure_match_count": (self.exact_successor_structure_match_count),
            "exact_successor_structure_match_rate_all": _rate(
                self.exact_successor_structure_match_count,
                self.example_denominator,
            ),
            "exact_successor_structure_match_rate_attempted": _rate(
                self.exact_successor_structure_match_count,
                self.attempted_example_denominator,
            ),
            "example_denominator": self.example_denominator,
            "k": self.k,
            "verifier_acceptance_rate_resolved": _rate(
                self.verifier_valid_candidate_count,
                self.verifier_resolved_candidate_denominator,
            ),
            "verifier_attempt_denominator": self.verifier_attempt_denominator,
            "verifier_resolved_candidate_denominator": (
                self.verifier_resolved_candidate_denominator
            ),
            "verifier_resolved_example_denominator": (self.verifier_resolved_example_denominator),
            "verifier_valid_candidate_count": self.verifier_valid_candidate_count,
            "verifier_valid_candidate_rate_attempted": _rate(
                self.verifier_valid_candidate_count,
                self.candidate_attempt_denominator,
            ),
            "verifier_valid_example_count": self.verifier_valid_example_count,
            "verifier_valid_success_rate_all": _rate(
                self.verifier_valid_example_count,
                self.example_denominator,
            ),
            "verifier_valid_success_rate_attempted": _rate(
                self.verifier_valid_example_count,
                self.attempted_example_denominator,
            ),
        }


@dataclass(frozen=True, slots=True)
class RuleAggregateV1:
    """Per-demonstrated-rule micro metrics, including explicit zero rows."""

    rule_id: str
    example_count: int
    top_k: tuple[TopKAggregateV1, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "example_count": self.example_count,
            "rule_id": self.rule_id,
            "top_k": [metric.as_dict() for metric in self.top_k],
        }


@dataclass(frozen=True, slots=True)
class MacroPerRuleV1:
    """Unweighted mean across rules that have demonstration examples."""

    k: int
    covered_rule_denominator: int
    demonstration_action_match_rate: float | None
    exact_successor_structure_match_rate: float | None
    verifier_valid_success_rate: float | None

    def __post_init__(self) -> None:
        _positive_integer(self.k, field="macro k")
        _nonnegative_integer(
            self.covered_rule_denominator,
            field="covered_rule_denominator",
        )
        for name in (
            "demonstration_action_match_rate",
            "exact_successor_structure_match_rate",
            "verifier_valid_success_rate",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be null or a finite rate")
        if self.covered_rule_denominator == 0 and any(
            getattr(self, name) is not None
            for name in (
                "demonstration_action_match_rate",
                "exact_successor_structure_match_rate",
                "verifier_valid_success_rate",
            )
        ):
            raise ValueError("zero-rule macro metrics must be null")

    def as_dict(self) -> dict[str, object]:
        return {
            "covered_rule_denominator": self.covered_rule_denominator,
            "demonstration_action_match_rate": self.demonstration_action_match_rate,
            "exact_successor_structure_match_rate": (self.exact_successor_structure_match_rate),
            "k": self.k,
            "verifier_valid_success_rate": self.verifier_valid_success_rate,
        }


@dataclass(frozen=True, slots=True)
class RuleCoverageV1:
    """Frozen-registry and observed rule/direction coverage."""

    registry_rule_ids: tuple[str, ...]
    registry_rule_directions: tuple[tuple[str, str], ...]
    demonstrated_rule_ids: tuple[str, ...]
    proposed_rule_ids: tuple[str, ...]
    unregistered_demonstrated_rule_ids: tuple[str, ...]
    unregistered_demonstrated_rule_directions: tuple[tuple[str, str], ...]
    unregistered_proposed_rule_ids: tuple[str, ...]
    unregistered_proposed_rule_directions: tuple[tuple[str, str], ...]
    zero_demonstration_rule_ids: tuple[str, ...]
    zero_proposal_rule_ids: tuple[str, ...]
    zero_demonstration_rule_directions: tuple[tuple[str, str], ...]
    zero_proposal_rule_directions: tuple[tuple[str, str], ...]
    demonstrated_rule_directions: tuple[tuple[str, str], ...]
    proposed_rule_directions: tuple[tuple[str, str], ...]
    covered_demonstrated_rule_directions: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "covered_demonstrated_rule_direction_count": len(
                self.covered_demonstrated_rule_directions
            ),
            "covered_demonstrated_rule_direction_rate": _rate(
                len(self.covered_demonstrated_rule_directions),
                len(self.demonstrated_rule_directions),
            ),
            "covered_demonstrated_rule_directions": [
                list(value) for value in self.covered_demonstrated_rule_directions
            ],
            "demonstrated_rule_directions": [
                list(value) for value in self.demonstrated_rule_directions
            ],
            "demonstrated_rule_ids": list(self.demonstrated_rule_ids),
            "demonstrated_rule_rate": _rate(
                len(self.demonstrated_rule_ids),
                len(self.registry_rule_ids),
            ),
            "proposed_rule_directions": [list(value) for value in self.proposed_rule_directions],
            "proposed_rule_ids": list(self.proposed_rule_ids),
            "proposed_rule_rate": _rate(
                len(self.proposed_rule_ids),
                len(self.registry_rule_ids),
            ),
            "registry_rule_ids": list(self.registry_rule_ids),
            "registry_rule_directions": [list(value) for value in self.registry_rule_directions],
            "unregistered_demonstrated_rule_ids": list(self.unregistered_demonstrated_rule_ids),
            "unregistered_demonstrated_rule_directions": [
                list(value) for value in self.unregistered_demonstrated_rule_directions
            ],
            "unregistered_proposed_rule_ids": list(self.unregistered_proposed_rule_ids),
            "unregistered_proposed_rule_directions": [
                list(value) for value in self.unregistered_proposed_rule_directions
            ],
            "zero_demonstration_rule_ids": list(self.zero_demonstration_rule_ids),
            "zero_demonstration_rule_directions": [
                list(value) for value in self.zero_demonstration_rule_directions
            ],
            "zero_proposal_rule_ids": list(self.zero_proposal_rule_ids),
            "zero_proposal_rule_directions": [
                list(value) for value in self.zero_proposal_rule_directions
            ],
        }


@dataclass(frozen=True, slots=True)
class BreakdownAggregateV1:
    """Micro metrics for one reconstructible stratum."""

    dimension: str
    value: str
    example_count: int
    top_k: tuple[TopKAggregateV1, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "example_count": self.example_count,
            "top_k": [metric.as_dict() for metric in self.top_k],
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class StepMetricAggregateV1:
    """Aggregate fully reconstructible from persisted per-example outcomes."""

    schema_version: str
    rule_registry_digest: str
    requested_top_ks: tuple[int, ...]
    total_examples: int
    example_status_counts: tuple[tuple[str, int], ...]
    candidate_status_counts: tuple[tuple[str, int], ...]
    top_k: tuple[TopKAggregateV1, ...]
    per_rule: tuple[RuleAggregateV1, ...]
    macro_per_rule: tuple[MacroPerRuleV1, ...]
    rule_coverage: RuleCoverageV1
    breakdowns: tuple[BreakdownAggregateV1, ...]

    def as_dict(self) -> dict[str, object]:
        example_status_counts = dict(self.example_status_counts)
        candidate_status_counts = dict(self.candidate_status_counts)
        candidate_denominator = sum(candidate_status_counts.values())
        return {
            "breakdowns": [breakdown.as_dict() for breakdown in self.breakdowns],
            "candidate_status_counts": candidate_status_counts,
            "candidate_status_denominator": candidate_denominator,
            "candidate_status_rates": {
                status: _rate(count, candidate_denominator)
                for status, count in self.candidate_status_counts
            },
            "example_status_counts": example_status_counts,
            "example_status_rates": {
                status: _rate(count, self.total_examples)
                for status, count in self.example_status_counts
            },
            "macro_per_rule": [metric.as_dict() for metric in self.macro_per_rule],
            "per_rule": [metric.as_dict() for metric in self.per_rule],
            "requested_top_ks": list(self.requested_top_ks),
            "rule_coverage": self.rule_coverage.as_dict(),
            "rule_registry_digest": self.rule_registry_digest,
            "schema_version": self.schema_version,
            "top_k": [metric.as_dict() for metric in self.top_k],
            "total_examples": self.total_examples,
        }


def _top_k_aggregate(
    rows: tuple[StepMetricOutcomeV1, ...],
    k: int,
) -> TopKAggregateV1:
    outcomes = tuple(row.at_k(k) for row in rows)
    return TopKAggregateV1(
        k=k,
        example_denominator=len(rows),
        attempted_example_denominator=sum(item.candidate_attempts > 0 for item in outcomes),
        verifier_resolved_example_denominator=sum(item.verifier_resolved > 0 for item in outcomes),
        demonstration_action_match_count=sum(item.demonstration_action_match for item in outcomes),
        exact_successor_structure_match_count=sum(
            item.exact_successor_structure_match for item in outcomes
        ),
        verifier_valid_example_count=sum(item.verifier_valid_success for item in outcomes),
        candidate_attempt_denominator=sum(item.candidate_attempts for item in outcomes),
        verifier_attempt_denominator=sum(item.verifier_attempts for item in outcomes),
        verifier_resolved_candidate_denominator=sum(item.verifier_resolved for item in outcomes),
        verifier_valid_candidate_count=sum(item.verifier_valid_candidates for item in outcomes),
    )


def _macro_per_rule(
    per_rule: tuple[RuleAggregateV1, ...],
    top_ks: tuple[int, ...],
) -> tuple[MacroPerRuleV1, ...]:
    covered = tuple(row for row in per_rule if row.example_count > 0)
    result: list[MacroPerRuleV1] = []
    for index, k in enumerate(top_ks):
        denominator = len(covered)
        if denominator == 0:
            demonstration_rate = successor_rate = verifier_rate = None
        else:
            demonstration_rate = (
                sum(
                    rule.top_k[index].demonstration_action_match_count / rule.example_count
                    for rule in covered
                )
                / denominator
            )
            successor_rate = (
                sum(
                    rule.top_k[index].exact_successor_structure_match_count / rule.example_count
                    for rule in covered
                )
                / denominator
            )
            verifier_rate = (
                sum(
                    rule.top_k[index].verifier_valid_example_count / rule.example_count
                    for rule in covered
                )
                / denominator
            )
        result.append(
            MacroPerRuleV1(
                k=k,
                covered_rule_denominator=denominator,
                demonstration_action_match_rate=demonstration_rate,
                exact_successor_structure_match_rate=successor_rate,
                verifier_valid_success_rate=verifier_rate,
            )
        )
    return tuple(result)


def _breakdown(
    rows: tuple[StepMetricOutcomeV1, ...],
    *,
    dimension: str,
    values: tuple[str, ...],
    selector: Callable[[StepMetricOutcomeV1, str], bool],
    top_ks: tuple[int, ...],
) -> tuple[BreakdownAggregateV1, ...]:
    selected: list[BreakdownAggregateV1] = []
    for value in values:
        stratum = tuple(row for row in rows if selector(row, value))
        selected.append(
            BreakdownAggregateV1(
                dimension=dimension,
                value=value,
                example_count=len(stratum),
                top_k=tuple(_top_k_aggregate(stratum, k) for k in top_ks),
            )
        )
    return tuple(selected)


def aggregate_step_metrics(
    rows: tuple[StepMetricOutcomeV1, ...],
) -> StepMetricAggregateV1:
    """Aggregate persisted rows without filtering failures or timeout outcomes."""

    if not isinstance(rows, tuple) or not rows:
        raise ValueError("rows must be a nonempty tuple of per-example outcomes")
    if not all(isinstance(row, StepMetricOutcomeV1) for row in rows):
        raise TypeError("every metric row must be a StepMetricOutcomeV1")
    if len({row.record_id for row in rows}) != len(rows):
        raise ValueError("metric rows contain duplicate record IDs")
    first = rows[0]
    for row in rows[1:]:
        if (
            row.registered_rule_ids != first.registered_rule_ids
            or row.registered_rule_directions != first.registered_rule_directions
            or row.rule_registry_digest != first.rule_registry_digest
            or row.requested_top_ks != first.requested_top_ks
            or row.family_evidence_manifest_digest != first.family_evidence_manifest_digest
            or row.training_family_inventory_digest != first.training_family_inventory_digest
        ):
            raise ValueError(
                "metric rows use incompatible registry, top-k, or family-evidence contracts"
            )

    status_counts = tuple(
        (status.value, sum(row.status is status for row in rows)) for status in ExampleMetricStatus
    )
    candidate_status_counts = tuple(
        (
            status.value,
            sum(candidate.status is status for row in rows for candidate in row.candidates),
        )
        for status in CandidateMetricStatus
    )
    per_rule = tuple(
        RuleAggregateV1(
            rule_id=rule_id,
            example_count=len(rule_rows),
            top_k=tuple(_top_k_aggregate(rule_rows, k) for k in first.requested_top_ks),
        )
        for rule_id in first.registered_rule_ids
        for rule_rows in (tuple(row for row in rows if row.demonstrated_rule_id == rule_id),)
    )

    demonstrated_actions = tuple(
        row.demonstration_action for row in rows if row.demonstration_action is not None
    )
    registered_demonstrated_actions = tuple(
        action
        for action in demonstrated_actions
        if (action.rule_id, action.direction) in first.registered_rule_directions
    )
    demonstrated_rules = tuple(
        sorted({action.rule_id for action in registered_demonstrated_actions})
    )
    unregistered_demonstrated_rules = tuple(
        sorted(
            {
                action.rule_id
                for action in demonstrated_actions
                if action.rule_id not in first.registered_rule_ids
            }
        )
    )
    unregistered_demonstrated_directions = tuple(
        sorted(
            {
                (action.rule_id, action.direction)
                for action in demonstrated_actions
                if (action.rule_id, action.direction) not in first.registered_rule_directions
            }
        )
    )
    proposed_actions = tuple(
        candidate.action
        for row in rows
        for candidate in row.candidates
        if candidate.action is not None
    )
    unregistered_proposed_rules = tuple(
        sorted(
            {
                action.rule_id
                for action in proposed_actions
                if action.rule_id not in first.registered_rule_ids
            }
        )
    )
    unregistered_proposed_directions = tuple(
        sorted(
            {
                (action.rule_id, action.direction)
                for action in proposed_actions
                if (action.rule_id, action.direction) not in first.registered_rule_directions
            }
        )
    )
    registered_proposed_actions = tuple(
        candidate.action
        for row in rows
        for candidate in row.candidates
        if candidate.action is not None
        and candidate.legality_status is LegalityStatus.LEGAL
        and (
            candidate.action.rule_id,
            candidate.action.direction,
        )
        in first.registered_rule_directions
    )
    proposed_rules = tuple(sorted({action.rule_id for action in registered_proposed_actions}))
    demonstrated_directions = tuple(
        sorted({(action.rule_id, action.direction) for action in registered_demonstrated_actions})
    )
    proposed_directions = tuple(
        sorted({(action.rule_id, action.direction) for action in registered_proposed_actions})
    )
    rule_coverage = RuleCoverageV1(
        registry_rule_ids=first.registered_rule_ids,
        registry_rule_directions=first.registered_rule_directions,
        demonstrated_rule_ids=demonstrated_rules,
        proposed_rule_ids=proposed_rules,
        unregistered_demonstrated_rule_ids=unregistered_demonstrated_rules,
        unregistered_demonstrated_rule_directions=(unregistered_demonstrated_directions),
        unregistered_proposed_rule_ids=unregistered_proposed_rules,
        unregistered_proposed_rule_directions=unregistered_proposed_directions,
        zero_demonstration_rule_ids=tuple(
            rule for rule in first.registered_rule_ids if rule not in demonstrated_rules
        ),
        zero_proposal_rule_ids=tuple(
            rule for rule in first.registered_rule_ids if rule not in proposed_rules
        ),
        zero_demonstration_rule_directions=tuple(
            pair for pair in first.registered_rule_directions if pair not in demonstrated_directions
        ),
        zero_proposal_rule_directions=tuple(
            pair for pair in first.registered_rule_directions if pair not in proposed_directions
        ),
        demonstrated_rule_directions=demonstrated_directions,
        proposed_rule_directions=proposed_directions,
        covered_demonstrated_rule_directions=tuple(
            pair for pair in demonstrated_directions if pair in proposed_directions
        ),
    )

    current_families = tuple(sorted({row.current_family for row in rows}))
    goal_families = tuple(sorted({row.goal_family for row in rows}))
    authoritative_splits = tuple(sorted({row.authoritative_split for row in rows}))
    family_statuses = tuple(status.value for status in FamilyGeneralization)
    unseen_family_roles = tuple(sorted({role for row in rows for role in row.unseen_family_roles}))
    remaining_lengths = tuple(
        str(value) for value in sorted({row.remaining_witness_steps for row in rows})
    )
    trace_lengths = tuple(str(value) for value in sorted({row.trace_length for row in rows}))
    evaluation_views = tuple(sorted({view for row in rows for view in row.evaluation_views}))
    breakdowns = (
        *_breakdown(
            rows,
            dimension="authoritative_split",
            values=authoritative_splits,
            selector=lambda row, value: row.authoritative_split == value,
            top_ks=first.requested_top_ks,
        ),
        *_breakdown(
            rows,
            dimension="current_family",
            values=current_families,
            selector=lambda row, value: row.current_family == value,
            top_ks=first.requested_top_ks,
        ),
        *_breakdown(
            rows,
            dimension="goal_family",
            values=goal_families,
            selector=lambda row, value: row.goal_family == value,
            top_ks=first.requested_top_ks,
        ),
        *_breakdown(
            rows,
            dimension="family_generalization",
            values=family_statuses,
            selector=lambda row, value: row.family_generalization.value == value,
            top_ks=first.requested_top_ks,
        ),
        *_breakdown(
            rows,
            dimension="unseen_family_role",
            values=unseen_family_roles,
            selector=lambda row, value: value in row.unseen_family_roles,
            top_ks=first.requested_top_ks,
        ),
        *_breakdown(
            rows,
            dimension="remaining_witness_steps",
            values=remaining_lengths,
            selector=lambda row, value: row.remaining_witness_steps == int(value),
            top_ks=first.requested_top_ks,
        ),
        *_breakdown(
            rows,
            dimension="trace_length",
            values=trace_lengths,
            selector=lambda row, value: row.trace_length == int(value),
            top_ks=first.requested_top_ks,
        ),
        *_breakdown(
            rows,
            dimension="evaluation_view",
            values=evaluation_views,
            selector=lambda row, value: value in row.evaluation_views,
            top_ks=first.requested_top_ks,
        ),
    )
    return StepMetricAggregateV1(
        schema_version=STEP_METRIC_AGGREGATE_SCHEMA_VERSION,
        rule_registry_digest=first.rule_registry_digest,
        requested_top_ks=first.requested_top_ks,
        total_examples=len(rows),
        example_status_counts=status_counts,
        candidate_status_counts=candidate_status_counts,
        top_k=tuple(_top_k_aggregate(rows, k) for k in first.requested_top_ks),
        per_rule=per_rule,
        macro_per_rule=_macro_per_rule(per_rule, first.requested_top_ks),
        rule_coverage=rule_coverage,
        breakdowns=breakdowns,
    )
