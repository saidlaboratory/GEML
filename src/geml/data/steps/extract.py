"""Goal-conditioned, verifier-replayable rewrite-step extraction.

Issue 7-0 owns the persisted :class:`StepRecordV1` contract, but not the
``RewriteTraceV1`` or ``RewriteActionV1`` producer contracts.  The normalized
views and protocols in this module are deliberately narrow integration shims:
the Workstream 1 adapter must authenticate its own producer schemas and hashes
before returning a view.

All persisted JSON is canonical, finite, and immutable.  Extraction validates
the complete trace before accepting any step so that
``remaining_witness_steps`` never describes a suffix containing a failed
transition.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

from geml.export.schema import ExportSchemaError, canonical_json_bytes

STEP_RECORD_SCHEMA_VERSION = "geml-step-record-v1"
STEP_FAILURE_SCHEMA_VERSION = "geml-step-failure-v1"
STEP_DATASET_CONFIG_VERSION = "geml-step-dataset-config-v1"
STEP_DATASET_MANIFEST_VERSION = "geml-step-dataset-manifest-v1"
STEP_SHARD_SCHEMA_VERSION = "geml-step-shard-v1"
STEP_REPLAY_AUDIT_VERSION = "geml-step-replay-audit-v1"
DEFAULT_STEP_OUTPUT_ROOT = Path("outputs/final/goal7/steps")

_SHA256_LENGTH = 64


class StepDatasetProtocolError(ValueError):
    """An input, record, or output violates the frozen step-dataset contract."""


class ResumeMismatchError(StepDatasetProtocolError):
    """Existing output differs from the deterministic resumed output."""


class SplitLeakageError(StepDatasetProtocolError):
    """A source, derived, trace, or e-class-relative group crosses splits."""


class SplitV1(StrEnum):
    """Authoritative source partitions inherited without reassignment."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST_IID = "test_iid"
    TEST_OOD = "test_ood"


class ActionDirectionV1(StrEnum):
    """The directed orientation of one producer-owned rewrite action."""

    FORWARD = "forward"
    BACKWARD = "backward"


class ReplayStatusV1(StrEnum):
    """Typed outcome from exact action enumeration and application."""

    APPLIED = "applied"
    AMBIGUOUS_SITE = "ambiguous_site"
    MISSING_RULE = "missing_rule"
    MISSING_DIRECTION = "missing_direction"
    INVALID_ARGUMENTS = "invalid_arguments"
    INVALID_SITE = "invalid_site"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    UNSUPPORTED_DOMAIN = "unsupported_domain"
    ERROR = "error"


class VerificationStatusV1(StrEnum):
    """Typed verifier outcome retained for every attempted transition."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    ERROR = "error"


class StepFailureCodeV1(StrEnum):
    """Primary reason one trace or transition did not yield training data."""

    TRACE_AUTHENTICATION_FAILED = "trace_authentication_failed"
    CORRUPT_TRACE = "corrupt_trace"
    INCOMPLETE_TRACE = "incomplete_trace"
    ZERO_LENGTH_TRACE = "zero_length_trace"
    GROUP_LEAKAGE = "group_leakage"
    AMBIGUOUS_SITE = "ambiguous_site"
    MISSING_RULE = "missing_rule"
    MISSING_DIRECTION = "missing_direction"
    INVALID_ARGUMENTS = "invalid_arguments"
    INVALID_SITE = "invalid_site"
    SOURCE_SIGNATURE_MISMATCH = "source_signature_mismatch"
    SUCCESSOR_SIGNATURE_MISMATCH = "successor_signature_mismatch"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    UNSUPPORTED_DOMAIN = "unsupported_domain"
    VERIFIER_REJECTED = "verifier_rejected"
    VERIFIER_UNSUPPORTED = "verifier_unsupported"
    VERIFIER_TIMEOUT = "verifier_timeout"
    VERIFIER_ERROR = "verifier_error"
    VERIFIER_IDENTITY_MISMATCH = "verifier_identity_mismatch"
    REPLAY_ERROR = "replay_error"


def _require_nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StepDatasetProtocolError(f"{label} must be a non-blank string")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StepDatasetProtocolError(
            f"{label} must be a 64-character lowercase hexadecimal SHA-256 digest"
        )
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise StepDatasetProtocolError(f"{label} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, label: str) -> int:
    value = _require_nonnegative_int(value, label)
    if value == 0:
        raise StepDatasetProtocolError(f"{label} must be a positive integer")
    return value


def _require_sorted_unique_strings(
    values: object,
    label: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise StepDatasetProtocolError(f"{label} must be a tuple")
    for value in values:
        _require_nonblank(value, f"{label} entry")
    if tuple(sorted(set(values))) != values:
        raise StepDatasetProtocolError(f"{label} must be sorted and unique")
    if not allow_empty and not values:
        raise StepDatasetProtocolError(f"{label} must not be empty")
    return values


def _canonical_bytes(value: object) -> bytes:
    try:
        return canonical_json_bytes(value)
    except ExportSchemaError as error:
        raise StepDatasetProtocolError(str(error)) from error


def _tagged_digest(tag: str, payload: object) -> str:
    _require_nonblank(tag, "digest tag")
    return hashlib.sha256(tag.encode("ascii") + b"\0" + _canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalJson:
    """An immutable snapshot of one strict finite JSON value."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise StepDatasetProtocolError("canonical JSON text must be a string")
        try:
            value = json.loads(
                self.text,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value!r}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise StepDatasetProtocolError(
                "canonical JSON text is not finite valid JSON"
            ) from error
        expected = _canonical_bytes(value).decode("utf-8")
        if self.text != expected:
            raise StepDatasetProtocolError("canonical JSON text is not uniquely serialized")

    @classmethod
    def from_value(cls, value: object) -> Self:
        """Snapshot ``value`` without retaining mutable caller-owned containers."""

        return cls(_canonical_bytes(value).decode("utf-8"))

    def to_value(self) -> object:
        """Return a fresh JSON-compatible value."""

        return json.loads(self.text)


@dataclass(frozen=True, slots=True)
class NormalizedStateV1:
    """Authenticated source-expression state exposed by the trace adapter."""

    state: CanonicalJson
    structural_signature: str
    family: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, CanonicalJson):
            raise StepDatasetProtocolError("state must be CanonicalJson")
        _require_sha256(self.structural_signature, "state structural_signature")
        _require_nonblank(self.family, "state family")


@dataclass(frozen=True, slots=True)
class NormalizedActionV1:
    """Read-only normalized view of a producer-owned ``RewriteActionV1``."""

    action: CanonicalJson
    action_digest: str
    rule_id: str
    direction: ActionDirectionV1
    occurrence_path: tuple[int, ...]
    ordered_arguments: tuple[CanonicalJson, ...]
    source_signature: str
    successor_signature: str
    assumptions: tuple[str, ...]
    domain_mode: str
    stored_verification_status: VerificationStatusV1
    stored_verifier_digest: str
    stored_verification_evidence_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, CanonicalJson):
            raise StepDatasetProtocolError("action must be CanonicalJson")
        _require_sha256(self.action_digest, "action_digest")
        _require_nonblank(self.rule_id, "rule_id")
        if not isinstance(self.direction, ActionDirectionV1):
            raise StepDatasetProtocolError("direction must be ActionDirectionV1")
        if not isinstance(self.occurrence_path, tuple) or any(
            type(slot) is not int or slot < 0 for slot in self.occurrence_path
        ):
            raise StepDatasetProtocolError(
                "occurrence_path must be a tuple of nonnegative child-slot integers"
            )
        if not isinstance(self.ordered_arguments, tuple) or any(
            not isinstance(argument, CanonicalJson) for argument in self.ordered_arguments
        ):
            raise StepDatasetProtocolError(
                "ordered_arguments must be a tuple of CanonicalJson values"
            )
        _require_sha256(self.source_signature, "action source_signature")
        _require_sha256(self.successor_signature, "action successor_signature")
        _require_sorted_unique_strings(self.assumptions, "action assumptions")
        _require_nonblank(self.domain_mode, "action domain_mode")
        if not isinstance(self.stored_verification_status, VerificationStatusV1):
            raise StepDatasetProtocolError(
                "stored_verification_status must be VerificationStatusV1"
            )
        _require_sha256(self.stored_verifier_digest, "stored_verifier_digest")
        _require_sha256(
            self.stored_verification_evidence_digest,
            "stored_verification_evidence_digest",
        )

    def as_dict(self) -> dict[str, object]:
        """Return the lossless normalized action boundary used by consumers."""

        return {
            "action": self.action.to_value(),
            "action_digest": self.action_digest,
            "assumptions": list(self.assumptions),
            "direction": self.direction.value,
            "domain_mode": self.domain_mode,
            "occurrence_path": list(self.occurrence_path),
            "ordered_arguments": [argument.to_value() for argument in self.ordered_arguments],
            "rule_id": self.rule_id,
            "source_signature": self.source_signature,
            "stored_verification_evidence_digest": (self.stored_verification_evidence_digest),
            "stored_verification_status": self.stored_verification_status.value,
            "stored_verifier_digest": self.stored_verifier_digest,
            "successor_signature": self.successor_signature,
        }


@dataclass(frozen=True, slots=True)
class NormalizedTraceV1:
    """Authenticated trace metadata and ordered states/actions from Workstream 1."""

    trace_schema_version: str
    trace_id: str
    trace_digest: str
    input_record_digest: str
    pair_id: str
    source_id: str
    source_group: str
    lineage_group_ids: tuple[str, ...]
    authoritative_split: SplitV1
    evaluation_views: tuple[str, ...]
    source_family: str
    domain_mode: str
    rewrite_mode: str
    rule_set_digest: str
    authentication_evidence_digest: str
    stored_replay_verified: bool
    states: tuple[NormalizedStateV1, ...]
    actions: tuple[NormalizedActionV1, ...]

    def __post_init__(self) -> None:
        _require_nonblank(self.trace_schema_version, "trace_schema_version")
        _require_nonblank(self.trace_id, "trace_id")
        _require_sha256(self.trace_digest, "trace_digest")
        _require_sha256(self.input_record_digest, "input_record_digest")
        _require_nonblank(self.pair_id, "pair_id")
        _require_nonblank(self.source_id, "source_id")
        _require_nonblank(self.source_group, "source_group")
        groups = _require_sorted_unique_strings(
            self.lineage_group_ids,
            "lineage_group_ids",
            allow_empty=False,
        )
        if self.source_group not in groups:
            raise StepDatasetProtocolError(
                "lineage_group_ids must include the authoritative source_group"
            )
        if not isinstance(self.authoritative_split, SplitV1):
            raise StepDatasetProtocolError("authoritative_split must be SplitV1")
        _require_sorted_unique_strings(self.evaluation_views, "evaluation_views")
        _require_nonblank(self.source_family, "source_family")
        _require_nonblank(self.domain_mode, "domain_mode")
        _require_nonblank(self.rewrite_mode, "rewrite_mode")
        _require_sha256(self.rule_set_digest, "rule_set_digest")
        _require_sha256(
            self.authentication_evidence_digest,
            "authentication_evidence_digest",
        )
        if type(self.stored_replay_verified) is not bool:
            raise StepDatasetProtocolError("stored_replay_verified must be a boolean")
        if not isinstance(self.states, tuple) or any(
            not isinstance(state, NormalizedStateV1) for state in self.states
        ):
            raise StepDatasetProtocolError("states must be a tuple of NormalizedStateV1")
        if not isinstance(self.actions, tuple) or any(
            not isinstance(action, NormalizedActionV1) for action in self.actions
        ):
            raise StepDatasetProtocolError("actions must be a tuple of NormalizedActionV1")


@dataclass(frozen=True, slots=True)
class ReplayResultV1:
    """Exact action application result returned by the injected replayer."""

    status: ReplayStatusV1
    reason: str
    successor_state: CanonicalJson | None = None
    successor_signature: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ReplayStatusV1):
            raise StepDatasetProtocolError("replay status must be ReplayStatusV1")
        _require_nonblank(self.reason, "replay reason")
        if self.status is ReplayStatusV1.APPLIED:
            if not isinstance(self.successor_state, CanonicalJson):
                raise StepDatasetProtocolError("applied replay requires a successor state")
            _require_sha256(self.successor_signature, "replay successor_signature")
        elif self.successor_state is not None or self.successor_signature is not None:
            raise StepDatasetProtocolError(
                "non-applied replay must not claim a successor state or signature"
            )


@dataclass(frozen=True, slots=True)
class VerificationResultV1:
    """Fresh verifier result for one replayed transition."""

    status: VerificationStatusV1
    reason: str
    verifier_digest: str
    evidence_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, VerificationStatusV1):
            raise StepDatasetProtocolError("verification status must be VerificationStatusV1")
        _require_nonblank(self.reason, "verification reason")
        _require_sha256(self.verifier_digest, "verifier_digest")
        _require_sha256(self.evidence_digest, "verification evidence_digest")


class TraceInputError(StepDatasetProtocolError):
    """Typed producer-adapter failure carrying the primary retained category."""

    def __init__(self, code: StepFailureCodeV1, reason: str) -> None:
        if not isinstance(code, StepFailureCodeV1):
            raise TypeError("code must be StepFailureCodeV1")
        self.code = code
        self.reason = _require_nonblank(reason, "trace input error reason")
        super().__init__(self.reason)


@runtime_checkable
class TraceAdapterV1(Protocol):
    """Authenticate and normalize the Workstream 1 trace without copying its schema."""

    def input_record_digest(self, trace: object) -> str:
        """Hash the exact producer record bytes, including malformed records."""

    def normalize_and_authenticate(self, trace: object) -> NormalizedTraceV1:
        """Validate producer schema/version/hashes and return an authenticated view."""

    def structural_signature(self, state: CanonicalJson) -> str:
        """Recompute the authoritative representation-independent state signature."""


@runtime_checkable
class ActionReplayerV1(Protocol):
    """Enumerate and apply one exact action under the frozen rule registry."""

    def apply(
        self,
        current_state: CanonicalJson,
        action: NormalizedActionV1,
    ) -> ReplayResultV1: ...


@runtime_checkable
class TransitionVerifierV1(Protocol):
    """Verify one replayed transition under its recorded assumptions."""

    @property
    def verifier_digest(self) -> str:
        """Return the pinned configured verifier implementation digest."""

    def verify(
        self,
        current_state: CanonicalJson,
        successor_state: CanonicalJson,
        action: NormalizedActionV1,
        *,
        assumptions: tuple[str, ...],
        domain_mode: str,
    ) -> VerificationResultV1: ...


def _record_identity_payload(record: StepRecordV1) -> dict[str, object]:
    return {
        "action": record.action.to_value(),
        "action_digest": record.action_digest,
        "action_source_signature": record.action_source_signature,
        "action_successor_signature": record.action_successor_signature,
        "assumptions": list(record.assumptions),
        "authoritative_split": record.authoritative_split.value,
        "current_signature": record.current_signature,
        "current_state": record.current_state.to_value(),
        "current_family": record.current_family,
        "direction": record.direction.value,
        "domain_mode": record.domain_mode,
        "evaluation_views": list(record.evaluation_views),
        "goal_family": record.goal_family,
        "goal_signature": record.goal_signature,
        "goal_state": record.goal_state.to_value(),
        "lineage_group_ids": list(record.lineage_group_ids),
        "next_signature": record.next_signature,
        "next_state": record.next_state.to_value(),
        "occurrence_path": list(record.occurrence_path),
        "ordered_arguments": [argument.to_value() for argument in record.ordered_arguments],
        "pair_id": record.pair_id,
        "remaining_witness_steps": record.remaining_witness_steps,
        "rewrite_mode": record.rewrite_mode,
        "rule_id": record.rule_id,
        "rule_set_digest": record.rule_set_digest,
        "source_family": record.source_family,
        "source_group": record.source_group,
        "source_id": record.source_id,
        "step_index": record.step_index,
        "trace_digest": record.trace_digest,
        "trace_id": record.trace_id,
        "trace_length": record.trace_length,
    }


@dataclass(frozen=True, slots=True)
class StepRecordV1:
    """One accepted goal-conditioned supervised rewrite step."""

    trace_id: str
    trace_digest: str
    pair_id: str
    source_id: str
    step_index: int
    trace_length: int
    current_state: CanonicalJson
    goal_state: CanonicalJson
    action: CanonicalJson
    next_state: CanonicalJson
    current_signature: str
    goal_signature: str
    action_source_signature: str
    action_successor_signature: str
    next_signature: str
    remaining_witness_steps: int
    source_group: str
    lineage_group_ids: tuple[str, ...]
    authoritative_split: SplitV1
    evaluation_views: tuple[str, ...]
    source_family: str
    current_family: str
    goal_family: str
    domain_mode: str
    rewrite_mode: str
    rule_set_digest: str
    action_digest: str
    rule_id: str
    direction: ActionDirectionV1
    occurrence_path: tuple[int, ...]
    ordered_arguments: tuple[CanonicalJson, ...]
    assumptions: tuple[str, ...]
    verification_evidence_digest: str
    verifier_digest: str
    supported: bool = True
    replay_status: ReplayStatusV1 = ReplayStatusV1.APPLIED
    verification_status: VerificationStatusV1 = VerificationStatusV1.ACCEPTED
    schema_version: str = STEP_RECORD_SCHEMA_VERSION
    record_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STEP_RECORD_SCHEMA_VERSION:
            raise StepDatasetProtocolError(f"schema_version must be {STEP_RECORD_SCHEMA_VERSION!r}")
        for label, value in (
            ("trace_id", self.trace_id),
            ("pair_id", self.pair_id),
            ("source_id", self.source_id),
            ("source_group", self.source_group),
            ("source_family", self.source_family),
            ("current_family", self.current_family),
            ("goal_family", self.goal_family),
            ("domain_mode", self.domain_mode),
            ("rewrite_mode", self.rewrite_mode),
            ("rule_id", self.rule_id),
        ):
            _require_nonblank(value, label)
        for label, value in (
            ("trace_digest", self.trace_digest),
            ("current_signature", self.current_signature),
            ("goal_signature", self.goal_signature),
            ("action_source_signature", self.action_source_signature),
            ("action_successor_signature", self.action_successor_signature),
            ("next_signature", self.next_signature),
            ("rule_set_digest", self.rule_set_digest),
            ("action_digest", self.action_digest),
            ("verification_evidence_digest", self.verification_evidence_digest),
            ("verifier_digest", self.verifier_digest),
        ):
            _require_sha256(value, label)
        _require_nonnegative_int(self.step_index, "step_index")
        _require_positive_int(self.trace_length, "trace_length")
        _require_positive_int(self.remaining_witness_steps, "remaining_witness_steps")
        if self.step_index >= self.trace_length:
            raise StepDatasetProtocolError("step_index must be smaller than trace_length")
        if self.remaining_witness_steps != self.trace_length - self.step_index:
            raise StepDatasetProtocolError(
                "remaining_witness_steps must equal trace_length - step_index"
            )
        for label, value in (
            ("current_state", self.current_state),
            ("goal_state", self.goal_state),
            ("action", self.action),
            ("next_state", self.next_state),
        ):
            if not isinstance(value, CanonicalJson):
                raise StepDatasetProtocolError(f"{label} must be CanonicalJson")
        groups = _require_sorted_unique_strings(
            self.lineage_group_ids,
            "lineage_group_ids",
            allow_empty=False,
        )
        if self.source_group not in groups:
            raise StepDatasetProtocolError("lineage_group_ids must include source_group")
        if not isinstance(self.authoritative_split, SplitV1):
            raise StepDatasetProtocolError("authoritative_split must be SplitV1")
        _require_sorted_unique_strings(self.evaluation_views, "evaluation_views")
        if not isinstance(self.direction, ActionDirectionV1):
            raise StepDatasetProtocolError("direction must be ActionDirectionV1")
        if not isinstance(self.occurrence_path, tuple) or any(
            type(slot) is not int or slot < 0 for slot in self.occurrence_path
        ):
            raise StepDatasetProtocolError(
                "occurrence_path must contain nonnegative child-slot integers"
            )
        if not isinstance(self.ordered_arguments, tuple) or any(
            not isinstance(argument, CanonicalJson) for argument in self.ordered_arguments
        ):
            raise StepDatasetProtocolError(
                "ordered_arguments must be a tuple of CanonicalJson values"
            )
        _require_sorted_unique_strings(self.assumptions, "assumptions")
        if type(self.supported) is not bool or not self.supported:
            raise StepDatasetProtocolError("accepted records must have supported=true")
        if self.replay_status is not ReplayStatusV1.APPLIED:
            raise StepDatasetProtocolError("accepted records require replay_status=applied")
        if self.verification_status is not VerificationStatusV1.ACCEPTED:
            raise StepDatasetProtocolError("accepted records require verification_status=accepted")
        if self.action_source_signature != self.current_signature:
            raise StepDatasetProtocolError("action source signature must match current signature")
        if self.action_successor_signature != self.next_signature:
            raise StepDatasetProtocolError("action successor signature must match next signature")
        expected_id = _tagged_digest(STEP_RECORD_SCHEMA_VERSION, _record_identity_payload(self))
        if self.record_id:
            _require_sha256(self.record_id, "record_id")
            if self.record_id != expected_id:
                raise StepDatasetProtocolError("record_id does not match scientific identity")
        else:
            object.__setattr__(self, "record_id", expected_id)

    @property
    def occurrence_depth(self) -> int:
        """Root-zero depth of the exact ordered occurrence path."""

        return len(self.occurrence_path)

    def as_dict(self) -> dict[str, object]:
        """Serialize the frozen public record with no mutable evidence omitted."""

        return {
            "action": self.action.to_value(),
            "action_digest": self.action_digest,
            "action_source_signature": self.action_source_signature,
            "action_successor_signature": self.action_successor_signature,
            "assumptions": list(self.assumptions),
            "authoritative_split": self.authoritative_split.value,
            "current_family": self.current_family,
            "current_signature": self.current_signature,
            "current_state": self.current_state.to_value(),
            "direction": self.direction.value,
            "domain_mode": self.domain_mode,
            "evaluation_views": list(self.evaluation_views),
            "goal_family": self.goal_family,
            "goal_signature": self.goal_signature,
            "goal_state": self.goal_state.to_value(),
            "lineage_group_ids": list(self.lineage_group_ids),
            "next_signature": self.next_signature,
            "next_state": self.next_state.to_value(),
            "occurrence_path": list(self.occurrence_path),
            "ordered_arguments": [argument.to_value() for argument in self.ordered_arguments],
            "pair_id": self.pair_id,
            "record_id": self.record_id,
            "remaining_witness_steps": self.remaining_witness_steps,
            "replay_status": self.replay_status.value,
            "rewrite_mode": self.rewrite_mode,
            "rule_id": self.rule_id,
            "rule_set_digest": self.rule_set_digest,
            "schema_version": self.schema_version,
            "source_family": self.source_family,
            "source_group": self.source_group,
            "source_id": self.source_id,
            "step_index": self.step_index,
            "supported": self.supported,
            "trace_digest": self.trace_digest,
            "trace_id": self.trace_id,
            "trace_length": self.trace_length,
            "verification_evidence_digest": self.verification_evidence_digest,
            "verification_status": self.verification_status.value,
            "verifier_digest": self.verifier_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        """Parse and revalidate one persisted accepted row."""

        required = set(_STEP_RECORD_FIELDS)
        if set(value) != required:
            raise StepDatasetProtocolError("step record fields differ from the frozen schema")
        return cls(
            trace_id=value["trace_id"],
            trace_digest=value["trace_digest"],
            pair_id=value["pair_id"],
            source_id=value["source_id"],
            step_index=value["step_index"],
            trace_length=value["trace_length"],
            current_state=CanonicalJson.from_value(value["current_state"]),
            goal_state=CanonicalJson.from_value(value["goal_state"]),
            action=CanonicalJson.from_value(value["action"]),
            next_state=CanonicalJson.from_value(value["next_state"]),
            current_signature=value["current_signature"],
            goal_signature=value["goal_signature"],
            action_source_signature=value["action_source_signature"],
            action_successor_signature=value["action_successor_signature"],
            next_signature=value["next_signature"],
            remaining_witness_steps=value["remaining_witness_steps"],
            source_group=value["source_group"],
            lineage_group_ids=_string_tuple(value["lineage_group_ids"], "lineage_group_ids"),
            authoritative_split=SplitV1(value["authoritative_split"]),
            evaluation_views=_string_tuple(value["evaluation_views"], "evaluation_views"),
            source_family=value["source_family"],
            current_family=value["current_family"],
            goal_family=value["goal_family"],
            domain_mode=value["domain_mode"],
            rewrite_mode=value["rewrite_mode"],
            rule_set_digest=value["rule_set_digest"],
            action_digest=value["action_digest"],
            rule_id=value["rule_id"],
            direction=ActionDirectionV1(value["direction"]),
            occurrence_path=_int_tuple(value["occurrence_path"], "occurrence_path"),
            ordered_arguments=_canonical_json_tuple(
                value["ordered_arguments"], "ordered_arguments"
            ),
            assumptions=_string_tuple(value["assumptions"], "assumptions"),
            verification_evidence_digest=value["verification_evidence_digest"],
            verifier_digest=value["verifier_digest"],
            supported=value["supported"],
            replay_status=ReplayStatusV1(value["replay_status"]),
            verification_status=VerificationStatusV1(value["verification_status"]),
            schema_version=value["schema_version"],
            record_id=value["record_id"],
        )


_STEP_RECORD_FIELDS = tuple(
    sorted(
        {
            "action",
            "action_digest",
            "action_source_signature",
            "action_successor_signature",
            "assumptions",
            "authoritative_split",
            "current_family",
            "current_signature",
            "current_state",
            "direction",
            "domain_mode",
            "evaluation_views",
            "goal_family",
            "goal_signature",
            "goal_state",
            "lineage_group_ids",
            "next_signature",
            "next_state",
            "occurrence_path",
            "ordered_arguments",
            "pair_id",
            "record_id",
            "remaining_witness_steps",
            "replay_status",
            "rewrite_mode",
            "rule_id",
            "rule_set_digest",
            "schema_version",
            "source_family",
            "source_group",
            "source_id",
            "step_index",
            "supported",
            "trace_digest",
            "trace_id",
            "trace_length",
            "verification_evidence_digest",
            "verification_status",
            "verifier_digest",
        }
    )
)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise StepDatasetProtocolError(f"{label} must be a JSON array of strings")
    return tuple(value)


def _int_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise StepDatasetProtocolError(f"{label} must be a JSON array of integers")
    return tuple(value)


def _canonical_json_tuple(value: object, label: str) -> tuple[CanonicalJson, ...]:
    if not isinstance(value, list):
        raise StepDatasetProtocolError(f"{label} must be a JSON array")
    return tuple(CanonicalJson.from_value(item) for item in value)


def _failure_identity_payload(failure: StepFailureV1) -> dict[str, object]:
    return {
        "action": None if failure.action is None else failure.action.to_value(),
        "action_digest": failure.action_digest,
        "authoritative_split": (
            None if failure.authoritative_split is None else failure.authoritative_split.value
        ),
        "code": failure.failure_code.value,
        "current_family": failure.current_family,
        "current_signature": failure.current_signature,
        "direction": None if failure.direction is None else failure.direction.value,
        "domain_mode": failure.domain_mode,
        "evaluation_views": list(failure.evaluation_views),
        "goal_family": failure.goal_family,
        "goal_signature": failure.goal_signature,
        "input_record_digest": failure.input_record_digest,
        "input_occurrence_index": failure.input_occurrence_index,
        "lineage_group_ids": list(failure.lineage_group_ids),
        "next_signature": failure.next_signature,
        "occurrence_path": (
            None if failure.occurrence_path is None else list(failure.occurrence_path)
        ),
        "ordered_arguments": (
            None
            if failure.ordered_arguments is None
            else [argument.to_value() for argument in failure.ordered_arguments]
        ),
        "pair_id": failure.pair_id,
        "producer_verification_status": (
            None
            if failure.producer_verification_status is None
            else failure.producer_verification_status.value
        ),
        "producer_verification_evidence_digest": failure.producer_verification_evidence_digest,
        "producer_verifier_digest": failure.producer_verifier_digest,
        "reason": failure.reason,
        "replay_status": (None if failure.replay_status is None else failure.replay_status.value),
        "rewrite_mode": failure.rewrite_mode,
        "rule_id": failure.rule_id,
        "rule_set_digest": failure.rule_set_digest,
        "source_family": failure.source_family,
        "source_group": failure.source_group,
        "source_id": failure.source_id,
        "step_index": failure.step_index,
        "supported": failure.supported,
        "trace_digest": failure.trace_digest,
        "trace_id": failure.trace_id,
        "trace_length": failure.trace_length,
        "verification_evidence_digest": failure.verification_evidence_digest,
        "verification_status": (
            None if failure.verification_status is None else failure.verification_status.value
        ),
        "verifier_digest": failure.verifier_digest,
    }


@dataclass(frozen=True, slots=True)
class StepFailureV1:
    """A typed retained row for a trace- or transition-level failure."""

    input_record_digest: str
    failure_code: StepFailureCodeV1
    reason: str
    trace_id: str | None = None
    trace_digest: str | None = None
    pair_id: str | None = None
    source_id: str | None = None
    step_index: int | None = None
    trace_length: int | None = None
    action: CanonicalJson | None = None
    action_digest: str | None = None
    rule_id: str | None = None
    direction: ActionDirectionV1 | None = None
    occurrence_path: tuple[int, ...] | None = None
    ordered_arguments: tuple[CanonicalJson, ...] | None = None
    current_signature: str | None = None
    goal_signature: str | None = None
    next_signature: str | None = None
    source_group: str | None = None
    lineage_group_ids: tuple[str, ...] = ()
    authoritative_split: SplitV1 | None = None
    evaluation_views: tuple[str, ...] = ()
    source_family: str | None = None
    current_family: str | None = None
    goal_family: str | None = None
    domain_mode: str | None = None
    rewrite_mode: str | None = None
    rule_set_digest: str | None = None
    supported: bool | None = None
    replay_status: ReplayStatusV1 | None = None
    verification_status: VerificationStatusV1 | None = None
    producer_verification_status: VerificationStatusV1 | None = None
    producer_verification_evidence_digest: str | None = None
    producer_verifier_digest: str | None = None
    verification_evidence_digest: str | None = None
    verifier_digest: str | None = None
    input_occurrence_index: int = 0
    schema_version: str = STEP_FAILURE_SCHEMA_VERSION
    failure_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STEP_FAILURE_SCHEMA_VERSION:
            raise StepDatasetProtocolError(
                f"schema_version must be {STEP_FAILURE_SCHEMA_VERSION!r}"
            )
        _require_sha256(self.input_record_digest, "input_record_digest")
        _require_nonnegative_int(
            self.input_occurrence_index,
            "input_occurrence_index",
        )
        if not isinstance(self.failure_code, StepFailureCodeV1):
            raise StepDatasetProtocolError("failure_code must be StepFailureCodeV1")
        _require_nonblank(self.reason, "failure reason")
        for label, value in (
            ("trace_id", self.trace_id),
            ("pair_id", self.pair_id),
            ("source_id", self.source_id),
            ("rule_id", self.rule_id),
            ("source_group", self.source_group),
            ("source_family", self.source_family),
            ("current_family", self.current_family),
            ("goal_family", self.goal_family),
            ("domain_mode", self.domain_mode),
            ("rewrite_mode", self.rewrite_mode),
        ):
            if value is not None:
                _require_nonblank(value, label)
        for label, value in (
            ("trace_digest", self.trace_digest),
            ("action_digest", self.action_digest),
            ("current_signature", self.current_signature),
            ("goal_signature", self.goal_signature),
            ("next_signature", self.next_signature),
            ("rule_set_digest", self.rule_set_digest),
            (
                "producer_verification_evidence_digest",
                self.producer_verification_evidence_digest,
            ),
            ("producer_verifier_digest", self.producer_verifier_digest),
            ("verification_evidence_digest", self.verification_evidence_digest),
            ("verifier_digest", self.verifier_digest),
        ):
            if value is not None:
                _require_sha256(value, label)
        if self.step_index is not None:
            _require_nonnegative_int(self.step_index, "step_index")
        if self.trace_length is not None:
            _require_nonnegative_int(self.trace_length, "trace_length")
        if self.action is not None and not isinstance(self.action, CanonicalJson):
            raise StepDatasetProtocolError("failure action must be CanonicalJson or None")
        if self.direction is not None and not isinstance(self.direction, ActionDirectionV1):
            raise StepDatasetProtocolError("failure direction must be ActionDirectionV1 or None")
        if self.occurrence_path is not None and (
            not isinstance(self.occurrence_path, tuple)
            or any(type(slot) is not int or slot < 0 for slot in self.occurrence_path)
        ):
            raise StepDatasetProtocolError(
                "failure occurrence_path must contain nonnegative integers"
            )
        if self.ordered_arguments is not None and (
            not isinstance(self.ordered_arguments, tuple)
            or any(not isinstance(argument, CanonicalJson) for argument in self.ordered_arguments)
        ):
            raise StepDatasetProtocolError("failure ordered_arguments must be CanonicalJson values")
        _require_sorted_unique_strings(self.lineage_group_ids, "lineage_group_ids")
        _require_sorted_unique_strings(self.evaluation_views, "evaluation_views")
        if self.authoritative_split is not None and not isinstance(
            self.authoritative_split, SplitV1
        ):
            raise StepDatasetProtocolError("failure authoritative_split must be SplitV1 or None")
        if self.supported is not None and type(self.supported) is not bool:
            raise StepDatasetProtocolError("failure supported must be boolean or None")
        if self.replay_status is not None and not isinstance(self.replay_status, ReplayStatusV1):
            raise StepDatasetProtocolError("failure replay_status must be ReplayStatusV1 or None")
        if self.verification_status is not None and not isinstance(
            self.verification_status, VerificationStatusV1
        ):
            raise StepDatasetProtocolError(
                "failure verification_status must be VerificationStatusV1 or None"
            )
        if self.producer_verification_status is not None and not isinstance(
            self.producer_verification_status,
            VerificationStatusV1,
        ):
            raise StepDatasetProtocolError(
                "failure producer_verification_status must be VerificationStatusV1 or None"
            )
        expected_id = _tagged_digest(
            STEP_FAILURE_SCHEMA_VERSION,
            _failure_identity_payload(self),
        )
        if self.failure_id:
            _require_sha256(self.failure_id, "failure_id")
            if self.failure_id != expected_id:
                raise StepDatasetProtocolError("failure_id does not match failure identity")
        else:
            object.__setattr__(self, "failure_id", expected_id)

    @property
    def occurrence_depth(self) -> int | None:
        return None if self.occurrence_path is None else len(self.occurrence_path)

    def as_dict(self) -> dict[str, object]:
        return {
            "action": None if self.action is None else self.action.to_value(),
            "action_digest": self.action_digest,
            "authoritative_split": (
                None if self.authoritative_split is None else self.authoritative_split.value
            ),
            "current_family": self.current_family,
            "current_signature": self.current_signature,
            "direction": None if self.direction is None else self.direction.value,
            "domain_mode": self.domain_mode,
            "evaluation_views": list(self.evaluation_views),
            "failure_code": self.failure_code.value,
            "failure_id": self.failure_id,
            "goal_family": self.goal_family,
            "goal_signature": self.goal_signature,
            "input_record_digest": self.input_record_digest,
            "input_occurrence_index": self.input_occurrence_index,
            "lineage_group_ids": list(self.lineage_group_ids),
            "next_signature": self.next_signature,
            "occurrence_path": (
                None if self.occurrence_path is None else list(self.occurrence_path)
            ),
            "ordered_arguments": (
                None
                if self.ordered_arguments is None
                else [argument.to_value() for argument in self.ordered_arguments]
            ),
            "pair_id": self.pair_id,
            "producer_verification_status": (
                None
                if self.producer_verification_status is None
                else self.producer_verification_status.value
            ),
            "producer_verification_evidence_digest": self.producer_verification_evidence_digest,
            "producer_verifier_digest": self.producer_verifier_digest,
            "reason": self.reason,
            "replay_status": (None if self.replay_status is None else self.replay_status.value),
            "rewrite_mode": self.rewrite_mode,
            "rule_id": self.rule_id,
            "rule_set_digest": self.rule_set_digest,
            "schema_version": self.schema_version,
            "source_family": self.source_family,
            "source_group": self.source_group,
            "source_id": self.source_id,
            "step_index": self.step_index,
            "supported": self.supported,
            "trace_digest": self.trace_digest,
            "trace_id": self.trace_id,
            "trace_length": self.trace_length,
            "verification_evidence_digest": self.verification_evidence_digest,
            "verification_status": (
                None if self.verification_status is None else self.verification_status.value
            ),
            "verifier_digest": self.verifier_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        if set(value) != set(_STEP_FAILURE_FIELDS):
            raise StepDatasetProtocolError("step failure fields differ from frozen schema")
        raw_direction = value["direction"]
        raw_split = value["authoritative_split"]
        raw_replay = value["replay_status"]
        raw_verification = value["verification_status"]
        raw_producer_verification = value["producer_verification_status"]
        raw_path = value["occurrence_path"]
        raw_arguments = value["ordered_arguments"]
        return cls(
            input_record_digest=value["input_record_digest"],
            failure_code=StepFailureCodeV1(value["failure_code"]),
            reason=value["reason"],
            trace_id=value["trace_id"],
            trace_digest=value["trace_digest"],
            pair_id=value["pair_id"],
            source_id=value["source_id"],
            step_index=value["step_index"],
            trace_length=value["trace_length"],
            action=(None if value["action"] is None else CanonicalJson.from_value(value["action"])),
            action_digest=value["action_digest"],
            rule_id=value["rule_id"],
            direction=(None if raw_direction is None else ActionDirectionV1(raw_direction)),
            occurrence_path=(None if raw_path is None else _int_tuple(raw_path, "occurrence_path")),
            ordered_arguments=(
                None
                if raw_arguments is None
                else _canonical_json_tuple(raw_arguments, "ordered_arguments")
            ),
            current_signature=value["current_signature"],
            goal_signature=value["goal_signature"],
            next_signature=value["next_signature"],
            source_group=value["source_group"],
            lineage_group_ids=_string_tuple(value["lineage_group_ids"], "lineage_group_ids"),
            authoritative_split=(None if raw_split is None else SplitV1(raw_split)),
            evaluation_views=_string_tuple(value["evaluation_views"], "evaluation_views"),
            source_family=value["source_family"],
            current_family=value["current_family"],
            goal_family=value["goal_family"],
            domain_mode=value["domain_mode"],
            rewrite_mode=value["rewrite_mode"],
            rule_set_digest=value["rule_set_digest"],
            supported=value["supported"],
            replay_status=(None if raw_replay is None else ReplayStatusV1(raw_replay)),
            verification_status=(
                None if raw_verification is None else VerificationStatusV1(raw_verification)
            ),
            producer_verification_status=(
                None
                if raw_producer_verification is None
                else VerificationStatusV1(raw_producer_verification)
            ),
            producer_verification_evidence_digest=value["producer_verification_evidence_digest"],
            producer_verifier_digest=value["producer_verifier_digest"],
            verification_evidence_digest=value["verification_evidence_digest"],
            verifier_digest=value["verifier_digest"],
            input_occurrence_index=value["input_occurrence_index"],
            schema_version=value["schema_version"],
            failure_id=value["failure_id"],
        )


_STEP_FAILURE_FIELDS = tuple(
    sorted(
        {
            "action",
            "action_digest",
            "authoritative_split",
            "current_family",
            "current_signature",
            "direction",
            "domain_mode",
            "evaluation_views",
            "failure_code",
            "failure_id",
            "goal_family",
            "goal_signature",
            "input_record_digest",
            "input_occurrence_index",
            "lineage_group_ids",
            "next_signature",
            "occurrence_path",
            "ordered_arguments",
            "pair_id",
            "producer_verification_status",
            "producer_verification_evidence_digest",
            "producer_verifier_digest",
            "reason",
            "replay_status",
            "rewrite_mode",
            "rule_id",
            "rule_set_digest",
            "schema_version",
            "source_family",
            "source_group",
            "source_id",
            "step_index",
            "supported",
            "trace_digest",
            "trace_id",
            "trace_length",
            "verification_evidence_digest",
            "verification_status",
            "verifier_digest",
        }
    )
)


@dataclass(frozen=True, slots=True)
class ReplayAuditV1:
    """Complete trace/step replay denominator accounting."""

    input_trace_count: int
    authenticated_trace_count: int
    accepted_trace_count: int
    failed_trace_count: int
    zero_length_trace_count: int
    attempted_step_count: int
    accepted_step_count: int
    failure_row_count: int
    failure_counts: tuple[tuple[str, int], ...]
    schema_version: str = STEP_REPLAY_AUDIT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STEP_REPLAY_AUDIT_VERSION:
            raise StepDatasetProtocolError("unexpected replay audit schema")
        for label, value in (
            ("input_trace_count", self.input_trace_count),
            ("authenticated_trace_count", self.authenticated_trace_count),
            ("accepted_trace_count", self.accepted_trace_count),
            ("failed_trace_count", self.failed_trace_count),
            ("zero_length_trace_count", self.zero_length_trace_count),
            ("attempted_step_count", self.attempted_step_count),
            ("accepted_step_count", self.accepted_step_count),
            ("failure_row_count", self.failure_row_count),
        ):
            _require_nonnegative_int(value, label)
        if self.accepted_trace_count + self.failed_trace_count != self.input_trace_count:
            raise StepDatasetProtocolError(
                "accepted_trace_count + failed_trace_count must equal input_trace_count"
            )
        if self.accepted_step_count + self.failure_row_count < self.attempted_step_count:
            raise StepDatasetProtocolError("step accounting cannot omit an attempted transition")
        if (
            not isinstance(self.failure_counts, tuple)
            or any(
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
                or type(row[1]) is not int
                or row[1] < 0
                for row in self.failure_counts
            )
            or tuple(sorted(self.failure_counts)) != self.failure_counts
        ):
            raise StepDatasetProtocolError("failure_counts must be sorted string/count pairs")
        if sum(count for _, count in self.failure_counts) != self.failure_row_count:
            raise StepDatasetProtocolError("failure_counts must reconstruct failure_row_count")

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted_step_count": self.accepted_step_count,
            "accepted_trace_count": self.accepted_trace_count,
            "attempted_step_count": self.attempted_step_count,
            "authenticated_trace_count": self.authenticated_trace_count,
            "failed_trace_count": self.failed_trace_count,
            "failure_counts": dict(self.failure_counts),
            "failure_row_count": self.failure_row_count,
            "input_trace_count": self.input_trace_count,
            "schema_version": self.schema_version,
            "zero_length_trace_count": self.zero_length_trace_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> ReplayAuditV1:
        expected = {
            "accepted_step_count",
            "accepted_trace_count",
            "attempted_step_count",
            "authenticated_trace_count",
            "failed_trace_count",
            "failure_counts",
            "failure_row_count",
            "input_trace_count",
            "schema_version",
            "zero_length_trace_count",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise StepDatasetProtocolError("replay audit fields differ from frozen schema")
        raw_failure_counts = value["failure_counts"]
        if not isinstance(raw_failure_counts, Mapping) or any(
            not isinstance(code, str) or type(count) is not int
            for code, count in raw_failure_counts.items()
        ):
            raise StepDatasetProtocolError(
                "replay audit failure_counts must be a string/integer object"
            )
        return cls(
            input_trace_count=value["input_trace_count"],  # type: ignore[arg-type]
            authenticated_trace_count=value["authenticated_trace_count"],  # type: ignore[arg-type]
            accepted_trace_count=value["accepted_trace_count"],  # type: ignore[arg-type]
            failed_trace_count=value["failed_trace_count"],  # type: ignore[arg-type]
            zero_length_trace_count=value["zero_length_trace_count"],  # type: ignore[arg-type]
            attempted_step_count=value["attempted_step_count"],  # type: ignore[arg-type]
            accepted_step_count=value["accepted_step_count"],  # type: ignore[arg-type]
            failure_row_count=value["failure_row_count"],  # type: ignore[arg-type]
            failure_counts=tuple(sorted(raw_failure_counts.items())),
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


def _replay_audit_from_rows(
    accepted: Sequence[StepRecordV1],
    failures: Sequence[StepFailureV1],
) -> ReplayAuditV1:
    """Reconstruct every replay-audit field from authenticated persisted rows."""

    accepted_by_trace: dict[str, list[StepRecordV1]] = {}
    for row in accepted:
        accepted_by_trace.setdefault(row.trace_id, []).append(row)
    for trace_id, trace_rows in accepted_by_trace.items():
        trace_lengths = {row.trace_length for row in trace_rows}
        trace_identities = {
            (
                row.trace_digest,
                row.pair_id,
                row.source_id,
                row.source_group,
                row.authoritative_split,
                row.goal_signature,
            )
            for row in trace_rows
        }
        if len(trace_lengths) != 1 or len(trace_identities) != 1:
            raise StepDatasetProtocolError(
                f"accepted trace rows disagree on trace identity: {trace_id}"
            )
        trace_length = next(iter(trace_lengths))
        if tuple(sorted(row.step_index for row in trace_rows)) != tuple(range(trace_length)):
            raise StepDatasetProtocolError(
                f"accepted trace rows do not cover every transition exactly once: {trace_id}"
            )

    failures_by_input: dict[tuple[str, int], list[StepFailureV1]] = {}
    for row in failures:
        key = (row.input_record_digest, row.input_occurrence_index)
        failures_by_input.setdefault(key, []).append(row)
    for key, trace_rows in failures_by_input.items():
        trace_ids = {row.trace_id for row in trace_rows}
        if len(trace_ids) != 1:
            raise StepDatasetProtocolError(
                f"failure rows disagree on trace identity for input occurrence {key!r}"
            )
        indexed_rows = [row for row in trace_rows if row.step_index is not None]
        if indexed_rows:
            if len(indexed_rows) != len(trace_rows):
                raise StepDatasetProtocolError(
                    "a failed trace cannot mix transition and trace-level failure rows"
                )
            trace_lengths = {row.trace_length for row in indexed_rows}
            if len(trace_lengths) != 1 or None in trace_lengths:
                raise StepDatasetProtocolError("transition failure rows disagree on trace length")
            trace_length = next(iter(trace_lengths))
            assert trace_length is not None
            if tuple(sorted(row.step_index for row in indexed_rows)) != tuple(range(trace_length)):
                raise StepDatasetProtocolError(
                    "transition failure rows do not cover every attempted transition"
                )
        elif len(trace_rows) != 1:
            raise StepDatasetProtocolError(
                "a trace-level failure input must produce exactly one retained row"
            )

    accepted_trace_ids = set(accepted_by_trace)
    failed_trace_ids = {row.trace_id for row in failures if row.trace_id is not None}
    overlap = sorted(accepted_trace_ids & failed_trace_ids)
    if overlap:
        raise StepDatasetProtocolError(
            "a trace identity cannot appear in accepted and failure rows: " + ", ".join(overlap)
        )

    authentication_failure_inputs = {
        key
        for key, trace_rows in failures_by_input.items()
        if any(
            row.failure_code is StepFailureCodeV1.TRACE_AUTHENTICATION_FAILED for row in trace_rows
        )
    }
    if any(len(failures_by_input[key]) != 1 for key in authentication_failure_inputs):
        raise StepDatasetProtocolError(
            "an authentication failure input must produce exactly one retained row"
        )

    failure_counts = tuple(sorted(Counter(row.failure_code.value for row in failures).items()))
    return ReplayAuditV1(
        input_trace_count=len(accepted_by_trace) + len(failures_by_input),
        authenticated_trace_count=(
            len(accepted_by_trace) + len(failures_by_input) - len(authentication_failure_inputs)
        ),
        accepted_trace_count=len(accepted_by_trace),
        failed_trace_count=len(failures_by_input),
        zero_length_trace_count=sum(
            any(row.failure_code is StepFailureCodeV1.ZERO_LENGTH_TRACE for row in trace_rows)
            for trace_rows in failures_by_input.values()
        ),
        attempted_step_count=(len(accepted) + sum(row.step_index is not None for row in failures)),
        accepted_step_count=len(accepted),
        failure_row_count=len(failures),
        failure_counts=failure_counts,
    )


@dataclass(frozen=True, slots=True)
class StepExtractionResultV1:
    """Typed in-memory extraction output before deterministic sharding."""

    accepted: tuple[StepRecordV1, ...]
    failures: tuple[StepFailureV1, ...]
    replay_audit: ReplayAuditV1
    input_digest: str
    rule_set_digest: str
    verifier_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, tuple) or any(
            not isinstance(row, StepRecordV1) for row in self.accepted
        ):
            raise StepDatasetProtocolError("accepted must be StepRecordV1 rows")
        if not isinstance(self.failures, tuple) or any(
            not isinstance(row, StepFailureV1) for row in self.failures
        ):
            raise StepDatasetProtocolError("failures must be StepFailureV1 rows")
        if not isinstance(self.replay_audit, ReplayAuditV1):
            raise StepDatasetProtocolError("replay_audit must be ReplayAuditV1")
        _require_sha256(self.input_digest, "input_digest")
        _require_sha256(self.rule_set_digest, "rule_set_digest")
        _require_sha256(self.verifier_digest, "verifier_digest")
        if any(row.rule_set_digest != self.rule_set_digest for row in self.accepted):
            raise StepDatasetProtocolError("every accepted row must use result.rule_set_digest")
        if any(row.verifier_digest != self.verifier_digest for row in self.accepted):
            raise StepDatasetProtocolError("every accepted row must use result.verifier_digest")
        if self.replay_audit.accepted_step_count != len(self.accepted):
            raise StepDatasetProtocolError("replay audit accepted count mismatch")
        if self.replay_audit.failure_row_count != len(self.failures):
            raise StepDatasetProtocolError("replay audit failure count mismatch")


def _failure_for_trace(
    trace: NormalizedTraceV1,
    code: StepFailureCodeV1,
    reason: str,
    *,
    step_index: int | None = None,
    action: NormalizedActionV1 | None = None,
    current: NormalizedStateV1 | None = None,
    goal: NormalizedStateV1 | None = None,
    next_state: NormalizedStateV1 | None = None,
    replay_status: ReplayStatusV1 | None = None,
    verification: VerificationResultV1 | None = None,
    trusted_input_record_digest: str | None = None,
) -> StepFailureV1:
    input_record_digest = (
        trace.input_record_digest
        if trusted_input_record_digest is None
        else _require_sha256(
            trusted_input_record_digest,
            "trusted input_record_digest",
        )
    )
    return StepFailureV1(
        input_record_digest=input_record_digest,
        failure_code=code,
        reason=reason,
        trace_id=trace.trace_id,
        trace_digest=trace.trace_digest,
        pair_id=trace.pair_id,
        source_id=trace.source_id,
        step_index=step_index,
        trace_length=len(trace.actions),
        action=None if action is None else action.action,
        action_digest=None if action is None else action.action_digest,
        rule_id=None if action is None else action.rule_id,
        direction=None if action is None else action.direction,
        occurrence_path=None if action is None else action.occurrence_path,
        ordered_arguments=None if action is None else action.ordered_arguments,
        current_signature=(None if current is None else current.structural_signature),
        goal_signature=None if goal is None else goal.structural_signature,
        next_signature=(None if next_state is None else next_state.structural_signature),
        source_group=trace.source_group,
        lineage_group_ids=trace.lineage_group_ids,
        authoritative_split=trace.authoritative_split,
        evaluation_views=trace.evaluation_views,
        source_family=trace.source_family,
        current_family=None if current is None else current.family,
        goal_family=None if goal is None else goal.family,
        domain_mode=trace.domain_mode,
        rewrite_mode=trace.rewrite_mode,
        rule_set_digest=trace.rule_set_digest,
        supported=_failure_supported(code),
        replay_status=replay_status,
        verification_status=(None if verification is None else verification.status),
        producer_verification_status=(
            None if action is None else action.stored_verification_status
        ),
        producer_verification_evidence_digest=(
            None if action is None else action.stored_verification_evidence_digest
        ),
        producer_verifier_digest=(None if action is None else action.stored_verifier_digest),
        verification_evidence_digest=(
            None if verification is None else verification.evidence_digest
        ),
        verifier_digest=(None if verification is None else verification.verifier_digest),
    )


def _failure_supported(code: StepFailureCodeV1) -> bool | None:
    if code in {
        StepFailureCodeV1.UNSUPPORTED_OPERATOR,
        StepFailureCodeV1.UNSUPPORTED_DOMAIN,
        StepFailureCodeV1.VERIFIER_UNSUPPORTED,
        StepFailureCodeV1.MISSING_RULE,
    }:
        return False
    if code in {
        StepFailureCodeV1.AMBIGUOUS_SITE,
        StepFailureCodeV1.MISSING_DIRECTION,
        StepFailureCodeV1.INVALID_ARGUMENTS,
        StepFailureCodeV1.INVALID_SITE,
        StepFailureCodeV1.SOURCE_SIGNATURE_MISMATCH,
        StepFailureCodeV1.SUCCESSOR_SIGNATURE_MISMATCH,
        StepFailureCodeV1.VERIFIER_REJECTED,
        StepFailureCodeV1.VERIFIER_TIMEOUT,
        StepFailureCodeV1.VERIFIER_ERROR,
        StepFailureCodeV1.VERIFIER_IDENTITY_MISMATCH,
    }:
        return True
    return None


_REPLAY_FAILURE_CODES: Mapping[ReplayStatusV1, StepFailureCodeV1] = {
    ReplayStatusV1.AMBIGUOUS_SITE: StepFailureCodeV1.AMBIGUOUS_SITE,
    ReplayStatusV1.MISSING_RULE: StepFailureCodeV1.MISSING_RULE,
    ReplayStatusV1.MISSING_DIRECTION: StepFailureCodeV1.MISSING_DIRECTION,
    ReplayStatusV1.INVALID_ARGUMENTS: StepFailureCodeV1.INVALID_ARGUMENTS,
    ReplayStatusV1.INVALID_SITE: StepFailureCodeV1.INVALID_SITE,
    ReplayStatusV1.UNSUPPORTED_OPERATOR: StepFailureCodeV1.UNSUPPORTED_OPERATOR,
    ReplayStatusV1.UNSUPPORTED_DOMAIN: StepFailureCodeV1.UNSUPPORTED_DOMAIN,
    ReplayStatusV1.ERROR: StepFailureCodeV1.REPLAY_ERROR,
}

_VERIFICATION_FAILURE_CODES: Mapping[VerificationStatusV1, StepFailureCodeV1] = {
    VerificationStatusV1.REJECTED: StepFailureCodeV1.VERIFIER_REJECTED,
    VerificationStatusV1.UNSUPPORTED: StepFailureCodeV1.VERIFIER_UNSUPPORTED,
    VerificationStatusV1.TIMEOUT: StepFailureCodeV1.VERIFIER_TIMEOUT,
    VerificationStatusV1.ERROR: StepFailureCodeV1.VERIFIER_ERROR,
}


def _evaluate_transition(
    trace: NormalizedTraceV1,
    step_index: int,
    *,
    replayer: ActionReplayerV1,
    state_signature_provider: TraceAdapterV1,
    verifier: TransitionVerifierV1,
    expected_verifier_digest: str,
) -> StepRecordV1 | StepFailureV1:
    current = trace.states[step_index]
    goal = trace.states[-1]
    next_state = trace.states[step_index + 1]
    action = trace.actions[step_index]

    if action.stored_verifier_digest != expected_verifier_digest:
        return _failure_for_trace(
            trace,
            StepFailureCodeV1.VERIFIER_IDENTITY_MISMATCH,
            "producer-stored verifier digest differs from the pinned configured verifier",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
        )
    if action.source_signature != current.structural_signature:
        return _failure_for_trace(
            trace,
            StepFailureCodeV1.SOURCE_SIGNATURE_MISMATCH,
            "stored action source signature differs from stored current state",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
        )
    if action.successor_signature != next_state.structural_signature:
        return _failure_for_trace(
            trace,
            StepFailureCodeV1.SUCCESSOR_SIGNATURE_MISMATCH,
            "stored action successor signature differs from stored next state",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
        )
    if action.domain_mode != trace.domain_mode:
        return _failure_for_trace(
            trace,
            StepFailureCodeV1.UNSUPPORTED_DOMAIN,
            "action domain mode differs from the authenticated trace domain",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
        )
    if action.stored_verification_status is not VerificationStatusV1.ACCEPTED:
        code = _VERIFICATION_FAILURE_CODES[action.stored_verification_status]
        return _failure_for_trace(
            trace,
            code,
            "producer trace does not contain an accepted stored transition verification",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
        )

    try:
        replay = replayer.apply(current.state, action)
    except TimeoutError:
        return _failure_for_trace(
            trace,
            StepFailureCodeV1.REPLAY_ERROR,
            "exact action replay raised TimeoutError",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
            replay_status=ReplayStatusV1.ERROR,
        )
    except Exception as error:
        return _failure_for_trace(
            trace,
            StepFailureCodeV1.REPLAY_ERROR,
            f"exact action replay raised {type(error).__name__}: {error}",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
            replay_status=ReplayStatusV1.ERROR,
        )
    if not isinstance(replay, ReplayResultV1):
        return _failure_for_trace(
            trace,
            StepFailureCodeV1.REPLAY_ERROR,
            "exact action replayer returned a value outside ReplayResultV1",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
            replay_status=ReplayStatusV1.ERROR,
        )
    if replay.status is not ReplayStatusV1.APPLIED:
        return _failure_for_trace(
            trace,
            _REPLAY_FAILURE_CODES[replay.status],
            replay.reason,
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
            replay_status=replay.status,
        )
    assert replay.successor_state is not None  # enforced by ReplayResultV1
    try:
        authoritative_replay_signature = state_signature_provider.structural_signature(
            replay.successor_state
        )
        _require_sha256(
            authoritative_replay_signature,
            "authoritative replay successor signature",
        )
    except Exception as error:
        return _failure_for_trace(
            trace,
            StepFailureCodeV1.REPLAY_ERROR,
            f"authoritative successor signature computation raised {type(error).__name__}: {error}",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
            replay_status=ReplayStatusV1.ERROR,
        )
    if (
        replay.successor_state != next_state.state
        or authoritative_replay_signature != next_state.structural_signature
        or replay.successor_signature != authoritative_replay_signature
    ):
        return _failure_for_trace(
            trace,
            StepFailureCodeV1.SUCCESSOR_SIGNATURE_MISMATCH,
            "replayed successor state or independently computed structural signature "
            "differs from the stored next state",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
            replay_status=replay.status,
        )

    try:
        verification = verifier.verify(
            current.state,
            replay.successor_state,
            action,
            assumptions=action.assumptions,
            domain_mode=trace.domain_mode,
        )
    except TimeoutError:
        return _failure_for_trace(
            trace,
            StepFailureCodeV1.VERIFIER_TIMEOUT,
            "transition verifier raised TimeoutError",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
            replay_status=replay.status,
        )
    except Exception as error:
        return _failure_for_trace(
            trace,
            StepFailureCodeV1.VERIFIER_ERROR,
            f"transition verifier raised {type(error).__name__}: {error}",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
            replay_status=replay.status,
        )
    if not isinstance(verification, VerificationResultV1):
        return _failure_for_trace(
            trace,
            StepFailureCodeV1.VERIFIER_ERROR,
            "transition verifier returned a value outside VerificationResultV1",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
            replay_status=replay.status,
        )
    if verification.verifier_digest != expected_verifier_digest:
        return _failure_for_trace(
            trace,
            StepFailureCodeV1.VERIFIER_IDENTITY_MISMATCH,
            "fresh verifier result digest differs from the pinned configured verifier",
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
            replay_status=replay.status,
            verification=verification,
        )
    if verification.status is not VerificationStatusV1.ACCEPTED:
        return _failure_for_trace(
            trace,
            _VERIFICATION_FAILURE_CODES[verification.status],
            verification.reason,
            step_index=step_index,
            action=action,
            current=current,
            goal=goal,
            next_state=next_state,
            replay_status=replay.status,
            verification=verification,
        )

    return StepRecordV1(
        trace_id=trace.trace_id,
        trace_digest=trace.trace_digest,
        pair_id=trace.pair_id,
        source_id=trace.source_id,
        step_index=step_index,
        trace_length=len(trace.actions),
        current_state=current.state,
        goal_state=goal.state,
        action=action.action,
        next_state=next_state.state,
        current_signature=current.structural_signature,
        goal_signature=goal.structural_signature,
        action_source_signature=action.source_signature,
        action_successor_signature=action.successor_signature,
        next_signature=next_state.structural_signature,
        remaining_witness_steps=len(trace.actions) - step_index,
        source_group=trace.source_group,
        lineage_group_ids=trace.lineage_group_ids,
        authoritative_split=trace.authoritative_split,
        evaluation_views=trace.evaluation_views,
        source_family=trace.source_family,
        current_family=current.family,
        goal_family=goal.family,
        domain_mode=trace.domain_mode,
        rewrite_mode=trace.rewrite_mode,
        rule_set_digest=trace.rule_set_digest,
        action_digest=action.action_digest,
        rule_id=action.rule_id,
        direction=action.direction,
        occurrence_path=action.occurrence_path,
        ordered_arguments=action.ordered_arguments,
        assumptions=action.assumptions,
        verification_evidence_digest=verification.evidence_digest,
        verifier_digest=verification.verifier_digest,
    )


def _extract_authenticated_trace(
    normalized: NormalizedTraceV1,
    *,
    replayer: ActionReplayerV1,
    state_signature_provider: TraceAdapterV1,
    verifier: TransitionVerifierV1,
    expected_verifier_digest: str,
) -> tuple[tuple[StepRecordV1, ...], tuple[StepFailureV1, ...], bool]:
    if len(normalized.states) != len(normalized.actions) + 1:
        return (
            (),
            (
                _failure_for_trace(
                    normalized,
                    StepFailureCodeV1.CORRUPT_TRACE,
                    "trace requires len(states) == len(actions) + 1",
                ),
            ),
            True,
        )
    if not normalized.actions:
        return (
            (),
            (
                _failure_for_trace(
                    normalized,
                    StepFailureCodeV1.ZERO_LENGTH_TRACE,
                    "zero-length positive trace cannot yield a supervised rewrite step",
                ),
            ),
            True,
        )
    if not normalized.stored_replay_verified:
        return (
            (),
            (
                _failure_for_trace(
                    normalized,
                    StepFailureCodeV1.CORRUPT_TRACE,
                    "producer trace is not marked complete and replay-verified",
                ),
            ),
            True,
        )

    outcomes = tuple(
        _evaluate_transition(
            normalized,
            step_index,
            replayer=replayer,
            state_signature_provider=state_signature_provider,
            verifier=verifier,
            expected_verifier_digest=expected_verifier_digest,
        )
        for step_index in range(len(normalized.actions))
    )
    transition_failures = {
        failure.step_index: failure for failure in outcomes if isinstance(failure, StepFailureV1)
    }
    if not transition_failures:
        return (
            tuple(record for record in outcomes if isinstance(record, StepRecordV1)),
            (),
            True,
        )

    goal = normalized.states[-1]
    failures: list[StepFailureV1] = []
    for step_index, outcome in enumerate(outcomes):
        if isinstance(outcome, StepFailureV1):
            failures.append(outcome)
            continue
        failures.append(
            _failure_for_trace(
                normalized,
                StepFailureCodeV1.INCOMPLETE_TRACE,
                "another transition in this trace failed; witnessed suffix is not accepted",
                step_index=step_index,
                action=normalized.actions[step_index],
                current=normalized.states[step_index],
                goal=goal,
                next_state=normalized.states[step_index + 1],
                replay_status=ReplayStatusV1.APPLIED,
                verification=VerificationResultV1(
                    status=outcome.verification_status,
                    reason=(
                        "fresh accepted verification retained after another transition "
                        "invalidated the trace"
                    ),
                    verifier_digest=outcome.verifier_digest,
                    evidence_digest=outcome.verification_evidence_digest,
                ),
            )
        )
    return (), tuple(failures), True


def extract_trace_steps(
    trace: object,
    *,
    adapter: TraceAdapterV1,
    replayer: ActionReplayerV1,
    verifier: TransitionVerifierV1,
    input_record_digest: str | None = None,
    expected_verifier_digest: str | None = None,
) -> tuple[tuple[StepRecordV1, ...], tuple[StepFailureV1, ...], bool]:
    """Authenticate and fully replay one trace.

    The boolean is true only when producer authentication succeeded.  If any
    transition fails, otherwise valid transitions become typed
    ``incomplete_trace`` rows rather than training examples.
    """

    if not isinstance(adapter, TraceAdapterV1):
        raise TypeError("adapter must implement TraceAdapterV1")
    if not isinstance(replayer, ActionReplayerV1):
        raise TypeError("replayer must implement ActionReplayerV1")
    if not isinstance(verifier, TransitionVerifierV1):
        raise TypeError("verifier must implement TransitionVerifierV1")
    configured_verifier_digest = _require_sha256(
        verifier.verifier_digest,
        "configured verifier_digest",
    )
    if expected_verifier_digest is not None:
        _require_sha256(expected_verifier_digest, "expected_verifier_digest")
        if configured_verifier_digest != expected_verifier_digest:
            raise StepDatasetProtocolError(
                "configured verifier digest differs from expected_verifier_digest"
            )
    source_digest = adapter.input_record_digest(trace)
    _require_sha256(source_digest, "input_record_digest")
    if input_record_digest is not None:
        _require_sha256(input_record_digest, "expected input_record_digest")
        if input_record_digest != source_digest:
            return (
                (),
                (
                    StepFailureV1(
                        input_record_digest=source_digest,
                        failure_code=StepFailureCodeV1.TRACE_AUTHENTICATION_FAILED,
                        reason=(
                            "provided input_record_digest differs from the adapter-computed "
                            "exact source record digest"
                        ),
                    ),
                ),
                False,
            )
    try:
        normalized = adapter.normalize_and_authenticate(trace)
    except TraceInputError as error:
        return (
            (),
            (
                StepFailureV1(
                    input_record_digest=source_digest,
                    failure_code=error.code,
                    reason=error.reason,
                ),
            ),
            False,
        )
    except Exception as error:
        return (
            (),
            (
                StepFailureV1(
                    input_record_digest=source_digest,
                    failure_code=StepFailureCodeV1.TRACE_AUTHENTICATION_FAILED,
                    reason=f"trace adapter raised {type(error).__name__}: {error}",
                ),
            ),
            False,
        )
    if not isinstance(normalized, NormalizedTraceV1):
        return (
            (),
            (
                StepFailureV1(
                    input_record_digest=source_digest,
                    failure_code=StepFailureCodeV1.TRACE_AUTHENTICATION_FAILED,
                    reason="trace adapter returned a value outside NormalizedTraceV1",
                ),
            ),
            False,
        )
    if normalized.input_record_digest != source_digest:
        return (
            (),
            (
                _failure_for_trace(
                    normalized,
                    StepFailureCodeV1.TRACE_AUTHENTICATION_FAILED,
                    "normalized input_record_digest differs from exact source record digest",
                    trusted_input_record_digest=source_digest,
                ),
            ),
            False,
        )
    return _extract_authenticated_trace(
        normalized,
        replayer=replayer,
        state_signature_provider=adapter,
        verifier=verifier,
        expected_verifier_digest=configured_verifier_digest,
    )


def _input_digest(record_digests: Sequence[str]) -> str:
    for digest in record_digests:
        _require_sha256(digest, "input record digest")
    return _tagged_digest(
        "geml-step-input-record-set-v1",
        sorted(record_digests),
    )


def _split_leakage(
    traces: Sequence[NormalizedTraceV1],
) -> dict[str, tuple[SplitV1, ...]]:
    memberships: dict[str, set[SplitV1]] = {}
    for trace in traces:
        for group_id in _partition_keys(trace):
            memberships.setdefault(group_id, set()).add(trace.authoritative_split)
    return {
        group_id: tuple(sorted(splits, key=lambda split: split.value))
        for group_id, splits in sorted(memberships.items())
        if len(splits) > 1
    }


def _partition_keys(trace: NormalizedTraceV1) -> tuple[str, ...]:
    return (
        f"pair:{trace.pair_id}",
        f"source:{trace.source_id}",
        f"trace:{trace.trace_id}",
        *(f"lineage:{group_id}" for group_id in trace.lineage_group_ids),
    )


def extract_step_dataset(
    traces: Iterable[object],
    *,
    adapter: TraceAdapterV1,
    replayer: ActionReplayerV1,
    verifier: TransitionVerifierV1,
    expected_verifier_digest: str,
    expected_input_digest: str | None = None,
    expected_rule_set_digest: str | None = None,
) -> StepExtractionResultV1:
    """Extract a deterministic dataset while retaining all typed failures."""

    _require_sha256(expected_verifier_digest, "expected_verifier_digest")
    if not isinstance(verifier, TransitionVerifierV1):
        raise TypeError("verifier must implement TransitionVerifierV1")
    configured_verifier_digest = _require_sha256(
        verifier.verifier_digest,
        "configured verifier_digest",
    )
    if configured_verifier_digest != expected_verifier_digest:
        raise StepDatasetProtocolError(
            "configured verifier digest differs from expected_verifier_digest"
        )
    materialized = tuple(traces)
    source_digests = tuple(adapter.input_record_digest(trace) for trace in materialized)
    for digest in source_digests:
        _require_sha256(digest, "adapter input record digest")
    seen_source_digests: Counter[str] = Counter()
    input_occurrence_indices: list[int] = []
    for digest in source_digests:
        input_occurrence_indices.append(seen_source_digests[digest])
        seen_source_digests[digest] += 1
    input_digest = _input_digest(source_digests)
    if expected_input_digest is not None:
        _require_sha256(expected_input_digest, "expected_input_digest")
        if input_digest != expected_input_digest:
            raise StepDatasetProtocolError(
                "authenticated input digest differs from expected_input_digest"
            )

    normalized_by_index: dict[int, NormalizedTraceV1] = {}
    preliminary_failures: dict[int, StepFailureV1] = {}
    for index, (trace, source_digest) in enumerate(zip(materialized, source_digests, strict=True)):
        try:
            normalized = adapter.normalize_and_authenticate(trace)
        except TraceInputError as error:
            preliminary_failures[index] = StepFailureV1(
                input_record_digest=source_digest,
                failure_code=error.code,
                reason=error.reason,
            )
            continue
        except Exception as error:
            preliminary_failures[index] = StepFailureV1(
                input_record_digest=source_digest,
                failure_code=StepFailureCodeV1.TRACE_AUTHENTICATION_FAILED,
                reason=f"trace adapter raised {type(error).__name__}: {error}",
            )
            continue
        if not isinstance(normalized, NormalizedTraceV1):
            preliminary_failures[index] = StepFailureV1(
                input_record_digest=source_digest,
                failure_code=StepFailureCodeV1.TRACE_AUTHENTICATION_FAILED,
                reason="trace adapter returned a value outside NormalizedTraceV1",
            )
            continue
        if normalized.input_record_digest != source_digest:
            preliminary_failures[index] = _failure_for_trace(
                normalized,
                StepFailureCodeV1.TRACE_AUTHENTICATION_FAILED,
                "normalized input digest differs from the exact source record digest",
                trusted_input_record_digest=source_digest,
            )
            continue
        normalized_by_index[index] = normalized

    authenticated = tuple(normalized_by_index.values())
    observed_rule_digests = {trace.rule_set_digest for trace in authenticated}
    if len(observed_rule_digests) > 1:
        raise StepDatasetProtocolError("authenticated traces bind more than one rule_set_digest")
    if observed_rule_digests:
        rule_set_digest = next(iter(observed_rule_digests))
    elif expected_rule_set_digest is not None:
        rule_set_digest = expected_rule_set_digest
    else:
        raise StepDatasetProtocolError(
            "an all-failed/empty input requires expected_rule_set_digest"
        )
    _require_sha256(rule_set_digest, "rule_set_digest")
    if expected_rule_set_digest is not None:
        _require_sha256(expected_rule_set_digest, "expected_rule_set_digest")
        if rule_set_digest != expected_rule_set_digest:
            raise StepDatasetProtocolError(
                "authenticated rule-set digest differs from expected_rule_set_digest"
            )

    leakage = _split_leakage(authenticated)
    leaking_trace_indexes = {
        index
        for index, trace in normalized_by_index.items()
        if any(group_id in leakage for group_id in _partition_keys(trace))
    }
    trace_id_counts = Counter(trace.trace_id for trace in authenticated)
    duplicate_trace_indexes = {
        index for index, trace in normalized_by_index.items() if trace_id_counts[trace.trace_id] > 1
    }
    duplicate_input_indexes = {
        index
        for index, trace in normalized_by_index.items()
        if seen_source_digests[trace.input_record_digest] > 1
    }

    accepted: list[StepRecordV1] = []
    failures: list[StepFailureV1] = []
    accepted_trace_count = 0
    authenticated_trace_count = len(authenticated)
    attempted_step_count = 0
    zero_length_count = 0
    for index in range(len(materialized)):
        if index in preliminary_failures:
            failures.append(
                replace(
                    preliminary_failures[index],
                    input_occurrence_index=input_occurrence_indices[index],
                    failure_id="",
                )
            )
            continue
        normalized = normalized_by_index[index]
        if index in leaking_trace_indexes:
            groups = sorted(
                group_id for group_id in _partition_keys(normalized) if group_id in leakage
            )
            failures.append(
                replace(
                    _failure_for_trace(
                        normalized,
                        StepFailureCodeV1.GROUP_LEAKAGE,
                        "lineage group crosses authoritative splits: " + ",".join(groups),
                    ),
                    input_occurrence_index=input_occurrence_indices[index],
                    failure_id="",
                )
            )
            continue
        if index in duplicate_trace_indexes or index in duplicate_input_indexes:
            duplicate_reasons: list[str] = []
            if index in duplicate_trace_indexes:
                duplicate_reasons.append("trace_id")
            if index in duplicate_input_indexes:
                duplicate_reasons.append("exact input record digest")
            failures.append(
                replace(
                    _failure_for_trace(
                        normalized,
                        StepFailureCodeV1.CORRUPT_TRACE,
                        "duplicate authenticated input identity: "
                        + " and ".join(duplicate_reasons),
                    ),
                    input_occurrence_index=input_occurrence_indices[index],
                    failure_id="",
                )
            )
            continue
        rows, trace_failures, _ = _extract_authenticated_trace(
            normalized,
            replayer=replayer,
            state_signature_provider=adapter,
            verifier=verifier,
            expected_verifier_digest=expected_verifier_digest,
        )
        attempted_step_count += len(rows) + sum(
            failure.step_index is not None for failure in trace_failures
        )
        if rows:
            accepted_trace_count += 1
            accepted.extend(rows)
        else:
            failures.extend(
                replace(
                    failure,
                    input_occurrence_index=input_occurrence_indices[index],
                    failure_id="",
                )
                for failure in trace_failures
            )
            if any(
                failure.failure_code is StepFailureCodeV1.ZERO_LENGTH_TRACE
                for failure in trace_failures
            ):
                zero_length_count += 1

    accepted.sort(key=lambda row: (row.trace_id, row.step_index, row.record_id))
    failures.sort(
        key=lambda row: (
            "" if row.trace_id is None else row.trace_id,
            -1 if row.step_index is None else row.step_index,
            row.failure_id,
        )
    )
    failure_counts = tuple(sorted(Counter(row.failure_code.value for row in failures).items()))
    audit = ReplayAuditV1(
        input_trace_count=len(materialized),
        authenticated_trace_count=authenticated_trace_count,
        accepted_trace_count=accepted_trace_count,
        failed_trace_count=len(materialized) - accepted_trace_count,
        zero_length_trace_count=zero_length_count,
        attempted_step_count=attempted_step_count,
        accepted_step_count=len(accepted),
        failure_row_count=len(failures),
        failure_counts=failure_counts,
    )
    return StepExtractionResultV1(
        accepted=tuple(accepted),
        failures=tuple(failures),
        replay_audit=audit,
        input_digest=input_digest,
        rule_set_digest=rule_set_digest,
        verifier_digest=expected_verifier_digest,
    )


@dataclass(frozen=True, slots=True)
class StepRuntimeIdentityV1:
    """Deterministic runtime/provenance fields required in every manifest."""

    git_commit: str
    python_version: str
    hardware: str
    package_versions: tuple[tuple[str, str], ...]
    deterministic_settings: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.git_commit, str)
            or len(self.git_commit) not in {40, 64}
            or self.git_commit.lower() != self.git_commit
            or any(character not in "0123456789abcdef" for character in self.git_commit)
            or len(set(self.git_commit)) < 2
        ):
            raise StepDatasetProtocolError(
                "git_commit must be a concrete 40- or 64-character lowercase hexadecimal SHA"
            )
        _require_nonblank(self.python_version, "python_version")
        _require_nonblank(self.hardware, "hardware")
        if (
            not isinstance(self.package_versions, tuple)
            or any(
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not isinstance(row[1], str)
                or not row[0].strip()
                or not row[1].strip()
                for row in self.package_versions
            )
            or tuple(sorted(self.package_versions)) != self.package_versions
        ):
            raise StepDatasetProtocolError(
                "package_versions must be sorted unique non-blank name/version pairs"
            )
        package_names = tuple(name for name, _ in self.package_versions)
        if len(set(package_names)) != len(package_names):
            raise StepDatasetProtocolError("package version names must be unique")
        _require_sorted_unique_strings(
            self.deterministic_settings,
            "deterministic_settings",
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "deterministic_settings": list(self.deterministic_settings),
            "git_commit": self.git_commit,
            "hardware": self.hardware,
            "package_versions": dict(self.package_versions),
            "python_version": self.python_version,
        }


@dataclass(frozen=True, slots=True)
class StepDatasetConfigV1:
    """Frozen scientific and persistence configuration for one extraction run."""

    seed: int
    shard_size: int
    expected_input_digest: str
    expected_rule_set_digest: str
    expected_verifier_digest: str
    exact_command: str
    runtime: StepRuntimeIdentityV1
    output_schema_version: str = STEP_DATASET_MANIFEST_VERSION
    schema_version: str = STEP_DATASET_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STEP_DATASET_CONFIG_VERSION:
            raise StepDatasetProtocolError("unexpected step config schema")
        if self.output_schema_version != STEP_DATASET_MANIFEST_VERSION:
            raise StepDatasetProtocolError("unexpected output manifest schema")
        _require_nonnegative_int(self.seed, "seed")
        _require_positive_int(self.shard_size, "shard_size")
        _require_sha256(self.expected_input_digest, "expected_input_digest")
        _require_sha256(self.expected_rule_set_digest, "expected_rule_set_digest")
        _require_sha256(self.expected_verifier_digest, "expected_verifier_digest")
        _require_nonblank(self.exact_command, "exact_command")
        if not isinstance(self.runtime, StepRuntimeIdentityV1):
            raise StepDatasetProtocolError("runtime must be StepRuntimeIdentityV1")

    def scientific_payload(self) -> dict[str, object]:
        return {
            "expected_input_digest": self.expected_input_digest,
            "expected_rule_set_digest": self.expected_rule_set_digest,
            "expected_verifier_digest": self.expected_verifier_digest,
            "output_schema_version": self.output_schema_version,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "shard_size": self.shard_size,
        }

    @property
    def config_digest(self) -> str:
        """Hash only the scientific configuration, not host/path metadata."""

        return _tagged_digest(STEP_DATASET_CONFIG_VERSION, self.scientific_payload())

    def as_dict(self) -> dict[str, object]:
        return {
            **self.scientific_payload(),
            "config_digest": self.config_digest,
            "exact_command": self.exact_command,
            "runtime": self.runtime.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class StepShardReceiptV1:
    """Immutable content receipt for one mixed accepted/failure JSONL shard."""

    shard_index: int
    relative_path: str
    row_count: int
    accepted_count: int
    failure_count: int
    byte_count: int
    sha256: str
    first_row_id: str
    last_row_id: str

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.shard_index, "shard_index")
        expected_path = f"shards/shard-{self.shard_index:05d}.jsonl"
        if self.relative_path != expected_path:
            raise StepDatasetProtocolError(
                f"relative_path must be the canonical shard path {expected_path!r}"
            )
        _require_positive_int(self.row_count, "row_count")
        _require_nonnegative_int(self.accepted_count, "accepted_count")
        _require_nonnegative_int(self.failure_count, "failure_count")
        _require_positive_int(self.byte_count, "byte_count")
        _require_sha256(self.sha256, "shard sha256")
        _require_sha256(self.first_row_id, "first_row_id")
        _require_sha256(self.last_row_id, "last_row_id")
        if self.accepted_count + self.failure_count != self.row_count:
            raise StepDatasetProtocolError("shard row type counts do not reconstruct row_count")

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted_count": self.accepted_count,
            "byte_count": self.byte_count,
            "failure_count": self.failure_count,
            "first_row_id": self.first_row_id,
            "last_row_id": self.last_row_id,
            "relative_path": self.relative_path,
            "row_count": self.row_count,
            "sha256": self.sha256,
            "shard_index": self.shard_index,
        }


@dataclass(frozen=True, slots=True)
class StepDatasetManifestV1:
    """Top-level deterministic handoff for the complete step dataset."""

    config_digest: str
    input_digest: str
    rule_set_digest: str
    verifier_digest: str
    dataset_digest: str
    accepted_count: int
    failure_count: int
    shard_receipts: tuple[StepShardReceiptV1, ...]
    sidecar_digests: tuple[tuple[str, str], ...]
    exact_command: str
    runtime: StepRuntimeIdentityV1
    seed: int
    schema_version: str = STEP_DATASET_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STEP_DATASET_MANIFEST_VERSION:
            raise StepDatasetProtocolError("unexpected step dataset manifest schema")
        for label, value in (
            ("config_digest", self.config_digest),
            ("input_digest", self.input_digest),
            ("rule_set_digest", self.rule_set_digest),
            ("verifier_digest", self.verifier_digest),
            ("dataset_digest", self.dataset_digest),
        ):
            _require_sha256(value, label)
        _require_nonnegative_int(self.accepted_count, "accepted_count")
        _require_nonnegative_int(self.failure_count, "failure_count")
        if not isinstance(self.shard_receipts, tuple) or any(
            not isinstance(receipt, StepShardReceiptV1) for receipt in self.shard_receipts
        ):
            raise StepDatasetProtocolError("shard_receipts must be StepShardReceiptV1 values")
        if tuple(receipt.shard_index for receipt in self.shard_receipts) != tuple(
            range(len(self.shard_receipts))
        ):
            raise StepDatasetProtocolError("shard indices must be contiguous from zero")
        if sum(receipt.accepted_count for receipt in self.shard_receipts) != self.accepted_count:
            raise StepDatasetProtocolError("manifest accepted_count mismatch")
        if sum(receipt.failure_count for receipt in self.shard_receipts) != self.failure_count:
            raise StepDatasetProtocolError("manifest failure_count mismatch")
        if (
            not isinstance(self.sidecar_digests, tuple)
            or any(
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not isinstance(row[1], str)
                for row in self.sidecar_digests
            )
            or tuple(sorted(set(self.sidecar_digests))) != self.sidecar_digests
        ):
            raise StepDatasetProtocolError("sidecar_digests must be sorted unique pairs")
        for name, digest in self.sidecar_digests:
            _require_nonblank(name, "sidecar name")
            _require_sha256(digest, f"sidecar digest for {name}")
        expected_sidecars = {
            "config.json",
            "per-rule-manifest.json",
            "replay-audit.json",
            "split-audit.json",
        }
        if {name for name, _ in self.sidecar_digests} != expected_sidecars:
            raise StepDatasetProtocolError("sidecar_digests must contain the four frozen sidecars")
        _require_nonblank(self.exact_command, "exact_command")
        if not isinstance(self.runtime, StepRuntimeIdentityV1):
            raise StepDatasetProtocolError("runtime must be StepRuntimeIdentityV1")
        _require_nonnegative_int(self.seed, "seed")

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted_count": self.accepted_count,
            "config_digest": self.config_digest,
            "dataset_digest": self.dataset_digest,
            "exact_command": self.exact_command,
            "failure_count": self.failure_count,
            "input_digest": self.input_digest,
            "rule_set_digest": self.rule_set_digest,
            "runtime": self.runtime.as_dict(),
            "schema_version": self.schema_version,
            "seed": self.seed,
            "shard_receipts": [receipt.as_dict() for receipt in self.shard_receipts],
            "sidecar_digests": dict(self.sidecar_digests),
            "verifier_digest": self.verifier_digest,
        }


def _dataset_digest(
    *,
    accepted_count: int,
    failure_count: int,
    input_digest: str,
    rule_set_digest: str,
    verifier_digest: str,
    receipts: Sequence[StepShardReceiptV1],
    sidecar_digests: Sequence[tuple[str, str]],
) -> str:
    return _tagged_digest(
        "geml-step-dataset-content-v1",
        {
            "accepted_count": accepted_count,
            "failure_count": failure_count,
            "input_digest": input_digest,
            "receipts": [receipt.as_dict() for receipt in receipts],
            "rule_set_digest": rule_set_digest,
            "sidecar_digests": dict(sidecar_digests),
            "verifier_digest": verifier_digest,
        },
    )


def _row_envelope(row: StepRecordV1 | StepFailureV1) -> dict[str, object]:
    if isinstance(row, StepRecordV1):
        return {
            "row": row.as_dict(),
            "row_id": row.record_id,
            "row_type": "accepted",
            "schema_version": STEP_SHARD_SCHEMA_VERSION,
        }
    return {
        "row": row.as_dict(),
        "row_id": row.failure_id,
        "row_type": "failure",
        "schema_version": STEP_SHARD_SCHEMA_VERSION,
    }


def _immutable_write(path: Path, data: bytes, *, resume: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if resume and path.is_file() and path.read_bytes() == data:
                return
            raise ResumeMismatchError(
                f"existing immutable output differs from resumed bytes: {path}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _json_file_bytes(value: object) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _shard_rows(
    rows: Sequence[StepRecordV1 | StepFailureV1],
    shard_size: int,
) -> tuple[tuple[StepRecordV1 | StepFailureV1, ...], ...]:
    return tuple(
        tuple(rows[start : start + shard_size]) for start in range(0, len(rows), shard_size)
    )


def _split_audit_payload(
    accepted: Sequence[StepRecordV1],
    failures: Sequence[StepFailureV1],
) -> dict[str, object]:
    def accumulate(
        rows: Sequence[StepRecordV1 | StepFailureV1],
    ) -> dict[str, set[str]]:
        memberships: dict[str, set[str]] = {}
        for row in rows:
            if row.authoritative_split is None:
                continue
            keys = [f"lineage:{group_id}" for group_id in row.lineage_group_ids]
            if row.trace_id is not None:
                keys.append(f"trace:{row.trace_id}")
            if row.pair_id is not None:
                keys.append(f"pair:{row.pair_id}")
            if row.source_id is not None:
                keys.append(f"source:{row.source_id}")
            for key in keys:
                memberships.setdefault(key, set()).add(row.authoritative_split.value)
        return memberships

    input_memberships = accumulate((*accepted, *failures))
    accepted_memberships = accumulate(accepted)
    detected_input_leaks = {
        group_id: sorted(splits)
        for group_id, splits in sorted(input_memberships.items())
        if len(splits) > 1
    }
    accepted_leaks = {
        group_id: sorted(splits)
        for group_id, splits in sorted(accepted_memberships.items())
        if len(splits) > 1
    }
    return {
        "accepted_group_count": len(accepted_memberships),
        "accepted_leakage_count": len(accepted_leaks),
        "accepted_leakages": accepted_leaks,
        "audited_input_group_count": len(input_memberships),
        "detected_input_leakage_count": len(detected_input_leaks),
        "detected_input_leakages": detected_input_leaks,
        "schema_version": "geml-step-split-audit-v1",
        "status": "passed" if not accepted_leaks else "failed",
    }


_TEMPORARY_NAME_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


def _is_writer_orphan_temporary(
    relative_path: str,
    *,
    allowed_files: set[str],
) -> bool:
    """Recognize only the exact temporary names emitted for an allowed output."""

    path = Path(relative_path)
    for allowed in allowed_files:
        destination = Path(allowed)
        if path.parent != destination.parent:
            continue
        prefix = f".{destination.name}."
        if not path.name.startswith(prefix) or not path.name.endswith(".tmp"):
            continue
        token = path.name[len(prefix) : -len(".tmp")]
        return len(token) == 8 and all(
            character in _TEMPORARY_NAME_CHARACTERS for character in token
        )
    return False


def write_step_dataset(
    result: StepExtractionResultV1,
    *,
    output_root: str | Path,
    config: StepDatasetConfigV1,
    frozen_registry: object,
    resume: bool = True,
) -> StepDatasetManifestV1:
    """Write immutable resumable shards and complete deterministic sidecars."""

    if not isinstance(result, StepExtractionResultV1):
        raise TypeError("result must be StepExtractionResultV1")
    if not isinstance(config, StepDatasetConfigV1):
        raise TypeError("config must be StepDatasetConfigV1")
    if type(resume) is not bool:
        raise TypeError("resume must be a boolean")
    if config.expected_input_digest != result.input_digest:
        raise StepDatasetProtocolError("config/input digest mismatch")
    if config.expected_rule_set_digest != result.rule_set_digest:
        raise StepDatasetProtocolError("config/rule-set digest mismatch")
    if config.expected_verifier_digest != result.verifier_digest:
        raise StepDatasetProtocolError("config/verifier digest mismatch")

    from geml.data.steps.stratify import (  # local import avoids a contract cycle
        RuleRegistrySnapshotV1,
        build_stratification_report,
    )

    registry = frozen_registry
    if not isinstance(registry, RuleRegistrySnapshotV1):
        raise TypeError("frozen_registry must be RuleRegistrySnapshotV1")
    if registry.authoritative_rule_set_digest != result.rule_set_digest:
        raise StepDatasetProtocolError(
            "frozen registry authoritative digest does not match result.rule_set_digest"
        )
    split_audit = _split_audit_payload(result.accepted, ())
    if split_audit["accepted_leakage_count"] != 0:
        raise SplitLeakageError(
            "accepted rows cross an authoritative pair/source/trace/lineage split boundary"
        )
    registry_by_key = {(entry.rule_id, entry.direction): entry for entry in registry.entries}
    for row in result.accepted:
        key = (row.rule_id, row.direction.value)
        entry = registry_by_key.get(key)
        if entry is None:
            raise StepDatasetProtocolError(
                f"accepted row references a rule/direction absent from the frozen registry: {key!r}"
            )
        if not entry.supported:
            raise StepDatasetProtocolError(
                f"accepted row references an unsupported frozen registry entry: {key!r}"
            )
    root = Path(output_root)
    rows: tuple[StepRecordV1 | StepFailureV1, ...] = tuple(
        sorted(
            (*result.accepted, *result.failures),
            key=lambda row: (
                row.trace_id or "" if isinstance(row, StepFailureV1) else row.trace_id,
                -1 if isinstance(row, StepFailureV1) and row.step_index is None else row.step_index,
                row.failure_id if isinstance(row, StepFailureV1) else row.record_id,
            ),
        )
    )
    shards = _shard_rows(rows, config.shard_size)
    expected_shard_names = {f"shards/shard-{index:05d}.jsonl" for index in range(len(shards))}
    allowed_files = expected_shard_names | {
        "config.json",
        "manifest.json",
        "per-rule-manifest.json",
        "replay-audit.json",
        "split-audit.json",
    }
    if root.is_dir():
        existing_paths = tuple(
            path for path in root.rglob("*") if path.is_file() or path.is_symlink()
        )
        existing_files: set[str] = set()
        for path in existing_paths:
            relative = path.relative_to(root).as_posix()
            if (
                path.is_file()
                and not path.is_symlink()
                and _is_writer_orphan_temporary(
                    relative,
                    allowed_files=allowed_files,
                )
            ):
                path.unlink()
                continue
            existing_files.add(relative)
        unexpected_files = sorted(existing_files - allowed_files)
        if unexpected_files:
            raise ResumeMismatchError(
                f"resume found unexpected files under the output root: {unexpected_files}"
            )
    shard_dir = root / "shards"
    if shard_dir.is_dir():
        existing_names = {
            path.relative_to(root).as_posix()
            for path in shard_dir.glob("*.jsonl")
            if path.is_file()
        }
        extras = sorted(existing_names - expected_shard_names)
        if extras:
            raise ResumeMismatchError(f"resume found stale unexpected shard files: {extras}")

    receipts: list[StepShardReceiptV1] = []
    for index, shard in enumerate(shards):
        envelopes = tuple(_row_envelope(row) for row in shard)
        data = b"".join(_canonical_bytes(envelope) + b"\n" for envelope in envelopes)
        relative = f"shards/shard-{index:05d}.jsonl"
        _immutable_write(root / relative, data, resume=resume)
        receipts.append(
            StepShardReceiptV1(
                shard_index=index,
                relative_path=relative,
                row_count=len(shard),
                accepted_count=sum(isinstance(row, StepRecordV1) for row in shard),
                failure_count=sum(isinstance(row, StepFailureV1) for row in shard),
                byte_count=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                first_row_id=envelopes[0]["row_id"],
                last_row_id=envelopes[-1]["row_id"],
            )
        )

    report = build_stratification_report(
        result.accepted,
        result.failures,
        registry,
    )
    sidecar_payloads = {
        "config.json": config.as_dict(),
        "per-rule-manifest.json": report.as_dict(),
        "replay-audit.json": result.replay_audit.as_dict(),
        "split-audit.json": _split_audit_payload(result.accepted, result.failures),
    }
    sidecar_digests: list[tuple[str, str]] = []
    for name, payload in sorted(sidecar_payloads.items()):
        data = _json_file_bytes(payload)
        _immutable_write(root / name, data, resume=resume)
        sidecar_digests.append((name, hashlib.sha256(data).hexdigest()))

    dataset_digest = _dataset_digest(
        accepted_count=len(result.accepted),
        failure_count=len(result.failures),
        input_digest=result.input_digest,
        rule_set_digest=result.rule_set_digest,
        verifier_digest=result.verifier_digest,
        receipts=receipts,
        sidecar_digests=sidecar_digests,
    )
    manifest = StepDatasetManifestV1(
        config_digest=config.config_digest,
        input_digest=result.input_digest,
        rule_set_digest=result.rule_set_digest,
        verifier_digest=result.verifier_digest,
        dataset_digest=dataset_digest,
        accepted_count=len(result.accepted),
        failure_count=len(result.failures),
        shard_receipts=tuple(receipts),
        sidecar_digests=tuple(sidecar_digests),
        exact_command=config.exact_command,
        runtime=config.runtime,
        seed=config.seed,
    )
    _immutable_write(
        root / "manifest.json",
        _json_file_bytes(manifest.as_dict()),
        resume=resume,
    )
    return manifest


def _parse_runtime(value: object) -> StepRuntimeIdentityV1:
    if not isinstance(value, dict) or set(value) != {
        "deterministic_settings",
        "git_commit",
        "hardware",
        "package_versions",
        "python_version",
    }:
        raise StepDatasetProtocolError("manifest runtime fields differ from schema")
    packages = value["package_versions"]
    if not isinstance(packages, dict) or any(
        not isinstance(key, str) or not isinstance(version, str)
        for key, version in packages.items()
    ):
        raise StepDatasetProtocolError("runtime package_versions must be an object")
    return StepRuntimeIdentityV1(
        git_commit=value["git_commit"],
        python_version=value["python_version"],
        hardware=value["hardware"],
        package_versions=tuple(sorted(packages.items())),
        deterministic_settings=_string_tuple(
            value["deterministic_settings"],
            "deterministic_settings",
        ),
    )


def _parse_step_dataset_config(value: object) -> StepDatasetConfigV1:
    expected_fields = {
        "config_digest",
        "exact_command",
        "expected_input_digest",
        "expected_rule_set_digest",
        "expected_verifier_digest",
        "output_schema_version",
        "runtime",
        "schema_version",
        "seed",
        "shard_size",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise StepDatasetProtocolError("config sidecar fields differ from frozen schema")
    config = StepDatasetConfigV1(
        seed=value["seed"],
        shard_size=value["shard_size"],
        expected_input_digest=value["expected_input_digest"],
        expected_rule_set_digest=value["expected_rule_set_digest"],
        expected_verifier_digest=value["expected_verifier_digest"],
        exact_command=value["exact_command"],
        runtime=_parse_runtime(value["runtime"]),
        output_schema_version=value["output_schema_version"],
        schema_version=value["schema_version"],
    )
    claimed_digest = _require_sha256(value["config_digest"], "config sidecar config_digest")
    if claimed_digest != config.config_digest:
        raise StepDatasetProtocolError(
            "config sidecar config_digest does not bind its scientific fields"
        )
    return config


def _parse_shard_rows(
    shard_data: bytes,
    receipt: StepShardReceiptV1,
) -> tuple[StepRecordV1 | StepFailureV1, ...]:
    if (
        len(shard_data) != receipt.byte_count
        or hashlib.sha256(shard_data).hexdigest() != receipt.sha256
    ):
        raise StepDatasetProtocolError(f"shard bytes differ from receipt: {receipt.relative_path}")
    if not shard_data.endswith(b"\n"):
        raise StepDatasetProtocolError(
            f"shard lacks its canonical trailing LF: {receipt.relative_path}"
        )

    rows: list[StepRecordV1 | StepFailureV1] = []
    for line in shard_data[:-1].split(b"\n"):
        try:
            envelope = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StepDatasetProtocolError(
                f"shard contains invalid JSON: {receipt.relative_path}"
            ) from error
        if line != _canonical_bytes(envelope):
            raise StepDatasetProtocolError(
                f"shard contains non-canonical JSON: {receipt.relative_path}"
            )
        if not isinstance(envelope, dict) or set(envelope) != {
            "row",
            "row_id",
            "row_type",
            "schema_version",
        }:
            raise StepDatasetProtocolError("shard envelope fields differ from schema")
        if envelope["schema_version"] != STEP_SHARD_SCHEMA_VERSION:
            raise StepDatasetProtocolError("unexpected shard schema version")
        payload = envelope["row"]
        if not isinstance(payload, dict):
            raise StepDatasetProtocolError("shard row payload must be an object")
        try:
            if envelope["row_type"] == "accepted":
                row: StepRecordV1 | StepFailureV1 = StepRecordV1.from_dict(payload)
                row_id = row.record_id
            elif envelope["row_type"] == "failure":
                row = StepFailureV1.from_dict(payload)
                row_id = row.failure_id
            else:
                raise StepDatasetProtocolError("unsupported shard row_type")
        except StepDatasetProtocolError:
            raise
        except (TypeError, ValueError) as error:
            raise StepDatasetProtocolError(
                f"shard contains an invalid typed row: {receipt.relative_path}"
            ) from error
        if envelope["row_id"] != row_id:
            raise StepDatasetProtocolError("shard envelope row_id mismatch")
        rows.append(row)

    accepted_count = sum(isinstance(row, StepRecordV1) for row in rows)
    failure_count = len(rows) - accepted_count
    row_ids = tuple(
        row.record_id if isinstance(row, StepRecordV1) else row.failure_id for row in rows
    )
    if (
        len(rows) != receipt.row_count
        or accepted_count != receipt.accepted_count
        or failure_count != receipt.failure_count
        or row_ids[0] != receipt.first_row_id
        or row_ids[-1] != receipt.last_row_id
    ):
        raise StepDatasetProtocolError(
            f"shard row accounting differs from receipt: {receipt.relative_path}"
        )
    return tuple(rows)


def _load_receipt_rows(
    root: Path,
    receipt: StepShardReceiptV1,
) -> tuple[StepRecordV1 | StepFailureV1, ...]:
    return _parse_shard_rows((root / receipt.relative_path).read_bytes(), receipt)


def load_step_dataset_manifest(
    path: str | Path,
    *,
    expected_config_digest: str | None = None,
    expected_input_digest: str | None = None,
    expected_rule_set_digest: str | None = None,
    expected_verifier_digest: str | None = None,
    verify_files: bool = True,
) -> StepDatasetManifestV1:
    """Load, authenticate, and optionally recursively verify a dataset manifest."""

    manifest_path = Path(path)
    if manifest_path.name != "manifest.json":
        raise StepDatasetProtocolError("dataset manifest must use the canonical name manifest.json")
    data = manifest_path.read_bytes()
    if not data.endswith(b"\n"):
        raise StepDatasetProtocolError("manifest must have exactly one trailing LF")
    try:
        value = json.loads(data[:-1])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StepDatasetProtocolError("manifest is not valid UTF-8 JSON") from error
    if data != _json_file_bytes(value):
        raise StepDatasetProtocolError("manifest bytes are not canonical")
    if not isinstance(value, dict) or set(value) != {
        "accepted_count",
        "config_digest",
        "dataset_digest",
        "exact_command",
        "failure_count",
        "input_digest",
        "rule_set_digest",
        "runtime",
        "schema_version",
        "seed",
        "shard_receipts",
        "sidecar_digests",
        "verifier_digest",
    }:
        raise StepDatasetProtocolError("manifest fields differ from frozen schema")
    raw_receipts = value["shard_receipts"]
    if not isinstance(raw_receipts, list):
        raise StepDatasetProtocolError("shard_receipts must be a JSON array")
    receipts: list[StepShardReceiptV1] = []
    for raw in raw_receipts:
        if not isinstance(raw, dict) or set(raw) != {
            "accepted_count",
            "byte_count",
            "failure_count",
            "first_row_id",
            "last_row_id",
            "relative_path",
            "row_count",
            "sha256",
            "shard_index",
        }:
            raise StepDatasetProtocolError("shard receipt fields differ from schema")
        receipts.append(StepShardReceiptV1(**raw))
    sidecars = value["sidecar_digests"]
    if not isinstance(sidecars, dict) or any(
        not isinstance(name, str) or not isinstance(digest, str)
        for name, digest in sidecars.items()
    ):
        raise StepDatasetProtocolError("sidecar_digests must be an object")
    manifest = StepDatasetManifestV1(
        config_digest=value["config_digest"],
        input_digest=value["input_digest"],
        rule_set_digest=value["rule_set_digest"],
        verifier_digest=value["verifier_digest"],
        dataset_digest=value["dataset_digest"],
        accepted_count=value["accepted_count"],
        failure_count=value["failure_count"],
        shard_receipts=tuple(receipts),
        sidecar_digests=tuple(sorted(sidecars.items())),
        exact_command=value["exact_command"],
        runtime=_parse_runtime(value["runtime"]),
        seed=value["seed"],
        schema_version=value["schema_version"],
    )
    expected_dataset_digest = _dataset_digest(
        accepted_count=manifest.accepted_count,
        failure_count=manifest.failure_count,
        input_digest=manifest.input_digest,
        rule_set_digest=manifest.rule_set_digest,
        verifier_digest=manifest.verifier_digest,
        receipts=manifest.shard_receipts,
        sidecar_digests=manifest.sidecar_digests,
    )
    if manifest.dataset_digest != expected_dataset_digest:
        raise StepDatasetProtocolError(
            "manifest dataset_digest does not bind its receipts and sidecars"
        )
    for label, expected, observed in (
        ("config", expected_config_digest, manifest.config_digest),
        ("input", expected_input_digest, manifest.input_digest),
        ("rule-set", expected_rule_set_digest, manifest.rule_set_digest),
        ("verifier", expected_verifier_digest, manifest.verifier_digest),
    ):
        if expected is not None:
            _require_sha256(expected, f"expected {label} digest")
            if expected != observed:
                raise StepDatasetProtocolError(
                    f"manifest {label} digest differs from expected digest"
                )
    if verify_files:
        root = manifest_path.parent
        allowed_files = {
            "manifest.json",
            *(receipt.relative_path for receipt in manifest.shard_receipts),
            *(name for name, _ in manifest.sidecar_digests),
        }
        observed_files: set[str] = set()
        for observed_path in root.rglob("*"):
            if not (observed_path.is_file() or observed_path.is_symlink()):
                continue
            relative = observed_path.relative_to(root).as_posix()
            if observed_path.is_symlink():
                raise StepDatasetProtocolError(
                    f"dataset bundle contains a symbolic-link file: {relative}"
                )
            observed_files.add(relative)
        unexpected_files = sorted(observed_files - allowed_files)
        if unexpected_files:
            raise StepDatasetProtocolError(
                f"dataset bundle contains unlisted files: {unexpected_files}"
            )
        missing_files = sorted(allowed_files - observed_files)
        if missing_files:
            raise StepDatasetProtocolError(
                f"dataset bundle is missing listed files: {missing_files}"
            )

        rows: list[StepRecordV1 | StepFailureV1] = []
        for receipt in manifest.shard_receipts:
            rows.extend(_load_receipt_rows(root, receipt))

        sidecar_values: dict[str, object] = {}
        for name, digest in manifest.sidecar_digests:
            sidecar_data = (root / name).read_bytes()
            if hashlib.sha256(sidecar_data).hexdigest() != digest:
                raise StepDatasetProtocolError(f"sidecar bytes differ from manifest digest: {name}")
            if not sidecar_data.endswith(b"\n"):
                raise StepDatasetProtocolError(f"sidecar lacks its canonical trailing LF: {name}")
            try:
                sidecar_value = json.loads(sidecar_data[:-1])
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise StepDatasetProtocolError(f"sidecar is not valid JSON: {name}") from error
            if sidecar_data != _json_file_bytes(sidecar_value):
                raise StepDatasetProtocolError(f"sidecar is not canonical JSON: {name}")
            sidecar_values[name] = sidecar_value

        config = _parse_step_dataset_config(sidecar_values["config.json"])
        if config.config_digest != manifest.config_digest:
            raise StepDatasetProtocolError("config sidecar digest differs from manifest")
        if config.expected_input_digest != manifest.input_digest:
            raise StepDatasetProtocolError("config input digest differs from manifest")
        if config.expected_rule_set_digest != manifest.rule_set_digest:
            raise StepDatasetProtocolError("config rule-set digest differs from manifest")
        if config.expected_verifier_digest != manifest.verifier_digest:
            raise StepDatasetProtocolError("config verifier digest differs from manifest")
        if config.exact_command != manifest.exact_command:
            raise StepDatasetProtocolError("config exact command differs from manifest")
        if config.runtime != manifest.runtime:
            raise StepDatasetProtocolError("config runtime identity differs from manifest")
        if config.seed != manifest.seed:
            raise StepDatasetProtocolError("config seed differs from manifest")
        for receipt in manifest.shard_receipts[:-1]:
            if receipt.row_count != config.shard_size:
                raise StepDatasetProtocolError(
                    "non-final shard row count differs from config shard_size"
                )
        if manifest.shard_receipts and manifest.shard_receipts[-1].row_count > config.shard_size:
            raise StepDatasetProtocolError("final shard exceeds config shard_size")

        accepted_rows = tuple(row for row in rows if isinstance(row, StepRecordV1))
        failure_rows = tuple(row for row in rows if isinstance(row, StepFailureV1))
        if len(accepted_rows) != manifest.accepted_count:
            raise StepDatasetProtocolError("loaded accepted row count differs from manifest")
        if len(failure_rows) != manifest.failure_count:
            raise StepDatasetProtocolError("loaded failure row count differs from manifest")
        if any(row.rule_set_digest != manifest.rule_set_digest for row in accepted_rows):
            raise StepDatasetProtocolError("accepted row rule_set_digest differs from manifest")
        if any(row.verifier_digest != manifest.verifier_digest for row in accepted_rows):
            raise StepDatasetProtocolError("accepted row verifier_digest differs from manifest")

        observed_replay_audit = ReplayAuditV1.from_dict(sidecar_values["replay-audit.json"])
        expected_replay_audit = _replay_audit_from_rows(
            accepted_rows,
            failure_rows,
        )
        if observed_replay_audit != expected_replay_audit:
            raise StepDatasetProtocolError(
                "replay audit does not reconstruct from the authenticated rows"
            )

        expected_split_audit = _split_audit_payload(accepted_rows, failure_rows)
        if _canonical_bytes(sidecar_values["split-audit.json"]) != _canonical_bytes(
            expected_split_audit
        ):
            raise StepDatasetProtocolError(
                "split audit does not reconstruct from the authenticated rows"
            )

        from geml.data.steps.stratify import (
            RuleRegistrySnapshotV1,
            StratificationProtocolError,
            build_stratification_report,
        )

        raw_stratification = sidecar_values["per-rule-manifest.json"]
        if not isinstance(raw_stratification, Mapping):
            raise StepDatasetProtocolError("per-rule manifest must be a JSON object")
        try:
            registry = RuleRegistrySnapshotV1.from_dict(raw_stratification.get("registry"))
            expected_stratification = build_stratification_report(
                accepted_rows,
                failure_rows,
                registry,
            )
        except StratificationProtocolError as error:
            raise StepDatasetProtocolError(
                f"per-rule manifest violates its typed schema: {error}"
            ) from error
        if registry.authoritative_rule_set_digest != manifest.rule_set_digest:
            raise StepDatasetProtocolError(
                "per-rule registry authoritative digest differs from manifest"
            )
        if _canonical_bytes(raw_stratification) != _canonical_bytes(
            expected_stratification.as_dict()
        ):
            raise StepDatasetProtocolError(
                "per-rule manifest does not reconstruct from the authenticated rows"
            )
    return manifest


def load_step_rows(
    manifest_path: str | Path,
) -> tuple[StepRecordV1 | StepFailureV1, ...]:
    """Authenticate a manifest and parse every typed shard row."""

    manifest = load_step_dataset_manifest(manifest_path)
    root = Path(manifest_path).parent
    return tuple(
        row for receipt in manifest.shard_receipts for row in _load_receipt_rows(root, receipt)
    )


def dataset_tree_digest(output_root: str | Path) -> str:
    """Hash names and exact bytes of every published file for two-run checks."""

    root = Path(output_root)
    if not root.is_dir() or not (root / "manifest.json").is_file():
        raise StepDatasetProtocolError("dataset_tree_digest requires a published dataset root")
    digest = hashlib.sha256(b"geml-step-dataset-tree-v1\0")
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()
