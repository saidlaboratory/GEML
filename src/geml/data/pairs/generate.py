"""Concrete, replayable equivalence-pair contracts for Goal 6.

Goal 4 saturation provenance is intentionally not treated as a proof trace: its
e-class identifiers are not stable application sites.  This module therefore
accepts only directed concrete trajectories whose actions identify a source-tree
occurrence by ordered child-slot path.  Production adapters may create these
records from the read-only rule engine; tests use injected replay verifiers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Protocol, Self, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StringConstraints,
    model_validator,
)

from geml.contracts.corpus import CorpusSplit

PAIR_SCHEMA_VERSION = "geml-pair-record-v1"
DERIVED_EXPRESSION_SCHEMA_VERSION = "geml-derived-expression-v1"
REWRITE_ACTION_SCHEMA_VERSION = "geml-rewrite-action-v1"
REWRITE_TRACE_SCHEMA_VERSION = "geml-rewrite-trace-v1"
PAIR_FIXTURE_MANIFEST_SCHEMA_VERSION = "geml-goal6-pair-fixture-manifest-v1"

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_Sha256Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]


class PairContractError(ValueError):
    """Raised when a pair cannot be represented as scientifically valid evidence."""


class PairStatus(StrEnum):
    """Retained status for both accepted and rejected candidate pairs."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class VerificationTier(StrEnum):
    """Strength of the evidence attached to a candidate record."""

    REPLAYED_RULE_ENGINE = "replayed_rule_engine"
    FORMAL_COUNTEREXAMPLE = "formal_counterexample"
    NUMERIC_COUNTEREXAMPLE = "numeric_counterexample"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class ReplayStatus(StrEnum):
    """Complete replay outcome for a directed trace."""

    PASSED = "passed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"


class _PairContract(BaseModel):
    """Fail-closed base for all persisted Goal 6 pair evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, strict=True)


def canonical_json_bytes(value: object) -> bytes:
    """Return strict, stable JSON bytes for scientific identifiers and content hashes."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PairContractError(f"value cannot be encoded as canonical JSON: {error}") from error


def sha256_digest(value: bytes) -> str:
    """Return a qualified SHA-256 digest rather than an unstable Python hash."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


class GroupLineageV1(_PairContract):
    """Source/e-class-relative grouping used to prevent split leakage."""

    group_id: _NonBlankStr
    relative_group_ids: tuple[_NonBlankStr, ...] = ()
    source_split: CorpusSplit

    @model_validator(mode="after")
    def validate_relatives(self) -> Self:
        if len(set(self.relative_group_ids)) != len(self.relative_group_ids):
            raise ValueError("relative_group_ids must not contain duplicates")
        if self.group_id in self.relative_group_ids:
            raise ValueError("relative_group_ids must not repeat group_id")
        if tuple(sorted(self.relative_group_ids)) != self.relative_group_ids:
            raise ValueError("relative_group_ids must be sorted canonically")
        return self

    @property
    def closure(self) -> tuple[str, ...]:
        """Return the canonical lineage closure including the primary group."""

        return tuple(sorted((self.group_id, *self.relative_group_ids)))


class ExpressionReferenceV1(_PairContract):
    """An endpoint identity with enough source context for replay and leakage audit."""

    expression_id: _NonBlankStr
    sympy_srepr: _NonBlankStr
    structural_signature: _NonBlankStr
    domain_mode: _NonBlankStr
    operator_family: _NonBlankStr
    source_split: CorpusSplit
    group: GroupLineageV1

    @model_validator(mode="after")
    def validate_group_split(self) -> Self:
        if self.group.source_split != self.source_split:
            raise ValueError("endpoint group source_split must match endpoint source_split")
        return self


class DerivedExpressionV1(_PairContract):
    """A trace-derived endpoint that is not added to the immutable Goal 1 corpus."""

    schema_version: str = DERIVED_EXPRESSION_SCHEMA_VERSION
    expression: ExpressionReferenceV1
    source_expression_id: _NonBlankStr
    source_group_id: _NonBlankStr
    rule_set_digest: _Sha256Digest
    generation_seed: StrictInt
    trace_digest: _Sha256Digest

    @model_validator(mode="after")
    def validate_inherited_group(self) -> Self:
        if self.expression.group.group_id != self.source_group_id:
            raise ValueError("derived expression must inherit its source group")
        return self


class OrderedBindingV1(_PairContract):
    """One ordered concrete argument binding for a rewrite action."""

    argument_index: _NonNegativeInt
    name: _NonBlankStr
    value: JsonValue


class RewriteActionV1(_PairContract):
    """A concrete action whose site is a stable child-slot path, never an e-class ID."""

    schema_version: str = REWRITE_ACTION_SCHEMA_VERSION
    rule_id: _NonBlankStr
    direction: _NonBlankStr
    occurrence_path: tuple[_NonNegativeInt, ...]
    bindings: tuple[OrderedBindingV1, ...] = ()
    source_structural_signature: _NonBlankStr
    successor_structural_signature: _NonBlankStr
    rule_assumptions: tuple[_NonBlankStr, ...] = ()
    domain_mode: _NonBlankStr
    semantic_digest: _Sha256Digest

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        indexes = tuple(binding.argument_index for binding in self.bindings)
        if indexes != tuple(range(len(self.bindings))):
            raise ValueError("binding argument_index values must be dense and ordered")
        names = tuple(binding.name for binding in self.bindings)
        if len(set(names)) != len(names):
            raise ValueError("binding names must be unique")
        if tuple(sorted(set(self.rule_assumptions))) != self.rule_assumptions:
            raise ValueError("rule_assumptions must be sorted and unique")
        if self.semantic_digest != self.expected_semantic_digest():
            raise ValueError(
                "semantic_digest does not bind the canonical action and source context"
            )
        return self

    def identity_payload(self) -> dict[str, object]:
        """Return exactly the action identity fields, excluding mutable execution evidence."""

        return {
            "bindings": [binding.model_dump(mode="json") for binding in self.bindings],
            "direction": self.direction,
            "domain_mode": self.domain_mode,
            "occurrence_path": list(self.occurrence_path),
            "rule_assumptions": list(self.rule_assumptions),
            "rule_id": self.rule_id,
            "schema_version": self.schema_version,
            "source_structural_signature": self.source_structural_signature,
            "successor_structural_signature": self.successor_structural_signature,
        }

    def expected_semantic_digest(self) -> str:
        """Compute the versioned action digest without status, timestamps, or verifier output."""

        return sha256_digest(canonical_json_bytes(self.identity_payload()))

    @classmethod
    def create(
        cls,
        *,
        rule_id: str,
        direction: str,
        occurrence_path: tuple[int, ...],
        bindings: tuple[OrderedBindingV1, ...],
        source_structural_signature: str,
        successor_structural_signature: str,
        rule_assumptions: tuple[str, ...],
        domain_mode: str,
    ) -> Self:
        """Create an action with its canonical digest filled in exactly once."""

        raw = {
            "rule_id": rule_id,
            "direction": direction,
            "occurrence_path": occurrence_path,
            "bindings": bindings,
            "source_structural_signature": source_structural_signature,
            "successor_structural_signature": successor_structural_signature,
            "rule_assumptions": rule_assumptions,
            "domain_mode": domain_mode,
            "semantic_digest": "sha256:" + "0" * 64,
        }
        provisional = cls.model_construct(**raw)
        return cls.model_validate(
            {**raw, "semantic_digest": provisional.expected_semantic_digest()}
        )


class TraceStateV1(_PairContract):
    """A concrete source-expression state in a directed rewrite trajectory."""

    expression_id: _NonBlankStr
    sympy_srepr: _NonBlankStr
    structural_signature: _NonBlankStr


class TransitionVerificationV1(_PairContract):
    """Verification evidence for one state-to-state action transition."""

    action_digest: _Sha256Digest
    source_structural_signature: _NonBlankStr
    successor_structural_signature: _NonBlankStr
    verifier_version: _NonBlankStr
    status: ReplayStatus
    evidence_digest: _Sha256Digest
    detail: str = ""


class RewriteTraceV1(_PairContract):
    """A full directed trajectory generated outside of Goal 4's e-class provenance rows."""

    schema_version: str = REWRITE_TRACE_SCHEMA_VERSION
    source: TraceStateV1
    goal: TraceStateV1
    states: tuple[TraceStateV1, ...] = Field(min_length=1)
    actions: tuple[RewriteActionV1, ...] = ()
    transitions: tuple[TransitionVerificationV1, ...] = ()
    verified_step_count: _NonNegativeInt
    rule_set_digest: _Sha256Digest
    policy_digest: _Sha256Digest
    domain_mode: _NonBlankStr
    generation_seed: StrictInt
    replay_status: ReplayStatus
    failure_type: _NonBlankStr | None = None
    failure_detail: str | None = None

    @model_validator(mode="after")
    def validate_complete_replay(self) -> Self:
        if self.states[0] != self.source:
            raise ValueError("trace source must be the first stored state")
        if self.states[-1] != self.goal:
            raise ValueError("trace goal must be the final stored state")
        if len(self.states) != len(self.actions) + 1:
            raise ValueError("a trace with n actions must contain exactly n + 1 states")
        if len(self.transitions) != len(self.actions):
            raise ValueError("every stored action must have one transition verification")
        if self.verified_step_count != sum(
            transition.status is ReplayStatus.PASSED for transition in self.transitions
        ):
            raise ValueError("verified_step_count must equal the number of passed transitions")
        for index, action in enumerate(self.actions):
            source = self.states[index]
            successor = self.states[index + 1]
            transition = self.transitions[index]
            if action.source_structural_signature != source.structural_signature:
                raise ValueError("action source signature does not match its stored source state")
            if action.successor_structural_signature != successor.structural_signature:
                raise ValueError("action successor signature does not match its stored next state")
            if transition.action_digest != action.semantic_digest:
                raise ValueError("transition action_digest does not match the action")
            if transition.source_structural_signature != source.structural_signature:
                raise ValueError("transition source signature does not match trace state")
            if transition.successor_structural_signature != successor.structural_signature:
                raise ValueError("transition successor signature does not match trace state")
        if self.replay_status is ReplayStatus.PASSED:
            if any(transition.status is not ReplayStatus.PASSED for transition in self.transitions):
                raise ValueError("passed trace cannot contain a failed transition")
            if self.failure_type is not None or self.failure_detail is not None:
                raise ValueError("passed trace cannot carry failure details")
        elif self.failure_type is None:
            raise ValueError("non-passed trace must retain a typed failure_type")
        return self

    @property
    def trace_digest(self) -> str:
        """Content digest used by pair and derived-expression provenance."""

        return sha256_digest(canonical_json_bytes(self.model_dump(mode="json")))


@runtime_checkable
class TransitionVerifier(Protocol):
    """Injected production verifier boundary for concrete-action replay."""

    def __call__(
        self,
        state: TraceStateV1,
        action: RewriteActionV1,
        expected_successor: TraceStateV1,
        *,
        domain_mode: str,
    ) -> TransitionVerificationV1: ...


def replay_trace(trace: RewriteTraceV1, verifier: TransitionVerifier) -> RewriteTraceV1:
    """Replay each concrete action and return fresh transition evidence.

    The caller supplies the authoritative rule-engine/verifier bridge.  This
    avoids inventing a generic semantic oracle from Goal 4 e-class state.
    """

    transitions = tuple(
        verifier(
            trace.states[index],
            action,
            trace.states[index + 1],
            domain_mode=trace.domain_mode,
        )
        for index, action in enumerate(trace.actions)
    )
    status = (
        ReplayStatus.PASSED
        if all(transition.status is ReplayStatus.PASSED for transition in transitions)
        else next(
            transition.status
            for transition in transitions
            if transition.status is not ReplayStatus.PASSED
        )
    )
    return RewriteTraceV1.model_validate(
        {
            **trace.model_dump(),
            "transitions": transitions,
            "verified_step_count": sum(
                transition.status is ReplayStatus.PASSED for transition in transitions
            ),
            "replay_status": status,
            "failure_type": None if status is ReplayStatus.PASSED else "replay_failed",
            "failure_detail": (
                None if status is ReplayStatus.PASSED else "one or more transitions failed"
            ),
        }
    )


class NonEquivalenceEvidenceV1(_PairContract):
    """Positive evidence for a negative pair; numeric disagreement is not automatically formal."""

    tier: VerificationTier
    evidence_digest: _Sha256Digest
    method: _NonBlankStr
    detail: _NonBlankStr
    rigorous: StrictBool

    @model_validator(mode="after")
    def validate_tier(self) -> Self:
        if self.tier is VerificationTier.FORMAL_COUNTEREXAMPLE and not self.rigorous:
            raise ValueError("formal_counterexample evidence must be rigorous")
        if self.tier is VerificationTier.NUMERIC_COUNTEREXAMPLE and self.rigorous:
            raise ValueError(
                "numeric_counterexample must not be labeled rigorous without formal bounds"
            )
        return self


class PairRecordV1(_PairContract):
    """One accepted or retained-rejected equivalence-pair candidate."""

    schema_version: str = PAIR_SCHEMA_VERSION
    pair_id: _Sha256Digest
    left: ExpressionReferenceV1
    right: ExpressionReferenceV1
    label: StrictBool | None
    pair_group_set: tuple[_NonBlankStr, ...]
    source_split: CorpusSplit
    evaluation_views: tuple[_NonBlankStr, ...] = ()
    verification_tier: VerificationTier
    trace: RewriteTraceV1 | None = None
    non_equivalence_evidence: NonEquivalenceEvidenceV1 | None = None
    status: PairStatus
    outcome_type: _NonBlankStr | None = None
    outcome_detail: str | None = None

    @model_validator(mode="after")
    def validate_scientific_identity(self) -> Self:
        if (
            self.left.source_split != self.source_split
            or self.right.source_split != self.source_split
        ):
            raise ValueError("both endpoints must belong to the record source_split")
        expected_groups = tuple(sorted(set((*self.left.group.closure, *self.right.group.closure))))
        if self.pair_group_set != expected_groups:
            raise ValueError("pair_group_set must be the canonical endpoint lineage union")
        if tuple(sorted(set(self.evaluation_views))) != self.evaluation_views:
            raise ValueError("evaluation_views must be sorted and unique")
        if self.status is PairStatus.ACCEPTED:
            if self.label is None:
                raise ValueError("accepted records must carry a binary label")
            if self.label:
                if self.trace is None or self.trace.replay_status is not ReplayStatus.PASSED:
                    raise ValueError(
                        "accepted positive pairs require a fully replayed concrete trace"
                    )
                if self.non_equivalence_evidence is not None:
                    raise ValueError("positive pair cannot also carry non-equivalence evidence")
                if self.left.group.group_id != self.right.group.group_id:
                    raise ValueError(
                        "positive endpoints must share their source/e-class-relative group"
                    )
                if (
                    self.trace.source.expression_id != self.left.expression_id
                    or self.trace.goal.expression_id != self.right.expression_id
                    or self.trace.source.structural_signature != self.left.structural_signature
                    or self.trace.goal.structural_signature != self.right.structural_signature
                ):
                    raise ValueError(
                        "positive trace direction must match the persisted left/right endpoints"
                    )
            else:
                if self.non_equivalence_evidence is None:
                    raise ValueError("accepted negative pairs require non-equivalence evidence")
                if not self.non_equivalence_evidence.rigorous:
                    raise ValueError(
                        "numeric disagreement alone cannot create an accepted negative"
                    )
                if self.trace is not None:
                    raise ValueError("accepted negative pair cannot carry a positive trace")
        elif self.label is not None:
            raise ValueError(
                "rejected or failed candidates must not be converted into labeled data"
            )
        if self.pair_id != self.expected_pair_id():
            raise ValueError("pair_id does not bind the complete canonical pair identity")
        return self

    def identity_payload(self) -> dict[str, object]:
        """Canonical identity payload with symmetric endpoint ordering for pair datasets."""

        endpoints = sorted(
            (self.left.model_dump(mode="json"), self.right.model_dump(mode="json")),
            key=canonical_json_bytes,
        )
        return {
            "endpoints": endpoints,
            "evaluation_views": list(self.evaluation_views),
            "label": self.label,
            "non_equivalence_evidence": (
                None
                if self.non_equivalence_evidence is None
                else self.non_equivalence_evidence.model_dump(mode="json")
            ),
            "outcome_detail": self.outcome_detail,
            "outcome_type": self.outcome_type,
            "pair_group_set": list(self.pair_group_set),
            "schema_version": self.schema_version,
            "source_split": self.source_split.value,
            "status": self.status.value,
            "trace_digest": None if self.trace is None else self.trace.trace_digest,
            "verification_tier": self.verification_tier.value,
        }

    def expected_pair_id(self) -> str:
        """Return a versioned SHA-256 pair identity, never a process-local hash."""

        return sha256_digest(canonical_json_bytes(self.identity_payload()))

    @classmethod
    def create(cls, **values: object) -> Self:
        """Construct a record only after deriving the canonical cryptographic pair ID."""

        provisional = cls.model_construct(pair_id="sha256:" + "0" * 64, **values)
        return cls.model_validate({**values, "pair_id": provisional.expected_pair_id()})


class PairFixtureManifestV1(_PairContract):
    """Minimal resumable fixture manifest; production writers use the same content digest rule."""

    schema_version: str = PAIR_FIXTURE_MANIFEST_SCHEMA_VERSION
    seed: StrictInt
    row_count: _NonNegativeInt
    content_digest: _Sha256Digest
    rejected_count: _NonNegativeInt
    failed_count: _NonNegativeInt


def deterministic_fixture_records(records: Iterable[PairRecordV1]) -> tuple[PairRecordV1, ...]:
    """Sort and validate fixture records without dropping rejected or failed evidence."""

    ordered = tuple(sorted(records, key=lambda record: record.pair_id))
    if len({record.pair_id for record in ordered}) != len(ordered):
        raise PairContractError("fixture pair IDs must be unique")
    return ordered


def write_fixture_pairs(
    records: Iterable[PairRecordV1],
    output_path: str | Path,
    *,
    seed: int,
) -> PairFixtureManifestV1:
    """Write deterministic JSONL fixture evidence and return its complete manifest."""

    ordered = deterministic_fixture_records(records)
    payload = b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in ordered
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return PairFixtureManifestV1(
        seed=seed,
        row_count=len(ordered),
        content_digest=sha256_digest(payload),
        rejected_count=sum(record.status is PairStatus.REJECTED for record in ordered),
        failed_count=sum(record.status is PairStatus.FAILED for record in ordered),
    )


def make_transition_verifier(
    apply: Callable[[TraceStateV1, RewriteActionV1, str], TraceStateV1 | None],
    *,
    verifier_version: str,
) -> TransitionVerifier:
    """Adapt a deterministic concrete-action applier to the replay-verifier protocol."""

    def verify(
        state: TraceStateV1,
        action: RewriteActionV1,
        expected_successor: TraceStateV1,
        *,
        domain_mode: str,
    ) -> TransitionVerificationV1:
        observed = apply(state, action, domain_mode)
        passed = observed == expected_successor
        evidence = {
            "action_digest": action.semantic_digest,
            "domain_mode": domain_mode,
            "expected": expected_successor.model_dump(mode="json"),
            "observed": None if observed is None else observed.model_dump(mode="json"),
            "verifier_version": verifier_version,
        }
        return TransitionVerificationV1(
            action_digest=action.semantic_digest,
            source_structural_signature=state.structural_signature,
            successor_structural_signature=expected_successor.structural_signature,
            verifier_version=verifier_version,
            status=ReplayStatus.PASSED if passed else ReplayStatus.FAILED,
            evidence_digest=sha256_digest(canonical_json_bytes(evidence)),
            detail="replayed" if passed else "concrete action did not reproduce expected successor",
        )

    return verify
