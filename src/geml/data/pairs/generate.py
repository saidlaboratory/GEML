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
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from random import Random
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
TRAJECTORY_POLICY_SCHEMA_VERSION = "geml-goal6-trajectory-policy-v1"
TRAJECTORY_OUTCOME_SCHEMA_VERSION = "geml-goal6-trajectory-outcome-v1"
PAIR_SHARD_MANIFEST_SCHEMA_VERSION = "geml-goal6-pair-shard-manifest-v1"
DERIVED_SEED_SCHEMA_VERSION = "geml-derived-seed-v1"

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_Sha256Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
_PositiveInt = Annotated[StrictInt, Field(ge=1)]


class PairContractError(ValueError):
    """Raised when a pair cannot be represented as scientifically valid evidence."""


class PairStatus(StrEnum):
    """Retained status for both accepted and rejected candidate pairs."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class TrajectoryOutcomeStatus(StrEnum):
    """Every outcome of a bounded concrete-trajectory attempt."""

    ACCEPTED = "accepted"
    EXHAUSTED = "exhausted"
    SHORTFALL = "shortfall"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"
    VERIFIER_ERROR = "verifier_error"
    TIMEOUT = "timeout"


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


class TrajectoryPolicyV1(_PairContract):
    """Frozen bounded-search policy for deterministic concrete trace generation."""

    schema_version: str = TRAJECTORY_POLICY_SCHEMA_VERSION
    min_trace_length: _PositiveInt
    max_trace_length: _PositiveInt
    max_attempts_per_source: _PositiveInt
    forbid_immediate_inverse: StrictBool = True
    action_selection: _NonBlankStr = "derived_seed_uniform_legal_action"

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.min_trace_length > self.max_trace_length:
            raise ValueError("min_trace_length cannot exceed max_trace_length")
        return self


class TrajectoryGenerationOutcomeV1(_PairContract):
    """A retained outcome from one source/attempt, including non-success paths."""

    schema_version: str = TRAJECTORY_OUTCOME_SCHEMA_VERSION
    source_expression_id: _NonBlankStr
    attempt_index: _NonNegativeInt
    derived_seed: StrictInt
    status: TrajectoryOutcomeStatus
    trace: RewriteTraceV1 | None = None
    detail: _NonBlankStr

    @model_validator(mode="after")
    def validate_trace_status(self) -> Self:
        if self.status is TrajectoryOutcomeStatus.ACCEPTED:
            if self.trace is None or self.trace.replay_status is not ReplayStatus.PASSED:
                raise ValueError("accepted trajectory outcomes require a passed concrete trace")
        elif self.trace is not None:
            raise ValueError(
                "non-accepted trajectory outcomes must retain no partial trace as success"
            )
        return self


def derive_trajectory_seed(
    *,
    component: str,
    run_seed: int,
    source_expression_id: str,
    attempt_index: int,
) -> int:
    """Derive one portable 64-bit seed from immutable, versioned identity fields.

    This is the executable `geml-derived-seed-v1` rule documented in the
    package-wide ML policy.  It deliberately uses canonical JSON and SHA-256,
    never process-randomized Python hashing or ambient global RNG state.
    """

    if not isinstance(component, str) or not component.strip():
        raise ValueError("component must be a nonblank string")
    if type(run_seed) is not int:
        raise ValueError("run_seed must be an integer")
    if not isinstance(source_expression_id, str) or not source_expression_id.strip():
        raise ValueError("source_expression_id must be a nonblank string")
    if type(attempt_index) is not int or attempt_index < 0:
        raise ValueError("attempt_index must be a nonnegative integer")
    payload = {
        "attempt_index": attempt_index,
        "component": component,
        "immutable_record_id": source_expression_id,
        "run_seed": run_seed,
        "schema_version": DERIVED_SEED_SCHEMA_VERSION,
    }
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:8], "big")


ConcreteActionEnumerator = Callable[[TraceStateV1, int, str], Iterable[RewriteActionV1]]
ConcreteActionApplier = Callable[[TraceStateV1, RewriteActionV1, str], TraceStateV1 | None]


def _is_immediate_inverse(previous: RewriteActionV1, candidate: RewriteActionV1) -> bool:
    """Identify the exact structural inverse of the immediately preceding action."""

    return (
        previous.rule_id == candidate.rule_id
        and previous.occurrence_path == candidate.occurrence_path
        and previous.source_structural_signature == candidate.successor_structural_signature
        and previous.successor_structural_signature == candidate.source_structural_signature
    )


def _trajectory_failure(
    *,
    source_expression_id: str,
    attempt_index: int,
    derived_seed: int,
    status: TrajectoryOutcomeStatus,
    detail: str,
) -> TrajectoryGenerationOutcomeV1:
    """Construct one typed retained non-success outcome."""

    return TrajectoryGenerationOutcomeV1(
        source_expression_id=source_expression_id,
        attempt_index=attempt_index,
        derived_seed=derived_seed,
        status=status,
        detail=detail,
    )


def generate_concrete_trajectory_attempts(
    source: TraceStateV1,
    *,
    run_seed: int,
    domain_mode: str,
    rule_set_digest: str,
    policy_digest: str,
    policy: TrajectoryPolicyV1,
    enumerate_actions: ConcreteActionEnumerator,
    apply_action: ConcreteActionApplier,
    verifier_version: str,
) -> tuple[TrajectoryGenerationOutcomeV1, ...]:
    """Generate bounded concrete trajectories and retain every failed attempt.

    The injected callbacks are the only bridge to the frozen Goal 4 rule
    engine.  This module neither reinterprets saturation provenance as a trace
    nor invents a second semantic verifier.  The first accepted trajectory may
    be used by a quota scheduler, but every preceding shortfall, timeout,
    unsupported, invalid, and verifier-error outcome remains in the returned
    ledger for persistence.
    """

    if not isinstance(domain_mode, str) or not domain_mode.strip():
        raise ValueError("domain_mode must be a nonblank string")
    if not isinstance(verifier_version, str) or not verifier_version.strip():
        raise ValueError("verifier_version must be a nonblank string")
    outcomes: list[TrajectoryGenerationOutcomeV1] = []
    for attempt_index in range(policy.max_attempts_per_source):
        derived_seed = derive_trajectory_seed(
            component="goal6.pairs.trajectory",
            run_seed=run_seed,
            source_expression_id=source.expression_id,
            attempt_index=attempt_index,
        )
        rng = Random(derived_seed)
        requested_length = rng.randint(policy.min_trace_length, policy.max_trace_length)
        states = [source]
        actions: list[RewriteActionV1] = []
        transitions: list[TransitionVerificationV1] = []
        failure: TrajectoryGenerationOutcomeV1 | None = None
        for _step_index in range(requested_length):
            current = states[-1]
            try:
                enumerated = tuple(enumerate_actions(current, derived_seed, domain_mode))
            except TimeoutError:
                failure = _trajectory_failure(
                    source_expression_id=source.expression_id,
                    attempt_index=attempt_index,
                    derived_seed=derived_seed,
                    status=TrajectoryOutcomeStatus.TIMEOUT,
                    detail="concrete action enumeration timed out",
                )
                break
            except NotImplementedError:
                failure = _trajectory_failure(
                    source_expression_id=source.expression_id,
                    attempt_index=attempt_index,
                    derived_seed=derived_seed,
                    status=TrajectoryOutcomeStatus.UNSUPPORTED,
                    detail="concrete action enumeration is unsupported",
                )
                break
            except Exception as error:  # callback boundary: evidence must retain its failure
                failure = _trajectory_failure(
                    source_expression_id=source.expression_id,
                    attempt_index=attempt_index,
                    derived_seed=derived_seed,
                    status=TrajectoryOutcomeStatus.VERIFIER_ERROR,
                    detail=f"concrete action enumeration failed: {type(error).__name__}",
                )
                break
            if any(
                action.source_structural_signature != current.structural_signature
                for action in enumerated
            ):
                failure = _trajectory_failure(
                    source_expression_id=source.expression_id,
                    attempt_index=attempt_index,
                    derived_seed=derived_seed,
                    status=TrajectoryOutcomeStatus.INVALID,
                    detail="enumerated action source signature differs from current concrete state",
                )
                break
            legal = tuple(
                action
                for action in sorted(enumerated, key=lambda action: action.semantic_digest)
                if not (
                    policy.forbid_immediate_inverse
                    and actions
                    and _is_immediate_inverse(actions[-1], action)
                )
            )
            if not legal:
                status = (
                    TrajectoryOutcomeStatus.EXHAUSTED
                    if not actions
                    else TrajectoryOutcomeStatus.SHORTFALL
                )
                failure = _trajectory_failure(
                    source_expression_id=source.expression_id,
                    attempt_index=attempt_index,
                    derived_seed=derived_seed,
                    status=status,
                    detail="no legal concrete action remains under the frozen trace policy",
                )
                break
            action = legal[rng.randrange(len(legal))]
            try:
                successor = apply_action(current, action, domain_mode)
            except TimeoutError:
                failure = _trajectory_failure(
                    source_expression_id=source.expression_id,
                    attempt_index=attempt_index,
                    derived_seed=derived_seed,
                    status=TrajectoryOutcomeStatus.TIMEOUT,
                    detail="concrete action application timed out",
                )
                break
            except NotImplementedError:
                failure = _trajectory_failure(
                    source_expression_id=source.expression_id,
                    attempt_index=attempt_index,
                    derived_seed=derived_seed,
                    status=TrajectoryOutcomeStatus.UNSUPPORTED,
                    detail="concrete action application is unsupported",
                )
                break
            except Exception as error:  # callback boundary: evidence must retain its failure
                failure = _trajectory_failure(
                    source_expression_id=source.expression_id,
                    attempt_index=attempt_index,
                    derived_seed=derived_seed,
                    status=TrajectoryOutcomeStatus.VERIFIER_ERROR,
                    detail=f"concrete action application failed: {type(error).__name__}",
                )
                break
            if successor is None:
                failure = _trajectory_failure(
                    source_expression_id=source.expression_id,
                    attempt_index=attempt_index,
                    derived_seed=derived_seed,
                    status=TrajectoryOutcomeStatus.UNSUPPORTED,
                    detail="concrete action has no supported successor",
                )
                break
            if successor.structural_signature != action.successor_structural_signature:
                failure = _trajectory_failure(
                    source_expression_id=source.expression_id,
                    attempt_index=attempt_index,
                    derived_seed=derived_seed,
                    status=TrajectoryOutcomeStatus.INVALID,
                    detail=(
                        "concrete action successor differs from its persisted structural signature"
                    ),
                )
                break
            evidence = {
                "action_digest": action.semantic_digest,
                "domain_mode": domain_mode,
                "schema_version": "geml-transition-verification-evidence-v1",
                "source": current.model_dump(mode="json"),
                "successor": successor.model_dump(mode="json"),
                "verifier_version": verifier_version,
            }
            actions.append(action)
            states.append(successor)
            transitions.append(
                TransitionVerificationV1(
                    action_digest=action.semantic_digest,
                    source_structural_signature=current.structural_signature,
                    successor_structural_signature=successor.structural_signature,
                    verifier_version=verifier_version,
                    status=ReplayStatus.PASSED,
                    evidence_digest=sha256_digest(canonical_json_bytes(evidence)),
                    detail="concrete action replayed to stored successor",
                )
            )
        if failure is not None:
            outcomes.append(failure)
            continue
        if len(actions) < policy.min_trace_length:
            outcomes.append(
                _trajectory_failure(
                    source_expression_id=source.expression_id,
                    attempt_index=attempt_index,
                    derived_seed=derived_seed,
                    status=TrajectoryOutcomeStatus.SHORTFALL,
                    detail="trajectory did not meet the frozen minimum trace length",
                )
            )
            continue
        trace = RewriteTraceV1(
            source=source,
            goal=states[-1],
            states=tuple(states),
            actions=tuple(actions),
            transitions=tuple(transitions),
            verified_step_count=len(transitions),
            rule_set_digest=rule_set_digest,
            policy_digest=policy_digest,
            domain_mode=domain_mode,
            generation_seed=derived_seed,
            replay_status=ReplayStatus.PASSED,
        )
        outcomes.append(
            TrajectoryGenerationOutcomeV1(
                source_expression_id=source.expression_id,
                attempt_index=attempt_index,
                derived_seed=derived_seed,
                status=TrajectoryOutcomeStatus.ACCEPTED,
                trace=trace,
                detail="accepted bounded concrete trajectory",
            )
        )
        break
    return tuple(outcomes)


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


class PairShardManifestV1(_PairContract):
    """Completion sidecar for one content-addressed, resumable pair shard."""

    schema_version: str = PAIR_SHARD_MANIFEST_SCHEMA_VERSION
    shard_id: _NonBlankStr
    config_digest: _Sha256Digest
    input_digest: _Sha256Digest
    rule_set_digest: _Sha256Digest
    content_digest: _Sha256Digest
    attempted_count: _NonNegativeInt
    record_count: _NonNegativeInt
    accepted_count: _NonNegativeInt
    rejected_count: _NonNegativeInt
    failed_count: _NonNegativeInt
    outcome_counts: dict[_NonBlankStr, _NonNegativeInt]
    split_counts: dict[_NonBlankStr, _NonNegativeInt]
    label_counts: dict[_NonBlankStr, _NonNegativeInt]
    endpoint_family_counts: dict[_NonBlankStr, _NonNegativeInt]
    endpoint_domain_counts: dict[_NonBlankStr, _NonNegativeInt]
    depth_counts: dict[_NonBlankStr, _NonNegativeInt]
    trace_length_counts: dict[_NonBlankStr, _NonNegativeInt]
    verification_tier_counts: dict[_NonBlankStr, _NonNegativeInt]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.attempted_count != self.record_count:
            raise ValueError("attempted_count must equal the retained shard record_count")
        if self.record_count != self.accepted_count + self.rejected_count + self.failed_count:
            raise ValueError("pair-shard status counts must sum to record_count")
        per_record = (
            self.outcome_counts,
            self.split_counts,
            self.label_counts,
            self.depth_counts,
            self.trace_length_counts,
            self.verification_tier_counts,
        )
        if any(sum(counts.values()) != self.record_count for counts in per_record):
            raise ValueError("every per-record pair-shard count must sum to record_count")
        if sum(self.endpoint_family_counts.values()) != 2 * self.record_count:
            raise ValueError("endpoint_family_counts must count both endpoints of each record")
        if sum(self.endpoint_domain_counts.values()) != 2 * self.record_count:
            raise ValueError("endpoint_domain_counts must count both endpoints of each record")
        for name, counts in (
            ("outcome_counts", self.outcome_counts),
            ("split_counts", self.split_counts),
            ("label_counts", self.label_counts),
            ("endpoint_family_counts", self.endpoint_family_counts),
            ("endpoint_domain_counts", self.endpoint_domain_counts),
            ("depth_counts", self.depth_counts),
            ("trace_length_counts", self.trace_length_counts),
            ("verification_tier_counts", self.verification_tier_counts),
        ):
            if tuple(counts) != tuple(sorted(counts)):
                raise ValueError(f"{name} keys must be sorted canonically")
        return self


def _publish_exact_bytes(destination: Path, payload: bytes, *, label: str) -> None:
    """Atomically publish bytes or verify that an existing resume target matches.

    A completion sidecar is published only after its shard bytes are available.
    Existing bytes are never overwritten: a resume either proves exact identity
    or fails loudly rather than accepting a stale or corrupt partial output.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file():
            raise PairContractError(f"{label} destination is not a regular file: {destination}")
        if destination.read_bytes() != payload:
            raise PairContractError(f"resumed {label} differs from the requested canonical bytes")
        return
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise PairContractError(
                    f"concurrently published {label} differs from the requested canonical bytes"
                ) from None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _canonical_counts(values: Iterable[str]) -> dict[str, int]:
    """Count one categorical evidence field with deterministic key ordering."""

    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def write_pair_shard(
    records: Iterable[PairRecordV1],
    output_path: str | Path,
    *,
    shard_id: str,
    config_digest: str,
    input_digest: str,
    rule_set_digest: str,
    depth_by_pair_id: Mapping[str, int],
) -> PairShardManifestV1:
    """Write or safely resume one canonical pair shard and completion manifest."""

    ordered = deterministic_fixture_records(records)
    missing_depths = tuple(
        record.pair_id for record in ordered if record.pair_id not in depth_by_pair_id
    )
    if missing_depths:
        raise PairContractError(f"pair shard is missing frozen depth metadata for {missing_depths}")
    if any(
        type(depth_by_pair_id[record.pair_id]) is not int or depth_by_pair_id[record.pair_id] < 0
        for record in ordered
    ):
        raise PairContractError("pair-shard depths must be nonnegative integers")
    payload = b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in ordered
    )
    destination = Path(output_path)
    _publish_exact_bytes(destination, payload, label="pair shard")
    manifest = PairShardManifestV1(
        shard_id=shard_id,
        config_digest=config_digest,
        input_digest=input_digest,
        rule_set_digest=rule_set_digest,
        content_digest=sha256_digest(payload),
        attempted_count=len(ordered),
        record_count=len(ordered),
        accepted_count=sum(record.status is PairStatus.ACCEPTED for record in ordered),
        rejected_count=sum(record.status is PairStatus.REJECTED for record in ordered),
        failed_count=sum(record.status is PairStatus.FAILED for record in ordered),
        outcome_counts=_canonical_counts(
            record.outcome_type if record.outcome_type is not None else record.status.value
            for record in ordered
        ),
        split_counts=_canonical_counts(record.source_split.value for record in ordered),
        label_counts=_canonical_counts(
            (
                "positive"
                if record.label is True
                else "negative"
                if record.label is False
                else "unlabeled"
            )
            for record in ordered
        ),
        endpoint_family_counts=_canonical_counts(
            endpoint.operator_family
            for record in ordered
            for endpoint in (record.left, record.right)
        ),
        endpoint_domain_counts=_canonical_counts(
            endpoint.domain_mode for record in ordered for endpoint in (record.left, record.right)
        ),
        depth_counts=_canonical_counts(str(depth_by_pair_id[record.pair_id]) for record in ordered),
        trace_length_counts=_canonical_counts(
            str(len(record.trace.actions)) if record.trace is not None else "none"
            for record in ordered
        ),
        verification_tier_counts=_canonical_counts(
            record.verification_tier.value for record in ordered
        ),
    )
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    _publish_exact_bytes(
        manifest_path,
        canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n",
        label="pair-shard manifest",
    )
    return manifest


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
    _publish_exact_bytes(destination, payload, label="fixture pair output")
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
