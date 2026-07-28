"""Verifier-replayable Goal 7 state-to-action records from Goal 6 positive traces.

The extractor accepts only Goal 6's concrete child-slot trajectories.  It never
interprets Goal 4 e-class provenance as a concrete application site and retains a
typed failure row whenever replay, graph binding, or input validation cannot proceed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints, model_validator

from geml.contracts.corpus import CorpusSplit
from geml.data.pairs.generate import (
    PairRecordV1,
    PairStatus,
    ReplayStatus,
    RewriteActionV1,
    TransitionVerifier,
    replay_trace,
)

STEP_RECORD_SCHEMA_VERSION = "geml-rewrite-step-v1"
STEP_FAILURE_SCHEMA_VERSION = "geml-rewrite-step-failure-v1"
STEP_FIXTURE_MANIFEST_SCHEMA_VERSION = "geml-goal7-step-fixture-manifest-v1"

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_Sha256Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]


class StepDatasetError(ValueError):
    """A trace cannot be safely turned into a supervised rewrite-step row."""


class StepFailureStatus(StrEnum):
    """Terminal non-success states retained in the extraction denominator."""

    INPUT_NOT_POSITIVE_TRACE = "input_not_positive_trace"
    REPLAY_FAILED = "replay_failed"
    REPLAY_UNSUPPORTED = "replay_unsupported"
    REPLAY_TIMEOUT = "replay_timeout"
    STATE_GRAPH_MISSING = "state_graph_missing"
    INVALID = "invalid"


class _StepContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)


def canonical_json_bytes(value: object) -> bytes:
    """Encode persisted step evidence in a stable, checksum-friendly representation."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


class RewriteStepRecordV1(_StepContract):
    """One replay-verified state/action/successor target with inherited split lineage."""

    schema_version: str = STEP_RECORD_SCHEMA_VERSION
    step_id: _Sha256Digest
    pair_id: _Sha256Digest
    trace_digest: _Sha256Digest
    step_index: _NonNegativeInt
    state_expression_id: _NonBlankStr
    state_structural_signature: _NonBlankStr
    state_graph_id: _NonBlankStr
    action: RewriteActionV1
    next_state_expression_id: _NonBlankStr
    next_state_structural_signature: _NonBlankStr
    remaining_step_distance: _NonNegativeInt
    source_split: CorpusSplit
    group_closure: tuple[_NonBlankStr, ...]
    operator_family: _NonBlankStr

    @model_validator(mode="after")
    def validate_transition_and_identity(self) -> Self:
        if self.action.source_structural_signature != self.state_structural_signature:
            raise ValueError("action source signature must match the stored state")
        if self.action.successor_structural_signature != self.next_state_structural_signature:
            raise ValueError("action successor signature must match the stored next state")
        if not self.group_closure or tuple(sorted(set(self.group_closure))) != self.group_closure:
            raise ValueError("group_closure must be nonempty, sorted, and unique")
        if self.step_id != self.expected_step_id():
            raise ValueError("step_id must bind the complete concrete supervised action")
        return self

    def identity_payload(self) -> dict[str, object]:
        return {
            "action": self.action.model_dump(mode="json"),
            "group_closure": list(self.group_closure),
            "next_state_expression_id": self.next_state_expression_id,
            "next_state_structural_signature": self.next_state_structural_signature,
            "operator_family": self.operator_family,
            "pair_id": self.pair_id,
            "remaining_step_distance": self.remaining_step_distance,
            "schema_version": self.schema_version,
            "source_split": self.source_split.value,
            "state_expression_id": self.state_expression_id,
            "state_graph_id": self.state_graph_id,
            "state_structural_signature": self.state_structural_signature,
            "step_index": self.step_index,
            "trace_digest": self.trace_digest,
        }

    def expected_step_id(self) -> str:
        return sha256_digest(canonical_json_bytes(self.identity_payload()))

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(step_id="sha256:" + "0" * 64, **values)
        return cls.model_validate({**values, "step_id": provisional.expected_step_id()})


class RewriteStepFailureV1(_StepContract):
    """An auditable input/replay/materialization failure; never silently deleted."""

    schema_version: str = STEP_FAILURE_SCHEMA_VERSION
    pair_id: _Sha256Digest
    trace_digest: _Sha256Digest | None = None
    source_split: CorpusSplit
    group_closure: tuple[_NonBlankStr, ...]
    status: StepFailureStatus
    detail: _NonBlankStr
    step_index: _NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        if not self.group_closure or tuple(sorted(set(self.group_closure))) != self.group_closure:
            raise ValueError("group_closure must be nonempty, sorted, and unique")
        return self


class StepFixtureManifestV1(_StepContract):
    """Checksum/count evidence for deterministic temporary-fixture extraction."""

    schema_version: str = STEP_FIXTURE_MANIFEST_SCHEMA_VERSION
    seed: StrictInt
    step_count: _NonNegativeInt
    failure_count: _NonNegativeInt
    content_digest: _Sha256Digest


def _failure_for_pair(
    record: PairRecordV1,
    *,
    status: StepFailureStatus,
    detail: str,
    trace_digest: str | None = None,
    step_index: int | None = None,
) -> RewriteStepFailureV1:
    return RewriteStepFailureV1(
        pair_id=record.pair_id,
        trace_digest=trace_digest,
        source_split=record.source_split,
        group_closure=record.pair_group_set,
        status=status,
        detail=detail,
        step_index=step_index,
    )


def _replay_failure_status(status: ReplayStatus) -> StepFailureStatus:
    return {
        ReplayStatus.FAILED: StepFailureStatus.REPLAY_FAILED,
        ReplayStatus.UNSUPPORTED: StepFailureStatus.REPLAY_UNSUPPORTED,
        ReplayStatus.TIMEOUT: StepFailureStatus.REPLAY_TIMEOUT,
    }[status]


def extract_trace_steps(
    record: PairRecordV1,
    *,
    state_graph_ids: Mapping[str, str],
    verifier: TransitionVerifier,
) -> tuple[tuple[RewriteStepRecordV1, ...], tuple[RewriteStepFailureV1, ...]]:
    """Replay one accepted positive trace and emit all steps or a retained failure row."""

    if record.status is not PairStatus.ACCEPTED or record.label is not True or record.trace is None:
        return (), (
            _failure_for_pair(
                record,
                status=StepFailureStatus.INPUT_NOT_POSITIVE_TRACE,
                detail=(
                    "only accepted positive pairs with a concrete trace can supply policy targets"
                ),
            ),
        )
    trace = record.trace
    try:
        replayed = replay_trace(trace, verifier)
    except Exception as error:
        return (), (
            _failure_for_pair(
                record,
                status=StepFailureStatus.INVALID,
                detail=f"replay raised {type(error).__name__}: {error}",
                trace_digest=trace.trace_digest,
            ),
        )
    if replayed.replay_status is not ReplayStatus.PASSED:
        return (), (
            _failure_for_pair(
                record,
                status=_replay_failure_status(replayed.replay_status),
                detail=replayed.failure_detail or "replay did not pass",
                trace_digest=trace.trace_digest,
            ),
        )
    missing = [
        state.expression_id
        for state in replayed.states
        if state.expression_id not in state_graph_ids
    ]
    if missing:
        return (), (
            _failure_for_pair(
                record,
                status=StepFailureStatus.STATE_GRAPH_MISSING,
                detail=f"no state graph ID for trace state(s): {', '.join(sorted(missing))}",
                trace_digest=trace.trace_digest,
            ),
        )
    return (
        tuple(
            RewriteStepRecordV1.create(
                pair_id=record.pair_id,
                trace_digest=trace.trace_digest,
                step_index=index,
                state_expression_id=state.expression_id,
                state_structural_signature=state.structural_signature,
                state_graph_id=state_graph_ids[state.expression_id],
                action=action,
                next_state_expression_id=next_state.expression_id,
                next_state_structural_signature=next_state.structural_signature,
                remaining_step_distance=len(replayed.actions) - index - 1,
                source_split=record.source_split,
                group_closure=record.pair_group_set,
                operator_family=record.left.operator_family,
            )
            for index, (state, action, next_state) in enumerate(
                zip(replayed.states[:-1], replayed.actions, replayed.states[1:], strict=True)
            )
        ),
        (),
    )


def extract_steps(
    records: Iterable[PairRecordV1],
    *,
    state_graph_ids: Mapping[str, str],
    verifier: TransitionVerifier,
) -> tuple[tuple[RewriteStepRecordV1, ...], tuple[RewriteStepFailureV1, ...]]:
    """Extract deterministically ordered steps and failures without filtering any input record."""

    steps: list[RewriteStepRecordV1] = []
    failures: list[RewriteStepFailureV1] = []
    for record in sorted(records, key=lambda item: item.pair_id):
        accepted, retained_failures = extract_trace_steps(
            record,
            state_graph_ids=state_graph_ids,
            verifier=verifier,
        )
        steps.extend(accepted)
        failures.extend(retained_failures)
    ordered_steps = tuple(sorted(steps, key=lambda item: (item.trace_digest, item.step_index)))
    accepted_pair_ids = {step.pair_id for step in ordered_steps}
    ordered_failures = tuple(
        sorted(
            (item for item in failures if item.pair_id not in accepted_pair_ids),
            key=lambda item: (item.pair_id, item.step_index or -1, item.status.value),
        )
    )
    return ordered_steps, ordered_failures


def write_fixture_steps(
    steps: Iterable[RewriteStepRecordV1],
    failures: Iterable[RewriteStepFailureV1],
    output_path: str | Path,
    *,
    seed: int,
) -> StepFixtureManifestV1:
    """Write byte-stable JSONL records for the tiny-fixture pathway."""

    ordered_steps = tuple(sorted(steps, key=lambda item: (item.trace_digest, item.step_index)))
    ordered_failures = tuple(
        sorted(failures, key=lambda item: (item.pair_id, item.step_index or -1, item.status.value))
    )
    payload = b"".join(
        canonical_json_bytes({"record_type": "step", **item.model_dump(mode="json")}) + b"\n"
        for item in ordered_steps
    ) + b"".join(
        canonical_json_bytes({"record_type": "failure", **item.model_dump(mode="json")}) + b"\n"
        for item in ordered_failures
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return StepFixtureManifestV1(
        seed=seed,
        step_count=len(ordered_steps),
        failure_count=len(ordered_failures),
        content_digest=sha256_digest(payload),
    )
