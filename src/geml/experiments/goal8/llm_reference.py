"""Read-only compatibility adapter for external #82 LLM reference rows.

This module intentionally contains no provider clients, credentials, prompts,
retry logic, model selection, or network operations.  External rows are context
only and are never eligible for a controlled Goal 8 gate comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

LLM_REFERENCE_ADAPTER_VERSION = "geml-goal8-llm-reference-adapter-v1"
_HEX = frozenset("0123456789abcdef")
_TRACKS = frozenset({"proof", "symbolic_regression"})
_STATUSES = frozenset(
    {
        "complete",
        "parse_failure",
        "refusal",
        "timeout",
        "api_error",
        "invalid",
        "unsupported",
    }
)


class LlmReferenceError(ValueError):
    """External evidence is present but malformed or unauthenticated."""


class ExternalReferenceState(StrEnum):
    """Availability state reported without affecting Gate G8."""

    MISSING = "missing"
    AVAILABLE = "available"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ExternalReferenceRow:
    """Strict projection of one retained #82 attempt."""

    schema_version: str
    attempt_id: str
    track: str
    task_id: str
    provider: str
    model: str
    status: str
    claimed_correct: bool | None
    verifier_confirmed_correct: bool | None
    parse_succeeded: bool
    prompt_sha256: str
    latency_seconds: float | None
    cost_usd: float | None

    @property
    def verified_correct(self) -> bool:
        return self.verifier_confirmed_correct is True

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "track": self.track,
            "task_id": self.task_id,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "claimed_correct": self.claimed_correct,
            "verifier_confirmed_correct": self.verifier_confirmed_correct,
            "parse_succeeded": self.parse_succeeded,
            "prompt_sha256": self.prompt_sha256,
            "latency_seconds": self.latency_seconds,
            "cost_usd": self.cost_usd,
            "role": "external_reference_only",
            "included_in_gate": False,
        }


@dataclass(frozen=True, slots=True)
class ExternalReferenceBundle:
    """Authenticated external rows or an explicit optional/missing state."""

    adapter_version: str
    state: ExternalReferenceState
    source_path: str | None
    source_sha256: str | None
    rows: tuple[ExternalReferenceRow, ...]
    reason: str | None = None

    @property
    def proof_rows(self) -> tuple[ExternalReferenceRow, ...]:
        return tuple(row for row in self.rows if row.track == "proof")

    def summary(self) -> dict[str, object]:
        proof = self.proof_rows
        return {
            "adapter_version": self.adapter_version,
            "state": self.state.value,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "reason": self.reason,
            "role": "external_reference_only",
            "included_in_gate": False,
            "row_count": len(self.rows),
            "proof_row_count": len(proof),
            "proof_claimed_correct_count": sum(row.claimed_correct is True for row in proof),
            "proof_verifier_confirmed_count": sum(row.verified_correct for row in proof),
            "status_counts": dict(sorted(Counter(row.status for row in self.rows).items())),
            "rows": [row.as_dict() for row in self.rows],
        }


def load_external_reference(
    path: str | Path | None,
    *,
    expected_sha256: str | None,
) -> ExternalReferenceBundle:
    """Load exact local JSONL bytes, or return an explicit missing state.

    A path and expected digest are both required when rows are available.  An
    omitted path represents the legitimate Phase-A state where #82 has not run.
    """

    if path is None:
        if expected_sha256 is not None:
            raise LlmReferenceError("expected_sha256 cannot be set when path is absent")
        return ExternalReferenceBundle(
            adapter_version=LLM_REFERENCE_ADAPTER_VERSION,
            state=ExternalReferenceState.MISSING,
            source_path=None,
            source_sha256=None,
            rows=(),
            reason="optional #82 output is not available",
        )
    if expected_sha256 is None:
        raise LlmReferenceError("expected_sha256 is required for external rows")
    expected = _sha256(expected_sha256, field="expected_sha256")
    try:
        source = Path(path).resolve(strict=True)
    except OSError as error:
        raise LlmReferenceError(f"cannot resolve external reference rows: {error}") from error
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise LlmReferenceError(f"cannot read external reference rows: {error}") from error
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise LlmReferenceError(f"external row SHA-256 mismatch: expected {expected}, got {actual}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LlmReferenceError("external reference rows are not UTF-8") from error
    rows: list[ExternalReferenceRow] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise LlmReferenceError(f"blank external JSONL row at line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise LlmReferenceError(f"invalid external JSON at line {line_number}") from error
        if not isinstance(value, Mapping):
            raise LlmReferenceError(f"external JSONL row {line_number} must be an object")
        row = parse_external_row(value)
        if row.attempt_id in seen:
            raise LlmReferenceError(f"duplicate external attempt_id {row.attempt_id!r}")
        seen.add(row.attempt_id)
        rows.append(row)
    return ExternalReferenceBundle(
        adapter_version=LLM_REFERENCE_ADAPTER_VERSION,
        state=ExternalReferenceState.AVAILABLE,
        source_path=str(source),
        source_sha256=actual,
        rows=tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.track,
                    row.provider,
                    row.model,
                    row.task_id,
                    row.attempt_id,
                ),
            )
        ),
    )


def parse_external_row(value: Mapping[str, object]) -> ExternalReferenceRow:
    """Project one #82 row without accepting untyped truthy values."""

    track = _string(value, "track")
    if track not in _TRACKS:
        raise LlmReferenceError(f"unknown external track {track!r}")
    status = _string(value, "status")
    if status not in _STATUSES:
        raise LlmReferenceError(f"unknown external status {status!r}")
    claimed = _optional_bool(value.get("claimed_correct"), field="claimed_correct")
    verified = _optional_bool(
        value.get("verifier_confirmed_correct"),
        field="verifier_confirmed_correct",
    )
    parsed = _bool(value, "parse_succeeded")
    if verified is True and (status != "complete" or not parsed):
        raise LlmReferenceError("verifier-confirmed correctness requires a parsed complete row")
    return ExternalReferenceRow(
        schema_version=_string(value, "schema_version"),
        attempt_id=_string(value, "attempt_id"),
        track=track,
        task_id=_string(value, "task_id"),
        provider=_string(value, "provider"),
        model=_string(value, "model"),
        status=status,
        claimed_correct=claimed,
        verifier_confirmed_correct=verified,
        parse_succeeded=parsed,
        prompt_sha256=_sha256(value.get("prompt_sha256"), field="prompt_sha256"),
        latency_seconds=_optional_number(value.get("latency_seconds"), field="latency_seconds"),
        cost_usd=_optional_number(value.get("cost_usd"), field="cost_usd"),
    )


def _string(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise LlmReferenceError(f"{key} must be a non-blank string")
    return item


def _bool(value: Mapping[str, object], key: str) -> bool:
    item = value.get(key)
    if type(item) is not bool:
        raise LlmReferenceError(f"{key} must be a boolean")
    return item


def _optional_bool(value: object, *, field: str) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise LlmReferenceError(f"{field} must be a boolean or null")
    return value


def _optional_number(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LlmReferenceError(f"{field} must be a finite nonnegative number or null")
    number = float(value)
    if number < 0.0 or not math.isfinite(number):
        raise LlmReferenceError(f"{field} must be a finite nonnegative number or null")
    return number


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise LlmReferenceError(f"{field} must be lowercase SHA-256 hexadecimal")
    return value
