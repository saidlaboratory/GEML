"""Deterministic descriptive coverage for the frozen rewrite-step dataset.

This module never samples or reweights rows.  Every dimension is descriptive,
and every frozen registry entry is emitted even when its observed count is
zero or it is unsupported.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from geml.export.schema import canonical_json_bytes

if TYPE_CHECKING:
    from geml.data.steps.extract import StepFailureV1, StepRecordV1

RULE_REGISTRY_SNAPSHOT_VERSION = "geml-step-rule-registry-snapshot-v1"
STRATIFICATION_REPORT_VERSION = "geml-step-stratification-report-v1"
_MISSING = "__missing__"


class StratificationProtocolError(ValueError):
    """Registry or report content violates the frozen descriptive contract."""


def _require_nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StratificationProtocolError(f"{label} must be a non-blank string")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StratificationProtocolError(
            f"{label} must be a 64-character lowercase hexadecimal SHA-256"
        )
    return value


def _nonnegative(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise StratificationProtocolError(f"{label} must be a nonnegative integer")
    return value


def _tagged_digest(tag: str, payload: object) -> str:
    return hashlib.sha256(tag.encode("ascii") + b"\0" + canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class RuleRegistryEntryV1:
    """One authenticated rule/direction inventory entry for zero coverage."""

    rule_id: str
    direction: str
    supported: bool

    def __post_init__(self) -> None:
        _require_nonblank(self.rule_id, "rule_id")
        if self.direction not in {"forward", "backward"}:
            raise StratificationProtocolError("registry direction must be 'forward' or 'backward'")
        if type(self.supported) is not bool:
            raise StratificationProtocolError("registry supported must be a boolean")

    def as_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "rule_id": self.rule_id,
            "supported": self.supported,
        }

    @classmethod
    def from_dict(cls, value: object) -> RuleRegistryEntryV1:
        expected = {"direction", "rule_id", "supported"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise StratificationProtocolError("registry entry fields differ from frozen schema")
        return cls(
            rule_id=value["rule_id"],  # type: ignore[arg-type]
            direction=value["direction"],  # type: ignore[arg-type]
            supported=value["supported"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RuleRegistrySnapshotV1:
    """Authenticated producer registry identity plus its normalized inventory."""

    authoritative_rule_set_digest: str
    source_schema_version: str
    source_content_digest: str
    entries: tuple[RuleRegistryEntryV1, ...]
    schema_version: str = RULE_REGISTRY_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RULE_REGISTRY_SNAPSHOT_VERSION:
            raise StratificationProtocolError("unexpected registry snapshot schema")
        _require_sha256(
            self.authoritative_rule_set_digest,
            "authoritative_rule_set_digest",
        )
        _require_nonblank(self.source_schema_version, "source_schema_version")
        _require_sha256(self.source_content_digest, "source_content_digest")
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, RuleRegistryEntryV1) for entry in self.entries
        ):
            raise StratificationProtocolError("entries must be a tuple of RuleRegistryEntryV1")
        keys = tuple((entry.rule_id, entry.direction) for entry in self.entries)
        if tuple(sorted(set(keys))) != keys:
            raise StratificationProtocolError(
                "registry entries must be unique and sorted by rule/direction"
            )

    @property
    def normalized_entry_digest(self) -> str:
        """Digest this compatibility view, separate from producer rule identity."""

        return _tagged_digest(
            RULE_REGISTRY_SNAPSHOT_VERSION,
            [entry.as_dict() for entry in self.entries],
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "authoritative_rule_set_digest": self.authoritative_rule_set_digest,
            "entries": [entry.as_dict() for entry in self.entries],
            "normalized_entry_digest": self.normalized_entry_digest,
            "schema_version": self.schema_version,
            "source_content_digest": self.source_content_digest,
            "source_schema_version": self.source_schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> RuleRegistrySnapshotV1:
        expected = {
            "authoritative_rule_set_digest",
            "entries",
            "normalized_entry_digest",
            "schema_version",
            "source_content_digest",
            "source_schema_version",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise StratificationProtocolError("registry snapshot fields differ from frozen schema")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list):
            raise StratificationProtocolError("registry snapshot entries must be a JSON array")
        snapshot = cls(
            authoritative_rule_set_digest=value["authoritative_rule_set_digest"],  # type: ignore[arg-type]
            source_schema_version=value["source_schema_version"],  # type: ignore[arg-type]
            source_content_digest=value["source_content_digest"],  # type: ignore[arg-type]
            entries=tuple(RuleRegistryEntryV1.from_dict(entry) for entry in raw_entries),
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )
        claimed_digest = _require_sha256(
            value["normalized_entry_digest"],
            "normalized_entry_digest",
        )
        if claimed_digest != snapshot.normalized_entry_digest:
            raise StratificationProtocolError(
                "normalized_entry_digest does not match the registry inventory"
            )
        return snapshot


@dataclass(frozen=True, slots=True)
class CoverageCountV1:
    """Accepted/failure denominator for one rule and direction."""

    rule_id: str
    direction: str
    registered: bool
    supported: bool | None
    accepted_count: int
    failure_count: int

    def __post_init__(self) -> None:
        _require_nonblank(self.rule_id, "coverage rule_id")
        if self.direction not in {"forward", "backward", _MISSING}:
            raise StratificationProtocolError("invalid coverage direction")
        if type(self.registered) is not bool:
            raise StratificationProtocolError("registered must be a boolean")
        if self.supported is not None and type(self.supported) is not bool:
            raise StratificationProtocolError("supported must be boolean or None")
        _nonnegative(self.accepted_count, "accepted_count")
        _nonnegative(self.failure_count, "failure_count")

    @property
    def total_count(self) -> int:
        return self.accepted_count + self.failure_count

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted_count": self.accepted_count,
            "direction": self.direction,
            "failure_count": self.failure_count,
            "registered": self.registered,
            "rule_id": self.rule_id,
            "supported": self.supported,
            "total_count": self.total_count,
        }


def _validate_counts(
    counts: tuple[tuple[str, int], ...],
    label: str,
    expected_total: int,
) -> None:
    if (
        not isinstance(counts, tuple)
        or any(
            not isinstance(row, tuple)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not row[0]
            or type(row[1]) is not int
            or row[1] < 0
            for row in counts
        )
        or tuple(sorted(set(counts))) != counts
    ):
        raise StratificationProtocolError(f"{label} must be sorted unique string/count pairs")
    if sum(count for _, count in counts) != expected_total:
        raise StratificationProtocolError(
            f"{label} counts must reconstruct the complete row denominator"
        )


@dataclass(frozen=True, slots=True)
class StratificationReportV1:
    """Complete descriptive counts without sampling or denominator changes."""

    registry: RuleRegistrySnapshotV1
    accepted_count: int
    failure_count: int
    rule_direction_coverage: tuple[CoverageCountV1, ...]
    occurrence_depth_counts: tuple[tuple[str, int], ...]
    current_family_counts: tuple[tuple[str, int], ...]
    goal_family_counts: tuple[tuple[str, int], ...]
    remaining_witness_length_counts: tuple[tuple[str, int], ...]
    trace_length_counts: tuple[tuple[str, int], ...]
    domain_mode_counts: tuple[tuple[str, int], ...]
    rewrite_mode_counts: tuple[tuple[str, int], ...]
    support_status_counts: tuple[tuple[str, int], ...]
    failure_code_counts: tuple[tuple[str, int], ...]
    schema_version: str = STRATIFICATION_REPORT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STRATIFICATION_REPORT_VERSION:
            raise StratificationProtocolError("unexpected stratification report schema")
        if not isinstance(self.registry, RuleRegistrySnapshotV1):
            raise StratificationProtocolError("registry must be RuleRegistrySnapshotV1")
        _nonnegative(self.accepted_count, "accepted_count")
        _nonnegative(self.failure_count, "failure_count")
        total = self.accepted_count + self.failure_count
        if not isinstance(self.rule_direction_coverage, tuple) or any(
            not isinstance(row, CoverageCountV1) for row in self.rule_direction_coverage
        ):
            raise StratificationProtocolError(
                "rule_direction_coverage must contain CoverageCountV1"
            )
        keys = tuple((row.rule_id, row.direction) for row in self.rule_direction_coverage)
        if tuple(sorted(set(keys))) != keys:
            raise StratificationProtocolError("rule_direction_coverage must be unique and sorted")
        if sum(row.total_count for row in self.rule_direction_coverage) != total:
            raise StratificationProtocolError(
                "rule coverage must reconstruct the complete row denominator"
            )
        for label, counts in (
            ("occurrence_depth_counts", self.occurrence_depth_counts),
            ("current_family_counts", self.current_family_counts),
            ("goal_family_counts", self.goal_family_counts),
            (
                "remaining_witness_length_counts",
                self.remaining_witness_length_counts,
            ),
            ("trace_length_counts", self.trace_length_counts),
            ("domain_mode_counts", self.domain_mode_counts),
            ("rewrite_mode_counts", self.rewrite_mode_counts),
            ("support_status_counts", self.support_status_counts),
        ):
            _validate_counts(counts, label, total)
        _validate_counts(
            self.failure_code_counts,
            "failure_code_counts",
            self.failure_count,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted_count": self.accepted_count,
            "current_family_counts": dict(self.current_family_counts),
            "domain_mode_counts": dict(self.domain_mode_counts),
            "failure_code_counts": dict(self.failure_code_counts),
            "failure_count": self.failure_count,
            "goal_family_counts": dict(self.goal_family_counts),
            "occurrence_depth_counts": dict(self.occurrence_depth_counts),
            "registry": self.registry.as_dict(),
            "remaining_witness_length_counts": dict(self.remaining_witness_length_counts),
            "rewrite_mode_counts": dict(self.rewrite_mode_counts),
            "rule_direction_coverage": [row.as_dict() for row in self.rule_direction_coverage],
            "schema_version": self.schema_version,
            "support_status_counts": dict(self.support_status_counts),
            "trace_length_counts": dict(self.trace_length_counts),
        }


def _count(values: Sequence[object]) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted(Counter(_MISSING if value is None else str(value) for value in values).items())
    )


def build_stratification_report(
    accepted: Sequence[StepRecordV1],
    failures: Sequence[StepFailureV1],
    registry: RuleRegistrySnapshotV1,
) -> StratificationReportV1:
    """Count the immutable rows exactly once in every descriptive dimension."""

    from geml.data.steps.extract import StepFailureV1, StepRecordV1

    if any(not isinstance(row, StepRecordV1) for row in accepted):
        raise TypeError("accepted must contain StepRecordV1 rows")
    if any(not isinstance(row, StepFailureV1) for row in failures):
        raise TypeError("failures must contain StepFailureV1 rows")
    if not isinstance(registry, RuleRegistrySnapshotV1):
        raise TypeError("registry must be RuleRegistrySnapshotV1")
    rows = (*accepted, *failures)

    accepted_by_key = Counter((row.rule_id, row.direction.value) for row in accepted)
    failure_by_key = Counter(
        (
            _MISSING if row.rule_id is None else row.rule_id,
            _MISSING if row.direction is None else row.direction.value,
        )
        for row in failures
    )
    registry_by_key = {(entry.rule_id, entry.direction): entry for entry in registry.entries}
    observed_keys = set(accepted_by_key) | set(failure_by_key)
    all_keys = sorted(set(registry_by_key) | observed_keys)
    coverage = tuple(
        CoverageCountV1(
            rule_id=rule_id,
            direction=direction,
            registered=(rule_id, direction) in registry_by_key,
            supported=(
                None
                if (rule_id, direction) not in registry_by_key
                else registry_by_key[(rule_id, direction)].supported
            ),
            accepted_count=accepted_by_key[(rule_id, direction)],
            failure_count=failure_by_key[(rule_id, direction)],
        )
        for rule_id, direction in all_keys
    )

    remaining: list[int | None] = []
    for row in rows:
        if isinstance(row, StepRecordV1):
            remaining.append(row.remaining_witness_steps)
        elif (
            row.step_index is not None
            and row.trace_length is not None
            and row.trace_length > row.step_index
        ):
            remaining.append(row.trace_length - row.step_index)
        else:
            remaining.append(None)
    support_values = [
        "supported" if row.supported else "unsupported" if row.supported is not None else "unknown"
        for row in rows
    ]
    return StratificationReportV1(
        registry=registry,
        accepted_count=len(accepted),
        failure_count=len(failures),
        rule_direction_coverage=coverage,
        occurrence_depth_counts=_count([row.occurrence_depth for row in rows]),
        current_family_counts=_count([row.current_family for row in rows]),
        goal_family_counts=_count([row.goal_family for row in rows]),
        remaining_witness_length_counts=_count(remaining),
        trace_length_counts=_count([row.trace_length for row in rows]),
        domain_mode_counts=_count([row.domain_mode for row in rows]),
        rewrite_mode_counts=_count([row.rewrite_mode for row in rows]),
        support_status_counts=_count(support_values),
        failure_code_counts=_count([row.failure_code.value for row in failures]),
    )
