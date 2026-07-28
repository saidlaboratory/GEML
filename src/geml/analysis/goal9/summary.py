"""Bounded SR aggregation that never uses numeric fit as exact-recovery evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from geml.data.sr.benchmark import ExactRecoveryStatus


class GateVerdict(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class Goal9SummaryV1:
    attempted: int
    verifier_exact: int
    numeric_only: int
    invalid_or_timeout: int
    verdict: GateVerdict


def summarize_rows(rows: tuple[dict[str, object], ...]) -> Goal9SummaryV1:
    statuses = Counter(str(row.get("exact_recovery_status")) for row in rows)
    invalid_or_timeout = sum(
        bool(row.get("timeout")) or row.get("error") is not None for row in rows
    )
    return Goal9SummaryV1(
        attempted=len(rows),
        verifier_exact=statuses[ExactRecoveryStatus.VERIFIED.value],
        numeric_only=statuses[ExactRecoveryStatus.NUMERIC_ONLY.value],
        invalid_or_timeout=invalid_or_timeout,
        verdict=GateVerdict.INSUFFICIENT_EVIDENCE,
    )
