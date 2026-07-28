"""Conservative Goal 8 aggregation and Gate G8 decision rules."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class GateVerdict(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class Goal8SummaryV1:
    attempted: int
    complete: int
    failed_or_timeout: int
    verifier_replayed_successes: int
    verdict: GateVerdict
    reasons: tuple[str, ...]


def summarize_rows(rows: Iterable[dict[str, object]]) -> Goal8SummaryV1:
    rows = tuple(rows)
    statuses = Counter(str(row.get("status")) for row in rows)
    successes = sum(bool(row.get("solved")) and bool(row.get("verifier_replayed")) for row in rows)
    reasons = ["no authenticated 256-problem and 1,000-expression production manifests are staged"]
    if statuses["pending"] or not rows:
        reasons.append("one or more predeclared execution cells remain pending")
    if statuses["failed"] or statuses["timeout"]:
        reasons.append("failed/timeout rows remain in the denominator")
    return Goal8SummaryV1(
        attempted=len(rows),
        complete=statuses["complete"],
        failed_or_timeout=statuses["failed"] + statuses["timeout"],
        verifier_replayed_successes=successes,
        verdict=GateVerdict.INSUFFICIENT_EVIDENCE,
        reasons=tuple(reasons),
    )
