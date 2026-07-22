"""Append-only provenance for every rewrite the engine considers.

Goal 4 requires that an optimization run be reconstructable after the fact.  That is only
possible if the log records *attempts*, not just successes: a rule that matched but was
rejected by its guard, or that produced a term the e-graph already contained, is evidence
about the run and must survive.  The reporting denominator defined by the rewrite policy
is attempted rewrites, so this module never discards a record.

Each record names the rule, its safety tier, the rewrite mode in force, the direction the
rule was applied in, the guard outcome, the assumptions the rule depended on, whether the
identity is branch sensitive, and the e-classes involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from geml.egraph.ir import EClassId
from geml.egraph.patterns import Substitution
from geml.egraph.policy import RewriteMode, RuleTier


class RewriteDirection(StrEnum):
    """Which orientation of a rule was applied.

    A bidirectional rule is expanded into two directed rules, so a record always names one
    concrete orientation rather than an ambiguous "either way".
    """

    FORWARD = "forward"
    BACKWARD = "backward"


class GuardOutcome(StrEnum):
    """Result of evaluating a rule's guard."""

    NOT_REQUIRED = "not_required"
    PASSED = "passed"
    FAILED = "failed"


class ApplicationOutcome(StrEnum):
    """What happened once a match was found.

    ``APPLIED`` means the rewrite equated two previously distinct e-classes.
    ``NO_CHANGE`` means the right-hand side was built but the e-graph already knew the
    equality, which is the normal signal that saturation is approaching a fixed point.
    """

    APPLIED = "applied"
    NO_CHANGE = "no_change"
    GUARD_REJECTED = "guard_rejected"
    UNSUPPORTED = "unsupported"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True, slots=True)
class RewriteRecord:
    """One immutable entry in a :class:`ProvenanceLog`."""

    sequence_index: int
    iteration: int
    rule_id: str
    rule_name: str
    tier: RuleTier
    mode: RewriteMode
    direction: RewriteDirection
    guard: GuardOutcome
    outcome: ApplicationOutcome
    branch_sensitive: bool
    assumptions: frozenset[str]
    source_eclass: EClassId
    result_eclass: EClassId | None
    substitution: Substitution
    detail: str = ""

    @property
    def applied(self) -> bool:
        """Return whether this record changed the e-graph."""
        return self.outcome is ApplicationOutcome.APPLIED


class ProvenanceLog:
    """An ordered, append-only sequence of :class:`RewriteRecord` entries."""

    __slots__ = ("_records",)

    def __init__(self) -> None:
        """Create an empty log."""
        self._records: list[RewriteRecord] = []

    def __len__(self) -> int:
        """Return how many attempts have been recorded."""
        return len(self._records)

    def record(
        self,
        *,
        iteration: int,
        rule_id: str,
        rule_name: str,
        tier: RuleTier,
        mode: RewriteMode,
        direction: RewriteDirection,
        guard: GuardOutcome,
        outcome: ApplicationOutcome,
        branch_sensitive: bool,
        assumptions: frozenset[str],
        source_eclass: EClassId,
        result_eclass: EClassId | None,
        substitution: Substitution,
        detail: str = "",
    ) -> RewriteRecord:
        """Append one attempt and return the stored record."""
        entry = RewriteRecord(
            sequence_index=len(self._records),
            iteration=iteration,
            rule_id=rule_id,
            rule_name=rule_name,
            tier=tier,
            mode=mode,
            direction=direction,
            guard=guard,
            outcome=outcome,
            branch_sensitive=branch_sensitive,
            assumptions=assumptions,
            source_eclass=source_eclass,
            result_eclass=result_eclass,
            substitution=substitution,
            detail=detail,
        )
        self._records.append(entry)
        return entry

    @property
    def records(self) -> tuple[RewriteRecord, ...]:
        """Return every recorded attempt in order."""
        return tuple(self._records)

    def records_for(self, rule_id: str) -> tuple[RewriteRecord, ...]:
        """Return every attempt made by one rule, in order."""
        return tuple(record for record in self._records if record.rule_id == rule_id)

    def attempt_counts(self) -> dict[str, int]:
        """Return attempts per rule identifier, ordered by first appearance."""
        counts: dict[str, int] = {}
        for record in self._records:
            counts[record.rule_id] = counts.get(record.rule_id, 0) + 1
        return counts

    def application_counts(self) -> dict[str, int]:
        """Return successful applications per rule identifier, ordered by first appearance."""
        counts: dict[str, int] = {}
        for record in self._records:
            counts.setdefault(record.rule_id, 0)
            if record.applied:
                counts[record.rule_id] += 1
        return counts

    def outcome_counts(self) -> dict[ApplicationOutcome, int]:
        """Return how many attempts ended in each outcome."""
        counts: dict[ApplicationOutcome, int] = dict.fromkeys(ApplicationOutcome, 0)
        for record in self._records:
            counts[record.outcome] += 1
        return counts

    def branch_sensitive_records(self) -> tuple[RewriteRecord, ...]:
        """Return every attempt that involved a branch-sensitive identity."""
        return tuple(record for record in self._records if record.branch_sensitive)

    def assumptions_used(self) -> frozenset[str]:
        """Return the union of assumptions relied on by applied rewrites."""
        used: set[str] = set()
        for record in self._records:
            if record.applied:
                used.update(record.assumptions)
        return frozenset(used)
