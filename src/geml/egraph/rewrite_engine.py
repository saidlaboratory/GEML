"""Bounded equality saturation over the GEML e-graph.

This module supplies the *machinery* for rewriting and deliberately contains no algebra:
not one mathematical identity is defined here.  Rules are data supplied by
:mod:`geml.egraph.rules_safe` and :mod:`geml.egraph.rules_domain`.

The saturation loop is the standard two-phase algorithm.  Each iteration first *searches*
every enabled rule against the current e-graph, collecting matches without mutating
anything, and then *applies* the whole batch.  Separating the phases keeps an iteration's
result independent of the order in which rules happen to grow the e-graph, which is what
makes a run reproducible.

An iteration ends with a congruence rebuild.  Saturation stops when a full iteration adds
no new equality, or when a resource limit is reached.  Every stop is reported with an
explicit status and a human-readable reason; the loop never returns quietly.

Guards are optional predicates attached to a rule.  A guard that fails is a recorded
outcome, not a dropped match: the provenance log keeps the rejection so that the reporting
denominator stays "attempted rewrites" as the rewrite policy requires.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from geml.egraph.core import EGraphStats, ResourceLimitError
from geml.egraph.ir import EClassId, EGraphError
from geml.egraph.patterns import (
    Match,
    Pattern,
    PatternNode,
    Substitution,
    instantiate,
    pattern_variables,
    search_pattern,
)
from geml.egraph.policy import (
    ExtractionStatus,
    ResourceLimits,
    RewriteMode,
    RulePolicy,
    RuleTier,
    SaturationReport,
)
from geml.egraph.provenance import (
    ApplicationOutcome,
    GuardOutcome,
    ProvenanceLog,
    RewriteDirection,
)

if TYPE_CHECKING:
    from geml.egraph.core import EGraph


class RuleConfigurationError(EGraphError):
    """Raised when a rule or rule set is configured in a way that cannot be executed."""


class Assumption(StrEnum):
    """A domain assumption a caller may declare about a source variable.

    These are *declared*, never inferred.  A rule that needs ``x > 0`` will not fire
    unless the caller has said so, because inferring positivity from syntax is exactly the
    kind of hidden reasoning Goal 4 forbids.
    """

    REAL = "real"
    POSITIVE = "positive"
    NONNEGATIVE = "nonnegative"
    NONZERO = "nonzero"


_ASSUMPTION_CLOSURE: dict[Assumption, frozenset[Assumption]] = {
    Assumption.REAL: frozenset({Assumption.REAL}),
    Assumption.POSITIVE: frozenset(
        {Assumption.POSITIVE, Assumption.NONNEGATIVE, Assumption.NONZERO, Assumption.REAL}
    ),
    Assumption.NONNEGATIVE: frozenset({Assumption.NONNEGATIVE, Assumption.REAL}),
    Assumption.NONZERO: frozenset({Assumption.NONZERO, Assumption.REAL}),
}


@dataclass(frozen=True, slots=True)
class AssumptionEnvironment:
    """Caller-declared assumptions, keyed by source variable name.

    The environment is closed under the obvious implications (a positive real is also
    nonnegative, nonzero, and real) so that a rule needing ``x != 0`` fires when the caller
    declared ``x > 0``.  No other inference happens.
    """

    declarations: tuple[tuple[str, frozenset[Assumption]], ...] = ()

    @classmethod
    def of(cls, **variables: Iterable[Assumption | str]) -> AssumptionEnvironment:
        """Build an environment from ``name=("positive", ...)`` keyword arguments."""
        rows: list[tuple[str, frozenset[Assumption]]] = []
        for name in sorted(variables):
            closed: set[Assumption] = set()
            for raw in variables[name]:
                assumption = Assumption(raw)
                closed |= _ASSUMPTION_CLOSURE[assumption]
            rows.append((name, frozenset(closed)))
        return cls(declarations=tuple(rows))

    def assumptions_for(self, name: str) -> frozenset[Assumption]:
        """Return the closed assumption set declared for ``name``."""
        for declared_name, assumptions in self.declarations:
            if declared_name == name:
                return assumptions
        return frozenset()

    def holds(self, name: str, assumption: Assumption) -> bool:
        """Return whether ``assumption`` was declared for ``name``."""
        return assumption in self.assumptions_for(name)


@dataclass(frozen=True, slots=True)
class RewriteContext:
    """Everything a guard or applier is allowed to consult besides the e-graph."""

    mode: RewriteMode = RewriteMode.SAFE_REAL
    assumptions: AssumptionEnvironment = field(default_factory=AssumptionEnvironment)


@runtime_checkable
class Guard(Protocol):
    """A named predicate deciding whether a matched rule may fire."""

    @property
    def name(self) -> str:
        """Return a stable identifier recorded in provenance when the guard rejects."""

    def __call__(self, egraph: EGraph, substitution: Substitution, context: RewriteContext) -> bool:
        """Return whether the rule may be applied under this match."""


@dataclass(frozen=True, slots=True)
class ApplierResult:
    """What an applier produced, or why it declined.

    Declining is a first-class outcome.  A folding applier that refuses a constant beyond
    its exactness bound reports that refusal rather than returning a wrong answer.
    """

    eclass: EClassId | None
    detail: str = ""

    @classmethod
    def produced(cls, eclass: EClassId) -> ApplierResult:
        """Return a successful result."""
        return cls(eclass=eclass)

    @classmethod
    def declined(cls, detail: str) -> ApplierResult:
        """Return a refusal carrying an explanation."""
        return cls(eclass=None, detail=detail)


@runtime_checkable
class Applier(Protocol):
    """Builds the right-hand side of a rule for one match."""

    @property
    def required_variables(self) -> frozenset[str]:
        """Return the pattern variables this applier reads."""

    def __call__(
        self, egraph: EGraph, substitution: Substitution, context: RewriteContext
    ) -> ApplierResult:
        """Insert the right-hand side and return the resulting e-class."""


@dataclass(frozen=True, slots=True)
class PatternApplier:
    """An applier that instantiates a fixed right-hand-side pattern."""

    rhs: Pattern

    @property
    def required_variables(self) -> frozenset[str]:
        """Return the variables occurring in the right-hand side."""
        return pattern_variables(self.rhs)

    def __call__(
        self, egraph: EGraph, substitution: Substitution, context: RewriteContext
    ) -> ApplierResult:
        """Insert the right-hand side under ``substitution``."""
        return ApplierResult.produced(instantiate(egraph, self.rhs, substitution))


@dataclass(frozen=True, slots=True)
class RewriteRule:
    """One directed, guarded rewrite together with its safety policy.

    A rule never carries its own notion of safety: the tier, assumptions, justification,
    and the modes it is enabled in all come from the :class:`RulePolicy` recorded in the
    frozen policy module.
    """

    policy: RulePolicy
    lhs: Pattern
    applier: Applier
    direction: RewriteDirection = RewriteDirection.FORWARD
    guard: Guard | None = None
    rhs: Pattern | None = None

    def __post_init__(self) -> None:
        """Reject rules that could not be executed soundly."""
        if not isinstance(self.lhs, PatternNode):
            raise RuleConfigurationError(
                f"rule {self.policy.rule_id}: the left-hand side must have an operator at its root"
            )
        available = pattern_variables(self.lhs)
        missing = self.applier.required_variables - available
        if missing:
            raise RuleConfigurationError(
                f"rule {self.policy.rule_id}: right-hand side uses unbound variables "
                f"{sorted(missing)}"
            )
        if self.policy.tier is RuleTier.UNSAFE:
            raise RuleConfigurationError(
                f"rule {self.policy.rule_id}: unsafe rules must never be constructed for execution"
            )

    @property
    def rule_id(self) -> str:
        """Return the policy rule identifier."""
        return self.policy.rule_id

    def enabled_in(self, mode: RewriteMode) -> bool:
        """Return whether this rule may run under ``mode``."""
        if self.policy.tier in (RuleTier.UNSAFE, RuleTier.UNCLASSIFIED):
            return False
        return mode in self.policy.enabled_in


def pattern_rule(
    policy: RulePolicy,
    lhs: Pattern,
    rhs: Pattern,
    *,
    guard: Guard | None = None,
    direction: RewriteDirection = RewriteDirection.FORWARD,
) -> RewriteRule:
    """Build a directed rule that rewrites ``lhs`` into ``rhs``."""
    return RewriteRule(
        policy=policy,
        lhs=lhs,
        applier=PatternApplier(rhs=rhs),
        direction=direction,
        guard=guard,
        rhs=rhs,
    )


def bidirectional_rules(
    policy: RulePolicy,
    left: Pattern,
    right: Pattern,
    *,
    guard: Guard | None = None,
) -> tuple[RewriteRule, RewriteRule]:
    """Build both orientations of an equivalence as two directed rules.

    They share a rule identifier and are told apart by their recorded direction, so
    provenance shows which way an equivalence was used.
    """
    return (
        pattern_rule(policy, left, right, guard=guard, direction=RewriteDirection.FORWARD),
        pattern_rule(policy, right, left, guard=guard, direction=RewriteDirection.BACKWARD),
    )


@dataclass(frozen=True, slots=True)
class RuleSet:
    """An ordered, immutable collection of rewrite rules."""

    rules: tuple[RewriteRule, ...] = ()

    def __iter__(self) -> Iterator[RewriteRule]:
        """Iterate the rules in declaration order."""
        return iter(self.rules)

    def __len__(self) -> int:
        """Return how many rules the set holds."""
        return len(self.rules)

    def enabled_for(self, mode: RewriteMode) -> RuleSet:
        """Return the subset of rules permitted to run under ``mode``."""
        return RuleSet(rules=tuple(rule for rule in self.rules if rule.enabled_in(mode)))

    def by_id(self, rule_id: str) -> tuple[RewriteRule, ...]:
        """Return every orientation of the rule with ``rule_id``."""
        return tuple(rule for rule in self.rules if rule.rule_id == rule_id)

    def merged_with(self, other: RuleSet) -> RuleSet:
        """Return the concatenation of two rule sets, preserving order."""
        return RuleSet(rules=self.rules + other.rules)


@dataclass(frozen=True, slots=True)
class SaturationLimits:
    """Resource bounds for one saturation run.

    Wraps the frozen :class:`~geml.egraph.policy.ResourceLimits` and adds the e-class bound
    the rewrite engine needs, which the policy module does not carry.
    """

    resources: ResourceLimits = field(default_factory=ResourceLimits)
    max_eclasses: int | None = None


@dataclass(frozen=True, slots=True)
class SaturationOutcome:
    """The complete, explicit result of a saturation run."""

    report: SaturationReport
    provenance: ProvenanceLog
    stats: EGraphStats


def saturate(
    egraph: EGraph,
    rules: RuleSet,
    context: RewriteContext | None = None,
    limits: SaturationLimits | None = None,
) -> SaturationOutcome:
    """Run bounded equality saturation and return an explicit outcome.

    Complexity is not bounded in general — equality saturation need not terminate, which is
    precisely why every run is capped by iteration count, node count, e-class count, and
    wall clock.  Within one iteration the cost is the search cost of every enabled rule
    plus one congruence rebuild.
    """
    active_context = context if context is not None else RewriteContext()
    active_limits = limits if limits is not None else SaturationLimits()
    resources = active_limits.resources
    active_rules = rules.enabled_for(active_context.mode)

    log = ProvenanceLog()
    started = time.monotonic()
    iterations = 0
    attempted = 0
    applied = 0
    saturated = False
    status: ExtractionStatus | None = None
    reason = ""

    if len(active_rules) == 0:
        return _finish(
            egraph,
            log,
            iterations=0,
            attempted=0,
            applied=0,
            saturated=True,
            status=ExtractionStatus.SUCCESS,
            reason="no rule is enabled in this rewrite mode",
        )

    while status is None:
        if iterations >= resources.max_iterations:
            status = ExtractionStatus.ITERATION_LIMIT
            reason = f"max_iterations={resources.max_iterations} reached"
            break
        if _expired(started, resources.timeout_seconds):
            status = ExtractionStatus.TIMEOUT
            reason = f"timeout_seconds={resources.timeout_seconds} elapsed"
            break

        matches = [
            (rule, match) for rule in active_rules for match in search_pattern(egraph, rule.lhs)
        ]
        if not matches:
            saturated = True
            status = ExtractionStatus.SUCCESS
            reason = "no enabled rule matched the e-graph"
            break

        changed = False
        for rule, match in matches:
            limit_status, limit_reason = _limit_hit(egraph, active_limits, started)
            if limit_status is not None:
                _record(
                    log,
                    iterations,
                    active_context.mode,
                    rule,
                    match,
                    ApplicationOutcome.LIMIT_REACHED,
                    limit_reason,
                )
                attempted += 1
                status = limit_status
                reason = limit_reason
                break

            attempted += 1
            guard_outcome = _evaluate_guard(rule, egraph, match.substitution, active_context)
            if guard_outcome is GuardOutcome.FAILED:
                _record(
                    log,
                    iterations,
                    active_context.mode,
                    rule,
                    match,
                    ApplicationOutcome.GUARD_REJECTED,
                    f"guard {rule.guard.name} rejected the match",
                    guard=guard_outcome,
                )
                continue

            try:
                result = rule.applier(egraph, match.substitution, active_context)
            except ResourceLimitError as error:
                _record(
                    log,
                    iterations,
                    active_context.mode,
                    rule,
                    match,
                    ApplicationOutcome.LIMIT_REACHED,
                    str(error),
                    guard=guard_outcome,
                )
                status = ExtractionStatus.NODE_LIMIT
                reason = str(error)
                break

            if result.eclass is None:
                _record(
                    log,
                    iterations,
                    active_context.mode,
                    rule,
                    match,
                    ApplicationOutcome.UNSUPPORTED,
                    result.detail,
                    guard=guard_outcome,
                )
                continue

            merged = egraph.merge(egraph.find(match.eclass), result.eclass)
            _record(
                log,
                iterations,
                active_context.mode,
                rule,
                match,
                ApplicationOutcome.APPLIED if merged else ApplicationOutcome.NO_CHANGE,
                result.detail,
                guard=guard_outcome,
                result_eclass=result.eclass,
            )
            if merged:
                applied += 1
                changed = True

        rebuild_report = egraph.rebuild()
        iterations += 1

        if status is not None:
            break
        if not rebuild_report.congruence_closed:
            status = ExtractionStatus.PARTIAL_SUCCESS
            reason = "congruence closure did not converge within max_iterations"
            break
        if not changed and not rebuild_report.changed:
            saturated = True
            status = ExtractionStatus.SUCCESS
            reason = "fixed point reached"

    return _finish(
        egraph,
        log,
        iterations=iterations,
        attempted=attempted,
        applied=applied,
        saturated=saturated,
        status=status if status is not None else ExtractionStatus.PARTIAL_SUCCESS,
        reason=reason,
    )


def _finish(
    egraph: EGraph,
    log: ProvenanceLog,
    *,
    iterations: int,
    attempted: int,
    applied: int,
    saturated: bool,
    status: ExtractionStatus,
    reason: str,
) -> SaturationOutcome:
    """Assemble the final outcome."""
    return SaturationOutcome(
        report=SaturationReport(
            iterations=iterations,
            rewrites_attempted=attempted,
            rewrites_applied=applied,
            status=status,
            saturated=saturated,
            reason=reason,
        ),
        provenance=log,
        stats=egraph.stats(),
    )


def _expired(started: float, timeout_seconds: int) -> bool:
    """Return whether the wall-clock budget is exhausted."""
    return time.monotonic() - started >= timeout_seconds


def _limit_hit(
    egraph: EGraph, limits: SaturationLimits, started: float
) -> tuple[ExtractionStatus | None, str]:
    """Return the limit status blocking further rewriting, if any."""
    stats = egraph.stats()
    if stats.node_count >= limits.resources.max_egraph_nodes:
        return (
            ExtractionStatus.NODE_LIMIT,
            f"max_egraph_nodes={limits.resources.max_egraph_nodes} reached",
        )
    if limits.max_eclasses is not None and stats.root_count >= limits.max_eclasses:
        return ExtractionStatus.NODE_LIMIT, f"max_eclasses={limits.max_eclasses} reached"
    if _expired(started, limits.resources.timeout_seconds):
        return (
            ExtractionStatus.TIMEOUT,
            f"timeout_seconds={limits.resources.timeout_seconds} elapsed",
        )
    return None, ""


def _evaluate_guard(
    rule: RewriteRule,
    egraph: EGraph,
    substitution: Substitution,
    context: RewriteContext,
) -> GuardOutcome:
    """Evaluate a rule's guard, if it has one."""
    if rule.guard is None:
        return GuardOutcome.NOT_REQUIRED
    return GuardOutcome.PASSED if rule.guard(egraph, substitution, context) else GuardOutcome.FAILED


def _record(
    log: ProvenanceLog,
    iteration: int,
    mode: RewriteMode,
    rule: RewriteRule,
    match: Match,
    outcome: ApplicationOutcome,
    detail: str,
    *,
    guard: GuardOutcome = GuardOutcome.NOT_REQUIRED,
    result_eclass: EClassId | None = None,
) -> None:
    """Append one attempt to the provenance log.

    The recorded mode is the mode the run was executed under, not a property of the rule,
    so a log row states the conditions the rewrite actually happened in.
    """
    log.record(
        iteration=iteration,
        rule_id=rule.policy.rule_id,
        rule_name=rule.policy.name,
        tier=rule.policy.tier,
        mode=mode,
        direction=rule.direction,
        guard=guard,
        outcome=outcome,
        branch_sensitive=rule.policy.branch_sensitive,
        assumptions=rule.policy.assumptions,
        source_eclass=match.eclass,
        result_eclass=result_eclass,
        substitution=match.substitution,
        detail=detail,
    )
