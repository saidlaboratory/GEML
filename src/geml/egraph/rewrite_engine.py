"""Bounded equality saturation over the GEML e-graph.

Machinery only; no algebraic identity is defined here. Each iteration searches every
enabled rule without mutating, applies the batch, then rebuilds congruence. Saturation
stops at a fixed point or a resource limit, always with an explicit status and reason. A
failed guard is a recorded outcome, not a dropped match.
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
from geml.eml.ir import is_valid_source_variable_name

if TYPE_CHECKING:
    from geml.egraph.core import EGraph


class RuleConfigurationError(EGraphError):
    """Raised when a rule or rule set is configured in a way that cannot be executed."""


class Assumption(StrEnum):
    """A domain assumption a caller may declare about a source variable; never inferred."""

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
    """Caller-declared assumptions per variable, closed under implication (positive => nonzero)."""

    declarations: tuple[tuple[str, frozenset[Assumption]], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.declarations, tuple):
            raise TypeError("assumption declarations must be a tuple")
        names: list[str] = []
        for declaration in self.declarations:
            if not isinstance(declaration, tuple) or len(declaration) != 2:
                raise TypeError("each assumption declaration must be a (name, assumptions) pair")
            name, assumptions = declaration
            if not is_valid_source_variable_name(name):
                raise ValueError(f"invalid source variable name in assumptions: {name!r}")
            if not isinstance(assumptions, frozenset) or any(
                not isinstance(value, Assumption) for value in assumptions
            ):
                raise TypeError("declared assumptions must be a frozenset of Assumption values")
            closed: set[Assumption] = set()
            for assumption in assumptions:
                closed |= _ASSUMPTION_CLOSURE[assumption]
            if frozenset(closed) != assumptions:
                raise ValueError(f"assumptions for {name!r} are not implication-closed")
            names.append(name)
        if names != sorted(set(names)):
            raise ValueError("assumption declarations must be unique and sorted by variable name")

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
        for declared_name, assumptions in self.declarations:
            if declared_name == name:
                return assumptions
        return frozenset()

    def holds(self, name: str, assumption: Assumption) -> bool:
        return assumption in self.assumptions_for(name)


@dataclass(frozen=True, slots=True)
class RewriteContext:
    """Everything a guard or applier is allowed to consult besides the e-graph."""

    mode: RewriteMode = RewriteMode.SAFE_REAL
    assumptions: AssumptionEnvironment = field(default_factory=AssumptionEnvironment)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RewriteMode):
            raise TypeError("rewrite mode must be a RewriteMode")
        if not isinstance(self.assumptions, AssumptionEnvironment):
            raise TypeError("assumptions must be an AssumptionEnvironment")


@runtime_checkable
class Guard(Protocol):
    """A named predicate deciding whether a matched rule may fire."""

    @property
    def name(self) -> str: ...

    def __call__(
        self, egraph: EGraph, substitution: Substitution, context: RewriteContext
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class ApplierResult:
    """What an applier produced, or why it declined; declining is a first-class outcome."""

    eclass: EClassId | None
    detail: str = ""

    @classmethod
    def produced(cls, eclass: EClassId) -> ApplierResult:
        return cls(eclass=eclass)

    @classmethod
    def declined(cls, detail: str) -> ApplierResult:
        return cls(eclass=None, detail=detail)


@runtime_checkable
class Applier(Protocol):
    """Builds the right-hand side of a rule for one match."""

    @property
    def required_variables(self) -> frozenset[str]: ...

    def __call__(
        self, egraph: EGraph, substitution: Substitution, context: RewriteContext
    ) -> ApplierResult: ...


@dataclass(frozen=True, slots=True)
class PatternApplier:
    """An applier that instantiates a fixed right-hand-side pattern."""

    rhs: Pattern

    @property
    def required_variables(self) -> frozenset[str]:
        return pattern_variables(self.rhs)

    def __call__(
        self, egraph: EGraph, substitution: Substitution, context: RewriteContext
    ) -> ApplierResult:
        return ApplierResult.produced(instantiate(egraph, self.rhs, substitution))


@dataclass(frozen=True, slots=True)
class RewriteRule:
    """One directed, guarded rewrite; its safety policy comes from the frozen policy module."""

    policy: RulePolicy
    lhs: Pattern
    applier: Applier
    direction: RewriteDirection = RewriteDirection.FORWARD
    guard: Guard | None = None
    rhs: Pattern | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy, RulePolicy):
            raise RuleConfigurationError("rewrite rule policy must be a RulePolicy")
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
        if (
            self.policy.tier
            in (
                RuleTier.GUARDED,
                RuleTier.VERIFIED_GUARDED,
                RuleTier.OPTIONAL,
            )
            and self.guard is None
        ):
            raise RuleConfigurationError(
                f"rule {self.policy.rule_id}: {self.policy.tier.value} rules require a guard"
            )

    @property
    def rule_id(self) -> str:
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
    """Build both orientations of an equivalence, sharing a rule id but differing in direction."""
    return (
        pattern_rule(policy, left, right, guard=guard, direction=RewriteDirection.FORWARD),
        pattern_rule(policy, right, left, guard=guard, direction=RewriteDirection.BACKWARD),
    )


@dataclass(frozen=True, slots=True)
class RuleSet:
    """An ordered, immutable collection of rewrite rules."""

    rules: tuple[RewriteRule, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.rules, tuple) or any(
            not isinstance(rule, RewriteRule) for rule in self.rules
        ):
            raise TypeError("rules must be a tuple of RewriteRule values")

    def __iter__(self) -> Iterator[RewriteRule]:
        return iter(self.rules)

    def __len__(self) -> int:
        return len(self.rules)

    def enabled_for(self, mode: RewriteMode) -> RuleSet:
        """Return the subset of rules permitted to run under ``mode``."""
        return RuleSet(rules=tuple(rule for rule in self.rules if rule.enabled_in(mode)))

    def by_id(self, rule_id: str) -> tuple[RewriteRule, ...]:
        return tuple(rule for rule in self.rules if rule.rule_id == rule_id)

    def merged_with(self, other: RuleSet) -> RuleSet:
        return RuleSet(rules=self.rules + other.rules)


@dataclass(frozen=True, slots=True)
class SaturationLimits:
    """Resource bounds for one run: the frozen ResourceLimits plus an e-class bound."""

    resources: ResourceLimits = field(default_factory=ResourceLimits)
    max_eclasses: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.resources, ResourceLimits):
            raise TypeError("resources must be ResourceLimits")
        if self.max_eclasses is not None and (
            type(self.max_eclasses) is not int or self.max_eclasses < 1
        ):
            raise ValueError("max_eclasses must be null or a positive integer")


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
    """Run bounded equality saturation, capped by iterations, node/e-class count, and wall clock."""
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

        remaining_attempts = resources.max_rewrite_attempts - attempted
        if remaining_attempts <= 0:
            status = ExtractionStatus.ITERATION_LIMIT
            reason = f"max_rewrite_attempts={resources.max_rewrite_attempts} reached"
            break
        matches: list[tuple[RewriteRule, Match]] = []
        for rule in active_rules:
            remaining = remaining_attempts - len(matches)
            if remaining <= 0:
                break
            matches.extend(
                (rule, match) for match in search_pattern(egraph, rule.lhs, max_matches=remaining)
            )
        if not matches:
            saturated = True
            status = ExtractionStatus.SUCCESS
            reason = "no enabled rule matched the e-graph"
            break

        changed = False
        for rule, match in matches:
            if attempted >= resources.max_rewrite_attempts:
                status = ExtractionStatus.ITERATION_LIMIT
                reason = f"max_rewrite_attempts={resources.max_rewrite_attempts} reached"
                break
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
        if attempted >= resources.max_rewrite_attempts:
            status = ExtractionStatus.ITERATION_LIMIT
            reason = f"max_rewrite_attempts={resources.max_rewrite_attempts} reached"
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


def _expired(started: float, timeout_seconds: float) -> bool:
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
        verifier_required=rule.policy.verifier_required,
        justification=rule.policy.justification,
        assumptions=rule.policy.assumptions,
        source_eclass=match.eclass,
        result_eclass=result_eclass,
        substitution=match.substitution,
        detail=detail,
    )
