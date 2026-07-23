"""Guarded, domain-restricted rewrite rules for the GEML e-graph.

Every identity here is conditional, none is a universal complex identity, and none runs in
``SAFE_REAL`` mode: they fire only in ``POSITIVE_REAL_FORMAL`` mode when a guard confirms
the caller declared the needed assumption. Guards are local (they inspect only the bound
e-class), so ``log(exp(x + y))`` does not fire even with x, y declared. See
``docs/specs/EGRAPH_DOMAIN_RULES.md`` for the statement and counterexample of each rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from geml.egraph.ir import Operator
from geml.egraph.patterns import (
    Pattern,
    PatternNode,
    PatternVar,
    Substitution,
    constant_value,
)
from geml.egraph.policy import RewriteMode, RulePolicy, RuleTier
from geml.egraph.rewrite_engine import (
    Assumption,
    Guard,
    RewriteContext,
    RuleSet,
    bidirectional_rules,
    pattern_rule,
)

if TYPE_CHECKING:
    from geml.egraph.core import EGraph

FORMAL_ONLY: frozenset[RewriteMode] = frozenset({RewriteMode.POSITIVE_REAL_FORMAL})

_A = PatternVar("a")
_B = PatternVar("b")
_X = PatternVar("x")


def _node(op: Operator, *children: Pattern) -> PatternNode:
    return PatternNode(op=op, children=children)


def _constant_satisfies(value: Fraction, assumption: Assumption) -> bool:
    if assumption is Assumption.REAL:
        return True
    if assumption is Assumption.POSITIVE:
        return value > 0
    if assumption is Assumption.NONNEGATIVE:
        return value >= 0
    return value != 0


@dataclass(frozen=True, slots=True)
class DeclaredAssumptionGuard:
    """Guard passing for an exact constant satisfying the assumption, or a variable declared so.

    A compound expression is never assumed to inherit a property from its operands.
    """

    variable: str
    assumption: Assumption

    @property
    def name(self) -> str:
        return f"{self.assumption.value}({self.variable})"

    def __call__(self, egraph: EGraph, substitution: Substitution, context: RewriteContext) -> bool:
        eclass = substitution[self.variable]
        value = constant_value(egraph, eclass)
        if value is not None:
            return _constant_satisfies(value, self.assumption)
        for node in egraph.nodes_of(eclass):
            if node.op is Operator.VARIABLE and context.assumptions.holds(
                str(node.payload), self.assumption
            ):
                return True
        return False


@dataclass(frozen=True, slots=True)
class AllOfGuard:
    """Guard that passes only when every component guard passes."""

    guards: tuple[Guard, ...]

    @property
    def name(self) -> str:
        return " and ".join(guard.name for guard in self.guards)

    def __call__(self, egraph: EGraph, substitution: Substitution, context: RewriteContext) -> bool:
        return all(guard(egraph, substitution, context) for guard in self.guards)


def _positive(variable: str) -> DeclaredAssumptionGuard:
    return DeclaredAssumptionGuard(variable=variable, assumption=Assumption.POSITIVE)


def _real(variable: str) -> DeclaredAssumptionGuard:
    return DeclaredAssumptionGuard(variable=variable, assumption=Assumption.REAL)


def _nonzero(variable: str) -> DeclaredAssumptionGuard:
    return DeclaredAssumptionGuard(variable=variable, assumption=Assumption.NONZERO)


def _policy(
    rule_id: str,
    name: str,
    assumptions: frozenset[str],
    justification: str,
    *,
    tier: RuleTier = RuleTier.GUARDED,
    branch_sensitive: bool = True,
) -> RulePolicy:
    return RulePolicy(
        rule_id=rule_id,
        name=name,
        tier=tier,
        assumptions=assumptions,
        justification=justification,
        enabled_in=FORMAL_ONLY,
        branch_sensitive=branch_sensitive,
        verifier_required=True,
    )


LOG_OF_EXP = pattern_rule(
    _policy(
        "DOMAIN-LOG-EXP",
        "logarithm of an exponential",
        frozenset({"x is real"}),
        "log(exp(x)) = x holds on the real line, but the principal complex logarithm "
        "gives log(exp(x)) = x only when Im(x) lies in (-pi, pi]; for example "
        "log(exp(3i*pi)) = i*pi, not 3i*pi. The GEML EML construction admits complex "
        "intermediates, so this identity is fenced behind a declared real argument.",
    ),
    _node(Operator.LOG, _node(Operator.EXP, _X)),
    _X,
    guard=_real("x"),
)

EXP_OF_LOG = pattern_rule(
    _policy(
        "DOMAIN-EXP-LOG",
        "exponential of a logarithm",
        frozenset({"x > 0"}),
        "exp(log(x)) = x requires x > 0 over the reals; log is undefined at x = 0 and "
        "not real valued for x < 0.",
    ),
    _node(Operator.EXP, _node(Operator.LOG, _X)),
    _X,
    guard=_positive("x"),
)

LOG_OF_PRODUCT = bidirectional_rules(
    _policy(
        "DOMAIN-LOG-PRODUCT",
        "logarithm of a product",
        frozenset({"a > 0", "b > 0"}),
        "log(a*b) = log(a) + log(b) requires both factors positive. With a = b = -1 the "
        "left side is log(1) = 0 while the right side is undefined over the reals, and "
        "over the principal complex branch it is 2i*pi.",
    ),
    _node(Operator.LOG, _node(Operator.MUL, _A, _B)),
    _node(Operator.ADD, _node(Operator.LOG, _A), _node(Operator.LOG, _B)),
    guard=AllOfGuard(guards=(_positive("a"), _positive("b"))),
)

EXP_OF_SUM = bidirectional_rules(
    _policy(
        "DOMAIN-EXP-SUM",
        "exponential of a sum",
        frozenset({"a is real", "b is real"}),
        "exp(a + b) = exp(a) * exp(b) holds for all finite complex arguments, so it is "
        "not branch sensitive. It is still fenced to the formal mode because extended "
        "real arguments break it: exp(inf + (-inf)) is undefined while "
        "exp(inf) * exp(-inf) is an indeterminate product.",
        branch_sensitive=False,
    ),
    _node(Operator.EXP, _node(Operator.ADD, _A, _B)),
    _node(Operator.MUL, _node(Operator.EXP, _A), _node(Operator.EXP, _B)),
    guard=AllOfGuard(guards=(_real("a"), _real("b"))),
)

LOG_OF_POWER = pattern_rule(
    _policy(
        "DOMAIN-LOG-POW",
        "logarithm of a power",
        frozenset({"a > 0"}),
        "log(a**b) = b * log(a) requires a > 0. With a = -1 and b = 2 the left side is "
        "log(1) = 0 while the right side is not real valued.",
    ),
    _node(Operator.LOG, _node(Operator.POW, _A, _B)),
    _node(Operator.MUL, _B, _node(Operator.LOG, _A)),
    guard=_positive("a"),
)


DIV_SELF = pattern_rule(
    _policy(
        "DOMAIN-DIV-SELF",
        "a value divided by itself",
        frozenset({"x != 0"}),
        "x / x = 1 requires x != 0; at x = 0 the quotient is undefined, so the rewrite "
        "would silently extend the domain of the expression.",
        tier=RuleTier.OPTIONAL,
        branch_sensitive=False,
    ),
    _node(Operator.DIV, _X, _X),
    PatternNode(op=Operator.CONSTANT, payload=Fraction(1)),
    guard=_nonzero("x"),
)

POWER_OF_POWER = pattern_rule(
    _policy(
        "DOMAIN-POW-POW",
        "a power raised to a power",
        frozenset({"a > 0"}),
        "(a**b)**c = a**(b*c) requires a > 0. With a = -1, b = 2, c = 1/2 the left side "
        "is 1 while the right side is (-1)**1 = -1.",
        tier=RuleTier.OPTIONAL,
    ),
    _node(Operator.POW, _node(Operator.POW, _A, _B), _X),
    _node(Operator.POW, _A, _node(Operator.MUL, _B, _X)),
    guard=_positive("a"),
)

PRODUCT_OF_POWERS = bidirectional_rules(
    _policy(
        "DOMAIN-POW-MUL",
        "a product of powers of one base",
        frozenset({"a > 0"}),
        "a**b * a**c = a**(b + c) requires a > 0 for real exponents; for a negative base "
        "the individual factors need not be real valued.",
        tier=RuleTier.OPTIONAL,
    ),
    _node(Operator.MUL, _node(Operator.POW, _A, _B), _node(Operator.POW, _A, _X)),
    _node(Operator.POW, _A, _node(Operator.ADD, _B, _X)),
    guard=_positive("a"),
)


DOMAIN_RULES: RuleSet = RuleSet(
    rules=(
        LOG_OF_EXP,
        EXP_OF_LOG,
        *LOG_OF_PRODUCT,
        *EXP_OF_SUM,
        LOG_OF_POWER,
    )
)

OPTIONAL_DOMAIN_RULES: RuleSet = RuleSet(
    rules=(
        DIV_SELF,
        POWER_OF_POWER,
        *PRODUCT_OF_POWERS,
    )
)


def domain_rules(*, include_optional: bool = False) -> RuleSet:
    """Return the domain rule set; the OPTIONAL tier is excluded unless requested."""
    if include_optional:
        return DOMAIN_RULES.merged_with(OPTIONAL_DOMAIN_RULES)
    return DOMAIN_RULES


DOMAIN_RULE_IDS: tuple[str, ...] = tuple(
    dict.fromkeys(rule.rule_id for rule in domain_rules(include_optional=True).rules)
)
