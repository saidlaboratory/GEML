"""Branch-insensitive rewrite rules for the GEML e-graph.

Every rule in this module is valid for all real values of its pattern variables, with one
explicitly fenced exception: the guarded power identities, which carry a nonzero-base
assumption and are therefore disabled in ``SAFE_REAL`` mode.

What is deliberately **absent** matters as much as what is present.  None of the following
appear here, because each is branch sensitive or needs a domain assumption, and they belong
to :mod:`geml.egraph.rules_domain`:

* ``log(exp(x)) -> x``
* ``exp(log(x)) -> x``
* ``log(x*y) -> log(x) + log(y)``
* ``exp(a + b) -> exp(a) * exp(b)``
* ``x / x -> 1``

Constant folding is *bounded exact*.  Folding uses :class:`fractions.Fraction` throughout,
so ``1/3`` stays ``1/3`` and never becomes a binary float, and an applier refuses any
result whose numerator or denominator exceeds a declared digit bound rather than letting a
saturating e-graph grow unbounded integers.  A refusal is reported through the provenance
log as an unsupported application, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from typing import TYPE_CHECKING

from geml.egraph.ir import ENode, Operator
from geml.egraph.patterns import (
    Pattern,
    PatternNode,
    PatternVar,
    Substitution,
    VarKind,
    constant_value,
)
from geml.egraph.policy import RewriteMode, RulePolicy, RuleTier
from geml.egraph.rewrite_engine import (
    ApplierResult,
    Assumption,
    RewriteContext,
    RewriteRule,
    RuleSet,
    bidirectional_rules,
    pattern_rule,
)

if TYPE_CHECKING:
    from geml.egraph.core import EGraph

ALL_MODES: frozenset[RewriteMode] = frozenset(
    {RewriteMode.SAFE_REAL, RewriteMode.POSITIVE_REAL_FORMAL}
)
FORMAL_ONLY: frozenset[RewriteMode] = frozenset({RewriteMode.POSITIVE_REAL_FORMAL})

_A = PatternVar("a")
_B = PatternVar("b")
_C = PatternVar("c")
_M = PatternVar("m", kind=VarKind.CONSTANT)
_N = PatternVar("n", kind=VarKind.CONSTANT)


def _literal(value: int | str) -> PatternNode:
    """Return a pattern matching one exact constant."""
    return PatternNode(op=Operator.CONSTANT, payload=Fraction(value))


def _node(op: Operator, *children: Pattern) -> PatternNode:
    """Return an operator pattern node."""
    return PatternNode(op=op, children=children)


@dataclass(frozen=True, slots=True)
class ExactBound:
    """A digit bound keeping constant folding finite.

    Equality saturation applied to folding rules can otherwise manufacture arbitrarily
    large exact rationals, so a fold whose result exceeds the bound is declined and
    reported rather than stored.
    """

    max_digits: int = 32

    def permits(self, value: Fraction) -> bool:
        """Return whether ``value`` is small enough to store."""
        return (
            len(str(abs(value.numerator))) <= self.max_digits
            and len(str(value.denominator)) <= self.max_digits
        )


class FoldOperation(StrEnum):
    """The arithmetic a :class:`ConstantFoldApplier` performs."""

    ADD = "add"
    MUL = "mul"
    SUB = "sub"
    NEG = "neg"
    DIV = "div"
    POW = "pow"


_MAX_FOLDED_EXPONENT = 64


@dataclass(frozen=True, slots=True)
class NonZeroGuard:
    """Guard requiring that a bound e-class is provably nonzero.

    "Provably" means one of exactly two things: the class denotes a nonzero exact constant,
    or it denotes a variable the caller explicitly declared nonzero.  Nothing is inferred
    from syntax.
    """

    variable: str

    @property
    def name(self) -> str:
        """Return the guard identifier recorded on rejection."""
        return f"nonzero({self.variable})"

    def __call__(self, egraph: EGraph, substitution: Substitution, context: RewriteContext) -> bool:
        """Return whether the bound class is known to be nonzero."""
        eclass = substitution[self.variable]
        value = constant_value(egraph, eclass)
        if value is not None:
            return value != 0
        for node in egraph.nodes_of(eclass):
            if node.op is Operator.VARIABLE and context.assumptions.holds(
                str(node.payload), Assumption.NONZERO
            ):
                return True
        return False


@dataclass(frozen=True, slots=True)
class ConstantFoldApplier:
    """Replaces an operator applied to exact constants with the exact result.

    The applier reads only the constants bound by the match, evaluates with
    :class:`fractions.Fraction`, and refuses rather than approximating whenever the
    operation is undefined, non-rational, or larger than :class:`ExactBound` allows.
    """

    operation: FoldOperation
    variables: tuple[str, ...]
    bound: ExactBound = field(default_factory=ExactBound)

    @property
    def required_variables(self) -> frozenset[str]:
        """Return the constant variables this applier reads."""
        return frozenset(self.variables)

    def __call__(
        self, egraph: EGraph, substitution: Substitution, context: RewriteContext
    ) -> ApplierResult:
        """Fold the bound constants, or decline with a reason."""
        values: list[Fraction] = []
        for name in self.variables:
            value = constant_value(egraph, substitution[name])
            if value is None:
                return ApplierResult.declined(f"{name} is not bound to an exact constant")
            values.append(value)

        folded, refusal = _evaluate(self.operation, values)
        if folded is None:
            return ApplierResult.declined(refusal)
        if not self.bound.permits(folded):
            return ApplierResult.declined(
                f"folded value exceeds the exact bound of {self.bound.max_digits} digits"
            )
        return ApplierResult.produced(egraph.add_node(ENode(op=Operator.CONSTANT, payload=folded)))


def _evaluate(operation: FoldOperation, values: list[Fraction]) -> tuple[Fraction | None, str]:
    """Return the exact fold result, or ``None`` with the reason it was refused."""
    if operation is FoldOperation.ADD:
        return values[0] + values[1], ""
    if operation is FoldOperation.MUL:
        return values[0] * values[1], ""
    if operation is FoldOperation.SUB:
        return values[0] - values[1], ""
    if operation is FoldOperation.NEG:
        return -values[0], ""
    if operation is FoldOperation.DIV:
        if values[1] == 0:
            return None, "refusing to fold a division by zero"
        return values[0] / values[1], ""
    base, exponent = values
    if exponent.denominator != 1:
        return None, "refusing to fold a non-integer exponent, which is branch sensitive"
    power = exponent.numerator
    if abs(power) > _MAX_FOLDED_EXPONENT:
        return None, f"refusing to fold an exponent beyond +/-{_MAX_FOLDED_EXPONENT}"
    if base == 0 and power <= 0:
        return None, "refusing to fold zero raised to a non-positive power"
    return Fraction(base) ** power, ""


def _always_safe(rule_id: str, name: str, justification: str) -> RulePolicy:
    """Return a policy for a universally valid identity."""
    return RulePolicy(
        rule_id=rule_id,
        name=name,
        tier=RuleTier.ALWAYS_SAFE,
        assumptions=frozenset(),
        justification=justification,
        enabled_in=ALL_MODES,
        branch_sensitive=False,
        verifier_required=False,
    )


def _guarded(
    rule_id: str, name: str, justification: str, assumptions: frozenset[str]
) -> RulePolicy:
    """Return a policy for an identity requiring a declared assumption."""
    return RulePolicy(
        rule_id=rule_id,
        name=name,
        tier=RuleTier.GUARDED,
        assumptions=assumptions,
        justification=justification,
        enabled_in=FORMAL_ONLY,
        branch_sensitive=False,
        verifier_required=True,
    )


ADD_COMMUTATIVITY = pattern_rule(
    _always_safe(
        "SAFE-ADD-COMM",
        "addition is commutative",
        "Real addition is commutative; the ordered child slots are swapped, not merged.",
    ),
    _node(Operator.ADD, _A, _B),
    _node(Operator.ADD, _B, _A),
)

MUL_COMMUTATIVITY = pattern_rule(
    _always_safe(
        "SAFE-MUL-COMM",
        "multiplication is commutative",
        "Real multiplication is commutative.",
    ),
    _node(Operator.MUL, _A, _B),
    _node(Operator.MUL, _B, _A),
)

ADD_ASSOCIATIVITY = bidirectional_rules(
    _always_safe(
        "SAFE-ADD-ASSOC",
        "addition is associative",
        "Real addition is associative; both re-groupings denote the same value.",
    ),
    _node(Operator.ADD, _node(Operator.ADD, _A, _B), _C),
    _node(Operator.ADD, _A, _node(Operator.ADD, _B, _C)),
)

MUL_ASSOCIATIVITY = bidirectional_rules(
    _always_safe(
        "SAFE-MUL-ASSOC",
        "multiplication is associative",
        "Real multiplication is associative.",
    ),
    _node(Operator.MUL, _node(Operator.MUL, _A, _B), _C),
    _node(Operator.MUL, _A, _node(Operator.MUL, _B, _C)),
)

ADD_IDENTITY = pattern_rule(
    _always_safe(
        "SAFE-ADD-ZERO",
        "zero is the additive identity",
        "x + 0 = x for every real x.",
    ),
    _node(Operator.ADD, _A, _literal(0)),
    _A,
)

MUL_IDENTITY = pattern_rule(
    _always_safe(
        "SAFE-MUL-ONE",
        "one is the multiplicative identity",
        "x * 1 = x for every real x.",
    ),
    _node(Operator.MUL, _A, _literal(1)),
    _A,
)

MUL_ZERO = pattern_rule(
    _always_safe(
        "SAFE-MUL-ZERO",
        "multiplication by zero annihilates",
        "x * 0 = 0 for every real x; no assumption on x is needed.",
    ),
    _node(Operator.MUL, _A, _literal(0)),
    _literal(0),
)

DOUBLE_NEGATION = pattern_rule(
    _always_safe(
        "SAFE-NEG-NEG",
        "double negation cancels",
        "-(-x) = x for every real x.",
    ),
    _node(Operator.NEG, _node(Operator.NEG, _A)),
    _A,
)

SUB_LOWERING = bidirectional_rules(
    _always_safe(
        "SAFE-SUB-LOWER",
        "subtraction lowers to addition of a negation",
        "a - b = a + (-b) by definition of subtraction over the reals. The backward "
        "orientation lets extraction recover a subtraction node when that is cheaper.",
    ),
    _node(Operator.SUB, _A, _B),
    _node(Operator.ADD, _A, _node(Operator.NEG, _B)),
)

ADDITIVE_INVERSE = pattern_rule(
    _always_safe(
        "SAFE-ADD-INVERSE",
        "a value plus its negation is zero",
        "x + (-x) = 0 for every real x.",
    ),
    _node(Operator.ADD, _A, _node(Operator.NEG, _A)),
    _literal(0),
)

POW_ONE = pattern_rule(
    _always_safe(
        "SAFE-POW-ONE",
        "an exponent of one is the identity",
        "x ** 1 = x for every real x, with no branch choice involved.",
    ),
    _node(Operator.POW, _A, _literal(1)),
    _A,
)

ONE_POW = pattern_rule(
    _always_safe(
        "SAFE-ONE-POW",
        "one raised to any power is one",
        "1 ** y = 1 for every real y; the principal branch is constant here.",
    ),
    _node(Operator.POW, _literal(1), _A),
    _literal(1),
)

POW_ZERO = pattern_rule(
    _guarded(
        "SAFE-POW-ZERO",
        "a nonzero base raised to zero is one",
        "x ** 0 = 1 requires x != 0 because 0 ** 0 is not defined. The assumption must be "
        "declared by the caller, so the rule is disabled in SAFE_REAL mode.",
        frozenset({"base != 0"}),
    ),
    _node(Operator.POW, _A, _literal(0)),
    _literal(1),
    guard=NonZeroGuard("a"),
)


def _fold(
    rule_id: str,
    name: str,
    justification: str,
    lhs: PatternNode,
    operation: FoldOperation,
    variables: tuple[str, ...],
) -> RewriteRule:
    """Build a bounded-exact constant folding rule."""
    return RewriteRule(
        policy=_always_safe(rule_id, name, justification),
        lhs=lhs,
        applier=ConstantFoldApplier(operation=operation, variables=variables),
    )


FOLD_ADD = _fold(
    "SAFE-FOLD-ADD",
    "fold a sum of exact constants",
    "Exact rational addition is closed and assumption free.",
    _node(Operator.ADD, _M, _N),
    FoldOperation.ADD,
    ("m", "n"),
)

FOLD_MUL = _fold(
    "SAFE-FOLD-MUL",
    "fold a product of exact constants",
    "Exact rational multiplication is closed and assumption free.",
    _node(Operator.MUL, _M, _N),
    FoldOperation.MUL,
    ("m", "n"),
)

FOLD_SUB = _fold(
    "SAFE-FOLD-SUB",
    "fold a difference of exact constants",
    "Exact rational subtraction is closed and assumption free.",
    _node(Operator.SUB, _M, _N),
    FoldOperation.SUB,
    ("m", "n"),
)

FOLD_NEG = _fold(
    "SAFE-FOLD-NEG",
    "fold the negation of an exact constant",
    "Exact rational negation is closed and assumption free.",
    _node(Operator.NEG, _M),
    FoldOperation.NEG,
    ("m",),
)

FOLD_DIV = _fold(
    "SAFE-FOLD-DIV",
    "fold a quotient of exact constants",
    "Exact rational division is closed away from a zero divisor; a zero divisor is "
    "refused and reported rather than folded.",
    _node(Operator.DIV, _M, _N),
    FoldOperation.DIV,
    ("m", "n"),
)

FOLD_POW = _fold(
    "SAFE-FOLD-POW",
    "fold an exact constant raised to an exact integer power",
    "Rational exponentiation by a bounded integer is closed and branch free. Non-integer "
    "exponents select a branch and are refused.",
    _node(Operator.POW, _M, _N),
    FoldOperation.POW,
    ("m", "n"),
)


SAFE_RULES: RuleSet = RuleSet(
    rules=(
        ADD_COMMUTATIVITY,
        MUL_COMMUTATIVITY,
        *ADD_ASSOCIATIVITY,
        *MUL_ASSOCIATIVITY,
        ADD_IDENTITY,
        MUL_IDENTITY,
        MUL_ZERO,
        DOUBLE_NEGATION,
        *SUB_LOWERING,
        ADDITIVE_INVERSE,
        POW_ONE,
        ONE_POW,
        POW_ZERO,
        FOLD_ADD,
        FOLD_MUL,
        FOLD_SUB,
        FOLD_NEG,
        FOLD_DIV,
        FOLD_POW,
    )
)

SAFE_RULE_IDS: tuple[str, ...] = tuple(dict.fromkeys(rule.rule_id for rule in SAFE_RULES.rules))
