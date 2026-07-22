"""Immutable intermediate representation consumed by the GEML e-graph.

This module defines the closed operator vocabulary of the Goal 4 e-graph, the ordered
``ENode`` representation stored inside e-classes, and the ordered ``Expr`` tree used to
insert expressions.  It contains no rewriting, matching, extraction, or cost logic.

Two structural invariants are enforced here rather than by convention:

* every operator has a fixed arity and a node carrying the wrong number of children is
  rejected at construction time;
* child slots are ordered, so ``Sub(a, b)`` and ``Sub(b, a)`` are distinct nodes even
  though a commutativity rule may later place them in the same e-class.

Structural identity is deliberately kept separate from semantic equivalence: two nodes are
equal here only when their operator, ordered children, and leaf payload agree.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from types import MappingProxyType
from typing import NewType

EClassId = NewType("EClassId", int)

type LeafPayload = str | Fraction


class Operator(StrEnum):
    """The closed set of operators the Goal 4 e-graph can represent.

    The vocabulary is intentionally small.  Anything outside it must be reported as an
    unsupported operator rather than approximated by a nearby node kind.
    """

    VARIABLE = "variable"
    CONSTANT = "constant"
    ADD = "add"
    MUL = "mul"
    NEG = "neg"
    SUB = "sub"
    DIV = "div"
    POW = "pow"
    EXP = "exp"
    LOG = "log"


OPERATOR_ARITY: Mapping[Operator, int] = MappingProxyType(
    {
        Operator.VARIABLE: 0,
        Operator.CONSTANT: 0,
        Operator.ADD: 2,
        Operator.MUL: 2,
        Operator.NEG: 1,
        Operator.SUB: 2,
        Operator.DIV: 2,
        Operator.POW: 2,
        Operator.EXP: 1,
        Operator.LOG: 1,
    }
)

LEAF_OPERATORS: frozenset[Operator] = frozenset({Operator.VARIABLE, Operator.CONSTANT})

OPERATOR_ORDER: Mapping[Operator, int] = MappingProxyType(
    {operator: index for index, operator in enumerate(Operator)}
)


class EGraphError(Exception):
    """Base class for every explicit Goal 4 e-graph failure."""


class UnsupportedOperatorError(EGraphError):
    """Raised when a value outside :class:`Operator` is used as an operator."""


class MalformedNodeError(EGraphError):
    """Raised when arity, payload, or child references violate the IR contract."""


def _validate_operator(operator: object) -> Operator:
    """Return ``operator`` as an :class:`Operator` or fail explicitly."""
    if not isinstance(operator, Operator):
        raise UnsupportedOperatorError(
            f"unsupported operator {operator!r}; supported operators are "
            f"{tuple(member.value for member in Operator)}"
        )
    return operator


def _validate_payload(operator: Operator, payload: LeafPayload | None) -> None:
    """Check that ``payload`` matches what ``operator`` requires."""
    if operator is Operator.VARIABLE:
        if not isinstance(payload, str) or not payload.strip():
            raise MalformedNodeError("a variable node requires a non-blank string payload")
        return
    if operator is Operator.CONSTANT:
        if not isinstance(payload, Fraction):
            raise MalformedNodeError(
                "a constant node requires an exact Fraction payload; floating point "
                "constants are not representable"
            )
        return
    if payload is not None:
        raise MalformedNodeError(f"operator {operator.value} does not accept a leaf payload")


@dataclass(frozen=True, slots=True)
class ENode:
    """One ordered node stored inside an e-class.

    Children are e-class identifiers, not nodes, which is what allows an e-graph to
    represent exponentially many equivalent trees in linear space.  Equality and hashing
    are structural over ``(op, children, payload)``, so a hash-cons table keyed by
    ``ENode`` gives perfect sharing of identical canonical nodes.
    """

    op: Operator
    children: tuple[EClassId, ...] = ()
    payload: LeafPayload | None = None

    def __post_init__(self) -> None:
        """Validate operator support, arity, child types, and payload shape."""
        operator = _validate_operator(self.op)
        if not isinstance(self.children, tuple):
            raise MalformedNodeError("children must be a tuple to keep child slots ordered")
        expected = OPERATOR_ARITY[operator]
        if len(self.children) != expected:
            raise MalformedNodeError(
                f"operator {operator.value} has arity {expected} but received "
                f"{len(self.children)} children"
            )
        for child in self.children:
            if not isinstance(child, int) or isinstance(child, bool):
                raise MalformedNodeError("every child must be an integer e-class identifier")
        _validate_payload(operator, self.payload)

    @property
    def is_leaf(self) -> bool:
        """Return whether this node has arity zero."""
        return self.op in LEAF_OPERATORS

    def canonicalize(self, resolve: Callable[[EClassId], EClassId]) -> ENode:
        """Return this node with every child replaced by its canonical e-class root.

        ``resolve`` is normally :meth:`geml.egraph.core.EGraph.find`.  Canonicalization is
        what makes congruence detectable: two nodes are congruent exactly when their
        canonical forms are equal.
        """
        if not self.children:
            return self
        canonical = tuple(resolve(child) for child in self.children)
        if canonical == self.children:
            return self
        return ENode(op=self.op, children=canonical, payload=self.payload)

    def sort_key(self) -> tuple[int, str, tuple[int, ...]]:
        """Return a total, deterministic ordering key over nodes."""
        return (OPERATOR_ORDER[self.op], _payload_key(self.payload), tuple(self.children))


def _payload_key(payload: LeafPayload | None) -> str:
    """Return a stable string key for a leaf payload."""
    if payload is None:
        return ""
    if isinstance(payload, Fraction):
        return f"{payload.numerator}/{payload.denominator}"
    return payload


@dataclass(frozen=True, slots=True)
class Expr:
    """An ordered expression tree used to insert expressions into an e-graph.

    ``Expr`` mirrors :class:`ENode` but holds child *trees* instead of e-class
    identifiers.  It is the only input shape :meth:`geml.egraph.core.EGraph.add` accepts,
    which keeps insertion free of ad-hoc parsing.
    """

    op: Operator
    children: tuple[Expr, ...] = ()
    payload: LeafPayload | None = None

    def __post_init__(self) -> None:
        """Validate operator support, arity, child types, and payload shape."""
        operator = _validate_operator(self.op)
        if not isinstance(self.children, tuple):
            raise MalformedNodeError("children must be a tuple to keep child slots ordered")
        expected = OPERATOR_ARITY[operator]
        if len(self.children) != expected:
            raise MalformedNodeError(
                f"operator {operator.value} has arity {expected} but received "
                f"{len(self.children)} children"
            )
        for child in self.children:
            if not isinstance(child, Expr):
                raise MalformedNodeError("every child of an Expr must itself be an Expr")
        _validate_payload(operator, self.payload)


def var(name: str) -> Expr:
    """Return a variable leaf named ``name``."""
    return Expr(op=Operator.VARIABLE, payload=name)


def const(value: Fraction | int | str) -> Expr:
    """Return an exact rational constant leaf.

    Floating point input is rejected: the e-graph folds constants exactly and a binary
    float would silently introduce a rounding assumption.
    """
    if isinstance(value, float):
        raise MalformedNodeError("constants must be exact; pass an int, str, or Fraction")
    return Expr(op=Operator.CONSTANT, payload=Fraction(value))


def add(left: Expr, right: Expr) -> Expr:
    """Return ``left + right``."""
    return Expr(op=Operator.ADD, children=(left, right))


def mul(left: Expr, right: Expr) -> Expr:
    """Return ``left * right``."""
    return Expr(op=Operator.MUL, children=(left, right))


def neg(operand: Expr) -> Expr:
    """Return ``-operand``."""
    return Expr(op=Operator.NEG, children=(operand,))


def sub(left: Expr, right: Expr) -> Expr:
    """Return ``left - right``."""
    return Expr(op=Operator.SUB, children=(left, right))


def div(left: Expr, right: Expr) -> Expr:
    """Return ``left / right``."""
    return Expr(op=Operator.DIV, children=(left, right))


def power(base: Expr, exponent: Expr) -> Expr:
    """Return ``base ** exponent``."""
    return Expr(op=Operator.POW, children=(base, exponent))


def exp(operand: Expr) -> Expr:
    """Return ``exp(operand)``."""
    return Expr(op=Operator.EXP, children=(operand,))


def log(operand: Expr) -> Expr:
    """Return ``log(operand)``."""
    return Expr(op=Operator.LOG, children=(operand,))
