"""Immutable IR for the GEML e-graph: the operator vocabulary, ordered ``ENode``, and ``Expr``.

Child slots are ordered, so ``Sub(a, b)`` and ``Sub(b, a)`` are structurally distinct.
No rewriting, matching, extraction, or cost logic lives here.
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
    """The closed set of operators the e-graph can represent."""

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
    if not isinstance(operator, Operator):
        raise UnsupportedOperatorError(
            f"unsupported operator {operator!r}; supported operators are "
            f"{tuple(member.value for member in Operator)}"
        )
    return operator


def _validate_payload(operator: Operator, payload: LeafPayload | None) -> None:
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
    """One ordered node stored inside an e-class; children are e-class identifiers.

    Equality and hashing are structural over ``(op, children, payload)``, so a hash-cons
    table keyed by ``ENode`` gives perfect sharing of identical canonical nodes.
    """

    op: Operator
    children: tuple[EClassId, ...] = ()
    payload: LeafPayload | None = None

    def __post_init__(self) -> None:
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
        return self.op in LEAF_OPERATORS

    def canonicalize(self, resolve: Callable[[EClassId], EClassId]) -> ENode:
        """Return this node with every child replaced by its canonical e-class root."""
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
    if payload is None:
        return ""
    if isinstance(payload, Fraction):
        return f"{payload.numerator}/{payload.denominator}"
    return payload


@dataclass(frozen=True, slots=True)
class Expr:
    """An ordered expression tree, the only input shape :meth:`EGraph.add` accepts."""

    op: Operator
    children: tuple[Expr, ...] = ()
    payload: LeafPayload | None = None

    def __post_init__(self) -> None:
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
    """Return an exact rational constant leaf; floating point input is rejected."""
    if isinstance(value, float):
        raise MalformedNodeError("constants must be exact; pass an int, str, or Fraction")
    return Expr(op=Operator.CONSTANT, payload=Fraction(value))


def add(left: Expr, right: Expr) -> Expr:
    return Expr(op=Operator.ADD, children=(left, right))


def mul(left: Expr, right: Expr) -> Expr:
    return Expr(op=Operator.MUL, children=(left, right))


def neg(operand: Expr) -> Expr:
    return Expr(op=Operator.NEG, children=(operand,))


def sub(left: Expr, right: Expr) -> Expr:
    return Expr(op=Operator.SUB, children=(left, right))


def div(left: Expr, right: Expr) -> Expr:
    return Expr(op=Operator.DIV, children=(left, right))


def power(base: Expr, exponent: Expr) -> Expr:
    return Expr(op=Operator.POW, children=(base, exponent))


def exp(operand: Expr) -> Expr:
    return Expr(op=Operator.EXP, children=(operand,))


def log(operand: Expr) -> Expr:
    return Expr(op=Operator.LOG, children=(operand,))
