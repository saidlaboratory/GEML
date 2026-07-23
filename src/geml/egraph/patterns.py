"""Pattern language and e-matching over e-classes.

Matching enumerates every consistent substitution (a repeated variable must bind the same
class everywhere), which keeps saturation reproducible. Recursion descends the finite
pattern, never the e-graph, so matching terminates on a cyclic e-graph. No rewriting here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import TYPE_CHECKING

from geml.egraph.ir import (
    LEAF_OPERATORS,
    OPERATOR_ARITY,
    EClassId,
    EGraphError,
    ENode,
    LeafPayload,
    MalformedNodeError,
    Operator,
)

if TYPE_CHECKING:
    from geml.egraph.core import EGraph


class PatternError(EGraphError):
    """Raised when a pattern is malformed or used outside its contract."""


class UnboundPatternVariableError(PatternError):
    """Raised when instantiation needs a variable the substitution does not bind."""


class ContradictoryConstantError(EGraphError):
    """Raised when one e-class contains two different exact constants."""


class VarKind(StrEnum):
    """Admissibility of a pattern variable: ``ANY``, or ``CONSTANT`` for numerals only."""

    ANY = "any"
    CONSTANT = "constant"


@dataclass(frozen=True, slots=True)
class PatternVar:
    """A named hole in a pattern."""

    name: str
    kind: VarKind = VarKind.ANY

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise PatternError("a pattern variable requires a non-blank name")
        if not isinstance(self.kind, VarKind):
            raise PatternError(f"unknown pattern variable kind {self.kind!r}")


@dataclass(frozen=True, slots=True)
class PatternNode:
    """An operator application inside a pattern."""

    op: Operator
    children: tuple[Pattern, ...] = ()
    payload: LeafPayload | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.op, Operator):
            raise PatternError(f"unsupported operator {self.op!r} in a pattern")
        expected = OPERATOR_ARITY[self.op]
        if len(self.children) != expected:
            raise PatternError(
                f"operator {self.op.value} has arity {expected} but the pattern supplies "
                f"{len(self.children)} children"
            )
        for child in self.children:
            if not isinstance(child, PatternVar | PatternNode):
                raise PatternError("every pattern child must be a PatternVar or PatternNode")
        if self.op in LEAF_OPERATORS and self.payload is None:
            raise PatternError(
                f"a {self.op.value} pattern node must carry a literal payload; use a "
                "PatternVar to match an arbitrary leaf"
            )
        if self.op not in LEAF_OPERATORS and self.payload is not None:
            raise PatternError(f"operator {self.op.value} does not accept a payload")


type Pattern = PatternVar | PatternNode


@dataclass(frozen=True, slots=True)
class Substitution:
    """A binding from pattern variables to e-classes, stored sorted by name for canonical eq."""

    bindings: tuple[tuple[str, EClassId], ...] = ()

    @classmethod
    def of(cls, mapping: dict[str, EClassId]) -> Substitution:
        return cls(bindings=tuple(sorted(mapping.items())))

    def get(self, name: str) -> EClassId | None:
        for bound_name, eclass in self.bindings:
            if bound_name == name:
                return eclass
        return None

    def __getitem__(self, name: str) -> EClassId:
        eclass = self.get(name)
        if eclass is None:
            raise UnboundPatternVariableError(f"pattern variable {name!r} is not bound")
        return eclass

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.bindings)


@dataclass(frozen=True, slots=True)
class Match:
    """One successful match of a pattern against an e-class."""

    eclass: EClassId
    substitution: Substitution


def pattern_variables(pattern: Pattern) -> frozenset[str]:
    """Return every variable name occurring in ``pattern``."""
    names: set[str] = set()
    stack: list[Pattern] = [pattern]
    while stack:
        current = stack.pop()
        if isinstance(current, PatternVar):
            names.add(current.name)
        else:
            stack.extend(current.children)
    return frozenset(names)


def constant_value(egraph: EGraph, eclass: EClassId) -> Fraction | None:
    """Return the exact constant an e-class denotes, or ``None``; raise on contradiction."""
    found: Fraction | None = None
    for node in egraph.nodes_of(eclass):
        if node.op is not Operator.CONSTANT:
            continue
        value = node.payload
        if not isinstance(value, Fraction):
            raise MalformedNodeError("a constant node must carry a Fraction payload")
        if found is not None and found != value:
            raise ContradictoryConstantError(
                f"e-class {egraph.find(eclass)} contains both {found} and {value}"
            )
        found = value
    return found


def match_in_eclass(egraph: EGraph, pattern: Pattern, eclass: EClassId) -> tuple[Substitution, ...]:
    """Return every substitution under which ``pattern`` matches ``eclass``, deduplicated."""
    solutions = _match(egraph, pattern, eclass, {})
    ordered: dict[Substitution, None] = {}
    for solution in solutions:
        ordered[Substitution.of(solution)] = None
    return tuple(ordered)


def search_pattern(egraph: EGraph, pattern: Pattern) -> tuple[Match, ...]:
    """Return every match of ``pattern`` in ``egraph``; the pattern root must be an operator."""
    if not isinstance(pattern, PatternNode):
        raise PatternError("a searchable pattern must have an operator at its root")
    matches: list[Match] = []
    for root in egraph.roots():
        for substitution in match_in_eclass(egraph, pattern, root):
            matches.append(Match(eclass=root, substitution=substitution))
    return tuple(matches)


def instantiate(egraph: EGraph, pattern: Pattern, substitution: Substitution) -> EClassId:
    """Insert ``pattern`` under ``substitution`` and return its e-class; every var must bind."""
    if isinstance(pattern, PatternVar):
        return egraph.find(substitution[pattern.name])
    children = tuple(instantiate(egraph, child, substitution) for child in pattern.children)
    return egraph.add_node(ENode(op=pattern.op, children=children, payload=pattern.payload))


def _match(
    egraph: EGraph,
    pattern: Pattern,
    eclass: EClassId,
    bindings: dict[str, EClassId],
) -> list[dict[str, EClassId]]:
    root = egraph.find(eclass)

    if isinstance(pattern, PatternVar):
        existing = bindings.get(pattern.name)
        if existing is not None:
            return [bindings] if egraph.find(existing) == root else []
        if pattern.kind is VarKind.CONSTANT and constant_value(egraph, root) is None:
            return []
        return [{**bindings, pattern.name: root}]

    solutions: list[dict[str, EClassId]] = []
    for node in egraph.nodes_of(root):
        if node.op is not pattern.op:
            continue
        if pattern.payload is not None and node.payload != pattern.payload:
            continue
        solutions.extend(_match_children(egraph, pattern, node, bindings))
    return solutions


def _match_children(
    egraph: EGraph,
    pattern: PatternNode,
    node: ENode,
    bindings: dict[str, EClassId],
) -> list[dict[str, EClassId]]:
    partial = [bindings]
    for child_pattern, child_class in zip(pattern.children, node.children, strict=True):
        extended: list[dict[str, EClassId]] = []
        for candidate in partial:
            extended.extend(_match(egraph, child_pattern, child_class, candidate))
        if not extended:
            return []
        partial = extended
    return partial
