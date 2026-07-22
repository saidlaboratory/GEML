"""Pattern language and e-matching over e-classes.

A *pattern* is a tree whose interior nodes are operators and whose leaves are either
literal values or pattern variables.  Matching a pattern against an e-class ("e-matching")
enumerates every substitution from pattern variables to e-classes under which some e-node
reachable from that class has the pattern's shape.

Two properties matter for Goal 4:

* **Totality of enumeration.**  A pattern variable that appears twice must bind to the same
  e-class in both positions, and *every* consistent binding is returned, not just the first
  one.  Dropping matches silently would make saturation non-reproducible.
* **Cycle safety.**  Recursion here descends the *pattern*, never the e-graph.  Because a
  pattern is a finite tree, matching terminates even when the e-graph contains cycles
  introduced by merging.

This module performs no rewriting: it reports matches and can instantiate a pattern, but
deciding whether to apply anything belongs to :mod:`geml.egraph.rewrite_engine`.
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
    """Raised when one e-class contains two different exact constants.

    This can only happen if an unsound rule equated two distinct numbers, so it is
    surfaced loudly instead of being resolved by picking one.
    """


class VarKind(StrEnum):
    """The admissibility constraint attached to a pattern variable.

    ``ANY`` binds to any e-class.  ``CONSTANT`` binds only to an e-class that contains an
    exact constant node, which is what lets folding rules match numerals without the
    pattern language needing numeric literals of its own.
    """

    ANY = "any"
    CONSTANT = "constant"


@dataclass(frozen=True, slots=True)
class PatternVar:
    """A named hole in a pattern."""

    name: str
    kind: VarKind = VarKind.ANY

    def __post_init__(self) -> None:
        """Reject blank names and unknown kinds."""
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
        """Validate operator support, arity, and leaf payload requirements."""
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
    """An immutable, canonically ordered binding from pattern variables to e-classes.

    Bindings are stored sorted by name so that two substitutions binding the same
    variables to the same classes compare and hash equal regardless of discovery order.
    """

    bindings: tuple[tuple[str, EClassId], ...] = ()

    @classmethod
    def of(cls, mapping: dict[str, EClassId]) -> Substitution:
        """Build a substitution from a plain mapping."""
        return cls(bindings=tuple(sorted(mapping.items())))

    def get(self, name: str) -> EClassId | None:
        """Return the class bound to ``name``, or ``None``."""
        for bound_name, eclass in self.bindings:
            if bound_name == name:
                return eclass
        return None

    def __getitem__(self, name: str) -> EClassId:
        """Return the class bound to ``name`` or fail explicitly."""
        eclass = self.get(name)
        if eclass is None:
            raise UnboundPatternVariableError(f"pattern variable {name!r} is not bound")
        return eclass

    @property
    def names(self) -> tuple[str, ...]:
        """Return the bound variable names in canonical order."""
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
    """Return the exact constant an e-class denotes, or ``None`` if it holds no constant.

    Raises :class:`ContradictoryConstantError` if the class contains two different
    constants, which would mean an unsound rule equated distinct numbers.
    """
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
    """Return every substitution under which ``pattern`` matches ``eclass``.

    Results are deterministic: e-nodes are visited in insertion order and duplicate
    substitutions are collapsed while preserving first-seen order.

    Complexity: worst case ``O(prod(|C_i|))`` over the e-classes visited at each pattern
    position, which is why patterns are kept small and shallow.
    """
    solutions = _match(egraph, pattern, eclass, {})
    ordered: dict[Substitution, None] = {}
    for solution in solutions:
        ordered[Substitution.of(solution)] = None
    return tuple(ordered)


def search_pattern(egraph: EGraph, pattern: Pattern) -> tuple[Match, ...]:
    """Return every match of ``pattern`` anywhere in ``egraph``.

    The pattern root must be an operator application.  A bare variable at the root would
    match every e-class, which is never a useful rewrite left-hand side and would make
    saturation quadratic for no benefit.
    """
    if not isinstance(pattern, PatternNode):
        raise PatternError("a searchable pattern must have an operator at its root")
    matches: list[Match] = []
    for root in egraph.roots():
        for substitution in match_in_eclass(egraph, pattern, root):
            matches.append(Match(eclass=root, substitution=substitution))
    return tuple(matches)


def instantiate(egraph: EGraph, pattern: Pattern, substitution: Substitution) -> EClassId:
    """Insert ``pattern`` under ``substitution`` into ``egraph`` and return its e-class.

    Every variable in ``pattern`` must be bound; an unbound variable is an explicit
    failure rather than a fresh e-class.
    """
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
    """Return every extension of ``bindings`` matching ``pattern`` against ``eclass``."""
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
    """Match every ordered child slot of ``pattern`` against ``node``, threading bindings."""
    partial = [bindings]
    for child_pattern, child_class in zip(pattern.children, node.children, strict=True):
        extended: list[dict[str, EClassId]] = []
        for candidate in partial:
            extended.extend(_match(egraph, child_pattern, child_class, candidate))
        if not extended:
            return []
        partial = extended
    return partial
