"""Cycle-safe, depth-indexed enumeration primitive for e-graph extraction.

This module owns the low-level mechanism that turns e-classes back into concrete
:class:`~geml.egraph.ir.Expr` trees.  It performs no scoring, no selection, and no cost
evaluation; those belong to Task 4-7.

Cycle safety.  Enumeration is indexed by a strictly decreasing ``remaining_depth`` budget.
A cyclic e-graph (for example one where ``x`` and ``x + 0`` were merged, or where an
inverse-rewrite pair equates ``log(exp(x))`` with ``x``) revisits an e-class with a smaller
budget on each descent until the budget reaches zero, at which point expansion stops.  The
budget therefore guarantees termination without any dependence on Python's recursion limit
beyond the configured maximum depth.

Memoization.  Enumeration of an e-class at a given remaining depth is a pure function of
``(canonical_eclass, remaining_depth)``: the only thing that stops the recursion is the
budget, never an ancestor context.  A memo keyed by that pair is therefore sound and
cannot be poisoned, because a partially computed result (one produced while a resource
limit was being hit) is never written to the memo.  Each cache entry is inserted only after
its full candidate list has been computed under a still-running budget.

Complexity.  Let ``D`` be the maximum depth, ``B`` the beam width, and ``N`` the number of
distinct ``(e-class, depth)`` pairs reachable within the budget.  Each pair is computed at
most once thanks to memoization, and each computation forms a bounded Cartesian product
capped at ``B`` combinations per e-node.  Time is ``O(N * B * arity)`` and memo space is
``O(N * B)`` expressions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction

from geml.egraph.core import EGraph
from geml.egraph.ir import EClassId, ENode, Expr, Operator
from geml.egraph.policy import ExtractionStatus

_MAX_SUPPORTED_DEPTH = 256


class ExtractionConfigurationError(ValueError):
    """Raised when extraction limits are internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    """Deterministic resource bounds for one enumeration.

    ``max_depth`` bounds the reconstructed tree height and is what makes cyclic e-graphs
    terminate.  ``beam_width`` bounds how many candidate sub-expressions survive per
    e-class.  ``max_candidates`` bounds the number of root candidates returned.
    ``max_nodes_visited`` and ``max_iterations`` are structural safety valves, and
    ``timeout_seconds`` is a wall-clock safety valve.
    """

    max_depth: int = 32
    beam_width: int = 16
    max_candidates: int = 256
    max_nodes_visited: int = 200_000
    max_iterations: int = 2_000_000
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        """Validate that every bound is positive and depth is representable."""
        for name in ("max_depth", "beam_width", "max_candidates", "max_nodes_visited"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ExtractionConfigurationError(f"{name} must be a positive integer")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ExtractionConfigurationError("max_iterations must be a positive integer")
        if self.max_depth > _MAX_SUPPORTED_DEPTH:
            raise ExtractionConfigurationError(
                f"max_depth must not exceed {_MAX_SUPPORTED_DEPTH} to keep recursion bounded"
            )
        if not isinstance(self.timeout_seconds, int | float) or self.timeout_seconds <= 0:
            raise ExtractionConfigurationError("timeout_seconds must be a positive number")


@dataclass(frozen=True, slots=True)
class EnumerationTelemetry:
    """Explicit account of one enumeration run.

    ``status`` is ``None`` while the run is in progress and is set to a halting status only
    when a resource limit stopped the run early.  ``exhaustive`` records whether any
    candidate was dropped by the depth or beam bounds; a run can complete without hitting a
    hard limit yet still be non-exhaustive because of beam truncation.
    """

    nodes_visited: int
    iterations: int
    elapsed_seconds: float
    exhaustive: bool
    halted_status: ExtractionStatus | None


def expr_signature(expr: Expr) -> str:
    """Return a canonical, deterministic string identity for an expression tree.

    The serialization is structural only.  It exists to give enumeration a stable total
    order and to deduplicate identical trees; it is explicitly not a cost or quality score.
    """
    if not expr.children:
        return f"{expr.op.value}:{_payload_token(expr.payload)}"
    inner = ",".join(expr_signature(child) for child in expr.children)
    return f"{expr.op.value}({inner})"


def expr_depth(expr: Expr) -> int:
    """Return the tree depth of ``expr`` with the convention that a leaf has depth zero."""
    if not expr.children:
        return 0
    return 1 + max(expr_depth(child) for child in expr.children)


def expr_node_count(expr: Expr) -> int:
    """Return the number of nodes in ``expr``."""
    return 1 + sum(expr_node_count(child) for child in expr.children)


def _payload_token(payload: str | Fraction | None) -> str:
    if payload is None:
        return ""
    if isinstance(payload, Fraction):
        return f"{payload.numerator}/{payload.denominator}"
    return payload


@dataclass(slots=True)
class _RunState:
    """Mutable per-call counters; never shared between calls, so no global state exists."""

    nodes_visited: int = 0
    iterations: int = 0
    exhaustive: bool = True
    halted_status: ExtractionStatus | None = None
    memo: dict[tuple[EClassId, int], tuple[Expr, ...]] = field(default_factory=dict)


class CycleSafeExtractor:
    """Enumerates concrete expressions from a (possibly cyclic) e-graph.

    A fresh instance is created per :meth:`enumerate` call, so the memo and counters carry
    no state across calls and repeated extraction is deterministic and independent.
    """

    __slots__ = ("_egraph", "_limits", "_started", "_state")

    def __init__(self, egraph: EGraph, limits: ExtractionLimits) -> None:
        """Bind the extractor to an e-graph and a validated set of limits."""
        self._egraph = egraph
        self._limits = limits
        self._state = _RunState()
        self._started = 0.0

    def enumerate(self, root: EClassId) -> tuple[tuple[Expr, ...], EnumerationTelemetry]:
        """Return canonical-ordered candidates for ``root`` and the run telemetry.

        The returned expressions are deduplicated and ordered by their canonical signature,
        which makes repeated calls produce identical output.  Candidates are capped at
        ``max_candidates``; internal e-classes are capped at ``beam_width``.
        """
        self._state = _RunState()
        self._started = time.monotonic()
        root_id = self._egraph.find(root)

        candidates = self._enumerate_root(root_id)
        telemetry = EnumerationTelemetry(
            nodes_visited=self._state.nodes_visited,
            iterations=self._state.iterations,
            elapsed_seconds=time.monotonic() - self._started,
            exhaustive=self._state.exhaustive and self._state.halted_status is None,
            halted_status=self._state.halted_status,
        )
        return candidates, telemetry

    def _enumerate_root(self, root_id: EClassId) -> tuple[Expr, ...]:
        """Enumerate the root e-class with the wider ``max_candidates`` cap."""
        return self._enumerate_eclass(root_id, self._limits.max_depth, self._limits.max_candidates)

    def _enumerate_eclass(
        self, eclass: EClassId, remaining_depth: int, cap: int
    ) -> tuple[Expr, ...]:
        """Return up to ``cap`` canonical-ordered candidates for one e-class.

        Internal calls use ``beam_width`` as the cap; only the root uses ``max_candidates``.
        A result is memoized only for the beam-width cap and only once fully computed under
        a live budget, which is what prevents memo poisoning.
        """
        canonical = self._egraph.find(eclass)
        use_memo = cap == self._limits.beam_width
        key = (canonical, remaining_depth)
        if use_memo and key in self._state.memo:
            return self._state.memo[key]

        if self._halted():
            return ()
        self._state.iterations += 1
        if self._state.iterations > self._limits.max_iterations:
            self._state.halted_status = ExtractionStatus.ITERATION_LIMIT
            return ()

        results = self._expand_nodes(canonical, remaining_depth, cap)
        ordered = self._order_and_cap(results, cap)

        if use_memo and self._state.halted_status is None:
            self._state.memo[key] = ordered
        return ordered

    def _expand_nodes(
        self,
        eclass: EClassId,
        remaining_depth: int,
        cap: int,
    ) -> list[Expr]:
        """Build a deterministic bounded prefix of candidates for one e-class.

        Stopping once ``cap`` distinct signatures are available is the operational beam
        bound.  Continuing to enumerate an exponentially large cyclic search space only to
        discard its suffix would turn the configured timeout into the de-facto algorithm.
        """
        collected: list[Expr] = []
        signatures: set[str] = set()
        for node in self._egraph.nodes_of(eclass):
            if self._halted():
                break
            self._state.nodes_visited += 1
            if self._state.nodes_visited > self._limits.max_nodes_visited:
                self._state.halted_status = ExtractionStatus.NODE_LIMIT
                break
            if self._timed_out():
                self._state.halted_status = ExtractionStatus.TIMEOUT
                break
            for expression in self._expand_node(node, remaining_depth):
                collected.append(expression)
                signatures.add(expr_signature(expression))
                if len(signatures) >= cap:
                    self._state.exhaustive = False
                    return collected
        return collected

    def _expand_node(self, node: ENode, remaining_depth: int) -> list[Expr]:
        """Build the candidates rooted at a single e-node."""
        if not node.children:
            return [Expr(op=node.op, payload=node.payload)]
        if remaining_depth <= 0:
            self._state.exhaustive = False
            return []

        child_options: list[tuple[Expr, ...]] = []
        for child in node.children:
            options = self._enumerate_eclass(child, remaining_depth - 1, self._limits.beam_width)
            if not options:
                return []
            child_options.append(options)
        return self._product(node.op, node.payload, child_options)

    def _product(
        self,
        op: Operator,
        payload: str | Fraction | None,
        child_options: list[tuple[Expr, ...]],
    ) -> list[Expr]:
        """Form a bounded Cartesian product of per-child options in deterministic order.

        The product is capped at ``beam_width`` combinations; if more combinations exist,
        the run is flagged non-exhaustive rather than silently completing.
        """
        combinations: list[tuple[Expr, ...]] = [()]
        for options in child_options:
            extended: list[tuple[Expr, ...]] = []
            for prefix in combinations:
                for option in options:
                    extended.append((*prefix, option))
                    if len(extended) >= self._limits.beam_width:
                        break
                if len(extended) >= self._limits.beam_width:
                    break
            if len(extended) < len(combinations) * len(options):
                self._state.exhaustive = False
            combinations = extended
        return [Expr(op=op, children=combo, payload=payload) for combo in combinations]

    def _order_and_cap(self, expressions: list[Expr], cap: int) -> tuple[Expr, ...]:
        """Deduplicate, order by canonical signature, and cap the candidate list."""
        unique: dict[str, Expr] = {}
        for expr in expressions:
            unique.setdefault(expr_signature(expr), expr)
        ordered = [unique[key] for key in sorted(unique)]
        if len(ordered) > cap:
            self._state.exhaustive = False
            ordered = ordered[:cap]
        return tuple(ordered)

    def _halted(self) -> bool:
        return self._state.halted_status is not None

    def _timed_out(self) -> bool:
        return time.monotonic() - self._started >= self._limits.timeout_seconds
