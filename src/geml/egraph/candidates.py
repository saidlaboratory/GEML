"""Public candidate-extraction API for the GEML e-graph.

This module orchestrates :mod:`geml.egraph.cycle_safe_extract` into a typed, deterministic
result.  It enumerates *possible equivalent expressions* for an e-class; it never scores,
selects, or ranks them, and it never consults any cost model.  A candidate is only a
possible equivalent expression, never an optimum.

The public :func:`extract_candidates` function is a pure function of the e-graph state and
the limits: repeated calls return identical candidates, identical ordering, and identical
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from geml.egraph.core import EGraph
from geml.egraph.cycle_safe_extract import (
    CycleSafeExtractor,
    EnumerationTelemetry,
    ExtractionLimits,
    expr_depth,
    expr_node_count,
    expr_signature,
)
from geml.egraph.ir import EClassId, Expr
from geml.egraph.policy import ExtractionStatus


@dataclass(frozen=True, slots=True)
class CandidateMetadata:
    """Deterministic, structural generation metadata for one candidate.

    Every field is derived from the expression's shape.  None of these is a cost or a
    quality score; ``enumeration_index`` is simply the candidate's position in the stable
    canonical ordering.
    """

    enumeration_index: int
    node_count: int
    signature: str


@dataclass(frozen=True, slots=True)
class Candidate:
    """One enumerated equivalent expression and where it came from.

    ``expression`` is a concrete :class:`~geml.egraph.ir.Expr`.  ``eclass`` is the canonical
    root e-class it was extracted from.  ``depth`` is the expression's own tree depth.
    """

    expression: Expr
    eclass: EClassId
    depth: int
    metadata: CandidateMetadata


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """The complete, explicit outcome of one extraction.

    ``status`` is drawn from the frozen :class:`~geml.egraph.policy.ExtractionStatus`.  A
    resource limit or an empty enumeration is reported through ``status`` and ``reason``,
    never by silently returning fewer candidates without explanation.
    """

    root: EClassId
    candidates: tuple[Candidate, ...]
    status: ExtractionStatus
    reason: str
    nodes_visited: int
    iterations: int
    elapsed_seconds: float

    @property
    def count(self) -> int:
        """Return how many candidates were enumerated."""
        return len(self.candidates)


def extract_candidates(
    egraph: EGraph,
    root: EClassId,
    limits: ExtractionLimits | None = None,
    *,
    required_expressions: tuple[Expr, ...] = (),
) -> ExtractionResult:
    """Enumerate candidate expressions for ``root`` under ``limits``.

    The enumeration is cycle-safe and deterministic.  Candidates are ordered by their
    canonical structural signature, so repeated calls with the same e-graph and limits
    return byte-identical results.
    """
    if not isinstance(egraph, EGraph):
        raise TypeError("extract_candidates requires an EGraph")
    active_limits = limits if limits is not None else ExtractionLimits()
    if not isinstance(required_expressions, tuple) or any(
        not isinstance(expression, Expr) for expression in required_expressions
    ):
        raise TypeError("required_expressions must be a tuple of Expr values")
    root_id = egraph.find(root)

    extractor = CycleSafeExtractor(egraph, active_limits)
    expressions, telemetry = extractor.enumerate(root_id)
    expressions, telemetry = _retain_required(
        egraph,
        root_id,
        expressions,
        required_expressions,
        active_limits,
        telemetry,
    )

    candidates = tuple(
        Candidate(
            expression=expression,
            eclass=root_id,
            depth=expr_depth(expression),
            metadata=CandidateMetadata(
                enumeration_index=index,
                node_count=expr_node_count(expression),
                signature=expr_signature(expression),
            ),
        )
        for index, expression in enumerate(expressions)
    )

    status, reason = _classify(candidates, telemetry)
    return ExtractionResult(
        root=root_id,
        candidates=candidates,
        status=status,
        reason=reason,
        nodes_visited=telemetry.nodes_visited,
        iterations=telemetry.iterations,
        elapsed_seconds=telemetry.elapsed_seconds,
    )


def _retain_required(
    egraph: EGraph,
    root: EClassId,
    expressions: tuple[Expr, ...],
    required: tuple[Expr, ...],
    limits: ExtractionLimits,
    telemetry: EnumerationTelemetry,
) -> tuple[tuple[Expr, ...], EnumerationTelemetry]:
    """Retain caller-designated root members as safety anchors.

    The experiment designates the source expression as one such anchor.  Including it
    guarantees that bounded enumeration cannot force selection of a more expensive form.
    Required expressions must already be present in the requested root e-class; this helper
    never mutates the e-graph.
    """
    required_by_signature: dict[str, Expr] = {}
    for expression in required:
        found = egraph.lookup_expr(expression)
        if found is None or egraph.find(found) != root:
            raise ValueError("a required expression is not present in the extraction root")
        required_by_signature.setdefault(expr_signature(expression), expression)
    if len(required_by_signature) > limits.max_candidates:
        raise ValueError("required expressions exceed the configured candidate limit")

    merged = {expr_signature(expression): expression for expression in expressions}
    merged.update(required_by_signature)
    ordered_keys = sorted(merged)
    if len(ordered_keys) <= limits.max_candidates:
        return tuple(merged[key] for key in ordered_keys), telemetry

    optional_keys = [key for key in ordered_keys if key not in required_by_signature]
    keep = set(required_by_signature)
    keep.update(optional_keys[: limits.max_candidates - len(keep)])
    retained = tuple(merged[key] for key in ordered_keys if key in keep)
    return (
        retained,
        EnumerationTelemetry(
            nodes_visited=telemetry.nodes_visited,
            iterations=telemetry.iterations,
            elapsed_seconds=telemetry.elapsed_seconds,
            exhaustive=False,
            halted_status=telemetry.halted_status,
        ),
    )


def _classify(
    candidates: tuple[Candidate, ...],
    telemetry: EnumerationTelemetry,
) -> tuple[ExtractionStatus, str]:
    """Map enumeration telemetry to an explicit status and human-readable reason."""
    if telemetry.halted_status is not None:
        if candidates:
            return telemetry.halted_status, _halt_reason(telemetry.halted_status, partial=True)
        return telemetry.halted_status, _halt_reason(telemetry.halted_status, partial=False)
    if not candidates:
        return (
            ExtractionStatus.FAILED,
            "no finite expression was reconstructable within the depth limit",
        )
    if not telemetry.exhaustive:
        return (
            ExtractionStatus.PARTIAL_SUCCESS,
            "enumeration was truncated by the depth or beam limits",
        )
    return ExtractionStatus.SUCCESS, "enumeration completed within all limits"


def _halt_reason(status: ExtractionStatus, *, partial: bool) -> str:
    """Return a reason string for a run that stopped on a resource limit."""
    prefix = "partial candidates returned" if partial else "no candidates returned"
    limit = {
        ExtractionStatus.TIMEOUT: "the wall-clock timeout was reached",
        ExtractionStatus.NODE_LIMIT: "the node-visit limit was reached",
        ExtractionStatus.ITERATION_LIMIT: "the iteration limit was reached",
    }.get(status, "a resource limit was reached")
    return f"{prefix}; {limit}"
