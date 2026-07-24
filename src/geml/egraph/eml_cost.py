"""Exact Goal 3 cost evaluation and deterministic selection for Goal 4 candidates.

The production path is deliberately non-materializing.  EML DAG cost comes from the Goal 3
direct source-AST compiler, while expanded EML tree size comes from the Goal 2 count-only
compiler.  This preserves exactness without allocating the recursively expanded EML tree.

Selection is allowed only when the caller-supplied source expression is itself present in
the candidate set.  That invariant makes degradation impossible: the source is a valid
fallback under the same cost and validation rules as every rewritten form.
"""

from __future__ import annotations

from dataclasses import dataclass

from geml.dag.ast import convert_with_stats
from geml.egraph.candidates import ExtractionResult
from geml.egraph.core import EGraph
from geml.egraph.ir import EClassId, Expr
from geml.egraph.validation import (
    ValidatedCandidate,
    ValidationStatus,
    VerificationContext,
    count_expr_eml_tree,
    expr_to_ast_tree,
    validate_candidate,
)
from geml.interfaces.eml_dag_cost import EMLDagCostStatus

_RANK_SENTINEL = 1 << 62


@dataclass(frozen=True, slots=True)
class CostVector:
    """Exact official costs and stable structural tie-break fields."""

    eml_dag_cost: int | None
    eml_tree_cost: int | None
    ast_dag_cost: int | None
    ast_tree_cost: int | None
    lexical: str

    @property
    def is_costed(self) -> bool:
        return self.eml_dag_cost is not None


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A retained validation row with exact cost provenance."""

    validated: ValidatedCandidate
    cost: CostVector
    goal3_status: EMLDagCostStatus | None
    cost_reason: str

    @property
    def rankable(self) -> bool:
        return self.validated.valid and self.cost.is_costed


@dataclass(frozen=True, slots=True)
class ASTBaseline:
    """Source-AST structural comparison baseline."""

    ast_tree_size: int
    ast_dag_size: int


@dataclass(frozen=True, slots=True)
class CostReport:
    """Complete cost and selection outcome over a retained candidate set."""

    root: EClassId
    scored: tuple[ScoredCandidate, ...]
    selected: ScoredCandidate | None
    ast_baseline: ASTBaseline | None
    estimated_eml_baseline: int | None
    total_count: int
    valid_count: int
    costed_count: int
    reference_in_candidates: bool
    reference_reason: str

    @property
    def retained_failures(self) -> tuple[ScoredCandidate, ...]:
        return tuple(row for row in self.scored if not row.rankable)


def compute_cost_vector(
    validated: ValidatedCandidate,
) -> tuple[CostVector, EMLDagCostStatus | None, str]:
    """Project validation-time exact compiler evidence into the ranking vector."""
    lexical = validated.candidate.metadata.signature
    dag_result = validated.dag_cost_result
    if (
        not validated.valid
        or validated.ast_tree is None
        or validated.eml_tree_count is None
        or dag_result is None
    ):
        return (
            CostVector(None, None, None, None, lexical),
            None if dag_result is None else dag_result.status,
            f"cost not available for {validated.status.value} candidate",
        )
    if dag_result.status is not EMLDagCostStatus.SUCCESS:
        return (
            CostVector(None, None, None, None, lexical),
            dag_result.status,
            (
                "official direct EML DAG cost failed: "
                f"{dag_result.error_type}: {dag_result.error_message}"
            ),
        )

    _graph, ast_dag_stats = convert_with_stats(validated.ast_tree)
    return (
        CostVector(
            eml_dag_cost=dag_result.eml_dag_node_count,
            eml_tree_cost=validated.eml_tree_count.node_count,
            ast_dag_cost=ast_dag_stats.dag_node_count,
            ast_tree_cost=ast_dag_stats.tree_node_count,
            lexical=lexical,
        ),
        dag_result.status,
        ("exact direct_hashcons EML DAG cost and exact count-only EML tree cost computed"),
    )


def _rank_key(scored: ScoredCandidate) -> tuple[int, int, int, int, int, str]:
    cost = scored.cost
    return (
        0 if scored.rankable else 1,
        _or_sentinel(cost.eml_dag_cost),
        _or_sentinel(cost.eml_tree_cost),
        _or_sentinel(cost.ast_dag_cost),
        _or_sentinel(cost.ast_tree_cost),
        cost.lexical,
    )


def _or_sentinel(value: int | None) -> int:
    return _RANK_SENTINEL if value is None else value


def rank_candidates(scored: tuple[ScoredCandidate, ...]) -> tuple[ScoredCandidate, ...]:
    return tuple(sorted(scored, key=_rank_key))


def select_best(ranked: tuple[ScoredCandidate, ...]) -> ScoredCandidate | None:
    """Return the best retained rankable candidate, without an optimality claim."""
    return next((row for row in ranked if row.rankable), None)


def ast_cost_baseline(expr: Expr) -> ASTBaseline:
    tree = expr_to_ast_tree(expr, expression_id="baseline")
    _graph, stats = convert_with_stats(tree)
    return ASTBaseline(
        ast_tree_size=stats.tree_node_count,
        ast_dag_size=stats.dag_node_count,
    )


def estimated_eml_baseline(expr: Expr) -> int:
    """Return the exact expanded EML tree count as a non-official comparison baseline."""
    return count_expr_eml_tree(expr).node_count


def evaluate_candidates(
    extraction: ExtractionResult,
    context: VerificationContext,
    egraph: EGraph,
) -> CostReport:
    """Validate, cost, rank, and select every extracted candidate.

    An explicit reference and source e-graph are mandatory.  No candidate is promoted to
    reference as a fallback, and selection is disabled unless the source reference appears
    in the extraction result.
    """
    if not isinstance(extraction, ExtractionResult):
        raise TypeError("evaluate_candidates requires an ExtractionResult")
    if not isinstance(context, VerificationContext):
        raise TypeError("evaluate_candidates requires a VerificationContext")
    if not isinstance(egraph, EGraph):
        raise TypeError("evaluate_candidates requires the source EGraph")

    validated = tuple(
        validate_candidate(candidate, extraction.root, context, egraph)
        for candidate in extraction.candidates
    )
    scored_rows: list[ScoredCandidate] = []
    for row in validated:
        cost, status, reason = compute_cost_vector(row)
        scored_rows.append(
            ScoredCandidate(
                validated=row,
                cost=cost,
                goal3_status=status,
                cost_reason=reason,
            )
        )
    ranked = rank_candidates(tuple(scored_rows))

    reference_in_candidates = context.reference is not None and any(
        candidate.expression == context.reference for candidate in extraction.candidates
    )
    if context.reference is None:
        reference_reason = "an explicit source reference was not supplied"
    elif not reference_in_candidates:
        reference_reason = "the source reference was not retained in the candidate set"
    else:
        reference_reason = "the source reference is retained and costed under identical rules"

    selected = select_best(ranked) if reference_in_candidates else None
    baseline_expr = context.reference
    ast_baseline = ast_cost_baseline(baseline_expr) if baseline_expr is not None else None
    estimated = _safe_estimate(baseline_expr)
    return CostReport(
        root=extraction.root,
        scored=ranked,
        selected=selected,
        ast_baseline=ast_baseline,
        estimated_eml_baseline=estimated,
        total_count=len(ranked),
        valid_count=sum(1 for row in ranked if row.validated.valid),
        costed_count=sum(1 for row in ranked if row.rankable),
        reference_in_candidates=reference_in_candidates,
        reference_reason=reference_reason,
    )


def _safe_estimate(expr: Expr | None) -> int | None:
    if expr is None:
        return None
    try:
        return estimated_eml_baseline(expr)
    except Exception:
        return None


__all__ = [
    "ASTBaseline",
    "CostReport",
    "CostVector",
    "ScoredCandidate",
    "ValidationStatus",
    "ast_cost_baseline",
    "compute_cost_vector",
    "estimated_eml_baseline",
    "evaluate_candidates",
    "rank_candidates",
    "select_best",
]
