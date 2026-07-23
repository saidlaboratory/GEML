"""Official EML cost evaluation, ranking, and selection for extracted candidates.

Every cost number here comes from a frozen interface.  The exact EML DAG cost is taken
only from the Goal 3 boundary :func:`geml.interfaces.eml_dag_cost.compute_eml_dag_cost`; the
exact EML tree cost from :func:`geml.eml.counting.count_materialized_eml`; and the AST DAG
and tree sizes from :func:`geml.dag.ast.convert_with_stats` and the frozen AST statistics.
Nothing in this module estimates a cost or counts nodes by hand.

Candidates are ranked by the deterministic tie-break order exact EML DAG cost, exact EML
tree cost, AST DAG size, AST tree size, and finally the stable lexical signature.  Invalid
candidates and candidates whose official cost could not be computed are retained and ranked
after every fully costed valid candidate; they are never discarded.

The selected candidate is only the best among the enumerated, validated candidates under
the configured limits.  It is never asserted to be optimal, minimal, or globally best.  The
AST and estimated-EML baselines are comparison references only, never an official or
optimal cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from geml.dag.ast import convert_with_stats
from geml.egraph.candidates import Candidate, ExtractionResult
from geml.egraph.ir import EClassId, Expr
from geml.egraph.validation import (
    ValidatedCandidate,
    ValidationStatus,
    VerificationContext,
    compile_expr_to_eml,
    expr_to_ast_tree,
    validate_candidate,
)
from geml.eml.counting import count_materialized_eml
from geml.interfaces.eml_dag_cost import (
    EMLDagCostStatus,
    compute_eml_dag_cost,
)

_RANK_SENTINEL = 1 << 62


@dataclass(frozen=True, slots=True)
class CostVector:
    """The frozen cost quantities used for tie-breaking.

    ``eml_dag_cost`` and ``eml_tree_cost`` are exact official EML costs; ``ast_dag_cost``
    and ``ast_tree_cost`` are source-AST baselines; ``lexical`` is the stable structural
    signature used as the final deterministic tie-breaker.
    """

    eml_dag_cost: int | None
    eml_tree_cost: int | None
    ast_dag_cost: int | None
    ast_tree_cost: int | None
    lexical: str

    @property
    def is_costed(self) -> bool:
        """Return whether the official exact EML DAG cost is present."""
        return self.eml_dag_cost is not None


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A validated candidate with its cost vector and cost provenance."""

    validated: ValidatedCandidate
    cost: CostVector
    goal3_status: EMLDagCostStatus | None
    cost_reason: str

    @property
    def rankable(self) -> bool:
        """Return whether this candidate is valid and fully costed."""
        return self.validated.valid and self.cost.is_costed


@dataclass(frozen=True, slots=True)
class ASTBaseline:
    """Source-AST size baseline for one reference expression."""

    ast_tree_size: int
    ast_dag_size: int


@dataclass(frozen=True, slots=True)
class CostReport:
    """The complete, explicit outcome of cost evaluation over a candidate set."""

    root: EClassId
    scored: tuple[ScoredCandidate, ...]
    selected: ScoredCandidate | None
    ast_baseline: ASTBaseline | None
    estimated_eml_baseline: int | None
    total_count: int
    valid_count: int
    costed_count: int

    @property
    def retained_failures(self) -> tuple[ScoredCandidate, ...]:
        """Return every retained candidate that is not a fully costed valid row."""
        return tuple(row for row in self.scored if not row.rankable)


def compute_cost_vector(
    validated: ValidatedCandidate,
) -> tuple[CostVector, EMLDagCostStatus | None, str]:
    """Compute the frozen cost vector for one validated candidate.

    Only a valid candidate is costed.  An invalid row keeps its lexical signature but no
    numeric costs, so it is retained and ranked last rather than dropped.
    """
    lexical = validated.candidate.metadata.signature
    if not validated.valid or validated.eml_term is None or validated.ast_tree is None:
        return (
            CostVector(None, None, None, None, lexical),
            None,
            f"cost not computed for {validated.status.value} candidate",
        )

    dag_result = compute_eml_dag_cost(validated.eml_term)
    if dag_result.status is not EMLDagCostStatus.SUCCESS:
        return (
            CostVector(None, None, None, None, lexical),
            dag_result.status,
            f"official EML DAG cost failed: {dag_result.error_type}: {dag_result.error_message}",
        )

    eml_tree_cost = count_materialized_eml(validated.eml_term).node_count
    _graph, ast_dag_stats = convert_with_stats(validated.ast_tree)
    cost = CostVector(
        eml_dag_cost=dag_result.eml_dag_node_count,
        eml_tree_cost=eml_tree_cost,
        ast_dag_cost=ast_dag_stats.dag_node_count,
        ast_tree_cost=ast_dag_stats.tree_node_count,
        lexical=lexical,
    )
    return cost, dag_result.status, "official exact costs computed"


def _rank_key(scored: ScoredCandidate) -> tuple[int, int, int, int, int, str]:
    """Return the deterministic total-order key implementing the tie-break order."""
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
    """Return ``value`` or a large sentinel that sorts after every real cost."""
    return _RANK_SENTINEL if value is None else value


def rank_candidates(scored: tuple[ScoredCandidate, ...]) -> tuple[ScoredCandidate, ...]:
    """Return the candidates in deterministic tie-break order."""
    return tuple(sorted(scored, key=_rank_key))


def select_best(ranked: tuple[ScoredCandidate, ...]) -> ScoredCandidate | None:
    """Return the best candidate among the enumerated, validated candidates.

    This is only the best of what was enumerated and validated under the configured limits;
    it is not asserted to be an optimum, a minimum, or globally best.
    """
    for row in ranked:
        if row.rankable:
            return row
    return None


def ast_cost_baseline(expr: Expr) -> ASTBaseline:
    """Return the source-AST size baseline for one expression.

    This is a comparison baseline only.  It is never the official or optimal cost.
    """
    tree = expr_to_ast_tree(expr, expression_id="baseline")
    _graph, stats = convert_with_stats(tree)
    return ASTBaseline(ast_tree_size=stats.tree_node_count, ast_dag_size=stats.dag_node_count)


def estimated_eml_baseline(expr: Expr) -> int:
    """Return an estimated EML baseline (the exact EML tree node count) for one expression.

    The EML tree size is an upper-bound proxy for the shared DAG cost and is provided only
    for comparison.  It is never the official cost, which comes solely from the Goal 3 DAG
    interface.
    """
    return count_materialized_eml(compile_expr_to_eml(expr)).node_count


def evaluate_candidates(
    extraction: ExtractionResult,
    context: VerificationContext | None = None,
) -> CostReport:
    """Validate, cost, rank, and select over an extraction result.

    Every input candidate produces exactly one retained row.  Failures at any stage are
    kept with an explicit status and reason, and the ranking places fully costed valid
    candidates ahead of every retained failure.
    """
    if not isinstance(extraction, ExtractionResult):
        raise TypeError("evaluate_candidates requires an ExtractionResult")
    active_context = context if context is not None else VerificationContext()

    reference_eml = _resolve_reference(extraction.candidates, active_context)
    validated = tuple(
        validate_candidate(candidate, extraction.root, active_context, reference_eml)
        for candidate in extraction.candidates
    )
    scored_rows: list[ScoredCandidate] = []
    for row in validated:
        cost, status, reason = compute_cost_vector(row)
        scored_rows.append(
            ScoredCandidate(validated=row, cost=cost, goal3_status=status, cost_reason=reason)
        )

    ranked = rank_candidates(tuple(scored_rows))
    selected = select_best(ranked)
    baseline_expr = _baseline_expression(active_context, extraction.candidates)
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
    )


def _resolve_reference(
    candidates: tuple[Candidate, ...],
    context: VerificationContext,
):
    """Return the compiled reference term used for semantic verification, if any."""
    if context.reference is not None:
        try:
            return compile_expr_to_eml(context.reference, compiler_mode=context.compiler_mode)
        except Exception:
            pass
    for candidate in candidates:
        try:
            return compile_expr_to_eml(candidate.expression, compiler_mode=context.compiler_mode)
        except Exception:
            continue
    return None


def _baseline_expression(
    context: VerificationContext,
    candidates: tuple[Candidate, ...],
) -> Expr | None:
    """Return the expression the baselines describe: the reference or first candidate."""
    if context.reference is not None:
        return context.reference
    if candidates:
        return candidates[0].expression
    return None


def _safe_estimate(expr: Expr | None) -> int | None:
    """Return the estimated EML baseline, or ``None`` if it cannot be compiled."""
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
