"""
failures.py - top successes/failures, and the explicit distinction
between "compresses well" and "becomes structurally competitive"

owned by 3-7

these are NOT the same claim. an expression can shrink a lot relative
to its own raw tree (high eml_compression) and still end up bigger
than a plain AST would've been (high dag_alpha_vs_ast_tree). this
module keeps the two axes separate on purpose, never merges them into
one "good/bad" score.

no e-graph or motif claims anywhere in here - this is DAG sharing
only, not goal 4's semantic rewriting or goal 5's motif compression
"""
from __future__ import annotations
from dataclasses import dataclass

from geml.analysis.goal3.metrics import AnalysisRow


@dataclass
class RankedResult:
    expression_id: str
    metric_value: float
    metrics: dict


def top_by_compression_ratio(rows: list[AnalysisRow], n: int = 5) -> list[RankedResult]:
    """ranks by eml_compression - how much smaller than its OWN raw tree
    the dag got. this is 'compresses well', nothing more."""
    valid = [r for r in rows if r.status == "success"]
    ranked = sorted(valid, key=lambda r: r.metrics["eml_compression"], reverse=True)
    return [RankedResult(r.expression_id, r.metrics["eml_compression"], r.metrics) for r in ranked[:n]]


def top_by_final_alpha(rows: list[AnalysisRow], n: int = 5) -> list[RankedResult]:
    """ranks by dag_alpha_vs_ast_tree, smallest first - the ABSOLUTE size
    relative to a plain AST. this is 'structurally competitive':
    approaching or beating 1.0 here, regardless of how much internal
    compression happened to get there."""
    valid = [r for r in rows if r.status == "success"]
    ranked = sorted(valid, key=lambda r: r.metrics["dag_alpha_vs_ast_tree"])
    return [RankedResult(r.expression_id, r.metrics["dag_alpha_vs_ast_tree"], r.metrics) for r in ranked[:n]]


def top_failures(rows: list[AnalysisRow], n: int = 5) -> list[RankedResult]:
    failures = [r for r in rows if r.status == "failure"]
    return [
        RankedResult(r.expression_id, 0.0, {})
        for r in failures[:n]
    ]


@dataclass
class DualClaim:
    expression_id: str
    compression_ratio: float        # eml_compression
    final_alpha: float               # dag_alpha_vs_ast_tree
    compresses_well: bool
    structurally_competitive: bool


def classify_dual(
    rows: list[AnalysisRow],
    compression_threshold: float = 3.0,
    competitive_threshold: float = 1.5,
) -> list[DualClaim]:
    """
    the concrete mechanism that keeps the two claims separate: each row
    gets both booleans independently, so a case with
    compresses_well=True and structurally_competitive=False is not
    just possible, it's exactly the situation this whole module exists
    to surface rather than hide behind one blended score
    """
    valid = [r for r in rows if r.status == "success"]
    out = []
    for r in valid:
        compression = r.metrics["eml_compression"]
        alpha = r.metrics["dag_alpha_vs_ast_tree"]
        out.append(DualClaim(
            expression_id=r.expression_id,
            compression_ratio=compression,
            final_alpha=alpha,
            compresses_well=compression >= compression_threshold,
            structurally_competitive=alpha <= competitive_threshold,
        ))
    return out
