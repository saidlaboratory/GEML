"""
eml_dag_cost.py - stable, frozen cost API for goal 4's e-graph extraction

owned by 3-8

deliberately minimal: goal 4 needs one thing - "how big would this
expression's EML-DAG be, and did it even work" - to pick the cheapest
candidate during extraction. nothing else.

FROZEN INTERFACE - this module intentionally imports only from the
already-frozen geml.graph.* / geml.dag.* modules, never from
geml.experiments.* or geml.analysis.* (goal 3's own internal research
code). goal 4 should depend on this module and never those.

assumptions and failure modes:
- only exp/ln (and compositions of them) are actually supported right
  now, same honest limit as the rest of goal 3 - anything requiring
  add/mul/pow constructors fails explicitly (status="failure"), it
  does not silently return a wrong or partial cost
- a "successful" result means the expression compiled, validated as
  pure eml, and its dag was built - it says nothing about whether the
  expression is mathematically meaningful beyond that
- tie-breaking: when two candidates have the same node_count,
  canonical_signature is exact and deterministic - goal 4 can use it
  (or depth, as a secondary axis) to break ties consistently rather
  than picking arbitrarily
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from geml.graph.schema import compute_statistics
from geml.graph.signatures import compute_signature
from geml.dag.eml import EmlNode, eml_to_dag, validate_eml_purity
from geml.dag.hashcons import HashConsTable
from geml.dag.direct_eml import compile_with_stats


@dataclass(frozen=True)
class EmlDagCostResult:
    status: str  # "success" | "failure" - never anything else
    node_count: int | None
    edge_count: int | None
    depth: int | None
    canonical_signature: str | None  # exact structural signature, for deterministic tie-breaking
    error_type: str | None = None
    error_message: str | None = None


def _failure(error_type: str, error_message: str) -> EmlDagCostResult:
    return EmlDagCostResult(
        status="failure", node_count=None, edge_count=None, depth=None,
        canonical_signature=None, error_type=error_type, error_message=error_message,
    )


def _cost_from_graph(graph, root_id: str) -> EmlDagCostResult:
    purity = validate_eml_purity(graph)
    if not purity.valid:
        return _failure("PurityError", "; ".join(purity.errors))

    stats = compute_statistics(graph)
    signature = compute_signature(graph, root_id)
    return EmlDagCostResult(
        status="success",
        node_count=stats.node_count,
        edge_count=stats.edge_count,
        depth=stats.max_depth,
        canonical_signature=signature,
    )


def compute_eml_dag_cost(build_eml_direct: Callable[[HashConsTable], str]) -> EmlDagCostResult:
    """
    the primary entry point: accepts a direct-construction recipe
    (build_fn(table) -> root_id, same shape used throughout goal 3) and
    returns its exact dag cost. this is the one function goal 4 should
    actually call - it never needs to touch Graph/GraphNode/
    HashConsTable directly, just this function and its result type
    """
    try:
        graph, root_id, _ = compile_with_stats(build_eml_direct)
    except Exception as error:
        return _failure(type(error).__name__, str(error))
    return _cost_from_graph(graph, root_id)


def compute_eml_dag_cost_from_tree(tree: EmlNode) -> EmlDagCostResult:
    """
    same cost computation, starting from an already-materialized
    EmlNode tree instead of a direct-construction recipe - useful when
    a candidate already exists in tree form (e.g. from an e-graph
    rewrite) rather than being built fresh
    """
    try:
        graph = eml_to_dag(tree)
        root_id = graph.roots[0]
    except Exception as error:
        return _failure(type(error).__name__, str(error))
    return _cost_from_graph(graph, root_id)
