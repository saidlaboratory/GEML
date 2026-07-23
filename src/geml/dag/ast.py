"""
ast.py - bottom-up hash-consing, ast trees to dags

owned by 3-2

reuses 3-1's real Graph/GraphNode/ChildRef/compute_signature, that
part's not a placeholder anymore since we built it ourselves already.
AstNode below still is though, pending 1-2
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from geml.graph.schema import Graph, GraphNode, ChildRef, compute_statistics
from geml.graph.signatures import compute_signature


# TODO(Sahil/Quang): placeholder pending 1-2's real ast contract, swap
# once it merges
@dataclass(frozen=True)
class AstNode:
    op: str
    children: tuple["AstNode", ...] = ()
    value: Any = None


def _ast_node_count(node: AstNode) -> int:
    # counts the original tree size, duplicates and all - this is what
    # the compression ratio gets measured against
    if not node.children:
        return 1
    return 1 + sum(_ast_node_count(c) for c in node.children)


def ast_to_dag(root: AstNode) -> Graph:
    # children get converted first, then each parent checks (via 3-1's
    # signature) if an identical node already exists before adding a
    # new one. only exact structural matches get merged - nothing
    # commutative or semantic, that just falls out of the signature
    # already caring about slot order and operator kind
    nodes: dict[str, GraphNode] = {}
    signature_to_id: dict[str, str] = {}
    counter = [0]

    def next_id() -> str:
        counter[0] += 1
        return f"n{counter[0]}"

    def convert(ast_node: AstNode) -> str:
        child_refs = tuple(
            ChildRef(slot=i, target_id=convert(child))
            for i, child in enumerate(ast_node.children)
        )

        tentative_id = next_id()
        candidate = GraphNode(
            node_id=tentative_id,
            family="ast",
            kind=ast_node.op,
            value=ast_node.value,
            children=child_refs,
        )
        nodes[tentative_id] = candidate  # temp registration so compute_signature can see the children
        sig = compute_signature(Graph(nodes=nodes, roots=(tentative_id,)), tentative_id)

        if sig in signature_to_id:
            del nodes[tentative_id]  # already exists, throw this one away
            return signature_to_id[sig]

        signature_to_id[sig] = tentative_id
        return tentative_id

    root_id = convert(root)
    return Graph(nodes=nodes, roots=(root_id,))


@dataclass
class DagConversionStats:
    ast_node_count: int
    dag_node_count: int
    dag_edge_count: int
    dag_max_depth: int
    compression_ratio: float


def convert_with_stats(root: AstNode) -> tuple[Graph, DagConversionStats]:
    graph = ast_to_dag(root)
    ast_count = _ast_node_count(root)
    dag_stats = compute_statistics(graph)
    ratio = ast_count / dag_stats.node_count if dag_stats.node_count else 0.0

    return graph, DagConversionStats(
        ast_node_count=ast_count,
        dag_node_count=dag_stats.node_count,
        dag_edge_count=dag_stats.edge_count,
        dag_max_depth=dag_stats.max_depth,
        compression_ratio=ratio,
    )
