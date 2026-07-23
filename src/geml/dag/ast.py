"""Exact structural sharing for validated binary AST trees."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from geml.contracts.ast import ASTNode, ASTTree
from geml.graph.schema import (
    AST_FAMILY,
    ChildRef,
    Graph,
    GraphNode,
    GraphRoot,
    compute_statistics,
)
from geml.graph.signatures import signature_from_parts
from geml.graph.validate import validate_graph

_NODE_ID_PREFIX = "ast-"


@dataclass(frozen=True, slots=True)
class ASTDagStatistics:
    """Exact source-tree and structurally shared DAG statistics."""

    tree_node_count: int
    dag_node_count: int
    dag_child_reference_count: int
    dag_depth: int
    compression_ratio: Fraction

    @property
    def dag_edge_count(self) -> int:
        """Compatibility name for the explicit child-reference count."""

        return self.dag_child_reference_count


def _children_by_node(tree: ASTTree) -> dict[str, tuple[tuple[int, str], ...]]:
    children: dict[str, list[tuple[int, str]]] = {node.node_id: [] for node in tree.nodes}
    for edge in tree.edges:
        children[edge.source_id].append((edge.child_slot, edge.target_id))
    return {node_id: tuple(sorted(node_children)) for node_id, node_children in children.items()}


def _postorder_node_ids(
    root_id: str,
    children: dict[str, tuple[tuple[int, str], ...]],
) -> list[str]:
    """Return iterative child-before-parent order for a validated AST."""

    order: list[str] = []
    stack: list[tuple[str, bool]] = [(root_id, False)]
    while stack:
        node_id, leaving = stack.pop()
        if leaving:
            order.append(node_id)
            continue
        stack.append((node_id, True))
        for _, child_id in reversed(children[node_id]):
            stack.append((child_id, False))
    return order


def _intern_node(
    source: ASTNode,
    child_nodes: tuple[tuple[int, str], ...],
    *,
    nodes: dict[str, GraphNode],
) -> str:
    signature = signature_from_parts(
        family=AST_FAMILY,
        kind=source.node_kind,
        label=source.label,
        value=source.value,
        children=((slot, child_id.removeprefix(_NODE_ID_PREFIX)) for slot, child_id in child_nodes),
    )
    node_id = f"{_NODE_ID_PREFIX}{signature}"
    if node_id in nodes:
        return node_id
    nodes[node_id] = GraphNode(
        node_id=node_id,
        family=AST_FAMILY,
        kind=source.node_kind,
        label=source.label,
        value=source.value,
        children=tuple(ChildRef(slot=slot, target_id=child_id) for slot, child_id in child_nodes),
    )
    return node_id


def ast_to_dag(tree: ASTTree) -> Graph:
    """Share only exactly identical structural subtrees in ``tree``."""

    if not isinstance(tree, ASTTree):
        raise TypeError("tree must be a validated ASTTree")

    source_nodes = {node.node_id: node for node in tree.nodes}
    source_children = _children_by_node(tree)
    source_to_dag: dict[str, str] = {}
    dag_nodes: dict[str, GraphNode] = {}

    for source_id in _postorder_node_ids(tree.root_id, source_children):
        child_nodes = tuple(
            (slot, source_to_dag[child_id]) for slot, child_id in source_children[source_id]
        )
        dag_id = _intern_node(
            source_nodes[source_id],
            child_nodes,
            nodes=dag_nodes,
        )
        source_to_dag[source_id] = dag_id

    graph = Graph(
        nodes=dag_nodes,
        roots=(
            GraphRoot(
                root_id=tree.expression_id,
                target_id=source_to_dag[tree.root_id],
                representation_mode=AST_FAMILY,
            ),
        ),
    )
    validation = validate_graph(graph)
    if not validation.valid:  # pragma: no cover - protects the public boundary
        raise RuntimeError(
            "AST-to-DAG conversion produced an invalid graph: " + "; ".join(validation.errors)
        )
    return graph


def convert_with_stats(tree: ASTTree) -> tuple[Graph, ASTDagStatistics]:
    """Convert one AST and return exact compression statistics."""

    graph = ast_to_dag(tree)
    graph_statistics = compute_statistics(graph)
    tree_node_count = tree.statistics.node_count
    return graph, ASTDagStatistics(
        tree_node_count=tree_node_count,
        dag_node_count=graph_statistics.node_count,
        dag_child_reference_count=graph_statistics.child_reference_count,
        dag_depth=graph_statistics.max_depth,
        compression_ratio=Fraction(tree_node_count, graph_statistics.node_count),
    )
