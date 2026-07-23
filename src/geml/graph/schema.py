"""
schema.py - generic graph/DAG data model, neutral across ast/eml/macro/motif

owned by 3-1
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChildRef:
    # slot+target pair. two children pointing at the same target still
    # show up as two separate refs - never collapsed into one
    slot: int
    target_id: str


@dataclass
class GraphNode:
    node_id: str
    family: str               # "ast" | "eml" | "macro" | "motif"
    kind: str                 # e.g. "Add", "eml", "eml_add", "motif_17"
    label: str | None = None
    value: Any = None         # leaf value only, None for internal nodes
    children: tuple[ChildRef, ...] = ()


@dataclass
class Graph:
    nodes: dict[str, GraphNode]
    roots: tuple[str, ...]  # can be more than one - many expressions can share one DAG


@dataclass
class GraphStatistics:
    node_count: int
    edge_count: int
    leaf_count: int
    root_count: int
    max_depth: int


def compute_statistics(graph: Graph) -> GraphStatistics:
    node_count = len(graph.nodes)
    edge_count = sum(len(n.children) for n in graph.nodes.values())
    leaf_count = sum(1 for n in graph.nodes.values() if not n.children)
    root_count = len(graph.roots)

    def depth_from(node_id: str, seen: frozenset[str]) -> int:
        # seen just guards against a hang if something's cyclic -
        # actual cycle detection lives in validate.py, this is a safety net
        if node_id in seen or node_id not in graph.nodes:
            return 0
        node = graph.nodes[node_id]
        if not node.children:
            return 0
        return 1 + max(
            depth_from(ref.target_id, seen | {node_id}) for ref in node.children
        )

    max_depth = max((depth_from(r, frozenset()) for r in graph.roots), default=0)

    return GraphStatistics(
        node_count=node_count,
        edge_count=edge_count,
        leaf_count=leaf_count,
        root_count=root_count,
        max_depth=max_depth,
    )
