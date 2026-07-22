"""
hashcons.py - generic interning table for building dags directly

owned by 3-4

difference from 3-3's approach: 3-3 builds the full tree first, then
runs hash-consing over the finished thing. this table gets checked at
every single node creation instead, so the uncompressed tree never
actually gets allocated - nodes only exist once, from the moment
they're first needed.

reuses 3-1's real compute_signature for every lookup on purpose, even
though it costs some recompute time - one source of truth for what
counts as identical structure beats two implementations that could
quietly drift apart
"""
from __future__ import annotations
from typing import Any

from geml.graph.schema import Graph, GraphNode, ChildRef
from geml.graph.signatures import compute_signature


class HashConsTable:
    def __init__(self, family: str):
        self.family = family
        self.nodes: dict[str, GraphNode] = {}
        self.signature_to_id: dict[str, str] = {}
        self._counter = 0
        self.peak_size = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"n{self._counter}"

    def intern_leaf(self, kind: str, value: Any) -> str:
        tentative_id = self._next_id()
        candidate = GraphNode(node_id=tentative_id, family=self.family, kind=kind, value=value)
        self.nodes[tentative_id] = candidate  # temp, so compute_signature can see it
        sig = compute_signature(Graph(nodes=self.nodes, roots=(tentative_id,)), tentative_id)

        if sig in self.signature_to_id:
            del self.nodes[tentative_id]
            return self.signature_to_id[sig]

        self.signature_to_id[sig] = tentative_id
        self.peak_size = max(self.peak_size, len(self.nodes))
        return tentative_id

    def intern_binary(self, kind: str, left_id: str, right_id: str) -> str:
        tentative_id = self._next_id()
        candidate = GraphNode(
            node_id=tentative_id,
            family=self.family,
            kind=kind,
            children=(ChildRef(0, left_id), ChildRef(1, right_id)),
        )
        self.nodes[tentative_id] = candidate
        sig = compute_signature(Graph(nodes=self.nodes, roots=(tentative_id,)), tentative_id)

        if sig in self.signature_to_id:
            del self.nodes[tentative_id]
            return self.signature_to_id[sig]

        self.signature_to_id[sig] = tentative_id
        self.peak_size = max(self.peak_size, len(self.nodes))
        return tentative_id

    def to_graph(self, root_id: str) -> Graph:
        return Graph(nodes=self.nodes, roots=(root_id,))
