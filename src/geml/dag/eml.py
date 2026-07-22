"""
eml.py - bottom-up hash-consing, pure eml trees to dags

owned by 3-3

reuses 3-1's real graph/signature stuff. EmlNode below is a
placeholder pending 2-1's real compiler output.

update: found the actual paper (arXiv:2603.21852v2, eq 3 and 5) and
confirmed two real constructions directly from it:
  exp(x) = eml(x, 1)
  ln(z)  = eml(1, eml(eml(1,z), 1))
both verified numerically, not just quoted. addition/multiplication/
powers are NOT in the parts of the paper fetched so far (Table 4
mentions multiplication needs depth 8, but doesn't give the tree) -
still placeholders below, asked Sahil whether he has the real compiler
output for those
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math

from geml.graph.schema import Graph, GraphNode, ChildRef, compute_statistics
from geml.graph.signatures import compute_signature


# TODO(Sahil/Quang): placeholder pending 2-1's real ir.py
@dataclass(frozen=True)
class EmlNode:
    kind: str  # "eml" | "Var" | "Const"
    children: tuple["EmlNode", ...] = ()
    value: Any = None  # should always be 1 for Const in real pure eml


def make_exp(arg: EmlNode) -> EmlNode:
    """exp(x) = eml(x, 1) - verified, arXiv:2603.21852v2 eq 3/example."""
    return EmlNode("eml", (arg, EmlNode("Const", value=1)))


def make_ln(arg: EmlNode) -> EmlNode:
    """ln(z) = eml(1, eml(eml(1,z), 1)) - verified, arXiv:2603.21852v2 eq 5."""
    inner = EmlNode("eml", (EmlNode("Const", value=1), arg))
    middle = EmlNode("eml", (inner, EmlNode("Const", value=1)))
    return EmlNode("eml", (EmlNode("Const", value=1), middle))


def eml_to_dag(root: EmlNode) -> Graph:
    # same hash-consing as 3-2's ast version, just restricted to
    # family="eml" and the eml/Var/Const node set
    nodes: dict[str, GraphNode] = {}
    signature_to_id: dict[str, str] = {}
    counter = [0]

    def next_id() -> str:
        counter[0] += 1
        return f"n{counter[0]}"

    def convert(eml_node: EmlNode) -> str:
        child_refs = tuple(
            ChildRef(slot=i, target_id=convert(child))
            for i, child in enumerate(eml_node.children)
        )
        tentative_id = next_id()
        candidate = GraphNode(
            node_id=tentative_id,
            family="eml",
            kind=eml_node.kind,
            value=eml_node.value,
            children=child_refs,
        )
        nodes[tentative_id] = candidate
        sig = compute_signature(Graph(nodes=nodes, roots=(tentative_id,)), tentative_id)

        if sig in signature_to_id:
            del nodes[tentative_id]
            return signature_to_id[sig]

        signature_to_id[sig] = tentative_id
        return tentative_id

    root_id = convert(root)
    return Graph(nodes=nodes, roots=(root_id,))


@dataclass
class EmlPurityResult:
    valid: bool
    errors: list[str]


def validate_eml_purity(graph: Graph) -> EmlPurityResult:
    # stricter than 3-1's generic family check - pure eml leaves aren't
    # just "any Const", they're supposed to be exactly the constant 1
    errors = []
    for node in graph.nodes.values():
        if node.kind not in ("eml", "Var", "Const"):
            errors.append(f"node {node.node_id!r} has kind {node.kind!r} - not eml/Var/Const")
        if node.kind == "Const" and node.value != 1:
            errors.append(f"node {node.node_id!r} is a Const with value {node.value!r} - pure eml leaves can only be 1")
        if node.kind == "eml" and len(node.children) != 2:
            errors.append(f"node {node.node_id!r} is kind eml but has {len(node.children)} children, needs exactly 2")
    return EmlPurityResult(valid=(len(errors) == 0), errors=errors)


@dataclass
class EmlDagStats:
    tree_node_count: int
    dag_node_count: int
    dag_edge_count: int
    dag_max_depth: int
    compression_ratio: float


def _tree_node_count(node: EmlNode) -> int:
    if not node.children:
        return 1
    return 1 + sum(_tree_node_count(c) for c in node.children)


def convert_with_stats(root: EmlNode) -> tuple[Graph, EmlDagStats]:
    graph = eml_to_dag(root)
    tree_count = _tree_node_count(root)
    dag_stats = compute_statistics(graph)
    ratio = tree_count / dag_stats.node_count if dag_stats.node_count else 0.0

    return graph, EmlDagStats(
        tree_node_count=tree_count,
        dag_node_count=dag_stats.node_count,
        dag_edge_count=dag_stats.edge_count,
        dag_max_depth=dag_stats.max_depth,
        compression_ratio=ratio,
    )


def evaluate_tree(node: EmlNode, bindings: dict[str, float]) -> float:
    # evaluates directly on the original tree, before any dedup. the
    # eml formula itself (exp(x) - ln(y)) is the one thing here i'm
    # actually sure about, it's the literal definition
    if node.kind == "Var":
        return bindings[node.value]
    if node.kind == "Const":
        return node.value
    if node.kind == "eml":
        x = evaluate_tree(node.children[0], bindings)
        y = evaluate_tree(node.children[1], bindings)
        return math.exp(x) - math.log(y)
    raise ValueError(f"can't evaluate kind {node.kind!r}")


def evaluate_dag(graph: Graph, node_id: str, bindings: dict[str, float]) -> float:
    # same evaluation but on the dag (post-sharing) - used to prove
    # sharing never changes the actual value
    node = graph.nodes[node_id]
    if node.kind == "Var":
        return bindings[node.value]
    if node.kind == "Const":
        return node.value
    if node.kind == "eml":
        ordered = sorted(node.children, key=lambda c: c.slot)
        x = evaluate_dag(graph, ordered[0].target_id, bindings)
        y = evaluate_dag(graph, ordered[1].target_id, bindings)
        return math.exp(x) - math.log(y)
    raise ValueError(f"can't evaluate kind {node.kind!r}")
