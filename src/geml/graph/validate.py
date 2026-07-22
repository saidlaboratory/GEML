"""
validate.py - acyclicity, reachability, root existence, slot sanity, purity

owned by 3-1

TODO(Sahil/Quang): AST_VOCAB/EML_VOCAB below are placeholders pending
1-2 (src/geml/contracts/ast.py) and 2-1 (src/geml/eml/ir.py) merging
"""
from __future__ import annotations
from dataclasses import dataclass, field

from geml.graph.schema import Graph

AST_VOCAB = {"Add", "Mul", "Pow", "Log", "Exp", "Neg", "Var", "Const"}
EML_VOCAB = {"eml", "Var", "Const"}

FAMILY_VOCABS = {
    "ast": AST_VOCAB,
    "eml": EML_VOCAB,
    # macro/motif are compiler-generated, no fixed vocab to check here
}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


def _check_roots_exist(graph: Graph, errors: list[str]) -> None:
    for root_id in graph.roots:
        if root_id not in graph.nodes:
            errors.append(f"root {root_id!r} does not exist in graph.nodes")


def _check_child_slots(graph: Graph, errors: list[str]) -> None:
    for node in graph.nodes.values():
        seen_slots = set()
        for ref in node.children:
            if ref.target_id not in graph.nodes:
                errors.append(f"node {node.node_id!r} has a child ref to missing node {ref.target_id!r}")
            if ref.slot in seen_slots:
                errors.append(f"node {node.node_id!r} has two children claiming slot {ref.slot}")
            seen_slots.add(ref.slot)


def _check_reachability(graph: Graph, errors: list[str]) -> None:
    visited = set()
    stack = [r for r in graph.roots if r in graph.nodes]
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        node = graph.nodes.get(nid)
        if node is None:
            continue
        for ref in node.children:
            stack.append(ref.target_id)
    for node_id in graph.nodes:
        if node_id not in visited:
            errors.append(f"node {node_id!r} is unreachable from any root")


def _check_acyclic(graph: Graph, errors: list[str]) -> None:
    # white/gray/black dfs - hitting gray means we looped back onto
    # something still on the current path
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {nid: WHITE for nid in graph.nodes}

    def visit(nid: str) -> None:
        color[nid] = GRAY
        node = graph.nodes[nid]
        for ref in node.children:
            target_color = color.get(ref.target_id)
            if target_color == GRAY:
                errors.append(f"cycle detected: node {nid!r} -> {ref.target_id!r}")
            elif target_color == WHITE:
                visit(ref.target_id)
        color[nid] = BLACK

    for root_id in graph.roots:
        if root_id in graph.nodes and color.get(root_id) == WHITE:
            visit(root_id)


def _check_purity(graph: Graph, errors: list[str]) -> None:
    for node in graph.nodes.values():
        vocab = FAMILY_VOCABS.get(node.family)
        if vocab is not None and node.kind not in vocab:
            errors.append(
                f"node {node.node_id!r} has kind {node.kind!r}, not in "
                f"the approved {node.family} vocabulary"
            )


def validate_graph(graph: Graph) -> ValidationResult:
    errors: list[str] = []
    _check_roots_exist(graph, errors)
    _check_child_slots(graph, errors)
    _check_reachability(graph, errors)
    _check_acyclic(graph, errors)
    _check_purity(graph, errors)
    return ValidationResult(valid=(len(errors) == 0), errors=errors)
