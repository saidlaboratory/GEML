"""
statistics.py - counts stuff on the tree + a signature for comparing them

owned by 1-6. leaf depth = 0 (not 1), matches the spec doc
"""
from __future__ import annotations
from geml.ast.builder import ASTNode


def node_count(node: ASTNode) -> int:
    if not node.children:
        return 1
    return 1 + sum(node_count(c) for c in node.children)


def depth(node: ASTNode) -> int:
    # leaf = 0, root of a single Add(x,y) = 1, etc
    if not node.children:
        return 0
    return 1 + max(depth(c) for c in node.children)


def edge_count(node: ASTNode) -> int:
    return node_count(node) - 1


def leaf_count(node: ASTNode) -> int:
    if not node.children:
        return 1
    return sum(leaf_count(c) for c in node.children)


def operator_count(node: ASTNode) -> int:
    # everything that isn't a leaf
    return node_count(node) - leaf_count(node)


def ast_stats(node: ASTNode) -> dict:
    return {
        "node_count": node_count(node),
        "edge_count": edge_count(node),
        "leaf_count": leaf_count(node),
        "operator_count": operator_count(node),
        "depth": depth(node),
    }


def structural_signature(node: ASTNode) -> str:
    # turns the tree into one string so two trees can be compared
    # without walking both by hand. same shape -> same string, always.
    # order matters here too (left/right, base/exp don't get shuffled)
    if not node.children:
        return f"{node.op}:{node.value}"
    child_sigs = ",".join(structural_signature(c) for c in node.children)
    return f"{node.op}({child_sigs})"
