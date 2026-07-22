"""Exact validated statistics and signatures for frozen binary AST records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from geml.contracts.ast import ASTEdge, ASTNode, ASTStatistics, ASTTree


class ASTStructureError(ValueError):
    """Node and edge records do not describe one rooted binary tree."""


def _validated_children(
    nodes: Sequence[ASTNode],
    edges: Sequence[ASTEdge],
    root_id: str,
) -> tuple[dict[str, ASTNode], dict[str, dict[int, str]]]:
    node_by_id: dict[str, ASTNode] = {}
    for node in nodes:
        if node.node_id in node_by_id:
            raise ASTStructureError(f"duplicate node_id: {node.node_id}")
        node_by_id[node.node_id] = node
    if root_id not in node_by_id:
        raise ASTStructureError("root_id must reference an existing node")

    children: dict[str, dict[int, str]] = {node_id: {} for node_id in node_by_id}
    incoming = {node_id: 0 for node_id in node_by_id}
    for edge in edges:
        if edge.source_id not in node_by_id or edge.target_id not in node_by_id:
            raise ASTStructureError("edge endpoint does not reference an existing node")
        source = node_by_id[edge.source_id]
        if edge.child_slot >= source.arity:
            raise ASTStructureError("child_slot exceeds its source-node arity")
        if edge.child_slot in children[edge.source_id]:
            raise ASTStructureError("a parent has duplicate child slots")
        children[edge.source_id][edge.child_slot] = edge.target_id
        incoming[edge.target_id] += 1
        if incoming[edge.target_id] > 1:
            raise ASTStructureError("a tree node cannot have multiple parents")

    if incoming[root_id] != 0:
        raise ASTStructureError("the root cannot have a parent")
    for node_id, node in node_by_id.items():
        if set(children[node_id]) != set(range(node.arity)):
            raise ASTStructureError("node arity does not match ordered child slots")
        if node_id != root_id and incoming[node_id] != 1:
            raise ASTStructureError("every non-root node must have exactly one parent")
    return node_by_id, children


def _tree_depth(
    node_by_id: dict[str, ASTNode],
    children: dict[str, dict[int, str]],
    root_id: str,
) -> int:
    """Validate reachability iteratively and return leaf-zero maximum depth."""

    depths = {root_id: 0}
    pending = [root_id]
    while pending:
        node_id = pending.pop()
        child_depth = depths[node_id] + 1
        for slot in reversed(range(node_by_id[node_id].arity)):
            child_id = children[node_id][slot]
            if child_id in depths:
                raise ASTStructureError("AST contains a cycle or repeated node reference")
            depths[child_id] = child_depth
            pending.append(child_id)
    if len(depths) != len(node_by_id):
        raise ASTStructureError("AST contains nodes unreachable from the root")
    return max(depths.values())


def calculate_statistics(
    nodes: Sequence[ASTNode],
    edges: Sequence[ASTEdge],
    root_id: str,
) -> ASTStatistics:
    """Compute exact counts and leaf-zero depth after validating tree topology."""

    if not nodes:
        raise ASTStructureError("an AST requires at least one node")
    node_by_id, children = _validated_children(nodes, edges, root_id)
    tree_depth = _tree_depth(node_by_id, children, root_id)
    leaf_count = sum(node.arity == 0 for node in nodes)
    return ASTStatistics(
        node_count=len(nodes),
        edge_count=len(edges),
        leaf_count=leaf_count,
        operator_count=len(nodes) - leaf_count,
        depth=tree_depth,
    )


def recompute_statistics(tree: ASTTree) -> ASTStatistics:
    """Recompute rather than trusting the statistics stored on a tree."""

    return calculate_statistics(tree.nodes, tree.edges, tree.root_id)


def structural_signature(tree: ASTTree) -> str:
    """Hash a canonical ordered structure independent of node identifiers."""

    node_by_id, children = _validated_children(tree.nodes, tree.edges, tree.root_id)
    _tree_depth(node_by_id, children, tree.root_id)

    digest = hashlib.sha256()
    events: list[str | bytes] = [tree.root_id]
    while events:
        event = events.pop()
        if isinstance(event, bytes):
            digest.update(event)
            continue

        node = node_by_id[event]
        digest.update(b"[")
        for index, value in enumerate((node.node_kind, node.label, node.arity, node.value)):
            if index:
                digest.update(b",")
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8", errors="backslashreplace")
            digest.update(encoded)
        digest.update(b",[")
        events.append(b"]]")
        for slot in reversed(range(node.arity)):
            events.append(children[event][slot])
            if slot > 0:
                events.append(b",")
    return digest.hexdigest()
