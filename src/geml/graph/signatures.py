"""Canonical, deterministic structural signatures for graph nodes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from pydantic import JsonValue

from geml.graph.schema import Graph, GraphNode, strict_json_snapshot

SIGNATURE_VERSION = "geml-graph-signature-v1"


def signature_from_parts(
    *,
    family: str,
    kind: str,
    label: str | None,
    value: JsonValue,
    children: Iterable[tuple[int, str]],
) -> str:
    """Hash one canonical node payload from already-computed child signatures."""

    ordered_children = sorted(children, key=lambda child: child[0])
    payload = {
        "arity": len(ordered_children),
        "children": [
            {"signature": child_signature, "slot": slot}
            for slot, child_signature in ordered_children
        ],
        "family": family,
        "kind": kind,
        "label": label,
        "value": strict_json_snapshot(value),
        "version": SIGNATURE_VERSION,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_signatures(graph: Graph, node_ids: Iterable[str]) -> dict[str, str]:
    """Compute signatures iteratively, memoizing shared descendants."""

    signatures: dict[str, str] = {}
    active: set[str] = set()

    for requested_id in node_ids:
        if requested_id not in graph.nodes:
            raise KeyError(f"graph node {requested_id!r} does not exist")
        stack: list[tuple[str, bool]] = [(requested_id, False)]
        while stack:
            node_id, leaving = stack.pop()
            if node_id in signatures:
                continue
            node = graph.nodes.get(node_id)
            if node is None:
                raise KeyError(f"graph node {node_id!r} does not exist")
            if not isinstance(node, GraphNode):
                raise TypeError(f"graph node {node_id!r} is not a GraphNode record")

            if leaving:
                signatures[node_id] = signature_from_parts(
                    family=node.family,
                    kind=node.kind,
                    label=node.label,
                    value=node.value,
                    children=((child.slot, signatures[child.target_id]) for child in node.children),
                )
                active.remove(node_id)
                continue

            if node_id in active:
                raise ValueError(f"cannot sign cyclic graph at node {node_id!r}")
            active.add(node_id)
            stack.append((node_id, True))
            for child in reversed(sorted(node.children, key=lambda ref: ref.slot)):
                if child.target_id in active:
                    raise ValueError(
                        f"cannot sign cyclic graph edge {node_id!r} -> {child.target_id!r}"
                    )
                if child.target_id not in graph.nodes:
                    raise KeyError(f"graph node {child.target_id!r} does not exist")
                if child.target_id not in signatures:
                    stack.append((child.target_id, False))

    return signatures


def compute_signature(graph: Graph, node_id: str) -> str:
    """Return one node's canonical 64-character structural signature."""

    return compute_signatures(graph, (node_id,))[node_id]
