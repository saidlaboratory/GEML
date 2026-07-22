"""
signatures.py - canonical structural signatures for graph nodes

owned by 3-1

structure only, not math meaning. x+x and 2*x mean the same thing but
get different signatures on purpose - proving they're actually equal
is goal 4's job (e-graphs), not this schema's
"""
from __future__ import annotations
from geml.graph.schema import Graph


def compute_signature(graph: Graph, node_id: str) -> str:
    node = graph.nodes[node_id]
    header = f"{node.family}:{node.kind}:{node.label}:{node.value}:arity{len(node.children)}"
    if not node.children:
        return header
    # slot embedded per child, so swapping child order changes the string
    child_sigs = ",".join(
        f"{ref.slot}={compute_signature(graph, ref.target_id)}"
        for ref in node.children
    )
    return f"{header}({child_sigs})"
