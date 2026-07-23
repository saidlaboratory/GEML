"""Validation for representation-neutral graph records."""

from __future__ import annotations

import math
from dataclasses import dataclass

from geml.eml.ir import is_valid_source_variable_name
from geml.graph.schema import (
    AST_FAMILY,
    EML_FAMILY,
    EML_ONE_KIND,
    EML_OPERATOR_KIND,
    EML_VARIABLE_KIND,
    REPRESENTATION_FAMILIES,
    ChildRef,
    Graph,
    GraphRoot,
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """The complete, deterministic result of graph validation."""

    valid: bool
    errors: tuple[str, ...] = ()


def _is_nonblank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _check_records(graph: Graph, errors: list[str]) -> None:
    if not graph.nodes:
        errors.append("graph must contain at least one node")
    if not graph.roots:
        errors.append("graph must contain at least one root reference")

    observed_families: set[str] = set()
    for key, node in graph.nodes.items():
        if key != node.node_id:
            errors.append(f"node mapping key {key!r} does not match node_id {node.node_id!r}")
        if not _is_nonblank(node.node_id):
            errors.append(f"node mapping key {key!r} has a blank or invalid node_id")
        if node.family not in REPRESENTATION_FAMILIES:
            errors.append(f"node {node.node_id!r} has unsupported family {node.family!r}")
        else:
            observed_families.add(node.family)
        if not _is_nonblank(node.kind):
            errors.append(f"node {node.node_id!r} has a blank or invalid kind")
        if node.label is not None and not _is_nonblank(node.label):
            errors.append(f"node {node.node_id!r} has a blank label")
        if not _is_json_value(node.value):
            errors.append(f"node {node.node_id!r} has a non-JSON or non-finite value")

    if len(observed_families) > 1:
        errors.append("all nodes in one graph must use the same representation family")


def _check_roots(graph: Graph, errors: list[str]) -> None:
    seen_root_ids: set[str] = set()
    for root in graph.roots:
        if not isinstance(root, GraphRoot):
            errors.append("graph contains a non-GraphRoot root record")
            continue
        if not _is_nonblank(root.root_id):
            errors.append(f"root identity {root.root_id!r} is blank or invalid")
        elif root.root_id in seen_root_ids:
            errors.append(f"duplicate root identity {root.root_id!r}")
        else:
            seen_root_ids.add(root.root_id)
        if not _is_nonblank(root.target_id):
            errors.append(f"root {root.root_id!r} has a blank or invalid target_id")
        elif root.target_id not in graph.nodes:
            errors.append(
                f"root {root.root_id!r} target {root.target_id!r} does not exist in graph.nodes"
            )
        if not _is_nonblank(root.representation_mode):
            errors.append(f"root {root.root_id!r} has a blank representation mode")


def _check_child_refs(graph: Graph, errors: list[str]) -> None:
    for node in graph.nodes.values():
        seen_slots: set[int] = set()
        for ref in node.children:
            if not isinstance(ref, ChildRef):
                errors.append(f"node {node.node_id!r} contains a non-ChildRef child record")
                continue
            if isinstance(ref.slot, bool) or not isinstance(ref.slot, int) or ref.slot < 0:
                errors.append(f"node {node.node_id!r} has invalid child slot {ref.slot!r}")
            elif ref.slot in seen_slots:
                errors.append(f"node {node.node_id!r} has duplicate child slot {ref.slot}")
            else:
                seen_slots.add(ref.slot)
            if not _is_nonblank(ref.target_id):
                errors.append(f"node {node.node_id!r} has a blank or invalid child target")
            elif ref.target_id not in graph.nodes:
                errors.append(f"node {node.node_id!r} references missing node {ref.target_id!r}")

        expected_slots = set(range(len(node.children)))
        if seen_slots != expected_slots:
            errors.append(
                f"node {node.node_id!r} child slots must be contiguous from zero; "
                f"expected {sorted(expected_slots)}, observed {sorted(seen_slots)}"
            )


def _reachable_nodes(graph: Graph) -> set[str]:
    visited: set[str] = set()
    stack = [
        root.target_id
        for root in graph.roots
        if isinstance(root, GraphRoot) and root.target_id in graph.nodes
    ]
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        for ref in graph.nodes[node_id].children:
            if isinstance(ref, ChildRef) and ref.target_id in graph.nodes:
                stack.append(ref.target_id)
    return visited


def _check_reachability(graph: Graph, errors: list[str]) -> None:
    reachable = _reachable_nodes(graph)
    for node_id in graph.nodes:
        if node_id not in reachable:
            errors.append(f"node {node_id!r} is unreachable from every root")


def _check_acyclic(graph: Graph, errors: list[str]) -> None:
    white, gray, black = 0, 1, 2
    color = {node_id: white for node_id in graph.nodes}

    for start_id in graph.nodes:
        if color[start_id] != white:
            continue
        stack: list[tuple[str, int]] = [(start_id, 0)]
        color[start_id] = gray
        while stack:
            node_id, child_index = stack[-1]
            children = graph.nodes[node_id].children
            if child_index >= len(children):
                color[node_id] = black
                stack.pop()
                continue
            stack[-1] = (node_id, child_index + 1)
            ref = children[child_index]
            if not isinstance(ref, ChildRef) or ref.target_id not in graph.nodes:
                continue
            target_color = color[ref.target_id]
            if target_color == gray:
                errors.append(f"cycle detected at edge {node_id!r} -> {ref.target_id!r}")
            elif target_color == white:
                color[ref.target_id] = gray
                stack.append((ref.target_id, 0))


def _check_ast_purity(graph: Graph, errors: list[str]) -> None:
    for node in graph.nodes.values():
        if node.family == AST_FAMILY and len(node.children) > 2:
            errors.append(f"AST node {node.node_id!r} exceeds the binary AST arity limit")


def _check_eml_purity(graph: Graph, errors: list[str]) -> None:
    for node in graph.nodes.values():
        if node.family != EML_FAMILY:
            continue
        if node.kind == EML_OPERATOR_KIND:
            if node.label != "eml" or node.value is not None or len(node.children) != 2:
                errors.append(
                    f"EML operator {node.node_id!r} must be labeled 'eml', "
                    "have no value, and have exactly two children"
                )
        elif node.kind == EML_VARIABLE_KIND:
            if (
                node.children
                or node.label != node.value
                or not is_valid_source_variable_name(node.value)
            ):
                errors.append(f"EML variable {node.node_id!r} must be a valid named leaf")
        elif node.kind == EML_ONE_KIND:
            if node.children or node.label != "1" or type(node.value) is not int or node.value != 1:
                errors.append(f"EML one {node.node_id!r} must be the primitive leaf labeled '1'")
        else:
            errors.append(f"node {node.node_id!r} has forbidden pure-EML kind {node.kind!r}")


def validate_graph(graph: Graph) -> ValidationResult:
    """Validate all graph invariants without dropping later failures."""

    if not isinstance(graph, Graph):
        return ValidationResult(False, ("value must be a Graph record",))

    errors: list[str] = []
    _check_records(graph, errors)
    _check_roots(graph, errors)
    _check_child_refs(graph, errors)
    _check_reachability(graph, errors)
    _check_acyclic(graph, errors)
    _check_ast_purity(graph, errors)
    _check_eml_purity(graph, errors)
    return ValidationResult(not errors, tuple(errors))
