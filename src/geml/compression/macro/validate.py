"""Strict validation for transparent official-construction macro DAGs."""

from __future__ import annotations

from dataclasses import dataclass

from geml.ast.statistics import structural_signature
from geml.contracts.ast import ASTTree
from geml.graph.schema import MACRO_FAMILY, GraphNode, GraphRoot
from geml.graph.validate import validate_graph

from .schema import (
    MACRO_NODE_KIND,
    MACRO_PAYLOAD_FIELD,
    MACRO_RULE_BY_ID,
    MACRO_RULE_BY_OPERATOR,
    MACRO_RULE_FIELD,
    MACRO_SCHEMA_VERSION,
    MACRO_VALUE_FIELDS,
    MacroGraphRecord,
    MacroRule,
    macro_payload_errors,
    macro_representation_mode,
    pure_eml_representation_mode,
)


@dataclass(frozen=True, slots=True)
class MacroGraphValidationResult:
    """The complete deterministic result of one macro validation."""

    valid: bool
    errors: tuple[str, ...] = ()


def _ordered_ast_children(tree: ASTTree) -> dict[str, tuple[str, ...]]:
    slots: dict[str, dict[int, str]] = {node.node_id: {} for node in tree.nodes}
    for edge in tree.edges:
        slots[edge.source_id][edge.child_slot] = edge.target_id
    return {
        node.node_id: tuple(slots[node.node_id][slot] for slot in range(node.arity))
        for node in tree.nodes
    }


def _validate_node(node_id: str, record: MacroGraphRecord, errors: list[str]) -> None:
    node = record.graph.nodes[node_id]
    if not isinstance(node, GraphNode):
        errors.append(f"macro node mapping key {node_id!r} is not a GraphNode")
        return
    if node.family != MACRO_FAMILY:
        errors.append(f"macro node {node_id!r} must use the macro family")
    if node.kind != MACRO_NODE_KIND:
        errors.append(f"macro node {node_id!r} must use kind {MACRO_NODE_KIND!r}")

    spec = MACRO_RULE_BY_OPERATOR.get(node.label or "")
    if spec is None:
        errors.append(f"macro node {node_id!r} has unknown operator {node.label!r}")

    if not isinstance(node.value, dict):
        errors.append(f"macro node {node_id!r} value must be a JSON object")
        return
    if set(node.value) != MACRO_VALUE_FIELDS:
        errors.append(
            f"macro node {node_id!r} value must contain exactly {sorted(MACRO_VALUE_FIELDS)}"
        )
        return

    rule_value = node.value.get(MACRO_RULE_FIELD)
    try:
        rule = MacroRule(rule_value)
    except (TypeError, ValueError):
        errors.append(f"macro node {node_id!r} has unknown expansion rule {rule_value!r}")
        return

    rule_spec = MACRO_RULE_BY_ID[rule]
    if spec is not None and rule_spec != spec:
        errors.append(
            f"macro node {node_id!r} operator {node.label!r} requires expansion "
            f"rule {spec.rule.value!r}, not {rule.value!r}"
        )
    if len(node.children) != rule_spec.arity:
        errors.append(
            f"macro node {node_id!r} expansion rule {rule.value!r} requires "
            f"arity {rule_spec.arity}, observed {len(node.children)}"
        )
    errors.extend(
        f"macro node {node_id!r}: {message}"
        for message in macro_payload_errors(
            rule_spec.operator,
            node.value.get(MACRO_PAYLOAD_FIELD),
        )
    )


def _validate_provenance(record: MacroGraphRecord, errors: list[str]) -> None:
    graph_node_ids = set(record.graph.nodes)
    forward = dict(record.source_to_macro_node)
    inverse = dict(record.macro_to_source_nodes)

    if not forward:
        errors.append("source-to-macro provenance must not be empty")
    unknown_targets = sorted(set(forward.values()) - graph_node_ids)
    if unknown_targets:
        errors.append(
            "source-to-macro provenance references missing macro nodes: "
            + ", ".join(repr(node_id) for node_id in unknown_targets)
        )
    if set(inverse) != graph_node_ids:
        missing = sorted(graph_node_ids - set(inverse))
        extra = sorted(set(inverse) - graph_node_ids)
        errors.append(
            f"inverse provenance keys must equal graph node IDs; missing={missing}, extra={extra}"
        )

    expected_inverse: dict[str, list[str]] = {macro_id: [] for macro_id in graph_node_ids}
    for source_id, macro_id in forward.items():
        if macro_id in expected_inverse:
            expected_inverse[macro_id].append(source_id)

    for macro_id in sorted(graph_node_ids & set(inverse)):
        source_ids = inverse[macro_id]
        expected = tuple(sorted(expected_inverse[macro_id]))
        if not source_ids:
            errors.append(f"inverse provenance for macro node {macro_id!r} must not be empty")
        if tuple(source_ids) != expected:
            errors.append(
                f"inverse provenance for macro node {macro_id!r} must be the "
                f"sorted exact inverse {expected!r}, observed {tuple(source_ids)!r}"
            )


def validate_macro_graph(record: object) -> MacroGraphValidationResult:
    """Validate the macro structure, rule bindings, metadata, and provenance."""

    if not isinstance(record, MacroGraphRecord):
        return MacroGraphValidationResult(
            False,
            ("value must be a MacroGraphRecord",),
        )

    errors: list[str] = []
    graph_validation = validate_graph(record.graph)
    errors.extend(graph_validation.errors)

    if record.schema_version != MACRO_SCHEMA_VERSION:
        errors.append(f"macro schema version must be {MACRO_SCHEMA_VERSION!r}")
    if record.is_pure_eml is not False:
        errors.append("macro record must explicitly declare is_pure_eml=false")

    if len(record.graph.roots) != 1:
        errors.append("a macro graph must contain exactly one root")
    else:
        root = record.graph.roots[0]
        if not isinstance(root, GraphRoot):
            errors.append("macro graph root must be a GraphRoot")
        else:
            expected_mode = macro_representation_mode(record.compiler_mode)
            if root.root_id != record.source_expression_id:
                errors.append("macro root identity must equal source_expression_id")
            if root.representation_mode != expected_mode:
                errors.append(f"macro root representation mode must be {expected_mode!r}")
            expected_target = record.source_to_macro_node.get(record.source_root_id)
            if expected_target is None:
                errors.append("source_root_id is absent from source provenance")
            elif root.target_id != expected_target:
                errors.append("macro root target must equal the source root's provenance target")

    for node_id in sorted(record.graph.nodes):
        _validate_node(node_id, record, errors)
    _validate_provenance(record, errors)

    expansion_cost = record.expansion_cost
    if expansion_cost.compiler_mode is not record.compiler_mode:
        errors.append("expansion-cost compiler mode must match the macro record")
    expected_pure_mode = pure_eml_representation_mode(record.compiler_mode)
    if expansion_cost.representation_mode != expected_pure_mode:
        errors.append(f"expansion-cost representation mode must be {expected_pure_mode!r}")
    if expansion_cost.construction_path != "direct_hashcons":
        errors.append("expansion cost must come from the direct_hashcons construction path")

    return MacroGraphValidationResult(not errors, tuple(errors))


def validate_macro_source_binding(
    record: object,
    source_ast: object,
) -> MacroGraphValidationResult:
    """Validate that provenance binds a macro DAG to one exact source AST."""

    validation = validate_macro_graph(record)
    errors = list(validation.errors)
    if not isinstance(record, MacroGraphRecord):
        return validation
    if not validation.valid:
        return validation
    if not isinstance(source_ast, ASTTree):
        errors.append("source_ast must be a validated ASTTree")
        return MacroGraphValidationResult(False, tuple(errors))

    if source_ast.expression_id != record.source_expression_id:
        errors.append("source AST expression_id does not match the macro record")
    if source_ast.root_id != record.source_root_id:
        errors.append("source AST root_id does not match the macro record")
    if structural_signature(source_ast) != record.source_ast_signature:
        errors.append("source AST structural signature does not match the macro record")

    source_nodes = {node.node_id: node for node in source_ast.nodes}
    if set(record.source_to_macro_node) != set(source_nodes):
        missing = sorted(set(source_nodes) - set(record.source_to_macro_node))
        extra = sorted(set(record.source_to_macro_node) - set(source_nodes))
        errors.append(
            "source provenance keys must equal source AST node IDs; "
            f"missing={missing}, extra={extra}"
        )
        return MacroGraphValidationResult(False, tuple(errors))

    children = _ordered_ast_children(source_ast)
    for source_id in sorted(source_nodes):
        source_node = source_nodes[source_id]
        macro_id = record.source_to_macro_node[source_id]
        macro_node = record.graph.nodes.get(macro_id)
        if macro_node is None:
            continue
        spec = MACRO_RULE_BY_OPERATOR.get(source_node.label)
        if spec is None:
            errors.append(
                f"source node {source_id!r} has unsupported operator {source_node.label!r}"
            )
            continue
        if macro_node.label != source_node.label:
            errors.append(f"source node {source_id!r} operator does not match its macro node")
        if not isinstance(macro_node.value, dict):
            continue
        if macro_node.value.get(MACRO_RULE_FIELD) != spec.rule.value:
            errors.append(f"source node {source_id!r} expansion rule does not match its operator")
        if macro_node.value.get(MACRO_PAYLOAD_FIELD) != source_node.value:
            errors.append(f"source node {source_id!r} payload does not match its macro node")
        expected_children = tuple(
            record.source_to_macro_node[child_id] for child_id in children[source_id]
        )
        observed_children = tuple(
            child.target_id for child in sorted(macro_node.children, key=lambda ref: ref.slot)
        )
        if observed_children != expected_children:
            errors.append(
                f"source node {source_id!r} ordered child bindings do not match its macro node"
            )

    return MacroGraphValidationResult(not errors, tuple(errors))
