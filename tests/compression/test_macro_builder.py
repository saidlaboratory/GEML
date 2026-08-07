"""Tests for exact transparent macro-graph construction."""

from __future__ import annotations

from dataclasses import fields

import pytest

from geml.ast.builder import build_ast_from_parsed
from geml.ast.statistics import calculate_statistics
from geml.compression.macro.builder import (
    MacroBuildFailureStage,
    MacroBuildStatus,
    build_macro_graph,
)
from geml.compression.macro.schema import (
    MACRO_NODE_KIND,
    MACRO_PAYLOAD_FIELD,
    MACRO_RULE_BY_OPERATOR,
    MACRO_RULE_FIELD,
    MacroGraphRecord,
)
from geml.compression.macro.validate import (
    validate_macro_graph,
    validate_macro_source_binding,
)
from geml.contracts.ast import ASTEdge, ASTNode, ASTTree
from geml.eml.compiler_core import CompilerMode
from geml.graph.schema import MACRO_FAMILY, Graph
from geml.interfaces.eml_dag_cost import EMLDagCostStatus, compute_eml_dag_cost
from geml.parsing.srepr import parse_srepr
from geml.spec.operators import OPERATOR_REGISTRY, EMLConstructionStatus


def _ast_from_srepr(source: str, *, expression_id: str = "expression") -> ASTTree:
    return build_ast_from_parsed(
        parse_srepr(source),
        expression_id=expression_id,
    )


def _manual_tree(
    label: str,
    arity: int,
    *,
    expression_id: str | None = None,
    repeated_leaf: bool = False,
) -> ASTTree:
    nodes = [ASTNode(node_id="root", node_kind="operator", label=label, arity=arity)]
    edges: list[ASTEdge] = []
    for slot in range(arity):
        assumptions = {"real": True}
        if label == "divide" and slot == 1:
            assumptions["nonzero"] = True
        name = "x" if repeated_leaf else ("x", "y")[slot]
        node_id = f"leaf-{slot}"
        nodes.append(
            ASTNode(
                node_id=node_id,
                node_kind="leaf",
                label="symbol",
                arity=0,
                value={"name": name, "assumptions": assumptions},
            )
        )
        edges.append(ASTEdge(source_id="root", target_id=node_id, child_slot=slot))
    return ASTTree(
        expression_id=expression_id or f"{label}-expression",
        root_id="root",
        nodes=tuple(nodes),
        edges=tuple(edges),
        statistics=calculate_statistics(nodes, edges, "root"),
    )


def _leaf_tree(label: str, value: object) -> ASTTree:
    nodes = (
        ASTNode(
            node_id="leaf",
            node_kind="leaf",
            label=label,
            arity=0,
            value=value,
        ),
    )
    return ASTTree(
        expression_id=f"{label}-expression",
        root_id="leaf",
        nodes=nodes,
        edges=(),
        statistics=calculate_statistics(nodes, (), "leaf"),
    )


_OPERATOR_SREPRS = {
    "symbol": "Symbol('x', real=True)",
    "one": "Integer(1)",
    "integer": "Integer(-3)",
    "rational": "Rational(2, 3)",
    "add": "Add(Symbol('x', real=True), Symbol('y', real=True))",
    "multiply": "Mul(Symbol('x', real=True), Symbol('y', real=True))",
    "power": "Pow(Symbol('x', positive=True), Rational(2, 3))",
    "exp": "exp(Symbol('x', real=True))",
    "log": "log(Symbol('x', positive=True))",
    "sin": "sin(Symbol('x', real=True))",
    "cos": "cos(Symbol('x', real=True))",
    "tan": "tan(Rational(1, 2))",
    "sinh": "sinh(Symbol('x', real=True))",
    "cosh": "cosh(Symbol('x', real=True))",
    "tanh": "tanh(Symbol('x', real=True))",
}
_MANUAL_ARITIES = {"subtract": 2, "divide": 2, "negate": 1}
_APPROVED_OPERATORS = tuple(sorted(_OPERATOR_SREPRS.keys() | _MANUAL_ARITIES.keys()))


def _operator_ast(operator: str) -> ASTTree:
    if operator in _OPERATOR_SREPRS:
        return _ast_from_srepr(
            _OPERATOR_SREPRS[operator],
            expression_id=f"{operator}-expression",
        )
    return _manual_tree(operator, _MANUAL_ARITIES[operator])


def _successful_record(
    tree: ASTTree,
    *,
    mode: CompilerMode | None = None,
) -> MacroGraphRecord:
    result = build_macro_graph(tree, compiler_mode=mode)
    assert result.status is MacroBuildStatus.SUCCESS
    assert result.macro_graph is not None
    return result.macro_graph


def test_rule_catalog_is_exactly_enabled_and_approved_registry() -> None:
    enabled_approved = {
        operator.name
        for operator in OPERATOR_REGISTRY.values()
        if operator.enabled_for_generation
        and operator.eml_construction_status is EMLConstructionStatus.APPROVED
    }

    assert set(MACRO_RULE_BY_OPERATOR) == enabled_approved
    assert set(MACRO_RULE_BY_OPERATOR) == set(_APPROVED_OPERATORS)
    assert {"e", "pi", "imaginary_unit"}.isdisjoint(MACRO_RULE_BY_OPERATOR)


@pytest.mark.parametrize("operator", _APPROVED_OPERATORS)
def test_every_approved_operator_builds_one_valid_official_macro(
    operator: str,
) -> None:
    tree = _operator_ast(operator)
    record = _successful_record(tree)
    validation = validate_macro_source_binding(record, tree)
    root_source = next(node for node in tree.nodes if node.node_id == tree.root_id)
    root_macro = record.graph.nodes[record.graph.roots[0].target_id]

    assert validation.valid, validation.errors
    assert root_macro.family == MACRO_FAMILY
    assert root_macro.kind == MACRO_NODE_KIND
    assert root_macro.label == operator
    assert root_macro.value == {
        MACRO_RULE_FIELD: MACRO_RULE_BY_OPERATOR[operator].rule.value,
        MACRO_PAYLOAD_FIELD: root_source.value,
    }
    assert tuple(ref.slot for ref in root_macro.children) == tuple(
        range(MACRO_RULE_BY_OPERATOR[operator].arity)
    )

    independent_cost = compute_eml_dag_cost(tree)
    assert independent_cost.status is EMLDagCostStatus.SUCCESS
    assert record.expansion_cost.eml_dag_node_count == independent_cost.eml_dag_node_count
    assert (
        record.expansion_cost.eml_dag_child_reference_count
        == independent_cost.eml_dag_child_reference_count
    )
    assert record.expansion_cost.eml_dag_depth == independent_cost.eml_dag_depth
    assert record.expansion_cost.root_signature == independent_cost.root_signature


def test_default_and_opt_in_modes_are_explicit_and_structurally_isolated() -> None:
    tree = _manual_tree("negate", 1)
    official = _successful_record(tree)
    clean = _successful_record(tree, mode=CompilerMode.CLEAN_NEGATION)

    assert official.compiler_mode is CompilerMode.OFFICIAL_V4
    assert official.graph.roots[0].representation_mode == ("macro:official_v4:is_pure_eml=false")
    assert official.expansion_cost.representation_mode == "pure_eml:official_v4"
    assert clean.compiler_mode is CompilerMode.CLEAN_NEGATION
    assert clean.graph.roots[0].representation_mode == ("macro:clean_negation:is_pure_eml=false")
    assert clean.expansion_cost.representation_mode == "pure_eml:clean_negation"
    assert official.expansion_cost.root_signature != clean.expansion_cost.root_signature
    assert official.is_pure_eml is False
    assert clean.is_pure_eml is False


def test_equal_source_subtrees_share_without_losing_occurrences() -> None:
    tree = _manual_tree("add", 2, repeated_leaf=True)
    record = _successful_record(tree)
    root = record.graph.nodes[record.graph.roots[0].target_id]

    assert len(record.graph.nodes) == 2
    assert root.children[0].target_id == root.children[1].target_id
    shared_id = root.children[0].target_id
    assert record.source_to_macro_node["leaf-0"] == shared_id
    assert record.source_to_macro_node["leaf-1"] == shared_id
    assert record.macro_to_source_nodes[shared_id] == ("leaf-0", "leaf-1")
    assert record.macro_to_source_nodes[root.node_id] == ("root",)


def test_provenance_is_immutable_and_absent_from_structural_node_values() -> None:
    record = _successful_record(_manual_tree("add", 2, repeated_leaf=True))

    with pytest.raises(TypeError):
        record.source_to_macro_node["new"] = "macro-new"  # type: ignore[index]
    with pytest.raises(TypeError):
        record.macro_to_source_nodes["new"] = ("source",)  # type: ignore[index]

    for node in record.graph.nodes.values():
        serialized_value = repr(node.value)
        assert "leaf-0" not in serialized_value
        assert "leaf-1" not in serialized_value
        assert "'root'" not in serialized_value


def test_construction_is_deterministic_but_ordered_slots_are_significant() -> None:
    first_tree = _manual_tree("subtract", 2, expression_id="same")
    second_tree = _manual_tree("subtract", 2, expression_id="same")
    first = _successful_record(first_tree)
    second = _successful_record(second_tree)

    assert dict(first.graph.nodes) == dict(second.graph.nodes)
    assert first.graph.roots == second.graph.roots
    assert first.source_ast_signature == second.source_ast_signature

    swapped = _manual_tree("subtract", 2, expression_id="same")
    swapped_nodes = tuple(
        node.model_copy(
            update={
                "value": {
                    "name": "y" if node.value["name"] == "x" else "x",
                    "assumptions": node.value["assumptions"],
                }
            }
        )
        if node.label == "symbol"
        else node
        for node in swapped.nodes
    )
    swapped = ASTTree(
        expression_id=swapped.expression_id,
        root_id=swapped.root_id,
        nodes=swapped_nodes,
        edges=swapped.edges,
        statistics=swapped.statistics,
    )
    swapped_record = _successful_record(swapped)

    assert first.graph.roots[0].target_id != swapped_record.graph.roots[0].target_id
    assert first.source_ast_signature != swapped_record.source_ast_signature


@pytest.mark.parametrize(
    ("tree", "expected_status"),
    [
        (_leaf_tree("e", None), MacroBuildStatus.UNSUPPORTED),
        (
            _leaf_tree(
                "rational",
                {"numerator": 2, "denominator": 4},
            ),
            MacroBuildStatus.INVALID_INPUT,
        ),
        (
            _leaf_tree(
                "symbol",
                {"name": "x", "assumptions": {}},
            ),
            MacroBuildStatus.INVALID_INPUT,
        ),
    ],
)
def test_invalid_and_unapproved_sources_are_retained(
    tree: ASTTree,
    expected_status: MacroBuildStatus,
) -> None:
    result = build_macro_graph(tree)

    assert result.status is expected_status
    assert result.failure_stage is MacroBuildFailureStage.INPUT_VALIDATION
    assert result.macro_graph is None
    assert result.error_type
    assert result.error_message


def test_source_node_kind_must_match_the_official_operator_shape() -> None:
    valid = _leaf_tree("integer", 2)
    wrong_kind_node = valid.nodes[0].model_copy(update={"node_kind": "operator"})
    wrong_kind = ASTTree(
        expression_id=valid.expression_id,
        root_id=valid.root_id,
        nodes=(wrong_kind_node,),
        edges=(),
        statistics=valid.statistics,
    )

    result = build_macro_graph(wrong_kind)

    assert result.status is MacroBuildStatus.INVALID_INPUT
    assert result.failure_stage is MacroBuildFailureStage.INPUT_VALIDATION
    assert "requires node_kind 'leaf'" in result.error_message


def test_macro_record_never_stores_original_ast_or_eml_graph() -> None:
    field_names = {field.name for field in fields(MacroGraphRecord)}

    assert "source_ast" not in field_names
    assert "eml_graph" not in field_names
    assert "original_graph" not in field_names
    assert field_names == {
        "graph",
        "compiler_mode",
        "source_expression_id",
        "source_root_id",
        "source_ast_signature",
        "source_to_macro_node",
        "macro_to_source_nodes",
        "expansion_cost",
        "schema_version",
        "is_pure_eml",
    }


def test_source_binding_rejects_a_different_ast() -> None:
    tree = _manual_tree("add", 2, expression_id="expression")
    record = _successful_record(tree)
    different = _manual_tree("subtract", 2, expression_id="expression")

    validation = validate_macro_source_binding(record, different)

    assert not validation.valid
    assert any("structural signature" in error for error in validation.errors)


def test_valid_macro_graph_passes_standalone_validation() -> None:
    record = _successful_record(_ast_from_srepr("sin(Symbol('x', real=True))"))

    validation = validate_macro_graph(record)

    assert validation.valid, validation.errors


def test_standalone_validation_retains_malformed_graph_records() -> None:
    record = _successful_record(_ast_from_srepr("Symbol('x', real=True)"))
    malformed = Graph(
        nodes={"bad": "not-a-node"},  # type: ignore[dict-item]
        roots=("not-a-root",),  # type: ignore[arg-type]
    )
    malformed_record = MacroGraphRecord(
        graph=malformed,
        compiler_mode=record.compiler_mode,
        source_expression_id=record.source_expression_id,
        source_root_id=record.source_root_id,
        source_ast_signature=record.source_ast_signature,
        source_to_macro_node=record.source_to_macro_node,
        macro_to_source_nodes=record.macro_to_source_nodes,
        expansion_cost=record.expansion_cost,
    )

    validation = validate_macro_graph(malformed_record)

    assert not validation.valid
    assert any("not a GraphNode" in error for error in validation.errors)
    assert any("GraphRoot" in error for error in validation.errors)
