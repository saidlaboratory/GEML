"""Tests for exact macro expansion and independent pure-EML comparison."""

from __future__ import annotations

from dataclasses import replace

import pytest

import geml.compression.macro.expand as expansion_module
from geml.ast.builder import build_ast_from_parsed
from geml.ast.statistics import calculate_statistics
from geml.compression.macro.builder import MacroBuildStatus, build_macro_graph
from geml.compression.macro.expand import (
    MacroExpansionError,
    MacroExpansionFailureStage,
    MacroExpansionStatus,
    expand_macro_graph,
    iter_validate_macro_expansions,
    validate_macro_expansion,
)
from geml.compression.macro.schema import MacroGraphRecord
from geml.contracts.ast import ASTEdge, ASTNode, ASTTree
from geml.dag.direct_eml import (
    UnsupportedASTOperatorError,
    compile_ast_to_eml_dag,
)
from geml.dag.eml import validate_eml_dag
from geml.eml.compiler_core import CompilerMode
from geml.graph.schema import (
    EML_FAMILY,
    EML_ONE_KIND,
    EML_OPERATOR_KIND,
    EML_VARIABLE_KIND,
)
from geml.parsing.srepr import parse_srepr


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
) -> ASTTree:
    nodes = [ASTNode(node_id="root", node_kind="operator", label=label, arity=arity)]
    edges: list[ASTEdge] = []
    for slot in range(arity):
        assumptions = {"real": True}
        if label == "divide" and slot == 1:
            assumptions["nonzero"] = True
        node_id = f"leaf-{slot}"
        nodes.append(
            ASTNode(
                node_id=node_id,
                node_kind="leaf",
                label="symbol",
                arity=0,
                value={
                    "name": ("x", "y")[slot],
                    "assumptions": assumptions,
                },
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


def _record(
    tree: ASTTree,
    *,
    mode: CompilerMode | None = None,
) -> MacroGraphRecord:
    build = build_macro_graph(tree, compiler_mode=mode)
    assert build.status is MacroBuildStatus.SUCCESS
    assert build.macro_graph is not None
    return build.macro_graph


@pytest.mark.parametrize("operator", _APPROVED_OPERATORS)
def test_every_approved_operator_expands_to_exact_reference_identity(
    operator: str,
) -> None:
    tree = _operator_ast(operator)
    record = _record(tree)

    result = validate_macro_expansion(record, tree)

    assert result.status is MacroExpansionStatus.SUCCESS
    assert result.expanded_identity == result.reference_identity
    assert result.expanded_graph is None
    assert result.error_type is None
    assert result.error_message is None
    assert result.failure_stage is None


@pytest.mark.parametrize(
    "mode",
    [CompilerMode.OFFICIAL_V4, CompilerMode.CLEAN_NEGATION],
)
def test_expansion_is_strictly_pure_eml_under_each_explicit_mode(
    mode: CompilerMode,
) -> None:
    tree = _manual_tree("negate", 1)
    record = _record(tree, mode=mode)

    result = validate_macro_expansion(
        record,
        tree,
        retain_expanded_graph=True,
    )

    assert result.status is MacroExpansionStatus.SUCCESS
    assert result.expanded_graph is not None
    validation = validate_eml_dag(result.expanded_graph)
    assert validation.valid, validation.errors
    assert result.expanded_graph.roots[0].representation_mode == (f"pure_eml:{mode.value}")
    assert all(node.family == EML_FAMILY for node in result.expanded_graph.nodes.values())
    assert {node.kind for node in result.expanded_graph.nodes.values()} <= {
        EML_OPERATOR_KIND,
        EML_VARIABLE_KIND,
        EML_ONE_KIND,
    }
    assert {
        node.label
        for node in result.expanded_graph.nodes.values()
        if node.kind == EML_OPERATOR_KIND
    } == {"eml"}


def test_official_v4_is_default_and_clean_negation_remains_distinct() -> None:
    tree = _manual_tree("negate", 1)
    official = validate_macro_expansion(_record(tree), tree)
    clean = validate_macro_expansion(
        _record(tree, mode=CompilerMode.CLEAN_NEGATION),
        tree,
    )

    assert official.status is MacroExpansionStatus.SUCCESS
    assert clean.status is MacroExpansionStatus.SUCCESS
    assert official.compiler_mode is CompilerMode.OFFICIAL_V4
    assert clean.compiler_mode is CompilerMode.CLEAN_NEGATION
    assert official.expanded_identity != clean.expanded_identity
    assert official.expanded_identity is not None
    assert clean.expanded_identity is not None
    assert official.expanded_identity.representation_mode == "pure_eml:official_v4"
    assert clean.expanded_identity.representation_mode == "pure_eml:clean_negation"


def test_expand_macro_graph_matches_the_reference_graph_exactly() -> None:
    tree = _ast_from_srepr(
        "Add(Mul(Rational(-2, 3), sin(Symbol('x', real=True))), "
        "Pow(log(Symbol('y', positive=True)), Integer(2)))"
    )
    record = _record(tree)

    expanded = expand_macro_graph(record)
    reference, reference_root, _ = compile_ast_to_eml_dag(tree)

    assert expanded.roots[0].target_id == reference_root
    assert dict(expanded.nodes) == dict(reference.nodes)
    assert expanded.roots == reference.roots


def test_stored_cost_mismatch_is_retained_without_a_partial_graph() -> None:
    tree = _ast_from_srepr("exp(Symbol('x', real=True))")
    record = _record(tree)
    wrong_cost = replace(
        record.expansion_cost,
        eml_dag_depth=record.expansion_cost.eml_dag_depth + 1,
    )
    corrupted = replace(record, expansion_cost=wrong_cost)

    result = validate_macro_expansion(
        corrupted,
        tree,
        retain_expanded_graph=True,
    )

    assert result.status is MacroExpansionStatus.MISMATCH
    assert result.failure_stage is MacroExpansionFailureStage.COMPARISON
    assert result.error_type == "MacroExpansionMismatch"
    assert "stored expansion depth" in result.error_message
    assert result.expanded_identity == result.reference_identity
    assert result.expanded_graph is None


def test_independent_reference_mismatch_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _manual_tree("subtract", 2, expression_id="expression")
    record = _record(tree)
    different = _manual_tree("add", 2, expression_id="expression")
    wrong_reference = compile_ast_to_eml_dag(different)

    def compile_wrong_reference(
        _tree: ASTTree,
        *,
        mode: CompilerMode,
    ):
        assert mode is CompilerMode.OFFICIAL_V4
        return wrong_reference

    monkeypatch.setattr(
        expansion_module,
        "compile_ast_to_eml_dag",
        compile_wrong_reference,
    )

    result = validate_macro_expansion(record, tree)

    assert result.status is MacroExpansionStatus.MISMATCH
    assert result.failure_stage is MacroExpansionFailureStage.COMPARISON
    assert result.expanded_identity != result.reference_identity
    assert "independent source-AST compile" in result.error_message


def test_reference_unsupported_failure_is_not_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = _ast_from_srepr("exp(Symbol('x', real=True))")
    record = _record(tree)

    def reject_reference(_tree: ASTTree, *, mode: CompilerMode):
        del mode
        raise UnsupportedASTOperatorError("synthetic rejection")

    monkeypatch.setattr(
        expansion_module,
        "compile_ast_to_eml_dag",
        reject_reference,
    )

    result = validate_macro_expansion(record, tree)

    assert result.status is MacroExpansionStatus.UNSUPPORTED
    assert result.failure_stage is MacroExpansionFailureStage.REFERENCE_COMPILATION
    assert result.error_type == "UnsupportedASTOperatorError"
    assert result.expanded_identity is not None
    assert result.expanded_graph is None


def test_invalid_macro_and_wrong_source_have_typed_distinct_stages() -> None:
    tree = _manual_tree("add", 2, expression_id="expression")
    record = _record(tree)
    invalid_record = replace(
        record,
        source_to_macro_node={
            source_id: macro_id
            for source_id, macro_id in record.source_to_macro_node.items()
            if source_id != "leaf-1"
        },
    )
    invalid_result = validate_macro_expansion(invalid_record, tree)
    wrong_source = _manual_tree("subtract", 2, expression_id="expression")
    wrong_source_result = validate_macro_expansion(record, wrong_source)

    assert invalid_result.status is MacroExpansionStatus.INVALID_INPUT
    assert invalid_result.failure_stage is MacroExpansionFailureStage.MACRO_VALIDATION
    assert wrong_source_result.status is MacroExpansionStatus.INVALID_INPUT
    assert wrong_source_result.failure_stage is MacroExpansionFailureStage.SOURCE_VALIDATION

    with pytest.raises(MacroExpansionError, match="invalid macro graph"):
        expand_macro_graph(invalid_record)


def test_streaming_validation_retains_each_row_in_order() -> None:
    first_tree = _ast_from_srepr("exp(Symbol('x', real=True))", expression_id="first")
    second_tree = _manual_tree("add", 2, expression_id="second")
    first_record = _record(first_tree)
    second_record = _record(second_tree)
    wrong_second_source = _manual_tree("subtract", 2, expression_id="second")

    stream = iter_validate_macro_expansions(
        (
            (first_record, first_tree),
            (second_record, wrong_second_source),
        )
    )
    first = next(stream)
    second = next(stream)

    assert first.status is MacroExpansionStatus.SUCCESS
    assert first.expanded_graph is None
    assert second.status is MacroExpansionStatus.INVALID_INPUT
    assert second.failure_stage is MacroExpansionFailureStage.SOURCE_VALIDATION
    with pytest.raises(StopIteration):
        next(stream)


def test_streaming_validation_retains_a_malformed_row() -> None:
    result = next(
        iter_validate_macro_expansions(
            [("record-only",)],  # type: ignore[list-item]
        )
    )

    assert result.status is MacroExpansionStatus.INVALID_INPUT
    assert result.failure_stage is MacroExpansionFailureStage.MACRO_VALIDATION
    assert result.error_type == "ValueError"
    assert result.error_message


def test_deep_macro_build_and_expansion_are_iterative() -> None:
    depth = 1_100
    nodes = [
        ASTNode(
            node_id=f"node-{index}",
            node_kind="operator",
            label="exp",
            arity=1,
        )
        for index in range(depth)
    ]
    nodes.append(
        ASTNode(
            node_id=f"node-{depth}",
            node_kind="leaf",
            label="symbol",
            arity=0,
            value={"name": "x", "assumptions": {"real": True}},
        )
    )
    edges = [
        ASTEdge(
            source_id=f"node-{index}",
            target_id=f"node-{index + 1}",
            child_slot=0,
        )
        for index in range(depth)
    ]
    tree = ASTTree(
        expression_id="deep-expression",
        root_id="node-0",
        nodes=tuple(nodes),
        edges=tuple(edges),
        statistics=calculate_statistics(nodes, edges, "node-0"),
    )

    record = _record(tree)
    result = validate_macro_expansion(record, tree)

    assert result.status is MacroExpansionStatus.SUCCESS
    assert result.expanded_identity is not None
    assert result.expanded_identity.eml_dag_depth == depth
