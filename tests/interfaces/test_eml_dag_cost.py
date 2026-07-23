"""Tests for the frozen exact pure-EML DAG cost interface."""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import FrozenInstanceError

import pytest

import geml.interfaces.eml_dag_cost as cost_module
from geml.ast.builder import build_ast_from_parsed
from geml.contracts.ast import ASTEdge, ASTNode, ASTStatistics, ASTTree
from geml.eml.compiler_core import CompilerMode, eml_exp, eml_negate
from geml.eml.ir import EML, One, Variable
from geml.graph.validate import ValidationResult
from geml.interfaces.eml_dag_cost import (
    EMLDagCostFailureStage,
    EMLDagCostInputKind,
    EMLDagCostResult,
    EMLDagCostStatus,
    compute_eml_dag_cost,
)
from geml.parsing.srepr import parse_srepr
from geml.spec.operators import OPERATORS, EMLConstructionStatus


def _ast_from_srepr(source: str, *, expression_id: str = "expression") -> ASTTree:
    return build_ast_from_parsed(
        parse_srepr(source),
        expression_id=expression_id,
    )


def _simple_operator_ast(label: str, arity: int) -> ASTTree:
    if arity not in (1, 2):
        raise ValueError("test operator arity must be one or two")
    variable_names = ("x", "y")
    nodes = [
        ASTNode(
            node_id="n000000",
            node_kind="operator",
            label=label,
            arity=arity,
        )
    ]
    edges = []
    for slot in range(arity):
        node_id = f"n{slot + 1:06d}"
        name = variable_names[slot]
        assumptions = {"real": True}
        if label == "divide" and slot == 1:
            assumptions["nonzero"] = True
        nodes.append(
            ASTNode(
                node_id=node_id,
                node_kind="leaf",
                label="symbol",
                arity=0,
                value={"name": name, "assumptions": assumptions},
            )
        )
        edges.append(
            ASTEdge(
                source_id="n000000",
                target_id=node_id,
                child_slot=slot,
            )
        )
    return ASTTree(
        expression_id=f"{label}-expression",
        root_id="n000000",
        nodes=tuple(nodes),
        edges=tuple(edges),
        statistics=ASTStatistics(
            node_count=arity + 1,
            edge_count=arity,
            leaf_count=arity,
            operator_count=1,
            depth=1,
        ),
    )


def _invalid_integer_ast() -> ASTTree:
    return ASTTree(
        expression_id="invalid-integer",
        root_id="n000000",
        nodes=(
            ASTNode(
                node_id="n000000",
                node_kind="leaf",
                label="integer",
                arity=0,
                value="not-an-integer",
            ),
        ),
        statistics=ASTStatistics(
            node_count=1,
            edge_count=0,
            leaf_count=1,
            operator_count=0,
            depth=0,
        ),
    )


def _deep_exp_ast(depth: int) -> ASTTree:
    nodes: list[ASTNode] = []
    edges: list[ASTEdge] = []
    for index in range(depth):
        node_id = f"n{index:06d}"
        child_id = f"n{index + 1:06d}"
        nodes.append(
            ASTNode(
                node_id=node_id,
                node_kind="operator",
                label="exp",
                arity=1,
            )
        )
        edges.append(
            ASTEdge(
                source_id=node_id,
                target_id=child_id,
                child_slot=0,
            )
        )
    nodes.append(
        ASTNode(
            node_id=f"n{depth:06d}",
            node_kind="leaf",
            label="symbol",
            arity=0,
            value={"name": "x", "assumptions": {"real": True}},
        )
    )
    return ASTTree(
        expression_id="deep-exp",
        root_id="n000000",
        nodes=tuple(nodes),
        edges=tuple(edges),
        statistics=ASTStatistics(
            node_count=depth + 1,
            edge_count=depth,
            leaf_count=1,
            operator_count=depth,
            depth=depth,
        ),
    )


def _assert_failure_has_no_partial_cost(result: EMLDagCostResult) -> None:
    assert result.status is not EMLDagCostStatus.SUCCESS
    assert result.eml_dag_node_count is None
    assert result.eml_dag_child_reference_count is None
    assert result.eml_dag_depth is None
    assert result.root_signature is None
    assert result.failure_stage is not None
    assert result.error_type
    assert result.error_message


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
_MANUAL_OPERATOR_ARITIES = {
    "subtract": 2,
    "divide": 2,
    "negate": 1,
}
_COSTED_OPERATORS = tuple(sorted(_OPERATOR_SREPRS.keys() | _MANUAL_OPERATOR_ARITIES.keys()))


def _operator_ast(name: str) -> ASTTree:
    if name in _OPERATOR_SREPRS:
        return _ast_from_srepr(
            _OPERATOR_SREPRS[name],
            expression_id=f"{name}-expression",
        )
    return _simple_operator_ast(name, _MANUAL_OPERATOR_ARITIES[name])


def test_source_ast_uses_exact_official_cost_by_default() -> None:
    tree = _ast_from_srepr("exp(Symbol('x', real=True))")

    result = compute_eml_dag_cost(tree)

    assert result.status is EMLDagCostStatus.SUCCESS
    assert result.input_kind is EMLDagCostInputKind.SOURCE_AST
    assert result.eml_dag_node_count == 3
    assert result.eml_dag_child_reference_count == 2
    assert result.eml_dag_depth == 1
    assert result.root_signature is not None
    assert re.fullmatch(r"[0-9a-f]{64}", result.root_signature)
    assert result.compiler_mode is CompilerMode.OFFICIAL_V4
    assert result.representation_mode == "pure_eml:official_v4"
    assert result.construction_path == "direct_hashcons"
    assert result.failure_stage is None
    assert result.error_type is None
    assert result.error_message is None


@pytest.mark.parametrize("operator_name", _COSTED_OPERATORS)
def test_cost_boundary_covers_every_enabled_approved_operator(
    operator_name: str,
) -> None:
    result = compute_eml_dag_cost(_operator_ast(operator_name))

    assert result.status is EMLDagCostStatus.SUCCESS
    assert result.eml_dag_node_count is not None
    assert result.eml_dag_node_count >= 1


def test_operator_cases_track_the_authoritative_registry() -> None:
    enabled_approved = {
        operator.name
        for operator in OPERATORS
        if operator.enabled_for_generation
        and operator.eml_construction_status is EMLConstructionStatus.APPROVED
    }

    assert set(_COSTED_OPERATORS) == enabled_approved
    assert {"e", "pi", "imaginary_unit"}.isdisjoint(_COSTED_OPERATORS)


@pytest.mark.parametrize("mode", tuple(CompilerMode))
def test_source_and_provided_tree_paths_match_exact_structure(
    mode: CompilerMode,
) -> None:
    source_tree = _simple_operator_ast("negate", 1)
    eml_tree = eml_negate(Variable("x"), mode=mode)

    direct = compute_eml_dag_cost(source_tree, compiler_mode=mode)
    posthoc = compute_eml_dag_cost(eml_tree)

    assert direct.status is EMLDagCostStatus.SUCCESS
    assert posthoc.status is EMLDagCostStatus.SUCCESS
    assert direct.eml_dag_node_count == posthoc.eml_dag_node_count
    assert direct.eml_dag_child_reference_count == posthoc.eml_dag_child_reference_count
    assert direct.eml_dag_depth == posthoc.eml_dag_depth
    assert direct.root_signature == posthoc.root_signature
    assert direct.compiler_mode is mode
    assert posthoc.compiler_mode is None
    assert posthoc.representation_mode == "pure_eml:provided_tree"
    assert posthoc.construction_path == "posthoc_hashcons"


def test_clean_negation_is_explicit_and_structurally_distinct() -> None:
    tree = _simple_operator_ast("negate", 1)

    default = compute_eml_dag_cost(tree)
    official = compute_eml_dag_cost(tree, compiler_mode=CompilerMode.OFFICIAL_V4)
    clean = compute_eml_dag_cost(tree, compiler_mode=CompilerMode.CLEAN_NEGATION)

    assert default.status is EMLDagCostStatus.SUCCESS
    assert official.status is EMLDagCostStatus.SUCCESS
    assert clean.status is EMLDagCostStatus.SUCCESS
    assert default.root_signature == official.root_signature
    assert default.compiler_mode is CompilerMode.OFFICIAL_V4
    assert clean.compiler_mode is CompilerMode.CLEAN_NEGATION
    assert clean.representation_mode == "pure_eml:clean_negation"
    assert clean.root_signature != official.root_signature


def test_provided_tree_shares_only_exact_repeated_structure() -> None:
    shared = eml_exp(Variable("x"))

    result = compute_eml_dag_cost(EML(shared, shared))

    assert result.status is EMLDagCostStatus.SUCCESS
    assert result.eml_dag_node_count == 4
    assert result.eml_dag_child_reference_count == 4
    assert result.eml_dag_depth == 2


def test_signature_is_deterministic_and_structure_sensitive() -> None:
    first = compute_eml_dag_cost(eml_exp(Variable("x")))
    repeated = compute_eml_dag_cost(eml_exp(Variable("x")))
    different = compute_eml_dag_cost(eml_exp(Variable("y")))

    assert first.status is EMLDagCostStatus.SUCCESS
    assert repeated.status is EMLDagCostStatus.SUCCESS
    assert different.status is EMLDagCostStatus.SUCCESS
    assert first.root_signature == repeated.root_signature
    assert first.eml_dag_node_count == different.eml_dag_node_count
    assert first.root_signature != different.root_signature


def test_deep_source_ast_is_costed_iteratively() -> None:
    depth = 1_200

    result = compute_eml_dag_cost(_deep_exp_ast(depth))

    assert result.status is EMLDagCostStatus.SUCCESS
    assert result.eml_dag_node_count == depth + 2
    assert result.eml_dag_child_reference_count == 2 * depth
    assert result.eml_dag_depth == depth


def test_result_is_frozen() -> None:
    result = compute_eml_dag_cost(One())

    with pytest.raises(FrozenInstanceError):
        result.status = EMLDagCostStatus.FAILURE  # type: ignore[misc]


def test_wrong_input_type_is_retained_as_invalid() -> None:
    result = compute_eml_dag_cost("exp(x)")  # type: ignore[arg-type]

    assert result.status is EMLDagCostStatus.INVALID_INPUT
    assert result.input_kind is None
    assert result.failure_stage is EMLDagCostFailureStage.INPUT_VALIDATION
    assert result.error_type == "TypeError"
    _assert_failure_has_no_partial_cost(result)


def test_invalid_compiler_mode_is_retained() -> None:
    result = compute_eml_dag_cost(
        _simple_operator_ast("negate", 1),
        compiler_mode="official_v4",  # type: ignore[arg-type]
    )

    assert result.status is EMLDagCostStatus.INVALID_INPUT
    assert result.failure_stage is EMLDagCostFailureStage.INPUT_VALIDATION
    assert result.error_type == "TypeError"
    _assert_failure_has_no_partial_cost(result)


def test_provided_tree_rejects_compiler_provenance_claim() -> None:
    result = compute_eml_dag_cost(
        eml_exp(Variable("x")),
        compiler_mode=CompilerMode.OFFICIAL_V4,
    )

    assert result.status is EMLDagCostStatus.INVALID_INPUT
    assert result.input_kind is EMLDagCostInputKind.PURE_EML_TREE
    assert result.failure_stage is EMLDagCostFailureStage.INPUT_VALIDATION
    _assert_failure_has_no_partial_cost(result)


def test_semantically_invalid_ast_payload_is_retained() -> None:
    result = compute_eml_dag_cost(_invalid_integer_ast())

    assert result.status is EMLDagCostStatus.INVALID_INPUT
    assert result.input_kind is EMLDagCostInputKind.SOURCE_AST
    assert result.failure_stage is EMLDagCostFailureStage.COMPILATION
    assert result.error_type == "ValueError"
    _assert_failure_has_no_partial_cost(result)


def test_unsupported_ast_operator_is_retained() -> None:
    result = compute_eml_dag_cost(_simple_operator_ast("sqrt", 1))

    assert result.status is EMLDagCostStatus.UNSUPPORTED
    assert result.failure_stage is EMLDagCostFailureStage.COMPILATION
    assert result.error_type == "UnsupportedASTOperatorError"
    _assert_failure_has_no_partial_cost(result)


def test_cyclic_eml_is_retained_as_invalid() -> None:
    cyclic = object.__new__(EML)
    object.__setattr__(cyclic, "left", cyclic)
    object.__setattr__(cyclic, "right", One())

    result = compute_eml_dag_cost(cyclic)

    assert result.status is EMLDagCostStatus.INVALID_INPUT
    assert result.failure_stage is EMLDagCostFailureStage.INPUT_VALIDATION
    assert result.error_type == "PureEMLValidationError"
    _assert_failure_has_no_partial_cost(result)


def test_unexpected_compilation_failure_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_compilation(*args: object, **kwargs: object) -> None:
        raise RuntimeError("deliberate compiler failure")

    monkeypatch.setattr(
        cost_module,
        "compile_ast_to_eml_dag",
        fail_compilation,
    )

    result = compute_eml_dag_cost(_ast_from_srepr("Symbol('x', real=True)"))

    assert result.status is EMLDagCostStatus.FAILURE
    assert result.failure_stage is EMLDagCostFailureStage.COMPILATION
    assert result.error_type == "RuntimeError"
    _assert_failure_has_no_partial_cost(result)


def test_dag_validation_failure_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cost_module,
        "validate_eml_dag",
        lambda graph: ValidationResult(False, ("deliberately invalid DAG",)),
    )

    result = compute_eml_dag_cost(One())

    assert result.status is EMLDagCostStatus.FAILURE
    assert result.failure_stage is EMLDagCostFailureStage.DAG_VALIDATION
    assert result.error_type == "EMLDagValidationError"
    assert "deliberately invalid DAG" in result.error_message
    _assert_failure_has_no_partial_cost(result)


def test_cost_computation_failure_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_statistics(graph: object) -> None:
        raise ArithmeticError("deliberate cost failure")

    monkeypatch.setattr(cost_module, "compute_statistics", fail_statistics)

    result = compute_eml_dag_cost(One())

    assert result.status is EMLDagCostStatus.FAILURE
    assert result.failure_stage is EMLDagCostFailureStage.COST_COMPUTATION
    assert result.error_type == "ArithmeticError"
    _assert_failure_has_no_partial_cost(result)


def test_result_contract_rejects_partial_failure_costs() -> None:
    with pytest.raises(ValueError, match="partial costs"):
        EMLDagCostResult(
            status=EMLDagCostStatus.FAILURE,
            input_kind=EMLDagCostInputKind.SOURCE_AST,
            eml_dag_node_count=3,
            failure_stage=EMLDagCostFailureStage.COST_COMPUTATION,
            error_type="Failure",
            error_message="partial cost must not escape",
        )


def test_result_contract_rejects_nonintegral_success_costs() -> None:
    with pytest.raises(ValueError, match="node count"):
        EMLDagCostResult(
            status=EMLDagCostStatus.SUCCESS,
            input_kind=EMLDagCostInputKind.SOURCE_AST,
            eml_dag_node_count=3.0,  # type: ignore[arg-type]
            eml_dag_child_reference_count=2,
            eml_dag_depth=1,
            root_signature="0" * 64,
            compiler_mode=CompilerMode.OFFICIAL_V4,
            representation_mode="pure_eml:official_v4",
            construction_path="direct_hashcons",
        )


def test_module_has_no_experiment_analysis_or_file_io_dependency() -> None:
    source = inspect.getsource(cost_module)
    syntax = ast.parse(source)
    forbidden_prefixes = (
        "geml.experiments",
        "geml.analysis",
        "geml.plots",
    )
    forbidden_modules = {"io", "os", "pathlib", "subprocess"}

    for node in ast.walk(syntax):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(forbidden_prefixes)
            assert node.module.split(".", maxsplit=1)[0] not in forbidden_modules
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(forbidden_prefixes)
                assert alias.name.split(".", maxsplit=1)[0] not in forbidden_modules
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open"

    assert "outputs/" not in source
