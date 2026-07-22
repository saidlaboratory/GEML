"""Hand-audited contract tests for deterministic binary AST construction."""

from __future__ import annotations

from collections import defaultdict

import pytest

from geml.ast.builder import build_ast
from geml.ast.statistics import structural_signature
from geml.contracts.ast import ASTNode, ASTTree
from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.parsing.srepr import UnsupportedNodeError


def _record(sympy_srepr: str, *, expression_id: str = "b" * 64) -> ExpressionRecord:
    return ExpressionRecord(
        expression_id=expression_id,
        sympy_srepr=sympy_srepr,
        display_text="fixture",
        split=CorpusSplit.TRAIN,
        operator_family="fixture",
        domain_mode="safe_real",
        variables=("x",),
        target_ast_size=1,
        target_depth=0,
        generator_seed=1,
        generator_metadata={},
    )


def _nodes(tree: ASTTree) -> dict[str, ASTNode]:
    return {node.node_id: node for node in tree.nodes}


def _children(tree: ASTTree) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = defaultdict(dict)
    for edge in tree.edges:
        result[edge.source_id][edge.child_slot] = edge.target_id
    return result


def test_same_srepr_produces_identical_contract_tree_and_signature() -> None:
    record = _record("Mul(Add(Symbol('x', real=True), Integer(1)), Rational(3, 2))")
    first = build_ast(record)
    second = build_ast(record)

    assert isinstance(first, ASTTree)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert structural_signature(first) == structural_signature(second)
    assert [node.node_id for node in first.nodes] == [
        f"n{index:06d}" for index in range(len(first.nodes))
    ]


@pytest.mark.parametrize(
    ("constructor", "label"),
    [("Add", "add"), ("Mul", "multiply")],
)
def test_nary_operators_fold_left_in_argument_order(constructor: str, label: str) -> None:
    tree = build_ast(
        _record(
            f"{constructor}(Symbol('x', real=True), Symbol('y', real=True), Symbol('z', real=True))"
        )
    )
    nodes = _nodes(tree)
    children = _children(tree)
    root = nodes[tree.root_id]
    inner = nodes[children[root.node_id][0]]
    right = nodes[children[root.node_id][1]]

    assert root.label == inner.label == label
    assert root.metadata == {
        "sympy_constructor": constructor,
        "binary_fold": "left",
        "source_arity": 3,
        "fold_step": 2,
    }
    assert inner.metadata["fold_step"] == 1
    assert right.value == {"name": "z", "assumptions": {"real": True}}
    assert nodes[children[inner.node_id][0]].value["name"] == "x"
    assert nodes[children[inner.node_id][1]].value["name"] == "y"


def test_power_preserves_base_and_exponent_slots() -> None:
    tree = build_ast(_record("Pow(Symbol('x', real=True), Integer(-3))"))
    nodes = _nodes(tree)
    children = _children(tree)[tree.root_id]

    assert nodes[tree.root_id].label == "power"
    assert nodes[children[0]].value == {"name": "x", "assumptions": {"real": True}}
    assert nodes[children[1]].value == -3


def test_exact_leaf_payloads_and_symbol_assumptions_are_preserved() -> None:
    tree = build_ast(
        _record("Add(Symbol('x', real=True, nonzero=True), Mul(Integer(1), Rational(-7, 5)))")
    )
    nodes = tree.nodes
    assert any(
        node.label == "symbol"
        and node.value == {"name": "x", "assumptions": {"nonzero": True, "real": True}}
        for node in nodes
    )
    assert any(node.label == "one" and node.value == 1 for node in nodes)
    assert any(
        node.label == "rational" and node.value == {"numerator": -7, "denominator": 5}
        for node in nodes
    )


def test_exp_log_and_lowered_arithmetic_encodings_are_supported() -> None:
    source = (
        "Add(Mul(Integer(-1), exp(Symbol('x', positive=True))), "
        "Mul(Symbol('x', positive=True), Pow(log(Symbol('x', positive=True)), Integer(-1))))"
    )
    tree = build_ast(_record(source))
    assert {node.label for node in tree.nodes} >= {
        "add",
        "multiply",
        "integer",
        "exp",
        "power",
        "log",
        "symbol",
    }
    assert tree.statistics.node_count == len(tree.nodes)
    assert tree.statistics.edge_count == len(tree.edges)


def test_repeated_subtrees_remain_distinct_ordered_occurrences() -> None:
    subtree = "Add(Symbol('x', real=True), Integer(1))"
    tree = build_ast(_record(f"Mul({subtree}, {subtree})"))
    nodes = _nodes(tree)
    root_children = _children(tree)[tree.root_id]

    assert root_children[0] != root_children[1]
    left = nodes[root_children[0]]
    right = nodes[root_children[1]]
    assert left.label == right.label == "add"
    assert left.node_id != right.node_id
    assert tree.statistics.node_count == 7
    assert tree.statistics.edge_count == 6
    assert tree.statistics.depth == 2


def test_large_nary_fold_builds_and_hashes_without_recursive_traversal() -> None:
    operand_count = 500
    source = "Add(" + ", ".join(f"Integer({index})" for index in range(operand_count)) + ")"

    tree = build_ast(_record(source))
    assert tree.statistics.node_count == 2 * operand_count - 1
    assert tree.statistics.edge_count == 2 * operand_count - 2
    assert tree.statistics.leaf_count == operand_count
    assert tree.statistics.operator_count == operand_count - 1
    assert tree.statistics.depth == operand_count - 1
    assert structural_signature(tree) == structural_signature(tree)


@pytest.mark.parametrize("constructor", ["sin", "cos", "tan", "sinh", "cosh", "tanh"])
def test_pending_trig_and_hyperbolic_nodes_fail_explicitly(constructor: str) -> None:
    with pytest.raises(UnsupportedNodeError) as captured:
        build_ast(_record(f"{constructor}(Symbol('x', real=True))"))
    assert captured.value.constructor == constructor
