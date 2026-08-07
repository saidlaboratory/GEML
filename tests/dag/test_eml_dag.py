"""Tests for exact pure EML tree-to-DAG structural sharing."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import mpmath as mp
import pytest

from geml.dag.eml import (
    convert_with_stats,
    dag_to_eml,
    eml_to_dag,
    validate_eml_dag,
)
from geml.eml.compiler_arithmetic import eml_multiply, eml_power
from geml.eml.compiler_core import eml_add, eml_exp, eml_log
from geml.eml.ir import EML, EMLTerm, One, Variable
from geml.eml.validate import validate_pure_eml
from geml.graph.schema import AST_FAMILY, ChildRef, Graph, GraphNode, GraphRoot
from geml.graph.signatures import compute_signature
from geml.verification.eml.numeric import NumericBackend, evaluate_pure_eml


def _approved_audit_trees() -> dict[str, Callable[[], EMLTerm]]:
    return {
        "exp": lambda: eml_exp(Variable("x")),
        "log": lambda: eml_log(Variable("x")),
        "add": lambda: eml_add(Variable("x"), Variable("y")),
        "multiply": lambda: eml_multiply(Variable("x"), Variable("y")),
        "power": lambda: eml_power(Variable("x"), Variable("y")),
    }


@pytest.mark.parametrize("name", _approved_audit_trees())
def test_actual_goal2_compiler_tree_converts_to_pure_dag(name: str) -> None:
    tree = _approved_audit_trees()[name]()

    graph, statistics = convert_with_stats(
        tree,
        root_id=name,
        representation_mode="pure_eml:official_v4",
    )

    assert validate_eml_dag(graph).valid
    assert statistics.tree_node_count == validate_pure_eml(tree).node_count
    assert statistics.dag_node_count <= statistics.tree_node_count
    assert statistics.compression_ratio >= 1
    assert {node.kind for node in graph.nodes.values()} <= {
        "eml",
        "variable",
        "one",
    }


def test_repeated_source_subexpression_shares_all_descendants() -> None:
    repeated = eml_exp(Variable("x"))
    tree = EML(repeated, repeated)

    graph, statistics = convert_with_stats(tree)
    root = graph.nodes[graph.roots[0].target_id]

    assert statistics.tree_node_count == 7
    assert statistics.dag_node_count == 4
    assert statistics.dag_child_reference_count == 4
    assert statistics.compression_ratio == Fraction(7, 4)
    assert root.children[0].target_id == root.children[1].target_id


def test_distinct_but_identical_trees_share() -> None:
    tree = EML(
        eml_exp(Variable("x")),
        eml_exp(Variable("x")),
    )

    graph = eml_to_dag(tree)
    root = graph.nodes[graph.roots[0].target_id]
    assert root.children[0].target_id == root.children[1].target_id


def test_ordered_children_are_not_commuted() -> None:
    first = eml_to_dag(EML(Variable("x"), Variable("y")))
    second = eml_to_dag(EML(Variable("y"), Variable("x")))

    assert compute_signature(first, first.roots[0].target_id) != compute_signature(
        second,
        second.roots[0].target_id,
    )


@pytest.mark.parametrize("name", _approved_audit_trees())
def test_dag_reconstruction_preserves_official_tree_evaluation(name: str) -> None:
    tree = _approved_audit_trees()[name]()
    graph = eml_to_dag(tree)
    reconstructed = dag_to_eml(graph, graph.roots[0].target_id)
    bindings = {"x": 2.0, "y": 3.0}

    expected, expected_extended = evaluate_pure_eml(
        tree,
        variables=bindings,
        backend=NumericBackend.MPMATH,
    )
    observed, observed_extended = evaluate_pure_eml(
        reconstructed,
        variables=bindings,
        backend=NumericBackend.MPMATH,
    )

    assert observed_extended is expected_extended
    if mp.isnan(expected):
        assert mp.isnan(observed)
    else:
        assert mp.almosteq(observed, expected)


def test_leaf_shapes_are_exact() -> None:
    one_graph, one_statistics = convert_with_stats(One())
    variable_graph, variable_statistics = convert_with_stats(Variable("x"))

    assert one_statistics.compression_ratio == 1
    assert variable_statistics.compression_ratio == 1
    assert one_statistics.dag_depth == variable_statistics.dag_depth == 0
    assert next(iter(one_graph.nodes.values())).kind == "one"
    assert next(iter(variable_graph.nodes.values())).kind == "variable"


def test_eml_validator_rejects_an_ast_graph() -> None:
    graph = Graph(
        nodes={
            "x": GraphNode(
                node_id="x",
                family=AST_FAMILY,
                kind="leaf",
                label="symbol",
                value="x",
            )
        },
        roots=(GraphRoot("expression", "x", "ast"),),
    )

    result = validate_eml_dag(graph)
    assert not result.valid
    assert any("eml family" in error for error in result.errors)


def test_generic_validation_rejects_malformed_eml_slots() -> None:
    graph = Graph(
        nodes={
            "root": GraphNode(
                node_id="root",
                family="eml",
                kind="eml",
                label="eml",
                children=(ChildRef(1, "x"), ChildRef(2, "one")),
            ),
            "x": GraphNode("x", "eml", "variable", "x", "x"),
            "one": GraphNode("one", "eml", "one", "1", 1),
        },
        roots=(GraphRoot("expression", "root", "pure_eml:official_v4"),),
    )

    assert not validate_eml_dag(graph).valid


def test_deep_tree_conversion_is_iterative() -> None:
    depth = 1_500
    tree: EMLTerm = Variable("x")
    for _ in range(depth):
        tree = EML(tree, One())

    _graph, statistics = convert_with_stats(tree)
    assert statistics.tree_node_count == 2 * depth + 1
    assert statistics.dag_node_count == depth + 2
    assert statistics.dag_depth == depth
