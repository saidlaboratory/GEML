"""Tests for the frozen binary-AST contracts."""

import json

import pytest
from pydantic import ValidationError

from geml.contracts.ast import ASTEdge, ASTNode, ASTStatistics, ASTTree


def _leaf_node(node_id: str, label: str = "x") -> ASTNode:
    return ASTNode(node_id=node_id, node_kind="variable", label=label, arity=0, value=label)


def _binary_tree() -> ASTTree:
    return ASTTree(
        expression_id="expr-000001",
        root_id="n0",
        nodes=(
            ASTNode(node_id="n0", node_kind="operator", label="Add", arity=2),
            _leaf_node("n1", "x"),
            ASTNode(node_id="n2", node_kind="constant", label="Integer", arity=0, value=1),
        ),
        edges=(
            ASTEdge(source_id="n0", target_id="n1", child_slot=0),
            ASTEdge(source_id="n0", target_id="n2", child_slot=1),
        ),
        statistics=ASTStatistics(
            node_count=3,
            edge_count=2,
            leaf_count=2,
            operator_count=1,
            depth=1,
        ),
    )


def test_valid_minimal_leaf_tree_uses_depth_zero() -> None:
    tree = ASTTree(
        expression_id="expr-leaf",
        root_id="n0",
        nodes=(_leaf_node("n0"),),
        statistics=ASTStatistics(
            node_count=1,
            edge_count=0,
            leaf_count=1,
            operator_count=0,
            depth=0,
        ),
    )

    assert tree.statistics.depth == 0


def test_valid_binary_operator_tree() -> None:
    tree = _binary_tree()

    assert [edge.child_slot for edge in tree.edges] == [0, 1]


def test_ast_json_round_trip() -> None:
    tree = _binary_tree()
    payload = tree.model_dump(mode="json")

    restored = ASTTree.model_validate(payload)

    assert restored == tree
    assert json.loads(json.dumps(payload)) == payload


def test_ast_rejects_duplicate_node_ids() -> None:
    with pytest.raises(ValidationError):
        ASTTree(
            expression_id="expr-duplicate",
            root_id="n0",
            nodes=(_leaf_node("n0"), _leaf_node("n0")),
            statistics=ASTStatistics(
                node_count=2,
                edge_count=0,
                leaf_count=2,
                operator_count=0,
                depth=0,
            ),
        )


def test_ast_rejects_missing_root() -> None:
    with pytest.raises(ValidationError):
        ASTTree(
            expression_id="expr-missing-root",
            root_id="missing",
            nodes=(_leaf_node("n0"),),
            statistics=ASTStatistics(
                node_count=1,
                edge_count=0,
                leaf_count=1,
                operator_count=0,
                depth=0,
            ),
        )


def test_ast_rejects_missing_edge_endpoint() -> None:
    with pytest.raises(ValidationError):
        ASTTree(
            expression_id="expr-missing-endpoint",
            root_id="n0",
            nodes=(ASTNode(node_id="n0", node_kind="operator", label="Neg", arity=1),),
            edges=(ASTEdge(source_id="n0", target_id="missing", child_slot=0),),
            statistics=ASTStatistics(
                node_count=1,
                edge_count=1,
                leaf_count=1,
                operator_count=0,
                depth=0,
            ),
        )


def test_ast_rejects_duplicate_child_slot() -> None:
    with pytest.raises(ValidationError):
        ASTTree(
            expression_id="expr-duplicate-slot",
            root_id="n0",
            nodes=(
                ASTNode(node_id="n0", node_kind="operator", label="Add", arity=2),
                _leaf_node("n1", "x"),
                _leaf_node("n2", "y"),
            ),
            edges=(
                ASTEdge(source_id="n0", target_id="n1", child_slot=0),
                ASTEdge(source_id="n0", target_id="n2", child_slot=0),
            ),
            statistics=ASTStatistics(
                node_count=3,
                edge_count=2,
                leaf_count=2,
                operator_count=1,
                depth=1,
            ),
        )


def test_ast_rejects_invalid_child_slot() -> None:
    with pytest.raises(ValidationError):
        ASTEdge(source_id="n0", target_id="n1", child_slot=2)


def test_ast_rejects_invalid_statistics() -> None:
    with pytest.raises(ValidationError):
        ASTStatistics(
            node_count=2,
            edge_count=0,
            leaf_count=1,
            operator_count=0,
            depth=0,
        )


def test_ast_statistics_reject_impossible_edge_count() -> None:
    with pytest.raises(ValidationError, match="edge_count must equal"):
        ASTStatistics(
            node_count=1,
            edge_count=99,
            leaf_count=1,
            operator_count=0,
            depth=0,
        )


def test_ast_statistics_reject_too_many_binary_leaves() -> None:
    with pytest.raises(ValidationError, match="leaf_count cannot exceed"):
        ASTStatistics(
            node_count=4,
            edge_count=3,
            leaf_count=3,
            operator_count=1,
            depth=1,
        )


def test_ast_statistics_reject_depth_exceeding_operator_count() -> None:
    with pytest.raises(ValidationError, match="depth cannot exceed"):
        ASTStatistics(
            node_count=3,
            edge_count=2,
            leaf_count=1,
            operator_count=2,
            depth=3,
        )


def test_ast_statistics_reject_depth_below_binary_capacity() -> None:
    with pytest.raises(ValidationError, match="depth is too small"):
        ASTStatistics(
            node_count=7,
            edge_count=6,
            leaf_count=4,
            operator_count=3,
            depth=1,
        )


def test_leaf_only_statistics_reject_nonzero_depth() -> None:
    with pytest.raises(ValidationError):
        ASTStatistics(
            node_count=1,
            edge_count=0,
            leaf_count=1,
            operator_count=0,
            depth=1,
        )


def test_ast_rejects_boolean_integer_fields() -> None:
    with pytest.raises(ValidationError):
        ASTNode(node_id="n0", node_kind="operator", label="Neg", arity=True)


def test_ast_rejects_disconnected_cycle() -> None:
    with pytest.raises(ValidationError, match="reachable from the root"):
        ASTTree(
            expression_id="expr-cycle",
            root_id="root",
            nodes=(
                _leaf_node("root"),
                ASTNode(node_id="n1", node_kind="operator", label="Neg", arity=1),
                ASTNode(node_id="n2", node_kind="operator", label="Neg", arity=1),
            ),
            edges=(
                ASTEdge(source_id="n1", target_id="n2", child_slot=0),
                ASTEdge(source_id="n2", target_id="n1", child_slot=0),
            ),
            statistics=ASTStatistics(
                node_count=3,
                edge_count=2,
                leaf_count=1,
                operator_count=2,
                depth=2,
            ),
        )


def test_ast_rejects_incorrect_structural_depth() -> None:
    with pytest.raises(ValidationError, match="depth does not match"):
        ASTTree(
            expression_id="expr-depth",
            root_id="n0",
            nodes=(
                ASTNode(node_id="n0", node_kind="operator", label="Neg", arity=1),
                ASTNode(node_id="n1", node_kind="operator", label="Neg", arity=1),
                ASTNode(node_id="n2", node_kind="operator", label="Add", arity=2),
                _leaf_node("n3", "x"),
                _leaf_node("n4", "y"),
            ),
            edges=(
                ASTEdge(source_id="n0", target_id="n1", child_slot=0),
                ASTEdge(source_id="n1", target_id="n2", child_slot=0),
                ASTEdge(source_id="n2", target_id="n3", child_slot=0),
                ASTEdge(source_id="n2", target_id="n4", child_slot=1),
            ),
            statistics=ASTStatistics(
                node_count=5,
                edge_count=4,
                leaf_count=2,
                operator_count=3,
                depth=2,
            ),
        )
