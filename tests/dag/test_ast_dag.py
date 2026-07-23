"""Tests for exact AST tree-to-DAG structural sharing."""

from __future__ import annotations

from fractions import Fraction

from geml.contracts.ast import ASTEdge, ASTNode, ASTStatistics, ASTTree
from geml.dag.ast import ast_to_dag, convert_with_stats
from geml.graph.signatures import compute_signature
from geml.graph.validate import validate_graph


def _leaf(node_id: str, label: str, value: str | int) -> ASTNode:
    return ASTNode(
        node_id=node_id,
        node_kind="leaf",
        label=label,
        arity=0,
        value=value,
    )


def _operator(node_id: str, label: str, arity: int = 2) -> ASTNode:
    return ASTNode(
        node_id=node_id,
        node_kind="operator",
        label=label,
        arity=arity,
    )


def _repeated_add_tree() -> ASTTree:
    """Return the source tree for ``(x + 1) * (x + 1)``."""

    return ASTTree(
        expression_id="repeated-add",
        root_id="multiply",
        nodes=(
            _operator("multiply", "multiply"),
            _operator("add-left", "add"),
            _leaf("x-left", "symbol", "x"),
            _leaf("one-left", "integer", 1),
            _operator("add-right", "add"),
            _leaf("x-right", "symbol", "x"),
            _leaf("one-right", "integer", 1),
        ),
        edges=(
            ASTEdge(source_id="multiply", target_id="add-left", child_slot=0),
            ASTEdge(source_id="multiply", target_id="add-right", child_slot=1),
            ASTEdge(source_id="add-left", target_id="x-left", child_slot=0),
            ASTEdge(source_id="add-left", target_id="one-left", child_slot=1),
            ASTEdge(source_id="add-right", target_id="x-right", child_slot=0),
            ASTEdge(source_id="add-right", target_id="one-right", child_slot=1),
        ),
        statistics=ASTStatistics(
            node_count=7,
            edge_count=6,
            leaf_count=4,
            operator_count=3,
            depth=2,
        ),
    )


def test_repeated_subtree_is_shared_exactly() -> None:
    graph, statistics = convert_with_stats(_repeated_add_tree())

    assert validate_graph(graph).valid
    assert statistics.tree_node_count == 7
    assert statistics.dag_node_count == 4
    assert statistics.dag_child_reference_count == 4
    assert statistics.dag_depth == 2
    assert statistics.compression_ratio == Fraction(7, 4)

    root = graph.nodes[graph.roots[0].target_id]
    assert len(root.children) == 2
    assert root.children[0].target_id == root.children[1].target_id


def test_distinct_repeated_leaves_become_duplicate_child_references() -> None:
    tree = ASTTree(
        expression_id="x-plus-x",
        root_id="add",
        nodes=(
            _operator("add", "add"),
            _leaf("x-left", "symbol", "x"),
            _leaf("x-right", "symbol", "x"),
        ),
        edges=(
            ASTEdge(source_id="add", target_id="x-left", child_slot=0),
            ASTEdge(source_id="add", target_id="x-right", child_slot=1),
        ),
        statistics=ASTStatistics(
            node_count=3,
            edge_count=2,
            leaf_count=2,
            operator_count=1,
            depth=1,
        ),
    )

    graph = ast_to_dag(tree)
    root = graph.nodes[graph.roots[0].target_id]
    assert len(graph.nodes) == 2
    assert [child.slot for child in root.children] == [0, 1]
    assert root.children[0].target_id == root.children[1].target_id


def test_commutative_reordering_is_not_shared() -> None:
    tree = ASTTree(
        expression_id="ordered-multiply",
        root_id="add",
        nodes=(
            _operator("add", "add"),
            _operator("xy", "multiply"),
            _leaf("x-left", "symbol", "x"),
            _leaf("y-left", "symbol", "y"),
            _operator("yx", "multiply"),
            _leaf("y-right", "symbol", "y"),
            _leaf("x-right", "symbol", "x"),
        ),
        edges=(
            ASTEdge(source_id="add", target_id="xy", child_slot=0),
            ASTEdge(source_id="add", target_id="yx", child_slot=1),
            ASTEdge(source_id="xy", target_id="x-left", child_slot=0),
            ASTEdge(source_id="xy", target_id="y-left", child_slot=1),
            ASTEdge(source_id="yx", target_id="y-right", child_slot=0),
            ASTEdge(source_id="yx", target_id="x-right", child_slot=1),
        ),
        statistics=ASTStatistics(
            node_count=7,
            edge_count=6,
            leaf_count=4,
            operator_count=3,
            depth=2,
        ),
    )

    graph = ast_to_dag(tree)
    root = graph.nodes[graph.roots[0].target_id]
    assert root.children[0].target_id != root.children[1].target_id


def test_semantically_related_multiply_and_power_are_not_shared() -> None:
    tree = ASTTree(
        expression_id="multiply-versus-power",
        root_id="add",
        nodes=(
            _operator("add", "add"),
            _operator("multiply", "multiply"),
            _leaf("x-a", "symbol", "x"),
            _leaf("x-b", "symbol", "x"),
            _operator("power", "power"),
            _leaf("x-c", "symbol", "x"),
            _leaf("two", "integer", 2),
        ),
        edges=(
            ASTEdge(source_id="add", target_id="multiply", child_slot=0),
            ASTEdge(source_id="add", target_id="power", child_slot=1),
            ASTEdge(source_id="multiply", target_id="x-a", child_slot=0),
            ASTEdge(source_id="multiply", target_id="x-b", child_slot=1),
            ASTEdge(source_id="power", target_id="x-c", child_slot=0),
            ASTEdge(source_id="power", target_id="two", child_slot=1),
        ),
        statistics=ASTStatistics(
            node_count=7,
            edge_count=6,
            leaf_count=4,
            operator_count=3,
            depth=2,
        ),
    )

    graph = ast_to_dag(tree)
    root = graph.nodes[graph.roots[0].target_id]
    assert root.children[0].target_id != root.children[1].target_id


def test_typed_values_do_not_false_share() -> None:
    tree = ASTTree(
        expression_id="typed-values",
        root_id="add",
        nodes=(
            _operator("add", "add"),
            _leaf("integer", "literal", 1),
            _leaf("string", "literal", "1"),
        ),
        edges=(
            ASTEdge(source_id="add", target_id="integer", child_slot=0),
            ASTEdge(source_id="add", target_id="string", child_slot=1),
        ),
        statistics=ASTStatistics(
            node_count=3,
            edge_count=2,
            leaf_count=2,
            operator_count=1,
            depth=1,
        ),
    )

    graph = ast_to_dag(tree)
    root = graph.nodes[graph.roots[0].target_id]
    assert root.children[0].target_id != root.children[1].target_id
    assert len(graph.nodes) == 3


def test_node_and_edge_order_do_not_affect_structural_identity() -> None:
    original = _repeated_add_tree()
    reordered = ASTTree(
        expression_id="different-expression-id",
        root_id=original.root_id,
        nodes=tuple(reversed(original.nodes)),
        edges=tuple(reversed(original.edges)),
        statistics=original.statistics,
    )

    first = ast_to_dag(original)
    second = ast_to_dag(reordered)
    first_root = first.roots[0].target_id
    second_root = second.roots[0].target_id

    assert first_root == second_root
    assert compute_signature(first, first_root) == compute_signature(second, second_root)


def test_source_ids_are_not_part_of_structural_identity() -> None:
    first = ASTTree(
        expression_id="first",
        root_id="root-a",
        nodes=(
            _operator("root-a", "negate", arity=1),
            _leaf("leaf-a", "symbol", "x"),
        ),
        edges=(ASTEdge(source_id="root-a", target_id="leaf-a", child_slot=0),),
        statistics=ASTStatistics(
            node_count=2,
            edge_count=1,
            leaf_count=1,
            operator_count=1,
            depth=1,
        ),
    )
    second = ASTTree(
        expression_id="second",
        root_id="root-b",
        nodes=(
            _operator("root-b", "negate", arity=1),
            _leaf("leaf-b", "symbol", "x"),
        ),
        edges=(ASTEdge(source_id="root-b", target_id="leaf-b", child_slot=0),),
        statistics=first.statistics,
    )

    assert ast_to_dag(first).roots[0].target_id == ast_to_dag(second).roots[0].target_id


def test_label_changes_prevent_sharing() -> None:
    tree = ASTTree(
        expression_id="labels",
        root_id="add",
        nodes=(
            _operator("add", "add"),
            _leaf("first", "integer", 1),
            _leaf("second", "one", 1),
        ),
        edges=(
            ASTEdge(source_id="add", target_id="first", child_slot=0),
            ASTEdge(source_id="add", target_id="second", child_slot=1),
        ),
        statistics=ASTStatistics(
            node_count=3,
            edge_count=2,
            leaf_count=2,
            operator_count=1,
            depth=1,
        ),
    )

    graph = ast_to_dag(tree)
    root = graph.nodes[graph.roots[0].target_id]
    assert root.children[0].target_id != root.children[1].target_id


def test_leaf_tree_has_unit_compression_and_zero_depth() -> None:
    tree = ASTTree(
        expression_id="leaf",
        root_id="x",
        nodes=(_leaf("x", "symbol", "x"),),
        statistics=ASTStatistics(
            node_count=1,
            edge_count=0,
            leaf_count=1,
            operator_count=0,
            depth=0,
        ),
    )

    graph, statistics = convert_with_stats(tree)
    assert len(graph.nodes) == 1
    assert statistics.dag_depth == 0
    assert statistics.compression_ratio == 1


def test_deep_ast_conversion_is_iterative() -> None:
    depth = 1_500
    nodes = [_leaf(f"node-{depth}", "symbol", "x")]
    edges: list[ASTEdge] = []
    for index in reversed(range(depth)):
        nodes.append(_operator(f"node-{index}", "exp", arity=1))
        edges.append(
            ASTEdge(
                source_id=f"node-{index}",
                target_id=f"node-{index + 1}",
                child_slot=0,
            )
        )
    tree = ASTTree(
        expression_id="deep",
        root_id="node-0",
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

    graph, statistics = convert_with_stats(tree)
    assert len(graph.nodes) == depth + 1
    assert statistics.dag_depth == depth
