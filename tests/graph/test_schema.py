"""Tests for the representation-neutral graph contract."""

from __future__ import annotations

import pytest

from geml.graph.schema import (
    AST_FAMILY,
    EML_FAMILY,
    EML_ONE_KIND,
    EML_OPERATOR_KIND,
    EML_VARIABLE_KIND,
    ChildRef,
    Graph,
    GraphNode,
    GraphRoot,
    compute_statistics,
)
from geml.graph.signatures import compute_signature
from geml.graph.validate import validate_graph


def _ast_leaf(node_id: str, value: str | int) -> GraphNode:
    label = "symbol" if isinstance(value, str) else "integer"
    return GraphNode(node_id, AST_FAMILY, "leaf", label, value)


def _add_xy() -> Graph:
    nodes = {
        "root": GraphNode(
            "root",
            AST_FAMILY,
            "operator",
            "add",
            children=(ChildRef(0, "x"), ChildRef(1, "y")),
        ),
        "x": _ast_leaf("x", "x"),
        "y": _ast_leaf("y", "y"),
    }
    return Graph(nodes, (GraphRoot("expression", "root", "ast"),))


def _shared_add() -> Graph:
    """Represent ``(x + 1) * (x + 1)`` with one shared Add node."""

    nodes = {
        "multiply": GraphNode(
            "multiply",
            AST_FAMILY,
            "operator",
            "multiply",
            children=(ChildRef(0, "add"), ChildRef(1, "add")),
        ),
        "add": GraphNode(
            "add",
            AST_FAMILY,
            "operator",
            "add",
            children=(ChildRef(0, "x"), ChildRef(1, "one")),
        ),
        "x": _ast_leaf("x", "x"),
        "one": _ast_leaf("one", 1),
    }
    return Graph(nodes, (GraphRoot("expression", "multiply", "ast"),))


def test_valid_graph_statistics_count_duplicate_references() -> None:
    graph = _shared_add()

    assert validate_graph(graph).valid
    assert len(graph.nodes["multiply"].children) == 2
    assert graph.nodes["multiply"].children[0].target_id == "add"
    assert graph.nodes["multiply"].children[1].target_id == "add"

    statistics = compute_statistics(graph)
    assert statistics.node_count == 4
    assert statistics.edge_count == 4
    assert statistics.child_reference_count == 4
    assert statistics.leaf_count == 2
    assert statistics.root_count == 1
    assert statistics.max_depth == 2


def test_graph_snapshots_mutable_inputs() -> None:
    nodes = {"x": _ast_leaf("x", "x")}
    graph = Graph(nodes, (GraphRoot("expression", "x", "ast"),))
    nodes["y"] = _ast_leaf("y", "y")

    assert tuple(graph.nodes) == ("x",)
    with pytest.raises(TypeError):
        graph.nodes["y"] = _ast_leaf("y", "y")  # type: ignore[index]


def test_identical_subtrees_have_identical_signatures() -> None:
    nodes = {
        "a": GraphNode(
            "a",
            AST_FAMILY,
            "operator",
            "add",
            children=(ChildRef(0, "x1"), ChildRef(1, "one1")),
        ),
        "x1": _ast_leaf("x1", "x"),
        "one1": _ast_leaf("one1", 1),
        "b": GraphNode(
            "b",
            AST_FAMILY,
            "operator",
            "add",
            children=(ChildRef(0, "x2"), ChildRef(1, "one2")),
        ),
        "x2": _ast_leaf("x2", "x"),
        "one2": _ast_leaf("one2", 1),
    }
    graph = Graph(
        nodes,
        (
            GraphRoot("expression-a", "a", "ast"),
            GraphRoot("expression-b", "b", "ast"),
        ),
    )

    signature_a = compute_signature(graph, "a")
    signature_b = compute_signature(graph, "b")
    assert signature_a == signature_b
    assert len(signature_a) == 64
    assert signature_a == signature_a.lower()


def test_ordered_slots_change_a_signature() -> None:
    nodes = {
        "xy": GraphNode(
            "xy",
            AST_FAMILY,
            "operator",
            "power",
            children=(ChildRef(0, "x"), ChildRef(1, "two")),
        ),
        "yx": GraphNode(
            "yx",
            AST_FAMILY,
            "operator",
            "power",
            # Tuple order is irrelevant; the explicit slots are authoritative.
            children=(ChildRef(1, "x"), ChildRef(0, "two")),
        ),
        "x": _ast_leaf("x", "x"),
        "two": _ast_leaf("two", 2),
    }
    graph = Graph(
        nodes,
        (
            GraphRoot("expression-xy", "xy", "ast"),
            GraphRoot("expression-yx", "yx", "ast"),
        ),
    )

    assert compute_signature(graph, "xy") != compute_signature(graph, "yx")


@pytest.mark.parametrize(
    ("changed_node", "family", "kind", "label", "value"),
    [
        ("family", EML_FAMILY, "leaf", "integer", 1),
        ("kind", AST_FAMILY, "constant", "integer", 1),
        ("label", AST_FAMILY, "leaf", "one", 1),
        ("value_type", AST_FAMILY, "leaf", "integer", "1"),
    ],
)
def test_every_canonical_header_field_affects_signature(
    changed_node: str,
    family: str,
    kind: str,
    label: str,
    value: str | int,
) -> None:
    del changed_node
    root = (GraphRoot("expression", "n", family),)
    baseline = Graph({"n": _ast_leaf("n", 1)}, root)
    changed = Graph({"n": GraphNode("n", family, kind, label, value)}, root)

    assert compute_signature(baseline, "n") != compute_signature(changed, "n")


def test_structural_identity_is_not_semantic_equivalence() -> None:
    add = Graph(
        {
            "root": GraphNode(
                "root",
                AST_FAMILY,
                "operator",
                "add",
                children=(ChildRef(0, "x1"), ChildRef(1, "x2")),
            ),
            "x1": _ast_leaf("x1", "x"),
            "x2": _ast_leaf("x2", "x"),
        },
        (GraphRoot("expression", "root", "ast"),),
    )
    multiply = Graph(
        {
            "root": GraphNode(
                "root",
                AST_FAMILY,
                "operator",
                "multiply",
                children=(ChildRef(0, "two"), ChildRef(1, "x")),
            ),
            "two": _ast_leaf("two", 2),
            "x": _ast_leaf("x", "x"),
        },
        (GraphRoot("expression", "root", "ast"),),
    )

    assert compute_signature(add, "root") != compute_signature(multiply, "root")


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (
            Graph(
                {"x": _ast_leaf("x", "x")},
                (GraphRoot("expression", "missing", "ast"),),
            ),
            "does not exist",
        ),
        (
            Graph(
                {
                    "root": GraphNode(
                        "root",
                        AST_FAMILY,
                        "operator",
                        "add",
                        children=(ChildRef(0, "x"), ChildRef(0, "y")),
                    ),
                    "x": _ast_leaf("x", "x"),
                    "y": _ast_leaf("y", "y"),
                },
                (GraphRoot("expression", "root", "ast"),),
            ),
            "duplicate child slot",
        ),
        (
            Graph(
                {
                    "root": GraphNode(
                        "root",
                        AST_FAMILY,
                        "operator",
                        "negate",
                        children=(ChildRef(1, "x"),),
                    ),
                    "x": _ast_leaf("x", "x"),
                },
                (GraphRoot("expression", "root", "ast"),),
            ),
            "contiguous from zero",
        ),
        (
            Graph(
                {
                    "a": GraphNode(
                        "a",
                        AST_FAMILY,
                        "operator",
                        "negate",
                        children=(ChildRef(0, "b"),),
                    ),
                    "b": GraphNode(
                        "b",
                        AST_FAMILY,
                        "operator",
                        "negate",
                        children=(ChildRef(0, "a"),),
                    ),
                },
                (GraphRoot("expression", "a", "ast"),),
            ),
            "cycle detected",
        ),
        (
            Graph(
                {
                    "root": _ast_leaf("root", "x"),
                    "orphan": _ast_leaf("orphan", "y"),
                },
                (GraphRoot("expression", "root", "ast"),),
            ),
            "unreachable",
        ),
    ],
)
def test_structural_validation_retains_failures(graph: Graph, message: str) -> None:
    result = validate_graph(graph)

    assert not result.valid
    assert any(message in error for error in result.errors)
    with pytest.raises(ValueError, match="invalid graph"):
        compute_statistics(graph)


def test_purity_rejects_mixed_representation_families() -> None:
    graph = Graph(
        {
            "root": GraphNode(
                "root",
                AST_FAMILY,
                "operator",
                "negate",
                children=(ChildRef(0, "x"),),
            ),
            "x": GraphNode("x", EML_FAMILY, EML_VARIABLE_KIND, "x", "x"),
        },
        (GraphRoot("expression", "root", "ast"),),
    )

    assert any("same representation family" in error for error in validate_graph(graph).errors)


def test_binary_ast_purity_rejects_arity_above_two() -> None:
    graph = Graph(
        {
            "root": GraphNode(
                "root",
                AST_FAMILY,
                "operator",
                "add",
                children=(
                    ChildRef(0, "x"),
                    ChildRef(1, "y"),
                    ChildRef(2, "z"),
                ),
            ),
            "x": _ast_leaf("x", "x"),
            "y": _ast_leaf("y", "y"),
            "z": _ast_leaf("z", "z"),
        },
        (GraphRoot("expression", "root", "ast"),),
    )

    assert any("binary AST arity" in error for error in validate_graph(graph).errors)


def test_exact_pure_eml_shapes_validate() -> None:
    graph = Graph(
        {
            "root": GraphNode(
                "root",
                EML_FAMILY,
                EML_OPERATOR_KIND,
                "eml",
                children=(ChildRef(0, "x"), ChildRef(1, "one")),
            ),
            "x": GraphNode("x", EML_FAMILY, EML_VARIABLE_KIND, "x", "x"),
            "one": GraphNode("one", EML_FAMILY, EML_ONE_KIND, "1", 1),
        },
        (GraphRoot("expression", "root", "pure_eml:official_v4"),),
    )

    assert validate_graph(graph).valid


@pytest.mark.parametrize(
    "invalid_leaf",
    [
        GraphNode("leaf", EML_FAMILY, EML_ONE_KIND, "1", 2),
        GraphNode("leaf", EML_FAMILY, EML_VARIABLE_KIND, "x+y", "x+y"),
        GraphNode("leaf", EML_FAMILY, "constant", "1", 1),
    ],
)
def test_pure_eml_rejects_hidden_or_nonprimitive_leaves(
    invalid_leaf: GraphNode,
) -> None:
    graph = Graph(
        {"leaf": invalid_leaf},
        (GraphRoot("expression", "leaf", "pure_eml:official_v4"),),
    )

    assert not validate_graph(graph).valid


@pytest.mark.parametrize("family", ["macro", "motif"])
def test_schema_remains_neutral_for_future_graph_families(family: str) -> None:
    graph = Graph(
        {
            "root": GraphNode(
                "root",
                family,
                "template",
                "future-representation",
                children=(ChildRef(0, "x"),),
            ),
            "x": GraphNode("x", family, "leaf", "x", "x"),
        },
        (GraphRoot("expression", "root", family),),
    )

    assert validate_graph(graph).valid
    assert compute_signature(graph, "root")
