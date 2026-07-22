"""tests/graph/test_schema.py - owned by 3-1. tiny fixtures only."""
from geml.graph.schema import Graph, GraphNode, ChildRef, compute_statistics
from geml.graph.signatures import compute_signature
from geml.graph.validate import validate_graph


def _add_xy():
    """x + y, a tiny plain AST graph."""
    nodes = {
        "root": GraphNode("root", family="ast", kind="Add", children=(ChildRef(0, "x"), ChildRef(1, "y"))),
        "x": GraphNode("x", family="ast", kind="Var", value="x"),
        "y": GraphNode("y", family="ast", kind="Var", value="y"),
    }
    return Graph(nodes=nodes, roots=("root",))


# ---------------------------------------------------------------------
# basic validity + stats
# ---------------------------------------------------------------------

def test_simple_graph_is_valid():
    g = _add_xy()
    result = validate_graph(g)
    assert result.valid
    assert result.errors == []


def test_stats_on_simple_graph():
    g = _add_xy()
    stats = compute_statistics(g)
    assert stats.node_count == 3
    assert stats.edge_count == 2
    assert stats.leaf_count == 2
    assert stats.root_count == 1
    assert stats.max_depth == 1


# ---------------------------------------------------------------------
# duplicate child references (the DAG-sharing case)
# ---------------------------------------------------------------------

def _mul_shared_add():
    """(x+1)*(x+1) as a DAG - Mul has two children, both pointing at the same Add node."""
    nodes = {
        "mul": GraphNode("mul", family="ast", kind="Mul", children=(ChildRef(0, "add"), ChildRef(1, "add"))),
        "add": GraphNode("add", family="ast", kind="Add", children=(ChildRef(0, "x"), ChildRef(1, "one"))),
        "x": GraphNode("x", family="ast", kind="Var", value="x"),
        "one": GraphNode("one", family="ast", kind="Const", value=1),
    }
    return Graph(nodes=nodes, roots=("mul",))


def test_duplicate_child_refs_stay_explicit():
    g = _mul_shared_add()
    mul_node = g.nodes["mul"]
    assert len(mul_node.children) == 2  # both refs present, not collapsed into one
    assert mul_node.children[0].target_id == mul_node.children[1].target_id == "add"


def test_shared_node_counted_once_in_stats():
    # DAG-sharing case: 4 unique nodes (mul, add, x, one), but 4 total
    # edges since 'add' is referenced twice - edge_count can exceed
    # node_count-1, which is the whole point of sharing vs a plain tree
    g = _mul_shared_add()
    stats = compute_statistics(g)
    assert stats.node_count == 4
    assert stats.edge_count == 4


def test_shared_node_graph_is_valid():
    g = _mul_shared_add()
    assert validate_graph(g).valid


# ---------------------------------------------------------------------
# signatures: structural equality
# ---------------------------------------------------------------------

def test_identical_subtrees_have_identical_signatures():
    """acceptance criterion: identical subtrees -> identical signatures."""
    nodes = {
        "a": GraphNode("a", family="ast", kind="Add", children=(ChildRef(0, "x1"), ChildRef(1, "one1"))),
        "x1": GraphNode("x1", family="ast", kind="Var", value="x"),
        "one1": GraphNode("one1", family="ast", kind="Const", value=1),
        "b": GraphNode("b", family="ast", kind="Add", children=(ChildRef(0, "x2"), ChildRef(1, "one2"))),
        "x2": GraphNode("x2", family="ast", kind="Var", value="x"),
        "one2": GraphNode("one2", family="ast", kind="Const", value=1),
    }
    g = Graph(nodes=nodes, roots=("a", "b"))
    assert compute_signature(g, "a") == compute_signature(g, "b")


def test_non_identical_ordered_subtrees_differ():
    """acceptance criterion: non-identical ordered subtrees -> different signatures."""
    nodes = {
        "pow_x2": GraphNode("pow_x2", family="ast", kind="Pow", children=(ChildRef(0, "x"), ChildRef(1, "two"))),
        "pow_2x": GraphNode("pow_2x", family="ast", kind="Pow", children=(ChildRef(0, "two"), ChildRef(1, "x"))),
        "x": GraphNode("x", family="ast", kind="Var", value="x"),
        "two": GraphNode("two", family="ast", kind="Const", value=2),
    }
    g = Graph(nodes=nodes, roots=("pow_x2", "pow_2x"))
    assert compute_signature(g, "pow_x2") != compute_signature(g, "pow_2x")


def test_structural_vs_semantic_equality():
    """
    fixture distinguishing structural equality (same shape) from
    semantic equality (same math, different shape). the schema only
    knows about structure - two graphs that mean the same thing but
    are built differently must NOT get the same signature. proving
    they're actually equal mathematically is goal 4's job (e-graphs),
    not this schema's.
    """
    # x + x  (structurally: Add(x, x))
    add_xx_nodes = {
        "root": GraphNode("root", family="ast", kind="Add", children=(ChildRef(0, "x1"), ChildRef(1, "x2"))),
        "x1": GraphNode("x1", family="ast", kind="Var", value="x"),
        "x2": GraphNode("x2", family="ast", kind="Var", value="x"),
    }
    g1 = Graph(nodes=add_xx_nodes, roots=("root",))

    # 2 * x  (structurally: Mul(2, x)) - mathematically equal to x+x, but a different shape
    mul_2x_nodes = {
        "root": GraphNode("root", family="ast", kind="Mul", children=(ChildRef(0, "two"), ChildRef(1, "x"))),
        "two": GraphNode("two", family="ast", kind="Const", value=2),
        "x": GraphNode("x", family="ast", kind="Var", value="x"),
    }
    g2 = Graph(nodes=mul_2x_nodes, roots=("root",))

    # different structure -> different signature, even though the maths matches
    assert compute_signature(g1, "root") != compute_signature(g2, "root")


# ---------------------------------------------------------------------
# validation failure modes
# ---------------------------------------------------------------------

def test_cycle_is_rejected():
    nodes = {
        "a": GraphNode("a", family="ast", kind="Add", children=(ChildRef(0, "b"),)),
        "b": GraphNode("b", family="ast", kind="Add", children=(ChildRef(0, "a"),)),
    }
    g = Graph(nodes=nodes, roots=("a",))
    result = validate_graph(g)
    assert not result.valid
    assert any("cycle" in e for e in result.errors)


def test_missing_root_is_rejected():
    nodes = {"a": GraphNode("a", family="ast", kind="Var", value="x")}
    g = Graph(nodes=nodes, roots=("nonexistent",))
    result = validate_graph(g)
    assert not result.valid
    assert any("does not exist" in e for e in result.errors)


def test_duplicate_slot_is_rejected():
    """two DIFFERENT children can't both claim the same slot number."""
    nodes = {
        "root": GraphNode("root", family="ast", kind="Add", children=(ChildRef(0, "x"), ChildRef(0, "y"))),
        "x": GraphNode("x", family="ast", kind="Var", value="x"),
        "y": GraphNode("y", family="ast", kind="Var", value="y"),
    }
    g = Graph(nodes=nodes, roots=("root",))
    result = validate_graph(g)
    assert not result.valid
    assert any("claiming slot" in e for e in result.errors)


def test_purity_violation_is_rejected():
    """sin isn't in the (provisional) AST vocabulary."""
    nodes = {
        "root": GraphNode("root", family="ast", kind="sin", children=(ChildRef(0, "x"),)),
        "x": GraphNode("x", family="ast", kind="Var", value="x"),
    }
    g = Graph(nodes=nodes, roots=("root",))
    result = validate_graph(g)
    assert not result.valid
    assert any("not in the approved" in e for e in result.errors)


def test_unreachable_node_is_rejected():
    nodes = {
        "root": GraphNode("root", family="ast", kind="Var", value="x"),
        "orphan": GraphNode("orphan", family="ast", kind="Var", value="y"),
    }
    g = Graph(nodes=nodes, roots=("root",))
    result = validate_graph(g)
    assert not result.valid
    assert any("unreachable" in e for e in result.errors)


def test_eml_family_purity():
    """eml family should reject AST-style operators like Add - pure EML
    trees may only contain eml nodes and leaves."""
    nodes = {
        "root": GraphNode("root", family="eml", kind="Add", children=(ChildRef(0, "x"), ChildRef(1, "one"))),
        "x": GraphNode("x", family="eml", kind="Var", value="x"),
        "one": GraphNode("one", family="eml", kind="Const", value=1),
    }
    g = Graph(nodes=nodes, roots=("root",))
    result = validate_graph(g)
    assert not result.valid
    assert any("eml" in e for e in result.errors)


def test_macro_family_is_valid_and_neutral():
    """
    macro nodes (compiler-generated shorthand like eml_add) aren't
    checked against a fixed vocabulary - there isn't a closed list the
    way ast/eml have one. this proves a macro graph validates cleanly
    and gets a real signature, not just "assumed to work" because no
    test ever built one.
    """
    nodes = {
        "root": GraphNode("root", family="macro", kind="eml_add", children=(ChildRef(0, "x"), ChildRef(1, "one"))),
        "x": GraphNode("x", family="macro", kind="Var", value="x"),
        "one": GraphNode("one", family="macro", kind="Const", value=1),
    }
    g = Graph(nodes=nodes, roots=("root",))
    result = validate_graph(g)
    assert result.valid
    assert result.errors == []
    sig = compute_signature(g, "root")
    assert sig  # non-empty, actually computed, not skipped


def test_motif_family_is_valid_and_neutral():
    """same as above but for motif family - a compressed, reused pattern
    node with no fixed vocabulary either."""
    nodes = {
        "root": GraphNode("root", family="motif", kind="motif_17", label="common_add_one", children=(ChildRef(0, "x"),)),
        "x": GraphNode("x", family="motif", kind="Var", value="x"),
    }
    g = Graph(nodes=nodes, roots=("root",))
    result = validate_graph(g)
    assert result.valid
    assert result.errors == []
    sig = compute_signature(g, "root")
    assert "motif" in sig
    assert "common_add_one" in sig  # label shows up in the signature too
