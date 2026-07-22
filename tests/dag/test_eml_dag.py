"""
tests/dag/test_eml_dag.py - owned by 3-3

update: found the real paper (arXiv:2603.21852v2) and confirmed exp(x)
and ln(z) are both real, verified eml constructions now (see make_exp/
make_ln in eml.py). x+y, x*y, and powers are still placeholders -
paper mentions multiplication needs depth 8 but doesn't give the tree
in the sections fetched so far, asked Sahil about it
"""
import math
from geml.dag.eml import (
    EmlNode, eml_to_dag, convert_with_stats,
    validate_eml_purity, evaluate_tree, evaluate_dag,
    make_exp, make_ln,
)
from geml.graph.schema import Graph, GraphNode, ChildRef
from geml.graph.validate import validate_graph


def _var(name):
    return EmlNode("Var", value=name)


def _one():
    return EmlNode("Const", value=1)


def _eml(a, b):
    return EmlNode("eml", (a, b))


def test_exp_construction_is_verified_correct():
    # exp(x) = eml(x, 1), arXiv:2603.21852v2 eq 3 - actually verified, not a guess
    tree = make_exp(_var("x"))
    val = evaluate_tree(tree, {"x": 2.0})
    assert abs(val - math.exp(2.0)) < 1e-9


def test_ln_construction_is_verified_correct():
    # ln(z) = eml(1, eml(eml(1,z), 1)), arXiv:2603.21852v2 eq 5 - actually verified
    tree = make_ln(_var("z"))
    val = evaluate_tree(tree, {"z": 3.5})
    assert abs(val - math.log(3.5)) < 1e-9


# audit case placeholders - still arbitrary shapes, not verified real
# compilations, for x+y/x*y/powers specifically

def test_audit_case_add_placeholder_dag_valid():
    tree = _eml(_var("x"), _eml(_var("y"), _one()))
    graph = eml_to_dag(tree)
    assert validate_graph(graph).valid
    assert validate_eml_purity(graph).valid


def test_audit_case_mul_placeholder_dag_valid():
    tree = _eml(_eml(_var("x"), _var("y")), _one())
    graph = eml_to_dag(tree)
    assert validate_graph(graph).valid
    assert validate_eml_purity(graph).valid


def test_audit_case_power_placeholder_dag_valid():
    tree = _eml(_eml(_var("x"), _one()), _eml(_var("x"), _one()))
    graph = eml_to_dag(tree)
    assert validate_graph(graph).valid
    assert validate_eml_purity(graph).valid


# repeated source subexpressions - the actual sharing behavior

def test_repeated_subexpression_shared():
    # same exp(x)-shaped subtree built twice independently, should
    # collapse to one shared node
    def make_exp_x():
        return _eml(_var("x"), _one())

    outer = _eml(make_exp_x(), make_exp_x())
    graph, stats = convert_with_stats(outer)
    assert stats.tree_node_count == 7
    assert stats.dag_node_count == 4
    assert stats.compression_ratio == 7 / 4


def test_shared_dag_is_valid_and_pure():
    def make_exp_x():
        return _eml(_var("x"), _one())
    outer = _eml(make_exp_x(), make_exp_x())
    graph = eml_to_dag(outer)
    assert validate_graph(graph).valid
    assert validate_eml_purity(graph).valid


def test_duplicate_refs_preserved_after_sharing():
    def make_exp_x():
        return _eml(_var("x"), _one())
    outer = _eml(make_exp_x(), make_exp_x())
    graph = eml_to_dag(outer)
    root_node = graph.nodes[graph.roots[0]]
    assert len(root_node.children) == 2
    assert root_node.children[0].target_id == root_node.children[1].target_id


# dag size can never exceed tree size

def test_dag_never_exceeds_tree_size():
    def make_exp_x():
        return _eml(_var("x"), _one())

    cases = [
        _eml(_var("x"), _one()),           # no duplication
        _eml(make_exp_x(), make_exp_x()),  # heavy duplication
        _eml(_var("x"), _var("y")),        # different leaves, nothing to share
    ]
    for case in cases:
        _, stats = convert_with_stats(case)
        assert stats.dag_node_count <= stats.tree_node_count


# no derived/macro nodes should ever sneak in

def test_no_derived_nodes_appear():
    def make_exp_x():
        return _eml(_var("x"), _one())
    graph = eml_to_dag(_eml(make_exp_x(), make_exp_x()))
    for node in graph.nodes.values():
        assert node.kind in ("eml", "Var", "Const")


def test_purity_rejects_non_one_constant():
    # pure eml leaves can only be the constant 1
    nodes = {
        "root": GraphNode("root", family="eml", kind="eml", children=(ChildRef(0, "x"), ChildRef(1, "five"))),
        "x": GraphNode("x", family="eml", kind="Var", value="x"),
        "five": GraphNode("five", family="eml", kind="Const", value=5),
    }
    g = Graph(nodes=nodes, roots=("root",))
    result = validate_eml_purity(g)
    assert not result.valid
    assert any("can only be 1" in e for e in result.errors)


def test_purity_rejects_non_eml_operator():
    # an ast-style operator like Add should never show up in a pure eml graph
    nodes = {
        "root": GraphNode("root", family="eml", kind="Add", children=(ChildRef(0, "x"), ChildRef(1, "one"))),
        "x": GraphNode("x", family="eml", kind="Var", value="x"),
        "one": GraphNode("one", family="eml", kind="Const", value=1),
    }
    g = Graph(nodes=nodes, roots=("root",))
    result = validate_eml_purity(g)
    assert not result.valid
    assert any("not eml/Var/Const" in e for e in result.errors)


# evaluation must match pre- and post-sharing

def test_all_audit_cases_evaluate_same_pre_and_post_sharing():
    bindings = {"x": 1.7, "y": 0.9}
    cases = [
        make_exp(_var("x")),
        make_ln(_var("x")),
        _eml(_var("x"), _eml(_var("y"), _one())),
        _eml(_eml(_var("x"), _var("y")), _one()),
    ]
    for tree in cases:
        graph = eml_to_dag(tree)
        tree_val = evaluate_tree(tree, bindings)
        dag_val = evaluate_dag(graph, graph.roots[0], bindings)
        assert abs(tree_val - dag_val) < 1e-9


def test_ln_of_exp_is_identity():
    """composing both real constructions: ln(exp(x)) should equal x -
    a genuine correctness check, not just an isolated formula check."""
    tree = make_ln(make_exp(_var("x")))
    val = evaluate_tree(tree, {"x": 4.2})
    assert abs(val - 4.2) < 1e-9
