"""tests/dag/test_ast_dag.py - owned by 3-2. tiny fixtures only."""
from geml.dag.ast import AstNode, ast_to_dag, convert_with_stats
from geml.graph.signatures import compute_signature
from geml.graph.validate import validate_graph


def _add_x1():
    """builds a fresh, independent Add(x, 1) subtree each call - as if
    generated separately, not sharing python objects"""
    return AstNode("Add", (AstNode("Var", value="x"), AstNode("Const", value=1)))


# ---------------------------------------------------------------------
# the exact acceptance-criterion example
# ---------------------------------------------------------------------

def test_repeated_subtree_shared():
    """(x+1)*(x+1) - the two independently-built Add(x,1) subtrees
    should collapse into one shared node."""
    mul = AstNode("Mul", (_add_x1(), _add_x1()))
    graph, stats = convert_with_stats(mul)

    assert stats.ast_node_count == 7   # Mul, Add, x, 1, Add, x, 1 (naive tree)
    assert stats.dag_node_count == 4   # Mul, Add, x, 1 (shared)
    assert stats.compression_ratio == 7 / 4


def test_shared_dag_is_valid():
    mul = AstNode("Mul", (_add_x1(), _add_x1()))
    graph = ast_to_dag(mul)
    assert validate_graph(graph).valid


def test_duplicate_refs_preserved_after_sharing():
    """mul's two children still both explicitly reference the shared
    add node - not collapsed into a single reference."""
    mul = AstNode("Mul", (_add_x1(), _add_x1()))
    graph = ast_to_dag(mul)
    mul_node = graph.nodes[graph.roots[0]]
    assert len(mul_node.children) == 2
    assert mul_node.children[0].target_id == mul_node.children[1].target_id


# ---------------------------------------------------------------------
# repeated leaves
# ---------------------------------------------------------------------

def test_repeated_leaf_shared():
    """x+x - both x leaves are the same variable, should share one node."""
    x = AstNode("Var", value="x")
    add_xx = AstNode("Add", (x, x))
    graph = ast_to_dag(add_xx)
    assert len(graph.nodes) == 2  # Add + one shared x


def test_different_leaves_not_shared():
    """x+y - different variables, must NOT be merged just because
    they're both leaves."""
    x = AstNode("Var", value="x")
    y = AstNode("Var", value="y")
    add_xy = AstNode("Add", (x, y))
    graph = ast_to_dag(add_xy)
    assert len(graph.nodes) == 3  # Add, x, y all separate


# ---------------------------------------------------------------------
# negative sharing - the "don't over-merge" tests
# ---------------------------------------------------------------------

def test_commutative_not_merged():
    """x*y and y*x are mathematically equal but structurally different
    (child order differs) - must get different signatures, never shared."""
    x = AstNode("Var", value="x")
    y = AstNode("Var", value="y")
    g1 = ast_to_dag(AstNode("Mul", (x, y)))
    g2 = ast_to_dag(AstNode("Mul", (y, x)))
    sig1 = compute_signature(g1, g1.roots[0])
    sig2 = compute_signature(g2, g2.roots[0])
    assert sig1 != sig2


def test_semantically_equal_but_structurally_different_not_merged():
    """x*x and x**2 are mathematically equal but built from different
    operators entirely (Mul vs Pow) - must never be merged."""
    x = AstNode("Var", value="x")
    g1 = ast_to_dag(AstNode("Mul", (x, x)))
    g2 = ast_to_dag(AstNode("Pow", (x, AstNode("Const", value=2))))
    sig1 = compute_signature(g1, g1.roots[0])
    sig2 = compute_signature(g2, g2.roots[0])
    assert sig1 != sig2


# ---------------------------------------------------------------------
# acceptance criterion: DAG size never exceeds tree size
# ---------------------------------------------------------------------

def test_dag_never_exceeds_tree_size_no_duplication():
    """x+y has no repeated structure - DAG size should equal tree size exactly."""
    x = AstNode("Var", value="x")
    y = AstNode("Var", value="y")
    _, stats = convert_with_stats(AstNode("Add", (x, y)))
    assert stats.dag_node_count == stats.ast_node_count == 3


def test_dag_never_exceeds_tree_size_with_duplication():
    """(x+1)*(x+1) has heavy duplication - DAG should be strictly smaller."""
    _, stats = convert_with_stats(AstNode("Mul", (_add_x1(), _add_x1())))
    assert stats.dag_node_count < stats.ast_node_count


def test_dag_never_exceeds_tree_size_general():
    """a few varied shapes - dag_node_count <= ast_node_count must always hold."""
    x = AstNode("Var", value="x")
    y = AstNode("Var", value="y")
    two = AstNode("Const", value=2)
    cases = [
        AstNode("Pow", (x, two)),
        AstNode("Add", (AstNode("Mul", (x, y)), AstNode("Mul", (x, y)))),  # repeated Mul(x,y)
        AstNode("Neg", (x,)),
    ]
    for case in cases:
        _, stats = convert_with_stats(case)
        assert stats.dag_node_count <= stats.ast_node_count


# ---------------------------------------------------------------------
# ordered slots survive conversion
# ---------------------------------------------------------------------

def test_pow_base_exponent_order_preserved():
    """x**2 - base (x) must stay in slot 0, exponent (2) in slot 1."""
    x = AstNode("Var", value="x")
    two = AstNode("Const", value=2)
    graph = ast_to_dag(AstNode("Pow", (x, two)))
    root = graph.nodes[graph.roots[0]]
    base_node = graph.nodes[root.children[0].target_id]
    exp_node = graph.nodes[root.children[1].target_id]
    assert base_node.value == "x"
    assert exp_node.value == 2
