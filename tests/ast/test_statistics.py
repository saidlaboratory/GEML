"""tests/ast/test_statistics.py - owned by [1-6]."""
import sympy as sp
from geml.parsing.srepr import ExpressionRecord
from geml.ast.builder import to_ast
from geml.ast.statistics import ast_stats, structural_signature


def test_spec_example_add_stats():
    """Spec sec 2.2/2.3: x+y -> 3 nodes, depth 1, 2 edges."""
    x, y = sp.symbols("x y")
    ast = to_ast(ExpressionRecord.from_expr("e1", x + y).parse())
    assert ast_stats(ast) == {
        "node_count": 3, "edge_count": 2, "leaf_count": 2,
        "operator_count": 1, "depth": 1,
    }


def test_leaf_depth_is_zero():
    x = sp.Symbol("x")
    ast = to_ast(ExpressionRecord.from_expr("e2", x).parse())
    assert ast_stats(ast)["depth"] == 0


def test_duplicate_subtree_stats_match_spec():
    """Spec sec 2.2: (x+1)(x+1) -> 7 nodes, depth 2, 6 edges."""
    x = sp.Symbol("x")
    e = sp.Mul(x + 1, x + 1, evaluate=False)
    ast = to_ast(ExpressionRecord.from_expr("e3", e).parse())
    stats = ast_stats(ast)
    assert stats["node_count"] == 7
    assert stats["depth"] == 2
    assert stats["edge_count"] == 6


def test_same_srepr_same_signature():
    """Acceptance criterion: same srepr always produces the same structural signature."""
    x = sp.Symbol("x")
    rec = ExpressionRecord.from_expr("e4", sp.Mul(x + 1, x + 1, evaluate=False))
    sig_a = structural_signature(to_ast(rec.parse()))
    sig_b = structural_signature(to_ast(rec.parse()))
    assert sig_a == sig_b


def test_different_structure_different_signature():
    """Structurally different expressions must not collide on signature."""
    x = sp.Symbol("x")
    ast1 = to_ast(ExpressionRecord.from_expr("e5", x + 1).parse())
    ast2 = to_ast(ExpressionRecord.from_expr("e6", x * 1).parse())
    assert structural_signature(ast1) != structural_signature(ast2)


def test_operand_order_affects_signature():
    """Left/right operand order must be preserved - not canonicalized away."""
    x = sp.Symbol("x")
    pow1 = to_ast(ExpressionRecord.from_expr("e7", sp.Pow(x, 2, evaluate=False)).parse())
    pow2 = to_ast(ExpressionRecord.from_expr("e8", sp.Pow(2, x, evaluate=False)).parse())
    assert structural_signature(pow1) != structural_signature(pow2)
