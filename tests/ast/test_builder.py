"""tests/ast/test_builder.py - owned by [1-6]."""
import sympy as sp
import pytest
from geml.parsing.srepr import ExpressionRecord, UnsupportedNodeError
from geml.ast.builder import to_ast


def test_spec_example_add():
    """Spec sec 2.2: x+y -> Add(x, y)."""
    x, y = sp.symbols("x y")
    ast = to_ast(ExpressionRecord.from_expr("e1", x + y).parse())
    assert ast.op == "Add"
    assert [c.value for c in ast.children] == ["x", "y"]


def test_spec_example_duplicate_subtree():
    """Spec sec 2.2: (x+1)(x+1) -> Mul(Add(x,1), Add(x,1)), both children identical."""
    x = sp.Symbol("x")
    e = sp.Mul(x + 1, x + 1, evaluate=False)
    ast = to_ast(ExpressionRecord.from_expr("e2", e).parse())
    assert ast.op == "Mul"
    assert len(ast.children) == 2
    assert repr(ast.children[0]) == repr(ast.children[1])


def test_pow_preserves_base_exponent_order():
    x = sp.Symbol("x")
    ast = to_ast(ExpressionRecord.from_expr("e3", sp.Pow(x, 3, evaluate=False)).parse())
    assert ast.op == "Pow"
    assert ast.children[0].value == "x"  # base first
    assert ast.children[1].value == sp.Integer(3)  # exponent second


def test_neg_detected_explicitly():
    x = sp.Symbol("x")
    ast = to_ast(ExpressionRecord.from_expr("e4", -x).parse())
    assert ast.op == "Neg"
    assert ast.children[0].value == "x"


def test_nary_add_folds_left_deterministically():
    """x+y+z should fold as Add(Add(x,y), z), not right-associated."""
    x, y, z = sp.symbols("x y z")
    ast = to_ast(ExpressionRecord.from_expr("e5", x + y + z).parse())
    assert ast.op == "Add"
    assert ast.children[1].value == "z"  # outermost fold applied last arg
    assert ast.children[0].op == "Add"


def test_unsupported_node_fails_explicitly():
    """sin(x) is outside the provisional registry - must raise, not silently pass."""
    x = sp.Symbol("x")
    rec = ExpressionRecord.from_expr("e6", sp.sin(x))
    with pytest.raises(UnsupportedNodeError):
        to_ast(rec.parse())


def test_tree_invariants_enforced_via_construction():
    """
    Acceptance criterion: 'tree invariants are enforced.'

    Per Sahil: no separate validate_invariants() needed - to_ast()
    already only ever builds correctly-shaped nodes (Pow always gets
    exactly base+exponent, Neg always gets exactly one child, etc.)
    because those shapes come straight from how each sympy node is
    unpacked. Anything that doesn't fit a known shape gets rejected
    instead of silently producing a malformed tree.

    Trig is a good test case for this specifically: it's in 1-3's
    registry but disabled right now, so it should still be rejected
    the same way any unsupported node is - proving the "reject rather
    than build broken" behavior holds even for things that exist in
    the wider spec but aren't switched on yet.
    """
    x = sp.Symbol("x")
    for trig_expr in (sp.sin(x), sp.cos(x), sp.tan(x)):
        rec = ExpressionRecord.from_expr("trig_check", trig_expr)
        with pytest.raises(UnsupportedNodeError):
            to_ast(rec.parse())
