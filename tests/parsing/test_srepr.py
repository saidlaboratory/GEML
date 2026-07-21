"""tests/parsing/test_srepr.py - owned by [1-6]."""
import sympy as sp
from geml.parsing.srepr import ExpressionRecord


def test_authoritative_form_survives_round_trip():
    """srepr round-trips back to an equal expression."""
    x, y = sp.symbols("x y")
    e = x + y
    rec = ExpressionRecord.from_expr("e1", e)
    assert sp.simplify(rec.parse() - e) == 0


def test_duplicate_subtree_survives_parse():
    """
    Regression test for a real bug found during development:
    sp.sympify(srepr_string) - even with evaluate=False - silently
    collapses Mul(Add(x,1), Add(x,1)) into Pow(Add(x,1), 2), destroying
    the duplicate-subtree structure the spec's own worked example
    (sec 2.1/2.2) depends on. This must never regress.
    """
    x = sp.Symbol("x")
    e = sp.Mul(x + 1, x + 1, evaluate=False)
    rec = ExpressionRecord.from_expr("e2", e)
    parsed = rec.parse()
    assert parsed.func == sp.Mul, (
        f"expected Mul, got {parsed.func} - the sympify Pow-collapse bug is back"
    )
    assert len(parsed.args) == 2


def test_same_srepr_produces_same_parse():
    """Same srepr string always parses to the same structure (determinism)."""
    x = sp.Symbol("x")
    rec = ExpressionRecord.from_expr("e3", sp.Mul(x + 1, x + 1, evaluate=False))
    a = rec.parse()
    b = rec.parse()
    assert sp.srepr(a) == sp.srepr(b)
