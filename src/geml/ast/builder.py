"""
builder.py - turns a parsed sympy expr into our tree format.

owned by 1-6

trig update from Sahil: 1-3's registry does include trig, but it's
disabled for now, so we're NOT marking trig as approved or building
fixtures for it yet. sticking with the current op list below until
that flips.

TODO(Sahil): once 1-2/1-3 actually merge, swap SUPPORTED_BINARY/UNARY
and ASTNode below for the real frozen versions, this is still just a
placeholder
"""
from __future__ import annotations
import sympy as sp
from geml.parsing.srepr import UnsupportedNodeError

# placeholder op list, not the real registry yet
SUPPORTED_BINARY = ("Add", "Mul")
SUPPORTED_UNARY = ("Pow", "Log", "Exp", "Neg")


class ASTNode:
    # temp shape, will get replaced by whatever 1-2 locks in
    __slots__ = ("op", "children", "value")

    def __init__(self, op: str, children: tuple["ASTNode", ...] = (), value=None):
        self.op = op
        self.children = children
        self.value = value

    def __repr__(self):
        # readable printing, e.g. Add(x, 1) instead of <object at 0x...>
        if self.op in ("Var", "Const"):
            return f"{self.value}"
        return f"{self.op}({', '.join(repr(c) for c in self.children)})"


def _fold_left(op_name: str, args: tuple[sp.Expr, ...]) -> ASTNode:
    # sympy gives us x+y+z as one flat Add with 3 args, but our tree is
    # strictly binary. fold left to right so it's always the same
    # shape for the same input - (x+y) then +z, not the other way
    nodes = [to_ast(a) for a in args]
    acc = nodes[0]
    for nxt in nodes[1:]:
        acc = ASTNode(op_name, (acc, nxt))
    return acc


def to_ast(expr: sp.Expr) -> ASTNode:
    if expr.is_Symbol:
        return ASTNode("Var", value=str(expr))

    if expr.is_Integer or expr.is_Number:
        return ASTNode("Const", value=expr)

    if expr.func == sp.Add:
        return _fold_left("Add", expr.args)

    if expr.func == sp.Mul:
        # sympy stores -x as Mul(-1, x) under the hood. want this as
        # an actual Neg node instead since that's in our op list
        if len(expr.args) == 2 and expr.args[0] == sp.Integer(-1):
            return ASTNode("Neg", (to_ast(expr.args[1]),))
        return _fold_left("Mul", expr.args)

    if expr.func == sp.Pow:
        base, exponent = expr.args  # order matters, x^2 != 2^x
        return ASTNode("Pow", (to_ast(base), to_ast(exponent)))

    if expr.func == sp.log:
        (arg,) = expr.args
        return ASTNode("Log", (to_ast(arg),))

    if expr.func == sp.exp:
        (arg,) = expr.args
        return ASTNode("Exp", (to_ast(arg),))

    # anything else = explicit error, not a guess
    raise UnsupportedNodeError(
        f"unsupported: {expr.func} in {expr} (not in the op list yet, pending 1-3)"
    )
