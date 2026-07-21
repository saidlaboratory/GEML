"""
srepr.py

Handles reading and writing expressions in their "authoritative" form -
the raw structural dump, not the pretty-printed version. Printing can
quietly reorder or simplify things, so we never trust it as the source
of truth.

Owned by [1-6], staying in scope.
"""
from __future__ import annotations
from dataclasses import dataclass
import sympy as sp


class UnsupportedNodeError(ValueError):
    """Basically a ValueError with a clearer name. Made it its own
    class so other code can catch "this operator isn't supported yet"
    specifically, instead of accidentally catching every other kind
    of error too."""


# TODO(Sahil): once [1-2] merges the real contracts, swap this out for
# whatever the actual frozen record shape ends up being. Right now I'm
# just guessing at expr_id + srepr - the real one might need more
# fields (family, split, checksum, whatever else corpus needs). Ping
# me once it's ready and I'll update this.
@dataclass(frozen=True)
class ExpressionRecord:
    expr_id: str
    srepr: str  # the real structural form - never the printed string

    @classmethod
    def from_expr(cls, expr_id: str, expr: sp.Expr) -> "ExpressionRecord":
        return cls(expr_id=expr_id, srepr=sp.srepr(expr))

    def parse(self) -> sp.Expr:
        """
        Turns the stored srepr text back into a real expression.

        Heads up - this isn't just sp.sympify(self.srepr). Tried that
        first and it quietly breaks things: (x+1)*(x+1) comes back out
        as (x+1)**2, even with evaluate=False passed in. SymPy
        re-simplifies it somewhere during parsing and sympify alone
        doesn't seem to stop that.

        What actually works: eval() the string ourselves inside an
        evaluate(False) block. More manual, but it actually keeps the
        structure intact the way it's supposed to.
        """
        with sp.evaluate(False):
            return eval(self.srepr, {k: v for k, v in vars(sp).items()})
