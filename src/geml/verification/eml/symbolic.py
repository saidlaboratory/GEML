"""Symbolic diagnostics for sourced elementary identities.

These checks are deliberately labelled diagnostics.  They neither establish
the pure-EML branch semantics nor replace numeric and structural evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sympy import I, Symbol, cos, cosh, exp, simplify, sin, sinh, tan, tanh, together


class SymbolicStatus(StrEnum):
    """Outcome reported by the non-authoritative SymPy diagnostic."""

    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    INDETERMINATE = "indeterminate"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SymbolicDiagnostic:
    """One explicit formula comparison and its assumptions."""

    operator: str
    status: SymbolicStatus
    method: str
    assumptions: tuple[str, ...]
    source_formula: str
    residual: str | None
    message: str | None
    proof_claimed: bool = False


_X = Symbol("x", real=True)
_FORMULAS = {
    "sin": -I * (exp(I * _X) - exp(-I * _X)) / 2,
    "cos": (exp(I * _X) + exp(-I * _X)) / 2,
    "tan": I * (-exp(I * _X) + exp(-I * _X)) / (exp(I * _X) + exp(-I * _X)),
    "sinh": (exp(2 * _X) - 1) / (2 * exp(_X)),
    "cosh": (exp(2 * _X) + 1) / (2 * exp(_X)),
    "tanh": (exp(2 * _X) - 1) / (exp(2 * _X) + 1),
}
_TARGETS = {
    "sin": sin(_X),
    "cos": cos(_X),
    "tan": tan(_X),
    "sinh": sinh(_X),
    "cosh": cosh(_X),
    "tanh": tanh(_X),
}


def diagnose_symbolic_identity(operator: str) -> SymbolicDiagnostic:
    """Compare the sourced formula after rewriting both sides to exponentials."""

    if operator not in _FORMULAS:
        raise ValueError(f"unsupported symbolic diagnostic operator: {operator!r}")
    assumptions = ["x is real", "complex exponentials use SymPy's principal conventions"]
    if operator == "tan":
        assumptions.append("cos(x) != 0")
    method = "SymPy together+simplify after rewriting the source function to exp"
    formula = _FORMULAS[operator]
    try:
        residual = simplify(together(formula - _TARGETS[operator].rewrite(exp)))
        if residual == 0:
            status = SymbolicStatus.EQUAL
        else:
            equality = residual.equals(0)
            if equality is True:
                status = SymbolicStatus.EQUAL
            elif equality is False:
                status = SymbolicStatus.NOT_EQUAL
            else:
                status = SymbolicStatus.INDETERMINATE
        return SymbolicDiagnostic(
            operator=operator,
            status=status,
            method=method,
            assumptions=tuple(assumptions),
            source_formula=str(formula),
            residual=str(residual),
            message=(
                "diagnostic agreement is not a proof of pure-EML branch correctness"
                if status is SymbolicStatus.EQUAL
                else "symbolic comparison did not reduce to zero"
            ),
        )
    except Exception as error:  # SymPy diagnostics must retain backend-specific failures
        return SymbolicDiagnostic(
            operator=operator,
            status=SymbolicStatus.ERROR,
            method=method,
            assumptions=tuple(assumptions),
            source_formula=str(formula),
            residual=None,
            message=f"{type(error).__name__}: {error}",
        )
