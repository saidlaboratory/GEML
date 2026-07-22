"""Symbolic diagnostics for sourced elementary identities.

These checks are deliberately labelled diagnostics.  They neither establish
the pure-EML branch semantics nor replace numeric and structural evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sympy import (
    Add,
    I,
    Integer,
    Mul,
    Pow,
    Rational,
    Symbol,
    cos,
    cosh,
    exp,
    log,
    simplify,
    sin,
    sinh,
    tan,
    tanh,
    together,
)


class SymbolicStatus(StrEnum):
    """Outcome reported by the non-authoritative SymPy diagnostic."""

    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    INDETERMINATE = "indeterminate"
    ERROR = "error"
    NOT_APPLICABLE = "not_applicable"


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
_Y = Symbol("y", real=True)
_POSITIVE_X = Symbol("x", positive=True)
_POSITIVE_Y = Symbol("y", positive=True)
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

_LOG_POSITIVE_X = log(_POSITIVE_X, evaluate=False)
_LOG_POSITIVE_Y = log(_POSITIVE_Y, evaluate=False)
_NEGATIVE_Y = Mul(Integer(-1), _Y, evaluate=False)
_GENERIC_FORMULAS = {
    "exp": (exp(_X, evaluate=False), exp(_X), ("x is finite and real",)),
    "log": (_LOG_POSITIVE_X, log(_POSITIVE_X), ("x is finite and strictly positive",)),
    "zero": (log(Integer(1), evaluate=False), Integer(0), ("the exact source constant is 1",)),
    "negate": (
        Add(Integer(0), Mul(Integer(-1), _X, evaluate=False), evaluate=False),
        -_X,
        ("x is finite and real",),
    ),
    "add": (
        Add(
            _X,
            Mul(Integer(-1), _NEGATIVE_Y, evaluate=False),
            evaluate=False,
        ),
        _X + _Y,
        ("x and y are finite and real",),
    ),
    "subtract": (
        Add(_X, Mul(Integer(-1), _Y, evaluate=False), evaluate=False),
        _X - _Y,
        ("x and y are finite and real",),
    ),
    "multiply": (
        exp(Add(_LOG_POSITIVE_X, _LOG_POSITIVE_Y, evaluate=False), evaluate=False),
        _POSITIVE_X * _POSITIVE_Y,
        ("x and y are finite and strictly positive",),
    ),
    "inverse": (
        exp(Mul(Integer(-1), _LOG_POSITIVE_X, evaluate=False), evaluate=False),
        1 / _POSITIVE_X,
        ("x is finite and strictly positive",),
    ),
    "divide": (
        exp(
            Add(
                _LOG_POSITIVE_X,
                Mul(Integer(-1), _LOG_POSITIVE_Y, evaluate=False),
                evaluate=False,
            ),
            evaluate=False,
        ),
        _POSITIVE_X / _POSITIVE_Y,
        ("x and y are finite and strictly positive",),
    ),
    "power": (
        exp(Mul(_Y, _LOG_POSITIVE_X, evaluate=False), evaluate=False),
        _POSITIVE_X**_Y,
        ("the base x is finite and strictly positive; exponent y is finite and real",),
    ),
}
_POWER_CASE_FORMULAS = {
    "power_square": (
        exp(Mul(Integer(2), _LOG_POSITIVE_X, evaluate=False), evaluate=False),
        Pow(_POSITIVE_X, Integer(2), evaluate=False),
        ("the base x is finite and strictly positive; exponent is the exact integer 2",),
    ),
    "power_half": (
        exp(Mul(Rational(1, 2), _LOG_POSITIVE_X, evaluate=False), evaluate=False),
        Pow(_POSITIVE_X, Rational(1, 2), evaluate=False),
        ("the base x is finite and strictly positive; exponent is the exact rational 1/2",),
    ),
    "power_negative_one": (
        exp(Mul(Integer(-1), _LOG_POSITIVE_X, evaluate=False), evaluate=False),
        Pow(_POSITIVE_X, Integer(-1), evaluate=False),
        ("the base x is finite and strictly positive; exponent is the exact integer -1",),
    ),
}
_SYMBOLICALLY_NOT_APPLICABLE = frozenset({"symbol", "one", "integer", "rational", "decimal"})


def _compare_formulas(
    *,
    operator: str,
    formula: object,
    target: object,
    assumptions: tuple[str, ...],
    method: str,
) -> SymbolicDiagnostic:
    """Reduce one independently stated construction/target pair diagnostically."""

    try:
        residual = simplify(together(formula - target))
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
            assumptions=assumptions,
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
            assumptions=assumptions,
            source_formula=str(formula),
            residual=None,
            message=f"{type(error).__name__}: {error}",
        )


def diagnose_symbolic_identity(operator: str) -> SymbolicDiagnostic:
    """Compare the sourced formula after rewriting both sides to exponentials."""

    if operator not in _FORMULAS:
        raise ValueError(f"unsupported symbolic diagnostic operator: {operator!r}")
    assumptions = ["x is real", "complex exponentials use SymPy's principal conventions"]
    if operator == "tan":
        assumptions.extend(("x is registry-certified in [-1, 1]", "cos(x) != 0"))
    method = "SymPy together+simplify after rewriting the source function to exp"
    formula = _FORMULAS[operator]
    return _compare_formulas(
        operator=operator,
        formula=formula,
        target=_TARGETS[operator].rewrite(exp),
        assumptions=tuple(assumptions),
        method=method,
    )


def diagnose_symbolic_formula(
    operator: str,
    *,
    case_id: str | None = None,
) -> SymbolicDiagnostic:
    """Diagnose an enabled construction identity without claiming a proof.

    Branch-sensitive arithmetic is intentionally checked only with positive
    symbols. Numeric audits retain the wider safe-real and nonzero-real cases.
    Leaf and exact-number constructors have no useful formula identity to
    simplify, so they return an explicit not-applicable diagnostic.
    """

    if operator in _FORMULAS:
        return diagnose_symbolic_identity(operator)
    if operator in _SYMBOLICALLY_NOT_APPLICABLE:
        return SymbolicDiagnostic(
            operator=operator,
            status=SymbolicStatus.NOT_APPLICABLE,
            method="symbolic diagnostic intentionally omitted for an exact source value",
            assumptions=("numeric reference is exact and independently stated",),
            source_formula=operator,
            residual=None,
            message="no symbolic identity is needed for this leaf or exact-number case",
        )
    try:
        formula, target, assumptions = (
            _POWER_CASE_FORMULAS[case_id]
            if operator == "power" and case_id in _POWER_CASE_FORMULAS
            else _GENERIC_FORMULAS[operator]
        )
    except KeyError as error:
        raise ValueError(f"unsupported symbolic diagnostic operator: {operator!r}") from error
    return _compare_formulas(
        operator=operator,
        formula=formula,
        target=target,
        assumptions=assumptions,
        method="SymPy together+simplify under the listed explicit assumptions",
    )
