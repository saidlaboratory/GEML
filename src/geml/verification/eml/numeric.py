"""Failure-aware numerical audits for source functions and pure EML trees."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import mpmath as mp
import numpy as np

from geml.eml.compiler_core import CompilerMode
from geml.eml.compiler_transcendental import eml_cosh, eml_sinh, eml_tanh
from geml.eml.compiler_trig import eml_cos, eml_sin, eml_tan
from geml.eml.ir import EMLTerm, One, Variable
from geml.eml.validate import validate_pure_eml


class NumericBackend(StrEnum):
    """Independent numeric execution paths used by the local audit."""

    MPMATH = "mpmath"
    NUMPY_COMPLEX128 = "numpy_complex128"


class ProbeStatus(StrEnum):
    """Explicit terminal classification for one requested sample point."""

    PASS = "pass"
    PASS_WITH_EXTENDED_INTERMEDIATE = "pass_with_extended_intermediate"
    SOURCE_DOMAIN_ERROR = "source_domain_error"
    SOURCE_NONFINITE = "source_nonfinite"
    EML_DOMAIN_ERROR = "eml_domain_error"
    EML_NONFINITE = "eml_nonfinite"
    EXTENDED_REAL_INTERMEDIATE = "extended_real_intermediate"
    OVERFLOW = "overflow"
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class NumericValue:
    """Stable printable real and imaginary components."""

    real: str
    imaginary: str


@dataclass(frozen=True, slots=True)
class NumericProbeResult:
    """Complete outcome for one operator/backend/input request."""

    operator: str
    domain_mode: str
    backend: NumericBackend
    method: str
    assumptions: tuple[str, ...]
    input_value: str
    status: ProbeStatus
    source_value: NumericValue | None
    eml_value: NumericValue | None
    absolute_error: str | None
    relative_error: str | None
    extended_intermediate: bool
    message: str | None


@dataclass(frozen=True, slots=True)
class NumericAuditResult:
    """Aggregate with a complete denominator and every retained probe row."""

    operator: str
    domain_mode: str
    backend: NumericBackend
    method: str
    assumptions: tuple[str, ...]
    requested_sample_count: int
    pass_count: int
    failure_count: int
    maximum_absolute_error: str | None
    maximum_relative_error: str | None
    results: tuple[NumericProbeResult, ...]


_SOURCE_FUNCTIONS = {
    "sin": mp.sin,
    "cos": mp.cos,
    "tan": mp.tan,
    "sinh": mp.sinh,
    "cosh": mp.cosh,
    "tanh": mp.tanh,
}
_COMPILERS = {
    "sin": eml_sin,
    "cos": eml_cos,
    "tan": eml_tan,
    "sinh": eml_sinh,
    "cosh": eml_cosh,
    "tanh": eml_tanh,
}
_DOMAIN_ASSUMPTIONS = {
    "safe_real": ("input is finite and real",),
    "positive_real": ("input is finite, real, and strictly positive",),
    "nonzero_real": ("input is finite, real, and nonzero",),
}
_PASS_STATUSES = frozenset({ProbeStatus.PASS, ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE})


def _is_finite_mpmath(value: Any) -> bool:
    try:
        return bool(mp.isfinite(mp.re(value)) and mp.isfinite(mp.im(value)))
    except (TypeError, ValueError):
        return False


def _is_finite_numpy(value: Any) -> bool:
    try:
        converted = np.complex128(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(converted.real) and np.isfinite(converted.imag))


def _format_mpmath(value: Any, *, digits: int) -> NumericValue:
    converted = mp.mpc(value)
    return NumericValue(
        real=mp.nstr(converted.real, digits),
        imaginary=mp.nstr(converted.imag, digits),
    )


def _format_numpy(value: Any) -> NumericValue:
    converted = np.complex128(value)
    return NumericValue(real=repr(float(converted.real)), imaginary=repr(float(converted.imag)))


def _require_tolerance(value: str, *, name: str) -> mp.mpf:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a decimal string")
    try:
        parsed = mp.mpf(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite nonnegative decimal") from error
    if not mp.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be a finite nonnegative decimal")
    return parsed


def _evaluate_mpmath(root: EMLTerm, variables: dict[str, mp.mpc]) -> tuple[mp.mpc, bool]:
    def visit(node: EMLTerm) -> tuple[mp.mpc, bool]:
        if isinstance(node, One):
            return mp.mpc(1), False
        if isinstance(node, Variable):
            try:
                return variables[node.name], False
            except KeyError as error:
                raise ValueError(f"no value supplied for EML variable {node.name!r}") from error
        left, left_extended = visit(node.left)
        right, right_extended = visit(node.right)
        value = mp.exp(left) - mp.log(right)
        extended = (
            left_extended
            or right_extended
            or not _is_finite_mpmath(left)
            or not _is_finite_mpmath(right)
        )
        return value, extended

    return visit(root)


def _evaluate_numpy(
    root: EMLTerm,
    variables: dict[str, np.complex128],
) -> tuple[np.complex128, bool]:
    def visit(node: EMLTerm) -> tuple[np.complex128, bool]:
        if isinstance(node, One):
            return np.complex128(1), False
        if isinstance(node, Variable):
            try:
                return variables[node.name], False
            except KeyError as error:
                raise ValueError(f"no value supplied for EML variable {node.name!r}") from error
        left, left_extended = visit(node.left)
        right, right_extended = visit(node.right)
        with np.errstate(all="ignore"):
            value = np.complex128(np.exp(left) - np.log(right))
        extended = (
            left_extended
            or right_extended
            or not _is_finite_numpy(left)
            or not _is_finite_numpy(right)
        )
        return value, extended

    return visit(root)


def evaluate_pure_eml(
    root: EMLTerm,
    *,
    variables: dict[str, complex | float | int],
    backend: NumericBackend,
    precision_digits: int = 100,
) -> tuple[Any, bool]:
    """Evaluate principal-branch ``eml(a,b)=exp(a)-Log(b)``.

    The boolean reports whether a strict descendant evaluated nonfinite. The caller
    separately classifies a nonfinite final result.
    """

    validate_pure_eml(root)
    if (
        isinstance(precision_digits, bool)
        or not isinstance(precision_digits, int)
        or precision_digits < 20
    ):
        raise ValueError("precision_digits must be an integer of at least 20")
    if backend is NumericBackend.MPMATH:
        with mp.workdps(precision_digits):
            converted = {name: mp.mpc(value) for name, value in variables.items()}
            return _evaluate_mpmath(root, converted)
    if backend is NumericBackend.NUMPY_COMPLEX128:
        converted = {name: np.complex128(value) for name, value in variables.items()}
        return _evaluate_numpy(root, converted)
    raise TypeError("backend must be a NumericBackend")


def _domain_error(
    operator: str,
    domain_mode: str,
    value: mp.mpf,
    *,
    pole_tolerance: mp.mpf,
) -> str | None:
    if not mp.isfinite(value):
        return "input is not finite"
    if domain_mode == "positive_real" and value <= 0:
        return "positive_real input must be strictly positive"
    if domain_mode == "nonzero_real" and value == 0:
        return "nonzero_real input must be nonzero"
    if operator == "tan" and abs(mp.cos(value)) <= pole_tolerance:
        return "tan source input is at or numerically indistinguishable from a real pole"
    return None


def _probe(
    *,
    operator: str,
    domain_mode: str,
    backend: NumericBackend,
    root: EMLTerm,
    raw_point: int | float | str,
    precision_digits: int,
    absolute_tolerance: mp.mpf,
    relative_tolerance: mp.mpf,
) -> NumericProbeResult:
    assumptions = (
        *_DOMAIN_ASSUMPTIONS[domain_mode],
        "pure EML uses the principal complex logarithm internally",
        *(("tan argument satisfies cos(argument) != 0",) if operator == "tan" else ()),
    )
    method = (
        f"mpmath principal-complex evaluation at {precision_digits} decimal digits"
        if backend is NumericBackend.MPMATH
        else "NumPy IEEE complex128 principal-log evaluation"
    )
    input_text = str(raw_point)
    with mp.workdps(precision_digits):
        try:
            point = mp.mpf(input_text)
        except (TypeError, ValueError) as error:
            return NumericProbeResult(
                operator=operator,
                domain_mode=domain_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                input_value=input_text,
                status=ProbeStatus.SOURCE_DOMAIN_ERROR,
                source_value=None,
                eml_value=None,
                absolute_error=None,
                relative_error=None,
                extended_intermediate=False,
                message=f"input cannot be represented as a real sample: {error}",
            )
        pole_tolerance = mp.power(10, -(precision_digits // 2))
        domain_message = _domain_error(
            operator,
            domain_mode,
            point,
            pole_tolerance=pole_tolerance,
        )
        if domain_message is not None:
            return NumericProbeResult(
                operator=operator,
                domain_mode=domain_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                input_value=input_text,
                status=ProbeStatus.SOURCE_DOMAIN_ERROR,
                source_value=None,
                eml_value=None,
                absolute_error=None,
                relative_error=None,
                extended_intermediate=False,
                message=domain_message,
            )

        try:
            source = mp.mpc(_SOURCE_FUNCTIONS[operator](point))
        except (ValueError, ZeroDivisionError) as error:
            return NumericProbeResult(
                operator=operator,
                domain_mode=domain_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                input_value=input_text,
                status=ProbeStatus.SOURCE_DOMAIN_ERROR,
                source_value=None,
                eml_value=None,
                absolute_error=None,
                relative_error=None,
                extended_intermediate=False,
                message=str(error),
            )
        if not _is_finite_mpmath(source):
            return NumericProbeResult(
                operator=operator,
                domain_mode=domain_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                input_value=input_text,
                status=ProbeStatus.SOURCE_NONFINITE,
                source_value=_format_mpmath(source, digits=precision_digits),
                eml_value=None,
                absolute_error=None,
                relative_error=None,
                extended_intermediate=False,
                message="source function evaluated to a nonfinite value",
            )

        try:
            if backend is NumericBackend.MPMATH:
                eml_value, extended = _evaluate_mpmath(root, {"x": mp.mpc(point)})
                eml_is_finite = _is_finite_mpmath(eml_value)
                formatted_eml = _format_mpmath(eml_value, digits=precision_digits)
                comparison_value = mp.mpc(eml_value)
            else:
                eml_value, extended = _evaluate_numpy(root, {"x": np.complex128(float(point))})
                eml_is_finite = _is_finite_numpy(eml_value)
                formatted_eml = _format_numpy(eml_value)
                comparison_value = mp.mpc(
                    str(float(np.real(eml_value))),
                    str(float(np.imag(eml_value))),
                )
        except OverflowError as error:
            status = ProbeStatus.OVERFLOW
            message = str(error)
            eml_value = None
            extended = False
            eml_is_finite = False
            formatted_eml = None
            comparison_value = None
        except (TypeError, ValueError, ZeroDivisionError) as error:
            status = ProbeStatus.EML_DOMAIN_ERROR
            message = str(error)
            eml_value = None
            extended = False
            eml_is_finite = False
            formatted_eml = None
            comparison_value = None
        else:
            status = ProbeStatus.PASS
            message = None

        if eml_value is None or comparison_value is None:
            return NumericProbeResult(
                operator=operator,
                domain_mode=domain_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                input_value=input_text,
                status=status,
                source_value=_format_mpmath(source, digits=precision_digits),
                eml_value=formatted_eml,
                absolute_error=None,
                relative_error=None,
                extended_intermediate=extended,
                message=message,
            )
        if not eml_is_finite:
            status = (
                ProbeStatus.EXTENDED_REAL_INTERMEDIATE if extended else ProbeStatus.EML_NONFINITE
            )
            nonfinite_message = "pure EML result is nonfinite"
            if extended:
                nonfinite_message += " after an extended-real intermediate"
            return NumericProbeResult(
                operator=operator,
                domain_mode=domain_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                input_value=input_text,
                status=status,
                source_value=_format_mpmath(source, digits=precision_digits),
                eml_value=formatted_eml,
                absolute_error=None,
                relative_error=None,
                extended_intermediate=extended,
                message=nonfinite_message,
            )

        absolute_error = abs(comparison_value - source)
        relative_error = absolute_error / abs(source) if source != 0 else absolute_error
        threshold = absolute_tolerance + relative_tolerance * abs(source)
        if absolute_error > threshold:
            status = ProbeStatus.MISMATCH
            message = f"absolute error {mp.nstr(absolute_error, 12)} exceeds tolerance"
        elif extended:
            status = ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE
            message = "final value passed, but at least one intermediate was nonfinite"
        return NumericProbeResult(
            operator=operator,
            domain_mode=domain_mode,
            backend=backend,
            method=method,
            assumptions=assumptions,
            input_value=input_text,
            status=status,
            source_value=_format_mpmath(source, digits=precision_digits),
            eml_value=formatted_eml,
            absolute_error=mp.nstr(absolute_error, precision_digits),
            relative_error=mp.nstr(relative_error, precision_digits),
            extended_intermediate=extended,
            message=message,
        )


def audit_numeric_operator(
    operator: str,
    points: tuple[int | float | str, ...],
    *,
    domain_mode: str = "safe_real",
    backend: NumericBackend = NumericBackend.MPMATH,
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4,
    precision_digits: int = 100,
    absolute_tolerance: str = "1e-70",
    relative_tolerance: str = "1e-70",
) -> NumericAuditResult:
    """Audit every requested point; invalid and failed rows remain in the result."""

    if operator not in _COMPILERS:
        raise ValueError(f"unsupported audit operator: {operator!r}")
    if domain_mode not in _DOMAIN_ASSUMPTIONS:
        raise ValueError(f"unsupported real audit domain: {domain_mode!r}")
    if not isinstance(backend, NumericBackend):
        raise TypeError("backend must be a NumericBackend")
    if not isinstance(points, tuple):
        raise TypeError("points must be a tuple so the requested denominator is stable")
    if not points:
        raise ValueError("points must contain at least one requested sample")
    if (
        isinstance(precision_digits, bool)
        or not isinstance(precision_digits, int)
        or precision_digits < 20
    ):
        raise ValueError("precision_digits must be an integer of at least 20")
    absolute_limit = _require_tolerance(absolute_tolerance, name="absolute_tolerance")
    relative_limit = _require_tolerance(relative_tolerance, name="relative_tolerance")

    root = _COMPILERS[operator](Variable("x"), mode=compiler_mode)
    results = tuple(
        _probe(
            operator=operator,
            domain_mode=domain_mode,
            backend=backend,
            root=root,
            raw_point=point,
            precision_digits=precision_digits,
            absolute_tolerance=absolute_limit,
            relative_tolerance=relative_limit,
        )
        for point in points
    )
    finite_errors = [
        (mp.mpf(row.absolute_error), mp.mpf(row.relative_error))
        for row in results
        if row.absolute_error is not None and row.relative_error is not None
    ]
    assumptions = (
        *_DOMAIN_ASSUMPTIONS[domain_mode],
        "pure EML uses the principal complex logarithm internally",
        *(("tan argument satisfies cos(argument) != 0",) if operator == "tan" else ()),
    )
    method = (
        f"mpmath principal-complex evaluation at {precision_digits} decimal digits"
        if backend is NumericBackend.MPMATH
        else "NumPy IEEE complex128 principal-log evaluation"
    )
    pass_count = sum(row.status in _PASS_STATUSES for row in results)
    return NumericAuditResult(
        operator=operator,
        domain_mode=domain_mode,
        backend=backend,
        method=method,
        assumptions=assumptions,
        requested_sample_count=len(points),
        pass_count=pass_count,
        failure_count=len(points) - pass_count,
        maximum_absolute_error=(
            mp.nstr(max(error[0] for error in finite_errors), precision_digits)
            if finite_errors
            else None
        ),
        maximum_relative_error=(
            mp.nstr(max(error[1] for error in finite_errors), precision_digits)
            if finite_errors
            else None
        ),
        results=results,
    )
