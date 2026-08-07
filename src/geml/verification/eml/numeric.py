"""Failure-aware numerical audits for source functions and pure EML trees."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import mpmath as mp
import numpy as np

from geml.eml.compiler_arithmetic import (
    eml_decimal,
    eml_divide,
    eml_integer,
    eml_inverse,
    eml_multiply,
    eml_power,
    eml_rational,
)
from geml.eml.compiler_core import (
    CompilerMode,
    eml_add,
    eml_exp,
    eml_log,
    eml_negate,
    eml_subtract,
    eml_zero,
    require_compiler_mode,
)
from geml.eml.compiler_transcendental import eml_cosh, eml_sinh, eml_tanh
from geml.eml.compiler_trig import eml_cos, eml_sin, eml_tan
from geml.eml.ir import EMLTerm, One, Variable
from geml.eml.validate import validate_pure_eml
from geml.spec.domains import DOMAIN_REGISTRY
from geml.spec.operators import OPERATOR_REGISTRY


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
    UNSUPPORTED = "unsupported"
    INVALID_SAMPLE = "invalid_sample"
    COMPILER_ERROR = "compiler_error"
    SOURCE_EVALUATION_ERROR = "source_evaluation_error"
    EML_EVALUATION_ERROR = "eml_evaluation_error"


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
    operator_family: str = ""
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4
    variable_assignments: tuple[tuple[str, str], ...] = ()
    sample_label: str | None = None

    def __post_init__(self) -> None:
        if not self.operator_family and self.operator in OPERATOR_REGISTRY:
            object.__setattr__(
                self,
                "operator_family",
                OPERATOR_REGISTRY[self.operator].operator_family,
            )
        if self.sample_label is None:
            object.__setattr__(self, "sample_label", self.input_value)


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
    operator_family: str = ""
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4
    status_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.operator_family and self.operator in OPERATOR_REGISTRY:
            object.__setattr__(
                self,
                "operator_family",
                OPERATOR_REGISTRY[self.operator].operator_family,
            )
        if not self.status_counts:
            object.__setattr__(self, "status_counts", _status_counts(self.results))

    def status_counts_dict(self) -> dict[str, int]:
        """Return a JSON-friendly copy of the complete terminal-status counts."""

        return dict(self.status_counts)


@dataclass(frozen=True, slots=True)
class SemanticSample:
    """One ordered, labeled variable assignment for a generic semantic case."""

    label: str
    variables: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("semantic sample label must be nonblank")
        if not isinstance(self.variables, tuple):
            raise TypeError("semantic sample variables must be an ordered tuple")
        for assignment in self.variables:
            if (
                not isinstance(assignment, tuple)
                or len(assignment) != 2
                or not isinstance(assignment[0], str)
                or not assignment[0]
            ):
                raise TypeError("each semantic assignment must be a nonblank name/value pair")


SourceEvaluator = Callable[[dict[str, mp.mpf]], Any]
SemanticCompiler = Callable[[dict[str, EMLTerm], CompilerMode], EMLTerm]
SemanticGuard = Callable[[dict[str, mp.mpf], int], str | None]


@dataclass(frozen=True, slots=True)
class SemanticCase:
    """Hand-audited source reference and compiler construction specification."""

    case_id: str
    operator: str
    operator_family: str
    variable_names: tuple[str, ...]
    source_evaluator: SourceEvaluator
    compiler: SemanticCompiler
    default_samples: tuple[SemanticSample, ...]
    assumptions: tuple[str, ...] = ()
    guard: SemanticGuard | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("semantic case_id must be nonblank")
        if not isinstance(self.operator, str) or not self.operator.strip():
            raise ValueError("semantic operator must be nonblank")
        if not isinstance(self.operator_family, str) or not self.operator_family.strip():
            raise ValueError("semantic operator family must be nonblank")
        if not isinstance(self.variable_names, tuple):
            raise TypeError("semantic variable names must be an ordered tuple")
        if any(not isinstance(name, str) or not name.strip() for name in self.variable_names):
            raise TypeError("semantic variable names must be nonblank strings")
        if len(self.variable_names) != len(set(self.variable_names)):
            raise ValueError("semantic variable names must be unique")
        if not callable(self.source_evaluator) or not callable(self.compiler):
            raise TypeError("semantic source evaluator and compiler must be callable")
        if not isinstance(self.default_samples, tuple):
            raise TypeError("semantic default samples must be an ordered tuple")
        if not self.default_samples:
            raise ValueError("semantic cases must provide at least one default sample")
        if not all(isinstance(sample, SemanticSample) for sample in self.default_samples):
            raise TypeError("every semantic default sample must be a SemanticSample")
        if not isinstance(self.assumptions, tuple) or any(
            not isinstance(assumption, str) or not assumption.strip()
            for assumption in self.assumptions
        ):
            raise TypeError("semantic assumptions must be a tuple of nonblank strings")
        if self.guard is not None and not callable(self.guard):
            raise TypeError("semantic guard must be callable or None")


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


def _status_counts(results: tuple[NumericProbeResult, ...]) -> tuple[tuple[str, int], ...]:
    counts = Counter(row.status for row in results)
    return tuple((status.value, counts[status]) for status in ProbeStatus)


def _aggregate_numeric_audit(
    *,
    operator: str,
    operator_family: str,
    domain_mode: str,
    backend: NumericBackend,
    compiler_mode: CompilerMode,
    method: str,
    assumptions: tuple[str, ...],
    results: tuple[NumericProbeResult, ...],
    precision_digits: int,
) -> NumericAuditResult:
    """Build one denominator-complete aggregate at the requested precision."""

    with mp.workdps(precision_digits):
        finite_errors = tuple(
            (mp.mpf(row.absolute_error), mp.mpf(row.relative_error))
            for row in results
            if row.absolute_error is not None and row.relative_error is not None
        )
        maximum_absolute_error = (
            mp.nstr(max(error[0] for error in finite_errors), precision_digits)
            if finite_errors
            else None
        )
        maximum_relative_error = (
            mp.nstr(max(error[1] for error in finite_errors), precision_digits)
            if finite_errors
            else None
        )
    pass_count = sum(row.status in _PASS_STATUSES for row in results)
    return NumericAuditResult(
        operator=operator,
        domain_mode=domain_mode,
        backend=backend,
        method=method,
        assumptions=assumptions,
        requested_sample_count=len(results),
        pass_count=pass_count,
        failure_count=len(results) - pass_count,
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
        results=results,
        operator_family=operator_family,
        compiler_mode=compiler_mode,
    )


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


def _require_tolerance(value: str, *, name: str, precision_digits: int) -> mp.mpf:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a decimal string")
    with mp.workdps(precision_digits):
        try:
            parsed = mp.mpf(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite nonnegative decimal") from error
        if not mp.isfinite(parsed) or parsed < 0:
            raise ValueError(f"{name} must be a finite nonnegative decimal")
        return +parsed


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


def _evaluate_numpy_with_diagnostics(
    root: EMLTerm,
    variables: dict[str, np.complex128],
) -> tuple[np.complex128, bool, bool]:
    """Evaluate NumPy complex128 while retaining explicit overflow provenance."""

    def visit(node: EMLTerm) -> tuple[np.complex128, bool, bool]:
        if isinstance(node, One):
            return np.complex128(1), False, False
        if isinstance(node, Variable):
            try:
                return variables[node.name], False, False
            except KeyError as error:
                raise ValueError(f"no value supplied for EML variable {node.name!r}") from error
        left, left_extended, left_overflow = visit(node.left)
        right, right_extended, right_overflow = visit(node.right)
        with np.errstate(all="ignore"):
            exponentiated = np.complex128(np.exp(left))
            logged = np.complex128(np.log(right))
            value = np.complex128(exponentiated - logged)
        operation_overflow = (_is_finite_numpy(left) and not _is_finite_numpy(exponentiated)) or (
            _is_finite_numpy(exponentiated)
            and _is_finite_numpy(logged)
            and not _is_finite_numpy(value)
        )
        extended = (
            left_extended
            or right_extended
            or not _is_finite_numpy(left)
            or not _is_finite_numpy(right)
        )
        return value, extended, left_overflow or right_overflow or operation_overflow

    return visit(root)


def _evaluate_numpy(
    root: EMLTerm,
    variables: dict[str, np.complex128],
) -> tuple[np.complex128, bool]:
    """Compatibility wrapper for callers that consume the original two-value API."""

    value, extended, _overflow = _evaluate_numpy_with_diagnostics(root, variables)
    return value, extended


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
    classify_overflow: bool = False,
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
        if not mp.isfinite(point):
            return NumericProbeResult(
                operator=operator,
                domain_mode=domain_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                input_value=input_text,
                status=ProbeStatus.SOURCE_NONFINITE,
                source_value=None,
                eml_value=None,
                absolute_error=None,
                relative_error=None,
                extended_intermediate=False,
                message="input is not finite",
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
                backend_overflow = False
                eml_value, extended = _evaluate_mpmath(root, {"x": mp.mpc(point)})
                eml_is_finite = _is_finite_mpmath(eml_value)
                formatted_eml = _format_mpmath(eml_value, digits=precision_digits)
                comparison_value = mp.mpc(eml_value)
            else:
                eml_value, extended, backend_overflow = _evaluate_numpy_with_diagnostics(
                    root,
                    {"x": np.complex128(float(point))},
                )
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
        if classify_overflow and backend_overflow:
            return NumericProbeResult(
                operator=operator,
                domain_mode=domain_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                input_value=input_text,
                status=ProbeStatus.OVERFLOW,
                source_value=_format_mpmath(source, digits=precision_digits),
                eml_value=formatted_eml,
                absolute_error=None,
                relative_error=None,
                extended_intermediate=extended,
                message="NumPy complex128 overflow occurred during pure EML evaluation",
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

    if not isinstance(points, tuple):
        raise TypeError("points must be a tuple so the requested denominator is stable")
    if not points:
        raise ValueError("points must contain at least one requested sample")
    if operator not in _COMPILERS:
        case_id = _UNARY_CASE_IDS.get(operator)
        if case_id is None:
            raise ValueError(f"unsupported audit operator: {operator!r}")
        samples = tuple(
            SemanticSample(label=str(point), variables=(("x", point),)) for point in points
        )
        return audit_semantic_case(
            SEMANTIC_CASE_REGISTRY[case_id],
            samples,
            domain_mode=domain_mode,
            backend=backend,
            compiler_mode=compiler_mode,
            precision_digits=precision_digits,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
    if not isinstance(backend, NumericBackend):
        raise TypeError("backend must be a NumericBackend")
    require_compiler_mode(compiler_mode)
    if domain_mode not in DOMAIN_REGISTRY:
        raise ValueError(f"unknown audit domain: {domain_mode!r}")
    if (
        isinstance(precision_digits, bool)
        or not isinstance(precision_digits, int)
        or precision_digits < 20
    ):
        raise ValueError("precision_digits must be an integer of at least 20")
    absolute_limit = _require_tolerance(
        absolute_tolerance,
        name="absolute_tolerance",
        precision_digits=precision_digits,
    )
    relative_limit = _require_tolerance(
        relative_tolerance,
        name="relative_tolerance",
        precision_digits=precision_digits,
    )

    if domain_mode == "complex":
        assumptions = (
            *DOMAIN_REGISTRY[domain_mode].variable_assumptions,
            "complex source auditing is reserved and disabled",
        )
        method = "unsupported reserved complex-domain audit"
        results = tuple(
            NumericProbeResult(
                operator=operator,
                domain_mode=domain_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                input_value=str(point),
                status=ProbeStatus.UNSUPPORTED,
                source_value=None,
                eml_value=None,
                absolute_error=None,
                relative_error=None,
                extended_intermediate=False,
                message="complex source mode is reserved and not enabled",
                compiler_mode=compiler_mode,
                variable_assignments=(("x", str(point)),),
            )
            for point in points
        )
        return _aggregate_numeric_audit(
            operator=operator,
            operator_family=OPERATOR_REGISTRY[operator].operator_family,
            domain_mode=domain_mode,
            backend=backend,
            compiler_mode=compiler_mode,
            method=method,
            assumptions=assumptions,
            results=results,
            precision_digits=precision_digits,
        )
    if domain_mode not in _DOMAIN_ASSUMPTIONS:
        raise ValueError(f"unsupported real audit domain: {domain_mode!r}")

    root = _COMPILERS[operator](Variable("x"), mode=compiler_mode)
    results = tuple(
        replace(
            _probe(
                operator=operator,
                domain_mode=domain_mode,
                backend=backend,
                root=root,
                raw_point=point,
                precision_digits=precision_digits,
                absolute_tolerance=absolute_limit,
                relative_tolerance=relative_limit,
                classify_overflow=True,
            ),
            compiler_mode=compiler_mode,
            variable_assignments=(("x", str(point)),),
        )
        for point in points
    )
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
    return _aggregate_numeric_audit(
        operator=operator,
        operator_family=OPERATOR_REGISTRY[operator].operator_family,
        domain_mode=domain_mode,
        backend=backend,
        compiler_mode=compiler_mode,
        method=method,
        assumptions=assumptions,
        results=results,
        precision_digits=precision_digits,
    )


def _sample(label: str, **variables: object) -> SemanticSample:
    return SemanticSample(label=label, variables=tuple(variables.items()))


def _guard_positive_x(values: dict[str, mp.mpf], _: int) -> str | None:
    return None if values["x"] > 0 else "source argument x must be strictly positive"


def _guard_nonzero_x(values: dict[str, mp.mpf], _: int) -> str | None:
    return None if values["x"] != 0 else "source denominator x must be nonzero"


def _guard_nonzero_y(values: dict[str, mp.mpf], _: int) -> str | None:
    return None if values["y"] != 0 else "source denominator y must be nonzero"


def _guard_tan(values: dict[str, mp.mpf], precision_digits: int) -> str | None:
    tolerance = mp.power(10, -(precision_digits // 2))
    if abs(mp.cos(values["x"])) <= tolerance:
        return "tan source input is at or numerically indistinguishable from a real pole"
    if abs(values["x"]) > 1:
        return "tan source input is outside the registry-certified interval [-1, 1]"
    return None


def _guard_positive_power_base(values: dict[str, mp.mpf], _: int) -> str | None:
    return None if values["x"] > 0 else "noninteger real power requires a positive base"


def _variables(names: tuple[str, ...]) -> dict[str, EMLTerm]:
    return {name: Variable(name) for name in names}


SEMANTIC_CASES: tuple[SemanticCase, ...] = (
    SemanticCase(
        "symbol",
        "symbol",
        "leaf",
        ("x",),
        lambda values: values["x"],
        lambda variables, _mode: variables["x"],
        (_sample("negative x", x=-1), _sample("positive x", x="0.5")),
    ),
    SemanticCase(
        "one",
        "one",
        "source_constant",
        (),
        lambda _values: mp.mpf(1),
        lambda _variables, _mode: One(),
        (_sample("primitive one"),),
    ),
    SemanticCase(
        "integer_zero",
        "integer",
        "exact_number",
        (),
        lambda _values: mp.mpf(0),
        lambda _variables, mode: eml_integer(0, mode=mode),
        (_sample("integer 0"),),
    ),
    SemanticCase(
        "integer_minus_one",
        "integer",
        "exact_number",
        (),
        lambda _values: mp.mpf(-1),
        lambda _variables, mode: eml_integer(-1, mode=mode),
        (_sample("integer -1"),),
    ),
    SemanticCase(
        "integer_two",
        "integer",
        "exact_number",
        (),
        lambda _values: mp.mpf(2),
        lambda _variables, mode: eml_integer(2, mode=mode),
        (_sample("integer 2"),),
    ),
    SemanticCase(
        "integer_three",
        "integer",
        "exact_number",
        (),
        lambda _values: mp.mpf(3),
        lambda _variables, mode: eml_integer(3, mode=mode),
        (_sample("integer 3"),),
    ),
    SemanticCase(
        "rational_half",
        "rational",
        "exact_number",
        (),
        lambda _values: mp.mpf(1) / 2,
        lambda _variables, mode: eml_rational(1, 2, mode=mode),
        (_sample("rational 1/2"),),
    ),
    SemanticCase(
        "decimal_eighth",
        "decimal",
        "exact_number",
        (),
        lambda _values: mp.mpf(1) / 8,
        lambda _variables, mode: eml_decimal("0.125", mode=mode),
        (_sample("decimal 0.125"),),
        assumptions=("decimal input is converted by the exact base-10 rational policy",),
    ),
    SemanticCase(
        "exp",
        "exp",
        "exp_log",
        ("x",),
        lambda values: mp.exp(values["x"]),
        lambda variables, _mode: eml_exp(variables["x"]),
        (_sample("negative x", x=-1), _sample("positive x", x=1)),
    ),
    SemanticCase(
        "log",
        "log",
        "exp_log",
        ("x",),
        lambda values: mp.log(values["x"]),
        lambda variables, _mode: eml_log(variables["x"]),
        (
            _sample("below one", x="0.5"),
            _sample("above one", x=2),
            _sample("invalid negative", x=-1),
        ),
        assumptions=("log source argument is strictly positive",),
        guard=_guard_positive_x,
    ),
    SemanticCase(
        "zero",
        "zero",
        "source_constant",
        (),
        lambda _values: mp.mpf(0),
        lambda _variables, _mode: eml_zero(),
        (_sample("exact zero"),),
    ),
    SemanticCase(
        "negate",
        "negate",
        "arithmetic",
        ("x",),
        lambda values: -values["x"],
        lambda variables, mode: eml_negate(variables["x"], mode=mode),
        (_sample("negative x", x=-1), _sample("positive x", x="0.5")),
    ),
    SemanticCase(
        "add",
        "add",
        "arithmetic",
        ("x", "y"),
        lambda values: values["x"] + values["y"],
        lambda variables, mode: eml_add(variables["x"], variables["y"], mode=mode),
        (
            _sample("negative left", x=-2, y="0.5"),
            _sample("ordered positive", x="0.5", y=2),
        ),
    ),
    SemanticCase(
        "subtract",
        "subtract",
        "arithmetic",
        ("x", "y"),
        lambda values: values["x"] - values["y"],
        lambda variables, _mode: eml_subtract(variables["x"], variables["y"]),
        (
            _sample("ordered subtraction", x=-1, y=2),
            _sample("positive variables", x="0.5", y=2),
        ),
    ),
    SemanticCase(
        "multiply",
        "multiply",
        "arithmetic",
        ("x", "y"),
        lambda values: values["x"] * values["y"],
        lambda variables, mode: eml_multiply(variables["x"], variables["y"], mode=mode),
        (
            _sample("negative product", x=-2, y="0.5"),
            _sample("positive product", x="0.5", y=2),
            _sample("zero product", x=0, y=2),
        ),
    ),
    SemanticCase(
        "inverse",
        "inverse",
        "arithmetic",
        ("x",),
        lambda values: 1 / values["x"],
        lambda variables, mode: eml_inverse(variables["x"], mode=mode),
        (
            _sample("negative inverse", x=-2),
            _sample("positive inverse below one", x="0.5"),
            _sample("positive inverse above one", x=2),
            _sample("invalid zero", x=0),
        ),
        assumptions=("inverse source denominator is nonzero",),
        guard=_guard_nonzero_x,
    ),
    SemanticCase(
        "divide",
        "divide",
        "arithmetic",
        ("x", "y"),
        lambda values: values["x"] / values["y"],
        lambda variables, mode: eml_divide(variables["x"], variables["y"], mode=mode),
        (
            _sample("ordered division", x=-1, y=2),
            _sample("positive division", x="0.5", y=2),
            _sample("zero numerator", x=0, y=2),
            _sample("invalid zero denominator", x=1, y=0),
        ),
        assumptions=("division source denominator y is nonzero",),
        guard=_guard_nonzero_y,
    ),
    SemanticCase(
        "power_square",
        "power",
        "power",
        ("x",),
        lambda values: values["x"] ** 2,
        lambda variables, mode: eml_power(
            variables["x"],
            eml_integer(2, mode=mode),
            mode=mode,
        ),
        (
            _sample("negative integer-power base", x=-2),
            _sample("positive base", x=2),
            _sample("zero integer-power base", x=0),
        ),
        assumptions=("exponent is the bounded exact integer 2",),
    ),
    SemanticCase(
        "power_half",
        "power",
        "power",
        ("x",),
        lambda values: mp.sqrt(values["x"]),
        lambda variables, mode: eml_power(
            variables["x"],
            eml_rational(1, 2, mode=mode),
            mode=mode,
        ),
        (_sample("positive half-power base", x=4), _sample("invalid negative base", x=-1)),
        assumptions=("noninteger exponent 1/2 requires a strictly positive base",),
        guard=_guard_positive_power_base,
    ),
    SemanticCase(
        "power_negative_one",
        "power",
        "power",
        ("x",),
        lambda values: 1 / values["x"],
        lambda variables, mode: eml_power(
            variables["x"],
            eml_integer(-1, mode=mode),
            mode=mode,
        ),
        (_sample("negative exponent", x=2), _sample("invalid zero base", x=0)),
        assumptions=("negative exponent -1 requires a nonzero base",),
        guard=_guard_nonzero_x,
    ),
    *tuple(
        SemanticCase(
            operator,
            operator,
            "trigonometric",
            ("x",),
            source,
            compiler,
            (
                _sample("negative x", x="-0.5"),
                _sample("zero x", x=0),
                _sample("positive x", x="0.5"),
            ),
            assumptions=(
                (
                    "tan source argument is registry-certified in [-1, 1]",
                    "tan source argument satisfies cos(x) != 0",
                )
                if operator == "tan"
                else ()
            ),
            guard=_guard_tan if operator == "tan" else None,
        )
        for operator, source, compiler in (
            (
                "sin",
                lambda values: mp.sin(values["x"]),
                lambda variables, mode: eml_sin(variables["x"], mode=mode),
            ),
            (
                "cos",
                lambda values: mp.cos(values["x"]),
                lambda variables, mode: eml_cos(variables["x"], mode=mode),
            ),
            (
                "tan",
                lambda values: mp.tan(values["x"]),
                lambda variables, mode: eml_tan(variables["x"], mode=mode),
            ),
        )
    ),
    *tuple(
        SemanticCase(
            operator,
            operator,
            "hyperbolic",
            ("x",),
            source,
            compiler,
            (
                _sample("negative x", x="-0.5"),
                _sample("zero x", x=0),
                _sample("positive x", x="0.5"),
            ),
        )
        for operator, source, compiler in (
            (
                "sinh",
                lambda values: mp.sinh(values["x"]),
                lambda variables, mode: eml_sinh(variables["x"], mode=mode),
            ),
            (
                "cosh",
                lambda values: mp.cosh(values["x"]),
                lambda variables, mode: eml_cosh(variables["x"], mode=mode),
            ),
            (
                "tanh",
                lambda values: mp.tanh(values["x"]),
                lambda variables, mode: eml_tanh(variables["x"], mode=mode),
            ),
        )
    ),
)

SEMANTIC_CASE_REGISTRY = MappingProxyType({case.case_id: case for case in SEMANTIC_CASES})
_UNARY_CASE_IDS = MappingProxyType(
    {
        "exp": "exp",
        "log": "log",
        "negate": "negate",
        "inverse": "inverse",
    }
)


def compile_semantic_case(
    case: SemanticCase,
    *,
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> EMLTerm:
    """Compile a generic semantic case using frozen constructors."""

    if not isinstance(case, SemanticCase):
        raise TypeError("case must be a SemanticCase")
    require_compiler_mode(compiler_mode)
    root = case.compiler(_variables(case.variable_names), compiler_mode)
    validate_pure_eml(root)
    return root


def _generic_row(
    *,
    case: SemanticCase,
    sample: SemanticSample,
    domain_mode: str,
    compiler_mode: CompilerMode,
    backend: NumericBackend,
    method: str,
    assumptions: tuple[str, ...],
    status: ProbeStatus,
    assignments: tuple[tuple[str, str], ...],
    source_value: NumericValue | None = None,
    eml_value: NumericValue | None = None,
    absolute_error: str | None = None,
    relative_error: str | None = None,
    extended_intermediate: bool = False,
    message: str | None = None,
) -> NumericProbeResult:
    return NumericProbeResult(
        operator=case.operator,
        domain_mode=domain_mode,
        backend=backend,
        method=method,
        assumptions=assumptions,
        input_value=sample.label,
        status=status,
        source_value=source_value,
        eml_value=eml_value,
        absolute_error=absolute_error,
        relative_error=relative_error,
        extended_intermediate=extended_intermediate,
        message=message,
        operator_family=case.operator_family,
        compiler_mode=compiler_mode,
        variable_assignments=assignments,
        sample_label=sample.label,
    )


def _format_assignments(
    sample: SemanticSample,
) -> tuple[tuple[tuple[str, str], ...], Exception | None]:
    """Stringify assignments without letting hostile sample values erase a row."""

    formatted: list[tuple[str, str]] = []
    first_error: Exception | None = None
    for name, value in sample.variables:
        try:
            value_text = str(value)
        except Exception as error:
            value_text = f"<unprintable {type(value).__name__}>"
            if first_error is None:
                first_error = error
        formatted.append((name, value_text))
    return tuple(formatted), first_error


def _generic_probe(
    *,
    case: SemanticCase,
    sample: SemanticSample,
    domain_mode: str,
    compiler_mode: CompilerMode,
    backend: NumericBackend,
    root: EMLTerm | None,
    compiler_error: Exception | None,
    precision_digits: int,
    absolute_tolerance: mp.mpf,
    relative_tolerance: mp.mpf,
) -> NumericProbeResult:
    assumptions = (
        *DOMAIN_REGISTRY[domain_mode].variable_assumptions,
        *case.assumptions,
        "pure EML uses the principal complex logarithm internally",
    )
    method = (
        f"mpmath independent source and principal-complex EML evaluation at "
        f"{precision_digits} decimal digits"
        if backend is NumericBackend.MPMATH
        else "independent mpmath source with NumPy IEEE complex128 EML evaluation"
    )
    assignments, assignment_error = _format_assignments(sample)
    if assignment_error is not None:
        return _generic_row(
            case=case,
            sample=sample,
            domain_mode=domain_mode,
            compiler_mode=compiler_mode,
            backend=backend,
            method=method,
            assumptions=assumptions,
            status=ProbeStatus.INVALID_SAMPLE,
            assignments=assignments,
            message=(
                "sample assignment could not be rendered: "
                f"{type(assignment_error).__name__}: {assignment_error}"
            ),
        )
    names = tuple(name for name, _ in sample.variables)
    if len(names) != len(set(names)) or names != case.variable_names:
        return _generic_row(
            case=case,
            sample=sample,
            domain_mode=domain_mode,
            compiler_mode=compiler_mode,
            backend=backend,
            method=method,
            assumptions=assumptions,
            status=ProbeStatus.INVALID_SAMPLE,
            assignments=assignments,
            message=(
                f"sample variables {names!r} do not match required ordered variables "
                f"{case.variable_names!r}"
            ),
        )

    assignment_text = dict(assignments)
    values: dict[str, mp.mpf] = {}
    with mp.workdps(precision_digits):
        for name, raw_value in sample.variables:
            if isinstance(raw_value, (bool, complex)):
                return _generic_row(
                    case=case,
                    sample=sample,
                    domain_mode=domain_mode,
                    compiler_mode=compiler_mode,
                    backend=backend,
                    method=method,
                    assumptions=assumptions,
                    status=ProbeStatus.INVALID_SAMPLE,
                    assignments=assignments,
                    message=f"sample value for {name!r} must be a finite real scalar",
                )
            try:
                value = mp.mpf(assignment_text[name])
            except (TypeError, ValueError) as error:
                return _generic_row(
                    case=case,
                    sample=sample,
                    domain_mode=domain_mode,
                    compiler_mode=compiler_mode,
                    backend=backend,
                    method=method,
                    assumptions=assumptions,
                    status=ProbeStatus.INVALID_SAMPLE,
                    assignments=assignments,
                    message=f"sample value for {name!r} is invalid: {error}",
                )
            if not mp.isfinite(value):
                return _generic_row(
                    case=case,
                    sample=sample,
                    domain_mode=domain_mode,
                    compiler_mode=compiler_mode,
                    backend=backend,
                    method=method,
                    assumptions=assumptions,
                    status=ProbeStatus.SOURCE_NONFINITE,
                    assignments=assignments,
                    message=f"sample value for {name!r} is nonfinite",
                )
            if domain_mode == "positive_real" and value <= 0:
                return _generic_row(
                    case=case,
                    sample=sample,
                    domain_mode=domain_mode,
                    compiler_mode=compiler_mode,
                    backend=backend,
                    method=method,
                    assumptions=assumptions,
                    status=ProbeStatus.SOURCE_DOMAIN_ERROR,
                    assignments=assignments,
                    message=f"positive_real variable {name!r} must be strictly positive",
                )
            if domain_mode == "nonzero_real" and value == 0:
                return _generic_row(
                    case=case,
                    sample=sample,
                    domain_mode=domain_mode,
                    compiler_mode=compiler_mode,
                    backend=backend,
                    method=method,
                    assumptions=assumptions,
                    status=ProbeStatus.SOURCE_DOMAIN_ERROR,
                    assignments=assignments,
                    message=f"nonzero_real variable {name!r} must be nonzero",
                )
            values[name] = value

        if case.guard is not None:
            try:
                guard_message = case.guard(values, precision_digits)
            except Exception as error:
                return _generic_row(
                    case=case,
                    sample=sample,
                    domain_mode=domain_mode,
                    compiler_mode=compiler_mode,
                    backend=backend,
                    method=method,
                    assumptions=assumptions,
                    status=ProbeStatus.SOURCE_EVALUATION_ERROR,
                    assignments=assignments,
                    message=f"source guard failed: {type(error).__name__}: {error}",
                )
            if guard_message is not None and (
                not isinstance(guard_message, str) or not guard_message.strip()
            ):
                return _generic_row(
                    case=case,
                    sample=sample,
                    domain_mode=domain_mode,
                    compiler_mode=compiler_mode,
                    backend=backend,
                    method=method,
                    assumptions=assumptions,
                    status=ProbeStatus.SOURCE_EVALUATION_ERROR,
                    assignments=assignments,
                    message="source guard must return a nonblank message or None",
                )
            if guard_message is not None:
                return _generic_row(
                    case=case,
                    sample=sample,
                    domain_mode=domain_mode,
                    compiler_mode=compiler_mode,
                    backend=backend,
                    method=method,
                    assumptions=assumptions,
                    status=ProbeStatus.SOURCE_DOMAIN_ERROR,
                    assignments=assignments,
                    message=guard_message,
                )
        if compiler_error is not None or root is None:
            return _generic_row(
                case=case,
                sample=sample,
                domain_mode=domain_mode,
                compiler_mode=compiler_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                status=ProbeStatus.COMPILER_ERROR,
                assignments=assignments,
                message=(
                    f"{type(compiler_error).__name__}: {compiler_error}"
                    if compiler_error is not None
                    else "compiler returned no tree"
                ),
            )
        try:
            source = mp.mpc(case.source_evaluator(values))
        except OverflowError as error:
            return _generic_row(
                case=case,
                sample=sample,
                domain_mode=domain_mode,
                compiler_mode=compiler_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                status=ProbeStatus.OVERFLOW,
                assignments=assignments,
                message=f"source reference overflowed: {error}",
            )
        except (ArithmeticError, ValueError) as error:
            return _generic_row(
                case=case,
                sample=sample,
                domain_mode=domain_mode,
                compiler_mode=compiler_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                status=ProbeStatus.SOURCE_DOMAIN_ERROR,
                assignments=assignments,
                message=f"{type(error).__name__}: {error}",
            )
        except Exception as error:
            return _generic_row(
                case=case,
                sample=sample,
                domain_mode=domain_mode,
                compiler_mode=compiler_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                status=ProbeStatus.SOURCE_EVALUATION_ERROR,
                assignments=assignments,
                message=f"source evaluator failed: {type(error).__name__}: {error}",
            )
        if not _is_finite_mpmath(source):
            return _generic_row(
                case=case,
                sample=sample,
                domain_mode=domain_mode,
                compiler_mode=compiler_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                status=ProbeStatus.SOURCE_NONFINITE,
                assignments=assignments,
                source_value=_format_mpmath(source, digits=precision_digits),
                message="source reference evaluated to a nonfinite value",
            )

        try:
            if backend is NumericBackend.MPMATH:
                backend_overflow = False
                eml_value, extended = _evaluate_mpmath(
                    root,
                    {name: mp.mpc(value) for name, value in values.items()},
                )
                eml_is_finite = _is_finite_mpmath(eml_value)
                formatted_eml = _format_mpmath(eml_value, digits=precision_digits)
                comparison_value = mp.mpc(eml_value)
            else:
                numpy_variables = {
                    name: np.complex128(float(value)) for name, value in values.items()
                }
                if any(not _is_finite_numpy(value) for value in numpy_variables.values()):
                    return _generic_row(
                        case=case,
                        sample=sample,
                        domain_mode=domain_mode,
                        compiler_mode=compiler_mode,
                        backend=backend,
                        method=method,
                        assumptions=assumptions,
                        status=ProbeStatus.OVERFLOW,
                        assignments=assignments,
                        source_value=_format_mpmath(source, digits=precision_digits),
                        message="sample conversion overflowed NumPy complex128",
                    )
                eml_value, extended, backend_overflow = _evaluate_numpy_with_diagnostics(
                    root,
                    numpy_variables,
                )
                eml_is_finite = _is_finite_numpy(eml_value)
                formatted_eml = _format_numpy(eml_value)
                comparison_value = mp.mpc(
                    str(float(np.real(eml_value))),
                    str(float(np.imag(eml_value))),
                )
        except OverflowError as error:
            return _generic_row(
                case=case,
                sample=sample,
                domain_mode=domain_mode,
                compiler_mode=compiler_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                status=ProbeStatus.OVERFLOW,
                assignments=assignments,
                source_value=_format_mpmath(source, digits=precision_digits),
                message=str(error),
            )
        except (TypeError, ValueError, ZeroDivisionError) as error:
            return _generic_row(
                case=case,
                sample=sample,
                domain_mode=domain_mode,
                compiler_mode=compiler_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                status=ProbeStatus.EML_DOMAIN_ERROR,
                assignments=assignments,
                source_value=_format_mpmath(source, digits=precision_digits),
                message=f"{type(error).__name__}: {error}",
            )
        except Exception as error:
            return _generic_row(
                case=case,
                sample=sample,
                domain_mode=domain_mode,
                compiler_mode=compiler_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                status=ProbeStatus.EML_EVALUATION_ERROR,
                assignments=assignments,
                source_value=_format_mpmath(source, digits=precision_digits),
                message=f"pure EML evaluator failed: {type(error).__name__}: {error}",
            )

        if backend_overflow:
            return _generic_row(
                case=case,
                sample=sample,
                domain_mode=domain_mode,
                compiler_mode=compiler_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                status=ProbeStatus.OVERFLOW,
                assignments=assignments,
                source_value=_format_mpmath(source, digits=precision_digits),
                eml_value=formatted_eml,
                extended_intermediate=extended,
                message="NumPy complex128 overflow occurred during pure EML evaluation",
            )
        if not eml_is_finite:
            status = (
                ProbeStatus.EXTENDED_REAL_INTERMEDIATE if extended else ProbeStatus.EML_NONFINITE
            )
            return _generic_row(
                case=case,
                sample=sample,
                domain_mode=domain_mode,
                compiler_mode=compiler_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                status=status,
                assignments=assignments,
                source_value=_format_mpmath(source, digits=precision_digits),
                eml_value=formatted_eml,
                extended_intermediate=extended,
                message=(
                    "pure EML result is nonfinite after an extended-real intermediate"
                    if extended
                    else "pure EML result is nonfinite"
                ),
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
        else:
            status = ProbeStatus.PASS
            message = None
        return _generic_row(
            case=case,
            sample=sample,
            domain_mode=domain_mode,
            compiler_mode=compiler_mode,
            backend=backend,
            method=method,
            assumptions=assumptions,
            status=status,
            assignments=assignments,
            source_value=_format_mpmath(source, digits=precision_digits),
            eml_value=formatted_eml,
            absolute_error=mp.nstr(absolute_error, precision_digits),
            relative_error=mp.nstr(relative_error, precision_digits),
            extended_intermediate=extended,
            message=message,
        )


def audit_semantic_case(
    case: SemanticCase,
    samples: tuple[SemanticSample, ...] | None = None,
    *,
    domain_mode: str = "safe_real",
    backend: NumericBackend = NumericBackend.MPMATH,
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4,
    precision_digits: int = 100,
    absolute_tolerance: str = "1e-70",
    relative_tolerance: str = "1e-70",
    _compiled_root: EMLTerm | None = None,
    _compiler_error: Exception | None = None,
) -> NumericAuditResult:
    """Audit a zero-, unary-, or binary-source case with one retained row per sample."""

    if not isinstance(case, SemanticCase):
        raise TypeError("case must be a SemanticCase")
    if samples is None:
        samples = case.default_samples
    if not isinstance(samples, tuple):
        raise TypeError("samples must be a tuple so the requested denominator is stable")
    if not samples:
        raise ValueError("samples must contain at least one requested sample")
    if not all(isinstance(sample, SemanticSample) for sample in samples):
        raise TypeError("every sample must be a SemanticSample")
    if _compiled_root is not None and _compiler_error is not None:
        raise ValueError("a precompiled root and compiler error cannot both be supplied")
    if _compiler_error is not None and not isinstance(_compiler_error, Exception):
        raise TypeError("_compiler_error must be an Exception or None")
    if not isinstance(backend, NumericBackend):
        raise TypeError("backend must be a NumericBackend")
    require_compiler_mode(compiler_mode)
    if domain_mode not in DOMAIN_REGISTRY:
        raise ValueError(f"unknown audit domain: {domain_mode!r}")
    if (
        isinstance(precision_digits, bool)
        or not isinstance(precision_digits, int)
        or precision_digits < 20
    ):
        raise ValueError("precision_digits must be an integer of at least 20")
    absolute_limit = _require_tolerance(
        absolute_tolerance,
        name="absolute_tolerance",
        precision_digits=precision_digits,
    )
    relative_limit = _require_tolerance(
        relative_tolerance,
        name="relative_tolerance",
        precision_digits=precision_digits,
    )
    assumptions = (
        *DOMAIN_REGISTRY[domain_mode].variable_assumptions,
        *case.assumptions,
        "pure EML uses the principal complex logarithm internally",
    )
    if domain_mode == "complex":
        method = "unsupported reserved complex-domain audit"
        results = tuple(
            _generic_row(
                case=case,
                sample=sample,
                domain_mode=domain_mode,
                compiler_mode=compiler_mode,
                backend=backend,
                method=method,
                assumptions=assumptions,
                status=ProbeStatus.UNSUPPORTED,
                assignments=_format_assignments(sample)[0],
                message="complex source mode is reserved and not enabled",
            )
            for sample in samples
        )
        return _aggregate_numeric_audit(
            operator=case.operator,
            operator_family=case.operator_family,
            domain_mode=domain_mode,
            backend=backend,
            compiler_mode=compiler_mode,
            method=method,
            assumptions=assumptions,
            results=results,
            precision_digits=precision_digits,
        )

    if _compiled_root is not None:
        try:
            validate_pure_eml(_compiled_root)
            root = _compiled_root
            compiler_error: Exception | None = None
        except Exception as error:
            root = None
            compiler_error = error
    elif _compiler_error is not None:
        root = None
        compiler_error = _compiler_error
    else:
        try:
            root = compile_semantic_case(case, compiler_mode=compiler_mode)
            compiler_error = None
        except Exception as error:
            root = None
            compiler_error = error
    results = tuple(
        _generic_probe(
            case=case,
            sample=sample,
            domain_mode=domain_mode,
            compiler_mode=compiler_mode,
            backend=backend,
            root=root,
            compiler_error=compiler_error,
            precision_digits=precision_digits,
            absolute_tolerance=absolute_limit,
            relative_tolerance=relative_limit,
        )
        for sample in samples
    )
    method = (
        f"mpmath independent source and principal-complex EML evaluation at "
        f"{precision_digits} decimal digits"
        if backend is NumericBackend.MPMATH
        else "independent mpmath source with NumPy IEEE complex128 EML evaluation"
    )
    return _aggregate_numeric_audit(
        operator=case.operator,
        operator_family=case.operator_family,
        domain_mode=domain_mode,
        backend=backend,
        compiler_mode=compiler_mode,
        method=method,
        assumptions=assumptions,
        results=results,
        precision_digits=precision_digits,
    )
