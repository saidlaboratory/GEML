"""Combined reproducible evidence for official pure-EML constructions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from geml.eml.compiler_core import CompilerMode
from geml.eml.compiler_transcendental import eml_cosh, eml_sinh, eml_tanh
from geml.eml.compiler_trig import eml_cos, eml_sin, eml_tan
from geml.eml.emitter import emit_eml
from geml.eml.ir import Variable
from geml.eml.validate import PureEMLStatistics, validate_pure_eml
from geml.spec.operators import OPERATOR_FAMILY_IDS, OPERATORS, EMLConstructionStatus
from geml.verification.eml.numeric import (
    SEMANTIC_CASES,
    NumericAuditResult,
    NumericBackend,
    SemanticCase,
    SemanticSample,
    audit_numeric_operator,
    audit_semantic_case,
    compile_semantic_case,
)
from geml.verification.eml.symbolic import (
    SymbolicDiagnostic,
    diagnose_symbolic_formula,
    diagnose_symbolic_identity,
)

PINNED_COMPILER_COMMIT = "b3da148261199b46247306dfd92068f589778260"
PINNED_V4_SOURCE_SHA256 = "7b147b564a952c8ef24f1b1f6bb2b443a68ce44100ad019d5222df9547b6da62"
PINNED_CLEAN_SOURCE_SHA256 = "85be0e39271856dfa26c6368ba031f5eb83aa35a7beea6aba472e5dca6448f03"

_COMPILERS = {
    "sin": eml_sin,
    "cos": eml_cos,
    "tan": eml_tan,
    "sinh": eml_sinh,
    "cosh": eml_cosh,
    "tanh": eml_tanh,
}
_EXPECTED_FINGERPRINTS = MappingProxyType(
    {
        CompilerMode.OFFICIAL_V4: MappingProxyType(
            {
                "sin": "d9fa0e691922aee5a0c57ac75e101e63abf9b9ee7f56841ff1820a4aa8cd6571",
                "cos": "1a0c5493b625c1fda4d4e4436e7e4c677eff304e01af1119e5c1c4fca530695f",
                "tan": "20c4f5fa49f4f62955d507c1da34cd5d898e61bda5155f5c3a3e61982dcf45b1",
                "sinh": "888c44fb76f939795e4943b500ad32c8cb223f903abc18bd008eede550213aa2",
                "cosh": "f40e6f5e9d6bc3db56f7b0ebc20df2e747b107acf6a06b83874cf10b8ec2609e",
                "tanh": "03aa8d0795c63db5c86c202342ce4f130a5b6858198fd76ec889cb3f79e0a943",
            }
        ),
        CompilerMode.CLEAN_NEGATION: MappingProxyType(
            {
                "sin": "d834c494688fdbfa764964a3762c02803afe3d76e26476d20b64d0ef545130a0",
                "cos": "d3b861ffd36f2c027eab1f6be6b7cbe8b3721e767b263f83894f00f7e7ba98b1",
                "tan": "7597f6360dcec6f277eb807fdd88dbab0868320afde639051a35ed1a7541938b",
                "sinh": "2f079ef578337e7fafb1a2633cbcbbfa691688bb6ea647fb92763939f8c50e95",
                "cosh": "33ead62ee5d3a657df52ca36e387561b59f78de1cfc3cb95c46652a8595a04ac",
                "tanh": "3d7ca7a475154e7ed9c8b8139dad855c92666867d101fb9938294830cc91b36a",
            }
        ),
    }
)


@dataclass(frozen=True, slots=True)
class ConstructionAudit:
    """Structural, fingerprint, symbolic, and two-backend evidence for one operator."""

    operator: str
    compiler_mode: CompilerMode
    pinned_commit: str
    emitted_sha256: str
    expected_sha256: str
    fingerprint_matches: bool
    statistics: PureEMLStatistics
    symbolic: SymbolicDiagnostic
    high_precision: NumericAuditResult
    complex128: NumericAuditResult


class EvidenceStatus(StrEnum):
    """Aggregate classification that never discards unsuccessful evidence rows."""

    COMPLETE = "complete"
    COMPLETE_WITH_RETAINED_FAILURES = "complete_with_retained_failures"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class SemanticConstructionAudit:
    """Structural, symbolic, and numeric evidence for one hand-audited case."""

    case_id: str
    operator: str
    operator_family: str
    domain_mode: str
    compiler_mode: CompilerMode
    pinned_commit: str
    emitted_sha256: str | None
    statistics: PureEMLStatistics | None
    symbolic: SymbolicDiagnostic
    numeric_audits: tuple[NumericAuditResult, ...]
    requested_sample_backend_count: int
    assumptions: tuple[str, ...]
    evidence_status: EvidenceStatus


@dataclass(frozen=True, slots=True)
class EnabledFamilyAudit:
    """Stable evidence matrix and explicit registry-coverage boundary."""

    domain_mode: str
    compiler_mode: CompilerMode
    approved_operators: tuple[str, ...]
    covered_operators: tuple[str, ...]
    covered_families: tuple[str, ...]
    excluded_source_constants: tuple[tuple[str, str], ...]
    cases: tuple[SemanticConstructionAudit, ...]


def expected_fingerprint(operator: str, mode: CompilerMode) -> str:
    """Return the offline golden derived from the pinned official source."""

    try:
        return _EXPECTED_FINGERPRINTS[mode][operator]
    except KeyError as error:
        raise ValueError(f"no pinned fingerprint for {operator!r} in mode {mode!r}") from error


def audit_construction(
    operator: str,
    *,
    points: tuple[int | float | str, ...] = (-1, -0.5, 0, 0.5, 1),
    domain_mode: str = "safe_real",
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> ConstructionAudit:
    """Run an offline construction audit without fetching the authoritative source."""

    try:
        tree = _COMPILERS[operator](Variable("x"), mode=compiler_mode)
    except KeyError as error:
        raise ValueError(f"unsupported construction audit operator: {operator!r}") from error
    emitted = emit_eml(tree)
    digest = hashlib.sha256(emitted.encode("utf-8")).hexdigest()
    expected = expected_fingerprint(operator, compiler_mode)
    return ConstructionAudit(
        operator=operator,
        compiler_mode=compiler_mode,
        pinned_commit=PINNED_COMPILER_COMMIT,
        emitted_sha256=digest,
        expected_sha256=expected,
        fingerprint_matches=digest == expected,
        statistics=validate_pure_eml(tree),
        symbolic=diagnose_symbolic_identity(operator),
        high_precision=audit_numeric_operator(
            operator,
            points,
            domain_mode=domain_mode,
            backend=NumericBackend.MPMATH,
            compiler_mode=compiler_mode,
            precision_digits=100,
            absolute_tolerance="1e-70",
            relative_tolerance="1e-70",
        ),
        complex128=audit_numeric_operator(
            operator,
            points,
            domain_mode=domain_mode,
            backend=NumericBackend.NUMPY_COMPLEX128,
            compiler_mode=compiler_mode,
            precision_digits=100,
            absolute_tolerance="1e-11",
            relative_tolerance="1e-11",
        ),
    )


def audit_all_six(
    *,
    points: tuple[int | float | str, ...] = (-1, -0.5, 0, 0.5, 1),
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4,
) -> tuple[ConstructionAudit, ...]:
    """Return stable-order evidence for all approved trig/hyperbolic candidates."""

    return tuple(
        audit_construction(operator, points=points, compiler_mode=compiler_mode)
        for operator in ("sin", "cos", "tan", "sinh", "cosh", "tanh")
    )


def audit_semantic_construction(
    case: SemanticCase,
    samples: tuple[SemanticSample, ...] | None = None,
    *,
    domain_mode: str = "safe_real",
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4,
    backends: tuple[NumericBackend, ...] = (
        NumericBackend.MPMATH,
        NumericBackend.NUMPY_COMPLEX128,
    ),
) -> SemanticConstructionAudit:
    """Combine non-proof evidence for one source/construction specification."""

    if not isinstance(backends, tuple):
        raise TypeError("backends must be a tuple so evidence ordering is stable")
    if not backends:
        raise ValueError("backends must contain at least one numeric backend")
    if not all(isinstance(backend, NumericBackend) for backend in backends):
        raise TypeError("every backend must be a NumericBackend")

    try:
        tree = compile_semantic_case(case, compiler_mode=compiler_mode)
        compiler_error: Exception | None = None
        emitted_sha256 = hashlib.sha256(emit_eml(tree).encode("utf-8")).hexdigest()
        statistics = validate_pure_eml(tree)
    except Exception as error:
        tree = None
        compiler_error = error
        emitted_sha256 = None
        statistics = None
    numeric_audits = tuple(
        audit_semantic_case(
            case,
            samples,
            domain_mode=domain_mode,
            backend=backend,
            compiler_mode=compiler_mode,
            absolute_tolerance=("1e-70" if backend is NumericBackend.MPMATH else "1e-11"),
            relative_tolerance=("1e-70" if backend is NumericBackend.MPMATH else "1e-11"),
            _compiled_root=tree,
            _compiler_error=compiler_error,
        )
        for backend in backends
    )
    failure_count = sum(audit.failure_count for audit in numeric_audits)
    unsupported_count = sum(audit.status_counts_dict()["unsupported"] for audit in numeric_audits)
    requested_count = sum(audit.requested_sample_count for audit in numeric_audits)
    if unsupported_count == requested_count:
        evidence_status = EvidenceStatus.UNSUPPORTED
    elif failure_count:
        evidence_status = EvidenceStatus.COMPLETE_WITH_RETAINED_FAILURES
    else:
        evidence_status = EvidenceStatus.COMPLETE
    assumptions = tuple(
        dict.fromkeys(assumption for audit in numeric_audits for assumption in audit.assumptions)
    )
    return SemanticConstructionAudit(
        case_id=case.case_id,
        operator=case.operator,
        operator_family=case.operator_family,
        domain_mode=domain_mode,
        compiler_mode=compiler_mode,
        pinned_commit=PINNED_COMPILER_COMMIT,
        emitted_sha256=emitted_sha256,
        statistics=statistics,
        symbolic=diagnose_symbolic_formula(case.operator, case_id=case.case_id),
        numeric_audits=numeric_audits,
        requested_sample_backend_count=requested_count,
        assumptions=assumptions,
        evidence_status=evidence_status,
    )


def audit_enabled_operator_families(
    *,
    domain_mode: str = "safe_real",
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4,
    backends: tuple[NumericBackend, ...] = (
        NumericBackend.MPMATH,
        NumericBackend.NUMPY_COMPLEX128,
    ),
) -> EnabledFamilyAudit:
    """Audit every approved source operator in stable hand-reviewed case order."""

    approved_operators = tuple(
        operator.name
        for operator in OPERATORS
        if operator.enabled_for_generation
        and operator.eml_construction_status is EMLConstructionStatus.APPROVED
    )
    covered_operators = tuple(dict.fromkeys(case.operator for case in SEMANTIC_CASES))
    missing = tuple(
        operator for operator in approved_operators if operator not in covered_operators
    )
    if missing:
        raise RuntimeError(f"semantic case registry omits approved operators: {missing!r}")
    cases = tuple(
        audit_semantic_construction(
            case,
            domain_mode=domain_mode,
            compiler_mode=compiler_mode,
            backends=backends,
        )
        for case in SEMANTIC_CASES
    )
    excluded_source_constants = tuple(
        (operator.name, operator.eml_construction_status.value)
        for operator in OPERATORS
        if operator.operator_family == "source_constant"
        and not (
            operator.enabled_for_generation
            and operator.eml_construction_status is EMLConstructionStatus.APPROVED
        )
    )
    return EnabledFamilyAudit(
        domain_mode=domain_mode,
        compiler_mode=compiler_mode,
        approved_operators=approved_operators,
        covered_operators=covered_operators,
        covered_families=tuple(
            family
            for family in OPERATOR_FAMILY_IDS
            if any(case.operator_family == family for case in SEMANTIC_CASES)
        ),
        excluded_source_constants=excluded_source_constants,
        cases=cases,
    )
