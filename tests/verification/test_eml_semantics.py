"""Domain-explicit semantic evidence for the six official constructions."""

from __future__ import annotations

import mpmath as mp
import pytest

from geml.eml.compiler_core import CompilerMode
from geml.eml.ir import EML, One, Variable
from geml.verification.eml.audit import audit_all_six, audit_construction
from geml.verification.eml.numeric import (
    NumericBackend,
    ProbeStatus,
    _probe,
    audit_numeric_operator,
)
from geml.verification.eml.symbolic import SymbolicStatus, diagnose_symbolic_identity


def test_all_six_pass_stratified_pole_safe_points_in_two_numeric_backends() -> None:
    audits = audit_all_six(points=(-1, -0.5, 0.5, 1))
    assert len(audits) == 6
    for audit in audits:
        assert audit.fingerprint_matches
        assert audit.symbolic.status is SymbolicStatus.EQUAL
        assert not audit.symbolic.proof_claimed
        assert audit.high_precision.requested_sample_count == 4
        assert audit.high_precision.pass_count == 4
        assert audit.high_precision.failure_count == 0
        assert audit.complex128.pass_count == 4
        assert audit.complex128.failure_count == 0
        assert all(row.assumptions and row.method for row in audit.high_precision.results)


def test_known_zero_extended_intermediates_are_retained_not_dropped() -> None:
    for operator in ("sinh", "tanh"):
        audit = audit_construction(operator, points=(0,))
        row = audit.high_precision.results[0]
        assert row.status is ProbeStatus.EXTENDED_REAL_INTERMEDIATE
        assert row.extended_intermediate
        assert audit.high_precision.requested_sample_count == 1
        assert audit.high_precision.failure_count == 1

        ieee_row = audit.complex128.results[0]
        assert ieee_row.status is ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE
        assert ieee_row.extended_intermediate


def test_nonfinite_root_with_finite_children_is_not_mislabeled_as_an_intermediate() -> None:
    result = _probe(
        operator="cos",
        domain_mode="safe_real",
        backend=NumericBackend.NUMPY_COMPLEX128,
        root=EML(Variable("x"), One()),
        raw_point=1_000,
        precision_digits=100,
        absolute_tolerance=mp.mpf("1e-12"),
        relative_tolerance=mp.mpf("1e-12"),
    )
    assert result.status is ProbeStatus.EML_NONFINITE
    assert not result.extended_intermediate
    assert result.message == "pure EML result is nonfinite"


def test_tan_pole_and_invalid_real_mode_points_are_explicit_domain_rows() -> None:
    with mp.workdps(100):
        pole = mp.nstr(mp.pi / 2, 100)
    tan_audit = audit_numeric_operator("tan", (pole,), precision_digits=100)
    assert tan_audit.requested_sample_count == tan_audit.failure_count == 1
    assert tan_audit.results[0].status is ProbeStatus.SOURCE_DOMAIN_ERROR
    assert "pole" in (tan_audit.results[0].message or "")

    positive = audit_numeric_operator("sin", (-1,), domain_mode="positive_real")
    nonzero = audit_numeric_operator("cos", (0,), domain_mode="nonzero_real")
    assert positive.results[0].status is ProbeStatus.SOURCE_DOMAIN_ERROR
    assert nonzero.results[0].status is ProbeStatus.SOURCE_DOMAIN_ERROR


@pytest.mark.parametrize("operator", ["sin", "cos", "tan", "sinh", "cosh", "tanh"])
def test_clean_negation_mode_also_passes_safe_numeric_points(operator: str) -> None:
    result = audit_numeric_operator(
        operator,
        (-1, -0.5, 0.5, 1),
        backend=NumericBackend.NUMPY_COMPLEX128,
        compiler_mode=CompilerMode.CLEAN_NEGATION,
        absolute_tolerance="1e-10",
        relative_tolerance="1e-10",
    )
    assert result.pass_count == 4
    assert result.failure_count == 0


def test_symbolic_checks_are_diagnostics_not_proof_substitutes() -> None:
    result = diagnose_symbolic_identity("tan")
    assert result.status is SymbolicStatus.EQUAL
    assert not result.proof_claimed
    assert "cos(x) != 0" in result.assumptions
    assert "not a proof" in (result.message or "")


@pytest.mark.parametrize("tolerance", ["nan", "inf", "-1e-12", "not-a-number"])
def test_numeric_audit_rejects_nonfinite_negative_or_invalid_tolerances(
    tolerance: str,
) -> None:
    with pytest.raises(ValueError, match="finite nonnegative decimal"):
        audit_numeric_operator("sin", (0,), absolute_tolerance=tolerance)


def test_numeric_audit_rejects_invalid_backend_and_empty_probe_grid() -> None:
    with pytest.raises(TypeError, match="NumericBackend"):
        audit_numeric_operator("sin", (0,), backend="mpmath")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        audit_numeric_operator("sin", ())
