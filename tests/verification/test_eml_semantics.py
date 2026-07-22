"""Domain-explicit semantic evidence for official pure-EML constructions."""

from __future__ import annotations

import mpmath as mp
import pytest

from geml.eml.compiler_core import CompilerMode
from geml.eml.ir import EML, One, Variable
from geml.spec.operators import OPERATOR_FAMILY_IDS, OPERATORS, EMLConstructionStatus
from geml.verification.eml import symbolic as symbolic_module
from geml.verification.eml.audit import (
    EvidenceStatus,
    audit_all_six,
    audit_construction,
    audit_enabled_operator_families,
    audit_semantic_construction,
)
from geml.verification.eml.numeric import (
    SEMANTIC_CASE_REGISTRY,
    SEMANTIC_CASES,
    NumericBackend,
    ProbeStatus,
    SemanticCase,
    SemanticSample,
    _probe,
    audit_numeric_operator,
    audit_semantic_case,
    evaluate_pure_eml,
)
from geml.verification.eml.symbolic import (
    SymbolicStatus,
    diagnose_symbolic_formula,
    diagnose_symbolic_identity,
)


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


def _sample(label: str, *assignments: tuple[str, object]) -> SemanticSample:
    return SemanticSample(label=label, variables=assignments)


def test_semantic_case_registry_covers_every_enabled_operator_and_family() -> None:
    approved = tuple(
        operator.name
        for operator in OPERATORS
        if operator.enabled_for_generation
        and operator.eml_construction_status is EMLConstructionStatus.APPROVED
    )
    covered = tuple(dict.fromkeys(case.operator for case in SEMANTIC_CASES))
    families = tuple(dict.fromkeys(case.operator_family for case in SEMANTIC_CASES))

    assert tuple(SEMANTIC_CASE_REGISTRY) == tuple(case.case_id for case in SEMANTIC_CASES)
    assert set(approved) <= set(covered)
    assert set(families) == set(OPERATOR_FAMILY_IDS)
    assert not {"e", "pi", "imaginary_unit"} & set(covered)
    assert {"decimal", "inverse"} <= set(covered)


@pytest.mark.parametrize("case", SEMANTIC_CASES, ids=lambda case: case.case_id)
def test_every_semantic_case_retains_one_complete_row_per_requested_sample(
    case: SemanticCase,
) -> None:
    audit = audit_semantic_case(case)

    assert audit.requested_sample_count == len(case.default_samples)
    assert len(audit.results) == audit.requested_sample_count
    assert audit.pass_count + audit.failure_count == audit.requested_sample_count
    assert sum(audit.status_counts_dict().values()) == audit.requested_sample_count
    assert tuple(audit.status_counts_dict()) == tuple(status.value for status in ProbeStatus)
    assert audit.operator_family == case.operator_family
    assert audit.compiler_mode is CompilerMode.OFFICIAL_V4
    assert all(row.assumptions and row.method for row in audit.results)
    assert all(row.operator_family == case.operator_family for row in audit.results)
    assert all(row.compiler_mode is CompilerMode.OFFICIAL_V4 for row in audit.results)
    assert all(row.sample_label for row in audit.results)


@pytest.mark.parametrize("backend", tuple(NumericBackend))
def test_generic_audit_supports_both_independent_numeric_backends(
    backend: NumericBackend,
) -> None:
    tolerance = "1e-11" if backend is NumericBackend.NUMPY_COMPLEX128 else "1e-70"
    audit = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["divide"],
        (_sample("ordered", ("x", -1), ("y", 2)),),
        backend=backend,
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
    )

    assert audit.pass_count == 1
    assert audit.results[0].source_value is not None
    assert mp.mpf(audit.results[0].source_value.real) == mp.mpf("-0.5")


def test_binary_samples_preserve_variable_and_argument_order() -> None:
    samples = (
        _sample("x minus y", ("x", 5), ("y", 2)),
        _sample("values swapped", ("x", 2), ("y", 5)),
        _sample("names out of order", ("y", 2), ("x", 5)),
        _sample("missing y", ("x", 5)),
    )
    audit = audit_semantic_case(SEMANTIC_CASE_REGISTRY["subtract"], samples)

    assert [row.status for row in audit.results] == [
        ProbeStatus.PASS,
        ProbeStatus.PASS,
        ProbeStatus.INVALID_SAMPLE,
        ProbeStatus.INVALID_SAMPLE,
    ]
    assert audit.results[0].source_value is not None
    assert audit.results[1].source_value is not None
    assert mp.mpf(audit.results[0].source_value.real) == 3
    assert mp.mpf(audit.results[1].source_value.real) == -3
    assert audit.results[0].variable_assignments == (("x", "5"), ("y", "2"))


@pytest.mark.parametrize("case_id", ["add", "multiply"])
def test_safe_arithmetic_accepts_negatives_that_positive_real_rejects(case_id: str) -> None:
    negative = (_sample("negative", ("x", -2), ("y", 1)),)
    safe = audit_semantic_case(SEMANTIC_CASE_REGISTRY[case_id], negative)
    positive = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY[case_id],
        negative,
        domain_mode="positive_real",
    )

    assert safe.results[0].status in {
        ProbeStatus.PASS,
        ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE,
    }
    assert positive.results[0].status is ProbeStatus.SOURCE_DOMAIN_ERROR
    assert all("strictly positive" not in assumption for assumption in safe.assumptions)
    assert all("strictly positive" in " ".join(row.assumptions) for row in positive.results)


def test_log_guards_are_stricter_than_safe_and_nonzero_variable_domains() -> None:
    safe_log = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["log"],
        (_sample("negative log", ("x", -1)),),
    )
    nonzero_log = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["log"],
        (_sample("negative nonzero log", ("x", -1)),),
        domain_mode="nonzero_real",
    )

    assert safe_log.results[0].status is ProbeStatus.SOURCE_DOMAIN_ERROR
    assert nonzero_log.results[0].status is ProbeStatus.SOURCE_DOMAIN_ERROR


def test_positive_real_pass_rows_cover_both_sides_of_one_with_explicit_assumptions() -> None:
    result = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["exp"],
        (
            _sample("below one", ("x", "0.5")),
            _sample("above one", ("x", 2)),
        ),
        domain_mode="positive_real",
    )

    assert result.pass_count == 2
    assert all("strictly positive" in " ".join(row.assumptions) for row in result.results)


def test_error_maxima_exclude_uncomparable_domain_failure_rows() -> None:
    result = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["log"],
        (
            _sample("valid", ("x", 2)),
            _sample("invalid", ("x", -1)),
        ),
    )

    assert result.maximum_absolute_error is not None
    assert result.maximum_relative_error is not None
    assert result.results[1].status is ProbeStatus.SOURCE_DOMAIN_ERROR
    assert result.results[1].absolute_error is None
    assert result.results[1].relative_error is None


@pytest.mark.parametrize(
    ("case_id", "sample"),
    [
        ("log", _sample("nonpositive log", ("x", 0))),
        ("inverse", _sample("zero inverse", ("x", 0))),
        ("divide", _sample("zero divisor", ("x", 1), ("y", 0))),
        ("power_half", _sample("negative fractional base", ("x", -1))),
        ("power_negative_one", _sample("zero negative-power base", ("x", 0))),
    ],
)
def test_operation_specific_guards_retain_source_domain_failures(
    case_id: str,
    sample: SemanticSample,
) -> None:
    audit = audit_semantic_case(SEMANTIC_CASE_REGISTRY[case_id], (sample,))

    assert audit.requested_sample_count == audit.failure_count == 1
    assert audit.results[0].status is ProbeStatus.SOURCE_DOMAIN_ERROR
    assert audit.results[0].source_value is None
    assert audit.results[0].absolute_error is None


def test_generic_tan_pole_nonfinite_input_and_complex_mode_are_explicit() -> None:
    with mp.workdps(100):
        pole = mp.nstr(mp.pi / 2, 100)
    tan = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["tan"],
        (_sample("pole", ("x", pole)),),
    )
    nonfinite = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["sin"],
        (_sample("nonfinite", ("x", "nan")),),
    )
    unsupported_value = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["sin"],
        (_sample("unsupported scalar", ("x", True)),),
    )
    complex_result = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["sin"],
        (_sample("reserved", ("x", 1)),),
        domain_mode="complex",
    )

    assert tan.results[0].status is ProbeStatus.SOURCE_DOMAIN_ERROR
    assert "pole" in (tan.results[0].message or "")
    assert nonfinite.results[0].status is ProbeStatus.SOURCE_NONFINITE
    assert unsupported_value.results[0].status is ProbeStatus.INVALID_SAMPLE
    assert complex_result.results[0].status is ProbeStatus.UNSUPPORTED
    assert complex_result.failure_count == 1


@pytest.mark.parametrize("case_id", ["sinh", "tanh"])
def test_negative_hyperbolic_sign_regression(case_id: str) -> None:
    result = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY[case_id],
        (_sample("negative", ("x", "-0.5")),),
    )
    row = result.results[0]

    assert row.status in {
        ProbeStatus.PASS,
        ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE,
    }
    assert row.source_value is not None
    assert mp.mpf(row.source_value.real) < 0
    assert row.eml_value is not None
    assert mp.mpf(row.eml_value.real) < 0


@pytest.mark.parametrize("operator", ["exp", "log", "negate", "inverse"])
def test_legacy_unary_audit_wrapper_supports_extended_unary_operators(
    operator: str,
) -> None:
    result = audit_numeric_operator(operator, (1,))

    assert result.requested_sample_count == 1
    assert result.results[0].variable_assignments == (("x", "1"),)
    assert result.compiler_mode is CompilerMode.OFFICIAL_V4


def test_pure_eml_evaluator_accepts_arbitrary_variable_mappings() -> None:
    tree = EML(Variable("y"), Variable("z"))
    expected = mp.exp(2) - mp.log(4)

    result, extended = evaluate_pure_eml(
        tree,
        variables={"y": 2, "z": 4},
        backend=NumericBackend.MPMATH,
    )

    assert mp.almosteq(result, expected)
    assert not extended


def test_compiler_failures_are_retained_for_every_requested_sample() -> None:
    invalid_case = SemanticCase(
        case_id="compiler_failure",
        operator="exp",
        operator_family="exp_log",
        variable_names=("x",),
        source_evaluator=lambda values: mp.exp(values["x"]),
        compiler=lambda _variables, _mode: object(),  # type: ignore[return-value]
        default_samples=(_sample("first", ("x", 1)), _sample("second", ("x", 2))),
    )
    result = audit_semantic_case(invalid_case)

    assert result.requested_sample_count == result.failure_count == 2
    assert {row.status for row in result.results} == {ProbeStatus.COMPILER_ERROR}


@pytest.mark.parametrize(
    "operator",
    ["exp", "log", "zero", "negate", "add", "subtract", "multiply", "inverse", "divide", "power"],
)
def test_generic_symbolic_diagnostics_are_explicit_nonproof_evidence(
    operator: str,
) -> None:
    result = diagnose_symbolic_formula(operator)

    assert result.status is SymbolicStatus.EQUAL
    assert result.assumptions
    assert result.method
    assert not result.proof_claimed
    assert "not a proof" in (result.message or "")


def test_exact_symbolic_diagnostic_is_explicitly_not_applicable() -> None:
    result = diagnose_symbolic_formula("rational")

    assert result.status is SymbolicStatus.NOT_APPLICABLE
    assert not result.proof_claimed
    assert result.message


def test_symbolic_indeterminate_and_error_paths_are_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IndeterminateResidual:
        def equals(self, _value: object) -> None:
            return None

        def __str__(self) -> str:
            return "undetermined"

    monkeypatch.setattr(symbolic_module, "simplify", lambda _value: IndeterminateResidual())
    indeterminate = diagnose_symbolic_formula("add")
    assert indeterminate.status is SymbolicStatus.INDETERMINATE
    assert indeterminate.residual == "undetermined"

    def fail(_value: object) -> object:
        raise RuntimeError("diagnostic backend failed")

    monkeypatch.setattr(symbolic_module, "simplify", fail)
    error = diagnose_symbolic_formula("add")
    assert error.status is SymbolicStatus.ERROR
    assert "diagnostic backend failed" in (error.message or "")
    assert not error.proof_claimed


def test_integrated_construction_audit_labels_modes_and_fingerprints_separately() -> None:
    sample = (_sample("positive", ("x", 1)),)
    official = audit_semantic_construction(
        SEMANTIC_CASE_REGISTRY["negate"],
        sample,
        backends=(NumericBackend.MPMATH,),
    )
    clean = audit_semantic_construction(
        SEMANTIC_CASE_REGISTRY["negate"],
        sample,
        compiler_mode=CompilerMode.CLEAN_NEGATION,
        backends=(NumericBackend.MPMATH,),
    )

    assert official.compiler_mode is CompilerMode.OFFICIAL_V4
    assert clean.compiler_mode is CompilerMode.CLEAN_NEGATION
    assert official.emitted_sha256 != clean.emitted_sha256
    assert official.statistics != clean.statistics
    assert official.requested_sample_backend_count == 1
    assert official.evidence_status is EvidenceStatus.COMPLETE


def test_enabled_family_audit_is_stable_complete_and_keeps_registry_boundaries() -> None:
    first = audit_enabled_operator_families(backends=(NumericBackend.MPMATH,))
    second = audit_enabled_operator_families(backends=(NumericBackend.MPMATH,))

    assert first.compiler_mode is CompilerMode.OFFICIAL_V4
    assert first.covered_families == OPERATOR_FAMILY_IDS
    assert first.approved_operators == second.approved_operators
    assert tuple(case.case_id for case in first.cases) == tuple(
        case.case_id for case in second.cases
    )
    assert dict(first.excluded_source_constants) == {
        "e": "pending_verification",
        "pi": "pending_verification",
        "imaginary_unit": "reserved",
    }
    assert all(case.numeric_audits for case in first.cases)
    assert all(case.assumptions for case in first.cases)
    assert all(case.symbolic.method for case in first.cases)


def test_integrated_complex_audit_is_explicitly_unsupported() -> None:
    result = audit_semantic_construction(
        SEMANTIC_CASE_REGISTRY["sin"],
        (_sample("reserved", ("x", 1)),),
        domain_mode="complex",
        backends=(NumericBackend.MPMATH,),
    )

    assert result.evidence_status is EvidenceStatus.UNSUPPORTED
    assert result.numeric_audits[0].results[0].status is ProbeStatus.UNSUPPORTED


def test_generic_audit_rejects_invalid_api_inputs() -> None:
    case = SEMANTIC_CASE_REGISTRY["sin"]
    sample = (_sample("valid", ("x", 1)),)

    with pytest.raises(TypeError, match="NumericBackend"):
        audit_semantic_case(case, sample, backend="mpmath")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite nonnegative decimal"):
        audit_semantic_case(case, sample, absolute_tolerance="nan")
    with pytest.raises(ValueError, match="at least one"):
        audit_semantic_case(case, ())
    with pytest.raises(ValueError, match="unknown audit domain"):
        audit_semantic_case(case, sample, domain_mode="unknown")
    with pytest.raises(TypeError, match="CompilerMode"):
        audit_semantic_case(case, sample, compiler_mode="official_v4")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SemanticSample"):
        audit_semantic_case(case, ("not a sample",))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="name/value pair"):
        SemanticSample(label="bad", variables=(("x",),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported audit operator"):
        audit_numeric_operator("unknown", (1,))
    with pytest.raises(TypeError, match="points must be a tuple"):
        audit_numeric_operator("exp", [1])  # type: ignore[arg-type]


def test_integrated_audit_rejects_invalid_backend_collection() -> None:
    case = SEMANTIC_CASE_REGISTRY["one"]

    with pytest.raises(TypeError, match="backends must be a tuple"):
        audit_semantic_construction(case, backends=[NumericBackend.MPMATH])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        audit_semantic_construction(case, backends=())
    with pytest.raises(TypeError, match="every backend"):
        audit_semantic_construction(case, backends=("mpmath",))  # type: ignore[arg-type]


def test_tolerance_and_aggregate_maxima_honor_requested_precision() -> None:
    tolerance = "0.123456789012345678901234567890123456789"
    threshold_case = SemanticCase(
        case_id="precision_threshold",
        operator="one",
        operator_family="source_constant",
        variable_names=(),
        source_evaluator=lambda _values: mp.mpf("1.123456789012345678901234567890123456789"),
        compiler=lambda _variables, _mode: One(),
        default_samples=(_sample("exact threshold"),),
    )
    threshold_audit = audit_semantic_case(
        threshold_case,
        precision_digits=100,
        absolute_tolerance=tolerance,
        relative_tolerance="0",
    )
    assert threshold_audit.results[0].absolute_error == tolerance
    assert threshold_audit.results[0].status is ProbeStatus.PASS

    maxima_audit = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["sin"],
        backend=NumericBackend.MPMATH,
        precision_digits=100,
        absolute_tolerance="1",
        relative_tolerance="1",
    )
    with mp.workdps(100):
        expected = max(
            mp.mpf(row.absolute_error)
            for row in maxima_audit.results
            if row.absolute_error is not None
        )
        assert mp.mpf(maxima_audit.maximum_absolute_error) == expected


def test_numpy_overflow_is_retained_as_overflow() -> None:
    audit = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["exp"],
        (_sample("complex128 overflow", ("x", 1_000)),),
        backend=NumericBackend.NUMPY_COMPLEX128,
        absolute_tolerance="1e-11",
        relative_tolerance="1e-11",
    )
    row = audit.results[0]

    assert row.status is ProbeStatus.OVERFLOW
    assert row.source_value is not None
    assert row.eml_value is not None
    assert "overflow" in (row.message or "").lower()


def test_generic_tan_enforces_certified_interval_without_narrowing_legacy_audit() -> None:
    generic = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["tan"],
        (_sample("outside certified interval", ("x", "1.25")),),
    )
    legacy = audit_numeric_operator("tan", ("1.25",))

    assert generic.results[0].status is ProbeStatus.SOURCE_DOMAIN_ERROR
    assert "[-1, 1]" in (generic.results[0].message or "")
    assert "tan source argument is registry-certified in [-1, 1]" in (
        generic.results[0].assumptions
    )
    assert legacy.results[0].status in {
        ProbeStatus.PASS,
        ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE,
    }
    assert all("[-1, 1]" not in assumption for assumption in legacy.results[0].assumptions)


def test_legacy_numpy_overflow_and_nonfinite_source_keep_precise_statuses() -> None:
    overflow = audit_numeric_operator(
        "sinh",
        (1_000,),
        backend=NumericBackend.NUMPY_COMPLEX128,
    )
    nonfinite = audit_numeric_operator("sin", ("nan",))

    assert overflow.results[0].status is ProbeStatus.OVERFLOW
    assert "overflow" in (overflow.results[0].message or "").lower()
    assert nonfinite.results[0].status is ProbeStatus.SOURCE_NONFINITE


def test_positive_real_family_audit_has_passing_evidence_for_guarded_arithmetic() -> None:
    family_audit = audit_enabled_operator_families(
        domain_mode="positive_real",
        backends=(NumericBackend.MPMATH,),
    )

    for operator in ("subtract", "inverse", "divide"):
        operator_audits = [case for case in family_audit.cases if case.operator == operator]
        assert operator_audits
        assert any(audit.numeric_audits[0].pass_count for audit in operator_audits)


@pytest.mark.parametrize("case_id", ["multiply", "divide", "power_square", "sinh", "tanh"])
def test_zero_point_backend_limitations_are_retained_in_generic_evidence(case_id: str) -> None:
    variables = (("x", 0), ("y", 2)) if case_id in {"multiply", "divide"} else (("x", 0),)
    sample = (_sample("zero edge", *variables),)
    high_precision = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY[case_id],
        sample,
        backend=NumericBackend.MPMATH,
    )
    complex128 = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY[case_id],
        sample,
        backend=NumericBackend.NUMPY_COMPLEX128,
        absolute_tolerance="1e-11",
        relative_tolerance="1e-11",
    )

    assert high_precision.results[0].status is ProbeStatus.EXTENDED_REAL_INTERMEDIATE
    assert complex128.results[0].status is ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE
    assert high_precision.results[0].extended_intermediate
    assert complex128.results[0].extended_intermediate


def test_unexpected_sample_guard_and_source_failures_each_retain_a_row() -> None:
    class Unprintable:
        def __str__(self) -> str:
            raise RuntimeError("cannot render")

    unprintable = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["sin"],
        (_sample("unprintable", ("x", Unprintable())),),
    )
    assert unprintable.results[0].status is ProbeStatus.INVALID_SAMPLE

    def fail_guard(_values: dict[str, mp.mpf], _precision: int) -> str | None:
        raise RuntimeError("guard backend failed")

    guard_case = SemanticCase(
        case_id="guard_failure",
        operator="symbol",
        operator_family="leaf",
        variable_names=("x",),
        source_evaluator=lambda values: values["x"],
        compiler=lambda variables, _mode: variables["x"],
        default_samples=(_sample("guard failure", ("x", 1)),),
        guard=fail_guard,
    )
    guard_audit = audit_semantic_case(guard_case)
    assert guard_audit.results[0].status is ProbeStatus.SOURCE_EVALUATION_ERROR

    def fail_source(_values: dict[str, mp.mpf]) -> mp.mpf:
        raise RuntimeError("source failed")

    source_case = SemanticCase(
        case_id="source_failure",
        operator="symbol",
        operator_family="leaf",
        variable_names=("x",),
        source_evaluator=fail_source,
        compiler=lambda variables, _mode: variables["x"],
        default_samples=(_sample("source failure", ("x", 1)),),
    )
    source_audit = audit_semantic_case(source_case)
    assert source_audit.results[0].status is ProbeStatus.SOURCE_EVALUATION_ERROR
    assert source_audit.requested_sample_count == source_audit.failure_count == 1


def test_sample_is_rendered_once_and_source_overflow_is_retained() -> None:
    class StatefulText:
        def __init__(self) -> None:
            self.calls = 0

        def __str__(self) -> str:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("rendered more than once")
            return "1"

    stateful_value = StatefulText()
    rendered_once = audit_semantic_case(
        SEMANTIC_CASE_REGISTRY["sin"],
        (_sample("stateful string", ("x", stateful_value)),),
    )

    def overflow_source(_values: dict[str, mp.mpf]) -> mp.mpf:
        raise OverflowError("reference overflow")

    overflow_case = SemanticCase(
        case_id="source_overflow",
        operator="symbol",
        operator_family="leaf",
        variable_names=("x",),
        source_evaluator=overflow_source,
        compiler=lambda variables, _mode: variables["x"],
        default_samples=(_sample("source overflow", ("x", 1)),),
    )
    source_overflow = audit_semantic_case(overflow_case)

    assert stateful_value.calls == 1
    assert rendered_once.requested_sample_count == 1
    assert rendered_once.results[0].status in {
        ProbeStatus.PASS,
        ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE,
    }
    assert source_overflow.results[0].status is ProbeStatus.OVERFLOW
    assert source_overflow.requested_sample_count == source_overflow.failure_count == 1


def test_integrated_audit_compiles_once_and_retains_compiler_failure_rows() -> None:
    call_count = 0

    def counted_compiler(_variables: dict[str, object], _mode: CompilerMode) -> One:
        nonlocal call_count
        call_count += 1
        return One()

    stable_case = SemanticCase(
        case_id="compile_once",
        operator="one",
        operator_family="source_constant",
        variable_names=(),
        source_evaluator=lambda _values: mp.mpf(1),
        compiler=counted_compiler,  # type: ignore[arg-type]
        default_samples=(_sample("one"),),
    )
    stable_audit = audit_semantic_construction(stable_case)
    assert call_count == 1
    assert stable_audit.statistics is not None
    assert stable_audit.emitted_sha256 is not None
    assert all(audit.pass_count == 1 for audit in stable_audit.numeric_audits)

    invalid_case = SemanticCase(
        case_id="integrated_compiler_failure",
        operator="one",
        operator_family="source_constant",
        variable_names=(),
        source_evaluator=lambda _values: mp.mpf(1),
        compiler=lambda _variables, _mode: object(),  # type: ignore[return-value]
        default_samples=(_sample("failure"),),
    )
    failed_audit = audit_semantic_construction(invalid_case)
    assert failed_audit.statistics is None
    assert failed_audit.emitted_sha256 is None
    assert failed_audit.evidence_status is EvidenceStatus.COMPLETE_WITH_RETAINED_FAILURES
    assert all(
        numeric.results[0].status is ProbeStatus.COMPILER_ERROR
        for numeric in failed_audit.numeric_audits
    )


def test_symbolic_diagnostics_preserve_stated_formulas_and_case_assumptions() -> None:
    zero = diagnose_symbolic_formula("zero")
    multiply = diagnose_symbolic_formula("multiply")
    half_power = diagnose_symbolic_formula("power", case_id="power_half")

    assert zero.source_formula == "log(1)"
    assert "exp" in multiply.source_formula
    assert "log" in multiply.source_formula
    assert "1/2" in " ".join(half_power.assumptions)
    assert zero.status is multiply.status is half_power.status is SymbolicStatus.EQUAL
