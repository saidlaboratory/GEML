"""Pinned formula, structural, numeric, and stress evidence for all six compilers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from types import MappingProxyType

import mpmath as mp
import pytest

from geml.eml.compiler_arithmetic import (
    eml_divide,
    eml_integer,
    eml_multiply,
    eml_power,
    eml_rational,
)
from geml.eml.compiler_constants import eml_internal_i_branch
from geml.eml.compiler_core import CompilerMode, eml_add, eml_exp, eml_subtract
from geml.eml.compiler_transcendental import (
    HYPERBOLIC_COMPILERS,
    eml_cosh,
    eml_sinh,
    eml_tanh,
)
from geml.eml.compiler_trig import TRIG_COMPILERS, eml_cos, eml_sin, eml_tan
from geml.eml.emitter import emit_eml
from geml.eml.ir import EML, EMLTerm, One, Variable
from geml.eml.validate import PureEMLValidationError, validate_pure_eml
from geml.spec.operators import OPERATOR_REGISTRY, EMLConstructionStatus
from geml.verification.eml.numeric import NumericBackend, ProbeStatus, audit_numeric_operator
from geml.verification.eml.symbolic import SymbolicStatus, diagnose_symbolic_identity

Compiler = Callable[..., EMLTerm]
_COMPILERS: dict[str, Compiler] = {
    "sin": eml_sin,
    "cos": eml_cos,
    "tan": eml_tan,
    "sinh": eml_sinh,
    "cosh": eml_cosh,
    "tanh": eml_tanh,
}
_CORE_POINTS = (-1, -0.5, 0, 0.5, 1)
_PASS_STATUSES = {ProbeStatus.PASS, ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE}

# Literal issue-owned goldens keep this test independent of the read-only verifier table.
_PINNED_EXPECTATIONS = {
    CompilerMode.OFFICIAL_V4: {
        "sin": (
            "d9fa0e691922aee5a0c57ac75e101e63abf9b9ee7f56841ff1820a4aa8cd6571",
            799,
            400,
            63,
        ),
        "cos": (
            "1a0c5493b625c1fda4d4e4436e7e4c677eff304e01af1119e5c1c4fca530695f",
            687,
            344,
            55,
        ),
        "tan": (
            "20c4f5fa49f4f62955d507c1da34cd5d898e61bda5155f5c3a3e61982dcf45b1",
            1183,
            592,
            75,
        ),
        "sinh": (
            "888c44fb76f939795e4943b500ad32c8cb223f903abc18bd008eede550213aa2",
            171,
            86,
            31,
        ),
        "cosh": (
            "f40e6f5e9d6bc3db56f7b0ebc20df2e747b107acf6a06b83874cf10b8ec2609e",
            187,
            94,
            31,
        ),
        "tanh": (
            "03aa8d0795c63db5c86c202342ce4f130a5b6858198fd76ec889cb3f79e0a943",
            157,
            79,
            28,
        ),
    },
    CompilerMode.CLEAN_NEGATION: {
        "sin": (
            "d834c494688fdbfa764964a3762c02803afe3d76e26476d20b64d0ef545130a0",
            1583,
            792,
            93,
        ),
        "cos": (
            "d3b861ffd36f2c027eab1f6be6b7cbe8b3721e767b263f83894f00f7e7ba98b1",
            1331,
            666,
            81,
        ),
        "tan": (
            "7597f6360dcec6f277eb807fdd88dbab0868320afde639051a35ed1a7541938b",
            2331,
            1166,
            105,
        ),
        "sinh": (
            "2f079ef578337e7fafb1a2633cbcbbfa691688bb6ea647fb92763939f8c50e95",
            311,
            156,
            45,
        ),
        "cosh": (
            "33ead62ee5d3a657df52ca36e387561b59f78de1cfc3cb95c46652a8595a04ac",
            355,
            178,
            45,
        ),
        "tanh": (
            "3d7ca7a475154e7ed9c8b8139dad855c92666867d101fb9938294830cc91b36a",
            297,
            149,
            42,
        ),
    },
}


def _sha256(tree: EMLTerm) -> str:
    return hashlib.sha256(emit_eml(tree).encode("utf-8")).hexdigest()


def _walk(root: EMLTerm) -> tuple[EMLTerm, ...]:
    nodes: list[EMLTerm] = []
    pending = [root]
    while pending:
        node = pending.pop()
        nodes.append(node)
        if isinstance(node, EML):
            pending.extend((node.right, node.left))
    return tuple(nodes)


def _node_ids(root: EMLTerm) -> set[int]:
    return {id(node) for node in _walk(root)}


def _ordered_oscillatory_terms(
    value: EMLTerm,
    *,
    mode: CompilerMode,
) -> tuple[EMLTerm, EMLTerm, EMLTerm, EMLTerm]:
    """Build the pinned terms independently from the trig compiler under test."""

    internal_i = eml_internal_i_branch(mode=mode)
    minus_one = eml_integer(-1, mode=mode)
    positive_exponent = eml_multiply(internal_i, value, mode=mode)
    negative_exponent = eml_multiply(
        eml_multiply(minus_one, internal_i, mode=mode),
        value,
        mode=mode,
    )
    return internal_i, minus_one, eml_exp(positive_exponent), eml_exp(negative_exponent)


def _ordered_formula_constructions(
    value: EMLTerm,
    *,
    mode: CompilerMode,
) -> dict[str, EMLTerm]:
    """Compose all six formulas with the exact pinned child/fold order."""

    internal_i, minus_one, exp_positive, exp_negative = _ordered_oscillatory_terms(
        value,
        mode=mode,
    )
    sine_difference = eml_add(
        eml_multiply(minus_one, exp_negative, mode=mode),
        exp_positive,
        mode=mode,
    )
    sine_coefficient = eml_multiply(
        eml_rational(-1, 2, mode=mode),
        internal_i,
        mode=mode,
    )
    half = eml_rational(1, 2, mode=mode)
    cosine_negative = eml_multiply(half, exp_negative, mode=mode)
    cosine_positive = eml_multiply(half, exp_positive, mode=mode)
    tangent_denominator = eml_add(exp_negative, exp_positive, mode=mode)
    tangent_reciprocal = eml_power(tangent_denominator, minus_one, mode=mode)
    tangent_numerator = eml_add(
        exp_negative,
        eml_multiply(minus_one, exp_positive, mode=mode),
        mode=mode,
    )

    exp_two_value = eml_exp(eml_add(value, value, mode=mode))
    exp_value = eml_exp(value)
    hyperbolic_denominator = eml_multiply(
        eml_integer(2, mode=mode),
        exp_value,
        mode=mode,
    )
    return {
        "sin": eml_multiply(sine_coefficient, sine_difference, mode=mode),
        "cos": eml_add(cosine_negative, cosine_positive, mode=mode),
        "tan": eml_multiply(
            eml_multiply(internal_i, tangent_reciprocal, mode=mode),
            tangent_numerator,
            mode=mode,
        ),
        "sinh": eml_divide(
            eml_subtract(exp_two_value, One()),
            hyperbolic_denominator,
            mode=mode,
        ),
        "cosh": eml_divide(
            eml_add(exp_two_value, One(), mode=mode),
            hyperbolic_denominator,
            mode=mode,
        ),
        "tanh": eml_divide(
            eml_subtract(exp_two_value, One()),
            eml_add(exp_two_value, One(), mode=mode),
            mode=mode,
        ),
    }


def _equivalent_reordered_constructions(
    value: EMLTerm,
    *,
    mode: CompilerMode,
) -> dict[str, EMLTerm]:
    """Build source-equivalent formulas with deliberately different fold order."""

    internal_i, minus_one, exp_positive, exp_negative = _ordered_oscillatory_terms(
        value,
        mode=mode,
    )
    difference = eml_add(
        eml_multiply(minus_one, exp_negative, mode=mode),
        exp_positive,
        mode=mode,
    )
    coefficient = eml_multiply(
        eml_rational(-1, 2, mode=mode),
        internal_i,
        mode=mode,
    )
    half = eml_rational(1, 2, mode=mode)
    negative_term = eml_multiply(half, exp_negative, mode=mode)
    positive_term = eml_multiply(half, exp_positive, mode=mode)
    tangent_denominator = eml_add(exp_positive, exp_negative, mode=mode)
    tangent_reciprocal = eml_power(tangent_denominator, minus_one, mode=mode)
    tangent_numerator = eml_add(
        exp_negative,
        eml_multiply(minus_one, exp_positive, mode=mode),
        mode=mode,
    )

    exp_two_value = eml_exp(eml_add(value, value, mode=mode))
    exp_value = eml_exp(value)
    reversed_hyperbolic_denominator = eml_multiply(
        exp_value,
        eml_integer(2, mode=mode),
        mode=mode,
    )
    return {
        "sin": eml_multiply(difference, coefficient, mode=mode),
        "cos": eml_add(positive_term, negative_term, mode=mode),
        "tan": eml_multiply(
            tangent_numerator,
            eml_multiply(internal_i, tangent_reciprocal, mode=mode),
            mode=mode,
        ),
        "sinh": eml_divide(
            eml_subtract(exp_two_value, One()),
            reversed_hyperbolic_denominator,
            mode=mode,
        ),
        "cosh": eml_divide(
            eml_add(exp_two_value, One(), mode=mode),
            reversed_hyperbolic_denominator,
            mode=mode,
        ),
        "tanh": eml_divide(
            eml_subtract(exp_two_value, One()),
            eml_add(One(), exp_two_value, mode=mode),
            mode=mode,
        ),
    }


@pytest.mark.parametrize("mode", tuple(CompilerMode))
@pytest.mark.parametrize("operator", tuple(_COMPILERS))
def test_literal_pinned_fingerprints_statistics_and_strict_purity(
    operator: str,
    mode: CompilerMode,
) -> None:
    tree = _COMPILERS[operator](Variable("x"), mode=mode)
    expected_digest, expected_nodes, expected_leaves, expected_depth = _PINNED_EXPECTATIONS[mode][
        operator
    ]
    assert _sha256(tree) == expected_digest

    statistics = validate_pure_eml(tree)
    assert (statistics.node_count, statistics.leaf_count, statistics.depth) == (
        expected_nodes,
        expected_leaves,
        expected_depth,
    )
    assert statistics.edge_count == statistics.node_count - 1
    assert statistics.operator_count == statistics.leaf_count - 1
    assert statistics.reused_object_count == 0
    nodes = _walk(tree)
    assert {type(node) for node in nodes} <= {EML, One, Variable}
    assert {node.name for node in nodes if isinstance(node, Variable)} == {"x"}


@pytest.mark.parametrize("mode", tuple(CompilerMode))
def test_complete_formulas_match_the_pinned_operation_order(mode: CompilerMode) -> None:
    value = Variable("x")
    expected = _ordered_formula_constructions(value, mode=mode)
    for operator, compiler in _COMPILERS.items():
        assert compiler(value, mode=mode) == expected[operator]


@pytest.mark.parametrize("mode", tuple(CompilerMode))
def test_source_equivalent_reordering_changes_every_structural_fingerprint(
    mode: CompilerMode,
) -> None:
    value = Variable("x")
    expected = _ordered_formula_constructions(value, mode=mode)
    reordered = _equivalent_reordered_constructions(value, mode=mode)
    for operator in _COMPILERS:
        assert _sha256(reordered[operator]) != _sha256(expected[operator]), operator


@pytest.mark.parametrize("mode", tuple(CompilerMode))
def test_trig_exponent_factor_order_is_structural(mode: CompilerMode) -> None:
    value = Variable("x")
    internal_i = eml_internal_i_branch(mode=mode)
    minus_one = eml_integer(-1, mode=mode)
    _, _, exp_positive, exp_negative = _ordered_oscillatory_terms(value, mode=mode)
    reordered_positive = eml_exp(eml_multiply(value, internal_i, mode=mode))
    reordered_negative = eml_exp(
        eml_multiply(
            value,
            eml_multiply(minus_one, internal_i, mode=mode),
            mode=mode,
        )
    )
    assert _sha256(exp_positive) != _sha256(reordered_positive)
    assert _sha256(exp_negative) != _sha256(reordered_negative)


def test_public_compiler_maps_exactly_follow_the_approved_enabled_registry() -> None:
    approved_by_family = {
        family: {
            name
            for name, record in OPERATOR_REGISTRY.items()
            if record.operator_family == family
            and record.enabled_for_generation
            and record.eml_construction_status is EMLConstructionStatus.APPROVED
        }
        for family in ("trigonometric", "hyperbolic")
    }
    assert isinstance(TRIG_COMPILERS, MappingProxyType)
    assert isinstance(HYPERBOLIC_COMPILERS, MappingProxyType)
    assert tuple(TRIG_COMPILERS) == ("sin", "cos", "tan")
    assert tuple(HYPERBOLIC_COMPILERS) == ("sinh", "cosh", "tanh")
    assert dict(TRIG_COMPILERS) == {
        "sin": eml_sin,
        "cos": eml_cos,
        "tan": eml_tan,
    }
    assert dict(HYPERBOLIC_COMPILERS) == {
        "sinh": eml_sinh,
        "cosh": eml_cosh,
        "tanh": eml_tanh,
    }
    assert set(TRIG_COMPILERS) == approved_by_family["trigonometric"]
    assert set(HYPERBOLIC_COMPILERS) == approved_by_family["hyperbolic"]
    assert set(TRIG_COMPILERS).isdisjoint({"e", "pi", "imaginary_unit"})
    with pytest.raises(TypeError):
        TRIG_COMPILERS["sinh"] = eml_sinh  # type: ignore[index]


def test_source_constants_remain_pending_or_reserved_and_disabled() -> None:
    expected = {
        "e": EMLConstructionStatus.PENDING_VERIFICATION,
        "pi": EMLConstructionStatus.PENDING_VERIFICATION,
        "imaginary_unit": EMLConstructionStatus.RESERVED,
    }
    for name, status in expected.items():
        record = OPERATOR_REGISTRY[name]
        assert record.eml_construction_status is status
        assert not record.enabled_for_generation


@pytest.mark.parametrize("operator", tuple(_COMPILERS))
def test_official_mode_is_default_and_clean_mode_is_explicit(operator: str) -> None:
    compiler = _COMPILERS[operator]
    value = Variable("x")
    default = compiler(value)
    official = compiler(value, mode=CompilerMode.OFFICIAL_V4)
    clean = compiler(value, mode=CompilerMode.CLEAN_NEGATION)
    assert default == official
    assert clean != official
    assert _sha256(default) == _PINNED_EXPECTATIONS[CompilerMode.OFFICIAL_V4][operator][0]
    assert _sha256(clean) == _PINNED_EXPECTATIONS[CompilerMode.CLEAN_NEGATION][operator][0]


@pytest.mark.parametrize("operator", tuple(_COMPILERS))
@pytest.mark.parametrize("mode", tuple(CompilerMode))
def test_public_compilers_preserve_input_and_expand_shared_children(
    operator: str,
    mode: CompilerMode,
) -> None:
    shared = Variable("source_name")
    source = EML(shared, shared)
    source_text = emit_eml(source)
    source_ids = _node_ids(source)
    assert validate_pure_eml(source).reused_object_count == 1

    result = _COMPILERS[operator](source, mode=mode)
    assert emit_eml(source) == source_text
    assert source_ids.isdisjoint(_node_ids(result))
    assert validate_pure_eml(result).reused_object_count == 0
    assert {node.name for node in _walk(result) if isinstance(node, Variable)} == {"source_name"}


@pytest.mark.parametrize("compiler", tuple(_COMPILERS.values()))
@pytest.mark.parametrize("mode", tuple(CompilerMode))
def test_public_compilers_reject_malformed_inputs(
    compiler: Compiler,
    mode: CompilerMode,
) -> None:
    with pytest.raises(PureEMLValidationError, match="forbidden pure EML node type"):
        compiler(object(), mode=mode)


@pytest.mark.parametrize("compiler", tuple(_COMPILERS.values()))
def test_public_compilers_reject_invalid_modes(compiler: Compiler) -> None:
    with pytest.raises(TypeError, match="CompilerMode"):
        compiler(Variable("x"), mode="official_v4")


@pytest.mark.parametrize("operator", tuple(_COMPILERS))
@pytest.mark.parametrize("mode", tuple(CompilerMode))
def test_repeated_compilation_is_deterministic(operator: str, mode: CompilerMode) -> None:
    compiler = _COMPILERS[operator]
    first = emit_eml(compiler(Variable("x"), mode=mode))
    second = emit_eml(compiler(Variable("x"), mode=mode))
    assert first == second


@pytest.mark.parametrize("backend", tuple(NumericBackend))
@pytest.mark.parametrize("operator", tuple(_COMPILERS))
def test_numeric_audit_retains_every_core_probe_and_honest_status(
    operator: str,
    backend: NumericBackend,
) -> None:
    tolerance = "1e-70" if backend is NumericBackend.MPMATH else "1e-11"
    audit = audit_numeric_operator(
        operator,
        _CORE_POINTS,
        backend=backend,
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
    )
    assert audit.requested_sample_count == len(_CORE_POINTS) == len(audit.results)
    assert audit.pass_count + audit.failure_count == len(_CORE_POINTS)
    assert tuple(row.input_value for row in audit.results) == tuple(map(str, _CORE_POINTS))
    assert all(row.operator == operator and row.backend is backend for row in audit.results)
    assert all(row.assumptions and row.method for row in audit.results)

    zero_is_nonfinite = backend is NumericBackend.MPMATH and operator in {"sinh", "tanh"}
    expected_failures = 1 if zero_is_nonfinite else 0
    assert audit.failure_count == expected_failures
    for row in audit.results:
        if zero_is_nonfinite and row.input_value == "0":
            assert row.status is ProbeStatus.EXTENDED_REAL_INTERMEDIATE
            assert row.extended_intermediate
        elif (
            backend is NumericBackend.NUMPY_COMPLEX128
            and operator in {"sinh", "tanh"}
            and row.input_value == "0"
        ):
            assert row.status is ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE
            assert row.extended_intermediate
        else:
            assert row.status in _PASS_STATUSES


@pytest.mark.parametrize("operator", ["sinh", "tanh"])
def test_negative_hyperbolic_sign_regression(operator: str) -> None:
    audit = audit_numeric_operator(operator, (-1, -0.5), backend=NumericBackend.MPMATH)
    assert audit.pass_count == 2
    for row in audit.results:
        assert row.status in _PASS_STATUSES
        assert row.source_value is not None
        assert mp.mpf(row.source_value.real) < 0
        assert row.eml_value is not None
        assert mp.mpf(row.eml_value.real) < 0


def test_cosh_remains_positive_on_the_core_real_probe_grid() -> None:
    audit = audit_numeric_operator("cosh", _CORE_POINTS, backend=NumericBackend.MPMATH)
    assert audit.pass_count == len(_CORE_POINTS)
    for row in audit.results:
        assert row.source_value is not None
        assert mp.mpf(row.source_value.real) > 0
        assert row.eml_value is not None
        assert mp.mpf(row.eml_value.real) > 0


def test_tan_safe_near_pole_and_exact_pole_rows_are_separate() -> None:
    safe = audit_numeric_operator("tan", (-1, 0, 1), backend=NumericBackend.MPMATH)
    assert safe.pass_count == safe.requested_sample_count == 3

    with mp.workdps(100):
        near_pole = mp.nstr(mp.pi / 2 - mp.mpf("1e-8"), 100)
        exact_pole = mp.nstr(mp.pi / 2, 100)
    stress = audit_numeric_operator(
        "tan",
        (near_pole,),
        backend=NumericBackend.MPMATH,
        precision_digits=100,
        absolute_tolerance="1e-60",
        relative_tolerance="1e-60",
    )
    domain_error = audit_numeric_operator(
        "tan",
        (exact_pole,),
        backend=NumericBackend.MPMATH,
        precision_digits=100,
    )
    assert stress.requested_sample_count == 1
    assert stress.results[0].status in _PASS_STATUSES
    assert domain_error.requested_sample_count == domain_error.failure_count == 1
    assert domain_error.results[0].status is ProbeStatus.SOURCE_DOMAIN_ERROR
    assert "pole" in (domain_error.results[0].message or "")


@pytest.mark.parametrize("operator", tuple(_COMPILERS))
def test_symbolic_identity_is_supporting_diagnostic_not_proof(operator: str) -> None:
    diagnostic = diagnose_symbolic_identity(operator)
    assert diagnostic.status is SymbolicStatus.EQUAL
    assert not diagnostic.proof_claimed


@pytest.mark.parametrize("operator", tuple(_COMPILERS))
@pytest.mark.parametrize("mode", tuple(CompilerMode))
def test_compound_child_stress_is_pure_bounded_and_deterministic(
    operator: str,
    mode: CompilerMode,
) -> None:
    compiler = _COMPILERS[operator]
    child = eml_exp(Variable("x"))
    first = compiler(child, mode=mode)
    second = compiler(child, mode=mode)
    statistics = validate_pure_eml(first)
    assert statistics.reused_object_count == 0
    assert emit_eml(first) == emit_eml(second)


def test_clean_atomic_stress_bounds_remain_explicit() -> None:
    for operator, compiler in _COMPILERS.items():
        statistics = validate_pure_eml(compiler(Variable("x"), mode=CompilerMode.CLEAN_NEGATION))
        assert statistics.node_count < 2_500, operator
        assert statistics.depth < 128, operator
