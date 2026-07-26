"""Grammar-v2 structure, purity, domain, and numeric conformance evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

import mpmath as mp
import numpy as np
import pytest

from geml.eml.compiler_arithmetic import (
    eml_divide,
    eml_integer,
    eml_multiply,
    eml_power,
    eml_rational,
)
from geml.eml.compiler_constants import eml_internal_i_branch, eml_internal_pi
from geml.eml.compiler_core import (
    CompilerMode,
    eml_add,
    eml_exp,
    eml_log,
    eml_negate,
    eml_subtract,
)
from geml.eml.compiler_transcendental import HYPERBOLIC_COMPILERS
from geml.eml.compiler_trig import (
    GRAMMAR_V2_COMPILERS,
    TRIG_COMPILERS,
    eml_acos,
    eml_asin,
    eml_atan,
    eml_e,
    eml_pi,
)
from geml.eml.emitter import emit_eml
from geml.eml.ir import EML, EMLTerm, One, Variable
from geml.eml.validate import validate_pure_eml
from geml.spec.domains import (
    DOMAIN_POLICIES,
    GRAMMAR_V2_DOMAIN_REGISTRY,
    GrammarV2InputStatus,
    GrammarVersion,
    classify_grammar_v2_real_input,
)
from geml.spec.operators import (
    GRAMMAR_V2_OPERATOR_OVERRIDES,
    GRAMMAR_V2_OPERATOR_REGISTRY,
    OPERATOR_REGISTRY,
    OPERATORS,
    EMLConstructionStatus,
    GrammarV2OperatorRecord,
)
from geml.verification.eml.numeric import NumericBackend, evaluate_pure_eml

Compiler = Callable[..., EMLTerm]

_EXPECTED = {
    CompilerMode.OFFICIAL_V4: {
        "asin": (
            "965217ec5f3a7ea923d5a057b177af2c95c8954d90e1bfb80d540c1779874e38",
            1969,
            985,
            75,
        ),
        "acos": (
            "302e4cd1a3de8849fafc5a98f0906a398781a772a06013988b34946aa2764160",
            2301,
            1151,
            77,
        ),
        "atan": (
            "af8a08abb86f2944434462a404bd11db30090d60f65c033027fafd318bd65bea",
            659,
            330,
            57,
        ),
        "pi": (
            "81a8cea02c08ae4dbc33f273bebe54337f61769688eccb801ab889c5e8218cd9",
            193,
            97,
            34,
        ),
        "e": (
            "f126a91c1429594f4cf1aa70d70f97c267859d06df9fab7c6904598826a2574c",
            3,
            2,
            1,
        ),
    },
    CompilerMode.CLEAN_NEGATION: {
        "asin": (
            "3464c3900691933037f8d268453c9c7bfea5c65e4e69a3ac3f094835f7d070d5",
            3677,
            1839,
            117,
        ),
        "acos": (
            "2fd2d8b9c9149575b573857b601372b9162d300de93c501cb51c890977b538e3",
            4317,
            2159,
            119,
        ),
        "atan": (
            "726f4f48a0c0f381b2f289ca4c2faa29c3856a95ad3bb49112a6cb67973c917e",
            1331,
            666,
            95,
        ),
        "pi": (
            "05b4e284e85c0aba4aa1f69bdef398ded48a96d415e34285560880b7b2f8931d",
            389,
            195,
            52,
        ),
        "e": (
            "f126a91c1429594f4cf1aa70d70f97c267859d06df9fab7c6904598826a2574c",
            3,
            2,
            1,
        ),
    },
}
_V1_TRANSCENDENTAL_SHA256 = {
    CompilerMode.OFFICIAL_V4: {
        "sin": "d9fa0e691922aee5a0c57ac75e101e63abf9b9ee7f56841ff1820a4aa8cd6571",
        "cos": "1a0c5493b625c1fda4d4e4436e7e4c677eff304e01af1119e5c1c4fca530695f",
        "tan": "20c4f5fa49f4f62955d507c1da34cd5d898e61bda5155f5c3a3e61982dcf45b1",
        "sinh": "888c44fb76f939795e4943b500ad32c8cb223f903abc18bd008eede550213aa2",
        "cosh": "f40e6f5e9d6bc3db56f7b0ebc20df2e747b107acf6a06b83874cf10b8ec2609e",
        "tanh": "03aa8d0795c63db5c86c202342ce4f130a5b6858198fd76ec889cb3f79e0a943",
    },
    CompilerMode.CLEAN_NEGATION: {
        "sin": "d834c494688fdbfa764964a3762c02803afe3d76e26476d20b64d0ef545130a0",
        "cos": "d3b861ffd36f2c027eab1f6be6b7cbe8b3721e767b263f83894f00f7e7ba98b1",
        "tan": "7597f6360dcec6f277eb807fdd88dbab0868320afde639051a35ed1a7541938b",
        "sinh": "2f079ef578337e7fafb1a2633cbcbbfa691688bb6ea647fb92763939f8c50e95",
        "cosh": "33ead62ee5d3a657df52ca36e387561b59f78de1cfc3cb95c46652a8595a04ac",
        "tanh": "3d7ca7a475154e7ed9c8b8139dad855c92666867d101fb9938294830cc91b36a",
    },
}


def _compile(name: str, mode: CompilerMode) -> EMLTerm:
    compiler = GRAMMAR_V2_COMPILERS[name]
    if name in {"e", "pi"}:
        return compiler(grammar_version=GrammarVersion.V2, mode=mode)
    return compiler(Variable("x"), grammar_version=GrammarVersion.V2, mode=mode)


def _literal_half_angle_components(
    value: EMLTerm,
    *,
    mode: CompilerMode,
) -> tuple[EMLTerm, EMLTerm]:
    internal_i = eml_internal_i_branch(mode=mode)
    root_one_plus_x = eml_power(
        eml_add(One(), value, mode=mode),
        eml_rational(1, 2, mode=mode),
        mode=mode,
    )
    root_one_minus_x = eml_power(
        eml_subtract(One(), value),
        eml_rational(1, 2, mode=mode),
        mode=mode,
    )
    numerator = eml_subtract(root_one_plus_x, root_one_minus_x)
    denominator = eml_add(root_one_plus_x, root_one_minus_x, mode=mode)
    imaginary_denominator = eml_multiply(internal_i, denominator, mode=mode)
    argument = eml_negate(
        eml_divide(
            eml_subtract(numerator, imaginary_denominator),
            eml_add(numerator, imaginary_denominator, mode=mode),
            mode=mode,
        ),
        mode=mode,
    )
    return internal_i, argument


def _literal_expected(name: str, mode: CompilerMode) -> EMLTerm:
    value = Variable("x")
    if name == "e":
        return eml_exp(One())
    if name == "pi":
        return eml_internal_pi(mode=mode)
    if name == "atan":
        internal_i = eml_internal_i_branch(mode=mode)
        ratio = eml_negate(
            eml_divide(
                eml_subtract(value, internal_i),
                eml_add(value, internal_i, mode=mode),
                mode=mode,
            ),
            mode=mode,
        )
        coefficient = eml_divide(
            eml_negate(internal_i, mode=mode),
            eml_integer(2, mode=mode),
            mode=mode,
        )
        return eml_multiply(coefficient, eml_log(ratio), mode=mode)

    internal_i, argument = _literal_half_angle_components(value, mode=mode)
    if name == "asin":
        return eml_multiply(
            eml_negate(internal_i, mode=mode),
            eml_log(argument),
            mode=mode,
        )
    half_pi = eml_multiply(
        eml_rational(1, 2, mode=mode),
        eml_internal_pi(mode=mode),
        mode=mode,
    )
    return eml_subtract(
        half_pi,
        eml_multiply(
            eml_negate(internal_i, mode=mode),
            eml_log(argument),
            mode=mode,
        ),
    )


@pytest.mark.parametrize("mode", list(CompilerMode))
@pytest.mark.parametrize("name", ["asin", "acos", "atan", "pi", "e"])
def test_v2_exact_literal_structure_fingerprint_and_purity(
    name: str,
    mode: CompilerMode,
) -> None:
    tree = _compile(name, mode)
    emitted = emit_eml(tree)
    statistics = validate_pure_eml(tree)
    expected_hash, expected_nodes, expected_leaves, expected_depth = _EXPECTED[mode][name]

    assert tree == _literal_expected(name, mode)
    assert hashlib.sha256(emitted.encode("utf-8")).hexdigest() == expected_hash
    assert statistics.node_count == expected_nodes
    assert statistics.edge_count == expected_nodes - 1
    assert statistics.leaf_count == expected_leaves
    assert statistics.operator_count == expected_nodes - expected_leaves
    assert statistics.depth == expected_depth
    assert statistics.reused_object_count == 0
    assert {type(node) for node in _walk(tree)} <= {EML, One, Variable}


def _walk(root: EMLTerm) -> tuple[EMLTerm, ...]:
    nodes: list[EMLTerm] = []
    pending = [root]
    while pending:
        node = pending.pop()
        nodes.append(node)
        if isinstance(node, EML):
            pending.extend((node.right, node.left))
    return tuple(nodes)


def test_v2_activation_is_required_and_v1_remains_the_default() -> None:
    with pytest.raises(TypeError, match="grammar_version"):
        eml_asin(Variable("x"))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="only in explicit grammar v2"):
        eml_asin(Variable("x"), grammar_version=GrammarVersion.V1)
    with pytest.raises(TypeError, match="GrammarVersion"):
        eml_asin(Variable("x"), grammar_version="v2")  # type: ignore[arg-type]

    assert tuple(TRIG_COMPILERS) == ("sin", "cos", "tan")
    assert tuple(GRAMMAR_V2_COMPILERS) == ("asin", "acos", "atan", "pi", "e")
    assert eml_asin(
        Variable("x"),
        grammar_version=GrammarVersion.V2,
    ) == eml_asin(
        Variable("x"),
        grammar_version=GrammarVersion.V2,
        mode=CompilerMode.OFFICIAL_V4,
    )


def test_v1_registry_rows_and_trig_fingerprints_remain_identical() -> None:
    operator_bytes = json.dumps(
        [row.model_dump(mode="json") for row in OPERATORS],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    domain_bytes = json.dumps(
        [row.model_dump(mode="json") for row in DOMAIN_POLICIES],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    assert hashlib.sha256(operator_bytes).hexdigest() == (
        "2fcf9b2efed56014c26f5b45549377adb563e39dc62d88b28ff47ac9ee3546a3"
    )
    assert hashlib.sha256(domain_bytes).hexdigest() == (
        "a812376613429aec30bef4e17ff76890151ffb0ef5ec8f1c5f472d72a4f7a1c6"
    )

    compilers = {**TRIG_COMPILERS, **HYPERBOLIC_COMPILERS}
    for mode in CompilerMode:
        for name, compiler in compilers.items():
            emitted = emit_eml(compiler(Variable("x"), mode=mode))
            assert (
                hashlib.sha256(emitted.encode()).hexdigest()
                == _V1_TRANSCENDENTAL_SHA256[mode][name]
            )


def test_v2_registry_is_a_separate_real_only_overlay() -> None:
    assert tuple(GRAMMAR_V2_OPERATOR_OVERRIDES) == ("asin", "acos", "atan", "pi", "e")
    assert tuple(GRAMMAR_V2_DOMAIN_REGISTRY) == (
        "safe_real",
        "positive_real",
        "nonzero_real",
    )
    assert not OPERATOR_REGISTRY["e"].enabled_for_generation
    assert not OPERATOR_REGISTRY["pi"].enabled_for_generation
    assert (
        OPERATOR_REGISTRY["e"].eml_construction_status is EMLConstructionStatus.PENDING_VERIFICATION
    )
    for name in ("asin", "acos", "atan", "pi", "e"):
        record = GRAMMAR_V2_OPERATOR_REGISTRY[name]
        assert isinstance(record, GrammarV2OperatorRecord)
        assert record.grammar_version is GrammarVersion.V2
        assert record.compiler_modes == ("official_v4", "clean_negation")
        assert not record.enabled_for_generation
        assert record.conformance_approved
        assert record.approval_scope == "bounded_compiler_conformance_only"
        assert record.eml_construction_status is EMLConstructionStatus.APPROVED
        assert "complex" not in record.domain_modes
    assert GRAMMAR_V2_OPERATOR_REGISTRY["imaginary_unit"] is OPERATOR_REGISTRY["imaginary_unit"]
    assert not GRAMMAR_V2_OPERATOR_REGISTRY["imaginary_unit"].enabled_for_generation


def test_v2_scalar_domain_classification_retains_every_boundary_kind() -> None:
    rows = {
        "negative_endpoint": classify_grammar_v2_real_input("asin", "-1"),
        "positive_endpoint": classify_grammar_v2_real_input("acos", "1"),
        "inside": classify_grammar_v2_real_input("asin", "0.999999999999999999"),
        "outside": classify_grammar_v2_real_input("acos", "1.000000000000000001"),
        "negative_zero": classify_grammar_v2_real_input("atan", -0.0),
        "positive_zero": classify_grammar_v2_real_input("atan", 0.0),
        "nonfinite": classify_grammar_v2_real_input("atan", "Infinity"),
        "invalid": classify_grammar_v2_real_input("atan", object()),
    }

    assert rows["negative_endpoint"].status is GrammarV2InputStatus.VALID_ENDPOINT
    assert rows["positive_endpoint"].status is GrammarV2InputStatus.VALID_ENDPOINT
    assert rows["inside"].status is GrammarV2InputStatus.VALID_INTERIOR
    assert rows["outside"].status is GrammarV2InputStatus.INVALID_DOMAIN
    assert rows["negative_zero"].zero_sign == "negative"
    assert rows["positive_zero"].zero_sign == "positive"
    assert rows["nonfinite"].status is GrammarV2InputStatus.NONFINITE
    assert rows["invalid"].status is GrammarV2InputStatus.INVALID_SAMPLE


@pytest.mark.parametrize(
    ("name", "compiler", "point"),
    [
        ("asin", eml_asin, "-0.5"),
        ("asin", eml_asin, "0.5"),
        ("acos", eml_acos, "-0.5"),
        ("acos", eml_acos, "0.5"),
        ("atan", eml_atan, "-10"),
        ("atan", eml_atan, "-0.5"),
        ("atan", eml_atan, "0.5"),
        ("atan", eml_atan, "10"),
    ],
)
def test_official_v2_high_precision_interior_audit(
    name: str,
    compiler: Compiler,
    point: str,
) -> None:
    tree = compiler(Variable("x"), grammar_version=GrammarVersion.V2)
    with mp.workdps(120):
        observed, extended = evaluate_pure_eml(
            tree,
            variables={"x": point},
            backend=NumericBackend.MPMATH,
            precision_digits=100,
        )
        expected = getattr(mp, name)(mp.mpf(point))
        assert abs(observed - expected) < mp.mpf("1e-90")
        assert extended


@pytest.mark.parametrize(("name", "compiler"), [("asin", eml_asin), ("acos", eml_acos)])
@pytest.mark.parametrize("point", ["-1", "1"])
def test_closed_domain_endpoints_are_retained_as_extended_value_rows(
    name: str,
    compiler: Compiler,
    point: str,
) -> None:
    tree = compiler(Variable("x"), grammar_version=GrammarVersion.V2)
    with mp.workdps(120):
        high_precision, extended = evaluate_pure_eml(
            tree,
            variables={"x": point},
            backend=NumericBackend.MPMATH,
            precision_digits=100,
        )
        expected = getattr(mp, name)(mp.mpf(point))
        assert not (mp.isfinite(high_precision.real) and mp.isfinite(high_precision.imag))
        assert extended
        ieee, ieee_extended = evaluate_pure_eml(
            tree,
            variables={"x": float(point)},
            backend=NumericBackend.NUMPY_COMPLEX128,
        )
        assert abs(ieee - complex(expected)) < 1e-12
        assert ieee_extended


def test_zero_values_are_finite_and_signed_zero_collapse_is_explicit() -> None:
    for name, compiler in (("asin", eml_asin), ("acos", eml_acos)):
        tree = compiler(Variable("x"), grammar_version=GrammarVersion.V2)
        with mp.workdps(120):
            observed, extended = evaluate_pure_eml(
                tree,
                variables={"x": "0"},
                backend=NumericBackend.MPMATH,
                precision_digits=100,
            )
            assert abs(observed - getattr(mp, name)(mp.mpf(0))) < mp.mpf("1e-90")
            assert extended

    for compiler in (eml_asin, eml_atan):
        tree = compiler(Variable("x"), grammar_version=GrammarVersion.V2)
        negative, _ = evaluate_pure_eml(
            tree,
            variables={"x": -0.0},
            backend=NumericBackend.NUMPY_COMPLEX128,
        )
        positive, _ = evaluate_pure_eml(
            tree,
            variables={"x": 0.0},
            backend=NumericBackend.NUMPY_COMPLEX128,
        )
        assert abs(negative) < 1e-12
        assert abs(positive) < 1e-12
        assert np.signbit(negative.real) == np.signbit(positive.real)


@pytest.mark.parametrize("mode", list(CompilerMode))
def test_v2_constants_are_real_with_explicit_mode_labels(mode: CompilerMode) -> None:
    e_tree = eml_e(grammar_version=GrammarVersion.V2, mode=mode)
    pi_tree = eml_pi(grammar_version=GrammarVersion.V2, mode=mode)
    with mp.workdps(120):
        observed_e, _ = evaluate_pure_eml(
            e_tree,
            variables={},
            backend=NumericBackend.MPMATH,
            precision_digits=100,
        )
        observed_pi, _ = evaluate_pure_eml(
            pi_tree,
            variables={},
            backend=NumericBackend.MPMATH,
            precision_digits=100,
        )
        assert abs(observed_e - mp.e) < mp.mpf("1e-95")
        assert abs(observed_pi - mp.pi) < mp.mpf("1e-90")
