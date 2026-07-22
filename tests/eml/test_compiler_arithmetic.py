"""Exact arithmetic, number, and internal-constant compiler coverage."""

from __future__ import annotations

import cmath
import hashlib
import math
from collections.abc import Callable
from decimal import Decimal

import pytest

from geml.eml.compiler_arithmetic import (
    eml_decimal,
    eml_divide,
    eml_integer,
    eml_multiply,
    eml_power,
    eml_rational,
)
from geml.eml.compiler_constants import (
    eml_internal_e,
    eml_internal_i_branch,
    eml_internal_pi,
)
from geml.eml.emitter import emit_eml
from geml.eml.ir import Variable
from geml.eml.validate import validate_pure_eml
from geml.verification.eml.numeric import NumericBackend, evaluate_pure_eml

_GOLDENS = {
    "zero": "15dbffb4e690501465f83c21bcfd38d5ab6d4360e9dba9267a7837433796fd24",
    "minus_one": "1b17a183a95fd41adde867c0d706d91507a9be2f4713b43dfd1f139cdc77e202",
    "two": "e333418203bc5355f8ab6a72d7e50f59f4bdbf9ef85322934dac1d8178fefb59",
    "three": "e54aa89d6fb434c0f674a7d48e06a75a40451ce5ac460c56647c247e94b7d8c9",
    "half": "43a9ad605bc5cd4869abbedaafa3ebf48b8b1e69b53a522820141d889e154054",
    "multiply": "a90fd0d34a94b1da3b4ba88d7b0f619834cbfdff3e96ae56f04d3a1b8588cd3d",
    "divide": "66f113aaa7e3e7d408ec0dfb8c235f2133f84a8c8c3d3abd12fbe49ccf860bc5",
    "power_two": "96917dba705a9910c8d4ab53b0e80a86b42a805423d58256033f9a9b58662039",
}


def _sha(tree: object) -> str:
    return hashlib.sha256(emit_eml(tree).encode("utf-8")).hexdigest()  # type: ignore[arg-type]


def test_official_exact_arithmetic_fingerprints() -> None:
    x = Variable("x")
    y = Variable("y")
    assert _sha(eml_integer(0)) == _GOLDENS["zero"]
    assert _sha(eml_integer(-1)) == _GOLDENS["minus_one"]
    assert _sha(eml_integer(2)) == _GOLDENS["two"]
    assert _sha(eml_integer(3)) == _GOLDENS["three"]
    assert _sha(eml_rational(1, 2)) == _GOLDENS["half"]
    assert _sha(eml_multiply(x, y)) == _GOLDENS["multiply"]
    assert _sha(eml_divide(x, y)) == _GOLDENS["divide"]
    assert _sha(eml_power(x, eml_integer(2))) == _GOLDENS["power_two"]


def test_decimal_inputs_compile_as_exact_base_ten_rationals() -> None:
    expected = emit_eml(eml_rational(5, 4))
    assert emit_eml(eml_decimal("1.25")) == expected
    assert emit_eml(eml_decimal(1.25)) == expected
    assert emit_eml(eml_decimal(Decimal("1.250"))) == expected


@pytest.mark.parametrize("value", [True, "nan", "inf", "not-a-number"])
def test_invalid_decimal_inputs_fail_explicitly(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        eml_decimal(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("parts", [(1, 0), (0, 2), (2, 4), (1, -2)])
def test_noncanonical_rationals_fail_explicitly(parts: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match=r"denominator|canonical|lowest"):
        eml_rational(*parts)


def test_internal_constants_are_fully_pure_and_do_not_add_source_leaves() -> None:
    for tree in (eml_internal_e(), eml_internal_i_branch(), eml_internal_pi()):
        statistics = validate_pure_eml(tree)
        emitted = emit_eml(tree)
        assert statistics.node_count > 0
        assert set(emitted.replace("EML", "")) <= set("[],1")


@pytest.mark.parametrize(
    ("tree", "variables", "expected"),
    [
        (eml_integer(0), {}, 0),
        (eml_integer(-1), {}, -1),
        (eml_integer(2), {}, 2),
        (eml_integer(3), {}, 3),
        (eml_rational(1, 2), {}, 0.5),
        (eml_multiply(Variable("x"), Variable("y")), {"x": 2, "y": 3}, 6),
        (eml_divide(Variable("x"), Variable("y")), {"x": 2, "y": 4}, 0.5),
        (eml_power(Variable("x"), eml_integer(2)), {"x": 3}, 9),
        (eml_internal_e(), {}, math.e),
        (eml_internal_pi(), {}, math.pi),
    ],
)
def test_arithmetic_and_internal_constants_match_safe_numeric_probes(
    tree: object,
    variables: dict[str, int],
    expected: complex,
) -> None:
    value, _ = evaluate_pure_eml(
        tree,  # type: ignore[arg-type]
        variables=variables,
        backend=NumericBackend.NUMPY_COMPLEX128,
    )
    assert complex(value) == pytest.approx(expected, rel=1e-11, abs=1e-11)


def test_exact_internal_i_convention_and_branch_cut_numeric_diagnostic() -> None:
    reconstructed_log_minus_one = -1j * math.pi
    intended_i = -cmath.exp(reconstructed_log_minus_one / 2)
    intended_pi = intended_i * reconstructed_log_minus_one
    assert intended_i == pytest.approx(1j, rel=1e-15, abs=1e-15)
    assert intended_pi == pytest.approx(math.pi, rel=1e-15, abs=1e-15)

    # Direct floating execution crosses an exact negative-real branch cut.
    # Check only the invariant magnitude and retain the extended-path flag;
    # either sign is numerical branch evidence, not the exact convention.
    for backend in NumericBackend:
        observed, extended = evaluate_pure_eml(
            eml_internal_i_branch(),
            variables={},
            backend=backend,
        )
        assert abs(complex(observed).real) < 1e-10
        assert abs(complex(observed).imag) == pytest.approx(1, rel=1e-10, abs=1e-10)
        assert extended


@pytest.mark.parametrize(
    "builder",
    [
        lambda: eml_integer(0, mode="invalid"),  # type: ignore[arg-type]
        lambda: eml_integer(1, mode="invalid"),  # type: ignore[arg-type]
        lambda: eml_rational(1, 1, mode="invalid"),  # type: ignore[arg-type]
        lambda: eml_decimal("1", mode="invalid"),  # type: ignore[arg-type]
        lambda: eml_internal_i_branch(mode="invalid"),  # type: ignore[arg-type]
    ],
)
def test_public_compilers_reject_invalid_modes_on_early_return_paths(
    builder: Callable[[], object],
) -> None:
    with pytest.raises(TypeError, match="CompilerMode"):
        builder()
