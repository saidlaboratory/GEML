"""Security and coverage tests for authoritative srepr parsing."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.parsing.srepr import (
    ParsedSreprNode,
    ParserLimits,
    SreprParseError,
    UnsupportedNodeError,
    parse_expression_record,
    parse_srepr,
)
from geml.spec.operators import OPERATOR_REGISTRY

_TRIG_HYPERBOLIC_CONSTRUCTORS = ("sin", "cos", "tan", "sinh", "cosh", "tanh")


def _record(sympy_srepr: str) -> ExpressionRecord:
    return ExpressionRecord(
        expression_id="a" * 64,
        sympy_srepr=sympy_srepr,
        display_text="fixture",
        split=CorpusSplit.TRAIN,
        operator_family="fixture",
        domain_mode="safe_real",
        variables=("x",),
        target_ast_size=1,
        target_depth=0,
        generator_seed=1,
        generator_metadata={},
    )


def test_parser_reads_authoritative_contract_field() -> None:
    parsed = parse_expression_record(_record("Add(Symbol('x', real=True), Rational(-3, 5))"))
    assert parsed == ParsedSreprNode(
        constructor="Add",
        children=(
            ParsedSreprNode(
                constructor="Symbol",
                value="x",
                assumptions=(("real", True),),
            ),
            ParsedSreprNode(constructor="Rational", value=(-3, 5)),
        ),
    )


def test_parser_supports_every_expected_syntax_constructor() -> None:
    source = (
        "Add(Mul(Integer(-1), Symbol('x', real=True)), "
        "Pow(exp(log(Rational(3, 2))), Integer(2)), "
        "sin(Symbol('x', real=True)), cos(Symbol('x', real=True)), "
        "tan(Symbol('x', real=True)), sinh(Symbol('x', real=True)), "
        "cosh(Symbol('x', real=True)), tanh(Symbol('x', real=True)))"
    )
    parsed = parse_srepr(source)
    constructors: set[str] = set()

    def visit(node: ParsedSreprNode) -> None:
        constructors.add(node.constructor)
        for child in node.children:
            visit(child)

    visit(parsed)
    assert constructors == {
        "Add",
        "Mul",
        "Integer",
        "Symbol",
        "Pow",
        "exp",
        "log",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "Rational",
    }
    enabled_operators = {
        name for name, operator in OPERATOR_REGISTRY.items() if operator.enabled_for_generation
    }
    assert enabled_operators == {
        "symbol",
        "one",
        "integer",
        "rational",
        "add",
        "subtract",
        "multiply",
        "divide",
        "negate",
        "power",
        "exp",
        "log",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
    }


@pytest.mark.parametrize("constructor", _TRIG_HYPERBOLIC_CONSTRUCTORS)
def test_trig_and_hyperbolic_constructors_preserve_their_argument(constructor: str) -> None:
    parsed = parse_srepr(f"{constructor}(Symbol('x', real=True))")

    assert parsed == ParsedSreprNode(
        constructor=constructor,
        children=(
            ParsedSreprNode(
                constructor="Symbol",
                value="x",
                assumptions=(("real", True),),
            ),
        ),
    )


@pytest.mark.parametrize("constructor", _TRIG_HYPERBOLIC_CONSTRUCTORS)
@pytest.mark.parametrize("arguments", ["", "Symbol('x', real=True), Integer(1)"])
def test_trig_and_hyperbolic_constructors_require_exactly_one_argument(
    constructor: str,
    arguments: str,
) -> None:
    with pytest.raises(SreprParseError, match="exactly one"):
        parse_srepr(f"{constructor}({arguments})")


def test_parser_never_evaluates_input(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist.txt"
    payload = "__import__('pathlib').Path(" + repr(marker.as_posix()) + ").write_text('executed')"
    with pytest.raises(SreprParseError, match="direct constructor"):
        parse_srepr(payload)
    assert not marker.exists()
    assert "eval(" not in inspect.getsource(parse_srepr)


@pytest.mark.parametrize(
    ("source", "constructor"),
    [
        ("erf(Symbol('x', real=True))", "erf"),
        ("Float('1.25')", "Float"),
        ("pi", "pi"),
        ("E", "E"),
        ("I", "I"),
    ],
)
def test_disabled_or_unknown_nodes_fail_explicitly(source: str, constructor: str) -> None:
    with pytest.raises(UnsupportedNodeError) as captured:
        parse_srepr(source)
    assert captured.value.constructor == constructor


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("", "nonblank"),
        ("Add(Symbol('x', real=True))", "at least two"),
        ("Pow(Symbol('x', real=True))", "base and exponent"),
        ("log(Symbol('x', real=True), Integer(2))", "exactly one"),
        ("Integer(True)", "literal base-10 integers"),
        ("Rational(1, 0)", "at least two"),
        ("Rational(1, 1)", "use Integer"),
        ("Rational(2, 4)", "lowest terms"),
        ("Symbol('x')", "must declare"),
        ("Symbol('x', complex=True)", "unsupported Symbol assumption"),
        ("Symbol('x', nonzero=True)", "real domain"),
        ("Add(*[Symbol('x', real=True), Integer(1)])", "starred positional"),
    ],
)
def test_malformed_subset_inputs_are_classified(source: str, message: str) -> None:
    with pytest.raises(SreprParseError, match=message):
        parse_srepr(source)


@pytest.mark.parametrize(
    "source",
    ["Integer(0x10)", "Integer(0b10)", "Integer(1_0)", "Integer(+1)"],
)
def test_integer_payloads_require_canonical_decimal_lexemes(source: str) -> None:
    with pytest.raises(SreprParseError, match="canonical literal base-10"):
        parse_srepr(source)


def test_integer_digit_limit_applies_to_integer_and_rational_payloads() -> None:
    limits = ParserLimits(maximum_integer_digits=3)
    assert parse_srepr("Integer(-999)", limits=limits).value == -999
    assert parse_srepr("Rational(999, 998)", limits=limits).value == (999, 998)
    with pytest.raises(SreprParseError, match="digit limit"):
        parse_srepr("Integer(1000)", limits=limits)
    with pytest.raises(SreprParseError, match="digit limit"):
        parse_srepr("Rational(1, 1009)", limits=limits)


def test_parser_limits_source_nodes_and_depth() -> None:
    source = "exp(exp(exp(Symbol('x', real=True))))"
    with pytest.raises(SreprParseError, match="source-length"):
        parse_srepr(source, limits=ParserLimits(maximum_source_characters=10))
    with pytest.raises(SreprParseError, match="node limit"):
        parse_srepr(source, limits=ParserLimits(maximum_nodes=2))
    with pytest.raises(SreprParseError, match="depth limit"):
        parse_srepr(source, limits=ParserLimits(maximum_depth=2))


@pytest.mark.parametrize("invalid_limit", [True, 0, -1, 1.5])
def test_parser_limits_are_strict_positive_integers(invalid_limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ParserLimits(maximum_nodes=invalid_limit)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        ParserLimits(maximum_integer_digits=invalid_limit)  # type: ignore[arg-type]
