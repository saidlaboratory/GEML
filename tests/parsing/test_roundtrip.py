"""Rendering, authority-boundary, and audit tests for issue 1-7 adapters."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest
from sympy import Add, Basic, Integer, Pow, Symbol, cos, cosh, erf, sin, sinh, tan, tanh

from geml.ast.builder import build_ast_from_parsed
from geml.contracts.ast import ASTEdge, ASTNode, ASTStatistics, ASTTree
from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.parsing import display, latex, roundtrip
from geml.parsing.display import (
    DISPLAY_SOURCE_OPERATORS,
    MalformedDisplayTreeError,
    UnsupportedDisplayNodeError,
    render_display,
)
from geml.parsing.latex import (
    LATEX_SOURCE_OPERATORS,
    MalformedLatexTreeError,
    UnsupportedLatexNodeError,
    render_latex,
)
from geml.parsing.roundtrip import (
    RoundTripComparisonError,
    RoundTripStatus,
    audit_latex_roundtrip,
    audit_source_roundtrip,
    compare_ast_structures,
)
from geml.parsing.srepr import parse_srepr
from geml.spec.operators import OPERATOR_REGISTRY, EMLConstructionStatus

TRIG_HYPERBOLIC_FUNCTIONS: dict[str, Callable[..., Basic]] = {
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "sinh": sinh,
    "cosh": cosh,
    "tanh": tanh,
}


def _tree(source: str, *, expression_id: str = "fixture") -> ASTTree:
    return build_ast_from_parsed(parse_srepr(source), expression_id=expression_id)


def _symbol(name: str) -> str:
    return f"Symbol({name!r}, real=True)"


def _record(source: str) -> ExpressionRecord:
    return ExpressionRecord(
        expression_id="a" * 64,
        sympy_srepr=source,
        display_text="non-authoritative fixture",
        latex_text=None,
        split=CorpusSplit.TRAIN,
        operator_family="fixture",
        domain_mode="safe_real",
        variables=("x",),
        target_ast_size=1,
        target_depth=0,
        generator_seed=1,
        generator_metadata={},
    )


SUPPORTED_OPERATOR_FIXTURES = {
    "symbol": "Symbol('x', real=True)",
    "one": "Integer(1)",
    "integer": "Integer(2)",
    "rational": "Rational(3, 5)",
    "add": "Add(Symbol('x', real=True), Symbol('y', real=True))",
    "subtract": ("Add(Symbol('x', real=True), Mul(Integer(-1), Symbol('y', real=True)))"),
    "multiply": "Mul(Symbol('x', real=True), Symbol('y', real=True))",
    "divide": ("Mul(Symbol('x', real=True), Pow(Symbol('y', real=True), Integer(-1)))"),
    "negate": "Mul(Integer(-1), Symbol('x', real=True))",
    "power": "Pow(Symbol('x', real=True), Integer(2))",
    "exp": "exp(Symbol('x', real=True))",
    "log": "log(Symbol('x', real=True))",
    "sin": "sin(Symbol('x', real=True))",
    "cos": "cos(Symbol('x', real=True))",
    "tan": "tan(Symbol('x', real=True))",
    "sinh": "sinh(Symbol('x', real=True))",
    "cosh": "cosh(Symbol('x', real=True))",
    "tanh": "tanh(Symbol('x', real=True))",
}


def _approved_enabled_operators() -> set[str]:
    return {
        name
        for name, operator in OPERATOR_REGISTRY.items()
        if operator.enabled_for_generation
        and operator.eml_construction_status is EMLConstructionStatus.APPROVED
    }


def test_renderer_coverage_tracks_supported_and_approved_source_operators() -> None:
    approved = _approved_enabled_operators()
    supported = set(SUPPORTED_OPERATOR_FIXTURES)
    assert approved == supported
    assert supported == DISPLAY_SOURCE_OPERATORS
    assert supported == LATEX_SOURCE_OPERATORS
    for source in SUPPORTED_OPERATOR_FIXTURES.values():
        tree = _tree(source)
        assert render_display(tree)
        assert render_latex(tree)


def test_other_pending_and_reserved_operators_are_not_falsely_covered() -> None:
    unsupported = set(OPERATOR_REGISTRY) - _approved_enabled_operators()
    assert unsupported.isdisjoint(DISPLAY_SOURCE_OPERATORS)
    assert unsupported.isdisjoint(LATEX_SOURCE_OPERATORS)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Symbol('x', real=True)", "x"),
        ("Integer(1)", "1"),
        ("Integer(-7)", "-7"),
        ("Rational(-3, 5)", "-3/5"),
        ("Add(Symbol('x', real=True), Symbol('y', real=True))", "x + y"),
        (
            "Add(Symbol('x', real=True), Mul(Integer(-1), Symbol('y', real=True)))",
            "x - y",
        ),
        ("Mul(Symbol('x', real=True), Symbol('y', real=True))", "x * y"),
        (
            "Mul(Symbol('x', real=True), Pow(Symbol('y', real=True), Integer(-1)))",
            "x / y",
        ),
        ("Mul(Integer(-1), Symbol('x', real=True))", "-x"),
        ("Pow(Symbol('x', real=True), Integer(2))", "x**2"),
        ("exp(Symbol('x', real=True))", "exp(x)"),
        ("log(Symbol('x', real=True))", "log(x)"),
        ("sin(Symbol('x', real=True))", "sin(x)"),
        ("cos(Symbol('x', real=True))", "cos(x)"),
        ("tan(Symbol('x', real=True))", "tan(x)"),
        ("sinh(Symbol('x', real=True))", "sinh(x)"),
        ("cosh(Symbol('x', real=True))", "cosh(x)"),
        ("tanh(Symbol('x', real=True))", "tanh(x)"),
    ],
)
def test_display_operator_forms(source: str, expected: str) -> None:
    assert render_display(_tree(source)) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "Mul(Add(Symbol('x', real=True), Symbol('y', real=True)), Symbol('z', real=True))",
            "(x + y) * z",
        ),
        (
            "Mul(Symbol('x', real=True), Add(Symbol('y', real=True), Symbol('z', real=True)))",
            "x * (y + z)",
        ),
        ("Mul(Integer(-1), Add(Symbol('x', real=True), Symbol('y', real=True)))", "-(x + y)"),
        (
            "Mul(Symbol('x', real=True), "
            "Pow(Mul(Symbol('y', real=True), Symbol('z', real=True)), Integer(-1)))",
            "x / (y * z)",
        ),
        (
            "Pow(Pow(Symbol('x', real=True), Symbol('y', real=True)), Symbol('z', real=True))",
            "(x**y)**z",
        ),
        (
            "Pow(Symbol('x', real=True), Pow(Symbol('y', real=True), Symbol('z', real=True)))",
            "x**y**z",
        ),
        ("exp(Add(Symbol('x', real=True), Symbol('y', real=True)))", "exp(x + y)"),
        ("log(Mul(Symbol('x', real=True), Symbol('y', real=True)))", "log(x * y)"),
        (
            "Add(Symbol('x', real=True), Add(Symbol('y', real=True), Symbol('z', real=True)))",
            "x + (y + z)",
        ),
    ],
)
def test_display_precedence_and_associativity(source: str, expected: str) -> None:
    tree = _tree(source)
    assert render_display(tree) == expected
    assert render_display(tree) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Symbol('x', real=True)", "x"),
        ("Integer(1)", "1"),
        ("Integer(-7)", "-7"),
        ("Rational(-3, 5)", r"-\frac{3}{5}"),
        ("Add(Symbol('x', real=True), Symbol('y', real=True))", "x + y"),
        (
            "Add(Symbol('x', real=True), Mul(Integer(-1), Symbol('y', real=True)))",
            "x - y",
        ),
        ("Mul(Symbol('x', real=True), Symbol('y', real=True))", r"x \cdot y"),
        (
            "Mul(Symbol('x', real=True), Pow(Symbol('y', real=True), Integer(-1)))",
            r"\frac{x}{y}",
        ),
        ("Mul(Integer(-1), Symbol('x', real=True))", "-x"),
        ("Pow(Symbol('x', real=True), Integer(2))", "x^{2}"),
        ("exp(Symbol('x', real=True))", r"\exp\left(x\right)"),
        ("log(Symbol('x', real=True))", r"\log\left(x\right)"),
        ("sin(Symbol('x', real=True))", r"\sin\left(x\right)"),
        ("cos(Symbol('x', real=True))", r"\cos\left(x\right)"),
        ("tan(Symbol('x', real=True))", r"\tan\left(x\right)"),
        ("sinh(Symbol('x', real=True))", r"\sinh\left(x\right)"),
        ("cosh(Symbol('x', real=True))", r"\cosh\left(x\right)"),
        ("tanh(Symbol('x', real=True))", r"\tanh\left(x\right)"),
    ],
)
def test_latex_operator_forms(source: str, expected: str) -> None:
    assert render_latex(_tree(source)) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "Mul(Add(Symbol('x', real=True), Symbol('y', real=True)), Symbol('z', real=True))",
            r"\left(x + y\right) \cdot z",
        ),
        (
            "Mul(Symbol('x', real=True), Add(Symbol('y', real=True), Symbol('z', real=True)))",
            r"x \cdot \left(y + z\right)",
        ),
        (
            "Mul(Integer(-1), Add(Symbol('x', real=True), Symbol('y', real=True)))",
            r"-\left(x + y\right)",
        ),
        (
            "Mul(Symbol('x', real=True), "
            "Pow(Mul(Symbol('y', real=True), Symbol('z', real=True)), Integer(-1)))",
            r"\frac{x}{y \cdot z}",
        ),
        (
            "Pow(Pow(Symbol('x', real=True), Symbol('y', real=True)), Symbol('z', real=True))",
            r"\left(x^{y}\right)^{z}",
        ),
        (
            "Pow(Symbol('x', real=True), Pow(Symbol('y', real=True), Symbol('z', real=True)))",
            "x^{y^{z}}",
        ),
        (
            "exp(Add(Symbol('x', real=True), Symbol('y', real=True)))",
            r"\exp\left(x + y\right)",
        ),
        (
            "log(Mul(Symbol('x', real=True), Symbol('y', real=True)))",
            r"\log\left(x \cdot y\right)",
        ),
    ],
)
def test_latex_grouping_is_stable(source: str, expected: str) -> None:
    tree = _tree(source)
    assert render_latex(tree) == expected
    assert render_latex(tree) == expected


def test_symbol_views_escape_names_without_reinterpreting_them() -> None:
    name = "rate_#%&{}\\"
    tree = _tree(_symbol(name))
    assert render_display(tree) == 'Symbol("rate_#%&{}\\\\")'
    assert render_latex(tree) == r"\mathrm{rate\_\#\%\&\{\}\backslash{}}"


def test_latex_symbol_whitespace_and_control_characters_are_unambiguous() -> None:
    simple = render_latex(_tree(_symbol("ab")))
    complex_name = render_latex(_tree(_symbol("a b\t\n^~")))
    assert simple == r"\mathrm{ab}"
    assert complex_name == (
        r"\mathrm{a\langle\mathtt{U+0020}\rangle"
        r"b\langle\mathtt{U+0009}\rangle"
        r"\langle\mathtt{U+000A}\rangle"
        r"\langle\mathtt{U+005E}\rangle"
        r"\langle\mathtt{U+007E}\rangle}"
    )
    assert complex_name != simple


def test_renderers_follow_child_slots_independently_of_edge_record_order() -> None:
    tree = _tree("Pow(Add(Symbol('x', real=True), Integer(1)), Integer(2))")
    reordered_edges = ASTTree(
        expression_id=tree.expression_id,
        root_id=tree.root_id,
        nodes=tree.nodes,
        edges=tuple(reversed(tree.edges)),
        statistics=tree.statistics,
    )
    assert render_display(reordered_edges) == "(x + 1)**2"
    assert render_latex(reordered_edges) == r"\left(x + 1\right)^{2}"


def test_renderers_do_not_delegate_to_simplification_or_string_authority() -> None:
    renderer_source = inspect.getsource(display) + inspect.getsource(latex)
    assert "simplify(" not in renderer_source
    assert "sympy_srepr" not in renderer_source


def _unsupported_tree() -> ASTTree:
    node = ASTNode(node_id="n0", node_kind="operator", label="erf", arity=1)
    child = ASTNode(
        node_id="n1",
        node_kind="leaf",
        label="symbol",
        arity=0,
        value={"name": "x", "assumptions": {"real": True}},
    )
    return ASTTree(
        expression_id="fixture",
        root_id="n0",
        nodes=(node, child),
        edges=(ASTEdge(source_id="n0", target_id="n1", child_slot=0),),
        statistics=ASTStatistics(
            node_count=2,
            edge_count=1,
            leaf_count=1,
            operator_count=1,
            depth=1,
        ),
    )


def test_unsupported_nodes_raise_typed_renderer_errors() -> None:
    tree = _unsupported_tree()
    with pytest.raises(UnsupportedDisplayNodeError, match="erf"):
        render_display(tree)
    with pytest.raises(UnsupportedLatexNodeError, match="erf"):
        render_latex(tree)


def test_malformed_nodes_and_non_ast_inputs_raise_typed_errors() -> None:
    malformed_node = ASTNode.model_construct(
        node_id="n0",
        node_kind="leaf",
        label="symbol",
        arity=0,
        value=None,
        metadata={},
    )
    malformed_tree = ASTTree.model_construct(
        expression_id="fixture",
        root_id="n0",
        nodes=(malformed_node,),
        edges=(),
        statistics=ASTStatistics(
            node_count=1,
            edge_count=0,
            leaf_count=1,
            operator_count=0,
            depth=0,
        ),
    )
    with pytest.raises(MalformedDisplayTreeError, match="symbol value"):
        render_display(malformed_tree)
    with pytest.raises(MalformedLatexTreeError, match="symbol value"):
        render_latex(malformed_tree)
    with pytest.raises(MalformedDisplayTreeError, match="ASTTree"):
        render_display("x")  # type: ignore[arg-type]
    with pytest.raises(MalformedLatexTreeError, match="ASTTree"):
        render_latex("x")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "malformation",
    ["missing_fields", "invalid_label_type", "wrong_arity", "operator_value"],
)
def test_known_malformed_node_shapes_raise_typed_errors(malformation: str) -> None:
    child = ASTNode(
        node_id="n1",
        node_kind="leaf",
        label="symbol",
        arity=0,
        value={"name": "x", "assumptions": {"real": True}},
    )
    if malformation in {"missing_fields", "invalid_label_type"}:
        root_fields: dict[str, object] = {
            "node_id": "n0",
            "arity": 0,
            "value": None,
            "metadata": {},
        }
        if malformation == "invalid_label_type":
            root_fields.update(node_kind="leaf", label=["bad"])
        root = ASTNode.model_construct(**root_fields)
        nodes = (root,)
        edges: tuple[ASTEdge, ...] = ()
        statistics = ASTStatistics(
            node_count=1,
            edge_count=0,
            leaf_count=1,
            operator_count=0,
            depth=0,
        )
    else:
        root = ASTNode(
            node_id="n0",
            node_kind="operator",
            label="add",
            arity=1 if malformation == "wrong_arity" else 2,
            value="structural-payload" if malformation == "operator_value" else None,
        )
        second_child = ASTNode(
            node_id="n2",
            node_kind="leaf",
            label="one",
            arity=0,
            value=1,
        )
        nodes = (root, child) if root.arity == 1 else (root, child, second_child)
        edges = (ASTEdge(source_id="n0", target_id="n1", child_slot=0),)
        if root.arity == 2:
            edges += (ASTEdge(source_id="n0", target_id="n2", child_slot=1),)
        statistics = ASTStatistics(
            node_count=len(nodes),
            edge_count=len(edges),
            leaf_count=len(nodes) - 1,
            operator_count=1,
            depth=1,
        )
    tree = ASTTree.model_construct(
        expression_id="fixture",
        root_id="n0",
        nodes=nodes,
        edges=edges,
        statistics=statistics,
    )
    with pytest.raises(MalformedDisplayTreeError):
        render_display(tree)
    with pytest.raises(MalformedLatexTreeError):
        render_latex(tree)


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("one", 2),
        ("integer", True),
        ("rational", {"numerator": 1}),
        ("symbol", {"name": "x", "assumptions": {}}),
    ],
)
def test_comparison_rejects_malformed_known_leaf_payloads(
    label: str,
    value: object,
) -> None:
    node = ASTNode(node_id="n0", node_kind="leaf", label=label, arity=0, value=value)
    tree = ASTTree(
        expression_id="fixture",
        root_id="n0",
        nodes=(node,),
        statistics=ASTStatistics(
            node_count=1,
            edge_count=0,
            leaf_count=1,
            operator_count=0,
            depth=0,
        ),
    )
    with pytest.raises(RoundTripComparisonError):
        compare_ast_structures(tree, tree)


def test_authoritative_source_roundtrip_is_exact() -> None:
    result = audit_source_roundtrip(
        _record("Pow(Add(Symbol('x', real=True), Integer(1)), Rational(3, 2))")
    )
    assert result.status is RoundTripStatus.EXACT
    assert result.exact_structural_equal is True
    assert result.commutative_normalized_equal is True
    assert result.semantic_equal is None
    assert result.original_signature == result.roundtrip_signature
    assert result.latex_text is None


@pytest.mark.parametrize("constructor", TRIG_HYPERBOLIC_FUNCTIONS)
def test_trig_and_hyperbolic_source_roundtrips_are_exact(constructor: str) -> None:
    result = audit_source_roundtrip(_record(f"{constructor}(Symbol('x', real=True))"))

    assert result.status is RoundTripStatus.EXACT
    assert result.exact_structural_equal is True
    assert result.parse_supported is None


def test_source_roundtrip_can_compare_an_expected_ast() -> None:
    record = _record("Add(Symbol('x', real=True), Integer(1))")
    expected = _tree("Add(Symbol('x', real=True), Integer(2))")
    result = audit_source_roundtrip(record, expected_tree=expected)
    assert result.status is RoundTripStatus.COMPARISON_INDETERMINATE
    assert result.exact_structural_equal is False
    assert result.commutative_normalized_equal is False
    assert result.semantic_equal is None


def test_source_roundtrip_retains_parse_failures() -> None:
    result = audit_source_roundtrip(_record("erf(Symbol('x', real=True))"))
    assert result.status is RoundTripStatus.OPERATOR_UNSUPPORTED
    assert result.parse_supported is False
    assert result.error_type == "UnsupportedNodeError"
    assert "erf" in (result.error_message or "")


def test_comparison_distinguishes_commutative_normalization() -> None:
    original = _tree("Add(Symbol('x', real=True), Symbol('y', real=True))")
    reordered = _tree("Add(Symbol('y', real=True), Symbol('x', real=True))")
    result = compare_ast_structures(original, reordered)
    assert result.status is RoundTripStatus.COMMUTATIVE_NORMALIZED
    assert result.exact_structural_equal is False
    assert result.commutative_normalized_equal is True
    assert result.semantic_equal is True
    assert result.normalization_detected is True


def test_comparison_distinguishes_semantic_only_equality() -> None:
    power = _tree("Pow(Symbol('x', real=True), Integer(2))")
    product = _tree("Mul(Symbol('x', real=True), Symbol('x', real=True))")
    result = compare_ast_structures(power, product)
    assert result.status is RoundTripStatus.SEMANTICALLY_EQUAL
    assert result.exact_structural_equal is False
    assert result.commutative_normalized_equal is False
    assert result.semantic_equal is True
    assert result.normalization_detected is True


@pytest.mark.parametrize(
    ("original", "reordered"),
    [
        (
            "Pow(Symbol('x', real=True), Symbol('y', real=True))",
            "Pow(Symbol('y', real=True), Symbol('x', real=True))",
        ),
        (
            "Mul(Symbol('x', real=True), Pow(Symbol('y', real=True), Integer(-1)))",
            "Mul(Pow(Symbol('y', real=True), Integer(-1)), Symbol('x', real=True))",
        ),
        (
            "Add(Symbol('x', real=True), Mul(Integer(-1), Symbol('y', real=True)))",
            "Add(Mul(Integer(-1), Symbol('y', real=True)), Symbol('x', real=True))",
        ),
    ],
)
def test_commutative_policy_does_not_reorder_power_division_or_subtraction(
    original: str,
    reordered: str,
) -> None:
    result = compare_ast_structures(_tree(original), _tree(reordered))
    assert result.exact_structural_equal is False
    assert result.commutative_normalized_equal is False


def test_commutative_policy_preserves_binary_association() -> None:
    left_grouped = _tree(
        "Add(Add(Symbol('x', real=True), Symbol('y', real=True)), Symbol('z', real=True))"
    )
    right_grouped = _tree(
        "Add(Symbol('x', real=True), Add(Symbol('y', real=True), Symbol('z', real=True)))"
    )
    result = compare_ast_structures(left_grouped, right_grouped)
    assert result.status is RoundTripStatus.SEMANTICALLY_EQUAL
    assert result.exact_structural_equal is False
    assert result.commutative_normalized_equal is False
    assert result.semantic_equal is True


def test_structural_comparison_handles_deep_valid_asts_iteratively() -> None:
    depth = 1_100
    nodes = tuple(
        ASTNode(
            node_id=f"n{index}",
            node_kind="operator" if index < depth else "leaf",
            label="exp" if index < depth else "symbol",
            arity=1 if index < depth else 0,
            value=(None if index < depth else {"name": "x", "assumptions": {"real": True}}),
        )
        for index in range(depth + 1)
    )
    edges = tuple(
        ASTEdge(source_id=f"n{index}", target_id=f"n{index + 1}", child_slot=0)
        for index in range(depth)
    )
    tree = ASTTree(
        expression_id="deep-fixture",
        root_id="n0",
        nodes=nodes,
        edges=edges,
        statistics=ASTStatistics(
            node_count=depth + 1,
            edge_count=depth,
            leaf_count=1,
            operator_count=depth,
            depth=depth,
        ),
    )
    result = compare_ast_structures(tree, tree, compare_semantics=False)
    assert result.status is RoundTripStatus.EXACT
    assert result.exact_structural_equal is True
    assert result.commutative_normalized_equal is True


def test_comparison_reports_non_equal_and_indeterminate_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    one = _tree("Add(Symbol('x', real=True), Integer(1))")
    two = _tree("Add(Symbol('x', real=True), Integer(2))")
    unequal = compare_ast_structures(one, two)
    monkeypatch.setattr(roundtrip, "_semantic_equality", lambda _left, _right: (None, None))
    indeterminate = compare_ast_structures(one, two)
    assert unequal.status is RoundTripStatus.NOT_EQUAL
    assert unequal.semantic_equal is False
    assert unequal.normalization_detected is False
    assert indeterminate.status is RoundTripStatus.COMPARISON_INDETERMINATE
    assert indeterminate.semantic_equal is None


def test_latex_roundtrip_reports_unavailable_optional_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        roundtrip,
        "_load_latex_backend",
        lambda: roundtrip._LatexBackend(
            None,
            "ImportError",
            "ANTLR 4.11 runtime is not installed",
        ),
    )
    result = audit_latex_roundtrip(_tree("Pow(Symbol('x', real=True), Integer(2))"))
    assert result.status is RoundTripStatus.PARSER_UNAVAILABLE
    assert result.latex_text == "x^{2}"
    assert result.parser_available is False
    assert result.parse_supported is True
    assert result.error_type == "ImportError"
    assert "ANTLR" in (result.error_message or "")
    assert result.exact_structural_equal is None


def test_latex_roundtrip_can_report_exact_structural_and_semantic_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def parser(_latex: str) -> Pow:
        return Pow(Symbol("x"), Integer(2), evaluate=False)

    monkeypatch.setattr(
        roundtrip,
        "_load_latex_backend",
        lambda: roundtrip._LatexBackend(parser),
    )
    result = audit_latex_roundtrip(_tree("Pow(Symbol('x', real=True), Integer(2))"))
    assert result.status is RoundTripStatus.EXACT
    assert result.parser_available is True
    assert result.parse_supported is True
    assert result.exact_structural_equal is True
    assert result.semantic_equal is True


@pytest.mark.parametrize(("constructor", "function"), TRIG_HYPERBOLIC_FUNCTIONS.items())
def test_trig_and_hyperbolic_ast_labels_map_to_the_matching_sympy_function(
    constructor: str,
    function: Callable[..., Basic],
) -> None:
    expression = roundtrip._ast_to_sympy(_tree(f"{constructor}(Symbol('x', real=True))"))
    assert expression.func is function


@pytest.mark.parametrize(("constructor", "function"), TRIG_HYPERBOLIC_FUNCTIONS.items())
def test_trig_and_hyperbolic_latex_roundtrips_support_semantic_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    constructor: str,
    function: Callable[..., Basic],
) -> None:
    def parser(_latex: str) -> Basic:
        return function(Symbol("x"), evaluate=False)

    monkeypatch.setattr(
        roundtrip,
        "_load_latex_backend",
        lambda: roundtrip._LatexBackend(parser),
    )
    result = audit_latex_roundtrip(_tree(f"{constructor}(Symbol('x', real=True))"))

    assert result.status is RoundTripStatus.EXACT
    assert result.parser_available is True
    assert result.parse_supported is True
    assert result.exact_structural_equal is True
    assert result.semantic_equal is True


def test_latex_roundtrip_allows_normalization_to_eliminate_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def parser(_latex: str) -> Integer:
        return Integer(0)

    monkeypatch.setattr(
        roundtrip,
        "_load_latex_backend",
        lambda: roundtrip._LatexBackend(parser),
    )
    tree = _tree("Add(Symbol('x', real=True), Mul(Integer(-1), Symbol('x', real=True)))")
    result = audit_latex_roundtrip(tree)
    assert result.status is RoundTripStatus.SEMANTICALLY_EQUAL
    assert result.exact_structural_equal is False
    assert result.commutative_normalized_equal is False
    assert result.semantic_equal is True
    assert result.normalization_detected is True


def test_latex_roundtrip_retains_semantic_comparison_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def parser(_latex: str) -> Integer:
        return Integer(0)

    monkeypatch.setattr(
        roundtrip,
        "_load_latex_backend",
        lambda: roundtrip._LatexBackend(parser),
    )
    monkeypatch.setattr(
        roundtrip,
        "_semantic_equality",
        lambda _left, _right: (None, RuntimeError("semantic comparison failed")),
    )
    result = audit_latex_roundtrip(_tree("Symbol('x', real=True)"))
    assert result.status is RoundTripStatus.COMPARISON_INDETERMINATE
    assert result.semantic_equal is None
    assert result.error_type == "RuntimeError"
    assert result.error_message == "semantic comparison failed"


def test_latex_roundtrip_retains_unsupported_and_malformed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsupported_name = audit_latex_roundtrip(_tree(_symbol("velocity")))
    assert unsupported_name.status is RoundTripStatus.OPERATOR_UNSUPPORTED
    assert unsupported_name.parser_available is None
    assert unsupported_name.parse_supported is False
    assert "single ASCII-letter" in (unsupported_name.error_message or "")

    def malformed_parser(_latex: str) -> Add:
        raise ValueError("malformed LaTeX fixture")

    monkeypatch.setattr(
        roundtrip,
        "_load_latex_backend",
        lambda: roundtrip._LatexBackend(malformed_parser),
    )
    parse_error = audit_latex_roundtrip(_tree("Symbol('x', real=True)"))
    assert parse_error.status is RoundTripStatus.PARSE_ERROR
    assert parse_error.parser_available is True
    assert parse_error.parse_supported is True
    assert parse_error.error_type == "ValueError"
    assert parse_error.error_message == "malformed LaTeX fixture"
    assert parse_error.latex_text == "x"


def test_latex_roundtrip_does_not_silently_accept_unsupported_parser_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def parser(_latex: str) -> Basic:
        return erf(Symbol("x"))

    monkeypatch.setattr(
        roundtrip,
        "_load_latex_backend",
        lambda: roundtrip._LatexBackend(parser),
    )
    result = audit_latex_roundtrip(_tree("Symbol('x', real=True)"))
    assert result.status is RoundTripStatus.OPERATOR_UNSUPPORTED
    assert result.parse_supported is False
    assert result.error_type == "UnsupportedNodeError"
    assert "erf" in (result.error_message or "")


def test_views_and_audits_preserve_authority_metrics_identity_and_files(tmp_path: Path) -> None:
    tree = _tree("Mul(Add(Symbol('x', real=True), Integer(1)), Integer(2))", expression_id="id")
    original_dump = tree.model_dump(mode="json")
    before = tuple(tmp_path.iterdir())

    render_display(tree)
    render_latex(tree)
    result = audit_source_roundtrip(
        _record("Mul(Add(Symbol('x', real=True), Integer(1)), Integer(2))")
    )

    assert tree.model_dump(mode="json") == original_dump
    assert tree.expression_id == "id"
    assert tree.statistics == ASTStatistics(
        node_count=5,
        edge_count=4,
        leaf_count=3,
        operator_count=2,
        depth=2,
    )
    assert tuple(tmp_path.iterdir()) == before
    assert "sympy_srepr" not in {field.name for field in fields(result)}
    with pytest.raises(FrozenInstanceError):
        result.status = RoundTripStatus.NOT_EQUAL  # type: ignore[misc]


def test_views_and_source_audit_do_not_read_files(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = _tree("Add(Symbol('x', real=True), Integer(1))")
    record = _record("Add(Symbol('x', real=True), Integer(1))")

    def forbidden_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("rendering and structural audits must not read files")

    monkeypatch.setattr("builtins.open", forbidden_open)
    monkeypatch.setattr(Path, "open", forbidden_open)
    assert render_display(tree) == "x + 1"
    assert render_latex(tree) == "x + 1"
    assert audit_source_roundtrip(record).status is RoundTripStatus.EXACT
