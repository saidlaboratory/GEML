"""Tests for direct official EML-to-DAG construction."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from geml.ast.builder import build_ast_from_parsed
from geml.dag import direct_eml
from geml.dag.direct_eml import (
    DIRECT_SOURCE_OPERATORS,
    DirectEMLCompiler,
    compile_ast_to_eml_dag,
    compile_with_stats,
)
from geml.dag.eml import eml_to_dag, validate_eml_dag
from geml.dag.hashcons import HashConsTable, InternedNode
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
)
from geml.eml.compiler_transcendental import eml_cosh, eml_sinh, eml_tanh
from geml.eml.compiler_trig import eml_cos, eml_sin, eml_tan
from geml.eml.ir import EMLTerm, One, Variable
from geml.experiments.goal2.run import materialize_ast_official
from geml.graph.schema import Graph
from geml.graph.signatures import compute_signature
from geml.parsing.srepr import parse_srepr
from geml.spec.operators import OPERATORS, EMLConstructionStatus

type TreeBuilder = Callable[[CompilerMode], EMLTerm]
type DirectBuilder = Callable[[DirectEMLCompiler], InternedNode]


def _formula_cases() -> tuple[tuple[str, TreeBuilder, DirectBuilder], ...]:
    return (
        ("one", lambda mode: One(), lambda compiler: compiler.emit_one()),
        (
            "variable",
            lambda mode: Variable("x"),
            lambda compiler: compiler.emit_variable("x"),
        ),
        (
            "exp",
            lambda mode: eml_exp(Variable("x")),
            lambda compiler: compiler.emit_exp(compiler.emit_variable("x")),
        ),
        (
            "log",
            lambda mode: eml_log(Variable("x")),
            lambda compiler: compiler.emit_log(compiler.emit_variable("x")),
        ),
        ("zero", lambda mode: eml_zero(), lambda compiler: compiler.emit_zero()),
        (
            "subtract",
            lambda mode: eml_subtract(Variable("x"), Variable("y")),
            lambda compiler: compiler.emit_subtract(
                compiler.emit_variable("x"),
                compiler.emit_variable("y"),
            ),
        ),
        (
            "negate",
            lambda mode: eml_negate(Variable("x"), mode=mode),
            lambda compiler: compiler.emit_negate(compiler.emit_variable("x")),
        ),
        (
            "add",
            lambda mode: eml_add(Variable("x"), Variable("y"), mode=mode),
            lambda compiler: compiler.emit_add(
                compiler.emit_variable("x"),
                compiler.emit_variable("y"),
            ),
        ),
        (
            "inverse",
            lambda mode: eml_inverse(Variable("x"), mode=mode),
            lambda compiler: compiler.emit_inverse(compiler.emit_variable("x")),
        ),
        (
            "multiply",
            lambda mode: eml_multiply(Variable("x"), Variable("y"), mode=mode),
            lambda compiler: compiler.emit_multiply(
                compiler.emit_variable("x"),
                compiler.emit_variable("y"),
            ),
        ),
        (
            "divide",
            lambda mode: eml_divide(Variable("x"), Variable("y"), mode=mode),
            lambda compiler: compiler.emit_divide(
                compiler.emit_variable("x"),
                compiler.emit_variable("y"),
            ),
        ),
        (
            "power",
            lambda mode: eml_power(Variable("x"), Variable("y"), mode=mode),
            lambda compiler: compiler.emit_power(
                compiler.emit_variable("x"),
                compiler.emit_variable("y"),
            ),
        ),
        (
            "integer",
            lambda mode: eml_integer(-5, mode=mode),
            lambda compiler: compiler.emit_integer(-5),
        ),
        (
            "rational",
            lambda mode: eml_rational(-2, 3, mode=mode),
            lambda compiler: compiler.emit_rational(-2, 3),
        ),
        (
            "decimal",
            lambda mode: eml_decimal("1.25", mode=mode),
            lambda compiler: compiler.emit_decimal("1.25"),
        ),
        (
            "sin",
            lambda mode: eml_sin(Variable("x"), mode=mode),
            lambda compiler: compiler.emit_sin(compiler.emit_variable("x")),
        ),
        (
            "cos",
            lambda mode: eml_cos(Variable("x"), mode=mode),
            lambda compiler: compiler.emit_cos(compiler.emit_variable("x")),
        ),
        (
            "tan",
            lambda mode: eml_tan(Variable("x"), mode=mode),
            lambda compiler: compiler.emit_tan(compiler.emit_variable("x")),
        ),
        (
            "sinh",
            lambda mode: eml_sinh(Variable("x"), mode=mode),
            lambda compiler: compiler.emit_sinh(compiler.emit_variable("x")),
        ),
        (
            "cosh",
            lambda mode: eml_cosh(Variable("x"), mode=mode),
            lambda compiler: compiler.emit_cosh(compiler.emit_variable("x")),
        ),
        (
            "tanh",
            lambda mode: eml_tanh(Variable("x"), mode=mode),
            lambda compiler: compiler.emit_tanh(compiler.emit_variable("x")),
        ),
    )


@pytest.mark.parametrize(
    ("name", "tree_builder", "direct_builder"),
    _formula_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
@pytest.mark.parametrize("mode", tuple(CompilerMode))
def test_direct_formulas_match_posthoc_dags_exactly(
    name: str,
    tree_builder: TreeBuilder,
    direct_builder: DirectBuilder,
    mode: CompilerMode,
) -> None:
    direct_graph, direct_root, statistics = compile_with_stats(
        direct_builder,
        mode=mode,
        root_id=name,
    )
    posthoc_graph = eml_to_dag(
        tree_builder(mode),
        root_id=name,
        representation_mode=f"pure_eml:{mode.value}",
    )
    posthoc_root = posthoc_graph.roots[0].target_id

    assert validate_eml_dag(direct_graph).valid
    assert direct_root == posthoc_root
    assert dict(direct_graph.nodes) == dict(posthoc_graph.nodes)
    assert compute_signature(direct_graph, direct_root) == compute_signature(
        posthoc_graph,
        posthoc_root,
    )
    assert statistics.final_node_count == len(direct_graph.nodes)
    assert statistics.peak_interning_table_size == len(direct_graph.nodes)
    assert statistics.compiler_mode is mode
    assert statistics.representation_mode == f"pure_eml:{mode.value}"
    assert statistics.elapsed_seconds >= 0


def test_official_v4_is_the_default_and_clean_negation_is_distinct() -> None:
    default_graph, default_root, default_stats = compile_with_stats(
        lambda compiler: compiler.emit_negate(compiler.emit_variable("x"))
    )
    official_graph, official_root, _ = compile_with_stats(
        lambda compiler: compiler.emit_negate(compiler.emit_variable("x")),
        mode=CompilerMode.OFFICIAL_V4,
    )
    clean_graph, clean_root, clean_stats = compile_with_stats(
        lambda compiler: compiler.emit_negate(compiler.emit_variable("x")),
        mode=CompilerMode.CLEAN_NEGATION,
    )

    assert default_stats.compiler_mode is CompilerMode.OFFICIAL_V4
    assert default_root == official_root
    assert dict(default_graph.nodes) == dict(official_graph.nodes)
    assert clean_stats.compiler_mode is CompilerMode.CLEAN_NEGATION
    assert clean_stats.representation_mode == "pure_eml:clean_negation"
    assert clean_root != official_root
    assert compute_signature(clean_graph, clean_root) != compute_signature(
        official_graph,
        official_root,
    )


def test_interning_uses_compact_stable_content_ids() -> None:
    def build(compiler: DirectEMLCompiler):
        value = compiler.emit_variable("x")
        for _ in range(250):
            value = compiler.emit_primitive(value, value)
        return value

    first, first_root, first_stats = compile_with_stats(build)
    second, second_root, second_stats = compile_with_stats(build)

    assert first_root == second_root
    assert dict(first.nodes) == dict(second.nodes)
    assert first_stats.final_node_count == second_stats.final_node_count == 251
    assert all(len(node_id) == len("eml-") + 64 for node_id in first.nodes)
    assert max(len(node.node_id) for node in first.nodes.values()) == 68


def test_duplicate_requests_are_cache_hits_and_keep_explicit_slots() -> None:
    graph, root_id, statistics = compile_with_stats(
        lambda compiler: compiler.emit_primitive(
            compiler.emit_exp(compiler.emit_variable("x")),
            compiler.emit_exp(compiler.emit_variable("x")),
        )
    )
    root = graph.nodes[root_id]

    assert root.children[0].target_id == root.children[1].target_id
    assert statistics.cache_hits > 0
    assert statistics.intern_requests > statistics.final_node_count


def test_elapsed_time_includes_graph_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalized = False
    timer_calls = 0
    original_to_graph = HashConsTable.to_graph

    def tracked_to_graph(
        table: HashConsTable,
        root: InternedNode,
        *,
        root_id: str,
        representation_mode: str,
    ) -> Graph:
        nonlocal finalized
        graph = original_to_graph(
            table,
            root,
            root_id=root_id,
            representation_mode=representation_mode,
        )
        finalized = True
        return graph

    def deterministic_timer() -> float:
        nonlocal timer_calls
        timer_calls += 1
        if timer_calls == 1:
            assert not finalized
            return 10.0
        assert finalized
        return 12.5

    monkeypatch.setattr(HashConsTable, "to_graph", tracked_to_graph)
    monkeypatch.setattr(direct_eml.time, "perf_counter", deterministic_timer)

    _, _, statistics = compile_with_stats(lambda compiler: compiler.emit_variable("x"))

    assert timer_calls == 2
    assert statistics.elapsed_seconds == 2.5


def test_cross_table_references_are_rejected() -> None:
    first = DirectEMLCompiler()
    second = DirectEMLCompiler()

    with pytest.raises(ValueError, match="different interning table"):
        second.emit_exp(first.emit_variable("x"))


def test_unreachable_table_entries_are_not_silently_dropped() -> None:
    compiler = DirectEMLCompiler()
    root = compiler.emit_variable("x")
    compiler.emit_variable("unused")

    with pytest.raises(ValueError, match="unreachable"):
        compiler.table.to_graph(
            root,
            root_id="expression",
            representation_mode="pure_eml:official_v4",
        )


def test_graph_snapshot_does_not_alias_the_live_table() -> None:
    compiler = DirectEMLCompiler()
    root = compiler.emit_variable("x")
    graph = compiler.table.to_graph(
        root,
        root_id="expression",
        representation_mode="pure_eml:official_v4",
    )
    compiler.emit_variable("later")

    assert len(graph.nodes) == 1


@pytest.mark.parametrize("name", ["", "x+y", "log(x)", "x y"])
def test_compound_or_blank_variable_names_are_rejected(name: str) -> None:
    compiler = DirectEMLCompiler()

    with pytest.raises(ValueError, match="variable names"):
        compiler.emit_variable(name)


_AST_SREPRS = (
    "Symbol('x', real=True)",
    "Integer(1)",
    "Integer(-3)",
    "Rational(2, 3)",
    "Add(Symbol('x', real=True), Symbol('y', real=True))",
    "Mul(Symbol('x', real=True), Symbol('y', real=True))",
    "Pow(Symbol('x', positive=True), Rational(2, 3))",
    "exp(Symbol('x', real=True))",
    "log(Symbol('x', positive=True))",
    "sin(Symbol('x', real=True))",
    "cos(Symbol('x', real=True))",
    "tan(Symbol('x', real=True))",
    "sinh(Symbol('x', real=True))",
    "cosh(Symbol('x', real=True))",
    "tanh(Symbol('x', real=True))",
    (
        "Add(Mul(Rational(-2, 3), sin(Symbol('x', real=True))), "
        "Pow(log(Symbol('y', positive=True)), Integer(2)), "
        "cosh(Symbol('x', real=True)))"
    ),
)


@pytest.mark.parametrize("sympy_srepr", _AST_SREPRS)
def test_authoritative_ast_direct_compilation_matches_goal2(
    sympy_srepr: str,
) -> None:
    tree = build_ast_from_parsed(
        parse_srepr(sympy_srepr),
        expression_id="expression",
    )
    direct_graph, direct_root, statistics = compile_ast_to_eml_dag(tree)
    posthoc_graph = eml_to_dag(
        materialize_ast_official(tree),
        root_id=tree.expression_id,
        representation_mode="pure_eml:official_v4",
    )

    assert direct_root == posthoc_graph.roots[0].target_id
    assert dict(direct_graph.nodes) == dict(posthoc_graph.nodes)
    assert statistics.compiler_mode is CompilerMode.OFFICIAL_V4


def test_direct_dispatch_covers_exactly_enabled_approved_registry() -> None:
    enabled_approved = {
        operator.name
        for operator in OPERATORS
        if operator.enabled_for_generation
        and operator.eml_construction_status is EMLConstructionStatus.APPROVED
    }

    assert enabled_approved == DIRECT_SOURCE_OPERATORS
    assert {"e", "pi", "imaginary_unit"}.isdisjoint(DIRECT_SOURCE_OPERATORS)
