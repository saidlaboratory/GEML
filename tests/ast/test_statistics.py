"""Exact statistics, invariant, and structural-signature tests."""

from __future__ import annotations

import re

import pytest

from geml.ast.builder import build_ast
from geml.ast.statistics import (
    ASTStructureError,
    calculate_statistics,
    recompute_statistics,
    structural_signature,
)
from geml.contracts.ast import ASTEdge, ASTNode, ASTStatistics
from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord


def _record(source: str, expression_id: str = "c" * 64) -> ExpressionRecord:
    return ExpressionRecord(
        expression_id=expression_id,
        sympy_srepr=source,
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


def test_leaf_depth_is_zero_and_statistics_use_frozen_contract() -> None:
    tree = build_ast(_record("Symbol('x', real=True)"))
    assert tree.statistics == ASTStatistics(
        node_count=1,
        edge_count=0,
        leaf_count=1,
        operator_count=0,
        depth=0,
    )
    assert recompute_statistics(tree) == tree.statistics


def test_nested_statistics_are_exact() -> None:
    tree = build_ast(_record("Add(Symbol('x', real=True), Mul(Rational(1, 2), exp(Integer(1))))"))
    assert tree.statistics == ASTStatistics(
        node_count=6,
        edge_count=5,
        leaf_count=3,
        operator_count=3,
        depth=3,
    )


def test_signature_is_stable_ordered_and_expression_id_independent() -> None:
    first = build_ast(_record("Pow(Symbol('x', real=True), Integer(2))", "a" * 64))
    same_structure = build_ast(_record("Pow(Symbol('x', real=True), Integer(2))", "b" * 64))
    reversed_operands = build_ast(_record("Pow(Integer(2), Symbol('x', real=True))", "a" * 64))

    signature = structural_signature(first)
    assert re.fullmatch(r"[0-9a-f]{64}", signature)
    assert signature == "ff0049794fc6ce60573d71d35a785a65a10aa78c50365357e57ead6a2eda9bed"
    assert signature == structural_signature(same_structure)
    assert signature != structural_signature(reversed_operands)


def test_signature_preserves_symbol_assumptions_and_escaped_names() -> None:
    real = build_ast(_record("Symbol('a,b', real=True)"))
    positive = build_ast(_record("Symbol('a,b', positive=True)"))
    other_name = build_ast(_record("Symbol('a', real=True)"))

    assert structural_signature(real) != structural_signature(positive)
    assert structural_signature(real) != structural_signature(other_name)

    surrogate = build_ast(_record("Symbol('\\ud800', real=True)"))
    surrogate_signature = structural_signature(surrogate)
    assert re.fullmatch(r"[0-9a-f]{64}", surrogate_signature)
    assert surrogate_signature == structural_signature(surrogate)


def test_statistics_reject_missing_slots() -> None:
    nodes = (
        ASTNode(node_id="root", node_kind="operator", label="add", arity=2),
        ASTNode(node_id="left", node_kind="leaf", label="integer", arity=0, value=1),
        ASTNode(node_id="orphan", node_kind="leaf", label="integer", arity=0, value=2),
    )
    edges = (ASTEdge(source_id="root", target_id="left", child_slot=0),)
    with pytest.raises(ASTStructureError, match="arity"):
        calculate_statistics(nodes, edges, "root")


def test_statistics_reject_unreachable_cyclic_component() -> None:
    nodes = (
        ASTNode(node_id="root", node_kind="leaf", label="integer", arity=0, value=1),
        ASTNode(node_id="a", node_kind="operator", label="exp", arity=1),
        ASTNode(node_id="b", node_kind="operator", label="log", arity=1),
    )
    edges = (
        ASTEdge(source_id="a", target_id="b", child_slot=0),
        ASTEdge(source_id="b", target_id="a", child_slot=0),
    )
    with pytest.raises(ASTStructureError, match="unreachable"):
        calculate_statistics(nodes, edges, "root")


def test_statistics_reject_multiple_parents() -> None:
    repeated_nodes = (
        ASTNode(node_id="root", node_kind="operator", label="add", arity=2),
        ASTNode(node_id="leaf", node_kind="leaf", label="integer", arity=0, value=1),
    )
    repeated_edges = (
        ASTEdge(source_id="root", target_id="leaf", child_slot=0),
        ASTEdge(source_id="root", target_id="leaf", child_slot=1),
    )
    with pytest.raises(ASTStructureError, match="multiple parents"):
        calculate_statistics(repeated_nodes, repeated_edges, "root")
