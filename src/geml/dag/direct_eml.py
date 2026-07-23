"""
direct_eml.py - direct/memoized eml dag construction, no intermediate tree

owned by 3-4

TODO(Sahil/Quang): only exp and ln are here since those are the only
two verified formulas we have so far (arXiv:2603.21852v2 eq 3 and 5,
cross-checked in 3-3). add/mul/pow constructors are pending 2-2/2-3/
2-4's real compiler output, not guessing at those
"""
from __future__ import annotations
from dataclasses import dataclass
import time

from geml.graph.schema import Graph
from geml.dag.hashcons import HashConsTable


def emit_var(table: HashConsTable, name: str) -> str:
    return table.intern_leaf("Var", name)


def emit_const_one(table: HashConsTable) -> str:
    return table.intern_leaf("Const", 1)


def emit_exp(table: HashConsTable, operand_id: str) -> str:
    # exp(x) = eml(x, 1), built node by node, checked against the
    # table every step - never assembled as a standalone tree first
    one_id = emit_const_one(table)
    return table.intern_binary("eml", operand_id, one_id)


def emit_ln(table: HashConsTable, operand_id: str) -> str:
    # ln(z) = eml(1, eml(eml(1,z), 1)), same direct approach - three
    # intern calls instead of building the tree then compressing it
    one_a = emit_const_one(table)
    inner = table.intern_binary("eml", one_a, operand_id)      # eml(1, z)
    one_b = emit_const_one(table)
    middle = table.intern_binary("eml", inner, one_b)           # eml(eml(1,z), 1)
    one_c = emit_const_one(table)
    return table.intern_binary("eml", one_c, middle)            # eml(1, eml(eml(1,z),1))


@dataclass
class ConstructionStats:
    elapsed_seconds: float
    peak_interning_table_size: int
    final_node_count: int


def compile_with_stats(build_fn) -> tuple[Graph, str, ConstructionStats]:
    # runs build_fn(table) -> root_id, timing it and recording peak
    # table size. build_fn is whatever emit_* composition the caller
    # wants, keeps the stats logic generic instead of duplicating it
    # per constructor
    table = HashConsTable(family="eml")
    start = time.perf_counter()
    root_id = build_fn(table)
    elapsed = time.perf_counter() - start

    stats = ConstructionStats(
        elapsed_seconds=elapsed,
        peak_interning_table_size=table.peak_size,
        final_node_count=len(table.nodes),
    )
    return table.to_graph(root_id), root_id, stats
