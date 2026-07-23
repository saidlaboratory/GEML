"""
run.py - per-expression AST tree/DAG + EML tree/DAG metrics, all 5
alpha/compression ratios

owned by 3-6

honesty note: only exp/ln compositions are actually runnable right now
(see 3-3/3-4/3-5) - the pipeline logic itself is general, but the
smoke-test fixtures are limited to what's actually verified. real
add/mul/pow support gets plugged in once those constructors exist.

uses direct EML-DAG construction (3-4), not the post-hoc path (3-3),
per this issue's instruction - the 3-5 audit already confirmed the two
paths agree, so direct is the one actually used for real work
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable

from geml.dag.ast import AstNode, convert_with_stats as ast_convert_with_stats
from geml.dag.eml import EmlNode
from geml.dag.hashcons import HashConsTable
from geml.dag.direct_eml import compile_with_stats
from geml.experiments.goal3.runtime import RowResult


def _count_eml_tree_nodes(node: EmlNode) -> int:
    # the raw (uncompressed) tree size, needed for raw_tree_alpha and
    # eml_compression - this is exactly the "big tree" 3-4 exists to
    # avoid allocating for real work, but we still need its SIZE for
    # these particular ratios, so it's materialized just for counting
    if not node.children:
        return 1
    return 1 + sum(_count_eml_tree_nodes(c) for c in node.children)


@dataclass
class Goal3Metrics:
    ast_tree_node_count: int
    ast_dag_node_count: int
    eml_tree_node_count: int
    eml_dag_node_count: int
    raw_tree_alpha: float           # |eml tree| / |ast tree|
    dag_alpha_vs_ast_tree: float    # |eml dag| / |ast tree|
    dag_alpha_vs_ast_dag: float     # |eml dag| / |ast dag|
    ast_compression: float          # |ast tree| / |ast dag|
    eml_compression: float          # |eml tree| / |eml dag|


def compute_goal3_metrics(
    source_ast: AstNode,
    eml_tree_builder: Callable[[], EmlNode],
    eml_direct_builder: Callable[[HashConsTable], str],
) -> Goal3Metrics:
    ast_graph, ast_conv_stats = ast_convert_with_stats(source_ast)
    ast_tree_count = ast_conv_stats.ast_node_count
    ast_dag_count = ast_conv_stats.dag_node_count

    eml_tree = eml_tree_builder()
    eml_tree_count = _count_eml_tree_nodes(eml_tree)

    eml_dag_graph, eml_root, construction_stats = compile_with_stats(eml_direct_builder)
    eml_dag_count = construction_stats.final_node_count

    return Goal3Metrics(
        ast_tree_node_count=ast_tree_count,
        ast_dag_node_count=ast_dag_count,
        eml_tree_node_count=eml_tree_count,
        eml_dag_node_count=eml_dag_count,
        raw_tree_alpha=eml_tree_count / ast_tree_count if ast_tree_count else 0.0,
        dag_alpha_vs_ast_tree=eml_dag_count / ast_tree_count if ast_tree_count else 0.0,
        dag_alpha_vs_ast_dag=eml_dag_count / ast_dag_count if ast_dag_count else 0.0,
        ast_compression=ast_tree_count / ast_dag_count if ast_dag_count else 0.0,
        eml_compression=eml_tree_count / eml_dag_count if eml_dag_count else 0.0,
    )


@dataclass
class Goal3InputRecord:
    expression_id: str
    build_ast: Callable[[], AstNode]
    build_eml_tree: Callable[[], EmlNode]
    build_eml_direct: Callable[[HashConsTable], str]


def process_one(record: Goal3InputRecord) -> RowResult:
    # every record gets a result - success or failure, never an
    # uncaught exception that silently drops the row
    try:
        source_ast = record.build_ast()
        metrics = compute_goal3_metrics(source_ast, record.build_eml_tree, record.build_eml_direct)
        return RowResult(
            expression_id=record.expression_id,
            status="success",
            metrics=asdict(metrics),
        )
    except Exception as error:
        return RowResult(
            expression_id=record.expression_id,
            status="failure",
            error_type=type(error).__name__,
            error_message=str(error),
        )
