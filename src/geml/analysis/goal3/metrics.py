"""
metrics.py - stratifies goal3 metrics by family/split/domain/size/depth,
plus reuse/sharing analysis on real graphs

owned by 3-7

reads goal3.run.py-shaped rows: expression_id, status, metrics dict
(the 5 alpha/compression ratios + node counts), plus stratification
tags. every aggregate reports BOTH all_processed_count and valid_count
so a reader always knows which denominator an average is really over
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

from geml.graph.schema import Graph


@dataclass
class AnalysisRow:
    expression_id: str
    status: str  # "success" | "failure"
    family: str | None = None
    split: str | None = None
    domain: str | None = None
    size_bucket: str | None = None
    depth_bucket: str | None = None
    metrics: dict | None = None  # only present when status == "success"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass
class GroupStats:
    key: str
    all_processed_count: int   # everything attempted in this group, success + failure
    valid_count: int            # only the successful ones - what the means below are actually averaged over
    mean_raw_tree_alpha: float | None
    mean_dag_alpha_vs_ast_tree: float | None
    mean_dag_alpha_vs_ast_dag: float | None
    mean_ast_compression: float | None
    mean_eml_compression: float | None


def stratify(rows: list[AnalysisRow], key_fn: Callable[[AnalysisRow], str]) -> dict[str, GroupStats]:
    groups: dict[str, list[AnalysisRow]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)

    out: dict[str, GroupStats] = {}
    for key, group_rows in groups.items():
        valid = [r for r in group_rows if r.status == "success"]
        out[key] = GroupStats(
            key=key,
            all_processed_count=len(group_rows),
            valid_count=len(valid),
            mean_raw_tree_alpha=_mean([r.metrics["raw_tree_alpha"] for r in valid]),
            mean_dag_alpha_vs_ast_tree=_mean([r.metrics["dag_alpha_vs_ast_tree"] for r in valid]),
            mean_dag_alpha_vs_ast_dag=_mean([r.metrics["dag_alpha_vs_ast_dag"] for r in valid]),
            mean_ast_compression=_mean([r.metrics["ast_compression"] for r in valid]),
            mean_eml_compression=_mean([r.metrics["eml_compression"] for r in valid]),
        )
    return out


def stratify_by_family(rows): return stratify(rows, lambda r: r.family or "unknown")
def stratify_by_split(rows): return stratify(rows, lambda r: r.split or "unknown")
def stratify_by_domain(rows): return stratify(rows, lambda r: r.domain or "unknown")
def stratify_by_size_bucket(rows): return stratify(rows, lambda r: r.size_bucket or "unknown")
def stratify_by_depth_bucket(rows): return stratify(rows, lambda r: r.depth_bucket or "unknown")


def stratify_by_signature(rows: list[AnalysisRow]) -> dict[str, GroupStats]:
    # groups by a truncated signature-ish key so structurally identical
    # shapes get compared against each other - falls back to size+depth
    # if a row doesn't carry an actual signature string
    def key(r: AnalysisRow) -> str:
        if r.metrics and "signature" in r.metrics:
            return r.metrics["signature"]
        return f"{r.size_bucket or '?'}:{r.depth_bucket or '?'}"
    return stratify(rows, key)


# ---------------------------------------------------------------------
# reuse/sharing analysis on real graphs (not just summary rows)
# ---------------------------------------------------------------------

@dataclass
class ReuseStats:
    node_count: int
    reused_subtree_count: int      # nodes referenced by more than one parent
    max_reuse_count: int            # how many times the single most-reused node is referenced
    sharing_concentration: float    # max_reuse / total_references - 0 if reuse is spread evenly, higher if concentrated in one node
    child_reference_overhead: int   # edge_count beyond what a plain tree would need (node_count - 1)
    mean_reuse_depth: float | None  # average depth-from-root of the reused nodes specifically


def _node_depths_from_roots(graph: Graph) -> dict[str, int]:
    depths: dict[str, int] = {}
    stack = [(root, 0) for root in graph.roots]
    while stack:
        node_id, depth = stack.pop()
        if node_id in depths:
            depths[node_id] = min(depths[node_id], depth)
        else:
            depths[node_id] = depth
        node = graph.nodes.get(node_id)
        if node:
            for ref in node.children:
                stack.append((ref.target_id, depth + 1))
    return depths


def analyze_reuse(graph: Graph) -> ReuseStats:
    reference_counts: Counter[str] = Counter()
    for node in graph.nodes.values():
        for ref in node.children:
            reference_counts[ref.target_id] += 1

    total_refs = sum(reference_counts.values())
    reused_ids = [nid for nid, count in reference_counts.items() if count > 1]
    max_reuse = max(reference_counts.values(), default=0)

    # concentration is only meaningful relative to nodes that are
    # actually reused - if nothing's reused at all, concentration is 0
    # by convention, not some leftover fraction from single-use nodes
    reused_reference_total = sum(count for count in reference_counts.values() if count > 1)
    sharing_concentration = (max_reuse / reused_reference_total) if reused_reference_total else 0.0

    depths = _node_depths_from_roots(graph)
    reuse_depths = [depths[nid] for nid in reused_ids if nid in depths]

    return ReuseStats(
        node_count=len(graph.nodes),
        reused_subtree_count=len(reused_ids),
        max_reuse_count=max_reuse,
        sharing_concentration=sharing_concentration,
        child_reference_overhead=total_refs - (len(graph.nodes) - 1) if graph.nodes else 0,
        mean_reuse_depth=_mean(reuse_depths),
    )
