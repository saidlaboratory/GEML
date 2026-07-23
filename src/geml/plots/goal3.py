"""
plots/goal3.py - prepares data series for goal 3 plots (compression/alpha
vs scale, runtime/throughput/memory vs scale)

owned by 3-7

returns plain data structures, not rendered images - keeps this
testable without needing a plotting library as a hard dependency.
actual rendering (matplotlib or whatever) can sit on top of these
series later; this module's job is just making sure the data going
into that plot is correct
"""
from __future__ import annotations
from dataclasses import dataclass

STANDARD_SCALE_CHECKPOINTS = (10_000, 50_000, 100_000, 250_000)


@dataclass
class ScalePoint:
    corpus_size: int
    mean_dag_alpha_vs_ast_tree: float
    mean_eml_compression: float
    elapsed_seconds: float
    throughput_per_second: float
    peak_memory_kb: float


def build_stability_curve(points: list[ScalePoint]) -> list[ScalePoint]:
    return sorted(points, key=lambda p: p.corpus_size)


def stability_delta(points: list[ScalePoint], metric: str) -> list[tuple[int, float]]:
    """
    (corpus_size, change_from_previous_point) pairs for one metric.
    a genuinely "stabilizing" metric should show these deltas shrinking
    toward zero as corpus_size grows - that's literally what a
    stability curve is checking for
    """
    curve = build_stability_curve(points)
    deltas: list[tuple[int, float]] = []
    previous = None
    for point in curve:
        value = getattr(point, metric)
        if previous is not None:
            deltas.append((point.corpus_size, value - previous))
        previous = value
    return deltas


def missing_checkpoints(points: list[ScalePoint]) -> list[int]:
    """which of the standard 10k/50k/100k/250k checkpoints are missing
    from the given points - so a caller can tell at a glance whether
    the stability curve is actually complete"""
    present = {p.corpus_size for p in points}
    return [c for c in STANDARD_SCALE_CHECKPOINTS if c not in present]
