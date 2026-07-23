"""tests/analysis/test_goal3_analysis.py - owned by 3-7. tiny fixtures only."""
import inspect
from geml.analysis.goal3.metrics import (
    AnalysisRow, stratify_by_family, stratify_by_split, analyze_reuse,
)
from geml.analysis.goal3.failures import (
    top_by_compression_ratio, top_by_final_alpha, top_failures, classify_dual,
)
from geml.plots.goal3 import ScalePoint, build_stability_curve, stability_delta, missing_checkpoints
from geml.graph.schema import Graph, GraphNode, ChildRef


def _row(expr_id, status="success", family=None, split=None, **metric_overrides):
    base_metrics = {
        "raw_tree_alpha": 1.5, "dag_alpha_vs_ast_tree": 1.5, "dag_alpha_vs_ast_dag": 1.5,
        "ast_compression": 1.0, "eml_compression": 1.0,
    }
    base_metrics.update(metric_overrides)
    return AnalysisRow(
        expression_id=expr_id, status=status, family=family, split=split,
        metrics=base_metrics if status == "success" else None,
    )


# ---------------------------------------------------------------------
# stratification + honest denominators
# ---------------------------------------------------------------------

def test_stratify_reports_both_denominators():
    rows = [
        _row("e1", family="exp"),
        _row("e2", family="exp"),
        _row("e3", status="failure", family="exp"),
        _row("e4", family="ln"),
    ]
    by_family = stratify_by_family(rows)
    assert by_family["exp"].all_processed_count == 3
    assert by_family["exp"].valid_count == 2
    assert by_family["ln"].all_processed_count == 1
    assert by_family["ln"].valid_count == 1


def test_stratify_mean_only_computed_over_valid_rows():
    rows = [
        _row("e1", family="exp", dag_alpha_vs_ast_tree=2.0),
        _row("e2", family="exp", dag_alpha_vs_ast_tree=4.0),
        _row("e3", status="failure", family="exp"),  # must not pull the mean toward 0/None
    ]
    stats = stratify_by_family(rows)["exp"]
    assert stats.mean_dag_alpha_vs_ast_tree == 3.0  # (2.0+4.0)/2, not divided by 3


def test_stratify_group_with_no_valid_rows_reports_none_not_crash():
    rows = [_row("e1", status="failure", family="broken")]
    stats = stratify_by_family(rows)["broken"]
    assert stats.valid_count == 0
    assert stats.mean_dag_alpha_vs_ast_tree is None


def test_stratify_by_split_groups_independently_of_family():
    rows = [_row("e1", family="exp", split="train"), _row("e2", family="ln", split="train")]
    by_split = stratify_by_split(rows)
    assert by_split["train"].all_processed_count == 2


# ---------------------------------------------------------------------
# reuse/sharing analysis on real graphs
# ---------------------------------------------------------------------

def _shared_graph() -> Graph:
    """(x+1)*(x+1) as a DAG - mul's two children both point at the same add node."""
    nodes = {
        "mul": GraphNode("mul", family="ast", kind="Mul", children=(ChildRef(0, "add"), ChildRef(1, "add"))),
        "add": GraphNode("add", family="ast", kind="Add", children=(ChildRef(0, "x"), ChildRef(1, "one"))),
        "x": GraphNode("x", family="ast", kind="Var", value="x"),
        "one": GraphNode("one", family="ast", kind="Const", value=1),
    }
    return Graph(nodes=nodes, roots=("mul",))


def test_analyze_reuse_detects_shared_subtree():
    stats = analyze_reuse(_shared_graph())
    assert stats.reused_subtree_count == 1
    assert stats.max_reuse_count == 2
    assert stats.child_reference_overhead == 1


def test_analyze_reuse_no_sharing_case():
    nodes = {
        "add": GraphNode("add", family="ast", kind="Add", children=(ChildRef(0, "x"), ChildRef(1, "y"))),
        "x": GraphNode("x", family="ast", kind="Var", value="x"),
        "y": GraphNode("y", family="ast", kind="Var", value="y"),
    }
    g = Graph(nodes=nodes, roots=("add",))
    stats = analyze_reuse(g)
    assert stats.reused_subtree_count == 0
    assert stats.sharing_concentration == 0.0
    assert stats.child_reference_overhead == 0


# ---------------------------------------------------------------------
# acceptance criterion: distinguish "compresses well" from "structurally competitive"
# ---------------------------------------------------------------------

def test_compression_and_final_alpha_rankings_can_diverge():
    """the actual proof the two claims are kept separate - constructs a
    case where they give opposite verdicts"""
    rows = [
        _row("big_but_compresses", raw_tree_alpha=40.0, dag_alpha_vs_ast_tree=4.0, eml_compression=10.0),
        _row("small_stays_small", raw_tree_alpha=1.5, dag_alpha_vs_ast_tree=1.2, eml_compression=1.2),
    ]
    best_compression = top_by_compression_ratio(rows)[0].expression_id
    best_alpha = top_by_final_alpha(rows)[0].expression_id
    assert best_compression == "big_but_compresses"
    assert best_alpha == "small_stays_small"
    assert best_compression != best_alpha  # genuinely different winners, not the same thing twice


def test_classify_dual_surfaces_compresses_well_but_not_competitive():
    rows = [_row("big_but_compresses", dag_alpha_vs_ast_tree=4.0, eml_compression=10.0)]
    claim = classify_dual(rows)[0]
    assert claim.compresses_well is True
    assert claim.structurally_competitive is False


def test_classify_dual_both_false_case():
    rows = [_row("neither", dag_alpha_vs_ast_tree=8.0, eml_compression=1.1)]
    claim = classify_dual(rows)[0]
    assert claim.compresses_well is False
    assert claim.structurally_competitive is False


def test_top_failures_never_crashes_on_missing_metrics():
    rows = [_row("e1", status="failure")]
    results = top_failures(rows)
    assert len(results) == 1
    assert results[0].expression_id == "e1"


# ---------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------

def test_stratify_is_reproducible():
    rows = [_row("e1", family="exp"), _row("e2", family="ln")]
    first = stratify_by_family(rows)
    second = stratify_by_family(rows)
    assert first["exp"].mean_dag_alpha_vs_ast_tree == second["exp"].mean_dag_alpha_vs_ast_tree


def test_classify_dual_is_reproducible():
    rows = [_row("e1", dag_alpha_vs_ast_tree=2.0, eml_compression=5.0)]
    assert classify_dual(rows) == classify_dual(rows)


# ---------------------------------------------------------------------
# stability curves
# ---------------------------------------------------------------------

def test_stability_curve_sorted_ascending():
    points = [
        ScalePoint(50_000, 2.1, 5.0, 10, 5000, 100_000),
        ScalePoint(10_000, 2.5, 4.5, 2, 5000, 20_000),
    ]
    curve = build_stability_curve(points)
    assert [p.corpus_size for p in curve] == [10_000, 50_000]


def test_stability_delta_shrinks_when_converging():
    points = [
        ScalePoint(10_000, 2.5, 4.5, 2, 5000, 20_000),
        ScalePoint(50_000, 2.1, 5.0, 10, 5000, 100_000),
        ScalePoint(100_000, 2.0, 5.1, 20, 5000, 200_000),
    ]
    deltas = stability_delta(points, "mean_dag_alpha_vs_ast_tree")
    assert abs(deltas[1][1]) < abs(deltas[0][1])  # second delta smaller - converging


def test_missing_checkpoints_detected():
    points = [ScalePoint(10_000, 2.5, 4.5, 2, 5000, 20_000)]
    assert missing_checkpoints(points) == [50_000, 100_000, 250_000]


def test_no_missing_checkpoints_when_complete():
    points = [
        ScalePoint(10_000, 0, 0, 0, 0, 0), ScalePoint(50_000, 0, 0, 0, 0, 0),
        ScalePoint(100_000, 0, 0, 0, 0, 0), ScalePoint(250_000, 0, 0, 0, 0, 0),
    ]
    assert missing_checkpoints(points) == []


# ---------------------------------------------------------------------
# acceptance criterion: no e-graph or motif claims anywhere in this code
# ---------------------------------------------------------------------

def test_no_egraph_or_motif_claims_in_source():
    """
    concrete check for an otherwise-subjective acceptance criterion:
    scans this issue's own owned modules for any FUNCTION, CLASS, or
    FIELD name referencing e-graph/motif concepts. that's the actual
    thing that would constitute a "claim" - a metric or type that
    implies this module computes something about e-graphs or motifs.

    prose explaining "this is NOT about motifs" is explicitly fine and
    expected (see failures.py's own docstring) - a disclaimer isn't a
    claim, it's the opposite. only checking identifiers, not comments,
    keeps this test meaningful without being fragile against how the
    disclaimer happens to be worded.
    """
    import geml.analysis.goal3.metrics as metrics_mod
    import geml.analysis.goal3.failures as failures_mod
    import geml.plots.goal3 as plots_mod

    forbidden = ("egraph", "e_graph", "motif")

    for module in (metrics_mod, failures_mod, plots_mod):
        for name, obj in vars(module).items():
            if name.startswith("_"):
                continue
            lowered = name.lower()
            for term in forbidden:
                assert term not in lowered, f"{module.__name__}.{name} references {term!r}"
            if inspect.isclass(obj) and hasattr(obj, "__dataclass_fields__"):
                for field_name in obj.__dataclass_fields__:
                    for term in forbidden:
                        assert term not in field_name.lower(), (
                            f"{module.__name__}.{name}.{field_name} references {term!r}"
                        )
