"""
tests/experiments/test_goal3_smoke.py - owned by 3-6. tiny fixtures only,
count-25 fresh-clone smoke pipeline, no production data.
"""
from pathlib import Path
from geml.dag.ast import AstNode
from geml.dag.eml import EmlNode, make_exp, make_ln
from geml.dag.direct_eml import emit_var, emit_exp, emit_ln
from geml.experiments.goal3.run import process_one, Goal3InputRecord, compute_goal3_metrics
from geml.experiments.goal3.runtime import process_corpus, load_completed_ids


def _exp_record(expr_id: str) -> Goal3InputRecord:
    return Goal3InputRecord(
        expression_id=expr_id,
        build_ast=lambda: AstNode("Exp", (AstNode("Var", value="x"),)),
        build_eml_tree=lambda: make_exp(EmlNode("Var", value="x")),
        build_eml_direct=lambda t: emit_exp(t, emit_var(t, "x")),
    )


def _ln_record(expr_id: str) -> Goal3InputRecord:
    return Goal3InputRecord(
        expression_id=expr_id,
        build_ast=lambda: AstNode("Log", (AstNode("Var", value="z"),)),
        build_eml_tree=lambda: make_ln(EmlNode("Var", value="z")),
        build_eml_direct=lambda t: emit_ln(t, emit_var(t, "z")),
    )


def _broken_record(expr_id: str) -> Goal3InputRecord:
    def broken():
        raise ValueError("deliberately broken - proves failures get reported, not dropped")
    return Goal3InputRecord(
        expression_id=expr_id, build_ast=broken, build_eml_tree=lambda: None, build_eml_direct=lambda t: None,
    )


def _make_count_25_corpus() -> list[Goal3InputRecord]:
    """the count-25 fresh-clone smoke corpus the issue asks for -
    23 runnable (mix of exp/ln), 2 deliberately broken to prove
    failure handling actually works, not just the happy path"""
    records = [_exp_record(f"exp_{i}") for i in range(12)]
    records += [_ln_record(f"ln_{i}") for i in range(11)]
    records += [_broken_record(f"broken_{i}") for i in range(2)]
    assert len(records) == 25
    return records


# ---------------------------------------------------------------------
# single-expression metric computation
# ---------------------------------------------------------------------

def test_metrics_computed_for_simple_exp():
    metrics = compute_goal3_metrics(
        AstNode("Exp", (AstNode("Var", value="x"),)),
        lambda: make_exp(EmlNode("Var", value="x")),
        lambda t: emit_exp(t, emit_var(t, "x")),
    )
    assert metrics.ast_tree_node_count == 2
    assert metrics.eml_tree_node_count == 3
    assert metrics.raw_tree_alpha == 1.5


def test_process_one_success_case():
    result = process_one(_exp_record("e1"))
    assert result.status == "success"
    assert result.metrics is not None
    assert result.error_type is None


def test_process_one_failure_case_never_crashes():
    result = process_one(_broken_record("e_broken"))
    assert result.status == "failure"
    assert result.error_type == "ValueError"
    assert result.metrics is None


# ---------------------------------------------------------------------
# acceptance criterion: every row gets metrics or an explicit failure
# ---------------------------------------------------------------------

def test_count_25_every_row_has_result(tmp_path):
    corpus = _make_count_25_corpus()
    out_path = tmp_path / "results.jsonl"
    summary = process_corpus(corpus, process_one, out_path, "smoke_run", "pytest tests/experiments/test_goal3_smoke.py")

    assert summary.processed_count == 25
    assert summary.success_count == 23
    assert summary.failure_count == 2

    rows = load_completed_ids(out_path)
    assert len(rows) == 25  # every single expression_id shows up, none silently missing


# ---------------------------------------------------------------------
# acceptance criterion: resuming produces identical final summaries
# ---------------------------------------------------------------------

def test_resume_matches_uninterrupted_run(tmp_path):
    corpus = _make_count_25_corpus()

    uninterrupted_path = tmp_path / "uninterrupted.jsonl"
    uninterrupted = process_corpus(
        corpus, process_one, uninterrupted_path, "run_full", "pytest ..."
    )

    interrupted_path = tmp_path / "interrupted.jsonl"
    # simulate a crash partway through: only process the first 15
    process_corpus(corpus[:15], process_one, interrupted_path, "run_partial", "pytest ...")
    # then "resume" with the full 25 - the first 15 should be skipped
    resumed = process_corpus(corpus, process_one, interrupted_path, "run_resumed", "pytest ...")

    assert (uninterrupted.processed_count, uninterrupted.success_count, uninterrupted.failure_count) == (
        resumed.processed_count, resumed.success_count, resumed.failure_count,
    )


def test_resume_actually_skips_completed_work(tmp_path):
    """confirms resume isn't just coincidentally correct - checks that
    the second call genuinely processed fewer new rows than the first"""
    corpus = _make_count_25_corpus()
    out_path = tmp_path / "results.jsonl"

    first = process_corpus(corpus[:15], process_one, out_path, "run1", "pytest ...")
    assert first.processed_count == 15

    # resuming with the same 15 again should add ZERO new rows - total
    # stays at 15, not 30
    second = process_corpus(corpus[:15], process_one, out_path, "run2", "pytest ...")
    assert second.processed_count == 15


def test_no_resume_starts_fresh(tmp_path):
    """resume=False should ignore any existing output and start over"""
    corpus = _make_count_25_corpus()
    out_path = tmp_path / "results.jsonl"

    process_corpus(corpus[:10], process_one, out_path, "run1", "pytest ...")
    fresh = process_corpus(corpus[:5], process_one, out_path, "run2", "pytest ...", resume=False)
    assert fresh.processed_count == 5  # not 15 - the old output got discarded


# ---------------------------------------------------------------------
# environment/throughput metadata gets recorded
# ---------------------------------------------------------------------

def test_run_summary_records_environment_and_throughput(tmp_path):
    corpus = _make_count_25_corpus()
    out_path = tmp_path / "results.jsonl"
    summary = process_corpus(corpus, process_one, out_path, "run1", "pytest ...")

    assert summary.environment.python_version
    assert summary.environment.platform
    assert summary.peak_memory_kb != 0  # -1.0 if unavailable, but never silently 0
    assert summary.throughput_per_second >= 0
    assert summary.reproduction_command == "pytest ..."
