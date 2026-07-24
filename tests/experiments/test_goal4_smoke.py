"""Fresh-clone smoke tests for the Goal 4 optimization experiment pipeline.

The full 25-expression stage is executed once through a module-scoped fixture; individual
assertions read its rows.  Mechanism tests (resume, timeout, determinism, subset) use a
small record set so the whole module stays fast without weakening the checks.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.experiments.goal4.run import (
    CONFIG_SCHEMA_VERSION,
    Goal4Config,
    StageStatus,
    item_from_record,
    load_checkpoint,
    load_goal4_config,
    process_expression,
    run_stage,
    select_subset,
)
from geml.experiments.goal4.runtime import Goal4RuntimeError, read_jsonl

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_CONFIG = _REPO_ROOT / "configs" / "goal4_final.yaml"

_REQUIRED_ROW_FIELDS = (
    "expression_id",
    "rewrite_mode",
    "domain_mode",
    "operator_family",
    "split",
    "rule_library",
    "declared_assumptions",
    "saturation_status",
    "extraction_status",
    "candidate_count",
    "stage_status",
    "failure_reason",
    "timeout",
    "resources",
    "provenance",
    "eml_dag_cost_before",
    "eml_dag_cost_after",
    "absolute_improvement",
    "relative_improvement",
)

_SPLITS = tuple(CorpusSplit)


def _expression_id(source: str, salt: int) -> str:
    return hashlib.sha256(f"geml-goal4-fixture\0{salt}\0{source}".encode()).hexdigest()


def _record(index: int) -> ExpressionRecord:
    """Return one tiny deterministic fixture record.

    Every seventh expression uses a trigonometric operator to exercise the retained
    unsupported-operator path; the rest are simple algebraic expressions.
    """
    if index % 7 == 0:
        srepr = "sin(Symbol('x', real=True))"
        family = "trigonometric"
    else:
        srepr = f"Add(Symbol('x', real=True), Integer({index}))"
        family = "algebraic_core"
    return ExpressionRecord(
        expression_id=_expression_id(srepr, index),
        sympy_srepr=srepr,
        display_text=srepr,
        latex_text=None,
        split=_SPLITS[index % len(_SPLITS)],
        operator_family=family,
        domain_mode="safe_real",
        variables=("x",),
        target_ast_size=3 + index % 5,
        target_depth=1,
        generator_seed=index,
        generator_metadata={},
    )


def _records(count: int = 25) -> list[ExpressionRecord]:
    return [_record(index) for index in range(count)]


def _config_dict(output_root: str, *, saturation_timeout: float = 5.0) -> dict:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "output_root": output_root,
        "include_optional_domain_rules": False,
        "modes": ["safe_real", "positive_real_formal"],
        "sampling": {
            "seed": 7,
            "target_size": 25,
            "balance_axes": ["operator_family", "domain_mode", "split", "size_bucket"],
            "size_bucket_edges": [4, 8, 16, 32],
        },
        "resources": {
            "max_iterations": 40,
            "max_egraph_nodes": 5000,
            "saturation_timeout_seconds": saturation_timeout,
            "max_eclasses": None,
            "extraction_max_depth": 12,
            "extraction_beam_width": 6,
            "extraction_max_candidates": 24,
            "extraction_max_nodes": 20000,
            "extraction_timeout_seconds": 5.0,
        },
        "processing": {"chunk_size": 8, "checkpoint_every_chunks": 1, "resume": True},
        "stages": {"smoke": {"expected_count": 25, "row_limit": 25}},
    }


def _config(tmp_path: Path, *, count: int = 25, **overrides) -> Goal4Config:
    data = _config_dict(str(tmp_path / "out"), **overrides)
    data["sampling"]["target_size"] = count
    data["stages"]["smoke"]["expected_count"] = count
    data["stages"]["smoke"]["row_limit"] = count
    return Goal4Config.model_validate(data)


@pytest.fixture(scope="module")
def default_stage(tmp_path_factory):
    """Run the full 25-expression smoke stage once and return its rows."""
    directory = tmp_path_factory.mktemp("goal4-default")
    config = Goal4Config.model_validate(_config_dict(str(directory / "out")))
    result = run_stage(config, "smoke", _records(), directory / "run")
    return result, read_jsonl(result.rows_path)


class TestConfigurationLoading:
    def test_production_config_loads(self):
        config = load_goal4_config(_PRODUCTION_CONFIG)
        assert config.schema_version == CONFIG_SCHEMA_VERSION
        assert 25_000 <= config.sampling.target_size <= 40_000
        assert config.resolved_modes()

    def test_temp_config_round_trips(self, tmp_path: Path):
        config_path = tmp_path / "goal4_smoke.yaml"
        config_path.write_text(yaml.safe_dump(_config_dict(str(tmp_path / "out"))))
        config = load_goal4_config(config_path)
        assert isinstance(config, Goal4Config)
        assert config.stages["smoke"].row_limit == 25

    def test_missing_config_is_explicit(self, tmp_path: Path):
        with pytest.raises(Goal4RuntimeError, match="missing Goal 4 config"):
            load_goal4_config(tmp_path / "absent.yaml")

    def test_wrong_schema_version_is_rejected(self, tmp_path: Path):
        data = _config_dict(str(tmp_path / "out"))
        data["schema_version"] = "wrong"
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(Goal4RuntimeError, match="schema version"):
            load_goal4_config(path)


class TestSubsetConstruction:
    def test_subset_is_deterministic(self, tmp_path: Path):
        config = _config(tmp_path)
        items = [item_from_record(record, config.sampling) for record in _records()]
        first = select_subset(items, config.sampling)
        second = select_subset(items, config.sampling)
        assert [item.expression_id for item in first] == [item.expression_id for item in second]

    def test_subset_respects_target_size(self, tmp_path: Path):
        config = _config(tmp_path)
        smaller = config.sampling.model_copy(update={"target_size": 10})
        items = [item_from_record(record, config.sampling) for record in _records()]
        assert len(select_subset(items, smaller)) == 10

    def test_subset_is_order_independent(self, tmp_path: Path):
        config = _config(tmp_path)
        items = [item_from_record(record, config.sampling) for record in _records()]
        forward = select_subset(items, config.sampling)
        backward = select_subset(list(reversed(items)), config.sampling)
        assert [item.expression_id for item in forward] == [item.expression_id for item in backward]


class TestPipelineExecution:
    def test_stage_runs_over_25_expressions_in_both_modes(self, default_stage):
        result, rows = default_stage
        assert result.total_units == 50
        assert result.completed_units == 50
        assert len(rows) == 50

    def test_modes_are_separated_and_balanced(self, default_stage):
        _result, rows = default_stage
        modes = Counter(row["rewrite_mode"] for row in rows)
        assert modes["safe_real"] == 25
        assert modes["positive_real_formal"] == 25

    def test_every_row_has_the_required_fields(self, default_stage):
        _result, rows = default_stage
        for row in rows:
            for field in _REQUIRED_ROW_FIELDS:
                assert field in row, field

    def test_unsupported_operators_are_retained(self, default_stage):
        _result, rows = default_stage
        statuses = Counter(row["stage_status"] for row in rows)
        assert statuses[StageStatus.UNSUPPORTED_OPERATOR.value] > 0

    def test_rule_library_matches_mode(self, default_stage):
        _result, rows = default_stage
        for row in rows:
            if row["rewrite_mode"] == "safe_real":
                assert row["rule_library"] == "safe_real"
            else:
                assert row["rule_library"] == "safe_plus_domain"

    def test_no_row_is_dropped(self, default_stage):
        _result, rows = default_stage
        unit_ids = {(row["expression_id"], row["rewrite_mode"]) for row in rows}
        assert len(unit_ids) == 50


class TestCheckpointAndResume:
    def test_checkpoint_is_created(self, default_stage):
        result, _rows = default_stage
        assert result.checkpoint_path.is_file()
        checkpoint = load_checkpoint(result.checkpoint_path)
        assert checkpoint.total_units == 50
        assert len(checkpoint.completed_ids) == 50

    def test_resume_recomputes_nothing(self, tmp_path: Path):
        config = _config(tmp_path, count=6)
        run_directory = tmp_path / "run"
        first = run_stage(config, "smoke", _records(6), run_directory)
        rows_after_first = len(read_jsonl(first.rows_path))
        second = run_stage(config, "smoke", _records(6), run_directory)
        rows_after_second = len(read_jsonl(second.rows_path))
        assert rows_after_first == rows_after_second == 12

    def test_resume_completes_a_partial_run(self, tmp_path: Path):
        config = _config(tmp_path, count=6)
        run_directory = tmp_path / "run"
        run_directory.mkdir(parents=True)
        partial = process_expression(
            item_from_record(_records(6)[0], config.sampling),
            config.resolved_modes()[0],
            config,
        )
        run_directory.joinpath("smoke.rows.jsonl").write_text(json.dumps(partial) + "\n")
        run_stage(config, "smoke", _records(6), run_directory)
        rows = read_jsonl(run_directory / "smoke.rows.jsonl")
        unit_ids = {(row["expression_id"], row["rewrite_mode"]) for row in rows}
        assert len(unit_ids) == 12
        assert len(rows) == 12

    def test_truncated_final_line_is_tolerated(self, tmp_path: Path):
        config = _config(tmp_path, count=6)
        run_directory = tmp_path / "run"
        run_directory.mkdir(parents=True)
        run_directory.joinpath("smoke.rows.jsonl").write_text(
            json.dumps({"expression_id": "x", "rewrite_mode": "safe_real"}) + "\n{partial"
        )
        result = run_stage(config, "smoke", _records(6), run_directory)
        assert result.completed_units >= 12


class TestTimeoutHandling:
    def test_zero_timeout_marks_every_supported_row(self, tmp_path: Path):
        config = _config(tmp_path, count=6, saturation_timeout=0.0)
        result = run_stage(config, "smoke", _records(6), tmp_path / "run")
        rows = read_jsonl(result.rows_path)
        supported = [row for row in rows if row["stage_status"] != "unsupported_operator"]
        assert supported
        assert all(row["saturation_status"] == "timeout" for row in supported)
        assert all(row["timeout"] for row in supported)

    def test_timeout_rows_are_still_complete(self, tmp_path: Path):
        config = _config(tmp_path, count=6, saturation_timeout=0.0)
        result = run_stage(config, "smoke", _records(6), tmp_path / "run")
        for row in read_jsonl(result.rows_path):
            assert "resources" in row
            assert row["failure_reason"] is None or isinstance(row["failure_reason"], str)


class TestResultGeneration:
    def test_results_are_deterministic(self, tmp_path: Path):
        config = _config(tmp_path, count=6)
        first = read_jsonl(run_stage(config, "smoke", _records(6), tmp_path / "a").rows_path)
        second = read_jsonl(run_stage(config, "smoke", _records(6), tmp_path / "b").rows_path)

        def key(rows):
            return sorted(
                (row["expression_id"], row["rewrite_mode"], row["stage_status"]) for row in rows
            )

        assert key(first) == key(second)

    def test_improvement_is_consistent_with_costs(self, default_stage):
        _result, rows = default_stage
        for row in rows:
            before = row["eml_dag_cost_before"]
            after = row["eml_dag_cost_after"]
            if before is not None and after is not None:
                assert row["absolute_improvement"] == before - after

    def test_optimized_rows_have_positive_improvement(self, default_stage):
        _result, rows = default_stage
        for row in rows:
            if row["stage_status"] == StageStatus.OPTIMIZED.value:
                assert row["absolute_improvement"] > 0
