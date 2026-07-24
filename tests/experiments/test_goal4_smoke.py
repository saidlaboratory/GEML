"""Fresh-clone smoke tests for the Goal 4 optimization experiment pipeline.

The full 25-expression stage is executed once through a module-scoped fixture; individual
assertions read its rows.  Mechanism tests (resume, timeout, determinism, subset) use a
small record set so the whole module stays fast without weakening the checks.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.experiments.goal4.run import (
    CONFIG_SCHEMA_VERSION,
    Goal4Config,
    StageStatus,
    _load_corpus_records,
    item_from_record,
    load_checkpoint,
    load_goal4_config,
    process_expression,
    run_stage,
    select_subset,
)
from geml.experiments.goal4.runtime import (
    Goal4RuntimeError,
    append_jsonl,
    read_jsonl,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_CONFIG = _REPO_ROOT / "configs" / "goal4_final.yaml"

_REQUIRED_ROW_FIELDS = (
    "expression_id",
    "schema_version",
    "run_id",
    "config_sha256",
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
    "difficulty_profile",
    "observed_ast_size",
    "resource_limits",
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
        generator_metadata={
            "achieved_source_ast_size": 3 + index % 5,
            "difficulty_profile": "stress" if index % 5 == 0 else "ordinary",
        },
    )


def _records(count: int = 25) -> list[ExpressionRecord]:
    return [_record(index) for index in range(count)]


def _config_dict(output_root: str, *, saturation_timeout: float = 0.2) -> dict:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "output_root": output_root,
        "include_optional_domain_rules": False,
        "modes": ["safe_real", "positive_real_formal"],
        "sampling": {
            "seed": 7,
            "target_size": 25,
            "balance_axes": [
                "operator_family",
                "domain_mode",
                "split",
                "size_bucket",
                "difficulty_profile",
            ],
            "size_bucket_edges": [4, 8, 16, 32],
        },
        "resources": {
            "max_iterations": 20,
            "max_egraph_nodes": 1000,
            "max_rewrite_attempts": 2000,
            "saturation_timeout_seconds": saturation_timeout,
            "max_eclasses": None,
            "extraction_max_depth": 6,
            "extraction_beam_width": 3,
            "extraction_max_candidates": 8,
            "extraction_max_nodes": 5000,
            "extraction_max_iterations": 10000,
            "extraction_timeout_seconds": 0.2,
        },
        "processing": {
            "chunk_size": 8,
            "checkpoint_every_chunks": 1,
            "worker_processes": 1,
            "resume": True,
        },
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
        assert "difficulty_profile" in config.sampling.balance_axes
        assert config.processing.worker_processes > 1

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

    @pytest.mark.parametrize(
        ("mutator", "message"),
        [
            (
                lambda data: data["modes"].pop(),
                "exactly safe_real and positive_real_formal",
            ),
            (
                lambda data: data["sampling"].update(
                    {"balance_axes": ["operator_family", "unknown"]}
                ),
                "unknown balance axes",
            ),
            (
                lambda data: data["sampling"].update({"size_bucket_edges": [8, 4]}),
                "strictly increasing",
            ),
        ],
    )
    def test_invalid_scientific_config_is_rejected(self, tmp_path: Path, mutator, message):
        data = _config_dict(str(tmp_path / "out"))
        mutator(data)
        with pytest.raises(ValueError, match=message):
            Goal4Config.model_validate(data)


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

    def test_balancing_uses_observed_size_and_difficulty(self, tmp_path: Path):
        config = _config(tmp_path)
        item = item_from_record(_record(5), config.sampling)
        assert item.observed_ast_size == 3
        assert item.observed_size_source.endswith("achieved_source_ast_size")
        assert item.difficulty_profile == "stress"

    def test_duplicate_expression_ids_are_rejected(self, tmp_path: Path):
        config = _config(tmp_path)
        item = item_from_record(_record(1), config.sampling)
        with pytest.raises(Goal4RuntimeError, match="duplicate expression_id"):
            select_subset([item, item], config.sampling)


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

    def test_rows_are_bound_to_one_run_identity(self, default_stage):
        result, rows = default_stage
        assert {row["run_id"] for row in rows} == {result.run_id}
        assert all(row["config_sha256"] for row in rows)

    def test_compact_provenance_retains_every_application(self, default_stage):
        _result, rows = default_stage
        for row in rows:
            provenance = row["provenance"]
            if provenance is None:
                continue
            assert provenance["application_log_complete"] is True
            assert provenance["attempt_aggregates_complete"] is True
            assert provenance["attempt_count"] == row["rewrites_attempted"]
            assert len(provenance["applications"]) == row["rewrites_applied"]
            catalog_size = len(provenance["rule_catalog"])
            assert all(
                0 <= application[2] < catalog_size for application in provenance["applications"]
            )

    def test_cost_provenance_uses_goal3_direct_source_ast(self, default_stage):
        _result, rows = default_stage
        costed = [row for row in rows if row["input_cost_provenance"]]
        assert costed
        assert all(
            row["input_cost_provenance"]["input_kind"] == "source_ast"
            and row["input_cost_provenance"]["construction_path"] == "direct_hashcons"
            for row in costed
        )


class TestCheckpointAndResume:
    def test_checkpoint_is_created(self, default_stage):
        result, _rows = default_stage
        assert result.checkpoint_path.is_file()
        checkpoint = load_checkpoint(result.checkpoint_path)
        assert checkpoint.run_id == result.run_id
        assert checkpoint.total_units == 50
        assert len(checkpoint.completed_ids) == 50
        assert result.run_manifest_path.is_file()

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
        first = run_stage(config, "smoke", _records(6), run_directory)
        partial = read_jsonl(first.rows_path)[0]
        first.rows_path.write_text(json.dumps(partial) + "\n", encoding="utf-8")
        first.checkpoint_path.unlink()
        run_stage(config, "smoke", _records(6), run_directory)
        rows = read_jsonl(run_directory / "smoke.rows.jsonl")
        unit_ids = {(row["expression_id"], row["rewrite_mode"]) for row in rows}
        assert len(unit_ids) == 12
        assert len(rows) == 12

    def test_truncated_final_line_is_tolerated(self, tmp_path: Path):
        config = _config(tmp_path, count=6)
        run_directory = tmp_path / "run"
        first = run_stage(config, "smoke", _records(6), run_directory)
        valid = read_jsonl(first.rows_path)[0]
        first.rows_path.write_text(
            json.dumps(valid) + "\n{partial",
            encoding="utf-8",
        )
        first.checkpoint_path.unlink()
        result = run_stage(config, "smoke", _records(6), run_directory)
        rows = read_jsonl(result.rows_path)
        assert result.completed_units == 12
        assert len(rows) == 12

    def test_stale_config_resume_is_rejected(self, tmp_path: Path):
        config = _config(tmp_path, count=6)
        run_directory = tmp_path / "run"
        run_stage(config, "smoke", _records(6), run_directory)
        changed_sampling = config.sampling.model_copy(update={"seed": config.sampling.seed + 1})
        changed = config.model_copy(update={"sampling": changed_sampling})
        with pytest.raises(Goal4RuntimeError, match="different content"):
            run_stage(changed, "smoke", _records(6), run_directory)

    def test_duplicate_completed_unit_is_rejected(self, tmp_path: Path):
        config = _config(tmp_path, count=3)
        result = run_stage(config, "smoke", _records(3), tmp_path / "run")
        row = read_jsonl(result.rows_path)[0]
        append_jsonl(result.rows_path, (row,))
        with pytest.raises(Goal4RuntimeError, match="duplicate work unit"):
            run_stage(config, "smoke", _records(3), tmp_path / "run")

    def test_corrupt_completed_line_is_not_silently_ignored(self, tmp_path: Path):
        config = _config(tmp_path, count=3)
        result = run_stage(config, "smoke", _records(3), tmp_path / "run")
        payload = result.rows_path.read_text(encoding="utf-8")
        result.rows_path.write_text("{broken}\n" + payload, encoding="utf-8")
        with pytest.raises(Goal4RuntimeError, match="invalid JSONL record"):
            run_stage(config, "smoke", _records(3), tmp_path / "run")


class TestTimeoutHandling:
    def test_nonpositive_timeout_is_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="greater than 0"):
            _config(tmp_path, count=6, saturation_timeout=0.0)

    def test_timeout_marks_supported_row(self, tmp_path: Path, monkeypatch):
        import geml.egraph.rewrite_engine as engine

        monkeypatch.setattr(
            engine,
            "_expired",
            lambda _started, _timeout_seconds: True,
        )
        config = _config(tmp_path, count=1)
        record = _record(1)
        row = process_expression(
            item_from_record(record, config.sampling),
            config.resolved_modes()[0],
            config,
        )
        assert row["saturation_status"] == "timeout"
        assert row["timeout"] is True

    def test_timeout_rows_are_still_complete(self, tmp_path: Path):
        config = _config(tmp_path, count=6, saturation_timeout=1e-9)
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

    def test_no_successful_row_degrades_cost(self, default_stage):
        _result, rows = default_stage
        for row in rows:
            if row["eml_dag_cost_after"] is not None:
                assert row["eml_dag_cost_after"] <= row["eml_dag_cost_before"]

    def test_domain_assumptions_come_from_corpus_domain_not_rewrite_mode(self, tmp_path: Path):
        config = _config(tmp_path, count=1)
        record = _record(1).model_copy(update={"domain_mode": "positive_real"})
        row = process_expression(
            item_from_record(record, config.sampling),
            config.resolved_modes()[1],
            config,
        )
        assert "positive" in row["declared_assumptions"]["x"]


class TestProductionCorpusLoader:
    def test_tiny_manifest_uses_split_shards_and_manifest_root(self, tmp_path: Path):
        from geml.data.storage.manifests import (
            build_corpus_manifest,
            build_split_manifest,
            write_corpus_manifest,
        )
        from geml.data.storage.shards import write_shards

        config_path = tmp_path / "fixture.yaml"
        config_path.write_text("fixture: true\n", encoding="utf-8")
        split_manifests = []
        expected = []
        for index, split in enumerate(CorpusSplit):
            record = _record(index).model_copy(update={"split": split})
            expected.append(record)
            shards = write_shards(
                (record,),
                tmp_path / "data" / split.value,
                corpus_id="goal4-loader-fixture",
                split=split,
                schema_version="fixture-v1",
                minimum_rows=1,
                maximum_rows=1,
                allow_small_fixture=True,
                manifest_root=tmp_path,
            )
            split_manifests.append(build_split_manifest(shards))
        manifest = build_corpus_manifest(
            split_manifests,
            corpus_id="goal4-loader-fixture",
            schema_version="fixture-v1",
            config_path=config_path,
            generator_seed=7,
            git_commit="fixture",
            created_at=datetime(2026, 7, 24, tzinfo=UTC),
        )
        manifest_path = tmp_path / "manifests" / "corpus.manifest.json"
        write_corpus_manifest(manifest, manifest_path)
        assert _load_corpus_records(manifest_path) == expected
