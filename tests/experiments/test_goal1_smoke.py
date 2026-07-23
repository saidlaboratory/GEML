"""Fresh-clone smoke coverage for the Goal 1 production integration path."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from geml.ast.builder import build_ast_from_parsed
from geml.data.generation.generator import (
    GeneratorConfigurationError,
    derive_expression_id,
    generate_expression,
)
from geml.data.storage.manifests import (
    ManifestIntegrityError,
    load_corpus_manifest,
    validate_manifest,
)
from geml.data.storage.splits import assign_splits
from geml.experiments.goal1 import run as goal1_run
from geml.experiments.goal1.qa import (
    _actual_shard_paths,
    _ast_triviality_counts,
    _audit_exact_number_bounds,
    _audit_log_arguments,
    _audit_power_and_division_guards,
    _audit_symbols,
    _audit_tan_arguments,
    _expected_required_operator_groups,
    _generator_provenance_violations,
    _metadata_counts,
    _missing_required_operator_groups,
    _nonnegative_int_mapping,
    _row_accounting_violations,
    _split_assignment_violations,
    _structural_identity,
    _target_support_violations,
    _triviality_admission_history_violations,
    _triviality_metadata_matches_ast,
    _validate_triviality_policy_error,
    compare_corpus_runs,
)
from geml.experiments.goal1.run import (
    Goal1ConfigurationError,
    Goal1Stage,
    StageGateError,
    StageOverride,
    StageRunResult,
    final_family_blockers,
    load_goal1_config,
    require_final_stage_ready,
    run_corpus_stage,
    run_selected_stage,
)
from geml.parsing.srepr import parse_srepr

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "goal1_final.yaml"
EXPECTED_TINY_CORPUS_HASH = "bf782b2ec446348d6a212522eb1a98333082facf5b9efd3e78d0b1cef883ad8c"
TINY_OVERRIDE = StageOverride(
    count=25,
    family_counts={
        "algebraic_core": 7,
        "powers_division_rationals": 4,
        "exp_log": 4,
        "trig_hyperbolic": 4,
        "mixed_elementary": 3,
        "ood_stress": 3,
    },
    split_counts={
        "train": 16,
        "validation": 3,
        "test_iid": 3,
        "test_ood": 3,
    },
)


def _create_directory_redirect(link: Path, target: Path) -> None:
    """Create a test-only directory redirect on POSIX or NTFS without privileges."""

    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"directory symlinks are unavailable: {error}")
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"directory redirects are unavailable: {completed.stderr.strip()}")


def _completed_stage_result(tmp_path: Path) -> StageRunResult:
    return StageRunResult(
        stage="final",
        run_label="run",
        run_id="final-run",
        output_root=str(tmp_path / "final" / "run"),
        manifest_path=str(tmp_path / "final" / "run" / "manifests" / "corpus.manifest.json"),
        qa_report_path=str(tmp_path / "final" / "run" / "qa.report.json"),
        run_metadata_path=str(tmp_path / "final" / "run" / "run.metadata.json"),
        passed=True,
        resumed=True,
        corpus_hash="a" * 64,
        row_accounting={"attempted": 250_000, "finalized_rows": 250_000},
        telemetry={"peak_resident_memory_bytes": 1_000_000},
        distributions={},
        caveats=(),
    )


def test_tiny_pipeline_is_deterministic_resumable_and_manifest_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    loaded = load_goal1_config(CONFIG_PATH)
    artifact_root = tmp_path / "isolated-artifacts"

    first = run_corpus_stage(
        loaded,
        stage="development",
        run_label="tiny-a",
        output_root=artifact_root,
        override=TINY_OVERRIDE,
    )
    assert first.passed
    assert first.corpus_hash == EXPECTED_TINY_CORPUS_HASH
    assert first.row_accounting == {
        "attempted": 25,
        "generated": 25,
        "accepted": 25,
        "duplicates": 0,
        "triviality_rejections": 0,
        "internal_triviality_retries": 0,
        "policy_rejections": 0,
        "unsupported": 0,
        "parse_failures": 0,
        "AST_validation_failures": 0,
        "display_failures": 0,
        "LaTeX_failures": 0,
        "roundtrip_audit_failures": 0,
        "storage_failures": 0,
        "finalized_rows": 25,
        "acceptance_rate": 1.0,
    }

    manifest = load_corpus_manifest(first.manifest_path)
    validation = validate_manifest(
        manifest,
        first.output_root,
        config_path=CONFIG_PATH,
        manifest_dir=Path(first.output_root) / "manifests",
    )
    assert validation.valid
    assert validation.validated_row_count == 25
    assert len({shard.checksum.digest for split in manifest.splits for shard in split.shards}) == 4

    qa_payload = json.loads(Path(first.qa_report_path).read_text(encoding="utf-8"))
    assert qa_payload["passed"]
    assert qa_payload["counts"]["unique_expression_ids"] == 25
    assert qa_payload["counts"]["unique_authoritative_srepr"] == 25
    assert qa_payload["adapters"]["parsed_and_ast_validated"] == 25
    assert qa_payload["adapters"]["display_validated"] == 25
    assert qa_payload["adapters"]["latex_validated"] == 25
    for adapter in ("parse", "ast", "display", "latex"):
        assert qa_payload["adapters"][adapter] == {"attempted": 25, "succeeded": 25}
    assert qa_payload["policy"]["manifest_seed_matches_generator_policy"]
    assert qa_payload["policy"]["frozen_split_assignment_check"]
    assert qa_payload["integrity"]["row_accounting"] == first.row_accounting
    assert qa_payload["integrity"]["deduplication"] == {
        "processed_count": 25,
        "unique_count": 25,
        "duplicate_count": 0,
        "identity_conflict_count": 0,
    }
    assert qa_payload["integrity"]["retained_error_stages"] == {}
    assert qa_payload["triviality"]["selection_policy"]["enforced"] is False
    assert (
        qa_payload["triviality"]["selection_policy"]["selected_record_counts"]
        == (qa_payload["triviality"]["record_counts"])
    )
    original_qa_bytes = Path(first.qa_report_path).read_bytes()

    resumed = run_corpus_stage(
        loaded,
        stage="development",
        run_label="tiny-a",
        output_root=artifact_root,
        override=TINY_OVERRIDE,
    )
    assert resumed.resumed
    assert resumed.corpus_hash == EXPECTED_TINY_CORPUS_HASH
    assert Path(first.qa_report_path).read_bytes() == original_qa_bytes

    second = run_corpus_stage(
        loaded,
        stage="development",
        run_label="tiny-b",
        output_root=artifact_root,
        override=TINY_OVERRIDE,
    )
    comparison = compare_corpus_runs(
        first.manifest_path,
        first.output_root,
        second.manifest_path,
        second.output_root,
    )
    assert comparison.passed
    assert comparison.first_corpus_hash == comparison.second_corpus_hash
    assert comparison.first_manifest_hash == comparison.second_manifest_hash
    assert not (tmp_path / "outputs").exists()


def test_final_gate_accepts_the_exact_six_family_policy(tmp_path: Path) -> None:
    loaded = load_goal1_config(CONFIG_PATH)
    assert final_family_blockers() == {}
    require_final_stage_ready(loaded)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("section", "field", "invalid_value"),
    [
        ("deduplication", "identity_fields", ["sympy_srepr"]),
        ("deduplication", "checkpoint_policy", "after_generation"),
        ("shards", "parquet_compression", "snappy"),
        ("shards", "checksum_algorithm", "md5"),
    ],
)
def test_upstream_storage_policy_drift_fails_closed(
    section: str,
    field: str,
    invalid_value: object,
) -> None:
    loaded = load_goal1_config(CONFIG_PATH)
    corpus_config = json.loads(json.dumps(loaded.corpus_config))
    corpus_config[section][field] = invalid_value
    drifted = goal1_run.LoadedGoal1Config(
        policy=loaded.policy,
        config_path=loaded.config_path,
        repository_root=loaded.repository_root,
        generator_config=loaded.generator_config,
        corpus_config=corpus_config,
    )

    with pytest.raises(Goal1ConfigurationError, match="policy conflicts"):
        goal1_run._validate_upstream_alignment(drifted)


@pytest.mark.parametrize("stage", [Goal1Stage.PILOT, Goal1Stage.FINAL])
def test_fixture_overrides_cannot_poison_production_stages(
    tmp_path: Path,
    stage: Goal1Stage,
) -> None:
    artifact_root = tmp_path / "isolated-artifacts"
    with pytest.raises(Goal1ConfigurationError, match="fixture overrides"):
        run_corpus_stage(
            load_goal1_config(CONFIG_PATH),
            stage=stage,
            output_root=artifact_root,
            override=TINY_OVERRIDE,
        )
    assert not artifact_root.exists()


def test_run_label_traversal_is_rejected_before_artifact_creation(tmp_path: Path) -> None:
    artifact_root = tmp_path / "isolated-artifacts"
    with pytest.raises(Goal1ConfigurationError, match="run_label"):
        run_corpus_stage(
            load_goal1_config(CONFIG_PATH),
            stage=Goal1Stage.DEVELOPMENT,
            run_label="../escape",
            output_root=artifact_root,
            override=TINY_OVERRIDE,
        )
    assert not artifact_root.exists()
    assert not (tmp_path / "escape").exists()


def test_public_final_api_requires_prior_gates_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_goal1_config(CONFIG_PATH)
    monkeypatch.setattr(goal1_run, "_find_completed_stage_run", lambda *args, **kwargs: None)

    def missing_gates(*args: Any, **kwargs: Any) -> Any:
        raise StageGateError("missing validated pilot gates")

    def unexpected_materialization(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("final materialization started before prior gates passed")

    monkeypatch.setattr(goal1_run, "_require_prior_gates", missing_gates)
    monkeypatch.setattr(goal1_run, "_run_corpus_stage_impl", unexpected_materialization)
    with pytest.raises(StageGateError, match="missing validated pilot gates"):
        run_corpus_stage(loaded, stage=Goal1Stage.FINAL)


def test_valid_completed_final_skips_launch_capacity_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_goal1_config(CONFIG_PATH)
    completed = _completed_stage_result(tmp_path)
    monkeypatch.setattr(
        goal1_run,
        "_find_completed_stage_run",
        lambda *args, **kwargs: completed,
    )

    def unexpected_gate(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("launch-only gate ran for an already validated final corpus")

    monkeypatch.setattr(goal1_run, "_require_prior_gates", unexpected_gate)
    monkeypatch.setattr(goal1_run, "_require_final_memory_capacity", unexpected_gate)
    monkeypatch.setattr(goal1_run, "_require_final_disk_capacity", unexpected_gate)
    monkeypatch.setattr(goal1_run, "_run_corpus_stage_impl", unexpected_gate)

    assert run_corpus_stage(loaded, stage=Goal1Stage.FINAL) is completed
    selected = run_selected_stage(loaded, stage=Goal1Stage.FINAL)
    assert selected["passed"]
    assert selected["preflight_status"] == "skipped_for_valid_completed_run"
    assert selected["run"] == completed.to_dict()


def test_run_lease_allows_exactly_one_owner_and_is_reusable(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    first = goal1_run._acquire_run_lease(run_root)
    try:
        with pytest.raises(StageGateError, match="another process owns"):
            goal1_run._acquire_run_lease(run_root)
    finally:
        first.release()

    second = goal1_run._acquire_run_lease(run_root)
    second.release()
    assert (run_root / "run.lease").is_file()
    assert (run_root / "run.lease").stat().st_size == 0
    assert (run_root / "run.lock.json").is_file()


@pytest.mark.parametrize("lease_name", ["run.lease", "run.lock.json"])
def test_run_lease_rejects_hard_link_without_mutating_target(
    tmp_path: Path,
    lease_name: str,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    sentinel = tmp_path / "external-sentinel.json"
    expected = b'{"sentinel":"do-not-modify"}\n'
    sentinel.write_bytes(expected)
    try:
        os.link(sentinel, run_root / lease_name)
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")

    with pytest.raises(ManifestIntegrityError, match="hard-linked"):
        goal1_run._acquire_run_lease(run_root)

    assert sentinel.read_bytes() == expected


def test_nested_shard_redirect_remains_lexically_visible_and_is_rejected(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    train_root = run_root / "data" / "train"
    alias_parent = run_root / "data" / "validation"
    alias_parent.mkdir(parents=True)
    train_root.mkdir(parents=True)
    shard_path = train_root / "train-00000.parquet"
    shard_path.write_bytes(b"not-a-real-parquet-file")
    alias_root = alias_parent / "nested"
    _create_directory_redirect(alias_root, train_root)

    actual_paths = _actual_shard_paths(run_root)
    assert Path("data/train/train-00000.parquet") in actual_paths
    assert Path("data/validation/nested/train-00000.parquet") in actual_paths
    with pytest.raises(ManifestIntegrityError, match="filesystem redirect"):
        goal1_run._require_safe_artifact_layout(run_root)


def test_external_shard_redirect_is_rejected_before_descent(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    alias_root = run_root / "data" / "validation" / "external"
    external_root = tmp_path / "external"
    alias_root.parent.mkdir(parents=True)
    external_root.mkdir()
    (external_root / "outside.parquet").write_bytes(b"not-a-real-parquet-file")
    _create_directory_redirect(alias_root, external_root)

    with pytest.raises(ManifestIntegrityError, match="escapes its root"):
        _actual_shard_paths(run_root)


def test_shard_redirect_cycle_terminates_without_duplicate_paths(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    train_root = run_root / "data" / "train"
    train_root.mkdir(parents=True)
    shard_path = train_root / "train-00000.parquet"
    shard_path.write_bytes(b"not-a-real-parquet-file")
    _create_directory_redirect(train_root / "loop", train_root)

    assert _actual_shard_paths(run_root) == {Path("data/train/train-00000.parquet")}


def test_redirected_dedup_state_is_rejected_without_deleting_target(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    external_state = tmp_path / "external-state"
    run_root.mkdir()
    external_state.mkdir()
    sentinel = external_state / "dedup.sqlite3"
    sentinel.write_bytes(b"do-not-delete")
    _create_directory_redirect(run_root / "state", external_state)

    with pytest.raises(ManifestIntegrityError, match="filesystem redirect"):
        goal1_run._require_safe_artifact_layout(run_root)
    assert sentinel.read_bytes() == b"do-not-delete"


def test_stage_directory_alias_is_rejected_even_when_target_stays_inside_output(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    aliased_target = output_root / "pilot-target"
    output_root.mkdir()
    aliased_target.mkdir()
    _create_directory_redirect(output_root / "pilot", aliased_target)

    with pytest.raises(Goal1ConfigurationError, match="filesystem redirect"):
        goal1_run._run_root(output_root, Goal1Stage.PILOT, "run-a")


def test_orphan_post_manifest_marker_blocks_replay_before_dedup_reset(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "isolated-artifacts"
    run_root = artifact_root / "development" / "orphan"
    run_root.mkdir(parents=True)
    (run_root / "stage.result.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ManifestIntegrityError, match=r"without corpus\.manifest\.json"):
        run_corpus_stage(
            load_goal1_config(CONFIG_PATH),
            stage=Goal1Stage.DEVELOPMENT,
            run_label="orphan",
            output_root=artifact_root,
            override=TINY_OVERRIDE,
        )
    assert not (run_root / "state" / "dedup.sqlite3").exists()


def test_boolean_peak_memory_cannot_satisfy_final_capacity_gate(tmp_path: Path) -> None:
    loaded = load_goal1_config(CONFIG_PATH)
    invalid = _completed_stage_result(tmp_path)
    invalid = StageRunResult(
        **{
            **invalid.to_dict(),
            "telemetry": {"peak_resident_memory_bytes": True},
        }
    )
    with pytest.raises(StageGateError, match="positive integer"):
        goal1_run._require_final_memory_capacity(loaded, (invalid, invalid))


def test_cli_reports_known_generator_configuration_failures_as_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_stage(*args: Any, **kwargs: Any) -> Any:
        raise GeneratorConfigurationError("invalid generator policy")

    monkeypatch.setattr(goal1_run, "run_selected_stage", fail_stage)
    exit_code = goal1_run.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--stage",
            "development",
            "--output-root",
            str(tmp_path / "artifacts"),
        ]
    )
    payload = json.loads(capsys.readouterr().err)
    assert exit_code == 2
    assert payload["error_type"] == "GeneratorConfigurationError"
    assert payload["message"] == "invalid generator policy"


def test_cli_reports_filesystem_failures_as_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_stage(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError()

    monkeypatch.setattr(goal1_run, "run_selected_stage", fail_stage)
    exit_code = goal1_run.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--stage",
            "development",
            "--output-root",
            str(tmp_path / "artifacts"),
        ]
    )
    payload = json.loads(capsys.readouterr().err)
    assert exit_code == 2
    assert payload["error_type"] == "PermissionError"
    assert payload["message"] == "PermissionError"


def test_completed_stage_metadata_tampering_fails_and_missing_marker_recovers(
    tmp_path: Path,
) -> None:
    loaded = load_goal1_config(CONFIG_PATH)
    artifact_root = tmp_path / "isolated-artifacts"
    first = run_corpus_stage(
        loaded,
        stage=Goal1Stage.DEVELOPMENT,
        run_label="metadata-audit",
        output_root=artifact_root,
        override=TINY_OVERRIDE,
    )
    run_root = Path(first.output_root)
    marker_path = run_root / "stage.result.json"
    metadata_path = run_root / "run.metadata.json"
    qa_path = run_root / "qa.report.json"
    original_marker = marker_path.read_text(encoding="utf-8")
    original_metadata = metadata_path.read_text(encoding="utf-8")

    marker = json.loads(original_marker)
    marker["telemetry"]["peak_resident_memory_bytes"] = True
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ManifestIntegrityError, match=r"metadata|peak-memory"):
        run_corpus_stage(
            loaded,
            stage=Goal1Stage.DEVELOPMENT,
            run_label="metadata-audit",
            output_root=artifact_root,
            override=TINY_OVERRIDE,
        )
    marker_path.write_text(original_marker, encoding="utf-8")

    metadata = json.loads(original_metadata)
    metadata["random_seed"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ManifestIntegrityError, match="run metadata"):
        run_corpus_stage(
            loaded,
            stage=Goal1Stage.DEVELOPMENT,
            run_label="metadata-audit",
            output_root=artifact_root,
            override=TINY_OVERRIDE,
        )
    metadata_path.write_text(original_metadata, encoding="utf-8")

    marker_path.unlink()
    metadata_path.unlink()
    qa_path.unlink()
    (run_root / "run.failure.json").write_text('{"stale":true}\n', encoding="utf-8")
    recovered = run_corpus_stage(
        loaded,
        stage=Goal1Stage.DEVELOPMENT,
        run_label="metadata-audit",
        output_root=artifact_root,
        override=TINY_OVERRIDE,
    )
    assert recovered.resumed
    assert recovered.telemetry["recovered_after_manifest"] is True
    assert not (run_root / "run.failure.json").exists()
    assert marker_path.is_file()
    assert metadata_path.is_file()
    assert qa_path.is_file()

    validated_again = run_corpus_stage(
        loaded,
        stage=Goal1Stage.DEVELOPMENT,
        run_label="metadata-audit",
        output_root=artifact_root,
        override=TINY_OVERRIDE,
    )
    assert validated_again.resumed
    assert validated_again.corpus_hash == first.corpus_hash


def test_post_manifest_failure_is_recoverable_and_retained_in_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_goal1_config(CONFIG_PATH)
    artifact_root = tmp_path / "isolated-artifacts"
    original_qa = goal1_run.run_corpus_qa

    def fail_qa(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("injected QA interruption")

    monkeypatch.setattr(goal1_run, "run_corpus_qa", fail_qa)
    with pytest.raises(RuntimeError, match="injected QA interruption"):
        run_corpus_stage(
            loaded,
            stage=Goal1Stage.DEVELOPMENT,
            run_label="qa-recovery",
            output_root=artifact_root,
            override=TINY_OVERRIDE,
        )

    run_root = artifact_root / "development" / "qa-recovery"
    assert (run_root / "manifests" / "corpus.manifest.json").is_file()
    assert (run_root / "run.failure.json").is_file()
    history_path = run_root / "run.failure-history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert len(history) == 1
    assert history[0]["pipeline_stage"] == "qa"
    assert history[0]["error_type"] == "RuntimeError"

    monkeypatch.setattr(goal1_run, "run_corpus_qa", original_qa)
    recovered = run_corpus_stage(
        loaded,
        stage=Goal1Stage.DEVELOPMENT,
        run_label="qa-recovery",
        output_root=artifact_root,
        override=TINY_OVERRIDE,
    )
    assert recovered.resumed
    assert recovered.passed
    assert any("prior failed or interrupted" in caveat for caveat in recovered.caveats)
    assert not (run_root / "run.failure.json").exists()
    assert json.loads(history_path.read_text(encoding="utf-8")) == history


@pytest.mark.parametrize(
    ("source", "expected_class"),
    [
        ("tan(sin(Symbol('x', real=True)))", "sin"),
        ("tan(cos(Symbol('x', real=True)))", "cos"),
        ("tan(tanh(Symbol('x', real=True)))", "tanh"),
        ("tan(Integer(-1))", "exact_constant"),
        ("tan(Integer(1))", "exact_constant"),
        ("tan(Rational(1, 2))", "exact_constant"),
    ],
)
def test_tan_argument_audit_accepts_only_certified_unit_intervals(
    source: str,
    expected_class: str,
) -> None:
    tree = build_ast_from_parsed(parse_srepr(source), expression_id="a" * 64)
    checked, classes, violations = _audit_tan_arguments(tree)
    assert checked == 1
    assert classes == {expected_class: 1}
    assert violations == ()


@pytest.mark.parametrize(
    "source",
    [
        "tan(Symbol('x', real=True))",
        "tan(Integer(2))",
    ],
)
def test_tan_argument_audit_rejects_uncertified_or_out_of_range_arguments(
    source: str,
) -> None:
    tree = build_ast_from_parsed(
        parse_srepr(source),
        expression_id="a" * 64,
    )
    checked, classes, violations = _audit_tan_arguments(tree)
    assert checked == 1
    assert classes == {}
    assert violations == ("n000000",)


@pytest.mark.parametrize(
    ("source", "domain_mode", "expected_class"),
    [
        ("log(Symbol('x', positive=True))", "positive_real", "positive_variable"),
        ("log(Integer(2))", "safe_real", "positive_constant"),
        ("log(exp(Symbol('x', real=True)))", "safe_real", "exp"),
        ("log(cosh(Symbol('x', real=True)))", "safe_real", "cosh"),
        ("log(Add(Integer(1), Integer(2)))", "safe_real", "positive_sum"),
        (
            "log(Mul(Integer(2), Rational(1, 2)))",
            "safe_real",
            "positive_product",
        ),
    ],
)
def test_log_argument_audit_accepts_only_structurally_positive_arguments(
    source: str,
    domain_mode: str,
    expected_class: str,
) -> None:
    tree = build_ast_from_parsed(parse_srepr(source), expression_id="a" * 64)
    checked, classes, violations = _audit_log_arguments(tree, domain_mode=domain_mode)
    assert checked == 1
    assert classes == {expected_class: 1}
    assert violations == ()


def test_log_argument_audit_rejects_unproved_real_symbol() -> None:
    tree = build_ast_from_parsed(
        parse_srepr("log(Symbol('x', real=True))"),
        expression_id="a" * 64,
    )
    checked, classes, violations = _audit_log_arguments(tree, domain_mode="safe_real")
    assert checked == 1
    assert classes == {}
    assert violations == ("n000000",)


def test_symbol_audit_requires_exact_domain_assumptions_names_and_vocabulary() -> None:
    grammar = load_goal1_config(CONFIG_PATH).generator_config.grammar
    valid = build_ast_from_parsed(
        parse_srepr("Add(Symbol('x', positive=True), Symbol('y', positive=True))"),
        expression_id="a" * 64,
    )
    assert not _audit_symbols(
        valid,
        variables=("x", "y"),
        domain_mode="positive_real",
        grammar=grammar,
    )

    invalid = build_ast_from_parsed(
        parse_srepr("Add(Symbol('x', real=True), Symbol('rogue', real=True))"),
        expression_id="b" * 64,
    )
    violations = _audit_symbols(
        invalid,
        variables=("x", "y"),
        domain_mode="positive_real",
        grammar=grammar,
    )
    assert any("outside the grammar" in violation for violation in violations)
    assert any("assumptions" in violation for violation in violations)
    assert any("record.variables" in violation for violation in violations)


def test_exact_number_audit_enforces_integer_and_rational_bounds() -> None:
    grammar = load_goal1_config(CONFIG_PATH).generator_config.grammar
    valid = build_ast_from_parsed(
        parse_srepr("Add(Integer(-9), Rational(8, 9))"),
        expression_id="a" * 64,
    )
    assert not _audit_exact_number_bounds(valid, grammar)

    invalid = build_ast_from_parsed(
        parse_srepr("Add(Integer(-10), Rational(10, 11))"),
        expression_id="b" * 64,
    )
    violations = _audit_exact_number_bounds(invalid, grammar)
    assert len(violations) == 2


def test_power_audit_enforces_configured_exponents_and_positive_reciprocals() -> None:
    generator_config = load_goal1_config(CONFIG_PATH).generator_config
    grammar = generator_config.grammar
    safe = build_ast_from_parsed(
        parse_srepr("Mul(Symbol('x', real=True), Pow(exp(Symbol('y', real=True)), Integer(-1)))"),
        expression_id="a" * 64,
    )
    negative_powers, reciprocals, violations = _audit_power_and_division_guards(
        safe,
        domain_mode="safe_real",
        grammar=grammar,
    )
    assert (negative_powers, reciprocals, violations) == (1, 1, ())

    unsafe = build_ast_from_parsed(
        parse_srepr("Pow(Symbol('x', real=True), Integer(-4))"),
        expression_id="b" * 64,
    )
    _, _, violations = _audit_power_and_division_guards(
        unsafe,
        domain_mode="safe_real",
        grammar=grammar,
    )
    assert any("configured exponent" in violation for violation in violations)

    unproved = build_ast_from_parsed(
        parse_srepr("Pow(Symbol('x', real=True), Integer(-1))"),
        expression_id="c" * 64,
    )
    _, _, violations = _audit_power_and_division_guards(
        unproved,
        domain_mode="safe_real",
        grammar=grammar,
    )
    assert any("positive/nonzero" in violation for violation in violations)

    generated = generate_expression(
        generator_config,
        expression_index=0,
        family_id="powers_division_rationals",
        split="train",
    )
    assert generated.generator_metadata["domain_guards"] == {
        "log_arguments": "positive_expression_grammar",
        "division_denominators": "positive_expression_grammar",
        "negative_power_bases": "positive_expression_grammar",
        "tan_arguments": "closed_unit_interval_structural_grammar",
    }


def test_required_operator_groups_come_from_exact_generator_family_policy() -> None:
    config = load_goal1_config(CONFIG_PATH).generator_config
    assert _expected_required_operator_groups(config, "mixed_elementary") == (
        ("exp", "log"),
        ("sin", "cos", "tan", "sinh", "cosh", "tanh"),
    )
    assert _expected_required_operator_groups(config, "ood_stress") == ()

    exp_only = build_ast_from_parsed(
        parse_srepr("exp(Symbol('x', real=True))"),
        expression_id="a" * 64,
    )
    missing_metadata, missing_structure = _missing_required_operator_groups(
        exp_only,
        {"exp": 1, "symbol": 1},
        _expected_required_operator_groups(config, "mixed_elementary"),
    )
    expected_trig_group = (("sin", "cos", "tan", "sinh", "cosh", "tanh"),)
    assert missing_metadata == expected_trig_group
    assert missing_structure == expected_trig_group


def test_ood_audit_requires_exact_criterion_profile_and_feasible_target() -> None:
    config = load_goal1_config(CONFIG_PATH).generator_config
    record = generate_expression(
        config,
        expression_index=7,
        family_id="ood_stress",
        split="test_ood",
    )
    assert not _target_support_violations(record, config)

    invalid = record.model_copy(
        update={
            "target_ast_size": 2,
            "generator_metadata": {
                **record.generator_metadata,
                "difficulty_profile": "ordinary",
                "stress_criterion": "not-the-frozen-criterion",
            },
        }
    )
    violations = _target_support_violations(invalid, config)
    assert any("difficulty profile" in violation for violation in violations)
    assert any("stress criterion" in violation for violation in violations)
    assert any("target size" in violation for violation in violations)


def test_generator_provenance_audit_accepts_generated_record() -> None:
    config = load_goal1_config(CONFIG_PATH).generator_config
    record = generate_expression(
        config,
        expression_index=7,
        family_id="mixed_elementary",
        split="train",
    )
    assert not _generator_provenance_violations(record, config)


@pytest.mark.parametrize(
    ("record_update", "metadata_update", "message"),
    [
        ({"generator_seed": 0}, {}, "generator seed"),
        ({"operator_family": "unknown-family"}, {}, "no configured generator policy"),
        ({}, {"expression_index": True}, "expression index"),
        ({}, {"difficulty_profile": ""}, "difficulty profile"),
        ({}, {"stress_criterion": ""}, "stress criterion"),
        ({}, {"attempts": True}, "generation attempts"),
        ({}, {"rejected_attempts": 1}, "rejected attempts"),
        ({}, {"labeling_rejected_attempts": 1}, "labeling rejection"),
        (
            {},
            {"labeling_attempts": 0, "labeling_rejected_attempts": 1},
            "exceed total labeling attempts",
        ),
        (
            {},
            {"labeling_attempts": 49},
            "per-attempt maximum",
        ),
        (
            {},
            {"rejection_reasons": {"grammar:not_a_generator_code": 1}},
            "not generator-defined",
        ),
        (
            {},
            {"labeling_rejection_reasons": {"variable_placement:tampered": 0}},
            "must be positive",
        ),
        (
            {},
            {"labeling_rejection_reasons": {"unknown:tampered": 0}},
            "not generator-defined",
        ),
        (
            {},
            {"rejection_reasons": {"grammar:labeling_exhausted": 0}},
            "positive count",
        ),
        (
            {},
            {"rejection_reasons": {"grammar:labeling_exhausted": 1}},
            "grammar failure count exceeds",
        ),
        (
            {},
            {
                "attempts": 2,
                "rejected_attempts": 1,
                "rejection_reasons": {
                    "missing_required_operator_group:exp|log": 1,
                    "triviality_cap:log_one": 1,
                },
            },
            "cannot account",
        ),
        (
            {},
            {
                "attempts": 4,
                "rejected_attempts": 3,
                "rejection_reasons": {"triviality_cap:log_one": 1},
            },
            "cannot account",
        ),
        ({}, {"unexpected": 0}, "metadata keys"),
    ],
)
def test_generator_provenance_audit_rejects_tampering(
    record_update: dict[str, object],
    metadata_update: dict[str, object],
    message: str,
) -> None:
    config = load_goal1_config(CONFIG_PATH).generator_config
    record = generate_expression(
        config,
        expression_index=7,
        family_id="mixed_elementary",
        split="train",
    )
    tampered = record.model_copy(
        update={
            **record_update,
            "generator_metadata": {**record.generator_metadata, **metadata_update},
        }
    )
    assert any(
        message in violation for violation in _generator_provenance_violations(tampered, config)
    )


def test_ast_triviality_counts_cross_check_lowered_logical_metadata() -> None:
    tree = build_ast_from_parsed(
        parse_srepr("Mul(Integer(-1), Integer(1))"),
        expression_id="a" * 64,
    )
    ast_counts = _ast_triviality_counts(tree)
    assert ast_counts["multiplication_by_one"] == 1
    assert ast_counts["constant_only_subtrees"] == 1

    logical_counts = {
        "multiplication_by_one": 0,
        "log_one": 0,
        "constant_only_subtrees": 1,
        "exp_log": 0,
        "log_exp": 0,
    }
    matches, _ = _triviality_metadata_matches_ast(
        tree,
        ast_counts,
        logical_counts,
        {"negate": 1, "one": 1},
    )
    assert matches
    matches, _ = _triviality_metadata_matches_ast(
        tree,
        ast_counts,
        {**logical_counts, "log_one": 1},
        {"negate": 1, "one": 1},
    )
    assert not matches


def test_unrelated_lowerings_cannot_hide_ast_triviality_events() -> None:
    multiplication_tree = build_ast_from_parsed(
        parse_srepr(
            "Add(Mul(Integer(1), Symbol('x', real=True)), Mul(Integer(-1), Symbol('y', real=True)))"
        ),
        expression_id="a" * 64,
    )
    multiplication_counts = _ast_triviality_counts(multiplication_tree)
    assert multiplication_counts["multiplication_by_one"] == 1
    metadata_counts = {
        "multiplication_by_one": 0,
        "log_one": 0,
        "constant_only_subtrees": 0,
        "exp_log": 0,
        "log_exp": 0,
    }
    matches, bounds = _triviality_metadata_matches_ast(
        multiplication_tree,
        multiplication_counts,
        metadata_counts,
        {"add": 1, "multiply": 1, "negate": 1, "one": 1, "symbol": 2},
    )
    assert not matches
    assert bounds["multiplication_by_one"] == (1, 1)

    constant_tree = build_ast_from_parsed(
        parse_srepr(
            "Add(Mul(Integer(2), Integer(3)), "
            "Add(Symbol('x', real=True), Mul(Integer(-1), Symbol('y', real=True))))"
        ),
        expression_id="b" * 64,
    )
    constant_counts = _ast_triviality_counts(constant_tree)
    assert constant_counts["constant_only_subtrees"] == 1
    matches, bounds = _triviality_metadata_matches_ast(
        constant_tree,
        constant_counts,
        metadata_counts,
        {"add": 1, "multiply": 1, "subtract": 1, "integer": 2, "symbol": 2},
    )
    assert not matches
    assert bounds["constant_only_subtrees"] == (1, 1)


@pytest.mark.parametrize(
    "raw",
    [
        {"count": True},
        {"count": -1},
        {"count": 1.0},
        {"count": "1"},
        {"": 1},
    ],
)
def test_nonnegative_metadata_count_parser_rejects_coercible_or_invalid_values(
    raw: object,
) -> None:
    parsed, detail = _nonnegative_int_mapping(raw)
    assert parsed is None
    assert detail


def test_nonnegative_metadata_count_parser_enforces_exact_keys() -> None:
    assert _nonnegative_int_mapping({"count": 0}, expected_keys={"count"}) == (
        {"count": 0},
        None,
    )
    parsed, detail = _nonnegative_int_mapping(
        {"count": 0, "extra": 0},
        expected_keys={"count"},
    )
    assert parsed is None
    assert "extra" in str(detail)

    failures = []
    assert (
        _metadata_counts(
            {"count": "1"},
            failures=failures,
            stage="operator_policy",
            error_type="InvalidCounts",
            message="bad counts",
            expression_id="a" * 64,
            expected_keys={"count"},
        )
        == {}
    )
    assert len(failures) == 1
    assert failures[0].error_type == "InvalidCounts"
    assert failures[0].expression_id == "a" * 64


def test_frozen_split_assignment_is_recomputed_without_materializing_another_corpus() -> None:
    config = load_goal1_config(CONFIG_PATH).generator_config
    source_records = tuple(
        generate_expression(
            config,
            expression_index=index,
            family_id=family,
            split="train",
        )
        for index, family in enumerate(
            ("algebraic_core", "powers_division_rationals", "exp_log", "ood_stress")
        )
    )
    assignment = assign_splits(
        source_records,
        {"train": 1, "validation": 1, "test_iid": 1, "test_ood": 1},
        seed=config.run_seed,
    )
    ordered = tuple(assignment.iter_records())
    assert not _split_assignment_violations(ordered, seed=config.run_seed)

    wrongly_assigned = (
        ordered[1].model_copy(update={"split": ordered[0].split}),
        ordered[0].model_copy(update={"split": ordered[1].split}),
        *ordered[2:],
    )
    assert _split_assignment_violations(wrongly_assigned, seed=config.run_seed)


def test_row_accounting_conserves_rows_without_treating_internal_retries_as_rows() -> None:
    accounting = {
        "attempted": 5,
        "generated": 4,
        "accepted": 2,
        "duplicates": 1,
        "triviality_rejections": 1,
        "internal_triviality_retries": 7,
        "policy_rejections": 1,
        "unsupported": 0,
        "parse_failures": 0,
        "AST_validation_failures": 0,
        "display_failures": 0,
        "LaTeX_failures": 0,
        "roundtrip_audit_failures": 0,
        "storage_failures": 0,
        "finalized_rows": 2,
    }
    assert not _row_accounting_violations(accounting, loaded_rows=2)

    invalid = {**accounting, "generated": 5}
    violations = _row_accounting_violations(invalid, loaded_rows=2)
    assert any(error_type == "RowAccountingConservationError" for error_type, _ in violations)


def test_corpus_triviality_limits_are_exact_and_reject_only_saturated_features() -> None:
    config = load_goal1_config(CONFIG_PATH).generator_config
    limits = goal1_run._corpus_triviality_record_limits(config, 250_000)
    assert limits == {
        "multiplication_by_one": 50_000,
        "log_one": 20_000,
        "constant_only_subtrees": 175_000,
        "exp_log": 87_500,
        "log_exp": 100_000,
    }

    selected_counts = {feature: 0 for feature in limits}
    selected_counts["multiplication_by_one"] = limits["multiplication_by_one"]
    assert goal1_run._corpus_triviality_rejection_features(
        ("multiplication_by_one", "log_one"),
        selected_counts,
        limits,
    ) == ("multiplication_by_one",)


def test_retained_triviality_rejection_proves_its_exact_decision() -> None:
    config = load_goal1_config(CONFIG_PATH).generator_config
    record = generate_expression(
        config,
        expression_index=0,
        family_id="algebraic_core",
        split="train",
    )
    counts = {feature: 0 for feature in config.triviality.per_expression_caps}
    counts["multiplication_by_one"] = 1
    record = record.model_copy(
        update={
            "generator_metadata": {
                **record.generator_metadata,
                "triviality_counts": counts,
            }
        }
    )
    limits = goal1_run._corpus_triviality_record_limits(config, 25)
    selected_counts = {feature: 0 for feature in limits}
    selected_counts["multiplication_by_one"] = limits["multiplication_by_one"]
    retained = goal1_run._triviality_policy_error_row(
        record,
        record_features=("multiplication_by_one",),
        blocked_features=("multiplication_by_one",),
        selected_record_counts=selected_counts,
        record_limits=limits,
    )
    assert (
        _validate_triviality_policy_error(retained, limits)
        == "corpus_triviality_cap:multiplication_by_one"
    )

    tampered = retained.model_copy(
        update={
            "metadata": {
                **retained.metadata,
                "blocked_features": ["log_one"],
            }
        }
    )
    with pytest.raises(ValueError, match="blocked triviality features"):
        _validate_triviality_policy_error(tampered, limits)


def test_triviality_admission_history_rejects_index_and_pre_count_tampering() -> None:
    zero_counts = {feature: 0 for feature in goal1_run.TRIVIALITY_FEATURES}
    one_log_count = {**zero_counts, "log_one": 1}
    accepted = ((0, ("log_one",)),)
    rejected = ((1, one_log_count),)
    assert not _triviality_admission_history_violations(
        accepted,
        rejected,
        final_selected_counts=one_log_count,
        attempted_count=2,
    )

    duplicate_index = _triviality_admission_history_violations(
        accepted,
        ((0, one_log_count),),
        final_selected_counts=one_log_count,
        attempted_count=2,
    )
    assert any(
        error_type == "DuplicateTrivialityAdmissionIndex" for error_type, _ in duplicate_index
    )

    wrong_pre_counts = _triviality_admission_history_violations(
        accepted,
        ((1, zero_counts),),
        final_selected_counts=one_log_count,
        attempted_count=2,
    )
    assert any(
        error_type == "TrivialityAdmissionHistoryMismatch" for error_type, _ in wrong_pre_counts
    )


def test_runner_enforces_and_reconciles_corpus_triviality_caps(tmp_path: Path) -> None:
    loaded = load_goal1_config(CONFIG_PATH)
    corpus_rate_caps = {
        **dict(loaded.generator_config.triviality.corpus_rate_caps),
        "multiplication_by_one": 0.0,
    }
    generator_config = loaded.generator_config.model_copy(
        update={
            "triviality": loaded.generator_config.triviality.model_copy(
                update={"corpus_rate_caps": corpus_rate_caps}
            )
        }
    )
    policy = loaded.policy.model_copy(
        update={
            "quality": loaded.policy.quality.model_copy(
                update={"triviality_rate_gate_minimum_rows": 1}
            )
        }
    )
    enforced = goal1_run.LoadedGoal1Config(
        policy=policy,
        config_path=loaded.config_path,
        repository_root=loaded.repository_root,
        generator_config=generator_config,
        corpus_config=loaded.corpus_config,
    )
    result = run_corpus_stage(
        enforced,
        stage=Goal1Stage.DEVELOPMENT,
        run_label="forced-triviality-cap",
        output_root=tmp_path / "isolated-artifacts",
        override=TINY_OVERRIDE,
    )

    assert result.passed
    assert result.row_accounting["accepted"] == TINY_OVERRIDE.count
    assert result.row_accounting["triviality_rejections"] > 0
    assert result.distributions["families"] == dict(TINY_OVERRIDE.family_counts)

    qa_payload = json.loads(Path(result.qa_report_path).read_text(encoding="utf-8"))
    assert qa_payload["passed"]
    assert qa_payload["triviality"]["record_counts"]["multiplication_by_one"] == 0
    assert qa_payload["triviality"]["selection_policy"]["enforced"] is True
    cap_rejections = result.row_accounting["triviality_rejections"]
    assert qa_payload["integrity"]["retained_triviality_policy_identities"] == cap_rejections
    assert qa_payload["integrity"]["deduplication"]["unique_count"] == (
        TINY_OVERRIDE.count + cap_rejections
    )

    rejection_counts = qa_payload["triviality"]["rejection_counts"]
    corpus_rate_rejections = {
        reason: count
        for reason, count in rejection_counts.items()
        if reason.startswith("corpus_triviality_cap:")
    }
    assert corpus_rate_rejections["corpus_triviality_cap:multiplication_by_one"] > 0
    assert sum(corpus_rate_rejections.values()) == cap_rejections

    retained_rows = [
        json.loads(line)
        for line in (Path(result.output_root) / "errors.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(retained_rows) == cap_rejections
    assert {row["stage"] for row in retained_rows} == {"triviality_policy"}

    errors_path = Path(result.output_root) / "errors.jsonl"
    original_errors = errors_path.read_text(encoding="utf-8")
    tampered_pre_counts = retained_rows[0]["metadata"]["selected_record_counts"]
    record_features = retained_rows[0]["metadata"]["record_triviality_features"]
    record_limits = retained_rows[0]["metadata"]["record_limits"]
    feature = next(
        name
        for name in goal1_run.TRIVIALITY_FEATURES
        if name not in record_features and record_limits[name] > 0
    )
    tampered_pre_counts[feature] = 1 if tampered_pre_counts[feature] != 1 else 0
    errors_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained_rows),
        encoding="utf-8",
    )
    tampered_pre_count_report = goal1_run.run_corpus_qa(
        result.manifest_path,
        result.output_root,
        config_path=CONFIG_PATH,
        generator_config=generator_config,
        expectations=goal1_run._qa_expectations(enforced, TINY_OVERRIDE),
        manifest_dir=Path(result.output_root) / "manifests",
    )
    assert any(
        failure["error_type"] == "TrivialityAdmissionHistoryMismatch"
        for failure in tampered_pre_count_report.failures
    )

    errors_path.write_text(original_errors, encoding="utf-8")
    retained_rows = [json.loads(line) for line in original_errors.splitlines()]
    manifest = load_corpus_manifest(result.manifest_path)
    first_shard = manifest.splits[0].shards[0]
    first_row = pq.read_table(Path(result.output_root) / first_shard.path).to_pylist()[0]
    accepted_index = json.loads(first_row["generator_metadata_json"])["expression_index"]
    retained_rows[0]["metadata"]["expression_index"] = accepted_index
    errors_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained_rows),
        encoding="utf-8",
    )
    tampered_index_report = goal1_run.run_corpus_qa(
        result.manifest_path,
        result.output_root,
        config_path=CONFIG_PATH,
        generator_config=generator_config,
        expectations=goal1_run._qa_expectations(enforced, TINY_OVERRIDE),
        manifest_dir=Path(result.output_root) / "manifests",
    )
    assert any(
        failure["error_type"] == "DuplicateTrivialityAdmissionIndex"
        for failure in tampered_index_report.failures
    )
    errors_path.write_text(original_errors, encoding="utf-8")


def test_deduplication_identity_includes_domain_mode() -> None:
    config = load_goal1_config(CONFIG_PATH).generator_config
    record = generate_expression(
        config,
        expression_index=0,
        family_id="algebraic_core",
        split="train",
    )
    alternate_domain = "safe_real" if record.domain_mode != "safe_real" else "positive_real"
    other_domain = record.model_copy(update={"domain_mode": alternate_domain})
    assert record.sympy_srepr == other_domain.sympy_srepr
    assert _structural_identity(record) != _structural_identity(other_domain)


def test_qa_rejects_duplicate_audit_identity_not_retained_by_pipeline(
    tmp_path: Path,
) -> None:
    loaded = load_goal1_config(CONFIG_PATH)
    result = run_corpus_stage(
        loaded,
        stage=Goal1Stage.DEVELOPMENT,
        run_label="orphaned-duplicate",
        output_root=tmp_path / "isolated-artifacts",
        override=TINY_OVERRIDE,
    )
    run_root = Path(result.output_root)
    orphan_srepr = "Integer(2)"
    orphan_id = derive_expression_id(
        domain_mode="safe_real",
        sympy_srepr=orphan_srepr,
    )
    duplicate_row = {
        "domain_mode": "safe_real",
        "duplicate_expression_id": orphan_id,
        "kept_expression_id": orphan_id,
        "sympy_srepr": orphan_srepr,
    }
    with (run_root / "duplicates.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(duplicate_row, sort_keys=True) + "\n")

    manifest = load_corpus_manifest(result.manifest_path)
    row_accounting = dict(manifest.metadata["row_accounting"])
    row_accounting.update(
        {
            "attempted": 26,
            "generated": 26,
            "duplicates": 1,
            "acceptance_rate": 25 / 26,
        }
    )
    tampered = manifest.model_copy(
        update={
            "metadata": {
                **manifest.metadata,
                "row_accounting": row_accounting,
                "deduplication": {
                    "processed_count": 26,
                    "unique_count": 25,
                    "duplicate_count": 1,
                    "identity_conflict_count": 0,
                },
            }
        }
    )
    tampered_path = run_root / "tampered.manifest.json"
    tampered_path.write_text(tampered.model_dump_json(indent=2) + "\n", encoding="utf-8")

    report = goal1_run.run_corpus_qa(
        tampered_path,
        run_root,
        config_path=CONFIG_PATH,
        generator_config=loaded.generator_config,
        expectations=goal1_run._qa_expectations(loaded, TINY_OVERRIDE),
    )
    assert not report.passed
    assert any(failure["error_type"] == "OrphanedDuplicateDecision" for failure in report.failures)


def test_qa_rejects_checksum_valid_generator_provenance_tampering(
    tmp_path: Path,
) -> None:
    loaded = load_goal1_config(CONFIG_PATH)
    result = run_corpus_stage(
        loaded,
        stage=Goal1Stage.DEVELOPMENT,
        run_label="generator-provenance-tamper",
        output_root=tmp_path / "isolated-artifacts",
        override=TINY_OVERRIDE,
    )
    run_root = Path(result.output_root)
    manifest = load_corpus_manifest(result.manifest_path)
    original_split = manifest.splits[0]
    original_shard = original_split.shards[0]
    shard_path = run_root / original_shard.path

    table = pq.read_table(shard_path)
    rows = table.to_pylist()
    first_metadata = json.loads(rows[0]["generator_metadata_json"])
    second_metadata = json.loads(rows[1]["generator_metadata_json"])
    first_metadata["expression_index"] = second_metadata["expression_index"]
    rows[0]["generator_metadata_json"] = json.dumps(
        first_metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=table.schema),
        shard_path,
        compression="zstd",
        data_page_version="2.0",
        use_dictionary=True,
        write_statistics=True,
    )

    tampered_shard = original_shard.model_copy(
        update={
            "byte_count": shard_path.stat().st_size,
            "checksum": original_shard.checksum.model_copy(
                update={"digest": goal1_run.sha256_file(shard_path)}
            ),
        }
    )
    tampered_split = original_split.model_copy(
        update={"shards": (tampered_shard, *original_split.shards[1:])}
    )
    tampered_manifest = manifest.model_copy(
        update={"splits": (tampered_split, *manifest.splits[1:])}
    )
    tampered_path = run_root / "generator-provenance-tampered.manifest.json"
    tampered_path.write_text(
        tampered_manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    report = goal1_run.run_corpus_qa(
        tampered_path,
        run_root,
        config_path=CONFIG_PATH,
        generator_config=loaded.generator_config,
        expectations=goal1_run._qa_expectations(loaded, TINY_OVERRIDE),
    )
    failure_types = {failure["error_type"] for failure in report.failures}
    assert not report.passed
    assert "GeneratorProvenanceMismatch" in failure_types
    assert "DuplicateExpressionIndex" in failure_types


def test_successful_record_can_retain_internal_triviality_retries() -> None:
    config = load_goal1_config(CONFIG_PATH).generator_config
    strict_triviality = config.triviality.model_copy(
        update={
            "per_expression_caps": {
                **dict(config.triviality.per_expression_caps),
                "constant_only_subtrees": 0,
            }
        }
    )
    strict_config = config.model_copy(update={"triviality": strict_triviality})
    record = generate_expression(
        strict_config,
        expression_index=0,
        family_id="powers_division_rationals",
        split="train",
    )
    assert (
        record.generator_metadata["rejection_reasons"]["triviality_cap:constant_only_subtrees"] > 0
    )
    assert record.generator_metadata["triviality_counts"]["constant_only_subtrees"] == 0
