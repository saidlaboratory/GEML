"""Issue 5-7 tests use only tiny hand-written and temporary fixtures."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.data.egraph_candidate_dataset import (
    Goal4RunContract,
    Goal4Unit,
    iter_replayed_candidate_groups,
    load_goal4_units,
    replay_candidate_group,
    summarize_candidate_groups,
)
from geml.data.storage.manifests import (
    build_corpus_manifest,
    build_split_manifest,
    write_manifest_bundle,
)
from geml.data.storage.shards import write_shards
from geml.egraph.ir import add, const, var
from geml.egraph.policy import RewriteMode
from geml.egraph.validation import ValidationStatus, expr_to_ast_tree
from geml.eml.compiler_core import CompilerMode
from geml.experiments.goal4.run import (
    Goal4Config,
    item_from_record,
    load_goal4_config,
    process_expression,
    run_stage,
)
from geml.experiments.goal5.neural_ranker import (
    NeuralRankerConfigurationError,
    load_neural_ranker_config,
    run_neural_ranker,
)
from geml.interfaces.eml_dag_cost import EMLDagCostStatus, compute_eml_dag_cost
from geml.learning.egraph_ranker import (
    FEATURE_NAMES,
    CandidateGroup,
    EGraphRanker,
    RankedCandidate,
    RankingMethod,
    candidate_feature_vector,
    candidate_group_id,
    evaluate_candidate_groups,
    fit_egraph_ranker,
    heuristics_outperform_neural,
)

_SOURCE_SHA = "a" * 64


def _record(
    source: str = "Add(Symbol('x', real=True), Integer(0))",
    *,
    split: CorpusSplit = CorpusSplit.TRAIN,
    family: str = "algebraic_core",
    salt: str = "0",
) -> ExpressionRecord:
    expression_id = hashlib.sha256(f"issue-5-7-fixture\0{salt}\0{source}".encode()).hexdigest()
    return ExpressionRecord(
        expression_id=expression_id,
        sympy_srepr=source,
        display_text=source,
        latex_text=None,
        split=split,
        operator_family=family,
        domain_mode="safe_real",
        variables=("x",),
        target_ast_size=3,
        target_depth=1,
        generator_seed=7,
        generator_metadata={
            "achieved_source_ast_size": 3,
            "difficulty_profile": "ordinary",
        },
    )


def _goal4_row(record: ExpressionRecord, mode: RewriteMode) -> dict[str, object]:
    config = load_goal4_config("configs/goal4_final.yaml")
    row = process_expression(
        item_from_record(record, config.sampling),
        mode,
        config,
    )
    row["source_manifest_sha256"] = _SOURCE_SHA
    return row


def _features(value: float) -> tuple[float, ...]:
    return (value,) + (0.0,) * (len(FEATURE_NAMES) - 1)


def _candidate(
    index: int,
    signature: str,
    cost: int | None,
    *,
    valid: bool = True,
    estimated: int | None = None,
    ast: int | None = None,
    feature_value: float | None = None,
) -> RankedCandidate:
    return RankedCandidate(
        candidate_index=index,
        signature=signature,
        features=_features(float(index) if feature_value is None else feature_value),
        official_eml_dag_cost=cost,
        estimated_eml_tree_cost=estimated if estimated is not None else cost,
        ast_dag_cost=ast if ast is not None else cost,
        ast_tree_cost=ast if ast is not None else cost,
        validation_status=(
            ValidationStatus.VALID.value if valid else ValidationStatus.SEMANTIC_MISMATCH.value
        ),
        validation_reason=(
            "validated against source" if valid else "independent semantic validation failed"
        ),
        official_cost_status=EMLDagCostStatus.SUCCESS.value,
        official_cost_reason="official OFFICIAL_V4 fixture cost",
        official_cost_scoring_seconds=0.01 * (index + 1),
    )


def _group(
    expression_id: str,
    split: CorpusSplit,
    candidates: tuple[RankedCandidate, ...],
    *,
    mode: str = RewriteMode.SAFE_REAL.value,
) -> CandidateGroup:
    return CandidateGroup(
        group_id=candidate_group_id(expression_id, mode),
        expression_id=expression_id,
        rewrite_mode=mode,
        split=split,
        candidates=candidates,
        source_stage_status="optimized",
        source_candidate_count=len(candidates),
        replay_status="matched",
        replay_reason="tiny fixture matches",
    )


def _constant_model() -> EGraphRanker:
    hidden_units = 2
    return EGraphRanker(
        feature_mean=(0.0,) * len(FEATURE_NAMES),
        feature_scale=(1.0,) * len(FEATURE_NAMES),
        hidden_weights=tuple((0.0,) * hidden_units for _ in FEATURE_NAMES),
        hidden_bias=(0.0,) * hidden_units,
        output_weights=(0.0,) * (hidden_units + 1),
        target_mean=0.0,
        target_scale=1.0,
        ridge=1.0,
        seed=7,
    )


def test_structural_features_do_not_contain_cost() -> None:
    expression = add(var("x"), const(Fraction(-1, 2)))
    features = candidate_feature_vector(expression)
    assert len(features) == len(FEATURE_NAMES)
    assert features[FEATURE_NAMES.index("depth")] == 1.0
    assert features[FEATURE_NAMES.index("rational_constant_count")] == 1.0
    assert features[FEATURE_NAMES.index("negative_constant_count")] == 1.0
    assert all("cost" not in name for name in FEATURE_NAMES)


def test_official_label_matches_frozen_pure_eml_dag_boundary() -> None:
    expression = add(var("x"), const(0))
    result = compute_eml_dag_cost(
        expr_to_ast_tree(expression, expression_id="label-fixture"),
        compiler_mode=CompilerMode.OFFICIAL_V4,
    )
    assert result.status is EMLDagCostStatus.SUCCESS
    candidate = _candidate(0, "source", result.eml_dag_node_count)
    assert candidate.official_eml_dag_cost == result.eml_dag_node_count
    assert candidate.rankable


def test_candidate_group_round_trip_preserves_failures() -> None:
    group = _group(
        "a" * 64,
        CorpusSplit.TRAIN,
        (
            _candidate(0, "a-invalid", 2, valid=False),
            _candidate(1, "z-valid", 3),
        ),
    )
    loaded = CandidateGroup.from_dict(group.as_dict())
    assert loaded == group
    assert loaded.retained_failure_count == 1
    assert loaded.exact_best.signature == "z-valid"


def test_goal4_replay_matches_summary_and_retains_every_candidate() -> None:
    record = _record()
    unit = Goal4Unit.from_row(_goal4_row(record, RewriteMode.SAFE_REAL))
    group = replay_candidate_group(
        record,
        unit,
        include_optional_domain_rules=False,
    )
    assert group.replay_status == "matched"
    assert len(group.candidates) == unit.source_candidate_count
    assert all(candidate.official_eml_dag_cost is not None for candidate in group.candidates)
    assert any(candidate.rankable for candidate in group.candidates)


def test_goal4_unsupported_source_is_retained_as_empty_matched_group() -> None:
    record = _record(
        "sin(Symbol('x', real=True))",
        family="trigonometric",
    )
    unit = Goal4Unit.from_row(_goal4_row(record, RewriteMode.SAFE_REAL))
    group = replay_candidate_group(
        record,
        unit,
        include_optional_domain_rules=False,
    )
    assert group.source_stage_status == "unsupported_operator"
    assert group.replay_status == "matched"
    assert not group.candidates
    summary = summarize_candidate_groups((group,))
    assert summary.empty_group_count == 1
    assert summary.group_count == 1


def test_goal4_rows_require_complete_grouped_modes(tmp_path: Path) -> None:
    record = _record()
    rows = [
        _goal4_row(record, RewriteMode.SAFE_REAL),
        _goal4_row(record, RewriteMode.POSITIVE_REAL_FORMAL),
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    first = rows[0]
    run = Goal4RunContract(
        run_id=first["run_id"],
        config_sha256=first["config_sha256"],
        source_manifest_sha256=_SOURCE_SHA,
        implementation_commit="fixture",
        selected_expression_count=1,
        modes=(RewriteMode.SAFE_REAL, RewriteMode.POSITIVE_REAL_FORMAL),
        include_optional_domain_rules=False,
    )
    units = load_goal4_units(path, run)
    assert [(unit.expression_id, unit.rewrite_mode) for unit in units] == [
        (record.expression_id, RewriteMode.POSITIVE_REAL_FORMAL),
        (record.expression_id, RewriteMode.SAFE_REAL),
    ]


def test_bounded_worker_replay_preserves_canonical_order() -> None:
    record = _record()
    units = tuple(
        Goal4Unit.from_row(_goal4_row(record, mode))
        for mode in (RewriteMode.POSITIVE_REAL_FORMAL, RewriteMode.SAFE_REAL)
    )
    groups = tuple(
        iter_replayed_candidate_groups(
            units,
            {record.expression_id: record},
            include_optional_domain_rules=False,
            worker_processes=2,
            chunksize=1,
        )
    )
    assert [group.rewrite_mode for group in groups] == [
        RewriteMode.POSITIVE_REAL_FORMAL.value,
        RewriteMode.SAFE_REAL.value,
    ]
    assert all(group.replay_status == "matched" for group in groups)


def test_grouped_train_validation_fit_is_deterministic() -> None:
    training = (
        _group(
            "1" * 64,
            CorpusSplit.TRAIN,
            (_candidate(0, "a", 4), _candidate(1, "b", 2)),
        ),
        _group(
            "2" * 64,
            CorpusSplit.TRAIN,
            (_candidate(0, "a", 8), _candidate(1, "b", 3)),
        ),
    )
    validation = (
        _group(
            "3" * 64,
            CorpusSplit.VALIDATION,
            (_candidate(0, "a", 5), _candidate(1, "b", 2)),
        ),
    )
    arguments = {
        "seed": 17,
        "hidden_units": 4,
        "ridge_values": (0.01, 0.1, 1.0),
    }
    first = fit_egraph_ranker(training, validation, **arguments)
    second = fit_egraph_ranker(training, validation, **arguments)
    assert first.model.as_dict() == second.model.as_dict()
    assert first.as_dict() == second.as_dict()
    assert first.training_group_count == 2
    assert first.validation_group_count == 1


def test_expression_identity_cannot_leak_across_grouped_splits() -> None:
    identity = "4" * 64
    training = (
        _group(identity, CorpusSplit.TRAIN, (_candidate(0, "a", 3), _candidate(1, "b", 2))),
    )
    validation = (
        _group(
            identity,
            CorpusSplit.VALIDATION,
            (_candidate(0, "a", 3), _candidate(1, "b", 2)),
        ),
    )
    with pytest.raises(ValueError, match="leakage"):
        fit_egraph_ranker(
            training,
            validation,
            seed=1,
            hidden_units=2,
            ridge_values=(1.0,),
        )


def test_failed_neural_selection_is_retained_in_metrics_and_outcomes() -> None:
    group = _group(
        "5" * 64,
        CorpusSplit.TEST_IID,
        (
            _candidate(0, "a-invalid", 1, valid=False),
            _candidate(1, "z-valid", 3, valid=True),
        ),
    )
    evaluation = evaluate_candidate_groups(
        (group,),
        split=CorpusSplit.TEST_IID,
        model=_constant_model(),
        random_seed=2,
    )
    neural = evaluation.metrics_for(RankingMethod.NEURAL)
    assert neural.attempted_group_count == 1
    assert neural.failed_selected_count == 1
    assert neural.validation_rate == 0.0
    outcome = next(item for item in evaluation.outcomes if item.method is RankingMethod.NEURAL)
    assert outcome.selected_signature == "a-invalid"
    assert outcome.failure_reason is not None


def test_baselines_report_exact_match_regret_and_cost_scoring_speedup() -> None:
    group = _group(
        "6" * 64,
        CorpusSplit.TEST_IID,
        (
            _candidate(0, "a", 6, estimated=2, ast=7),
            _candidate(1, "b", 3, estimated=4, ast=1),
            _candidate(2, "c", 5, estimated=3, ast=4),
        ),
    )
    evaluation = evaluate_candidate_groups(
        (group,),
        split=CorpusSplit.TEST_IID,
        model=_constant_model(),
        random_seed=3,
    )
    exact = evaluation.metrics_for(RankingMethod.EXACT)
    estimated = evaluation.metrics_for(RankingMethod.ESTIMATED_EML)
    ast = evaluation.metrics_for(RankingMethod.AST)
    assert exact.exact_best_match_rate == 1.0
    assert exact.mean_regret == 0.0
    assert exact.official_cost_scoring_calls == 3
    assert estimated.mean_regret == 3.0
    assert estimated.call_count_speedup_vs_exact == 3.0
    assert ast.exact_best_match_rate == 1.0


def test_heuristic_outperformance_is_stated_from_complete_metrics() -> None:
    group = _group(
        "7" * 64,
        CorpusSplit.TEST_IID,
        (
            _candidate(0, "a-neural", 9, estimated=9, ast=9),
            _candidate(1, "z-best", 2, estimated=1, ast=1),
        ),
    )
    evaluation = evaluate_candidate_groups(
        (group,),
        split=CorpusSplit.TEST_IID,
        model=_constant_model(),
        random_seed=4,
    )
    outperformers = heuristics_outperform_neural(evaluation)
    assert RankingMethod.ESTIMATED_EML in outperformers
    assert RankingMethod.AST in outperformers


def test_config_is_strict_and_loads_without_production_outputs(tmp_path: Path) -> None:
    loaded = load_neural_ranker_config(
        "configs/goal5_neural_ranker.yaml",
        require_inputs=False,
    )
    assert loaded.config.dataset.retain_failed_candidates is True
    assert loaded.config.evaluation.speedup_scope == "official_eml_dag_cost_scoring_only"

    raw = yaml.safe_load(Path("configs/goal5_neural_ranker.yaml").read_text(encoding="utf-8"))
    raw["model"]["unexpected"] = True
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(NeuralRankerConfigurationError, match="unexpected"):
        load_neural_ranker_config(bad, require_inputs=False)


def test_ranker_artifact_round_trip() -> None:
    model = _constant_model()
    loaded = EGraphRanker.from_dict(model.as_dict())
    assert loaded == model
    assert loaded.predict(_features(3.0)) == pytest.approx(0.0)


def test_end_to_end_runner_uses_only_temporary_goal4_and_corpus_artifacts(
    tmp_path: Path,
) -> None:
    records = tuple(
        _record(
            split=split,
            salt=str(index),
        )
        for index, split in enumerate(CorpusSplit)
    )
    corpus_root = tmp_path / "corpus"
    source_config = tmp_path / "source-config.json"
    source_config.write_text("{}\n", encoding="utf-8")
    split_manifests = []
    for split in CorpusSplit:
        shards = write_shards(
            [record for record in records if record.split is split],
            corpus_root / "shards",
            corpus_id="issue-5-7-fixture",
            split=split,
            schema_version="geml-corpus-v1",
            minimum_rows=1,
            maximum_rows=1,
            allow_small_fixture=True,
            manifest_root=corpus_root,
        )
        split_manifests.append(build_split_manifest(shards))
    corpus = build_corpus_manifest(
        split_manifests,
        corpus_id="issue-5-7-fixture",
        schema_version="geml-corpus-v1",
        config_path=source_config,
        generator_seed=7,
        git_commit="fixture",
        created_at=datetime(2026, 7, 24, tzinfo=UTC),
    )
    corpus_path = write_manifest_bundle(
        corpus,
        corpus_root / "manifests",
        artifact_root=corpus_root,
        config_path=source_config,
    ).corpus_manifest
    corpus_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    goal4_config = Goal4Config.model_validate(
        {
            "schema_version": "geml-goal4-config-v2",
            "output_root": str(tmp_path / "goal4"),
            "include_optional_domain_rules": False,
            "modes": ["safe_real", "positive_real_formal"],
            "sampling": {
                "seed": 7,
                "target_size": 4,
                "balance_axes": ["split"],
                "size_bucket_edges": [4],
            },
            "resources": {
                "max_iterations": 10,
                "max_egraph_nodes": 100,
                "max_rewrite_attempts": 100,
                "saturation_timeout_seconds": 2.0,
                "max_eclasses": None,
                "extraction_max_depth": 4,
                "extraction_beam_width": 3,
                "extraction_max_candidates": 6,
                "extraction_max_nodes": 1000,
                "extraction_max_iterations": 5000,
                "extraction_timeout_seconds": 2.0,
            },
            "processing": {
                "chunk_size": 4,
                "checkpoint_every_chunks": 1,
                "worker_processes": 1,
                "resume": True,
            },
            "stages": {"final": {"expected_count": 4, "row_limit": None}},
        }
    )
    goal4_result = run_stage(
        goal4_config,
        "final",
        records,
        tmp_path / "goal4",
        source_identity={
            "kind": "corpus_manifest",
            "manifest_sha256": corpus_sha,
            "record_count": 4,
        },
        implementation_commit="fixture",
    )
    rows_sha = hashlib.sha256(goal4_result.rows_path.read_bytes()).hexdigest()
    run_sha = hashlib.sha256(goal4_result.run_manifest_path.read_bytes()).hexdigest()
    run_payload = json.loads(goal4_result.run_manifest_path.read_text(encoding="utf-8"))

    neural_config = {
        "schema_version": "geml-goal5-neural-ranker-config-v1",
        "goal4": {
            "rows_path": str(goal4_result.rows_path),
            "rows_sha256": rows_sha,
            "run_manifest_path": str(goal4_result.run_manifest_path),
            "run_manifest_sha256": run_sha,
            "run_id": goal4_result.run_id,
            "corpus_manifest_path": str(corpus_path),
            "corpus_manifest_sha256": corpus_sha,
        },
        "dataset": {
            "method": "fixed_cycle_safe_goal4_replay_v1",
            "exact_group_key": "expression_id_and_rewrite_mode",
            "label": "official_v4_pure_eml_dag_node_count",
            "retain_failed_candidates": True,
            "require_summary_match": True,
            "worker_processes": 1,
            "worker_chunksize": 1,
            "checkpoint_every_groups": 1,
            "log_every_groups": 8,
        },
        "model": {
            "method": "geml-egraph-random-feature-ranker-v1",
            "seed": 7,
            "hidden_units": 4,
            "ridge_values": [0.1, 1.0],
            "target_transform": "log1p",
            "group_equal_weighting": True,
            "validation_selects_ridge": True,
        },
        "evaluation": {
            "methods": [method.value for method in RankingMethod],
            "random_seed": 7,
            "primary_test_split": "test_iid",
            "report_test_ood_separately": True,
            "speedup_scope": "official_eml_dag_cost_scoring_only",
            "retain_failed_selections": True,
            "heuristic_comparison_order": (
                "validation_rate_then_mean_regret_then_exact_best_match"
            ),
        },
        "runtime": {
            "output_root": str(tmp_path / "neural"),
            "resume": True,
            "atomic_finalization": True,
        },
    }
    assert run_payload["source_manifest_sha256"] == corpus_sha
    config_path = tmp_path / "neural.yaml"
    config_path.write_text(yaml.safe_dump(neural_config, sort_keys=False), encoding="utf-8")

    first = run_neural_ranker(config_path)
    second = run_neural_ranker(config_path)
    assert first.completion_path == second.completion_path
    assert first.report["dataset"]["group_count"] == 8
    assert first.report["dataset"]["replay_mismatch_count"] == 0
    assert first.report["runtime"]["candidate_cost_scoring_observation_count"] > 0
    assert first.report["test_iid"]["evaluable_group_count"] == 2
