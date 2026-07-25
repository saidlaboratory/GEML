from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

import geml.compression.motif.mdl as mdl_module
import geml.experiments.goal5.motif_sweeps as sweep_module
from geml.compression.motif.compress import (
    MotifCompressionFailureStage,
    MotifCompressionResult,
    MotifCompressionStatus,
)
from geml.compression.motif.mdl import (
    graph_mdl_cost,
    motif_graph_mdl_result,
    summarize_split_mdl,
    universal_integer_bits,
    vocabulary_mdl_bits,
)
from geml.compression.motif.vocabulary import (
    MotifChildRef,
    MotifNode,
    MotifPool,
    MotifTargetKind,
    build_motif_template,
    build_motif_vocabulary,
)
from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.data.storage.manifests import (
    build_corpus_manifest,
    build_split_manifest,
    write_manifest_bundle,
)
from geml.data.storage.shards import write_shards
from geml.experiments.goal5.motif_sweeps import (
    StageConfig,
    SweepCandidateResult,
    build_sweep_vocabularies,
    evaluate_graph_sweep,
    select_validation_winner,
    vocabulary_from_payload,
    vocabulary_payload,
)
from geml.graph.schema import ChildRef, Graph, GraphNode, GraphRoot


def _graph(
    *,
    renamed: bool = False,
    repeated: bool = True,
    reverse_child_storage: bool = False,
) -> Graph:
    prefix = "renamed-" if renamed else ""
    leaf_id = f"{prefix}leaf"
    other_id = f"{prefix}other"
    root_id = f"{prefix}root"
    children = (
        ChildRef(slot=0, target_id=leaf_id),
        ChildRef(slot=1, target_id=leaf_id if repeated else other_id),
    )
    nodes = {
        leaf_id: GraphNode(
            node_id=leaf_id,
            family="macro",
            kind="official_construction",
            label="symbol",
            value={"name": "x"},
        ),
        root_id: GraphNode(
            node_id=root_id,
            family="macro",
            kind="official_construction",
            label="add",
            value=None,
            children=tuple(reversed(children)) if reverse_child_storage else children,
        ),
    }
    if not repeated:
        nodes[other_id] = GraphNode(
            node_id=other_id,
            family="macro",
            kind="official_construction",
            label="symbol",
            value={"name": "x"},
        )
    return Graph(
        nodes=nodes,
        roots=(
            GraphRoot(
                root_id=f"{prefix}expression",
                target_id=root_id,
                representation_mode="macro:official_v4:is_pure_eml=false",
            ),
        ),
    )


def _template():
    return build_motif_template(
        source_family="macro",
        representation_mode="macro:official_v4:is_pure_eml=false",
        nodes=(
            MotifNode(
                kind="official_construction",
                label="add",
                value=None,
                children=(
                    MotifChildRef(
                        slot=0,
                        target_kind=MotifTargetKind.BOUNDARY,
                        target_index=0,
                    ),
                    MotifChildRef(
                        slot=1,
                        target_kind=MotifTargetKind.BOUNDARY,
                        target_index=0,
                    ),
                ),
            ),
        ),
        boundary_count=1,
        support_count=3,
        occurrence_count=4,
    )


def _two_node_template():
    return build_motif_template(
        source_family="macro",
        representation_mode="macro:official_v4:is_pure_eml=false",
        nodes=(
            MotifNode(
                kind="official_construction",
                label="add",
                value=None,
                children=(
                    MotifChildRef(
                        slot=0,
                        target_kind=MotifTargetKind.INTERNAL,
                        target_index=1,
                    ),
                    MotifChildRef(
                        slot=1,
                        target_kind=MotifTargetKind.INTERNAL,
                        target_index=1,
                    ),
                ),
            ),
            MotifNode(
                kind="official_construction",
                label="symbol",
                value={"name": "x"},
                children=(),
            ),
        ),
        boundary_count=0,
        support_count=3,
        occurrence_count=3,
    )


def _candidate_vocabulary():
    return build_motif_vocabulary(
        pool=MotifPool.MACRO,
        min_size=1,
        max_size=2,
        min_support_count=1,
        vocabulary_limit=None,
        training_transaction_count=3,
        processed_count=3,
        failure_count=0,
        training_fingerprint="c" * 64,
        templates=(_template(), _two_node_template()),
    )


def test_graph_mdl_is_identifier_invariant_and_counts_repeated_refs() -> None:
    original = graph_mdl_cost(_graph())
    renamed = graph_mdl_cost(_graph(renamed=True))
    slot_permuted = graph_mdl_cost(_graph(reverse_child_storage=True))
    duplicated_leaf = graph_mdl_cost(_graph(repeated=False))

    assert original == renamed
    assert original == slot_permuted
    assert original.child_reference_count == 2
    assert duplicated_leaf.node_count == original.node_count + 1
    assert duplicated_leaf.total_bits > original.total_bits


def test_dictionary_cost_is_charged_in_addition_to_graph_data() -> None:
    template = _template()
    empty = vocabulary_mdl_bits(())
    populated = vocabulary_mdl_bits((template,))

    assert populated > empty
    assert populated - empty >= template.dictionary_cost_bits


def test_compressed_data_cost_is_reconstructible_and_decomposed() -> None:
    template = _template()
    vocabulary = build_motif_vocabulary(
        pool=MotifPool.MACRO,
        min_size=1,
        max_size=1,
        min_support_count=1,
        vocabulary_limit=None,
        training_transaction_count=3,
        processed_count=3,
        failure_count=0,
        training_fingerprint="a" * 64,
        templates=(template,),
    )

    result = motif_graph_mdl_result(_graph(), vocabulary)
    summary = summarize_split_mdl((result,), vocabulary)

    assert result.success
    assert result.selected_occurrence_count == 1
    assert result.reconstruction_failure_count == 0
    assert result.conditional_data_bits == (
        result.framing_bits + result.residual_bits + result.occurrence_bits
    )
    assert summary.processed_count == 1
    assert summary.dictionary_bits >= template.dictionary_cost_bits
    assert summary.total_mdl_bits == summary.dictionary_bits + result.conditional_data_bits


def test_failed_compression_falls_back_without_dropping_the_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _template()
    vocabulary = build_motif_vocabulary(
        pool=MotifPool.MACRO,
        min_size=1,
        max_size=1,
        min_support_count=1,
        vocabulary_limit=None,
        training_transaction_count=3,
        processed_count=3,
        failure_count=0,
        training_fingerprint="a" * 64,
        templates=(template,),
    )

    def fail_compression(_graph, _vocabulary):
        return MotifCompressionResult(
            status=MotifCompressionStatus.FAILURE,
            compressed=None,
            candidate_occurrence_count=2,
            selected_occurrence_count=1,
            failure_stage=MotifCompressionFailureStage.MATCHING,
            error_type="InjectedFailure",
            error_message="retained",
        )

    monkeypatch.setattr(mdl_module, "compress_graph", fail_compression)
    result = motif_graph_mdl_result(_graph(), vocabulary)
    summary = summarize_split_mdl((result,), vocabulary)

    assert not result.success
    assert result.conditional_data_bits == result.baseline_bits + 1
    assert result.selected_occurrence_count == 0
    assert result.attempted_selected_occurrence_count == 1
    assert summary.processed_count == 1
    assert summary.reconstruction_failure_count == 1


def test_post_selection_coding_failure_retains_attempt_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _template()
    vocabulary = build_motif_vocabulary(
        pool=MotifPool.MACRO,
        min_size=1,
        max_size=1,
        min_support_count=1,
        vocabulary_limit=None,
        training_transaction_count=3,
        processed_count=3,
        failure_count=0,
        training_fingerprint="a" * 64,
        templates=(template,),
    )

    def fail_after_selection(*_args, **_kwargs):
        raise RuntimeError("injected post-selection coding failure")

    monkeypatch.setattr(mdl_module, "compressed_data_mdl_cost", fail_after_selection)
    result = motif_graph_mdl_result(_graph(), vocabulary)

    assert not result.success
    assert result.conditional_data_bits == result.baseline_bits + 1
    assert result.selected_occurrence_count == 0
    assert result.attempted_selected_occurrence_count == 1
    assert result.error_type == "RuntimeError"


def test_nonnegative_universal_integer_code_is_fixed() -> None:
    assert [universal_integer_bits(value) for value in range(8)] == [1, 4, 4, 5, 5, 5, 5, 8]


def _candidate(
    digest: str,
    *,
    validation_bits: int,
    vocabulary_size: int,
    maximum_size: int,
    failures: int = 0,
    complete: bool = True,
) -> SweepCandidateResult:
    full_digest = digest * 64
    return SweepCandidateResult(
        configuration_digest=full_digest,
        minimum_motif_size=2,
        maximum_motif_size=maximum_size,
        requested_vocabulary_size=vocabulary_size,
        actual_vocabulary_size=vocabulary_size,
        train_total_mdl_bits=1,
        validation_total_mdl_bits=validation_bits,
        reconstruction_failure_count=failures,
        mining_complete=complete,
        vocabulary_digest="f" * 64,
        metadata={},
    )


def test_validation_winner_excludes_failures_and_uses_frozen_ties() -> None:
    candidates = (
        _candidate("c", validation_bits=90, vocabulary_size=64, maximum_size=4, failures=1),
        _candidate("b", validation_bits=100, vocabulary_size=64, maximum_size=6),
        _candidate("a", validation_bits=100, vocabulary_size=64, maximum_size=4),
        _candidate("d", validation_bits=80, vocabulary_size=64, maximum_size=4, complete=False),
    )

    winner = select_validation_winner(candidates)

    assert winner.configuration_digest == "a" * 64


def test_tiny_sweep_runs_all_nested_vocabularies_and_round_trips_artifacts() -> None:
    candidates = _candidate_vocabulary()
    stage = StageConfig(
        train_limit=2,
        validation_limit=1,
        test_iid_limit=1,
        test_ood_limit=1,
        minimum_support_count=1,
        vocabulary_sizes=(1, 2),
        size_ranges=((2, 2),),
    )
    sweeps = build_sweep_vocabularies(
        candidates,
        stage,
        scientific_protocol_digest="d" * 64,
    )

    evaluations = evaluate_graph_sweep(
        (("first", _graph()), ("second", _graph(renamed=True))),
        sweeps,
    )

    assert len(sweeps) == 3  # two headline budgets plus one size-one ablation
    assert sum(sweep.is_one_node_ablation for sweep in sweeps) == 1
    assert all(evaluation.summary.processed_count == 2 for evaluation in evaluations.values())
    assert all(
        evaluation.summary.reconstruction_failure_count == 0 for evaluation in evaluations.values()
    )
    assert vocabulary_from_payload(vocabulary_payload(candidates)) == candidates


def test_persisted_vocabulary_decoder_rejects_noncanonical_types_and_order() -> None:
    canonical = vocabulary_payload(_candidate_vocabulary())
    malformed: list[dict[str, object]] = []

    support_as_text = json.loads(json.dumps(canonical))
    support_as_text["templates"][0]["support_count"] = "3"
    malformed.append(support_as_text)

    dictionary_cost_as_float = json.loads(json.dumps(canonical))
    dictionary_cost_as_float["templates"][0]["dictionary_cost_bits"] = float(
        dictionary_cost_as_float["templates"][0]["dictionary_cost_bits"]
    )
    malformed.append(dictionary_cost_as_float)

    slot_as_text = json.loads(json.dumps(canonical))
    slot_as_text["templates"][0]["nodes"][0]["children"][0]["slot"] = "0"
    malformed.append(slot_as_text)

    kind_as_number = json.loads(json.dumps(canonical))
    kind_as_number["templates"][0]["nodes"][0]["kind"] = 7
    malformed.append(kind_as_number)

    extra_field = json.loads(json.dumps(canonical))
    extra_field["unexpected"] = True
    malformed.append(extra_field)

    reordered = json.loads(json.dumps(canonical))
    reordered["templates"].reverse()
    malformed.append(reordered)

    for payload in malformed:
        with pytest.raises((TypeError, ValueError)):
            vocabulary_from_payload(payload)


def test_persisted_graph_decoder_rejects_coercion_invalidity_and_noncanonical_order() -> None:
    canonical = sweep_module.graph_payload(_graph())
    malformed: list[dict[str, object]] = []

    slot_as_text = json.loads(json.dumps(canonical))
    slot_as_text["nodes"][1]["children"][0]["slot"] = "0"
    malformed.append(slot_as_text)

    family_as_number = json.loads(json.dumps(canonical))
    family_as_number["nodes"][0]["family"] = 5
    malformed.append(family_as_number)

    missing_target = json.loads(json.dumps(canonical))
    missing_target["nodes"][1]["children"][0]["target_id"] = "missing"
    malformed.append(missing_target)

    extra_field = json.loads(json.dumps(canonical))
    extra_field["nodes"][0]["unexpected"] = None
    malformed.append(extra_field)

    reordered = json.loads(json.dumps(canonical))
    reordered["nodes"].reverse()
    malformed.append(reordered)

    for payload in malformed:
        with pytest.raises((TypeError, ValueError)):
            sweep_module.graph_from_payload(payload)


def test_goal5_json_loader_requires_exact_canonical_bytes(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"b": 2, "a": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical JSON"):
        sweep_module._load_json_mapping(path)

    canonical = {"a": 1, "b": 2}
    path.write_bytes(sweep_module._canonical_json(canonical) + b"\n")
    assert sweep_module._load_json_mapping(path) == canonical


def _expression_record(index: int, split: CorpusSplit) -> ExpressionRecord:
    return ExpressionRecord(
        expression_id=f"{index + 1:064x}",
        sympy_srepr="Add(Symbol('x', real=True), Symbol('y', real=True))",
        display_text="x + y",
        latex_text=None,
        split=split,
        operator_family="algebraic_core",
        domain_mode="safe_real",
        variables=("x", "y"),
        target_ast_size=3,
        target_depth=1,
        generator_seed=index,
        generator_metadata={"fixture": True},
    )


def _tiny_corpus(root: Path) -> Path:
    run_root = root / "input"
    split_manifests = []
    cursor = 0
    for split in CorpusSplit:
        records = tuple(_expression_record(cursor + offset, split) for offset in range(2))
        cursor += len(records)
        shards = write_shards(
            records,
            run_root / "data" / split.value,
            corpus_id="goal5-fixture",
            split=split,
            schema_version="geml-expression-record-v1",
            minimum_rows=1,
            maximum_rows=2,
            allow_small_fixture=True,
            manifest_root=run_root,
        )
        split_manifests.append(build_split_manifest(shards))
    source_config = root / "source.yaml"
    source_config.write_text("schema_version: fixture-v1\n", encoding="utf-8")
    manifest = build_corpus_manifest(
        split_manifests,
        corpus_id="goal5-fixture",
        schema_version="geml-expression-record-v1",
        config_path=source_config,
        generator_seed=1,
        git_commit="fixture",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        package_names=("geml",),
    )
    return write_manifest_bundle(
        manifest,
        run_root / "manifests",
        artifact_root=run_root,
    ).corpus_manifest


def _tiny_runner_config(root: Path, manifest_path: Path) -> Path:
    stage = {
        "train_limit": 2,
        "validation_limit": 2,
        "test_iid_limit": 2,
        "test_ood_limit": 2,
        "minimum_support_count": 1,
        "vocabulary_sizes": [1],
        "size_ranges": [[2, 2]],
    }
    final = {
        "train_limit": None,
        "validation_limit": None,
        "test_iid_limit": None,
        "test_ood_limit": None,
        "minimum_support_count": 1,
        "vocabulary_sizes": [64, 256, 512, 1024],
        "size_ranges": [[2, 4], [2, 6], [2, 8]],
    }
    config = {
        "schema_version": "geml-goal5-motif-sweeps-config-v1",
        "input_manifest": manifest_path.relative_to(root).as_posix(),
        "output_root": "output",
        "compiler_mode": "official_v4",
        "graph_family": "macro",
        "representation_mode": "macro:official_v4:is_pure_eml=false",
        "stages": {"smoke": stage, "final": final},
        "mining": {
            "support_unit": "graph_transaction",
            "boundary_order": "canonical_first_encounter",
            "maximum_motif_size": 8,
            "retain_one_node_ablation": True,
            "allow_silent_truncation": False,
        },
        "compression": {
            "occurrence_policy": "deterministic_safe_greedy_v1",
            "codec": "geml-motif-mdl-v1",
            "require_exact_reconstruction": True,
            "failure_fallback": "encode_original_graph",
        },
        "selection": {
            "metric": "validation_total_mdl_bits",
            "tie_break": [
                "actual_vocabulary_size",
                "maximum_motif_size",
                "configuration_digest",
            ],
            "evaluate_tests_once_after_lock": True,
        },
        "runtime": {
            "shard_rows": 2,
            "resume": True,
            "atomic_finalization": True,
        },
    }
    path = root / "configs" / "goal5.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _prepare_tiny_runner(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (root / "src" / "geml").mkdir(parents=True)
    config_path = _tiny_runner_config(root, _tiny_corpus(root))
    monkeypatch.setattr(
        sweep_module,
        "sweep_implementation_digest",
        lambda _root: "a" * 64,
    )
    return config_path


def test_tiny_runner_locks_before_heldout_and_resumes_without_re_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _prepare_tiny_runner(tmp_path, monkeypatch)

    result = sweep_module.run_motif_sweeps(config_path, stage_name="smoke")
    lock = json.loads(result.selection_lock_path.read_text(encoding="utf-8"))
    heldout = json.loads(result.heldout_results_path.read_text(encoding="utf-8"))

    assert lock["heldout_artifacts_absent_at_lock"] is True
    assert lock["selected_configuration"]["actual_vocabulary_size"] == 1
    assert set(heldout["splits"]) == {"test_iid", "test_ood"}
    assert all(
        split["summary"]["reconstruction_failure_count"] == 0
        for split in heldout["splits"].values()
    )

    def unexpected_rebuild(*_args, **_kwargs):
        raise AssertionError("a completed run must not rebuild or re-evaluate graph caches")

    monkeypatch.setattr(sweep_module, "build_graph_cache", unexpected_rebuild)
    resumed = sweep_module.run_motif_sweeps(config_path, stage_name="smoke")
    assert resumed == result


def test_resume_reuses_mining_and_split_sweep_receipts_after_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _prepare_tiny_runner(tmp_path, monkeypatch)
    immutable_json = sweep_module._immutable_json

    def interrupt_before_sweep_table(payload, path):
        if path.name == "sweep_table.json":
            raise RuntimeError("simulated interruption")
        return immutable_json(payload, path)

    monkeypatch.setattr(sweep_module, "_immutable_json", interrupt_before_sweep_table)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        sweep_module.run_motif_sweeps(config_path, stage_name="smoke")

    run_dirs = tuple((tmp_path / "output" / "smoke").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "mining.json").is_file()
    assert (run_dir / "sweep.train.json").is_file()
    assert (run_dir / "sweep.validation.json").is_file()
    assert not (run_dir / "sweep_table.json").exists()

    def unexpected_mining(*_args, **_kwargs):
        raise AssertionError("resume must reuse the immutable mining checkpoint")

    evaluated_sweep_counts: list[int] = []
    evaluate_graph_sweep = sweep_module.evaluate_graph_sweep

    def evaluate_heldout_only(graphs, sweeps):
        evaluated_sweep_counts.append(len(sweeps))
        if len(sweeps) != 1:
            raise AssertionError("resume must reuse train and validation sweep receipts")
        return evaluate_graph_sweep(graphs, sweeps)

    monkeypatch.setattr(sweep_module, "_immutable_json", immutable_json)
    monkeypatch.setattr(sweep_module, "mine_motifs", unexpected_mining)
    monkeypatch.setattr(sweep_module, "evaluate_graph_sweep", evaluate_heldout_only)

    result = sweep_module.run_motif_sweeps(config_path, stage_name="smoke")

    assert result.completion_path.is_file()
    assert evaluated_sweep_counts == [1, 1]


def test_completed_run_rejects_corrupt_split_sweep_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _prepare_tiny_runner(tmp_path, monkeypatch)
    result = sweep_module.run_motif_sweeps(config_path, stage_name="smoke")
    sweep_table = json.loads((result.run_dir / "sweep_table.json").read_text(encoding="utf-8"))
    receipt_path = result.run_dir / sweep_table["split_receipts"]["train"]["path"]
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="missing or corrupt"):
        sweep_module.run_motif_sweeps(config_path, stage_name="smoke")


def test_completed_run_rejects_corrupt_nested_cache_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _prepare_tiny_runner(tmp_path, monkeypatch)
    result = sweep_module.run_motif_sweeps(config_path, stage_name="smoke")
    completion = json.loads(result.completion_path.read_text(encoding="utf-8"))
    train_manifest_path = result.run_dir / completion["graph_caches"]["train"]["path"]
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    parquet_path = result.run_dir / train_manifest["data"]["path"]
    parquet_path.write_bytes(parquet_path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="missing or corrupt"):
        sweep_module.run_motif_sweeps(config_path, stage_name="smoke")


def test_graph_cache_loader_rejects_coerced_manifest_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _prepare_tiny_runner(tmp_path, monkeypatch)
    result = sweep_module.run_motif_sweeps(config_path, stage_name="smoke")
    completion = json.loads(result.completion_path.read_text(encoding="utf-8"))
    train_manifest_path = result.run_dir / completion["graph_caches"]["train"]["path"]
    manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    manifest["processed_count"] = str(manifest["processed_count"])
    train_manifest_path.write_bytes(sweep_module._canonical_json(manifest) + b"\n")

    with pytest.raises(ValueError, match="processed count"):
        sweep_module.load_completed_graph_cache(
            train_manifest_path,
            run_dir=result.run_dir,
            config_digest=manifest["config_digest"],
            input_manifest_sha256=manifest["input_manifest_sha256"],
            implementation_digest=manifest["implementation_digest"],
        )
