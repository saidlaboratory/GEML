from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import geml.experiments.goal5.learned_motifs as learned_motifs
from geml.compression.motif.mdl import MotifGraphMDLResult, graph_mdl_cost
from geml.compression.motif.vocabulary import (
    MotifChildRef,
    MotifNode,
    MotifPool,
    MotifTargetKind,
    build_motif_template,
    build_motif_vocabulary,
)
from geml.contracts.corpus import CorpusSplit
from geml.experiments.goal5.learned_motifs import (
    ExperimentMethod,
    Goal5LearnedMotifConfig,
    GraphExample,
    GraphInputFailure,
    HeldoutClaimStatus,
    LearnedMotifReplayInputs,
    SplitGraphBatch,
    VocabularyArm,
    build_exact_training_targets,
    deterministic_train_partition,
    evaluate_vocabulary_arms,
    fit_and_lock_selection,
    learned_config_digest,
    run_learned_motif_experiment,
)
from geml.experiments.goal5.motif_sweeps import vocabulary_payload
from geml.graph.schema import ChildRef, Graph, GraphNode, GraphRoot
from geml.learning.motif_selector import (
    FEATURE_NAMES,
    SelectorTrainingExample,
    candidate_pool_digest,
    fit_ridge_selector,
    motif_features,
    select_frequent_templates,
    select_learned_templates,
    select_natural_mdl_templates,
    select_random_templates,
    selector_from_payload,
    selector_payload,
    selector_payload_digest,
)

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "goal5_learned_motifs.yaml"
_PARTITION_DIGEST = "c" * 64


def _template(
    label: str,
    *,
    support: int,
    occurrences: int,
    two_nodes: bool = False,
):
    boundary = MotifChildRef(
        slot=0,
        target_kind=MotifTargetKind.BOUNDARY,
        target_index=0,
    )
    if two_nodes:
        internal = MotifChildRef(
            slot=0,
            target_kind=MotifTargetKind.INTERNAL,
            target_index=1,
        )
        nodes = (
            MotifNode(kind="operator", label=label, value=None, children=(internal,)),
            MotifNode(kind="operator", label="child", value=None, children=(boundary,)),
        )
    else:
        nodes = (MotifNode(kind="operator", label=label, value=None, children=(boundary,)),)
    return build_motif_template(
        source_family="macro",
        representation_mode="macro:official_v4:is_pure_eml=false",
        nodes=nodes,
        boundary_count=1,
        support_count=support,
        occurrence_count=occurrences,
    )


def _vocabulary():
    templates = (
        _template("a", support=10, occurrences=11),
        _template("b", support=30, occurrences=42, two_nodes=True),
        _template("c", support=20, occurrences=25),
        _template("d", support=5, occurrences=9, two_nodes=True),
    )
    return build_motif_vocabulary(
        pool=MotifPool.MACRO,
        min_size=1,
        max_size=2,
        min_support_count=1,
        vocabulary_limit=None,
        training_transaction_count=40,
        processed_count=40,
        failure_count=0,
        training_fingerprint="a" * 64,
        templates=templates,
    )


def _examples(vocabulary):
    return tuple(
        SelectorTrainingExample(
            motif_id=template.motif_id,
            split=CorpusSplit.TRAIN,
            mdl_gain_bits=10 * template.support_count - template.dictionary_cost_bits,
        )
        for template in vocabulary.templates
    )


def _selected_frequent(vocabulary, *, budget: int = 2):
    templates = select_frequent_templates(vocabulary, budget=budget)
    return build_motif_vocabulary(
        pool=vocabulary.pool,
        min_size=vocabulary.min_size,
        max_size=vocabulary.max_size,
        min_support_count=vocabulary.min_support_count,
        vocabulary_limit=budget,
        training_transaction_count=vocabulary.training_transaction_count,
        processed_count=vocabulary.processed_count,
        failure_count=vocabulary.failure_count,
        training_fingerprint=vocabulary.training_fingerprint,
        templates=templates,
    )


def _large_vocabulary(size: int):
    training_count = size + 64
    templates = tuple(
        _template(
            f"motif-{index:04d}",
            support=training_count - index,
            occurrences=training_count - index + (index % 3),
        )
        for index in range(size)
    )
    return build_motif_vocabulary(
        pool=MotifPool.MACRO,
        min_size=1,
        max_size=1,
        min_support_count=1,
        vocabulary_limit=None,
        training_transaction_count=training_count,
        processed_count=training_count,
        failure_count=0,
        training_fingerprint="e" * 64,
        templates=templates,
    )


def _graph(label: str, *, prefix: str) -> Graph:
    root_id = f"{prefix}-root"
    leaf_id = f"{prefix}-leaf"
    nodes = {
        leaf_id: GraphNode(
            node_id=leaf_id,
            family="macro",
            kind="leaf",
            label="x",
            value="x",
        )
    }
    if label in {"b", "d"}:
        child_id = f"{prefix}-child"
        nodes[child_id] = GraphNode(
            node_id=child_id,
            family="macro",
            kind="operator",
            label="child",
            children=(ChildRef(slot=0, target_id=leaf_id),),
        )
        target_id = child_id
    else:
        target_id = leaf_id
    nodes[root_id] = GraphNode(
        node_id=root_id,
        family="macro",
        kind="operator",
        label=label,
        children=(ChildRef(slot=0, target_id=target_id),),
    )
    return Graph(
        nodes=nodes,
        roots=(
            GraphRoot(
                root_id=f"{prefix}-graph-root",
                target_id=root_id,
                representation_mode="macro:official_v4:is_pure_eml=false",
            ),
        ),
    )


def _batch(split: CorpusSplit, labels: tuple[str, ...]) -> SplitGraphBatch:
    return SplitGraphBatch(
        split=split,
        records=tuple(
            GraphExample(
                expression_id=f"{split.value}-{index}",
                split=split,
                graph=_graph(label, prefix=f"{split.value}-{index}"),
            )
            for index, label in enumerate(labels)
        ),
    )


def _config() -> Goal5LearnedMotifConfig:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return Goal5LearnedMotifConfig.model_validate(raw)


def _replay_inputs(
    config: Goal5LearnedMotifConfig,
    vocabulary,
    batches: dict[CorpusSplit, SplitGraphBatch],
) -> LearnedMotifReplayInputs:
    return LearnedMotifReplayInputs(
        config=config,
        candidate_pool=vocabulary,
        split_loader=batches.__getitem__,
    )


def _complete_fixture_run(
    run_dir: Path,
) -> tuple[
    Goal5LearnedMotifConfig,
    object,
    object,
    dict[CorpusSplit, SplitGraphBatch],
]:
    config = _config()
    vocabulary = _vocabulary()
    selected_frequent = _selected_frequent(vocabulary)
    batches = {
        CorpusSplit.TRAIN: _batch(
            CorpusSplit.TRAIN,
            ("a", "b", "c", "d", "b"),
        ),
        CorpusSplit.VALIDATION: _batch(
            CorpusSplit.VALIDATION,
            ("a", "b", "c", "d"),
        ),
        CorpusSplit.TEST_IID: _batch(
            CorpusSplit.TEST_IID,
            ("a", "a", "b", "c"),
        ),
        CorpusSplit.TEST_OOD: _batch(
            CorpusSplit.TEST_OOD,
            ("d", "c", "b", "a"),
        ),
    }
    run_learned_motif_experiment(
        batches.__getitem__,
        candidate_pool=vocabulary,
        selected_frequent=selected_frequent,
        config=config,
        config_digest=learned_config_digest(config),
        output_dir=run_dir,
        reproduction_command="python -m geml.experiments.goal5.learned_motifs",
    )
    return config, vocabulary, selected_frequent, batches


def _leave_iid_receipt_after_crash(
    run_dir: Path,
) -> tuple[
    Goal5LearnedMotifConfig,
    object,
    object,
    dict[CorpusSplit, SplitGraphBatch],
]:
    config = _config()
    vocabulary = _vocabulary()
    selected_frequent = _selected_frequent(vocabulary)
    batches = {
        CorpusSplit.TRAIN: _batch(
            CorpusSplit.TRAIN,
            ("a", "b", "c", "d", "b"),
        ),
        CorpusSplit.VALIDATION: _batch(
            CorpusSplit.VALIDATION,
            ("a", "b", "c", "d"),
        ),
        CorpusSplit.TEST_IID: _batch(
            CorpusSplit.TEST_IID,
            ("a", "a", "b", "c"),
        ),
    }

    def crashing_loader(split: CorpusSplit) -> SplitGraphBatch:
        if split is CorpusSplit.TEST_OOD:
            raise RuntimeError("fixture crash after IID receipt")
        return batches[split]

    with pytest.raises(RuntimeError, match="fixture crash"):
        run_learned_motif_experiment(
            crashing_loader,
            candidate_pool=vocabulary,
            selected_frequent=selected_frequent,
            config=config,
            config_digest=learned_config_digest(config),
            output_dir=run_dir,
            reproduction_command="python -m geml.experiments.goal5.learned_motifs",
        )
    assert (run_dir / "heldout.test_iid.json").is_file()
    assert not (run_dir / "heldout.test_ood.json").exists()
    return config, vocabulary, selected_frequent, batches


def _rewrite_canonical_json(path: Path, payload: object) -> str:
    data = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_fit_is_deterministic_and_never_uses_ids_as_features() -> None:
    vocabulary = _vocabulary()
    first = fit_ridge_selector(
        vocabulary,
        _examples(vocabulary),
        ridge_lambda=0.01,
        train_partition_digest=_PARTITION_DIGEST,
        training_seed=19,
    )
    second = fit_ridge_selector(
        vocabulary,
        tuple(reversed(_examples(vocabulary))),
        ridge_lambda=0.01,
        train_partition_digest=_PARTITION_DIGEST,
        training_seed=19,
    )

    assert first == second
    assert first.candidate_pool_digest == candidate_pool_digest(vocabulary)
    assert all("id" not in name for name in FEATURE_NAMES)
    assert select_learned_templates(vocabulary, first, budget=2) == select_learned_templates(
        vocabulary,
        second,
        budget=2,
    )
    assert first.training_seed == 19
    assert selector_from_payload(selector_payload(first)) == first
    assert len(selector_payload_digest(first)) == 64

    corrupt = selector_payload(first)
    corrupt["feature_names"] = [*FEATURE_NAMES, "expression_id"]
    with pytest.raises(ValueError, match="feature_names"):
        selector_from_payload(corrupt)


def test_non_train_training_target_is_rejected_at_the_boundary() -> None:
    motif_id = _vocabulary().templates[0].motif_id
    with pytest.raises(ValueError, match="TRAIN"):
        SelectorTrainingExample(
            motif_id=motif_id,
            split=CorpusSplit.VALIDATION,
            mdl_gain_bits=1,
        )


def test_frequent_and_random_baselines_use_the_exact_budget() -> None:
    vocabulary = _vocabulary()
    frequent = select_frequent_templates(vocabulary, budget=3)
    random_first = select_random_templates(vocabulary, budget=3, seed=17)
    random_repeat = select_random_templates(vocabulary, budget=3, seed=17)

    assert len(frequent) == len(random_first) == 3
    assert frequent[0].support_count == max(
        template.support_count for template in vocabulary.templates
    )
    assert random_first == random_repeat
    assert len({template.motif_id for template in random_first}) == 3


def test_zero_variance_features_and_unregularized_fit_remain_finite() -> None:
    template = _template("only", support=2, occurrences=2)
    vocabulary = build_motif_vocabulary(
        pool=MotifPool.MACRO,
        min_size=1,
        max_size=1,
        min_support_count=1,
        vocabulary_limit=None,
        training_transaction_count=2,
        processed_count=2,
        failure_count=0,
        training_fingerprint="b" * 64,
        templates=(template,),
    )
    selector = fit_ridge_selector(
        vocabulary,
        (
            SelectorTrainingExample(
                motif_id=template.motif_id,
                split=CorpusSplit.TRAIN,
                mdl_gain_bits=-3,
            ),
        ),
        ridge_lambda=0.0,
        train_partition_digest=_PARTITION_DIGEST,
    )

    assert selector.predict(template) == pytest.approx(-3.0)
    assert select_natural_mdl_templates(vocabulary, selector) == ()


def test_feature_vector_exposes_structural_and_train_only_counts() -> None:
    template = _template("shared", support=7, occurrences=9, two_nodes=True)
    vector = motif_features(template)

    assert vector.motif_id == template.motif_id
    assert vector.values[0] == 2.0
    assert vector.values[5] == float(template.dictionary_cost_bits)
    assert vector.values[6] > 0.0
    assert vector.values[7] > 0.0


def test_candidate_digest_detects_train_count_drift() -> None:
    original = _vocabulary()
    changed_template = replace(
        original.templates[0],
        support_count=original.templates[0].support_count + 1,
        occurrence_count=original.templates[0].occurrence_count + 1,
    )
    changed = build_motif_vocabulary(
        pool=original.pool,
        min_size=original.min_size,
        max_size=original.max_size,
        min_support_count=original.min_support_count,
        vocabulary_limit=original.vocabulary_limit,
        training_transaction_count=original.training_transaction_count,
        processed_count=original.processed_count,
        failure_count=original.failure_count,
        training_fingerprint=original.training_fingerprint,
        templates=(changed_template, *original.templates[1:]),
    )

    assert candidate_pool_digest(original) != candidate_pool_digest(changed)


def test_train_partition_is_deterministic_and_train_only() -> None:
    batch = _batch(CorpusSplit.TRAIN, ("a", "b", "c", "d"))
    first = deterministic_train_partition(batch, fraction=0.5, seed=23)
    second = deterministic_train_partition(
        SplitGraphBatch(
            split=CorpusSplit.TRAIN,
            records=tuple(reversed(batch.records)),
        ),
        fraction=0.5,
        seed=23,
    )

    assert first == second
    assert len(first.batch.records) == 2
    assert len(first.digest) == 64
    with pytest.raises(ValueError, match="TRAIN"):
        deterministic_train_partition(
            _batch(CorpusSplit.VALIDATION, ("a",)),
            fraction=1.0,
            seed=23,
        )


def test_streaming_train_partition_retains_only_bounded_graph_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_count = 5_000
    maximum_graphs = 32
    vocabulary = _vocabulary()
    descriptors = {
        split: learned_motifs.GraphCacheDescriptor(
            split=split,
            data_path=tmp_path / f"{split.value}.parquet",
            manifest_path=tmp_path / f"{split.value}.json",
            data_sha256="a" * 64,
            manifest_sha256="b" * 64,
            requested_limit=None,
            processed_count=full_count if split is CorpusSplit.TRAIN else 1,
            success_count=full_count if split is CorpusSplit.TRAIN else 1,
            failure_count=0,
            failures=(),
        )
        for split in CorpusSplit
    }
    references = {
        split: learned_motifs.ArtifactReference(
            path=f"{split.value}.json",
            sha256="b" * 64,
        )
        for split in CorpusSplit
    }
    inputs = learned_motifs.FrequentSweepInputs(
        run_dir=tmp_path,
        candidate_pool=vocabulary,
        selected_frequent=_selected_frequent(vocabulary),
        selection_lock_sha256="c" * 64,
        run_complete_sha256="d" * 64,
        config_digest="e" * 64,
        input_manifest_sha256="f" * 64,
        implementation_digest="1" * 64,
        selected_configuration_digest="2" * 64,
        graph_cache_references=references,
        graph_cache_descriptors=descriptors,
    )

    class TrackingGraphExample:
        live_count = 0
        peak_count = 0

        def __init__(
            self,
            *,
            expression_id: str,
            split: CorpusSplit,
            graph: Graph,
        ) -> None:
            self.expression_id = expression_id
            self.split = split
            self.graph = graph
            type(self).live_count += 1
            type(self).peak_count = max(
                type(self).peak_count,
                type(self).live_count,
            )

        def __del__(self) -> None:
            type(self).live_count -= 1

    def streamed_graphs(_descriptor):
        for index in range(full_count):
            yield (
                f"train-source-{index:05d}",
                _graph("a", prefix=f"streamed-{index:05d}"),
            )

    monkeypatch.setattr(
        learned_motifs,
        "_load_frequent_graph_cache",
        lambda _inputs, _split: descriptors[CorpusSplit.TRAIN],
    )
    monkeypatch.setattr(learned_motifs, "iter_cached_graphs", streamed_graphs)
    monkeypatch.setattr(learned_motifs, "GraphExample", TrackingGraphExample)

    partition = learned_motifs.load_frequent_train_partition(
        inputs,
        fraction=0.8,
        seed=17,
        maximum_graphs=maximum_graphs,
    )

    assert len(partition.batch.records) == maximum_graphs
    assert partition.full_train_graph_count == full_count
    assert len(partition.source_expression_ids) == full_count
    assert TrackingGraphExample.peak_count <= maximum_graphs + 4


def test_positive_lambda_matches_the_full_ridge_solution() -> None:
    vocabulary = _vocabulary()
    examples = _examples(vocabulary)
    ridge_lambda = 0.5
    selector = fit_ridge_selector(
        vocabulary,
        examples,
        ridge_lambda=ridge_lambda,
        train_partition_digest=_PARTITION_DIGEST,
    )
    ordered = sorted(examples, key=lambda example: example.motif_id)
    by_id = {template.motif_id: template for template in vocabulary.templates}
    matrix = np.asarray(
        [motif_features(by_id[example.motif_id]).values for example in ordered],
        dtype=np.float64,
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0.0] = 1.0
    standardized = (matrix - means) / scales
    targets = np.asarray([example.mdl_gain_bits for example in ordered], dtype=np.float64)
    centered = targets - targets.mean()
    expected = np.linalg.solve(
        standardized.T @ standardized + ridge_lambda * np.eye(standardized.shape[1]),
        standardized.T @ centered,
    )

    assert selector.coefficients == pytest.approx(tuple(expected))


def test_all_upstream_and_reconstruction_failures_remain_in_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocabulary = _vocabulary()
    selected = _selected_frequent(vocabulary)
    graph = _graph("b", prefix="failure")
    baseline_bits = graph_mdl_cost(graph).total_bits

    def failed_result(
        _graph: Graph,
        _vocabulary,
        *,
        occurrences=None,
        prepared_graph=None,
    ) -> MotifGraphMDLResult:
        del occurrences, prepared_graph
        return MotifGraphMDLResult(
            success=False,
            baseline_bits=baseline_bits,
            conditional_data_bits=baseline_bits + 1,
            framing_bits=1,
            residual_bits=baseline_bits,
            occurrence_bits=0,
            selected_occurrence_count=0,
            candidate_occurrence_count=0,
            reconstruction_failure_count=1,
            attempted_selected_occurrence_count=1,
            error_type="ForcedFailure",
            error_message="fixture reconstruction failure",
        )

    monkeypatch.setattr(learned_motifs, "motif_graph_mdl_result", failed_result)
    batch = SplitGraphBatch(
        split=CorpusSplit.VALIDATION,
        records=(
            GraphExample(
                expression_id="evaluated",
                split=CorpusSplit.VALIDATION,
                graph=graph,
            ),
        ),
        failures=(
            GraphInputFailure(
                expression_id="upstream",
                split=CorpusSplit.VALIDATION,
                stage="macro_build",
                error_type="Unsupported",
                error_message="fixture unsupported row",
            ),
        ),
    )
    arm = VocabularyArm(
        arm_id="frequent",
        method=ExperimentMethod.FREQUENT,
        vocabulary=selected,
        budget=len(selected.templates),
    )

    evaluation = evaluate_vocabulary_arms(
        batch,
        (arm,),
        candidate_pool=vocabulary,
    )[0]

    assert evaluation.summary.processed_count == 1
    assert evaluation.summary.reconstruction_failure_count == 1
    assert evaluation.summary.conditional_data_bits == baseline_bits + 1
    assert evaluation.source_row_count == 2
    assert evaluation.total_failure_count == 2
    assert not evaluation.eligible
    assert evaluation.reconstruction_failures[0].error_type == "ForcedFailure"
    assert evaluation.reconstruction_failures[0].attempted_selected_occurrence_count == 1
    assert evaluation.upstream_failures[0].expression_id == "upstream"


def test_sparse_training_targets_equal_brute_force_singleton_mdl() -> None:
    vocabulary = _vocabulary()
    partition = deterministic_train_partition(
        _batch(CorpusSplit.TRAIN, ("a", "b", "c", "d", "b")),
        fraction=1.0,
        seed=31,
    )

    audit = build_exact_training_targets(vocabulary, partition)
    arms = tuple(
        VocabularyArm(
            arm_id=template.motif_id,
            method=ExperimentMethod.LEARNED,
            vocabulary=learned_motifs._selected_vocabulary(
                vocabulary,
                (template,),
                budget=1,
            ),
            budget=1,
            ridge_lambda=0.0,
        )
        for template in vocabulary.templates
    )
    brute_force = []
    for arm in arms:
        accumulator = learned_motifs._MDLAccumulator(arm.vocabulary)
        for record in partition.batch.records:
            occurrences = learned_motifs.find_vocabulary_occurrences(
                record.graph,
                arm.vocabulary,
            )
            accumulator.add(
                record.expression_id,
                learned_motifs.motif_graph_mdl_result(
                    record.graph,
                    arm.vocabulary,
                    occurrences=occurrences,
                ),
            )
        brute_force.append(
            accumulator.finish(
                split=CorpusSplit.TRAIN,
                arm=arm,
                upstream_failures=partition.batch.failures,
            )
        )

    assert tuple(target.summary for target in audit.targets) == tuple(
        evaluation.summary for evaluation in brute_force
    )
    assert audit.union_match_pass_count == len(partition.batch.records)
    assert audit.base_singleton_evaluation_count == len(partition.batch.records)
    assert audit.sparse_singleton_evaluation_count < (
        len(partition.batch.records) * len(vocabulary.templates)
    )


def test_sparse_target_scaling_at_1024_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocabulary = _large_vocabulary(1024)
    batch = _batch(
        CorpusSplit.TRAIN,
        tuple(f"motif-{index:04d}" for index in range(128)),
    )
    partition = deterministic_train_partition(
        batch,
        fraction=1.0,
        seed=37,
        maximum_graphs=128,
    )
    calls = 0
    exact_result = learned_motifs.motif_graph_mdl_result

    def counted_result(*args, **kwargs):
        nonlocal calls
        calls += 1
        return exact_result(*args, **kwargs)

    monkeypatch.setattr(
        learned_motifs,
        "motif_graph_mdl_result",
        counted_result,
    )
    started = time.perf_counter()
    audit = build_exact_training_targets(vocabulary, partition)
    elapsed = time.perf_counter() - started

    brute_force_calls = len(vocabulary.templates) * len(batch.records)
    assert len(audit.targets) == 1024
    assert calls == (
        audit.base_singleton_evaluation_count + audit.sparse_singleton_evaluation_count
    )
    assert calls <= 2 * len(batch.records)
    assert calls * 100 < brute_force_calls
    assert elapsed < 15.0


def test_thirty_seven_arm_evaluation_reuses_matches_and_duplicate_vocabularies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocabulary = _large_vocabulary(1024)
    budget = 256
    frequent = _selected_frequent(vocabulary, budget=budget)
    arms = (
        *(
            VocabularyArm(
                arm_id=f"learned:{ridge_lambda}",
                method=ExperimentMethod.LEARNED,
                vocabulary=frequent,
                budget=budget,
                ridge_lambda=ridge_lambda,
            )
            for ridge_lambda in (0.0, 1.0e-6, 1.0e-4, 0.01, 1.0, 100.0)
        ),
        VocabularyArm(
            arm_id="frequent",
            method=ExperimentMethod.FREQUENT,
            vocabulary=frequent,
            budget=budget,
        ),
        *(
            VocabularyArm(
                arm_id=f"random:{seed}",
                method=ExperimentMethod.RANDOM,
                vocabulary=learned_motifs._selected_vocabulary(
                    vocabulary,
                    select_random_templates(
                        vocabulary,
                        budget=budget,
                        seed=seed,
                    ),
                    budget=budget,
                ),
                budget=budget,
                random_seed=seed,
            )
            for seed in range(30)
        ),
    )
    batch = _batch(
        CorpusSplit.VALIDATION,
        tuple(f"motif-{index:04d}" for index in range(32)),
    )
    match_calls = 0
    mdl_calls = 0
    exact_match = learned_motifs.find_vocabulary_occurrences
    exact_result = learned_motifs.motif_graph_mdl_result

    def counted_match(*args, **kwargs):
        nonlocal match_calls
        match_calls += 1
        return exact_match(*args, **kwargs)

    def counted_result(*args, **kwargs):
        nonlocal mdl_calls
        mdl_calls += 1
        return exact_result(*args, **kwargs)

    monkeypatch.setattr(
        learned_motifs,
        "find_vocabulary_occurrences",
        counted_match,
    )
    monkeypatch.setattr(
        learned_motifs,
        "motif_graph_mdl_result",
        counted_result,
    )
    started = time.perf_counter()
    evaluations = evaluate_vocabulary_arms(
        batch,
        arms,
        candidate_pool=vocabulary,
    )
    elapsed = time.perf_counter() - started
    unique_vocabulary_count = len({arm.vocabulary.vocabulary_id for arm in arms})

    assert len(arms) == len(evaluations) == 37
    assert match_calls == len(batch.records)
    assert mdl_calls == unique_vocabulary_count * len(batch.records)
    assert unique_vocabulary_count <= 31
    assert elapsed < 20.0


def test_runner_locks_before_test_and_test_changes_cannot_change_selection(
    tmp_path: Path,
) -> None:
    config = _config()
    vocabulary = _vocabulary()
    selected_frequent = _selected_frequent(vocabulary)
    fixed_batches = {
        CorpusSplit.TRAIN: _batch(CorpusSplit.TRAIN, ("a", "b", "c", "d", "b")),
        CorpusSplit.VALIDATION: _batch(
            CorpusSplit.VALIDATION,
            ("a", "b", "c", "d"),
        ),
    }

    def run_once(
        run_dir: Path,
        *,
        test_iid_labels: tuple[str, ...],
    ):
        calls: list[CorpusSplit] = []
        heldout_batches = {
            CorpusSplit.TEST_IID: _batch(CorpusSplit.TEST_IID, test_iid_labels),
            CorpusSplit.TEST_OOD: _batch(
                CorpusSplit.TEST_OOD,
                ("d", "c", "b", "a"),
            ),
        }

        def loader(split: CorpusSplit) -> SplitGraphBatch:
            calls.append(split)
            if split in {CorpusSplit.TEST_IID, CorpusSplit.TEST_OOD}:
                assert (run_dir / "selection.lock.json").is_file()
                assert not (run_dir / "heldout_results.json").exists()
            return fixed_batches[split] if split in fixed_batches else heldout_batches[split]

        result = run_learned_motif_experiment(
            loader,
            candidate_pool=vocabulary,
            selected_frequent=selected_frequent,
            config=config,
            config_digest=learned_config_digest(config),
            output_dir=run_dir,
            reproduction_command="python -m geml.experiments.goal5.learned_motifs",
        )
        return result, tuple(calls), (run_dir / "selection.lock.json").read_bytes()

    first, first_calls, first_lock = run_once(
        tmp_path / "first",
        test_iid_labels=("a", "a", "b", "c"),
    )
    second, second_calls, second_lock = run_once(
        tmp_path / "second",
        test_iid_labels=("d", "d", "d", "d"),
    )

    assert first_calls == second_calls == tuple(CorpusSplit)
    assert first_lock == second_lock
    assert first.locked.lock_digest == second.locked.lock_digest
    assert len(first.locked.random_arms) == 30
    assert len({arm.random_seed for arm in first.locked.random_arms}) == 30
    assert {
        first.locked.learned_arm.budget,
        first.locked.frequent_arm.budget,
        *(arm.budget for arm in first.locked.random_arms),
    } == {len(selected_frequent.templates)}
    assert first.test_iid.split is CorpusSplit.TEST_IID
    assert first.test_ood.split is CorpusSplit.TEST_OOD
    assert first.claim_status in {
        HeldoutClaimStatus.SUPPORTED,
        HeldoutClaimStatus.NULL_RESULT,
    }
    assert (tmp_path / "first" / "run.complete.json").is_file()
    assert json.loads(first_lock)["heldout_artifacts_absent_at_lock"] is True
    first_replay_batches = {
        **fixed_batches,
        CorpusSplit.TEST_IID: _batch(
            CorpusSplit.TEST_IID,
            ("a", "a", "b", "c"),
        ),
        CorpusSplit.TEST_OOD: _batch(
            CorpusSplit.TEST_OOD,
            ("d", "c", "b", "a"),
        ),
    }
    replay_inputs = _replay_inputs(
        config,
        vocabulary,
        first_replay_batches,
    )
    completed = learned_motifs.load_completed_learned_motif_run(
        tmp_path / "first",
        require_current_implementation=False,
        replay_inputs=replay_inputs,
    )
    assert completed.learned_vocabulary == first.locked.learned_arm.vocabulary
    assert completed.frequent_vocabulary == selected_frequent
    assert completed.claim_status is first.claim_status

    resume_calls: list[CorpusSplit] = []

    def resume_loader(split: CorpusSplit) -> SplitGraphBatch:
        resume_calls.append(split)
        if split in {CorpusSplit.TEST_IID, CorpusSplit.TEST_OOD}:
            raise AssertionError("immutable held-out receipts must be reused")
        return fixed_batches[split]

    resumed = run_learned_motif_experiment(
        resume_loader,
        candidate_pool=vocabulary,
        selected_frequent=selected_frequent,
        config=config,
        config_digest=learned_config_digest(config),
        output_dir=tmp_path / "first",
        reproduction_command=("python -m geml.experiments.goal5.learned_motifs"),
    )
    assert resume_calls == [CorpusSplit.TRAIN, CorpusSplit.VALIDATION]
    assert resumed.test_iid == first.test_iid
    assert resumed.test_ood == first.test_ood

    (tmp_path / "first" / "selector.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(
        learned_motifs.LearnedMotifConfigurationError,
        match="checksum mismatch",
    ):
        learned_motifs.load_completed_learned_motif_run(
            tmp_path / "first",
            require_current_implementation=False,
            replay_inputs=replay_inputs,
        )


def test_resume_rejects_iid_ood_identity_overlap_after_iid_receipt(
    tmp_path: Path,
) -> None:
    config, vocabulary, selected_frequent, batches = _leave_iid_receipt_after_crash(tmp_path)
    iid_expression_id = batches[CorpusSplit.TEST_IID].records[0].expression_id
    overlapping_ood = SplitGraphBatch(
        split=CorpusSplit.TEST_OOD,
        records=(
            GraphExample(
                expression_id=iid_expression_id,
                split=CorpusSplit.TEST_OOD,
                graph=_graph("d", prefix="overlapping-ood"),
            ),
        ),
    )
    calls: list[CorpusSplit] = []

    def resume_loader(split: CorpusSplit) -> SplitGraphBatch:
        calls.append(split)
        if split is CorpusSplit.TEST_IID:
            raise AssertionError("IID receipt must be reused")
        if split is CorpusSplit.TEST_OOD:
            return overlapping_ood
        return batches[split]

    with pytest.raises(
        learned_motifs.LearnedMotifProtocolError,
        match="reuses expression",
    ):
        run_learned_motif_experiment(
            resume_loader,
            candidate_pool=vocabulary,
            selected_frequent=selected_frequent,
            config=config,
            config_digest=learned_config_digest(config),
            output_dir=tmp_path,
            reproduction_command="python -m geml.experiments.goal5.learned_motifs",
        )
    assert calls == [
        CorpusSplit.TRAIN,
        CorpusSplit.VALIDATION,
        CorpusSplit.TEST_OOD,
    ]


def test_resume_rejects_inconsistent_iid_receipt_arm_binding(
    tmp_path: Path,
) -> None:
    config, vocabulary, selected_frequent, batches = _leave_iid_receipt_after_crash(tmp_path)
    receipt_path = tmp_path / "heldout.test_iid.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["report"]["learned"]["arm_id"] = "learned:inconsistent"
    _rewrite_canonical_json(receipt_path, receipt)
    calls: list[CorpusSplit] = []

    def resume_loader(split: CorpusSplit) -> SplitGraphBatch:
        calls.append(split)
        if split in {CorpusSplit.TEST_IID, CorpusSplit.TEST_OOD}:
            raise AssertionError("inconsistent receipt must fail before held-out loading")
        return batches[split]

    with pytest.raises(
        learned_motifs.LearnedMotifProtocolError,
        match="not bound to locked arm",
    ):
        run_learned_motif_experiment(
            resume_loader,
            candidate_pool=vocabulary,
            selected_frequent=selected_frequent,
            config=config,
            config_digest=learned_config_digest(config),
            output_dir=tmp_path,
            reproduction_command="python -m geml.experiments.goal5.learned_motifs",
        )
    assert calls == [CorpusSplit.TRAIN, CorpusSplit.VALIDATION]


def test_completed_loader_rejects_coordinated_receipt_arm_inconsistency(
    tmp_path: Path,
) -> None:
    config = _config()
    vocabulary = _vocabulary()
    selected_frequent = _selected_frequent(vocabulary)
    batches = {
        CorpusSplit.TRAIN: _batch(
            CorpusSplit.TRAIN,
            ("a", "b", "c", "d", "b"),
        ),
        CorpusSplit.VALIDATION: _batch(
            CorpusSplit.VALIDATION,
            ("a", "b", "c", "d"),
        ),
        CorpusSplit.TEST_IID: _batch(
            CorpusSplit.TEST_IID,
            ("a", "a", "b", "c"),
        ),
        CorpusSplit.TEST_OOD: _batch(
            CorpusSplit.TEST_OOD,
            ("d", "c", "b", "a"),
        ),
    }
    run_learned_motif_experiment(
        batches.__getitem__,
        candidate_pool=vocabulary,
        selected_frequent=selected_frequent,
        config=config,
        config_digest=learned_config_digest(config),
        output_dir=tmp_path,
        reproduction_command="python -m geml.experiments.goal5.learned_motifs",
    )

    receipt_path = tmp_path / "heldout.test_iid.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["report"]["learned"]["arm_id"] = "learned:inconsistent"
    receipt_sha256 = _rewrite_canonical_json(receipt_path, receipt)

    heldout_path = tmp_path / "heldout_results.json"
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
    heldout["split_receipts"][CorpusSplit.TEST_IID.value]["sha256"] = receipt_sha256
    heldout["test_iid"] = receipt["report"]
    heldout_sha256 = _rewrite_canonical_json(heldout_path, heldout)

    result_path = tmp_path / "experiment.result.json"
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    result_payload["test_iid"] = receipt["report"]
    result_sha256 = _rewrite_canonical_json(result_path, result_payload)

    completion_path = tmp_path / "run.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["artifacts"]["test_iid_receipt"]["sha256"] = receipt_sha256
    completion["artifacts"]["heldout_results"]["sha256"] = heldout_sha256
    completion["artifacts"]["experiment_result"]["sha256"] = result_sha256
    _rewrite_canonical_json(completion_path, completion)

    with pytest.raises(
        learned_motifs.LearnedMotifProtocolError,
        match="inconsistent with locked learned arm",
    ):
        learned_motifs.load_completed_learned_motif_run(
            tmp_path,
            require_current_implementation=False,
            replay_inputs=_replay_inputs(config, vocabulary, batches),
        )


def test_completed_loader_recursively_reauthenticates_frequent_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    vocabulary = _vocabulary()
    provenance = learned_motifs.FrequentSweepProvenance(
        run_directory="outputs/final/goal5/frequent/final/fake",
        selection_lock_sha256="1" * 64,
        run_complete_sha256="2" * 64,
        config_digest="3" * 64,
        input_manifest_sha256="4" * 64,
        implementation_digest="5" * 64,
        selected_configuration_digest="6" * 64,
    )
    source_references = {
        split: learned_motifs.ArtifactReference(
            path=f"graph_cache/{split.value}.manifest.json",
            sha256=hashlib.sha256(split.value.encode("ascii")).hexdigest(),
        )
        for split in CorpusSplit
    }
    batches = {
        CorpusSplit.TRAIN: _batch(
            CorpusSplit.TRAIN,
            ("a", "b", "c", "d", "b"),
        ),
        CorpusSplit.VALIDATION: _batch(
            CorpusSplit.VALIDATION,
            ("a", "b", "c", "d"),
        ),
        CorpusSplit.TEST_IID: _batch(
            CorpusSplit.TEST_IID,
            ("a", "a", "b", "c"),
        ),
        CorpusSplit.TEST_OOD: _batch(
            CorpusSplit.TEST_OOD,
            ("d", "c", "b", "a"),
        ),
    }
    run_learned_motif_experiment(
        batches.__getitem__,
        candidate_pool=vocabulary,
        selected_frequent=_selected_frequent(vocabulary),
        config=config,
        config_digest=learned_config_digest(config),
        output_dir=tmp_path,
        reproduction_command="python -m geml.experiments.goal5.learned_motifs",
        frequent_provenance=provenance,
        source_graph_cache_references=source_references,
    )

    def reject_source(_run_dir):
        raise learned_motifs.LearnedMotifConfigurationError("fixture upstream mismatch")

    monkeypatch.setattr(
        learned_motifs,
        "load_frequent_sweep_inputs",
        reject_source,
    )
    with pytest.raises(
        learned_motifs.LearnedMotifProtocolError,
        match="recursive authentication",
    ):
        learned_motifs.load_completed_learned_motif_run(
            tmp_path,
            require_current_implementation=False,
        )


def test_selector_decoder_requires_exact_json_float_types() -> None:
    vocabulary = _vocabulary()
    selector = fit_ridge_selector(
        vocabulary,
        _examples(vocabulary),
        ridge_lambda=0.01,
        train_partition_digest=_PARTITION_DIGEST,
    )
    payload = selector_payload(selector)
    payload["intercept"] = int(selector.intercept)

    with pytest.raises(ValueError, match="exact float"):
        selector_from_payload(payload)


def test_selector_decoder_requires_a_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        selector_from_payload([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "path",
    [
        ".",
        "./selector.json",
        "artifacts//selector.json",
        "artifacts/../selector.json",
        "../selector.json",
        "/selector.json",
        "C:/selector.json",
        r"artifacts\selector.json",
    ],
)
def test_artifact_references_require_canonical_relative_posix_paths(path: str) -> None:
    with pytest.raises(ValueError, match="canonical relative POSIX"):
        learned_motifs.ArtifactReference(path=path, sha256="a" * 64)

    reference = learned_motifs.ArtifactReference(
        path="artifacts/selector.json",
        sha256="a" * 64,
    )
    assert reference.path == "artifacts/selector.json"


def test_completed_loader_requires_canonical_json_bytes(tmp_path: Path) -> None:
    config, vocabulary, _frequent, batches = _complete_fixture_run(tmp_path)
    completion = tmp_path / "run.complete.json"
    completion.write_bytes(completion.read_bytes().rstrip(b"\n"))

    with pytest.raises(
        learned_motifs.LearnedMotifConfigurationError,
        match="not canonical JSON",
    ):
        learned_motifs.load_completed_learned_motif_run(
            tmp_path,
            require_current_implementation=False,
            replay_inputs=_replay_inputs(config, vocabulary, batches),
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "total_failure_count",
        "candidate_prefilter",
        "computation",
        "partition",
        "target_failure",
    ],
)
def test_completed_loader_rejects_inconsistent_training_target_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    original_payload = learned_motifs._target_payload

    def corrupt_target_payload(audit):
        payload = original_payload(audit)
        if corruption == "total_failure_count":
            payload["total_failure_count"] = 1
        elif corruption == "candidate_prefilter":
            payload["candidate_prefilter"] = {"unexpected": True}
        elif corruption == "computation":
            payload["computation"]["union_match_pass_count"] += 1
        elif corruption == "partition":
            payload["partition"]["fraction"] = 0.5
        else:
            row = payload["targets"][0]
            row["summary"]["success_count"] -= 1
            row["summary"]["reconstruction_failure_count"] += 1
            row["reconstruction_failure_count"] = 1
            row["reconstruction_failures"] = [
                {
                    "attempted_selected_occurrence_count": 0,
                    "error_message": "fixture retained failure",
                    "error_type": "FixtureFailure",
                    "expression_id": "train-0",
                }
            ]
        return payload

    with monkeypatch.context() as patch:
        patch.setattr(
            learned_motifs,
            "_target_payload",
            corrupt_target_payload,
        )
        config, vocabulary, _frequent, batches = _complete_fixture_run(tmp_path)

    with pytest.raises(learned_motifs.LearnedMotifProtocolError):
        learned_motifs.load_completed_learned_motif_run(
            tmp_path,
            require_current_implementation=False,
            replay_inputs=_replay_inputs(config, vocabulary, batches),
        )


def test_completed_loader_refits_selector_from_exact_train_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_fit = learned_motifs.fit_ridge_selector

    def biased_fit(*args, **kwargs):
        selector = exact_fit(*args, **kwargs)
        return replace(selector, intercept=selector.intercept + 1_000_000.0)

    with monkeypatch.context() as patch:
        patch.setattr(learned_motifs, "fit_ridge_selector", biased_fit)
        config, vocabulary, _frequent, batches = _complete_fixture_run(tmp_path)

    with pytest.raises(
        learned_motifs.LearnedMotifProtocolError,
        match="exact TRAIN refit",
    ):
        learned_motifs.load_completed_learned_motif_run(
            tmp_path,
            require_current_implementation=False,
            replay_inputs=_replay_inputs(config, vocabulary, batches),
        )


@pytest.mark.parametrize(
    "corruption",
    ["better_nonselected", "selected_failure"],
)
def test_completed_loader_replays_validation_model_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    exact_payload = learned_motifs._validation_payload

    def corrupt_validation_payload(report):
        payload = exact_payload(report)
        if corruption == "better_nonselected":
            summary = payload["candidates"][1]["evaluation"]["summary"]
            summary["conditional_data_bits"] = 0
            summary["total_mdl_bits"] = summary["dictionary_bits"]
            summary["framing_bits"] = 0
            summary["residual_bits"] = 0
            summary["occurrence_bits"] = 0
            summary["candidate_occurrence_count"] = 0
            summary["selected_occurrence_count"] = 0
            summary["selected_motif_counts"] = []
            summary["savings_bits"] = summary["baseline_total_bits"] - summary["total_mdl_bits"]
        else:
            evaluation = payload["candidates"][payload["selected_candidate_index"]]["evaluation"]
            evaluation["summary"]["success_count"] -= 1
            evaluation["summary"]["reconstruction_failure_count"] += 1
            evaluation["reconstruction_failures"] = [
                {
                    "attempted_selected_occurrence_count": 0,
                    "error_message": "fixture retained failure",
                    "error_type": "FixtureFailure",
                    "expression_id": "validation-0",
                }
            ]
            evaluation["total_failure_count"] = 1
        return payload

    with monkeypatch.context() as patch:
        patch.setattr(
            learned_motifs,
            "_validation_payload",
            corrupt_validation_payload,
        )
        config, vocabulary, _frequent, batches = _complete_fixture_run(tmp_path)

    with pytest.raises(learned_motifs.LearnedMotifProtocolError):
        learned_motifs.load_completed_learned_motif_run(
            tmp_path,
            require_current_implementation=False,
            replay_inputs=_replay_inputs(config, vocabulary, batches),
        )


def test_completed_loader_replays_heldout_aggregates_from_graphs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_payload = learned_motifs._heldout_payload

    def corrupt_heldout_payload(report):
        payload = exact_payload(report)
        if report.split is CorpusSplit.TEST_IID:
            summary = payload["learned"]["summary"]
            summary["conditional_data_bits"] += 1
            summary["total_mdl_bits"] += 1
            summary["residual_bits"] += 1
            summary["savings_bits"] -= 1
        return payload

    with monkeypatch.context() as patch:
        patch.setattr(
            learned_motifs,
            "_heldout_payload",
            corrupt_heldout_payload,
        )
        config, vocabulary, _frequent, batches = _complete_fixture_run(tmp_path)

    with pytest.raises(
        learned_motifs.LearnedMotifProtocolError,
        match="deterministic replay",
    ):
        learned_motifs.load_completed_learned_motif_run(
            tmp_path,
            require_current_implementation=False,
            replay_inputs=_replay_inputs(config, vocabulary, batches),
        )


def test_checked_in_config_digest_is_recursively_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    source_run = repository / "outputs" / "frequent"
    config_path = repository / "configs" / "goal5.yaml"
    source_run.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    config_path.write_text("fixture: true\n", encoding="utf-8")
    completion = {
        "config_digest": "a" * 64,
        "reproduction_command": (
            "python -m geml.experiments.goal5.learned_motifs "
            "--config configs/goal5.yaml "
            "--frequent-sweep-run-dir outputs/frequent"
        ),
    }
    loaded = SimpleNamespace(
        config_digest="b" * 64,
        frequent_sweep_run_dir=source_run.resolve(),
    )
    monkeypatch.setattr(
        learned_motifs,
        "_repository_root",
        lambda _path: repository.resolve(),
    )
    monkeypatch.setattr(
        learned_motifs,
        "load_learned_motif_config",
        lambda *_args, **_kwargs: loaded,
    )

    with pytest.raises(
        learned_motifs.LearnedMotifProtocolError,
        match="checked-in config",
    ):
        learned_motifs._learned_config_from_completion(
            repository / "outputs" / "learned",
            completion,
            inputs=SimpleNamespace(run_dir=source_run.resolve()),
            implementation_digest="c" * 64,
        )


def test_reproduction_config_must_reside_in_config_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    source_run = repository / "outputs" / "frequent"
    misplaced_config = repository / "outputs" / "goal5.yaml"
    source_run.mkdir(parents=True)
    misplaced_config.write_text("fixture: true\n", encoding="utf-8")
    completion = {
        "config_digest": "a" * 64,
        "reproduction_command": (
            "python -m geml.experiments.goal5.learned_motifs "
            "--config outputs/goal5.yaml "
            "--frequent-sweep-run-dir outputs/frequent"
        ),
    }
    monkeypatch.setattr(
        learned_motifs,
        "_repository_root",
        lambda _path: repository.resolve(),
    )

    with pytest.raises(
        learned_motifs.LearnedMotifProtocolError,
        match="checked-in config directory",
    ):
        learned_motifs._learned_config_from_completion(
            repository / "outputs" / "learned",
            completion,
            inputs=SimpleNamespace(run_dir=source_run.resolve()),
            implementation_digest="c" * 64,
        )


def test_validation_lock_contains_no_heldout_fields() -> None:
    config = _config()
    vocabulary = _vocabulary()
    selected_frequent = _selected_frequent(vocabulary)
    locked, report = fit_and_lock_selection(
        _batch(CorpusSplit.TRAIN, ("a", "b", "c", "d", "b")),
        _batch(CorpusSplit.VALIDATION, ("a", "b", "c", "d")),
        candidate_pool=vocabulary,
        selected_frequent=selected_frequent,
        config=config,
        config_digest=learned_config_digest(config),
    )

    assert report.selected.evaluation.split is CorpusSplit.VALIDATION
    assert locked.training_target_audit.targets
    assert {target.example.split for target in locked.training_target_audit.targets} == {
        CorpusSplit.TRAIN
    }
    assert "test" not in repr(locked).lower()
    assert report.selected.selector.training_target_digest


def test_config_freezes_posthoc_reporting_and_thirty_random_repetitions() -> None:
    config = _config()

    assert config.frequent_sweep_run_dir is None
    assert config.baselines.random_repetitions == 30
    assert config.baselines.equal_motif_budget is True
    assert config.evaluation.validation_selects_hyperparameters is True
    assert config.evaluation.evaluate_tests_once_after_lock is True
    assert config.evaluation.posthoc_claim_rule.startswith("report_only")
    assert config.selector.candidate_prefilter == "train-frequency-rank-v1"
    assert config.selector.maximum_candidate_motifs == 4096
    assert config.selector.maximum_target_graphs == 4096
    assert config.selector.target_definition == ("exact-singleton-two-part-mdl-gain-v1")
    assert vocabulary_payload(_vocabulary())["training_fingerprint"] == "a" * 64
