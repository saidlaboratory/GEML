from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import geml.learning.value.train as value_train
from geml.learning.value.head import (
    GoalConditionedValueHead,
    ValueHeadConfig,
    ValueHeadParameters,
    ValuePredictionV1,
)
from geml.learning.value.train import (
    PRODUCTION_SEEDS,
    FixtureBenchmarkExclusionsV1,
    FrozenBenchmarkReferenceV1,
    ValueRowStatus,
    ValueSeedResultV1,
    ValueTrainingConfigurationError,
    ValueTrainingConfigV1,
    ValueTrainingEnvironmentV1,
    ValueTrainingInputIdentityV1,
    ValueTrainingStatus,
    evaluate_value_head,
    hybrid_value_loss,
    run_three_seed_training,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "goal8_value.yaml"


class FixtureEncoder:
    embedding_dimension = 2
    encoder_digest = hashlib.sha256(b"fixture-shared-encoder").hexdigest()
    trainable_parameter_count = 50

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def encode_batch(
        self,
        states: Sequence[str],
        *,
        training: bool,
    ) -> Sequence[Sequence[float]]:
        self.calls.append((tuple(states), training))
        embeddings = {
            "a": (1.0, 0.0),
            "b": (0.0, 1.0),
            "c": (1.0, 1.0),
        }
        return tuple(embeddings[state] for state in states)


def _head_config(**overrides: int | str) -> ValueHeadConfig:
    values: dict[str, int | str] = {
        "embedding_dimension": 2,
        "shared_encoder_digest": FixtureEncoder.encoder_digest,
        "hidden_dimension": 2,
        "ordinal_thresholds": 4,
        "maximum_witness_steps": 4,
        "maximum_trainable_parameters": 100,
    }
    values.update(overrides)
    return ValueHeadConfig(**values)


def _directional_parameters() -> ValueHeadParameters:
    # Feature order is current, goal, current-goal, current*goal.
    return ValueHeadParameters(
        hidden_weights=(
            (1.0, 0.0),
            (0.0, 0.0),
            (-1.0, 0.0),
            (0.0, 0.0),
            (0.5, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ),
        hidden_bias=(0.0, 0.0),
        regression_weights=(1.0, 0.0),
        regression_bias=0.0,
        ordinal_weights=((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)),
        ordinal_bias=(0.0, 0.0, 0.0, 0.0),
    )


def test_head_is_goal_conditioned_directional_and_deterministic() -> None:
    encoder = FixtureEncoder()
    head = GoalConditionedValueHead(
        encoder,
        _head_config(),
        seed=20260726,
        parameters=_directional_parameters(),
    )

    forward = head.predict_batch(["a"], ["b"])
    repeated = head.predict_batch(["a"], ["b"])
    reversed_pair = head.predict_batch(["b"], ["a"])

    assert forward == repeated
    assert forward != reversed_pair
    assert head.score_batch(["a"], ["b"]) == (forward[0].remaining_witness_steps,)
    assert head.score_search_batch(["a"], "b") == head.score_batch(["a"], ["b"])
    assert all(not training for _, training in encoder.calls)
    assert encoder.calls[0][0] == ("a",)
    assert encoder.calls[1][0] == ("b",)


def test_stable_cryptographic_initialization_and_compact_budget() -> None:
    first = GoalConditionedValueHead(FixtureEncoder(), _head_config(), seed=20260726)
    second = GoalConditionedValueHead(FixtureEncoder(), _head_config(), seed=20260726)
    different = GoalConditionedValueHead(FixtureEncoder(), _head_config(), seed=20260727)

    assert first.parameters == second.parameters
    assert first.parameters != different.parameters
    assert first.trainable_parameter_count == 33
    assert first.total_trainable_parameter_count == 83
    assert first.shared_encoder_digest == FixtureEncoder.encoder_digest
    with pytest.raises(ValueError, match="exceeds"):
        GoalConditionedValueHead(
            FixtureEncoder(),
            _head_config(maximum_trainable_parameters=32),
            seed=20260726,
        )
    with pytest.raises(ValueError, match="shared encoder plus value head"):
        GoalConditionedValueHead(
            FixtureEncoder(),
            _head_config(
                maximum_trainable_parameters=82,
                maximum_total_trainable_parameters=82,
            ),
            seed=20260726,
        )
    with pytest.raises(ValueError, match="digest does not match"):
        GoalConditionedValueHead(
            FixtureEncoder(),
            _head_config(shared_encoder_digest="0" * 64),
            seed=20260726,
        )


def test_head_and_training_configs_reject_boolean_numeric_fields() -> None:
    with pytest.raises(ValueError, match="dimensions and limits"):
        _head_config(embedding_dimension=True)
    with pytest.raises(ValueError, match="seed"):
        GoalConditionedValueHead(FixtureEncoder(), _head_config(), seed=True)
    with pytest.raises(ValueError, match="finite real"):
        ValuePredictionV1(True, (0.5,) * 4)
    with pytest.raises(ValueError, match="training limits"):
        ValueTrainingConfigV1(head=_head_config(), maximum_epochs=True)


def test_goal8_value_yaml_matches_the_executable_scientific_contract() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "geml-goal8-value-config-v1"
    assert payload["stage"] == "production"
    target = payload["target"]
    architecture = payload["architecture"]
    loss = payload["loss"]
    training = payload["training"]
    leakage = payload["leakage"]

    assert architecture["shared_encoder_digest"] is None
    assert architecture["embedding_dimension"] is None
    head = ValueHeadConfig(
        embedding_dimension=2,
        shared_encoder_digest=FixtureEncoder.encoder_digest,
        hidden_dimension=architecture["hidden_dimension"],
        ordinal_thresholds=architecture["ordinal_thresholds"],
        maximum_witness_steps=target["maximum_witness_steps"],
        maximum_trainable_parameters=architecture["maximum_trainable_parameters"],
        maximum_total_trainable_parameters=architecture["maximum_total_trainable_parameters"],
        target_policy=target["clipping_policy"],
    )
    config = ValueTrainingConfigV1(
        head=head,
        seeds=tuple(training["seeds"]),
        maximum_epochs=training["maximum_epochs"],
        early_stopping_patience=training["early_stopping_patience"],
        batch_size=training["batch_size"],
        regression_loss="right_censored_huber",
        huber_delta=loss["huber_delta"],
        regression_weight=loss["regression_weight"],
        ordinal_weight=loss["ordinal_weight"],
        selection_metric=training["selection_metric"],
    )

    assert target["name"] == "remaining_witness_steps"
    assert target["shortest_path_claim"] is False
    assert loss["regression"] == "one_sided_huber_for_right_censored_targets"
    assert loss["ordinal"] == "binary_cross_entropy_over_remaining_distance_thresholds"
    assert config.seeds == PRODUCTION_SEEDS
    assert leakage["authenticated_goal8_benchmark_manifest_required"] is True
    assert payload["phase_a"]["production_training"] is False


def test_ordinal_probabilities_are_cumulative_and_monotone() -> None:
    prediction = GoalConditionedValueHead(
        FixtureEncoder(),
        _head_config(),
        seed=20260726,
    ).predict_batch(["a"], ["b"])[0]

    probabilities = prediction.ordinal_probabilities
    assert all(
        probabilities[index] >= probabilities[index + 1] for index in range(len(probabilities) - 1)
    )
    with pytest.raises(ValueError, match="non-increasing"):
        ValuePredictionV1(1.0, (0.1, 0.9))


def test_hybrid_loss_uses_huber_and_right_censored_ordinal_targets() -> None:
    config = ValueTrainingConfigV1(head=_head_config())
    predictions = (
        ValuePredictionV1(remaining_witness_steps=0.0, ordinal_probabilities=(0.5,) * 4),
        ValuePredictionV1(remaining_witness_steps=4.0, ordinal_probabilities=(0.5,) * 4),
    )

    loss = hybrid_value_loss(predictions, (0, 10), config)

    assert loss.regression == 0.0
    assert loss.ordinal == pytest.approx(math.log(2.0))
    assert loss.total == pytest.approx(0.25 * math.log(2.0))
    assert loss.censored_count == 1
    assert loss.attempted_count == 2


def test_right_censored_regression_is_a_one_sided_lower_bound() -> None:
    config = ValueTrainingConfigV1(head=_head_config())

    at_bound = hybrid_value_loss(
        (ValuePredictionV1(4.0, (0.5,) * 4),),
        (10,),
        config,
    )
    above_bound = hybrid_value_loss(
        (ValuePredictionV1(10.0, (0.5,) * 4),),
        (10,),
        config,
    )
    below_bound = hybrid_value_loss(
        (ValuePredictionV1(2.0, (0.5,) * 4),),
        (10,),
        config,
    )

    assert at_bound.regression == 0.0
    assert above_bound.regression == 0.0
    assert below_bound.regression == pytest.approx(1.5)

    exact_at_boundary = hybrid_value_loss(
        (ValuePredictionV1(6.0, (0.5,) * 4),),
        (4,),
        config,
    )
    censored_above_boundary = hybrid_value_loss(
        (ValuePredictionV1(6.0, (0.5,) * 4),),
        (5,),
        config,
    )
    assert exact_at_boundary.regression == pytest.approx(1.5)
    assert exact_at_boundary.censored_count == 0
    assert censored_above_boundary.regression == 0.0
    assert censored_above_boundary.censored_count == 1


@dataclass(frozen=True)
class FixtureRow:
    record: str
    group: str
    partition: str
    current: str
    goal: str
    remaining: int | None
    relatives: tuple[str, ...] = ()
    replayed: bool = True
    row_status: str = ValueRowStatus.ACCEPTED
    failure: str | None = None


class FixtureRowAdapter:
    def record_id(self, row: FixtureRow) -> str:
        return row.record

    def group_id(self, row: FixtureRow) -> str:
        return row.group

    def related_group_ids(self, row: FixtureRow) -> Sequence[str]:
        return tuple(sorted({row.group, *row.relatives}))

    def record_digest(self, row: FixtureRow) -> str:
        return hashlib.sha256(repr(row).encode()).hexdigest()

    def split(self, row: FixtureRow) -> str:
        return row.partition

    def current_state(self, row: FixtureRow) -> Any:
        return row.current

    def goal_state(self, row: FixtureRow) -> Any:
        return row.goal

    def remaining_witness_steps(self, row: FixtureRow) -> int | None:
        return row.remaining

    def replay_verified(self, row: FixtureRow) -> bool:
        return row.replayed

    def status(self, row: FixtureRow) -> str:
        return row.row_status

    def failure_detail(self, row: FixtureRow) -> str | None:
        return row.failure


class FixtureTrainingHarness:
    def __init__(self, fail_seed: int | None = None) -> None:
        self.requests = []
        self.fail_seed = fail_seed

    def run_value_training_cell(self, request: Any) -> ValueSeedResultV1:
        self.requests.append(request)
        if request.seed == self.fail_seed:
            raise RuntimeError("fixture training failure")
        validation_mae = {
            20260726: 0.5,
            20260727: 0.2,
            20260728: 0.3,
        }[request.seed]
        return ValueSeedResultV1(
            seed=request.seed,
            status=ValueTrainingStatus.COMPLETE,
            validation_mae=validation_mae,
            checkpoint_digest=hashlib.sha256(str(request.seed).encode()).hexdigest(),
            epochs_completed=4,
            resumed=request.resume,
            request_digest=request.digest,
            resumed_from_checkpoint_digest=(
                hashlib.sha256(f"prior-{request.seed}".encode()).hexdigest()
                if request.resume
                else None
            ),
        )


def _training_rows() -> tuple[tuple[FixtureRow, ...], tuple[FixtureRow, ...]]:
    return (
        (
            FixtureRow("train-1", "train-group-1", "train", "a", "b", 2),
            FixtureRow("train-2", "train-group-2", "train", "b", "c", 1),
        ),
        (FixtureRow("valid-1", "valid-group", "validation", "a", "c", 3),),
    )


def _fixture_exclusions() -> FixtureBenchmarkExclusionsV1:
    return FixtureBenchmarkExclusionsV1.create(("benchmark-group",))


def _environment() -> ValueTrainingEnvironmentV1:
    return ValueTrainingEnvironmentV1(
        git_commit="fixture-commit",
        python_version="3.12.fixture",
        package_versions=(("geml", "fixture"),),
        hardware="fixture-cpu",
        exact_command="python -m fixture.value_train",
    )


def _production_environment() -> ValueTrainingEnvironmentV1:
    return ValueTrainingEnvironmentV1(
        git_commit="ab" * 20,
        python_version="3.12.4",
        package_versions=(
            ("geml", "0.1.0"),
            ("numpy", "2.3.0"),
            ("torch", "2.7.0"),
        ),
        hardware="NVIDIA H100 80GB",
        exact_command="python -m geml.learning.value.train --config configs/goal8_value.yaml",
    )


def _input_identity() -> ValueTrainingInputIdentityV1:
    return ValueTrainingInputIdentityV1(
        step_dataset_sha256=hashlib.sha256(b"fixture-steps").hexdigest(),
        split_manifest_sha256=hashlib.sha256(b"fixture-splits").hexdigest(),
        shared_encoder_checkpoint_sha256=FixtureEncoder.encoder_digest,
        shared_training_harness_sha256=hashlib.sha256(b"fixture-common-harness").hexdigest(),
    )


def test_training_delegates_exactly_three_seeds_and_retains_failure_rows() -> None:
    training, validation = _training_rows()
    harness = FixtureTrainingHarness(fail_seed=20260727)

    summary = run_three_seed_training(
        training_rows=training,
        validation_rows=validation,
        adapter=FixtureRowAdapter(),
        harness=harness,
        config=ValueTrainingConfigV1(head=_head_config()),
        environment=_environment(),
        input_identity=_input_identity(),
        benchmark_exclusions=_fixture_exclusions(),
    )

    assert tuple(request.seed for request in harness.requests) == PRODUCTION_SEEDS
    assert all(len(request.config_digest) == 64 for request in harness.requests)
    assert all(request.environment_digest == _environment().digest for request in harness.requests)
    assert len({request.digest for request in harness.requests}) == 3
    assert tuple(result.seed for result in summary.seed_results) == PRODUCTION_SEEDS
    assert summary.seed_results[1].status is ValueTrainingStatus.FAILED
    assert summary.seed_results[1].failure_reason == "fixture training failure"
    assert summary.seed_results[1].resumed is None
    assert summary.selected_seed == 20260728
    assert summary.selection_metric == "validation_mae"
    assert summary.benchmark_exclusion_file_sha256 is None
    assert summary.input_audit.attempted_count == 3
    assert summary.input_audit.included_count == 3


def test_training_resume_flag_is_strict() -> None:
    training, validation = _training_rows()
    harness = FixtureTrainingHarness()
    with pytest.raises(ValueTrainingConfigurationError, match="strict Boolean"):
        run_three_seed_training(
            training_rows=training,
            validation_rows=validation,
            adapter=FixtureRowAdapter(),
            harness=harness,
            config=ValueTrainingConfigV1(head=_head_config()),
            environment=_environment(),
            input_identity=_input_identity(),
            benchmark_exclusions=_fixture_exclusions(),
            resume=1,  # type: ignore[arg-type]
        )
    assert harness.requests == []


def test_production_refuses_missing_or_self_issued_fixture_exclusions() -> None:
    training, validation = _training_rows()
    common = {
        "training_rows": training,
        "validation_rows": validation,
        "adapter": FixtureRowAdapter(),
        "harness": FixtureTrainingHarness(),
        "environment": _production_environment(),
        "input_identity": _input_identity(),
    }
    production_config = ValueTrainingConfigV1(
        head=_head_config(),
        stage="production",
        benchmark_manifest_file_sha256="a" * 64,
        benchmark_manifest_content_sha256="b" * 64,
    )
    with pytest.raises(ValueTrainingConfigurationError, match="preregistered"):
        run_three_seed_training(
            **common,
            config=ValueTrainingConfigV1(head=_head_config(), stage="production"),
            benchmark_exclusions=_fixture_exclusions(),
        )
    with pytest.raises(ValueTrainingConfigurationError, match="required"):
        run_three_seed_training(
            **common,
            config=production_config,
            benchmark_exclusions=None,
        )
    forged_groups = FixtureBenchmarkExclusionsV1.create(
        tuple(f"claimed-production-group-{index:03d}" for index in range(256))
    )
    with pytest.raises(ValueTrainingConfigurationError, match="authenticated frozen"):
        run_three_seed_training(
            **common,
            config=production_config,
            benchmark_exclusions=forged_groups,
        )


def test_production_refuses_placeholder_environment_before_training() -> None:
    training, validation = _training_rows()
    harness = FixtureTrainingHarness()
    with pytest.raises(ValueError, match="unique"):
        ValueTrainingEnvironmentV1(
            git_commit="fixture",
            python_version="fixture",
            package_versions=(("numpy", "1"), ("numpy", "2")),
            hardware="fixture",
            exact_command="fixture",
        )
    production_environment = _production_environment()
    production_environment.require_production_ready()
    for invalid_commit in ("a" * 40, "AB" * 20):
        with pytest.raises(ValueTrainingConfigurationError, match="git_commit"):
            replace(
                production_environment,
                git_commit=invalid_commit,
            ).require_production_ready()
    with pytest.raises(ValueTrainingConfigurationError, match="package_versions"):
        replace(
            production_environment,
            package_versions=(
                ("geml", "0.1.0"),
                ("numpy", "unknown"),
                ("torch", "2.7.0"),
            ),
        ).require_production_ready()
    config = ValueTrainingConfigV1(
        head=_head_config(),
        stage="production",
        benchmark_manifest_file_sha256="a" * 64,
        benchmark_manifest_content_sha256="b" * 64,
    )

    with pytest.raises(ValueTrainingConfigurationError, match="git_commit"):
        run_three_seed_training(
            training_rows=training,
            validation_rows=validation,
            adapter=FixtureRowAdapter(),
            harness=harness,
            config=config,
            environment=_environment(),
            input_identity=_input_identity(),
            benchmark_exclusions=None,
        )
    assert harness.requests == []


def test_benchmark_manifest_reference_requires_exact_authenticated_bytes(
    tmp_path: Path,
) -> None:
    training, validation = _training_rows()
    source = tmp_path / "manifest.json"
    source.write_text("{}\n", encoding="utf-8")
    wrong_digest = "0" * 64

    with pytest.raises(ValueTrainingConfigurationError, match="authentication failed"):
        run_three_seed_training(
            training_rows=training,
            validation_rows=validation,
            adapter=FixtureRowAdapter(),
            harness=FixtureTrainingHarness(),
            config=ValueTrainingConfigV1(head=_head_config()),
            environment=_environment(),
            input_identity=_input_identity(),
            benchmark_exclusions=FrozenBenchmarkReferenceV1(
                path=str(source),
                expected_file_sha256=wrong_digest,
            ),
        )


def test_benchmark_manifest_is_reauthenticated_after_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts = iter((object(), object()))
    monkeypatch.setattr(
        value_train,
        "verify_frozen_manifest",
        lambda *_args, **_kwargs: next(receipts),
    )
    monkeypatch.setattr(
        value_train,
        "load_benchmark_manifest",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(ValueTrainingConfigurationError, match="changed while"):
        value_train._resolve_benchmark_exclusions(
            FrozenBenchmarkReferenceV1(
                path="fixture-benchmark.json",
                expected_file_sha256="a" * 64,
            ),
            config=ValueTrainingConfigV1(head=_head_config()),
        )


def test_training_refuses_an_encoder_identity_mismatch() -> None:
    training, validation = _training_rows()
    mismatched = ValueTrainingInputIdentityV1(
        step_dataset_sha256=hashlib.sha256(b"fixture-steps").hexdigest(),
        split_manifest_sha256=hashlib.sha256(b"fixture-splits").hexdigest(),
        shared_encoder_checkpoint_sha256="0" * 64,
        shared_training_harness_sha256=hashlib.sha256(b"fixture-common-harness").hexdigest(),
    )

    with pytest.raises(ValueTrainingConfigurationError, match="encoder does not match"):
        run_three_seed_training(
            training_rows=training,
            validation_rows=validation,
            adapter=FixtureRowAdapter(),
            harness=FixtureTrainingHarness(),
            config=ValueTrainingConfigV1(head=_head_config()),
            environment=_environment(),
            input_identity=mismatched,
            benchmark_exclusions=_fixture_exclusions(),
        )


@pytest.mark.parametrize("leak_split", ["train", "validation"])
def test_benchmark_groups_cannot_enter_training_or_model_selection(
    leak_split: str,
) -> None:
    training, validation = _training_rows()
    if leak_split == "train":
        training = (FixtureRow("leak", "benchmark-group", "train", "a", "b", 1),)
    else:
        validation = (FixtureRow("leak", "benchmark-group", "validation", "a", "b", 1),)

    with pytest.raises(ValueTrainingConfigurationError, match="leaked"):
        run_three_seed_training(
            training_rows=training,
            validation_rows=validation,
            adapter=FixtureRowAdapter(),
            harness=FixtureTrainingHarness(),
            config=ValueTrainingConfigV1(head=_head_config()),
            environment=_environment(),
            input_identity=_input_identity(),
            benchmark_exclusions=_fixture_exclusions(),
        )


def test_benchmark_and_cross_split_relative_groups_are_rejected() -> None:
    training, validation = _training_rows()
    leaked_training = (
        FixtureRow(
            "relative-leak",
            "different-primary",
            "train",
            "a",
            "b",
            1,
            relatives=("benchmark-group",),
        ),
    )
    with pytest.raises(ValueTrainingConfigurationError, match="relative leaked"):
        run_three_seed_training(
            training_rows=leaked_training,
            validation_rows=validation,
            adapter=FixtureRowAdapter(),
            harness=FixtureTrainingHarness(),
            config=ValueTrainingConfigV1(head=_head_config()),
            environment=_environment(),
            input_identity=_input_identity(),
            benchmark_exclusions=_fixture_exclusions(),
        )

    cross_validation = (
        FixtureRow(
            "relative-cross",
            "different-validation-primary",
            "validation",
            "a",
            "b",
            1,
            relatives=("train-group-1",),
        ),
    )
    with pytest.raises(ValueTrainingConfigurationError, match="cross train/validation"):
        run_three_seed_training(
            training_rows=training,
            validation_rows=cross_validation,
            adapter=FixtureRowAdapter(),
            harness=FixtureTrainingHarness(),
            config=ValueTrainingConfigV1(head=_head_config()),
            environment=_environment(),
            input_identity=_input_identity(),
            benchmark_exclusions=_fixture_exclusions(),
        )


def test_training_retains_and_excludes_source_failure_rows() -> None:
    training, validation = _training_rows()
    training = (
        *training,
        FixtureRow(
            "unsupported-train",
            "unsupported-group",
            "train",
            "a",
            "b",
            None,
            row_status=ValueRowStatus.UNSUPPORTED,
            failure="operator unavailable",
        ),
    )
    harness = FixtureTrainingHarness()

    summary = run_three_seed_training(
        training_rows=training,
        validation_rows=validation,
        adapter=FixtureRowAdapter(),
        harness=harness,
        config=ValueTrainingConfigV1(head=_head_config()),
        environment=_environment(),
        input_identity=_input_identity(),
        benchmark_exclusions=_fixture_exclusions(),
    )

    assert summary.input_audit.attempted_count == 4
    assert summary.input_audit.included_count == 3
    assert summary.input_audit.unsupported_count == 1
    assert summary.input_audit.failure_count == 0
    retained = next(row for row in summary.input_audit.rows if row.record_id == "unsupported-train")
    assert retained.status is ValueRowStatus.UNSUPPORTED
    assert retained.failure_reason == "operator unavailable"
    assert all(len(request.training_rows) == 2 for request in harness.requests)


@pytest.mark.parametrize(
    ("bad_request_digest", "epochs", "message"),
    [
        ("0" * 64, 1, "not bound"),
        (None, 101, "maximum_epochs"),
    ],
)
def test_training_rejects_unbound_or_overbudget_checkpoint_results(
    bad_request_digest: str | None,
    epochs: int,
    message: str,
) -> None:
    class InvalidResultHarness:
        def run_value_training_cell(self, request: Any) -> ValueSeedResultV1:
            return ValueSeedResultV1(
                seed=request.seed,
                status=ValueTrainingStatus.COMPLETE,
                validation_mae=0.1,
                checkpoint_digest=hashlib.sha256(b"checkpoint").hexdigest(),
                epochs_completed=epochs,
                resumed=False,
                request_digest=bad_request_digest or request.digest,
            )

    training, validation = _training_rows()
    summary = run_three_seed_training(
        training_rows=training,
        validation_rows=validation,
        adapter=FixtureRowAdapter(),
        harness=InvalidResultHarness(),
        config=ValueTrainingConfigV1(head=_head_config()),
        environment=_environment(),
        input_identity=_input_identity(),
        benchmark_exclusions=_fixture_exclusions(),
    )

    assert summary.selected_seed is None
    assert all(result.status is ValueTrainingStatus.FAILED for result in summary.seed_results)
    assert all(message in result.failure_reason for result in summary.seed_results)
    assert all(result.resumed is None for result in summary.seed_results)


def test_seed_result_requires_strict_resume_truth() -> None:
    with pytest.raises(ValueError, match="Boolean or None"):
        ValueSeedResultV1(
            seed=PRODUCTION_SEEDS[0],
            status=ValueTrainingStatus.COMPLETE,
            validation_mae=0.1,
            checkpoint_digest=hashlib.sha256(b"checkpoint").hexdigest(),
            epochs_completed=1,
            resumed=1,  # type: ignore[arg-type]
            request_digest=hashlib.sha256(b"request").hexdigest(),
        )


def test_metrics_retain_unsupported_and_failed_denominators() -> None:
    rows = (
        FixtureRow("r0", "g0", "test", "a", "b", 0),
        FixtureRow("r1", "g1", "test", "a", "b", 1),
        FixtureRow("r3", "g3", "test", "a", "b", 3),
        FixtureRow("r8", "g8", "test", "a", "b", 8),
        FixtureRow(
            "unsupported",
            "gu",
            "test",
            "a",
            "b",
            2,
            row_status=ValueRowStatus.UNSUPPORTED,
        ),
        FixtureRow("failed", "gf", "test", "a", "b", 2, replayed=False),
    )
    target_by_current_call = iter(((1.0, 2.0, 4.0, 9.0),))

    report = evaluate_value_head(
        rows=rows,
        adapter=FixtureRowAdapter(),
        score_batch=lambda current, goal: next(target_by_current_call),
    )

    assert report.metrics.attempted_count == 6
    assert report.metrics.valid_count == 4
    assert report.metrics.unsupported_count == 1
    assert report.metrics.failure_count == 1
    assert report.metrics.mean_absolute_error == 1.0
    assert report.metrics.spearman_rank_correlation == pytest.approx(1.0)
    assert tuple(tier.count for tier in report.metrics.calibration_by_witness_tier) == (
        1,
        1,
        1,
        0,
        1,
    )
    two_to_three = report.metrics.calibration_by_witness_tier[2]
    assert two_to_three.attempted_count == 3
    assert two_to_three.valid_count == 1
    assert two_to_three.unsupported_count == 1
    assert two_to_three.failure_count == 1
    assert report.rows[-2].prediction is None
    assert report.rows[-1].status is ValueRowStatus.REPLAY_FAILED


def test_scoring_failure_is_retained_for_every_valid_attempt() -> None:
    rows = (
        FixtureRow("r1", "g1", "test", "a", "b", 1),
        FixtureRow("r2", "g2", "test", "a", "b", 2),
    )

    def fail(current: Sequence[Any], goal: Sequence[Any]) -> Sequence[float]:
        del current, goal
        raise RuntimeError("model unavailable")

    report = evaluate_value_head(
        rows=rows,
        adapter=FixtureRowAdapter(),
        score_batch=fail,
    )

    assert report.metrics.valid_count == 0
    assert report.metrics.failure_count == 2
    assert all(row.status is ValueRowStatus.SCORING_ERROR for row in report.rows)
    assert all(row.failure_reason == "model unavailable" for row in report.rows)


def test_evaluation_retains_rows_without_distance_targets() -> None:
    rows = (
        FixtureRow("valid", "g-valid", "test", "a", "b", 1),
        FixtureRow(
            "unsupported",
            "g-unsupported",
            "test",
            "a",
            "b",
            None,
            row_status=ValueRowStatus.UNSUPPORTED,
            failure="unsupported operator",
        ),
        FixtureRow(
            "invalid",
            "g-invalid",
            "test",
            "a",
            "b",
            None,
            row_status=ValueRowStatus.INVALID,
            failure="invalid witness",
        ),
    )

    report = evaluate_value_head(
        rows=rows,
        adapter=FixtureRowAdapter(),
        score_batch=lambda current, goal: (1.0,),
    )

    assert report.metrics.attempted_count == 3
    assert report.metrics.valid_count == 1
    assert report.metrics.unsupported_count == 1
    assert report.metrics.failure_count == 1
    assert report.metrics.unstratified_count == 2
    assert report.rows[1].failure_reason == "unsupported operator"
    assert report.rows[2].failure_reason == "invalid witness"


def test_nonfinite_encoder_output_is_rejected() -> None:
    class InvalidEncoder(FixtureEncoder):
        def encode_batch(
            self,
            states: Sequence[str],
            *,
            training: bool,
        ) -> Sequence[Sequence[float]]:
            del states, training
            return ((np.nan, 0.0),)

    head = GoalConditionedValueHead(InvalidEncoder(), _head_config(), seed=20260726)
    with pytest.raises(ValueError, match="non-finite"):
        head.predict_batch(["a"], ["b"])
