"""Issue 6-4 tests use only tiny hand-written fixtures and temporary directories.

The determinism and resume-equivalence tests deliberately avoid PyTorch so that they run on a
core-only clone; the harness contract they exercise is the same one the ML cells rely on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from geml.learning.harness.config import (
    DEFAULT_MAX_EPOCHS,
    HARNESS_CONFIG_SCHEMA_VERSION,
    BatchBudget,
    ComparisonUnit,
    HarnessConfig,
    HarnessConfigError,
    Precision,
    RunEnvelope,
    config_digest,
    load_harness_config,
)
from geml.learning.harness.seeds import (
    PRODUCTION_SEEDS,
    SeedPolicyError,
    derive_child_seed,
    require_production_seed,
    seed_everything,
    worker_seed,
)
from geml.learning.harness.train import (
    BatchPlan,
    Cell,
    CellScheduler,
    CellStatus,
    CheckpointMismatchError,
    HarnessError,
    TrainableTask,
    TrainingHarness,
    atomic_write_json,
    epoch_order,
    plan_batches,
    resolve_precision,
    sha256_file,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = REPOSITORY_ROOT / "configs" / "goal6_harness_defaults.yaml"


class CountingTask:
    """A deterministic arithmetic stand-in for a real training task.

    The "loss" is a pure function of the epoch and the batch contents, so an uninterrupted run and
    an interrupted-then-resumed run must agree exactly, with no floating-point excuse available.
    """

    def __init__(self, count: int = 12, failing_epoch: int | None = None) -> None:
        self.count = count
        self.failing_epoch = failing_epoch
        self.value = 0.0
        self.steps = 0

    def example_count(self, split: str) -> int:
        return self.count if split == "train" else max(1, self.count // 2)

    def example_cost(self, split: str, index: int) -> tuple[int, int]:
        return (index % 4 + 1, index % 3 + 1)

    def train_step(self, batch: BatchPlan, epoch: int) -> float:
        if self.failing_epoch is not None and epoch == self.failing_epoch:
            raise RuntimeError("injected training failure")
        self.steps += 1
        self.value += sum(batch.indices) / 100.0
        return float(len(batch.indices) + epoch)

    def evaluate(self, split: str, batches: list[BatchPlan]) -> dict[str, float]:
        return {"validation_loss": 1.0 / (1.0 + self.steps), "accuracy": 0.5}

    def state_dict(self) -> dict[str, object]:
        return {"value": self.value, "steps": self.steps}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.value = float(state["value"])
        self.steps = int(state["steps"])

    def parameter_count(self) -> int:
        return 1234


def make_config(tmp_path: Path, **overrides: object) -> HarnessConfig:
    payload: dict[str, object] = {
        "run_name": "fixture-cell",
        "output_root": str(tmp_path),
        "seed": PRODUCTION_SEEDS[0],
        "max_epochs": 4,
    }
    payload.update(overrides)
    return HarnessConfig.model_validate(payload)


def test_shipped_defaults_file_validates_against_the_frozen_schema() -> None:
    payload = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))
    config = load_harness_config(payload)
    assert config.schema_version == HARNESS_CONFIG_SCHEMA_VERSION
    assert config.seed in PRODUCTION_SEEDS
    assert config.max_epochs <= DEFAULT_MAX_EPOCHS
    assert config.compute_match.comparison_unit is ComparisonUnit.PER_OPTIMIZER_STEP
    assert config.compute_match.vocabulary_embeddings_count_toward_match is True


def test_unknown_configuration_keys_are_rejected() -> None:
    with pytest.raises(HarnessConfigError, match=r"extra_inputs_not_permitted|Extra inputs"):
        load_harness_config(
            {
                "run_name": "x",
                "output_root": "out",
                "seed": PRODUCTION_SEEDS[0],
                "learnign_rate": 0.1,
            }
        )


def test_non_production_seed_is_refused_by_default() -> None:
    with pytest.raises(HarnessConfigError, match="preregistered production seeds"):
        load_harness_config({"run_name": "x", "output_root": "out", "seed": 12345})


def test_fixture_runs_may_opt_out_of_the_production_seed_policy() -> None:
    config = load_harness_config(
        {"run_name": "x", "output_root": "out", "seed": 7, "require_production_seed": False}
    )
    assert config.seed == 7


def test_epoch_budget_above_the_cap_requires_a_documented_reason() -> None:
    with pytest.raises(HarnessConfigError, match="max_epochs_exception_reason"):
        load_harness_config(
            {
                "run_name": "x",
                "output_root": "out",
                "seed": PRODUCTION_SEEDS[0],
                "max_epochs": 60,
            }
        )
    approved = load_harness_config(
        {
            "run_name": "x",
            "output_root": "out",
            "seed": PRODUCTION_SEEDS[0],
            "max_epochs": 60,
            "max_epochs_exception_reason": "approved in the issue thread for the pilot",
        }
    )
    assert approved.max_epochs == 60


def test_config_hash_is_stable_and_sensitive(tmp_path: Path) -> None:
    first = make_config(tmp_path)
    second = make_config(tmp_path)
    assert config_digest(first) == config_digest(second) == first.config_hash
    changed = make_config(tmp_path, seed=PRODUCTION_SEEDS[1])
    assert config_digest(changed) != config_digest(first)


def test_config_hash_ignores_key_order(tmp_path: Path) -> None:
    forward = HarnessConfig.model_validate(
        {"run_name": "a", "output_root": str(tmp_path), "seed": PRODUCTION_SEEDS[0]}
    )
    reversed_keys = HarnessConfig.model_validate(
        {"seed": PRODUCTION_SEEDS[0], "output_root": str(tmp_path), "run_name": "a"}
    )
    assert config_digest(forward) == config_digest(reversed_keys)


def test_production_seeds_are_exactly_the_preregistered_three() -> None:
    assert PRODUCTION_SEEDS == (20260726, 20260727, 20260728)
    assert require_production_seed(20260727) == 20260727
    with pytest.raises(SeedPolicyError, match="not a preregistered production seed"):
        require_production_seed(20260729)


def test_child_seed_derivation_is_stable_and_label_sensitive() -> None:
    base = derive_child_seed(PRODUCTION_SEEDS[0], "epoch", "3")
    assert base == derive_child_seed(PRODUCTION_SEEDS[0], "epoch", "3")
    assert base != derive_child_seed(PRODUCTION_SEEDS[0], "epoch", "4")
    assert base != derive_child_seed(PRODUCTION_SEEDS[1], "epoch", "3")
    assert 0 <= base < 2**63
    assert worker_seed(PRODUCTION_SEEDS[0], 0) != worker_seed(PRODUCTION_SEEDS[0], 1)


def test_child_seed_derivation_rejects_degenerate_labels() -> None:
    with pytest.raises(SeedPolicyError, match="at least one label"):
        derive_child_seed(1)
    with pytest.raises(SeedPolicyError, match="NUL"):
        derive_child_seed(1, "bad\0label")
    with pytest.raises(SeedPolicyError, match="non-negative"):
        worker_seed(1, -1)


def test_seed_everything_reports_what_it_could_not_control() -> None:
    report = seed_everything(PRODUCTION_SEEDS[0])
    payload = report.as_dict()
    assert payload["root_seed"] == PRODUCTION_SEEDS[0]
    assert isinstance(payload["unsupported"], list)


def test_epoch_order_is_deterministic_and_epoch_specific() -> None:
    first = epoch_order(10, PRODUCTION_SEEDS[0], 0)
    assert first == epoch_order(10, PRODUCTION_SEEDS[0], 0)
    assert first != epoch_order(10, PRODUCTION_SEEDS[0], 1)
    assert sorted(first) == list(range(10))


def test_batches_respect_node_edge_and_example_budgets() -> None:
    task = CountingTask(count=10)
    batches = plan_batches(task, "train", list(range(10)), 6, 6, 3)
    assert sum(len(batch.indices) for batch in batches) == 10
    assert all(len(batch.indices) <= 3 for batch in batches)
    covered = [index for batch in batches for index in batch.indices]
    assert sorted(covered) == list(range(10))


def test_oversized_example_still_forms_its_own_batch() -> None:
    """Dropping an over-budget example would silently change the denominator."""

    task = CountingTask(count=4)
    batches = plan_batches(task, "train", list(range(4)), 1, 1, 8)
    covered = [index for batch in batches for index in batch.indices]
    assert sorted(covered) == [0, 1, 2, 3]


def test_training_run_produces_a_complete_result_and_checkpoints(tmp_path: Path) -> None:
    harness = TrainingHarness(make_config(tmp_path), CountingTask())
    result = harness.run()
    assert result.status is CellStatus.COMPLETE
    assert result.epochs_completed == 4
    assert result.parameters == 1234
    assert harness.latest_checkpoint_path.exists()
    assert harness.best_checkpoint_path.exists()
    assert harness.result_path.exists()

    payload = json.loads(harness.result_path.read_text(encoding="utf-8"))
    assert payload["config_hash"] == harness.config_hash
    assert len(payload["telemetry"]["train_loss_by_epoch"]) == 4


def test_interrupted_run_resumes_to_the_same_result(tmp_path: Path) -> None:
    """The acceptance criterion for issue 6-4: resume equals uninterrupted."""

    uninterrupted = TrainingHarness(make_config(tmp_path / "a"), CountingTask()).run()

    partial_harness = TrainingHarness(make_config(tmp_path / "b"), CountingTask())
    partial_harness.run(max_epochs=2)
    resumed = TrainingHarness(make_config(tmp_path / "b"), CountingTask()).run()

    assert resumed.status is CellStatus.COMPLETE
    assert resumed.epochs_completed == uninterrupted.epochs_completed
    assert resumed.telemetry.train_loss_by_epoch == uninterrupted.telemetry.train_loss_by_epoch
    assert resumed.best_criterion_value == uninterrupted.best_criterion_value
    assert resumed.best_epoch == uninterrupted.best_epoch
    assert resumed.resume_lineage, "a resumed run must record its checkpoint lineage"


def test_resume_refuses_a_mismatched_configuration(tmp_path: Path) -> None:
    TrainingHarness(make_config(tmp_path), CountingTask()).run(max_epochs=1)
    other = make_config(tmp_path, seed=PRODUCTION_SEEDS[1])
    harness = TrainingHarness(other, CountingTask())
    harness.output_directory.mkdir(parents=True, exist_ok=True)
    with pytest.raises(CheckpointMismatchError, match="config_hash"):
        harness.load_checkpoint(Path(tmp_path) / "fixture-cell" / "checkpoint.latest.json")


def test_training_failure_becomes_a_row_rather_than_a_crash(tmp_path: Path) -> None:
    harness = TrainingHarness(make_config(tmp_path), CountingTask(failing_epoch=1))
    result = harness.run()
    assert result.status is CellStatus.FAILED
    assert result.failure_reason is not None
    assert "injected training failure" in result.failure_reason
    assert harness.result_path.exists()
    assert result.telemetry.failed_epochs == 1


def test_missing_early_stopping_criterion_is_reported(tmp_path: Path) -> None:
    class NoCriterionTask(CountingTask):
        def evaluate(self, split: str, batches: list[BatchPlan]) -> dict[str, float]:
            return {"accuracy": 1.0}

    result = TrainingHarness(make_config(tmp_path), NoCriterionTask()).run()
    assert result.status is CellStatus.FAILED
    assert "validation_loss" in str(result.failure_reason)


def test_counting_task_satisfies_the_trainable_protocol() -> None:
    assert isinstance(CountingTask(), TrainableTask)


def test_atomic_write_replaces_without_truncating(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "payload.json"
    atomic_write_json(target, {"b": 2, "a": 1})
    assert target.read_text(encoding="utf-8") == '{"a":1,"b":2}\n'
    atomic_write_json(target, {"a": 3})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 3}
    assert not list(target.parent.glob(".*tmp*"))
    assert len(sha256_file(target)) == 64


def test_scheduler_resolves_each_cell_status(tmp_path: Path) -> None:
    scheduler = CellScheduler(tmp_path)
    cell = Cell(cell_id="arm-a-seed-0", config=make_config(tmp_path))
    assert scheduler.plan([cell]) == [(cell, "run")]

    def write(status: CellStatus) -> None:
        atomic_write_json(
            scheduler.result_path(cell),
            {"status": str(status), "config_hash": config_digest(cell.config)},
        )

    write(CellStatus.COMPLETE)
    assert scheduler.plan([cell])[0][1] == "skip_and_verify"
    write(CellStatus.INCOMPLETE)
    assert scheduler.plan([cell])[0][1] == "resume"
    write(CellStatus.FAILED)
    assert scheduler.plan([cell])[0][1] == "retain_failure_requires_explicit_retry"
    assert list(scheduler.iter_runnable([cell])) == []


def test_scheduler_refuses_to_overwrite_a_different_configuration(tmp_path: Path) -> None:
    scheduler = CellScheduler(tmp_path)
    cell = Cell(cell_id="arm-a-seed-0", config=make_config(tmp_path))
    atomic_write_json(
        scheduler.result_path(cell),
        {"status": "complete", "config_hash": "0" * 64},
    )
    with pytest.raises(HarnessError, match="refusing to overwrite"):
        scheduler.existing_status(cell)


def test_cell_content_hash_is_identity_sensitive(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert Cell("a", config).content_hash != Cell("b", config).content_hash


def test_run_envelope_rejects_impossible_denominators() -> None:
    payload = {
        "config_hash": "a" * 64,
        "git_commit": "41e0dc8cc6315fcc5593355a38c7e317d88eeafe",
        "python_version": "3.12.11",
        "package_versions": {"torch": "2.6.0"},
        "seed": PRODUCTION_SEEDS[0],
        "device": "cpu",
        "precision": Precision.FLOAT32,
        "deterministic_algorithms": True,
        "started_at": "2026-07-26T00:00:00Z",
        "attempted": 2,
        "succeeded": 3,
        "reproduction_command": "python -m geml.experiments.goal6.run_grid",
    }
    with pytest.raises(ValueError, match="exceed attempted"):
        RunEnvelope.model_validate(payload)

    payload["succeeded"] = 1
    payload["failed"] = 1
    assert RunEnvelope.model_validate(payload).attempted == 2


def test_batch_budget_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        BatchBudget(max_nodes_per_batch=0)


def test_precision_is_resolved_and_recorded(tmp_path: Path) -> None:
    result = TrainingHarness(make_config(tmp_path), CountingTask()).run()
    assert result.precision is not None
    assert result.precision.effective == str(Precision.FLOAT32)
    assert json.loads(harness_result_text(tmp_path))["precision"]["effective"] == "float32"


def harness_result_text(tmp_path: Path) -> str:
    return (Path(tmp_path) / "fixture-cell" / "result.json").read_text(encoding="utf-8")


def test_bfloat16_falls_back_on_an_unsupported_device_with_a_reason(tmp_path: Path) -> None:
    config = make_config(tmp_path, precision="bfloat16", device="cuda:0")
    resolved = resolve_precision(config)
    assert resolved.requested == str(Precision.BFLOAT16)
    if resolved.effective == str(Precision.FLOAT32):
        assert resolved.fallback_reason
