"""Contract tests for the phase-A compact-model training harness."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from geml.learning.harness.config import (
    DynamicBatchConfig,
    OutcomeCountsV1,
    PrecisionMode,
    RunEnvelopeV1,
    TrainingConfigV1,
    TrainingStatus,
    sha256_digest,
)
from geml.learning.harness.seeds import seed_everything
from geml.learning.harness.train import BatchCost, dynamic_budget_batches, train_resumable

try:
    import torch
    from torch import nn
    from torch.nn import functional
except ImportError:  # pragma: no cover - core-only environment.
    torch = None
    nn = None
    functional = None


def _config(tmp_path: Path) -> TrainingConfigV1:
    return TrainingConfigV1(
        run_id="fixture-run",
        seed=20260726,
        epochs=3,
        optimizer="sgd",
        learning_rate=0.05,
        weight_decay=0.0,
        precision=PrecisionMode.FLOAT32,
        deterministic_algorithms_requested=True,
        dynamic_batch=DynamicBatchConfig(node_budget=8, edge_budget=16),
        checkpoint_path=str(tmp_path / "latest.pt"),
        result_path=str(tmp_path / "results.json"),
    )


def test_dynamic_budget_batches_preserve_order_and_oversized_items() -> None:
    batches = dynamic_budget_batches(
        (
            BatchCost("a", nodes=3, edges=2),
            BatchCost("b", nodes=7, edges=1),
            BatchCost("c", nodes=2, edges=12),
        ),
        node_budget=8,
        edge_budget=8,
    )

    assert [[item.item_id for item in batch] for batch in batches] == [["a"], ["b"], ["c"]]


def test_training_config_is_content_addressed_and_outcomes_are_complete(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.config_hash == _config(tmp_path).config_hash
    assert config.config_hash.startswith("sha256:")
    assert (
        OutcomeCountsV1(
            attempted=4,
            successful=1,
            failed=1,
            unsupported=1,
            invalid=0,
            timeout=1,
        ).attempted
        == 4
    )
    with pytest.raises(ValidationError, match="complete terminal-outcome denominator"):
        OutcomeCountsV1(
            attempted=1,
            successful=1,
            failed=1,
            unsupported=0,
            invalid=0,
            timeout=0,
        )


def test_run_envelope_binds_configuration_content() -> None:
    configuration = {"arm": "ast", "seed": 20260726}
    envelope = RunEnvelopeV1(
        configuration=configuration,
        configuration_hash=sha256_digest(configuration),
        git_commit="998a139b09d232db5ed4ef4222d1e0dc778d3542",
        package_versions={"geml": "0.1.0"},
        seeds=(20260726,),
        hardware={"accelerator": "cpu"},
        precision=PrecisionMode.FLOAT32,
        deterministic_settings={"requested": True},
        input_checksums={"fixture": "sha256:input"},
        output_checksums={"fixture": "sha256:output"},
        started_at="2026-07-27T00:00:00+00:00",
        resource_telemetry={"peak_memory_bytes": 0},
        outcomes=OutcomeCountsV1(
            attempted=1,
            successful=1,
            failed=0,
            unsupported=0,
            invalid=0,
            timeout=0,
        ),
        resume_lineage={"resumed": False},
        reproduction_command="python -m pytest tests/learning/test_harness.py",
    )

    assert envelope.configuration_hash == sha256_digest(configuration)
    invalid = envelope.model_dump()
    invalid["configuration_hash"] = "sha256:not-the-configuration"
    with pytest.raises(ValidationError, match="configuration_hash"):
        RunEnvelopeV1.model_validate(invalid)


def test_seed_everything_replays_python_and_numpy_sequences() -> None:
    seed_everything(17, deterministic_algorithms=True)
    first = (random.random(), float(np.random.rand()))
    seed_everything(17, deterministic_algorithms=True)
    second = (random.random(), float(np.random.rand()))

    assert first == second


@pytest.mark.skipif(torch is None, reason="optional [ml] extra is not installed")
def test_resume_matches_uninterrupted_state_and_retains_failed_attempt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    inputs = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    targets = torch.tensor([[2.0], [4.0], [6.0]], dtype=torch.float32)
    batches = ((inputs, targets),)

    torch.manual_seed(91)
    template = nn.Sequential(nn.Linear(1, 4), nn.Dropout(p=0.25), nn.Linear(4, 1))
    initial_state = copy.deepcopy(template.state_dict())

    def make_model() -> nn.Module:
        model = nn.Sequential(nn.Linear(1, 4), nn.Dropout(p=0.25), nn.Linear(4, 1))
        model.load_state_dict(copy.deepcopy(initial_state))
        return model

    def run(model: nn.Module, *, resume: bool, observer=None):
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

        def step(batch, *, training: bool):
            features, labels = batch
            return functional.mse_loss(model(features), labels)

        return train_resumable(
            config=config,
            model=model,
            optimizer=optimizer,
            train_batches=batches,
            validation_batches=batches,
            step=step,
            resume=resume,
            on_checkpoint=observer,
        )

    uninterrupted = run(make_model(), resume=False)
    latest = tmp_path / "latest.pt"
    best = tmp_path / "latest.best.pt"
    latest.unlink()
    best.unlink()

    def stop_after_first_epoch(metric) -> None:
        if metric.epoch == 1:
            raise RuntimeError("injected supervisor interruption")

    with pytest.raises(RuntimeError, match="injected supervisor interruption"):
        run(make_model(), resume=False, observer=stop_after_first_epoch)
    resumed = run(make_model(), resume=True)

    assert resumed.resumed is True
    assert resumed.state_digest == uninterrupted.state_digest
    assert [(m.epoch, m.train_loss, m.validation_loss) for m in resumed.metrics] == [
        (m.epoch, m.train_loss, m.validation_loss) for m in uninterrupted.metrics
    ]
    evidence = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert [attempt["status"] for attempt in evidence["attempts"]] == [
        TrainingStatus.COMPLETE.value,
        TrainingStatus.FAILED.value,
        TrainingStatus.COMPLETE.value,
    ]
    assert evidence["attempts"][1]["checkpoint_digest"] is not None
