"""Task-neutral seeded, resumable compact-model training utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from geml.learning.harness.config import TrainingConfigV1, TrainingStatus
from geml.learning.harness.seeds import seed_everything

try:  # No torch dependency for core imports or configuration parsing.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - optional-ML path.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


CHECKPOINT_SCHEMA_VERSION = "geml-training-checkpoint-v1"
RESULTS_SCHEMA_VERSION = "geml-training-results-v1"


class HarnessError(RuntimeError):
    """Raised for unsafe resume state, invalid dynamic batches, or persistence failures."""


@dataclass(frozen=True, slots=True)
class BatchCost:
    """Task-neutral size accounting used for dynamic node/edge-budget batching."""

    item_id: str
    nodes: int
    edges: int

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id must be nonblank")
        if self.nodes < 0 or self.edges < 0:
            raise ValueError("nodes and edges must be nonnegative")


def dynamic_budget_batches(
    costs: Sequence[BatchCost],
    *,
    node_budget: int,
    edge_budget: int,
) -> tuple[tuple[BatchCost, ...], ...]:
    """Create deterministic consecutive batches without dropping oversized items."""

    if node_budget < 1 or edge_budget < 1:
        raise ValueError("node_budget and edge_budget must be positive")
    batches: list[tuple[BatchCost, ...]] = []
    current: list[BatchCost] = []
    nodes = 0
    edges = 0
    for cost in costs:
        would_exceed = current and (
            nodes + cost.nodes > node_budget or edges + cost.edges > edge_budget
        )
        if would_exceed:
            batches.append(tuple(current))
            current = []
            nodes = 0
            edges = 0
        current.append(cost)
        nodes += cost.nodes
        edges += cost.edges
    if current:
        batches.append(tuple(current))
    return tuple(batches)


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """One retained train/validation observation."""

    epoch: int
    train_loss: float
    validation_loss: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Terminal outcome stored beside checkpoint/resume evidence."""

    config_hash: str
    status: TrainingStatus
    best_epoch: int
    metrics: tuple[EpochMetrics, ...]
    resumed: bool
    checkpoint_digest: str | None
    best_checkpoint_digest: str | None
    state_digest: str | None


class TrainingStep(Protocol):
    """Task callback that returns a scalar differentiable loss for a supplied batch."""

    def __call__(self, batch: Any, *, training: bool) -> Tensor: ...


CheckpointObserver = Callable[[EpochMetrics], None]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: object) -> None:
    """Write a checkpoint atomically, leaving the preceding checkpoint recoverable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require_torch() -> None:
    if torch is None:
        raise HarnessError("install GEML with `.[ml]` before running a training cell")


def _checkpoint_payload(
    *,
    config: TrainingConfigV1,
    epoch: int,
    best_epoch: int,
    best_validation: float,
    patience_used: int,
    metrics: list[EpochMetrics],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_hash": config.config_hash,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "patience_used": patience_used,
        "metrics": [asdict(metric) for metric in metrics],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_random_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _load_checkpoint(
    path: Path,
    *,
    config: TrainingConfigV1,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int, float, int, list[EpochMetrics]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise HarnessError("checkpoint schema is unsupported")
    if payload.get("config_hash") != config.config_hash:
        raise HarnessError(
            "checkpoint config hash does not match the active training configuration"
        )
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_rng_state"])
    cuda_rng_state_all = payload.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state_all)
    metric_rows = payload.get("metrics")
    if not isinstance(metric_rows, list):
        raise HarnessError("checkpoint metrics are malformed")
    metrics = [EpochMetrics(**row) for row in metric_rows]
    return (
        int(payload["epoch"]),
        int(payload["best_epoch"]),
        float(payload["best_validation"]),
        int(payload["patience_used"]),
        metrics,
    )


def _best_checkpoint_path(latest_checkpoint_path: Path) -> Path:
    """Derive the paired best checkpoint without changing frozen configuration identity."""

    return latest_checkpoint_path.with_name(
        f"{latest_checkpoint_path.stem}.best{latest_checkpoint_path.suffix}"
    )


def _digest_value(digest: Any, value: object) -> None:
    """Hash state recursively without process-local object identities or wall telemetry."""

    if torch is not None and isinstance(value, torch.Tensor):
        digest.update(b"tensor\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        raw_bytes = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(raw_bytes)
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=repr):
            _digest_value(digest, key)
            _digest_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _digest_value(digest, item)
    elif value is None or isinstance(value, str | int | float | bool):
        digest.update(b"scalar\0")
        digest.update(
            json.dumps(
                value,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
    else:
        raise HarnessError(f"unsupported value in reproducibility state digest: {type(value)!r}")


def _state_digest(
    *,
    config: TrainingConfigV1,
    epoch: int,
    best_epoch: int,
    best_validation: float,
    patience_used: int,
    metrics: Sequence[EpochMetrics],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> str:
    """Hash replay-relevant state while deliberately excluding nondeterministic timings."""

    digest = hashlib.sha256()
    _digest_value(
        digest,
        {
            "config_hash": config.config_hash,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "patience_used": patience_used,
            "metrics": [
                {
                    "epoch": metric.epoch,
                    "train_loss": metric.train_loss,
                    "validation_loss": metric.validation_loss,
                }
                for metric in metrics
            ],
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
    )
    return f"sha256:{digest.hexdigest()}"


def _result_row(
    *,
    result: TrainingResult,
    started_at: str,
    ended_at: str,
    error: BaseException | None,
) -> dict[str, object]:
    """Build one terminal record retained separately from resumable checkpoints."""

    return {
        "best_checkpoint_digest": result.best_checkpoint_digest,
        "best_epoch": result.best_epoch,
        "checkpoint_digest": result.checkpoint_digest,
        "config_hash": result.config_hash,
        "ended_at": ended_at,
        "error": None if error is None else {"message": str(error), "type": type(error).__name__},
        "metrics": [asdict(metric) for metric in result.metrics],
        "resumed": result.resumed,
        "started_at": started_at,
        "state_digest": result.state_digest,
        "status": result.status.value,
    }


def _append_result_evidence(path: Path, row: dict[str, object]) -> None:
    """Retain every terminal attempt when updating machine-readable result evidence."""

    if path.is_file():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(prior, dict) or prior.get("schema_version") != RESULTS_SCHEMA_VERSION:
            raise HarnessError("existing result evidence uses an incompatible schema")
        attempts = prior.get("attempts")
        if not isinstance(attempts, list):
            raise HarnessError("existing result evidence attempts are malformed")
    else:
        attempts = []
    attempts.append(row)
    _atomic_json(path, {"attempts": attempts, "schema_version": RESULTS_SCHEMA_VERSION})


def train_resumable(
    *,
    config: TrainingConfigV1,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_batches: Iterable[Any],
    validation_batches: Iterable[Any],
    step: TrainingStep,
    resume: bool,
    on_checkpoint: CheckpointObserver | None = None,
) -> TrainingResult:
    """Train a compact task head with deterministic resume and retained run evidence.

    The callback owns task-specific labels and loss construction. The harness
    owns seed state, epoch/early-stop policy, checkpoint compatibility, and
    complete terminal status. ``on_checkpoint`` is an observability hook called
    only after the latest checkpoint has committed; it enables job supervisors
    to stop a process without losing a resumable epoch boundary. No test split
    is accepted or evaluated here.
    """

    started_at = _utc_now()
    started = time.monotonic()
    checkpoint_path = Path(config.checkpoint_path)
    best_checkpoint_path = _best_checkpoint_path(checkpoint_path)
    result_path = Path(config.result_path)
    resumed = False
    best_epoch = 0
    metrics: list[EpochMetrics] = []
    status = TrainingStatus.FAILED
    state_digest: str | None = None

    try:
        _require_torch()
        seed_everything(
            config.seed,
            deterministic_algorithms=config.deterministic_algorithms_requested,
        )
        train_items = tuple(train_batches)
        validation_items = tuple(validation_batches)
        if not train_items or not validation_items:
            raise HarnessError("training and validation batches must both be nonempty")
        resumed = resume and checkpoint_path.is_file()
        if resumed:
            epoch, best_epoch, best_validation, patience_used, metrics = _load_checkpoint(
                checkpoint_path, config=config, model=model, optimizer=optimizer
            )
            start_epoch = epoch + 1
        else:
            start_epoch = 1
            best_validation = (
                float("inf") if config.early_stopping.minimize_metric else float("-inf")
            )
            patience_used = 0
        status = TrainingStatus.COMPLETE
        for epoch in range(start_epoch, config.epochs + 1):
            elapsed = time.monotonic() - started
            if config.timeout_seconds is not None and elapsed > config.timeout_seconds:
                status = TrainingStatus.TIMEOUT
                break
            model.train()
            train_loss = 0.0
            for batch in train_items:
                optimizer.zero_grad(set_to_none=True)
                loss = step(batch, training=True)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise HarnessError("training step must return one finite scalar loss")
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach())
            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for batch in validation_items:
                    loss = step(batch, training=False)
                    if loss.ndim != 0 or not torch.isfinite(loss):
                        raise HarnessError("validation step must return one finite scalar loss")
                    validation_loss += float(loss)
            metric = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss / len(train_items),
                validation_loss=validation_loss / len(validation_items),
                wall_seconds=time.monotonic() - started,
            )
            metrics.append(metric)
            improved = (
                metric.validation_loss < best_validation - config.early_stopping.minimum_delta
                if config.early_stopping.minimize_metric
                else metric.validation_loss > best_validation + config.early_stopping.minimum_delta
            )
            if improved:
                best_epoch = epoch
                best_validation = metric.validation_loss
                patience_used = 0
            else:
                patience_used += 1
            payload = _checkpoint_payload(
                config=config,
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation=best_validation,
                patience_used=patience_used,
                metrics=metrics,
                model=model,
                optimizer=optimizer,
            )
            _atomic_torch(checkpoint_path, payload)
            if improved:
                _atomic_torch(best_checkpoint_path, payload)
            if on_checkpoint is not None:
                on_checkpoint(metric)
            if not improved and patience_used >= config.early_stopping.patience:
                status = TrainingStatus.EARLY_STOPPED
                break
        latest_epoch = metrics[-1].epoch if metrics else 0
        state_digest = _state_digest(
            config=config,
            epoch=latest_epoch,
            best_epoch=best_epoch,
            best_validation=best_validation,
            patience_used=patience_used,
            metrics=metrics,
            model=model,
            optimizer=optimizer,
        )
    except Exception as error:
        if torch is None:
            status = TrainingStatus.UNSUPPORTED
        checkpoint_digest = _sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
        best_digest = _sha256_file(best_checkpoint_path) if best_checkpoint_path.is_file() else None
        result = TrainingResult(
            config_hash=config.config_hash,
            status=status,
            best_epoch=best_epoch,
            metrics=tuple(metrics),
            resumed=resumed,
            checkpoint_digest=checkpoint_digest,
            best_checkpoint_digest=best_digest,
            state_digest=state_digest,
        )
        _append_result_evidence(
            result_path,
            _result_row(result=result, started_at=started_at, ended_at=_utc_now(), error=error),
        )
        raise

    checkpoint_digest = _sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    best_digest = _sha256_file(best_checkpoint_path) if best_checkpoint_path.is_file() else None
    result = TrainingResult(
        config_hash=config.config_hash,
        status=status,
        best_epoch=best_epoch,
        metrics=tuple(metrics),
        resumed=resumed,
        checkpoint_digest=checkpoint_digest,
        best_checkpoint_digest=best_digest,
        state_digest=state_digest,
    )
    _append_result_evidence(
        result_path,
        _result_row(result=result, started_at=started_at, ended_at=_utc_now(), error=None),
    )
    return result
