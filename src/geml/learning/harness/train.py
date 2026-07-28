"""Issue 6-4: one deterministic, resumable training harness for Goals 6-9.

Design commitments
------------------
* **No task or representation fork.**  The harness consumes a :class:`TrainableTask` and never
  inspects the channel, the representation family, or the goal number.  Anything channel-specific
  belongs in the task, not here.
* **Resume equivalence.**  Each epoch derives its randomness from ``(root_seed, "epoch", n)``
  rather than from the accumulated RNG stream, so an interrupted-and-resumed run reproduces an
  uninterrupted run exactly instead of approximately.
* **Evidence is never erased.**  Failures, timeouts, and incomplete cells stay as typed rows with
  their denominators.  There is no hidden automatic retry: a technical retry is explicit, keeps the
  same seed, and is recorded alongside the original failure.
* **Atomic writes.**  Checkpoints and result files are written to a temporary path and moved into
  place with :func:`os.replace`, so an interrupted write cannot corrupt earlier evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from geml.learning.harness.config import (
    HarnessConfig,
    Precision,
    canonical_json_bytes,
    config_digest,
)
from geml.learning.harness.seeds import derive_child_seed, seed_everything

CHECKPOINT_SCHEMA_VERSION = "geml-goal6-checkpoint-v1"
RESULT_SCHEMA_VERSION = "geml-goal6-training-result-v1"


class HarnessError(RuntimeError):
    """The harness cannot proceed without discarding or fabricating evidence."""


class CheckpointMismatchError(HarnessError):
    """A checkpoint does not belong to the configuration that is being resumed."""


class CellStatus(StrEnum):
    """Typed lifecycle status for one independent training cell."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class BatchPlan:
    """One dynamic batch: the example indices it covers and its measured budget usage."""

    indices: tuple[int, ...]
    node_count: int
    edge_count: int


@runtime_checkable
class TrainableTask(Protocol):
    """The narrow task interface the harness drives.

    Issue 6-3 owns the models and issue 6-2/#56 owns the data; this protocol is the seam between
    them.  It deliberately exposes no channel identity, so the harness cannot branch on one.
    """

    def example_count(self, split: str) -> int:
        """Return the number of examples in ``split``."""

    def example_cost(self, split: str, index: int) -> tuple[int, int]:
        """Return ``(node_count, edge_count)`` for one example, for dynamic batching."""

    def train_step(self, batch: BatchPlan, epoch: int) -> float:
        """Run one optimizer-visible step and return its scalar loss."""

    def evaluate(self, split: str, batches: Sequence[BatchPlan]) -> dict[str, float]:
        """Return metrics for ``split``; keys must include the early-stopping criterion."""

    def state_dict(self) -> dict[str, object]:
        """Return the complete resumable task state (model, optimizer, scheduler, scaler)."""

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore task state produced by :meth:`state_dict`."""

    def parameter_count(self) -> int:
        """Return the code-measured trainable parameter count."""


def plan_batches(
    task: TrainableTask,
    split: str,
    order: Sequence[int],
    max_nodes: int,
    max_edges: int,
    max_examples: int,
) -> list[BatchPlan]:
    """Group ``order`` into batches bounded by node, edge, and example budgets.

    An example whose own cost exceeds a budget still forms its own single-example batch: dropping
    it would silently change the denominator, which the reporting policy forbids.
    """

    batches: list[BatchPlan] = []
    current: list[int] = []
    nodes = 0
    edges = 0
    for index in order:
        example_nodes, example_edges = task.example_cost(split, index)
        exceeds = (
            nodes + example_nodes > max_nodes
            or edges + example_edges > max_edges
            or len(current) + 1 > max_examples
        )
        if current and exceeds:
            batches.append(BatchPlan(tuple(current), nodes, edges))
            current, nodes, edges = [], 0, 0
        current.append(index)
        nodes += example_nodes
        edges += example_edges
    if current:
        batches.append(BatchPlan(tuple(current), nodes, edges))
    return batches


def epoch_order(count: int, seed: int, epoch: int) -> list[int]:
    """Return the deterministic example order for one epoch.

    Derived from ``(seed, "epoch", epoch)`` so that resuming at epoch *n* reproduces exactly the
    order an uninterrupted run would have used, without replaying the whole RNG stream.
    """

    import random as _random

    order = list(range(count))
    _random.Random(derive_child_seed(seed, "epoch", str(epoch))).shuffle(order)
    return order


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` atomically, never truncating an existing file in place."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: object) -> None:
    """Atomically write a canonically serialized JSON document."""

    atomic_write_bytes(path, canonical_json_bytes(payload) + b"\n")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class EarlyStopState:
    """Mutable early-stopping bookkeeping that must survive a resume."""

    best_value: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement: int = 0

    def improved(self, value: float, mode: str, min_delta: float) -> bool:
        if self.best_value is None:
            return True
        if mode == "min":
            return value < self.best_value - min_delta
        return value > self.best_value + min_delta

    def as_dict(self) -> dict[str, object]:
        return {
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> EarlyStopState:
        return cls(
            best_value=payload.get("best_value"),  # type: ignore[arg-type]
            best_epoch=payload.get("best_epoch"),  # type: ignore[arg-type]
            epochs_without_improvement=int(payload.get("epochs_without_improvement", 0)),
        )


@dataclass
class TrainingTelemetry:
    """Curves and counters recorded for every run."""

    train_loss_by_epoch: list[float] = field(default_factory=list)
    validation_by_epoch: list[dict[str, float]] = field(default_factory=list)
    epoch_wall_seconds: list[float] = field(default_factory=list)
    attempted_epochs: int = 0
    failed_epochs: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "train_loss_by_epoch": list(self.train_loss_by_epoch),
            "validation_by_epoch": [dict(row) for row in self.validation_by_epoch],
            "epoch_wall_seconds": list(self.epoch_wall_seconds),
            "attempted_epochs": self.attempted_epochs,
            "failed_epochs": self.failed_epochs,
        }


@dataclass
class TrainingResult:
    """The complete, denominator-explicit outcome of one training cell."""

    status: CellStatus
    config_hash: str
    seed: int
    epochs_completed: int
    best_criterion_value: float | None
    best_epoch: int | None
    telemetry: TrainingTelemetry
    parameters: int
    wall_seconds: float
    failure_reason: str | None = None
    resume_lineage: tuple[str, ...] = ()
    precision: ResolvedPrecision | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "precision": self.precision.as_dict() if self.precision else None,
            "status": str(self.status),
            "config_hash": self.config_hash,
            "seed": self.seed,
            "epochs_completed": self.epochs_completed,
            "best_criterion_value": self.best_criterion_value,
            "best_epoch": self.best_epoch,
            "telemetry": self.telemetry.as_dict(),
            "parameters": self.parameters,
            "wall_seconds": round(self.wall_seconds, 6),
            "failure_reason": self.failure_reason,
            "resume_lineage": list(self.resume_lineage),
        }


@dataclass(frozen=True)
class ResolvedPrecision:
    """The precision actually used, plus why it may differ from the request."""

    requested: str
    effective: str
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "effective": self.effective,
            "fallback_reason": self.fallback_reason,
        }


def resolve_precision(config: HarnessConfig) -> ResolvedPrecision:
    """Resolve the requested precision against what the device actually supports.

    BF16 is the intended H100 production precision.  On a device that does not support it the
    harness falls back to float32 and records the reason, rather than either crashing or silently
    running at a precision the result row does not mention.
    """

    requested = str(config.precision)
    if config.precision is not Precision.BFLOAT16:
        return ResolvedPrecision(requested=requested, effective=requested)

    try:
        import torch
    except ImportError:
        return ResolvedPrecision(requested, str(Precision.FLOAT32), "torch is not installed")

    device = config.device
    if device.startswith("cuda"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return ResolvedPrecision(requested, requested)
        return ResolvedPrecision(
            requested, str(Precision.FLOAT32), f"bfloat16 unsupported on device {device!r}"
        )
    if not hasattr(torch, "bfloat16"):  # pragma: no cover - very old torch
        return ResolvedPrecision(requested, str(Precision.FLOAT32), "torch lacks bfloat16")
    return ResolvedPrecision(
        requested, requested, f"bfloat16 on {device!r} is supported but not hardware-accelerated"
    )


def environment_report() -> dict[str, str]:
    """Record interpreter and library versions for the run envelope."""

    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("torch", "numpy", "pydantic"):
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = "not-installed"
        else:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


class TrainingHarness:
    """Drive one independent training cell to completion, or to an explicit failure row."""

    def __init__(self, config: HarnessConfig, task: TrainableTask) -> None:
        self.config = config
        self.task = task
        self.config_hash = config_digest(config)
        self.output_directory = Path(config.output_root) / config.run_name
        self.telemetry = TrainingTelemetry()
        self.early_stop = EarlyStopState()
        self._resume_lineage: list[str] = []

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.output_directory / "checkpoint.latest.json"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.output_directory / "checkpoint.best.json"

    @property
    def result_path(self) -> Path:
        return self.output_directory / "result.json"

    def _checkpoint_payload(self, epoch: int) -> dict[str, object]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "seed": self.config.seed,
            "epoch": epoch,
            "early_stop": self.early_stop.as_dict(),
            "telemetry": self.telemetry.as_dict(),
            "task_state": self.task.state_dict(),
            "environment": environment_report(),
        }

    def save_checkpoint(self, epoch: int, *, is_best: bool) -> None:
        payload = self._checkpoint_payload(epoch)
        atomic_write_json(self.latest_checkpoint_path, payload)
        if is_best:
            atomic_write_json(self.best_checkpoint_path, payload)

    def load_checkpoint(self, path: Path) -> int:
        """Restore state from ``path`` and return the epoch already completed.

        A checkpoint from a different configuration or seed is refused outright: silently resuming
        onto a mismatched configuration would produce a row that no configuration explains.
        """

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointMismatchError(
                f"checkpoint schema {payload.get('schema_version')!r} is not "
                f"{CHECKPOINT_SCHEMA_VERSION!r}"
            )
        if payload.get("config_hash") != self.config_hash:
            raise CheckpointMismatchError(
                "checkpoint config_hash does not match this configuration; refusing to resume "
                "onto a different experiment"
            )
        if payload.get("seed") != self.config.seed:
            raise CheckpointMismatchError(
                f"checkpoint seed {payload.get('seed')} does not match configured seed "
                f"{self.config.seed}"
            )
        self.task.load_state_dict(payload["task_state"])  # type: ignore[index]
        self.early_stop = EarlyStopState.from_dict(payload["early_stop"])  # type: ignore[index]
        telemetry = payload["telemetry"]  # type: ignore[index]
        self.telemetry = TrainingTelemetry(
            train_loss_by_epoch=list(telemetry["train_loss_by_epoch"]),
            validation_by_epoch=[dict(row) for row in telemetry["validation_by_epoch"]],
            epoch_wall_seconds=list(telemetry["epoch_wall_seconds"]),
            attempted_epochs=int(telemetry["attempted_epochs"]),
            failed_epochs=int(telemetry["failed_epochs"]),
        )
        self._resume_lineage.append(sha256_file(path))
        return int(payload["epoch"])  # type: ignore[index]

    def _split_batches(self, split: str, epoch: int, shuffle: bool) -> list[BatchPlan]:
        count = self.task.example_count(split)
        order = epoch_order(count, self.config.seed, epoch) if shuffle else list(range(count))
        budget = self.config.batch_budget
        return plan_batches(
            self.task,
            split,
            order,
            budget.max_nodes_per_batch,
            budget.max_edges_per_batch,
            budget.max_examples_per_batch,
        )

    def run(self, *, resume: bool = True, max_epochs: int | None = None) -> TrainingResult:
        """Train to completion, early stopping, or an explicit failure row."""

        started = time.monotonic()
        seed_everything(self.config.seed, deterministic=self.config.deterministic)

        start_epoch = 0
        if resume and self.latest_checkpoint_path.exists():
            start_epoch = self.load_checkpoint(self.latest_checkpoint_path)

        limit = max_epochs if max_epochs is not None else self.config.max_epochs
        stopping = self.config.early_stopping
        status = CellStatus.COMPLETE
        failure_reason: str | None = None
        epoch = start_epoch

        try:
            for epoch in range(start_epoch, limit):
                epoch_started = time.monotonic()
                self.telemetry.attempted_epochs += 1

                losses = [
                    self.task.train_step(batch, epoch)
                    for batch in self._split_batches("train", epoch, shuffle=True)
                ]
                mean_loss = sum(losses) / len(losses) if losses else 0.0
                self.telemetry.train_loss_by_epoch.append(mean_loss)

                metrics = self.task.evaluate(
                    "validation", self._split_batches("validation", epoch, shuffle=False)
                )
                if stopping.criterion not in metrics:
                    raise HarnessError(
                        f"early-stopping criterion {stopping.criterion!r} is absent from the "
                        f"reported validation metrics {sorted(metrics)}"
                    )
                self.telemetry.validation_by_epoch.append(dict(metrics))
                self.telemetry.epoch_wall_seconds.append(time.monotonic() - epoch_started)

                value = metrics[stopping.criterion]
                is_best = self.early_stop.improved(value, stopping.mode, stopping.min_delta)
                if is_best:
                    self.early_stop.best_value = value
                    self.early_stop.best_epoch = epoch
                    self.early_stop.epochs_without_improvement = 0
                else:
                    self.early_stop.epochs_without_improvement += 1

                self.save_checkpoint(epoch + 1, is_best=is_best)
                if self.early_stop.epochs_without_improvement >= stopping.patience:
                    break
        except KeyboardInterrupt:
            status = CellStatus.INCOMPLETE
            failure_reason = "interrupted"
        except Exception as error:
            status = CellStatus.FAILED
            failure_reason = f"{type(error).__name__}: {error}"
            self.telemetry.failed_epochs += 1

        result = TrainingResult(
            status=status,
            config_hash=self.config_hash,
            seed=self.config.seed,
            epochs_completed=len(self.telemetry.train_loss_by_epoch),
            best_criterion_value=self.early_stop.best_value,
            best_epoch=self.early_stop.best_epoch,
            telemetry=self.telemetry,
            parameters=self.task.parameter_count(),
            wall_seconds=time.monotonic() - started,
            failure_reason=failure_reason,
            resume_lineage=tuple(self._resume_lineage),
            precision=resolve_precision(self.config),
        )
        atomic_write_json(self.result_path, result.as_dict())
        return result


@dataclass(frozen=True)
class Cell:
    """One independently schedulable unit of work."""

    cell_id: str
    config: HarnessConfig
    device: str = "cpu"

    @property
    def content_hash(self) -> str:
        payload = canonical_json_bytes(
            {"cell_id": self.cell_id, "config_hash": config_digest(self.config)}
        )
        return hashlib.sha256(payload).hexdigest()


class CellScheduler:
    """Idempotent scheduling of independent cells by complete content hash.

    Resolution policy, applied before any work is done:

    ``complete`` match
        skip and verify.
    ``incomplete`` match
        resume.
    ``failed`` match
        retain the failure and require an explicit documented technical retry.
    hash mismatch
        refuse to overwrite.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def result_path(self, cell: Cell) -> Path:
        return self.root / cell.cell_id / "result.json"

    def existing_status(self, cell: Cell) -> CellStatus:
        path = self.result_path(cell)
        if not path.exists():
            return CellStatus.NOT_STARTED
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("config_hash") != config_digest(cell.config):
            raise HarnessError(
                f"cell {cell.cell_id} already holds a result for a different configuration; "
                "refusing to overwrite earlier evidence"
            )
        return CellStatus(payload["status"])

    def plan(self, cells: Iterable[Cell]) -> list[tuple[Cell, str]]:
        """Return the action for each cell without executing anything."""

        actions: list[tuple[Cell, str]] = []
        for cell in cells:
            status = self.existing_status(cell)
            if status is CellStatus.COMPLETE:
                actions.append((cell, "skip_and_verify"))
            elif status is CellStatus.NOT_STARTED:
                actions.append((cell, "run"))
            elif status in {CellStatus.INCOMPLETE, CellStatus.RUNNING}:
                actions.append((cell, "resume"))
            else:
                actions.append((cell, "retain_failure_requires_explicit_retry"))
        return actions

    def iter_runnable(self, cells: Iterable[Cell]) -> Iterator[tuple[Cell, str]]:
        """Yield only the cells that may proceed without an explicit human decision."""

        for cell, action in self.plan(cells):
            if action in {"run", "resume"}:
                yield cell, action
