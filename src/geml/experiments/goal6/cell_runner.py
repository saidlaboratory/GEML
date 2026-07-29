"""Issue 6-5: the production cell runner behind the six-arm Goal 6 grid.

This is the wiring between three frozen contracts: the issue 6-2/#56 channel datasets, the
issue 6-3 backbones, and the issue 6-4 training harness.  One :class:`TrainableTask` exists per
arm family -- the shared-encoder Siamese graph task, the compute-matched prefix-sequence task, and
the transparent trivial floor -- and each drives its real backbone through the real
:class:`TrainingHarness`.

Two conventions are enforced rather than assumed:

**Data is injected, never invented.**
    Every arm consumes an explicitly bound dataset provider.  Production binds providers that read
    the materialized channel shards; tests bind tiny in-memory fixtures.  A missing provider is an
    explicit FAILED row naming the missing production input -- the runner never substitutes data,
    so fixture inputs can never masquerade as production output.

**Hyperparameters come from configuration, not from this module.**
    Widths, layers, budgets, and epochs arrive through :class:`HarnessConfig` and the runner's
    explicit backbone configuration arguments.  The compact-model hidden width is pending the
    issue 6-3 preflight freeze, so :class:`ProductionCellRunner` requires an explicit
    :class:`EncoderConfig` and adds no width default of its own; backbone-class defaults remain
    the backbones' to own.
"""

from __future__ import annotations

import base64
import io
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from geml.experiments.goal6.run_grid import (
    ArmFamily,
    ArmSpec,
    EvaluationView,
    GridRow,
    grid_cell_id,
)
from geml.learning.backbones.gin import (
    CompactGraphEncoder,
    EncoderConfig,
    EquivalencePairModel,
    forward_flop_estimate,
    parameter_report,
)
from geml.learning.backbones.prefix_transformer import (
    PAD_TOKEN,
    PrefixSequenceEncoder,
    PrefixTransformerConfig,
    PrefixTransformerPairModel,
    prefix_forward_flop_estimate,
)
from geml.learning.backbones.trivial import (
    StructuralCounts,
    TrivialFloorPairModel,
    counts_to_tensor,
    trivial_parameter_report,
)
from geml.learning.harness.config import HarnessConfig, config_digest
from geml.learning.harness.seeds import seed_everything
from geml.learning.harness.train import (
    BatchPlan,
    CellStatus,
    TrainingHarness,
    plan_batches,
)

#: The evaluation views a provider may populate; any other name is a typo, not a new view.
KNOWN_VIEWS: frozenset[str] = frozenset(view.value for view in EvaluationView)

#: Views a cell cannot train and early-stop without.
REQUIRED_VIEWS: tuple[str, ...] = (
    EvaluationView.TRAIN.value,
    EvaluationView.VALIDATION.value,
)


class CellRunnerError(RuntimeError):
    """A cell cannot run without inventing data; surfaced as a FAILED row, never raised out."""


@dataclass(frozen=True)
class GraphExample:
    """One endpoint graph in the exact per-node/per-message layout the encoder consumes.

    ``messages`` rows are ``(source, target, direction, slot, role)`` tuples, matching the
    materialized message-edge contract: every logical edge appears as forward and reverse message
    edges, and repeated references stay distinct.  Optional columns default to the same neutral
    values the issue 6-3 encoder fixtures use.
    """

    node_kind: tuple[int, ...]
    node_label: tuple[int, ...]
    messages: tuple[tuple[int, int, int, int, int], ...]
    node_value_category: tuple[int, ...] | None = None
    node_value_sign: tuple[int, ...] | None = None
    node_value_magnitude: tuple[tuple[float, float], ...] | None = None
    node_value_bucket: tuple[int, ...] | None = None
    node_is_root: tuple[int, ...] | None = None
    node_root_order: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PairExample:
    """One labelled pair.  The label lives here and only here, never inside an endpoint."""

    left: object
    right: object
    label: bool


#: One arm's datasets: evaluation-view name mapped to its labelled pairs.
PairDataset = Mapping[str, Sequence[PairExample]]

#: Called with the arm so a production provider can locate that arm's materialized shards.
DatasetProvider = Callable[[ArmSpec], PairDataset]


class CollatedGraphBatch:
    """A batch of :class:`GraphExample` graphs concatenated with node-index offsets.

    Satisfies the encoder's ``GraphBatchProtocol`` structurally; the encoder re-validates shapes
    and endpoint ranges itself, so this class only assembles tensors.
    """

    def __init__(self, examples: Sequence[GraphExample]) -> None:
        if not examples:
            raise CellRunnerError("cannot collate zero graphs into a batch")
        kinds: list[int] = []
        labels: list[int] = []
        categories: list[int] = []
        signs: list[int] = []
        magnitudes: list[tuple[float, float]] = []
        buckets: list[int] = []
        is_root: list[int] = []
        root_order: list[int] = []
        sources: list[int] = []
        targets: list[int] = []
        directions: list[int] = []
        slots: list[int] = []
        roles: list[int] = []
        graph_index: list[int] = []

        offset = 0
        for number, example in enumerate(examples):
            count = len(example.node_kind)
            kinds.extend(example.node_kind)
            labels.extend(example.node_label)
            categories.extend(example.node_value_category or (0,) * count)
            signs.extend(example.node_value_sign or (2,) * count)
            magnitudes.extend(example.node_value_magnitude or ((0.0, 0.0),) * count)
            buckets.extend(example.node_value_bucket or (0,) * count)
            is_root.extend(example.node_is_root or (0,) * count)
            root_order.extend(example.node_root_order or (0,) * count)
            for source, target, direction, slot, role in example.messages:
                sources.append(source + offset)
                targets.append(target + offset)
                directions.append(direction)
                slots.append(slot)
                roles.append(role)
            graph_index.extend((number,) * count)
            offset += count

        self.node_kind = torch.tensor(kinds, dtype=torch.long)
        self.node_label = torch.tensor(labels, dtype=torch.long)
        self.node_value_category = torch.tensor(categories, dtype=torch.long)
        self.node_value_sign = torch.tensor(signs, dtype=torch.long)
        self.node_value_magnitude = torch.tensor(magnitudes, dtype=torch.float32)
        self.node_value_bucket = torch.tensor(buckets, dtype=torch.long)
        self.node_is_root = torch.tensor(is_root, dtype=torch.long)
        self.node_root_order = torch.tensor(root_order, dtype=torch.long)
        self.message_source = torch.tensor(sources, dtype=torch.long)
        self.message_target = torch.tensor(targets, dtype=torch.long)
        self.message_direction = torch.tensor(directions, dtype=torch.long)
        self.message_slot = torch.tensor(slots, dtype=torch.long)
        self.message_role = torch.tensor(roles, dtype=torch.long)
        self.graph_index = torch.tensor(graph_index, dtype=torch.long)
        self.num_graphs = len(examples)


class _PairTaskBase:
    """The harness-facing plumbing shared by all three arm families.

    Checkpoint state is a ``torch.save`` blob carried as base64 so the model and optimizer survive
    the harness's JSON checkpoint format byte-exactly.
    """

    endpoint_type: type = object

    def __init__(self, dataset: PairDataset, model: nn.Module, config: HarnessConfig) -> None:
        self.dataset = {view: tuple(examples) for view, examples in dataset.items()}
        for view, examples in self.dataset.items():
            for example in examples:
                if not isinstance(example, PairExample):
                    raise CellRunnerError(f"view {view!r} holds a non-PairExample entry")
                if not (
                    isinstance(example.left, self.endpoint_type)
                    and isinstance(example.right, self.endpoint_type)
                ):
                    raise CellRunnerError(
                        f"view {view!r} holds endpoints that are not "
                        f"{self.endpoint_type.__name__}; {type(self).__name__} cannot consume "
                        "another family's data"
                    )
        self.model = model
        policy = config.optimizer
        self.gradient_clip_norm = policy.gradient_clip_norm
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=policy.learning_rate,
            weight_decay=policy.weight_decay,
        )
        self.loss = nn.BCEWithLogitsLoss()

    def example_count(self, split: str) -> int:
        return len(self.dataset.get(split, ()))

    def _examples(self, split: str, indices: Sequence[int]) -> list[PairExample]:
        examples = self.dataset[split]
        return [examples[index] for index in indices]

    @staticmethod
    def _labels(examples: Sequence[PairExample]) -> Tensor:
        return torch.tensor(
            [1.0 if example.label else 0.0 for example in examples], dtype=torch.float32
        )

    def model_inputs(self, split: str, indices: Sequence[int]) -> tuple[tuple, Tensor]:
        """Return ``(model_arguments, labels)`` for one batch; labels never enter the arguments."""

        raise NotImplementedError

    def train_step(self, batch: BatchPlan, epoch: int) -> float:
        self.model.train()
        inputs, labels = self.model_inputs(EvaluationView.TRAIN.value, batch.indices)
        self.optimizer.zero_grad()
        loss = self.loss(self.model(*inputs), labels)
        loss.backward()
        if self.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        return float(loss.detach())

    def evaluate(self, split: str, batches: Sequence[BatchPlan]) -> dict[str, float]:
        if not batches:
            raise CellRunnerError(f"split {split!r} produced no batches to evaluate")
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in batches:
                inputs, labels = self.model_inputs(split, batch.indices)
                logits = self.model(*inputs)
                total_loss += float(self.loss(logits, labels)) * labels.numel()
                correct += int(((logits > 0.0) == (labels > 0.5)).sum())
                total += labels.numel()
        self.model.train()
        return {"validation_loss": total_loss / total, "accuracy": correct / total}

    def state_dict(self) -> dict[str, object]:
        buffer = io.BytesIO()
        torch.save(
            {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, buffer
        )
        return {"torch_state_b64": base64.b64encode(buffer.getvalue()).decode("ascii")}

    def load_state_dict(self, state: dict[str, object]) -> None:
        blob = base64.b64decode(str(state["torch_state_b64"]))
        payload = torch.load(io.BytesIO(blob), weights_only=True)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])


class GraphPairTask(_PairTaskBase):
    """The Siamese equivalence task over collated graph endpoints."""

    endpoint_type = GraphExample

    def example_cost(self, split: str, index: int) -> tuple[int, int]:
        example = self.dataset[split][index]
        return (
            len(example.left.node_kind) + len(example.right.node_kind),
            len(example.left.messages) + len(example.right.messages),
        )

    def model_inputs(self, split: str, indices: Sequence[int]) -> tuple[tuple, Tensor]:
        examples = self._examples(split, indices)
        left = CollatedGraphBatch([example.left for example in examples])
        right = CollatedGraphBatch([example.right for example in examples])
        return (left, right), self._labels(examples)

    def parameter_count(self) -> int:
        return parameter_report(self.model).total_parameters

    def reference_forward_flops(self, batch: BatchPlan) -> int:
        """Estimate one pair-batch forward pass under the declared reference batch.

        ``forward_flop_estimate`` doubles its single-endpoint workload for the shared encoder, so
        averaging the left-batch and right-batch estimates yields exactly
        ``encoder(left) + encoder(right) + head`` without modifying the frozen utility.
        """

        (left, right), _ = self.model_inputs(EvaluationView.TRAIN.value, batch.indices)
        left_estimate = forward_flop_estimate(self.model, left).forward_flops
        right_estimate = forward_flop_estimate(self.model, right).forward_flops
        return (left_estimate + right_estimate) // 2


class PrefixPairTask(_PairTaskBase):
    """The compute-matched sequence-control task over already-serialized prefix tokens."""

    endpoint_type = tuple

    def example_cost(self, split: str, index: int) -> tuple[int, int]:
        example = self.dataset[split][index]
        return (len(example.left) + len(example.right), 0)

    @staticmethod
    def _pad(sequences: Sequence[tuple[int, ...]]) -> Tensor:
        length = max(len(sequence) for sequence in sequences)
        rows = [list(sequence) + [PAD_TOKEN] * (length - len(sequence)) for sequence in sequences]
        return torch.tensor(rows, dtype=torch.long)

    def model_inputs(self, split: str, indices: Sequence[int]) -> tuple[tuple, Tensor]:
        examples = self._examples(split, indices)
        left = self._pad([example.left for example in examples])
        right = self._pad([example.right for example in examples])
        return (left, right), self._labels(examples)

    def parameter_count(self) -> int:
        return parameter_report(self.model).total_parameters

    def reference_forward_flops(self, batch: BatchPlan) -> int:
        (left, right), _ = self.model_inputs(EvaluationView.TRAIN.value, batch.indices)
        left_estimate = prefix_forward_flop_estimate(self.model, left.shape[0], left.shape[1])
        right_estimate = prefix_forward_flop_estimate(self.model, right.shape[0], right.shape[1])
        return (int(left_estimate["forward_flops"]) + int(right_estimate["forward_flops"])) // 2


class TrivialFloorTask(_PairTaskBase):
    """The transparent structural-counts floor over predeclared features."""

    endpoint_type = StructuralCounts

    def example_cost(self, split: str, index: int) -> tuple[int, int]:
        example = self.dataset[split][index]
        return (
            example.left.node_count + example.right.node_count,
            example.left.edge_count + example.right.edge_count,
        )

    def model_inputs(self, split: str, indices: Sequence[int]) -> tuple[tuple, Tensor]:
        examples = self._examples(split, indices)
        left = counts_to_tensor(tuple(example.left for example in examples))
        right = counts_to_tensor(tuple(example.right for example in examples))
        return (left, right), self._labels(examples)

    def parameter_count(self) -> int:
        return int(trivial_parameter_report(self.model)["total_parameters"])

    def reference_forward_flops(self, batch: BatchPlan) -> int:
        per_pair = int(trivial_parameter_report(self.model)["forward_flops"])
        return per_pair * len(batch.indices)


def resolve_commit(repository_root: Path | None = None) -> str:
    """Resolve the commit via ``git rev-parse``, failing before any cell evidence exists."""

    root = repository_root or Path(__file__).resolve().parents[4]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CellRunnerError(f"could not resolve the git commit under {root}") from error
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not commit:
        raise CellRunnerError(
            f"git rev-parse failed under {root}: {completed.stderr.strip() or 'no output'}; "
            "pass commit= explicitly when running outside a checkout"
        )
    return commit


def _validate_dataset(arm: ArmSpec, dataset: Mapping[str, Sequence[PairExample]]) -> None:
    unknown = sorted(set(dataset) - KNOWN_VIEWS)
    if unknown:
        raise CellRunnerError(
            f"arm {arm.arm_id!r} provided unknown evaluation views {unknown}; views must use the "
            f"declared names {sorted(KNOWN_VIEWS)}"
        )
    empty = sorted(view for view, examples in dataset.items() if not examples)
    if empty:
        raise CellRunnerError(
            f"arm {arm.arm_id!r} provided empty evaluation views {empty}; a zero-example view "
            "can never be evaluated, so the cell fails before training instead of after it"
        )
    for view in REQUIRED_VIEWS:
        if not dataset.get(view):
            raise CellRunnerError(
                f"arm {arm.arm_id!r} has no {view!r} examples; a cell is never trained or "
                "early-stopped on invented data"
            )


class ProductionCellRunner:
    """Run one (arm, seed) cell on the real backbones through the real harness.

    ``encoder_config`` has no default here because the compact-model hidden width is pending the
    issue 6-3 preflight freeze: the caller states the width, the runner adds no layer of defaults.
    ``run_name`` in the incoming configuration is replaced with :func:`grid_cell_id` so the three
    seed cells of one arm can never share checkpoints; the recorded ``config_hash`` is the digest
    of the configuration that actually ran.  When metrics are reported, they come from the
    early-stopping-selected best checkpoint, not from whatever epoch happened to run last.
    """

    def __init__(
        self,
        providers: Mapping[str, DatasetProvider],
        *,
        encoder_config: EncoderConfig,
        prefix_config: PrefixTransformerConfig | None = None,
        commit: str | None = None,
        repository_root: Path | None = None,
    ) -> None:
        self.providers = dict(providers)
        self.encoder_config = encoder_config
        self.prefix_config = prefix_config
        self.commit = commit if commit is not None else resolve_commit(repository_root)

    def run_cell(self, arm: ArmSpec, seed: int, config: HarnessConfig) -> GridRow:
        """Run one cell to a COMPLETE or explicit FAILED row; exceptions never escape."""

        started = time.monotonic()
        config_hash = ""  # replaced inside the try; empty only if hashing itself failed
        try:
            config_hash = config_digest(config)
            provider = self.providers.get(arm.arm_id)
            if provider is None:
                missing = (
                    f"materialized shards for channel {arm.channel.channel_name!r}"
                    if arm.channel is not None
                    else f"the {arm.family.value} control dataset"
                )
                raise CellRunnerError(
                    f"no dataset provider bound for arm {arm.arm_id!r}: {missing} are not wired, "
                    "and the runner never invents data in their place"
                )
            effective = self._effective_config(arm, seed, config)
            config_hash = config_digest(effective)
            dataset = {view: tuple(examples) for view, examples in dict(provider(arm)).items()}
            _validate_dataset(arm, dataset)

            seed_everything(effective.seed, deterministic=effective.deterministic)
            task = self._build_task(arm, dataset, effective)
            harness = TrainingHarness(effective, task)
            result = harness.run()

            row = self._row(arm, seed, config_hash, result.status)
            row.parameters = task.parameter_count()
            row.failure_reason = result.failure_reason
            if result.status is CellStatus.COMPLETE:
                harness.load_checkpoint(harness.best_checkpoint_path)
                row.forward_flops = task.reference_forward_flops(
                    self._plan(task, EvaluationView.TRAIN.value, effective)[0]
                )
                for view in sorted(dataset):
                    batches = self._plan(task, view, effective)
                    count = task.example_count(view)
                    row.metrics_by_view[view] = task.evaluate(view, batches)
                    # evaluate() scores every example or raises into a FAILED row, so the
                    # denominator cannot silently shrink.
                    row.denominators_by_view[view] = {
                        "attempted": count,
                        "valid": count,
                        "failed": 0,
                        "unsupported": 0,
                        "timed_out": 0,
                    }
            row.wall_seconds = time.monotonic() - started
            return row
        except Exception as error:
            return self._row(
                arm,
                seed,
                config_hash,
                CellStatus.FAILED,
                failure_reason=f"{type(error).__name__}: {error}",
                wall_seconds=time.monotonic() - started,
            )

    def _build_task(
        self, arm: ArmSpec, dataset: PairDataset, config: HarnessConfig
    ) -> _PairTaskBase:
        if arm.family is ArmFamily.GRAPH:
            model = EquivalencePairModel(CompactGraphEncoder(self.encoder_config))
            return GraphPairTask(dataset, model, config)
        if arm.family is ArmFamily.PREFIX_TRANSFORMER:
            model = PrefixTransformerPairModel(PrefixSequenceEncoder(self.prefix_config))
            return PrefixPairTask(dataset, model, config)
        return TrivialFloorTask(dataset, TrivialFloorPairModel(), config)

    def _row(
        self, arm: ArmSpec, seed: int, config_hash: str, status: CellStatus, **fields: object
    ) -> GridRow:
        return GridRow(
            arm_id=arm.arm_id,
            family=arm.family,
            channel_name=arm.channel.channel_name if arm.channel else None,
            representation_mode=arm.channel.representation_mode if arm.channel else None,
            seed=seed,
            status=status,
            config_hash=config_hash,
            commit=self.commit,
            **fields,
        )

    @staticmethod
    def _effective_config(arm: ArmSpec, seed: int, config: HarnessConfig) -> HarnessConfig:
        payload = config.model_dump()
        payload.update(seed=seed, run_name=grid_cell_id(arm, seed))
        return HarnessConfig.model_validate(payload)

    @staticmethod
    def _plan(task: _PairTaskBase, view: str, config: HarnessConfig) -> list[BatchPlan]:
        budget = config.batch_budget
        return plan_batches(
            task,
            view,
            list(range(task.example_count(view))),
            budget.max_nodes_per_batch,
            budget.max_edges_per_batch,
            budget.max_examples_per_batch,
        )
