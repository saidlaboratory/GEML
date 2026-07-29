"""Bind persisted issue 6-2 channel shards to the issue 6-5 production cell runner.

This is the production side of the runner's injected-dataset seam: it loads
:class:`GraphTensorV1` records from the shards written by :func:`write_channel_shard`, verifies
each shard against its completion manifest before use, and converts every record into a fully
populated :class:`GraphExample`.  Nothing here defaults a model-plane column -- every value-plane
and root column is derived explicitly from the persisted record and checked with
:meth:`GraphExample.require_complete`, so a loader regression fails loudly instead of silently
zeroing the value plane the way the in-memory fixture defaults would.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import ValidationError

from geml.contracts.corpus import CorpusSplit
from geml.experiments.goal6.cell_runner import GraphExample, PairExample
from geml.experiments.goal6.run_grid import ArmSpec, EvaluationView
from geml.learning.backbones.gin import GraphVocabularySizes, value_magnitude_features
from geml.learning.datasets.channels import RepresentationChannel
from geml.learning.datasets.materialize import (
    CategoricalVocabularyV1,
    ChannelFailureV1,
    ChannelShardManifestV1,
    EndpointRole,
    GraphTensorV1,
    MessageDirection,
    MessageRole,
    build_training_categorical_vocabulary,
    canonical_json_bytes,
    sha256_digest,
)


class ShardProviderError(RuntimeError):
    """A persisted channel shard cannot be bound to the runner without inventing data."""


class ShardIntegrityError(ShardProviderError):
    """A shard or its completion manifest is missing, unreadable, or fails its checksum."""


class ShardRecordError(ShardProviderError):
    """A shard record is malformed, misaligned, or incomplete for pair construction."""


#: Goal 1's stored test_ood is one combined stress profile; it is reported under its own view
#: name and never relabelled strict depth- or family-OOD (configs/goal6_grid.yaml).
SPLIT_VIEWS: dict[CorpusSplit, str] = {
    CorpusSplit.TRAIN: EvaluationView.TRAIN.value,
    CorpusSplit.VALIDATION: EvaluationView.VALIDATION.value,
    CorpusSplit.TEST_IID: EvaluationView.TEST_IID.value,
    CorpusSplit.TEST_OOD: EvaluationView.TEST_OOD_STRESS.value,
}

#: The closed exact-value category set; it must fit the encoder's value_category vocabulary.
VALUE_CATEGORIES: tuple[str, ...] = (
    "null",
    "bool",
    "integer",
    "rational",
    "number",
    "string",
    "array",
    "object",
)

_DIRECTION_INDEX = {MessageDirection.FORWARD: 0, MessageDirection.REVERSE: 1}
_ROLE_INDEX = {MessageRole.PARENT_TO_CHILD: 0, MessageRole.CHILD_TO_PARENT: 1}


def _sign(value: float) -> int:
    return 0 if value < 0 else 2 if value > 0 else 1


def _value_plane(value: object) -> tuple[int, int, tuple[float, float]]:
    """Return ``(category, sign, magnitude)`` derived from one exact stored value."""

    if value is None:
        return VALUE_CATEGORIES.index("null"), 1, (0.0, 0.0)
    if isinstance(value, bool):
        return VALUE_CATEGORIES.index("bool"), 1, (0.0, 0.0)
    if isinstance(value, int):
        return VALUE_CATEGORIES.index("integer"), _sign(value), value_magnitude_features(value)
    if isinstance(value, float):
        magnitude = (math.log1p(abs(value)), math.log1p(1))
        return VALUE_CATEGORIES.index("number"), _sign(value), magnitude
    if isinstance(value, str):
        return VALUE_CATEGORIES.index("string"), 1, (0.0, 0.0)
    if isinstance(value, list):
        return VALUE_CATEGORIES.index("array"), 1, (0.0, 0.0)
    if isinstance(value, dict):
        numerator = value.get("numerator")
        denominator = value.get("denominator")
        if (
            set(value) == {"numerator", "denominator"}
            and type(numerator) is int
            and type(denominator) is int
            and denominator != 0
        ):
            magnitude = value_magnitude_features(numerator, denominator)
            return VALUE_CATEGORIES.index("rational"), _sign(numerator * denominator), magnitude
        return VALUE_CATEGORIES.index("object"), 1, (0.0, 0.0)
    raise ShardRecordError(f"exact value of type {type(value).__name__} is not JSON-shaped")


def _value_bucket(value: object, bucket_count: int) -> int:
    """Hash the exact canonical value string into a bounded, run-independent bucket."""

    digest = hashlib.sha256(canonical_json_bytes(value)).digest()
    return int.from_bytes(digest[:8], "big") % bucket_count


def _index_map(values: Sequence[str], size: int, *, field: str) -> dict[str, int]:
    """Map known training values to ``1..n``, reserving index 0 for out-of-vocabulary."""

    if len(values) + 1 > size:
        raise ShardRecordError(
            f"training vocabulary holds {len(values)} {field} entries but the encoder reserves "
            f"only {size} indices including the unknown slot"
        )
    return {value: position + 1 for position, value in enumerate(values)}


def graph_example_from_tensor(
    tensor: GraphTensorV1,
    *,
    vocabulary: CategoricalVocabularyV1,
    sizes: GraphVocabularySizes | None = None,
) -> GraphExample:
    """Convert one persisted tensor record with every model-plane column populated.

    Kind and label indices come from the train-only categorical vocabulary with index 0 reserved
    for out-of-vocabulary values, so held-out splits can never grow the vocabulary.
    """

    sizes = sizes or GraphVocabularySizes()
    if len(VALUE_CATEGORIES) > sizes.value_category:
        raise ShardRecordError(
            f"the encoder's value_category vocabulary ({sizes.value_category}) cannot hold the "
            f"{len(VALUE_CATEGORIES)} declared exact-value categories"
        )
    kinds = _index_map(vocabulary.node_kinds, sizes.kind, field="node kind")
    labels = _index_map(vocabulary.node_labels, sizes.label, field="node label")
    categories: list[int] = []
    signs: list[int] = []
    magnitudes: list[tuple[float, float]] = []
    buckets: list[int] = []
    is_root: list[int] = []
    root_order: list[int] = []
    for node in tensor.nodes:
        category, sign, magnitude = _value_plane(node.exact_value)
        categories.append(category)
        signs.append(sign)
        magnitudes.append(magnitude)
        buckets.append(_value_bucket(node.exact_value, sizes.value_bucket))
        is_root.append(int(node.root_indicator))
        order = node.root_orders[0] if node.root_orders else 0
        if order >= sizes.root_order:
            raise ShardRecordError(
                f"graph {tensor.graph_id!r} node {node.index} has root order {order}, outside "
                f"the encoder's root_order vocabulary of {sizes.root_order}"
            )
        root_order.append(order)
    messages: list[tuple[int, int, int, int, int]] = []
    for edge in tensor.message_edges:
        if edge.child_slot >= sizes.slot:
            raise ShardRecordError(
                f"graph {tensor.graph_id!r} message edge uses child slot {edge.child_slot}, "
                f"outside the encoder's slot vocabulary of {sizes.slot}"
            )
        messages.append(
            (
                edge.source_index,
                edge.target_index,
                _DIRECTION_INDEX[edge.direction],
                edge.child_slot,
                _ROLE_INDEX[edge.role],
            )
        )
    return GraphExample(
        node_kind=tuple(kinds.get(node.node_kind, 0) for node in tensor.nodes),
        node_label=tuple(
            labels.get(node.node_label, 0) if node.node_label else 0 for node in tensor.nodes
        ),
        messages=tuple(messages),
        node_value_category=tuple(categories),
        node_value_sign=tuple(signs),
        node_value_magnitude=tuple(magnitudes),
        node_value_bucket=tuple(buckets),
        node_is_root=tuple(is_root),
        node_root_order=tuple(root_order),
    ).require_complete()


def load_channel_shard(
    path: str | Path, *, channel: RepresentationChannel
) -> tuple[tuple[GraphTensorV1, ...], tuple[ChannelFailureV1, ...]]:
    """Load one shard's tensor and failure rows after verifying its completion manifest."""

    shard_path = Path(path)
    manifest_path = shard_path.with_suffix(shard_path.suffix + ".manifest.json")
    if not shard_path.is_file():
        raise ShardIntegrityError(f"channel shard is missing: {shard_path}")
    if not manifest_path.is_file():
        raise ShardIntegrityError(f"channel-shard manifest is missing: {manifest_path}")
    try:
        manifest = ChannelShardManifestV1.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except ValidationError as error:
        raise ShardIntegrityError(f"channel-shard manifest is invalid: {manifest_path}") from error
    if manifest.channel is not channel:
        raise ShardRecordError(
            f"shard {shard_path} materializes channel {manifest.channel.value!r}, "
            f"not {channel.value!r}"
        )
    payload = shard_path.read_bytes()
    observed = sha256_digest(payload)
    if observed != manifest.content_digest:
        raise ShardIntegrityError(
            f"shard {shard_path} fails its checksum: manifest records "
            f"{manifest.content_digest} but the file hashes to {observed}"
        )
    tensors: list[GraphTensorV1] = []
    failures: list[ChannelFailureV1] = []
    for number, line in enumerate(payload.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ShardRecordError(f"shard {shard_path} line {number} is not JSON") from error
        record_type = row.pop("record_type", None) if isinstance(row, dict) else None
        model = {"tensor": GraphTensorV1, "failure": ChannelFailureV1}.get(record_type)
        if model is None:
            raise ShardRecordError(
                f"shard {shard_path} line {number} has unknown record_type {record_type!r}"
            )
        try:
            # Strict contracts only coerce arrays/enums in JSON mode, so validate JSON bytes.
            record = model.model_validate_json(json.dumps(row))
        except ValidationError as error:
            raise ShardRecordError(
                f"shard {shard_path} line {number} is not a valid {record_type} record"
            ) from error
        if record.channel is not channel:
            raise ShardRecordError(
                f"shard {shard_path} line {number} carries channel {record.channel.value!r} "
                f"inside a {channel.value!r} shard"
            )
        (tensors if record_type == "tensor" else failures).append(record)
    if len(tensors) != manifest.tensor_count or len(failures) != manifest.failure_count:
        raise ShardIntegrityError(
            f"shard {shard_path} holds {len(tensors)} tensors and {len(failures)} failures but "
            f"its manifest records {manifest.tensor_count} and {manifest.failure_count}"
        )
    return tuple(tensors), tuple(failures)


def pair_dataset_from_tensors(
    tensors: Iterable[GraphTensorV1],
    *,
    vocabulary: CategoricalVocabularyV1,
    sizes: GraphVocabularySizes | None = None,
) -> dict[str, tuple[PairExample, ...]]:
    """Group aligned endpoint tensors into per-view labelled pairs, refusing partial pairs."""

    by_pair: dict[str, dict[EndpointRole, GraphTensorV1]] = defaultdict(dict)
    for tensor in tensors:
        endpoints = by_pair[tensor.pair_id]
        if tensor.endpoint_role in endpoints:
            raise ShardRecordError(
                f"pair {tensor.pair_id!r} holds duplicate {tensor.endpoint_role.value} endpoints"
            )
        endpoints[tensor.endpoint_role] = tensor
    views: dict[str, list[PairExample]] = defaultdict(list)
    for pair_id in sorted(by_pair):
        endpoints = by_pair[pair_id]
        missing = [role.value for role in EndpointRole if role not in endpoints]
        if missing:
            raise ShardRecordError(
                f"pair {pair_id!r} is missing its {' and '.join(missing)} endpoint; partial "
                "pairs are never trained on"
            )
        left = endpoints[EndpointRole.LEFT]
        right = endpoints[EndpointRole.RIGHT]
        if left.split is not right.split:
            raise ShardRecordError(f"pair {pair_id!r} endpoints disagree on split")
        if left.target_label != right.target_label:
            raise ShardRecordError(f"pair {pair_id!r} endpoints disagree on target label")
        views[SPLIT_VIEWS[left.split]].append(
            PairExample(
                left=graph_example_from_tensor(left, vocabulary=vocabulary, sizes=sizes),
                right=graph_example_from_tensor(right, vocabulary=vocabulary, sizes=sizes),
                label=left.target_label,
            )
        )
    return {view: tuple(examples) for view, examples in views.items()}


class ShardChannelProvider:
    """A dataset provider reading one channel's persisted, checksum-verified shards.

    Retained failure ledger rows fail the arm closed: they mean the channel's aligned pair
    denominator is not intact, so the runner must record an explicit FAILED row rather than
    train on a silently smaller subset.
    """

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        *,
        sizes: GraphVocabularySizes | None = None,
    ) -> None:
        self.shard_paths = tuple(Path(path) for path in shard_paths)
        if not self.shard_paths:
            raise ShardProviderError("a shard-backed provider needs at least one shard path")
        self.sizes = sizes or GraphVocabularySizes()

    def __call__(self, arm: ArmSpec) -> dict[str, tuple[PairExample, ...]]:
        if arm.channel is None:
            raise ShardRecordError(
                f"arm {arm.arm_id!r} declares no graph channel; shards cannot feed it"
            )
        try:
            channel = RepresentationChannel(arm.channel.channel_name)
        except ValueError as error:
            raise ShardRecordError(
                f"arm {arm.arm_id!r} names unknown representation channel "
                f"{arm.channel.channel_name!r}"
            ) from error
        tensors: list[GraphTensorV1] = []
        failures: list[ChannelFailureV1] = []
        for path in self.shard_paths:
            shard_tensors, shard_failures = load_channel_shard(path, channel=channel)
            tensors.extend(shard_tensors)
            failures.extend(shard_failures)
        if failures:
            slots = sorted(
                f"{row.pair_id}/{row.endpoint_role.value}:{row.status.value}" for row in failures
            )
            listed = ", ".join(slots[:5]) + (", ..." if len(slots) > 5 else "")
            raise ShardRecordError(
                f"channel {channel.value!r} retains {len(failures)} failure rows ({listed}); "
                "the arm fails closed instead of training on a reduced pair subset"
            )
        training = tuple(row for row in tensors if row.split is CorpusSplit.TRAIN)
        if not training:
            raise ShardRecordError(
                f"channel {channel.value!r} shards hold no training tensors to fit the "
                "categorical vocabulary"
            )
        vocabulary = build_training_categorical_vocabulary(training)
        return pair_dataset_from_tensors(tensors, vocabulary=vocabulary, sizes=self.sizes)
