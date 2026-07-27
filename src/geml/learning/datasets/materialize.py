"""Deterministic PyG-ready tensors that preserve ordered DAG semantics.

The records deliberately remain independent of a torch installation.  They are
the typed, serializable boundary consumed by a later optional PyG adapter, and
keep graph/pair/split/label identity outside the model feature plane.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StringConstraints,
    model_validator,
)

from geml.contracts.corpus import CorpusSplit
from geml.export.schema import canonicalize_graph
from geml.graph.schema import Graph
from geml.graph.validate import validate_graph
from geml.learning.datasets.channels import (
    ChannelState,
    RepresentationChannel,
    channel_source_note,
    channel_state,
    require_channel_available,
)

GRAPH_TENSOR_SCHEMA_VERSION = "geml-graph-tensor-v1"
CHANNEL_FAILURE_SCHEMA_VERSION = "geml-channel-failure-v1"
MATERIALIZATION_MANIFEST_SCHEMA_VERSION = "geml-goal6-channel-materialization-v1"

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_Sha256Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]


class MaterializationError(ValueError):
    """Raised when a channel cannot preserve the canonical source graph contract."""


class EndpointRole(StrEnum):
    """The ordered pair role attached outside of the model feature plane."""

    LEFT = "left"
    RIGHT = "right"


class MessageDirection(StrEnum):
    """Direction label for one derived message edge."""

    FORWARD = "forward"
    REVERSE = "reverse"


class MessageRole(StrEnum):
    """Role label retained alongside direction for message-passing implementations."""

    PARENT_TO_CHILD = "parent_to_child"
    CHILD_TO_PARENT = "child_to_parent"


class ChannelFailureStatus(StrEnum):
    """Typed non-success outcomes that keep the alignment denominator intact."""

    BLOCKED = "blocked"
    SOURCE_MISSING = "source_missing"
    INVALID_SOURCE_GRAPH = "invalid_source_graph"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class _TensorContract(BaseModel):
    """Strict shared serialization behavior for tensor and ledger records."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, strict=True)


def canonical_json_bytes(value: object) -> bytes:
    """Encode one finite JSON value in a unique deterministic form."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MaterializationError(f"record cannot be canonicalized as JSON: {error}") from error


def sha256_digest(data: bytes) -> str:
    """Return an algorithm-qualified content digest."""

    return f"sha256:{hashlib.sha256(data).hexdigest()}"


class TensorNodeV1(_TensorContract):
    """A node with exact value and all model-consumable categorical fields."""

    index: _NonNegativeInt
    node_kind: _NonBlankStr
    node_label: _NonBlankStr | None = None
    exact_value: JsonValue
    root_indicator: StrictBool
    root_orders: tuple[_NonNegativeInt, ...] = ()

    @model_validator(mode="after")
    def validate_roots(self) -> Self:
        if self.root_indicator != bool(self.root_orders):
            raise ValueError("root_indicator must agree with root_orders")
        if tuple(sorted(set(self.root_orders))) != self.root_orders:
            raise ValueError("root_orders must be sorted and unique")
        return self


class LogicalEdgeV1(_TensorContract):
    """One source parent-to-child edge; repeated references remain distinct by child slot."""

    index: _NonNegativeInt
    parent_index: _NonNegativeInt
    child_index: _NonNegativeInt
    child_slot: _NonNegativeInt


class MessageEdgeV1(_TensorContract):
    """One directed message edge derived losslessly from a logical source edge."""

    logical_edge_index: _NonNegativeInt
    source_index: _NonNegativeInt
    target_index: _NonNegativeInt
    child_slot: _NonNegativeInt
    direction: MessageDirection
    role: MessageRole


class GraphTensorV1(_TensorContract):
    """Serializable graph tensor plus external pair/label/split metadata.

    `model_feature_payload` is the sole model plane.  Pair identity, endpoint
    role, target label, and split live on this envelope and must not be passed
    to a model.
    """

    schema_version: str = GRAPH_TENSOR_SCHEMA_VERSION
    channel: RepresentationChannel
    source_note: _NonBlankStr
    representation_family: _NonBlankStr
    representation_mode: _NonBlankStr
    graph_id: _NonBlankStr
    expression_id: _NonBlankStr
    pair_id: _NonBlankStr
    endpoint_role: EndpointRole
    target_label: StrictBool
    split: CorpusSplit
    nodes: tuple[TensorNodeV1, ...] = Field(min_length=1)
    logical_edges: tuple[LogicalEdgeV1, ...] = ()
    message_edges: tuple[MessageEdgeV1, ...] = ()

    @model_validator(mode="after")
    def validate_tensor_topology(self) -> Self:
        if self.source_note != channel_source_note(self.channel):
            raise ValueError("source_note must match the declared honest channel provenance")
        indexes = tuple(node.index for node in self.nodes)
        if indexes != tuple(range(len(self.nodes))):
            raise ValueError("tensor node indexes must be dense and ordered")
        all_indexes = set(indexes)
        root_orders = tuple(order for node in self.nodes for order in node.root_orders)
        if root_orders != tuple(range(len(root_orders))):
            raise ValueError("root orders must be dense globally and in order")
        logical_indexes = tuple(edge.index for edge in self.logical_edges)
        if logical_indexes != tuple(range(len(self.logical_edges))):
            raise ValueError("logical edge indexes must be dense and ordered")
        logical_keys: set[tuple[int, int]] = set()
        for edge in self.logical_edges:
            if edge.parent_index not in all_indexes or edge.child_index not in all_indexes:
                raise ValueError("logical edge endpoint is outside the node table")
            key = (edge.parent_index, edge.child_slot)
            if key in logical_keys:
                raise ValueError("logical edges must not duplicate a parent child-slot")
            logical_keys.add(key)
        by_logical: dict[int, list[MessageEdgeV1]] = defaultdict(list)
        for edge in self.message_edges:
            if edge.logical_edge_index not in set(logical_indexes):
                raise ValueError("message edge references an unknown logical edge")
            if edge.source_index not in all_indexes or edge.target_index not in all_indexes:
                raise ValueError("message edge endpoint is outside the node table")
            by_logical[edge.logical_edge_index].append(edge)
        if len(self.message_edges) != 2 * len(self.logical_edges):
            raise ValueError(
                "every logical edge must produce exactly forward and reverse message edges"
            )
        for logical in self.logical_edges:
            expected = {
                (
                    logical.parent_index,
                    logical.child_index,
                    logical.child_slot,
                    MessageDirection.FORWARD,
                    MessageRole.PARENT_TO_CHILD,
                ),
                (
                    logical.child_index,
                    logical.parent_index,
                    logical.child_slot,
                    MessageDirection.REVERSE,
                    MessageRole.CHILD_TO_PARENT,
                ),
            }
            observed = {
                (edge.source_index, edge.target_index, edge.child_slot, edge.direction, edge.role)
                for edge in by_logical[logical.index]
            }
            if observed != expected:
                raise ValueError(
                    "message edges must losslessly encode each logical edge in both directions"
                )
        return self

    def model_feature_payload(self) -> dict[str, object]:
        """Return the typed feature plane without labels, splits, or pair identities."""

        return {
            "message_edges": [edge.model_dump(mode="json") for edge in self.message_edges],
            "nodes": [node.model_dump(mode="json") for node in self.nodes],
            "representation_family": self.representation_family,
            "representation_mode": self.representation_mode,
            "schema_version": self.schema_version,
        }


class ChannelFailureV1(_TensorContract):
    """An alignment-ledger row for a channel that did not yield a tensor."""

    schema_version: str = CHANNEL_FAILURE_SCHEMA_VERSION
    pair_id: _NonBlankStr
    endpoint_role: EndpointRole
    channel: RepresentationChannel
    status: ChannelFailureStatus
    detail: _NonBlankStr


class ChannelMaterializationManifestV1(_TensorContract):
    """Digest/count report for one deterministic temporary or production shard."""

    schema_version: str = MATERIALIZATION_MANIFEST_SCHEMA_VERSION
    channel: RepresentationChannel
    content_digest: _Sha256Digest
    tensor_count: _NonNegativeInt
    failure_count: _NonNegativeInt
    pair_count: _NonNegativeInt
    production_ready: StrictBool


def materialize_graph(
    graph: Graph,
    *,
    channel: RepresentationChannel,
    graph_id: str,
    expression_id: str,
    pair_id: str,
    endpoint_role: EndpointRole,
    target_label: bool,
    split: CorpusSplit,
) -> GraphTensorV1:
    """Materialize a validated source graph without flattening sharing or child-slot order."""

    require_channel_available(channel)
    source_validation = validate_graph(graph)
    if not source_validation.valid:
        raise MaterializationError("invalid source graph: " + "; ".join(source_validation.errors))
    canonical = canonicalize_graph(graph)
    _validate_channel_representation(
        channel,
        canonical.representation_family,
        canonical.representation_mode,
    )
    root_orders_by_target: dict[int, list[int]] = defaultdict(list)
    for root in canonical.roots:
        root_orders_by_target[root.target_ordinal].append(root.root_order)
    nodes = tuple(
        TensorNodeV1(
            index=node.ordinal,
            node_kind=node.kind,
            node_label=node.label,
            exact_value=node.value,
            root_indicator=node.ordinal in root_orders_by_target,
            root_orders=tuple(root_orders_by_target[node.ordinal]),
        )
        for node in canonical.nodes
    )
    logical_edges = tuple(
        LogicalEdgeV1(
            index=edge_index,
            parent_index=node.ordinal,
            child_index=child.target_ordinal,
            child_slot=child.slot,
        )
        for edge_index, (node, child) in enumerate(
            (node, child) for node in canonical.nodes for child in node.children
        )
    )
    message_edges = tuple(
        edge
        for logical in logical_edges
        for edge in (
            MessageEdgeV1(
                logical_edge_index=logical.index,
                source_index=logical.parent_index,
                target_index=logical.child_index,
                child_slot=logical.child_slot,
                direction=MessageDirection.FORWARD,
                role=MessageRole.PARENT_TO_CHILD,
            ),
            MessageEdgeV1(
                logical_edge_index=logical.index,
                source_index=logical.child_index,
                target_index=logical.parent_index,
                child_slot=logical.child_slot,
                direction=MessageDirection.REVERSE,
                role=MessageRole.CHILD_TO_PARENT,
            ),
        )
    )
    return GraphTensorV1(
        channel=channel,
        source_note=channel_source_note(channel),
        representation_family=canonical.representation_family,
        representation_mode=canonical.representation_mode,
        graph_id=graph_id,
        expression_id=expression_id,
        pair_id=pair_id,
        endpoint_role=endpoint_role,
        target_label=target_label,
        split=split,
        nodes=nodes,
        logical_edges=logical_edges,
        message_edges=message_edges,
    )


def _validate_channel_representation(
    channel: RepresentationChannel,
    representation_family: str,
    representation_mode: str,
) -> None:
    """Reject a source graph whose family/mode would dishonestly relabel a channel."""

    exact = {
        RepresentationChannel.AST_DAG: ("ast", "ast"),
        RepresentationChannel.PURE_EML_DAG: ("eml", "pure_eml:official_v4"),
    }
    if channel in exact:
        expected_family, expected_mode = exact[channel]
        if (representation_family, representation_mode) != (expected_family, expected_mode):
            raise MaterializationError(
                f"{channel.value} requires family/mode {(expected_family, expected_mode)!r}, "
                f"not {(representation_family, representation_mode)!r}"
            )
        return
    if channel is RepresentationChannel.FREQUENT_MACRO_MOTIF_DAG:
        prefix = "motif:frequent:motif-vocabulary:"
        suffix = ":macro:macro:official_v4:is_pure_eml=false"
        if (
            representation_family != "motif"
            or not representation_mode.startswith(prefix)
            or not representation_mode.endswith(suffix)
        ):
            raise MaterializationError(
                "frequent_macro_motif_dag requires the immutable frequent macro-motif source mode"
            )


def blocked_channel_failure(
    *,
    pair_id: str,
    endpoint_role: EndpointRole,
    channel: RepresentationChannel,
) -> ChannelFailureV1:
    """Produce a retained alignment row for a contract-blocked channel."""

    if channel_state(channel) is not ChannelState.BLOCKED:
        raise ValueError("blocked_channel_failure is only valid for a declared blocked channel")
    return ChannelFailureV1(
        pair_id=pair_id,
        endpoint_role=endpoint_role,
        channel=channel,
        status=ChannelFailureStatus.BLOCKED,
        detail=channel_source_note(channel),
    )


def validate_channel_alignment(
    tensors: Iterable[GraphTensorV1],
    failures: Iterable[ChannelFailureV1],
    *,
    expected_pair_ids: Iterable[str],
) -> None:
    """Require each pair/channel/endpoint slot to be a tensor or an explicit ledger row.

    This validates alignment even while the motif-AST fair-control contract is
    unresolved.  It does not mark the four-channel production comparison ready.
    """

    expected = tuple(sorted(set(expected_pair_ids)))
    if not expected:
        raise MaterializationError("expected_pair_ids cannot be empty")
    tensor_slots: dict[tuple[str, RepresentationChannel, EndpointRole], GraphTensorV1] = {}
    for tensor in tensors:
        slot = (tensor.pair_id, tensor.channel, tensor.endpoint_role)
        if slot in tensor_slots:
            raise MaterializationError("duplicate tensor alignment slot")
        tensor_slots[slot] = tensor
    failure_slots: dict[tuple[str, RepresentationChannel, EndpointRole], ChannelFailureV1] = {}
    for failure in failures:
        slot = (failure.pair_id, failure.channel, failure.endpoint_role)
        if slot in failure_slots:
            raise MaterializationError("duplicate failure alignment slot")
        failure_slots[slot] = failure
    overlap = set(tensor_slots) & set(failure_slots)
    if overlap:
        raise MaterializationError("a tensor slot cannot also carry a failure row")
    expected_slots = {
        (pair_id, channel, role)
        for pair_id in expected
        for channel in RepresentationChannel
        for role in EndpointRole
    }
    actual_slots = set(tensor_slots) | set(failure_slots)
    if actual_slots != expected_slots:
        missing = sorted(expected_slots - actual_slots, key=lambda item: tuple(map(str, item)))
        unexpected = sorted(actual_slots - expected_slots, key=lambda item: tuple(map(str, item)))
        raise MaterializationError(
            f"channel alignment ledger mismatch: missing={missing}, unexpected={unexpected}"
        )
    for pair_id in expected:
        labels = {
            tensor.target_label for slot, tensor in tensor_slots.items() if slot[0] == pair_id
        }
        if len(labels) > 1:
            raise MaterializationError("aligned tensors for one pair disagree on target label")
        splits = {tensor.split for slot, tensor in tensor_slots.items() if slot[0] == pair_id}
        if len(splits) > 1:
            raise MaterializationError("aligned tensors for one pair disagree on split")


def write_fixture_channel(
    tensors: Iterable[GraphTensorV1],
    failures: Iterable[ChannelFailureV1],
    output_path: str | Path,
    *,
    channel: RepresentationChannel,
) -> ChannelMaterializationManifestV1:
    """Write deterministic fixture JSONL and retain every channel failure row."""

    tensor_rows = tuple(sorted(tensors, key=lambda row: (row.pair_id, row.endpoint_role.value)))
    failure_rows = tuple(sorted(failures, key=lambda row: (row.pair_id, row.endpoint_role.value)))
    if any(row.channel is not channel for row in (*tensor_rows, *failure_rows)):
        raise MaterializationError("fixture writer accepts exactly one representation channel")
    rows = (
        *({"record_type": "tensor", **row.model_dump(mode="json")} for row in tensor_rows),
        *({"record_type": "failure", **row.model_dump(mode="json")} for row in failure_rows),
    )
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    pair_ids = {row.pair_id for row in (*tensor_rows, *failure_rows)}
    return ChannelMaterializationManifestV1(
        channel=channel,
        content_digest=sha256_digest(payload),
        tensor_count=len(tensor_rows),
        failure_count=len(failure_rows),
        pair_count=len(pair_ids),
        production_ready=(channel_state(channel) is ChannelState.AVAILABLE and not failure_rows),
    )
