"""Issue 6-3: one compact edge-aware GINE-style encoder shared by every graph channel.

Scope and honesty notes
-----------------------
This module implements a *project-specific* three-layer message-passing encoder that is
inspired by GIN (Xu et al., 2019, https://arxiv.org/abs/1810.00826) and by the edge-feature
variant exposed as ``GINEConv`` in PyTorch Geometric.  It is **not** an exact reproduction of
either paper or of the PyTorch Geometric implementation, and it must never be described as one.
The deviations are enumerated in :data:`ENCODER_DEVIATIONS` and are asserted by the tests.

The encoder deliberately depends only on ``torch``.  The repository's declared optional ``[ml]``
extra (owned by issue 6-0/#54) pins both ``torch`` and ``torch-geometric``, but this module keeps
to plain ``torch`` so a torch-only environment can still run it.
Message scattering is therefore implemented with ``Tensor.index_add_``.

Representation neutrality
-------------------------
There is exactly one encoder implementation and no channel-specific branch.  The encoder never
sees a label, a split, a pair identity, an evaluation outcome, or any channel name: it consumes
only the model-plane tensor fields described by :class:`GraphBatchProtocol`.

Directionality
--------------
:meth:`CompactGraphEncoder.encode` returns both node-level and graph-level embeddings so that the
Goal 7/8 policy and value heads can reuse this encoder.  Those heads are **goal-conditioned and
order-sensitive**: they must consume the directional outputs of :meth:`encode` directly and must
never reuse :func:`symmetric_pair_features`, which is swap-invariant by construction and therefore
destroys the current/goal distinction that directional proof prediction depends on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

ENCODER_SCHEMA_VERSION = "geml-goal6-compact-encoder-v1"

#: Widths permitted for the single allowed development comparison (issue 6-3).
PERMITTED_HIDDEN_WIDTHS: tuple[int, ...] = (64, 96)

#: The approximate total-parameter target stated by issue 6-3 for the whole task model.
PARAMETER_TARGET_RANGE: tuple[int, int] = (200_000, 1_000_000)

#: Measured with the *fixture* vocabulary defaults in :class:`GraphVocabularySizes`.  Production
#: label vocabularies are owned by issue 6-2/#56 and differ per channel, so these numbers move once
#: real channels exist.  They are recorded here so the width freeze is an auditable decision rather
#: than a guess, and they are asserted by the tests to catch silent architectural drift.
MEASURED_FIXTURE_PARAMETERS: dict[tuple[int, bool], int] = {
    (64, False): 185_732,
    (64, True): 210_692,
    (96, False): 343_044,
    (96, True): 398_916,
}


def parameter_target_status(total_parameters: int) -> str:
    """Classify a measured parameter count against the issue 6-3 target band.

    ``below_target`` is a legitimate, reportable outcome: the width-64 configuration with the
    fixture vocabulary measures roughly 0.19M, marginally under the stated ``0.2M`` lower bound.
    The correct response is to record it, not to pad the architecture until the number looks right.
    """

    low, high = PARAMETER_TARGET_RANGE
    if total_parameters < low:
        return "below_target"
    if total_parameters > high:
        return "above_target"
    return "within_target"


#: Documented, test-asserted deviations from the published components this encoder borrows from.
ENCODER_DEVIATIONS: tuple[str, ...] = (
    "epsilon is a learnable scalar per layer, matching GIN's trainable-epsilon variant rather "
    "than the fixed-zero simplification",
    "edge conditioning sums separate direction, ordered-slot, and role embeddings before the "
    "GINE-style ReLU(h_u + W a_uv) term, instead of consuming a single opaque edge vector",
    "each message-passing block is followed by a residual connection, a normalization layer, "
    "dropout, and a compact two-layer feed-forward block; published GIN does not specify this "
    "block structure",
    "readout is sum pooling over nodes only; the optional virtual node is added as a separate "
    "configuration-controlled global exchange and is not part of published GIN",
    "exact node values are encoded through a documented category/sign/log-magnitude/bucket path "
    "rather than as raw floats",
)


class EncoderConfigurationError(ValueError):
    """The requested encoder configuration violates the frozen issue 6-3 contract."""


class InvalidGraphBatchError(ValueError):
    """A graph batch is empty, ragged, or otherwise not consumable by the encoder."""


@runtime_checkable
class GraphBatchProtocol(Protocol):
    """The exact model-plane tensor fields this encoder consumes.

    This is a narrow *consumption* protocol, not a competing tensor schema.  Issue 6-2/#56 owns
    the canonical ``GraphTensorV1`` materializer; after that branch merges, its batch type should
    satisfy this protocol structurally and :class:`GraphBatch` below becomes fixture-only code.

    Field contract
    --------------
    ``node_kind``/``node_label``/``node_value_category``
        Long tensors of shape ``[num_nodes]`` holding vocabulary indices.
    ``node_value_sign``
        Long tensor ``[num_nodes]`` in ``{0, 1, 2}`` for negative/zero/positive.
    ``node_value_magnitude``
        Float tensor ``[num_nodes, 2]`` holding ``log1p(|numerator|)`` and ``log1p(|denominator|)``
        of the exact stored value.  Never a lossy decimal rendering of the value.
    ``node_value_bucket``
        Long tensor ``[num_nodes]``: a bounded hash bucket of the *exact* canonical value string,
        so distinct exact constants remain distinguishable without an unbounded vocabulary.  This
        is an approximation with a documented collision probability; the exact value itself is
        preserved in the source payload, never here.
    ``node_is_root``/``node_root_order``
        Long tensors ``[num_nodes]`` giving the root indicator and the ordered root index.  Root
        identity and root order are semantically material and must not be discarded.
    ``message_source``/``message_target``
        Long tensors ``[num_messages]``.  Every logical parent-to-child edge is materialized as
        two *message* edges with different direction labels.  The underlying logical edge is never
        mutated into an undirected edge, and repeated logical references are never deduplicated.
    ``message_direction``/``message_slot``/``message_role``
        Long tensors ``[num_messages]`` giving the direction label, the ordered child slot, and the
        message role.
    ``graph_index``
        Long tensor ``[num_nodes]`` mapping each node to its graph in ``[0, num_graphs)``.
    """

    node_kind: Tensor
    node_label: Tensor
    node_value_category: Tensor
    node_value_sign: Tensor
    node_value_magnitude: Tensor
    node_value_bucket: Tensor
    node_is_root: Tensor
    node_root_order: Tensor
    message_source: Tensor
    message_target: Tensor
    message_direction: Tensor
    message_slot: Tensor
    message_role: Tensor
    graph_index: Tensor
    num_graphs: int


@dataclass(frozen=True)
class GraphVocabularySizes:
    """Bounded vocabulary sizes for every embedded discrete field."""

    kind: int = 16
    label: int = 256
    value_category: int = 8
    value_bucket: int = 1024
    root_order: int = 8
    direction: int = 2
    slot: int = 8
    role: int = 4

    def __post_init__(self) -> None:
        for name, size in self.as_dict().items():
            if size < 1:
                raise EncoderConfigurationError(f"vocabulary {name} must be positive, got {size}")

    def as_dict(self) -> dict[str, int]:
        return {
            "kind": self.kind,
            "label": self.label,
            "value_category": self.value_category,
            "value_bucket": self.value_bucket,
            "root_order": self.root_order,
            "direction": self.direction,
            "slot": self.slot,
            "role": self.role,
        }


@dataclass(frozen=True)
class EncoderConfig:
    """The frozen, auditable encoder configuration."""

    hidden_width: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    normalization: str = "layer_norm"
    use_virtual_node: bool = False
    vocabulary: GraphVocabularySizes = GraphVocabularySizes()
    schema_version: str = ENCODER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.hidden_width not in PERMITTED_HIDDEN_WIDTHS:
            raise EncoderConfigurationError(
                f"hidden_width must be one of {PERMITTED_HIDDEN_WIDTHS}, got {self.hidden_width}"
            )
        if self.num_layers != 3:
            raise EncoderConfigurationError(
                "issue 6-3 freezes the encoder at three message-passing layers"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise EncoderConfigurationError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.normalization not in {"layer_norm", "graph_norm"}:
            raise EncoderConfigurationError(
                f"normalization must be 'layer_norm' or 'graph_norm', got {self.normalization!r}"
            )


@dataclass(frozen=True)
class EncoderOutput:
    """Directional encoder outputs reused by Goals 7-9."""

    node_embeddings: Tensor
    graph_embeddings: Tensor


class GraphNorm(nn.Module):
    """Per-graph mean/variance normalization with a learnable mean-subtraction weight.

    This follows the GraphNorm formulation (normalize within each graph in the batch rather than
    across the whole batch).  It is implemented directly here rather than imported, because the
    declared ``[ml]`` extra provides only ``torch``.
    """

    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.mean_scale = nn.Parameter(torch.ones(width))

    def forward(self, values: Tensor, graph_index: Tensor, num_graphs: int) -> Tensor:
        counts = torch.zeros(num_graphs, device=values.device, dtype=values.dtype)
        counts.index_add_(0, graph_index, torch.ones_like(graph_index, dtype=values.dtype))
        counts = counts.clamp(min=1.0).unsqueeze(-1)

        totals = torch.zeros(num_graphs, values.shape[-1], device=values.device, dtype=values.dtype)
        totals.index_add_(0, graph_index, values)
        mean = (totals / counts)[graph_index]

        centered = values - self.mean_scale * mean
        sq_totals = torch.zeros_like(totals)
        sq_totals.index_add_(0, graph_index, centered * centered)
        variance = (sq_totals / counts)[graph_index]
        return self.weight * centered / torch.sqrt(variance + self.eps) + self.bias


class _Normalization(nn.Module):
    """Dispatch wrapper so the message-passing block is normalization-agnostic."""

    def __init__(self, width: int, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.layer = nn.LayerNorm(width) if kind == "layer_norm" else GraphNorm(width)

    def forward(self, values: Tensor, graph_index: Tensor, num_graphs: int) -> Tensor:
        if self.kind == "layer_norm":
            return self.layer(values)
        return self.layer(values, graph_index, num_graphs)


class GineLayer(nn.Module):
    r"""One edge-aware GINE-style message-passing layer.

    The edge embedding for message edge :math:`u \to v` is

    .. math:: a_{uv} = E_{\mathrm{direction}}(d_{uv}) + E_{\mathrm{slot}}(s_{uv})
              + E_{\mathrm{role}}(r_{uv})

    and the pre-residual node update is

    .. math:: \tilde h_v^{(\ell+1)} = \mathrm{MLP}_\ell\left((1+\epsilon_\ell) h_v^{(\ell)}
              + \sum_{u \in \mathcal N(v)} \mathrm{ReLU}\left(h_u^{(\ell)}
              + W_\ell a_{uv}\right)\right).

    The sum is taken over incoming *message* edges, so a repeated logical reference contributes
    once per occurrence and is never collapsed.
    """

    def __init__(self, width: int, dropout: float, normalization: str) -> None:
        super().__init__()
        self.epsilon = nn.Parameter(torch.zeros(1))
        self.edge_projection = nn.Linear(width, width)
        self.message_mlp = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.norm = _Normalization(width, normalization)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.ReLU(),
            nn.Linear(2 * width, width),
        )
        self.feed_forward_norm = _Normalization(width, normalization)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_states: Tensor,
        edge_features: Tensor,
        source: Tensor,
        target: Tensor,
        graph_index: Tensor,
        num_graphs: int,
    ) -> Tensor:
        aggregated = torch.zeros_like(node_states)
        if source.numel() > 0:
            messages = torch.relu(node_states[source] + self.edge_projection(edge_features))
            aggregated = aggregated.index_add(0, target, messages)

        updated = self.message_mlp((1.0 + self.epsilon) * node_states + aggregated)
        updated = self.dropout(updated)
        node_states = self.norm(node_states + updated, graph_index, num_graphs)

        forwarded = self.dropout(self.feed_forward(node_states))
        return self.feed_forward_norm(node_states + forwarded, graph_index, num_graphs)


class CompactGraphEncoder(nn.Module):
    """The single shared encoder used by every graph channel."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        width = self.config.hidden_width
        vocab = self.config.vocabulary

        self.kind_embedding = nn.Embedding(vocab.kind, width)
        self.label_embedding = nn.Embedding(vocab.label, width)
        self.value_category_embedding = nn.Embedding(vocab.value_category, width)
        self.value_sign_embedding = nn.Embedding(3, width)
        self.value_bucket_embedding = nn.Embedding(vocab.value_bucket, width)
        self.value_magnitude_projection = nn.Linear(2, width)
        self.root_indicator_embedding = nn.Embedding(2, width)
        self.root_order_embedding = nn.Embedding(vocab.root_order, width)

        self.direction_embedding = nn.Embedding(vocab.direction, width)
        self.slot_embedding = nn.Embedding(vocab.slot, width)
        self.role_embedding = nn.Embedding(vocab.role, width)

        self.layers = nn.ModuleList(
            GineLayer(width, self.config.dropout, self.config.normalization)
            for _ in range(self.config.num_layers)
        )
        if self.config.use_virtual_node:
            self.virtual_node_mlp = nn.ModuleList(
                nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
                for _ in range(self.config.num_layers)
            )
        else:
            self.virtual_node_mlp = None

    @property
    def output_width(self) -> int:
        return self.config.hidden_width

    def _validate(self, batch: GraphBatchProtocol) -> None:
        num_nodes = int(batch.node_kind.shape[0])
        if num_nodes == 0:
            raise InvalidGraphBatchError("encoder received a batch with zero nodes")
        if batch.num_graphs <= 0:
            raise InvalidGraphBatchError("encoder received a batch with zero graphs")
        node_fields = {
            "node_label": batch.node_label,
            "node_value_category": batch.node_value_category,
            "node_value_sign": batch.node_value_sign,
            "node_value_bucket": batch.node_value_bucket,
            "node_is_root": batch.node_is_root,
            "node_root_order": batch.node_root_order,
            "graph_index": batch.graph_index,
        }
        for name, tensor in node_fields.items():
            if tensor.shape[0] != num_nodes:
                raise InvalidGraphBatchError(
                    f"{name} has {tensor.shape[0]} rows but the batch has {num_nodes} nodes"
                )
        if batch.node_value_magnitude.shape != (num_nodes, 2):
            raise InvalidGraphBatchError(
                "node_value_magnitude must have shape [num_nodes, 2]; exact numerator and "
                "denominator magnitudes may not be discarded"
            )
        num_messages = int(batch.message_source.shape[0])
        message_fields = {
            "message_target": batch.message_target,
            "message_direction": batch.message_direction,
            "message_slot": batch.message_slot,
            "message_role": batch.message_role,
        }
        for name, tensor in message_fields.items():
            if tensor.shape[0] != num_messages:
                raise InvalidGraphBatchError(
                    f"{name} has {tensor.shape[0]} rows but the batch has {num_messages} messages"
                )
        if num_messages > 0:
            endpoints = torch.cat([batch.message_source, batch.message_target])
            if int(endpoints.max()) >= num_nodes or int(endpoints.min()) < 0:
                raise InvalidGraphBatchError("message endpoints reference nodes outside the batch")
        if int(batch.graph_index.max()) >= batch.num_graphs or int(batch.graph_index.min()) < 0:
            raise InvalidGraphBatchError("graph_index references a graph outside the batch")

    def _node_features(self, batch: GraphBatchProtocol) -> Tensor:
        return (
            self.kind_embedding(batch.node_kind)
            + self.label_embedding(batch.node_label)
            + self.value_category_embedding(batch.node_value_category)
            + self.value_sign_embedding(batch.node_value_sign)
            + self.value_bucket_embedding(batch.node_value_bucket)
            + self.value_magnitude_projection(batch.node_value_magnitude)
            + self.root_indicator_embedding(batch.node_is_root)
            + self.root_order_embedding(batch.node_root_order)
        )

    def _edge_features(self, batch: GraphBatchProtocol) -> Tensor:
        return (
            self.direction_embedding(batch.message_direction)
            + self.slot_embedding(batch.message_slot)
            + self.role_embedding(batch.message_role)
        )

    def encode(self, batch: GraphBatchProtocol) -> EncoderOutput:
        """Return directional node-level and graph-level embeddings."""

        self._validate(batch)
        node_states = self._node_features(batch)
        edge_features = self._edge_features(batch)

        for index, layer in enumerate(self.layers):
            node_states = layer(
                node_states,
                edge_features,
                batch.message_source,
                batch.message_target,
                batch.graph_index,
                batch.num_graphs,
            )
            if self.virtual_node_mlp is not None:
                pooled = self._sum_pool(node_states, batch.graph_index, batch.num_graphs)
                node_states = node_states + self.virtual_node_mlp[index](pooled)[batch.graph_index]

        graph_embeddings = self._sum_pool(node_states, batch.graph_index, batch.num_graphs)
        return EncoderOutput(node_embeddings=node_states, graph_embeddings=graph_embeddings)

    @staticmethod
    def _sum_pool(values: Tensor, graph_index: Tensor, num_graphs: int) -> Tensor:
        pooled = torch.zeros(num_graphs, values.shape[-1], device=values.device, dtype=values.dtype)
        return pooled.index_add(0, graph_index, values)

    def forward(self, batch: GraphBatchProtocol) -> Tensor:
        return self.encode(batch).graph_embeddings


def symmetric_pair_features(left: Tensor, right: Tensor) -> Tensor:
    r"""Return the swap-invariant pair composition frozen by issue 6-3.

    .. math:: p(a, b) = \left[g_a + g_b,\; |g_a - g_b|,\; g_a \odot g_b\right]

    Every component is symmetric in its two arguments, so swap invariance holds *by construction*
    and not merely by training.  This function must not be used for directional proof prediction:
    a swap-invariant feature cannot distinguish a source from a goal.
    """

    if left.shape != right.shape:
        raise InvalidGraphBatchError("pair endpoints must have identical embedding shapes")
    return torch.cat([left + right, (left - right).abs(), left * right], dim=-1)


class EquivalencePairModel(nn.Module):
    """Shared-weight Siamese equivalence classifier over two graph endpoints."""

    def __init__(self, encoder: CompactGraphEncoder | None = None, head_width: int = 64) -> None:
        super().__init__()
        self.encoder = encoder or CompactGraphEncoder()
        width = self.encoder.output_width
        self.head = nn.Sequential(
            nn.Linear(3 * width, head_width),
            nn.ReLU(),
            nn.Linear(head_width, 1),
        )

    def forward(self, left: GraphBatchProtocol, right: GraphBatchProtocol) -> Tensor:
        if left.num_graphs != right.num_graphs:
            raise InvalidGraphBatchError(
                "pair endpoints must contain the same number of graphs; a channel that drops an "
                "endpoint belongs in the alignment failure ledger, not in a silently smaller batch"
            )
        left_embedding = self.encoder.encode(left).graph_embeddings
        right_embedding = self.encoder.encode(right).graph_embeddings
        return self.head(symmetric_pair_features(left_embedding, right_embedding)).squeeze(-1)


@dataclass(frozen=True)
class ParameterReport:
    """Code-measured parameter accounting for the frozen compute-matching policy."""

    total_parameters: int
    trainable_parameters: int
    per_component: dict[str, int]
    embedding_parameters: int

    def as_dict(self) -> dict[str, object]:
        return {
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "per_component": dict(sorted(self.per_component.items())),
            "embedding_parameters": self.embedding_parameters,
        }


def parameter_report(module: nn.Module) -> ParameterReport:
    """Measure parameters by code, never by hand-maintained constants."""

    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    per_component: dict[str, int] = {}
    for name, child in module.named_children():
        per_component[name] = sum(parameter.numel() for parameter in child.parameters())
    embedding = sum(
        parameter.numel()
        for child in module.modules()
        if isinstance(child, nn.Embedding)
        for parameter in child.parameters()
    )
    return ParameterReport(
        total_parameters=total,
        trainable_parameters=trainable,
        per_component=per_component,
        embedding_parameters=embedding,
    )


@dataclass(frozen=True)
class FlopReport:
    """A declared-workload forward-FLOP estimate with explicit limitations."""

    forward_flops: int
    num_nodes: int
    num_messages: int
    num_graphs: int
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "forward_flops": self.forward_flops,
            "num_nodes": self.num_nodes,
            "num_messages": self.num_messages,
            "num_graphs": self.num_graphs,
            "limitations": list(self.limitations),
        }


FLOP_LIMITATIONS: tuple[str, ...] = (
    "counts only dense matrix-multiply multiply-accumulate pairs as 2 FLOPs each",
    "embedding lookups, normalization, dropout, activation, and scatter-add are excluded",
    "this is an analytic estimate for a declared reference workload, not a hardware measurement",
    "training FLOPs are approximated downstream as three times the forward estimate",
)


def forward_flop_estimate(model: nn.Module, batch: GraphBatchProtocol) -> FlopReport:
    """Estimate forward FLOPs for one declared reference batch.

    Equal parameter counts do not imply equal compute, because pure-EML graphs are typically much
    larger than AST or macro graphs for the same expression.  The grid therefore matches on this
    measured estimate under one frozen reference workload, in addition to parameter count.
    """

    encoder = model.encoder if isinstance(model, EquivalencePairModel) else model
    if not isinstance(encoder, CompactGraphEncoder):
        raise EncoderConfigurationError("forward_flop_estimate requires a CompactGraphEncoder")

    width = encoder.output_width
    num_nodes = int(batch.node_kind.shape[0])
    num_messages = int(batch.message_source.shape[0])

    per_layer = 0
    per_layer += 2 * num_messages * width * width  # edge projection
    per_layer += 2 * num_nodes * width * width * 2  # two-layer message MLP
    per_layer += 2 * num_nodes * width * (2 * width) * 2  # two-layer feed-forward block
    total = per_layer * encoder.config.num_layers
    total += 2 * num_nodes * 2 * width  # exact-value magnitude projection
    if encoder.config.use_virtual_node:
        total += 2 * batch.num_graphs * width * width * 2 * encoder.config.num_layers
    if isinstance(model, EquivalencePairModel):
        head_in = 3 * width
        head_width = model.head[0].out_features
        total = 2 * total  # both endpoints pass through the shared encoder
        total += 2 * batch.num_graphs * (head_in * head_width + head_width)
    return FlopReport(
        forward_flops=int(total),
        num_nodes=num_nodes,
        num_messages=num_messages,
        num_graphs=int(batch.num_graphs),
        limitations=FLOP_LIMITATIONS,
    )


def value_magnitude_features(numerator: int, denominator: int = 1) -> tuple[float, float]:
    """Return the documented ``log1p`` magnitude pair for an exact rational value.

    Exact integers and rationals are never rendered as lossy floats.  The sign travels in
    ``node_value_sign`` and the exact canonical string travels in ``node_value_bucket``; this
    function supplies only the continuous magnitude channel.
    """

    if denominator == 0:
        raise InvalidGraphBatchError("exact rational values may not have a zero denominator")
    return (math.log1p(abs(numerator)), math.log1p(abs(denominator)))
