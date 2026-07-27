"""Issue 6-3: the compute-matched prefix-sequence transformer control.

This is a *real control*, not a deliberately weakened baseline.  It receives the same node kinds,
labels, exact values, root order, and ordered child structure that the graph encoder receives; the
only difference is that the graph is linearized into a prefix token sequence instead of being
consumed as a graph.  Any capacity difference between this control and the graph encoder must come
from the frozen parameter/FLOP matching policy, never from a quiet handicap.

Tokenization is a frozen, versioned contract (:data:`PREFIX_TOKENIZATION_VERSION`).  Exact integers
and rationals are serialized as structured sign/digit/solidus tokens, never as lossy float strings,
so the sequence control sees the same exact values the graph control sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor, nn

PREFIX_TOKENIZATION_VERSION = "geml-goal6-prefix-tokens-v1"
PREFIX_MODEL_SCHEMA_VERSION = "geml-goal6-prefix-transformer-v1"

#: Reserved control tokens.  Ordinary vocabulary entries start at :data:`FIRST_CONTENT_TOKEN`.
PAD_TOKEN = 0
START_TOKEN = 1
NODE_TOKEN = 2
ROOT_TOKEN = 3
SLOT_TOKEN = 4
VALUE_START_TOKEN = 5
NEGATIVE_TOKEN = 6
SOLIDUS_TOKEN = 7
DIGIT_TOKEN_BASE = 8  # digits 0-9 occupy 8..17
FIRST_CONTENT_TOKEN = 18


class TokenizationStatus(StrEnum):
    """Typed outcome of one serialization attempt; failures stay visible as rows."""

    OK = "ok"
    OVERLENGTH = "overlength"
    INVALID_TOKEN = "invalid_token"


class PrefixTokenizationError(ValueError):
    """A prefix sequence could not be produced under the frozen contract."""


@dataclass(frozen=True)
class SerializedPrefix:
    """One serialized graph together with its explicit outcome."""

    tokens: tuple[int, ...]
    status: TokenizationStatus
    truncated_from: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "tokens": list(self.tokens),
            "status": str(self.status),
            "truncated_from": self.truncated_from,
            "tokenization_version": PREFIX_TOKENIZATION_VERSION,
        }


@dataclass(frozen=True)
class PrefixNode:
    """The exact per-node information the tokenizer is allowed to see.

    Deliberately mirrors the model-plane allowlist: kind, label, exact value, root order, and
    ordered child slots.  No split, label, pair identity, or evaluation outcome appears here.
    """

    kind_id: int
    label_id: int
    value_numerator: int | None = None
    value_denominator: int | None = None
    root_order: int | None = None
    child_slots: tuple[int, ...] = ()


def _integer_tokens(value: int) -> list[int]:
    tokens: list[int] = []
    if value < 0:
        tokens.append(NEGATIVE_TOKEN)
    for character in str(abs(value)):
        tokens.append(DIGIT_TOKEN_BASE + int(character))
    return tokens


def serialize_prefix(
    nodes: tuple[PrefixNode, ...],
    max_length: int,
    kind_offset: int = FIRST_CONTENT_TOKEN,
    label_offset: int = FIRST_CONTENT_TOKEN + 32,
) -> SerializedPrefix:
    """Serialize a graph into the frozen prefix token contract.

    Overlength sequences are an explicit, recorded outcome.  They are truncated only so a batch can
    still be formed, and the original length is retained so the loss of information is auditable
    rather than silent.
    """

    if max_length < 2:
        raise PrefixTokenizationError("max_length must leave room for at least a start token")
    if not nodes:
        raise PrefixTokenizationError("cannot serialize a graph with zero nodes")

    tokens: list[int] = [START_TOKEN]
    for node in nodes:
        if node.kind_id < 0 or node.label_id < 0:
            return SerializedPrefix((), TokenizationStatus.INVALID_TOKEN)
        tokens.append(NODE_TOKEN)
        tokens.append(kind_offset + node.kind_id)
        tokens.append(label_offset + node.label_id)
        if node.root_order is not None:
            tokens.append(ROOT_TOKEN)
            tokens.extend(_integer_tokens(node.root_order))
        if node.value_numerator is not None:
            tokens.append(VALUE_START_TOKEN)
            tokens.extend(_integer_tokens(node.value_numerator))
            if node.value_denominator is not None and node.value_denominator != 1:
                tokens.append(SOLIDUS_TOKEN)
                tokens.extend(_integer_tokens(node.value_denominator))
        for slot in node.child_slots:
            tokens.append(SLOT_TOKEN)
            tokens.extend(_integer_tokens(slot))

    if len(tokens) > max_length:
        return SerializedPrefix(
            tuple(tokens[:max_length]),
            TokenizationStatus.OVERLENGTH,
            truncated_from=len(tokens),
        )
    return SerializedPrefix(tuple(tokens), TokenizationStatus.OK)


@dataclass(frozen=True)
class PrefixTransformerConfig:
    """Frozen configuration for the sequence control."""

    vocabulary_size: int = 256
    hidden_width: int = 64
    num_layers: int = 3
    num_heads: int = 4
    feedforward_width: int = 128
    dropout: float = 0.1
    max_length: int = 512
    schema_version: str = PREFIX_MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.hidden_width % self.num_heads != 0:
            raise PrefixTokenizationError("hidden_width must be divisible by num_heads")
        if self.max_length < 2:
            raise PrefixTokenizationError("max_length must be at least 2")


class PrefixSequenceEncoder(nn.Module):
    """A compact transformer encoder over prefix token sequences."""

    def __init__(self, config: PrefixTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or PrefixTransformerConfig()
        width = self.config.hidden_width
        self.token_embedding = nn.Embedding(
            self.config.vocabulary_size, width, padding_idx=PAD_TOKEN
        )
        self.register_buffer(
            "positional_encoding",
            _sinusoidal_positions(self.config.max_length, width),
            persistent=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_width,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.num_layers)

    @property
    def output_width(self) -> int:
        return self.config.hidden_width

    def forward(self, tokens: Tensor) -> Tensor:
        """Return one pooled embedding per sequence.

        ``tokens`` is a long tensor ``[batch, length]`` padded with :data:`PAD_TOKEN`.  Padding is
        masked out of both attention and the pooled mean, so padding length cannot leak into the
        representation.
        """

        if tokens.dim() != 2:
            raise PrefixTokenizationError("tokens must have shape [batch, length]")
        if tokens.shape[1] > self.config.max_length:
            raise PrefixTokenizationError(
                f"sequence length {tokens.shape[1]} exceeds the frozen max_length "
                f"{self.config.max_length}; overlength inputs must be recorded, not silently run"
            )
        padding_mask = tokens == PAD_TOKEN
        if bool(padding_mask.all(dim=1).any()):
            raise PrefixTokenizationError("a fully padded sequence has no content to encode")

        embedded = self.token_embedding(tokens)
        embedded = embedded + self.positional_encoding[: tokens.shape[1]].unsqueeze(0)
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)

        keep = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
        return (encoded * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)


def _sinusoidal_positions(length: int, width: int) -> Tensor:
    """Bounded, non-learned positional encoding."""

    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width)
    )
    encoding = torch.zeros(length, width)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding


class PrefixTransformerPairModel(nn.Module):
    """Swap-invariant equivalence classifier over two prefix sequences.

    Both endpoints pass through one shared encoder and are combined with the *same* symmetric
    composition used by the graph arm, so the two arms differ only in how the expression is
    represented, not in how the pair is formed.
    """

    def __init__(
        self,
        encoder: PrefixSequenceEncoder | None = None,
        head_width: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = encoder or PrefixSequenceEncoder()
        width = self.encoder.output_width
        self.head = nn.Sequential(
            nn.Linear(3 * width, head_width),
            nn.ReLU(),
            nn.Linear(head_width, 1),
        )

    def forward(self, left_tokens: Tensor, right_tokens: Tensor) -> Tensor:
        if left_tokens.shape[0] != right_tokens.shape[0]:
            raise PrefixTokenizationError("pair endpoints must have the same batch size")
        left = self.encoder(left_tokens)
        right = self.encoder(right_tokens)
        features = torch.cat([left + right, (left - right).abs(), left * right], dim=-1)
        return self.head(features).squeeze(-1)


def prefix_forward_flop_estimate(
    model: PrefixTransformerPairModel,
    batch_size: int,
    sequence_length: int,
) -> dict[str, object]:
    """Estimate forward FLOPs for the sequence control under a declared workload.

    Reported with the same limitations as the graph estimate so the two arms are matched on
    comparable quantities rather than on parameter count alone.
    """

    config = model.encoder.config
    width = config.hidden_width
    per_layer = 0
    per_layer += 2 * sequence_length * width * width * 4  # q, k, v, output projections
    per_layer += 2 * 2 * sequence_length * sequence_length * width  # attention scores and context
    per_layer += 2 * sequence_length * width * config.feedforward_width * 2  # feed-forward block
    total = per_layer * config.num_layers * batch_size
    total *= 2  # both endpoints pass through the shared encoder
    head_width = model.head[0].out_features
    total += 2 * batch_size * (3 * width * head_width + head_width)
    return {
        "forward_flops": int(total),
        "batch_size": int(batch_size),
        "sequence_length": int(sequence_length),
        "tokenization_version": PREFIX_TOKENIZATION_VERSION,
        "limitations": [
            "counts only dense matrix-multiply multiply-accumulate pairs as 2 FLOPs each",
            "embedding lookups, normalization, dropout, softmax, and masking are excluded",
            "an analytic estimate for a declared reference workload, not a hardware measurement",
        ],
    }
