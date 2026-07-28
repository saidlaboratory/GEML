"""Goal-conditioned proposer adapter for the shared prefix transformer.

This module does not implement another transformer or rule registry.  It adds
explicit current/separator/goal roles to already-tokenized prefix inputs, calls
the injected Workstream-2 backbone, and delegates concrete-action scoring and
proposal construction to :mod:`geml.learning.policy.head`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from geml.learning.policy.head import (
    ActionFeatureResolverProtocol,
    ActionInventoryV1,
    FactoredActionScorer,
    GraphEncodingProtocol,
    InvalidPolicyStateError,
    ModelIdentityV1,
    PolicyScoreBatchV1,
    ProposalV1,
    RuleVocabularyV1,
    UnsupportedPolicyStateError,
    _ModuleBase,
    _propose_with_scorer,
    _raise_for_inventory_status,
    _require_torch,
    nn,
    torch,
)


@dataclass(frozen=True, slots=True)
class RoleTokenIdsV1:
    """Reserved prefix-token IDs used to preserve current/goal direction."""

    current: int
    separator: int
    goal: int
    padding: int

    def __post_init__(self) -> None:
        values = (self.current, self.separator, self.goal, self.padding)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise TypeError("role and padding token IDs must be nonnegative integers")
        if len(set(values)) != len(values):
            raise ValueError("current, separator, goal, and padding token IDs must be distinct")

    @property
    def reserved(self) -> frozenset[int]:
        return frozenset((self.current, self.separator, self.goal, self.padding))


@dataclass(frozen=True, slots=True)
class RoleSeparatedPrefixV1:
    """One masked transformer input with explicit directional roles."""

    token_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    current_positions: tuple[int, ...]
    goal_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, tuple) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.token_ids
        ):
            raise TypeError("token_ids must be a tuple of nonnegative integers")
        if (
            not isinstance(self.attention_mask, tuple)
            or len(self.attention_mask) != len(self.token_ids)
            or any(not isinstance(value, bool) for value in self.attention_mask)
        ):
            raise TypeError("attention_mask must be a bool tuple aligned with token_ids")
        for name, positions in (
            ("current_positions", self.current_positions),
            ("goal_positions", self.goal_positions),
        ):
            if (
                not isinstance(positions, tuple)
                or not positions
                or any(
                    isinstance(position, bool)
                    or not isinstance(position, int)
                    or position < 0
                    or position >= len(self.token_ids)
                    for position in positions
                )
            ):
                raise TypeError(f"{name} must contain valid nonnegative token positions")
            if tuple(sorted(set(positions))) != positions:
                raise ValueError(f"{name} must be strictly increasing")
            if any(not self.attention_mask[position] for position in positions):
                raise ValueError(f"{name} cannot reference masked padding")
        if set(self.current_positions) & set(self.goal_positions):
            raise ValueError("current and goal token positions must be disjoint")
        observed_padding = False
        for attended in self.attention_mask:
            if not attended:
                observed_padding = True
            elif observed_padding:
                raise ValueError("masked padding must be a suffix")

    @classmethod
    def build(
        cls,
        current_tokens: tuple[int, ...],
        goal_tokens: tuple[int, ...],
        *,
        role_tokens: RoleTokenIdsV1,
        pad_to_length: int | None = None,
    ) -> RoleSeparatedPrefixV1:
        """Build ``CURRENT current SEP GOAL goal [PAD...]`` deterministically."""

        _validate_state_tokens(current_tokens, role_tokens, "current_tokens")
        _validate_state_tokens(goal_tokens, role_tokens, "goal_tokens")
        if not current_tokens or not goal_tokens:
            raise InvalidPolicyStateError("current and goal token sequences must be nonempty")
        unpadded = (
            role_tokens.current,
            *current_tokens,
            role_tokens.separator,
            role_tokens.goal,
            *goal_tokens,
        )
        if pad_to_length is None:
            length = len(unpadded)
        else:
            if (
                isinstance(pad_to_length, bool)
                or not isinstance(pad_to_length, int)
                or pad_to_length < len(unpadded)
            ):
                raise InvalidPolicyStateError(
                    "pad_to_length must be an integer at least the unpadded sequence length"
                )
            length = pad_to_length
        padding_count = length - len(unpadded)
        current_start = 1
        goal_start = len(current_tokens) + 3
        return cls(
            token_ids=(*unpadded, *((role_tokens.padding,) * padding_count)),
            attention_mask=(*((True,) * len(unpadded)), *((False,) * padding_count)),
            current_positions=tuple(range(current_start, current_start + len(current_tokens))),
            goal_positions=tuple(range(goal_start, goal_start + len(goal_tokens))),
        )


def _validate_state_tokens(
    tokens: object,
    role_tokens: RoleTokenIdsV1,
    name: str,
) -> None:
    if not isinstance(tokens, tuple) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens
    ):
        raise TypeError(f"{name} must be a tuple of nonnegative integer token IDs")
    collision = role_tokens.reserved.intersection(tokens)
    if collision:
        raise InvalidPolicyStateError(
            f"{name} contains reserved role/padding tokens: {sorted(collision)}"
        )


@runtime_checkable
class PrefixPairEncodingProtocol(Protocol):
    """Minimum directional output of the injected prefix transformer."""

    current_embedding: object
    goal_embedding: object
    current_token_embeddings: object


@runtime_checkable
class SharedPrefixEncoderProtocol(Protocol):
    """Injected Workstream-2 prefix-transformer boundary."""

    def forward(self, pair: RoleSeparatedPrefixV1) -> PrefixPairEncodingProtocol:
        """Encode one role-separated, masked current/goal prefix pair."""


@dataclass(frozen=True, slots=True)
class _CurrentPrefixEncoding:
    """Adapter exposing current tokens through the common resolver protocol."""

    graph_embedding: object
    node_embeddings: object


class PrefixTransformerProposer(_ModuleBase):
    """Concrete-action proposer reusing the shared compact prefix transformer."""

    def __init__(
        self,
        *,
        encoder: SharedPrefixEncoderProtocol,
        action_resolver: ActionFeatureResolverProtocol,
        vocabulary: RuleVocabularyV1,
        model_identity: ModelIdentityV1,
        role_tokens: RoleTokenIdsV1,
        hidden_width: int,
        max_arguments: int = 8,
        probability_temperature: float = 1.0,
    ) -> None:
        _require_torch()
        super().__init__()
        if model_identity.model_family != "prefix_transformer":
            raise ValueError("PrefixTransformerProposer requires model_family='prefix_transformer'")
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be an injected torch.nn.Module")
        self.encoder = encoder
        self.action_resolver = action_resolver
        self.vocabulary = vocabulary
        self.model_identity = model_identity
        self.role_tokens = role_tokens
        self.scorer = FactoredActionScorer(
            hidden_width=hidden_width,
            vocabulary=vocabulary,
            max_arguments=max_arguments,
            probability_temperature=probability_temperature,
        )

    def score_inventory(
        self,
        current_tokens: tuple[int, ...],
        goal_tokens: tuple[int, ...],
        inventory: ActionInventoryV1,
        *,
        pad_to_length: int | None = None,
    ) -> PolicyScoreBatchV1:
        """Return differentiable scores under the shared action inventory/mask."""

        if not isinstance(inventory, ActionInventoryV1):
            raise TypeError("inventory must be an ActionInventoryV1")
        _raise_for_inventory_status(inventory)
        if inventory.unknown_rule_ids:
            raise UnsupportedPolicyStateError(
                "inventory contains unregistered rules: " + ", ".join(inventory.unknown_rule_ids)
            )
        if inventory.legal_action_count == 0:
            empty_or_zero = next(self.parameters()).new_zeros((len(inventory.actions),))
            mask = torch.tensor(
                inventory.legal_mask,
                dtype=torch.bool,
                device=empty_or_zero.device,
            )
            return PolicyScoreBatchV1(
                logits=empty_or_zero,
                probabilities=empty_or_zero,
                legal_mask=mask,
            )
        pair = RoleSeparatedPrefixV1.build(
            current_tokens,
            goal_tokens,
            role_tokens=self.role_tokens,
            pad_to_length=pad_to_length,
        )
        encoded = self.encoder(pair)
        if not isinstance(encoded, PrefixPairEncodingProtocol):
            raise TypeError("prefix encoder output must implement PrefixPairEncodingProtocol")
        current_encoding: GraphEncodingProtocol = _CurrentPrefixEncoding(
            graph_embedding=encoded.current_embedding,
            node_embeddings=encoded.current_token_embeddings,
        )
        return self.scorer(
            encoded.current_embedding,
            encoded.goal_embedding,
            current_encoding,
            inventory,
            self.action_resolver,
        )

    def forward(
        self,
        current_tokens: tuple[int, ...],
        goal_tokens: tuple[int, ...],
        inventory: ActionInventoryV1,
        *,
        pad_to_length: int | None = None,
    ) -> PolicyScoreBatchV1:
        return self.score_inventory(
            current_tokens,
            goal_tokens,
            inventory,
            pad_to_length=pad_to_length,
        )

    def propose(
        self,
        current_tokens: tuple[int, ...],
        goal_tokens: tuple[int, ...],
        inventory: ActionInventoryV1,
        *,
        top_k: int,
        pad_to_length: int | None = None,
    ) -> ProposalV1:
        """Return the same typed output and tie policy as the GNN proposer."""

        return _propose_with_scorer(
            inventory,
            top_k=top_k,
            model_identity=self.model_identity,
            probability_temperature=self.scorer.probability_temperature,
            score=lambda: self.score_inventory(
                current_tokens,
                goal_tokens,
                inventory,
                pad_to_length=pad_to_length,
            ),
        )
