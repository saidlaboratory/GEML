"""Shared typed legal-action masking and optional compact-GNN rewrite policy head."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from geml.data.pairs.generate import RewriteActionV1

try:  # Keep contracts usable without the optional ML extra.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - core-only path.
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


class PolicyContractError(ValueError):
    """A candidate action, registry, mask, or typed proposal is unsafe."""


class ProposalStatus(StrEnum):
    PROPOSED = "proposed"
    NO_LEGAL_ACTION = "no_legal_action"
    INVALID_STATE = "invalid_state"


def validate_registered_rule_ids(rule_ids: tuple[str, ...]) -> None:
    """Reject an ambiguous or empty registry order before it reaches either proposer."""

    if not rule_ids or tuple(sorted(set(rule_ids))) != rule_ids:
        raise PolicyContractError("registered_rule_ids must be nonempty, sorted, and unique")


@dataclass(frozen=True, slots=True)
class LegalActionSetV1:
    """Registry-derived legal actions for one exact structural state."""

    state_structural_signature: str
    registered_rule_ids: tuple[str, ...]
    legal_actions: tuple[RewriteActionV1, ...]

    def __post_init__(self) -> None:
        if not self.state_structural_signature.strip():
            raise PolicyContractError("state_structural_signature must be nonblank")
        validate_registered_rule_ids(self.registered_rule_ids)
        action_ids = tuple(action.semantic_digest for action in self.legal_actions)
        if len(set(action_ids)) != len(action_ids):
            raise PolicyContractError("legal actions must have distinct semantic digests")
        for action in self.legal_actions:
            if action.rule_id not in self.registered_rule_ids:
                raise PolicyContractError("legal action uses an unregistered rule")
            if action.source_structural_signature != self.state_structural_signature:
                raise PolicyContractError("legal action source must match this exact state")

    @property
    def legal_action_digests(self) -> tuple[str, ...]:
        return tuple(action.semantic_digest for action in self.legal_actions)


@dataclass(frozen=True, slots=True)
class ActionProposalV1:
    """A calibrated top-k proposal; only legal registered actions can appear here."""

    action: RewriteActionV1
    rank: int
    score: float
    probability: float

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise PolicyContractError("proposal rank must be nonnegative")
        if not math.isfinite(self.score) or not math.isfinite(self.probability):
            raise PolicyContractError("proposal score and probability must be finite")
        if not 0.0 <= self.probability <= 1.0:
            raise PolicyContractError("proposal probability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PolicyOutputV1:
    """One family-neutral top-k result consumed unchanged by metrics and search."""

    state_structural_signature: str
    status: ProposalStatus
    proposals: tuple[ActionProposalV1, ...]
    masked_action_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.state_structural_signature.strip():
            raise PolicyContractError("state_structural_signature must be nonblank")
        if self.status is ProposalStatus.PROPOSED:
            if not self.proposals:
                raise PolicyContractError("a proposed output must contain at least one proposal")
            ranks = tuple(item.rank for item in self.proposals)
            if ranks != tuple(range(len(self.proposals))):
                raise PolicyContractError("proposal ranks must be dense and ordered")
            probabilities = sum(item.probability for item in self.proposals)
            if probabilities > 1.0 + 1e-7:
                raise PolicyContractError("top-k proposal mass cannot exceed one")
        elif self.proposals:
            raise PolicyContractError(
                "no-action and invalid-state outputs cannot contain proposals"
            )
        if tuple(sorted(set(self.masked_action_digests))) != self.masked_action_digests:
            raise PolicyContractError("masked action digests must be sorted and unique")


def masked_top_k(
    action_set: LegalActionSetV1,
    *,
    candidate_scores: dict[str, float],
    top_k: int,
) -> PolicyOutputV1:
    """Calibrate scores over exactly the registry-derived legal-action mask.

    Candidate action IDs outside the legal set receive no probability mass and
    are retained as masked IDs. Missing legal-action scores are rejected rather
    than silently imputed, which prevents model-specific mask behavior.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    legal_by_digest = {action.semantic_digest: action for action in action_set.legal_actions}
    masked = tuple(sorted(set(candidate_scores) - set(legal_by_digest)))
    if not legal_by_digest:
        return PolicyOutputV1(
            state_structural_signature=action_set.state_structural_signature,
            status=ProposalStatus.NO_LEGAL_ACTION,
            proposals=(),
            masked_action_digests=masked,
        )
    missing = sorted(set(legal_by_digest) - set(candidate_scores))
    if missing:
        raise PolicyContractError(f"model did not score every legal action: {missing}")
    if any(not math.isfinite(candidate_scores[digest]) for digest in legal_by_digest):
        raise PolicyContractError("legal action scores must be finite")
    maximum = max(candidate_scores[digest] for digest in legal_by_digest)
    unnormalized = {
        digest: math.exp(candidate_scores[digest] - maximum) for digest in legal_by_digest
    }
    normalizer = sum(unnormalized.values())
    ordered = sorted(
        legal_by_digest,
        key=lambda digest: (-candidate_scores[digest], digest),
    )[:top_k]
    return PolicyOutputV1(
        state_structural_signature=action_set.state_structural_signature,
        status=ProposalStatus.PROPOSED,
        proposals=tuple(
            ActionProposalV1(
                action=legal_by_digest[digest],
                rank=rank,
                score=candidate_scores[digest],
                probability=unnormalized[digest] / normalizer,
            )
            for rank, digest in enumerate(ordered)
        ),
        masked_action_digests=masked,
    )


if torch is not None:

    class SharedGNNPolicyHead(nn.Module):
        """Rule/site scorer that reuses an existing compact GNN encoder by reference.

        Per-action features are the registered rule index, the ordered slot path,
        and binding count. The caller still derives legal candidates from the rule
        registry; this head cannot create an unregistered action.
        """

        def __init__(self, encoder: nn.Module, *, registered_rule_ids: tuple[str, ...]) -> None:
            super().__init__()
            validate_registered_rule_ids(registered_rule_ids)
            hidden_width = getattr(encoder, "hidden_width", None)
            if hidden_width not in {64, 96}:
                raise PolicyContractError(
                    "policy requires the shared compact 64- or 96-wide encoder"
                )
            self.encoder = encoder
            self.hidden_width = hidden_width
            self.rule_to_index = {
                rule_id: index for index, rule_id in enumerate(registered_rule_ids)
            }
            self.rule_embedding = nn.Embedding(len(registered_rule_ids), hidden_width)
            self.slot_embedding = nn.Embedding(65, hidden_width)
            self.binding_projection = nn.Linear(1, hidden_width)
            self.scorer = nn.Sequential(
                nn.Linear(hidden_width * 2, hidden_width),
                nn.GELU(),
                nn.Linear(hidden_width, 1),
            )

        def score(self, graph_batch: object, action_set: LegalActionSetV1) -> dict[str, float]:
            """Score all legal actions for one graph; batching remains at the encoder boundary."""

            if getattr(graph_batch, "graph_count", None) != 1:
                raise PolicyContractError("policy action scoring currently expects one state graph")
            encoding = self.encoder(graph_batch)
            graph_embedding = encoding.graph_embeddings[0]
            scores: dict[str, float] = {}
            for action in action_set.legal_actions:
                rule_index = self.rule_to_index[action.rule_id]
                slots = torch.tensor(
                    action.occurrence_path or (0,),
                    dtype=torch.long,
                    device=graph_embedding.device,
                ).clamp_max(64)
                action_embedding = self.rule_embedding.weight[rule_index]
                action_embedding = action_embedding + self.slot_embedding(slots).mean(dim=0)
                binding_count = torch.tensor(
                    [[len(action.bindings)]],
                    dtype=graph_embedding.dtype,
                    device=graph_embedding.device,
                )
                action_embedding = action_embedding + self.binding_projection(
                    binding_count
                ).squeeze(0)
                scores[action.semantic_digest] = float(
                    self.scorer(torch.cat((graph_embedding, action_embedding))).squeeze(0)
                )
            return scores

else:

    class SharedGNNPolicyHead:  # pragma: no cover - optional-ML error path.
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("install GEML with `.[ml]` to use the compact GNN policy head")
