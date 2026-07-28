"""Prefix-transformer rewrite proposer using the shared legal-action output contract."""

from __future__ import annotations

from typing import Any

from geml.learning.policy.head import (
    LegalActionSetV1,
    PolicyContractError,
    validate_registered_rule_ids,
)

try:  # Keep policy contracts importable without torch.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - core-only path.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if torch is not None:

    class PrefixTransformerPolicyProposer(nn.Module):
        """Score exactly the same legal action set as the GNN policy head."""

        def __init__(self, encoder: nn.Module, *, registered_rule_ids: tuple[str, ...]) -> None:
            super().__init__()
            validate_registered_rule_ids(registered_rule_ids)
            hidden_width = getattr(encoder, "hidden_width", None)
            if hidden_width not in {64, 96}:
                raise PolicyContractError(
                    "proposer requires the compact 64- or 96-wide transformer"
                )
            self.encoder = encoder
            self.rule_to_index = {
                rule_id: index for index, rule_id in enumerate(registered_rule_ids)
            }
            self.rule_embedding = nn.Embedding(len(registered_rule_ids), hidden_width)
            self.slot_embedding = nn.Embedding(65, hidden_width)
            self.scorer = nn.Sequential(
                nn.Linear(hidden_width * 2, hidden_width),
                nn.GELU(),
                nn.Linear(hidden_width, 1),
            )

        def score(self, prefix_batch: Any, action_set: LegalActionSetV1) -> dict[str, float]:
            """Score legal actions for one prefix-serialized state without changing masks."""

            if prefix_batch.token_ids.shape[0] != 1:
                raise PolicyContractError("policy action scoring currently expects one state graph")
            state_embedding = self.encoder(prefix_batch)[0]
            scores: dict[str, float] = {}
            for action in action_set.legal_actions:
                slots = torch.tensor(
                    action.occurrence_path or (0,),
                    dtype=torch.long,
                    device=state_embedding.device,
                ).clamp_max(64)
                action_embedding = self.rule_embedding.weight[self.rule_to_index[action.rule_id]]
                action_embedding = action_embedding + self.slot_embedding(slots).mean(dim=0)
                scores[action.semantic_digest] = float(
                    self.scorer(torch.cat((state_embedding, action_embedding))).squeeze(0)
                )
            return scores

else:

    class PrefixTransformerPolicyProposer:  # pragma: no cover - optional-ML error path.
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("install GEML with `.[ml]` to use the prefix policy proposer")
