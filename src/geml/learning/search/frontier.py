"""Deterministic bounded frontier shared by every Goal 8 search mode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class SearchMode(StrEnum):
    """Supported search policies; terminal and budget semantics are mode-independent."""

    UNIFORM = "uniform"
    POLICY = "policy"
    POLICY_VALUE = "policy_value"
    TRANSFORMER = "transformer"


@dataclass(frozen=True, slots=True)
class FrontierEntry:
    """One frontier state with a stable, fully ordered priority."""

    priority: tuple[float, int, str, str]
    state_signature: str
    encoded_state: Any
    depth: int
    path_cost: float

    def __post_init__(self) -> None:
        if type(self.priority) is not tuple or len(self.priority) != 4:
            raise TypeError("frontier priority must be an exact four-field tuple")
        score, priority_depth, signature, action_digest = self.priority
        if type(score) is not float or not math.isfinite(score):
            raise TypeError("frontier priority score must be a finite float")
        if type(priority_depth) is not int or priority_depth < 0:
            raise TypeError("frontier priority depth must be a non-negative integer")
        if type(signature) is not str or not signature:
            raise TypeError("frontier priority signature must be a non-blank string")
        if type(action_digest) is not str or not action_digest:
            raise TypeError("frontier priority action identity must be a non-blank string")
        if type(self.state_signature) is not str or not self.state_signature:
            raise TypeError("state_signature must be a non-blank string")
        if type(self.depth) is not int or self.depth < 0:
            raise TypeError("depth must be a non-negative integer")
        if priority_depth != self.depth or signature != self.state_signature:
            raise ValueError("frontier priority depth/signature must match the entry")
        if (
            type(self.path_cost) is not float
            or not math.isfinite(self.path_cost)
            or self.path_cost < 0
        ):
            raise TypeError("path_cost must be a finite non-negative float")

    def to_json(self) -> dict[str, Any]:
        """Return a checkpoint-safe representation."""

        return {
            "priority": list(self.priority),
            "state_signature": self.state_signature,
            "encoded_state": self.encoded_state,
            "depth": self.depth,
            "path_cost": self.path_cost,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> FrontierEntry:
        """Restore an entry written by :meth:`to_json`."""

        if type(payload) is not dict:
            raise TypeError("checkpoint frontier entry must be an object")
        expected_fields = {
            "priority",
            "state_signature",
            "encoded_state",
            "depth",
            "path_cost",
        }
        if set(payload) != expected_fields:
            raise ValueError("checkpoint frontier entry fields are incomplete or unknown")
        raw_priority = payload["priority"]
        if type(raw_priority) is not list or len(raw_priority) != 4:
            raise ValueError("frontier priority must contain exactly four fields")
        return cls(
            priority=(
                raw_priority[0],
                raw_priority[1],
                raw_priority[2],
                raw_priority[3],
            ),
            state_signature=payload["state_signature"],
            encoded_state=payload["encoded_state"],
            depth=payload["depth"],
            path_cost=payload["path_cost"],
        )


class DeterministicFrontier:
    """A small bounded priority frontier with canonical tie-breaking.

    The lowest priority is expanded first. When the frontier exceeds ``beam_width``,
    the deterministically worst entry is evicted. Search modes differ only in how
    they produce the first priority component.
    """

    def __init__(self, beam_width: int) -> None:
        if type(beam_width) is not int or beam_width <= 0:
            raise TypeError("beam_width must be a positive integer")
        self._beam_width = beam_width
        self._entries: list[FrontierEntry] = []

    @property
    def beam_width(self) -> int:
        return self._beam_width

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[FrontierEntry, ...]:
        """Return an immutable ordered view for checkpoint-integrity checks."""

        return tuple(self._entries)

    def push(self, entry: FrontierEntry) -> FrontierEntry | None:
        """Add ``entry`` and return an evicted entry, if the beam overflowed."""

        self._entries.append(entry)
        self._entries.sort(key=lambda item: item.priority)
        if len(self._entries) <= self._beam_width:
            return None
        return self._entries.pop()

    def pop(self) -> FrontierEntry:
        """Remove the deterministically best entry."""

        if not self._entries:
            raise IndexError("pop from empty frontier")
        return self._entries.pop(0)

    def snapshot(self) -> list[dict[str, Any]]:
        """Return the exact ordered frontier for checkpointing."""

        return [entry.to_json() for entry in self._entries]

    @classmethod
    def restore(
        cls,
        *,
        beam_width: int,
        payload: list[dict[str, Any]],
        maximum_depth: int | None = None,
    ) -> DeterministicFrontier:
        """Restore an exact ordered frontier and reject malformed checkpoints."""

        if type(payload) is not list:
            raise TypeError("checkpoint frontier must be a list")
        if maximum_depth is not None and (type(maximum_depth) is not int or maximum_depth < 0):
            raise TypeError("maximum_depth must be a non-negative integer or None")
        frontier = cls(beam_width)
        entries = [FrontierEntry.from_json(item) for item in payload]
        if entries != sorted(entries, key=lambda item: item.priority):
            raise ValueError("checkpoint frontier is not in canonical priority order")
        if len(entries) > beam_width:
            raise ValueError("checkpoint frontier exceeds the configured beam width")
        if maximum_depth is not None and any(entry.depth > maximum_depth for entry in entries):
            raise ValueError("checkpoint frontier exceeds the configured proof-depth budget")
        frontier._entries = entries
        return frontier


__all__ = [
    "DeterministicFrontier",
    "FrontierEntry",
    "SearchMode",
]
