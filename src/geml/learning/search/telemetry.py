"""Typed telemetry for verifier-gated Goal 8 proof search."""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import psutil


@dataclass(frozen=True, slots=True)
class SearchTelemetryV1:
    """Immutable persisted telemetry included in every search result."""

    expanded: int
    generated: int
    valid: int
    invalid: int
    unsupported: int
    duplicate: int
    beam_pruned: int
    verifier_calls: int
    verifier_cache_hits: int
    verifier_attempt_reuses: int
    verifier_failures: int
    verifier_timeouts: int
    adapter_failures: int
    frontier_peak: int
    max_depth_reached: int
    proof_length: int | None
    model_scoring_batches: int
    model_scoring_seconds: float
    expanded_state_signatures: tuple[str, ...]
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        integer_values = {
            "expanded": self.expanded,
            "generated": self.generated,
            "valid": self.valid,
            "invalid": self.invalid,
            "unsupported": self.unsupported,
            "duplicate": self.duplicate,
            "beam_pruned": self.beam_pruned,
            "verifier_calls": self.verifier_calls,
            "verifier_cache_hits": self.verifier_cache_hits,
            "verifier_attempt_reuses": self.verifier_attempt_reuses,
            "verifier_failures": self.verifier_failures,
            "verifier_timeouts": self.verifier_timeouts,
            "adapter_failures": self.adapter_failures,
            "frontier_peak": self.frontier_peak,
            "max_depth_reached": self.max_depth_reached,
            "model_scoring_batches": self.model_scoring_batches,
            "peak_rss_bytes": self.peak_rss_bytes,
        }
        invalid_integer = next(
            (name for name, value in integer_values.items() if type(value) is not int or value < 0),
            None,
        )
        if invalid_integer is not None:
            raise TypeError(f"telemetry counter {invalid_integer} must be a non-negative integer")
        if (
            type(self.model_scoring_seconds) is not float
            or not math.isfinite(self.model_scoring_seconds)
            or self.model_scoring_seconds < 0
        ):
            raise TypeError("model_scoring_seconds must be a finite non-negative float")
        if type(self.expanded_state_signatures) is not tuple or any(
            type(signature) is not str or not signature
            for signature in self.expanded_state_signatures
        ):
            raise TypeError("expanded_state_signatures must be a tuple of non-blank strings")
        if len(self.expanded_state_signatures) != self.expanded:
            raise ValueError("expanded state order must align with expanded count")
        if self.proof_length is not None and (
            type(self.proof_length) is not int or self.proof_length < 0
        ):
            raise TypeError("proof_length must be a non-negative integer or None")
        if self.proof_length is not None and self.proof_length > self.max_depth_reached:
            raise ValueError("proof_length cannot exceed max_depth_reached")
        # Transition outcomes consume generated rows; an additional terminal
        # verification outcome may be retained for an exact-goal candidate.
        if self.valid + self.invalid + self.unsupported > self.generated + 1:
            raise ValueError("verification outcome counts exceed generated plus terminal rows")

    @property
    def attempted_transitions(self) -> int:
        """Persisted denominator for transition-level rates."""

        return self.generated

    @property
    def valid_transitions(self) -> int:
        """Persisted numerator for verifier-valid transition rates."""

        return self.valid


@dataclass(slots=True)
class SearchTelemetry:
    """Mutable counters whose snapshot is persisted in ``SearchResultV1``."""

    expanded: int = 0
    generated: int = 0
    valid: int = 0
    invalid: int = 0
    unsupported: int = 0
    duplicate: int = 0
    beam_pruned: int = 0
    verifier_calls: int = 0
    verifier_cache_hits: int = 0
    verifier_attempt_reuses: int = 0
    verifier_failures: int = 0
    verifier_timeouts: int = 0
    adapter_failures: int = 0
    frontier_peak: int = 0
    max_depth_reached: int = 0
    proof_length: int | None = None
    model_scoring_batches: int = 0
    model_scoring_seconds: float = 0.0
    expanded_state_signatures: list[str] = field(default_factory=list)
    peak_rss_bytes: int = 0

    @property
    def attempted_transitions(self) -> int:
        """Number of generated candidates retained in the scientific denominator."""

        return self.generated

    @property
    def valid_transitions(self) -> int:
        return self.valid

    def observe_frontier(self, size: int) -> None:
        if type(size) is not int or size < 0:
            raise TypeError("observed frontier size must be a non-negative integer")
        self.frontier_peak = max(self.frontier_peak, size)

    def observe_depth(self, depth: int) -> None:
        """Record the deepest verifier-valid structural state reached."""

        if type(depth) is not int or depth < 0:
            raise TypeError("observed search depth must be a non-negative integer")
        self.max_depth_reached = max(self.max_depth_reached, depth)

    def observe_resources(self) -> None:
        """Sample resident memory without making search correctness depend on it."""

        try:
            rss = psutil.Process(os.getpid()).memory_info().rss
        except (OSError, psutil.Error):
            return
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)

    def record_model_batch(self, started_at: float) -> None:
        self.model_scoring_batches += 1
        self.model_scoring_seconds += max(0.0, time.monotonic() - started_at)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def snapshot(self) -> SearchTelemetryV1:
        return SearchTelemetryV1(
            expanded=self.expanded,
            generated=self.generated,
            valid=self.valid,
            invalid=self.invalid,
            unsupported=self.unsupported,
            duplicate=self.duplicate,
            beam_pruned=self.beam_pruned,
            verifier_calls=self.verifier_calls,
            verifier_cache_hits=self.verifier_cache_hits,
            verifier_attempt_reuses=self.verifier_attempt_reuses,
            verifier_failures=self.verifier_failures,
            verifier_timeouts=self.verifier_timeouts,
            adapter_failures=self.adapter_failures,
            frontier_peak=self.frontier_peak,
            max_depth_reached=self.max_depth_reached,
            proof_length=self.proof_length,
            model_scoring_batches=self.model_scoring_batches,
            model_scoring_seconds=self.model_scoring_seconds,
            expanded_state_signatures=tuple(self.expanded_state_signatures),
            peak_rss_bytes=self.peak_rss_bytes,
        )

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> SearchTelemetry:
        if type(payload) is not dict:
            raise TypeError("checkpoint telemetry must be an object")
        known_fields = cls.__dataclass_fields__
        unknown = set(payload).difference(known_fields)
        if unknown:
            raise ValueError(f"unknown telemetry fields: {sorted(unknown)}")
        telemetry = cls(**payload)
        # Reuse the immutable snapshot's complete type, range, and relationship checks.
        telemetry.snapshot()
        return telemetry


__all__ = ["SearchTelemetry", "SearchTelemetryV1"]
