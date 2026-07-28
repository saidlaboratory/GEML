"""Immutable, leakage-free selected proof problems for Goal 8."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from geml.data.pairs.generate import (
    PairRecordV1,
    PairStatus,
    ReplayStatus,
    TransitionVerifier,
    replay_trace,
)
from geml.data.proofs.tiers import ProofTier

BENCHMARK_SCHEMA_VERSION = "geml-goal8-proof-benchmark-v1"


class ProofBenchmarkError(ValueError):
    """A benchmark candidate or immutable frozen selection is scientifically unsafe."""


@dataclass(frozen=True, slots=True)
class ProofCandidateV1:
    pair: PairRecordV1
    tier: ProofTier

    def __post_init__(self) -> None:
        accepted_positive_trace = (
            self.pair.status is PairStatus.ACCEPTED
            and self.pair.label is True
            and self.pair.trace is not None
        )
        if not accepted_positive_trace:
            raise ProofBenchmarkError(
                "benchmark candidates require accepted positive replayable pairs"
            )

    @property
    def family(self) -> str:
        return self.pair.left.operator_family

    @property
    def group_closure(self) -> tuple[str, ...]:
        return self.pair.pair_group_set


@dataclass(frozen=True, slots=True)
class FrozenProofProblemV1:
    problem_id: str
    pair_id: str
    source_expression_id: str
    target_expression_id: str
    family: str
    tier: ProofTier
    domain_mode: str
    group_closure: tuple[str, ...]
    trace_digest: str


@dataclass(frozen=True, slots=True)
class ProofBenchmarkManifestV1:
    schema_version: str
    content_digest: str
    problems: tuple[FrozenProofProblemV1, ...]
    excluded_count: int

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_SCHEMA_VERSION:
            raise ProofBenchmarkError("unexpected proof benchmark schema")
        if len({problem.problem_id for problem in self.problems}) != len(self.problems):
            raise ProofBenchmarkError("proof problem IDs must be unique")
        if self.content_digest != benchmark_digest(self.problems):
            raise ProofBenchmarkError("benchmark digest does not bind frozen problem rows")


def benchmark_digest(problems: Iterable[FrozenProofProblemV1]) -> str:
    payload = [asdict(problem) | {"tier": problem.tier.value} for problem in problems]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def freeze_benchmark(
    candidates: Iterable[ProofCandidateV1],
    *,
    quotas: Mapping[tuple[str, ProofTier], int],
    training_group_ids: Iterable[str],
    verifier: TransitionVerifier,
    required_count: int = 256,
) -> ProofBenchmarkManifestV1:
    """Select canonical candidates under frozen quotas and replay every known proof."""

    if required_count < 1 or sum(quotas.values()) != required_count:
        raise ProofBenchmarkError("quota sum must equal the required immutable benchmark count")
    training_groups = set(training_group_ids)
    grouped: dict[tuple[str, ProofTier], list[ProofCandidateV1]] = {}
    excluded = 0
    for candidate in sorted(candidates, key=lambda item: item.pair.pair_id):
        if training_groups & set(candidate.group_closure):
            excluded += 1
            continue
        assert candidate.pair.trace is not None
        replayed = replay_trace(candidate.pair.trace, verifier)
        if replayed.replay_status is not ReplayStatus.PASSED:
            excluded += 1
            continue
        grouped.setdefault((candidate.family, candidate.tier), []).append(candidate)
    selected: list[ProofCandidateV1] = []
    for key, quota in sorted(quotas.items(), key=lambda item: (item[0][0], item[0][1].value)):
        available = grouped.get(key, [])
        if len(available) < quota:
            raise ProofBenchmarkError(
                f"quota shortfall for family/tier {key!r}: {len(available)} < {quota}"
            )
        selected.extend(available[:quota])
    if len(selected) != required_count:
        raise ProofBenchmarkError("selection did not produce exactly the required benchmark count")
    problems = tuple(
        FrozenProofProblemV1(
            problem_id=f"proof:{candidate.pair.pair_id}",
            pair_id=candidate.pair.pair_id,
            source_expression_id=candidate.pair.left.expression_id,
            target_expression_id=candidate.pair.right.expression_id,
            family=candidate.family,
            tier=candidate.tier,
            domain_mode=candidate.pair.left.domain_mode,
            group_closure=candidate.group_closure,
            trace_digest=candidate.pair.trace.trace_digest,
        )
        for candidate in selected
    )
    return ProofBenchmarkManifestV1(
        schema_version=BENCHMARK_SCHEMA_VERSION,
        content_digest=benchmark_digest(problems),
        problems=problems,
        excluded_count=excluded,
    )
