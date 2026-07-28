"""Leakage exclusion and deterministic training-cell helpers for Goal 8 value estimates."""

from __future__ import annotations

from collections.abc import Iterable

from geml.learning.value.head import ValueContractError, ValueExampleV1


def exclude_benchmark_groups(
    examples: Iterable[ValueExampleV1],
    *,
    benchmark_group_ids: Iterable[str],
) -> tuple[ValueExampleV1, ...]:
    """Drop every training candidate related to a frozen proof-benchmark group."""

    frozen_groups = set(benchmark_group_ids)
    if not frozen_groups:
        raise ValueContractError("benchmark_group_ids must be explicit before value training")
    retained = tuple(
        example for example in examples if not (frozen_groups & set(example.group_closure))
    )
    if any(frozen_groups & set(example.group_closure) for example in retained):
        raise AssertionError("benchmark leakage exclusion failed")
    return retained


def require_three_seed_rows(seed_rows: Iterable[int]) -> tuple[int, int, int]:
    """Fail closed unless a production table retains the frozen three explicit seeds."""

    observed = tuple(sorted(set(seed_rows)))
    if len(observed) != 3:
        raise ValueContractError(
            "value training requires three retained seed rows or explicit failures"
        )
    return observed  # Production validation additionally matches harness preregistered seeds.
