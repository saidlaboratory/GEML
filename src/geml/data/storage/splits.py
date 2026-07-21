"""
splits.py - deterministic train/validation/IID-test/OOD-test assignment

owned by 1-5

real target counts are 175k/25k/25k/25k (250k total), but the actual
function takes split_sizes as a param so tests can use tiny numbers
instead of the real corpus size
"""
from __future__ import annotations
from dataclasses import dataclass
import random

from geml.data.storage.dedup import ExpressionRecord

# real corpus target - used by the actual 250k run, not by tests
GOAL1_SPLIT_SIZES = {
    "train": 175_000,
    "validation": 25_000,
    "iid_test": 25_000,
    "ood_test": 25_000,
}


class SplitSizeMismatchError(ValueError):
    """raised when the record count doesn't exactly match the sum of
    split_sizes - per spec, this should fail loud, not silently drop or
    pad rows to make the counts work"""


@dataclass
class SplitResult:
    splits: dict[str, list[ExpressionRecord]]
    seed: int


def assign_splits(
    records: list[ExpressionRecord],
    split_sizes: dict[str, int],
    seed: int = 42,
) -> SplitResult:
    """
    deterministically assigns every record to exactly one split, with
    exact counts matching split_sizes. same records + same split_sizes
    + same seed always produces the exact same assignment - shuffles a
    COPY of the list (never touches the caller's original order) using
    a seeded Random instance, then slices it up in a fixed split order.
    """
    total_requested = sum(split_sizes.values())
    if total_requested != len(records):
        raise SplitSizeMismatchError(
            f"got {len(records)} records but split_sizes wants exactly "
            f"{total_requested} - these must match exactly, not "
            f"approximately"
        )

    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    result: dict[str, list[ExpressionRecord]] = {}
    cursor = 0
    for split_name, size in split_sizes.items():
        result[split_name] = shuffled[cursor : cursor + size]
        cursor += size

    return SplitResult(splits=result, seed=seed)
