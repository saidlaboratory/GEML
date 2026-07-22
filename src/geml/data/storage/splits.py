"""Deterministic semantic assignment to the four frozen corpus splits."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from geml.contracts.corpus import (
    FINAL_CORPUS_SPLIT_COUNTS,
    FINAL_CORPUS_TOTAL_COUNT,
    CorpusSplit,
)
from geml.contracts.expression import ExpressionRecord

GOAL1_SPLIT_SIZES = FINAL_CORPUS_SPLIT_COUNTS
_IID_SPLIT_ORDER = (
    CorpusSplit.TRAIN,
    CorpusSplit.VALIDATION,
    CorpusSplit.TEST_IID,
)


class SplitAssignmentError(ValueError):
    """Input records cannot satisfy the requested semantic split policy."""


class SplitSizeMismatchError(SplitAssignmentError):
    """Input population size differs from the exact requested split total."""


@dataclass(frozen=True)
class SplitAssignment:
    """Immutable assigned records in canonical split order."""

    records_by_split: Mapping[CorpusSplit, tuple[ExpressionRecord, ...]]
    seed: int

    @property
    def total_count(self) -> int:
        return sum(len(records) for records in self.records_by_split.values())

    def iter_records(self) -> Iterable[ExpressionRecord]:
        for split in (*_IID_SPLIT_ORDER, CorpusSplit.TEST_OOD):
            yield from self.records_by_split[split]


def _validated_counts(
    split_sizes: Mapping[CorpusSplit | str, int],
) -> Mapping[CorpusSplit, int]:
    counts: dict[CorpusSplit, int] = {}
    for raw_split, count in split_sizes.items():
        try:
            split = raw_split if isinstance(raw_split, CorpusSplit) else CorpusSplit(raw_split)
        except ValueError as error:
            raise SplitAssignmentError(f"unknown corpus split: {raw_split!r}") from error
        if split in counts:
            raise SplitAssignmentError(f"duplicate split count for {split.value!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SplitAssignmentError(
                f"split count for {split.value!r} must be a nonnegative integer"
            )
        counts[split] = count

    required = set(CorpusSplit)
    if set(counts) != required:
        missing = sorted(split.value for split in required - set(counts))
        raise SplitAssignmentError(f"split counts must name every frozen split; missing={missing}")
    return MappingProxyType(counts)


def _assignment_key(record: ExpressionRecord, seed: int) -> tuple[bytes, str]:
    payload = f"geml-split-v1\0{seed}\0{record.expression_id}".encode()
    return hashlib.sha256(payload).digest(), record.expression_id


def assign_splits(
    records: Iterable[ExpressionRecord],
    split_sizes: Mapping[CorpusSplit | str, int] = GOAL1_SPLIT_SIZES,
    *,
    seed: int = 42,
    ood_operator_families: tuple[str, ...] = ("ood_stress",),
) -> SplitAssignment:
    """Assign exact counts while keeping OOD-stress expressions out of IID splits.

    Assignment depends only on the seed and stable expression identities, never input
    order or Python's process-randomized ``hash()``.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitAssignmentError("split seed must be an integer")
    if not ood_operator_families or any(not name.strip() for name in ood_operator_families):
        raise SplitAssignmentError("ood_operator_families must contain nonblank names")
    if len(set(ood_operator_families)) != len(ood_operator_families):
        raise SplitAssignmentError("ood_operator_families must be unique")

    counts = _validated_counts(split_sizes)
    materialized = tuple(records)
    requested_total = sum(counts.values())
    if len(materialized) != requested_total:
        raise SplitSizeMismatchError(
            f"received {len(materialized)} records but exact split counts require {requested_total}"
        )
    expression_ids = [record.expression_id for record in materialized]
    if len(set(expression_ids)) != len(expression_ids):
        raise SplitAssignmentError("split input contains duplicate expression_id values")

    ood_families = frozenset(ood_operator_families)
    ood_records = tuple(record for record in materialized if record.operator_family in ood_families)
    iid_records = tuple(
        record for record in materialized if record.operator_family not in ood_families
    )
    if len(ood_records) != counts[CorpusSplit.TEST_OOD]:
        raise SplitSizeMismatchError(
            f"found {len(ood_records)} OOD records but test_ood requires "
            f"{counts[CorpusSplit.TEST_OOD]}"
        )
    required_iid = sum(counts[split] for split in _IID_SPLIT_ORDER)
    if len(iid_records) != required_iid:
        raise SplitSizeMismatchError(
            f"found {len(iid_records)} IID records but IID splits require {required_iid}"
        )

    sorted_iid = sorted(iid_records, key=lambda record: _assignment_key(record, seed))
    sorted_ood = sorted(ood_records, key=lambda record: _assignment_key(record, seed))
    assigned: dict[CorpusSplit, tuple[ExpressionRecord, ...]] = {}
    cursor = 0
    for split in _IID_SPLIT_ORDER:
        size = counts[split]
        assigned[split] = tuple(
            record.model_copy(update={"split": split})
            for record in sorted_iid[cursor : cursor + size]
        )
        cursor += size
    assigned[CorpusSplit.TEST_OOD] = tuple(
        record.model_copy(update={"split": CorpusSplit.TEST_OOD}) for record in sorted_ood
    )
    return SplitAssignment(records_by_split=MappingProxyType(assigned), seed=seed)


def validate_final_split_counts(assignment: SplitAssignment) -> None:
    """Raise when an assignment is not the frozen 250,000-record final layout."""

    observed = {split: len(assignment.records_by_split.get(split, ())) for split in CorpusSplit}
    if observed != dict(FINAL_CORPUS_SPLIT_COUNTS):
        raise SplitSizeMismatchError(
            f"final split counts must be {dict(FINAL_CORPUS_SPLIT_COUNTS)!r}; got {observed!r}"
        )
    if assignment.total_count != FINAL_CORPUS_TOTAL_COUNT:
        raise SplitSizeMismatchError(
            f"final corpus must contain exactly {FINAL_CORPUS_TOTAL_COUNT} records"
        )
