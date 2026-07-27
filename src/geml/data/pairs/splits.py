"""Split and evaluation-view checks for Goal 6 pair evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from geml.contracts.corpus import CorpusSplit
from geml.data.pairs.generate import PairRecordV1, PairStatus


class PairSplitError(ValueError):
    """Raised when source/e-class-relative lineage crosses a protected partition."""


def validate_group_isolation(records: Iterable[PairRecordV1]) -> None:
    """Ensure every endpoint group and relative occurs in exactly one source partition."""

    memberships: dict[str, set[CorpusSplit]] = defaultdict(set)
    for record in records:
        for endpoint in (record.left, record.right):
            for group_id in endpoint.group.closure:
                memberships[group_id].add(record.source_split)
    crossing = {group: splits for group, splits in memberships.items() if len(splits) > 1}
    if crossing:
        rendered = ", ".join(
            f"{group}: {sorted(split.value for split in splits)}"
            for group, splits in sorted(crossing.items())
        )
        raise PairSplitError(f"source/e-class-relative group leakage across partitions: {rendered}")


def validate_fixed_base_counts(
    records: Iterable[PairRecordV1],
    *,
    expected_train: int,
    expected_validation: int,
    expected_test: int,
) -> None:
    """Validate accepted base-record quotas without treating OOD views as extra rows."""

    expected = {
        CorpusSplit.TRAIN: expected_train,
        CorpusSplit.VALIDATION: expected_validation,
        CorpusSplit.TEST_IID: expected_test,
        CorpusSplit.TEST_OOD: expected_test,
    }
    observed: dict[CorpusSplit, int] = defaultdict(int)
    for record in records:
        if record.status is PairStatus.ACCEPTED:
            observed[record.source_split] += 1
    if dict(observed) != expected:
        rendered = {split.value: observed.get(split, 0) for split in CorpusSplit}
        wanted = {split.value: count for split, count in expected.items()}
        raise PairSplitError(f"pair base-count mismatch: expected {wanted}, observed {rendered}")


def require_evaluation_view(records: Iterable[PairRecordV1], view: str) -> tuple[PairRecordV1, ...]:
    """Return an immutable evaluation view without changing the underlying base records."""

    if not isinstance(view, str) or not view.strip():
        raise ValueError("evaluation view must be a nonblank string")
    return tuple(
        record
        for record in records
        if record.status is PairStatus.ACCEPTED and view in record.evaluation_views
    )
