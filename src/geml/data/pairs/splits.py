"""Split and evaluation-view checks for Goal 6 pair evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from geml.contracts.corpus import CorpusSplit
from geml.data.pairs.generate import PairRecordV1, PairStatus


class PairSplitError(ValueError):
    """Raised when source/e-class-relative lineage crosses a protected partition."""


OOD_STRESS_VIEW = "ood_stress"
STRICT_DEPTH_OOD_VIEW = "strict_depth_ood"
FAMILY_OOD_VIEW = "family_ood"


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


def require_ood_stress_view(records: Iterable[PairRecordV1]) -> tuple[PairRecordV1, ...]:
    """Select the stored combined OOD-stress split without relabeling it as depth OOD."""

    selected = require_evaluation_view(records, OOD_STRESS_VIEW)
    invalid = tuple(
        record.pair_id for record in selected if record.source_split is not CorpusSplit.TEST_OOD
    )
    if invalid:
        raise PairSplitError(
            f"ood_stress view may contain only authoritative test_ood records; found {invalid}"
        )
    return selected


def require_strict_depth_ood_view(
    records: Iterable[PairRecordV1],
    *,
    training_depths: Iterable[int],
    evaluation_depth_by_pair_id: Mapping[str, int],
) -> tuple[PairRecordV1, ...]:
    """Select strict depth OOD only after proving disjoint depth support."""

    train = frozenset(training_depths)
    if any(type(depth) is not int or depth < 0 for depth in train):
        raise ValueError("training_depths must contain nonnegative integers")
    selected = require_evaluation_view(records, STRICT_DEPTH_OOD_VIEW)
    if not selected:
        return selected
    if any(record.source_split is not CorpusSplit.TEST_OOD for record in selected):
        raise PairSplitError(
            "strict_depth_ood must be evaluated from the authoritative test_ood split"
        )
    missing = tuple(
        record.pair_id for record in selected if record.pair_id not in evaluation_depth_by_pair_id
    )
    if missing:
        raise PairSplitError(f"strict_depth_ood records lack frozen depth metadata: {missing}")
    evaluation = frozenset(evaluation_depth_by_pair_id[record.pair_id] for record in selected)
    if any(type(depth) is not int or depth < 0 for depth in evaluation):
        raise ValueError("evaluation depths must be nonnegative integers")
    overlap = train & evaluation
    if overlap:
        raise PairSplitError(
            "strict_depth_ood depth supports overlap between training and evaluation: "
            f"{sorted(overlap)}"
        )
    return selected


def require_family_ood_view(
    records: Iterable[PairRecordV1],
    *,
    training_families: Iterable[str],
) -> tuple[PairRecordV1, ...]:
    """Select family OOD only when every evaluated endpoint family was excluded from training."""

    training = frozenset(training_families)
    if any(not isinstance(family, str) or not family.strip() for family in training):
        raise ValueError("training_families must contain nonblank strings")
    selected = require_evaluation_view(records, FAMILY_OOD_VIEW)
    if not selected:
        return selected
    evaluated = {
        family
        for record in selected
        for family in (record.left.operator_family, record.right.operator_family)
    }
    overlap = training & evaluated
    if overlap:
        raise PairSplitError(
            f"family_ood contains families observed in training: {sorted(overlap)}"
        )
    return selected
