"""Split-isolation and reporting helpers for Goal 7 rewrite-step evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from geml.contracts.corpus import CorpusSplit
from geml.data.steps.extract import RewriteStepRecordV1


class StepSplitError(ValueError):
    """A group-relative rewrite step crosses a protected source split."""


def validate_step_group_isolation(records: Iterable[RewriteStepRecordV1]) -> None:
    """Prove that inherited group closures remain in exactly one source partition."""

    memberships: dict[str, set[CorpusSplit]] = defaultdict(set)
    for record in records:
        for group_id in record.group_closure:
            memberships[group_id].add(record.source_split)
    crossing = {group: splits for group, splits in memberships.items() if len(splits) > 1}
    if crossing:
        rendered = ", ".join(
            f"{group}: {sorted(split.value for split in splits)}"
            for group, splits in sorted(crossing.items())
        )
        raise StepSplitError(f"rewrite-step group leakage across partitions: {rendered}")


def rule_coverage(
    records: Iterable[RewriteStepRecordV1],
    *,
    registered_rule_ids: Iterable[str],
) -> dict[str, int]:
    """Report every registered rule, including rules with zero supervised targets."""

    expected = tuple(sorted(set(registered_rule_ids)))
    if not expected or any(not rule_id.strip() for rule_id in expected):
        raise ValueError("registered_rule_ids must be a nonempty set of nonblank rule IDs")
    observed = Counter(record.action.rule_id for record in records)
    unknown = sorted(set(observed) - set(expected))
    if unknown:
        raise StepSplitError(f"step rows contain unregistered rule IDs: {unknown}")
    return {rule_id: observed[rule_id] for rule_id in expected}


def family_coverage(records: Iterable[RewriteStepRecordV1]) -> dict[str, int]:
    """Report source-family counts without resampling the evaluation population."""

    return dict(sorted(Counter(record.operator_family for record in records).items()))
