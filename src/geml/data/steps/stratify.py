"""
stratify.py - per-rule coverage tables and split-inheritance checks

owned by 7-0

pure bookkeeping over StepRecords, so this half is finished rather than
skeletal. the rule registry it checks against is 4-4/4-5's; pass the ids
in, because a hardcoded list here would go stale the moment they add a
rule.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from geml.data.steps.extract import StepRecord


@dataclass(frozen=True)
class RuleCoverage:
    rule_id: str
    total: int
    per_split: Mapping[str, int]
    min_remaining: int | None    # shortest distance-to-target this rule appears at
    max_remaining: int | None


def rule_coverage(steps: Sequence[StepRecord]) -> tuple[RuleCoverage, ...]:
    """
    one row per rule that actually fired, rarest first - that ordering is
    the point of the table, since the rare rules are the ones a grid
    average will hide.
    """
    by_rule: dict[str, list[StepRecord]] = defaultdict(list)
    for step in steps:
        by_rule[step.rule_id].append(step)

    rows = [
        RuleCoverage(
            rule_id=rule_id,
            total=len(group),
            per_split=Counter(s.split for s in group),
            min_remaining=min(s.remaining_steps for s in group),
            max_remaining=max(s.remaining_steps for s in group),
        )
        for rule_id, group in by_rule.items()
    ]
    return tuple(sorted(rows, key=lambda r: (r.total, r.rule_id)))


def dead_rules(steps: Iterable[StepRecord], registry_rule_ids: Iterable[str]) -> tuple[str, ...]:
    """
    registered rules with zero step records.

    a rule that never shows up is either unreachable in the corpus or
    broken in the rule library, and both are findings. "absent from the
    table" is not a report, so this returns them explicitly.
    """
    fired = {step.rule_id for step in steps}
    return tuple(sorted(set(registry_rule_ids) - fired))


def split_violations(
    steps: Iterable[StepRecord],
    trace_splits: Mapping[str, str],
) -> tuple[str, ...]:
    """
    step ids whose split disagrees with their source trace's split.

    steps inherit splits from 6-1 pairs; if extraction ever reassigns one
    it breaks the group-split guarantee upstream, which is exactly the
    leak a model would happily exploit. empty tuple means clean.
    """
    return tuple(
        sorted(
            step.step_id for step in steps
            if step.pair_id in trace_splits and step.split != trace_splits[step.pair_id]
        )
    )


def unverified_steps(steps: Iterable[StepRecord]) -> tuple[str, ...]:
    """step ids that were never actually replayed - reported, not assumed fine."""
    return tuple(sorted(step.step_id for step in steps if step.replay_status != "verified"))
