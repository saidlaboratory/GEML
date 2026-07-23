"""
extract.py - turn 6-1 pair traces into supervised rewrite-step records

owned by 7-0

PARTIAL START. the extraction, the replay check and the failure rows are
here and tested; the production run waits on 6-1's real traces and on
4-3's rewrite application API.

on the two protocols below: they describe 6-1's PairRecord and
RuleApplication structurally rather than importing them, so this branch
and the 6-1 branch can land in either order. once 6-1 merges they bind
to the real classes with no change here - the field names are already
theirs. same story for `apply_rule`, which is 4-3.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, Sequence


class TraceStep(Protocol):
    """structurally 6-1's RuleApplication."""
    rule_id: str
    rule_name: str
    tier: str
    mode: str
    site_id: str
    result_signature: str
    assumptions: tuple[str, ...]


class PairTrace(Protocol):
    """structurally the positive half of 6-1's PairRecord."""
    pair_id: str
    split: str
    family: str
    group_id: str
    left_signature: str | None
    rule_sequence: tuple[TraceStep, ...]


@dataclass(frozen=True)
class StepRecord:
    """one (state, rule, site) training example."""
    step_id: str
    pair_id: str
    step_index: int
    state_signature: str        # the state this step is applied to
    next_signature: str         # the state it produces
    rule_id: str
    rule_name: str
    tier: str
    mode: str
    site_id: str
    remaining_steps: int        # steps from this state to the end of the trace, this one included
    split: str
    family: str
    group_id: str
    assumptions: tuple[str, ...] = ()
    replay_status: str = "unchecked"   # "verified" once 4-3 has actually re-applied it


@dataclass(frozen=True)
class StepError:
    pair_id: str | None
    step_index: int | None
    stage: str
    error_type: str
    message: str


@dataclass
class StepBuildResult:
    steps: list[StepRecord] = field(default_factory=list)
    errors: list[StepError] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "attempted": len(self.steps) + len(self.errors),
            "steps": len(self.steps),
            "errors": len(self.errors),
        }


# 4-3: apply this rule at this site to this state, give back the resulting
# signature. None means it didn't apply.
ApplyRule = Callable[[str, str, str], str | None]


def extract_steps(
    traces: Iterable[PairTrace],
    *,
    apply_rule: ApplyRule | None = None,
) -> StepBuildResult:
    """
    walk each trace and emit one record per step.

    the state chain is left_signature -> step[0].result_signature ->
    step[1].result_signature -> ..., so a trace with no starting state
    can't be walked at all and is reported whole rather than half
    extracted.

    pass 4-3's `apply_rule` to actually re-apply every step and check it
    lands on the stored next state; without it records come out
    `replay_status="unchecked"`, which is honest about what CI proved.
    """
    result = StepBuildResult()
    for trace in traces:
        if not trace.rule_sequence:
            continue
        if not trace.left_signature:
            result.errors.append(
                StepError(trace.pair_id, None, "extract", "MissingInitialState",
                          "trace has no left_signature to replay from")
            )
            continue

        state = trace.left_signature
        total = len(trace.rule_sequence)
        for index, step in enumerate(trace.rule_sequence):
            status = "unchecked"
            if apply_rule is not None:
                try:
                    produced = apply_rule(state, step.rule_id, step.site_id)
                except Exception as error:
                    result.errors.append(
                        StepError(trace.pair_id, index, "replay", type(error).__name__, str(error))
                    )
                    break
                if produced != step.result_signature:
                    result.errors.append(
                        StepError(
                            trace.pair_id, index, "replay", "ReplayMismatch",
                            f"{step.rule_id} at {step.site_id} gave {produced!r}, "
                            f"trace says {step.result_signature!r}",
                        )
                    )
                    # the rest of the trace hangs off a state we can't reproduce
                    break
                status = "verified"

            result.steps.append(
                StepRecord(
                    step_id=f"{trace.pair_id}#{index}",
                    pair_id=trace.pair_id,
                    step_index=index,
                    state_signature=state,
                    next_signature=step.result_signature,
                    rule_id=step.rule_id,
                    rule_name=step.rule_name,
                    tier=step.tier,
                    mode=step.mode,
                    site_id=step.site_id,
                    remaining_steps=total - index,
                    split=trace.split,
                    family=trace.family,
                    group_id=trace.group_id,
                    assumptions=tuple(step.assumptions),
                    replay_status=status,
                )
            )
            state = step.result_signature
    return result


def find_ambiguous(steps: Sequence[StepRecord]) -> dict[tuple[str, str, str], tuple[str, ...]]:
    """
    (state, rule, site) keys that lead to more than one next state.

    those records aren't a function of their input, so a policy head
    trained on them is being asked to predict a coin flip. they get
    reported here and dropped by `drop_ambiguous`, never left in quietly.
    """
    seen: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for step in steps:
        seen[(step.state_signature, step.rule_id, step.site_id)].add(step.next_signature)
    return {key: tuple(sorted(nexts)) for key, nexts in seen.items() if len(nexts) > 1}


def drop_ambiguous(result: StepBuildResult) -> StepBuildResult:
    """move every ambiguous record out of `steps` and into `errors`."""
    ambiguous = find_ambiguous(result.steps)
    if not ambiguous:
        return result

    kept = []
    for step in result.steps:
        key = (step.state_signature, step.rule_id, step.site_id)
        if key in ambiguous:
            result.errors.append(
                StepError(
                    step.pair_id, step.step_index, "extract", "AmbiguousStep",
                    f"{step.rule_id} at {step.site_id} leads to {ambiguous[key]}",
                )
            )
        else:
            kept.append(step)
    result.steps = kept
    return result
