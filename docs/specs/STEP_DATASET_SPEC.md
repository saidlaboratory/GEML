# Rewrite-step dataset (frozen by 7-0)

One supervised example per rewrite step: given a state, which rule fired
and where. 7-1's policy head trains on these, 7-2 evaluates against
them, 8-1 uses `remaining_steps` as its value target.

**Status: partial start.** Extraction, replay checking, ambiguity
handling and the coverage tables are implemented and tested
(`tests/data/test_steps.py`). Production extraction waits on 6-1's real
traces and 4-3's rewrite application API.

## Where the records come from

A 6-1 positive pair is a start state plus an ordered rule sequence. The
state chain is:

```
state_0 = pair.left_signature
state_k = rule_sequence[k-1].result_signature
```

so step *k* is `(state_k, rule_k, site_k) -> state_k+1`. A trace with no
`left_signature` can't be walked at all; it's reported whole rather than
half extracted.

## `StepRecord` (`src/geml/data/steps/extract.py`)

| Field | Notes |
|---|---|
| `step_id` | `{pair_id}#{index}` |
| `pair_id`, `step_index` | back-reference into 6-1 |
| `state_signature` / `next_signature` | before and after |
| `rule_id`, `rule_name`, `tier`, `mode`, `site_id`, `assumptions` | straight off the 6-1 `RuleApplication` |
| `remaining_steps` | steps from this state to the end of the trace, **this one included** - so the final step is 1, never 0 |
| `split`, `family`, `group_id` | inherited from the pair, never recomputed |
| `replay_status` | `unchecked` until 4-3 actually re-applies the step, then `verified` |

`StepError` keeps the failures: pair id, step index, stage, error type,
message. `StepBuildResult.counts` reports `attempted = steps + errors`.

## Reading 6-1 without importing it

`extract.py` declares `TraceStep` and `PairTrace` as `Protocol`s with
6-1's exact field names, so the two branches can land in either order.
They're a structural read of 6-1's contract, not a second definition of
it - when 6-1 merges the real `PairRecord` already satisfies them and
nothing here changes. Same for `apply_rule`, which is 4-3's
`(state, rule_id, site_id) -> next_state | None`.

## Replay is the acceptance criterion

"Every step record replays" only means something if something replayed
it. With `apply_rule` supplied, every step is re-applied and compared to
the stored next state:

| Outcome | What happens |
|---|---|
| matches | `replay_status="verified"` |
| different state | `ReplayMismatch` error row, **and the rest of the trace is abandoned** - it hangs off a state we can't reproduce |
| rule didn't apply (`None`) | same, message records `gave None` |
| engine raised | error row with the exception type, trace abandoned |

Without `apply_rule` the records still come out, marked `unchecked`.
`unverified_steps()` lists them. That's fine for the fixture smoke path
and not fine for the production run - `configs/goal7_steps.yaml` sets
`replay.verify: true`.

## Ambiguity

`find_ambiguous` reports any `(state, rule, site)` key that leads to more
than one next state. Those records aren't a function of their input, so
a policy head trained on them is being asked to predict a coin flip.
`drop_ambiguous` moves them into `errors`; they're never left in quietly
and never dropped silently either.

## Coverage

`rule_coverage` returns one row per rule that fired, **rarest first** -
that ordering is the point, since rare rules are exactly what a grid
average hides. Each row carries the total, per-split counts, and the
distance band (`min_remaining`/`max_remaining`) the rule appears at.

`dead_rules(steps, registry_ids)` names registered rules with zero
records. A rule that never fires is either unreachable in the corpus or
broken in the library, and both are findings - "absent from the table"
is not a report.

`split_violations` checks every step still carries its pair's split.
Steps inherit splits from 6-1; reassigning one here would break the
group-split guarantee upstream, which is the first leak a model would
find.

## Not in here

No policy head (7-1), no metrics (7-2), no training runs (7-3).
