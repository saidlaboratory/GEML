# Equivalence-pair dataset (frozen by 6-1)

The dataset Goals 6, 7 and 8 all sit on. A pair is two expression ids, a
label, and enough provenance that anyone can replay how we decided the
label. 6-2 tensorizes these records, 7-0 replays the traces into steps,
8-x uses the step distance as a value target - so the shapes below are a
contract, not an implementation detail.

**Status: partial start.** Schema, builder control flow, negative
matching and split bookkeeping are implemented and tested against fakes
(`tests/data/test_pairs.py`). The production build waits on 1-8 (final
corpus), 4-4/4-5 (rule libraries) and 2-6 (verifier). Those come in
through the two protocols below - this module imports none of them.

## Records (`src/geml/data/pairs/generate.py`)

`PairRecord` - one row of the dataset.

| Field | Notes |
|---|---|
| `pair_id` | `left~right` for positives, `left!right` for negatives |
| `left_expression_id` / `right_expression_id` | 1-2 corpus ids |
| `label` | `"equivalent"` / `"not_equivalent"` |
| `split` | one of 1-2's four splits; both sides must agree, see below |
| `group_id` | the leakage unit - e-class id when 4-2 gives us one, expression id until then |
| `family`, `max_depth`, `left_size`, `right_size` | stratification and OOD slicing |
| `verification` | tier + status, on every pair including negatives |
| `left_signature` | 3-1 signature of the state the trace starts from - 7-0 replays step 0 against it |
| `rule_sequence` | positives only: the ordered `RuleApplication` trace |
| `step_distance` | `len(rule_sequence)`; None for negatives |
| `negative_kind` | negatives only |
| `eval_tags` | `depth_ood` / `family_ood`, assigned by `tag_ood` |

`RuleApplication` - one rewrite step. `rule_id`, `rule_name`, 4-1 `tier`
and `mode`, the `site_id` it fired on, the `result_signature` after the
step, and any `assumptions` the mode required. **7-0 replays against
`result_signature`**, so a step without one is not replayable and the
pair carrying it is not usable. Chained with `left_signature` this gives
the full state sequence: state 0 is `left_signature`, state *k* is step
*k-1*'s `result_signature`.

`Verification` - `tier` (`egraph_proof` | `symbolic` | `numeric`) and
`status` (`verified` | `refuted` | `unsupported`). Recorded on every
pair. `unsupported` is a real outcome we keep, not a silent skip.

`PairError` - retained failure row: expression id, stage, error type,
message. Same fields 1-2's `ErrorRow` wants, without importing it while
that branch is unmerged.

## Injected interfaces

Two protocols, both satisfied by objects the owning issues already
describe:

- `SaturationEngine.equivalents(source) -> Iterable[Equivalent]` - 4-2
  through 4-5. Each `Equivalent` carries the id it reached, its size and
  depth, and the full rule sequence that got there.
- `Verifier.verify(left, right, rule_sequence) -> Verification` - 2-6.

Passing them in is what lets the builder be finished and tested before
either exists. It also means the fixture tests and the production run
exercise the same code path.

## What gets rejected

A positive survives only if it has a non-empty trace **and** verification
returns `verified`. Everything else lands in `errors`:

| Case | Why it isn't a pair |
|---|---|
| engine raised | saturation hit a limit or blew up; recorded with the exception type |
| empty rule sequence | nothing for 7-0 to replay |
| `refuted` / `unsupported` | we don't have the proof, so we don't claim the label |
| negative that verifies as equal | a near-miss edit that preserved meaning - real, and reported, but not a negative |
| no size-matched candidate | reported rather than fixed by widening the tolerance |

`BuildResult.counts` reports `attempted = pairs + errors`. Attempted is
the denominator, per 4-1's reporting policy.

## Hard negatives

Same family, same split, node count within tolerance (default 2, see
`configs/goal6_pairs.yaml`), nearest match wins, ties broken by
expression id so the draw is reproducible. Non-equivalence has to be
positively refuted by the verifier - "we couldn't prove them equal" is
not a negative label.

The tolerance is the difficulty knob. Widen it and "which side is
bigger" starts being a winning strategy, which is exactly the shortcut
this dataset exists to close off.

## Splits and leakage

Both sides of a pair come from the same corpus split - `pair_split`
raises `CrossSplitPair` otherwise. Group splits work on `group_id`, so
an expression and its e-class relatives can never straddle a split.
`find_leaks(pairs)` returns every group that appears in more than one
split; the acceptance criterion is that it returns `{}` on the real
build.

Until 4-2 merges, `group_id` falls back to the expression id. That
fallback cannot see that two ids are e-class relatives, so the
production build must run with `eclass_id` populated - a clean
`find_leaks` under the fallback is weaker evidence than it looks.

`tag_ood` measures OOD against what train actually contained: deeper
than any train pair, or a family train never saw. Measured, not
declared, so the tags can't drift away from the data as the corpus
changes.

## Not in here

No tensorization (6-2), no step extraction (7-0), no training. This spec
covers the records and how they're built, nothing downstream.
