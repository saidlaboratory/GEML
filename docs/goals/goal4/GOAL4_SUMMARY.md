# Goal 4 summary: verifier-gated e-graph optimization

## Objective

Goal 4 adds an optional, deterministic, non-ML search over source forms on top of Goals
1–3. It uses equality saturation to discover equivalent expressions and selects by the
official Goal 3 exact Pure EML DAG cost. Pure EML remains canonical.

No result is described as a global optimum. The selected form is only the cheapest
candidate retained and validated under the configured resource limits.

## Clean-room lineage

The Goal 4 branch is built directly on the completed Goal 3 branch. Its implementation
authority is the current repository specifications, issues 4-1 through 4-9, the
authoritative sources named by those specifications, and official dependency
documentation. No prototype implementation or prototype history is an implementation
source.

## Pipeline

For every selected expression, the runner creates two independent work units:
`safe_real` and `positive_real_formal`.

1. Build the frozen source AST and convert it to the closed e-graph vocabulary.
2. Compute the input cost with Goal 3's direct source-AST-to-Pure-EML-DAG compiler.
3. Run bounded equality saturation with complete application provenance and exact attempt
   aggregates.
4. Run bounded, cycle-safe candidate extraction while forcing the source expression to
   remain as a safety anchor.
5. Independently confirm each candidate's concrete membership in the root e-class.
6. Require an explicit source reference, official direct compilation, exact count-only EML
   tree accounting, and a deterministic domain-aware source-semantics audit.
7. Rank valid candidates by exact EML DAG, EML tree, AST DAG, AST tree, and stable
   signature.
8. Retain one terminal row for the work unit, including failures and resource stops.

The source anchor is costed by the same path as rewritten candidates. Therefore a
successful selection cannot be more expensive than the source. Any observed degradation is
classified as an integrity failure rather than reported as unchanged.

## Rewrite modes and assumptions

- `safe_real` uses the branch-insensitive finite-real library.
- `positive_real_formal` adds the guarded domain library.

Mode selection does not itself declare a variable positive. The runner derives declarations
from each corpus record's `domain_mode`: `positive_real` supplies positivity,
`nonzero_real` supplies nonzero, and other supported modes supply realness only. Every row
records those declarations and every domain-rule application records its guard and
assumption.

Positive-real results are conditional findings. They are not presented as universal
complex identities.

## Deterministic subset

The final stage selects 30,000 expressions, within the required 25,000–40,000 range.
Round-robin strata include:

- operator family;
- corpus domain mode;
- split;
- achieved source-AST-size bucket; and
- generator `difficulty_profile`.

Achieved AST size comes from `generator_metadata.achieved_source_ast_size`; a row explicitly
labels the rare fallback to `target_ast_size`. A seeded SHA-256 rank makes selection
independent of input order.

## Resource policy

The production configuration records and enforces:

- 50 saturation iterations;
- 500 e-nodes;
- 2,500 rewrite attempts;
- 0.5 seconds of saturation wall time;
- extraction depth 8, beam width 3, and 12 root candidates;
- 10,000 extraction node visits and 20,000 extraction iterations;
- 0.25 seconds of extraction wall time; and
- eight worker processes.

These bounds are scientific parameters, not hidden implementation details. A structural or
wall-clock stop retains its partial result and explicit reason. Wall-clock stops can depend
on machine load, so reproducibility claims are strongest for rows stopped by structural
bounds or a fixed point.

## Artifact integrity and resume

The run ID hashes the configuration, schema versions, manifest SHA-256, corpus identity,
selected expression IDs and strata, compiler mode, both rewrite modes, and implementation
commit. Rows and checkpoints carry this ID.

Resume rejects:

- an incompatible schema or run ID;
- duplicate or unexpected work units;
- malformed completed JSONL records;
- stale checkpoints; and
- a checkpoint that claims a row absent from durable storage.

A crash-truncated final JSONL fragment is the only repaired case. Valid rows are flushed and
`fsync`ed before a checkpoint advances.

## Metrics and denominators

Each mode and stratum reports:

- `improved / costed`: the success-only after-rate;
- `improved / processed`: the all-processed after-rate;
- `costed / processed`: cost coverage;
- improved, unchanged, degraded, failure, and timeout counts;
- signed and positive-only total improvement; and
- exact mean absolute and relative improvement ratios.

Unsupported operators, timeouts, validation failures, and internal failures stay in the
processed denominator. `costed / processed` is not mislabeled as an all-processed success
rate.

## Production commands

Run from a clean committed checkout:

```bash
python -m geml.experiments.goal4.run \
  --config configs/goal4_final.yaml \
  --stage final \
  --manifest outputs/final/goal1/final/run/manifests/corpus.manifest.json
```

Then generate the strict summary, failure audit, plot data, and six plots:

```bash
python -m geml.analysis.goal4.summary \
  --rows outputs/final/goal4/final/final.rows.jsonl \
  --output-dir outputs/final/goal4/analysis
```

## Production result

<!-- production-results:start -->
### Frozen artifact identity

Only the following audited run is a Goal 4 production result:

| Field | Value |
|---|---|
| Implementation commit | `3f590fcde0dd85ca0db140ce77af8553ac04aeb7` |
| Run ID | `9c26ec3036c45bd3bf24256d9a57fa4e1e48d016cd2cee446a47918667cf2536` |
| Goal 1 manifest SHA-256 | `77fce5779b3d2c2f3cdf2b9f49da54cd14474d37ab128337bdf4fcc52afd4f0d` |
| Goal 4 config SHA-256 | `adb7e72830d96540c65989becc35fc05ce1a6d2d24cadecac105d8dd55bd9b0a` |
| Selection SHA-256 | `8cfba717f7cad75f3d9b0b7e4a532439f4578f21038dcc941aee2e7d6ded2942` |
| Final rows SHA-256 | `f8fd2e6db597da465d4367ce402fc69598eac45031fa9afe52fedc17011a2c31` |
| Expressions / paired rows | 30,000 / 60,000 |

The checkpoint contains exactly the same 60,000 unique `(expression_id, mode)` keys as the
rows file. Every expression has one row in each mode.

### Paired outcomes

| Mode | Processed | Costed | Improved / costed | Improved / processed | Unchanged | Retained failures | Signed EML-DAG nodes saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| `safe_real` | 30,000 | 18,210 (60.700%) | 4,349 / 18,210 (23.882%) | 4,349 / 30,000 (14.497%) | 13,861 | 11,790 | 95,596 |
| `positive_real_formal` | 30,000 | 18,210 (60.700%) | 5,026 / 18,210 (27.600%) | 5,026 / 30,000 (16.753%) | 13,184 | 11,790 | 104,996 |

Mean signed savings over costed rows were 5.250 nodes in `safe_real` and 5.766 in
`positive_real_formal`. Mean relative savings over all costed rows, including unchanged
rows, were 2.361% and 2.849%, respectively. The largest retained reduction was 302 exact
EML-DAG nodes in both modes.

The formal library produced 677 more improved rows and 9,400 more saved nodes than the safe
library. The extra findings were concentrated in `exp_log` (1,560 versus 2,131 improved
rows) and `ood_stress` (289 versus 395). `algebraic_core` and
`powers_division_rationals` had identical improved counts in the two modes, as expected
when no additional guarded rule changes the selected form.

### Coverage, failures, and resource stops

Each mode retained:

- 11,566 unsupported-operator rows (38.553% of processed);
- 224 work-unit validation failures (0.747%);
- zero timeouts;
- zero internal errors;
- zero cost failures or missing-source-candidate failures; and
- zero degraded selections.

All 5,903 `trig_hyperbolic` and 5,663 `mixed_elementary` rows in each mode were outside the
closed Goal 4 vocabulary, which explains the 60.700% cost coverage. Among supported
families, validation failure rates were 2.3% for `exp_log` and 8.1% for `ood_stress`; they
were zero for the other supported families. There was no timeout bias because no work unit
timed out.

Saturation reached a fixed point for 9,558 safe and 9,489 formal rows. Node limits stopped
6,427 and 6,458 rows; rewrite-attempt limits stopped 2,449 and 2,487. These are retained
partial searches, not mislabeled fixed points. Extraction was explicitly partial for
17,015 safe and 17,280 formal rows. The observed maxima were the configured 500 e-nodes
and 2,500 rewrite attempts; extraction visited at most 3,127 of its allowed 10,000 nodes.

### Validation and provenance audit

The independent streaming audit checked every row, not a sample. Every selected form:

- remained in the claimed root with the original source anchor;
- had exact non-increasing before/after cost arithmetic;
- passed direct Goal 3 compilation and independent source-semantics probing; and
- carried complete compact application provenance plus exact attempt aggregates.

The 224 failed rows per mode were conservative zero-finite-evidence outcomes. Across their
retained candidates there were 660 `safe_real` and 773 `positive_real_formal`
`inconclusive` verdicts.

One additional, non-selected formal candidate on expression
`ba621a8167a97f1debc522f024888a32b2dbfe6d1ec5b3fb2c69cf4e064e2d77`
was retained as `domain_mismatch`. Reproduction traced it to floating evaluation of the
exact identity `exp(log(9)) - 7 = 2` when used as an exponent of `-1`: the near-integer
float makes the probe evaluator report undefinedness. The source anchor was selected
unchanged, so no reported improvement depends on this candidate. The audit deliberately
keeps this conservative false positive instead of weakening the definedness check with an
unjustified near-integer heuristic.

All six generated plots were rendered and visually checked:
`success_rate.png`, `improvement_distribution.png`, `runtime_distribution.png`,
`failure_breakdown.png`, `family_improvements.png`, and `memory_availability.png`. RSS
snapshots were available on all 60,000 rows.

### Superseded audit runs

No earlier local run contributes to these findings. Review first exposed a misleading
top-level validation-failure reason and then a rare incremental congruence-rekey defect
that could lose or over-merge a retained node near the e-node limit. The latter reduced to
a 19-node regression and was replaced by atomic whole-graph canonical repair. Runs from
the pre-fix commits were preserved only as audit evidence and superseded; the table above
uses the clean post-fix implementation and its distinct content-addressed run ID.
<!-- production-results:end -->

## Claim boundaries

- Same-e-class membership is formal evidence relative to the enabled rule set and recorded
  assumptions; numeric probing is an independent bug-detection audit, not a theorem prover.
- Candidate extraction is bounded and beam-limited.
- Trigonometric and hyperbolic source operators are outside the Goal 4 e-graph vocabulary
  and are retained as unsupported rows.
- The study does not claim global minimality, symbolic completeness, unrestricted complex
  equivalence, or machine-independent wall-clock frontiers.
