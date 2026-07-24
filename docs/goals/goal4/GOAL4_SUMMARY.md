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
The corrected production run has not yet been inserted into this document. Do not infer a
scientific after-rate from the smoke fixtures.
<!-- production-results:end -->

## Claim boundaries

- Same-e-class membership is formal evidence relative to the enabled rule set and recorded
  assumptions; numeric probing is an independent bug-detection audit, not a theorem prover.
- Candidate extraction is bounded and beam-limited.
- Trigonometric and hyperbolic source operators are outside the Goal 4 e-graph vocabulary
  and are retained as unsupported rows.
- The study does not claim global minimality, symbolic completeness, unrestricted complex
  equivalence, or machine-independent wall-clock frontiers.
