# Gate G7: goal-conditioned rewrite policy

Status: **insufficient evidence / production pending**

This document freezes Gate G7 before any production test-grid output is
inspected. Tiny fixtures exercise the same runner, metric, analysis, and plot
paths, but fixture output can never produce a scientific pass or fail.

## Primary comparison

The predeclared primary arm is `gnn:pure_eml_dag`. It is compared, seed for
seed and step for step, with `uniform_valid`. The comparison cutoff is top-1.
The transformer and the other three graph channels remain required report
arms, but they are not substituted for the primary arm after observing test
results.

All six arms use the same accepted `StepRecordV1` rows, frozen
`test_iid`/`test_ood` evaluation splits, registry-derived legal-action mask,
top-k values, node/edge batch budget, optimizer-step budget,
epoch/early-stop policy, complete training-configuration digest, and seeds:

```text
20260726, 20260727, 20260728
```

## Frozen quantitative rule

Gate G7 returns `pass` only if all of these conditions hold:

1. Top-1 demonstration-action match exceeds the uniform-valid rate by at least
   `0.05`, and the lower bound of its 95% paired group-bootstrap interval is
   greater than zero.
2. Top-1 exact-successor-structure match exceeds the uniform-valid rate by at
   least `0.05`, with the same positive lower-bound requirement.
3. Top-1 verifier-valid success is at least `0.99`.
4. Its verifier-valid margin relative to uniform-valid is at least `-0.01`.
5. No registered rule is dead in the primary arm.

The five-percentage-point exact-metric margin is the predeclared minimum
practical improvement for this bounded workshop experiment, rather than a
post-result significance target. The 0.99 safety floor reflects that learned
ranking is useful only if verifier-confirmed legal progress remains nearly
universal at top-1. One hundred distinct source groups is a conservative
minimum below which the grouped interval is treated as descriptive only.

The five learned arms must also be compute matched within each seed: the
relative parameter-count span may not exceed `0.05`, and the estimated-FLOP
span on the frozen reference workload may not exceed `0.10`. The tighter
parameter tolerance enforces substantially the same capacity; the wider FLOP
tolerance allows discrete batching and graph/sequence padding without calling
materially different workloads matched. Outside-tolerance or missing telemetry
is insufficient evidence, not a favorable gate result.

The interval resamples connected dependency components formed from
`source_group`, `trace_id`, and every retained lineage-group ID. Transitive
overlap merges components. Each selected component retains all of its steps
and seed repetitions. The fixed bootstrap uses 2,000 resamples and seed
`20260726`; at least 100 distinct components are required. Raw seed rows are
always published because three seeds do not justify treating seeds as an
asymptotic sample.

`demonstration-action match` means the exact
`(rule_id, direction, occurrence_path, ordered_arguments)` action. It is not a
claim that the demonstrated route is uniquely mathematically correct.
`exact-successor-structure match` separately credits distinct actions that
replay to the exact stored successor signature. Verifier-valid success is a
safety metric and never substitutes semantic equivalence for either exact
metric.

## Verdict states

- `pass`: every evidence precondition and every quantitative rule passes.
- `fail`: production evidence is complete and sufficient, but at least one
  quantitative rule fails.
- `insufficient_evidence`: fixture stage; missing or non-complete cells; fewer
  than three complete primary seeds; fewer than 100 groups; unavailable paired
  rows; a missing authenticated completion ledger; unavailable/out-of-tolerance
  compute matching; a missing production analysis run envelope; or a registered
  rule with no evaluation example. Any ranked candidate outside the shared
  legal mask or frozen directed registry also makes the evidence insufficient.

Failures, timeouts, unsupported cells, no-action rows, invalid proposals, and
verifier errors remain in the raw tables and denominators. They are never
filtered to manufacture a pass.

Production analysis rejects any Gate-policy override: primary/baseline arms,
cutoff, margins, safety floor, bootstrap settings, and dead-rule rule are
frozen above before result inspection. The analysis run envelope must bind the
run, config, step manifest, rule registry, verifier, implementation, complete
training configuration, authenticated step-population digest, authenticated
training-family inventory, and exact predeclared analysis command. Each cell
also records actual epochs, optimizer steps, maximum node/edge batch,
early-stopping state, stop reason, executor-reported wall time, and independent
runner-observed wall time. Any cap violation is retained as invalid evidence.
Learned-arm parameter counts must be exact across seeds and
reference-workload FLOPs must remain within the predeclared tolerance, in
addition to within-seed arm matching.

An unseen-family row is reportable only when its label is derived from a
training-family inventory bound to the exact step-manifest digest. The
step-manifest authenticator must independently return the frozen inventory
digest, and every production metric row must match it. Caller labels and
family-name heuristics cannot create held-out evidence.

The step-manifest authenticator separately derives a digest over the sorted
accepted step scientific identities. Every complete cell must match that exact
population; equal row counts and agreement among arms are not sufficient.

## Current production dependencies

The four-channel contract is resolved in `docs/specs/PRE_PHASE_B_DECISIONS.md`.
The existing frequent motif artifact remains labeled **macro-derived**, and the
train-only motif-AST control uses an equal entry/MDL budget. Production hashes
and the step denominator remain null until Phase B materializes the aligned
channels, step manifest, dataset/registry/verifier digests, and harness/model
digests. The runner refuses production execution until those evidence
identities are supplied.
