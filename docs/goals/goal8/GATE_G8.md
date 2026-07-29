# Gate G8 preregistration

**Gate schema:** `geml-goal8-gate-policy-v1`
**Policy ID:** `goal8-primary-efficiency-v1`
**Policy SHA-256:** `b83ffafe13d5eb56a9572b82fa09b019aed81b6087cef4e172433056054c0649`
**Current verdict:** `insufficient_evidence` (production studies have not run)

Gate G8 asks one bounded primary question:

> On the frozen 256-problem benchmark, under identical budgets, does the
> preregistered GNN policy-plus-value search reduce expanded nodes relative to
> uniform-valid search while maintaining comparable verifier-confirmed exact
> target success and accepting no invalid proof transition?

Simplification results are required as a complete, verifier-normalized
1,000-expression companion study, but there is no post-hoc simplification
quality threshold in the primary proof-efficiency gate.

## Frozen primary comparison

| Field | Frozen value |
|---|---|
| Baseline | `uniform` |
| Primary guided method | `policy_value` |
| Other descriptive controlled methods | `policy`, `transformer` |
| Required proof result schema | `geml-goal8-atp-cell-v1` |
| Simplification methods | `uniform`, `policy`, and deterministic `sympy` |
| Simplification descriptive baseline | `uniform` |
| Required simplification result schema | `geml-goal8-simplify-cell-v1` |
| Stochastic proof seeds (`uniform`) | `20260726`, `20260727`, `20260728` |
| Deterministic proof seed (`policy`, `policy_value`, `transformer`) | canonical `20260726` only |
| Stochastic simplification seeds (`uniform`) | `20260726`, `20260727`, `20260728` |
| Deterministic seeded simplification (`policy`) | canonical `20260726` only |
| Deterministic unseeded simplification (`sympy`) | one null-seed cell |
| Proof tasks | exactly 256 frozen IDs |
| Simplification tasks | exactly 1,000 frozen IDs |
| Minimum group-mean success difference | guided − uniform ≥ −0.02 |
| Minimum all-attempt group-mean node reduction | 0.10 |
| Node-reduction interval | deterministic 95% group bootstrap, 2,000 resamples |
| Interval rule | lower endpoint must be > 0 |
| Minimum paired problem groups | 64 |
| Maximum accepted invalid proof transitions | 0 |

Every simplification expression requires all three stochastic `uniform` cells,
one deterministic `policy` cell at the canonical seed, and one deterministic
`sympy` cell with a null seed: five cells per expression. A missing cell
remains missing evidence; it is not imputed as failure or zero improvement.

Deterministic ATP methods are not rerun under nominally different seeds.
Pairing with uniform therefore uses the shared canonical seed; the two
additional stochastic-uniform rows remain raw descriptive evidence and are
not turned into pseudo-replicates for a deterministic method.

The 0.02 success margin defines *comparable* exact-target success. The 10%
node threshold is applied to all paired attempted rows, not only solved rows.
For conservative accounting, every unsuccessful row is charged at least its
configured expanded-node budget. Missing telemetry is never imputed, and any
missing paired seed/group telemetry makes the evidence insufficient. The
primary method was named before production results; the best observed method
cannot be substituted after evaluation.

## Resampling unit

Rows first pair on `(problem_id, seed)`. Repeated rows are collapsed by the
frozen `group_id`, and only those group summaries are resampled. Search nodes
and repeated seeds are never treated as independent observations. Raw seed
rows remain published.

## Verdict rules

### `insufficient_evidence`

Return `insufficient_evidence` before considering favorable values if any of
the following is true:

- either manifest is fixture-scoped, mixed-scope, unauthenticated, or unfrozen;
- the production proof manifest was not accepted by issue #67's canonical
  producer loader and exact external byte checksum;
- proof or simplification rows were supplied in memory instead of through a
  complete reauthenticated producer shard/cell bundle;
- either producer bundle lacks independently frozen expected aggregate and
  pre-run config digests, or its exact bytes/config disagree with those
  external trust anchors;
- any producer shard is missing/duplicated, any completion count/digest
  disagrees, any canonical producer `run_id` cannot be re-derived, or any
  expected cell file is missing/extra;
- the frozen populations are not exactly 256 and 1,000 unique IDs;
- any required proof problem/method/seed cell is absent;
- any required simplification expression/method/seed cell is absent;
- any unexpected method/seed cell or noncanonical deterministic seed is
  present;
- any controlled row is not an authenticated projection of the frozen #68 or
  #69 producer result schema;
- any row's group/stratum metadata or full ATP projection digest disagrees
  with the authoritative frozen manifest record;
- more than one controlled ATP budget digest occurs anywhere in the grid;
- any primary paired seed row lacks expanded-node telemetry, or any paired
  group is incomplete for conservative node accounting;
- required difficulty/OOD stratum metadata is absent from saved rows;
- any row lacks SHA-shaped run/cell/config/artifact provenance, a concrete Git
  commit, runtime/hardware/package identity, or a fully rendered exact-shard
  reproduction command;
- run/config identities conflict across rows, per-method checkpoints conflict,
  or row identities do not bind to the authenticated manifests;
- any retained proof or GEML-simplification execution attestation disagrees
  with its frozen method/checkpoint/rule/verifier/implementation/budget cell,
  or a SymPy implementation/wall-budget attestation disagrees;
- the primary contrast is absent;
- fewer than 64 paired problem groups are available.

Tiny fixture data therefore cannot pass Gate G8, regardless of its numbers.
Protocol incompleteness is not converted into a negative scientific result.
The current #68/#69 run layouts do not publish an enclosing independently
retained aggregate/config record, so that release/orchestration record remains
a production-integration prerequisite rather than something this analyzer may
self-issue.

### `fail`

After complete production evidence is established, return `fail` if any of
the following is true:

- an invalid transition entered a counted proof;
- a claimed proof is not exact-target, replay-confirmed, and terminal-verified;
- a claimed simplification/no-change output is not verifier-confirmed;
- primary guided success is more than 0.02 below uniform;
- mean all-attempt node reduction is below 0.10;
- the 95% group-bootstrap lower endpoint is not strictly positive.

Rejected invalid action proposals remain safety telemetry and do not themselves
mean that an invalid transition was accepted. Because #68 does not persist a
general accepted-invalid-transition counter, missing telemetry is never
reported as zero: counted successes establish zero through complete verified
replay, while every unverified claim is a gate failure.

### `pass`

Return `pass` only after every completeness/safety requirement is satisfied and
all three primary quantitative conditions hold:

1. success difference ≥ −0.02;
2. mean all-attempt node reduction ≥ 0.10;
3. the 95% group-bootstrap lower endpoint is > 0.

The resulting claim remains limited to the fixed benchmark, budgets, methods,
seeds, rule set, verifier version, and hardware/configuration evidence bound by
the saved manifests.

## External LLM rows

Rows produced by #82 are optional external context. They are never a controlled
baseline, never enter the paired contrasts, and cannot change this verdict.
Missing rows are reported as `missing`, not as zero accuracy.

## Current evidence

No production ATP or simplification study was run in Phase A. The current
verdict is therefore `insufficient_evidence`; there are no values to report or
interpret.
