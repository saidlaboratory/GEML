# Goal 8 analysis: Phase-A contract

**Issue:** #70 / 8-5
**Status:** `phase_a_implemented`; production evidence is pending
**Analysis schema:** `geml-goal8-analysis-v1`

This issue analyzes saved Goal 8 proof-search and simplification rows. It does
not run search, train a model, call SymPy, call an LLM provider, or modify any
Goals 1–5 artifact. Phase A contains only typed readers, deterministic
aggregation, plots, table writers, gate logic, and tiny fixtures. No production
result or scientific verdict is reported here.

## Evidence boundary

The analyzer consumes two byte-authenticated frozen manifests:

- the ordered proof-problem population;
- the ordered simplification-expression population.

Issue #70 does not copy either producer's persisted result model. The generic
`authenticate_manifest` hashes exact JSON bytes and is fixture-safe only.
Production proof evidence must use
`authenticate_proof_benchmark_manifest`, which combines the external byte
checksum with issue #67's canonical producer loader and internal content
digest. Its full ordered ATP problem projection is re-derived from the
accepted problems and bound to every row's `benchmark_projection_digest`;
group, family, difficulty, and OOD metadata are compared task by task before
any pairing or bootstrap. The issue #69 sample projector independently verifies its producer
content digest and preserves the authenticated source/exclusion-manifest
digests embedded in that sample. Its complete record metadata (group, family,
depth/size stratum, domain, and split) is likewise authoritative. Saved #69
rows must bind to all three identities. Unknown task IDs, duplicate
task/method/seed cells, inconsistent cross-method task metadata, a checksum
mismatch, and duplicate manifest IDs fail loudly.

Production rows are not accepted as loose mappings or preconstructed typed
observations. `authenticate_producer_run` reloads the native #68/#69
`shard.complete.json` and per-cell JSON layout, requires every configured shard
exactly once, verifies a single run/config and input binding, checks exact
expected/attempted cell IDs and counts, rejects missing/extra/duplicate cells,
checks both cell self-digests and completion digests, and records an aggregate
SHA-256 over every exact completion/cell byte stream and relative path. It also
re-derives the producer's domain-separated canonical `run_id` from the config,
frozen-input bindings, and rule/verifier/implementation attestations. Files
outside that exact evidence set are rejected, except for each shard's optional
mutable `runner.checkpoint.json`, which is deliberately not evidence and is
excluded from the aggregate. The bundle is reauthenticated when analysis
begins. In-memory inputs remain useful for tiny fixtures, but are explicitly
unauthenticated and can never decide a production gate.

The aggregate is a trust check only when its expected value and the expected
config digest were frozen outside the analyzed run directory. The external
aggregate is SHA-256 over lexicographically ordered completion/cell paths,
framing each UTF-8 relative path and exact file bytes with an unsigned
eight-byte big-endian length. The expected config digest must come from the
pre-run resolved configuration, and the resulting aggregate/config pair must
be recorded in an independently retained release or orchestration manifest.
The analyzer never promotes its own discovered values into trust anchors.
The current #68/#69 publishers emit shard completions but no enclosing,
independently retained run-index digest; creating and freezing that external
record is therefore a real production-integration blocker for Gate G8, not
evidence that Phase A can manufacture for itself.

The raw result readers are strict scientific projections, not replacements for
the producer-owned `SearchResultV1` or simplification schema. Integration must
map those producer rows without changing their meaning.
Production Gate G8 evidence additionally requires the exact
`geml-goal8-atp-cell-v1` and `geml-goal8-simplify-cell-v1` producer schema
versions. Fixture or ad-hoc projections can render reports but cannot satisfy
the frozen production gate. Each producer cell's intrinsic canonical-content
digest is checked before any scientific field is projected.

Every raw projection retains the run, cell, config, benchmark/sample,
checkpoint, rule-set, verifier, implementation, runtime, package/hardware, and
reproduction-command identity supplied by its producer. Production provenance
requires SHA-shaped run/cell/config/artifact identities; a concrete Git commit,
Python/platform/machine/processor identity, and required package versions; and
consistent run/config identities across the study. Each row must retain a
fully rendered `--shard-index` reproduction command; commands legitimately
differ by shard and unresolved template tokens are rejected. Row identities
must also bind to the authenticated manifests. Missing, placeholder,
inconsistent, or
unbound provenance makes the gate evidence insufficient rather than silently
unattributed. The deterministic SymPy arm's producer-owned
`checkpoint_digest="not_applicable"` is the sole checkpoint-digest exception.
Issue #68 and #69 execution attestations must agree with the retained method,
checkpoint, rule set, verifier, implementation, and budget. SymPy rows bind
their implementation and wall budget through their comparator-budget digest.
An attestation mismatch remains an explicit invalid row, but it cannot support
the gate.

The accepted method/seed grid is exact, not a minimum. Unexpected methods,
extra stochastic seeds, and noncanonical deterministic seeds remain listed as
unexpected raw cells, are excluded from controlled summaries and paired
statistics (including diagnostic and gate aggregates), and make production
evidence insufficient.

## Proof accounting

A proof counts as successful only when all of the following are true:

1. the retained status is the producer's verified terminal-success state
   (`complete` in the generic projection or `success` in #68);
2. the exact target structural signature was reached;
3. the complete transition trace replayed successfully;
4. the terminal claim was verifier-confirmed;
5. no invalid transition was accepted;
6. no verifier error contaminated the claimed proof.

Semantic equivalence, same-e-class membership, a lower cost, a target string,
or a provider's claim is not proof success. Any row that claims completion or
target attainment without the conditions above is retained as an
`unverifiable_claim` and contributes zero successes.

The two producer flags `claimed_success` and `exact_target_reached` are
independent evidence. Claim-only and target-only outcomes are retained as
`replay_failed`; neither is rejected merely because the flags differ.
The explicit producer claim is retained verbatim even when a later exception,
invalid outcome, or timeout determines the final status; such a partial claim
is still an unverifiable claim rather than an ordinary failure. Fixture rows
that do not expose this producer field retain `null` and rely only on their
status/exact-target evidence. Likewise, a replay exception may truthfully
report zero or partial attempted transitions for a nonempty trace. Only a
successful proof must report replay coverage equal to the full trace.

For every method, the report publishes:

- attempted, valid, exact-success, failed, timeout, unsupported, invalid,
  verifier-error, and unverifiable-claim counts;
- exact success ratios over both attempted and valid denominators;
- observed expanded nodes over the full attempted population and separately
  over verified successes;
- proof length only for verified proofs;
- independently runner-measured search wall time and peak-memory availability;
- invalid actions, available accepted-invalid-transition telemetry, verifier
  errors, and verifier timeouts;
- termination reasons and raw seed rows;
- family, difficulty, OOD, and seed breakdowns.

Here, a *valid result row* means a verifier-safe, analyzable completed,
exhausted, budget-exhausted, or timed-out search cell. It does not mean a
successful proof. Unsupported, invalid, failed, verifier-error, and
unverifiable-claim rows remain in the attempted denominator.

Rejected illegal action proposals are reported as `invalid_action_count`.
They are distinct from `invalid_transition_count`, which means an invalid
transition entered a claimed proof and is a gate safety violation. Issue #68
does not persist a general accepted-invalid-transition counter, so analysis
does not default that field to zero. A counted successful proof establishes
zero through its complete verified replay; failed/unverified rows retain
unavailable telemetry and any unverifiable claim independently fails the gate.

## Paired statistics

All controlled methods must see the same problem, seed, and complete
`SearchBudget`:

- beam width;
- expanded-node budget;
- generated-state budget;
- proof-depth limit;
- wall-time limit;
- verifier-call budget.

For each guided method, comparisons pair on `(problem_id, seed)` and then
collapse repeated tasks/seeds to the frozen `group_id`. Stochastic uniform
search retains all three preregistered seeds; deterministic policy, policy plus
value, and transformer methods use only the canonical first seed. The paired
contrast uses only genuine shared cells and never manufactures deterministic
pseudo-replicates. Bootstrap resampling is performed over problem groups,
never over search nodes or raw repeated seed rows. The report retains:

- raw paired seed-row count;
- independent paired-group count;
- group-mean exact-success difference;
- group-mean relative expanded-node reduction over **all attempted rows**;
- the corresponding jointly-solved descriptive contrast;
- a deterministic group-bootstrap interval.

The primary contrast uses conservative all-attempt accounting: every
unsuccessful row is charged at least its configured expanded-node budget, so a
zero-work crash, unsupported row, timeout, or failed proof cannot look like an
efficiency improvement. Raw observed node counts remain published separately.
Every paired seed row and every group must have expanded-node telemetry before
the gate can decide; partial groups are never silently averaged. The
jointly-solved contrast is descriptive and cannot hide expensive failures.

## Simplification accounting

A row is a verifier-confirmed changed output only if it:

- is labeled `simplified` in the generic projection or `complete` by #69;
- is structurally changed;
- is semantic-verifier confirmed.

It is labeled an exact-cost simplification only when both source and result
costs use the same preregistered `pure_eml:official_v4` objective and prove a
strict reduction. This rule applies equally to GEML and the independent SymPy
comparator. A verifier-confirmed SymPy change with unavailable exact cost
remains a valid changed comparator output, but it is never mislabeled as an
exact-cost reduction.

A verifier-confirmed unchanged output requires semantic verification and
structural identity. It is labeled exact-cost `no_change` only when both costs
exist and are equal. Unverified outputs count as neither confirmed changes nor
confirmed unchanged outcomes. Every timeout, unsupported case, invalid result,
failure, verifier error, and unverified claim remains in the attempted
population.

Simplification reports retain producer termination reasons, error types, and
available verifier-call/error/timeout counts. Missing count mappings (including
the independent SymPy arm where search counters are not applicable) remain
null with an explicit telemetry denominator. Every stratum publishes the exact
`valid/attempted` ratio, which is the semantic-verification availability rate;
it is not inferred from a success-only metric.

Reports include exact cost, structural size, and depth changes only where the
corresponding before/after values exist. SymPy is an independently produced
method label; this analysis does not run it and never feeds a SymPy result to
GEML search. Raw projections preserve distinct source-expression,
method-result, and verification-evidence IDs, plus the domain mode and exact
five-field sample stratum. The source and result IDs are re-derived from the
canonical `geml-expression-v1\0{domain_mode}\0{sympy_srepr}` payload rather
than trusted as display labels. Availability counts accompany every
complete-case cost/size/depth mean.

Each expression has five controlled comparator cells: three stochastic
`uniform` seeds, one deterministic `policy` cell at the canonical first seed,
and one deterministic unseeded `sympy` cell. Deterministic methods are not
turned into pseudo-replicates.

Descriptive simplification contrasts pair methods on the same expression,
average repeated stochastic cells within that expression, and then collapse
expressions by frozen group before deterministic bootstrap resampling. They
report verifier-confirmed-change and exact-cost-reduction rate differences;
cost-change differences use only groups with complete exact-cost pairs and
publish their explicit availability denominators. These descriptive contrasts
do not introduce a post-hoc Gate G8 threshold.

## Deterministic products

`write_analysis_tables` atomically rebuilds:

- `goal8_summary.json`;
- `proof_methods.csv`;
- `simplification_methods.csv`;
- `paired_proof_contrasts.csv`;
- `paired_simplification_contrasts.csv`.

`render_plots` uses six fixed filenames:

- `proof_coverage.png`;
- `proof_nodes.png`;
- `proof_verifier_safety.png`;
- `simplification_outcomes.png`;
- `proof_success_by_family.png`;
- `paired_node_reduction.png`.

Plot filenames never derive from untrusted method/task labels. Plot values come
only from a validated `Goal8Report`; no hand-entered scientific values are
accepted.

## External LLM compatibility

`src/geml/experiments/goal8/llm_reference.py` is a local,
checksum-authenticated JSONL reader only. It contains no provider client,
credential handling, prompt construction, retry loop, model selection, or
network operation.

Missing #82 output is the explicit optional state `missing`. Available rows
must arrive as the checksum-authenticated `ExternalReferenceBundle`; analysis
derives state, row count, source path, SHA-256, status counts, and rows from
that bundle rather than accepting loose caller-supplied state/count values.
Rows are labeled `external_reference_only`; claimed correctness and
verifier-confirmed correctness are separate. External rows cannot alter Gate
G8 and are absent from controlled paired comparisons.

## Fixed limitations

- The production proof claim is bounded to exactly 256 frozen problems.
- The simplification claim is bounded to exactly 1,000 frozen existing-v1
  expressions.
- Goal 4 currently excludes trig/hyperbolic operators; unsupported families
  must remain visible rather than being replaced.
- Three stochastic seeds are retained individually, while deterministic
  methods have one canonical seed. They do not justify strong asymptotic
  significance claims.
- No claim generalizes to unbounded theorem proving, all v1 expressions, or
  external mathematical corpora.
- Until issues #65–#69 produce frozen authenticated results, the only honest
  gate state is `insufficient_evidence`.
- The #68 adapter consumes the producer's explicit difficulty, witness-length,
  rule-diversity, OOD, and length-OOD fields. The #69 adapter consumes and
  cross-checks the producer's depth bucket, size bucket, domain, split, and
  five-field sample stratum.

## Production integration checklist

1. Authenticate #67 through its producer loader, authenticate #69's sample
   content digest, and bind every row's task metadata to those frozen records.
2. Independently freeze each run's pre-run config digest and post-finalization
   aggregate exact-byte/path checksum in a release/orchestration manifest, then
   authenticate the complete producer shard/cell directories against those
   external values.
3. Confirm the exact 256/1,000 task populations and all required method/seed
   cells.
4. Confirm one identical global controlled ATP budget, exact simplification
   component/budget attestations, and complete paired node telemetry before
   looking at comparative results.
5. Rebuild tables and plots only from the saved rows.
6. Publish every raw row and failure category alongside aggregate claims.
7. Evaluate the source-frozen Gate G8 policy without changing thresholds.
