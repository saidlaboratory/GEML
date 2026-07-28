# Goal 8 proof-benchmark specification

## Status and scope

Issue 8-2 is `phase_a_implemented` and `production_pending`. This specification
freezes the schema, tier boundaries, quota table, deterministic selector,
leakage audit, replay gate, and immutability protocol. It does **not** freeze a
production manifest or any production task ID. Production requires the merged,
authenticated outputs of issues 6-1, 7-0, and 7-1 before model selection.

The final artifact belongs under:

```text
outputs/final/goal8/benchmark/
```

Production must contain exactly 256 unique directed proof problems. A fixture
manifest may be smaller but is labeled `fixture`, cannot be presented as the
production benchmark, and is used only by tests.

## Scientific meaning of one task

A problem is ordered:

```text
source/current expression
target/goal expression
domain and assumptions
rule-set and verifier identity
```

Source and target are already semantically equivalent by construction.
Equivalence, membership in one e-class, or a lower structural cost is not proof
success. A task is eligible only when its complete known concrete witness
replays transition by transition and reaches the exact target structural
signature.

Structural signatures are representation-independent source-expression
signatures. Channel graph hashes, e-class IDs, union-find IDs, and mutable DAG
node IDs are not task identity.

## Phase-A integration boundary

`BenchmarkCandidateV1` is a narrow adapter, not a competing persisted pair or
trace schema. At integration:

- issue 6-1 adapts `PairRecordV1`, `RewriteTraceV1`, split/group lineage, and
  source manifests into candidates;
- issue 7-0 supplies step/action lineage used by policy/value exclusion
  manifests;
- Goal 4 supplies an injected concrete proof replayer;
- issue 8-2 never copies or modifies those producer contracts.

The adapter records:

- pair ID;
- ordered source and target expression IDs and structural signatures;
- authoritative group plus all source/trace-relative groups;
- all e-class-relative IDs;
- authoritative Goal 1 split, family, domain, and sorted explicit assumptions;
- complete concrete trace states, state sizes, action digests, and rule IDs
  when the producer reports the candidate ready;
- pair/trace source-manifest IDs;
- rule-set digest, verifier version, generation seed, and source row;
- producer capability status and detail.

Unsupported, invalid, or producer-error candidates may lack a trace and still
remain explicit exclusion rows. A `ready` candidate and every accepted problem
must have one.

For `n` actions the trace has `n+1` state signatures and state sizes. Its first
and last signatures must equal the candidate source and target. Source and
target structures must differ, so zero-step identity tasks cannot enter the
benchmark.

## Stable problem identity

`problem_id` is lowercase SHA-256 over:

```text
UTF8("geml-proof-problem-v1\0") ||
canonical_json({
  schema_version,
  pair_id,
  ordered source and target expression IDs,
  ordered source and target structural signatures,
  authoritative group and sorted complete lineage groups,
  sorted e-class-relative IDs,
  split,
  family,
  domain,
  ordered assumptions,
  trace ID,
  SHA-256 of the complete canonical trace payload,
  rule-set SHA-256,
  verifier version
})
```

Canonical JSON is UTF-8, key-sorted, compact, and contains no NaN or infinity.
The ID excludes selection rank, quota, model results, search outcomes,
timestamps, hardware, and file paths. It is directional: swapping source and
target changes the ID. Changing a state signature/size, action digest, or rule
ID also changes it even if a producer accidentally reuses a trace ID. Python
`hash()` is never used.

Every accepted record revalidates its problem ID from its complete candidate.
The manifest also rejects duplicate IDs and duplicate directed
`(source signature, target signature, domain, assumptions, rule-set digest)`
tasks.

## Predeclared tiers

Tiers use only a known replayable witness and static state structure. They
cannot use benchmark search success, expansion count, runtime, model score, or
any post-evaluation result.

### Known witness length

The production boundaries are:

| Tier | Known replayable steps |
|---|---:|
| `short` | 1–2 |
| `medium` | 3–4 |
| `long` | 5–8 |
| `length_ood` | at least 9 |

Before curation, the authenticated development ledger must prove that 8 is the
largest witness length exposed to training or model selection. Otherwise these
rows are not valid strict length-OOD evidence and production must stop.

The known witness is not claimed shortest unless a separate shortest-path proof
exists. Thus the tier means known-witness length, not mathematical minimum
proof distance.

### Rule diversity

Rule diversity is the number of distinct `rule_id` values in the witness:

| Tier | Distinct rules |
|---|---:|
| `single` | 1 |
| `moderate` | 2 |
| `high` | at least 3 |

### Static difficulty

Difficulty uses:

```text
score =
    known_witness_length
  + 2 * (distinct_rule_count - 1)
  + ceil(maximum_witness_state_size / 8)
```

`maximum_witness_state_size` is the maximum canonical source-AST tree node
count across the concrete witness states. It counts ordered occurrences rather
than channel-specific DAG nodes, so the tier does not depend on a learned graph
representation.

| Tier | Score |
|---|---:|
| `easy` | at most 8 |
| `medium` | 9–16 |
| `hard` | at least 17 |

The formula is a preregistered structural grouping device, not a calibrated
probability of ATP success.

### OOD labels

Length and family support are represented independently of the source split:

- `length_family_in_distribution`;
- `length_ood`;
- `family_ood`;
- `length_and_family_ood`.

Goal 1 `test_ood` remains its documented combined `ood_stress` profile. It is
not renamed strict depth-OOD, and
`length_family_in_distribution` does not relabel a `test_ood` row as globally
IID. The authoritative `split` remains a separate quota axis.

`exp_log` is the preregistered family-OOD view because Goal 4 supports exp/log.
This label is conditional: the complete development ledger must prove
`exp_log` absent from all train, validation, policy-training,
policy-selection, value-training, and value-selection sources. If the
integrated training design does not make that true, curation stops; it must not
silently relabel or replace the family.

## Frozen production quotas

The selector uses 32 exact, nonoverlapping composite cells of eight tasks each.
The checked-in YAML is the authority for every cell. Its marginals are:

| Axis | Frozen totals |
|---|---|
| Family | 96 `algebraic_core`, 96 `powers_division_rationals`, 64 `exp_log` |
| Known witness length | 64 each: short, medium, long, length-OOD |
| Rule diversity | 96 single, 80 moderate, 80 high |
| Static difficulty | 48 easy, 128 medium, 80 hard |
| Strict length/family support | 144 length/family in-distribution, 48 length-OOD, 48 family-OOD, 16 combined |
| Authoritative source split | 128 test-IID, 128 combined test-OOD-stress |

These figures are derived from the composite cells; they are not separately
optimized marginals. A candidate belongs to at most one cell. Any cell
shortfall stops publication and is retained in `CurationReportV1` as a
`quota_shortfall` row.

Plan validation proves that every declared witness-length, rule-diversity, and
difficulty combination is mathematically inhabitable. In particular, a
witness cannot contain more distinct rules than actions. The two additional
short algebraic/power cells therefore use `single`/`medium`, while corresponding
long `single`/`medium` cells use `high`/`medium`; this preserves every frozen
marginal above without retaining impossible short/high cells.

The accepted quota table uses only the current Goal 4 algebraic, power,
exp/log-capable families. Candidates from `trig_hyperbolic`,
`mixed_elementary`, or `ood_stress` that require unsupported trig/hyperbolic
operators remain explicit `unsupported` or outside-quota exclusions. They are
not relabeled as covered tasks. If eligible replayable candidates cannot fill
an exact supported cell, the correct result is a loud production blocker.

## Deterministic selection algorithm

The algorithm version is:

```text
geml-proof-benchmark-selection-v1
```

It performs these steps:

1. Validate the plan, exact quota sum, source roles, and family/OOD
   consistency.
2. Authenticate the raw bytes of all five source artifacts.
3. Validate the complete leakage ledger and held-out-family evidence.
4. Derive every problem ID and sort candidates independently of input order.
   Candidates sharing a scientific ID are tie-broken by their full canonical
   candidate bytes, so stable-sort input order can never select the survivor.
5. Retain duplicate complete identities as `duplicate_candidate`.
6. Reject train/validation source rows and every development-lineage overlap.
7. Retain producer `unsupported`, `invalid`, and `error` statuses.
8. Assign tiers from the preregistered static formula.
9. Retain candidates outside the exact quota table.
10. Replay every otherwise eligible known proof. Exceptions, timeouts,
    unsupported transitions, invalid transitions, identity mismatches, and
    wrong terminal structures are failures.
11. When multiple replayable traces represent the same directed task, retain
    the shortest known replayed witness, breaking ties by problem ID. All
    alternatives remain `duplicate_task` rows. A failed witness never blocks a
    replayable witness.
12. Within each cell, rank by:

    ```text
    SHA256(
      UTF8(
        "geml-proof-benchmark-selection-v1\0"
        + selection_seed + "\0"
        + quota_id + "\0"
        + problem_id
      )
    ),
    then problem_id
    ```

13. Select the exact cell count and retain lower-ranked replayable candidates
    as `quota_filled`.
14. Refuse to construct a publishable manifest if any cell is short.

The production seed is `20260726`. Seed, cells, and tier boundaries are frozen
before policy/value model selection and cannot be changed after search results
are observed.

## Leakage proof

`LeakageLedgerV1` contains eight explicit scopes:

1. train pairs;
2. validation pairs;
3. train traces;
4. validation traces;
5. policy training;
6. policy model selection;
7. value training;
8. value model selection.

Each scope supplies sorted unique:

- group IDs;
- e-class-relative IDs;
- pair IDs;
- trace IDs;
- family inventory.

The ledger also records the authenticated maximum witness length exposed
anywhere in development. It must equal the preregistered
`in_distribution_witness_max` whenever length-OOD quotas exist.

For every candidate, the curator checks the authoritative group and every
lineage group, every e-class relative, pair ID, and trace ID against every
scope. It records every concrete hit with its scope and identity type.
Accepted overlap count must be zero. Excluded overlaps remain counted and
visible.

Family-OOD publication additionally requires a declaration that the
development family inventory is complete. An empty or partial inventory is not
evidence of absence: every development role must provide a nonempty family
inventory before a held-out-family claim is accepted.

The benchmark manifest and future policy/value checkpoints must consume the
same frozen exclusion manifest. Benchmark groups must be frozen before
production policy/value model selection.

## Replay gate and current verifier coverage

The injected `ProofReplayer` returns:

- verifier version;
- rule-set SHA-256;
- one typed status for every transition;
- exact final structural signature;
- complete status and detail.

Accepted problems require:

- producer status `ready`;
- replay verifier/rule identities equal to both candidate and plan;
- exactly one `verified` outcome per witness action;
- complete status `verified`;
- final signature exactly equal to the directed target.

A verifier exception is caught and retained as `replay_error`; it can never
become acceptance.

Goal 4 currently supports algebraic, power, exp, and log operators. It does not
support trig or hyperbolic operators. The benchmark therefore does not claim
full-v1 family coverage. Capability failures remain ledger rows and quota
shortfalls remain blockers.

## Authenticated source artifacts

Every plan and manifest includes exactly one of each source:

- pair manifest;
- trace manifest;
- split manifest;
- leakage manifest;
- rule registry;

Each reference binds an artifact ID, role, path, schema version, and lowercase
raw-file SHA-256. Production curation requires `source_root`, reads every file,
and rejects missing or mismatched bytes before candidate selection. Each file
is read exactly once: its digest is checked and every parser consumes that same
authenticated byte snapshot, so a path cannot change between hashing and
projection.
Candidate pair/trace provenance and the leakage-ledger ID must name the
corresponding authenticated artifacts.

Pair, trace, and split sources are narrow
`SourceBindingManifestV1` JSON views over the authoritative producer
manifests. They do not copy producer contracts. Each sorted binding contains a
source-row ID, role-specific record ID, and SHA-256 of a canonical projection:

- pair: ordered pair/expression IDs and exact endpoint signatures;
- trace: complete trace payload, producer status/detail, and generation seed;
- split: group/lineage/e-class IDs, authoritative split, family, domain, and
  assumptions.

`derive_source_binding` applies the domain-separated
`geml-proof-source-binding-v1` digest. Production curation recomputes all three
bindings for every input candidate and rejects a missing or mismatched row.
The authenticated `RuleRegistryEvidenceV1` must match the plan's rule-set
digest and contain every rule ID used by every concrete trace. The leakage
source is validated as a `LeakageLedgerV1` JSON document and must equal the
in-memory ledger exactly. Every parsed artifact's schema version and artifact
identity must equal its plan entry; the curator cannot authenticate one file
while selecting from unrelated in-memory rows.

Manifest loading repeats the scientific identity checks independently of the
curator. Every accepted candidate's pair/trace artifact IDs and every candidate
and replay rule-set/verifier identity must equal the manifest's embedded plan.
Rehashing a detached accepted row therefore cannot create a valid manifest.

The benchmark config is authenticated separately by `config_sha256`; it is not
listed as its own source artifact because embedding a file's checksum inside
that same file would create an invalid self-reference.

The curation environment records:

- implementation commit;
- Python and package versions;
- platform and hardware;
- exact reproduction command.

Production rejects fixture, pending, unknown, or otherwise placeholder
environment values. The implementation commit must be a full concrete
lowercase hexadecimal Git hash, and Python/platform/hardware/command plus every
package version must be concrete. Fixture manifests may retain lightweight
fixture metadata.

The plan records the selection seed, config SHA-256, rule-set SHA-256, verifier
version, and all source checksums.

## Failure and exclusion accounting

Every input candidate produces either one accepted problem or an explicit
ledger outcome. Typed reasons include:

- duplicate candidate or directed task;
- train/validation source row;
- development leakage;
- producer unsupported, invalid, or error;
- outside the preregistered quota;
- replay unsupported, invalid, timeout, error, or identity mismatch;
- valid replay below a full quota cutoff;
- aggregate quota shortfall.

Candidate exclusions retain pair ID, derived problem ID, producer status,
quota where known, detail, and replay evidence where attempted. Quota results
record required, eligible, accepted, and missing counts. Failures are never
silently dropped or replaced by a favorable denominator.

## Manifest and immutability

`BenchmarkManifestV1` includes:

- schema and benchmark identity;
- selection algorithm and seed;
- target count;
- embedded held-out-family/selection plan plus plan and content digests;
- authenticated source artifacts;
- tier policy and exact composite cells;
- per-cell denominators;
- accepted records and full exclusions;
- leakage audit;
- config/rule/verifier identity;
- curation environment.

`content_sha256` authenticates canonical JSON for every field except itself.
The manifest file is compact, key-sorted canonical UTF-8 JSON with one terminal
LF. The file SHA-256 authenticates those exact bytes.

First publication uses atomic create-without-replacement. If a path already
exists:

- identical bytes are verified and returned;
- different bytes are rejected;
- nothing is overwritten.

After the production file checksum is copied into
`frozen_manifest_sha256`, every rerun uses verify-only mode. The file must
already exist, match the configured SHA-256, validate its internal digest, and
equal the newly curated bytes. Verify-only mode never rewrites it.

Any scientifically necessary task change requires a new schema/benchmark
version and a documented reason. Task IDs, quotas, memberships, or tier
boundaries cannot change after model or benchmark-result inspection.

## Production procedure

Do not execute this procedure until Workstreams 1–3 are integrated:

1. Freeze pair/trace/split outputs and the complete development-exclusion
   ledger.
2. Generate authenticated pair/trace/split binding views and rule-registry
   evidence from those frozen sources.
3. Confirm the nonempty family inventories and maximum development
   witness-length claim in the authenticated ledger.
4. Replace every null source path/schema/checksum in
   `configs/goal8_benchmark.yaml`.
5. Compute and record the final config SHA-256 without changing quotas.
6. Adapt producer records to `BenchmarkCandidateV1`.
7. Curate with the integrated Goal 4 replayer and authenticated `source_root`.
8. Audit every exclusion and exact quota denominator.
9. Require exactly 256 accepted unique problems and zero accepted leakage.
10. Publish atomically, record the file checksum, and switch future runs to
   verify-only mode.
11. Freeze benchmark groups before policy/value training or checkpoint
    selection.

No production manifest, ATP run, model training, or benchmark tuning was
performed in Phase A.
