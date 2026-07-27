# Goal-conditioned rewrite-step dataset

**Contract:** `geml-step-record-v1`
**Issue:** 7-0 / #61
**Production root:** `outputs/final/goal7/steps/`
**Phase-A state:** implemented against tiny injected fixtures; production pending

## Scientific target

Each accepted row is the directional supervised tuple

```text
current_state
goal_state
action
next_state
remaining_witness_steps
```

The goal is mandatory. The same current expression can lie on demonstrations
toward different equivalent targets, with different correct next actions.
Current and goal are ordered roles and are never swapped or canonicalized.

`remaining_witness_steps = trace_length - step_index` is the length of the
verified suffix in the stored demonstration. It is a witnessed upper bound on
distance to this particular goal. It is not labeled as a minimum or
shortest-path distance unless a separate future procedure proves optimality.

## Producer authority and Phase-A adapter

Issue #55 owns `RewriteTraceV1` and `RewriteActionV1`. Issue #61 does not
duplicate either persisted schema. The temporary `NormalizedTraceV1`,
`NormalizedStateV1`, and `NormalizedActionV1` classes are narrow read-only
compatibility views. After Workstream 1 integration, its adapter must:

1. hash the exact source record bytes, even when the record is malformed;
2. validate the producer schema version and all producer-defined IDs/hashes;
3. validate the producer's complete-replay flag and verification evidence;
4. return canonical source-expression states and representation-independent
   structural signatures;
5. return the producer's authoritative rule-set/action/verifier digests;
6. include the source, trace, derived-relative, and e-class-relative lineage
   groups needed for partition auditing.

The normalized action view preserves:

- rule ID and explicit `forward`/`backward` direction;
- the ordered root-to-occurrence child-slot path;
- ordered arguments or bindings as lossless canonical JSON values;
- source and successor structural signatures;
- assumptions and domain mode;
- the semantic action digest;
- producer verifier status, verifier digest, and separate evidence digest.

An e-class ID, union-find ID, DAG node ID, or substitution map alone is not an
application-site identity. Ordered paths distinguish repeated occurrences.
The action digest is owned by the producer and is not redefined here.

The authoritative `rule_set_digest` in a step row is the same digest that Goal
7 proposals must call `rule_registry_digest` for the same executable frozen
registry. The descriptive registry sidecar also has a separate normalized-entry
digest; that sidecar digest is not a replacement for executable rule identity.

The production configuration also pins one `expected_verifier_digest`.
Acceptance requires exact equality among that configured digest, the verifier
digest authenticated in the producer transition, the injected verifier's
declared digest, and the digest returned by the fresh verification call. This
is intentionally an exact-identity policy: no unspecified compatibility map is
invented. Producer-stored and freshly generated evidence digests remain
separate evidence for different executions and are **not** required to equal
one another. Producer status, identity, and evidence occupy dedicated failure
fields; fresh verification fields remain null unless the injected verifier was
actually called and returned a typed result. An evidence digest is never
substituted for a verifier identity.

## `StepRecordV1`

Every accepted record contains:

| Category | Fields |
|---|---|
| Schema and identity | `schema_version`, `record_id` |
| Source lineage | `trace_id`, `trace_digest`, `pair_id`, `source_id`, `source_group`, `lineage_group_ids`, `authoritative_split`, `evaluation_views` |
| Position | `step_index`, `trace_length`, `remaining_witness_steps` |
| Ordered states | `current_state`, `goal_state`, `next_state` |
| Structural identity | `current_signature`, `goal_signature`, `next_signature`, `action_source_signature`, `action_successor_signature` |
| Action | `action`, `action_digest`, `rule_id`, `direction`, `occurrence_path`, `ordered_arguments`, `assumptions` |
| Scientific strata | `source_family`, `current_family`, `goal_family`, `domain_mode`, `rewrite_mode` |
| Rule/replay evidence | `rule_set_digest`, `supported`, `replay_status`, `verification_status`, `verifier_digest`, `verification_evidence_digest` |

All states and action data are immutable `CanonicalJson` snapshots. Persisted
JSON decodes them to their original JSON type; canonicalization does not
stringify or reorder arrays. JSON objects use sorted keys, ASCII escaping,
compact separators, finite numbers, and no duplicate/non-finite values.

Accepted rows require:

- `step_index < trace_length`;
- `remaining_witness_steps == trace_length - step_index`;
- action source signature equals current signature;
- action successor signature equals next signature;
- exact replay returns the stored next canonical state, and the adapter
  independently recomputes that state's stored structural signature;
- fresh verifier status is `accepted`;
- producer, configured, injected, and fresh-result verifier digests agree;
- `supported`, replay `applied`, and verifier `accepted`.

### Stable record ID

`record_id` is:

```text
lowercase_hex_sha256(
  UTF8("geml-step-record-v1\0")
  || canonical_json(scientific_identity)
)
```

`scientific_identity` binds the ordered current/goal/next state snapshots and
signatures, complete normalized action and action digest, rule/direction/path
and ordered arguments, step/trace position, trace/pair/source identity,
source/group/split lineage, families/domain/mode, assumptions, and rule-set
digest. It deliberately excludes timestamps, filesystem paths, fresh verifier
status text, and mutable runtime telemetry. Python `hash()` is never used.

## Replay and acceptance

For each authenticated positive trace:

1. require `len(states) == len(actions) + 1`;
2. reject a zero-action trace as a typed `zero_length_trace` row;
3. require the producer's complete-replay flag;
4. for each step, compare action source/successor signatures to stored states;
5. enumerate and apply the exact action, including direction, ordered path,
   and ordered arguments;
6. compare the replayed canonical state to the stored next state, independently
   recompute its structural signature through the authoritative adapter, and
   bind both to the stored next signature (the replayer's claimed signature is
   never trusted by itself);
7. run the configured verifier under recorded assumptions/domain;
8. require its declared and returned identity to equal the configured and
   producer-stored verifier digest;
9. retain the fresh evidence digest separately from producer evidence.

The whole trace is authenticated and replayed before accepting any step. If
one transition fails, that transition keeps its primary typed failure and
otherwise valid transitions become `incomplete_trace` rows. This prevents a
row from claiming a witnessed suffix that contains an unverified transition.
No trace is repaired, truncated, or silently skipped.

## Failure rows

`geml-step-failure-v1` retains a stable `failure_id`, adapter-computed exact
input-record digest,
deterministic `input_occurrence_index` for repeated byte-identical inputs,
reason, and every available trace/action/group/stratum/replay/verifier field.
Primary failure codes are:

```text
trace_authentication_failed
corrupt_trace
incomplete_trace
zero_length_trace
group_leakage
ambiguous_site
missing_rule
missing_direction
invalid_arguments
invalid_site
source_signature_mismatch
successor_signature_mismatch
unsupported_operator
unsupported_domain
verifier_rejected
verifier_unsupported
verifier_timeout
verifier_error
verifier_identity_mismatch
replay_error
```

Unexpected adapter, replay, and verifier exceptions are converted to explicit
failure rows. They are never interpreted as negative examples or dropped from
denominators.

## Partition and leakage policy

Rows inherit source group and authoritative split exactly. The extractor does
not reassign a row. It audits namespaced trace, pair, source, and all supplied
lineage-group identities. If any identity occurs in more than one split, all
affected traces are rejected as `group_leakage`; no accepted row crosses a
partition. Duplicate trace IDs or exact input-record digests in one split are
also rejected rather than duplicated into training.

`evaluation_views` and full lineage remain in the record so proof-benchmark
groups can later be excluded from training. A filtered test family is not
called held out unless it was absent from training.

## Descriptive stratification

`geml-step-stratification-report-v1` counts every accepted and failure row
without sampling or reweighting:

- rule and direction;
- root-zero occurrence depth;
- current and goal family;
- remaining witnessed suffix length;
- trace length;
- domain and rewrite mode;
- supported, unsupported, or unknown status;
- typed failure code.

The authenticated frozen registry inventory contributes an explicit row for
every rule/direction, including zero-count and unsupported entries. Evaluation
rows are never oversampled. A future training sampler is a separate concern and
must preserve original weights and denominators.

## Deterministic shards and resume

The exact input digest is a version-tagged SHA-256 over the sorted multiset of
exact producer-record digests. Input order therefore cannot change output row
order or content, while duplicates remain detectable. Accepted and failure
rows are canonically sorted and written together to:

```text
shards/shard-00000.jsonl
shards/shard-00001.jsonl
...
```

Every line has `schema_version`, `row_type`, `row_id`, and the typed `row`.
Shard receipts retain path, index, byte/row/type counts, first/last row IDs, and
SHA-256. Immutable publication uses a temporary file and no-replacement link.
Resume accepts an existing file only when its exact bytes match, and rejects
changed or stale extra shards. Before resuming, the writer may remove only its
own exact orphan form `.<allowed-destination>.<8-char-tempfile-token>.tmp`;
lookalikes, unrelated temporary files, symbolic links, and all other unlisted
files fail closed.

Before creating any output, the writer reruns the accepted-row partition audit
and requires every accepted `(rule_id, direction)` to be present and supported
in the frozen registry. A leaking, absent, or unsupported accepted row cannot
publish even a partial bundle.

Sidecars are:

```text
config.json
per-rule-manifest.json
replay-audit.json
split-audit.json
manifest.json
```

The manifest binds config, input, authoritative rule-set, verifier identity,
every shard, every sidecar, counts, seed, deterministic runtime identity, and
the exact reproduction command. Loading recursively verifies canonical bytes,
hashes, receipt counts, typed row schemas and derived IDs, and expected
config/input/rule/verifier digests. It parses `config.json` as
`StepDatasetConfigV1` and cross-binds its digests, seed, shard size, runtime,
and exact command to the manifest. It typed-parses the replay and registry
evidence, then reconstructs the replay audit, split audit, and complete
stratification/per-rule report from the authenticated rows. A sidecar therefore
cannot become false merely by rehashing it and its manifest. Loading also
rejects symbolic links and any file not listed by the manifest, including stale
shards and writer temporaries.
`dataset_tree_digest()` supports identical-two-run fixture and production
audits.

## Phase-A and production boundary

`configs/goal7_steps.yaml` intentionally leaves the three production providers,
expected input/rule/verifier digests, and `production_command` null. Its command
status is explicitly `pending_workstream_1_integration`; it does not advertise
a no-op `python -m` command. That is an explicit blocker, not a fallback. Phase
B must integrate the authenticated Workstream 1 trace/action producer and Goal
4 replayer/verifier, replace those null values, and record a concrete
executable command and environment.

Phase A uses only tiny hand-written fixtures and temporary output directories.
It does not read `outputs/`, generate production traces, extract the production
step dataset, train a policy, or modify Goals 1–5 artifacts.
