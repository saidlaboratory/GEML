# Goal 6 trace-rich equivalence pairs

Schema version: `geml-pair-record-v1`.

This contract defines the future 50,000 train / 5,000 validation / 5,000 test
base records. Phase-A fixtures use the same records but must not be presented
as production data. Goals 1--5 artifacts remain read-only.

## Pair identity and partitions

`pair_id` is a version-tagged SHA-256 digest over canonical JSON containing
the two endpoint identities in canonical unordered order, source split, full
lineage union, label/status/evaluation views, evidence identity, and any typed
outcome. It never uses Python `hash()`, a timestamp, display text, a seed, or
a shard location.

Each endpoint carries `expression_id`, authoritative `sympy_srepr`, structural
signature, family/domain metadata, source split, and `GroupLineageV1`. The
lineage contains a primary source/e-class-relative group plus its canonical
relative closure. Both endpoints and every relative must belong to one source
partition only. Positive derived endpoints inherit their trace source group.

The base counts are fixed: 50,000 train, 5,000 validation, and 5,000 each in
`test_iid` and `test_ood`. IID, strict depth-OOD, and family-OOD are explicit
views over those records, not additional or oversampled datasets. The stored
Goal 1 `test_ood` profile must not be renamed strict depth-OOD.

## Concrete action and trace evidence

`RewriteActionV1` persists `rule_id`, direction, ordered root-to-occurrence
child-slot path, ordered argument bindings, source and successor structural
signatures, assumptions/domain, and a semantic digest. It must never use an
e-class, union-find, or shared-DAG node ID as its application-site identity.
The semantic digest deliberately excludes mutable verifier outcomes,
timestamps, and execution evidence.

`RewriteTraceV1` stores a directed source/goal, `n + 1` concrete states for
`n` actions, and one verifier outcome per transition. Replaying an action from
each state must reproduce the next stored structural signature. Goal 4
provenance is not itself a concrete trace and cannot be relabeled as one.

Every accepted positive has a passed full trace. A failed, unsupported, or
unreplayable trace remains a retained rejection/failure row.

## Deterministic concrete trajectories

`generate_concrete_trajectory_attempts` in
`geml.data.pairs.generate` is the only Phase-A trajectory orchestration path.
It accepts injected read-only callbacks to enumerate legal actions and apply a
concrete action through the authoritative rule engine/verifier. It never turns
Goal 4 saturation provenance into a trace.

For source expression `E` and zero-based attempt `a`, it derives a local RNG
seed using `geml-derived-seed-v1` over canonical JSON with the component
`goal6.pairs.trajectory`, the frozen run seed, immutable source ID `E`, and
`a`. The first eight SHA-256 bytes are an unsigned big-endian 64-bit integer.
The local `random.Random` instance selects only from a lexicographically
digest-sorted legal action list; no ambient global RNG is used.

`TrajectoryPolicyV1` fixes minimum/maximum trace length, attempts per source,
and whether the exact inverse of the immediately previous action is excluded.
Each attempt produces an `TrajectoryGenerationOutcomeV1` row. Accepted rows
contain a complete passed `RewriteTraceV1`; exhausted, shortfall, invalid,
unsupported, verifier-error, and timeout rows carry a typed outcome instead of
being silently retried away. A quota scheduler may use the first accepted row
only while persisting all earlier outcomes.

## Negatives

Negatives are size-matched structural near misses or same-family candidates
with recorded non-equivalence evidence. A high-precision numeric disagreement
is retained as `numeric_counterexample` evidence but cannot become an accepted
negative without rigorous interval/error-bound support. Accepted negatives
therefore require `formal_counterexample` evidence marked rigorous. Unresolved
candidates remain rejected/failure rows and never receive a negative label.

`assess_hard_negative` applies that predeclared admission order: reject an
exact structural match, reject a size-tolerance violation, retain a candidate
without rigorous formal evidence as `unresolved`, and accept only a rigorous
formal counterexample. The candidate's source operator family remains attached
for auditable same-family balance accounting; it is never used as a shortcut to
assign a label.

## Output and resumability

Production writes are sharded under `outputs/final/goal6/pairs/` with canonical
JSONL content hashes, config hash, rule/verifier provenance, rejection/failure
statistics, and explicit attempted/accepted denominators. Writers must use
atomic finalization without overwriting earlier failure evidence. Unit tests use
only hand-written or temporary fixture records.

`write_pair_shard` persists canonical JSONL sorted by `pair_id` and a
`geml-goal6-pair-shard-manifest-v1` completion sidecar. The sidecar binds the
shard ID, configuration/input/rule-set hashes, content hash, and accepted,
rejected, and failed counts plus outcome, split, label, endpoint
family/domain, frozen depth, trace-length, and verifier-tier denominators.
`write_pair_shard` requires an explicit frozen depth for every pair rather than
substituting an unknown bin. Resuming is exact-bytes-only: a pre-existing shard
or manifest with different content raises an integrity error rather than being
overwritten. Completion is published after the shard bytes, so a missing
sidecar remains visibly incomplete.
