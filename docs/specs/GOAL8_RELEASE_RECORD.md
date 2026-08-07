# Goal 8 release record specification

## Status and ownership

Gate G8 (`docs/goals/goal8/GATE_G8.md`) returns `insufficient_evidence`
whenever "either producer bundle lacks independently frozen expected aggregate
and pre-run config digests, or its exact bytes/config disagree with those
external trust anchors". The #68/#69 runners publish immutable per-cell JSON
and per-shard `shard.complete.json` completions, but no enclosing record of
those anchors, and the #70 analyzer is explicitly forbidden from promoting its
own discovered values into trust anchors.

This specification assigns ownership of that record to the Goal 8 run
orchestration step (ws4). The record is:

- **emitted** by `emit_release_record` in
  `src/geml/experiments/goal8/release_record.py`, run by the operator who
  finalizes a production study;
- **consumed** by issue #70's `authenticate_producer_run` through
  `load_release_record`, which supplies `expected_aggregate_sha256` and
  `expected_config_digest` for each bundle.

The analyzer never issues, repairs, or regenerates this record. A missing or
invalid record leaves Gate G8 at `insufficient_evidence`; it is never treated
as a scientific failure.

## Record schema

One JSON object with schema `geml-goal8-release-record-v1`:

```json
{
  "schema_version": "geml-goal8-release-record-v1",
  "bundles": {
    "proof_benchmark": {
      "kind": "proof_benchmark",
      "run_id": "<sha256>",
      "config_digest": "<sha256>",
      "aggregate_sha256": "<sha256>",
      "shard_count": 8,
      "cell_count": 2304,
      "byte_count": 123456789,
      "cell_schema_version": "geml-goal8-atp-cell-v1",
      "shard_schema_version": "geml-goal8-atp-shard-v1"
    },
    "simplification_benchmark": {
      "kind": "simplification_benchmark",
      "run_id": "<sha256>",
      "config_digest": "<sha256>",
      "aggregate_sha256": "<sha256>",
      "shard_count": 8,
      "cell_count": 5000,
      "byte_count": 123456789,
      "cell_schema_version": "geml-goal8-simplify-cell-v1",
      "shard_schema_version": "geml-goal8-simplify-shard-v1"
    }
  },
  "content_digest": "<sha256>"
}
```

`bundles` must contain exactly the two producer kinds: the #68 ATP run and the
#69 simplification run. Every digest is a lowercase SHA-256 hexadecimal
string; counts are strict positive integers. `content_digest` is SHA-256 over
`geml-goal8-release-record-v1\0` plus the canonical JSON (sorted keys, compact
separators, ASCII, no NaN) of the record without the `content_digest` field.
The loader recomputes and enforces it, so an edited record fails loudly. A
record whose checksum chain was reforged wholesale still cannot authenticate a
run: its `aggregate_sha256`/`config_digest` then disagree with the exact run
bytes or the pre-run configuration inside `authenticate_producer_run`.

## The two trust anchors

**`config_digest` (pre-run).** The canonical-JSON digest of the resolved
runner configuration, exactly as printed by
`python -m geml.experiments.goal8.run_atp <config> --validate-only` and
`python -m geml.experiments.goal8.run_simplify <config> --validate-only`
before any shard executes. The operator captures this value before the study
runs and passes it to the emitter. The emitter never derives it from the run
outputs it is anchoring; a run directory whose retained `config_digest`
disagrees with the supplied pre-run value is refused.

**`aggregate_sha256` (post-finalization).** SHA-256 over the finalized run
directory's evidence files — every `shards/shard-NNNNN/shard.complete.json`
and every file under `cells/` — in lexicographic path order, framing each
file's UTF-8 relative path and exact bytes with unsigned eight-byte big-endian
lengths. This is byte-for-byte the aggregate `authenticate_producer_run`
computes; each shard's mutable `runner.checkpoint.json` is deliberately not
evidence and is excluded. Freezing this value outside the run directory is
what turns the analyzer's aggregate from a self-check into a trust check.

## Who emits it, and when

The run orchestrator (the operator finalizing a production study, ws4 lane)
emits the record exactly once per release:

1. after **every** shard of **both** runs has published its
   `shard.complete.json` — the emitter refuses partial runs;
2. before any #70 analysis consumes the runs;
3. to a path outside both run directories, on independently retained storage
   (the release/orchestration area, not the producer output roots).

## Fail-closed emission

`emit_release_record` walks each run directory, computes the aggregate, and
then revalidates the complete run through the analysis-side
`authenticate_producer_run` with the supplied pre-run config digest and the
freshly computed aggregate as anchors. Emission is refused — nothing is
written — when any of the following holds:

- a run directory cannot be resolved, is not named by its re-derivable
  canonical `run_id`, or is missing shards, cells, or completions;
- any shard is missing/duplicated, any cell is missing/extra, any
  completion/cell digest chain disagrees, or the shards do not belong to one
  run/configuration;
- the run's retained `config_digest` differs from the supplied pre-run value;
- the record path lies inside either run directory;
- a record already exists at the path with different content (byte-identical
  re-emission is an accepted resume; the record is otherwise immutable).

The emitter records the schema versions it validated against. Production runs
must carry the frozen `geml-goal8-atp-cell-v1` / `geml-goal8-simplify-cell-v1`
schemas; the schema-override parameters exist only so tiny fixture runs can
exercise the protocol, mirroring `authenticate_producer_run`, and a
fixture-schema record can never satisfy the production gate.

## Consumption by the analyzer

Issue #70 loads the record with `load_release_record` (which verifies the
schema, field shapes, and `content_digest`) and passes each bundle's
`aggregate_sha256` and `config_digest` to `authenticate_producer_run`. Only a
bundle whose exact bytes and configuration match these externally retained
values reports `trust_anchor_verified`, which Gate G8 requires before any
production verdict. Tampered run bytes, a tampered record, or a
reforged-checksum record all fail authentication rather than degrading to a
warning.
