# Corpus and run-manifest schema

`geml.contracts.corpus` freezes split names and final-corpus counts and defines validated
shard/split/corpus manifests, result-bearing run metadata, and retained error rows. The module
performs record validation only; it does not read files, generate data, calculate checksums, or
shard a corpus.

## Final result-bearing corpus

The final production corpus specification is exposed as `FINAL_CORPUS_SPLIT_COUNTS` and
`FINAL_CORPUS_TOTAL_COUNT`:

| Split | Rows |
|---|---:|
| `train` | 175,000 |
| `validation` | 25,000 |
| `test_iid` | 25,000 |
| `test_ood` | 25,000 |
| **total** | **250,000** |

These constants specify the final run; generic manifests do not enforce them. Unit tests and
local validation may use tiny temporary corpora with any internally consistent nonnegative row
counts.

## Fields

### `ChecksumRecord`

Both fields are required: nonempty `algorithm` and an even-length hexadecimal `digest`. The
producing issue selects and calculates the checksum algorithm.

### `CorpusShardManifest`

Required fields are `schema_version`, `corpus_id`, `shard_id`, `path`, `split`, `shard_index`,
`row_count`, and `checksum`. `byte_count` is nullable and defaults to null. `error_row_count`
defaults to zero, and JSON-compatible `metadata` defaults to `{}`. Indexes and counts are
nonnegative.

### `SplitManifest`

Required fields are `schema_version`, `corpus_id`, `split`, a nonempty ordered `shards` array,
and `total_row_count`. `total_error_row_count` and JSON-compatible `metadata` are optional with
zero/empty defaults. Shard IDs must be unique. Shard indexes must be zero-based, contiguous, and
match array order. Shard schema, corpus, and split fields must match the parent; declared totals
must equal shard sums.

### `CorpusManifest`

Required fields are `schema_version`, `corpus_id`, a nonempty ordered `splits` array,
`total_row_count`, `config_hash`, `generator_seed`, timezone-aware `created_at`, `git_commit`,
`python_version`, `platform`, and a nonempty `package_versions` object.
`total_error_row_count` and JSON-compatible `metadata` have zero/empty defaults. Split names and
all shard IDs must be unique, child schema/corpus IDs must match, and declared totals must equal
the split sums. `created_at` validates from a timezone-aware Python datetime or ISO 8601 string;
numeric epoch coercion is not permitted. The manifest counts and checksums are authoritative for
the artifact it declares.

### `RunMetadata`

All reproducibility/accounting fields are required: `run_id`, `stage`, `config_hash`,
`random_seed`, `git_commit`, `python_version`, `platform`, nonempty `package_versions`,
timezone-aware `started_at` and `ended_at`, nonnegative `elapsed_seconds`, nonempty ordered
`input_manifests`, `processed_count`, `success_count`, `failure_count`, and
`reproduction_command`. JSON-compatible `metadata` defaults to `{}`. End time cannot precede
start time, and processed count must equal successes plus failures. Timestamps validate from
timezone-aware Python datetimes or ISO 8601 strings; numeric epoch coercion is not permitted.

### `ErrorRow`

`expression_id` and `shard_id` are optional nullable identities. Required fields are nonempty
`stage`, `error_type`, `message`, and `status`, plus boolean `recoverable`. JSON-compatible
`metadata` defaults to `{}`. Status remains an explicit producer-defined string so future issue
work can freeze a status registry without creating a competing registry here.

Model field reassignment is blocked, ordered collections validate into tuples, undeclared fields
are rejected, and metadata must be JSON-compatible. JSON objects remain ordinary dictionaries,
so callers that require deep immutability must not mutate nested metadata. Integer fields reject
booleans instead of coercing them to `0` or `1`.

## JSON-compatible example

```json
{
  "schema_version": "1.0.0",
  "corpus_id": "tiny-fixture",
  "splits": [
    {
      "schema_version": "1.0.0",
      "corpus_id": "tiny-fixture",
      "split": "train",
      "shards": [
        {
          "schema_version": "1.0.0",
          "corpus_id": "tiny-fixture",
          "shard_id": "train-00000",
          "path": "train/train-00000.parquet",
          "split": "train",
          "shard_index": 0,
          "row_count": 2,
          "byte_count": 512,
          "checksum": {"algorithm": "sha256", "digest": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
          "error_row_count": 0,
          "metadata": {}
        }
      ],
      "total_row_count": 2,
      "total_error_row_count": 0,
      "metadata": {}
    }
  ],
  "total_row_count": 2,
  "total_error_row_count": 0,
  "config_hash": "config-sha256-example",
  "generator_seed": 1729,
  "created_at": "2026-01-01T00:00:00Z",
  "git_commit": "0123456789abcdef",
  "python_version": "3.12.0",
  "platform": "example-platform",
  "package_versions": {"geml": "0.1.0"},
  "metadata": {"fixture": true}
}
```

## Scope and consumers

Issue 1-5 uses these records for deterministic generation, immutable sharding, checksums, and
checkpointed runs. Every later result-bearing Goals 1–5 study consumes the manifest and run
accounting fields. File formats, path resolution, file I/O, checksum calculation, generation,
deduplication, checkpointing, and final-count enforcement are out of scope for this contract.
