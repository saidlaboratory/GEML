# Goals 6--12 ML reproducibility and bounded-compute policy

This policy owns the common learning environment contract for Goals 6--12.
It applies to fixture, training, search, and analysis cells that consume an ML
checkpoint. Goals 1--5 remain immutable, and the core package must remain
usable without PyTorch.

## Environment

Use Python 3.12 and install the optional extra with:

```bash
python -m pip install -e '.[ml]'
```

The frozen optional package versions are `torch==2.5.1` and
`torch-geometric==2.6.1`. CPU-only tests must skip their ML-specific checks
when this extra is not installed; no test may download packages, use a GPU, or
read production outputs.

`configs/ml_env.yaml` is the machine-readable source for the package pins,
three production seeds, precision policy, batch policy, and hardware profile.
It is a policy artifact, not a hardware discovery result.

## RunEnvelopeV1

Every generated dataset, training cell, search shard, external-reference
attempt, and analysis run must persist a `geml-run-envelope-v1` object with at
least these fields:

| Field | Requirement |
| --- | --- |
| `schema_version` | Exactly `geml-run-envelope-v1`. |
| `configuration` / `configuration_hash` | Canonical configuration content and its SHA-256 digest. |
| `git_commit` | Full commit SHA that produced the output. |
| `package_versions` | Python and relevant package versions. |
| `seeds` | Every seed used, including data-loader and search seeds where distinct. |
| `hardware` / `precision` | Accelerator/host description, requested and effective precision. |
| `determinism` | Requested deterministic settings plus every known nondeterministic operation. |
| `input_checksums` / `output_checksums` | Content checksums of declared inputs and produced files. |
| `started_at` / `ended_at` / `wall_seconds` | UTC timing information. |
| `resource_telemetry` | Parameters, FLOPs where defined, memory, and effective batch/node/edge budget. |
| `outcomes` | Attempted, successful, failed, unsupported, invalid, and timeout counts. |
| `resume_lineage` | Prior checkpoint/output identity and whether the cell resumed. |
| `reproduction_command` | Exact command needed to reproduce the cell. |

Run envelopes are append-only evidence. Atomic finalization may replace an
incomplete *temporary* file, but it must not overwrite a previous failed row
or conceal a failed attempt.

## Seeds and selection

The only production stochastic seeds are `20260726`, `20260727`, and
`20260728`. A technical rerun uses the same seed and records the old row, new
row, code/config hashes, and failure reason. Validation data selects one model
configuration before test evaluation; broad architecture or hyperparameter
sweeps are prohibited.

## Compute profile

The default production profile is two H100 80 GB GPUs, at least 64 effective
CPU threads, 256 GB RAM, and 1 TB fast local NVMe. Run independent experiment
or seed cells by default. Four H100 GPUs are allowed only after a recorded
30--60 minute throughput pilot shows a material wall-clock benefit without
starving shared CPU, RAM, or NVMe resources. Report total GPU-hours separately
from parallel wall time; never claim linear speedup.

Use BF16/mixed precision on H100 only when supported, record the effective
precision and nondeterministic operations, and use dynamic batches constrained
by node and edge budgets. CPU CI uses tiny float32 fixtures.

All ordinary validation commands are capped at approximately 30 minutes.
Longer production work must be independently sharded, checkpointed, resumable,
and audit-linked.
