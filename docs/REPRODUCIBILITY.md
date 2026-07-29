# GEML reproducibility guide

Status: Phase-A package for issues 12-1 and 12-2, prepared 2026-07-26.

This guide separates fast, offline contract smokes from production science. A smoke checks
wiring with tiny fixtures; it does not reproduce a paper result. Production work is
sharded, checkpointed, resumable, and expected to exceed 30 minutes. Goals 1-5 and their
250k-v1 corpus, EML trees, DAGs, motifs, and reports are immutable inputs and must not be
regenerated.

The machine-readable public artifact registry is
`scripts/repro/ARTIFACT_SOURCES.json`. Reproducibility commands use only the Python standard
library unless a command explicitly says otherwise. Tests never need a network connection,
GPU, API key, production `outputs/`, or the 35 GB artifact archive.

## Current blockers and non-claims

- Integration selected `requirements-lock.txt` as the single core/development lock. Its
  preproduction SHA-256 is
  `1ec81a07d64f969a58abb7ce205107e8e23b360258e330d134be1ccedae87c51`.
  Optional CUDA-dependent ML versions remain pinned in `configs/ml_env.yaml`.
- All six Phase-A workstreams are integrated. Their smokes establish wiring and contracts,
  not production results.
- Goal 10 and Goals 11-12 files in this workstream are Phase-A fixtures/scaffolds, not
  production results.
- Goals 6-12 production artifacts are currently `missing` or `deferred` in the artifact
  registry. Public locations and checksums must be added only after verified publication.
- Production runtime and price are not claimed before the required measured pilot and a
  contemporaneous host quote. The formulas and stop conditions below prevent an estimate
  from being mistaken for observed telemetry.

## Fresh-clone setup

Use Python 3.12 or newer. Replace `<release-commit>` with the frozen 40- or 64-hex commit
reported by the release. Do not reproduce from a moving branch name.

### Linux or macOS

```bash
set -euo pipefail
git clone https://github.com/saidlaboratory/GEML.git GEML-release
cd GEML-release
git fetch --tags --force
git checkout --detach '<release-commit>'
test "$(git rev-parse HEAD)" = '<release-commit>'

python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

### Windows PowerShell

```powershell
$ErrorActionPreference = "Stop"
git clone https://github.com/saidlaboratory/GEML.git GEML-release
Set-Location GEML-release
git fetch --tags --force
git checkout --detach "<release-commit>"
if ((git rev-parse HEAD) -ne "<release-commit>") {
    throw "Release commit mismatch"
}

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

The editable package install is dependency-free; all resolved core/development packages come
from the selected lock. A release is not environment-reproducible until a clean install from
that lock passes:

```bash
python -m pytest
python -m ruff check .
python -m ruff format . --check
```

Record the release commit, lock SHA-256, Python version, package inventory, operating system,
CPU, RAM, storage, GPU model/driver, CUDA version, deterministic settings, and precision in
the final run envelope.

## Authenticated Goals 1-5 artifact handoff

Public folder:
<https://drive.google.com/drive/folders/1zK5HhWeaFtVJwtby15dO6LDdybVckbfF>

The link-readable folder contains exactly five published files:

| File | Bytes | Drive ID | SHA-256 |
|---|---:|---|---|
| `GEML_artifacts_goals_1-5_2026-07-25.tar` | 35,835,445,760 | `1tCxLbx_hNn-JY373YoS4amvl0hVS2Gky` | `438b11726bd108b2fe971063d8dffbdd580c0f4ec7c42947047693f818290f3e` |
| `GEML_artifacts_goals_1-5_2026-07-25_SHA256.txt` | 106 | `1Ug7RcUEYGBfovjDef4SzOi6f-LvAsZJd` | `d2707643bdec0f59b6353a3249e221df27cd072b539854d258acd4de988384d3` |
| `GEML_artifacts_extract_and_verify_macos_windows.txt` | 3,587 | `1_iGZ20yhB202VU1sUw9cKenP-qg-8e4U` | `d4c37799c2544d3ad94f608661b96d14d658051f4beeca77bb33600c43d09c76` |
| `GEML_artifacts_extract_and_verify_windows.ps1` | 3,005 | `1IOV2S05lmP6TA6kRpa11kLNq7yBcr1ag` | `10829265099a7dbaa289d4354bbe276667676c57e31679e15f91f3c81d6ac217` |
| `ARTIFACT_INDEX.md` | 7,526 | `1KycCGZaJgj_oeVae-g7Yo19LR81bHwIM` | `b3a52cf022c96bf42bf5ab52c15b3d258945f94f5a76f688804d420bd94b91a6` |

Authenticate all five downloads before executing either downloaded script or trusting the
downloaded index/checksum files. The archive is an uncompressed TAR, not a ZIP. Provision at
least 80 GB free space and at least 1.3 million free inodes. Extract outside the Git checkout.
When workstreams share a host, download and extract once, make the tree read-only, and share
only its absolute path.

### Resumable download

`gdown` is used only for this explicit network step. Keep it out of the locked scientific
environment: create a throwaway downloader environment and remove that environment after all
five downloads authenticate.

```bash
python3.12 -m venv /absolute/path/to/geml-gdown-venv
/absolute/path/to/geml-gdown-venv/bin/python -m pip install gdown==6.0.0
GDOWN=/absolute/path/to/geml-gdown-venv/bin/gdown
mkdir -p GEML_artifacts_delivery
cd GEML_artifacts_delivery

"${GDOWN}" --continue \
  'https://drive.google.com/file/d/1tCxLbx_hNn-JY373YoS4amvl0hVS2Gky/view' \
  -O GEML_artifacts_goals_1-5_2026-07-25.tar
"${GDOWN}" 'https://drive.google.com/file/d/1Ug7RcUEYGBfovjDef4SzOi6f-LvAsZJd/view' \
  -O GEML_artifacts_goals_1-5_2026-07-25_SHA256.txt
"${GDOWN}" 'https://drive.google.com/file/d/1_iGZ20yhB202VU1sUw9cKenP-qg-8e4U/view' \
  -O GEML_artifacts_extract_and_verify_macos_windows.txt
"${GDOWN}" 'https://drive.google.com/file/d/1IOV2S05lmP6TA6kRpa11kLNq7yBcr1ag/view' \
  -O GEML_artifacts_extract_and_verify_windows.ps1
"${GDOWN}" 'https://drive.google.com/file/d/1KycCGZaJgj_oeVae-g7Yo19LR81bHwIM/view' \
  -O ARTIFACT_INDEX.md
```

On PowerShell, create the throwaway environment with `py -3.12 -m venv
C:\absolute\path\to\geml-gdown-venv`, install with that environment's `python.exe`, and invoke
its `gdown.exe`; the same five URLs/output names work with backticks for line continuation. If
Drive terminates the archive transfer, rerun its command unchanged with `--continue`. A byte
count or filename is not authentication.

### Verify every downloaded file

Run the verifier from the Git checkout and pass the absolute delivery path (the `scripts`
namespace is intentionally not installed as a library):

```bash
cd /absolute/path/to/GEML
python -m scripts.repro verify-delivery /absolute/path/to/GEML_artifacts_delivery
python -m scripts.repro preflight-archive \
  /absolute/path/to/GEML_artifacts_delivery/GEML_artifacts_goals_1-5_2026-07-25.tar
```

The command exits nonzero and retains `missing`, `size_mismatch`, or `checksum_mismatch` for
each bad entry. The second command then scans the authenticated TAR member table without
extracting it. It rejects absolute paths, `..`, backslashes, platform-colliding or reserved
names, duplicate members, links, devices, FIFOs, sparse members, unexpected roots, and
declared file/byte totals that differ from the frozen contract. Run both commands immediately
before extraction; a member preflight is not a substitute for the archive SHA-256. On
Linux/macOS an independent checksum check is:

```bash
printf '%s  %s\n' \
  '438b11726bd108b2fe971063d8dffbdd580c0f4ec7c42947047693f818290f3e' \
  'GEML_artifacts_goals_1-5_2026-07-25.tar' \
  'd2707643bdec0f59b6353a3249e221df27cd072b539854d258acd4de988384d3' \
  'GEML_artifacts_goals_1-5_2026-07-25_SHA256.txt' \
  'd4c37799c2544d3ad94f608661b96d14d658051f4beeca77bb33600c43d09c76' \
  'GEML_artifacts_extract_and_verify_macos_windows.txt' \
  '10829265099a7dbaa289d4354bbe276667676c57e31679e15f91f3c81d6ac217' \
  'GEML_artifacts_extract_and_verify_windows.ps1' \
  'b3a52cf022c96bf42bf5ab52c15b3d258945f94f5a76f688804d420bd94b91a6' \
  'ARTIFACT_INDEX.md' |
  sha256sum -c -
```

Use `shasum -a 256 -c -` instead of `sha256sum -c -` on macOS.

PowerShell independent check:

```powershell
$ExpectedHashes = [ordered]@{
    "GEML_artifacts_goals_1-5_2026-07-25.tar" =
        "438b11726bd108b2fe971063d8dffbdd580c0f4ec7c42947047693f818290f3e"
    "GEML_artifacts_goals_1-5_2026-07-25_SHA256.txt" =
        "d2707643bdec0f59b6353a3249e221df27cd072b539854d258acd4de988384d3"
    "GEML_artifacts_extract_and_verify_macos_windows.txt" =
        "d4c37799c2544d3ad94f608661b96d14d658051f4beeca77bb33600c43d09c76"
    "GEML_artifacts_extract_and_verify_windows.ps1" =
        "10829265099a7dbaa289d4354bbe276667676c57e31679e15f91f3c81d6ac217"
    "ARTIFACT_INDEX.md" =
        "b3a52cf022c96bf42bf5ab52c15b3d258945f94f5a76f688804d420bd94b91a6"
}
foreach ($FileName in $ExpectedHashes.Keys) {
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $FileName).Hash.ToLowerInvariant()
    if ($Actual -ne $ExpectedHashes[$FileName]) {
        throw "SHA-256 mismatch for ${FileName}"
    }
}
```

### Extract and authenticate every root file

Expected extracted root:

```text
GEML_artifacts/
files: 1,210,913
bytes: 34,810,631,623
```

Linux:

```bash
cd /absolute/path/to/GEML_artifacts_delivery
mkdir -p extracted
test ! -e extracted/GEML_artifacts || {
  echo 'Extraction target already exists; refusing overwrite' >&2
  exit 1
}
tar -xf GEML_artifacts_goals_1-5_2026-07-25.tar -C extracted
cd /absolute/path/to/GEML
python -m scripts.repro verify-tree \
  /absolute/path/to/GEML_artifacts_delivery/extracted/GEML_artifacts
export GEML_ARTIFACTS_ROOT="$(
  realpath /absolute/path/to/GEML_artifacts_delivery/extracted/GEML_artifacts
)"
```

macOS:

```bash
DESTINATION_PARENT="${HOME}/GEML_artifacts_delivery"
test ! -e "${DESTINATION_PARENT}/GEML_artifacts" || {
  echo 'Extraction target already exists; refusing overwrite' >&2
  exit 1
}
mkdir -p "${DESTINATION_PARENT}"
tar -xf GEML_artifacts_goals_1-5_2026-07-25.tar -C "${DESTINATION_PARENT}"
cd /absolute/path/to/GEML
python -m scripts.repro verify-tree "${DESTINATION_PARENT}/GEML_artifacts"
export GEML_ARTIFACTS_ROOT="${DESTINATION_PARENT}/GEML_artifacts"
```

Windows, only after authenticating the downloaded script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
    -File .\GEML_artifacts_extract_and_verify_windows.ps1
$env:GEML_ARTIFACTS_ROOT =
    (Resolve-Path ".\extracted\GEML_artifacts").Path
Push-Location C:\absolute\path\to\GEML
python -m scripts.repro verify-tree $env:GEML_ARTIFACTS_ROOT
Pop-Location
```

If any hash/count differs, stop. Do not repair a partial archive. Never use the known stale
directory `GEML_artifacts - Copy`, which omits Goal 5-9.

Important authenticated paths include:

```text
$GEML_ARTIFACTS_ROOT/1-8_source_expression_corpus_250k/manifests/corpus.manifest.json
$GEML_ARTIFACTS_ROOT/2-7_2-8_official_pure_eml_corpus/manifest.json
$GEML_ARTIFACTS_ROOT/3-1_to_3-8_graph_corpus_and_costs/manifest.json
$GEML_ARTIFACTS_ROOT/4-1_to_4-9_goal4_egraph_study/outputs/final/goal4/final/final.rows.jsonl
$GEML_ARTIFACTS_ROOT/5-8_goal6_ready_graph_export/run.complete.json
$GEML_ARTIFACTS_ROOT/5-8_goal6_ready_graph_export/batches/
$GEML_ARTIFACTS_ROOT/5-8_goal6_ready_graph_export/blobs/sha256/
$GEML_ARTIFACTS_ROOT/5-9_goals1_to_5_final_report/integration.evidence.json
$GEML_ARTIFACTS_ROOT/5-9_goals1_to_5_final_report/run.complete.json
```

The ten canonical directory tree hashes and their exact schemes are preserved in
`scripts/repro/ARTIFACT_SOURCES.json`. `verify-tree` recomputes each published digest, checks
the root `ARTIFACT_INDEX.md` body, rejects links and unexpected root entries, and then checks
the aggregate file/byte totals. The eight directories through issue 5-7 use SHA-256 over the
LF-joined, ordinal sequence `relative/path<NUL>lowercase_file_sha256` with no trailing LF.
Issues 5-8 and 5-9 use `geml-artifact-tree-v1`: SHA-256 over the tag followed by LF and a
per-directory ordinal traversal of
`relative/path<NUL>byte-size<NUL>raw-file-sha256<LF>`. This full content verification opens
all 1,210,913 files and can be I/O intensive; it is production artifact authentication, not a
smoke test.

## Bounded smoke entry points

List the exact state and commands:

```bash
python -m scripts.repro smoke
```

Execute one goal:

```bash
python -m scripts.repro smoke --goal 1 --execute
```

Every entry point removes credential-like environment variables, hides CUDA GPUs, enables
mocked/offline mode, sets a fixed Python hash seed, and enforces a 1,740-second hard cap. The
network rule is a cooperative test contract, not an operating-system network sandbox; the
fresh-clone audit must still verify that no smoke opens a network connection. Each uses only
temporary/tiny fixtures. Goal 10 includes its compiler/e-graph structure tests as well as its
three experiment-level conformance tests. The same form applies on Windows with
`.\.venv\Scripts\python.exe`.

| Goal | Entry point | Phase-A state |
|---:|---|---|
| 1 | `python -m scripts.repro smoke --goal 1 --execute` | ready |
| 2 | `python -m scripts.repro smoke --goal 2 --execute` | ready |
| 3 | `python -m scripts.repro smoke --goal 3 --execute` | ready |
| 4 | `python -m scripts.repro smoke --goal 4 --execute` | ready |
| 5 | `python -m scripts.repro smoke --goal 5 --execute` | ready |
| 6 | `python -m scripts.repro smoke --goal 6 --execute` | ready |
| 7 | `python -m scripts.repro smoke --goal 7 --execute` | ready |
| 8 | `python -m scripts.repro smoke --goal 8 --execute` | ready |
| 9 | `python -m scripts.repro smoke --goal 9 --execute` | ready |
| 10 | `python -m scripts.repro smoke --goal 10 --execute` | ready |
| 11 | `python -m scripts.repro smoke --goal 11 --execute` | ready; provider calls mocked |
| 12 | `python -m scripts.repro smoke --goal 12 --execute` | ready |

A missing target returns exit code 3 and its exact path. After merge, run the listing again;
every row must say `ready`, then execute all 12 commands separately and record command, start
and end time, return code, elapsed time, test counts, skips, and failures. Never combine the
12 commands under one 30-minute claim.

## Production compute profiles

Production work uses a separate environment with #54's frozen ML dependencies.

### CPU profile

- Generation, parsing, graph materialization, verification, e-graph replay, PySR/GP, proof
  search, simplification, manifest/report assembly, and Goal 10 conformance.
- At least 64 effective CPU threads, preferably 96-128.
- At least 256 GB RAM.
- At least 1 TB fast local NVMe and 80 GB free before artifact extraction.
- Process-level shards with an explicit RAM ceiling; do not starve GPU data loaders.

### Default 2xH100 profile

- Two H100 80 GB GPUs.
- Prefer independent ready experiment cells per GPU over DDP for compact models.
- BF16 only where supported and recorded.
- Dynamic node/edge-budget batches.
- Three frozen seeds; approximately 30-epoch cap; early stopping.
- Atomic `latest` and `best` checkpoints and exact resume lineage.
- No broad hyperparameter sweep.

### Conditional 4xH100 profile

Use four H100s only after a 30-60 minute measured pilot shows that four independent cells
reduce wall time without CPU, RAM, loader, or NVMe starvation. Do not assume linear scaling.
Record both total GPU-hours and elapsed wall time. A 4-GPU host that is not materially faster
than two GPUs fails this condition.

### Runtime and cost budgeting

Before launch, measure a representative pilot and freeze:

```text
expected_cell_hours =
    ceil(remaining_work_units / measured_work_units_per_hour * 1.20)
expected_instance_hours =
    sequential_setup_and_preprocessing_hours
    + max(parallel_GPU_critical_path_hours, parallel_CPU_analysis_hours)
    + sequential_collection_and_validation_hours
expected_compute_cost_usd =
    current_instance_quote_usd_per_hour * expected_instance_hours
expected_total_cost_usd =
    expected_compute_cost_usd
    + storage_cost_usd + egress_cost_usd + API_cost_usd
```

The 20% factor is a declared scheduling reserve, not evidence. The dependency-aware wall-time
formula adds sequential phases and takes a maximum only across work actually run in parallel;
do not use `max(CPU, GPU)` when preprocessing must finish before training. Store the measured
throughput, quote provider/instance ID/access time, storage/egress/API charges, formula inputs,
expected total cost, and user-approved total ceiling. Include the throughput pilot itself.
Abort before the total of compute, storage, egress, and API spend can exceed the ceiling.
Replace estimates with observed wall time, GPU-hours, and cost in final artifacts. The 72-hour
project window is a planning constraint, not a performance result.

## Production run plan

These are production plans, not smoke commands. Before launch, copy exact CLI flags from
each integrated pipeline's `--help` and frozen config; do not invent an adapter flag. Bind
every cell to its config content/hash, release commit,
package lock/hash, seeds, inputs/checksums, profile, shard ID/count, checkpoint directory,
output directory, and exact command in `RunEnvelopeV1`.

| Goal | Immutable inputs / production cell | Sharding and resume | Profile and current state |
|---:|---|---|---|
| 1-5 | No production command: use authenticated immutable artifacts | Never regenerate or overwrite; checksum input tree | CPU authentication only; complete public handoff |
| 6 | Pair generation, four aligned channels, six arms x three seeds, analysis/Gate G6 | Pair/channel shards by frozen source group; training cell = `(arm, seed)`; resume same config digest from atomic latest checkpoint | CPU plus 2xH100; integrated, production pending |
| 7 | Step extraction, policy/transformer cells, fixed grid, Gate G7 | Step shards by trace group; training cells by model/seed; checkpoint model/optimizer/scheduler/RNG and replay ledger | CPU plus 2xH100; integrated, production pending |
| 8 | Value training, 256-proof ATP comparison, 1,000-expression simplification | Value cells by seed; ATP/simplification by frozen problem-ID ranges; completed rows content-addressed and never overwritten | CPU-heavy plus 2xH100; integrated, production pending |
| 9 | Frozen SR benchmark, EML/AST guided search, PySR or labeled GP fallback, transformer-SR, Gate G9 | Shard by frozen task-ID ranges and seed; checkpoint search frontier/model state; preserve timeouts/invalid outputs | CPU-heavy plus 2xH100; verifier scope frozen, production pending |
| 10 | v2 conformance generation, repeat hash, structure/domain/numeric audits, compatibility checks | CPU shards by operator/domain/case; atomic rows and audit checkpoints; no training and no Goals 1-5 regeneration | CPU; conformance scope resolved, production pending |
| 11 | Workshop manifest, fixed-scale efficiency, no-retraining synthesis, external LLM panel | Analysis consumes immutable manifests; LLM has exactly 200 frozen attempts/model with bounded concurrency/retries and one retained row/task | CPU/API; integrated, paid calls still separately gated |
| 12 | Final report, checksum index, reproducibility verification, paper/release | Rebuild deterministically from frozen manifests; publication uploads resumable; never mutate evidence | CPU; integrated scaffold, authenticated results pending |

For every executable pipeline, production launch must follow this shape after the real merged
CLI is verified:

```bash
python -m '<merged.pipeline.module>' --help
# Freeze the exact command emitted/documented by that module, including:
# --config <frozen-config> --shard-index <i> --shard-count <n>
# --checkpoint-dir <immutable-lineage> --output-dir <unique-cell> --resume
```

Do not run the placeholder above. The coordinator must replace each placeholder with the
pipeline's tested CLI in this document after merge. A resumable command must:

1. refuse a checkpoint whose config, commit, seed, input, or schema digest differs;
2. write checkpoints and result files atomically;
3. retain earlier failed/timeout/unsupported evidence;
4. skip only checksum-verified completed shards;
5. record parent checkpoint and exact resume command;
6. never overwrite a successful cell with a rerun.

## Checkpoint resume protocol

1. Stop the producer cleanly when possible; never edit a checkpoint.
2. Copy the failed cell's run envelope, stderr/stdout, latest atomic checkpoint, partial-row
   ledger, and checksums to durable storage.
3. Recreate the same release commit and locked environment.
4. Reauthenticate all input artifacts and compare config/commit/seed/schema hashes.
5. Invoke the producer's explicit `--resume` path with the same cell/shard identity.
6. Require the producer to reject mismatched lineage.
7. Append a resume event with old/new host metadata and wall-time segments.
8. After completion, verify output checksums and preserve the interrupted evidence.

Hardware-independent bitwise GPU identity is not promised. Determinism settings,
nondeterministic kernels, precision, and any numeric tolerance must be reported.

## Output collection

Create an index outside the directory being indexed:

```bash
python -m scripts.repro collect outputs/final/goal6 \
  --output release_manifests/goal6.artifacts.json \
  --artifact-id goal6-final \
  --config configs/goal6_grid.yaml \
  --seed 20260726 --seed 20260727 --seed 20260728 \
  --profile 2xh100
```

The collector:

- rejects an output manifest inside its source tree;
- rejects linked/junction roots, linked entries/configs, and empty output roots;
- hashes regular files in canonical relative-path order;
- records byte totals and a digest over the file ledger;
- records the release commit, Python/platform/package inventory, profile, seeds, and config
  checksums;
- preserves missing configs and read failures as explicit failure states;
- writes atomically.

Run it separately for each goal/cell collection. Then assemble issue 11-0's final workshop
manifest and verify every referenced checksum. Do not publish a final report when any required
input is `missing`, `failed`, `unsupported`, `checksum_mismatch`, or `schema_mismatch`.

## Provider preflight, mocks, and cost guards

Tests and default smokes use mock mode and remove all provider keys:

```bash
python -m scripts.repro provider-preflight --provider openai --mode mock
python -m scripts.repro provider-preflight --provider anthropic --mode mock
python -m scripts.repro provider-preflight --provider google --mode mock
python -m scripts.repro provider-preflight --provider moonshot --mode mock
```

Copy `scripts/repro/.env.providers.example` to an untracked, access-restricted location only
for production. Never print, log, archive, commit, or place keys in shell history. Provider
SDK/dependency policy remains a #82 integration decision; do not install an undeclared SDK.

Local live-mode validation makes no network call and is intentionally insufficient by itself:

```bash
python -m scripts.repro provider-preflight \
  --provider '<provider>' \
  --mode live \
  --model-id '<exact-operator-selected-api-model-id>' \
  --attempts 200 \
  --estimated-cost-usd '<estimate-from-current-rates>' \
  --max-cost-usd '<user-approved-ceiling>' \
  --spend-approval I-CONFIRM-PAID-API-CALLS
```

It rejects missing keys, mutable aliases such as `latest`, any attempt count other than 200,
an estimate above the cap, and absent explicit approval. It never returns the secret.

Immediately before paid execution, the merged #82 adapter must additionally:

- call or inspect the official provider model-list/API documentation and pin the exact
  available model ID; never silently substitute a provider/model;
- verify exactly 100 frozen proof and 100 frozen SR IDs and their list hashes;
- show estimated tokens/cost and obtain fresh user approval;
- enforce concurrency/retry/token/time/cost ceilings;
- store all 200 success/failure rows per model, raw responses, parse/refusal/timeout/API
  failures, usage, latency, prompts/hashes, exact model/date, and verifier results;
- stop before the approved cap and never count retries as new benchmark tasks.

Frontier-LLM results are external, verifier-normalized context, not controlled Gate G6-G11
evidence.

## Fresh-clone verification

After all six workstreams merge and the lock is frozen:

1. Create a new temporary clone, detached at the release commit.
2. Install only from the approved lock.
3. Leave `GEML_ARTIFACTS_ROOT`, provider keys, GPU visibility, and `outputs/` absent.
4. Run all 12 smoke commands separately and record elapsed times.
5. Run the full standard validation.
6. Authenticate all five delivery files, preflight the TAR member table, then authenticate
   every extracted canonical-directory digest on the production host.
7. Check every production command's `--help`, config, shard, resume, and cost guard without
   starting a paid/full run.
8. Store the verification log and its SHA-256 in the release manifest.

No smoke is accepted if it exceeds approximately 30 minutes or touches network, GPU, API
credentials, or production artifacts.

## Instance teardown checklist

Before destroying a production host:

- Stop launchers and flush atomic writers; record all active/incomplete cell states.
- Collect stdout/stderr, run envelopes, configs, config hashes, commit/lock hashes, seeds,
  package and hardware metadata, checkpoints, failure ledgers, raw provider responses, usage,
  and cost records.
- Run checksum collection and copy verified manifests plus artifacts to durable storage.
- Test-restore at least one checkpoint and open representative output shards.
- Confirm every expected shard is complete or explicitly failed/timeout/unsupported.
- Revoke or rotate API credentials. On the dedicated ephemeral host, remove only
  secret-bearing history entries and credential files; do not erase unrelated operator
  history or audit logs.
- Delete task-scoped local secret files and provider credential caches after the evidence
  bundle is safely collected.
- Unmount/delete the extracted 35 GB tree only after its public source and local verification
  are recorded; never delete the public handoff.
- Release cloud GPUs/storage/IPs and verify billing stopped.
- Record final observed instance-hours, GPU-hours, storage/egress/API cost, teardown time, and
  operator.

## Release-time standard validation

```bash
python -m pytest tests/test_repro_scripts.py
python -m pytest
python -m ruff check .
python -m ruff format . --check
git diff --check
git status --short
```

Record exact pass/fail/skip counts and runtimes. A green targeted test does not prove the full
release, and a missing production result must never be presented as zero or success.
