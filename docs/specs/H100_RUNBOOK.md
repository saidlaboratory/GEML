# 4xH100 Phase-B execution runbook

Status: prepared 2026-07-30 against the local `release/mathnlp-2026` branch
(`5d6abd5`). Companion tooling: `scripts/h100/bootstrap.sh`,
`scripts/h100/run_pipeline.sh`, `scripts/h100/schedule_cells.py`.

Honesty rule for every number in this document: a value is either **measured**
(with the machine and command that produced it), an **estimate** (derived from
a named measurement, never presented as measured), or **pending** (no number
exists). Nothing here was measured on an H100; this repo has never run on one.

## PREREQUISITES — what blocks a clean run today

1. **Branches must be pushed/merged.** `release/mathnlp-2026` exists only on
   the development laptop. It merges three reviewed local branches on top of
   `main` (`503f435`):
   - `submission/pairs-build-loop` (`975bfb8`) — the real production
     pair-build loop behind `python -m geml.data.pairs`;
   - `submission/final-report-binding` (`866949a`);
   - `submission/sr-benchmark-freeze` (`49a9913`).
   Push the release branch (or land it on `origin/main`) so the node can clone
   at that commit, and merge this kit's branch (`h100/runbook-and-scheduler`)
   into it. `bootstrap.sh` refuses to guess: it takes `GEML_COMMIT` explicitly.

2. **Artifact delivery vs regeneration — decide and record.**
   - *Option A (recommended, policy-compliant):* download the authenticated
     Goals 1–5 delivery (35.8 GB tar, checksums in
     `scripts/repro/ARTIFACT_SOURCES.json`), verify with
     `python -m scripts.repro verify-delivery / preflight-archive /
     verify-tree`, extract (needs ≥ 80 GB free and ≥ 1.3 M free inodes), and
     set `GEML_ARTIFACTS_ROOT`. `docs/REPRODUCIBILITY.md` states Goals 1–5
     production artifacts are never regenerated.
   - *Option B (fallback only):* regenerate the corpus with the Goal 1
     dev→pilot→final stages below. A regenerated corpus is a **new corpus
     identity** unless byte-identical to the delivered one: every downstream
     frozen digest (corpus manifest hash, pair-build input digest, goal7 step
     identities) would have to be re-derived from it, and the deviation from
     the immutability policy must be recorded as a lead decision. Do not mix
     Option A and Option B artifacts in one run.

3. **Goal 7 null identities must be filled from real artifacts.** Verified
   today via `scripts/h100/run_pipeline.sh goal7-validate` (runs on the
   laptop, CPU): `configs/goal7_grid.yaml` reports 15 production blockers —
   `expected_step_count`, `step_manifest`, `step_manifest_sha256`,
   `rule_registry_sha256`, `verifier_sha256`, `shared_harness_sha256`,
   `shared_gnn_architecture_sha256`, `transformer_architecture_sha256`,
   `compute_reference_sha256`, `implementation_sha256`,
   `training_config_sha256`, `training_family_inventory_sha256`,
   `step_population_sha256`, `reproduction_command`, and
   `analysis_reproduction_command` are all null. They may only be filled from
   the authenticated 7-0 step manifest and the integrated Workstream-1/2
   providers — never typed in by hand. `configs/goal7_steps.yaml` likewise has
   null `production_providers` and null expected digests, and the goal7
   `run_grid` CLI currently implements `--validate-only` plus injected
   executors only.

4. **Missing production entry points (implementation gaps, not config gaps):**
   - a channel-materialization driver: pair shards → four
     `GraphTensorV1` channel shard sets.
     `geml.learning.datasets.materialize` has the full API
     (`materialize_graph`, `mine_motif_ast_control_vocabulary`,
     `compress_motif_ast_control`, `write_channel_shard`) but no CLI;
   - a production `ChannelRegistry` implementation for
     `geml.experiments.goal6.run_grid.build_manifest` (today only the Protocol
     and test fixtures exist), which also writes `grid.manifest.json`;
   - a per-cell goal6 production entry point binding `ShardChannelProvider` +
     `ProductionCellRunner` + `GridRunner` for one `(arm, seed)` — this is the
     command `schedule_cells.py` needs in `--cmd`.

5. **Goal 6 width freeze.** `configs/goal6_grid.yaml` has
   `hidden_width: pending_preflight_freeze` and
   `use_virtual_node: pending_preflight_freeze` (permitted widths 64/96). The
   preflight tool is being built on the sibling branch `h100/width-preflight`;
   run it after channel shards exist, freeze the two values in the config,
   then start the grid. `ProductionCellRunner` deliberately has no width
   default.

6. **4-GPU use is conditional.** `docs/specs/ML_REPRODUCIBILITY.md` and
   `configs/ml_env.yaml` authorize 4 GPUs only after a recorded 30–60 minute
   throughput pilot shows material wall-clock benefit over 2 GPUs without
   CPU/RAM/NVMe starvation. Step (e) below schedules that pilot explicitly.
   Report total GPU-hours separately from wall time; never claim linear
   speedup.

7. **CUDA pins are H100-compatible.** `configs/ml_env.yaml` pins
   `torch==2.5.1` + `torch-geometric==2.6.1`. The default-index (PyPI) Linux
   x86_64 wheel of torch 2.5.1 bundles the CUDA 12.4 runtime and ships sm_90
   kernels, so H100 (compute capability 9.0) is supported, and it reports
   version exactly `2.5.1`, which the pin test requires. Two cautions:
   - do **not** install from `download.pytorch.org/whl/cu124` — those wheels
     report `2.5.1+cu124` and fail
     `tests/learning/test_ml_env.py::test_ml_environment_versions_when_optional_extra_is_installed`;
   - the node needs NVIDIA driver ≥ 550.54 (CUDA 12.4). `bootstrap.sh`
     enforces both with a runtime `sm_90`/bf16 self-check;
   - **sympy conflict, handled by bootstrap:** torch 2.5.1's metadata pins
     `sympy==1.13.1`, while the frozen core pin (`pyproject.toml`,
     `requirements-lock.txt`) is `sympy==1.14.0` — and core code that runs on
     the node (`geml.parsing`, `geml.verification`, the pair build's
     `sympy_srepr` handling) depends on it. Measured on the dev laptop
     (pip 24.2, clean 3.12 venv): installing torch after the lock silently
     downgrades sympy to 1.13.1. `bootstrap.sh` therefore reinstalls
     `sympy==1.14.0` after the torch wheels and hard-asserts it; torch 2.5.1
     imports and trains under sympy 1.14.0 (verified, CPU), and the harness
     never calls `torch.compile`, so torch's own sympy use stays inert.
     Expect `pip check` to report exactly this one torch-metadata conflict —
     that is the accepted state, anything else is a real problem. torch's
     remaining transitive deps (`filelock`, `fsspec`, `jinja2`, `networkx`,
     `MarkupSafe`, plus `triton==3.1.0` and the `nvidia-*` cu124 runtime
     wheels on Linux) are not named by the 32-line core lock and float to
     install-day versions; only sympy overlaps the lock.
   The pin test must PASS on the node (it fails/skips on the dev laptop; that
   is a known local-env condition, not a waiver for the node).

8. **Spend ceiling.** Before renting hours, fill the cost formula in
   `docs/REPRODUCIBILITY.md` ("Runtime and cost budgeting") with the measured
   pilot throughput and the actual instance quote, and record the approved
   ceiling. No number is prefilled here because none has been measured.

## Node bring-up

```bash
GEML_REPO_URL=git@github.com:saidlaboratory/GEML.git \
GEML_COMMIT=<pushed release/mathnlp-2026 tip> \
bash bootstrap.sh          # or scripts/h100/bootstrap.sh from a scratch clone
```

`bootstrap.sh` is idempotent: NVIDIA driver/GPU sanity via `nvidia-smi`,
apt git/python3.12, clone at the exact commit (detached), `.venv` with
`requirements-lock.txt`, the pinned `[ml]` CUDA wheels, `pip install -e .
--no-deps`, a torch sm_90/bf16 device check, and
`tests/learning/test_ml_env.py` (all four tests must pass, no skips).
Optionally `GEML_BOOTSTRAP_FULL_TESTS=1` runs the full CPU suite (measured on
the dev M1 laptop at this kit's commit: 2757 passed, 2 failed — the 2
failures are the known local-env failures that must NOT reproduce on the
node: the goal5 export digest and the ml_env pin).

## Ordered execution plan

Run each step with `scripts/h100/run_pipeline.sh <step>` from the repo root
with the venv active. Steps refuse (exit 3) while their prerequisite is
unmet.

| # | Step | Command (via run_pipeline.sh) | Wall | RAM | Label |
|---|------|-------------------------------|------|-----|-------|
| a1 | `verify-artifacts` | `scripts.repro verify-delivery/preflight-archive/verify-tree` | network-bound + tar extract | small | pending (never timed) |
| a2 | `corpus-dev` (fallback) | `geml.experiments.goal1.run --stage development` | ? | ? | pending |
| a3 | `corpus-pilot` (fallback) | `--stage pilot` (two deterministic runs) | ? | ? | pending |
| a4 | `corpus-final` (fallback) | `--stage final` (250k rows) | ? | **~28 GB** | RAM measured (memory-gated on the 16 GB M1 laptop; fits the 256 GB+ node); wall pending |
| b | `pairs` | `python -m geml.data.pairs --config configs/goal6_pairs.yaml` | ~3.7 h at the measured M1 rate; **2–4 h estimate** on server CPU | < 8 GB estimate | measured pilot: **34 min / 10k rows, M1 CPU, single process**; 65k-row scale-up is an estimate |
| c | `channels` | BLOCKED (prereq 4) — four channel shard sets from pair shards | ? | ? | pending |
| d | `width-preflight` | BLOCKED (prereq 5) — sibling branch `h100/width-preflight` | ? | ? | pending |
| e1 | `goal6-pilot` | `schedule_cells.py --gpus 0,1` on 2 cells | 30–60 min | ? | wall fixed by policy; throughput is the output |
| e2 | `goal6-grid` | `schedule_cells.py --gpus 0,1,2,3`, 18 cells | pending pilot; caps: 30 epochs / 20k steps / patience 5 per cell | ? | pending |
| e3 | `goal7-validate` | `goal7.run_grid --validate-only` | seconds | small | measured today on M1 (prints 15 blockers) |
| e4 | `goal7-grid` | BLOCKED (prereq 3); 18 cells | ≤ 2 h/cell → **≤ 36 GPU-h hard cap** (`wall_time_seconds: 7200`) | ? | cap by construction, not an estimate |
| f | `analysis` | goal6/goal7 summaries + gates + final report | minutes | small | estimate |

Step details:

**(a) Corpus: deliver (A) or regenerate dev→pilot→final (B).** Option B runs
`python -m geml.experiments.goal1.run --config configs/goal1_final.yaml
--stage {development,pilot,final}` — 1k, 10k×2 (with a determinism
comparison), then 250k rows. The final stage is memory-gated on the dev
laptop (verified: it refuses/fails on 16 GB) and needs roughly 28 GB RSS;
that fits the H100 node's RAM trivially. Wall time has never been measured
for any stage. Remember prerequisite 2: Option B output is a new corpus
identity.

**(b) Production pair build.** Real, landed on the release branch:

```bash
GEML_ARTIFACTS_ROOT=/abs/path/GEML_artifacts \
PYTHONPATH=src python3 -m geml.data.pairs --config configs/goal6_pairs.yaml
```

Builds 50k/5k/5k/5k pairs into `outputs/final/goal6/pairs/` as resumable,
atomically finalized shards; a corpus that cannot reach the targets ends
`complete_short` with per-split shortfalls rather than inventing records.
Measured: 34 min / 10k rows on the M1 CPU at pilot scale (single process).
Linear scaling to 65k rows gives ~3.7 h; treat 2–4 h on the node's server CPU
as an **estimate** — the build is sequential, so more cores do not help
without further work.

**(c) Channel materialization.** Blocked on the driver (prerequisite 4). Once
it exists: materialize all four channels (`ast_dag`, `pure_eml_dag`,
`frequent_macro_motif_dag`, `motif_ast_fair_control` — the last one mines its
train-only vocabulary under the frozen macro dictionary/MDL budget), write
checksum-manifested shards under `outputs/final/goal6/channels/`, and keep
the failure ledger: `ShardChannelProvider` fails an arm closed if any failure
rows remain.

**(d) Width preflight.** Reference only — built on `h100/width-preflight`,
not here. It must freeze `hidden_width` ∈ {64, 96} and `use_virtual_node` in
`configs/goal6_grid.yaml` from a measured parameter/FLOP comparison over the
real channel vocabularies, before any grid cell runs.

**(e) Grids on 4 GPUs.** First the mandatory 2-GPU pilot (30–60 min, record
throughput, then decide 4-GPU use — prerequisite 6). Then:

```bash
# goal6: 6 arms x 3 seeds = 18 cells
python3 scripts/h100/schedule_cells.py \
  --manifest outputs/final/goal6/grid/grid.manifest.json \
  --gpus 0,1,2,3 \
  --cmd '<per-cell entry point> --arm {arm_id} --seed {seed}'

# goal7: after prerequisite 3 resolves; the plan's per-cell
# reproduction_command is the source of truth for --cmd
python3 scripts/h100/schedule_cells.py \
  --manifest outputs/final/goal7/grid/<run_id>/run.plan.json \
  --gpus 0,1,2,3 --cmd '<reproduction_command with {cell_id}>'
```

The scheduler runs one cell per GPU via `CUDA_VISIBLE_DEVICES`, queues the
rest, and is resume-safe by reading the same content-addressed row files the
runners write: goal6 rows at `cells/<cell_id>.json` (the
`GridRunner.existing_row` identity — only status `complete` is skipped, so a
re-run never loses a retained failure), goal7 rows at
`cells/<id[:2]>/<id>.json` (immutable; a retained failure there needs a new
run identity). Failures are retained, never deleted; `--retry-failed`
explicitly requeues goal6 non-complete rows. Events go to
`<root>/scheduler.log.jsonl`, per-cell worker output to
`<root>/scheduler/<cell_id>.log`. CPU-tested here:
`tests/test_schedule_cells.py` schedules fixture cells across two fake
devices.

**(f) Analysis and gates.** After a grid is complete: goal6 aggregation and
Gate G6 via `geml.analysis.goal6.summary` (`aggregate`, `evaluate_gate_g6`);
goal7 via `geml.analysis.goal7.summary` (`build_goal7_summary`,
`write_goal7_summary`), invoked through the `analysis_reproduction_command`
frozen into the goal7 config; the final report CLI exists today:

```bash
PYTHONPATH=src python3 -m geml.analysis.final.report \
  --manifest <manifest> --sections <sections> \
  --artifact-root "${GEML_ARTIFACTS_ROOT}" --markdown-out <out.md>
```

Collect every produced output directory with
`python -m scripts.repro collect <root> --output <manifest> --artifact-id
<id> --profile 4xh100-conditional` so checksums and profile are bound to the
evidence.

## Scheduler reference

```
schedule_cells.py --manifest M [--root R] [--gpus 0,1,2,3] --cmd TEMPLATE
                  [--retry-failed] [--cell-timeout S] [--log PATH] [--dry-run]
```

`--cmd` placeholders: `{cell_id}`, `{arm_id}`, `{seed}`, `{gpu}`. Outcomes per
cell are decided from the persisted row file, not the exit code: `complete`,
`retained_failure` (row exists, non-complete), `no_row` (worker died before
writing evidence), `timeout` (killed by `--cell-timeout`; the harness's own
wall budget remains the scientific limit). Exit code 0 only when every
enumerated cell has a complete row.
