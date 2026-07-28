# Reproducibility package

## Fresh clone and bounded local validation

Use Python 3.12, create a virtual environment, install the exact requirements, then install GEML:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e '.[dev]'
python scripts/repro/run_smoke.py --goal all
```

The smoke entry points are fixture-only, require neither `outputs/`, GPUs, nor network access, and
are designed to finish within 30 minutes. Goal 10 is deliberately excluded: its grammar-v2 compiler
dependency is blocked and must not be substituted with an invented formula or mode.

## Artifact staging and production work

The public archive is available from the [GEML Drive folder](https://drive.google.com/drive/folders/1zK5HhWeaFtVJwtby15dO6LDdybVckbfF).
It is 33.37 GB and needs at least 40 GB of writable staging capacity. Download and authenticate the
published archive/index before extraction; do not regenerate Goals 1-5. The current workspace has
only about 21 GiB free, so it cannot safely stage the archive.

Production cells are not runnable claims: they require staged checksum-verified artifacts, frozen
configs, and the existing checkpoint/resume interfaces. Use the declared 2x H100 (80 GB) default and
conditional 4x H100 configuration only after recording actual hardware, wall time, total GPU-hours,
and interruption/resume metadata. No 10-100x scaling run is planned or implied.

## External LLM reference preflight

Set a provider key only in the environment, never in a file or commit. For example:

```bash
export OPENAI_API_KEY='...'
python scripts/repro/preflight_llm.py --provider openai --require-key
```

The preflight makes zero network and paid calls. A 200-attempt panel is allowed only after user
credentials and an explicit spend approval; raw responses, exact model IDs, prompts, usage, latency,
cost, and verifier outcomes must be retained. LLM rows remain external references, never controlled
gate inputs.

## Artifact availability

The Goal 11 manifest validator retains missing, checksum-mismatched, and explicitly deferred rows.
At this revision, production result tables/checkpoints and the archive have not been staged locally;
they are unavailable rather than silently omitted. See `configs/goal11_corpus_v3.yaml` and
`docs/goals/FINAL_REPORT.md` for the current evidence boundary.
