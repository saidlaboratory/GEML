# Contributing to GEML

Thanks for helping with GEML. This is a preregistered representation study, so the
contribution rules are a little stricter than a typical library: they exist to keep the
science honest and the results reproducible. Please read this once before your first
change — the clean-room and claim-discipline sections in particular are non-negotiable.

If anything here is ambiguous, stop and ask on the relevant issue rather than guessing.
See also [`AGENTS.md`](AGENTS.md) and [`docs/CLEANROOM_RULES.md`](docs/CLEANROOM_RULES.md),
which this guide summarizes and points back to.

## The three rules that matter most

1. **Clean-room.** The v0 prototype ([geml_experiments](https://github.com/sahilsinghthefirst/geml_experiments))
   is historical motivation only. Do not read, copy, port, or use its code, tests, schemas,
   helpers, architecture, or commit history as a template. Implement only from the current
   repository specifications, your assigned issue, the authoritative public sources named in
   [`docs/specs/EML_SOURCE_LEDGER.md`](docs/specs/EML_SOURCE_LEDGER.md), and official
   dependency documentation. The full rule set is in
   [`docs/CLEANROOM_RULES.md`](docs/CLEANROOM_RULES.md).

2. **Claim discipline.** Every number in code, docs, comments, or a PR must trace to a goal
   summary under `docs/goals/goalN/` and its checksummed manifest — never to memory or a
   guess. Nulls and gate failures are first-class results here: report them with the same
   prominence as wins and never soften the wording. GEML has published null results (learned
   motif selection does not beat frequency ranking) and a published gate **fail** (Gate G10,
   the eight preregistered `asin`/`acos` endpoint cells). Do not round those away or reframe
   them.

3. **Exclusive write scope.** Each issue owns an explicit set of write paths. Edit only the
   paths your issue assigns; never change another issue's contracts or frozen interfaces. If
   a shared interface looks wrong, raise it on the issue — don't fork a competing version.

## Development setup

GEML is a Python **3.12** package with a `src/` layout. Runtime code lives under
`src/geml/`; tests use only small hand-written or temporary fixtures.

```bash
git clone https://github.com/saidlaboratory/GEML.git
cd GEML

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Install the package plus the tooling you need:

```bash
# core runtime + tests + linter (what most contributors need)
python -m pip install -e ".[dev]"

# the learning / GPU extra (Goals 6-12): pinned torch + torch-geometric
python -m pip install -e ".[ml]"

# everything at once
python -m pip install -e ".[dev,ml]"
```

Two things worth knowing about the extras:

- **`[dev]`** pulls in `pytest` and `ruff` on top of the core dependencies
  (`sympy==1.14.0`, `numpy`, `pandas`, `pyarrow`, `pydantic`, and friends — see
  [`pyproject.toml`](pyproject.toml)).
- **`[ml]`** pins `torch==2.5.1` and `torch-geometric==2.6.1`. These are the exact CUDA/
  training versions the learning goals were run against, and the pins are enforced by a test
  (see below). Core GEML never imports torch, so you only need `[ml]` when touching Goals 6+.

## Running the checks

The three commands CI runs (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

```bash
python -m pytest
python -m ruff check .
python -m ruff format . --check
```

If you haven't done the editable install, you can still run the suite by putting `src` on the
path:

```bash
PYTHONPATH=src python3 -m pytest
```

Run all three before you push. `ruff check` catches lint (the config selects `B,E,F,I,N,PT,
RUF,SIM,UP`); `ruff format --check` verifies formatting without rewriting files — drop
`--check` to actually apply it.

### Two tests that are *supposed* to fail on a mismatched machine

These are correct-by-design guards, not repo bugs. If you hit them, fix your environment —
don't patch the test.

- **`tests/export/test_goal5_export.py`** imports the `geml` package directly. It only passes
  once the package is importable — i.e. after `pip install -e .` (or with `PYTHONPATH=src`).
  A bare `pytest` in a fresh checkout with no install will fail it at import time. Installing
  the package fixes it.
- **`tests/learning/test_ml_env.py`** enforces the exact pinned ML versions from
  `configs/ml_env.yaml`. If you have the `[ml]` extra installed but at a different torch/PyG
  version than the pins (e.g. a system torch 2.9.x instead of the pinned 2.5.1), that test
  will correctly fail — it is guarding training reproducibility. Match the pins, or leave the
  `[ml]` extra uninstalled (the test skips when torch isn't present).

## Branches and pull requests

- **One issue per PR.** If your work spans multiple issues, split it into one PR each. Don't
  bundle.
- **Branch naming.** Use a short prefix and a concise slug. Issue-tracked goal work follows
  `issue/<goal>-<sub>-<slug>` (e.g. `issue/6-1-pairs-cli`); other work uses a category prefix
  like `enh/<slug>` or `fix/<slug>` (e.g. `enh/lock-transitive-pins`). Match what's already
  in the history.
- **Stay inside your write scope.** The PR diff should touch only the paths your issue owns.
- **Before you open the PR**, confirm: all three checks pass; every new number traces to a
  goal summary + manifest; failures and nulls are reported plainly; no v0-prototype code was
  consulted; no production corpus or `outputs/` artifact was made a test dependency.

The [pull request template](.github/PULL_REQUEST_TEMPLATE.md) walks through this checklist.

## Tests and generated data

Tests must run on a fresh clone with **no** `outputs/` directory and **no** production
artifacts present. Use tiny inline or `tmp_path` fixtures. Production corpus shards and
generated artifacts must never become test dependencies — that keeps CI hermetic and the
provenance clean.

## Scientific integrity

- Document every domain assumption and every metric-definition change.
- Keep failures, unsupported inputs, timeouts, and validation errors **visible** in the
  accounting; never silently discard them.
- Never hide unsupported operators in derived leaves, and never relabel macro/motif nodes as
  pure EML to improve a reported number.
- Keep structural identity separate from semantic equivalence; preserve ordered child slots
  and repeated references wherever the governing contract requires them.

That's it. When in doubt, over-communicate on the issue and keep the numbers honest.
