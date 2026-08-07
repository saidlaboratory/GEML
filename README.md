# GEML — Graph Learning over a Single-Operator Representation of Mathematics

CI is defined in `.github/workflows/ci.yml` and is run on the anonymous review snapshot.
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![tests](https://img.shields.io/badge/tests-2805%20passing-brightgreen)

> **A controlled representation study.** GEML collapses every elementary math operator into one primitive — `eml(x, y) = exp(x) − ln(y)` — so any expression becomes a graph where every internal node is the same operation. The project has delivered the first exact cost profile of that representation at 250,000-expression scale and the first learning verdicts on it: the single-operator form is learnable despite its **~40× median expansion**, and the preregistered diagnostics pinpoint usable search guidance as the open frontier.

**Paper:** [*One Operator, Measured Exactly*](paper/main.pdf) ·
**Target venue:** [MathNLP 2026](https://sites.google.com/view/mathnlp2026) ·
**Project site:** [`docs/`](docs/index.html)

**Authors:** Anonymous submission. Author identities are intentionally omitted for double-blind review.

---

## 1. What GEML is

Odrzywołek (2026) showed that the standard elementary functions and constants (π, e, i) can be
built from the constant **1** and a single binary operator:

```
eml(x, y) = exp(x) − ln(y)
```

Any continuous mathematical expression therefore maps to a strict binary tree in which **every
internal node is the same operation**. GEML asks whether that homogeneity is *useful for machine
learning*. Instead of feeding math to a model as a token sequence (`[sin, (, x, +, 1, )]`), it
feeds the structural topology of the EML tree to a graph neural network that only has to learn
*where things connect*, never *what the operator is*. The one quantitative question the project
measures is whether that representation — far larger than an ordinary AST — earns its keep on
symbolic reasoning, and it commits to publishing the answer whichever way it lands.

> **Contributors / coding agents:** a v0 prototype exists but is off-limits. Build clean-room from
> the current repository specs and issues, never from the prototype's code, tests, schemas, or
> history. See [`AGENTS.md`](AGENTS.md) and [`docs/CLEANROOM_RULES.md`](docs/CLEANROOM_RULES.md).

### Quickstart

GEML is a Python 3.12 package with a `src/` layout; runtime code lives under `src/geml/`.

```bash
python -m pip install -e ".[dev]"   # install package + dev tools
python -m pytest                     # run the suite (2,805 tests)
python -m ruff check . && python -m ruff format . --check
```

Where to look: per-goal results are under [`docs/goals/`](docs/goals/), the frozen contracts under
[`docs/specs/`](docs/specs/), and the rendered overview in [`docs/index.html`](docs/index.html).
For the full dev guide — environment, clean-room rules, contribution workflow — see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## 2. Framework

### Topology, not vocabulary

A conventional model of mathematics has to learn a vocabulary: `sin`, `exp`, `+`, `/`, and dozens
more, each a distinct token or node type. The EML rewrite erases that vocabulary — one operator
does everything — so the only thing left to learn is *structure*. A graph neural network over an
EML tree never sees an operator label; it sees how the graph is wired. GEML's bet is that a model
freed from operator identity might generalize better across the shape of an expression. The cost of
that bet is measured, not assumed, and the measurement is the α threshold.

### The α threshold — the central quantitative question

Let `α = |T_EML| / |T_AST|` be the expansion factor when an AST is rewritten in pure EML form.
Counting the representable expressions of size *x* as `N(x) = C_x · K^x · L^(x+1)` (Catalan tree
shapes × operator labels × leaf symbols, with `C_x → 4^x / (x^{3/2}√π)`), the reduced EML operator
vocabulary is only a net win when

```
α < 1 + ln(K) / ln(4L)
```

where `K` is the number of operator types and `L` the number of leaf symbols. For the full grammar
this counting break-even is ≈ **1.56**; across the six preregistered per-family grammars the
operative thresholds run from **1.29 to 1.50** ([Goal 2](docs/goals/goal2/GOAL2_SUMMARY.md)) — even
the most generous bar sits near 1.5. Whether raw EML clears that bar, and whether compression can
bring it under, is the quantitative spine of the whole project. (It does not: see
[Results](#3-results).)

### Four representation channels

Every learning experiment is run over the same four channels so a win can be attributed to the
right cause — graph sharing vs. the EML rewrite vs. motif compression:

1. **AST-DAG** — the source AST with exact subtree sharing. The fairness baseline that separates
   "graph sharing helps" from "EML helps."
2. **Pure EML-DAG** — the canonical single-operator channel, assumption-free. Every EML claim
   traces back to it.
3. **Frequent-motif EML-DAG** — a lossless dictionary of common EML subgraphs; the "practical EML"
   channel (smaller, still reconstructable).
4. **Motif-AST** — the same motif compression applied to the AST, a train-only fair control so a
   motif gain is never mistaken for an EML gain.

### Semantic verification — e-graph, verifier-gated

Equivalence in GEML is **semantic, not structural**: two equal expressions are generally not
isomorphic graphs. Equivalence pairs and rewrite candidates come from equality saturation over an
e-graph, and every candidate is checked two ways — same-e-class membership (formal evidence
relative to the enabled rule set and each row's recorded domain assumptions) plus an independent
domain-aware numeric probe that catches bugs the formal check cannot. Every rewrite, proof, and
simplification step passes a verifier before it counts. The gate is **sound, not complete**: it
never accepts an unverified step, but its bounded rule set and bounded search mean it can miss valid
rewrites rather than emit wrong ones. Pure EML stays canonical throughout; positive-real results are
labelled as conditional findings, never as universal complex identities.

### The experimental program

The structural layer above (Goals 1–5) feeds a four-track learning program. Each track is
representation-agnostic and ends at an explicit **gate** — a pass/fail rule that decides whether the
next track proceeds, proceeds narrowed, or stops. **The first verdicts are in**
(2026-07-31, four-H100 run; full ledger in [`results/`](results/)): the single-operator
representation is *learnable* — two of three equivalence seeds clear the operative
chance floor despite the 40× expansion — and the preregistered rank/OOD diagnostics proved
their worth by catching a value-head guidance signal that error metrics alone would have
passed. The discriminative EML-vs-AST comparison, symbolic regression, and the capstone
synthesis are still production-pending, and every missing run yields an explicit
missing-state, never a plausible-looking number.

- **Equivalence learning** ([Goal 6](docs/goals/goal6/GOAL6_SUMMARY.md), *first verdict:
  learnable*) — can a GNN learn `E₁ ≡ E₂`, and under which channel? **First GPU verdict:** trained
  on pure EML-DAG cells, two of three seeds reach equivalence validation loss ≈ 0.53 BCE. The
  operative chance floor is not the balanced-label 0.69: accepted pairs are ~74% positive, so a
  constant majority-rate predictor already attains ≈ 0.58, and the converged seeds clear that floor
  by 0.04–0.05 nats — real but modest learning despite the 40× expansion (the third seed diverged,
  at 2.39). The full six-arm EML-vs-AST grid × three seeds is still pending, so **Gate
  G6** (every GNN arm beats the trivial floor; EML-vs-AST recorded either way) is not yet decided.
- **Rewrite-step prediction** ([Goal 7](docs/goals/goal7/GOAL7_SUMMARY.md), *partial*) — from a
  state graph, predict the next `(rule id, application site)`, scored by top-k *verifier-valid* step
  accuracy. **First GPU run:** 13 of 18 logical cells complete (5 invalid, kept in the denominator);
  a separate retrieval grid is 15/15. **Gate G7** stays open until the grid completes.
- **Verified proof paths** ([Goal 8](docs/goals/goal8/GOAL8_SUMMARY.md),
  *first verdict: guidance not yet learned*) — best-first/beam search over rewrites with the
  verifier gating every step. **First GPU verdict:** the reduced-budget goal-conditioned value
  head reaches low error (MAE ≈ 0.83) but ranks at chance (test Spearman ≈ 0) with a *constant*
  out-of-domain predictor — exactly the failure mode the preregistered rank and OOD diagnostics
  exist to catch, and MAE alone would have missed. Recorded as a methodological finding;
  **Gate G8** is not yet passed.
- **Symbolic regression** ([Goal 9](docs/goals/goal9/GOAL9_SUMMARY.md), *`insufficient_evidence`*) —
  recover an in-grammar expression from numeric samples via encoder-guided search, EML-space vs.
  AST-space, against PySR/GP and transformer-SR references. **Gate G9:** exact-recovery above the GP
  baseline at matched budget, or a documented negative.

Domain expansion ([Goal 10](docs/goals/goal10/GOAL10_SUMMARY.md)) adds a grammar-v2 surface
(`asin`, `acos`, `atan`, `π`, `e`) and re-audits the compiler — that audit produced a final
published result (see below). A scale-up and parameter-efficiency comparison
([Goal 11](docs/goals/goal11/GOAL11_SUMMARY.md), *production pending*) is the capstone that
synthesizes the four tracks without retraining.

## 3. Results

The structural measurement (Goals 1–5) is complete on the full 250,000-expression corpus — an
exact, provenance-bound cost profile of the single-operator representation. Every row links to
its full, machine-generated summary.

| Goal | Result |
|---|---|
| [1](docs/goals/goal1/GOAL1_SUMMARY.md) | 250,000 unique expressions, QA-gated, split exactly 175k / 25k / 25k / 25k (train / validation / test_iid / test_ood). |
| [2](docs/goals/goal2/GOAL2_SUMMARY.md) | Raw pure-EML expansion: **median α = 40.6602**, mean 952.1371 (p99 ≈ 10,448.6 — a heavy right tail). **0 / 250,000** expressions fall below the preregistered 1.29–1.50 per-family thresholds. |
| [3](docs/goals/goal3/GOAL3_SUMMARY.md) | Lossless DAG sharing compresses the expanded EML tree **39.375× on average** — yet the EML DAG still beats the AST tree on **0 / 250,000** expressions (best remaining ratio 8/7). Compressing well and becoming competitive are different claims. |
| [4](docs/goals/goal4/GOAL4_SUMMARY.md) | Verifier-gated e-graph rewriting improves **23.9%** (`safe_real`) / **27.6%** (`positive_real_formal`) of costed rows at **2.4–2.8%** mean relative savings, on **60.7%** vocabulary coverage. |
| [5](docs/goals/goal5/GOAL5_SUMMARY.md) | Preregistered comparisons: the equal-budget frequent-motif baseline edges the learned vocabulary (317.7M vs. 324.5M MDL bits, test_iid), and a structural heuristic edges the neural ranker — transparent baselines remain the bar to beat. |
| [10](docs/goals/goal10/GOAL10_SUMMARY.md) | Grammar-v2 conformance audit recorded (Gate G10): the 8 preregistered `asin`/`acos` endpoint cells are nonfinite at 100-digit precision; scoped to the opt-in v2 surface only. |

**What the numbers say.** The single-operator rewrite sits well above the counting threshold
(median α 40.66 vs. a ~1.5 bar), and exact sharing — though it shrinks the expanded form ~39× —
does not close the gap to the ordinary AST. Homogeneity is therefore not free structure, which
makes the learning verdict the interesting one: despite that cost, the representation is
learnable on binary equivalence, and what remains to be won — usable search guidance —
is exactly what the frozen tracks in [the framework](#2-framework) are built to measure next.

All numbers above come from clean-committed production runs; the exact commands, config hashes,
and content hashes are in each goal's summary.

## References

- **Paper:** *One Operator, Measured Exactly: Cost, Learnability, and the Guidance Frontier in a
  Single-Operator Representation of Mathematics* — [`paper/main.pdf`](paper/main.pdf)
  (MathNLP 2026 submission).
- Odrzywołek, A. (2026). *The EML function* — reduction of elementary functions to `exp(x) − ln(y)`.
  The official EML compiler is used for all pure-EML conversions: no abbreviations, no hidden derived
  leaves.
- Per-goal machine-generated summaries and QA evidence: [`docs/goals/`](docs/goals/); frozen
  contracts: [`docs/specs/`](docs/specs/).
- The Goals 1–5 artifact archive (including the graph exports) and the full
  reproducibility protocol live in [`CONTRIBUTING.md`](CONTRIBUTING.md#reproducibility).

## License

GEML is released under the [MIT License](LICENSE), Copyright (c) 2026 GEML contributors.
