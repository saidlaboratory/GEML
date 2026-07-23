# GEML — Graph Learning over a Single-Operator Representation of Mathematics

> **A controlled representation study:** does collapsing every elementary math operator into the single primitive `eml(x, y) = exp(x) − ln(y)` help graph neural networks learn symbolic reasoning — given that it *inflates* tree size?

**Target venue:** [MathNLP 2026](https://sites.google.com/view/mathnlp2026) ·
**Project site:** [`docs/`](docs/index.html) ·
**v0 prototype (Goals 1–5, complete):** [sahilsinghthefirst/geml_experiments](https://github.com/sahilsinghthefirst/geml_experiments)

**Lead:** Quang Bui ([@duckyquang](https://github.com/duckyquang)) ·
**Members:** Muhammad Rayyan ([@Mray229](https://github.com/Mray229)), Sibi Gokul ([@Sisigoks](https://github.com/Sisigoks)), Daksh Jain ([@Daksh-QE](https://github.com/Daksh-QE)), Sahil Singh ([@gensahilsingh](https://github.com/gensahilsingh))

---

## 1. What GEML is

Odrzywołek (2026) showed that the standard elementary functions and constants (π, e, i) can be constructed from the constant **1** and a single binary operator:

```
eml(x, y) = exp(x) − ln(y)
```

Any continuous mathematical expression therefore maps to a strict binary tree in which **every internal node is the same operation**. GEML asks whether that homogeneity is *useful for machine learning*: instead of feeding math to a model as a token sequence (`[sin, (, x, +, 1, )]`), we feed the structural topology of the EML tree to a graph neural network that only has to learn *where things connect*, never *what the operator is*.

This repository hosts the clean-room production rebuild of Goals 1–5. The v0 prototype (data generation, expansion study, and three families of compression) lives in [geml_experiments](https://github.com/sahilsinghthefirst/geml_experiments) and is **complete through Goal 5R** with 262/262 tests green.

> ⚠️ **Clean-room warning for contributors and coding agents:** the prototype is historical context only. Do not inspect or reuse its code, tests, schemas, helpers, architecture, or history. Implement from the current repository specifications, assigned issues, and authoritative public sources. See [`AGENTS.md`](AGENTS.md) and [`docs/CLEANROOM_RULES.md`](docs/CLEANROOM_RULES.md).

### Development setup

GEML is a Python 3.12 package using a `src/` layout. Runtime code lives under
`src/geml/`, while unit tests use only small local or temporary fixtures.

Install the package and development tools:

```bash
python -m pip install -e ".[dev]"
```

Run the standard validation suite:

```bash
python -m pytest
python -m ruff check .
python -m ruff format . --check
```

## 2. What changed since the original proposal

The original proposal ("EML-Native Foundation Models for Universal Mathematical Reasoning") framed EML compression as a premise. The v0 data says otherwise, so the project is **reframed from a universal-reasoning moonshot to a controlled representation study**. Every claim below was revised against measured evidence:

| Original claim | Revised claim | Why |
|---|---|---|
| Shrinking the math vocabulary to one operator reduces model size | **Hypothesis under test:** raw EML trees are ~11–12× *larger* than ASTs. Compression (DAG + motifs) makes EML *trainable*; whether homogeneity then *generalizes better* is the open question | Goal 2: mean α ≈ 10.6, median ≈ 11.4, essentially never below the theoretical threshold (~1.56) |
| "GPT-4-level mathematical competency at a fraction of the parameters" | **Competitive accuracy with fewer parameters on a bounded algebraic domain** | Unfalsifiable as stated; parameter-efficiency is now an explicit experiment (Goal 11.2) |
| Universal / zero-shot automated theorem proving | **Verified equational rewrite-proofs on a bounded domain** | GEML operates on a bounded algebraic fragment; proofs are verifier-gated rewrite paths (Goal 8) |
| "Equivalence as graph isomorphism" | **Equivalence is semantic, not structural.** Learned from e-graph-generated equivalence pairs with rule-sequence provenance, always behind a verifier | Two equivalent expressions are generally *not* isomorphic graphs |
| EML as a universal "machine code for AI" | Removed from claims (at most a speculative closing remark) | No evidence; claim discipline (Goal 12.2) |

## 3. The central question: the α threshold

Let `α = |T_EML| / |T_AST|` be the expansion factor when an AST is rewritten in pure EML form. Counting representable expressions of size *x* as `N(x) = C_x · K^x · L^(x+1)` (Catalan shapes × operator labels × leaf symbols) and using `C_x → 4^x / (x^{3/2}√π)`, EML's reduced operator vocabulary wins only if:

```
α < 1 + log_{4L}(K)
```

where `K` = number of operator types and `L` = number of leaf symbols. For the current grammar the threshold is **≈ 1.56** (with e.g. K = 20, L = 3 it would be ≈ 2.21). Raw EML sits at **α ≈ 11.4** — an order of magnitude over. The entire compression program (Goals 3–5) exists to close that gap honestly, and the learning program (Goals 6–11) tests whether what remains is worth it.

## 4. Evidence so far — v0 prototype (Goals 1–5, 10k-sample corpus)

| Goal | Deliverable | Status | Headline result |
|---|---|---|---|
| 1 | Expression generator, AST/EML converters, metrics | ✅ | Bi-directional SymPy/LaTeX ↔ binary-tree pipeline; strict no-hidden-leaves accounting |
| 2 | Expansion study (raw EML vs AST) | ✅ | Raw EML ~11× larger; α essentially never below threshold |
| 3 | Exact structural DAG compression | ✅ | Median α 11.4 → 3.5 (~3.1× shrink), lossless |
| 3R | Generator repair → v1 corpus | ✅ | Fixed depth/log/duplicate biases; byte-reproducible |
| 4 | E-graph (semantic) compression | ✅ | ~73% of expressions improved; still only ~5% below threshold |
| 5 | ML-facing compression (macro/motif/neural/hierarchical) | ✅ | Motifs strongest; learned selection adds nothing yet |
| 5R | Repair pass (tests, real reconstruction, honest framing) | ✅ | 262/262 tests green on fresh clone |
| 6+ | GNN training onward | ⏳ | This repo — see roadmap below |

**Compression results** (v1 corpus; gains relative to the Goal 3 pure EML-DAG baseline):

| Method | What it does | Median gain | Notes |
|---|---|---|---|
| Pure EML-DAG | Exact subtree sharing | 1.0× (baseline) | Lossless, assumption-free — the non-negotiable control |
| E-graph, positive-real | Semantic rewrites + extract | ~1.2× | Needs positivity assumptions; keep as optional pre-pass |
| Macro graph | Relabeled compact AST-DAG | 5.25× (α = 0.78) | Structurally ≈ AST-DAG; doubles as interpretable control |
| **Frequent motif** | Dictionary of common subgraphs | **7.40×** | Best simple compressor, reconstruction-verified |
| Learned motif | Scored dictionary | 7.11× | ≈ random/frequency — **null result, dropped** |
| Neural e-graph ranker | Fast candidate selection | ~1.0× | 109× *scoring* speedup only; loses to hand-coded heuristic |

**Null results are recorded with equal prominence:** learned motif selection does not beat frequency ranking; the neural ranker is a speed baseline, not a compression win.

### Representation channels going forward

- **Core (A/B-tested in every learning goal):**
  1. **Pure EML-DAG** — canonical, assumption-free control; every EML claim traces back to it.
  2. **Frequent-motif-compressed EML-DAG** — the "practical EML" channel (7.4× smaller, lossless, cheap).
  3. **AST-DAG** — the fairness baseline that separates "graph sharing helps" from "EML helps."
- **Auxiliary:** e-graph positive-real as canonicalization pre-pass; neural ranker only if extraction becomes a training-loop bottleneck.
- **Dropped from critical path:** learned motif selection (revisit only at larger corpus scale).

## 5. Roadmap — Goals 6–12

Each goal ends at a **gate**: an explicit pass/fail criterion that decides whether the next goal proceeds, proceeds narrowed, or stops. Reserved repair passes (6R–11R) follow the prototype's discipline.

### Goal 6 — Equivalence Learning Grid *(the discriminative foundation)*
Can a GNN learn `E₁ ≡ E₂`, and under which representation?

| Sub | Task |
|---|---|
| 6.0 | ML dependency decision (PyTorch + PyG `[ml]` extra), training-reproducibility policy (pinned deps, seeds, metadata) |
| 6.1 | Trace-rich equivalence-pair dataset: e-graph positives **with rule-sequence provenance + step-distance**, size-matched hard negatives, tiered verification, group splits, depth-OOD + family-OOD sets (≥50k/5k/5k pairs — the Goal 7/8 fuel) |
| 6.2 | Graph materialization: AST-DAG, pure EML-DAG, motif-EML (train-mined vocab), motif-AST (fair-compression control) as PyG datasets |
| 6.3 | Reusable backbones: GIN encoder (node-type-as-feature, virtual node), compute-matched prefix transformer, trivial op-count baseline |
| 6.4 | Training harness: YAML configs, early stopping, 3 seeds/cell, CPU smoke test in CI |
| 6.5 | Baseline grid: 6 arms × metrics (acc, F1, both OODs, sample efficiency, time, memory, α) |
| 6.6 | Analysis: accuracy-vs-α curve, graph-benefit vs EML-benefit separation, honest report |

**Gate G6:** trivial baseline beaten by all GNN arms (else stop and fix the dataset); EML-vs-AST verdict recorded either way. If pure EML-DAG loses decisively on OOD, later goals proceed with the EML claim narrowed — the verifier-guided pipeline is representation-agnostic and survives.

### Goal 7 — Rewrite-Step Prediction *(the generative skill)*

| Sub | Task |
|---|---|
| 7.0 | Step dataset from 6.1 traces: (state graph) → (rule id, application site); per-rule stratification |
| 7.1 | Policy head on shared encoders (per-node site scoring + rule classification); transformer step-proposer baseline |
| 7.2 | Metrics: top-k *verifier-valid* step accuracy, rule coverage, unseen-family generalization |
| 7.3 | Grid across channels — does homogeneous topology help rule-application transfer? (the sharpest EML test) |

**Gate G7:** learned policy beats uniform-random valid-step baseline by a wide margin; no dead rules.

### Goal 8 — Verified Proof-Path Generation *(simplification + equational ATP)*

| Sub | Task |
|---|---|
| 8.0 | Search harness: best-first/beam over rewrites, verifier gate on every step (e-graph/SymPy/numeric tiers), budgets + telemetry |
| 8.1 | Value head trained on step-distance labels to guide search |
| 8.2 | Proof benchmark: held-out identity families, length-OOD, difficulty tiers |
| 8.3 | ATP comparison: guided-GNN vs uniform search vs transformer-proposer, all verifier-gated |
| 8.4 | Simplification mode: search for minimal form using Goal 4/5 exact-cost machinery; compare vs SymPy `simplify` |
| 8.5 | Analysis + frontier-LLM *reference* run (LLM as step-proposer through the same verifier — reference ceiling, not controlled baseline) |

**Gate G8:** guided search beats uniform search on nodes-expanded at equal success rate; **zero invalid steps emitted**.

### Goal 9 — Symbolic Regression Track *(EML's native home)*

| Sub | Task |
|---|---|
| 9.0 | SR task spec: numeric samples → in-grammar expression; synthetic benchmark + restricted Feynman-style subset |
| 9.1 | Encoder-guided search: EML-space vs AST-space (the SR version of the representation question) |
| 9.2 | Baselines: PySR/GP reference, transformer-SR |
| 9.3 | Metrics: exact-recovery rate, accuracy–complexity Pareto, wall-clock |

**Gate G9:** exact-recovery above GP baseline at matched budget, or a documented negative.

### Goal 10 — Domain Expansion *(add trig)*

| Sub | Task |
|---|---|
| 10.0 | Compiler + rule-set extension: sin/cos/tan, negative-domain policy, wider constants; soundness tiers per rule |
| 10.1 | Corpus v2 with expanded grammar (3R generator discipline) |
| 10.2 | Re-run α/DAG/motif studies on v2 — **the stress test** (trig may explode EML; measure, don't assume) |
| 10.3 | Re-run Goal 6/7 grids where the verdict could flip |

**Gate G10:** expanded compiler passes purity + numeric audits; documented α behavior for trig.

### Goal 11 — Scale-Up & Final Comparison

| Sub | Task |
|---|---|
| 11.0 | Corpus v3 + pairs at 10–100× |
| 11.1 | Scaling curves per channel (does the EML/AST gap grow, shrink, or invert?) |
| 11.2 | Full three-track evaluation at scale; **parameter-efficiency hypothesis tested explicitly** (small GNN vs larger transformer at matched accuracy) |
| 11.3 | Frontier-LLM reference comparison, verifier-normalized (verified-correct vs claimed-correct rate) |

**Gate G11:** every headline number has both denominators, multi-seed variance, and scaling caveats resolved or stated.

### Goal 12 — Consolidation & Release

- 12.0 Final findings report: the α-story, the representation verdict (positive **or null — published with equal prominence**), verified-soundness result, parameter-efficiency verdict.
- 12.1 Reproducibility package: one-command re-run per goal, pinned environment, archived artifacts.
- 12.2 Paper/preprint + repo release under the agreed claim discipline.

## 6. Predictions (falsifiable, to be verified by experiment)

| Task | Prediction |
|---|---|
| Direct expression evaluation | **Weak** — nested exp/ln trees explode |
| Symbolic equivalence classification | Decent at depths 1–6; degrades beyond |
| Symbolic regression | **Strongest track** — EML's native home per the original paper |
| Proof generation | Needs different architecture for complex proofs (substitutions, lemmas, case work); especially weak in geometry (construction-based) |

## 7. Team plan & working agreement

- **Group A — pipeline rebuild:** reimplement Goals 1–5 at large scale from scratch (larger dataset; heavy agent-assisted development with thorough human + agent code review). Compute: H100 if hours allow, else a personal RTX 5090 over SSH (sufficient for dataset generation and motif/macro extraction).
- **Group B — architecture & learning:** GNN/ML-experienced members build the model zoo (a Siamese GIN is a candidate) and drive Goals 6–12. This is the compute-heavy half (H100 needed).
- **Overlap:** 2–3 people bridge both groups as supervisors.
- **Open design decision:** which constants to implement — π and e are candidates; **i is likely out** (it would change the pipeline and project fundamentally).
- **Documentation is continuous:** every member documents their own work as it lands — clean, organized, usable. No end-of-project documentation crunch.
- **Paper:** ~5–7 days before deadline, 2 members start writing the core paper (codebase need not be 100% complete); remaining members switch to reviewing once code is done.

## 8. Resources

- **Compute:** 1× NVIDIA H100 (preemptible/spot) for ~3 months; RTX 5090 fallback for generation workloads.
- **Software:** PyTorch + PyTorch Geometric, SymPy, e-graph tooling; Lean 4 under consideration for verification tiers.

## 9. References

- Odrzywołek, A. (2026). *The EML function* — reduction of elementary functions to `exp(x) − ln(y)`. (Official EML compiler used for all pure-EML conversions — no abbreviations, no hidden derived leaves.)
- v0 prototype and per-goal summaries: [geml_experiments](https://github.com/sahilsinghthefirst/geml_experiments) (`docs/goal1/`–`docs/goal5/`).
