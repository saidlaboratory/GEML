# Goal 6 — equivalence learning grid

**Status:** `phase_a_implemented` / `production_pending`.
**No production result in this document is real, because no production run has happened.**

Every numeric field below is an explicit machine-readable missing state. This file is regenerated
from saved result rows by `src/geml/analysis/goal6/summary.py`; it is never hand-edited with
plausible numbers.

---

## 1. What Goal 6 asks

At the fixed 250k-v1 structural scale, does pure-EML structural regularity or motif-aware graph
structure improve equivalence classification relative to honest AST and sequence controls?

The comparison is six arms on three preregistered seeds:

| Arm | Family | Input |
|---|---|---|
| 1 | graph | approved channel 1 |
| 2 | graph | approved channel 2 |
| 3 | graph | approved channel 3 |
| 4 | graph | approved channel 4 |
| 5 | prefix transformer | prefix token sequence (`geml-goal6-prefix-tokens-v1`) |
| 6 | trivial floor | 19 predeclared surface counts (`geml-goal6-trivial-floor-v1`) |

The four graph arms share **one** encoder implementation, one weight-initialization policy, one
optimizer, one epoch budget, and one task head. They differ only by input channel.

## 2. Results

```
status: missing
reason: cell_missing
detail: no production cells have been executed; Phase A implemented code, configs, and fixture
        tests only
```

Expected cells: 18 (6 arms x 3 seeds). Present cells: 0.

## 3. Blocking dependencies

### 3.1 Issue 6-2 / #56 four-channel contract — **resolved before production**

The integration decision in `docs/specs/PRE_PHASE_B_DECISIONS.md` approves AST-DAG,
pure-EML-DAG, the honestly labeled macro-derived frequent-motif DAG, and a new train-only
motif-AST fair control. The control is capped by both the reference vocabulary entry limit
and exact dictionary MDL bits, and every compressed graph must reconstruct.

Production artifacts are not yet materialized, but the channel identity/scope blocker is
closed. The runner still refuses missing or misaligned channel rows and dishonest relabeling.

### 3.2 Encoder width and virtual-node freeze — **pending preflight**

Issue 6-3 requires the hidden width (64 or 96) and the virtual-node setting to be frozen once, by
measured parameter/FLOP matching. A fair measurement needs production channel vocabularies, which
do not exist yet, so `configs/goal6_grid.yaml` records both as `pending_preflight_freeze` rather
than guessing.

Measured with the fixture vocabulary defaults:

| Width | Virtual node | Total parameters | Target status |
|---:|---|---:|---|
| 64 | no | 185,732 | below_target |
| 64 | yes | 210,692 | within_target |
| 96 | no | 343,044 | within_target |
| 96 | yes | 398,916 | within_target |

The width-64 configuration without a virtual node sits marginally under the issue's approximate
0.2M lower bound. That is recorded rather than corrected: padding the architecture to reach a round
number would be tuning to a target instead of measuring one. Production label vocabularies differ
per channel and will move these totals, which is precisely why vocabulary embeddings count toward
the parameter match.

### 3.3 Strict OOD views — conditional

- `test_ood_stress` is Goal 1's stored combined stress profile across size, depth, variable count,
  and composition. It is reported under that name and is **not** relabelled strict depth-OOD.
- `test_depth_ood` requires predeclared, disjoint depth support between training and evaluation.
  Absent that, the runner records `unsupported_not_disjoint`.
- `test_family_ood` requires at least one family genuinely excluded from training, not merely
  filtered out of a test table. Absent that, the runner records `unsupported_not_excluded`.

## 4. Reporting commitments

- Raw per-seed rows are always published; the mean never replaces them.
- A missing or failed cell is an explicit missing state, never a zero.
- Every aggregate reconstructs its own denominator.
- Quality is reported jointly with parameters, FLOPs, wall time, and memory. Equal parameter count
  is not equal compute, because pure-EML graphs are much larger than AST or macro graphs for the
  same expression.
- Structural metrics are per-channel and carry a `comparable_across_channels` flag. There is no
  pooled cross-channel alpha column, and the plotting code refuses to draw one.
- Contrasts are paired within seed and within identical pair identities, with group-level
  resampling. Correlated pair rows are never treated as independent samples.
- Null and negative findings are reported plainly and are not resolved by rerunning.

## 5. Fixed-scale limitations

All Goal 6 claims are bounded by the 50,000 training pairs and the frozen 250k-v1 corpus. No
scaling law is fitted and no result is extrapolated to a larger corpus, a longer training budget,
or a different operator grammar. With three seeds, reported effect sizes are descriptive; no
asymptotic significance is claimed.
