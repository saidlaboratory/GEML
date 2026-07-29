# Gate G6 — predeclared decision rules

**Status:** rules frozen, verdict not yet issued.
**Schema:** `geml-goal6-gate-v1`
**Implementation:** `src/geml/analysis/goal6/summary.py::evaluate_gate_g6`

This document is written **before** any production result exists. That ordering is the point: a
gate whose rules are chosen after the numbers are visible is not a gate.

---

## 1. The three permitted states

| State | Meaning |
|---|---|
| `pass` | The predeclared evidence requirements are met **and** at least one predeclared contrast reaches the preregistered effect-size threshold in the positive direction. |
| `fail` | The evidence requirements are met, and no predeclared contrast reaches the threshold. This is a **reportable null result**. |
| `insufficient_evidence` | The evidence requirements are not met: arms, seeds, or cells are missing or failed, or the report was generated from fixture rows. |

There is no fourth state, no "partial pass", and no provisional verdict.

## 2. Evidence requirements (all must hold for `pass` or `fail`)

1. All six arms are present: four approved graph channels, the compute-matched prefix transformer,
   and the trivial floor.
2. Every arm has a result row for **all three** preregistered seeds `20260726`, `20260727`,
   `20260728`.
3. Every cell has status `complete`. A failed, timed-out, or incomplete cell forces
   `insufficient_evidence` — it is never dropped to make the remaining cells look complete.
4. Every reported view reconstructs its own denominator: `attempted`, `valid`, `failed`,
   `unsupported`, and `timed_out`, summing exactly to `attempted`.
5. All four graph channels were scored on **identical** pair identities. Any misalignment sends the
   affected pairs to the failure ledger rather than quietly shrinking one arm's subset.

If any requirement fails, the verdict is `insufficient_evidence` and every unmet requirement is
listed by name in `unmet_requirements`.

## 3. Predeclared contrasts

Contrasts are **paired within seed** and within identical pair identities. Independently pooled
means are not used: the arms share seeds and share pairs, so pooling would discard the pairing that
makes the comparison informative.

The contrast set is registered as `(left_arm, right_arm, view, metric)` tuples and is frozen
before production. The intended primary set is:

- each graph channel against the trivial floor, on `test_iid`, metric `accuracy`;
- each graph channel against the prefix transformer, on `test_iid`, metric `accuracy`;
- the pure-EML channel against the AST channel, on `test_iid`, metric `accuracy`.

The exact registered tuples must be recorded in the run manifest at freeze time.

## 4. Effect-size threshold

`minimum_effect_size = 0.5`, applied to the **paired** standardized mean difference across the
three seeds, with the mean difference required to be positive for a `pass`.

With three seeds this is explicitly a *descriptive* effect size. The gate does not compute a
p-value and does not claim asymptotic significance. Raw per-seed differences are always published
alongside it so a reader can see the whole sample.

## 5. What the gate deliberately does not do

- It does not fit a scaling law or extrapolate beyond 50,000 training pairs and the fixed 250k-v1
  corpus.
- It does not reduce an arm to a single score; quality is always reported jointly with parameters,
  FLOPs, wall time, and memory.
- It does not pool structural metrics across representations. Pure-EML alpha, ordinary node/edge
  size, macro size, and dictionary-inclusive motif MDL are different quantities and are never
  averaged or plotted as one "alpha".
- It does not permit a rerun to change a seed. A technical rerun keeps the same seed and records
  the failure cause, the old row, the new row, the code/config hash, and the reason.
- It does not treat a `fail` as a problem to be fixed. A null result at this scale is a first-class
  finding for the workshop paper.

## 6. Current verdict

```
state: insufficient_evidence
reason: no production result rows exist; Phase A implemented the analysis and gate machinery only
```

The blocking dependencies are recorded in `GOAL6_SUMMARY.md`.
