# Gate G9 — bounded symbolic regression

**Owner:** issue [9-3] (#74) · **Schema:** `geml-gate-g9-v1`
**Current state:** `insufficient_evidence`
**Decision procedure:** `geml.analysis.goal9.summary.decide_gate`

> The state above is the value the implemented decision procedure returns for the inputs that
> exist today. It is not a placeholder. No Goal 9 production search, baseline, or analysis has
> been run, and no gate verdict may be written into this file by hand.

---

## 1. Predeclared states

Exactly three, frozen before any result exists:

```text
pass
fail
insufficient_evidence
```

---

## 2. The intended comparison

Verifier-confirmed **exact symbolic recovery** by the EML-guided arm, against the pinned
PySR reference — or the explicitly labelled in-repository GP fallback when PySR is genuinely
unavailable — **under the matched budget**, on the frozen benchmark.

| Role | Method |
|---|---|
| Treatment | `eml_guided` |
| Representation control | `ast_guided` |
| Reference | `pysr`, or the explicitly labelled `gp_fallback` |
| External context, never in the gate | issue 11-3 LLM rows |

An explicit negative result is a first-class outcome. `fail` is a publishable finding, not a
reason to rerun, reseed, or reduce the benchmark.

---

## 3. Decision rule

The procedure is evaluated in this order and stops at the first match.

1. **`insufficient_evidence` if exact-verification coverage is unknown.** Coverage is known
   only when every benchmark manifest has status `complete`. While the Goal 9
   verification-scope decision is open, every manifest is stamped
   `blocked_pending_verifier_decision`, so the gate cannot pass. This is a hard interlock:
   there is no code path from an unknown-coverage benchmark to `pass`.
2. **`insufficient_evidence` if the report is not complete.** Any blocking validation issue —
   a missing manifest, missing rows, a schema-version mismatch, or unmatched budgets between
   the two representation arms — forces this state. A fixture-only report is always
   `insufficient_evidence`.
3. **`insufficient_evidence` if either side of the comparison produced no rows.** Both the
   EML-guided arm and at least one pinned reference must have results.
4. Otherwise compare verifier-confirmed exact recoveries under the matched budget:
   * **`pass`** if the EML-guided arm achieved strictly more verifier-confirmed exact
     recoveries than the pinned reference;
   * **`fail`** otherwise, recorded as an explicit negative result with both counts.

---

## 4. What can never move the gate

* **Numeric fit.** A recovery is counted only from a `verified` equivalence outcome. No
  root-mean-squared error, however small, can produce a recovery.
* **LLM rows.** `CONTROLLED_METHODS` contains no LLM method, so issue 11-3 rows cannot enter
  a method summary, a contrast, a Pareto point, or the gate. They are counted separately as
  `external_reference_rows` and labelled external.
* **Post-hoc subsetting.** Denominators are attempted-first. Dropping unsupported, invalid,
  or timed-out rows from a denominator is not available to any method.
* **Unmatched budgets.** If the two representation arms did not share one budget digest,
  validation raises `unmatched_budgets` and the gate returns `insufficient_evidence`.
* **Fixture data.** `summarize(..., fixture_only=True)` forces `fixture_only` completeness
  and therefore `insufficient_evidence`, and `geml.plots.goal9.render_plots` refuses to draw
  figures at all for a non-complete report.

---

## 5. Recorded with every verdict

`GateG9` persists: the state, the rationale string, `verifier_coverage_known`,
`verifier_supported_tasks` out of `benchmark_tasks`, the treatment, control, and reference
methods, both verified-recovery counts, whether external LLM rows were considered (always
`false`), and the caveat list.

The enclosing `Goal9Summary` additionally persists `rows_digest`, a SHA-256 over the exact
result rows the verdict was computed from, so a published gate can be tied back to the
evidence.

---

## 6. Blocking conditions today

| Blocker | Effect on the gate |
|---|---|
| Goal 9 verification-scope decision is open | manifests are not frozen → coverage unknown → cannot pass |
| No production search or baseline rows | report is incomplete → `insufficient_evidence` |
| Workstream 2 backbone unmerged | transformer-SR reports `dependency_unavailable` |
| PySR install decision not made | reference side may be `gp_fallback`, labelled as such |

---

## 7. Bounded-benchmark caveats attached to any future verdict

1. Fixed-scale benchmark — 256 synthetic tasks plus a frozen restricted Feynman-style subset.
   No claim extends to corpus-size scaling or to general symbolic-regression superiority.
2. Three seeds: raw variation is published; no asymptotic significance is claimed.
3. Exact-recovery coverage is bounded by the assigned verifier's declared capability; tasks
   outside it are `unsupported`, not failures to recover.
