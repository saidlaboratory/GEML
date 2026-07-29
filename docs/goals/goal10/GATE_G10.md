# Gate G10: opt-in compiler conformance

Possible outcomes are `pass`, `fail`, and `insufficient_evidence`.

A final `pass` requires:

- an authenticated final bounded conformance manifest;
- an exact match to the preregistered conformance-configuration SHA-256;
- every required constructor/compiler-mode cell;
- exact preregistered fingerprints, node counts, depths, and strict purity;
- all expected-valid numeric cases within preregistered tolerances;
- correct retention of invalid and unsupported rows;
- exact v1 registry, domain, domain-rule-ID, trig, and hyperbolic formula hashes;
- a known implementation commit produced from a clean worktree;
- no unresolved ownership or signed-zero evidence limitation.

Any integrity, exact-structure, v1-drift, coverage, or expected-valid numeric
failure yields `fail`. Fixture evidence, missing inputs, unsupported
expected-valid cases, or unresolved ownership blockers yield
`insufficient_evidence`.

The current Phase-A bounded audit is `fail`: required exact endpoints are valid
at the source level but nonfinite in the pinned high-precision raw-tree
evaluation. A missing or fixture-only audit with no demonstrated violation
would be `insufficient_evidence`; observed required-case failures are not
downgraded to insufficient evidence.

**Published verdict.** The final-tier audit was executed and published on
2026-07-29 (lead decision): Gate G10 = `fail`, 74 retained rows, zero
integrity/coverage/v1 failures, exactly the eight preregistered `asin`/`acos`
endpoint cells failing as `nonfinite_result`. Full provenance hashes are
recorded in `GOAL10_SUMMARY.md`; the immutable machine-rendered artifacts live
under `outputs/final/goal10/audit/`. No criteria revision is pre-authorized:
any later revision would require a new explicitly recorded decision as a
separate step and would produce a new, separately-versioned verdict.

Gate G10 concerns compiler conformance only. It does not authorize claims
about corpus v2, alpha, DAGs, motifs, compression, Goal 6/7 models, or other
learned effects.

Run and publish the bounded audit with:

```bash
python -m geml.experiments.goal10.rerun_studies \
  --conformance-config configs/goal10_corpus_v2.yaml \
  --audit-config configs/goal10_rerun.yaml \
  --run-dir outputs/final/goal10/conformance \
  --output-dir outputs/final/goal10/audit \
  --evidence-tier final
```

The command independently rebuilds every configured case/mode result, rejects
self-consistent tampering, writes `audit.json` and `GOAL10_SUMMARY.md`
immutably, and exits 0/1/2 for pass/fail/insufficient evidence. Final evidence
also requires a known commit and clean worktree.
