# GEML results ledger

This directory is the compact, reviewable numerical record for the GEML project.  It
contains no corpus shards, model checkpoints, graph archives, logs, or other large
binary artifacts.  The ledger records the results that can be independently traced to
the final Goal 1--5 manifests/reports and to the authenticated Phase-B GPU evidence
package.

## Scope and interpretation

* `GOALS_1_5.md` is the finalized CPU/data-pipeline record for Goals 1--5.
* `PHASE_B_GPU.md` is the record of the 2026-07-31 Phase-B run.  It preserves exact
  per-seed metrics, denominators, invalid cells, null outcomes, package hashes, and
  hardware/software provenance.
* `GOALS_6_12.md` distinguishes authenticated Phase-B outcomes from goals that are
  still unrun, stale in the repository release checklist, or scientifically null.
* `results.json` is the machine-readable ledger.  Missing or unavailable values are
  represented by JSON `null` with a status explaining why; no values are imputed.
* `PROVENANCE.md` maps every headline number to a source path, schema, commit, or
  artifact digest.  The source artifacts remain in the separate artifact store.

The authoritative repository snapshot used for this ledger is
`origin/main` at commit `5de31bf34945fcdaef5dbe10d9df819fa98b1ca5`.  The GPU package
was produced from runtime source commit
`53a34d2d37e0912bd17feb01c84c97ad35e4455b` and is not silently substituted for the
repository's older Phase-A documentation.

## Important caveats

1. Goal 7 has 18 preregistered logical cells: 13 completed and 5 invalid under the
   frozen wall budget.  The raw status directory also contains eight retained failed
   attempts (26 raw status records).  Invalid cells are counted, not discarded.
2. Goal 8's reduced-budget value head completed three 1,500-step runs, but the
   scientific review is `null_or_collapsed_value_head`: its validation loss is
   reportable, while its test ranking is not a useful proof-search/simplification
   result.  OOD Spearman is null because the measured values were constant.
3. The Phase-B package does not constitute completed Goal 9, 10, 11, or 12 evidence.
   See `GOALS_6_12.md` for the exact boundary.
4. The ledger is intentionally small.  Use `ARTIFACT_INDEX.md` and the Phase-B
   package's verification script when raw artifacts are needed.
