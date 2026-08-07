# Artifact coverage and cleanup boundary

This is a compact inventory of what is covered by the authoritative artifact stores.
It intentionally does **not** copy any corpus rows, graph archives, checkpoints, or
logs into the Git repository.

## Authoritative Goals 1--5 coverage

The anonymous artifact tar
`GEML_artifacts_goals_1-5_2026-07-25.tar` is the authoritative consolidated store
for the finalized Goals 1--5 artifacts.  Its published size is 35,835,445,760
bytes and its SHA-256 is
`438b11726bd108b2fe971063d8dffbdd580c0f4ec7c42947047693f818290f3e`.
The extracted inventory contains 1,210,913 files totaling 34,810,631,623 bytes.
The compact ledger covers the numerical manifests, summaries, reports, and result
claims from this inventory; it does not duplicate the inventory itself.

The coverage includes:

* Goal 1 source-corpus manifest, QA report, split/domain/family/variable counts, and
  generator policy results.
* Goal 2 official-v4 pure-EML manifest and semantic-audit summary.
* Goal 3 AST/EML DAG summaries, cost ratios, and analysis tables.
* Goal 4 safe-real and positive-real-formal e-graph study summary.
* Goal 5 frequent and learned motif selection, neural ranker analysis, final export
  manifest, and final report/plot data.

See `GOALS_1_5.md` and `PROVENANCE.md` for the exact source paths and values.

## Authoritative Phase-B coverage

The anonymous Phase-B artifact package is the authoritative Phase-B package.  It
contains the final cache, results, runtime-provenance, and state archives, together
with `FINAL_VALIDATION.json`, its README, package manifest, packaging status log,
extraction/verification instructions, and `SHA256SUMS.txt`.  The exact archive and
manifest hashes are listed in `PROVENANCE.md`; the final validation hash is
`7d949c3b69d1baffa6903b1c23eb376e37607fddced345cc1f259d3b00e2e758`.

The old local Goal 6 archive for snapshot `ANON-G6-LEGACY-SNAPSHOT` is not uploaded as a separate
authoritative artifact.  It is scientifically superseded by, and represented inside,
the final results archive for runtime snapshot `ANON-PHASE-B-SNAPSHOT`.  The compact ledger therefore
does not count that old tar as missing coverage.

The obsolete retrieval variants `verbose_v1` and `target_size_bug` are intentionally
not authoritative and are not preserved as final deliverables.  Their absence is
not a data-loss claim: only the frozen, authenticated retrieval grid in the final
Phase-B package is used by `PHASE_B_GPU.md`.

## What is not implied

"Covered" means that the final store has the evidence needed to reproduce or audit
the reported number; it does not mean every intermediate byte, failed attempt, cache,
or temporary worktree is retained.  In particular, Goal 7's invalid cells and failed
attempt records are represented by counts and final validation metadata, while large
per-cell logs remain in the Phase-B results archive.  No local deletion is authorized
by this inventory.  Before deleting any local copy, verify the artifact package by
running its extraction/hash script and comparing all hashes in `PROVENANCE.md`.
