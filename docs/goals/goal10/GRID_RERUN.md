# Goal 10 compatibility matrix

Despite the historical `rerun_grids.py` filename, **no Goal 6 or Goal 7
learning grid is run here**. The module exercises four tiny
grammar-version/compiler-mode cells and verifies that selected immutable Goals
1–5 text files still match their coordinated SHA-256 values. These hashes use
Git-compatible LF-normalized bytes, so the same repository content authenticates
identically on Windows and Linux checkouts.

The omitted experiment option resolves to `v1` with `official_v4`. Grammar v2
must be requested explicitly and every resulting row is labeled
`pure_eml:grammar=v2:compiler=<mode>`. Mixed grammar versions or compiler modes
cannot enter an aggregate unless the caller first groups them under explicit
version/mode keys.

The following questions remain deferred:

- effects of grammar v2 on corpus statistics or pure-EML alpha;
- effects on AST, EML, DAG, macro, or motif representations;
- effects on Goal 6 equivalence learning;
- effects on Goal 7 rewrite-policy learning;
- production corpus-v2 or learned-effect claims.

Goals 1–5 and all currently reported Goal 6/7 results are v1-only. This
compatibility check generates no corpus, graph, motif, checkpoint, or production
artifact and trains no model.

The current Phase-A check exercises the owned compatibility-row boundary only.
The Goal 6/7 producer interfaces are being implemented in separate workstreams,
so an end-to-end downstream-consumer adapter remains a merge-time acceptance
blocker rather than a completed claim.
