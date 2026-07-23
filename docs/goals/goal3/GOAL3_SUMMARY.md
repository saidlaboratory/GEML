# Goal 3 Summary

Goal 3 implements exact structural DAG sharing for the source AST and for
strictly pure EML, audits the direct EML-DAG compiler against post-hoc
hash-consing, runs the audited path over the final 250,000-expression corpus,
and freezes the cost boundary that Goal 4 must use.

This report presents structural evidence only. It does not establish semantic
equivalence, domain validity beyond the upstream contracts, e-graph
effectiveness, or motif effectiveness.

## Delivered components

- A representation-neutral graph schema with ordered child slots, canonical
  signatures, validation, and structural statistics.
- Iterative exact hash-consing for source AST and pure-EML trees.
- A direct pure-EML DAG compiler that avoids constructing the expanded EML
  tree.
- A 23-case audit showing exact agreement between direct construction and
  post-hoc construction for every enabled, approved operator.
- A resumable, immutable, sharded Goal 3 runner with retained failures and
  process-tree runtime and memory telemetry.
- Reproducible stratified analysis by family, operator signature, source size,
  source depth, split, domain, and reuse pattern.
- The read-only `compute_eml_dag_cost()` interface for later extraction
  stages.

## Production run

The final run used eight workers, `official_v4`, and `direct_hashcons` on the
authoritative Goal 1 corpus. No other compiler mode contributes to the
reported production metrics:

| Item | Result |
| --- | ---: |
| Processed expressions | 250,000 |
| Successful expressions | 250,000 |
| Retained failures | 0 |
| Metric shards | 10 |
| Processing wall time | 1,535.705 s |
| Throughput | 162.792 rows/s |
| Peak runner-plus-workers RSS | 1,444,044,800 bytes |

The processing time excludes input reads and includes direct row processing,
Parquet construction, and process-tree RSS sampling. Peak RSS is the maximum
sampled simultaneous resident memory of the runner and its live worker
descendants; it is not a whole-machine measurement.

The run manifest SHA-256 is
`279b1d016bf8ff3295cf183cee9929dd69315ef21fedf58f3d63bb74414b5000`.
The direct-versus-post-hoc audit fingerprint is
`1af4b4efadb880af9f068232626b78994ae4129fc88098d413d545e41415cf86`.

## Structural verdict

The corpus-wide arithmetic means are:

| Metric | Mean |
| --- | ---: |
| Raw EML tree / AST tree | 952.1371252900 |
| EML DAG / AST tree | 8.3344012718 |
| EML DAG / AST DAG | 10.4749538902 |
| AST tree / AST DAG | 1.3617076634 |
| EML tree / EML DAG | 39.3750077169 |

Exact sharing therefore compresses the expanded EML representation strongly,
but it does not make pure EML structurally competitive with the source AST on
this corpus. None of the 250,000 expressions has
`eml_dag_node_count / ast_tree_node_count <= 1`. The best individual remaining
ratio is the exact fraction `8/7`; the largest individual EML compression is
`28087277/180`, but that same expression still has a remaining ratio of
`540/13`. “Compresses well” and “becomes structurally competitive” are
different claims.

## Denominator and numerical policy

Every group and checkpoint reports all-processed, valid, and failure counts.
All five overall means above use 250,000 valid rows out of 250,000 processed
rows. No failed row is silently removed from a denominator.

Each per-expression ratio is derived from exact integer counts and retained as
a reduced rational. Aggregate ratio means use deterministic compensated
`Decimal` accumulation in authoritative corpus order, with an explicitly
pinned 80-digit working context and 50 reported digits. Aggregate means are
therefore labeled approximate; exact per-row values and ranking comparisons
remain rational. These means are means of per-expression ratios, not ratios of
corpus-wide count totals.

The 10k, 50k, and 100k points are deterministic training-corpus prefixes; the
250k point also includes later splits and OOD data. They are not independent
samples. They show reasonable stability with some composition-driven drift;
they are not confidence intervals and do not prove statistical convergence or
OOD generalization.

## Frozen Goal 4 cost boundary

Goal 4 must call:

```python
from geml.interfaces.eml_dag_cost import compute_eml_dag_cost
```

The function accepts either a validated `ASTTree` or a strictly pure
`EMLTerm`. A successful result contains:

- exact DAG node count, child-reference count, and depth;
- a lowercase SHA-256 signature of exact ordered structure;
- input kind, representation mode, and construction path;
- compiler mode when and only when compiler provenance is available.

For a source AST, `official_v4` is the default and direct hash-consing is used.
`clean_negation` is available only by explicit request and remains separately
labeled. Results from different compiler or representation modes must not be
mixed. A supplied EML tree is shared post hoc and cannot claim compiler
provenance.

The primary extraction cost is exact EML-DAG node count. If a caller needs a
deterministic tie-break between equal primary costs within the same
representation and compiler mode, it may compare `root_signature`
lexicographically. Child-reference count and depth are descriptive metadata,
not implicit secondary scientific objectives.

Non-success results retain a status, failure stage, exception type, and
message, and expose no partial cost:

- `invalid_input`: invalid type, invalid compiler-mode use, or rejected input;
- `unsupported`: a source operator has no approved direct compiler path;
- `failure`: compilation, DAG validation, or exact cost computation failed.

Callers own outer timeouts and candidate limits. The interface owns only exact
compilation, validation, counting, and signing. The interface imports no Goal
3 experiment or analysis module.

## Reproduction and validation

Run from the repository root:

```bash
python -m geml.experiments.goal3.run \
  --config configs/goal3_final.yaml \
  --stage final

python -m geml.analysis.goal3.metrics \
  --manifest outputs/final/goal3/final/manifest.json \
  --output-dir outputs/final/goal3_analysis/reproduction

python -m pytest tests/interfaces/test_eml_dag_cost.py
python -m pytest
python -m ruff check .
python -m ruff format . --check
```

The canonical saved analysis fingerprint is
`149a7be3c23624c5719f3dded0e36ad0390f8b11644082f0481b53fe4f67168e`.
Two independent analysis passes produced that same fingerprint and identical
product hashes.
