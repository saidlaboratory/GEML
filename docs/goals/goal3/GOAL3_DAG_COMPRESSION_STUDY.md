# Goal 3 DAG Compression Study

## Scope

This study measures exact structural sharing. Nodes are shared only when their
labels, values, ordered child slots, and recursively referenced structure are
identical. Repeated references and child order are preserved.

The evidence does not identify mathematical equivalence between different
structures. It makes no e-graph or motif claim and does not establish that one
expression is mathematically preferable to another.

## Metrics

For each successfully processed expression:

- `raw_tree_alpha = eml_tree_node_count / ast_tree_node_count`
- `dag_alpha_vs_ast_tree = eml_dag_node_count / ast_tree_node_count`
- `dag_alpha_vs_ast_dag = eml_dag_node_count / ast_dag_node_count`
- `ast_compression = ast_tree_node_count / ast_dag_node_count`
- `eml_compression = eml_tree_node_count / eml_dag_node_count`

“Compresses well” means `eml_compression` is large. “Becomes structurally
competitive” means `dag_alpha_vs_ast_tree` approaches or falls below one.
These predicates are deliberately reported independently.

Reuse depth is the minimum child-reference distance from the root. Sharing
concentration is the largest reused node's excess-reference count divided by
all excess references in that expression. Child-reference overhead is the sum
of `indegree - 1` across reused nodes.

## Corpus-wide result

All 250,000 processed expressions were valid for analysis; no processing
failure occurred.

| Metric | Arithmetic mean of per-expression ratios |
| --- | ---: |
| Raw tree alpha | 952.1371252900 |
| DAG alpha versus AST tree | 8.3344012718 |
| DAG alpha versus AST DAG | 10.4749538902 |
| AST compression | 1.3617076634 |
| EML compression | 39.3750077169 |

Exact EML sharing removes substantial repetition, but the resulting EML DAG
remains larger than the source AST tree for every expression. The most
competitive expression has exact remaining alpha `8/7`. The least competitive
has `172/3`.

The clearest counterexample to conflating the two claims is the expression
with the greatest EML compression: its exact compression is
`28087277/180` (about 156,040.43), while its EML DAG is still `540/13`
(about 41.54) times the size of its AST tree.

## Results by source family

| Family | Rows | Mean remaining alpha | Mean EML compression |
| --- | ---: | ---: | ---: |
| `algebraic_core` | 70,000 | 4.5536 | 5.9948 |
| `exp_log` | 40,000 | 5.7684 | 5.6667 |
| `mixed_elementary` | 35,000 | 13.5022 | 39.9935 |
| `ood_stress` | 25,000 | 5.4188 | 8.5150 |
| `powers_division_rationals` | 40,000 | 6.7068 | 6.9495 |
| `trig_hyperbolic` | 40,000 | 16.4449 | 182.6705 |

The trig/hyperbolic family achieves by far the greatest internal EML
compression and also leaves the largest mean EML-DAG-to-AST gap. This is
structural evidence that a very repetitive expansion can shrink dramatically
without becoming small relative to its source representation. Families differ
in operator composition and source complexity, so this table does not isolate
operator-level causality.

## Reuse structure

| Reuse measure | AST DAG | EML DAG |
| --- | ---: | ---: |
| Total reused nodes | 857,703 | 4,532,684 |
| Total references to reused nodes | 3,439,858 | 69,048,387 |
| Total excess references | 2,582,155 | 64,515,703 |
| Maximum reused-node indegree | 63 | 1,939 |
| Mean reused nodes per expression | 3.430812 | 18.130736 |
| Mean references to reused nodes per expression | 13.759432 | 276.193548 |
| Reused-node-weighted mean minimum root depth | 4.6422 | 27.7800 |
| Mean per-expression sharing concentration | 0.5699 | 0.7978 |

Both AST and EML reuse occurs in 222,830 expressions. A further 27,167 have
EML reuse but no AST reuse, and three have no reuse in either representation.
No expression has AST-only reuse. The EML-only group has mean remaining alpha
17.2996 and mean EML compression 124.2207, again separating compression from
competitiveness.

## Operator-signature stratification

The corpus contains 188,368 distinct operator signatures, so the complete
stratification is stored as deterministic canonical JSONL in
`operator-signature.strata.jsonl.gz` instead of as one large nested JSON
object. It contains all 250,000 rows across its groups and no failures.

The sidecar's compressed SHA-256 is
`0374c037a50b23d45c4491fd879ec6419f114fe4aced1a481b3b80db07ffcf26`;
its logical uncompressed-content SHA-256 is
`257ad765ab25deae68425e1bf57a783090ccd791a2a814de896c9820cc74e968`.
The public streaming reader validates the manifest binding, gzip integrity,
canonical encoding, group order and uniqueness, group invariants, and total
denominators while consuming the file.

## Scale and runtime

| Prefix | Raw alpha | Remaining alpha | EML compression | Throughput (rows/s) | Peak RSS (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 870.3787 | 8.6492 | 37.4157 | 191.72 | 1,103.2 |
| 50,000 | 962.8730 | 8.6202 | 40.3686 | 190.68 | 1,242.8 |
| 100,000 | 959.0146 | 8.6333 | 40.1252 | 186.95 | 1,268.5 |
| 250,000 | 952.1371 | 8.3344 | 39.3750 | 162.79 | 1,377.1 |

The 10k, 50k, and 100k points are training-corpus prefixes, while the 250k
point includes later splits and OOD data. The sampled metrics are reasonably
stable, with some drift from corpus composition. In particular, the
100k-to-250k change is -6.8775 for raw alpha, -0.2989 for remaining alpha, and
-0.7502 for EML compression. These deterministic prefixes are useful
engineering evidence, not independent samples, confidence intervals, proof of
convergence, or proof of OOD generalization.

Runtime telemetry follows the scopes stated in the Goal 3 summary. The final
processing time was 1,535.705 seconds; the complete invocation, including
excluded input work, took 1,568.032 seconds on one Windows AMD64 run with
eight workers. These measurements are engineering observations, not universal
benchmarks. Analysis itself was exercised twice over the full output and
produced the same fingerprint and byte-identical products; the monitored
analysis pass peaked at about 328.9 MiB RSS.

## Numerical and denominator caveats

Per-row ratios and ranking comparisons are exact reduced rationals. Aggregate
ratio means are deterministic compensated decimal approximations, accumulated
in authoritative row order at 80 working digits and reported at 50 digits
under a fully pinned context. A displayed group or overall mean is the
arithmetic mean of per-expression ratios, not a ratio of summed node counts.

Every overall, stratum, and checkpoint record retains:

- `all_processed_count`;
- `valid_count`;
- `failure_count`.

No failure is silently discarded. The final corpus happens to have zero
failures, but the analysis and interface preserve unsupported inputs,
validation failures, and computation failures as explicit records. Zero
failures demonstrates coverage of this pipeline run; it is not a universal
proof of compiler semantics or mathematical correctness.

## Reproduction

From the repository root:

```bash
python -m geml.experiments.goal3.run \
  --config configs/goal3_final.yaml \
  --stage final

python -m geml.analysis.goal3.metrics \
  --manifest outputs/final/goal3/final/manifest.json \
  --output-dir outputs/final/goal3_analysis/reproduction

python -m pytest tests/graph tests/dag
python -m pytest tests/experiments/test_goal3_audit.py
python -m pytest tests/experiments/test_goal3_smoke.py
python -m pytest tests/analysis/test_goal3_analysis.py
python -m pytest tests/interfaces/test_eml_dag_cost.py
```

The production manifest SHA-256 is
`279b1d016bf8ff3295cf183cee9929dd69315ef21fedf58f3d63bb74414b5000`.
The analysis fingerprint is
`149a7be3c23624c5719f3dded0e36ad0390f8b11644082f0481b53fe4f67168e`.
