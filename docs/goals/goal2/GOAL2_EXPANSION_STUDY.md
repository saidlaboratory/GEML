# Goal 2 pure-EML expansion study

## 1. Question and scope

This study asks how much a deterministic source-expression tree grows when compiled into the exact, fully expanded, official-v4 pure EML tree. It covers the validated 250,000-expression Goal 1 corpus and reports structural counts, stratified raw expansion, bounded semantic-audit outcomes, failure survivorship, and pilot-to-final stability.

The study does not evaluate a learned model. It makes no claim about prediction accuracy, reasoning, theorem proving, parameter efficiency, or later compressed representations. No DAG, macro, motif, or e-graph statistic is called pure-EML alpha.

## 2. Structural definitions

For each authoritative source expression:

- `T_AST` is its validated deterministic source AST, preserving ordered child slots.
- `T_EML` is the fully expanded pure EML tree produced by exact official-v4 count-only compilation.
- `|T|` is exact expanded-tree node count.
- `alpha_tree = |T_EML| / |T_AST|`.

The metric row stores `eml_node_count` as the exact numerator, `ast_node_count` as the exact positive denominator, the stable exact-ratio text, and a derived finite float for analysis. A missing, invalid, or nonpositive AST denominator would produce an explicit failed row rather than a substituted denominator. No such row occurred in the final corpus.

## 3. Compiler and source provenance

The only primary compiler mode is `CompilerMode.OFFICIAL_V4`. The runner consumes authoritative `sympy_srepr`, parses it through the frozen s-expression parser and AST contracts, validates the AST, and dispatches its ordered structure to the frozen Bundle I count-only constructors. Supported dispatch covers one, variable, integer, rational, add, subtract, multiply, divide, negate, power, exp, log, sin, cos, tan, sinh, cosh, and tanh. Unsupported labels fail explicitly; no display string, LaTeX, `sympy.simplify`, or materialized counting path is used.

The input corpus manifest SHA-256 is `77fce5779b3d2c2f3cdf2b9f49da54cd14474d37ab128337bdf4fcc52afd4f0d`. The run records repository commit `baf0b1c3a746bfcedb29222a495d9d05d599620f`, config hash `878bdfeeac2b8b0936b19d223b6eeecb6e7c2321cc4dcc5e443522d73aa57fb7`, and runner fingerprint `3c57ac753442149e2c70cd94085b6b0f97d90d64d252782cefe3c93edf580c5e`.

## 4. Corpus and splits

The final Goal 1 corpus passed its manifest, shard checksum, uniqueness, split, operator-policy, and QA gates before Goal 2 ran. It contains 250,000 unique expression IDs and 250,000 unique authoritative `sympy_srepr` values:

| Split | Rows |
| --- | ---: |
| train | 175,000 |
| validation | 25,000 |
| test IID | 25,000 |
| test OOD | 25,000 |

The six source families contain 70,000 algebraic-core, 40,000 powers/division/rational, 40,000 exp/log, 40,000 trig/hyperbolic, 35,000 mixed-elementary, and 25,000 OOD-stress rows. Pending or unsupported source operators are absent.

## 5. Counting and materialization policy

Exact count-only compilation ran on every row. It did not materialize EML to obtain counts. The primary schema records source metadata; AST nodes, edges, leaves, operators, and depth; exact EML counterparts; compiler-operation counts; alpha components; threshold outcomes; timing; and explicit processing, materialization, semantic, and error statuses.

Materialization was restricted to deterministically selected semantic-audit rows whose exact preflight did not exceed 25,000 nodes, depth 256, or 100,000 construction steps. The final study materialized 273 rows; seven selected rows exceeded the node limit. Materialized trees were validated as pure EML and checked against count-only statistics.

## 6. Semantic audit and denominators

Structural and semantic denominators are intentionally separate:

| Denominator | Count |
| --- | ---: |
| all processed | 250,000 |
| exact count successes | 250,000 |
| exact count failures | 0 |
| deterministically selected for semantic work | 280 |
| semantic audited/materialized | 273 |
| semantic valid | 195 |

Semantic selection is a deterministic, structure-keyed SHA-256 hash sample over expression identity and family, split, size, and depth metadata, with a common modulus of 1,000 for the final run; it is not quota-controlled stratified sampling. Each materialized row requested two deterministic domain-aware assignments and evaluated them through 80-digit `mpmath` and NumPy `complex128`. Positive-real assignments were strictly positive; nonzero-real assignments avoided zero and could be negative; safe-real assignments could be negative. Domain, pole, range, nonfinite, overflow, mismatch, and backend outcomes were retained rather than resampled away.

All 273 materialized rows contain exactly four detailed backend-by-probe records, for 1,092 retained results. The row-level semantic counts are 195 passed, 53 nonfinite, 22 overflow, and 3 mismatch. The remaining 249,720 rows are explicitly `semantic_not_selected`; seven selected rows are `not_materialized_node_limit`. Consequently, semantic evidence is a bounded audit and must not be described as validation of all 250,000 expressions.

The saved `semantic_backend_status_counts` table exhaustively crosses each configured backend with every probe status. It reports status rates within each backend and distinct-expression incidence against backend-audited, selected, and materialized populations, together with unique-assignment totals and per-selected/materialized coverage. Deterministic variable-bearing probes are collision-free; constant-only expressions necessarily reuse their sole empty assignment vector. An all-processed incidence is retained for completeness but explicitly labeled as selection-diluted; it is not a corpus-wide semantic failure rate.

## 7. Threshold theory and derivations

Each named family scenario uses:

`alpha_threshold = 1 + log_(4L)(K) = 1 + ln(K) / ln(4L)`.

`K` is the declared source operator-label count and `L` is the declared bounded leaf-label count. The full leaf vocabulary, where enabled, has six variable names, 19 integers from -9 through 9, and 92 distinct reduced nonintegral rationals. Pass is the strict, unrounded comparison `alpha_tree < alpha_threshold`.

| Scenario | Family derivation | K | L | Threshold |
| --- | --- | ---: | ---: | ---: |
| `algebraic_core_bounded_v1` | add, subtract, multiply, negate; variables + integers | 4 | 25 | 1.301029996 |
| `power_division_rational_bounded_v1` | preceding algebraic labels + divide + power; full leaves | 6 | 117 | 1.291415582 |
| `exp_log_bounded_v1` | power/division/algebraic labels + exp + log; full leaves | 8 | 117 | 1.338204808 |
| `trig_hyperbolic_bounded_v1` | four algebraic labels + six trig/hyperbolic labels; no rationals | 10 | 25 | 1.500000000 |
| `mixed_elementary_bounded_v1` | four algebraic + divide, power, exp, log + six trig/hyperbolic; full leaves | 14 | 117 | 1.429221914 |
| `ood_stress_bounded_v1` | add, subtract, multiply, divide, negate, power, exp, log; full leaves | 8 | 117 | 1.338204808 |

All scenarios are defined; none required an invented or observed-data leaf count. Thresholds are descriptive combinatorial references, not empirical laws.

## 8. Overall distribution

All 250,000 rows had a valid alpha, so all-processed and valid-alpha structural summaries coincide.

| Statistic | Alpha |
| --- | ---: |
| mean | 952.137125 |
| median | 40.660189 |
| population standard deviation (`ddof=0`) | 23,155.559427 |
| minimum | 1.500000 |
| p10 | 21.033842 |
| p25 | 26.380039 |
| p75 | 75.255038 |
| p90 | 385.072286 |
| p95 | 1,187.824000 |
| p99 | 10,448.597826 |
| maximum | 6,481,679.307692 |

Quantiles use the documented linear method. Mean and median AST nodes were 40.791744 and 29. Mean and median EML nodes were 21,736.236440 and 1,501. Mean and median EML depth were 75.777420 and 71. The mean, high quantiles, and maximum demonstrate an extreme right tail.

Threshold outcomes use family-specific denominators. All six valid-only pass counts are zero: 0/70,000 algebraic core, 0/40,000 powers/division/rationals, 0/40,000 exp/log, 0/40,000 trig/hyperbolic, 0/35,000 mixed elementary, and 0/25,000 OOD stress. Because alpha was valid for every row, all-processed pass rates are also zero within each family.

## 9. Stratified results

The saved stratified table contains every required dimension: AST-size bucket, AST depth, source family, stable logical-generator operator signature, variable count, canonical-srepr AST constant category/count, domain, split, IID/OOD, count status, semantic status, and the family-size, family-depth, family-domain, and split-family cross-strata. It contains 188,680 final groups; groups below the configured minimum count of 30 remain present with `underpowered=true`.

The compatibility metric fields are named `source_constant_counts_json` and `source_constant_count`, but their values count constant leaves in the canonical-srepr AST after SymPy lowering. They are therefore labeled `canonical_ast_constant_category` and `canonical_ast_constant_count` in analysis and must not be conflated with logical generator metadata. Lowering adds integer leaves relative to the logical construction metadata in 229,483/250,000 rows, for 1,577,150 additional integer-leaf occurrences in aggregate; operator signatures and operator counts remain logical-generator metadata.

### Family

| Family | Rows | Median alpha | p90 | Mean EML nodes |
| --- | ---: | ---: | ---: | ---: |
| algebraic core | 70,000 | 25.5778 | 33.3448 | 1,105.99 |
| exp/log | 40,000 | 32.7872 | 58.7143 | 1,353.87 |
| powers/division/rationals | 40,000 | 44.1579 | 67.4111 | 1,969.27 |
| OOD stress | 25,000 | 45.7377 | 70.4144 | 4,251.65 |
| mixed elementary | 35,000 | 133.9289 | 987.5028 | 20,086.72 |
| trig/hyperbolic | 40,000 | 270.4602 | 4,566.1510 | 110,359.70 |

Trig/hyperbolic and mixed-elementary structure dominates expansion. Algebraic-core structure is the smallest family by median and tail quantiles, but even its minimum alpha is 9 and it has no threshold pass.

### AST size

Median alpha by configured bucket was 110.1429 for 1-8 nodes, 38.6923 for 9-16, 33.8800 for 17-32, 38.8298 for 33-64, 46.8676 for 65-128, and 50.4473 for 10,810 rows outside the configured 1-128 policy range. The nonmonotonic pattern shows that source size alone does not explain raw expansion; operator composition is a major confounder.

### Domain and IID/OOD

Median alpha was 46.0909 for nonzero-real rows (68,345), 38.4884 for positive-real rows (84,843), and 39.6988 for safe-real rows (96,812). The 225,000 IID rows had median 39.4237 and a very heavy maximum of 6,481,679.3077. The 25,000 OOD rows had median 45.7377 and maximum 120.8692; that narrower range reflects the OOD-stress family definition and must not be read as a general claim that OOD inputs expand less.

Operator-signature and cross-stratum results are retained in the table rather than summarized selectively. Of 188,368 final operator-signature groups, 188,075 have fewer than 30 rows and are explicitly flagged underpowered.

## 10. Scaling results

The saved scaling table has 446 rows describing:

- EML nodes versus AST nodes;
- alpha versus AST nodes;
- alpha versus AST depth;
- EML depth versus AST depth;
- compiler-operation counts versus AST size;
- runtime versus AST and EML size.

The scatter source is a deterministic 20,000-row sample. Exact counts, not materialized trees, drive the structural scaling tables. The relation is strongly heterogeneous: trig/hyperbolic and mixed expressions occupy the largest EML and alpha ranges at modest AST sizes, while algebraic, exp/log, power, and OOD-stress families form much tighter bands. No fitted trend or model-performance scaling law is claimed.

## 11. Top expansions

The highest-alpha, largest-node, and largest-operation case is expression `54fa4d5bbf529ecdcfc6ba55bd6f65f380efd12d572dc0d22e157766216ca098` from the trig/hyperbolic family:

| Metric | Value |
| --- | ---: |
| AST nodes / depth | 13 / 12 |
| EML nodes / depth | 84,261,831 / 342 |
| compiler operations | 84,486,577 |
| alpha | 6,481,679.307692 |

The deepest EML case is expression `72afa39d38eac52ad5ad7cbf7053ece0caf3277d0818c4028d3939b367611464`, at depth 381 with 33,378,349 EML nodes and alpha 2,086,146.8125. The largest successful semantic-audit tree has 24,113 EML nodes and passed. The slowest count-successful, semantic-passed row took 2.003267 seconds. Six independently ranked top-100 lists are saved with deterministic expression-ID tie breaking and authoritative `sympy_srepr` audit context.

## 12. Failure and survivorship analysis

There were no parsing, AST validation, unsupported-operator, compiler, count, or alpha failures. The failure tables retain 85 selected-row terminal materialization/semantic issues:

| Status | Rows |
| --- | ---: |
| materialization node limit | 7 |
| semantic nonfinite | 53 |
| semantic overflow | 22 |
| semantic mismatch | 3 |

By family these 85 rows comprise 27 algebraic-core, 9 exp/log, 6 mixed-elementary, 21 OOD-stress, 4 powers/division/rational, and 18 trig/hyperbolic rows. These are counts within a deterministically sampled semantic workflow, not corpus-wide semantic failure-rate estimates. Comparable summaries use selected rows (29/92 pilot versus 85/280 final) or materialized rows with a nonpassing semantic status (23/86 versus 78/273); the all-processed incidence is only a sampling-diluted descriptive quantity because the pilot and final selection moduli differ.

All 250,000 rows retained valid structural alpha, including all 85 terminal-issue rows. Therefore valid-alpha survivorship excludes zero rows and does not suppress high-expansion rows from the structural summaries. The companion all-processed denominator is still reported throughout.

The three mismatches remain open scientific-review cases:

1. `0cad3e801650e041e5826dad317ca2881e64f7f1902aafece1a0a03138e73bfa` (trig/hyperbolic): source values were zero at both probes, while official-v4 EML produced extreme `mpmath` values and NumPy overflow.
2. `cbbbf1e6962397e9b8e649a48bce7f314b834d77eb1654a78cc1575beb1a6cb3` (mixed elementary): `mpmath` hit a bounded “too many digits in integer” range condition, while NumPy values differed by about 19.4% relative error at both probes.
3. `0ddfd77a80d7bdc272c2c089f50f595eb5339bd564c231be0ec7931df3d76040` (OOD stress): high-precision probes passed with extended intermediates, while NumPy produced one mismatch and one overflow.

These rows were not silently dropped, resampled, or reclassified as successes.

## 13. Pilot-to-final stability

The pilot is the separately generated Goal 1 pilot run-a, not a deterministic subset of the final corpus. Both artifacts validate and use official-v4 metrics, but they are not statistically independent: 3,180/10,000 pilot expression IDs (31.8%) also occur in final. The overlap is strongly family-structured: all 2,800 pilot algebraic-core rows overlap, followed by 132 exp/log, 113 trig/hyperbolic, 70 mixed elementary, 65 powers/division/rationals, and no OOD-stress rows. All shared IDs retain the same split and family, and 2,800 share the generator seed.

| Metric | Pilot 10k | Final 250k | Final - pilot |
| --- | ---: | ---: | ---: |
| mean alpha | 745.901811 | 952.137125 | +206.235314 |
| median alpha | 40.172598 | 40.660189 | +0.487591 |
| p90 alpha | 378.275630 | 385.072286 | +6.796655 |
| p95 alpha | 1,092.941223 | 1,187.824000 | +94.882777 |
| selected-row terminal-issue rate | 0.315217 | 0.303571 | -0.011646 |
| materialized semantic-nonpass rate | 0.267442 | 0.285714 | +0.018272 |

All six family median ranks are unchanged, with rank correlation 1.0. Family median changes range from +0.1567 for exp/log to +8.6432 for mixed elementary; trig/hyperbolic p90 increased by 598.6218. The largest AST-size-bucket share change is -3.3916 percentage points for the 1-8 bucket. Central tendency and family order are descriptively stable, but the structured ID overlap mechanically couples the comparison and tail-sensitive statistics remain meaningfully sample-dependent.

## 14. Runtime, memory, and throughput

The final run used eight worker processes and ten 25,000-row atomic metric shards. It completed in 2,348.020064 seconds (39.13 minutes), at 106.472685 rows/second. Peak process-tree resident memory was 1,567,707,136 bytes (about 1.46 GiB). It processed 250,000 rows successfully and wrote zero primary count-failure rows.

The separately generated 10k pilot completed in 87.502431 seconds at 114.282539 rows/second, with peak resident memory 880,295,936 bytes. Its manifest SHA-256 is `80f96f420486948a7846bc2e3fde4d026a813e66dcb24a1afe75703052b7fe2f`.

## 15. Limitations

- Semantic verification is sampled: only 273 materialized rows were audited, and only 195 received a passing row status.
- Materialization is bounded, so seven selected rows above 25,000 nodes were not evaluated.
- Two probes per row cannot establish functional equivalence over an entire domain.
- NumPy `complex128` and even high-precision evaluation can encounter severe range and conditioning problems in large official-v4 expansions.
- The theoretical thresholds depend on declared bounded grammar vocabularies and are descriptive scenarios, not universal laws.
- Operator signatures are often unique; most signature groups are underpowered.
- This run was performed on a dirty, unstaged implementation tree as required by the bundle. Its results are provisional until regenerated from the reviewed clean commit.
- Raw tree expansion does not predict downstream model behavior or the benefit of future structural sharing.

## 16. Reproduction

Run from the repository root after validating the Goal 1 artifacts:

```text
python -m geml.experiments.goal2.run --config configs/goal2_final.yaml --stage smoke
python -m geml.experiments.goal2.run --config configs/goal2_final.yaml --stage pilot
python -m geml.experiments.goal2.run --config configs/goal2_final.yaml --stage final
python -m geml.analysis.goal2.stratified --metrics-manifest outputs/final/goal2/final/manifest.json --pilot-manifest outputs/final/goal2/pilot/manifest.json --config configs/goal2_final.yaml --output-dir outputs/final/goal2/analysis
python -m geml.plots.goal2 --analysis-manifest outputs/final/goal2/analysis/manifest.json
```

The runner validates input manifests and checksums before processing. Checkpoints and output shards are atomic; a valid shard is checksum-validated before resume. Analysis validates the final and pilot metric manifests and never modifies them. Plots read only saved analysis tables.

## 17. Artifact paths and checksums

| Artifact | SHA-256 |
| --- | --- |
| Goal 1 final input manifest | `77fce5779b3d2c2f3cdf2b9f49da54cd14474d37ab128337bdf4fcc52afd4f0d` |
| Goal 2 smoke manifest | `b21e1a093e6484f44e2ab412cc88ad05c3eb8e14e7d8ee8f4bd5f7af16b748fd` |
| Goal 2 pilot manifest | `80f96f420486948a7846bc2e3fde4d026a813e66dcb24a1afe75703052b7fe2f` |
| Goal 2 final metrics manifest | `22940d0afd75fae908bd6834816ba5c3729127733c491052c468ee6c3f061466` |
| Goal 2 analysis manifest | `efbfdee7c30b135493dd4548776124cd77f074a12724265526596fc8743fcdb5` |
| Goal 2 plot manifest | `c4fb1024024eba6c81fa6372b2d11f5a5dc7cea873c3ed49f1cb45cac471e609` |

The metrics live under `outputs/final/goal2/{smoke,pilot,final}/`; the 16 saved analysis tables live under `outputs/final/goal2/analysis/tables/`; and 10 PNG plus 10 SVG plots live under `outputs/final/goal2/analysis/plots/`. Every listed manifest and subordinate artifact passed its checksum validator after generation.

## 18. Handoff to Goal 3

Raw official-v4 pure EML is structurally expensive on this corpus. The median 40.66-fold expansion is already substantial, and the trig/hyperbolic and mixed-elementary tails reach millions-fold expansion. Goal 3 should treat the exact per-row counts, deterministic top cases, family/size cross-strata, and retained semantic issues as the baseline against which any explicitly different shared, compressed, or learned representation is compared.

Any later comparison must preserve representation labels and denominators. A DAG, macro, motif, or other shared representation may be scientifically useful, but its size ratio is not this study's raw pure-EML tree alpha.
