# Goal 2 summary: raw pure-EML expansion

## Result

Goal 2 measured the exact expanded-tree cost of compiling the validated 250,000-expression Goal 1 corpus with `CompilerMode.OFFICIAL_V4`. Every input row produced a successful exact count and a valid tree-alpha value. The primary structural result is unfavorable to raw expansion: median alpha was **40.6602**, mean alpha was **952.1371**, p90 was **385.0723**, p99 was **10,448.5978**, and the maximum was **6,481,679.3077**.

The result is structural only. It does not measure model accuracy, reasoning quality, parameter efficiency, or the effect of any future DAG, macro, or motif representation.

## Corpus and denominators

| Population | Count |
| --- | ---: |
| All processed | 250,000 |
| Exact count successes | 250,000 |
| Exact count failures | 0 |
| Valid alpha rows | 250,000 |
| Deterministically selected for semantic work | 280 |
| Materialized and semantically audited | 273 |
| Semantic-valid audited rows | 195 |
| Selected but over the 25,000-node materialization limit | 7 |

The split counts were 175,000 train, 25,000 validation, 25,000 IID test, and 25,000 OOD test. Structural statistics use all 250,000 valid-alpha rows. Semantic results use only the explicitly sampled audit population and are not extrapolated to unaudited rows.

## Expansion distribution

| Statistic | Tree alpha | AST nodes | EML nodes | EML depth |
| --- | ---: | ---: | ---: | ---: |
| Mean | 952.1371 | 40.7917 | 21,736.2364 | 75.7774 |
| Median | 40.6602 | 29 | 1,501 | 71 |

Tree alpha had standard deviation 23,155.5594, minimum 1.5, p10 21.0338, p25 26.3800, p75 75.2550, p95 1,187.8240, and maximum 6,481,679.3077. The very large mean-to-median gap records a heavy right tail rather than a typical 952-fold expansion.

## Family results

| Source family | Rows | Median alpha | p90 alpha | Maximum alpha |
| --- | ---: | ---: | ---: | ---: |
| algebraic core | 70,000 | 25.5778 | 33.3448 | 88.3333 |
| exp/log | 40,000 | 32.7872 | 58.7143 | 128.8077 |
| powers/division/rationals | 40,000 | 44.1579 | 67.4111 | 170.3333 |
| OOD stress | 25,000 | 45.7377 | 70.4144 | 120.8692 |
| mixed elementary | 35,000 | 133.9289 | 987.5028 | 849,331.1364 |
| trig/hyperbolic | 40,000 | 270.4602 | 4,566.1510 | 6,481,679.3077 |

Trig/hyperbolic expressions dominate the high-expansion tail; mixed elementary expressions are the second-largest family by median and p90. The largest case is expression `54fa4d5bbf529ecdcfc6ba55bd6f65f380efd12d572dc0d22e157766216ca098`, with 13 AST nodes, 84,261,831 exact EML nodes, and alpha 6,481,679.3077.

## Named threshold scenarios

Pass means the strict inequality `alpha_tree < 1 + ln(K) / ln(4L)`. No scenario had a strict pass.

| Scenario | Family | K | L | Threshold | Passes / applicable |
| --- | --- | ---: | ---: | ---: | ---: |
| `algebraic_core_bounded_v1` | algebraic core | 4 | 25 | 1.301030 | 0 / 70,000 |
| `power_division_rational_bounded_v1` | powers/division/rationals | 6 | 117 | 1.291416 | 0 / 40,000 |
| `exp_log_bounded_v1` | exp/log | 8 | 117 | 1.338205 | 0 / 40,000 |
| `trig_hyperbolic_bounded_v1` | trig/hyperbolic | 10 | 25 | 1.500000 | 0 / 40,000 |
| `mixed_elementary_bounded_v1` | mixed elementary | 14 | 117 | 1.429222 | 0 / 35,000 |
| `ood_stress_bounded_v1` | OOD stress | 8 | 117 | 1.338205 | 0 / 25,000 |

All six scenarios have explicit bounded vocabularies; none is undefined. Both valid-only and all-processed pass rates are zero. These scenarios are descriptive combinatorial references, not empirical performance laws.

## Semantic audit and retained issues

The audit used two deterministic domain-aware probes with both 80-digit `mpmath` and NumPy `complex128`. It retained 1,092 backend-by-probe results for the 273 materialized rows. Row statuses were 195 passed, 53 semantic nonfinite, 22 semantic overflow, and 3 semantic mismatch; seven additional selected rows exceeded the materialization node limit. No count or alpha failure occurred, so failure survivorship excluded zero rows from the structural distribution.

The saved backend/status taxonomy reports probe rates within each backend plus expression-incidence rates against the audited, selected, and materialized populations. It also retains unique-assignment totals and per-selected/materialized coverage; deterministic variable-bearing probes are collision-free, while constant-only expressions necessarily reuse their sole empty assignment vector. Its all-processed incidence is explicitly selection-diluted and is not a corpus-wide semantic failure rate.

The three mismatch rows are retained for scientific review. Their structural alpha values are approximately 43.10, 139.31, and 311.68 (2,543 to 8,219 EML nodes), so the evidence points to extreme intermediate numeric magnitude, cancellation, and backend range/precision behavior rather than exceptional tree expansion. They were not relabeled, resampled, or omitted.

## Pilot stability

The comparison uses the separately generated Goal 1 pilot run-a, not a subset of the final corpus, but it is not statistically independent: 3,180/10,000 pilot expression IDs (31.8%) also occur in final. All 2,800 pilot algebraic-core rows overlap; the other shared counts are 132 exp/log, 113 trig/hyperbolic, 70 mixed elementary, 65 powers/division/rationals, and 0 OOD stress. Every shared ID keeps the same split and family, and 2,800 share the generator seed. From pilot to final, alpha mean changed by +206.2353, median by +0.4876, p90 by +6.7967, and p95 by +94.8828. The selected-row terminal-issue rate was 29/92 (31.52%) versus 85/280 (30.36%); the materialized semantic-nonpass rate was 23/86 (26.74%) versus 78/273 (28.57%). All six family median rankings were unchanged (rank correlation 1.0), but that stability is partly mechanically coupled by the structured overlap and the tail-sensitive statistics remain sample-sensitive.

## Runtime and artifacts

The final eight-worker run completed in 2,348.0201 seconds at 106.4727 rows/second, with peak process-tree resident memory of 1,567,707,136 bytes. It is provisional because it was executed from the required unstaged dirty implementation tree. The archival run should be regenerated from the reviewed clean commit.

Primary artifacts:

- `outputs/final/goal2/final/manifest.json` — SHA-256 `22940d0afd75fae908bd6834816ba5c3729127733c491052c468ee6c3f061466`
- `outputs/final/goal2/analysis/manifest.json` — SHA-256 `efbfdee7c30b135493dd4548776124cd77f074a12724265526596fc8743fcdb5`
- `outputs/final/goal2/analysis/plots/manifest.json` — SHA-256 `c4fb1024024eba6c81fa6372b2d11f5a5dc7cea873c3ed49f1cb45cac471e609`

Reproduction commands:

```text
python -m geml.experiments.goal2.run --config configs/goal2_final.yaml --stage smoke
python -m geml.experiments.goal2.run --config configs/goal2_final.yaml --stage pilot
python -m geml.experiments.goal2.run --config configs/goal2_final.yaml --stage final
python -m geml.analysis.goal2.stratified --metrics-manifest outputs/final/goal2/final/manifest.json --pilot-manifest outputs/final/goal2/pilot/manifest.json --config configs/goal2_final.yaml --output-dir outputs/final/goal2/analysis
python -m geml.plots.goal2 --analysis-manifest outputs/final/goal2/analysis/manifest.json
```
