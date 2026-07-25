# Goal 5 compression study

**Evidence status:** `complete`

> This document is generated only from the normalized integration evidence. It does not infer missing producer results.

## Denominators and metrics

- Every graph row reports successes and retained failures over all attempted inputs.
- Node count, edge count, and MDL each declare availability, exact observed/missing denominators, and authenticated reasons when unavailable.
- Standalone graph MDL is not pooled with dictionary-inclusive motif MDL.
- Structural validation, reconstruction, and expansion separately declare available, unavailable, or not-applicable state and exact pass/fail counts.
- Runtime and memory separately declare availability and observed/missing counts.
- Neural exact-best match, validation rate, regret, and cost-scoring runtime use candidate-group denominators; failed selected candidates remain present.
- Goal 4 e-graph all-processed rows and issue 5-8 full-corpus rows retain their different exact denominators; cross-track subset comparisons use exact ID joins.

## Production export coverage

- 2500 immutable batches cover 250000 expressions, 1250000 graphs, and 250000 hierarchy records.
- Validation failures: 0; reconstruction failures: 0.
- 5-8 subset labels are unavailable: issue 5-8 freezes explicit-only-default-empty subset labels; Goal 4 nontrivial membership is joined externally by exact expression ID
- 5-8 runtime is unavailable: the issue 5-8 production completion and batch schemas do not publish per-graph or run runtime observations
- 5-8 memory is unavailable: the issue 5-8 production completion and batch schemas do not publish peak-memory observations

## Subset definitions

- `all` (`all`, IDs `b9f10864c0e4b7eb977043866f4990f6046e43685790a050454e62e13f7585b3`): Every expression in the authenticated issue 5-8 full-corpus export; track-specific all-processed denominators remain explicit.; all authenticated production corpus expressions. Sources: `issue_5_8_production_export`
- `safe_nontrivial` (`safe_real`, IDs `7f3885d04385f9a586a7ae94661e1897bb0e15018b3669d217cf85d2c3ed369f`): Exact Goal 4 expression IDs whose safe_real row records rewrites_applied > 0.; branch_insensitive_finite_real. Sources: `goal4_rows`
- `domain_nontrivial` (`positive_real_formal`, IDs `974be40e2469d78787118af6e63525bf6b40c66050e8e8db79172f39b1afd99d`): Exact Goal 4 expression IDs whose positive_real_formal row records rewrites_applied > 0.; conditional_positive_real_formal_under_recorded_assumptions. Sources: `goal4_rows`

## Exact cross-track cohort joins

| Split | Subset | Expressions | ID digest | Tracks | Sources |
|---|---|---:|---|---:|---|
| `train` | `safe_nontrivial` | 6367 | `efe60b31bdb8d35b25eb4a282b6eea47719eca753c6ef7022132b1d1d498c865` | 7 | `goal4_rows`, `issue_5_8_production_export` |
| `train` | `domain_nontrivial` | 6386 | `159c303a969f15e79530671fa727850aa58f07a4af10c601436868aa8d26d4d5` | 7 | `goal4_rows`, `issue_5_8_production_export` |
| `validation` | `safe_nontrivial` | 5444 | `ad1fa9b7b3a2e8b1e70839781d065d4234a23257f757be2a5e63ff178b574c96` | 7 | `goal4_rows`, `issue_5_8_production_export` |
| `validation` | `domain_nontrivial` | 5453 | `43df8616ab982b3f0fe0cad03fc75267e4a1dca66122edbb14d7667be1cfe0da` | 7 | `goal4_rows`, `issue_5_8_production_export` |
| `test_iid` | `safe_nontrivial` | 5395 | `51e4eed6fbecd4960e4e77767c138d7abff77c9842a769bf4e44b4d4d99385c8` | 7 | `goal4_rows`, `issue_5_8_production_export` |
| `test_iid` | `domain_nontrivial` | 5403 | `bd6c1144ac26cbc2c86d72d3ccce7cb66b58a5e7599d87136e14c9bedbe8a769` | 7 | `goal4_rows`, `issue_5_8_production_export` |
| `test_ood` | `safe_nontrivial` | 1107 | `6268ce8de99e6127db867323e32d4c82a142bcc8f9a924e131d89516c8dd42b6` | 7 | `goal4_rows`, `issue_5_8_production_export` |
| `test_ood` | `domain_nontrivial` | 1107 | `6268ce8de99e6127db867323e32d4c82a142bcc8f9a924e131d89516c8dd42b6` | 7 | `goal4_rows`, `issue_5_8_production_export` |


## Neural ranker

| Method | Split | Subset | Evaluable / all | Valid / all | Validation rate | Exact-best rate | Failed selected | Mean regret | Cost-scoring seconds | Speedup vs exact |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `exact_official_eml_dag` | `validation` | `all` | 10840/18168 (59.665%) | 10840/18168 (59.665%) | 10840/10840 (100.000%) | 10840/10840 (100.000%) | 0 | `0/10840` | 2220.45577 | 1x |
| `exact_official_eml_dag` | `validation` | `safe_nontrivial` | 5398/5444 (99.155%) | 5398/5444 (99.155%) | 5398/5398 (100.000%) | 5398/5398 (100.000%) | 0 | `0/5398` | 1107.62608 | 1x |
| `exact_official_eml_dag` | `validation` | `domain_nontrivial` | 5407/5453 (99.156%) | 5407/5453 (99.156%) | 5407/5407 (100.000%) | 5407/5407 (100.000%) | 0 | `0/5407` | 1112.80165 | 1x |
| `exact_official_eml_dag` | `test_iid` | `all` | 10752/18050 (59.568%) | 10752/18050 (59.568%) | 10752/10752 (100.000%) | 10752/10752 (100.000%) | 0 | `0/10752` | 2210.23259 | 1x |
| `exact_official_eml_dag` | `test_iid` | `safe_nontrivial` | 5350/5395 (99.166%) | 5350/5395 (99.166%) | 5350/5350 (100.000%) | 5350/5350 (100.000%) | 0 | `0/5350` | 1102.51944 | 1x |
| `exact_official_eml_dag` | `test_iid` | `domain_nontrivial` | 5358/5403 (99.167%) | 5358/5403 (99.167%) | 5358/5358 (100.000%) | 5358/5358 (100.000%) | 0 | `0/5358` | 1107.68582 | 1x |
| `exact_official_eml_dag` | `test_ood` | `all` | 2034/2214 (91.870%) | 2034/2214 (91.870%) | 2034/2034 (100.000%) | 2034/2034 (100.000%) | 0 | `0/2034` | 222.258886 | 1x |
| `exact_official_eml_dag` | `test_ood` | `safe_nontrivial` | 1017/1107 (91.870%) | 1017/1107 (91.870%) | 1017/1017 (100.000%) | 1017/1017 (100.000%) | 0 | `0/1017` | 104.648625 | 1x |
| `exact_official_eml_dag` | `test_ood` | `domain_nontrivial` | 1017/1107 (91.870%) | 1017/1107 (91.870%) | 1017/1017 (100.000%) | 1017/1017 (100.000%) | 0 | `0/1017` | 117.610261 | 1x |
| `neural_ranker` | `validation` | `all` | 10840/18168 (59.665%) | 10840/18168 (59.665%) | 10840/10840 (100.000%) | 8440/10840 (77.860%) | 0 | `33457/10840` | 150.309005 | 14.7726x |
| `neural_ranker` | `validation` | `safe_nontrivial` | 5398/5444 (99.155%) | 5398/5444 (99.155%) | 5398/5398 (100.000%) | 4251/5398 (78.751%) | 0 | `16034/5398` | 75.2227969 | 14.7246x |
| `neural_ranker` | `validation` | `domain_nontrivial` | 5407/5453 (99.156%) | 5407/5453 (99.156%) | 5407/5407 (100.000%) | 4154/5407 (76.826%) | 0 | `17423/5407` | 75.0581692 | 14.8259x |
| `neural_ranker` | `test_iid` | `all` | 10752/18050 (59.568%) | 10752/18050 (59.568%) | 10752/10752 (100.000%) | 8349/10752 (77.651%) | 0 | `31669/10752` | 147.189521 | 15.0162x |
| `neural_ranker` | `test_iid` | `safe_nontrivial` | 5350/5395 (99.166%) | 5350/5395 (99.166%) | 5350/5350 (100.000%) | 4189/5350 (78.299%) | 0 | `15351/5350` | 73.4186896 | 15.0169x |
| `neural_ranker` | `test_iid` | `domain_nontrivial` | 5358/5403 (99.167%) | 5358/5403 (99.167%) | 5358/5358 (100.000%) | 4116/5358 (76.820%) | 0 | `16318/5358` | 73.743504 | 15.0208x |
| `neural_ranker` | `test_ood` | `all` | 2034/2214 (91.870%) | 2034/2214 (91.870%) | 2034/2034 (100.000%) | 1499/2034 (73.697%) | 0 | `6298/2034` | 43.118034 | 5.15466x |
| `neural_ranker` | `test_ood` | `safe_nontrivial` | 1017/1107 (91.870%) | 1017/1107 (91.870%) | 1017/1017 (100.000%) | 796/1017 (78.269%) | 0 | `2588/1017` | 21.7025681 | 4.82195x |
| `neural_ranker` | `test_ood` | `domain_nontrivial` | 1017/1107 (91.870%) | 1017/1107 (91.870%) | 1017/1017 (100.000%) | 703/1017 (69.125%) | 0 | `3710/1017` | 21.4154659 | 5.49184x |
| `estimated_eml_tree_cost` | `validation` | `all` | 10840/18168 (59.665%) | 10840/18168 (59.665%) | 10840/10840 (100.000%) | 8933/10840 (82.408%) | 0 | `29247/10840` | 144.973142 | 15.3163x |
| `estimated_eml_tree_cost` | `validation` | `safe_nontrivial` | 5398/5444 (99.155%) | 5398/5444 (99.155%) | 5398/5398 (100.000%) | 4446/5398 (82.364%) | 0 | `14386/5398` | 72.5067039 | 15.2762x |
| `estimated_eml_tree_cost` | `validation` | `domain_nontrivial` | 5407/5453 (99.156%) | 5407/5453 (99.156%) | 5407/5407 (100.000%) | 4452/5407 (82.338%) | 0 | `14861/5407` | 72.4383994 | 15.362x |
| `estimated_eml_tree_cost` | `test_iid` | `all` | 10752/18050 (59.568%) | 10752/18050 (59.568%) | 10752/10752 (100.000%) | 8796/10752 (81.808%) | 0 | `28953/10752` | 141.821239 | 15.5846x |
| `estimated_eml_tree_cost` | `test_iid` | `safe_nontrivial` | 5350/5395 (99.166%) | 5350/5395 (99.166%) | 5350/5350 (100.000%) | 4377/5350 (81.813%) | 0 | `14233/5350` | 71.0774969 | 15.5115x |
| `estimated_eml_tree_cost` | `test_iid` | `domain_nontrivial` | 5358/5403 (99.167%) | 5358/5403 (99.167%) | 5358/5358 (100.000%) | 4375/5358 (81.654%) | 0 | `14720/5358` | 70.7164144 | 15.6638x |
| `estimated_eml_tree_cost` | `test_ood` | `all` | 2034/2214 (91.870%) | 2034/2214 (91.870%) | 2034/2034 (100.000%) | 1555/2034 (76.450%) | 0 | `9192/2034` | 43.2768803 | 5.13574x |
| `estimated_eml_tree_cost` | `test_ood` | `safe_nontrivial` | 1017/1107 (91.870%) | 1017/1107 (91.870%) | 1017/1017 (100.000%) | 798/1017 (78.466%) | 0 | `4020/1017` | 21.6921294 | 4.82427x |
| `estimated_eml_tree_cost` | `test_ood` | `domain_nontrivial` | 1017/1107 (91.870%) | 1017/1107 (91.870%) | 1017/1017 (100.000%) | 757/1017 (74.435%) | 0 | `5172/1017` | 21.5847509 | 5.44877x |
| `ast_dag_cost` | `validation` | `all` | 10840/18168 (59.665%) | 10840/18168 (59.665%) | 10840/10840 (100.000%) | 8867/10840 (81.799%) | 0 | `21535/10840` | 147.184014 | 15.0863x |
| `ast_dag_cost` | `validation` | `safe_nontrivial` | 5398/5444 (99.155%) | 5398/5444 (99.155%) | 5398/5398 (100.000%) | 4411/5398 (81.715%) | 0 | `10787/5398` | 73.6082493 | 15.0476x |
| `ast_dag_cost` | `validation` | `domain_nontrivial` | 5407/5453 (99.156%) | 5407/5453 (99.156%) | 5407/5407 (100.000%) | 4421/5407 (81.764%) | 0 | `10748/5407` | 73.5477256 | 15.1303x |
| `ast_dag_cost` | `test_iid` | `all` | 10752/18050 (59.568%) | 10752/18050 (59.568%) | 10752/10752 (100.000%) | 8661/10752 (80.552%) | 0 | `23519/10752` | 143.917064 | 15.3577x |
| `ast_dag_cost` | `test_iid` | `safe_nontrivial` | 5350/5395 (99.166%) | 5350/5395 (99.166%) | 5350/5350 (100.000%) | 4312/5350 (80.598%) | 0 | `11666/5350` | 71.8930326 | 15.3356x |
| `ast_dag_cost` | `test_iid` | `domain_nontrivial` | 5358/5403 (99.167%) | 5358/5403 (99.167%) | 5358/5358 (100.000%) | 4305/5358 (80.347%) | 0 | `11853/5358` | 71.9967035 | 15.3852x |
| `ast_dag_cost` | `test_ood` | `all` | 2034/2214 (91.870%) | 2034/2214 (91.870%) | 2034/2034 (100.000%) | 1565/2034 (76.942%) | 0 | `5290/2034` | 42.8510489 | 5.18678x |
| `ast_dag_cost` | `test_ood` | `safe_nontrivial` | 1017/1107 (91.870%) | 1017/1107 (91.870%) | 1017/1017 (100.000%) | 796/1017 (78.269%) | 0 | `2443/1017` | 21.4674391 | 4.87476x |
| `ast_dag_cost` | `test_ood` | `domain_nontrivial` | 1017/1107 (91.870%) | 1017/1107 (91.870%) | 1017/1017 (100.000%) | 769/1017 (75.615%) | 0 | `2847/1017` | 21.3836098 | 5.50002x |
| `deterministic_random` | `validation` | `all` | 10840/18168 (59.665%) | 10840/18168 (59.665%) | 10840/10840 (100.000%) | 2218/10840 (20.461%) | 0 | `595752/10840` | 246.049815 | 9.02442x |
| `deterministic_random` | `validation` | `safe_nontrivial` | 5398/5444 (99.155%) | 5398/5444 (99.155%) | 5398/5398 (100.000%) | 1119/5398 (20.730%) | 0 | `296217/5398` | 123.445273 | 8.97261x |
| `deterministic_random` | `validation` | `domain_nontrivial` | 5407/5453 (99.156%) | 5407/5453 (99.156%) | 5407/5407 (100.000%) | 1064/5407 (19.678%) | 0 | `299535/5407` | 122.576502 | 9.07843x |
| `deterministic_random` | `test_iid` | `all` | 10752/18050 (59.568%) | 10752/18050 (59.568%) | 10752/10752 (100.000%) | 2158/10752 (20.071%) | 0 | `574885/10752` | 239.85196 | 9.21499x |
| `deterministic_random` | `test_iid` | `safe_nontrivial` | 5350/5395 (99.166%) | 5350/5395 (99.166%) | 5350/5350 (100.000%) | 1081/5350 (20.206%) | 0 | `286458/5350` | 119.588061 | 9.21931x |
| `deterministic_random` | `test_iid` | `domain_nontrivial` | 5358/5403 (99.167%) | 5358/5403 (99.167%) | 5358/5358 (100.000%) | 1033/5358 (19.280%) | 0 | `288427/5358` | 120.236571 | 9.21255x |
| `deterministic_random` | `test_ood` | `all` | 2034/2214 (91.870%) | 2034/2214 (91.870%) | 2034/2034 (100.000%) | 1078/2034 (52.999%) | 0 | `22362/2034` | 44.8440689 | 4.95626x |
| `deterministic_random` | `test_ood` | `safe_nontrivial` | 1017/1107 (91.870%) | 1017/1107 (91.870%) | 1017/1017 (100.000%) | 576/1017 (56.637%) | 0 | `10158/1017` | 22.5003358 | 4.65098x |
| `deterministic_random` | `test_ood` | `domain_nontrivial` | 1017/1107 (91.870%) | 1017/1107 (91.870%) | 1017/1017 (100.000%) | 502/1017 (49.361%) | 0 | `12204/1017` | 22.3437331 | 5.26368x |

Issue 5-7 process-wide runtime: 10721.4889s replay, 24.5323453s fit/evaluation; peak sampled process-tree RSS 1700249600 bytes across 604 samples.

## Artifact provenance

| Name | Schema | Size | SHA-256 | Path |
|---|---|---:|---|---|
| `goal1_corpus` | geml-corpus-v1 | 12003 | `77fce5779b3d2c2f3cdf2b9f49da54cd14474d37ab128337bdf4fcc52afd4f0d` | `outputs/final/goal1/final/run/manifests/corpus.manifest.json` |
| `goal2_final` | geml-goal2-manifest-v1 | 18290 | `06d129f427dc376190fcee38217a6bc78f35c49a61bc8e453849473ec96e8e32` | `outputs/final/goal2/final/manifest.json` |
| `goal3_final` | geml-goal3-manifest-v1 | 22885 | `279b1d016bf8ff3295cf183cee9929dd69315ef21fedf58f3d63bb74414b5000` | `outputs/final/goal3/final/manifest.json` |
| `goal4_rows` | geml-goal4-row-v2 | 752883521 | `f8fd2e6db597da465d4367ce402fc69598eac45031fa9afe52fedc17011a2c31` | `outputs/final/goal4/final/final.rows.jsonl` |
| `goal4_run` | geml-goal4-run-v1 | 2390 | `e616a4fb4354fbb64d786d9da22be2923d5e18868f64dad8f8f5e6d10e608fc4` | `outputs/final/goal4/final/final.run.json` |
| `issue_5_5_frequent_motifs` | geml-goal5-frequent-run-complete-v1 | 2159 | `9310a6fe1cafb101418e55c310aa87baade7cccb9a72a3fe1817a21d9302240f` | `outputs/final/goal5/motif_sweeps/final/a9a271583707-77fce5779b3d-28618ea156fd/run.complete.json` |
| `issue_5_5_heldout_results` | geml-goal5-frequent-heldout-results-v1 | 191229 | `b3442598104c2799e71a1ea299f349ce19ffcb1b038b4075b9e9107a9600a0f3` | `outputs/final/goal5/motif_sweeps/final/a9a271583707-77fce5779b3d-28618ea156fd/heldout_results.json` |
| `issue_5_5_sweep_table` | geml-goal5-frequent-sweep-table-v1 | 1135326 | `682685833a4ac3b510083f38e9446c042346f9ba48ac819ea6d63fe92b74a9be` | `outputs/final/goal5/motif_sweeps/final/a9a271583707-77fce5779b3d-28618ea156fd/sweep_table.json` |
| `issue_5_6_experiment_result` | geml-goal5-learned-results-v1 | 9580860 | `af59794a1710316218f33e65cd5e6f4bced4b97b50f360742e444183629820b3` | `outputs/final/goal5/learned_motifs/2521e53bc741-882ceb5185be-5842901d3d15/experiment.result.json` |
| `issue_5_6_heldout_results` | geml-goal5-learned-heldout-results-v1 | 5878315 | `ff577249c5010264934663931853538f2d910a23e4575c85e14ff697175ebf57` | `outputs/final/goal5/learned_motifs/2521e53bc741-882ceb5185be-5842901d3d15/heldout_results.json` |
| `issue_5_6_learned_motifs` | geml-goal5-learned-run-complete-v1 | 2597 | `eee61141e7322fa12dd8990712b4aade83c08bb73c005a943b5a414fb9dbec3e` | `outputs/final/goal5/learned_motifs/2521e53bc741-882ceb5185be-5842901d3d15/run.complete.json` |
| `issue_5_6_validation_results` | geml-goal5-learned-validation-results-v1 | 3701639 | `67807cd681d5457b7c840beb953fdc3ab1ba2a9aa343533f6ded3dab4e8cd66d` | `outputs/final/goal5/learned_motifs/2521e53bc741-882ceb5185be-5842901d3d15/validation_results.json` |
| `issue_5_7_candidate_groups` | geml-egraph-candidate-group-v1 | 530841706 | `915c214fc7efab9ca79fcc26e482127176ba6f8b8714060b9c2741d2197b1e00` | `outputs/final/goal5/neural_ranker/beb427df6191-f8fd2e6db597-97a163047ff7/candidate_groups.jsonl` |
| `issue_5_7_neural_ranker` | geml-goal5-neural-ranker-complete-v1 | 2746 | `8be559c78a12c8bcd42886217f71236c395c6eb9a7f916d0097bb9c0d4e0a961` | `outputs/final/goal5/neural_ranker/beb427df6191-f8fd2e6db597-97a163047ff7/run.complete.json` |
| `issue_5_7_report` | geml-goal5-neural-ranker-report-v1 | 1003902 | `fdd5db9a6d82d2c8de2d3fb431c0736b05ace21ce2a1540a9aada32d380648be` | `outputs/final/goal5/neural_ranker/beb427df6191-f8fd2e6db597-97a163047ff7/report.json` |
| `issue_5_7_run_manifest` | geml-goal5-neural-ranker-run-v1 | 3047 | `8e6c996f07d0a241275a6cc68833e4f0237435425746f124e8b68ba49aac96db` | `outputs/final/goal5/neural_ranker/beb427df6191-f8fd2e6db597-97a163047ff7/run.manifest.json` |
| `issue_5_7_test_iid_outcomes` | explicitly unversioned: issue 5-7 outcome rows predate a row-level schema_version; their exact bytes are authenticated by the atomic completion | 75113669 | `c28579750d90b7e5295ce03f747e3d7bcef66bd2f9c18a4d382935495b1e900d` | `outputs/final/goal5/neural_ranker/beb427df6191-f8fd2e6db597-97a163047ff7/test_iid.outcomes.jsonl` |
| `issue_5_7_test_ood_outcomes` | explicitly unversioned: issue 5-7 outcome rows predate a row-level schema_version; their exact bytes are authenticated by the atomic completion | 15772619 | `a734932bcbfa99883ab927e4221cee98f3883e8441f7b31f13a791a828770849` | `outputs/final/goal5/neural_ranker/beb427df6191-f8fd2e6db597-97a163047ff7/test_ood.outcomes.jsonl` |
| `issue_5_7_validation_outcomes` | explicitly unversioned: issue 5-7 outcome rows predate a row-level schema_version; their exact bytes are authenticated by the atomic completion | 76527855 | `cde361dccbaf218d0bac6e3d47cbff5120a90aadcc4b4c40e1582d03d0463028` | `outputs/final/goal5/neural_ranker/beb427df6191-f8fd2e6db597-97a163047ff7/validation.outcomes.jsonl` |
| `issue_5_8_production_export` | geml-goal5-production-export-v1 | 2104849 | `54a2ead4d9219172d4e7c819cfb4404e09176923d09fd664586e06b658b7082d` | `outputs/final/goal5/export/run-7a82a05b64211cd6e911f5174a1326e2563037e17c3ac474158d4b463e05bc6e/run.complete.json` |

## Interpretation boundaries

- Pure EML-DAG and both e-graph-selected EML-DAG tracks contain only pure EML nodes.
- Macro and motif nodes are compression vocabulary nodes, not pure EML nodes.
- Macro graphs are structurally close to labeled compiler/AST graphs and must not be described as single-operator EML.
- Safe-real nontrivial cohorts use branch-insensitive finite-real rewrites; domain-aware cohorts are conditional positive-real-formal results under each row's recorded assumptions.
- Standalone graph MDL and dictionary-inclusive motif MDL are distinct scopes and are never combined into one aggregate.
- Neural-ranker speedup is scoped only to candidate cost scoring; it is not an end-to-end pipeline or mathematical-reasoning speedup.
- Every failure, unsupported case, validation failure, and missing resource observation remains visible in its all-attempted denominator.
- All-processed denominators remain track-specific: the Goal 4 e-graph study covers its frozen 30,000-expression selection, while issue 5-8 covers the full 250,000-expression corpus. Exact cross-track comparisons use the separately joined nontrivial cohorts.
