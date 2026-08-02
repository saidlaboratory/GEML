# Goals 1--5: finalized CPU and corpus results

The values below are transcribed from the finalized manifests, summaries, plot data,
and reports in `GEML_artifacts`.  They are not estimates and do not include raw
shards.  Each number is mapped to its source in `PROVENANCE.md`.

## Goal 1 -- source-expression corpus

The final corpus is `geml-goal1-final`, schema `geml-corpus-v1`, generated with seed
20260721.  It contains 250,000 accepted rows split 175,000/25,000/25,000/25,000
(train/validation/test-IID/test-OOD), with zero storage, AST, parse, LaTeX, or
round-trip failures.  The generator attempted 286,413 rows: 35,768 were duplicate
attempts and 645 were rejected by the explicit triviality policy; 250,000 were
accepted, giving an acceptance rate of 0.8728654076456027.  Unsupported rows and
identity conflicts were both zero.

Generation took 620.2928234 s.  Accepted-row throughput was 403.0354544966034
rows/s, attempted-row throughput was 461.7383745149426 rows/s, and peak RSS was
2,193,186,816 bytes.  Domain counts were 68,345 nonzero-real, 84,843 positive-real,
and 96,812 safe-real.  Family counts were 70,000 algebraic-core, 40,000 exp/log,
35,000 mixed-elementary, 25,000 OOD-stress, 40,000 powers/division/rationals, and
40,000 trig/hyperbolic.  Variable counts for 1 through 6 variables were
68,934, 61,244, 46,743, 34,511, 24,095, and 14,473.

The policy audit reports all trig operators covered, approved-domain checks true,
blanket `log(exp(...))` wrapping false, 196,656 certified log arguments, 28,648
certified tan arguments, 257,698 lowered reciprocal candidates, and 292,783 negative
power arguments.  A LaTeX parser was unavailable for 64 checks, but no round-trip
failure was recorded; this is a tooling limitation, not a silent success.

QA found 250,000 unique authoritative s-expressions, expression IDs, and structural
identities; cross-split ID collisions and duplicate authoritative occurrences were
zero.  Triviality feature counts were constant-only 159,469, exp/log 34,685,
log/exp 42,093, log(1) 6,400, and multiplication-by-one 50,000, subject to the
configured caps.

## Goal 2 -- pure EML

The official-v4 compiler processed all 250,000 rows with zero conversion failures.
Elapsed time was 2,557.261734 s, processing throughput 99.16671338859994 rows/s,
aggregation throughput 97.76081840827327 rows/s, and peak RSS 1,617,088,512 bytes.
The semantic audit selected 280 rows: 273 were materialized and audited (203 passed,
3 mismatch, 45 nonfinite, and 22 overflow), while 7 exceeded the node limit before
materialization.  A further 249,720 rows were not selected for that audit.  These
audit statuses are not corpus conversion failures.

Raw pure-EML expansion had median alpha 40.6602, mean 952.1371, and approximate p99
10,448.6.  Zero of 250,000 rows fell below the preregistered 1.29--1.50 thresholds.

## Goal 3 -- AST/EML DAGs and costs

Direct hash-consing processed 250,000 rows with zero failures, audit gate ready, in
1,568.0315 s (159.4356 rows/s; cumulative 162.7917 rows/s) and peak RSS
1,444,044,800 bytes.  AST-DAG maximum indegree was 63; EML-DAG maximum indegree was
1,939.  Mean reused nodes were 3.430812 (AST) and 18.130736 (EML); mean reused
references were 13.759432 and 276.193548; mean excess references were 10.32862 and
258.062812.  Mean reuse depth was 4.6421511875 and 27.7799859862, with sharing
concentration 0.5698556037 and 0.7978003017.

The mean raw-tree alpha was 952.1371252900141.  Reported means were 8.334401271758448
for EML-DAG versus AST-tree, 10.474953890182578 for EML-DAG versus AST-DAG,
1.361707663394114 AST compression, and 39.37500771693145 EML compression.  No row
was structurally competitive under the published criterion; the best remaining
ratio was 8/7.

## Goal 4 -- verifier-gated e-graph study

The study selected 30,000 expressions and evaluated two 30,000-row modes.  Each mode
costed 18,210 rows, had 11,790 failures, and zero timeouts.  For `safe_real`, 4,349
costed rows improved (23.88248215% of costed; 14.49666667% of processed), with mean
signed improvement 5.2496430533 nodes and mean relative improvement 0.0236117313.
For `positive_real_formal`, 5,026 improved (27.60021966% of costed; 16.75333333%
of processed), with mean signed improvement 5.7658429434 nodes and relative
improvement 0.02848704193.  Failures were principally unsupported trig/hyperbolic
operators or independent-validation rejection; no timeout occurred.

## Goal 5 -- motifs, ranker, and export

The selected frequent vocabulary has 1,024 motifs of sizes 2--4 with minimum support
32.  On test-IID it used 317,678,264 bits versus a 572,524,716-bit baseline,
saving 254,846,452 bits (44.512742398%); on test-OOD it used 818,099,441 versus
1,453,062,331 bits, saving 634,962,890 bits (43.69825550%).  Reconstruction and
source failures were zero.

The learned vocabulary also has size 1,024.  Its test-IID total was 324,485,346
bits, versus 317,678,264 for equal-budget frequent, and random median 368,943,015
bits.  This is a null result against equal-budget frequent, positive against random
and uncompressed macro baselines.  Test-OOD totals were 821,843,999 learned and
818,099,441 equal-budget frequent bits.

The neural ranker used 30,000 expressions, 60,000 groups, and 299,645 candidates
(298,211 valid, 1,434 failed), with 23,132 empty groups and zero replay mismatches.
On test-IID, 10,752 groups were evaluable and 7,298 unevaluable: neural exact-best
was 8,349/10,752 (77.65067%), mean regret 2.9454055, total regret 31,669, and
15.0162x exact-scoring speedup.  EML-tree and AST-DAG heuristics were exact-best on
8,796 and 8,661 groups, respectively, and therefore outperform the neural ranker on
the preregistered comparison.  On test-OOD, neural was 1,499/2,034 (73.69715%),
mean regret 3.09636185, total regret 6,298, and 5.15466x speedup; AST-DAG was
1,565/2,034 and again outperformed neural.  These comparisons are null, not a claim
of neural superiority.

The production export contains exactly 250,000 expressions, 1,250,000 graph views,
250,000 hierarchies, and 2,500 batches, with zero validation or reconstruction
failures.  The published learned-motif MDL denominator is 75,000 rows for its MDL
aggregate; it must not be presented as a 250,000-row denominator.
