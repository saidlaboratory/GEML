# Goal 3 Summary

what actually got built, what's actually proven, and what isn't yet.
structural evidence only - no claims here about math correctness
beyond what's been directly tested, no e-graph or motif claims.

## what goal 3 covers

a generic graph/DAG schema (3-1) neutral across ast/eml/macro/motif
representations, exact structural sharing for both AST trees (3-2) and
pure EML trees (3-3), a direct/memoized construction path that skips
allocating the full uncompressed tree (3-4), an audit proving the
direct and post-hoc paths produce identical results (3-5), a metrics
pipeline with checkpoint/resume built for the real 250k corpus (3-6),
stratified analysis of that pipeline's output (3-7), and this - the
frozen cost interface goal 4 actually consumes (3-8).

## what's genuinely verified

- exp(x) = eml(x, 1) and ln(z) = eml(1, eml(eml(1,z), 1)) - both
  pulled directly from the real paper (arXiv:2603.21852v2, eq 3 and
  5), checked numerically against math.exp/math.log, and checked
  composed together (ln(exp(x)) = x).
- (x+1)*(x+1)-shaped duplication collapses from 7 nodes to 4, for both
  AST and EML DAGs, tested directly.
- direct construction and post-hoc construction produce byte-identical
  canonical signatures for every case both paths can currently build.
- the resume/checkpoint guarantee: running interrupted-then-resumed
  produces identical final success/failure counts to running straight
  through, tested with an actual simulated interruption, not just
  assumed.

## what's NOT yet covered

- add, multiply, divide, power - the real formulas exist now (found on
  the goal2 branch, in compiler_arithmetic.py/compiler_core.py) but
  aren't wired into goal 3's DAG modules yet. everything built so far
  is honestly scoped to exp/ln and compositions of them.
- the real 250k corpus run. the corpus itself was genuinely generated
  and QA-passed (confirmed via goal1's CORPUS_QA.md - exact 250,000
  rows, correct split counts, all checksums valid), but the actual
  data files aren't committed to git, so goal 3's pipeline hasn't
  actually processed them yet.
- trig/hyperbolic functions - out of scope for this stage entirely.

## denominator and scaling caveats

every aggregate stat reported anywhere in goal 3's analysis (3-7)
distinguishes all_processed_count from valid_count - an average is
never silently computed over failed rows, and a reader always knows
which denominator a given number is really over. the 10k/50k/100k/250k
stability-curve machinery is built and tested, but has no real data
points yet since the real corpus hasn't been run through the pipeline.

## the frozen interface (this issue)

`compute_eml_dag_cost()` / `compute_eml_dag_cost_from_tree()` in
`src/geml/interfaces/eml_dag_cost.py` are the only things goal 4 should
ever call for a dag cost number. this module deliberately never
imports from `geml.experiments.*` or `geml.analysis.*` - only from the
already-frozen `geml.graph.*`/`geml.dag.*` modules - so goal 4 never
has to pull in goal 3's internal research code just to get a cost.
