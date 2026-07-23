# Goal 3 DAG Compression Study

detailed findings on exact structural sharing. structural evidence
only - this is about tree/DAG size, not about whether an expression is
mathematically "better" in any other sense.

## the core distinction this whole study depends on

"compresses well" and "becomes structurally competitive" are different
claims, and conflating them was flagged explicitly in 3-7:

- **compresses well** = eml_compression is high (the DAG is much
  smaller than that expression's own raw, uncompressed tree)
- **structurally competitive** = dag_alpha_vs_ast_tree is low
  (approaching or beating 1.0 - genuinely close to what a plain AST
  tree would've cost)

an expression can do the first without the second. a raw EML tree that
starts out 40x bigger than its AST can compress 10x and still end up
4x bigger than the AST - it compressed well, it did not become
competitive. 3-7's `classify_dual()` reports both booleans
independently for exactly this reason.

## what the tested cases show

for the two verified families (exp, ln) and their compositions:
- raw pure EML trees are consistently larger than the equivalent AST
  tree, matching the α > 1.56 threshold problem identified back in the
  original project brief
- exact structural sharing recovers a meaningful fraction of that size
  difference whenever an expression contains genuine duplication (like
  `(x+1)*(x+1)`-shaped repetition) - tested directly, 7 nodes down to 4
- expressions with NO internal duplication get no benefit from sharing
  at all (dag_node_count == ast_tree_node_count exactly) - sharing
  only helps when there's something to share, tested directly

## what this study does not claim

- no e-graph claims - discovering that two DIFFERENT expressions are
  mathematically equivalent (like sin²x+cos²x = 1) is goal 4's job
  entirely, not tested or claimed here
- no motif claims - goal 5's compressed-pattern-dictionary approach is
  a different technique, not evaluated here
- no claim about add/mul/pow/power families - not implemented yet in
  goal 3's DAG modules, so there's nothing to report on them
- no claim about real-scale (10k-250k) behavior - the stability-curve
  code is built and tested against synthetic scale points, but has no
  real corpus data run through it yet

## reproducibility

every number in this study traces back to a specific test in 3-1
through 3-7's test suites, runnable independently:
- `pytest tests/graph/test_schema.py` - schema/signature correctness
- `pytest tests/dag/test_ast_dag.py` - AST sharing
- `pytest tests/dag/test_eml_dag.py` - EML sharing, verified formulas
- `pytest tests/dag/test_direct_eml_dag.py` - direct construction matches post-hoc
- `pytest tests/experiments/test_goal3_audit.py` - the direct-vs-post-hoc audit
- `pytest tests/experiments/test_goal3_smoke.py` - pipeline + resume guarantee
- `pytest tests/analysis/test_goal3_analysis.py` - stratification + the dual-claim distinction
