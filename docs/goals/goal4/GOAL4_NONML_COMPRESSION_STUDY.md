# Goal 4 non-ML compression study

## What is optimized

Goal 4 optimizes the exact node count of the official Pure EML DAG produced from a source
AST. Three structural quantities remain separate:

- source AST tree size;
- source AST DAG size after exact structural sharing; and
- official Pure EML DAG size after direct compilation and hash-consing.

A source operator can expand into many primitive EML nodes. Equivalent source forms can
therefore have different exact EML DAG costs even when their AST sizes are similar.

## Why equality saturation can help

An e-graph stores multiple source forms in shared equivalence classes. Documented rewrites
add forms; cycle-safe extraction turns a bounded portion of that space back into concrete
expressions; the frozen Goal 3 cost boundary ranks them.

The process is not learned. It has:

- a static rule catalogue;
- explicit guards;
- exact rational folding;
- deterministic structural tie-breaks; and
- externally configured limits.

The source expression is forcibly retained among the candidates. Thus the search can
improve or preserve the official cost, but it cannot legitimately degrade it.

## Sources of reductions

### Operand order and grouping

Addition and multiplication are commutative and associative under the finite-real source
contract. Different orderings and groupings can expose different exact sharing after EML
compilation. The cost boundary, not an AST heuristic, decides which retained form is
cheapest.

### Identities and cancellation

The safe library includes additive and multiplicative identities, multiplication by zero,
double negation, additive inverse, and subtraction lowering. Rules that delete an operand
are scoped to finite real operands on the validated source domain; this assumption is
documented in `EGRAPH_SEMANTICS.md`.

Subtraction lowering is bidirectional. This lets extraction recover a `sub` node when the
official subtraction formula is cheaper than an equivalent addition-of-negation form.

### Exact constant folding

Integer and rational folds use `fractions.Fraction`. Division by zero, non-integer power
folds, zero to a non-positive power, exponents outside the configured bound, and results
outside the exact digit bound are explicitly declined and counted. No float approximation
or hidden simplifier participates.

### Guarded domain rules

The formal mode can collapse structures such as `log(exp(x))` or `exp(log(x))` only when a
named guard confirms the needed declaration. Product-log, exponential-sum, and power rules
retain their direction, branch flag, assumptions, and counterexample-backed justification
in provenance.

These gains are conditional on the declared real or positive-real domain and must not be
reported as universal complex identities.

## Exact scalable costing

Goal 4 does not materialize expanded EML trees in production:

- EML DAG cost uses Goal 3's direct source-AST compiler and structural hash-consing.
- EML tree size uses Goal 2's count-only counterparts of the frozen compiler formulas.
- AST DAG and tree sizes use the frozen AST graph/statistics interfaces.

Materializing the recursive EML expansion would create an avoidable memory risk and would
discard Goal 3's scalable direct-cost interface.

## Validation layers

Each selected form is supported by several independent checks:

1. extraction metadata names the root e-class;
2. non-mutating structural lookup finds the concrete expression in that root;
3. the original source is independently found in the same root;
4. the candidate compiles through the official direct EML DAG boundary;
5. exact count-only construction succeeds; and
6. deterministic real-domain probes find neither a value mismatch nor a definedness
   mismatch and produce finite evidence.

The e-class relationship is the formal evidence under the enabled rules and assumptions.
The numeric layer is deliberately described as an audit: finitely many probes cannot prove
a universal identity. Zero finite evidence is inconclusive, not valid.

## Provenance representation

Every e-graph-changing application is durable. A per-row rule catalog stores the rule ID,
name, tier, mode, direction, justification, assumptions, branch flag, verifier flag, and
substitution field order once. Compact application tuples then retain sequence, iteration,
catalog index, guard outcome, source/result e-classes, substitution values, and detail.

All no-op, guard-rejected, unsupported, and limit-stopped attempts remain in exact per-rule
outcome aggregates. An ordered-log digest provides an integrity binding without inflating
every row with repeated policy text.

## Audited production findings

The frozen 30,000-expression study is identified in `GOAL4_SUMMARY.md`. On the 18,210
costed rows per mode, `safe_real` improved 4,349 (23.882%) and
`positive_real_formal` improved 5,026 (27.600%). Their all-processed rates were 14.497% and
16.753%. Exact signed savings were 95,596 and 104,996 EML-DAG nodes, with no degraded
selection.

The guarded formal rules added 677 improved rows. Their observed benefit was concentrated
in `exp_log` and `ood_stress`; algebraic and powers/division/rational improved counts did
not change. This is an empirical result for the frozen balanced selection and configured
bounds, not a claim about every expression in those families.

Cost coverage was 60.700%. The remaining rows were retained rather than filtered:
11,566 unsupported trigonometric/mixed rows and 224 conservative validation failures per
mode. There were no timeouts or internal errors in the audited final run.

One non-selected formal candidate received a conservative floating-probe
`domain_mismatch` because an exact integer exponent reached the evaluator through
`exp(log(9)) - 7`. The source anchor was retained and selected unchanged. This example is
reported because validation failures are evidence about the audit boundary, even when they
do not affect a selected result.

## Honest limitations

- The candidate set is depth-, beam-, count-, node-, iteration-, and time-bounded.
- A cheaper equivalent expression may exist outside the retained frontier.
- Wall-clock limits can stop at different frontiers under different machine load.
- Finite-real and declared-domain assumptions do not imply unrestricted complex
  equivalence.
- Unsupported trigonometric/hyperbolic rows are failures in the all-processed denominator,
  not silently removed.
- Floating numeric probes can conservatively reject an exact identity near a real-domain
  boundary; rejected candidates are retained and never promoted.
- “Best” always means best among retained, validated candidates under the stated limits.
