# Goal 4 Non-ML Compression Study

This study explains, **without invoking any machine learning**, why the Goal 4 e-graph
stage can reduce the official Pure EML DAG cost of an expression, and exactly where those
reductions come from. Every mechanism here is a deterministic rewrite or an exact
arithmetic fold; none of it is learned, statistical, or heuristic beyond the fixed,
configured resource limits.

## The three size measures

Three structural sizes are relevant, and they are not interchangeable:

- **AST tree size** — the node count of the source expression as a plain tree.
- **AST DAG size** — the node count after exact structural sharing of identical AST
  subtrees.
- **EML DAG size** — the node count of the expression after official compilation to Pure
  EML and exact structural sharing (the frozen Goal 3 cost).

The EML DAG size is the quantity Goal 4 optimizes, because it is the canonical, official
cost. A single source operator can expand into many primitive EML nodes (the `eml(x, y) =
exp(x) − ln(y)` construction is not one-to-one with source operators), so two source
expressions that are semantically equal can have very different EML DAG sizes. That gap is
the opportunity Goal 4 exploits.

## Why an e-graph helps

An e-graph represents many equivalent expressions in one shared structure. Equality
saturation applies the rewrite rules to a fixed point (or a resource limit), populating the
e-graph with every expression reachable from the input by the enabled rules. Extraction
then enumerates concrete expressions from that structure, and the cost stage picks the one
whose official EML DAG cost is smallest.

The key point is that the *choice of source form* changes the EML DAG cost even when the
mathematical value is identical. For example, the two operand orders of a commutative
operator compile to structurally different EML DAGs with different amounts of sharing, so
one order can be strictly cheaper than the other. The e-graph makes both orders available;
extraction and the Goal 3 cost pick the cheaper one. No assumption and no approximation is
involved — only a search over equivalent forms and an exact cost comparison.

## Where the improvements come from

### Easy identities and safe rewrites

The safe rule library (branch-insensitive, enabled in both modes) supplies the
assumption-free reductions:

- commutativity and associativity, which expose cheaper operand orders and groupings;
- additive and multiplicative identities (`x + 0`, `x · 1`) and multiplication by zero,
  which delete whole subtrees;
- double negation and additive inverse, which cancel structure;
- subtraction lowering (`a − b = a + (−b)`), whose backward orientation lets extraction
  recover a `sub` node when that compiles to a smaller EML DAG.

These never require a domain assumption and are the source of every improvement in
`safe_real` mode.

### Constant folding

Exact constant folding replaces an operator applied to exact rational constants with the
exact result, using `fractions.Fraction` throughout. Folding removes operator nodes
outright and is bounded: a result beyond a digit bound, a division by zero, a non-integer
exponent, or zero raised to a non-positive power is declined and recorded, never
approximated. Because a folded constant compiles to a smaller EML construction than the
un-folded arithmetic, folding is a direct EML DAG reduction.

### Domain-aware rewrites

In `positive_real_formal` mode only, the guarded domain rules add reductions that are valid
only under explicit assumptions — for example `log(exp(x)) → x` when `x` is declared real,
or `exp(log(x)) → x` when `x` is declared positive. Each such rule fires only when a guard
confirms the caller declared the required property; nothing is inferred from the shape of
the expression. These rules can collapse large EML constructions (a `log`/`exp` pair expands
into many primitive nodes), so where they apply they can produce the largest reductions —
but strictly within the declared domain.

## Relationship between the layers

The pipeline composes these layers in a fixed order:

```
source AST  --(compile)-->  e-graph expression
     |                              |
     |                     equality saturation (safe rules; + domain rules in formal mode)
     |                              |
     |                     cycle-safe bounded extraction  -->  candidate expressions
     |                              |
     |                     validation (compile / semantic / domain)  -->  valid candidates
     |                              |
     +----- Goal 3 exact EML DAG cost of input      Goal 3 exact EML DAG cost of candidates
                              \            /
                               deterministic ranking + selection
```

Rule application widens the space of equivalent forms; extraction turns that space back into
concrete expressions; the Goal 3 cost decides which form is cheapest. Compression is thus a
*consequence* of searching equivalent forms and costing them officially — not a heuristic
size estimate and not a learned model.

## Limitations

- **Resource constraints.** Saturation and extraction are bounded. A cheaper equivalent form
  can exist beyond the iteration, node, depth, beam, candidate, or time limits and simply not
  be found. Every stop reason is recorded.
- **Candidate limits.** Extraction enumerates a bounded, beam-limited set of candidates. The
  selected candidate is the best of those enumerated, not of all equivalent expressions.
- **Rewrite limits.** Only the enabled rules participate. The safe library is intentionally
  small, and the domain library is intentionally guarded; identities outside them are not
  applied.
- **Branch-sensitive assumptions.** Every domain-aware reduction is conditional on
  caller-declared assumptions and is confined to `positive_real_formal` mode. Without those
  assumptions, the corresponding reductions do not occur, by design.

The honest reading is therefore: Goal 4 compresses EML DAG cost by searching a bounded space
of provably equivalent (or, in formal mode, assumption-conditioned) forms and selecting the
cheapest under the official cost — a deterministic, auditable, non-ML procedure whose reach
is exactly the reach of its rules and its resource budget.
