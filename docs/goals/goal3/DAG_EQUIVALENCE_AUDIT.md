# Goal 3 DAG Equivalence Audit (3-5)

what this actually proves, and what it doesn't - read this before
citing the audit results anywhere.

## what's covered right now

only `exp` and `ln`, plus compositions of them (`ln(exp(x))`,
`exp(exp(x))`), plus one repeated-subexpression case. those are the
only two eml constructions currently verified against the real paper
(arXiv:2603.21852v2, eq 3 and 5 - see 3-3/3-4). everything runnable in
the audit set traces back to just these two building blocks.

6 cases run and match exactly across every comparison axis (signature,
node count, edge count, depth, evaluation, purity, both directions).

## what's NOT covered

`add`, `mul`, `pow` - none of these have real constructors yet, since
2-2/2-3/2-4's actual compiler formulas aren't merged. there's one
deliberately included case (`add_family_blocked`) that exists purely
to prove the audit harness reports this gap explicitly instead of
silently pretending everything's covered - it shows up as `blocked`,
not as a pass, not as a skip.

**do not read "6/6 passing" as "goal 3's dag conversion is proven
correct in general."** it's proven correct for exactly the two
families tested. every other family is an open question until it has
its own real constructor and its own audit case.

## why this matters

the whole point of 3-4 (direct construction) is a performance
optimization over 3-3 (build-then-compress) - same math, different
allocation strategy. this audit is the thing that actually backs that
claim up with numbers, rather than just asserting it. if a future
family's direct and post-hoc paths ever disagree, that's a real bug in
one of the two implementations, and this audit is what's supposed to
catch it before it goes further.

## scale note

this audit runs on hand-built tiny fixtures only, not the real 250k
corpus - that's explicitly out of scope for 3-5 (see issue: "do not
run the full 250k pipeline"). once the real corpus exists, a similar
but much larger-scale audit will be needed as part of a later goal 3
task, not this one.

## how to extend this later

once 2-2/2-3/2-4 land with real add/mul/pow formulas, the
`add_family_blocked` case (and equivalents for mul/pow) should get
replaced with real runnable cases the same way `exp`/`ln` are handled
here - build both a direct and post-hoc version, add eval bindings,
drop it into `STRATIFIED_AUDIT_SET` in `equivalence_audit.py`.
