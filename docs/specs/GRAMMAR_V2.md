# Grammar v2 contract status

## Status: blocked, not activated

Grammar v2 is an opt-in extension proposal for `asin`, `acos`, `atan`, `e`, and `pi`. It is
**not** enabled in the v1 generator, parser, e-graph, corpus, or any Goals 1-5 artifact. No v2
record may be aggregated with v1 records unless a future manifest explicitly groups the versions.

## Mathematical source and required semantics

The proposed principal-value identities are from the NIST Digital Library of Mathematical
Functions (DLMF): [inverse sine, Eq. 4.23.19](https://dlmf.nist.gov/4.23.E19), [inverse cosine,
Eq. 4.23.22](https://dlmf.nist.gov/4.23.E22), [inverse tangent, Eq. 4.23.26](https://dlmf.nist.gov/4.23.E26),
and [the inverse-cosine/inverse-sine relation, Eq. 4.23.16](https://dlmf.nist.gov/4.23.E16).
They use principal square roots and logarithms. The real `asin`/`acos` contract must accept
`[-1, 1]`, document signed zero and both endpoints, and retain typed invalid-domain outcomes for
arguments outside that interval. `atan` is real for finite real arguments, but its construction
still crosses a branch-sensitive complex representation. `e` must be constructed as `exp(1)`;
the pure-EML construction of `pi` must retain an explicit branch convention.

Every future constructor must publish its exact pure-EML emission fingerprint, node count, depth,
source provenance, and high-precision audit ledger. Final trees must contain only `eml`, source
variables, and primitive `1`; an imaginary unit is never a source leaf.

## Current implementation blocker

The issue-owned v2 paths exclude both `src/geml/egraph/ir.py` and `src/geml/eml/compiler_core.py`.
The present e-graph `Operator` enum has no `asin`, `acos`, or `atan` entries, so no sound v2
rewrite rule can be constructed in `rules_domain.py` alone. The current core `CompilerMode` also
has only `official_v4` and `clean_negation`; changing it from the v2 issue would exceed ownership.

Enabling a new operator in the existing global `OPERATORS` registry would cause the v1 generator
to select it, violating byte-identical v1 behavior. A safe implementation therefore requires an
explicitly approved shared-interface change: a separate versioned registry/dispatch boundary and
e-graph vocabulary with v1-default isolation. Until that approval is recorded, there is no v2
compiler, conformance benchmark, Gate G10, or v2 learning claim.
