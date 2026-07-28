# GEML grammar v2: real inverse trigonometry and named constants

## Scope and compatibility boundary

Grammar v2 is an explicit compiler-conformance extension containing only
`asin`, `acos`, `atan`, `pi`, and `e` in addition to the frozen v1 vocabulary.
It does not replace v1 and does not authorize a new corpus, graph, motif,
compression, or learning run.

The following remain authoritative and unchanged:

- `OPERATORS`, `OPERATOR_REGISTRY`, `DOMAIN_POLICIES`, and `DOMAIN_REGISTRY`;
- the default `CompilerMode.OFFICIAL_V4`;
- the opt-in `CompilerMode.CLEAN_NEGATION`;
- `TRIG_COMPILERS`, including the exact v1 `sin`, `cos`, and `tan` trees;
- every Goal 1–5 source artifact and fingerprint.

The v2 surface is available only through `GRAMMAR_V2_OPERATOR_REGISTRY`,
`GRAMMAR_V2_DOMAIN_REGISTRY`, and constructors that require the keyword
`grammar_version=GrammarVersion.V2`. Omitting that keyword is an error. Passing
`GrammarVersion.V1` is also an error. Existing v1 APIs therefore cannot activate
v2 implicitly.

## Authorities

All sources were accessed on 2026-07-26.

1. Andrzej Odrzywołek, *All elementary functions from a single operator*,
   arXiv:2603.21852v2:
   <https://arxiv.org/abs/2603.21852v2>. Section 4.1 documents the reconstructed
   logarithm, the internal branch correction, extended-value paths, and
   isolated real-axis limitations. Its supplementary Part II treats recursive
   EML witnesses as partial complex maps:
   <https://arxiv.org/src/2603.21852v2/anc/SupplementaryInformation.pdf>.
2. The official public `eml_compiler_v4.py` at immutable commit
   `b3da148261199b46247306dfd92068f589778260`:
   <https://github.com/VA00/SymbolicRegressionPackage/blob/b3da148261199b46247306dfd92068f589778260/EML_toolkit/EmL_compiler/eml_compiler_v4.py>.
   It supplies the direct `e`, internal branch, `pi`, and `atan`
   constructions. Its direct normalized `asin`/`acos` form is retained as
   provenance but is not used because the current recursive EML evaluator
   becomes nonfinite at zero on that lowering.
3. NIST Digital Library of Mathematical Functions, §4.23, principal inverse
   trigonometric branches:
   - `asin`, equation 4.23.19: <https://dlmf.nist.gov/4.23.E19>;
   - `acos`, equation 4.23.22: <https://dlmf.nist.gov/4.23.E22>;
   - `atan`, equation 4.23.26: <https://dlmf.nist.gov/4.23.E26>.
4. SymPy 1.14.0 structural authority at commit
   `fe935ceb303891d1f8bea4c03b19fd9ec9464b02`:
   <https://github.com/sympy/sympy/tree/fe935ceb303891d1f8bea4c03b19fd9ec9464b02>.
   For a real atomic `x`, its reference logarithmic rewrites are
   `-I*log(I*x + sqrt(1 - x**2))` and
   `I*log(I*x + sqrt(1 - x**2)) + pi/2`.

The unused convenience expressions in the upstream compiler are not treated as
proof. The implemented `e`, `pi`, and `atan` orders come from executed direct
compiler paths. The `asin`/`acos` half-angle construction below is an
independent algebraic derivation from the same DLMF principal branches and the
audited `atan`; it is used specifically to avoid the direct lowering's
nonfinite zero path.

## Public API

```python
from geml.eml.compiler_trig import (
    GRAMMAR_V2_COMPILERS,
    eml_acos,
    eml_asin,
    eml_atan,
    eml_e,
    eml_pi,
)
from geml.spec.domains import GrammarVersion

tree = eml_atan(variable, grammar_version=GrammarVersion.V2)
```

Each constructor also accepts an explicit `mode=CompilerMode...` argument.
`OFFICIAL_V4` remains its default core mode. These functions are grammar-v2
extensions built on that core; the representation key is the tuple

```text
(grammar_version="v2", source_operator, compiler_mode, raw_tree_sha256)
```

The raw tree hash alone is not a version or mode label. This matters for `e`,
whose tree is mode independent, and for `pi`, whose public v2 tree is also an
existing internal compatibility subtree.

The v2 registry rows are `GrammarV2OperatorRecord` values carrying
`grammar_version="v2"`, both explicit compiler-mode labels, and
`approval_scope="bounded_compiler_conformance_only"`. Their EML constructions
are approved, but `enabled_for_generation` remains false: this work does not
authorize a v2 corpus or modify the immutable v1 generation registry.

## Exact constructions and order

Let:

```text
F(a, b) = eml(a, b)
L(x)    = the frozen reconstructed logarithm
J       = the frozen internal branch subtree
```

`J` is not an IR node, source leaf, public operator, or complex-domain mode.
It expands completely into `eml` and primitive `1`. The paired branch
convention satisfies `L(-1) = -J*pi`. Principal-complex numeric backends can
select the conjugate sign for both quantities; the real formulas below are
invariant when that paired sign is used consistently.

### Constants

```text
e  = exp(1)
pi = J * L(-1)
```

The exact construction order is the pinned public compiler order. No `e`,
`pi`, or `J` label remains in the returned tree.

### Principal real `atan`

The pinned direct compiler order is:

```text
ratio       = negate(divide(subtract(x, J), add(x, J)))
coefficient = divide(negate(J), integer(2))
atan(x)     = multiply(coefficient, L(ratio))
```

For finite real `x`,

```text
ratio = (J - x) / (J + x) = exp(2 J atan(x)).
```

Its principal argument is strictly between `-pi` and `pi`, so the DLMF
principal logarithm gives the range `(-pi/2, pi/2)`. The exact-zero lowering
still passes through a zero factor inside the generic multiplication macro;
that operational caveat is retained below.

### Principal real `asin` and `acos`

For `x` in `[-1,1]`, define:

```text
a = sqrt(1 + x)
b = sqrt(1 - x)
n = a - b
d = a + b
r = -((n - J*d) / (n + J*d))

asin(x) = (-J) * L(r)
acos(x) = pi/2 - asin(x)
```

Write `theta=asin(x)` on `[-pi/2,pi/2]`. Because both square roots above are
principal and nonnegative,

```text
n/d = tan(theta/2).
```

Substituting `n/d` into the audited DLMF principal-`atan` logarithm and
cancelling the common denominator yields `r`. Since `theta/2` lies in
`[-pi/4,pi/4]`, the principal logarithm remains on the intended branch and
`(-J)L(r)=theta`. DLMF 4.23.22 then gives
`acos(x)=pi/2-theta` in `[0,pi]`.

The cancellation is performed before EML lowering. This is essential at
`x=0`: it avoids materializing the zero half-angle quotient and keeps both
returned trees finite in the high-precision evaluator. It does not erase the
recursive EML square-root witness's partial-map limitation at exact endpoints.

## Exact atomic structure

SHA-256 is over the UTF-8 `emit_eml` output for `Variable("x")`, or the
constant tree for `e` and `pi`. Counts are expanded tree occurrences, a leaf
has depth zero, and every row has `reused_object_count == 0`.

| Operator | Core mode | SHA-256 | Nodes | Leaves | Depth |
|---|---|---|---:|---:|---:|
| `asin` | `official_v4` | `965217ec5f3a7ea923d5a057b177af2c95c8954d90e1bfb80d540c1779874e38` | 1,969 | 985 | 75 |
| `acos` | `official_v4` | `302e4cd1a3de8849fafc5a98f0906a398781a772a06013988b34946aa2764160` | 2,301 | 1,151 | 77 |
| `atan` | `official_v4` | `af8a08abb86f2944434462a404bd11db30090d60f65c033027fafd318bd65bea` | 659 | 330 | 57 |
| `pi` | `official_v4` | `81a8cea02c08ae4dbc33f273bebe54337f61769688eccb801ab889c5e8218cd9` | 193 | 97 | 34 |
| `e` | `official_v4` | `f126a91c1429594f4cf1aa70d70f97c267859d06df9fab7c6904598826a2574c` | 3 | 2 | 1 |
| `asin` | `clean_negation` | `3464c3900691933037f8d268453c9c7bfea5c65e4e69a3ac3f094835f7d070d5` | 3,677 | 1,839 | 117 |
| `acos` | `clean_negation` | `2fd2d8b9c9149575b573857b601372b9162d300de93c501cb51c890977b538e3` | 4,317 | 2,159 | 119 |
| `atan` | `clean_negation` | `726f4f48a0c0f381b2f289ca4c2faa29c3856a95ad3bb49112a6cb67973c917e` | 1,331 | 666 | 95 |
| `pi` | `clean_negation` | `05b4e284e85c0aba4aa1f69bdef398ded48a96d415e34285560880b7b2f8931d` | 389 | 195 | 52 |
| `e` | `clean_negation` | `f126a91c1429594f4cf1aa70d70f97c267859d06df9fab7c6904598826a2574c` | 3 | 2 | 1 |

The final node-type set is exactly a subset of `EML`, `Variable`, and `One`.
There is no node labeled `asin`, `acos`, `atan`, `sqrt`, `pi`, `e`, or `i`.

## Domain and terminal-status policy

The public mathematical branches are:

```text
atan : R       -> (-pi/2, pi/2)
asin : [-1, 1] -> [-pi/2, pi/2]
acos : [-1, 1] -> [0, pi]
```

All requested source inputs first receive one of these typed statuses:

| Status | Meaning |
|---|---|
| `valid_interior` | finite real input accepted by the source-domain policy |
| `valid_endpoint` | exact `asin`/`acos` endpoint; the lowered root reaches zero |
| `invalid_domain` | finite real `asin`/`acos` input outside `[-1,1]` |
| `nonfinite` | NaN or infinity, not a finite source input |
| `invalid_sample` | value cannot be represented as an accepted real scalar |

`classify_grammar_v2_real_input` preserves the sign of requested `+0.0` and
`-0.0` in its `zero_sign` field. A symbolic constructor cannot inspect future
variable assignments, so consumers must run the classifier before numeric
evaluation. Invalid-domain rows are never silently evaluated as complex source
expressions.

### Endpoint and signed-zero limitations

The supplementary proof certifies its square-root witness for a strictly
positive base and states `asin`/`acos` after recursive substitution on the open
interval `|x|<1`. At `x=±1`, the high-level DLMF formula has the correct
principal value, but one lowered root reaches `power(0,1/2)` and therefore uses
the existing extended-value boundary. At the pinned 100-digit audit precision,
mpmath retains a nonfinite raw result while IEEE complex128 reaches the
principal endpoint within `1e-12`; both outcomes are retained. Endpoint rows
are typed `valid_endpoint`, not ordinary interior evidence or compiler passes.

The half-angle lowering avoids the prior `L(0)` path: 100-decimal mpmath
evaluation is finite and agrees with `asin(0)=0` and `acos(0)=pi/2`. Raw-tree
`asin(+0)`/`asin(-0)` and `atan(+0)`/`atan(-0)` still do not reliably preserve
distinct IEEE output sign bits.

The requested input sign remains recorded, but no signed-zero preservation
claim is made. A new total endpoint/signed-zero contract requires ownership of
the arithmetic/verification layer; it must not be simulated by deleting or
clamping probes.

## Deterministic numeric audit

The issue-owned test performs:

- mpmath principal-complex evaluation at 100 decimal digits;
- NumPy IEEE complex128 evaluation for signed-zero and extended-value behavior;
- ordinary interior values at both signs;
- exact `asin`/`acos` endpoints;
- large positive and negative `atan`;
- typed values immediately inside and outside `[-1,1]`;
- NaN, infinity, and invalid sample classification;
- retained nonfinite and extended-intermediate observations.

The preregistered test thresholds are:

- ordinary 100-digit interior error below `1e-90`;
- exact endpoint mpmath nonfinite status retained, with the independent IEEE
  complex128 value within `1e-12` of the principal reference;
- `e` error below `1e-95`;
- `pi` error below `1e-90`;
- complex128 signed-zero magnitude below `1e-12`, with sign collapse retained.

Run the bounded audit with:

```bash
python -m pytest tests/eml/test_compiler_trig_v2.py -q
python -m geml.experiments.goal10.corpus_v2 \
  --config configs/goal10_corpus_v2.yaml \
  --output-dir outputs/final/goal10/conformance
python -m geml.experiments.goal10.rerun_studies \
  --conformance-config configs/goal10_corpus_v2.yaml \
  --audit-config configs/goal10_rerun.yaml \
  --run-dir outputs/final/goal10/conformance \
  --output-dir outputs/final/goal10/audit \
  --evidence-tier final
```

Passing sampled points supports the sourced construction; it is not a proof
that every recursive EML intermediate is total.

## E-graph boundary

`src/geml/egraph/ir.py` defines a closed `Operator` enum containing neither
trigonometric nor inverse-trigonometric operators. Its rewrite context also has
no interval assumptions such as `x in [-1,1]` or
`x in [-pi/2,pi/2]`. The conformance-only Goal 10 scope does not require
either contract.

`GRAMMAR_V2_RULE_CAPABILITIES` records the desired identities, guards, version,
and missing executable capabilities as nonexecutable metadata. It does not add
opaque string leaves, pretend another operator is inverse trig, or modify the
v1 `DOMAIN_RULES`. Executable rules require an explicit ownership extension for
at least:

1. the e-graph `Operator` enum, arity map, parsers, and serializers; and
2. a declared interval-assumption representation and provenance-preserving
   guards.

No executable inverse-trig rule is claimed, and this is an intentional scope
boundary rather than a blocker to compiler conformance.

## Remaining scientific-review items

1. Decide whether the project formally adopts extended-real EML evaluation at
   exact zero and square-root endpoints, or keeps those rows as partial-map
   limitations.
2. Decide whether IEEE signed-zero parity is a required numeric-output
   contract. The current pure symbolic IR has no signed-zero literal.
3. Extend the e-graph IR and interval-assumption contract only in a separately
   authorized future experiment.
4. Keep all Goal 10 results labeled grammar-v2 conformance. No v2 learning or
   Goals 1–5 regeneration follows from this compiler extension.
