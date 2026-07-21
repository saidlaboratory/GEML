# EML authoritative source ledger

This ledger records the public clean-room evidence used by
`geml.spec.operators`. It is provenance and approval metadata only: it does not
copy or implement an EML construction. Retrieval date for every source below is
**2026-07-20**.

An `approved` decision requires both an authoritative construction claim and a
source encoding compatible with a currently enabled domain policy. A mention in
an operator list is insufficient by itself. `pending_verification` means useful
primary evidence exists but the current project has not closed its mathematical
or domain review. `reserved` means the current project policy intentionally
excludes the entry.

## Source: EML-PAPER-2603.21852-V2

- **Title:** *All elementary functions from a single operator*, Andrzej
  Odrzywołek.
- **Public URL:** <https://arxiv.org/abs/2603.21852v2>
- **Immutable revision:** arXiv:2603.21852v2, submitted 2026-04-04; PDF dated
  2026-04-07.
- **Source type / authority:** primary paper; authoritative scientific source.
- **Scope used:** abstract and Section 1 for the primitive source grammar;
  Table 1 for the calculator source vocabulary; Section 3 and Table 4 for
  construction coverage; Section 4.1 for the compiler and domain caveats;
  Section 4.2 for variable terminals.
- **Relevant entries:** `symbol`, `one`, `integer`, `rational`, `add`,
  `subtract`, `multiply`, `divide`, `negate`, `power`, `exp`, `log`, `sin`,
  `cos`, `tan`, `sinh`, `cosh`, `tanh`, `e`, `pi`, `imaginary_unit`.
- **Domain caveats:** Section 4.1 says real-axis results can have isolated
  exceptional points, that trigonometric constructions use complex
  intermediates, and that principal-log branches and extended-real behavior
  require care. Those caveats prevent current trig/hyperbolic approval under the
  disabled complex policy.
- **Evidence decision:** sufficient, together with the pinned compiler, for
  guarded real algebraic, exact-number, power, `exp`, and `log` entries. Only
  pending evidence for trig/hyperbolic entries and optional `e`/`pi` source
  leaves under current project policy. `imaginary_unit` remains reserved.

## Source: EML-COMPILER-B3DA1482

- **Title:** official public `eml_compiler_v4.py` in
  `VA00/SymbolicRegressionPackage`.
- **Public URL:**
  <https://github.com/VA00/SymbolicRegressionPackage/blob/b3da148261199b46247306dfd92068f589778260/EML_toolkit/EmL_compiler/eml_compiler_v4.py>
- **Immutable revision:** commit
  [`b3da148261199b46247306dfd92068f589778260`](https://github.com/VA00/SymbolicRegressionPackage/commit/b3da148261199b46247306dfd92068f589778260),
  authored 2026-04-25.
- **Source type / authority:** official reference compiler; authoritative
  implementation evidence, not copied into GEML.
- **Scope used:** `eml_int`, `eml_rational`, `eml_sinh`, `eml_cosh`,
  `eml_tanh`, `eml_atan`, `eml_from_number`, `compile_to_eml`, and
  `normalize_to_exp_log`; the `Integer`, `Rational`, `Symbol`, `Add`, `Mul`,
  `Pow`, `exp`, and `log` dispatch branches; the declared input function map.
- **Relevant entries:** every operator in the registry.
- **Domain caveats:** compiler tests operate primarily on real grids, skip
  points outside a source function's domain, and include special handling for
  branch-sensitive constants. This project must retain such failures rather
  than treating skipped points as verification.
- **Evidence decision:** corroborates `approved` source coverage for exact
  numbers, arithmetic, bounded/guarded power, `exp`, and positive-argument
  `log`. Trig/hyperbolic functions remain `pending_verification` because their
  official lowering and branch behavior have not passed GEML Goal 2's declared
  domain audit. No upstream code or formula is reused here.

## Source: SYMPY-1.14-STRUCTURE

- **Title:** SymPy 1.14.0 core and advanced expression-manipulation
  documentation.
- **Public URLs:**
  <https://docs.sympy.org/latest/tutorials/intro-tutorial/manipulation.html> and
  <https://docs.sympy.org/latest/modules/core.html>.
- **Immutable revision:** SymPy 1.14.0 source commit
  [`fe935ceb303891d1f8bea4c03b19fd9ec9464b02`](https://github.com/sympy/sympy/tree/fe935ceb303891d1f8bea4c03b19fd9ec9464b02);
  relevant documentation source is pinned at
  [`manipulation.rst`](https://github.com/sympy/sympy/blob/fe935ceb303891d1f8bea4c03b19fd9ec9464b02/doc/src/tutorials/intro-tutorial/manipulation.rst)
  and
  [`core.rst`](https://github.com/sympy/sympy/blob/fe935ceb303891d1f8bea4c03b19fd9ec9464b02/doc/src/modules/core.rst).
- **Source type / authority:** official library documentation and pinned
  documentation source.
- **Scope used:** `srepr`; `Symbol`, `Integer`, `Rational`, `Add`, `Mul`, and
  `Pow`; structural representation of negation and reciprocal; construction
  with `evaluate=False`; function classes for `exp`, `log`, trig, and
  hyperbolic entries.
- **Relevant entries:** every operator in the registry.
- **Domain caveats:** `evaluate=False` preserves construction intent at the
  point of construction but does not prevent all later evaluation. SymPy's
  absence of dedicated subtraction, division, and negation node classes is why
  this registry records their explicit `Add`/`Mul`/`Pow` lowerings.
- **Evidence decision:** sufficient for the declared source encodings only. It
  does not establish the correctness of any pure-EML construction.

## Approval boundary

Goal 2 owns implementation and GEML-local symbolic/numeric verification of
official pure-EML formulas. A future status change must cite new evidence,
document branch and singularity behavior, enable any required domain policy,
and update both registry tests and this ledger. Status must never be changed
only to satisfy a corpus quota.
