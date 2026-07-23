# Corpus v2 grammar — design draft (10-0)

**Status: draft, doc only.** No registry, compiler or rule-set code in
this branch. Those files belong to 1-3, 2-4 and 4-5, and 10-0 is the
approved contract change to them — it can't touch them until all three
are merged and their write-set leases are released. This is the part
that can be settled now: what v2 adds, what it must not break, and which
decisions are still open.

Everything below is checked against the merged v1 state on `goal1` /
`goal2` (`geml.spec.operators`, `geml.spec.domains`,
`geml.eml.compiler_trig`, `docs/specs/EML_SOURCE_LEDGER.md`).

## 1. Trig is already in v1

The issue title says "extend ... with trig", which reads as if trig were
missing. It isn't. In the merged v1 registry:

| Operator | status | `enabled_for_generation` |
|---|---|---|
| `sin`, `cos`, `tan` | `approved` | **true** |
| `sinh`, `cosh`, `tanh` | `approved` | **true** |

2-4 shipped pinned pure-EML constructions for all six with exact
fingerprints (`EML_TRANSCENDENTAL_FORMULAS.md`), and the v1 generator
already emits them — `grammar.py` imports `sin`/`cos`/`tan`/`sinh`/
`cosh`/`tanh` and carries a `TAN_ARGUMENT_CLASSES` guard.

So v2 is not "add trig". It is four separate things, and they should be
argued separately because their evidence and their risk are different:

1. inverse trig (`atan`, then maybe `asin`/`acos`);
2. wider constants (π, e as **source leaves**);
3. an explicit negative-domain policy;
4. domain-aware trig rewrite rules with soundness tiers for the e-graph.

Worth a note on the issue before anyone starts the code half — otherwise
the obvious first move is to re-add operators that are already there.

## 2. Inverse trig

**`atan` first.** The pinned official compiler
(`EML-COMPILER-B3DA1482`) has a direct `eml_atan`, already listed in the
ledger's "scope used". It's total on ℝ — no pole, no argument guard, no
new domain machinery. That makes it the one v2 operator that costs
nothing but the construction work and its fingerprint audit.

**`asin`/`acos` are a bigger ask.** The pinned file has no direct
construction for either, so they'd have to be derived:

```
asin(x) = atan(x / sqrt(1 - x^2))       needs |x| < 1
acos(x) = pi/2 - asin(x)                needs pi
```

Two consequences. First, a derived construction is not a sourced one —
under the clean-room rule it stays `pending_verification` and disabled
until it has its own evidence entry, purity audit and pinned
fingerprints. Second, **`acos` depends on the π decision**, so it can't
be settled before §3 is.

Recommendation: `atan` in v2, `asin`/`acos` registered as
`pending_verification` and disabled, revisited only if a task track
actually needs them. Neither Goal 6 (equivalence) nor Goal 7 (rewrite
steps) does.

## 3. Constants — the open decision (issue #2)

Current merged state: `e` and `pi` are `pending_verification` with
`enabled_for_generation=False`; `imaginary_unit` is `reserved` and
bound to the `complex` domain, which is itself disabled.

### π and e: in, but measure first

Both are constructible from `1` under the paper's scope, so evidence
isn't the blocker — cost and blast radius are.

- **Cost is unmeasured.** A π or e leaf expands to a fixed pure-EML
  subtree. Nobody has counted it. If that subtree is large, every
  expression containing the constant carries it, and the α distribution
  moves — which is precisely the "trig may explode EML" stress test
  10-2 exists to run. So: measure the two subtrees, publish the node
  counts, *then* enable. No number goes in this doc that hasn't been
  measured.
- **π makes exact poles constructible.** Today `tan`'s argument is
  structurally confined to `[-1, 1]`, comfortably inside the poles at
  ±π/2 ≈ ±1.5708. With π available as a leaf, `tan(pi/2)` and
  `1/cos(pi/2)` become expressible. The guard must stay **structural** —
  a generator-level proof about the argument's shape, not a numeric
  probe that happens to miss the pole. Concretely: an argument
  containing a π leaf is not certifiable under the current
  `[-1, 1]` rule and must be rejected.
- **e interacts with the exp/log rules.** `log(e) -> 1` becomes an
  `ALWAYS_SAFE` rewrite, and `exp(1)` vs `e` becomes a canonicalization
  choice with a size consequence. It also adds a triviality pattern
  (`log(e)`), so 10-1's generator config needs a cap entry for it
  alongside the existing `log_one` / `exp_log` / `log_exp` ones.

### i: stays out

Recommending we close issue #2's third question as **no**, on the
grounds that i isn't a grammar addition at all:

- the `complex` domain policy is disabled and its branch conventions are
  explicitly deferred;
- 2-6's verifier tiers are real-valued — numeric probes over ℝ, symbolic
  checks under real assumptions;
- every 4-1 rule soundness label assumes real semantics. Branch cuts
  turn `ALWAYS_SAFE` entries into `GUARDED` ones case by case.

Enabling i means rewriting the verifier tiers and re-auditing the whole
rule set. That's a goal, not a grammar knob, and nothing in Goals 6–11
needs it.

One thing worth stating plainly because it looks like a contradiction:
`compiler_trig.py` **does** use an internal imaginary branch
(`eml_internal_i_branch`) inside the sin/cos/tan constructions. That is a
compiler internal built from primitive `1`. It is not a source leaf, it
doesn't appear in generated expressions, and it doesn't approve complex
source values. Keeping i `reserved` and keeping that internal branch are
not in tension.

## 4. Negative-domain policy

v1 has three enabled modes — `safe_real`, `positive_real`,
`nonzero_real` — and each already carries operation constraints. v2
doesn't add a mode; it writes down the negative-argument rules that are
currently implicit in per-operator notes, so the new operators have
something to be checked against.

| Case | v2 policy |
|---|---|
| `log` of a possibly-negative argument | forbidden structurally, in every mode. A nonzero assumption is not enough — this is already v1's rule and it carries forward unchanged |
| possibly-negative base, non-integer exponent | forbidden. Bounded approved integer exponents only — v1 rule, unchanged |
| negative argument to `sin`/`cos`/`tan`/`sinh`/`cosh`/`tanh` | allowed. These are total on ℝ (modulo `tan`'s poles) and the sign identities in §5 depend on it |
| `tan` argument | structurally certified in `[-1, 1]`; unchanged, and **not** widened by π becoming available (see §3) |
| `atan` argument | unrestricted — total on ℝ |
| `asin`/`acos` argument | would need a structural `[-1, 1]` certificate in the same style as `tan`'s. Moot while both stay disabled |
| negative constants as leaves | already allowed via bounded integers/rationals; π and e are positive, so they add nothing here |

The load-bearing word is **structural**. Every guard above is a property
the generator proves about the shape it built, not something a numeric
probe discovers afterwards. Probes report failures; they don't establish
domains.

## 5. Rewrite rules and soundness tiers

New rules for `rules_domain.py`, tiered per 4-1. Nothing here is enabled
by default until its tier is justified in the PR that adds it.

| Rule | Tier | Guard / note |
|---|---|---|
| `sin(-x) -> -sin(x)` | `ALWAYS_SAFE` | odd function, total |
| `cos(-x) -> cos(x)` | `ALWAYS_SAFE` | even function, total |
| `tan(-x) -> -tan(x)` | `ALWAYS_SAFE` | poles are symmetric, so the guard is unchanged either way |
| `sin(x)^2 + cos(x)^2 -> 1` | `ALWAYS_SAFE` | total |
| `sin(2x) -> 2 sin(x) cos(x)` | `ALWAYS_SAFE` | total |
| `cos(2x) -> 1 - 2 sin(x)^2` | `ALWAYS_SAFE` | total |
| `tan(x) -> sin(x)/cos(x)` | `GUARDED` | needs `cos(x) != 0`; the structural `[-1, 1]` guard gives it, but the rule must record the assumption rather than inherit it silently |
| `sinh(-x) -> -sinh(x)`, `cosh(-x) -> cosh(x)` | `ALWAYS_SAFE` | total |
| `atan(tan(x)) -> x` | `VERIFIED_GUARDED` | 4-1 excludes this from `SAFE_REAL` for general x, correctly. It *is* valid on `(-pi/2, pi/2)`, and our corpus `tan` arguments are structurally confined to `[-1, 1]`, which sits inside that. Only admissible if the guard is proved from the expression's structure — if the argument certificate is ever weakened, this rule has to go back to excluded |
| `sin(asin(x)) -> x` | `GUARDED` | needs `|x| <= 1`; disabled while `asin` is |
| `asin(sin(x)) -> x` | `UNSAFE` | branch-sensitive, already excluded by 4-1. Stays excluded |
| `log(e) -> 1` | `ALWAYS_SAFE` | only exists if `e` ships |
| `exp(1) -> e` | `OPTIONAL` | canonicalization direction, not a simplification — off by default |

The `atan(tan(x))` row is the interesting one and the one most likely to
be got wrong later: it's sound *because of a generator guarantee*, not
because of anything local to the rule. If 10-1 ever widens the `tan`
argument policy, this rule breaks silently. Whoever implements it should
wire the guard to the same structural certificate the generator emits,
not to a comment.

## 6. Keeping v1 byte-identical

The registry has an invariant worth knowing before planning any of this:
`validate_operator_registry()` refuses `enabled_for_generation=True` on
anything that isn't `approved`. So enabling an operator is not a flag
flip — it needs a status change, which needs a ledger entry, which needs
evidence. That's the right order and v2 shouldn't work around it.

Rules for the code half when it lands:

- new operators enter the registry **disabled**, and a v2 config enables
  them; the v1 default path produces byte-identical output;
- new constructions get pinned fingerprints in the same style as 2-4's
  table before anything is enabled;
- unverified constructions stay disabled — an operator nobody audited
  doesn't ship because the corpus needs filling;
- the ledger gets a per-entry evidence record; status never changes to
  satisfy a quota.

## 7. Gate G10 preconditions

Before 10-1 generates a single v2 expression:

1. π and e subtree node counts measured and published (§3);
2. `atan` construction audited: purity, fingerprints, numeric probes,
   with failures retained;
3. every new rule carries a tier and provenance (§5);
4. full v1 suite green, unchanged, with v2 config off;
5. issue #2 closed with a recorded decision and rationale.

## 8. Still open

- **Issue #2** — the π/e/i decision. §3 is a recommendation, not a
  verdict; it needs both groups.
- Whether `asin`/`acos` are wanted at all. Nothing in Goals 6–9 asks for
  them.
- Who lands the registry edit, and when 1-3 / 2-4 / 4-5 release their
  leases.
