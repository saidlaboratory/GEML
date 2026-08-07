# Expression record schema

`geml.contracts.expression.ExpressionRecord` is the validated interchange record for one source
expression. It contains data shape and provenance only; it does not generate, parse, hash, or
simplify expressions.

## Authority

`sympy_srepr` is the sole authoritative structural expression. `display_text` and `latex_text`
are non-authoritative presentation fields and must never be parsed to recover structure or used
to resolve a disagreement with `sympy_srepr`.

## Fields

| Field | Required | Type | Meaning |
|---|---:|---|---|
| `expression_id` | yes | nonempty string | Stable identity assigned by the producing issue. |
| `sympy_srepr` | yes | nonempty string | Authoritative SymPy structural representation. |
| `display_text` | yes | nonempty string | Non-authoritative plain-text display. |
| `latex_text` | no | nonempty string or null | Non-authoritative LaTeX display. |
| `split` | yes | split string | One of `train`, `validation`, `test_iid`, or `test_ood`. |
| `operator_family` | yes | nonempty string | Family name supplied by the future approved registry. |
| `domain_mode` | yes | nonempty string | Domain policy name supplied by the future approved registry. |
| `variables` | yes | array of unique nonempty strings | Ordered source-variable names; an empty array is permitted for a constant-only expression. |
| `target_ast_size` | yes | nonnegative integer | Generator target size, not a measured AST statistic. |
| `target_depth` | yes | nonnegative integer | Generator target depth, not a measured AST statistic. |
| `generator_seed` | yes | integer | Seed used by the producer. |
| `generator_metadata` | yes | JSON object | JSON-compatible producer metadata; may be `{}` when no additional metadata applies. |

Field reassignment is blocked and undeclared fields are rejected. Required strings must contain
a non-whitespace character and are preserved exactly rather than silently normalized. Numeric
fields require JSON integers, so booleans are not accepted as integer values. Duplicate
variables, invalid splits, negative targets, and non-JSON metadata are rejected.
Family and domain values are intentionally not hard-coded here; their approved registries belong
to issue 1-3.

## JSON-compatible example

```json
{
  "expression_id": "expr-000001",
  "sympy_srepr": "Add(Symbol('x'), Integer(1))",
  "display_text": "x + 1",
  "latex_text": "x + 1",
  "split": "train",
  "operator_family": "algebraic_core",
  "domain_mode": "safe_real",
  "variables": ["x"],
  "target_ast_size": 3,
  "target_depth": 1,
  "generator_seed": 1729,
  "generator_metadata": {"generator": "example"}
}
```

## Scope and consumers

Corpus generation and sharding work consumes this record, issue 1-6 uses it as the source for
deterministic parsing/AST construction, and later representation goals retain its stable
`expression_id`. Operator approval, expression generation, ID derivation, SymPy parsing,
normalization, deduplication, and file I/O are out of scope.
