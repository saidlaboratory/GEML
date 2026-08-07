# Macro graph contract

Status: Goal 5 official-construction contract (`geml-macro-v1`).

## Purpose and representation boundary

A macro graph is a transparent, lossless DAG of approved **source
constructions**. Each node names the exact official constructor that expands
that source operation into pure EML. It is not itself a pure-EML graph:

- `MacroGraphRecord.is_pure_eml` is exactly `false`;
- every graph root has representation mode
  `macro:<compiler-mode>:is_pure_eml=false`; and
- macro node labels, rule names, and source payloads must never be counted as
  EML nodes.

The exact pure-EML DAG cost of the full expansion is separate metadata. It is
never called macro size, EML alpha, or a node count of the macro graph. Pure
EML produced by expansion has representation mode
`pure_eml:<compiler-mode>`.

This distinction prevents compact source-level names from being mistaken for
hidden primitive EML operations.

## Structural model

The graph follows the standard term-graph model: a node has a label and an
ordered successor list, and identical terms may share a node. Child slot
number is semantic. Repeated references are retained, so `f(x, x)` has two
child references even when both target the same interned node. The stored
graph is finite, rooted, reachable, and acyclic.

The implementation uses exact structural hash-consing. A macro node's
structural identity contains:

1. representation family `macro`;
2. kind `official_construction`;
3. source operator label;
4. the exact JSON value described below; and
5. the ordered child signatures, including every repeated slot.

Source node IDs do not enter structural identity. This makes renaming source
occurrences irrelevant while preserving the mapping back to every occurrence.
Commutative mathematical equivalence is also irrelevant: swapping child slots
changes structural identity.

Term graphs with ordered successors and explicit sharing are described by
[König and Zantema](https://www.ti.inf.uni-due.de/publications/koenig/termgraph02.pdf).
The lossless straight-line grammar interpretation—one deterministic expansion
for each nonterminal application—follows the graph-compression model surveyed
by [Maneth and Peternek](https://www.pure.ed.ac.uk/ws/portalfiles/portal/23486332/Maneth_et_al_2016_compressing_graphs.pdf).

## Node schema and official rules

Every node value is exactly:

```json
{
  "expansion_rule": "<fully-qualified stable rule ID>",
  "payload": null
}
```

Only leaves replace `null` with their canonical payload. No source occurrence
ID, compiler output, hidden helper, or display text may occur in the value.

| Source operator | Arity | Expansion rule |
| --- | ---: | --- |
| `symbol` | 0 | `geml.eml.ir.Variable` |
| `one` | 0 | `geml.eml.ir.One` |
| `integer` | 0 | `geml.eml.compiler_arithmetic.eml_integer` |
| `rational` | 0 | `geml.eml.compiler_arithmetic.eml_rational` |
| `add` | 2 | `geml.eml.compiler_core.eml_add` |
| `subtract` | 2 | `geml.eml.compiler_core.eml_subtract` |
| `multiply` | 2 | `geml.eml.compiler_arithmetic.eml_multiply` |
| `divide` | 2 | `geml.eml.compiler_arithmetic.eml_divide` |
| `negate` | 1 | `geml.eml.compiler_core.eml_negate` |
| `power` | 2 | `geml.eml.compiler_arithmetic.eml_power` |
| `exp` | 1 | `geml.eml.compiler_core.eml_exp` |
| `log` | 1 | `geml.eml.compiler_core.eml_log` |
| `sin` | 1 | `geml.eml.compiler_trig.eml_sin` |
| `cos` | 1 | `geml.eml.compiler_trig.eml_cos` |
| `tan` | 1 | `geml.eml.compiler_trig.eml_tan` |
| `sinh` | 1 | `geml.eml.compiler_transcendental.eml_sinh` |
| `cosh` | 1 | `geml.eml.compiler_transcendental.eml_cosh` |
| `tanh` | 1 | `geml.eml.compiler_transcendental.eml_tanh` |

This catalog must equal, at import time, the merged operator registry entries
that are both generation-enabled and EML-approved. A pending or disabled
operator cannot silently acquire a macro rule.

Leaf payloads are canonical:

- `symbol`: exactly `name` and a nonempty set of approved true assumptions;
- `one`: exact integer `1`;
- `integer`: an exact Python integer, never a boolean; and
- `rational`: exact integer numerator and positive denominator in lowest
  terms, with zero represented as `0/1`.

All nonleaf payloads are JSON `null`.

## Source provenance

`source_to_macro_node` maps every AST occurrence ID to its shared macro node.
`macro_to_source_nodes` is its exact inverse: it has one key for every macro
node and a sorted, nonempty tuple of all source occurrences mapped there.
`source_root_id` maps to the graph root target.

The sidecars are immutable snapshots. They are deliberately outside node
values so that repeated equal subtrees share structurally without erasing
occurrence-level auditability. The record also stores the source expression
ID and an ID-independent canonical source-AST signature. It does **not** store
the original AST or any original EML graph as a comparison shortcut.

## Exact expansion-cost metadata

The builder calls the frozen direct-hash-cons EML cost boundary under the
record's compiler mode. A successful record stores:

- pure-EML DAG node count;
- ordered child-reference count;
- leaf-zero maximum depth;
- canonical pure-EML root signature;
- compiler mode;
- pure-EML representation mode; and
- construction path `direct_hashcons`.

All fields describe the complete expanded expression. Failed, invalid, or
unsupported cost computations are returned as typed build failures; no
partial record or estimated cost is emitted.

`OFFICIAL_V4` is the default. `CLEAN_NEGATION` is available only through an
explicit compiler-mode argument and is kept under its distinct macro and
pure-EML representation labels. The two modes must not share reported
structural metrics.

## Expansion and independent validation

Expansion is deterministic and iterative:

1. validate the generic graph, macro schema, rule catalog, payloads, root,
   provenance, and mode-bound cost metadata;
2. traverse unique macro nodes in child-before-parent order;
3. dispatch solely from each stored fully qualified rule ID;
4. bind children by ascending slot without deduplicating repeated references;
5. hash-cons every produced primitive EML node under the record's explicit
   compiler mode; and
6. strictly validate the result as a pure-EML DAG.

The expanded graph may contain only:

- binary nodes labeled `eml`;
- valid source-variable leaves; and
- primitive-one leaves.

Validation then compiles the supplied, provenance-bound AST through the
independent official source-AST compiler. It compares the canonical pure-EML
root signature, DAG node count, ordered child-reference count, depth, and
representation mode. It also compares those fields with the stored expansion
metadata. Equality is structural, not merely numerical or symbolic.

Every mismatch, unsupported row, invalid input, and internal failure has a
typed status, stage, error type, and message. The streaming API processes one
record at a time and, by default, retains compact identities rather than
expanded graphs. Thus a 250k-row audit is bounded by the largest current row
unless the caller explicitly accumulates results or requests graph retention.

## Compression and motif implications

Macro graph size means the number of unique official-construction nodes and
ordered references. Pure-EML expansion size remains separate. Later motif or
grammar compression must preserve external attachment slots, rule identity,
and ordered repeated references to remain lossless. Description-length
selection must account for both dictionary and encoded-data costs, following
the minimum-description-length principle introduced by
[Rissanen](https://doi.org/10.1016/0005-1098(78)90005-5); a smaller replacement
count alone is not a proof of better compression.

## Scientific and domain assumptions

The macro layer introduces no new mathematical identity. It delegates every
formula and domain condition to the approved Goal 2 compiler and the merged
operator/domain registry. In particular, internal complex intermediates in
approved trigonometric formulas do not widen the source domain. Validation is
structural and exact; numerical agreement cannot substitute for an identical
canonical pure-EML signature.
