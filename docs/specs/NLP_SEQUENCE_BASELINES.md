# Supplementary NLP sequence baselines

Tokenization version: `geml-supplementary-text-tokens-v1`.
Module: `geml.learning.backbones.text_tokens`.

These baselines treat a mathematical expression as *language*: a flat token
stream over its textual rendering, with no access to the tree or graph
structure the other arms consume. Sequence-vs-graph is the venue-native
question for MathNLP -- if a plain text view of the formula carries the
equivalence signal, the structural representations have to earn their cost
against it, not against a strawman.

## Status: supplementary, never in the frozen grid

The Goal 6 grid (`geml.experiments.goal6.run_grid`) is preregistered: exactly
four aligned graph channels, one prefix transformer, one trivial floor, three
fixed seeds. Widening that grid after preregistration would change what was
compared without changing what was promised. These arms therefore report
*alongside* the grid in supplementary tables. `SupplementaryArmSpec` enforces
this mechanically: every entry must declare `supplementary=True` and
`in_frozen_grid=False`, and construction fails otherwise. Nothing in
`run_grid.py` is modified.

## The two token schemes

Both schemes are deterministic character-class scanners. There is no learned
vocabulary, no merges, and no external tokenizer dependency; the id table is
a fixed list derived from what the frozen renderers can emit.

- **`latex_chars`** tokenizes the issue 1-7 LaTeX rendering
  (`geml.parsing.latex.render_latex`) into commands (`\frac`, `\cdot`),
  escaped characters, braces, digit runs, letter runs, and single
  punctuation. This is the "math as it is written" view: the surface form a
  human or an NLP system would actually read.
- **`srepr_chars`** tokenizes the authoritative `sympy_srepr` text into
  constructor names, digit runs, and single punctuation. This is the
  "math as it is stored" view: fully explicit structure, but serialized as a
  linear string.

Digit runs and out-of-list identifiers spell out per character; characters
outside the fixed table map to an explicit unknown id. Round-trip contract:
joining the tokens reproduces the source text modulo whitespace, and
re-tokenizing the space-joined tokens reproduces the tokens exactly.

## Model interface

`build_sequence_example` / `pad_examples` produce the same padded
`[batch, length]` long tensor the frozen prefix control consumes, sharing its
`PAD_TOKEN` and `START_TOKEN`, so `PrefixTransformerPairModel` runs on these
sequences **unchanged**. Any difference between these arms and the prefix arm
is therefore attributable to the input view, not to the architecture.

Examples carry only token ids and tokenization provenance. The shared
leakage guard (`assert_no_leaked_fields`) is applied on every serialization,
so no split, label, pair identity, or verification outcome can enter the
feature plane.

No production numbers exist for these arms yet; any future results must state
the run that produced them.
