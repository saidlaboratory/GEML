"""SUPPLEMENTARY math-as-language token baselines.  Not part of the frozen six-arm grid.

The Goal 6 grid (:mod:`geml.experiments.goal6.run_grid`) is a preregistered contract: exactly four
aligned graph channels, one prefix transformer, one trivial floor, three fixed seeds.  Adding an
arm to that grid after preregistration would change what was compared without changing what was
promised, so these baselines report *alongside* the grid, never inside it.  Every registry entry
here is mechanically forced to declare ``supplementary=True`` and ``in_frozen_grid=False``.

What the baselines add is the venue-native NLP question: does the equivalence signal survive when
the expression is treated as *language* rather than structure?  Two deterministic token schemes
over the frozen 1-7 textual views:

``latex_chars``
    Character-class tokens over the 1-7 LaTeX rendering (:func:`geml.parsing.latex.render_latex`):
    commands, braces, digit runs, letter runs, single punctuation.  No learned vocabulary, no
    external tokenizer dependency.

``srepr_chars``
    The same discipline over the authoritative ``sympy_srepr`` text: constructor names, digit
    runs, single punctuation.

Both schemes feed the *unchanged* :class:`~geml.learning.backbones.prefix_transformer.
PrefixTransformerPairModel` through the same padded long-tensor interface the frozen prefix
control consumes, so sequence-vs-graph stays a representation comparison, not a model comparison.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from geml.contracts.ast import ASTTree
from geml.learning.backbones.prefix_transformer import (
    PAD_TOKEN,
    START_TOKEN,
    TokenizationStatus,
)
from geml.learning.backbones.trivial import assert_no_leaked_fields
from geml.parsing.latex import render_latex

TEXT_TOKENIZATION_VERSION = "geml-supplementary-text-tokens-v1"

LATEX_SCHEME = "latex_chars"
SREPR_SCHEME = "srepr_chars"

#: Reserved ids.  PAD and START are shared with the frozen prefix contract so the encoder's
#: padding mask and start convention apply unchanged.
UNKNOWN_TOKEN = 2
FIRST_TEXT_TOKEN = 3

# Character-class scanners.  Alternatives are ordered command > escape > digits > letters > any
# non-space, so every non-whitespace character is consumed by exactly one deterministic rule.
_LATEX_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|[0-9]+|[A-Za-z]+|\S")
_SREPR_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|\S")

# The closed vocabulary is derived from the frozen renderers, not learned from data.  Everything
# the 1-7 LaTeX adapter can emit and every srepr constructor the safe parser accepts is listed;
# multi-character tokens outside this list fall back to per-character ids, and characters outside
# it map to UNKNOWN_TOKEN.
_LATEX_COMMANDS = (
    r"\frac", r"\cdot", r"\left", r"\right", r"\mathrm", r"\mathtt",
    r"\langle", r"\rangle", r"\backslash",
    r"\exp", r"\log", r"\sin", r"\cos", r"\tan", r"\sinh", r"\cosh", r"\tanh",
    r"\{", r"\}", r"\_", r"\%", r"\$", r"\#", r"\&",
)  # fmt: skip
_SREPR_NAMES = (
    "Symbol", "Integer", "Rational", "Add", "Mul", "Pow",
    "exp", "log", "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "real", "positive", "nonzero", "True",
)  # fmt: skip
_CHARS = tuple(string.digits + string.ascii_letters + "+-^_{}()[].,='/*<>|!:;?@\"~`\\")

_VOCABULARY: tuple[str, ...] = _LATEX_COMMANDS + _SREPR_NAMES + _CHARS
_TOKEN_IDS = {token: FIRST_TEXT_TOKEN + index for index, token in enumerate(_VOCABULARY)}

#: Fits inside the prefix transformer's default 256-entry embedding, asserted by the tests.
TEXT_VOCABULARY_SIZE = FIRST_TEXT_TOKEN + len(_VOCABULARY)


class TextTokenizationError(ValueError):
    """A supplementary text sequence could not be produced."""


def tokenize_latex(text: str) -> tuple[str, ...]:
    """Split a LaTeX rendering into deterministic character-class tokens."""

    return tuple(_LATEX_TOKEN.findall(text))


def tokenize_srepr(text: str) -> tuple[str, ...]:
    """Split authoritative srepr text into deterministic character-class tokens."""

    return tuple(_SREPR_TOKEN.findall(text))


def detokenize(tokens: Sequence[str]) -> str:
    """Rejoin tokens so that re-tokenizing reproduces them exactly (the round-trip contract)."""

    return " ".join(tokens)


_SCHEME_TOKENIZERS = {LATEX_SCHEME: tokenize_latex, SREPR_SCHEME: tokenize_srepr}


def _token_ids(token: str) -> list[int]:
    known = _TOKEN_IDS.get(token)
    if known is not None:
        return [known]
    # digit runs, multi-letter identifiers, and anything unlisted spell out per character
    return [_TOKEN_IDS.get(character, UNKNOWN_TOKEN) for character in token]


@dataclass(frozen=True)
class SequenceExample:
    """One tokenized expression with its explicit outcome.

    Deliberately carries only the token ids and tokenization provenance.  No split, label, pair
    identity, or verification outcome may appear here; :func:`as_dict` re-applies the shared
    leakage guard on every serialization.
    """

    scheme: str
    tokens: tuple[int, ...]
    status: TokenizationStatus
    truncated_from: int | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "scheme": self.scheme,
            "tokens": list(self.tokens),
            "status": str(self.status),
            "truncated_from": self.truncated_from,
            "tokenization_version": TEXT_TOKENIZATION_VERSION,
        }
        assert_no_leaked_fields(payload)
        return payload


def build_sequence_example(text: str, scheme: str, max_length: int) -> SequenceExample:
    """Encode one textual rendering under the same outcome discipline as the prefix contract.

    Overlength sequences are truncated only so a batch can still form, with the original length
    recorded so the loss is auditable rather than silent.
    """

    if scheme not in _SCHEME_TOKENIZERS:
        raise TextTokenizationError(f"unknown token scheme {scheme!r}")
    if max_length < 2:
        raise TextTokenizationError("max_length must leave room for at least a start token")
    if not isinstance(text, str) or not text.strip():
        raise TextTokenizationError("cannot tokenize blank text")

    ids = [START_TOKEN]
    for token in _SCHEME_TOKENIZERS[scheme](text):
        ids.extend(_token_ids(token))
    if len(ids) > max_length:
        return SequenceExample(
            scheme,
            tuple(ids[:max_length]),
            TokenizationStatus.OVERLENGTH,
            truncated_from=len(ids),
        )
    return SequenceExample(scheme, tuple(ids), TokenizationStatus.OK)


def latex_sequence_example(tree: ASTTree, max_length: int) -> SequenceExample:
    """Encode the 1-7 LaTeX rendering of a frozen AST."""

    return build_sequence_example(render_latex(tree), LATEX_SCHEME, max_length)


def pad_examples(examples: Sequence[SequenceExample]) -> Tensor:
    """Stack examples into the padded ``[batch, length]`` long tensor the prefix encoder consumes.

    Padding uses the shared :data:`PAD_TOKEN`, so the encoder's attention and pooling masks apply
    to these sequences exactly as they do to the frozen prefix control's.
    """

    if not examples:
        raise TextTokenizationError("cannot pad an empty batch")
    width = max(len(example.tokens) for example in examples)
    batch = torch.full((len(examples), width), PAD_TOKEN, dtype=torch.long)
    for row, example in enumerate(examples):
        batch[row, : len(example.tokens)] = torch.tensor(example.tokens, dtype=torch.long)
    return batch


@dataclass(frozen=True)
class SupplementaryArmSpec:
    """A sequence baseline that reports alongside, never inside, the frozen six-arm grid.

    The Goal 6 grid was preregistered with exactly six arms and three seeds; quietly widening it
    would misrepresent what was preregistered.  ``supplementary`` and ``in_frozen_grid`` are
    validated rather than trusted so this contract cannot be flipped by a config edit.
    """

    arm_id: str
    scheme: str
    supplementary: bool = True
    in_frozen_grid: bool = False

    def __post_init__(self) -> None:
        if not self.arm_id.startswith("supplementary::"):
            raise TextTokenizationError(
                f"supplementary arm ids must carry the supplementary:: prefix, got {self.arm_id!r}"
            )
        if not self.supplementary or self.in_frozen_grid:
            raise TextTokenizationError(
                "sequence baselines are supplementary by contract and may never claim membership "
                "in the frozen six-arm grid"
            )
        if self.scheme not in _SCHEME_TOKENIZERS:
            raise TextTokenizationError(f"unknown token scheme {self.scheme!r}")

    def as_dict(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "scheme": self.scheme,
            "supplementary": self.supplementary,
            "in_frozen_grid": self.in_frozen_grid,
            "tokenization_version": TEXT_TOKENIZATION_VERSION,
        }


SUPPLEMENTARY_SEQUENCE_ARMS: tuple[SupplementaryArmSpec, ...] = (
    SupplementaryArmSpec(arm_id="supplementary::latex_tokens", scheme=LATEX_SCHEME),
    SupplementaryArmSpec(arm_id="supplementary::srepr_tokens", scheme=SREPR_SCHEME),
)
