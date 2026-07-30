"""Supplementary math-as-language baseline tests: tiny fixtures only, never ``outputs/``.

Model tests skip cleanly when the optional ``[ml]`` extra is absent, mirroring the issue 6-3
backbone tests.
"""

from __future__ import annotations

import re
from dataclasses import fields

import pytest

torch = pytest.importorskip("torch", reason="the optional [ml] extra is not installed")

from geml.ast.builder import build_ast_from_parsed  # noqa: E402
from geml.learning.backbones.prefix_transformer import (  # noqa: E402
    PAD_TOKEN,
    START_TOKEN,
    PrefixTransformerConfig,
    PrefixTransformerPairModel,
    TokenizationStatus,
)
from geml.learning.backbones.text_tokens import (  # noqa: E402
    LATEX_SCHEME,
    SREPR_SCHEME,
    SUPPLEMENTARY_SEQUENCE_ARMS,
    TEXT_VOCABULARY_SIZE,
    UNKNOWN_TOKEN,
    SequenceExample,
    SupplementaryArmSpec,
    TextTokenizationError,
    build_sequence_example,
    detokenize,
    latex_sequence_example,
    pad_examples,
    tokenize_latex,
    tokenize_srepr,
)
from geml.learning.backbones.trivial import FORBIDDEN_FEATURE_FIELDS  # noqa: E402
from geml.parsing.latex import render_latex  # noqa: E402
from geml.parsing.srepr import parse_srepr  # noqa: E402

FIXTURE_SOURCES = (
    "Symbol('x', real=True)",
    "Rational(-3, 5)",
    "Add(Symbol('x', real=True), Mul(Integer(-1), Symbol('y', real=True)))",
    "Mul(Symbol('x', real=True), Pow(Symbol('y', real=True), Integer(-1)))",
    "Pow(Add(Symbol('x', real=True), Integer(2)), Rational(3, 5))",
    "exp(Mul(Integer(-1), Pow(Symbol('x', real=True), Integer(2))))",
    "sin(Add(Symbol('longname', real=True), cos(Symbol('y', real=True))))",
)


def _tree(source: str):
    return build_ast_from_parsed(parse_srepr(source), expression_id="fixture")


def _fixture_texts() -> list[tuple[str, str]]:
    pairs = [(LATEX_SCHEME, render_latex(_tree(source))) for source in FIXTURE_SOURCES]
    pairs.extend((SREPR_SCHEME, source) for source in FIXTURE_SOURCES)
    return pairs


def test_tokenizers_are_deterministic_and_round_trip_stable() -> None:
    tokenizers = {LATEX_SCHEME: tokenize_latex, SREPR_SCHEME: tokenize_srepr}
    for scheme, text in _fixture_texts():
        tokenize = tokenizers[scheme]
        tokens = tokenize(text)
        assert tokens, text
        assert tokenize(text) == tokens
        # no character is lost or invented, and re-tokenizing the joined form is stable
        assert "".join(tokens) == re.sub(r"\s+", "", text)
        assert tokenize(detokenize(tokens)) == tokens


def test_latex_tokens_split_on_character_classes() -> None:
    text = render_latex(_tree(FIXTURE_SOURCES[3]))
    assert text == r"\frac{x}{y}"
    assert tokenize_latex(text) == ("\\frac", "{", "x", "}", "{", "y", "}")
    assert tokenize_srepr("Integer(-13)") == ("Integer", "(", "-", "13", ")")


def test_encoding_is_deterministic_and_fits_the_unchanged_prefix_model() -> None:
    default_vocabulary = PrefixTransformerConfig().vocabulary_size
    assert default_vocabulary >= TEXT_VOCABULARY_SIZE
    for scheme, text in _fixture_texts():
        example = build_sequence_example(text, scheme, max_length=512)
        assert example == build_sequence_example(text, scheme, max_length=512)
        assert example.status is TokenizationStatus.OK
        assert example.tokens[0] == START_TOKEN
        assert PAD_TOKEN not in example.tokens
        assert all(0 <= token < default_vocabulary for token in example.tokens)


def test_unknown_characters_map_to_the_explicit_unknown_token() -> None:
    example = build_sequence_example("x + π", LATEX_SCHEME, max_length=32)
    assert UNKNOWN_TOKEN in example.tokens
    assert example.status is TokenizationStatus.OK


def test_overlength_is_an_explicit_recorded_outcome() -> None:
    text = render_latex(_tree(FIXTURE_SOURCES[4]))
    example = build_sequence_example(text, LATEX_SCHEME, max_length=4)
    assert example.status is TokenizationStatus.OVERLENGTH
    assert len(example.tokens) == 4
    assert example.truncated_from is not None
    assert example.truncated_from > 4


def test_degenerate_inputs_are_rejected() -> None:
    with pytest.raises(TextTokenizationError, match="unknown token scheme"):
        build_sequence_example("x", "word_piece", max_length=8)
    with pytest.raises(TextTokenizationError, match="max_length"):
        build_sequence_example("x", LATEX_SCHEME, max_length=1)
    with pytest.raises(TextTokenizationError, match="blank"):
        build_sequence_example("   ", SREPR_SCHEME, max_length=8)
    with pytest.raises(TextTokenizationError, match="empty batch"):
        pad_examples(())


def test_padding_matches_the_frozen_tensor_interface() -> None:
    examples = [
        build_sequence_example(text, scheme, max_length=512)
        for scheme, text in _fixture_texts()
        if scheme == LATEX_SCHEME
    ]
    batch = pad_examples(examples)
    assert batch.dtype == torch.long
    assert batch.shape == (len(examples), max(len(e.tokens) for e in examples))
    for row, example in enumerate(examples):
        assert batch[row, : len(example.tokens)].tolist() == list(example.tokens)
        assert (batch[row, len(example.tokens) :] == PAD_TOKEN).all()


def test_pair_model_runs_on_latex_tokens_unchanged() -> None:
    torch.manual_seed(7)
    left = pad_examples(
        [latex_sequence_example(_tree(source), max_length=512) for source in FIXTURE_SOURCES[:4]]
    )
    right = pad_examples(
        [latex_sequence_example(_tree(source), max_length=512) for source in FIXTURE_SOURCES[3:]]
    )
    model = PrefixTransformerPairModel()
    logits = model(left, right)
    assert logits.shape == (4,)

    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and bool(g.abs().sum() > 0) for g in grads)

    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(left, right), model(right, left), atol=1e-6)


def test_no_target_bearing_field_reaches_the_feature_plane() -> None:
    field_names = {field.name for field in fields(SequenceExample)}
    assert FORBIDDEN_FEATURE_FIELDS.isdisjoint(field_names)
    example = build_sequence_example("x + y", LATEX_SCHEME, max_length=16)
    assert FORBIDDEN_FEATURE_FIELDS.isdisjoint(example.as_dict())


def test_supplementary_arms_stay_outside_the_frozen_grid() -> None:
    assert len(SUPPLEMENTARY_SEQUENCE_ARMS) == 2
    arm_ids = [arm.arm_id for arm in SUPPLEMENTARY_SEQUENCE_ARMS]
    assert len(set(arm_ids)) == len(arm_ids)
    frozen_arm_ids = {"prefix_transformer", "trivial_floor"}
    for arm in SUPPLEMENTARY_SEQUENCE_ARMS:
        assert arm.supplementary is True
        assert arm.in_frozen_grid is False
        assert arm.arm_id.startswith("supplementary::")
        assert arm.arm_id not in frozen_arm_ids
        assert not arm.arm_id.startswith("graph::")
        assert "supplementary" in arm.as_dict()

    with pytest.raises(TextTokenizationError, match="frozen six-arm grid"):
        SupplementaryArmSpec(
            arm_id="supplementary::latex_tokens", scheme=LATEX_SCHEME, in_frozen_grid=True
        )
    with pytest.raises(TextTokenizationError, match="supplementary:: prefix"):
        SupplementaryArmSpec(arm_id="latex_tokens", scheme=LATEX_SCHEME)
