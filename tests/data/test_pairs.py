"""
tests/data/test_pairs.py - owned by 6-1. tiny fixtures only.

the e-graph and the verifier are faked here on purpose: the point of the
partial start is that the builder's control flow, the negative matching
and the split bookkeeping are provable before 4-4/4-5/2-6 merge. when
they do, the fakes get swapped for the real objects and these tests keep
their meaning.
"""
import pytest

from geml.data.pairs.generate import (
    Equivalent, PairRecord, RuleApplication, SourceExpression, Verification,
    build_positive_pairs,
)
from geml.data.pairs.negatives import build_negative_pairs, size_matched
from geml.data.pairs.splits import CrossSplitPair, find_leaks, pair_split, tag_ood


def _expr(expr_id, *, split="train", family="algebraic", size=10, depth=3, eclass=None):
    return SourceExpression(
        expression_id=expr_id, split=split, family=family, size=size, max_depth=depth,
        signature=f"sig:{expr_id}", eclass_id=eclass,
    )


def _step(rule_id="R1", site="n0"):
    return RuleApplication(
        rule_id=rule_id, rule_name="add_zero", tier="ALWAYS_SAFE", mode="SAFE_REAL",
        site_id=site, result_signature=f"sig:{rule_id}:{site}",
    )


class FakeEngine:
    """saturation stand-in. maps expression_id -> the hits it 'reaches'."""

    def __init__(self, hits, explode=()):
        self.hits = hits
        self.explode = set(explode)

    def equivalents(self, source):
        if source.expression_id in self.explode:
            raise RuntimeError("node limit")
        return self.hits.get(source.expression_id, [])


class FakeVerifier:
    """returns whatever status the fixture asks for, defaulting to verified."""

    def __init__(self, statuses=None, default="verified"):
        self.statuses = statuses or {}
        self.default = default

    def verify(self, left, right, rule_sequence):
        status = self.statuses.get((left, right), self.default)
        return Verification(tier="egraph_proof", status=status, detail="fixture")


# ---------------------------------------------------------------------
# positives: provenance and honest failure accounting
# ---------------------------------------------------------------------

def test_positive_pair_carries_replayable_trace():
    source = _expr("e1")
    engine = FakeEngine({"e1": [Equivalent("e2", 11, 3, (_step("R1"), _step("R2", "n4")))]})
    result = build_positive_pairs([source], engine, FakeVerifier())

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.label == "equivalent"
    assert pair.step_distance == 2 == len(pair.rule_sequence)
    assert [s.rule_id for s in pair.rule_sequence] == ["R1", "R2"]
    assert pair.verification.tier == "egraph_proof"
    # 7-0 replays from left_signature; a trace without a starting state is dead weight
    assert pair.left_signature == "sig:e1"


def test_traceless_equivalent_is_rejected_not_kept():
    """a positive with no rule sequence is useless to 7-0, so it can't be a pair."""
    engine = FakeEngine({"e1": [Equivalent("e2", 10, 3, ())]})
    result = build_positive_pairs([_expr("e1")], engine, FakeVerifier())

    assert result.pairs == []
    assert result.errors[0].error_type == "EmptyRuleSequence"


def test_unverified_equivalent_becomes_an_error_row():
    engine = FakeEngine({"e1": [Equivalent("e2", 10, 3, (_step(),))]})
    verifier = FakeVerifier({("e1", "e2"): "unsupported"})
    result = build_positive_pairs([_expr("e1")], engine, verifier)

    assert result.pairs == []
    assert result.errors[0].stage == "verification"


def test_engine_failure_is_retained_and_counted():
    engine = FakeEngine({"e2": [Equivalent("e3", 10, 3, (_step(),))]}, explode=["e1"])
    result = build_positive_pairs([_expr("e1"), _expr("e2")], engine, FakeVerifier())

    assert result.counts == {"attempted": 2, "pairs": 1, "errors": 1}
    assert result.errors[0].error_type == "RuntimeError"


def test_max_per_source_caps_the_fan_out():
    hits = [Equivalent(f"e{i}", 10, 3, (_step(),)) for i in range(5)]
    result = build_positive_pairs(
        [_expr("e0")], FakeEngine({"e0": hits}), FakeVerifier(), max_per_source=2
    )
    assert len(result.pairs) == 2


# ---------------------------------------------------------------------
# negatives: size matching is the whole difficulty knob
# ---------------------------------------------------------------------

def test_size_match_picks_nearest_and_is_deterministic():
    pool = [_expr("a", size=20), _expr("b", size=12), _expr("c", size=12)]
    # b and c tie on distance; expression_id breaks it the same way every run
    assert size_matched(11, pool, tolerance=5).expression_id == "b"
    assert size_matched(11, pool, tolerance=5) is size_matched(11, list(reversed(pool)), tolerance=5)


def test_size_match_refuses_to_widen_the_window():
    assert size_matched(10, [_expr("a", size=40)], tolerance=3) is None


def test_negative_pair_needs_a_refutation():
    source = _expr("e1", size=10)
    pool = [_expr("e9", size=11)]
    result = build_negative_pairs([source], pool, FakeVerifier(default="refuted"), tolerance=2)

    assert len(result.pairs) == 1
    assert result.pairs[0].label == "not_equivalent"
    assert result.pairs[0].negative_kind == "same_family"


def test_accidentally_equivalent_candidate_is_not_labelled_negative():
    """a near-miss edit that turns out to preserve meaning gets reported, not mislabelled."""
    result = build_negative_pairs(
        [_expr("e1", size=10)], [_expr("e9", size=10)], FakeVerifier(default="verified"),
        tolerance=2,
    )
    assert result.pairs == []
    assert result.errors[0].error_type == "not_refuted:verified"


def test_negative_never_crosses_family_or_split():
    source = _expr("e1", size=10, family="trig", split="train")
    pool = [_expr("wrong_family", size=10, family="algebraic"),
            _expr("wrong_split", size=10, family="trig", split="validation")]
    result = build_negative_pairs([source], pool, FakeVerifier(default="refuted"), tolerance=2)

    assert result.pairs == []
    assert result.errors[0].error_type == "NoSizeMatch"


# ---------------------------------------------------------------------
# splits: leakage is the acceptance criterion that actually bites
# ---------------------------------------------------------------------

def _pair(pair_id, *, split, group, family="algebraic", depth=3):
    return PairRecord(
        pair_id=pair_id, left_expression_id="l", right_expression_id="r", label="equivalent",
        split=split, group_id=group, family=family, max_depth=depth, left_size=10, right_size=10,
        verification=Verification("egraph_proof", "verified"),
    )


def test_cross_split_pair_is_rejected():
    with pytest.raises(CrossSplitPair):
        pair_split("train", "validation")
    with pytest.raises(CrossSplitPair):
        pair_split("dev", "dev")
    assert pair_split("train", "train") == "train"


def test_clean_splits_report_no_leaks():
    pairs = [_pair("p1", split="train", group="g1"), _pair("p2", split="test_iid", group="g2")]
    assert find_leaks(pairs) == {}


def test_group_on_both_sides_of_a_split_is_caught():
    pairs = [_pair("p1", split="train", group="g1"), _pair("p2", split="test_iid", group="g1")]
    assert find_leaks(pairs) == {"g1": ("test_iid", "train")}


def test_ood_tags_are_measured_against_train_not_declared():
    pairs = [
        _pair("t1", split="train", group="g1", family="algebraic", depth=4),
        _pair("d1", split="test_ood", group="g2", family="algebraic", depth=9),
        _pair("f1", split="test_ood", group="g3", family="trig", depth=2),
        _pair("i1", split="test_iid", group="g4", family="algebraic", depth=3),
    ]
    tagged = {p.pair_id: p.eval_tags for p in tag_ood(pairs)}

    assert tagged["d1"] == ("depth_ood",)
    assert tagged["f1"] == ("family_ood",)
    assert tagged["i1"] == ()      # inside train's depth and family coverage
    assert tagged["t1"] == ()      # train defines the reference, can't be OOD against itself
