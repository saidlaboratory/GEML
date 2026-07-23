"""
tests/data/test_steps.py - owned by 7-0. tiny fixtures only.

the traces here are hand-built duck-typed stand-ins for 6-1 PairRecords
(same field names), and `apply_rule` stands in for 4-3. both get swapped
for the real thing when those branches merge; the assertions don't move.
"""
from dataclasses import dataclass

from geml.data.steps.extract import (
    drop_ambiguous, extract_steps, find_ambiguous,
)
from geml.data.steps.stratify import (
    dead_rules, rule_coverage, split_violations, unverified_steps,
)


@dataclass(frozen=True)
class Step:
    rule_id: str
    site_id: str
    result_signature: str
    rule_name: str = "add_zero"
    tier: str = "ALWAYS_SAFE"
    mode: str = "SAFE_REAL"
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Trace:
    pair_id: str
    left_signature: str | None
    rule_sequence: tuple[Step, ...]
    split: str = "train"
    family: str = "algebraic"
    group_id: str = "g1"


def _two_step_trace(pair_id="p1", **kwargs):
    return Trace(
        pair_id=pair_id,
        left_signature="s0",
        rule_sequence=(Step("R1", "n0", "s1"), Step("R2", "n3", "s2")),
        **kwargs,
    )


def _replay(state, rule_id, site_id):
    """honest fake of 4-3: knows the two transitions the fixture trace uses."""
    return {("s0", "R1", "n0"): "s1", ("s1", "R2", "n3"): "s2"}.get((state, rule_id, site_id))


# ---------------------------------------------------------------------
# extraction: the state chain
# ---------------------------------------------------------------------

def test_state_chain_walks_from_the_initial_signature():
    result = extract_steps([_two_step_trace()])
    assert [(s.state_signature, s.next_signature) for s in result.steps] == [
        ("s0", "s1"), ("s1", "s2"),
    ]


def test_remaining_steps_counts_down_to_the_target():
    result = extract_steps([_two_step_trace()])
    assert [s.remaining_steps for s in result.steps] == [2, 1]


def test_trace_without_a_starting_state_is_reported_whole():
    """half-extracting a trace we can't replay would be worse than skipping it."""
    trace = Trace("p1", None, (Step("R1", "n0", "s1"),))
    result = extract_steps([trace])

    assert result.steps == []
    assert result.errors[0].error_type == "MissingInitialState"


def test_step_inherits_split_family_and_group():
    result = extract_steps([_two_step_trace(split="test_ood", family="trig", group_id="g7")])
    assert all(s.split == "test_ood" and s.family == "trig" and s.group_id == "g7"
               for s in result.steps)


# ---------------------------------------------------------------------
# replay: the acceptance criterion
# ---------------------------------------------------------------------

def test_replayed_steps_are_marked_verified():
    result = extract_steps([_two_step_trace()], apply_rule=_replay)
    assert [s.replay_status for s in result.steps] == ["verified", "verified"]
    assert result.errors == []


def test_unreplayed_steps_admit_they_are_unchecked():
    result = extract_steps([_two_step_trace()])
    assert unverified_steps(result.steps) == ("p1#0", "p1#1")


def test_replay_mismatch_stops_the_trace_and_is_recorded():
    """once a state can't be reproduced, everything after it hangs off fiction."""
    trace = Trace("p1", "s0", (Step("R1", "n0", "WRONG"), Step("R2", "n3", "s2")))
    result = extract_steps([trace], apply_rule=_replay)

    assert result.steps == []
    assert result.errors[0].error_type == "ReplayMismatch"
    assert result.counts == {"attempted": 1, "steps": 0, "errors": 1}


def test_rule_that_does_not_apply_is_a_mismatch_not_a_crash():
    trace = Trace("p1", "s0", (Step("R9", "n0", "s1"),))
    result = extract_steps([trace], apply_rule=_replay)

    assert result.steps == []
    assert "gave None" in result.errors[0].message


def test_replay_exception_is_retained():
    def explode(state, rule_id, site_id):
        raise RuntimeError("rewrite engine timeout")

    result = extract_steps([_two_step_trace()], apply_rule=explode)
    assert result.errors[0].error_type == "RuntimeError"


# ---------------------------------------------------------------------
# ambiguity
# ---------------------------------------------------------------------

def test_same_state_rule_site_with_two_outcomes_is_ambiguous():
    a = Trace("p1", "s0", (Step("R1", "n0", "s1"),))
    b = Trace("p2", "s0", (Step("R1", "n0", "s99"),))
    result = extract_steps([a, b])

    assert find_ambiguous(result.steps) == {("s0", "R1", "n0"): ("s1", "s99")}


def test_ambiguous_records_are_moved_to_errors():
    a = Trace("p1", "s0", (Step("R1", "n0", "s1"),))
    b = Trace("p2", "s0", (Step("R1", "n0", "s99"),))
    c = Trace("p3", "t0", (Step("R2", "n1", "t1"),))
    result = drop_ambiguous(extract_steps([a, b, c]))

    assert [s.step_id for s in result.steps] == ["p3#0"]
    assert {e.error_type for e in result.errors} == {"AmbiguousStep"}
    assert len(result.errors) == 2


def test_clean_extraction_has_nothing_to_drop():
    result = extract_steps([_two_step_trace()])
    assert drop_ambiguous(result).steps == result.steps


# ---------------------------------------------------------------------
# stratification and coverage
# ---------------------------------------------------------------------

def test_coverage_table_is_rarest_rule_first():
    traces = [_two_step_trace("p1"), Trace("p2", "s0", (Step("R1", "n0", "s1"),))]
    rows = rule_coverage(extract_steps(traces).steps)

    assert [r.rule_id for r in rows] == ["R2", "R1"]
    assert rows[0].total == 1 and rows[1].total == 2
    assert rows[1].per_split == {"train": 2}


def test_coverage_records_the_distance_band_a_rule_appears_at():
    bands = {
        r.rule_id: (r.min_remaining, r.max_remaining)
        for r in rule_coverage(extract_steps([_two_step_trace()]).steps)
    }
    assert bands == {"R1": (2, 2), "R2": (1, 1)}   # R2 only ever fires as the last step


def test_dead_rules_are_named_not_omitted():
    steps = extract_steps([_two_step_trace()]).steps
    assert dead_rules(steps, ["R1", "R2", "R3", "R4"]) == ("R3", "R4")
    assert dead_rules(steps, ["R1", "R2"]) == ()


def test_split_reassignment_is_caught():
    steps = extract_steps([_two_step_trace(split="train")]).steps
    assert split_violations(steps, {"p1": "train"}) == ()
    assert split_violations(steps, {"p1": "validation"}) == ("p1#0", "p1#1")
