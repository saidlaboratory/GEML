"""
tests/dag/test_direct_eml_dag.py - owned by 3-4. tiny fixtures only.

only exp and ln are tested here since those are the only two verified
constructions available (see 3-3). add/mul/pow direct constructors
are pending 2-2/2-3/2-4's real compiler output.
"""
import math
from geml.dag.hashcons import HashConsTable
from geml.dag.direct_eml import emit_var, emit_const_one, emit_exp, emit_ln, compile_with_stats
from geml.graph.signatures import compute_signature
from geml.graph.validate import validate_graph
from geml.dag.eml import EmlNode, eml_to_dag, make_exp, make_ln, evaluate_dag, validate_eml_purity


# ---------------------------------------------------------------------
# acceptance criterion: direct signatures match post-hoc signatures
# ---------------------------------------------------------------------

def test_direct_exp_matches_posthoc_exp():
    def build(table):
        x_id = emit_var(table, "x")
        return emit_exp(table, x_id)

    direct_graph, direct_root, _ = compile_with_stats(build)
    direct_sig = compute_signature(direct_graph, direct_root)

    posthoc_graph = eml_to_dag(make_exp(EmlNode("Var", value="x")))
    posthoc_sig = compute_signature(posthoc_graph, posthoc_graph.roots[0])

    assert direct_sig == posthoc_sig


def test_direct_ln_matches_posthoc_ln():
    def build(table):
        z_id = emit_var(table, "z")
        return emit_ln(table, z_id)

    direct_graph, direct_root, _ = compile_with_stats(build)
    direct_sig = compute_signature(direct_graph, direct_root)

    posthoc_graph = eml_to_dag(make_ln(EmlNode("Var", value="z")))
    posthoc_sig = compute_signature(posthoc_graph, posthoc_graph.roots[0])

    assert direct_sig == posthoc_sig


# ---------------------------------------------------------------------
# direct construction actually shares while building, not just after
# ---------------------------------------------------------------------

def test_repeated_request_returns_same_node():
    """asking for exp(x) twice in a row should return the exact same
    node id the second time - proving sharing happens live, not as a
    separate cleanup pass afterward."""
    table = HashConsTable("eml")
    x_id = emit_var(table, "x")
    a = emit_exp(table, x_id)
    b = emit_exp(table, x_id)
    assert a == b


def test_repeated_subexpression_shared_during_direct_build():
    def build(table):
        x_id = emit_var(table, "x")
        exp_a = emit_exp(table, x_id)
        exp_b = emit_exp(table, x_id)
        return table.intern_binary("eml", exp_a, exp_b)

    graph, root, stats = compile_with_stats(build)
    # outer eml + shared exp node + x + const 1 = 4, not 7
    assert stats.final_node_count == 4
    assert validate_graph(graph).valid


# ---------------------------------------------------------------------
# construction is deterministic
# ---------------------------------------------------------------------

def test_direct_construction_is_deterministic():
    def build(table):
        x_id = emit_var(table, "x")
        exp_a = emit_exp(table, x_id)
        exp_b = emit_exp(table, x_id)
        return table.intern_binary("eml", exp_a, exp_b)

    g1, r1, _ = compile_with_stats(build)
    g2, r2, _ = compile_with_stats(build)
    assert compute_signature(g1, r1) == compute_signature(g2, r2)


# ---------------------------------------------------------------------
# construction stats are recorded and sane
# ---------------------------------------------------------------------

def test_construction_stats_recorded():
    def build(table):
        x_id = emit_var(table, "x")
        return emit_exp(table, x_id)

    _, _, stats = compile_with_stats(build)
    assert stats.elapsed_seconds >= 0
    assert stats.peak_interning_table_size == 3  # x, const 1, eml
    assert stats.final_node_count == 3


def test_peak_size_can_exceed_final_size():
    """if a node briefly exists as a duplicate candidate before being
    discarded, peak can be >= final - just confirming this is tracked,
    not asserting a specific relationship, since exact intermediate
    behavior is an implementation detail"""
    def build(table):
        x_id = emit_var(table, "x")
        exp_a = emit_exp(table, x_id)
        exp_b = emit_exp(table, x_id)
        return table.intern_binary("eml", exp_a, exp_b)

    _, _, stats = compile_with_stats(build)
    assert stats.peak_interning_table_size >= stats.final_node_count


# ---------------------------------------------------------------------
# no macro/helper nodes ever appear - only eml/Var/Const
# ---------------------------------------------------------------------

def test_no_helper_nodes_survive():
    def build(table):
        z_id = emit_var(table, "z")
        return emit_ln(table, z_id)

    graph, _, _ = compile_with_stats(build)
    for node in graph.nodes.values():
        assert node.kind in ("eml", "Var", "Const")


def test_direct_build_passes_strict_eml_purity():
    def build(table):
        z_id = emit_var(table, "z")
        return emit_ln(table, z_id)

    graph, _, _ = compile_with_stats(build)
    result = validate_eml_purity(graph)
    assert result.valid
    assert result.errors == []


# ---------------------------------------------------------------------
# evaluates to the correct real number, not just structurally identical
# ---------------------------------------------------------------------

def test_direct_exp_evaluates_correctly():
    def build(table):
        x_id = emit_var(table, "x")
        return emit_exp(table, x_id)

    graph, root, _ = compile_with_stats(build)
    val = evaluate_dag(graph, root, {"x": 2.0})
    assert abs(val - math.exp(2.0)) < 1e-9


def test_direct_ln_evaluates_correctly():
    def build(table):
        z_id = emit_var(table, "z")
        return emit_ln(table, z_id)

    graph, root, _ = compile_with_stats(build)
    val = evaluate_dag(graph, root, {"z": 3.5})
    assert abs(val - math.log(3.5)) < 1e-9
