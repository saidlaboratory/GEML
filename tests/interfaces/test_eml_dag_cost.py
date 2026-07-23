"""tests/interfaces/test_eml_dag_cost.py - owned by 3-8. tiny fixtures only."""
import ast
import inspect
from geml.interfaces.eml_dag_cost import compute_eml_dag_cost, compute_eml_dag_cost_from_tree
from geml.dag.direct_eml import emit_var, emit_exp, emit_ln
from geml.dag.eml import EmlNode, make_exp, make_ln


# ---------------------------------------------------------------------
# basic success cases
# ---------------------------------------------------------------------

def test_direct_path_success():
    result = compute_eml_dag_cost(lambda t: emit_exp(t, emit_var(t, "x")))
    assert result.status == "success"
    assert result.node_count == 3
    assert result.canonical_signature is not None


def test_tree_path_success():
    result = compute_eml_dag_cost_from_tree(make_exp(EmlNode("Var", value="x")))
    assert result.status == "success"
    assert result.node_count == 3


# ---------------------------------------------------------------------
# acceptance criterion: cost results match goal 3 metrics on fixtures
# (both entry points agree exactly, since they're the same underlying
# graph, just built two different ways)
# ---------------------------------------------------------------------

def test_direct_and_tree_paths_agree_on_cost():
    direct = compute_eml_dag_cost(lambda t: emit_exp(t, emit_var(t, "x")))
    tree = compute_eml_dag_cost_from_tree(make_exp(EmlNode("Var", value="x")))
    assert direct.node_count == tree.node_count
    assert direct.edge_count == tree.edge_count
    assert direct.depth == tree.depth
    assert direct.canonical_signature == tree.canonical_signature


def test_repeated_subexpression_costed_correctly():
    """(exp(x), exp(x)) - direct construction should share, cost should
    reflect the shared structure, not the naive duplicated size"""
    def build(t):
        a = emit_exp(t, emit_var(t, "x"))
        b = emit_exp(t, emit_var(t, "x"))
        return t.intern_binary("eml", a, b)
    result = compute_eml_dag_cost(build)
    assert result.status == "success"
    assert result.node_count == 4  # outer eml + shared exp-node + x + const 1, not 7


# ---------------------------------------------------------------------
# failure handling - never silently drops, never crashes the caller
# ---------------------------------------------------------------------

def test_unsupported_operation_reported_as_failure():
    def broken(t):
        raise NotImplementedError("add not supported yet")
    result = compute_eml_dag_cost(broken)
    assert result.status == "failure"
    assert result.node_count is None
    assert result.error_type == "NotImplementedError"
    assert result.error_message


def test_failure_never_raises_to_caller():
    """the whole point of a stable interface: goal 4 should never need
    a try/except around this call - failures come back as data"""
    def broken(t):
        raise ValueError("deliberately broken")
    result = compute_eml_dag_cost(broken)  # must not raise
    assert result.status == "failure"


# ---------------------------------------------------------------------
# tie-breaking metadata
# ---------------------------------------------------------------------

def test_same_cost_different_signature_enables_tie_breaking():
    """exp(x) and exp(y) - genuinely same cost (same shape), different
    variable - canonical_signature is what lets goal 4 break the tie
    deterministically instead of picking arbitrarily"""
    cost_x = compute_eml_dag_cost(lambda t: emit_exp(t, emit_var(t, "x")))
    cost_y = compute_eml_dag_cost(lambda t: emit_exp(t, emit_var(t, "y")))
    assert cost_x.node_count == cost_y.node_count
    assert cost_x.canonical_signature != cost_y.canonical_signature


def test_signature_deterministic_across_repeated_calls():
    a = compute_eml_dag_cost(lambda t: emit_exp(t, emit_var(t, "x")))
    b = compute_eml_dag_cost(lambda t: emit_exp(t, emit_var(t, "x")))
    assert a.canonical_signature == b.canonical_signature


# ---------------------------------------------------------------------
# acceptance criterion: goal 4 can use this without importing goal 3
# experiment code - concretely checked by scanning this module's own
# import statements
# ---------------------------------------------------------------------

def test_module_never_imports_experiment_or_analysis_code():
    import geml.interfaces.eml_dag_cost as cost_mod
    source = inspect.getsource(cost_mod)
    tree = ast.parse(source)

    forbidden_prefixes = ("geml.experiments", "geml.analysis")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for prefix in forbidden_prefixes:
                assert not node.module.startswith(prefix), (
                    f"eml_dag_cost.py imports from {node.module!r} - "
                    f"this must only depend on graph/dag, never experiment code"
                )
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden_prefixes:
                    assert not alias.name.startswith(prefix), (
                        f"eml_dag_cost.py imports {alias.name!r} - "
                        f"this must only depend on graph/dag, never experiment code"
                    )
