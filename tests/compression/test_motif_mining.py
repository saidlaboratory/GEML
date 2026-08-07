"""Tests for train-only exact rooted DAG motif discovery."""

from __future__ import annotations

from dataclasses import replace

import pytest

from geml.compression.motif.mine import (
    MotifMiningConfig,
    MotifMiningRecord,
    mine_motifs,
)
from geml.compression.motif.vocabulary import (
    MotifChildRef,
    MotifNode,
    MotifPool,
    MotifTargetKind,
    build_motif_template,
    build_motif_vocabulary,
)
from geml.contracts.corpus import CorpusSplit
from geml.graph.schema import (
    EML_FAMILY,
    EML_ONE_KIND,
    MACRO_FAMILY,
    ChildRef,
    Graph,
    GraphNode,
    GraphRoot,
)

_MODE = "macro:official_v4"


def _macro_node(
    node_id: str,
    label: str,
    *children: tuple[int, str],
) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        family=MACRO_FAMILY,
        kind="rule" if children else "leaf",
        label=label,
        value=None if children else label,
        children=tuple(ChildRef(slot, target) for slot, target in children),
    )


def _shared_graph(prefix: str = "") -> Graph:
    """A root with two branches sharing a non-root descendant."""

    def node(name: str) -> str:
        return f"{prefix}{name}"

    nodes = {
        node("root"): _macro_node(
            node("root"),
            "root-rule",
            (0, node("left")),
            (1, node("right")),
        ),
        node("left"): _macro_node(
            node("left"),
            "left-rule",
            (0, node("shared")),
            (1, node("x")),
        ),
        node("right"): _macro_node(
            node("right"),
            "right-rule",
            (0, node("shared")),
            (1, node("y")),
        ),
        node("shared"): _macro_node(
            node("shared"),
            "shared-rule",
            (0, node("z")),
        ),
        node("x"): _macro_node(node("x"), "x"),
        node("y"): _macro_node(node("y"), "y"),
        node("z"): _macro_node(node("z"), "z"),
    }
    return Graph(
        nodes,
        (GraphRoot(f"{prefix}expression", node("root"), _MODE),),
    )


def _record(
    expression_id: str,
    graph: Graph,
    split: CorpusSplit = CorpusSplit.TRAIN,
) -> MotifMiningRecord:
    return MotifMiningRecord(
        expression_id=expression_id,
        split=split,
        graph=graph,
    )


def _unshared_graph() -> Graph:
    """Unfold the shared node while preserving every root subtree signature."""

    nodes = dict(_shared_graph().nodes)
    left = nodes["left"]
    right = nodes["right"]
    shared = nodes.pop("shared")
    nodes["left-shared"] = replace(shared, node_id="left-shared")
    nodes["right-shared"] = replace(shared, node_id="right-shared")
    nodes["left"] = replace(
        left,
        children=(ChildRef(0, "left-shared"), left.children[1]),
    )
    nodes["right"] = replace(
        right,
        children=(ChildRef(0, "right-shared"), right.children[1]),
    )
    return Graph(nodes, (GraphRoot("expression", "root", _MODE),))


def test_train_only_support_is_per_graph_and_ids_ignore_node_names() -> None:
    first = _shared_graph()
    renamed = _shared_graph("renamed-")
    validation_only = Graph(
        {"special": _macro_node("special", "validation-only")},
        (GraphRoot("validation", "special", _MODE),),
    )

    result = mine_motifs(
        (
            _record("train-a", first),
            _record("train-b", renamed),
            _record("validation", validation_only, CorpusSplit.VALIDATION),
        ),
        MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=1,
            max_size=4,
            min_support_count=2,
        ),
    )

    assert result.processed_count == 3
    assert result.success_count == 2
    assert result.failure_count == 1
    assert result.failures[0].error_type == "NonTrainingRecord"
    assert all(
        node.label != "validation-only"
        for template in result.vocabulary.templates
        for node in template.nodes
    )
    assert all(template.support_count == 2 for template in result.vocabulary.templates)
    assert result.vocabulary.training_transaction_count == 2

    repeated = mine_motifs(
        (_record("train-a", first), _record("train-b", renamed)),
        MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=1,
            max_size=4,
            min_support_count=2,
        ),
    )
    assert [template.motif_id for template in repeated.vocabulary.templates] == [
        template.motif_id for template in result.vocabulary.templates
    ]
    assert repeated.vocabulary.vocabulary_id == result.vocabulary.vocabulary_id


def test_dag_occurrence_preserves_shared_internal_reference() -> None:
    result = mine_motifs(
        (_record("a", _shared_graph()), _record("b", _shared_graph("b-"))),
        MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=4,
            max_size=4,
            min_support_count=2,
        ),
    )

    root_template = next(
        template
        for template in result.vocabulary.templates
        if template.nodes[0].label == "root-rule"
        and any(node.label == "shared-rule" for node in template.nodes)
    )
    shared_index = next(
        index for index, node in enumerate(root_template.nodes) if node.label == "shared-rule"
    )
    internal_targets = [
        child.target_index
        for node in root_template.nodes
        for child in node.children
        if child.target_kind is MotifTargetKind.INTERNAL and child.target_index == shared_index
    ]

    assert internal_targets == [shared_index, shared_index]
    assert root_template.support_count == 2
    assert root_template.occurrence_count == 2


def test_boundary_slots_follow_order_and_merge_repeated_targets() -> None:
    repeated = Graph(
        {
            "root": _macro_node(
                "root",
                "pair",
                (0, "shared"),
                (1, "shared"),
            ),
            "shared": _macro_node("shared", "x"),
        },
        (GraphRoot("expression", "root", _MODE),),
    )
    result = mine_motifs(
        (_record("repeated", repeated),),
        MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=1,
            max_size=1,
            min_support_count=1,
        ),
    )

    template = next(
        template for template in result.vocabulary.templates if template.nodes[0].label == "pair"
    )
    assert template.boundary_count == 1
    assert [child.target_index for child in template.nodes[0].children] == [0, 0]
    assert [child.slot for child in template.nodes[0].children] == [0, 1]


def test_shared_descendant_with_an_outside_parent_is_not_internalized_early() -> None:
    result = mine_motifs(
        (_record("shared", _shared_graph()),),
        MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=2,
            max_size=2,
            min_support_count=1,
        ),
    )

    assert not any(
        template.nodes[0].label == "left-rule"
        and any(node.label == "shared-rule" for node in template.nodes[1:])
        for template in result.vocabulary.templates
    )
    left_with_x = next(
        template
        for template in result.vocabulary.templates
        if template.nodes[0].label == "left-rule"
    )
    assert [node.label for node in left_with_x.nodes] == ["left-rule", "x"]


def test_whole_transaction_support_deduplicates_roots_and_occurrences() -> None:
    graph = Graph(
        {
            "a": _macro_node("a", "same"),
            "b": _macro_node("b", "same"),
        },
        (
            GraphRoot("root-a", "a", _MODE),
            GraphRoot("root-b", "b", _MODE),
        ),
    )
    result = mine_motifs(
        (_record("multi-root", graph),),
        MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=1,
            max_size=1,
            min_support_count=1,
        ),
    )

    template = result.vocabulary.templates[0]
    assert template.support_count == 1
    assert template.occurrence_count == 2


def test_secondary_graph_root_is_an_external_entry_not_an_internal_node() -> None:
    graph = Graph(
        {
            "primary": _macro_node("primary", "primary", (0, "secondary")),
            "secondary": _macro_node("secondary", "secondary"),
        },
        (
            GraphRoot("primary-root", "primary", _MODE),
            GraphRoot("secondary-root", "secondary", _MODE),
        ),
    )
    result = mine_motifs(
        (_record("multi-entry", graph),),
        MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=2,
            max_size=2,
            min_support_count=1,
        ),
    )

    assert result.vocabulary.templates == ()


def test_duplicate_transaction_is_retained_as_a_failure() -> None:
    record = _record("duplicate", _shared_graph())
    result = mine_motifs(
        (record, record),
        MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=1,
            max_size=1,
            min_support_count=1,
        ),
    )

    assert result.processed_count == 2
    assert result.success_count == 1
    assert result.failure_count == 1
    assert result.failures[0].error_type == "DuplicateTransaction"


def test_mixed_pool_keeps_family_specific_motif_identities() -> None:
    eml = Graph(
        {"one": GraphNode("one", EML_FAMILY, EML_ONE_KIND, "1", 1)},
        (GraphRoot("eml", "one", "pure_eml:official_v4"),),
    )
    macro = Graph(
        {"one": _macro_node("one", "1")},
        (GraphRoot("macro", "one", _MODE),),
    )
    result = mine_motifs(
        (_record("eml", eml), _record("macro", macro)),
        MotifMiningConfig(
            pool=MotifPool.MIXED,
            min_size=1,
            max_size=1,
            min_support_count=1,
        ),
    )

    assert {template.source_family for template in result.vocabulary.templates} == {
        EML_FAMILY,
        MACRO_FAMILY,
    }
    assert len({template.motif_id for template in result.vocabulary.templates}) == 2


def test_callable_source_is_replayed_once_per_mined_size() -> None:
    calls = 0
    records = (_record("a", _shared_graph()), _record("b", _shared_graph("b-")))

    def source() -> tuple[MotifMiningRecord, ...]:
        nonlocal calls
        calls += 1
        return records

    result = mine_motifs(
        source,
        MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=1,
            max_size=3,
            min_support_count=2,
        ),
    )

    assert calls == 3
    assert result.failure_count == 0
    assert result.candidate_count_by_size[-1][0] == 3


def test_callable_source_detects_dag_sharing_changes_between_passes() -> None:
    calls = 0

    def changing_source() -> tuple[MotifMiningRecord, ...]:
        nonlocal calls
        calls += 1
        graph = _shared_graph() if calls == 1 else _unshared_graph()
        return (_record("same-expression", graph),)

    with pytest.raises(RuntimeError, match="changed between exact mining passes"):
        mine_motifs(
            changing_source,
            MotifMiningConfig(
                pool=MotifPool.MACRO,
                min_size=1,
                max_size=2,
                min_support_count=1,
            ),
        )


def test_motif_identity_excludes_support_and_dictionary_statistics() -> None:
    mined = mine_motifs(
        (_record("a", _shared_graph()),),
        MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=1,
            max_size=1,
            min_support_count=1,
        ),
    ).vocabulary.templates[0]
    rebuilt = build_motif_template(
        source_family=mined.source_family,
        representation_mode=mined.representation_mode,
        nodes=mined.nodes,
        boundary_count=mined.boundary_count,
        support_count=99,
        occurrence_count=101,
    )

    assert rebuilt.signature == mined.signature
    assert rebuilt.motif_id == mined.motif_id
    assert rebuilt.dictionary_cost_bits == mined.dictionary_cost_bits
    assert replace(rebuilt, support_count=98).motif_id == mined.motif_id


def test_templates_reject_noncanonical_internal_index_permutations() -> None:
    with pytest.raises(ValueError, match="canonical first-path index order"):
        build_motif_template(
            source_family=MACRO_FAMILY,
            representation_mode=_MODE,
            nodes=(
                MotifNode(
                    kind="root",
                    children=(
                        MotifChildRef(0, MotifTargetKind.INTERNAL, 2),
                        MotifChildRef(1, MotifTargetKind.INTERNAL, 1),
                    ),
                ),
                MotifNode(kind="leaf", label="right"),
                MotifNode(kind="leaf", label="left"),
            ),
            boundary_count=0,
        )


def test_templates_reject_noncanonical_boundary_slot_permutations() -> None:
    with pytest.raises(ValueError, match="canonical first-encounter slot order"):
        build_motif_template(
            source_family=MACRO_FAMILY,
            representation_mode=_MODE,
            nodes=(
                MotifNode(
                    kind="root",
                    children=(
                        MotifChildRef(0, MotifTargetKind.BOUNDARY, 1),
                        MotifChildRef(1, MotifTargetKind.BOUNDARY, 0),
                    ),
                ),
            ),
            boundary_count=2,
        )


def test_motif_node_normalizes_child_tuple_order() -> None:
    canonical = build_motif_template(
        source_family=MACRO_FAMILY,
        representation_mode=_MODE,
        nodes=(
            MotifNode(
                kind="root",
                children=(
                    MotifChildRef(0, MotifTargetKind.BOUNDARY, 0),
                    MotifChildRef(1, MotifTargetKind.BOUNDARY, 1),
                ),
            ),
        ),
        boundary_count=2,
    )
    reversed_input = build_motif_template(
        source_family=MACRO_FAMILY,
        representation_mode=_MODE,
        nodes=(
            MotifNode(
                kind="root",
                children=(
                    MotifChildRef(1, MotifTargetKind.BOUNDARY, 1),
                    MotifChildRef(0, MotifTargetKind.BOUNDARY, 0),
                ),
            ),
        ),
        boundary_count=2,
    )

    assert reversed_input == canonical


def test_vocabulary_support_is_bounded_and_covered_by_its_identity() -> None:
    shape = mine_motifs(
        (_record("a", _shared_graph()),),
        MotifMiningConfig(
            pool=MotifPool.MACRO,
            min_size=1,
            max_size=1,
            min_support_count=1,
        ),
    ).vocabulary.templates[0]
    once = build_motif_template(
        source_family=shape.source_family,
        representation_mode=shape.representation_mode,
        nodes=shape.nodes,
        boundary_count=shape.boundary_count,
        support_count=1,
        occurrence_count=1,
    )
    twice = replace(once, support_count=2, occurrence_count=2)

    once_vocabulary = build_motif_vocabulary(
        pool=MotifPool.MACRO,
        min_size=1,
        max_size=1,
        min_support_count=1,
        vocabulary_limit=None,
        training_transaction_count=2,
        processed_count=2,
        failure_count=0,
        training_fingerprint="0" * 64,
        templates=(once,),
    )
    twice_vocabulary = build_motif_vocabulary(
        pool=MotifPool.MACRO,
        min_size=1,
        max_size=1,
        min_support_count=1,
        vocabulary_limit=None,
        training_transaction_count=2,
        processed_count=2,
        failure_count=0,
        training_fingerprint="0" * 64,
        templates=(twice,),
    )

    assert once_vocabulary.vocabulary_id != twice_vocabulary.vocabulary_id
    with pytest.raises(ValueError, match="cannot exceed training_transaction_count"):
        build_motif_vocabulary(
            pool=MotifPool.MACRO,
            min_size=1,
            max_size=1,
            min_support_count=1,
            vocabulary_limit=None,
            training_transaction_count=1,
            processed_count=1,
            failure_count=0,
            training_fingerprint="0" * 64,
            templates=(twice,),
        )
