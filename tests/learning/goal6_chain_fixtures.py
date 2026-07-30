"""Tiny hand-written pair-shard and vocabulary fixtures for the goal6 channel chain tests."""

from __future__ import annotations

import json
from pathlib import Path

from geml.ast.builder import build_ast_from_parsed
from geml.compression.macro.builder import build_macro_graph
from geml.compression.motif.mine import MotifMiningConfig, MotifMiningRecord, mine_motifs
from geml.compression.motif.vocabulary import MotifPool
from geml.contracts.corpus import CorpusSplit
from geml.data.pairs.generate import (
    ExpressionReferenceV1,
    GroupLineageV1,
    PairRecordV1,
    PairStatus,
    VerificationTier,
    write_pair_shard,
)
from geml.data.pairs.negatives import formal_counterexample_evidence
from geml.experiments.goal5.motif_sweeps import vocabulary_payload
from geml.parsing.srepr import parse_srepr

#: Expressions sharing macro substructure so mining finds real motifs at support >= 2.
SREPRS = {
    "fx-t1": "Mul(Add(Symbol('x', real=True), Integer(1)), Symbol('y', real=True))",
    "fx-t2": "Add(Mul(Symbol('x', real=True), Symbol('x', real=True)), Integer(1))",
    "fx-t3": "Pow(Add(Symbol('x', real=True), Integer(1)), Integer(2))",
    "fx-t4": "Add(Symbol('x', real=True), Integer(2))",
    "fx-v1": "Add(Symbol('x', real=True), Integer(3))",
    "fx-v2": "Mul(Symbol('x', real=True), Integer(3))",
    "fx-i1": "Add(Symbol('y', real=True), Integer(1))",
    "fx-i2": "Mul(Symbol('y', real=True), Integer(2))",
    "fx-o1": "Pow(Symbol('x', real=True), Integer(3))",
    "fx-o2": "Mul(Symbol('x', real=True), Integer(4))",
}

#: (split, left expression, right expression) for every accepted fixture negative pair.
PAIR_PLAN = (
    (CorpusSplit.TRAIN, "fx-t1", "fx-t2"),
    (CorpusSplit.TRAIN, "fx-t3", "fx-t4"),
    (CorpusSplit.VALIDATION, "fx-v1", "fx-v2"),
    (CorpusSplit.TEST_IID, "fx-i1", "fx-i2"),
    (CorpusSplit.TEST_OOD, "fx-o1", "fx-o2"),
)

BROKEN_SREPR = "Function('nope')(Symbol('x', real=True))"


def _endpoint(name: str, srepr: str, split: CorpusSplit) -> ExpressionReferenceV1:
    return ExpressionReferenceV1(
        expression_id=name,
        sympy_srepr=srepr,
        structural_signature=f"{name}-signature",
        domain_mode="safe_real",
        operator_family="algebraic_core",
        source_split=split,
        group=GroupLineageV1(group_id=name, source_split=split),
    )


def _negative(split: CorpusSplit, left_name: str, right_name: str) -> PairRecordV1:
    left = _endpoint(left_name, SREPRS.get(left_name, BROKEN_SREPR), split)
    right = _endpoint(right_name, SREPRS.get(right_name, BROKEN_SREPR), split)
    evidence = formal_counterexample_evidence(
        method="fixture-exact-point",
        detail="fixture endpoints differ at a shared rational point",
        witness={"x": "1", "left": f"{left_name}:2", "right": "-1"},
    )
    return PairRecordV1.create(
        left=left,
        right=right,
        label=False,
        pair_group_set=tuple(sorted({left_name, right_name})),
        source_split=split,
        evaluation_views=(),
        verification_tier=VerificationTier.FORMAL_COUNTEREXAMPLE,
        trace=None,
        non_equivalence_evidence=evidence,
        status=PairStatus.ACCEPTED,
        outcome_type=None,
        outcome_detail=None,
    )


def build_pair_shards(root: Path, *, include_broken_pair: bool = False) -> dict[str, list[str]]:
    """Write fixture pair shards; return the accepted pair ids per split value."""

    plan = list(PAIR_PLAN)
    if include_broken_pair:
        plan.append((CorpusSplit.TRAIN, "fx-t1", "fx-broken"))
    by_split: dict[CorpusSplit, list[PairRecordV1]] = {}
    for split, left, right in plan:
        by_split.setdefault(split, []).append(_negative(split, left, right))
    root.mkdir(parents=True, exist_ok=True)
    pair_ids: dict[str, list[str]] = {}
    for split, records in by_split.items():
        write_pair_shard(
            records,
            root / f"pairs-{split.value}-00000.jsonl",
            shard_id=f"geml-goal6-pairs:{split.value}:00000",
            config_digest="sha256:" + "c" * 64,
            input_digest="sha256:" + "d" * 64,
            rule_set_digest="sha256:" + "e" * 64,
            depth_by_pair_id={record.pair_id: 3 for record in records},
        )
        pair_ids[split.value] = sorted(record.pair_id for record in records)
    return pair_ids


def write_macro_vocabulary(path: Path) -> None:
    """Mine a real macro-motif vocabulary from the training fixtures and persist it."""

    records = []
    for name in ("fx-t1", "fx-t2", "fx-t3", "fx-t4"):
        tree = build_ast_from_parsed(parse_srepr(SREPRS[name]), expression_id=name)
        built = build_macro_graph(tree)
        assert built.macro_graph is not None, built.error_message
        records.append(
            MotifMiningRecord(
                expression_id=name, split=CorpusSplit.TRAIN, graph=built.macro_graph.graph
            )
        )
    mined = mine_motifs(
        records,
        MotifMiningConfig(
            pool=MotifPool.MACRO, min_size=2, max_size=3, min_support_count=2, vocabulary_limit=4
        ),
    )
    assert mined.vocabulary.templates, "the fixture must mine at least one real macro motif"
    path.write_text(json.dumps(vocabulary_payload(mined.vocabulary)), encoding="utf-8")
