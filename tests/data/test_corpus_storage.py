"""tests/data/test_corpus_storage.py - owned by 1-5. tiny fixtures only, tmp dirs only, no production data."""
import os
import pytest
from geml.data.storage.dedup import ExpressionRecord, DedupStats, deduplicate
from geml.data.storage.splits import assign_splits, SplitSizeMismatchError
from geml.data.storage.shards import write_shards, read_shard, ShardSizeError
from geml.data.storage.manifests import build_manifest, write_manifest, load_manifest, validate_manifest


# ---------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------

def test_dedup_keeps_first_occurrence_only():
    records = [
        ExpressionRecord("e1", "Add(x,1)"),
        ExpressionRecord("e2", "Add(x,1)"),  # dup of e1
        ExpressionRecord("e3", "Mul(x,2)"),
    ]
    stats = DedupStats()
    unique = list(deduplicate(records, stats))
    assert [r.expr_id for r in unique] == ["e1", "e3"]


def test_dedup_stats_never_silently_drop():
    records = [
        ExpressionRecord("e1", "Add(x,1)"),
        ExpressionRecord("e2", "Add(x,1)"),
    ]
    stats = DedupStats()
    list(deduplicate(records, stats))
    assert stats.seen == 2
    assert stats.unique == 1
    assert stats.duplicates == 1
    assert stats.duplicate_ids == ["e2"]  # exactly which id got rejected, not just a count


# ---------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------

def test_split_exact_counts():
    records = [ExpressionRecord(f"e{i}", f"s{i}") for i in range(10)]
    sizes = {"train": 6, "validation": 2, "iid_test": 1, "ood_test": 1}
    result = assign_splits(records, sizes, seed=1)
    assert {name: len(recs) for name, recs in result.splits.items()} == sizes


def test_split_deterministic_same_seed():
    records = [ExpressionRecord(f"e{i}", f"s{i}") for i in range(10)]
    sizes = {"train": 6, "validation": 2, "iid_test": 1, "ood_test": 1}
    a = assign_splits(records, sizes, seed=7)
    b = assign_splits(records, sizes, seed=7)
    assert a.splits == b.splits


def test_split_mismatch_raises():
    records = [ExpressionRecord(f"e{i}", f"s{i}") for i in range(10)]
    with pytest.raises(SplitSizeMismatchError):
        assign_splits(records, {"train": 5}, seed=1)  # 5 != 10


# ---------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------

def test_shard_chunking_and_roundtrip(tmp_path):
    records = [ExpressionRecord(f"e{i}", f"srepr_{i}") for i in range(7)]
    shards = write_shards(records, tmp_path, max_rows=3, min_rows=1, format="parquet")
    assert [s.row_count for s in shards] == [3, 3, 1]

    read_back = read_shard(shards[0].path, format="parquet")
    assert [r.expr_id for r in read_back] == ["e0", "e1", "e2"]


def test_shard_is_immutable_after_write(tmp_path):
    records = [ExpressionRecord("e0", "s0")]
    shards = write_shards(records, tmp_path, max_rows=5, min_rows=1, format="parquet")
    mode = oct(os.stat(shards[0].path).st_mode)[-3:]
    assert mode == "444"


def test_shard_resume_skips_existing(tmp_path):
    records = [ExpressionRecord(f"e{i}", f"s{i}") for i in range(6)]
    first = write_shards(records, tmp_path, max_rows=3, min_rows=1, format="parquet")
    # running again should skip both existing shards without error
    # (they're read-only, so a naive re-write would crash)
    second = write_shards(records, tmp_path, max_rows=3, min_rows=1, format="parquet", resume=True)
    assert len(first) == len(second) == 2


def test_shard_jsonl_format_roundtrip(tmp_path):
    records = [ExpressionRecord(f"e{i}", f"s{i}") for i in range(4)]
    shards = write_shards(records, tmp_path, max_rows=4, min_rows=1, format="jsonl")
    read_back = read_shard(shards[0].path, format="jsonl")
    assert [r.expr_id for r in read_back] == ["e0", "e1", "e2", "e3"]


def test_shard_max_less_than_min_raises(tmp_path):
    records = [ExpressionRecord("e0", "s0")]
    with pytest.raises(ShardSizeError):
        write_shards(records, tmp_path, max_rows=1, min_rows=5, format="parquet")


# ---------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------

def test_manifest_valid_when_shards_intact(tmp_path):
    records = [ExpressionRecord(f"e{i}", f"s{i}") for i in range(5)]
    shards = write_shards(records, tmp_path, max_rows=5, min_rows=1, format="parquet")
    manifest = build_manifest(shards)
    write_manifest(manifest, tmp_path / "manifest.json")

    loaded = load_manifest(tmp_path / "manifest.json")
    result = validate_manifest(loaded)
    assert result.valid
    assert result.errors == []


def test_manifest_catches_corrupted_shard(tmp_path):
    records = [ExpressionRecord(f"e{i}", f"s{i}") for i in range(5)]
    shards = write_shards(records, tmp_path, max_rows=5, min_rows=1, format="parquet")
    manifest = build_manifest(shards)

    # corrupt the shard after the manifest was built (make writable first, it's read-only)
    os.chmod(shards[0].path, 0o644)
    with open(shards[0].path, "wb") as f:
        f.write(b"not a real parquet file")

    result = validate_manifest(manifest)
    assert not result.valid
    assert any("checksum mismatch" in e for e in result.errors)


def test_manifest_catches_missing_shard(tmp_path):
    records = [ExpressionRecord(f"e{i}", f"s{i}") for i in range(5)]
    shards = write_shards(records, tmp_path, max_rows=5, min_rows=1, format="parquet")
    manifest = build_manifest(shards)

    os.chmod(shards[0].path, 0o644)
    shards[0].path.unlink()

    result = validate_manifest(manifest)
    assert not result.valid
    assert any("missing shard" in e for e in result.errors)


def test_manifest_captures_metadata(tmp_path):
    records = [ExpressionRecord("e0", "s0")]
    shards = write_shards(records, tmp_path, max_rows=5, min_rows=1, format="parquet")
    manifest = build_manifest(shards)

    assert manifest.total_rows == 1
    assert "pyarrow" in manifest.package_versions
    assert manifest.python_version  # non-empty
    assert manifest.timestamp > 0
    # git_commit_hash may be None (e.g. tmp_path isn't a git repo) - that's
    # allowed per spec ("when available"), just shouldn't crash either way
