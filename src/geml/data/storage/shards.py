"""
shards.py - immutable shard files (parquet primary, JSONL for debugging)

owned by 1-5

writes are atomic (temp file + rename) so a crash never leaves a
half-written file sitting at the real path, and resumable (skips
shards whose final file already exists, so a re-run picks up where a
crashed run left off instead of redoing everything)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import gzip
import json
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from geml.data.storage.dedup import ExpressionRecord

# real target range for production shards. kept as defaults, not
# hardcoded, so tests can pass tiny values instead of needing 10k+ fake
# records just to exercise the chunking logic
MIN_SHARD_ROWS = 10_000
MAX_SHARD_ROWS = 25_000


class ShardSizeError(ValueError):
    pass


@dataclass
class ShardInfo:
    path: Path
    row_count: int
    format: str


def _atomic_write(path: Path, write_fn) -> None:
    """
    write to a temp file in the same directory, then os.replace() it
    into place - that's atomic on the same filesystem, so anyone
    reading `path` either sees nothing (not written yet), the complete
    old file, or the complete new file. never something half-written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    # make it read-only after writing - shards are supposed to be
    # immutable once finalized, this enforces that at the OS level too,
    # not just "we promise not to touch it"
    os.chmod(path, 0o444)


def _write_parquet(records: list[ExpressionRecord], tmp_path: Path) -> None:
    table = pa.table({
        "expr_id": [r.expr_id for r in records],
        "srepr": [r.srepr for r in records],
    })
    pq.write_table(table, tmp_path)


def _write_jsonl_gz(records: list[ExpressionRecord], tmp_path: Path) -> None:
    with gzip.open(tmp_path, "wt") as f:
        for r in records:
            f.write(json.dumps({"expr_id": r.expr_id, "srepr": r.srepr}) + "\n")


def write_shards(
    records: list[ExpressionRecord],
    output_dir: Path,
    max_rows: int = MAX_SHARD_ROWS,
    min_rows: int = MIN_SHARD_ROWS,
    format: str = "parquet",
    resume: bool = True,
) -> list[ShardInfo]:
    """
    chunks records into shard files of at most max_rows each. last
    shard can end up smaller than min_rows if the total doesn't divide
    evenly - known, accepted edge case, not treated as an error.

    resume=True (default) skips any shard whose file already exists at
    the target path instead of rewriting it - so if a run crashes
    halfway through writing 10 shards, running it again just picks up
    from wherever it stopped.
    """
    if max_rows < min_rows:
        raise ShardSizeError(f"max_rows ({max_rows}) can't be less than min_rows ({min_rows})")

    output_dir = Path(output_dir)
    ext = "parquet" if format == "parquet" else "jsonl.gz"
    writer = _write_parquet if format == "parquet" else _write_jsonl_gz

    shard_infos = []
    for i, start in enumerate(range(0, len(records), max_rows)):
        chunk = records[start : start + max_rows]
        path = output_dir / f"shard_{i:04d}.{ext}"

        if resume and path.exists():
            shard_infos.append(ShardInfo(path=path, row_count=len(chunk), format=format))
            continue

        _atomic_write(path, lambda tmp, c=chunk: writer(c, tmp))
        shard_infos.append(ShardInfo(path=path, row_count=len(chunk), format=format))

    return shard_infos


def read_shard(path: Path, format: str = "parquet") -> list[ExpressionRecord]:
    path = Path(path)
    if format == "parquet":
        table = pq.read_table(path)
        d = table.to_pydict()
        return [ExpressionRecord(eid, s) for eid, s in zip(d["expr_id"], d["srepr"])]
    elif format == "jsonl":
        out = []
        with gzip.open(path, "rt") as f:
            for line in f:
                obj = json.loads(line)
                out.append(ExpressionRecord(obj["expr_id"], obj["srepr"]))
        return out
    raise ValueError(f"unknown format: {format}")
