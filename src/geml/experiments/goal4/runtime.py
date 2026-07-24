"""Deterministic runtime primitives for the Goal 4 optimization experiment.

This module owns the reusable, dependency-light machinery the Goal 4 runner needs: stable
JSON and JSONL artifacts, create-only atomic publication, resumable checkpoints, optional
resource sampling, and the source-expression to e-graph conversion.  It performs no
optimization itself; the pipeline in :mod:`geml.experiments.goal4.run` composes it with the
frozen Goal 4 interfaces.

Two optional third-party packages are treated as strictly optional so the pipeline and its
smoke test run on a fresh clone: ``psutil`` (peak memory sampling degrades to ``None``) and
any parquet reader (the runner uses JSONL exclusively).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from geml.contracts.ast import ASTTree
from geml.egraph.ir import Expr, add, const, div, exp, log, mul, neg, power, sub, var
from geml.egraph.rewrite_engine import Assumption, AssumptionEnvironment

try:  # pragma: no cover - exercised only when psutil is installed
    import psutil
except ImportError:  # pragma: no cover - the fresh-clone default
    psutil = None


class Goal4RuntimeError(RuntimeError):
    """A Goal 4 runtime artifact is missing, corrupt, or inconsistent with a resume."""


class UnsupportedSourceOperatorError(ValueError):
    """A source expression uses an operator outside the e-graph vocabulary."""


_UNARY_BUILDERS = {
    "negate": neg,
    "exp": exp,
    "log": log,
}
_BINARY_BUILDERS = {
    "add": add,
    "subtract": sub,
    "multiply": mul,
    "divide": div,
    "power": power,
}


def canonical_json(value: object) -> str:
    """Serialize a JSON value deterministically, rejecting non-finite floats."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_hex(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes, *, resume_identical: bool = True) -> Path:
    """Publish bytes create-only after an fsync.

    A resumed caller may accept an existing byte-identical artifact; different content is
    never overwritten, so completed work cannot be clobbered.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-goal4-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            if resume_identical and destination.read_bytes() == payload:
                return destination
            raise Goal4RuntimeError(
                f"immutable artifact already exists with different content: {destination}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_write_json(path: str | Path, payload: object, *, resume_identical: bool = True) -> Path:
    """Publish a stable, human-readable JSON artifact create-only."""
    encoded = (
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return atomic_write_bytes(path, encoded, resume_identical=resume_identical)


def atomic_replace_json(path: str | Path, payload: object) -> Path:
    """Atomically publish a replaceable JSON snapshot after flushing it to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".geml-goal4-checkpoint-",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_json(path: str | Path, *, label: str) -> Any:
    """Load one JSON document with a useful artifact error."""
    source = Path(path)
    if not source.is_file():
        raise Goal4RuntimeError(f"missing {label}: {source}")
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except Exception as error:
        raise Goal4RuntimeError(f"invalid {label}: {source}") from error


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> int:
    """Append rows to a JSONL result file, flushing and fsyncing before returning.

    Appending is the durable unit of progress: a row written here survives interruption and
    is what resume reads back to skip completed work.  Returns the number of rows written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _repair_trailing_jsonl_record(destination)
    written = 0
    with destination.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
            written += 1
        stream.flush()
        os.fsync(stream.fileno())
    return written


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield rows while tolerating only one trailing partial record.

    Keeping this reader lazy is important for production analysis: provenance-rich Goal 4
    rows can be large even though the parsed analysis projection is compact.
    """
    source = Path(path)
    if not source.is_file():
        return
    size = source.stat().st_size
    with source.open("rb") as stream:
        line_number = 0
        while True:
            raw = stream.readline()
            if not raw:
                break
            line_number += 1
            is_final = stream.tell() == size
            terminated = raw.endswith(b"\n")
            try:
                text = raw.decode("utf-8")
                if not text.strip():
                    raise ValueError("blank JSONL records are not permitted")
                value = json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                if is_final and not terminated:
                    break
                raise Goal4RuntimeError(
                    f"invalid JSONL record at {source}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise Goal4RuntimeError(f"JSONL record at {source}:{line_number} must be an object")
            yield value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read all rows into memory.

    Callers processing production artifacts should prefer :func:`iter_jsonl`; this eager
    compatibility helper remains useful for bounded resume state and tests.
    """
    return list(iter_jsonl(path))


def _repair_trailing_jsonl_record(path: Path) -> None:
    """Repair a crash-truncated tail before appending.

    A valid final object missing only its newline gets that newline.  An invalid final
    fragment is truncated back to the previous newline.  Corruption in any completed line
    remains a hard error and is detected by :func:`read_jsonl`.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("r+b") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        stream.seek(end - 1)
        if stream.read(1) == b"\n":
            return

        cursor = end
        start = 0
        block_size = 8192
        while cursor > 0:
            block_start = max(0, cursor - block_size)
            stream.seek(block_start)
            block = stream.read(cursor - block_start)
            newline = block.rfind(b"\n")
            if newline >= 0:
                start = block_start + newline + 1
                break
            cursor = block_start
        stream.seek(start)
        tail = stream.read(end - start)
        valid = False
        try:
            decoded = tail.decode("utf-8")
            value = json.loads(decoded)
            valid = isinstance(value, dict)
        except (UnicodeDecodeError, json.JSONDecodeError):
            valid = False
        if valid:
            stream.seek(0, os.SEEK_END)
            stream.write(b"\n")
        else:
            stream.truncate(start)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """A resource snapshot for one processed expression.

    RSS is sampled honestly before and after the unit.  It is not mislabeled as a
    per-expression peak; both values are ``None`` when ``psutil`` is unavailable.
    """

    wall_seconds: float
    cpu_seconds: float | None
    rss_bytes_before: int | None
    rss_bytes_after: int | None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "rss_bytes_before": self.rss_bytes_before,
            "rss_bytes_after": self.rss_bytes_after,
        }


def sample_process_memory() -> int | None:
    """Return current process RSS in bytes, or ``None`` when psutil is unavailable."""
    if psutil is None:
        return None
    try:  # pragma: no cover - depends on psutil availability
        return int(psutil.Process().memory_info().rss)
    except Exception:  # pragma: no cover - platform dependent
        return None


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """Resumable progress for one experiment stage.

    ``completed_ids`` names every ``(expression_id, mode)`` pair already recorded, so a
    resume recomputes nothing.  ``chunk_index`` records how many chunks have been finalized.
    """

    schema_version: str
    run_id: str
    stage: str
    total_units: int
    completed_ids: tuple[str, ...]
    chunk_index: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly copy."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "stage": self.stage,
            "total_units": self.total_units,
            "completed_ids": list(self.completed_ids),
            "chunk_index": self.chunk_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CheckpointState:
        """Rebuild a checkpoint from its JSON form."""
        required = {
            "schema_version",
            "run_id",
            "stage",
            "total_units",
            "completed_ids",
            "chunk_index",
        }
        missing = required - value.keys()
        if missing:
            raise Goal4RuntimeError(f"checkpoint is missing fields: {sorted(missing)}")
        completed = value.get("completed_ids", [])
        if not isinstance(completed, list):
            raise Goal4RuntimeError("checkpoint completed_ids must be a list")
        if any(not isinstance(item, str) or not item for item in completed):
            raise Goal4RuntimeError("checkpoint completed_ids must contain non-blank strings")
        if len(set(completed)) != len(completed):
            raise Goal4RuntimeError("checkpoint completed_ids contains duplicates")
        schema_version = value["schema_version"]
        run_id = value["run_id"]
        stage = value["stage"]
        total_units = value["total_units"]
        chunk_index = value["chunk_index"]
        if any(not isinstance(item, str) or not item for item in (schema_version, run_id, stage)):
            raise Goal4RuntimeError("checkpoint identity fields must be non-blank strings")
        if isinstance(total_units, bool) or not isinstance(total_units, int) or total_units < 0:
            raise Goal4RuntimeError("checkpoint total_units must be nonnegative")
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
            raise Goal4RuntimeError("checkpoint chunk_index must be nonnegative")
        return cls(
            schema_version=schema_version,
            run_id=run_id,
            stage=stage,
            total_units=total_units,
            completed_ids=tuple(completed),
            chunk_index=chunk_index,
        )


def unit_key(expression_id: str, mode: str) -> str:
    """Return the stable per-mode work-unit identifier used in checkpoints and rows."""
    return f"{expression_id}::{mode}"


def assumption_environment_for(domain_mode: str, variables: Iterable[str]) -> AssumptionEnvironment:
    """Return the declared assumptions implied by a corpus domain mode.

    Assumptions are declared from the frozen domain mode, never inferred from the
    expression: ``positive_real`` declares every variable positive, ``nonzero_real``
    declares every variable nonzero, and every other mode declares only real variables.
    """
    names = tuple(variables)
    if len(set(names)) != len(names):
        raise Goal4RuntimeError("source variables must be unique")
    if domain_mode == "positive_real":
        assumption: tuple[Assumption, ...] = (Assumption.POSITIVE,)
    elif domain_mode == "nonzero_real":
        assumption = (Assumption.NONZERO,)
    else:
        assumption = (Assumption.REAL,)
    return AssumptionEnvironment.of(**{name: assumption for name in names})


def ast_tree_to_expr(tree: ASTTree) -> Expr:
    """Convert a validated source :class:`ASTTree` into an e-graph expression.

    Trigonometric and hyperbolic operators have no e-graph representation and raise
    :class:`UnsupportedSourceOperatorError`, which the runner records as a retained
    unsupported-operator failure rather than dropping the row.
    """
    children_by_id: dict[str, list[tuple[int, str]]] = {node.node_id: [] for node in tree.nodes}
    for edge in tree.edges:
        children_by_id[edge.source_id].append((edge.child_slot, edge.target_id))
    node_by_id = {node.node_id: node for node in tree.nodes}

    memo: dict[str, Expr] = {}
    order: list[str] = []
    stack: list[tuple[str, bool]] = [(tree.root_id, False)]
    while stack:
        node_id, expanded = stack.pop()
        if expanded:
            order.append(node_id)
            continue
        stack.append((node_id, True))
        for _slot, child_id in sorted(children_by_id[node_id]):
            stack.append((child_id, False))

    for node_id in order:
        node = node_by_id[node_id]
        ordered_children = [memo[child_id] for _slot, child_id in sorted(children_by_id[node_id])]
        memo[node_id] = _build_expr(node.label, node.value, ordered_children)
    return memo[tree.root_id]


def _build_expr(label: str, value: object, children: list[Expr]) -> Expr:
    """Build one e-graph node from an AST label, value, and converted children."""
    if label == "symbol":
        if not isinstance(value, Mapping) or not isinstance(value.get("name"), str):
            raise UnsupportedSourceOperatorError("symbol node is missing a source name")
        return var(value["name"])
    if label == "one":
        return const(1)
    if label == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise UnsupportedSourceOperatorError("integer node has a non-integer value")
        return const(value)
    if label == "rational":
        if not isinstance(value, Mapping):
            raise UnsupportedSourceOperatorError("rational node is missing its payload")
        numerator = value.get("numerator")
        denominator = value.get("denominator")
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            raise UnsupportedSourceOperatorError("rational node has non-integer parts")
        return const(Fraction(numerator, denominator))
    if label in _UNARY_BUILDERS:
        return _UNARY_BUILDERS[label](children[0])
    if label in _BINARY_BUILDERS:
        return _BINARY_BUILDERS[label](children[0], children[1])
    raise UnsupportedSourceOperatorError(f"operator {label!r} is outside the e-graph vocabulary")


def iter_chunks(items: list[Any], chunk_size: int) -> Iterator[list[Any]]:
    """Yield successive chunks of ``items`` of at most ``chunk_size`` elements."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]
