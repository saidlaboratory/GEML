"""Grammar-version compatibility checks for Goal 10.

The historical module name is retained for issue compatibility.  This module
does not run a learning grid: it labels tiny compiler records, proves that v1
remains the omitted-option default, and rejects mixed-version aggregation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from geml.eml.compiler_core import CompilerMode
from geml.eml.compiler_trig import eml_atan, eml_sin
from geml.eml.emitter import emit_eml
from geml.eml.ir import Variable
from geml.spec.domains import GrammarVersion

COMPATIBILITY_SCHEMA_VERSION = "geml-goal10-compatibility-v1"


class CompatibilityError(ValueError):
    """A grammar selection or compatibility cohort is invalid."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GrammarSelection(_FrozenModel):
    """Explicit experiment-boundary grammar/compiler selection.

    The default is deliberately v1.  Grammar v2 callers must opt in and must
    retain the resulting version label on every downstream record.
    """

    grammar_version: GrammarVersion = GrammarVersion.V1
    compiler_mode: CompilerMode = CompilerMode.OFFICIAL_V4


class ImmutableReference(_FrozenModel):
    """One read-only Goals 1-5 text file and its canonical-LF SHA-256."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CompatibilityConfig(_FrozenModel):
    """Frozen compatibility-only configuration."""

    schema_version: str
    default_selection: GrammarSelection
    matrix: tuple[GrammarSelection, ...]
    immutable_v1_references: tuple[ImmutableReference, ...]
    deferred_questions: tuple[str, ...]

    @model_validator(mode="after")
    def validate_contract(self) -> CompatibilityConfig:
        if self.schema_version != COMPATIBILITY_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {COMPATIBILITY_SCHEMA_VERSION!r}")
        if self.default_selection != GrammarSelection():
            raise ValueError("the omitted-option default must remain v1/official_v4")
        expected = {
            (GrammarVersion.V1, CompilerMode.OFFICIAL_V4),
            (GrammarVersion.V1, CompilerMode.CLEAN_NEGATION),
            (GrammarVersion.V2, CompilerMode.OFFICIAL_V4),
            (GrammarVersion.V2, CompilerMode.CLEAN_NEGATION),
        }
        observed = {(row.grammar_version, row.compiler_mode) for row in self.matrix}
        if observed != expected or len(self.matrix) != len(expected):
            raise ValueError("matrix must contain each version/mode cell exactly once")
        if not self.immutable_v1_references:
            raise ValueError("at least one immutable Goals 1-5 reference is required")
        if len({row.path for row in self.immutable_v1_references}) != len(
            self.immutable_v1_references
        ):
            raise ValueError("immutable reference paths must be unique")
        if not self.deferred_questions:
            raise ValueError("deferred scientific questions must be explicit")
        return self


class CompatibilityRow(_FrozenModel):
    """One tiny downstream-compatible record with an inseparable version label."""

    schema_version: str = COMPATIBILITY_SCHEMA_VERSION
    record_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    grammar_version: GrammarVersion
    compiler_mode: CompilerMode
    representation_label: str
    source_expression: str
    pure_eml_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_label(self) -> CompatibilityRow:
        expected = (
            f"pure_eml:grammar={self.grammar_version.value}:compiler={self.compiler_mode.value}"
        )
        if self.representation_label != expected:
            raise ValueError("representation label disagrees with grammar/compiler fields")
        return self


def resolve_selection(payload: Mapping[str, Any] | None = None) -> GrammarSelection:
    """Resolve an experiment selection, with omission intentionally selecting v1."""

    return GrammarSelection.model_validate({} if payload is None else dict(payload))


def representation_label(selection: GrammarSelection) -> str:
    """Return the mandatory non-ambiguous representation label."""

    return (
        f"pure_eml:grammar={selection.grammar_version.value}:"
        f"compiler={selection.compiler_mode.value}"
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_repository_text_bytes(path: Path) -> bytes:
    """Return Git-compatible text bytes independent of checkout line endings."""

    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def make_tiny_compatibility_row(selection: GrammarSelection) -> CompatibilityRow:
    """Compile one deterministic fixture through the selected compiler path."""

    variable = Variable("x")
    if selection.grammar_version is GrammarVersion.V1:
        source_expression = "sin(x)"
        tree = eml_sin(variable, mode=selection.compiler_mode)
    else:
        source_expression = "atan(x)"
        tree = eml_atan(
            variable,
            grammar_version=GrammarVersion.V2,
            mode=selection.compiler_mode,
        )
    emitted = emit_eml(tree)
    label = representation_label(selection)
    record_id = _sha256_text(
        "\0".join(
            (
                COMPATIBILITY_SCHEMA_VERSION,
                label,
                source_expression,
                emitted,
            )
        )
    )
    return CompatibilityRow(
        record_id=record_id,
        grammar_version=selection.grammar_version,
        compiler_mode=selection.compiler_mode,
        representation_label=label,
        source_expression=source_expression,
        pure_eml_sha256=_sha256_text(emitted),
    )


def require_homogeneous_rows(rows: Iterable[CompatibilityRow]) -> tuple[CompatibilityRow, ...]:
    """Reject silent aggregation across grammar versions or compiler modes."""

    materialized = tuple(rows)
    if not materialized:
        raise CompatibilityError("compatibility aggregation cannot be empty")
    cohorts = {(row.grammar_version, row.compiler_mode) for row in materialized}
    if len(cohorts) != 1:
        raise CompatibilityError("mixed grammar/compiler rows require explicit grouped aggregation")
    return materialized


def group_rows_explicitly(
    rows: Iterable[CompatibilityRow],
) -> dict[tuple[GrammarVersion, CompilerMode], tuple[CompatibilityRow, ...]]:
    """Group mixed rows under explicit version/mode keys."""

    grouped: dict[tuple[GrammarVersion, CompilerMode], list[CompatibilityRow]] = {}
    for row in rows:
        grouped.setdefault((row.grammar_version, row.compiler_mode), []).append(row)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def audit_immutable_references(
    repository_root: str | Path,
    references: Iterable[ImmutableReference],
) -> tuple[str, ...]:
    """Return every missing, unsafe, or checksum-mismatched v1 reference."""

    root = Path(repository_root).resolve()
    failures: list[str] = []
    for reference in references:
        candidate = (root / reference.path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            failures.append(f"{reference.path}: path escapes repository root")
            continue
        if not candidate.is_file():
            failures.append(f"{reference.path}: missing")
            continue
        observed = hashlib.sha256(_canonical_repository_text_bytes(candidate)).hexdigest()
        if observed != reference.sha256:
            failures.append(
                f"{reference.path}: checksum mismatch "
                f"(expected {reference.sha256}, observed {observed})"
            )
    return tuple(failures)


def run_compatibility_checks(
    config: CompatibilityConfig,
    *,
    repository_root: str | Path,
) -> tuple[tuple[CompatibilityRow, ...], tuple[str, ...]]:
    """Run only the four tiny compiler cells and immutable-reference checks."""

    rows = tuple(make_tiny_compatibility_row(selection) for selection in config.matrix)
    failures = audit_immutable_references(
        repository_root,
        config.immutable_v1_references,
    )
    return rows, failures
