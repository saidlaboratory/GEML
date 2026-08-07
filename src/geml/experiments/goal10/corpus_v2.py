"""Bounded deterministic grammar-v2 compiler conformance set.

The historical filename is retained for issue compatibility.  This module
never builds a corpus, shards, learning splits, graphs, motifs, or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import mpmath as mp
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from geml.eml.compiler_core import CompilerMode
from geml.eml.compiler_trig import (
    eml_acos,
    eml_asin,
    eml_atan,
    eml_e,
    eml_pi,
)
from geml.eml.emitter import emit_eml
from geml.eml.ir import EML, EMLTerm, One, Variable
from geml.eml.validate import validate_pure_eml
from geml.spec.domains import (
    GrammarV2InputClassification,
    GrammarV2InputStatus,
    GrammarVersion,
    classify_grammar_v2_real_input,
)

CONFORMANCE_SCHEMA_VERSION = "geml-goal10-conformance-v1"
CONFORMANCE_ID_PREFIX = "geml-goal10-conformance-record-v1"
CONFORMANCE_MAXIMUM_RECORDS = 1_000
CONFORMANCE_DOMAIN_MODE = "safe_real"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


class ConformanceError(ValueError):
    """A conformance configuration or deterministic run is invalid."""


class ProbeRegion(StrEnum):
    """Predeclared mathematical region for a compiler probe."""

    CONSTANT = "constant"
    INTERIOR = "interior"
    NEAR_BOUNDARY = "near_boundary"
    ENDPOINT = "endpoint"
    SIGNED_ZERO = "signed_zero"
    LARGE_MAGNITUDE = "large_magnitude"
    INVALID = "invalid"
    NONFINITE = "nonfinite"
    COMPOSITION = "composition"


class ExpectedCaseStatus(StrEnum):
    """Domain-level expectation declared before evaluation."""

    VALID = "valid"
    INVALID_DOMAIN = "invalid_domain"
    NONFINITE = "nonfinite"
    UNSUPPORTED = "unsupported"


class ProbeStatus(StrEnum):
    """Observed typed result for one retained probe."""

    PASS = "pass"
    PASS_WITH_EXTENDED_INTERMEDIATE = "pass_with_extended_intermediate"
    INVALID_DOMAIN = "invalid_domain"
    NONFINITE_INPUT = "nonfinite_input"
    UNSUPPORTED = "unsupported"
    COMPILE_FAILURE = "compile_failure"
    NUMERIC_FAILURE = "numeric_failure"
    NONFINITE_RESULT = "nonfinite_result"
    SINGULAR = "singular"
    OVERFLOW = "overflow"
    UNDERFLOW = "underflow"
    TIMEOUT = "timeout"


class VerifierStatus(StrEnum):
    """Independent comparison status retained on every row."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"
    UNSUPPORTED = "unsupported"


class SignedZeroStatus(StrEnum):
    """Signed-zero evidence status for a numeric probe."""

    NOT_APPLICABLE = "not_applicable"
    BACKEND_CANNOT_REPRESENT = "backend_cannot_represent"
    PRESERVED = "preserved"
    NOT_PRESERVED = "not_preserved"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ConformanceCase(_FrozenModel):
    """One predeclared source-level compiler probe."""

    case_name: str = Field(min_length=1)
    constructor: str = Field(pattern=r"^(e|pi|atan|asin|acos|atan_asin|acos_atan)$")
    input_text: str | None
    region: ProbeRegion
    expected_status: ExpectedCaseStatus

    @model_validator(mode="after")
    def validate_input_shape(self) -> ConformanceCase:
        if self.constructor in {"e", "pi"}:
            if self.input_text is not None or self.region is not ProbeRegion.CONSTANT:
                raise ValueError("constant cases require no input and the constant region")
        elif self.input_text is None:
            raise ValueError("unary and composition cases require input_text")
        if self.region is ProbeRegion.COMPOSITION and self.constructor not in {
            "atan_asin",
            "acos_atan",
        }:
            raise ValueError("composition region requires a composition constructor")
        if self.constructor in {"atan_asin", "acos_atan"} and (
            self.region is not ProbeRegion.COMPOSITION
        ):
            raise ValueError("composition constructors require the composition region")
        if (
            self.expected_status is ExpectedCaseStatus.INVALID_DOMAIN
            and self.region is not ProbeRegion.INVALID
        ):
            raise ValueError("invalid-domain cases must use the invalid region")
        if (
            self.expected_status is ExpectedCaseStatus.NONFINITE
            and self.region is not ProbeRegion.NONFINITE
        ):
            raise ValueError("nonfinite cases must use the nonfinite region")
        return self


class ConformanceConfig(_FrozenModel):
    """Frozen bounded conformance protocol."""

    schema_version: str
    seed: int = Field(ge=0)
    maximum_records: int = Field(ge=1, le=CONFORMANCE_MAXIMUM_RECORDS)
    precision_digits: int = Field(ge=50, le=500)
    case_timeout_seconds: float = Field(gt=0, le=60)
    absolute_tolerance: str
    relative_tolerance: str
    imaginary_tolerance: str
    compiler_modes: tuple[CompilerMode, ...]
    cases: tuple[ConformanceCase, ...]
    operator_quotas: dict[str, int]
    region_quotas: dict[ProbeRegion, int]
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")

    @model_validator(mode="after")
    def validate_protocol(self) -> ConformanceConfig:
        if self.schema_version != CONFORMANCE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CONFORMANCE_SCHEMA_VERSION!r}")
        if self.compiler_modes != tuple(CompilerMode):
            raise ValueError("compiler_modes must contain both modes in stable enum order")
        if not self.cases:
            raise ValueError("at least one conformance case is required")
        if len({case.case_name for case in self.cases}) != len(self.cases):
            raise ValueError("case_name values must be unique")
        expected_records = len(self.cases) * len(self.compiler_modes)
        if expected_records > self.maximum_records:
            raise ValueError("predeclared cases exceed maximum_records")
        observed_operators = Counter(case.constructor for case in self.cases)
        observed_regions = Counter(case.region for case in self.cases)
        multiplier = len(self.compiler_modes)
        if self.operator_quotas != {
            name: count * multiplier for name, count in sorted(observed_operators.items())
        }:
            raise ValueError("operator_quotas must exactly match cases across compiler modes")
        if self.region_quotas != {
            name: count * multiplier for name, count in sorted(observed_regions.items())
        }:
            raise ValueError("region_quotas must exactly match cases across compiler modes")
        with mp.workdps(self.precision_digits):
            for field_name in (
                "absolute_tolerance",
                "relative_tolerance",
                "imaginary_tolerance",
            ):
                tolerance = mp.mpf(getattr(self, field_name))
                if not mp.isfinite(tolerance) or tolerance < 0:
                    raise ValueError(f"{field_name} must be finite and nonnegative")
        return self


class NumericComparison(_FrozenModel):
    """High-precision source/EML comparison retained without float coercion."""

    reference_real: str
    reference_imaginary: str
    observed_real: str
    observed_imaginary: str
    absolute_error: str
    relative_error: str
    precision_unit_error: str
    precision_digits: int
    extended_intermediate: bool
    signed_zero_status: SignedZeroStatus


class ConformanceRecord(_FrozenModel):
    """One deterministic conformance row, including typed unsuccessful rows."""

    schema_version: str = CONFORMANCE_SCHEMA_VERSION
    record_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_name: str
    constructor: str
    source_expression: str
    input_text: str | None
    domain_mode: str = CONFORMANCE_DOMAIN_MODE
    domain_status: GrammarV2InputStatus | None
    input_zero_sign: str | None
    region: ProbeRegion
    expected_status: ExpectedCaseStatus
    observed_status: ProbeStatus
    verifier_status: VerifierStatus
    grammar_version: GrammarVersion = GrammarVersion.V2
    compiler_mode: CompilerMode
    pure_eml_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    node_count: int | None = Field(default=None, ge=1)
    depth: int | None = Field(default=None, ge=0)
    numeric: NumericComparison | None = None
    failure_type: str | None = None
    failure_message: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> ConformanceRecord:
        has_tree = self.pure_eml_fingerprint is not None
        if has_tree != (self.node_count is not None and self.depth is not None):
            raise ValueError("tree fingerprint and statistics must appear together")
        if self.observed_status not in {
            ProbeStatus.PASS,
            ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE,
        } and (self.failure_type is None or self.failure_message is None):
            raise ValueError("unsuccessful rows require a typed failure")
        if self.verifier_status is VerifierStatus.PASSED and self.numeric is None:
            raise ValueError("passed numeric verification requires a comparison")
        return self


class ConformanceManifest(_FrozenModel):
    """Stable manifest over one deterministic bounded set."""

    schema_version: str = CONFORMANCE_SCHEMA_VERSION
    benchmark_kind: str = "bounded_compiler_conformance_not_a_corpus"
    seed: int
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    implementation_commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|unknown)$")
    worktree_dirty: bool
    grammar_version: GrammarVersion = GrammarVersion.V2
    record_count: int = Field(ge=1, le=CONFORMANCE_MAXIMUM_RECORDS)
    compiler_modes: tuple[CompilerMode, ...]
    operator_counts: dict[str, int]
    region_counts: dict[ProbeRegion, int]
    status_counts: dict[ProbeStatus, int]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_metadata: dict[str, str]
    production_root: str = "outputs/final/goal10/conformance"
    learning_split: None = None


class ConformanceRun(_FrozenModel):
    """Manifest plus its ordered content rows."""

    manifest: ConformanceManifest
    records: tuple[ConformanceRecord, ...]


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for scientific identities."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _runtime_metadata() -> dict[str, str]:
    """Capture reproducibility context without placing it in content identities."""

    def distribution_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "source-tree"

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "geml": distribution_version("geml"),
        "mpmath": distribution_version("mpmath"),
        "pydantic": distribution_version("pydantic"),
        "pyyaml": distribution_version("PyYAML"),
        "platform": platform.platform(),
        "machine": platform.machine() or "unknown",
        "processor": platform.processor() or "unknown",
    }


def _git_state() -> tuple[str, bool]:
    """Return the executable revision and whether uncommitted files are present."""

    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            cwd=REPOSITORY_ROOT,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=normal"),
            check=True,
            capture_output=True,
            cwd=REPOSITORY_ROOT,
            text=True,
            timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown", True
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        return "unknown", True
    return revision, bool(status)


def load_conformance_config(path: str | Path) -> ConformanceConfig:
    """Load and strictly validate a bounded conformance YAML file."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ConformanceConfig.model_validate(payload)


def _source_expression(case: ConformanceCase) -> str:
    if case.constructor in {"e", "pi"}:
        return case.constructor
    if case.constructor == "atan_asin":
        return f"atan(asin({case.input_text}))"
    if case.constructor == "acos_atan":
        return f"acos(atan({case.input_text}))"
    return f"{case.constructor}({case.input_text})"


def _record_id(case: ConformanceCase, mode: CompilerMode) -> str:
    payload = b"\0".join(
        (
            CONFORMANCE_ID_PREFIX.encode("utf-8"),
            canonical_json_bytes(case.model_dump(mode="json")),
            mode.value.encode("utf-8"),
            GrammarVersion.V2.value.encode("utf-8"),
        )
    )
    return _sha256(payload)


def _parse_finite_input(case: ConformanceCase) -> mp.mpf:
    if case.input_text is None:
        raise ConformanceError("numeric case has no input")
    try:
        value = mp.mpf(case.input_text)
    except ValueError as error:
        raise ConformanceError(f"invalid numeric input {case.input_text!r}") from error
    if not mp.isfinite(value):
        raise ConformanceError(f"nonfinite numeric input {case.input_text!r}")
    return value


def _compile_case(case: ConformanceCase, mode: CompilerMode) -> EMLTerm:
    version = GrammarVersion.V2
    if case.constructor == "e":
        return eml_e(grammar_version=version, mode=mode)
    if case.constructor == "pi":
        return eml_pi(grammar_version=version, mode=mode)
    value = Variable("x")
    unary: dict[str, Callable[..., EMLTerm]] = {
        "atan": eml_atan,
        "asin": eml_asin,
        "acos": eml_acos,
    }
    if case.constructor in unary:
        return unary[case.constructor](value, grammar_version=version, mode=mode)
    if case.constructor == "atan_asin":
        inner = eml_asin(value, grammar_version=version, mode=mode)
        return eml_atan(inner, grammar_version=version, mode=mode)
    if case.constructor == "acos_atan":
        inner = eml_atan(value, grammar_version=version, mode=mode)
        return eml_acos(inner, grammar_version=version, mode=mode)
    raise ConformanceError(f"unsupported constructor {case.constructor!r}")


def _classify_case(case: ConformanceCase) -> GrammarV2InputClassification | None:
    if case.constructor in {"e", "pi"}:
        return None
    if case.input_text is None:
        raise ConformanceError("nonconstant case has no input")
    if case.constructor in {"atan", "asin", "acos"}:
        return classify_grammar_v2_real_input(case.constructor, case.input_text)
    inner_name, outer_name = {
        "atan_asin": ("asin", "atan"),
        "acos_atan": ("atan", "acos"),
    }[case.constructor]
    inner = classify_grammar_v2_real_input(inner_name, case.input_text)
    if inner.status not in {
        GrammarV2InputStatus.VALID_INTERIOR,
        GrammarV2InputStatus.VALID_ENDPOINT,
    }:
        return inner
    with mp.workdps(100):
        value = mp.mpf(case.input_text)
        inner_value = getattr(mp, inner_name)(value)
        return classify_grammar_v2_real_input(
            outer_name,
            mp.nstr(inner_value, 100),
        )


def _expected_status_for_domain(
    classification: GrammarV2InputClassification | None,
) -> ExpectedCaseStatus:
    if classification is None or classification.status in {
        GrammarV2InputStatus.VALID_INTERIOR,
        GrammarV2InputStatus.VALID_ENDPOINT,
    }:
        return ExpectedCaseStatus.VALID
    if classification.status is GrammarV2InputStatus.INVALID_DOMAIN:
        return ExpectedCaseStatus.INVALID_DOMAIN
    if classification.status is GrammarV2InputStatus.NONFINITE:
        return ExpectedCaseStatus.NONFINITE
    return ExpectedCaseStatus.UNSUPPORTED


def _evaluate_iterative(
    root: EMLTerm,
    *,
    variable_value: mp.mpf | None,
    deadline: float,
) -> tuple[mp.mpc, bool]:
    """Evaluate a pure EML tree without Python recursion."""

    values: dict[int, tuple[mp.mpc, bool]] = {}
    events: list[tuple[EMLTerm, bool]] = [(root, False)]
    while events:
        if time.monotonic() > deadline:
            raise TimeoutError("pure-EML numeric evaluation exceeded the per-case deadline")
        node, leaving = events.pop()
        node_id = id(node)
        if node_id in values:
            continue
        if isinstance(node, One):
            values[node_id] = (mp.mpc(1), False)
            continue
        if isinstance(node, Variable):
            if variable_value is None:
                raise ConformanceError("variable tree requires a numeric input")
            values[node_id] = (mp.mpc(variable_value), False)
            continue
        if not isinstance(node, EML):
            raise ConformanceError(f"forbidden EML node {type(node).__name__}")
        if not leaving:
            events.append((node, True))
            events.append((node.right, False))
            events.append((node.left, False))
            continue
        left, left_extended = values[id(node.left)]
        right, right_extended = values[id(node.right)]
        try:
            result = mp.exp(left) - mp.log(right)
        except ValueError as error:
            message = f"EML evaluation failed: {type(error).__name__}: {error}"
            raise ConformanceError(message) from error
        finite = bool(mp.isfinite(result.real) and mp.isfinite(result.imag))
        extended = (
            left_extended
            or right_extended
            or not mp.isfinite(left.real)
            or not mp.isfinite(left.imag)
            or not mp.isfinite(right.real)
            or not mp.isfinite(right.imag)
            or not finite
        )
        values[node_id] = (mp.mpc(result), extended)
    return values[id(root)]


def _reference_value(case: ConformanceCase, value: mp.mpf | None) -> mp.mpc:
    if case.constructor == "e":
        return mp.mpc(mp.e)
    if case.constructor == "pi":
        return mp.mpc(mp.pi)
    if value is None:
        raise ConformanceError("reference evaluator requires an input")
    functions: Mapping[str, Callable[[mp.mpf], Any]] = {
        "atan": mp.atan,
        "asin": mp.asin,
        "acos": mp.acos,
        "atan_asin": lambda x: mp.atan(mp.asin(x)),
        "acos_atan": lambda x: mp.acos(mp.atan(x)),
    }
    return mp.mpc(functions[case.constructor](value))


def _format(value: mp.mpf) -> str:
    return mp.nstr(value, n=max(20, mp.mp.dps), strip_zeros=False)


def _numeric_comparison(
    case: ConformanceCase,
    tree: EMLTerm,
    config: ConformanceConfig,
) -> tuple[NumericComparison, bool, bool]:
    value = None if case.input_text is None else _parse_finite_input(case)
    reference = _reference_value(case, value)
    observed, extended = _evaluate_iterative(
        tree,
        variable_value=value,
        deadline=time.monotonic() + config.case_timeout_seconds,
    )
    absolute_error = abs(observed - reference)
    scale = max(mp.mpf(1), abs(reference))
    relative_error = absolute_error / scale
    precision_unit = mp.power(10, -config.precision_digits)
    imaginary_error = abs(observed.imag - reference.imag)
    passed = bool(
        absolute_error <= mp.mpf(config.absolute_tolerance)
        or relative_error <= mp.mpf(config.relative_tolerance)
    ) and bool(imaginary_error <= mp.mpf(config.imaginary_tolerance))

    signed_zero_status = SignedZeroStatus.NOT_APPLICABLE
    if case.region is ProbeRegion.SIGNED_ZERO and case.constructor in {"asin", "atan"}:
        # mpmath canonicalizes +0 and -0 to the same mpf value.  Retain the
        # probe as unsupported signed-zero evidence instead of claiming a pass.
        signed_zero_status = SignedZeroStatus.BACKEND_CANNOT_REPRESENT
        passed = False

    comparison = NumericComparison(
        reference_real=_format(reference.real),
        reference_imaginary=_format(reference.imag),
        observed_real=_format(observed.real),
        observed_imaginary=_format(observed.imag),
        absolute_error=_format(absolute_error),
        relative_error=_format(relative_error),
        precision_unit_error=_format(absolute_error / precision_unit),
        precision_digits=config.precision_digits,
        extended_intermediate=extended,
        signed_zero_status=signed_zero_status,
    )
    return comparison, passed, extended


def _typed_precompile_record(
    case: ConformanceCase,
    mode: CompilerMode,
    classification: GrammarV2InputClassification | None,
    *,
    fingerprint: str,
    node_count: int,
    depth: int,
) -> ConformanceRecord | None:
    if case.expected_status is ExpectedCaseStatus.VALID:
        return None
    status = {
        ExpectedCaseStatus.INVALID_DOMAIN: ProbeStatus.INVALID_DOMAIN,
        ExpectedCaseStatus.NONFINITE: ProbeStatus.NONFINITE_INPUT,
        ExpectedCaseStatus.UNSUPPORTED: ProbeStatus.UNSUPPORTED,
    }[case.expected_status]
    return ConformanceRecord(
        record_id=_record_id(case, mode),
        case_name=case.case_name,
        constructor=case.constructor,
        source_expression=_source_expression(case),
        input_text=case.input_text,
        domain_status=None if classification is None else classification.status,
        input_zero_sign=None if classification is None else classification.zero_sign,
        region=case.region,
        expected_status=case.expected_status,
        observed_status=status,
        verifier_status=(
            VerifierStatus.NOT_RUN
            if status in {ProbeStatus.INVALID_DOMAIN, ProbeStatus.NONFINITE_INPUT}
            else VerifierStatus.UNSUPPORTED
        ),
        compiler_mode=mode,
        pure_eml_fingerprint=fingerprint,
        node_count=node_count,
        depth=depth,
        failure_type=status.value,
        failure_message=(
            (
                "input is outside the declared principal-real interval"
                if status is ProbeStatus.INVALID_DOMAIN
                else "inverse-trig source inputs must be finite real scalars"
            )
            if status in {ProbeStatus.INVALID_DOMAIN, ProbeStatus.NONFINITE_INPUT}
            else "input policy is explicitly unsupported by the grammar-v2 contract"
        ),
    )


def _run_case(
    case: ConformanceCase,
    mode: CompilerMode,
    config: ConformanceConfig,
    compiled_cache: dict[tuple[str, CompilerMode], tuple[EMLTerm, int, int, str]],
) -> ConformanceRecord:
    classification = _classify_case(case)
    classified_expectation = _expected_status_for_domain(classification)
    if classified_expectation is not case.expected_status:
        raise ConformanceError(
            f"case {case.case_name!r} expected {case.expected_status.value} but the "
            f"grammar-v2 domain contract classifies it as {classified_expectation.value}"
        )
    common = {
        "record_id": _record_id(case, mode),
        "case_name": case.case_name,
        "constructor": case.constructor,
        "source_expression": _source_expression(case),
        "input_text": case.input_text,
        "domain_status": None if classification is None else classification.status,
        "input_zero_sign": None if classification is None else classification.zero_sign,
        "region": case.region,
        "expected_status": case.expected_status,
        "compiler_mode": mode,
    }
    try:
        cache_key = (case.constructor, mode)
        cached = compiled_cache.get(cache_key)
        if cached is None:
            tree = _compile_case(case, mode)
            statistics = validate_pure_eml(tree)
            fingerprint = _sha256(emit_eml(tree).encode("utf-8"))
            compiled_cache[cache_key] = (
                tree,
                statistics.node_count,
                statistics.depth,
                fingerprint,
            )
        else:
            tree, node_count, depth, fingerprint = cached
    except (ConformanceError, TypeError, ValueError, RuntimeError) as error:
        return ConformanceRecord(
            **common,
            observed_status=ProbeStatus.COMPILE_FAILURE,
            verifier_status=VerifierStatus.FAILED,
            failure_type=type(error).__name__,
            failure_message=str(error),
        )
    if cached is None:
        node_count = statistics.node_count
        depth = statistics.depth
    precompiled = _typed_precompile_record(
        case,
        mode,
        classification,
        fingerprint=fingerprint,
        node_count=node_count,
        depth=depth,
    )
    if precompiled is not None:
        return precompiled
    try:
        with mp.workdps(config.precision_digits):
            numeric, passed, extended = _numeric_comparison(case, tree, config)
    except (TimeoutError, OverflowError, ZeroDivisionError) as error:
        if isinstance(error, TimeoutError):
            status = ProbeStatus.TIMEOUT
        elif isinstance(error, OverflowError):
            status = ProbeStatus.OVERFLOW
        else:
            status = ProbeStatus.SINGULAR
        return ConformanceRecord(
            **common,
            observed_status=status,
            verifier_status=VerifierStatus.FAILED,
            pure_eml_fingerprint=fingerprint,
            node_count=node_count,
            depth=depth,
            failure_type=type(error).__name__,
            failure_message=str(error) or status.value,
        )
    except (ConformanceError, ValueError) as error:
        return ConformanceRecord(
            **common,
            observed_status=ProbeStatus.NUMERIC_FAILURE,
            verifier_status=VerifierStatus.FAILED,
            pure_eml_fingerprint=fingerprint,
            node_count=node_count,
            depth=depth,
            failure_type=type(error).__name__,
            failure_message=str(error),
        )
    if numeric.signed_zero_status is SignedZeroStatus.BACKEND_CANNOT_REPRESENT:
        return ConformanceRecord(
            **common,
            observed_status=ProbeStatus.UNSUPPORTED,
            verifier_status=VerifierStatus.UNSUPPORTED,
            pure_eml_fingerprint=fingerprint,
            node_count=node_count,
            depth=depth,
            numeric=numeric,
            failure_type="signed_zero_backend_limit",
            failure_message="mpmath does not retain the sign bit of zero",
        )
    finite = all(
        mp.isfinite(mp.mpf(component))
        for component in (
            numeric.observed_real,
            numeric.observed_imaginary,
            numeric.absolute_error,
            numeric.relative_error,
        )
    )
    if not finite:
        observed_status = ProbeStatus.NONFINITE_RESULT
    elif not passed:
        observed_status = ProbeStatus.NUMERIC_FAILURE
    elif extended:
        observed_status = ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE
    else:
        observed_status = ProbeStatus.PASS
    failure_type = None
    failure_message = None
    if observed_status is ProbeStatus.NONFINITE_RESULT:
        failure_type = "nonfinite_result"
        failure_message = "the pure-EML evaluation produced a nonfinite component"
    elif observed_status is ProbeStatus.NUMERIC_FAILURE:
        failure_type = "numeric_tolerance_exceeded"
        failure_message = "absolute/relative/imaginary error exceeded preregistered tolerances"
    return ConformanceRecord(
        **common,
        observed_status=observed_status,
        verifier_status=(
            VerifierStatus.PASSED
            if observed_status in {ProbeStatus.PASS, ProbeStatus.PASS_WITH_EXTENDED_INTERMEDIATE}
            else VerifierStatus.FAILED
        ),
        pure_eml_fingerprint=fingerprint,
        node_count=node_count,
        depth=depth,
        numeric=numeric,
        failure_type=failure_type,
        failure_message=failure_message,
    )


def build_conformance_run(config: ConformanceConfig) -> ConformanceRun:
    """Build the deterministic bounded record set and self-authenticating manifest."""

    compiled_cache: dict[
        tuple[str, CompilerMode],
        tuple[EMLTerm, int, int, str],
    ] = {}
    records = tuple(
        _run_case(case, mode, config, compiled_cache)
        for case in config.cases
        for mode in config.compiler_modes
    )
    if len(records) > config.maximum_records:
        raise ConformanceError("generated record count exceeds configured maximum")
    operator_counts = dict(sorted(Counter(row.constructor for row in records).items()))
    region_counts = dict(sorted(Counter(row.region for row in records).items()))
    if operator_counts != config.operator_quotas or region_counts != config.region_quotas:
        raise ConformanceError("generated coverage disagrees with predeclared quotas")
    content = b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in records
    )
    config_hash = _sha256(canonical_json_bytes(config.model_dump(mode="json")))
    content_hash = _sha256(content)
    implementation_commit, worktree_dirty = _git_state()
    manifest_payload = {
        "schema_version": CONFORMANCE_SCHEMA_VERSION,
        "benchmark_kind": "bounded_compiler_conformance_not_a_corpus",
        "seed": config.seed,
        "base_commit": config.base_commit,
        "implementation_commit": implementation_commit,
        "worktree_dirty": worktree_dirty,
        "grammar_version": GrammarVersion.V2.value,
        "record_count": len(records),
        "compiler_modes": [mode.value for mode in config.compiler_modes],
        "operator_counts": operator_counts,
        "region_counts": {key.value: value for key, value in region_counts.items()},
        "status_counts": dict(
            sorted(Counter(row.observed_status.value for row in records).items())
        ),
        "config_sha256": config_hash,
        "content_sha256": content_hash,
        "runtime_metadata": _runtime_metadata(),
        "production_root": "outputs/final/goal10/conformance",
        "learning_split": None,
    }
    manifest = ConformanceManifest.model_validate(
        {
            **manifest_payload,
            "manifest_sha256": _sha256(canonical_json_bytes(manifest_payload)),
        }
    )
    return ConformanceRun(manifest=manifest, records=records)


def write_conformance_run(run: ConformanceRun, output_directory: str | Path) -> None:
    """Write two deterministic files without overwriting different evidence."""

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "manifest.json": canonical_json_bytes(run.manifest.model_dump(mode="json")) + b"\n",
        "records.jsonl": b"".join(
            canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in run.records
        ),
    }
    _write_immutable_payloads(
        {root / name: payload for name, payload in payloads.items()},
        error_type=ConformanceError,
        description="conformance evidence",
    )


def _write_immutable_payloads(
    payloads: Mapping[Path, bytes],
    *,
    error_type: type[Exception],
    description: str,
) -> None:
    """Preflight all conflicts, then atomically create every absent file."""

    conflicts = [
        path for path, payload in payloads.items() if path.exists() and path.read_bytes() != payload
    ]
    if conflicts:
        joined = ", ".join(str(path) for path in conflicts)
        raise error_type(f"refusing to overwrite different {description}: {joined}")

    staged: list[tuple[Path, Path]] = []
    try:
        for path, payload in payloads.items():
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((temporary, path))
        for temporary, path in staged:
            os.replace(temporary, path)
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicitly bounded compiler-conformance command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    run = build_conformance_run(load_conformance_config(arguments.config))
    write_conformance_run(run, arguments.output_dir)
    print(run.manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the public command
    raise SystemExit(main())
