"""Lazy, typed hierarchy references and reconstruction validation.

Hierarchy records are metadata.  They contain content-addressed references, not
eagerly embedded expansions, so model-feature shards remain limited to the
allowlisted graph payload in :mod:`geml.export.schema`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from geml.compression.macro.schema import (
    MACRO_NODE_KIND,
    MACRO_PAYLOAD_FIELD,
    MACRO_RULE_BY_OPERATOR,
    MACRO_RULE_FIELD,
    MACRO_SCHEMA_VERSION,
    MacroRule,
    macro_node_value,
)
from geml.compression.motif.compress import CompressedMotifGraph
from geml.compression.motif.vocabulary import (
    MotifChildRef,
    MotifNode,
    MotifPool,
    MotifTargetKind,
    MotifTemplate,
    MotifVocabulary,
)
from geml.contracts.corpus import CorpusSplit
from geml.dag.direct_eml import DirectEMLCompiler
from geml.eml.compiler_core import CompilerMode
from geml.experiments.goal5.motif_sweeps import vocabulary_payload_digest
from geml.export.schema import (
    ContentDescriptor,
    ExportSchemaError,
    canonical_json_bytes,
    canonicalize_graph,
    decode_canonical_json_bytes,
    model_payload_digest,
    model_payload_from_graph,
    sha256_digest,
    sharing_graph_digest,
)
from geml.graph.schema import (
    MACRO_FAMILY,
    MOTIF_FAMILY,
    ChildRef,
    Graph,
    GraphNode,
    GraphRoot,
)
from geml.graph.validate import validate_graph

HIERARCHY_SCHEMA_VERSION = "geml-goal5-hierarchy-v1"
EXPANSION_BUNDLE_MEDIA_TYPE = "application/vnd.geml.expansion-bundle.v1+json"
BINDING_BUNDLE_MEDIA_TYPE = "application/vnd.geml.expansion-bindings.v1+json"
AST_TO_MACRO_HOOK = "geml.hierarchy.ast_to_macro.v1"
MACRO_TO_EML_HOOK = "geml.hierarchy.macro_to_pure_eml.v1"
MOTIF_TO_SOURCE_HOOK = "geml.hierarchy.motif_to_source.v1"
MACRO_EXPANSION_BUNDLE_VERSION = "geml-macro-expansion-bundle-v1"
EMPTY_BINDING_BUNDLE_VERSION = "geml-empty-binding-bundle-v1"
MACRO_BINDING_BUNDLE_VERSION = "geml-ast-macro-binding-bundle-v1"
MOTIF_EXPANSION_BUNDLE_VERSION = "geml-motif-expansion-bundle-v1"
MOTIF_BINDING_BUNDLE_VERSION = "geml-motif-binding-bundle-v1"

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
_Sha256Digest = Annotated[
    str,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
]


class _HierarchyContract(BaseModel):
    """Shared strictness for hierarchy metadata."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        populate_by_name=True,
        strict=True,
    )


class HierarchyRelation(StrEnum):
    """Typed relation between two representation levels."""

    EXPANDS_TO = "expands_to"
    LOWERS_TO = "lowers_to"
    DERIVED_FROM = "derived_from"


class HierarchyLevelRef(_HierarchyContract):
    """Lazy reference to one graph level in a representation hierarchy."""

    level_order: _NonNegativeInt
    level_name: _NonBlankStr
    representation_family: _NonBlankStr
    representation_mode: _NonBlankStr
    graph_digest: _Sha256Digest
    model_payload_digest: _Sha256Digest


class HierarchyLink(_HierarchyContract):
    """Content-addressed expansion and binding data for one directed link."""

    link_order: _NonNegativeInt
    relation: HierarchyRelation
    source_level: _NonBlankStr
    target_level: _NonBlankStr
    source_node_ordinal: _NonNegativeInt | None = None
    expansion_bundle: ContentDescriptor
    binding_bundle: ContentDescriptor
    reconstruction_hook: _NonBlankStr
    expected_target_graph_digest: _Sha256Digest
    selected_representation: (
        Literal[
            "frequent_motif_dag",
            "learned_motif_dag",
        ]
        | None
    ) = None
    vocabulary_id: _NonBlankStr | None = None
    vocabulary_digest: _Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_bundle_media_types(self) -> Self:
        """Keep expansion templates distinct from per-use binding maps."""

        if self.expansion_bundle.media_type != EXPANSION_BUNDLE_MEDIA_TYPE:
            raise ValueError("expansion_bundle must use the Goal 5 expansion bundle media type")
        if self.binding_bundle.media_type != BINDING_BUNDLE_MEDIA_TYPE:
            raise ValueError("binding_bundle must use the Goal 5 binding bundle media type")
        motif_identity = (
            self.selected_representation,
            self.vocabulary_id,
            self.vocabulary_digest,
        )
        if self.reconstruction_hook == MOTIF_TO_SOURCE_HOOK:
            if any(value is None for value in motif_identity):
                raise ValueError("motif hierarchy links require complete vocabulary identity")
            expected_source_level = {
                "frequent_motif_dag": "frequent_motif",
                "learned_motif_dag": "learned_motif",
            }[self.selected_representation]
            if self.source_level != expected_source_level or self.target_level != "macro":
                raise ValueError(
                    "motif hierarchy link levels must match the selected representation"
                )
        elif any(value is not None for value in motif_identity):
            raise ValueError("non-motif hierarchy links cannot carry vocabulary identity")
        return self


def _validate_subset_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("subset_labels must contain only nonblank strings")
    if len(set(labels)) != len(labels):
        raise ValueError("subset_labels must not contain duplicates")
    return labels


class HierarchyRecord(_HierarchyContract):
    """A lazy, acyclic set of graph levels for one expression."""

    schema_version: Literal["geml-goal5-hierarchy-v1"] = HIERARCHY_SCHEMA_VERSION
    expression_id: _NonBlankStr
    split: CorpusSplit
    subset_labels: tuple[_NonBlankStr, ...] = ()
    levels: tuple[HierarchyLevelRef, ...] = Field(min_length=2)
    links: tuple[HierarchyLink, ...] = Field(min_length=1)

    @field_validator("subset_labels")
    @classmethod
    def validate_subset_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        """Require caller-supplied, ordered, unique subset labels."""

        return _validate_subset_labels(labels)

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        """Require dense orders, valid targets, matching digests, and no cycles."""

        level_orders = tuple(level.level_order for level in self.levels)
        if level_orders != tuple(range(len(self.levels))):
            raise ValueError("hierarchy level_order values must be dense and match level order")
        level_names = tuple(level.level_name for level in self.levels)
        if len(set(level_names)) != len(level_names):
            raise ValueError("hierarchy level_name values must be unique")

        link_orders = tuple(link.link_order for link in self.links)
        if link_orders != tuple(range(len(self.links))):
            raise ValueError("hierarchy link_order values must be dense and match link order")
        levels_by_name = {level.level_name: level for level in self.levels}
        adjacency: dict[str, list[str]] = {name: [] for name in level_names}
        for link in self.links:
            if link.source_level not in levels_by_name:
                raise ValueError(f"unknown hierarchy source level {link.source_level!r}")
            if link.target_level not in levels_by_name:
                raise ValueError(f"unknown hierarchy target level {link.target_level!r}")
            if link.source_level == link.target_level:
                raise ValueError("hierarchy links cannot be self-referential")
            target = levels_by_name[link.target_level]
            if link.expected_target_graph_digest != target.graph_digest:
                raise ValueError(
                    f"link {link.link_order} expected target digest does not match "
                    f"level {target.level_name!r}"
                )
            adjacency[link.source_level].append(link.target_level)

        state: dict[str, int] = dict.fromkeys(level_names, 0)
        for start in level_names:
            if state[start] != 0:
                continue
            stack: list[tuple[str, int]] = [(start, 0)]
            state[start] = 1
            while stack:
                node, edge_index = stack[-1]
                if edge_index == len(adjacency[node]):
                    state[node] = 2
                    stack.pop()
                    continue
                target = adjacency[node][edge_index]
                stack[-1] = (node, edge_index + 1)
                if state[target] == 1:
                    raise ValueError(f"hierarchy cycle detected at {node!r} -> {target!r}")
                if state[target] == 0:
                    state[target] = 1
                    stack.append((target, 0))
        return self


class HierarchyValidationResult(_HierarchyContract):
    """Complete validation outcome for lazy hierarchy reconstruction."""

    valid: StrictBool
    errors: tuple[_NonBlankStr, ...] = ()
    checked_link_count: _NonNegativeInt

    @model_validator(mode="after")
    def validate_error_accounting(self) -> Self:
        """Keep the Boolean result consistent with the retained errors."""

        if self.valid == bool(self.errors):
            raise ValueError("valid must be true exactly when errors is empty")
        return self


type GraphLoader = Callable[[HierarchyLevelRef], Graph]
type BundleLoader = Callable[[ContentDescriptor], bytes]
type HierarchyReconstructionHook = Callable[
    [Graph, bytes, bytes, HierarchyLink],
    Graph,
]


@dataclass(slots=True)
class LazyHierarchyResolver:
    """Resolve and verify hierarchy content only when a caller requests it."""

    record: HierarchyRecord
    graph_loader: GraphLoader
    bundle_loader: BundleLoader
    reconstruction_hooks: Mapping[str, HierarchyReconstructionHook]
    _graph_cache: dict[str, Graph] = field(default_factory=dict, init=False)
    _bundle_cache: dict[str, bytes] = field(default_factory=dict, init=False)

    def _level(self, level_name: str) -> HierarchyLevelRef:
        for level in self.record.levels:
            if level.level_name == level_name:
                return level
        raise KeyError(f"unknown hierarchy level {level_name!r}")

    @staticmethod
    def _authenticate_level(
        level_name: str,
        level: HierarchyLevelRef,
        graph: Graph,
    ) -> None:
        """Validate one loaded graph against every field in its level reference."""

        validation = validate_graph(graph)
        if not validation.valid:
            raise ExportSchemaError(
                f"loaded hierarchy level {level_name!r} is invalid: " + "; ".join(validation.errors)
            )
        canonical = canonicalize_graph(graph)
        if canonical.digest != level.graph_digest:
            raise ExportSchemaError(
                f"loaded hierarchy level {level_name!r} digest mismatch: "
                f"expected {level.graph_digest}, observed {canonical.digest}"
            )
        if canonical.representation_family != level.representation_family:
            raise ExportSchemaError(
                f"loaded hierarchy level {level_name!r} family mismatch: "
                f"expected {level.representation_family!r}, "
                f"observed {canonical.representation_family!r}"
            )
        if canonical.representation_mode != level.representation_mode:
            raise ExportSchemaError(
                f"loaded hierarchy level {level_name!r} mode mismatch: "
                f"expected {level.representation_mode!r}, "
                f"observed {canonical.representation_mode!r}"
            )
        observed_payload_digest = model_payload_digest(model_payload_from_graph(graph))
        if observed_payload_digest != level.model_payload_digest:
            raise ExportSchemaError(
                f"loaded hierarchy level {level_name!r} model payload digest mismatch: "
                f"expected {level.model_payload_digest}, observed {observed_payload_digest}"
            )

    def load_level(self, level_name: str) -> Graph:
        """Load and authenticate one graph level, including on cache hits."""

        level = self._level(level_name)
        graph = self._graph_cache.get(level.graph_digest)
        if graph is None:
            graph = self.graph_loader(level)
        self._authenticate_level(level_name, level, graph)
        self._graph_cache[level.graph_digest] = graph
        return graph

    def load_bundle(self, descriptor: ContentDescriptor) -> bytes:
        """Load and authenticate one expansion or binding bundle on every access."""

        data = self._bundle_cache.get(descriptor.digest)
        if data is None:
            data = self.bundle_loader(descriptor)
        errors = descriptor.verify(data)
        if errors:
            raise ExportSchemaError("; ".join(errors))
        self._bundle_cache[descriptor.digest] = data
        return data

    def validate_link(self, link: HierarchyLink) -> tuple[str, ...]:
        """Run the named reconstruction hook and retain every observed failure."""

        errors: list[str] = []
        source_graph: Graph | None = None
        target_graph: Graph | None = None
        expansion: bytes | None = None
        bindings: bytes | None = None
        try:
            source_graph = self.load_level(link.source_level)
        except Exception as error:
            errors.append(f"source level {link.source_level!r}: {type(error).__name__}: {error}")
        try:
            target_graph = self.load_level(link.target_level)
        except Exception as error:
            errors.append(f"target level {link.target_level!r}: {type(error).__name__}: {error}")
        try:
            expansion = self.load_bundle(link.expansion_bundle)
        except Exception as error:
            errors.append(f"expansion bundle: {type(error).__name__}: {error}")
        try:
            bindings = self.load_bundle(link.binding_bundle)
        except Exception as error:
            errors.append(f"binding bundle: {type(error).__name__}: {error}")
        if errors:
            return tuple(errors)

        hook = self.reconstruction_hooks.get(link.reconstruction_hook)
        if hook is None:
            errors.append(f"missing reconstruction hook {link.reconstruction_hook!r}")
            return tuple(errors)
        if source_graph is None or target_graph is None or expansion is None or bindings is None:
            raise AssertionError(
                "successful hierarchy loads must produce all reconstruction inputs"
            )
        if link.source_node_ordinal is not None and link.source_node_ordinal >= len(
            canonicalize_graph(source_graph).nodes
        ):
            errors.append(
                f"source node ordinal {link.source_node_ordinal} does not exist "
                f"in level {link.source_level!r}"
            )
            return tuple(errors)
        try:
            reconstructed = hook(source_graph, expansion, bindings, link)
        except Exception as error:
            errors.append(f"reconstruction hook raised {type(error).__name__}: {error}")
            return tuple(errors)

        validation = validate_graph(reconstructed)
        errors.extend(f"reconstructed graph invalid: {error}" for error in validation.errors)
        if validation.valid:
            observed = sharing_graph_digest(reconstructed)
            if observed != link.expected_target_graph_digest:
                errors.append(
                    "reconstructed graph digest mismatch: "
                    f"expected {link.expected_target_graph_digest}, observed {observed}"
                )
            target_digest = sharing_graph_digest(target_graph)
            if observed != target_digest:
                errors.append(
                    "reconstructed graph does not match loaded target level: "
                    f"target {target_digest}, observed {observed}"
                )
        return tuple(errors)

    def validate_all(self) -> HierarchyValidationResult:
        """Validate every link without short-circuiting later failures."""

        errors = tuple(
            f"link {link.link_order}: {error}"
            for link in self.record.links
            for error in self.validate_link(link)
        )
        return HierarchyValidationResult(
            valid=not errors,
            errors=errors,
            checked_link_count=len(self.record.links),
        )


def _json_object(data: bytes, *, label: str) -> dict[str, object]:
    """Decode a canonical strict-JSON hierarchy bundle."""

    payload = decode_canonical_json_bytes(data, label=label, trailing_lf=False)
    if not isinstance(payload, dict):
        raise ExportSchemaError(f"{label} must contain a JSON object")
    return payload


def macro_expansion_bundle_bytes(*, compiler_mode: CompilerMode) -> bytes:
    """Encode the shared official-construction mode used by two hierarchy links."""

    if not isinstance(compiler_mode, CompilerMode):
        raise TypeError("compiler_mode must be a CompilerMode")
    rule_catalog = {
        operator: {
            "arity": spec.arity,
            "rule": spec.rule.value,
        }
        for operator, spec in sorted(MACRO_RULE_BY_OPERATOR.items())
    }
    return canonical_json_bytes(
        {
            "compiler_mode": compiler_mode.value,
            "macro_schema_version": MACRO_SCHEMA_VERSION,
            "rule_catalog": rule_catalog,
            "rule_catalog_digest": sha256_digest(canonical_json_bytes(rule_catalog)),
            "schema_version": MACRO_EXPANSION_BUNDLE_VERSION,
        }
    )


def empty_binding_bundle_bytes() -> bytes:
    """Return the canonical no-binding bundle for deterministic graph transforms."""

    return canonical_json_bytes({"schema_version": EMPTY_BINDING_BUNDLE_VERSION})


def macro_binding_bundle_bytes(ast_graph: Graph, macro_graph: Graph) -> bytes:
    """Preserve every ordered AST occurrence to macro-node expansion mapping."""

    ast_validation = validate_graph(ast_graph)
    macro_validation = validate_graph(macro_graph)
    if (
        not ast_validation.valid
        or not macro_validation.valid
        or len(ast_graph.roots) != 1
        or len(macro_graph.roots) != 1
    ):
        raise ExportSchemaError("macro binding bundle requires valid single-root graphs")
    ast_ids = _canonical_node_ids(ast_graph)
    macro_ids = _canonical_node_ids(macro_graph)
    ast_ordinals = {node_id: ordinal for ordinal, node_id in enumerate(ast_ids)}
    macro_ordinals = {node_id: ordinal for ordinal, node_id in enumerate(macro_ids)}
    occurrences: list[dict[str, object]] = []
    stack: list[tuple[str, str, tuple[int, ...]]] = [
        (
            ast_graph.roots[0].target_id,
            macro_graph.roots[0].target_id,
            (),
        )
    ]
    while stack:
        ast_id, macro_id, path = stack.pop()
        occurrences.append(
            {
                "ast_node_ordinal": ast_ordinals[ast_id],
                "macro_node_ordinal": macro_ordinals[macro_id],
                "source_path": list(path),
            }
        )
        ast_children = sorted(ast_graph.nodes[ast_id].children, key=lambda child: child.slot)
        macro_children = sorted(
            macro_graph.nodes[macro_id].children,
            key=lambda child: child.slot,
        )
        if tuple(child.slot for child in ast_children) != tuple(
            child.slot for child in macro_children
        ):
            raise ExportSchemaError("AST and macro child slots do not align")
        stack.extend(
            (
                ast_child.target_id,
                macro_child.target_id,
                (*path, ast_child.slot),
            )
            for ast_child, macro_child in reversed(
                tuple(zip(ast_children, macro_children, strict=True))
            )
        )
    occurrences.sort(key=lambda item: tuple(item["source_path"]))  # type: ignore[arg-type]
    return canonical_json_bytes(
        {
            "occurrences": occurrences,
            "schema_version": MACRO_BINDING_BUNDLE_VERSION,
        }
    )


def motif_expansion_bundle_bytes(
    vocabulary: MotifVocabulary,
    *,
    selected_representation: Literal[
        "frequent_motif_dag",
        "learned_motif_dag",
    ],
) -> bytes:
    """Encode only the selected structural templates needed for lazy expansion."""

    if not isinstance(vocabulary, MotifVocabulary):
        raise TypeError("vocabulary must be a MotifVocabulary")
    if selected_representation not in {
        "frequent_motif_dag",
        "learned_motif_dag",
    }:
        raise ValueError("selected_representation must identify one production motif class")
    vocabulary_payload = {
        "failure_count": vocabulary.failure_count,
        "max_size": vocabulary.max_size,
        "min_size": vocabulary.min_size,
        "min_support_count": vocabulary.min_support_count,
        "pool": vocabulary.pool.value,
        "processed_count": vocabulary.processed_count,
        "templates": [
            {
                "boundary_count": template.boundary_count,
                "dictionary_cost_bits": template.dictionary_cost_bits,
                "motif_id": template.motif_id,
                "nodes": [
                    {
                        "children": [
                            {
                                "slot": child.slot,
                                "target_index": child.target_index,
                                "target_kind": child.target_kind.value,
                            }
                            for child in node.children
                        ],
                        "kind": node.kind,
                        "label": node.label,
                        "value": node.value,
                    }
                    for node in template.nodes
                ],
                "occurrence_count": template.occurrence_count,
                "representation_mode": template.representation_mode,
                "signature": template.signature,
                "source_family": template.source_family,
                "support_count": template.support_count,
            }
            for template in vocabulary.templates
        ],
        "training_fingerprint": vocabulary.training_fingerprint,
        "training_transaction_count": vocabulary.training_transaction_count,
        "vocabulary_id": vocabulary.vocabulary_id,
        "vocabulary_limit": vocabulary.vocabulary_limit,
    }
    return canonical_json_bytes(
        {
            "schema_version": MOTIF_EXPANSION_BUNDLE_VERSION,
            "selected_representation": selected_representation,
            "selected_vocabulary_digest": (f"sha256:{vocabulary_payload_digest(vocabulary)}"),
            "vocabulary": vocabulary_payload,
            "vocabulary_digest": sha256_digest(canonical_json_bytes(vocabulary_payload)),
            "vocabulary_id": vocabulary.vocabulary_id,
        }
    )


def _canonical_node_ids(graph: Graph) -> tuple[str, ...]:
    """Return the same first-encounter order used by exported model payloads."""

    validation = validate_graph(graph)
    if not validation.valid:
        raise ExportSchemaError("cannot order invalid graph nodes: " + "; ".join(validation.errors))
    visited: set[str] = set()
    ordered: list[str] = []
    for root in graph.roots:
        stack = [root.target_id]
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            ordered.append(node_id)
            children = sorted(graph.nodes[node_id].children, key=lambda child: child.slot)
            stack.extend(child.target_id for child in reversed(children))
    return tuple(ordered)


def motif_binding_bundle_bytes(compressed: CompressedMotifGraph) -> bytes:
    """Encode ID-independent per-occurrence bindings in canonical ordinal space."""

    if not isinstance(compressed, CompressedMotifGraph):
        raise TypeError("compressed must be a CompressedMotifGraph")
    ordered_ids = _canonical_node_ids(compressed.graph)
    ordinals = {node_id: ordinal for ordinal, node_id in enumerate(ordered_ids)}
    binding_payloads = [
        {
            "boundary_target_ordinals": [
                ordinals[target_id] for target_id in binding.boundary_target_ids
            ],
            "motif_id": binding.motif_id,
            "placeholder_ordinal": ordinals[binding.placeholder_id],
        }
        for binding in compressed.bindings
    ]
    binding_payloads.sort(
        key=lambda payload: (
            payload["placeholder_ordinal"],
            payload["motif_id"],
            payload["boundary_target_ordinals"],
        )
    )
    return canonical_json_bytes(
        {
            "bindings": binding_payloads,
            "schema_version": MOTIF_BINDING_BUNDLE_VERSION,
            "source_family": compressed.source_family,
            "source_representation_mode": compressed.source_representation_mode,
        }
    )


def _require_empty_binding(data: bytes) -> None:
    payload = _json_object(data, label="empty hierarchy binding bundle")
    if payload != {"schema_version": EMPTY_BINDING_BUNDLE_VERSION}:
        raise ExportSchemaError("deterministic graph-transform binding bundle is invalid")


def _exact_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ExportSchemaError(f"{label} must be a nonnegative exact integer")
    return value


def _exact_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExportSchemaError(f"{label} must be a nonempty JSON string")
    return value


def _compiler_mode(data: bytes) -> CompilerMode:
    payload = _json_object(data, label="macro expansion bundle")
    expected_fields = {
        "compiler_mode",
        "macro_schema_version",
        "rule_catalog",
        "rule_catalog_digest",
        "schema_version",
    }
    if set(payload) != expected_fields:
        raise ExportSchemaError("macro expansion bundle fields do not match the schema")
    if payload["schema_version"] != MACRO_EXPANSION_BUNDLE_VERSION:
        raise ExportSchemaError("unsupported macro expansion bundle schema")
    catalog = payload["rule_catalog"]
    if not isinstance(catalog, dict) or set(catalog) != set(MACRO_RULE_BY_OPERATOR):
        raise ExportSchemaError("macro rule catalog operators do not match the approved catalog")
    for operator, raw_rule in catalog.items():
        if not isinstance(raw_rule, dict) or set(raw_rule) != {"arity", "rule"}:
            raise ExportSchemaError(
                f"macro rule catalog entry {operator!r} fields do not match the schema"
            )
        _exact_nonnegative_int(
            raw_rule["arity"],
            label=f"macro rule catalog entry {operator!r} arity",
        )
        _exact_nonempty_string(
            raw_rule["rule"],
            label=f"macro rule catalog entry {operator!r} rule",
        )
    _exact_nonempty_string(
        payload["macro_schema_version"],
        label="macro expansion macro_schema_version",
    )
    _exact_nonempty_string(
        payload["rule_catalog_digest"],
        label="macro expansion rule_catalog_digest",
    )
    compiler_mode = _exact_nonempty_string(
        payload["compiler_mode"],
        label="macro expansion compiler_mode",
    )
    current_catalog = {
        operator: {
            "arity": spec.arity,
            "rule": spec.rule.value,
        }
        for operator, spec in sorted(MACRO_RULE_BY_OPERATOR.items())
    }
    if (
        payload["macro_schema_version"] != MACRO_SCHEMA_VERSION
        or payload["rule_catalog"] != current_catalog
        or payload["rule_catalog_digest"] != sha256_digest(canonical_json_bytes(current_catalog))
    ):
        raise ExportSchemaError(
            "macro expansion bundle does not match the current approved rule catalog"
        )
    try:
        return CompilerMode(compiler_mode)
    except ValueError as error:
        raise ExportSchemaError("macro expansion bundle has an invalid compiler mode") from error


def _decode_macro_bindings(data: bytes) -> dict[str, object]:
    payload = _json_object(data, label="AST-to-macro binding bundle")
    if (
        set(payload) != {"occurrences", "schema_version"}
        or payload["schema_version"] != MACRO_BINDING_BUNDLE_VERSION
        or not isinstance(payload["occurrences"], list)
    ):
        raise ExportSchemaError("AST-to-macro binding bundle fields do not match the schema")
    for raw_occurrence in payload["occurrences"]:
        if not isinstance(raw_occurrence, dict) or set(raw_occurrence) != {
            "ast_node_ordinal",
            "macro_node_ordinal",
            "source_path",
        }:
            raise ExportSchemaError("AST-to-macro binding record fields do not match the schema")
        _exact_nonnegative_int(
            raw_occurrence["ast_node_ordinal"],
            label="AST-to-macro ast_node_ordinal",
        )
        _exact_nonnegative_int(
            raw_occurrence["macro_node_ordinal"],
            label="AST-to-macro macro_node_ordinal",
        )
        source_path = raw_occurrence["source_path"]
        if not isinstance(source_path, list):
            raise ExportSchemaError("AST-to-macro source_path must be a JSON array")
        for slot in source_path:
            _exact_nonnegative_int(slot, label="AST-to-macro source_path slot")
    return payload


def reconstruct_ast_to_macro(
    source_graph: Graph,
    expansion: bytes,
    bindings: bytes,
    _link: HierarchyLink,
) -> Graph:
    """Rebuild the transparent macro DAG from an exported AST-DAG."""

    mode = _compiler_mode(expansion)
    validation = validate_graph(source_graph)
    if not validation.valid or any(node.family != "ast" for node in source_graph.nodes.values()):
        raise ExportSchemaError("AST-to-macro reconstruction requires a valid AST graph")

    nodes: dict[str, GraphNode] = {}
    for node_id, node in source_graph.nodes.items():
        spec = MACRO_RULE_BY_OPERATOR.get(node.label or "")
        if spec is None:
            raise ExportSchemaError(
                f"AST-to-macro reconstruction found unsupported operator {node.label!r}"
            )
        nodes[node_id] = GraphNode(
            node_id=node_id,
            family=MACRO_FAMILY,
            kind=MACRO_NODE_KIND,
            label=node.label,
            value=macro_node_value(spec.rule, node.value),
            children=node.children,
        )
    result = Graph(
        nodes=nodes,
        roots=tuple(
            GraphRoot(
                root_id=root.root_id,
                target_id=root.target_id,
                representation_mode=f"macro:{mode.value}:is_pure_eml=false",
            )
            for root in source_graph.roots
        ),
    )
    result_validation = validate_graph(result)
    if not result_validation.valid:
        raise ExportSchemaError(
            "AST-to-macro reconstruction produced an invalid graph: "
            + "; ".join(result_validation.errors)
        )
    binding_payload = _decode_macro_bindings(bindings)
    observed = _decode_macro_bindings(macro_binding_bundle_bytes(source_graph, result))
    if binding_payload != observed:
        raise ExportSchemaError("AST-to-macro expansion mapping does not match the source graph")
    return result


def _postorder(graph: Graph) -> tuple[str, ...]:
    order: list[str] = []
    completed: set[str] = set()
    for root in graph.roots:
        stack: list[tuple[str, bool]] = [(root.target_id, False)]
        while stack:
            node_id, leaving = stack.pop()
            if node_id in completed:
                continue
            if leaving:
                completed.add(node_id)
                order.append(node_id)
                continue
            stack.append((node_id, True))
            stack.extend(
                (child.target_id, False)
                for child in reversed(
                    sorted(graph.nodes[node_id].children, key=lambda child: child.slot)
                )
                if child.target_id not in completed
            )
    return tuple(order)


def _emit_macro_node(
    compiler: DirectEMLCompiler,
    rule: MacroRule,
    payload: object,
    children: tuple[object, ...],
):
    if rule is MacroRule.VARIABLE:
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            raise ExportSchemaError("variable macro payload is invalid")
        return compiler.emit_variable(payload["name"])
    if rule is MacroRule.ONE:
        return compiler.emit_one()
    if rule is MacroRule.INTEGER:
        if type(payload) is not int:
            raise ExportSchemaError("integer macro payload is invalid")
        return compiler.emit_integer(payload)
    if rule is MacroRule.RATIONAL:
        if not isinstance(payload, dict):
            raise ExportSchemaError("rational macro payload is invalid")
        numerator = payload.get("numerator")
        denominator = payload.get("denominator")
        if type(numerator) is not int or type(denominator) is not int:
            raise ExportSchemaError("rational macro payload is invalid")
        return compiler.emit_rational(numerator, denominator)
    dispatch = {
        MacroRule.NEGATE: compiler.emit_negate,
        MacroRule.EXP: compiler.emit_exp,
        MacroRule.LOG: compiler.emit_log,
        MacroRule.SIN: compiler.emit_sin,
        MacroRule.COS: compiler.emit_cos,
        MacroRule.TAN: compiler.emit_tan,
        MacroRule.SINH: compiler.emit_sinh,
        MacroRule.COSH: compiler.emit_cosh,
        MacroRule.TANH: compiler.emit_tanh,
        MacroRule.ADD: compiler.emit_add,
        MacroRule.SUBTRACT: compiler.emit_subtract,
        MacroRule.MULTIPLY: compiler.emit_multiply,
        MacroRule.DIVIDE: compiler.emit_divide,
        MacroRule.POWER: compiler.emit_power,
    }
    try:
        constructor = dispatch[rule]
    except KeyError as error:  # pragma: no cover - exhaustive enum guard
        raise ExportSchemaError(f"unsupported macro expansion rule {rule.value!r}") from error
    return constructor(*children)


def reconstruct_macro_to_pure_eml(
    source_graph: Graph,
    expansion: bytes,
    bindings: bytes,
    _link: HierarchyLink,
) -> Graph:
    """Expand exported macro rules through the approved official compiler."""

    mode = _compiler_mode(expansion)
    _require_empty_binding(bindings)
    validation = validate_graph(source_graph)
    if not validation.valid or any(
        node.family != MACRO_FAMILY for node in source_graph.nodes.values()
    ):
        raise ExportSchemaError("macro-to-EML reconstruction requires a valid macro graph")

    compiler = DirectEMLCompiler(mode=mode)
    expanded: dict[str, object] = {}
    for node_id in _postorder(source_graph):
        node = source_graph.nodes[node_id]
        if (
            node.kind != MACRO_NODE_KIND
            or not isinstance(node.value, dict)
            or set(node.value) != {MACRO_RULE_FIELD, MACRO_PAYLOAD_FIELD}
        ):
            raise ExportSchemaError(f"macro node {node_id!r} has invalid rule metadata")
        try:
            rule = MacroRule(node.value[MACRO_RULE_FIELD])
        except (TypeError, ValueError) as error:
            raise ExportSchemaError(f"macro node {node_id!r} has an invalid rule") from error
        children = tuple(
            expanded[child.target_id]
            for child in sorted(node.children, key=lambda child: child.slot)
        )
        expanded[node_id] = _emit_macro_node(
            compiler,
            rule,
            node.value[MACRO_PAYLOAD_FIELD],
            children,
        )
    if len(source_graph.roots) != 1:
        raise ExportSchemaError("macro-to-EML reconstruction requires exactly one root")
    root = source_graph.roots[0]
    return compiler.table.to_graph(
        expanded[root.target_id],
        root_id=root.root_id,
        representation_mode=f"pure_eml:{mode.value}",
    )


def _decode_motif_vocabulary(
    data: bytes,
    *,
    link: HierarchyLink,
) -> MotifVocabulary:
    """Rebuild the strict core vocabulary contract from a hierarchy bundle."""

    payload = _json_object(data, label="motif expansion bundle")
    if set(payload) != {
        "schema_version",
        "selected_representation",
        "selected_vocabulary_digest",
        "vocabulary",
        "vocabulary_digest",
        "vocabulary_id",
    }:
        raise ExportSchemaError("motif expansion bundle fields do not match the schema")
    if payload["schema_version"] != MOTIF_EXPANSION_BUNDLE_VERSION:
        raise ExportSchemaError("unsupported motif expansion bundle schema")
    selected_representation = payload["selected_representation"]
    selected_vocabulary_digest = payload["selected_vocabulary_digest"]
    vocabulary_id = payload["vocabulary_id"]
    if (
        not isinstance(selected_representation, str)
        or selected_representation not in {"frequent_motif_dag", "learned_motif_dag"}
        or not isinstance(selected_vocabulary_digest, str)
        or not selected_vocabulary_digest.startswith("sha256:")
        or len(selected_vocabulary_digest) != len("sha256:") + 64
        or any(
            character not in "0123456789abcdef"
            for character in selected_vocabulary_digest.removeprefix("sha256:")
        )
        or not isinstance(vocabulary_id, str)
        or not vocabulary_id
    ):
        raise ExportSchemaError("motif expansion identity fields are invalid")
    raw_vocabulary = payload["vocabulary"]
    if not isinstance(raw_vocabulary, dict):
        raise ExportSchemaError("motif expansion vocabulary must be an object")
    if payload["vocabulary_digest"] != sha256_digest(canonical_json_bytes(raw_vocabulary)):
        raise ExportSchemaError("motif expansion vocabulary digest is invalid")
    expected_vocabulary_fields = {
        "failure_count",
        "max_size",
        "min_size",
        "min_support_count",
        "pool",
        "processed_count",
        "templates",
        "training_fingerprint",
        "training_transaction_count",
        "vocabulary_id",
        "vocabulary_limit",
    }
    if set(raw_vocabulary) != expected_vocabulary_fields:
        raise ExportSchemaError("motif vocabulary fields do not match the core contract")

    raw_templates = raw_vocabulary["templates"]
    if not isinstance(raw_templates, list):
        raise ExportSchemaError("motif expansion templates must be a list")
    templates: list[MotifTemplate] = []
    for raw_template in raw_templates:
        expected_template_fields = {
            "boundary_count",
            "dictionary_cost_bits",
            "motif_id",
            "nodes",
            "occurrence_count",
            "representation_mode",
            "signature",
            "source_family",
            "support_count",
        }
        if not isinstance(raw_template, dict) or set(raw_template) != expected_template_fields:
            raise ExportSchemaError("motif template fields do not match the core contract")
        raw_nodes = raw_template["nodes"]
        if not isinstance(raw_nodes, list):
            raise ExportSchemaError("motif template nodes must be a list")
        nodes: list[MotifNode] = []
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict) or set(raw_node) != {
                "children",
                "kind",
                "label",
                "value",
            }:
                raise ExportSchemaError("motif node fields do not match the core contract")
            raw_children = raw_node["children"]
            if not isinstance(raw_children, list):
                raise ExportSchemaError("motif node children must be a list")
            children: list[MotifChildRef] = []
            for raw_child in raw_children:
                if not isinstance(raw_child, dict) or set(raw_child) != {
                    "slot",
                    "target_index",
                    "target_kind",
                }:
                    raise ExportSchemaError("motif child fields do not match the core contract")
                slot = _exact_nonnegative_int(
                    raw_child["slot"],
                    label="motif child slot",
                )
                target_index = _exact_nonnegative_int(
                    raw_child["target_index"],
                    label="motif child target_index",
                )
                try:
                    children.append(
                        MotifChildRef(
                            slot=slot,
                            target_kind=MotifTargetKind(raw_child["target_kind"]),
                            target_index=target_index,
                        )
                    )
                except (TypeError, ValueError) as error:
                    raise ExportSchemaError("motif child failed validation") from error
            try:
                nodes.append(
                    MotifNode(
                        kind=raw_node["kind"],  # type: ignore[arg-type]
                        label=raw_node["label"],  # type: ignore[arg-type]
                        value=raw_node["value"],  # type: ignore[arg-type]
                        children=tuple(children),
                    )
                )
            except (TypeError, ValueError) as error:
                raise ExportSchemaError("motif node failed validation") from error
        try:
            templates.append(
                MotifTemplate(
                    motif_id=raw_template["motif_id"],  # type: ignore[arg-type]
                    signature=raw_template["signature"],  # type: ignore[arg-type]
                    source_family=raw_template["source_family"],  # type: ignore[arg-type]
                    representation_mode=raw_template["representation_mode"],  # type: ignore[arg-type]
                    nodes=tuple(nodes),
                    boundary_count=_exact_nonnegative_int(
                        raw_template["boundary_count"],
                        label="motif boundary_count",
                    ),
                    support_count=_exact_nonnegative_int(
                        raw_template["support_count"],
                        label="motif support_count",
                    ),
                    occurrence_count=_exact_nonnegative_int(
                        raw_template["occurrence_count"],
                        label="motif occurrence_count",
                    ),
                    dictionary_cost_bits=_exact_nonnegative_int(
                        raw_template["dictionary_cost_bits"],
                        label="motif dictionary_cost_bits",
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            raise ExportSchemaError("motif template failed structural validation") from error

    vocabulary_limit = raw_vocabulary["vocabulary_limit"]
    if vocabulary_limit is not None:
        vocabulary_limit = _exact_nonnegative_int(
            vocabulary_limit,
            label="motif vocabulary_limit",
        )
        if vocabulary_limit == 0:
            raise ExportSchemaError("motif vocabulary_limit must be positive when present")
    try:
        vocabulary = MotifVocabulary(
            vocabulary_id=raw_vocabulary["vocabulary_id"],  # type: ignore[arg-type]
            pool=MotifPool(raw_vocabulary["pool"]),
            min_size=_exact_nonnegative_int(
                raw_vocabulary["min_size"],
                label="motif min_size",
            ),
            max_size=_exact_nonnegative_int(
                raw_vocabulary["max_size"],
                label="motif max_size",
            ),
            min_support_count=_exact_nonnegative_int(
                raw_vocabulary["min_support_count"],
                label="motif min_support_count",
            ),
            vocabulary_limit=vocabulary_limit,
            training_transaction_count=_exact_nonnegative_int(
                raw_vocabulary["training_transaction_count"],
                label="motif training_transaction_count",
            ),
            processed_count=_exact_nonnegative_int(
                raw_vocabulary["processed_count"],
                label="motif processed_count",
            ),
            failure_count=_exact_nonnegative_int(
                raw_vocabulary["failure_count"],
                label="motif failure_count",
            ),
            training_fingerprint=raw_vocabulary["training_fingerprint"],  # type: ignore[arg-type]
            templates=tuple(templates),
        )
    except (TypeError, ValueError) as error:
        raise ExportSchemaError("motif vocabulary failed structural validation") from error
    observed_selected_digest = f"sha256:{vocabulary_payload_digest(vocabulary)}"
    expected_source_level = {
        "frequent_motif_dag": "frequent_motif",
        "learned_motif_dag": "learned_motif",
    }[selected_representation]
    if (
        vocabulary_id != vocabulary.vocabulary_id
        or selected_vocabulary_digest != observed_selected_digest
        or link.selected_representation != selected_representation
        or link.vocabulary_id != vocabulary.vocabulary_id
        or link.vocabulary_digest != observed_selected_digest
        or link.source_level != expected_source_level
        or link.target_level != "macro"
    ):
        raise ExportSchemaError(
            "motif expansion identity does not match its selected hierarchy link"
        )
    expected_bundle = motif_expansion_bundle_bytes(
        vocabulary,
        selected_representation=selected_representation,
    )
    if data != expected_bundle:
        raise ExportSchemaError(
            "motif expansion bundle does not exactly round-trip through its producer"
        )
    return vocabulary


def reconstruct_motif_to_source(
    source_graph: Graph,
    expansion: bytes,
    bindings: bytes,
    _link: HierarchyLink,
) -> Graph:
    """Expand selected motif placeholders using canonical ordinal bindings."""

    vocabulary = _decode_motif_vocabulary(expansion, link=_link)
    templates = vocabulary.by_id()
    binding_payload = _json_object(bindings, label="motif binding bundle")
    if set(binding_payload) != {
        "bindings",
        "schema_version",
        "source_family",
        "source_representation_mode",
    }:
        raise ExportSchemaError("motif binding bundle fields do not match the schema")
    if binding_payload["schema_version"] != MOTIF_BINDING_BUNDLE_VERSION:
        raise ExportSchemaError("unsupported motif binding bundle schema")
    source_family = binding_payload["source_family"]
    source_mode = binding_payload["source_representation_mode"]
    raw_bindings = binding_payload["bindings"]
    if (
        not isinstance(source_family, str)
        or not source_family
        or not isinstance(source_mode, str)
        or not source_mode
        or not isinstance(raw_bindings, list)
    ):
        raise ExportSchemaError("motif binding bundle metadata is invalid")
    motif_kind = "frequent" if _link.selected_representation == "frequent_motif_dag" else "learned"
    expected_source_mode = (
        f"motif:{motif_kind}:{vocabulary.vocabulary_id}:{source_family}:{source_mode}"
    )
    if any(root.representation_mode != expected_source_mode for root in source_graph.roots):
        raise ExportSchemaError(
            "motif source graph mode does not match its selected representation"
        )

    validation = validate_graph(source_graph)
    if not validation.valid or any(
        node.family != MOTIF_FAMILY for node in source_graph.nodes.values()
    ):
        raise ExportSchemaError("motif expansion requires a valid motif-family graph")
    ordered_ids = _canonical_node_ids(source_graph)
    placeholder_ids: set[str] = set()
    decoded: list[tuple[str, MotifTemplate, tuple[str, ...]]] = []
    binding_order: list[tuple[int, str, tuple[int, ...]]] = []
    for raw_binding in raw_bindings:
        if not isinstance(raw_binding, dict) or set(raw_binding) != {
            "boundary_target_ordinals",
            "motif_id",
            "placeholder_ordinal",
        }:
            raise ExportSchemaError("motif binding record fields do not match the schema")
        motif_id = raw_binding["motif_id"]
        placeholder_ordinal = raw_binding["placeholder_ordinal"]
        boundary_ordinals = raw_binding["boundary_target_ordinals"]
        if (
            not isinstance(motif_id, str)
            or type(placeholder_ordinal) is not int
            or not isinstance(boundary_ordinals, list)
            or any(type(value) is not int for value in boundary_ordinals)
        ):
            raise ExportSchemaError("motif binding record has invalid types")
        if placeholder_ordinal < 0 or any(value < 0 for value in boundary_ordinals):
            raise ExportSchemaError("motif binding ordinals must be nonnegative")
        if len(boundary_ordinals) != len(set(boundary_ordinals)):
            raise ExportSchemaError("motif binding boundary ordinals must be unique")
        binding_order.append(
            (
                placeholder_ordinal,
                motif_id,
                tuple(boundary_ordinals),
            )
        )
        try:
            placeholder_id = ordered_ids[placeholder_ordinal]
            boundary_ids = tuple(ordered_ids[value] for value in boundary_ordinals)
            template = templates[motif_id]
        except (IndexError, KeyError) as error:
            raise ExportSchemaError(
                "motif binding references an unknown ordinal/template"
            ) from error
        if placeholder_id in placeholder_ids:
            raise ExportSchemaError("motif binding bundle repeats a placeholder")
        placeholder = source_graph.nodes[placeholder_id]
        if (
            placeholder.kind != "motif_reference"
            or placeholder.label != motif_id
            or placeholder.value != {"motif_id": motif_id}
        ):
            raise ExportSchemaError("motif binding does not match its placeholder identity")
        placeholder_boundaries = tuple(
            child.target_id for child in sorted(placeholder.children, key=lambda child: child.slot)
        )
        if boundary_ids != placeholder_boundaries:
            raise ExportSchemaError("motif binding boundaries do not match its placeholder edges")
        placeholder_ids.add(placeholder_id)
        decoded.append((placeholder_id, template, boundary_ids))
    if binding_order != sorted(binding_order):
        raise ExportSchemaError("motif bindings must use canonical placeholder order")

    graph_placeholders = {
        node_id for node_id, node in source_graph.nodes.items() if node.kind == "motif_reference"
    }
    if graph_placeholders != placeholder_ids:
        raise ExportSchemaError("motif placeholders do not match the binding bundle")

    expanded_ids: dict[str, tuple[str, ...]] = {}
    for binding_index, (placeholder_id, template, _boundary_ids) in enumerate(decoded):
        expanded_ids[placeholder_id] = tuple(
            f"hierarchy-expanded-{binding_index}-{node_index}"
            for node_index in range(len(template.nodes))
        )
    if set(source_graph.nodes).intersection(
        node_id for node_ids in expanded_ids.values() for node_id in node_ids
    ):
        raise ExportSchemaError("motif expansion node IDs collide with source IDs")

    nodes: dict[str, GraphNode] = {}
    for node_id, node in source_graph.nodes.items():
        if node_id in placeholder_ids:
            continue
        nodes[node_id] = GraphNode(
            node_id=node_id,
            family=source_family,
            kind=node.kind,
            label=node.label,
            value=node.value,
            children=tuple(
                ChildRef(
                    slot=child.slot,
                    target_id=(
                        expanded_ids[child.target_id][0]
                        if child.target_id in expanded_ids
                        else child.target_id
                    ),
                )
                for child in sorted(node.children, key=lambda child: child.slot)
            ),
        )

    for placeholder_id, template, boundary_ids in decoded:
        node_ids = expanded_ids[placeholder_id]
        if template.boundary_count != len(boundary_ids):
            raise ExportSchemaError("motif boundary count does not match its binding")
        if template.source_family != source_family or template.representation_mode != source_mode:
            raise ExportSchemaError("motif template source family/mode is inconsistent")
        for node_index, motif_node in enumerate(template.nodes):
            children: list[ChildRef] = []
            for motif_child in motif_node.children:
                try:
                    target_id = (
                        node_ids[motif_child.target_index]
                        if motif_child.target_kind is MotifTargetKind.INTERNAL
                        else boundary_ids[motif_child.target_index]
                    )
                except IndexError as error:
                    raise ExportSchemaError("motif child target index is out of range") from error
                children.append(
                    ChildRef(
                        slot=motif_child.slot,
                        target_id=target_id,
                    )
                )
            nodes[node_ids[node_index]] = GraphNode(
                node_id=node_ids[node_index],
                family=source_family,
                kind=motif_node.kind,
                label=motif_node.label,
                value=motif_node.value,
                children=tuple(children),
            )

    result = Graph(
        nodes=nodes,
        roots=tuple(
            GraphRoot(
                root_id=root.root_id,
                target_id=(
                    expanded_ids[root.target_id][0]
                    if root.target_id in expanded_ids
                    else root.target_id
                ),
                representation_mode=source_mode,
            )
            for root in source_graph.roots
        ),
    )
    result_validation = validate_graph(result)
    if not result_validation.valid:
        raise ExportSchemaError(
            "motif hierarchy expansion produced an invalid graph: "
            + "; ".join(result_validation.errors)
        )
    return result


def default_hierarchy_reconstruction_hooks() -> Mapping[str, HierarchyReconstructionHook]:
    """Return the immutable production hook registry for Goal 5 exports."""

    return {
        AST_TO_MACRO_HOOK: reconstruct_ast_to_macro,
        MACRO_TO_EML_HOOK: reconstruct_macro_to_pure_eml,
        MOTIF_TO_SOURCE_HOOK: reconstruct_motif_to_source,
    }
