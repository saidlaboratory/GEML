"""Strict, leakage-resistant contracts for Goal 5 graph exports.

The model plane is intentionally smaller than the source graph contract.  It is
an explicit allowlist of fields that a learner may consume.  Expression
identity, split membership, subset labels, validation results, reconstruction
results, and evaluation metrics live in the metadata or audit planes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
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
    macro_payload_errors,
)
from geml.contracts.corpus import CorpusSplit
from geml.contracts.expression import ExpressionRecord
from geml.eml.ir import is_valid_source_variable_name
from geml.graph.schema import (
    AST_FAMILY,
    EML_FAMILY,
    EML_ONE_KIND,
    EML_OPERATOR_KIND,
    EML_VARIABLE_KIND,
    MACRO_FAMILY,
    MOTIF_FAMILY,
    ChildRef,
    Graph,
    GraphNode,
    GraphRoot,
    strict_json_snapshot,
)
from geml.graph.validate import validate_graph
from geml.spec.operators import OPERATOR_REGISTRY

EXPORT_SCHEMA_VERSION = "geml-goal5-graph-export-v1"
MODEL_FEATURE_SCHEMA_VERSION = "geml-goal5-model-features-v1"
SHARING_GRAPH_DIGEST_VERSION = "geml-sharing-graph-digest-v1"
PRODUCTION_EXPORT_SCHEMA_VERSION = "geml-goal5-production-export-v1"
EDGE_TYPE_CHILD = "child"
SUBSET_LABEL_POLICY = "explicit-only-default-empty"

MODEL_FEATURE_ALLOWLIST = (
    "schema_version",
    "representation_family",
    "representation_mode",
    "nodes",
    "edges",
    "roots",
)

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
_Sha256Digest = Annotated[
    str,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
]
_Sha256Hex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]

type GraphReconstructionHook = Callable[[Graph], Graph]


class ExportSchemaError(ValueError):
    """Raised when an object cannot be represented by the export schema."""


class _ExportContract(BaseModel):
    """Shared strictness policy for all serialized export records."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        populate_by_name=True,
        strict=True,
    )


class ExportPlane(StrEnum):
    """Physical isolation boundary for exported data."""

    METADATA = "metadata"
    MODEL = "model"
    AUDIT = "audit"


class ShardRecordType(StrEnum):
    """The one record schema carried by a shard."""

    EXPRESSION_METADATA = "expression_metadata"
    GRAPH_METADATA = "graph_metadata"
    HIERARCHY_METADATA = "hierarchy_metadata"
    MODEL_GRAPH = "model_graph"
    GRAPH_AUDIT = "graph_audit"


class ValidationStatus(StrEnum):
    """Whether the exported source graph passed all graph checks."""

    PASSED = "passed"
    FAILED = "failed"


class ReconstructionStatus(StrEnum):
    """Outcome of a requested graph reconstruction check."""

    NOT_REQUESTED = "not_requested"
    PASSED = "passed"
    FAILED = "failed"


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one strict JSON-compatible value deterministically."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ExportSchemaError(f"value is not strict canonical JSON: {error}") from error


def decode_canonical_json_bytes(
    data: bytes,
    *,
    label: str,
    trailing_lf: bool,
) -> object:
    """Decode unique-key finite JSON and require exact producer serialization."""

    if not isinstance(data, bytes):
        raise ExportSchemaError(f"{label} must be encoded as bytes")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def finite_float(value: str) -> float:
        decoded = float(value)
        if not math.isfinite(decoded):
            raise ValueError(f"non-finite JSON number {value!r}")
        return decoded

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            data,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ExportSchemaError(f"{label} is not valid finite unique-key UTF-8 JSON") from error

    expected = canonical_json_bytes(payload) + (b"\n" if trailing_lf else b"")
    if data != expected:
        suffix = " with one trailing LF" if trailing_lf else ""
        raise ExportSchemaError(f"{label} is not canonical JSON{suffix}")
    return payload


def sha256_digest(data: bytes) -> str:
    """Return an OCI-style algorithm-qualified SHA-256 digest."""

    return f"sha256:{hashlib.sha256(data).hexdigest()}"


class ContentDescriptor(_ExportContract):
    """An OCI-style descriptor for one immutable byte sequence."""

    media_type: _NonBlankStr = Field(alias="mediaType")
    digest: _Sha256Digest
    size: _NonNegativeInt

    @classmethod
    def from_bytes(cls, data: bytes, *, media_type: str) -> Self:
        """Describe ``data`` without persisting it."""

        return cls(media_type=media_type, digest=sha256_digest(data), size=len(data))

    def verify(self, data: bytes) -> tuple[str, ...]:
        """Return every size or digest mismatch for ``data``."""

        errors: list[str] = []
        if len(data) != self.size:
            errors.append(f"size mismatch: expected {self.size}, observed {len(data)}")
        observed_digest = sha256_digest(data)
        if observed_digest != self.digest:
            errors.append(f"digest mismatch: expected {self.digest}, observed {observed_digest}")
        return tuple(errors)


class CanonicalChild(_ExportContract):
    """One sharing-sensitive edge in canonical ordinal space."""

    slot: _NonNegativeInt
    target_ordinal: _NonNegativeInt
    edge_type: Literal["child"] = EDGE_TYPE_CHILD


class CanonicalNode(_ExportContract):
    """One canonical node used only to compute a sharing-sensitive digest."""

    ordinal: _NonNegativeInt
    family: _NonBlankStr
    kind: _NonBlankStr
    label: _NonBlankStr | None = None
    value: JsonValue = None
    children: tuple[CanonicalChild, ...] = ()

    @field_validator("value")
    @classmethod
    def snapshot_value(cls, value: JsonValue) -> JsonValue:
        """Freeze nested JSON so a computed digest cannot drift later."""

        return strict_json_snapshot(value)


class CanonicalRoot(_ExportContract):
    """One ordered root in canonical ordinal space."""

    root_order: _NonNegativeInt
    representation_mode: _NonBlankStr
    target_ordinal: _NonNegativeInt


class CanonicalGraph(_ExportContract):
    """Canonical graph form that preserves sharing and repeated references."""

    digest_version: Literal["geml-sharing-graph-digest-v1"] = SHARING_GRAPH_DIGEST_VERSION
    representation_family: _NonBlankStr
    representation_mode: _NonBlankStr
    nodes: tuple[CanonicalNode, ...] = Field(min_length=1)
    roots: tuple[CanonicalRoot, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ordinals_and_references(self) -> Self:
        """Require dense first-encounter ordinals and exact references."""

        ordinals = tuple(node.ordinal for node in self.nodes)
        if ordinals != tuple(range(len(self.nodes))):
            raise ValueError("canonical node ordinals must be dense and match node order")
        root_orders = tuple(root.root_order for root in self.roots)
        if root_orders != tuple(range(len(self.roots))):
            raise ValueError("canonical root orders must be dense and match root order")

        valid_ordinals = set(ordinals)
        for node in self.nodes:
            slots = tuple(child.slot for child in node.children)
            if slots != tuple(range(len(node.children))):
                raise ValueError(
                    f"canonical node {node.ordinal} child slots must be dense and ordered"
                )
            if any(child.target_ordinal not in valid_ordinals for child in node.children):
                raise ValueError(f"canonical node {node.ordinal} has an invalid target ordinal")
        if any(root.target_ordinal not in valid_ordinals for root in self.roots):
            raise ValueError("canonical root has an invalid target ordinal")
        return self

    @property
    def digest(self) -> str:
        """Return the sharing-sensitive digest of this canonical graph."""

        return sha256_digest(canonical_json_bytes(self.model_dump(mode="json")))


def canonicalize_graph(graph: Graph) -> CanonicalGraph:
    """Assign deterministic first-encounter ordinals to a valid graph.

    Traversal follows ordered roots and then child slots.  Original node and
    root identifiers are deliberately excluded.  A repeated reference reuses
    its first ordinal, so a shared node hashes differently from two otherwise
    identical duplicated nodes.
    """

    validation = validate_graph(graph)
    if not validation.valid:
        raise ExportSchemaError(
            "cannot canonicalize invalid graph: " + "; ".join(validation.errors)
        )

    representation_modes = {root.representation_mode for root in graph.roots}
    if len(representation_modes) != 1:
        raise ExportSchemaError(
            "one exported graph must have exactly one representation_mode across all roots"
        )
    representation_mode = next(iter(representation_modes))

    ordinals: dict[str, int] = {}
    node_ids: list[str] = []
    for root in graph.roots:
        stack = [root.target_id]
        while stack:
            node_id = stack.pop()
            if node_id in ordinals:
                continue
            ordinals[node_id] = len(node_ids)
            node_ids.append(node_id)
            children = sorted(graph.nodes[node_id].children, key=lambda child: child.slot)
            stack.extend(child.target_id for child in reversed(children))

    nodes = tuple(
        CanonicalNode(
            ordinal=ordinals[node_id],
            family=graph.nodes[node_id].family,
            kind=graph.nodes[node_id].kind,
            label=graph.nodes[node_id].label,
            value=graph.nodes[node_id].value,
            children=tuple(
                CanonicalChild(
                    slot=child.slot,
                    target_ordinal=ordinals[child.target_id],
                )
                for child in sorted(
                    graph.nodes[node_id].children,
                    key=lambda child: child.slot,
                )
            ),
        )
        for node_id in node_ids
    )
    roots = tuple(
        CanonicalRoot(
            root_order=root_order,
            representation_mode=root.representation_mode,
            target_ordinal=ordinals[root.target_id],
        )
        for root_order, root in enumerate(graph.roots)
    )
    return CanonicalGraph(
        representation_family=nodes[0].family,
        representation_mode=representation_mode,
        nodes=nodes,
        roots=roots,
    )


def sharing_graph_digest(graph: Graph) -> str:
    """Return the versioned digest that distinguishes sharing from duplication."""

    return canonicalize_graph(graph).digest


class ModelNode(_ExportContract):
    """Allowlisted node fields visible to model loaders."""

    ordinal: _NonNegativeInt
    family: _NonBlankStr
    kind: _NonBlankStr
    label: _NonBlankStr | None = None
    value: JsonValue = None

    @field_validator("value")
    @classmethod
    def snapshot_value(cls, value: JsonValue) -> JsonValue:
        """Freeze nested feature values after strict JSON validation."""

        return strict_json_snapshot(value)


class ModelEdge(_ExportContract):
    """Allowlisted child reference visible to model loaders."""

    source_ordinal: _NonNegativeInt
    target_ordinal: _NonNegativeInt
    slot: _NonNegativeInt
    edge_type: Literal["child"] = EDGE_TYPE_CHILD


class ModelRoot(_ExportContract):
    """Allowlisted ordered root reference visible to model loaders."""

    root_order: _NonNegativeInt
    target_ordinal: _NonNegativeInt


def _macro_value_errors(node: ModelNode) -> tuple[str, ...]:
    if node.kind != MACRO_NODE_KIND:
        return (f"unsupported macro node kind {node.kind!r}",)
    if node.label not in MACRO_RULE_BY_OPERATOR:
        return (f"unsupported macro operator {node.label!r}",)
    if not isinstance(node.value, dict) or set(node.value) != {
        MACRO_RULE_FIELD,
        MACRO_PAYLOAD_FIELD,
    }:
        return ("macro value must contain exactly expansion_rule and payload",)
    spec = MACRO_RULE_BY_OPERATOR[node.label]
    if node.value[MACRO_RULE_FIELD] != spec.rule.value:
        return ("macro expansion_rule does not match the node operator",)
    return macro_payload_errors(node.label, node.value[MACRO_PAYLOAD_FIELD])


def _eml_value_errors(node: ModelNode) -> tuple[str, ...]:
    if node.kind == EML_OPERATOR_KIND:
        return (
            ()
            if node.label == "eml" and node.value is None
            else ("EML operator value must be null and label must be 'eml'",)
        )
    if node.kind == EML_VARIABLE_KIND:
        return (
            ()
            if is_valid_source_variable_name(node.value) and node.value == node.label
            else ("EML variable value must equal its valid source-variable label",)
        )
    if node.kind == EML_ONE_KIND:
        return (
            ()
            if node.label == "1" and type(node.value) is int and node.value == 1
            else ("EML one value must be exact integer 1",)
        )
    return (f"unsupported EML node kind {node.kind!r}",)


def _ast_value_errors(node: ModelNode) -> tuple[str, ...]:
    if node.kind == "operator":
        if node.label not in OPERATOR_REGISTRY:
            return (f"unsupported AST operator {node.label!r}",)
        return () if node.value is None else ("AST operator value must be null",)
    if node.kind != "leaf" or node.label not in {
        "symbol",
        "one",
        "integer",
        "rational",
    }:
        return (f"unsupported AST leaf kind/label {node.kind!r}/{node.label!r}",)
    return macro_payload_errors(node.label, node.value)


def _model_node_value_errors(node: ModelNode) -> tuple[str, ...]:
    """Apply a family/kind-specific structural value allowlist."""

    if node.family == AST_FAMILY:
        return _ast_value_errors(node)
    if node.family == EML_FAMILY:
        return _eml_value_errors(node)
    if node.family == MACRO_FAMILY:
        return _macro_value_errors(node)
    if node.family != MOTIF_FAMILY:
        return (f"unsupported model node family {node.family!r}",)
    if node.kind == "motif_reference":
        expected = {"motif_id": node.label}
        if (
            not isinstance(node.label, str)
            or not node.label.startswith("motif:")
            or len(node.label) != len("motif:") + 64
            or any(character not in "0123456789abcdef" for character in node.label[6:])
            or node.value != expected
        ):
            return ("motif reference value must contain only its canonical motif_id",)
        return ()
    if node.kind == MACRO_NODE_KIND:
        return _macro_value_errors(node)
    if node.kind in {EML_OPERATOR_KIND, EML_VARIABLE_KIND, EML_ONE_KIND}:
        return _eml_value_errors(node)
    return (f"unsupported motif-carried node kind {node.kind!r}",)


def _model_node_arity_errors(node: ModelNode, child_count: int) -> tuple[str, ...]:
    """Validate source-construction arity without interpreting motif references."""

    expected_arity: int | None = None
    if node.family == AST_FAMILY:
        operator = OPERATOR_REGISTRY.get(node.label or "")
        expected_arity = None if operator is None else operator.arity
    elif node.family == MACRO_FAMILY or (
        node.family == MOTIF_FAMILY and node.kind == MACRO_NODE_KIND
    ):
        rule = MACRO_RULE_BY_OPERATOR.get(node.label or "")
        expected_arity = None if rule is None else rule.arity
    elif node.family == EML_FAMILY or (
        node.family == MOTIF_FAMILY
        and node.kind in {EML_OPERATOR_KIND, EML_VARIABLE_KIND, EML_ONE_KIND}
    ):
        expected_arity = 2 if node.kind == EML_OPERATOR_KIND else 0

    if expected_arity is None or child_count == expected_arity:
        return ()
    return (
        f"{node.family}/{node.kind}/{node.label} requires {expected_arity} "
        f"children, observed {child_count}",
    )


class ModelGraphPayload(_ExportContract):
    """The complete fail-closed model-feature allowlist."""

    schema_version: Literal["geml-goal5-model-features-v1"] = MODEL_FEATURE_SCHEMA_VERSION
    representation_family: _NonBlankStr
    representation_mode: _NonBlankStr
    nodes: tuple[ModelNode, ...] = Field(min_length=1)
    edges: tuple[ModelEdge, ...] = ()
    roots: tuple[ModelRoot, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph_shape(self) -> Self:
        """Validate dense ordinals, slots, roots, and edge references."""

        ordinals = tuple(node.ordinal for node in self.nodes)
        if ordinals != tuple(range(len(self.nodes))):
            raise ValueError("model node ordinals must be dense and match node order")
        if any(node.family != self.representation_family for node in self.nodes):
            raise ValueError("every model node family must match representation_family")
        valid_ordinals = set(ordinals)
        root_orders = tuple(root.root_order for root in self.roots)
        if root_orders != tuple(range(len(self.roots))):
            raise ValueError("model root orders must be dense and match root order")
        if any(root.target_ordinal not in valid_ordinals for root in self.roots):
            raise ValueError("model root has an invalid target ordinal")

        slots_by_source: dict[int, list[int]] = {}
        edge_order = tuple(
            (edge.source_ordinal, edge.slot, edge.target_ordinal) for edge in self.edges
        )
        if edge_order != tuple(sorted(edge_order)):
            raise ValueError(
                "model edges must be ordered by source ordinal, slot, then target ordinal"
            )
        for edge in self.edges:
            if edge.source_ordinal not in valid_ordinals:
                raise ValueError("model edge has an invalid source ordinal")
            if edge.target_ordinal not in valid_ordinals:
                raise ValueError("model edge has an invalid target ordinal")
            slots_by_source.setdefault(edge.source_ordinal, []).append(edge.slot)
        for source, slots in slots_by_source.items():
            if slots != list(range(len(slots))):
                raise ValueError(
                    f"model edges for source ordinal {source} must have dense ordered slots"
                )
        semantic_errors = tuple(
            f"model node {node.ordinal}: {error}"
            for node in self.nodes
            for error in (
                *_model_node_value_errors(node),
                *_model_node_arity_errors(
                    node,
                    len(slots_by_source.get(node.ordinal, ())),
                ),
            )
        )
        if semantic_errors:
            raise ValueError("; ".join(semantic_errors))
        try:
            self.to_graph()
        except ExportSchemaError as error:
            raise ValueError(str(error)) from error
        return self

    def to_graph(self) -> Graph:
        """Reconstruct and fully validate a frozen graph from allowlisted fields."""

        children_by_source: dict[int, list[ChildRef]] = {node.ordinal: [] for node in self.nodes}
        for edge in self.edges:
            children_by_source[edge.source_ordinal].append(
                ChildRef(
                    slot=edge.slot,
                    target_id=f"node-{edge.target_ordinal}",
                )
            )
        graph = Graph(
            nodes={
                f"node-{node.ordinal}": GraphNode(
                    node_id=f"node-{node.ordinal}",
                    family=node.family,
                    kind=node.kind,
                    label=node.label,
                    value=node.value,
                    children=tuple(
                        sorted(
                            children_by_source[node.ordinal],
                            key=lambda child: child.slot,
                        )
                    ),
                )
                for node in self.nodes
            },
            roots=tuple(
                GraphRoot(
                    root_id=f"root-{root.root_order}",
                    target_id=f"node-{root.target_ordinal}",
                    representation_mode=self.representation_mode,
                )
                for root in self.roots
            ),
        )
        validation = validate_graph(graph)
        if not validation.valid:
            raise ExportSchemaError(
                "model payload does not reconstruct a valid graph: " + "; ".join(validation.errors)
            )
        return graph


def model_payload_from_graph(graph: Graph) -> ModelGraphPayload:
    """Project a frozen graph through the complete model-feature allowlist."""

    canonical = canonicalize_graph(graph)
    return ModelGraphPayload(
        representation_family=canonical.representation_family,
        representation_mode=canonical.representation_mode,
        nodes=tuple(
            ModelNode(
                ordinal=node.ordinal,
                family=node.family,
                kind=node.kind,
                label=node.label,
                value=node.value,
            )
            for node in canonical.nodes
        ),
        edges=tuple(
            ModelEdge(
                source_ordinal=node.ordinal,
                target_ordinal=child.target_ordinal,
                slot=child.slot,
            )
            for node in canonical.nodes
            for child in node.children
        ),
        roots=tuple(
            ModelRoot(
                root_order=root.root_order,
                target_ordinal=root.target_ordinal,
            )
            for root in canonical.roots
        ),
    )


def graph_from_model_payload(payload: ModelGraphPayload) -> Graph:
    """Return the fully validated graph reconstructed from a model payload."""

    return payload.to_graph()


def model_payload_digest(payload: ModelGraphPayload) -> str:
    """Return the content digest used to join metadata to model features."""

    return sha256_digest(canonical_json_bytes(payload.model_dump(mode="json")))


def _validate_subset_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("subset_labels must contain only nonblank strings")
    if len(set(labels)) != len(labels):
        raise ValueError("subset_labels must not contain duplicates")
    return labels


class ExpressionMetadataRecord(_ExportContract):
    """Source/provenance metadata that is physically isolated from features."""

    expression_id: _NonBlankStr
    sympy_srepr: _NonBlankStr
    display_text: _NonBlankStr
    latex_text: _NonBlankStr | None = None
    split: CorpusSplit
    subset_labels: tuple[_NonBlankStr, ...] = ()
    operator_family: _NonBlankStr
    domain_mode: _NonBlankStr
    variables: tuple[_NonBlankStr, ...]
    target_ast_size: _NonNegativeInt
    target_depth: _NonNegativeInt
    generator_seed: StrictInt
    generator_metadata: dict[str, JsonValue]

    @field_validator("subset_labels")
    @classmethod
    def validate_subset_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        """Keep caller-supplied subset labels explicit, ordered, and unique."""

        return _validate_subset_labels(labels)

    @field_validator("generator_metadata")
    @classmethod
    def snapshot_generator_metadata(
        cls,
        metadata: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        """Freeze nested provenance JSON after Pydantic validates its shape."""

        snapshot = strict_json_snapshot(metadata)
        if not isinstance(snapshot, dict):
            raise ValueError("generator_metadata must be a JSON object")
        return snapshot

    @classmethod
    def from_expression(
        cls,
        expression: ExpressionRecord,
        *,
        subset_labels: tuple[str, ...] = (),
    ) -> Self:
        """Snapshot an expression without inferring any subset membership."""

        return cls(
            **expression.model_dump(),
            subset_labels=subset_labels,
        )


class GraphMetadataRecord(_ExportContract):
    """Join keys and corpus membership for one successfully exported graph."""

    expression_id: _NonBlankStr
    split: CorpusSplit
    subset_labels: tuple[_NonBlankStr, ...] = ()
    representation_family: _NonBlankStr
    representation_mode: _NonBlankStr
    graph_digest: _Sha256Digest
    model_payload_digest: _Sha256Digest

    @field_validator("subset_labels")
    @classmethod
    def validate_subset_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        """Require explicit, nonduplicated subset labels."""

        return _validate_subset_labels(labels)


class ModelPlaneRecord(_ExportContract):
    """Content-addressed wrapper; loaders expose only ``payload`` as features."""

    model_payload_digest: _Sha256Digest
    payload: ModelGraphPayload

    @model_validator(mode="after")
    def validate_payload_digest(self) -> Self:
        """Require the join key to authenticate the exact allowlisted payload."""

        observed = model_payload_digest(self.payload)
        if self.model_payload_digest != observed:
            raise ValueError("model_payload_digest does not match the canonical model payload")
        return self


class GraphAuditRecord(_ExportContract):
    """Validation, reconstruction, and metric data excluded from model features."""

    expression_id: _NonBlankStr
    split: CorpusSplit
    subset_labels: tuple[_NonBlankStr, ...] = ()
    representation_family: _NonBlankStr | None = None
    representation_mode: _NonBlankStr | None = None
    graph_digest: _Sha256Digest | None = None
    validation_status: ValidationStatus
    validation_errors: tuple[_NonBlankStr, ...] = ()
    reconstruction_status: ReconstructionStatus
    reconstruction_errors: tuple[_NonBlankStr, ...] = ()
    failure_stage: _NonBlankStr | None = None
    error_type: _NonBlankStr | None = None
    error_message: _NonBlankStr | None = None
    metrics: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("subset_labels")
    @classmethod
    def validate_subset_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        """Keep audit joins bound to the exact explicit subset assignment."""

        return _validate_subset_labels(labels)

    @field_validator("metrics")
    @classmethod
    def snapshot_metrics(
        cls,
        metrics: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        """Freeze nested audit JSON so it remains descriptor-stable."""

        snapshot = strict_json_snapshot(metrics)
        if not isinstance(snapshot, dict):
            raise ValueError("metrics must be a JSON object")
        return snapshot

    @model_validator(mode="after")
    def validate_outcome_details(self) -> Self:
        """Require failure details and prohibit contradictory error lists."""

        if self.validation_status is ValidationStatus.PASSED and self.validation_errors:
            raise ValueError("passed graph validation cannot contain validation_errors")
        if self.validation_status is ValidationStatus.FAILED and not self.validation_errors:
            raise ValueError("failed graph validation must retain validation_errors")
        if self.reconstruction_status is ReconstructionStatus.PASSED and self.reconstruction_errors:
            raise ValueError("passed reconstruction cannot contain reconstruction_errors")
        if (
            self.reconstruction_status is ReconstructionStatus.FAILED
            and not self.reconstruction_errors
        ):
            raise ValueError("failed reconstruction must retain reconstruction_errors")
        if (
            self.reconstruction_status is ReconstructionStatus.NOT_REQUESTED
            and self.reconstruction_errors
        ):
            raise ValueError("unrequested reconstruction cannot contain errors")
        failure_details = (
            self.failure_stage,
            self.error_type,
            self.error_message,
        )
        if any(value is None for value in failure_details) != all(
            value is None for value in failure_details
        ):
            raise ValueError(
                "failure_stage, error_type, and error_message must be supplied together"
            )
        if self.failure_stage is not None and self.validation_status is not ValidationStatus.FAILED:
            raise ValueError("structured build failures require failed validation status")
        return self


class PreparedGraphExport(_ExportContract):
    """The three planes prepared from one expression/graph pair."""

    expression_metadata: ExpressionMetadataRecord
    graph_metadata: GraphMetadataRecord | None = None
    model_record: ModelPlaneRecord | None = None
    audit_record: GraphAuditRecord

    @model_validator(mode="after")
    def validate_plane_alignment(self) -> Self:
        """Keep the three planes aligned without moving fields between them."""

        if self.expression_metadata.expression_id != self.audit_record.expression_id:
            raise ValueError("expression and audit records disagree on expression_id")
        if self.expression_metadata.split != self.audit_record.split:
            raise ValueError("expression and audit records disagree on split")
        if self.expression_metadata.subset_labels != self.audit_record.subset_labels:
            raise ValueError("expression and audit records disagree on subset_labels")
        if (self.graph_metadata is None) != (self.model_record is None):
            raise ValueError("graph metadata and model record must be present together")
        if self.graph_metadata is not None and self.model_record is not None:
            if self.graph_metadata.expression_id != self.expression_metadata.expression_id:
                raise ValueError("graph metadata has the wrong expression_id")
            if self.graph_metadata.split != self.expression_metadata.split:
                raise ValueError("graph metadata has the wrong split")
            if self.graph_metadata.subset_labels != self.expression_metadata.subset_labels:
                raise ValueError("graph metadata has the wrong subset_labels")
            if self.graph_metadata.model_payload_digest != self.model_record.model_payload_digest:
                raise ValueError("metadata/model payload digests disagree")
        return self


def _graph_family_and_mode(graph: Graph) -> tuple[str | None, str | None]:
    families = {
        node.family
        for node in graph.nodes.values()
        if hasattr(node, "family") and isinstance(node.family, str) and node.family.strip()
    }
    modes = {
        root.representation_mode
        for root in graph.roots
        if hasattr(root, "representation_mode")
        and isinstance(root.representation_mode, str)
        and root.representation_mode.strip()
    }
    family = next(iter(families)) if len(families) == 1 else None
    mode = next(iter(modes)) if len(modes) == 1 else None
    return family, mode


def _run_reconstruction_check(
    graph: Graph,
    *,
    reconstruction_hook: GraphReconstructionHook | None,
    expected_reconstruction_graph: Graph | None,
) -> tuple[ReconstructionStatus, tuple[str, ...]]:
    if reconstruction_hook is None and expected_reconstruction_graph is None:
        return ReconstructionStatus.NOT_REQUESTED, ()
    if reconstruction_hook is None:
        return (
            ReconstructionStatus.FAILED,
            ("expected_reconstruction_graph was supplied without a reconstruction_hook",),
        )
    if expected_reconstruction_graph is None:
        return (
            ReconstructionStatus.FAILED,
            ("reconstruction_hook was supplied without expected_reconstruction_graph",),
        )

    expected_validation = validate_graph(expected_reconstruction_graph)
    if not expected_validation.valid:
        return (
            ReconstructionStatus.FAILED,
            tuple(
                f"expected reconstruction graph invalid: {error}"
                for error in expected_validation.errors
            ),
        )

    try:
        reconstructed = reconstruction_hook(graph)
    except Exception as error:
        return (
            ReconstructionStatus.FAILED,
            (f"reconstruction hook raised {type(error).__name__}: {error}",),
        )
    reconstructed_validation = validate_graph(reconstructed)
    if not reconstructed_validation.valid:
        return (
            ReconstructionStatus.FAILED,
            tuple(
                f"reconstructed graph invalid: {error}" for error in reconstructed_validation.errors
            ),
        )

    observed = sharing_graph_digest(reconstructed)
    expected = sharing_graph_digest(expected_reconstruction_graph)
    if observed != expected:
        return (
            ReconstructionStatus.FAILED,
            (f"reconstructed graph digest mismatch: expected {expected}, observed {observed}",),
        )
    return ReconstructionStatus.PASSED, ()


def prepare_graph_export(
    expression: ExpressionRecord,
    graph: Graph,
    *,
    subset_labels: tuple[str, ...] = (),
    reconstruction_hook: GraphReconstructionHook | None = None,
    expected_reconstruction_graph: Graph | None = None,
    audit_metrics: Mapping[str, JsonValue] | None = None,
) -> PreparedGraphExport:
    """Prepare isolated metadata/model/audit records without dropping failures."""

    expression_metadata = ExpressionMetadataRecord.from_expression(
        expression,
        subset_labels=subset_labels,
    )
    family, mode = _graph_family_and_mode(graph)
    validation = validate_graph(graph)
    validation_errors = list(validation.errors)
    if validation.valid and mode is None:
        validation_errors.append(
            "one exported graph must have exactly one representation_mode across all roots"
        )

    if validation_errors:
        reconstruction_requested = (
            reconstruction_hook is not None or expected_reconstruction_graph is not None
        )
        audit = GraphAuditRecord(
            expression_id=expression.expression_id,
            split=expression.split,
            subset_labels=subset_labels,
            representation_family=family,
            representation_mode=mode,
            validation_status=ValidationStatus.FAILED,
            validation_errors=tuple(validation_errors),
            reconstruction_status=(
                ReconstructionStatus.FAILED
                if reconstruction_requested
                else ReconstructionStatus.NOT_REQUESTED
            ),
            reconstruction_errors=(
                ("reconstruction was requested but source graph validation failed",)
                if reconstruction_requested
                else ()
            ),
            metrics=dict(audit_metrics or {}),
        )
        return PreparedGraphExport(
            expression_metadata=expression_metadata,
            audit_record=audit,
        )

    canonical = canonicalize_graph(graph)
    payload = model_payload_from_graph(graph)
    payload_digest = model_payload_digest(payload)
    graph_metadata = GraphMetadataRecord(
        expression_id=expression.expression_id,
        split=expression.split,
        subset_labels=subset_labels,
        representation_family=canonical.representation_family,
        representation_mode=canonical.representation_mode,
        graph_digest=canonical.digest,
        model_payload_digest=payload_digest,
    )
    reconstruction_status, reconstruction_errors = _run_reconstruction_check(
        graph,
        reconstruction_hook=reconstruction_hook,
        expected_reconstruction_graph=expected_reconstruction_graph,
    )
    audit = GraphAuditRecord(
        expression_id=expression.expression_id,
        split=expression.split,
        subset_labels=subset_labels,
        representation_family=canonical.representation_family,
        representation_mode=canonical.representation_mode,
        graph_digest=canonical.digest,
        validation_status=ValidationStatus.PASSED,
        reconstruction_status=reconstruction_status,
        reconstruction_errors=reconstruction_errors,
        metrics=dict(audit_metrics or {}),
    )
    return PreparedGraphExport(
        expression_metadata=expression_metadata,
        graph_metadata=graph_metadata,
        model_record=ModelPlaneRecord(
            model_payload_digest=payload_digest,
            payload=payload,
        ),
        audit_record=audit,
    )


def prepare_graph_failure_export(
    expression: ExpressionRecord,
    *,
    representation_family: str,
    representation_mode: str,
    failure_stage: str,
    error_type: str,
    error_message: str,
    subset_labels: tuple[str, ...] = (),
    reconstruction_required: bool = False,
    audit_metrics: Mapping[str, JsonValue] | None = None,
) -> PreparedGraphExport:
    """Retain a typed pre-graph failure without fabricating a partial graph."""

    details = {
        "representation_family": representation_family,
        "representation_mode": representation_mode,
        "failure_stage": failure_stage,
        "error_type": error_type,
        "error_message": error_message,
    }
    for name, value in details.items():
        if not isinstance(value, str) or not value.strip():
            raise ExportSchemaError(f"{name} must be a nonblank string")
    if not isinstance(reconstruction_required, bool):
        raise ExportSchemaError("reconstruction_required must be a boolean")
    return PreparedGraphExport(
        expression_metadata=ExpressionMetadataRecord.from_expression(
            expression,
            subset_labels=subset_labels,
        ),
        audit_record=GraphAuditRecord(
            expression_id=expression.expression_id,
            split=expression.split,
            subset_labels=subset_labels,
            representation_family=representation_family,
            representation_mode=representation_mode,
            validation_status=ValidationStatus.FAILED,
            validation_errors=(f"{failure_stage}: {error_type}: {error_message}",),
            reconstruction_status=(
                ReconstructionStatus.FAILED
                if reconstruction_required
                else ReconstructionStatus.NOT_REQUESTED
            ),
            reconstruction_errors=(
                ("reconstruction was required but graph construction failed",)
                if reconstruction_required
                else ()
            ),
            failure_stage=failure_stage,
            error_type=error_type,
            error_message=error_message,
            metrics=dict(audit_metrics or {}),
        ),
    )


class ShardDescriptor(_ExportContract):
    """Typed descriptor for one deterministic split/mode shard."""

    shard_id: _NonBlankStr
    path: _NonBlankStr
    plane: ExportPlane
    record_type: ShardRecordType
    split: CorpusSplit
    representation_mode: _NonBlankStr | None = None
    shard_index: _NonNegativeInt
    row_count: _NonNegativeInt
    content: ContentDescriptor

    @model_validator(mode="after")
    def validate_plane_and_mode(self) -> Self:
        """Require record types to remain in their declared physical plane."""

        expected_plane = {
            ShardRecordType.EXPRESSION_METADATA: ExportPlane.METADATA,
            ShardRecordType.GRAPH_METADATA: ExportPlane.METADATA,
            ShardRecordType.HIERARCHY_METADATA: ExportPlane.METADATA,
            ShardRecordType.MODEL_GRAPH: ExportPlane.MODEL,
            ShardRecordType.GRAPH_AUDIT: ExportPlane.AUDIT,
        }[self.record_type]
        if self.plane is not expected_plane:
            raise ValueError(
                f"{self.record_type.value} records must use the {expected_plane.value} plane"
            )
        mode_required = self.record_type in {
            ShardRecordType.GRAPH_METADATA,
            ShardRecordType.MODEL_GRAPH,
            ShardRecordType.GRAPH_AUDIT,
        }
        if mode_required and self.representation_mode is None:
            raise ValueError(f"{self.record_type.value} shards require representation_mode")
        if (
            self.record_type is ShardRecordType.EXPRESSION_METADATA
            and self.representation_mode is not None
        ):
            raise ValueError("expression metadata shards cannot have representation_mode")
        return self


class ExportManifest(_ExportContract):
    """Top-level immutable manifest for a Goal 5 graph export."""

    schema_version: Literal["geml-goal5-graph-export-v1"] = EXPORT_SCHEMA_VERSION
    dataset_id: _NonBlankStr
    sharing_graph_digest_version: Literal["geml-sharing-graph-digest-v1"] = (
        SHARING_GRAPH_DIGEST_VERSION
    )
    edge_type: Literal["child"] = EDGE_TYPE_CHILD
    subset_label_policy: Literal["explicit-only-default-empty"] = SUBSET_LABEL_POLICY
    model_feature_allowlist: tuple[_NonBlankStr, ...] = MODEL_FEATURE_ALLOWLIST
    shards: tuple[ShardDescriptor, ...]
    expression_count: _NonNegativeInt
    graph_count: _NonNegativeInt
    validation_failure_count: _NonNegativeInt
    reconstruction_failure_count: _NonNegativeInt

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Require deterministic uniqueness and the frozen allowlist."""

        if self.model_feature_allowlist != MODEL_FEATURE_ALLOWLIST:
            raise ValueError("model_feature_allowlist must equal the frozen allowlist")
        shard_ids = tuple(shard.shard_id for shard in self.shards)
        if len(set(shard_ids)) != len(shard_ids):
            raise ValueError("export manifest contains duplicate shard_id values")
        paths = tuple(shard.path for shard in self.shards)
        if len(set(paths)) != len(paths):
            raise ValueError("export manifest contains duplicate shard paths")
        groups: dict[
            tuple[ShardRecordType, CorpusSplit, str | None],
            list[int],
        ] = {}
        for shard in self.shards:
            key = (shard.record_type, shard.split, shard.representation_mode)
            groups.setdefault(key, []).append(shard.shard_index)
        for key, shard_indexes in groups.items():
            if shard_indexes != list(range(len(shard_indexes))):
                raise ValueError(
                    "shard_index values must be contiguous and match manifest order "
                    f"within group {key!r}"
                )
        return self


class SourceArtifactDescriptor(_ExportContract):
    """Authenticated metadata-plane provenance for one production input."""

    name: _NonBlankStr
    path: _NonBlankStr
    content: ContentDescriptor
    semantic_digest: _Sha256Hex | None = None


class ProductionRepresentation(_ExportContract):
    """One separately sharded representation selected for every expression."""

    name: Literal[
        "ast_dag",
        "pure_eml_dag",
        "macro_dag",
        "frequent_motif_dag",
        "learned_motif_dag",
    ]
    representation_family: _NonBlankStr
    representation_mode: _NonBlankStr
    selected_vocabulary_digest: _Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_frozen_binding(self) -> Self:
        """Bind every public representation name to its exact family and mode class."""

        fixed = {
            "ast_dag": (AST_FAMILY, "ast"),
            "pure_eml_dag": (EML_FAMILY, "pure_eml:official_v4"),
            "macro_dag": (MACRO_FAMILY, "macro:official_v4:is_pure_eml=false"),
        }
        if self.name in fixed:
            expected_family, expected_mode = fixed[self.name]
            if (
                self.representation_family != expected_family
                or self.representation_mode != expected_mode
            ):
                raise ValueError(
                    f"{self.name} must use family {expected_family!r} and mode {expected_mode!r}"
                )
            if self.selected_vocabulary_digest is not None:
                raise ValueError(f"{self.name} cannot carry a selected vocabulary digest")
            return self

        motif_kind = "frequent" if self.name == "frequent_motif_dag" else "learned"
        mode_prefix = f"motif:{motif_kind}:motif-vocabulary:"
        mode_suffix = ":macro:macro:official_v4:is_pure_eml=false"
        vocabulary_id_digest = self.representation_mode.removeprefix(mode_prefix).removesuffix(
            mode_suffix
        )
        if (
            self.representation_family != MOTIF_FAMILY
            or not self.representation_mode.startswith(mode_prefix)
            or not self.representation_mode.endswith(mode_suffix)
            or len(vocabulary_id_digest) != 64
            or any(character not in "0123456789abcdef" for character in vocabulary_id_digest)
        ):
            raise ValueError(f"{self.name} must use its selected official-v4 macro motif mode")
        if self.selected_vocabulary_digest is None:
            raise ValueError(f"{self.name} requires a selected vocabulary digest")
        return self


class ProductionBatchDescriptor(_ExportContract):
    """One bounded, immutable export checkpoint."""

    batch_id: _NonBlankStr
    path: _NonBlankStr
    split: CorpusSplit
    source_shard_index: _NonNegativeInt
    source_batch_index: _NonNegativeInt
    source_records_digest: _Sha256Digest
    expression_count: _NonNegativeInt
    graph_count: _NonNegativeInt
    hierarchy_count: _NonNegativeInt
    validation_failure_count: _NonNegativeInt
    reconstruction_failure_count: _NonNegativeInt
    first_expression_id: _NonBlankStr
    last_expression_id: _NonBlankStr
    manifest: ContentDescriptor

    @model_validator(mode="after")
    def validate_batch_counts(self) -> Self:
        """Keep aggregate batch counters internally possible."""

        if self.expression_count < 1:
            raise ValueError("production batches must contain at least one expression")
        if self.graph_count < self.expression_count:
            raise ValueError("production graph_count cannot be smaller than expression_count")
        if self.graph_count != self.expression_count * 5:
            raise ValueError("production batches require five successful graphs per expression")
        if self.hierarchy_count > self.expression_count:
            raise ValueError("hierarchy_count cannot exceed expression_count")
        if self.validation_failure_count > self.graph_count:
            raise ValueError("validation_failure_count cannot exceed graph_count")
        if self.reconstruction_failure_count > self.graph_count:
            raise ValueError("reconstruction_failure_count cannot exceed graph_count")
        if self.validation_failure_count or self.reconstruction_failure_count:
            raise ValueError("completed production batches cannot contain failed graph attempts")
        if self.first_expression_id > self.last_expression_id:
            raise ValueError("batch expression ID bounds must be increasing")
        return self


class ProductionExportManifest(_ExportContract):
    """Atomic completion contract over every bounded Goal 5 export batch."""

    schema_version: Literal["geml-goal5-production-export-v1"] = PRODUCTION_EXPORT_SCHEMA_VERSION
    dataset_id: _NonBlankStr
    config_digest: _Sha256Hex
    implementation_digest: _Sha256Hex
    source_artifacts: tuple[SourceArtifactDescriptor, ...] = Field(min_length=3)
    representations: tuple[ProductionRepresentation, ...] = Field(min_length=5, max_length=5)
    hierarchy_enabled: StrictBool
    batches: tuple[ProductionBatchDescriptor, ...] = Field(min_length=1)
    expression_count: _NonNegativeInt
    graph_count: _NonNegativeInt
    hierarchy_count: _NonNegativeInt
    validation_failure_count: _NonNegativeInt
    reconstruction_failure_count: _NonNegativeInt
    reproduction_command: _NonBlankStr

    @model_validator(mode="after")
    def validate_completion_totals(self) -> Self:
        """Require exact five-mode coverage and exact aggregate counters."""

        expected_names = (
            "ast_dag",
            "pure_eml_dag",
            "macro_dag",
            "frequent_motif_dag",
            "learned_motif_dag",
        )
        if tuple(item.name for item in self.representations) != expected_names:
            raise ValueError("production representations must use the frozen five-mode order")
        modes = tuple(item.representation_mode for item in self.representations)
        if len(set(modes)) != len(modes):
            raise ValueError("production representation modes must be distinct")
        source_names = tuple(item.name for item in self.source_artifacts)
        if len(set(source_names)) != len(source_names):
            raise ValueError("source artifact names must be unique")
        batch_ids = tuple(batch.batch_id for batch in self.batches)
        batch_paths = tuple(batch.path for batch in self.batches)
        if len(set(batch_ids)) != len(batch_ids) or len(set(batch_paths)) != len(batch_paths):
            raise ValueError("production batch IDs and paths must be unique")
        ordered_batches = tuple(
            sorted(
                self.batches,
                key=lambda batch: (
                    list(CorpusSplit).index(batch.split),
                    batch.source_shard_index,
                    batch.source_batch_index,
                ),
            )
        )
        if self.batches != ordered_batches:
            raise ValueError("production batches must use canonical split/source order")
        totals = {
            "expression_count": sum(batch.expression_count for batch in self.batches),
            "graph_count": sum(batch.graph_count for batch in self.batches),
            "hierarchy_count": sum(batch.hierarchy_count for batch in self.batches),
            "validation_failure_count": sum(
                batch.validation_failure_count for batch in self.batches
            ),
            "reconstruction_failure_count": sum(
                batch.reconstruction_failure_count for batch in self.batches
            ),
        }
        for name, observed in totals.items():
            if getattr(self, name) != observed:
                raise ValueError(f"{name} must equal the sum of batch values")
        if self.graph_count != self.expression_count * len(self.representations):
            raise ValueError(
                "production export requires exactly five graph attempts per expression"
            )
        if self.validation_failure_count or self.reconstruction_failure_count:
            raise ValueError("completed production exports cannot contain failed graph attempts")
        expected_hierarchy_count = self.expression_count if self.hierarchy_enabled else 0
        if self.hierarchy_count != expected_hierarchy_count:
            raise ValueError(
                "production hierarchy coverage must match the configured hierarchy policy"
            )
        return self
