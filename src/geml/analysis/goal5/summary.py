"""Strict, denominator-explicit Goal 5 integration evidence and summaries.

The integration layer deliberately consumes a small normalized evidence manifest rather
than guessing at producer internals.  Every numeric claim keeps its all-attempted
denominator and names the authenticated artifacts that support it.  The runner verifies
those artifact bytes before this module derives rates or renders prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from geml.export.hierarchical import HIERARCHY_SCHEMA_VERSION
from geml.export.schema import (
    MODEL_FEATURE_SCHEMA_VERSION,
    PRODUCTION_EXPORT_SCHEMA_VERSION,
)

INTEGRATION_EVIDENCE_SCHEMA_VERSION = "geml-goal5-integration-evidence-v1"
INTEGRATION_SUMMARY_SCHEMA_VERSION = "geml-goal5-integration-summary-v1"

GOAL6_EXPORT_ROOT = "outputs/final/goal5/export"
GOAL6_RUN_DIRECTORY_PATTERN = "outputs/final/goal5/export/run-{run_digest}"
GOAL6_COMPLETION_FILENAME = "run.complete.json"

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
_NonNegativeFloat = Annotated[StrictFloat, Field(ge=0.0)]


class IntegrationEvidenceError(ValueError):
    """Normalized Goal 5 evidence violates the scientific reporting contract."""


class EvidenceStatus(StrEnum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"


class SplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST_IID = "test_iid"
    TEST_OOD = "test_ood"


class GraphTrackName(StrEnum):
    AST_DAG = "ast_dag"
    PURE_EML_DAG = "pure_eml_dag"
    SAFE_EGRAPH_EML_DAG = "safe_egraph_eml_dag"
    DOMAIN_EGRAPH_EML_DAG = "domain_egraph_eml_dag"
    MACRO_DAG = "macro_dag"
    FREQUENT_MOTIF_DAG = "frequent_motif_dag"
    LEARNED_MOTIF_DAG = "learned_motif_dag"


class RankerMethod(StrEnum):
    EXACT_EML_COST = "exact_official_eml_dag"
    NEURAL = "neural_ranker"
    ESTIMATED_EML_COST = "estimated_eml_tree_cost"
    AST_COST = "ast_dag_cost"
    RANDOM = "deterministic_random"


class MetricAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class ArtifactSchemaState(StrEnum):
    VERSIONED = "versioned"
    EXPLICITLY_UNVERSIONED = "explicitly_unversioned"


class MdlScope(StrEnum):
    STANDALONE_GRAPH = "standalone_graph"
    DICTIONARY_INCLUSIVE_MOTIF = "dictionary_inclusive_motif"


class ClaimOutcome(StrEnum):
    POSITIVE = "positive"
    NULL_RESULT = "null_result"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"


class ClaimId(StrEnum):
    LEARNED_VS_FREQUENT = "learned_vs_frequent"
    LEARNED_VS_RANDOM = "learned_vs_random"
    LEARNED_VS_MACRO = "learned_vs_macro"
    NEURAL_VS_HEURISTICS = "neural_vs_heuristics"


class GoalCompletionStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    BLOCKED = "blocked"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class _ArtifactBackedModel(_FrozenModel):
    """Shared canonical source references for every evidence-bearing object."""

    source_artifacts: tuple[_NonBlankStr, ...] = Field(min_length=1)

    @field_validator("source_artifacts")
    @classmethod
    def validate_source_order(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or value != tuple(sorted(value)):
            raise ValueError("source_artifacts must be unique and sorted")
        return value


class FrozenGoal6Export(_FrozenModel):
    """Goal 6-facing paths and schema versions frozen by issue 5-9."""

    export_root: Literal["outputs/final/goal5/export"] = GOAL6_EXPORT_ROOT
    run_directory_pattern: Literal["outputs/final/goal5/export/run-{run_digest}"] = (
        GOAL6_RUN_DIRECTORY_PATTERN
    )
    completion_filename: Literal["run.complete.json"] = GOAL6_COMPLETION_FILENAME
    production_manifest_schema_version: Literal["geml-goal5-production-export-v1"] = (
        PRODUCTION_EXPORT_SCHEMA_VERSION
    )
    model_feature_schema_version: Literal["geml-goal5-model-features-v1"] = (
        MODEL_FEATURE_SCHEMA_VERSION
    )
    hierarchy_schema_version: Literal["geml-goal5-hierarchy-v1"] = HIERARCHY_SCHEMA_VERSION


class SourceArtifact(_FrozenModel):
    """One immutable producer artifact cited by normalized evidence."""

    name: _NonBlankStr
    path: _NonBlankStr
    sha256: _Sha256Hex
    size_bytes: _NonNegativeInt
    media_type: _NonBlankStr
    schema_state: ArtifactSchemaState
    schema_version: _NonBlankStr | None = None
    unversioned_reason: _NonBlankStr | None = None

    @field_validator("path")
    @classmethod
    def validate_repository_relative_path(cls, value: str) -> str:
        """Require one canonical POSIX path below the repository root."""

        if "\\" in value:
            raise ValueError("artifact paths must use POSIX separators")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != value
        ):
            raise ValueError("artifact paths must be canonical repository-relative paths")
        return value

    @model_validator(mode="after")
    def validate_schema_declaration(self) -> Self:
        if self.schema_state is ArtifactSchemaState.VERSIONED:
            if self.schema_version is None:
                raise ValueError("versioned artifacts require schema_version")
            if self.unversioned_reason is not None:
                raise ValueError("versioned artifacts cannot carry an unversioned reason")
            return self
        if self.schema_version is not None:
            raise ValueError("explicitly unversioned artifacts cannot invent schema_version")
        if self.unversioned_reason is None:
            raise ValueError("explicitly unversioned artifacts require an authenticated reason")
        return self


class SubsetDefinition(_ArtifactBackedModel):
    """A named denominator slice with its scientific membership rule."""

    name: _NonBlankStr
    definition: _NonBlankStr
    is_nontrivial: bool
    expression_ids_sha256: _Sha256Hex
    rewrite_mode: Literal["all", "safe_real", "positive_real_formal"]
    semantics: _NonBlankStr

    @model_validator(mode="after")
    def validate_subset_semantics(self) -> Self:
        expected_nontrivial = self.rewrite_mode != "all"
        if self.is_nontrivial is not expected_nontrivial:
            raise ValueError("nontrivial flag must match the subset rewrite mode")
        expected_semantics = {
            "safe_real": "branch_insensitive_finite_real",
            "positive_real_formal": ("conditional_positive_real_formal_under_recorded_assumptions"),
        }
        if (
            self.rewrite_mode in expected_semantics
            and self.semantics != expected_semantics[self.rewrite_mode]
        ):
            raise ValueError("nontrivial subset semantics do not match the rewrite mode")
        return self


class CrossTrackJoinEvidence(_ArtifactBackedModel):
    """A split-specific expression-ID cohort joined across every graph track."""

    split: SplitName
    subset: Literal["safe_nontrivial", "domain_nontrivial"]
    expression_count: _NonNegativeInt
    expression_ids_sha256: _Sha256Hex
    track_names: tuple[GraphTrackName, ...] = Field(min_length=1)


class _MetricObservation(_ArtifactBackedModel):
    """Shared availability and authenticated-source contract for one metric."""

    availability: MetricAvailability
    denominator_count: _NonNegativeInt
    observation_count: _NonNegativeInt
    missing_count: _NonNegativeInt
    unavailable_reason: _NonBlankStr | None = None

    def _validate_availability(self, *, allow_not_applicable: bool = False) -> None:
        if self.observation_count + self.missing_count != self.denominator_count:
            raise ValueError("observed/missing counts must partition denominator_count")
        if self.availability is MetricAvailability.AVAILABLE:
            if self.denominator_count > 0 and self.observation_count == 0:
                raise ValueError("available metrics require at least one observation")
            if self.unavailable_reason is not None:
                raise ValueError("available metrics cannot carry an unavailable reason")
            return
        if self.availability is MetricAvailability.NOT_APPLICABLE and not allow_not_applicable:
            raise ValueError("this metric cannot be marked not_applicable")
        if self.observation_count:
            raise ValueError("unavailable metrics cannot contain observations")
        if self.unavailable_reason is None:
            raise ValueError("unavailable metrics require an authenticated reason")


class IntegerMetricObservation(_MetricObservation):
    """Exact nonnegative integer total with explicit availability."""

    total: _NonNegativeInt | None
    unit: _NonBlankStr

    @model_validator(mode="after")
    def validate_metric(self) -> Self:
        self._validate_availability()
        if (self.availability is MetricAvailability.AVAILABLE) != (self.total is not None):
            raise ValueError("integer total is present exactly when the metric is available")
        return self


class FloatMetricObservation(_MetricObservation):
    """Exact source float total with explicit availability."""

    total: _NonNegativeFloat | None
    unit: _NonBlankStr

    @model_validator(mode="after")
    def validate_metric(self) -> Self:
        self._validate_availability()
        if (self.availability is MetricAvailability.AVAILABLE) != (self.total is not None):
            raise ValueError("float total is present exactly when the metric is available")
        return self


class PeakMetricObservation(_MetricObservation):
    """Peak resource value; aggregation across slices uses the maximum."""

    peak: _NonNegativeInt | None
    unit: _NonBlankStr

    @model_validator(mode="after")
    def validate_metric(self) -> Self:
        self._validate_availability()
        if (self.observation_count > 0) != (self.peak is not None):
            raise ValueError("peak value is present exactly when observations exist")
        return self


class MdlMetricObservation(_MetricObservation):
    """MDL total whose scope prevents unlike costs from being conflated."""

    total_bits: _NonNegativeInt | None
    codec: _NonBlankStr | None
    scope: MdlScope

    @model_validator(mode="after")
    def validate_metric(self) -> Self:
        self._validate_availability()
        available = self.availability is MetricAvailability.AVAILABLE
        if available != (self.total_bits is not None):
            raise ValueError("MDL total is present exactly when MDL is available")
        if available != (self.codec is not None):
            raise ValueError("MDL codec is present exactly when MDL is available")
        return self


class AuditCounts(_ArtifactBackedModel):
    """Exact availability and attempted/passed/failed accounting for one validity gate."""

    availability: MetricAvailability
    denominator_count: _NonNegativeInt
    attempted_count: _NonNegativeInt
    passed_count: _NonNegativeInt
    failed_count: _NonNegativeInt
    unobserved_count: _NonNegativeInt
    unavailable_reason: _NonBlankStr | None = None

    @model_validator(mode="after")
    def validate_accounting(self) -> Self:
        if self.passed_count + self.failed_count != self.attempted_count:
            raise ValueError("audit pass/fail counts must partition attempted_count")
        if self.attempted_count + self.unobserved_count != self.denominator_count:
            raise ValueError("audit attempted/unobserved counts must partition denominator_count")
        if self.availability is MetricAvailability.AVAILABLE:
            if self.denominator_count > 0 and self.attempted_count == 0:
                raise ValueError("available audits require at least one attempt")
            if self.unavailable_reason is not None:
                raise ValueError("available audits cannot carry an unavailable reason")
            return self
        if any((self.attempted_count, self.passed_count, self.failed_count)):
            raise ValueError("unavailable and not-applicable audits must use zero counts")
        if self.unavailable_reason is None:
            raise ValueError("unavailable audits require an authenticated reason")
        return self


class GraphSliceEvidence(_ArtifactBackedModel):
    """Exact aggregate evidence for one graph track, split, and subset."""

    split: SplitName
    subset: _NonBlankStr
    denominator_count: _NonNegativeInt
    success_count: _NonNegativeInt
    failure_count: _NonNegativeInt
    failure_counts: dict[str, _NonNegativeInt]
    node_count: IntegerMetricObservation
    edge_count: IntegerMetricObservation
    mdl_cost: MdlMetricObservation
    structural_validation: AuditCounts
    reconstruction: AuditCounts
    expansion: AuditCounts
    runtime: FloatMetricObservation
    memory: PeakMetricObservation

    @field_validator("failure_counts")
    @classmethod
    def validate_failure_categories(
        cls,
        value: dict[str, int],
    ) -> dict[str, int]:
        if any(not key or count < 1 for key, count in value.items()):
            raise ValueError("failure_counts must map nonblank labels to positive counts")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def validate_denominators(self) -> Self:
        if self.success_count + self.failure_count != self.denominator_count:
            raise ValueError("success/failure counts must partition denominator_count")
        if sum(self.failure_counts.values()) != self.failure_count:
            raise ValueError("failure categories must sum to failure_count")
        for name, metric in (
            ("node count", self.node_count),
            ("edge count", self.edge_count),
            ("MDL", self.mdl_cost),
        ):
            if metric.denominator_count != self.success_count:
                raise ValueError(f"{name} denominator must equal success_count")
        for name, metric in (("runtime", self.runtime), ("memory", self.memory)):
            if metric.denominator_count != self.denominator_count:
                raise ValueError(f"{name} denominator must equal denominator_count")
        for name, audit in (
            ("structural validation", self.structural_validation),
            ("reconstruction", self.reconstruction),
            ("expansion", self.expansion),
        ):
            if audit.denominator_count != self.success_count:
                raise ValueError(f"{name} denominator must equal success_count")
        return self


_PURE_EML_TRACKS = frozenset(
    {
        GraphTrackName.PURE_EML_DAG,
        GraphTrackName.SAFE_EGRAPH_EML_DAG,
        GraphTrackName.DOMAIN_EGRAPH_EML_DAG,
    }
)


class GraphTrackEvidence(_FrozenModel):
    """One structural representation and all of its denominator slices."""

    name: GraphTrackName
    display_name: _NonBlankStr
    representation_family: _NonBlankStr
    representation_mode: _NonBlankStr
    is_pure_eml: bool
    slices: tuple[GraphSliceEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_purity_and_keys(self) -> Self:
        expected_purity = self.name in _PURE_EML_TRACKS
        if self.is_pure_eml is not expected_purity:
            raise ValueError(
                f"{self.name.value} must declare is_pure_eml={str(expected_purity).lower()}"
            )
        keys = tuple((item.split, item.subset) for item in self.slices)
        if len(set(keys)) != len(keys):
            raise ValueError("graph track contains duplicate split/subset slices")
        reconstruction_required = self.name in {
            GraphTrackName.FREQUENT_MOTIF_DAG,
            GraphTrackName.LEARNED_MOTIF_DAG,
        }
        expansion_required = self.name is GraphTrackName.MACRO_DAG
        expected_mdl_scope = (
            MdlScope.DICTIONARY_INCLUSIVE_MOTIF
            if reconstruction_required
            else MdlScope.STANDALONE_GRAPH
        )
        goal4_egraph_track = self.name in {
            GraphTrackName.SAFE_EGRAPH_EML_DAG,
            GraphTrackName.DOMAIN_EGRAPH_EML_DAG,
        }
        for item in self.slices:
            if (
                item.node_count.unit != "nodes"
                or item.edge_count.unit != "edges"
                or item.runtime.unit != "seconds"
                or item.memory.unit != "bytes"
            ):
                raise ValueError("graph metric units must use nodes, edges, seconds, and bytes")
            if item.mdl_cost.scope is not expected_mdl_scope:
                raise ValueError("MDL scope does not match the frozen track contract")
            if goal4_egraph_track and (
                item.edge_count.availability is not MetricAvailability.UNAVAILABLE
                or item.mdl_cost.availability is not MetricAvailability.UNAVAILABLE
            ):
                raise ValueError(
                    "Goal 4 e-graph rows expose post-rewrite node cost only; "
                    "non-reversible selected_signature cannot support edges or MDL"
                )
            if item.structural_validation.availability is MetricAvailability.NOT_APPLICABLE:
                raise ValueError("structural validation cannot be marked not_applicable")
            self._validate_required_audit(
                item.reconstruction,
                required=reconstruction_required,
                label="reconstruction",
            )
            self._validate_required_audit(
                item.expansion,
                required=expansion_required,
                label="expansion",
            )
        codecs = {
            item.mdl_cost.codec
            for item in self.slices
            if item.mdl_cost.availability is MetricAvailability.AVAILABLE
        }
        if len(codecs) > 1:
            raise ValueError("one graph track cannot aggregate different MDL codecs")
        return self

    @staticmethod
    def _validate_required_audit(
        audit: AuditCounts,
        *,
        required: bool,
        label: str,
    ) -> None:
        if required and audit.availability is MetricAvailability.NOT_APPLICABLE:
            raise ValueError(f"required {label} cannot be marked not_applicable")
        if not required and audit.availability is not MetricAvailability.NOT_APPLICABLE:
            raise ValueError(f"{label} availability does not match the frozen track contract")


class RankerMethodSliceEvidence(_ArtifactBackedModel):
    """Candidate-selection evidence for one method, split, and subset."""

    method: RankerMethod
    split: SplitName
    subset: _NonBlankStr
    denominator_count: _NonNegativeInt
    evaluable_group_count: _NonNegativeInt
    unevaluable_group_count: _NonNegativeInt
    attempted_group_count: _NonNegativeInt
    validated_selection_count: _NonNegativeInt
    failed_selected_count: _NonNegativeInt
    exact_best_match_count: _NonNegativeInt
    regret_group_count: _NonNegativeInt
    total_regret_eml_dag_nodes: _NonNegativeInt
    max_regret_eml_dag_nodes: _NonNegativeInt | None
    official_cost_scoring_calls: _NonNegativeInt
    official_cost_scoring_seconds: _NonNegativeFloat

    @model_validator(mode="after")
    def validate_denominators(self) -> Self:
        if self.evaluable_group_count + self.unevaluable_group_count != self.denominator_count:
            raise ValueError("evaluable/unevaluable groups must partition denominator_count")
        if self.attempted_group_count != self.evaluable_group_count:
            raise ValueError("every evaluable group must be attempted by every ranker method")
        if (
            self.validated_selection_count + self.failed_selected_count
            != self.attempted_group_count
        ):
            raise ValueError("validated/failed selections must partition attempted groups")
        if self.exact_best_match_count > self.validated_selection_count:
            raise ValueError("exact-best matches cannot exceed validated selections")
        if self.regret_group_count > self.validated_selection_count:
            raise ValueError("regret groups cannot exceed validated selections")
        if (self.regret_group_count == 0) != (self.max_regret_eml_dag_nodes is None):
            raise ValueError("maximum regret is present exactly when regret groups exist")
        if (
            self.max_regret_eml_dag_nodes is not None
            and self.max_regret_eml_dag_nodes > self.total_regret_eml_dag_nodes
        ):
            raise ValueError("maximum regret cannot exceed total regret")
        return self


class RankerDatasetEvidence(_ArtifactBackedModel):
    """Complete issue 5-7 replay and official-label denominators."""

    group_count: _NonNegativeInt
    expression_count: _NonNegativeInt
    candidate_count: _NonNegativeInt
    valid_candidate_count: _NonNegativeInt
    failed_candidate_count: _NonNegativeInt
    official_cost_label_count: _NonNegativeInt
    replay_mismatch_count: _NonNegativeInt
    empty_group_count: _NonNegativeInt
    groups_by_split: dict[str, _NonNegativeInt]
    groups_by_source_status: dict[str, _NonNegativeInt]

    @model_validator(mode="after")
    def validate_dataset_counts(self) -> Self:
        if self.valid_candidate_count + self.failed_candidate_count != self.candidate_count:
            raise ValueError("valid/failed candidates must partition candidate_count")
        if self.official_cost_label_count > self.candidate_count:
            raise ValueError("official labels cannot exceed candidate_count")
        if self.empty_group_count > self.group_count:
            raise ValueError("empty groups cannot exceed group_count")
        if sum(self.groups_by_split.values()) != self.group_count:
            raise ValueError("split group counts must sum to group_count")
        if sum(self.groups_by_source_status.values()) != self.group_count:
            raise ValueError("source-status group counts must sum to group_count")
        if set(self.groups_by_split) != {split.value for split in SplitName}:
            raise ValueError("ranker dataset must report every corpus split")
        return self


class RankerFitEvidence(_ArtifactBackedModel):
    """Leakage-safe TRAIN fit and VALIDATION model-selection counts."""

    training_group_count: _NonNegativeInt
    training_candidate_count: _NonNegativeInt
    validation_group_count: _NonNegativeInt
    selected_ridge: _NonNegativeFloat


class RankerRuntimeEvidence(_ArtifactBackedModel):
    """Process-wide runtime and sampled process-tree memory evidence."""

    candidate_cost_scoring_observation_count: _NonNegativeInt
    candidate_cost_scoring_total_seconds: _NonNegativeFloat
    candidate_replay_active_wall_seconds: _NonNegativeFloat
    finalizing_invocation_wall_seconds_before_report: _NonNegativeFloat
    model_fit_and_evaluation_wall_seconds: _NonNegativeFloat
    peak_process_tree_rss_bytes: _NonNegativeInt
    rss_sample_count: _NonNegativeInt
    rss_sampling_policy: _NonBlankStr
    memory_scope: _NonBlankStr
    worker_processes: Annotated[StrictInt, Field(gt=0)]
    speedup_scope: Literal["official_eml_dag_cost_scoring_only"]


class NeuralRankerEvidence(_FrozenModel):
    """The exact-cost ground truth and every issue 5-7 comparator."""

    ground_truth_cost: Literal["official_pure_eml_dag_nodes"]
    dataset: RankerDatasetEvidence
    fit: RankerFitEvidence
    runtime: RankerRuntimeEvidence
    slices: tuple[RankerMethodSliceEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_keys(self) -> Self:
        keys = tuple((item.method, item.split, item.subset) for item in self.slices)
        if len(set(keys)) != len(keys):
            raise ValueError("neural-ranker evidence contains duplicate method/split/subset slices")
        grouped_counts: dict[
            tuple[SplitName, str],
            tuple[int, int, int],
        ] = {}
        for item in self.slices:
            group_key = (item.split, item.subset)
            counts = (
                item.denominator_count,
                item.evaluable_group_count,
                item.unevaluable_group_count,
            )
            prior = grouped_counts.setdefault(group_key, counts)
            if counts != prior:
                raise ValueError("ranker methods must share all/evaluable/unevaluable denominators")
            if item.method is RankerMethod.EXACT_EML_COST and (
                item.exact_best_match_count != item.validated_selection_count
                or item.failed_selected_count != 0
                or item.regret_group_count != item.validated_selection_count
                or item.total_regret_eml_dag_nodes != 0
                or item.max_regret_eml_dag_nodes not in {None, 0}
            ):
                raise ValueError(
                    "the exact EML-cost ground truth must have full exact-best matches "
                    "and zero regret"
                )
            if item.subset == "all":
                expected = self.dataset.groups_by_split.get(item.split.value)
                if expected is None or item.denominator_count != expected:
                    raise ValueError("all-subset ranker denominators must match the dataset split")
        return self


class ScientificClaim(_ArtifactBackedModel):
    """One artifact-supported directional or null conclusion."""

    claim_id: ClaimId
    outcome: ClaimOutcome
    statement: _NonBlankStr
    metric: _NonBlankStr
    split: SplitName
    subset: _NonBlankStr
    exact_denominator_count: _NonNegativeInt
    subject_value: _NonBlankStr
    baseline_value: _NonBlankStr


class GoalStatusEvidence(_ArtifactBackedModel):
    """Artifact-backed status for one cumulative project goal."""

    goal_number: Annotated[StrictInt, Field(ge=1, le=5)]
    status: GoalCompletionStatus
    summary: _NonBlankStr


class ProductionExportEvidence(_ArtifactBackedModel):
    """Authenticated issue 5-8 corpus coverage and validation totals."""

    batch_count: _NonNegativeInt
    expression_count: _NonNegativeInt
    graph_count: _NonNegativeInt
    hierarchy_count: _NonNegativeInt
    validation_failure_count: _NonNegativeInt
    reconstruction_failure_count: _NonNegativeInt
    representation_names: tuple[GraphTrackName, ...] = Field(min_length=5, max_length=5)
    subset_labels_available: bool
    subset_label_reason: _NonBlankStr
    runtime_available: bool
    runtime_reason: _NonBlankStr
    memory_available: bool
    memory_reason: _NonBlankStr

    @model_validator(mode="after")
    def validate_export_totals(self) -> Self:
        expected_names = (
            GraphTrackName.AST_DAG,
            GraphTrackName.PURE_EML_DAG,
            GraphTrackName.MACRO_DAG,
            GraphTrackName.FREQUENT_MOTIF_DAG,
            GraphTrackName.LEARNED_MOTIF_DAG,
        )
        if self.representation_names != expected_names:
            raise ValueError(
                "production export representations must use the frozen five-mode order"
            )
        if self.batch_count < 1 or self.expression_count < 1:
            raise ValueError("complete production export requires batches and expressions")
        if self.graph_count != self.expression_count * len(self.representation_names):
            raise ValueError("production graph count must provide five graphs per expression")
        if self.hierarchy_count != self.expression_count:
            raise ValueError("production hierarchy count must cover every expression")
        if self.validation_failure_count > self.graph_count:
            raise ValueError("production validation failures cannot exceed graph count")
        if self.reconstruction_failure_count > self.graph_count:
            raise ValueError("production reconstruction failures cannot exceed graph count")
        if self.subset_labels_available:
            raise ValueError(
                "issue 5-8 uses explicit-only empty subset labels; derived cohorts stay external"
            )
        if self.runtime_available or self.memory_available:
            raise ValueError("issue 5-8 does not publish runtime or memory observations")
        return self


class Goal5IntegrationEvidence(_FrozenModel):
    """Normalized, hash-bound input to the final Goal 5 integration."""

    schema_version: Literal["geml-goal5-integration-evidence-v1"] = (
        INTEGRATION_EVIDENCE_SCHEMA_VERSION
    )
    status: EvidenceStatus
    dataset_id: _NonBlankStr
    goal6_export: FrozenGoal6Export
    source_artifacts: tuple[SourceArtifact, ...] = Field(min_length=1)
    subset_definitions: tuple[SubsetDefinition, ...] = Field(min_length=2)
    cohort_joins: tuple[CrossTrackJoinEvidence, ...] = ()
    production_export: ProductionExportEvidence | None = None
    graph_tracks: tuple[GraphTrackEvidence, ...]
    neural_ranker: NeuralRankerEvidence | None
    claims: tuple[ScientificClaim, ...]
    goal_statuses: tuple[GoalStatusEvidence, ...]
    reproduction_commands: tuple[_NonBlankStr, ...] = Field(min_length=1)
    missing_requirements: tuple[_NonBlankStr, ...]

    @model_validator(mode="after")
    def validate_complete_or_explicitly_incomplete(self) -> Self:
        artifact_names = tuple(item.name for item in self.source_artifacts)
        if len(set(artifact_names)) != len(artifact_names) or artifact_names != tuple(
            sorted(artifact_names)
        ):
            raise ValueError("source artifacts must have unique names in sorted order")

        subset_names = tuple(item.name for item in self.subset_definitions)
        if len(set(subset_names)) != len(subset_names):
            raise ValueError("subset definitions must have unique names")
        if subset_names[0] != "all":
            raise ValueError("the first subset definition must be the all-processed denominator")
        if not any(item.is_nontrivial for item in self.subset_definitions):
            raise ValueError("at least one explicitly defined nontrivial subset is required")

        referenced = self._referenced_artifacts()
        missing_artifacts = referenced.difference(artifact_names)
        if missing_artifacts:
            raise ValueError(
                "evidence references unknown source artifacts: "
                + ", ".join(sorted(missing_artifacts))
            )
        unused_artifacts = set(artifact_names).difference(referenced)
        if unused_artifacts:
            raise ValueError(
                "source artifacts must support at least one reported item: "
                + ", ".join(sorted(unused_artifacts))
            )

        track_names = tuple(item.name for item in self.graph_tracks)
        if len(set(track_names)) != len(track_names):
            raise ValueError("graph track names must be unique")
        claim_ids = tuple(item.claim_id for item in self.claims)
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("scientific claim IDs must be unique")
        goal_numbers = tuple(item.goal_number for item in self.goal_statuses)
        if len(set(goal_numbers)) != len(goal_numbers):
            raise ValueError("goal status entries must be unique")

        if self.status is EvidenceStatus.INCOMPLETE:
            if not self.missing_requirements:
                raise ValueError("incomplete evidence must enumerate missing requirements")
            return self

        if self.missing_requirements:
            raise ValueError("complete evidence cannot enumerate missing requirements")
        if subset_names != ("all", "safe_nontrivial", "domain_nontrivial"):
            raise ValueError(
                "complete evidence requires all, safe_nontrivial, and "
                "domain_nontrivial subsets in frozen order"
            )
        expected_subset_modes = ("all", "safe_real", "positive_real_formal")
        if tuple(item.rewrite_mode for item in self.subset_definitions) != expected_subset_modes:
            raise ValueError("subset rewrite modes do not match the frozen cohort contract")
        expected_tracks = tuple(GraphTrackName)
        if track_names != expected_tracks:
            raise ValueError("complete evidence must contain every graph track in frozen order")
        expected_claims = tuple(ClaimId)
        if set(claim_ids) != set(expected_claims):
            raise ValueError("complete evidence must contain every required comparison claim")
        if goal_numbers != (1, 2, 3, 4, 5) or any(
            item.status is not GoalCompletionStatus.COMPLETE for item in self.goal_statuses
        ):
            raise ValueError("complete evidence requires complete, ordered Goal 1-5 statuses")
        if self.neural_ranker is None:
            raise ValueError("complete evidence requires issue 5-7 neural-ranker results")
        if self.production_export is None:
            raise ValueError("complete evidence requires issue 5-8 production-export results")

        expected_slice_keys = {(split, subset) for split in SplitName for subset in subset_names}
        reference_nontrivial_denominators: dict[tuple[SplitName, str], int] | None = None
        for track in self.graph_tracks:
            observed = {(item.split, item.subset) for item in track.slices}
            if observed != expected_slice_keys:
                raise ValueError(f"{track.name.value} must contain every split/subset denominator")
            denominators = {
                (item.split, item.subset): item.denominator_count for item in track.slices
            }
            nontrivial_denominators = {
                key: count for key, count in denominators.items() if key[1] != "all"
            }
            if reference_nontrivial_denominators is None:
                reference_nontrivial_denominators = nontrivial_denominators
            elif nontrivial_denominators != reference_nontrivial_denominators:
                raise ValueError(
                    "graph tracks must use exact nontrivial cross-track cohort denominators"
                )
            for split in SplitName:
                all_count = denominators[(split, "all")]
                if any(
                    denominators[(split, subset)] > all_count
                    for subset in ("safe_nontrivial", "domain_nontrivial")
                ):
                    raise ValueError("nontrivial graph cohorts cannot exceed the all cohort")
        expected_join_keys = tuple(
            (split, subset)
            for split in SplitName
            for subset in ("safe_nontrivial", "domain_nontrivial")
        )
        observed_join_keys = tuple((item.split, item.subset) for item in self.cohort_joins)
        if observed_join_keys != expected_join_keys:
            raise ValueError("complete evidence requires every split-specific cohort join")
        if (
            reference_nontrivial_denominators is None
        ):  # pragma: no cover - frozen tracks are nonempty
            raise ValueError("complete evidence requires graph-track denominators")
        for join in self.cohort_joins:
            if join.track_names != tuple(GraphTrackName):
                raise ValueError("cohort joins must include every graph track in frozen order")
            if (
                join.expression_count
                != reference_nontrivial_denominators[(join.split, join.subset)]
            ):
                raise ValueError("cohort join count must match graph-track denominator")
        expected_ranker_keys = {
            (method, split, subset)
            for method in RankerMethod
            for split in (
                SplitName.VALIDATION,
                SplitName.TEST_IID,
                SplitName.TEST_OOD,
            )
            for subset in subset_names
        }
        observed_ranker = {
            (item.method, item.split, item.subset) for item in self.neural_ranker.slices
        }
        if observed_ranker != expected_ranker_keys:
            raise ValueError(
                "complete neural-ranker evidence must contain every method/split/subset"
            )
        denominator_by_slice: dict[tuple[SplitName, str], int] = {}
        for item in self.neural_ranker.slices:
            key = (item.split, item.subset)
            previous = denominator_by_slice.setdefault(key, item.denominator_count)
            if previous != item.denominator_count:
                raise ValueError("ranker methods must use exact shared cohort denominators")
        for split in (SplitName.VALIDATION, SplitName.TEST_IID, SplitName.TEST_OOD):
            all_count = denominator_by_slice[(split, "all")]
            if any(
                denominator_by_slice[(split, subset)] > all_count
                for subset in ("safe_nontrivial", "domain_nontrivial")
            ):
                raise ValueError("nontrivial ranker cohorts cannot exceed the all cohort")
            for subset in ("safe_nontrivial", "domain_nontrivial"):
                if (
                    denominator_by_slice[(split, subset)]
                    != reference_nontrivial_denominators[(split, subset)]
                ):
                    raise ValueError(
                        "ranker nontrivial denominators must match exact expression-ID joins"
                    )
        neural_denominators = {
            (item.split, item.subset): item.denominator_count
            for item in self.neural_ranker.slices
            if item.method is RankerMethod.NEURAL
        }
        for claim in self.claims:
            key = (claim.split, claim.subset)
            expected_denominator = (
                neural_denominators.get(key)
                if claim.claim_id is ClaimId.NEURAL_VS_HEURISTICS
                else self._motif_claim_denominator(key)
            )
            if expected_denominator is None:
                raise ValueError("scientific claim references an unknown split/subset")
            if claim.exact_denominator_count != expected_denominator:
                raise ValueError("scientific claim denominator must match its authenticated cohort")
        return self

    def _referenced_artifacts(self) -> set[str]:
        referenced: set[str] = set()
        for subset in self.subset_definitions:
            referenced.update(subset.source_artifacts)
        for join in self.cohort_joins:
            referenced.update(join.source_artifacts)
        if self.production_export is not None:
            referenced.update(self.production_export.source_artifacts)
        for track in self.graph_tracks:
            for item in track.slices:
                referenced.update(item.source_artifacts)
                for metric in (
                    item.node_count,
                    item.edge_count,
                    item.mdl_cost,
                    item.structural_validation,
                    item.reconstruction,
                    item.expansion,
                    item.runtime,
                    item.memory,
                ):
                    referenced.update(metric.source_artifacts)
        if self.neural_ranker is not None:
            referenced.update(self.neural_ranker.dataset.source_artifacts)
            referenced.update(self.neural_ranker.fit.source_artifacts)
            referenced.update(self.neural_ranker.runtime.source_artifacts)
            for item in self.neural_ranker.slices:
                referenced.update(item.source_artifacts)
        for item in self.claims:
            referenced.update(item.source_artifacts)
        for item in self.goal_statuses:
            referenced.update(item.source_artifacts)
        return referenced

    def _motif_claim_denominator(self, key: tuple[SplitName, str]) -> int | None:
        for track in self.graph_tracks:
            if track.name is GraphTrackName.LEARNED_MOTIF_DAG:
                return next(
                    (
                        item.denominator_count
                        for item in track.slices
                        if (item.split, item.subset) == key
                    ),
                    None,
                )
        return None


@dataclass(frozen=True, slots=True)
class ExactRatio:
    """A non-lossy fraction, including an explicit zero-denominator sentinel."""

    numerator: int
    denominator: int

    @property
    def fraction(self) -> Fraction | None:
        if self.denominator == 0:
            return None
        return Fraction(self.numerator, self.denominator)

    @property
    def exact(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    def as_dict(self) -> dict[str, int | str | None]:
        value = self.fraction
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "exact": self.exact,
            "decimal": None if value is None else f"{float(value):.12g}",
        }


@dataclass(frozen=True, slots=True)
class GraphSliceSummary:
    evidence: GraphSliceEvidence
    success_rate: ExactRatio
    mean_nodes: ExactRatio
    mean_edges: ExactRatio
    mean_mdl_bits: ExactRatio

    def as_dict(self) -> dict[str, object]:
        result = self.evidence.model_dump(mode="json")
        result.update(
            {
                "success_rate": self.success_rate.as_dict(),
                "mean_nodes": self.mean_nodes.as_dict(),
                "mean_edges": self.mean_edges.as_dict(),
                "mean_mdl_bits": self.mean_mdl_bits.as_dict(),
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class GraphTrackSummary:
    name: GraphTrackName
    display_name: str
    representation_family: str
    representation_mode: str
    is_pure_eml: bool
    slices: tuple[GraphSliceSummary, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name.value,
            "display_name": self.display_name,
            "representation_family": self.representation_family,
            "representation_mode": self.representation_mode,
            "is_pure_eml": self.is_pure_eml,
            "slices": [item.as_dict() for item in self.slices],
        }


@dataclass(frozen=True, slots=True)
class ScopedSpeedup:
    """Exact-scoring seconds divided by comparator-scoring seconds."""

    exact_scoring_seconds: float
    comparator_scoring_seconds: float
    available: bool
    unavailable_reason: str | None

    @property
    def value(self) -> float | None:
        if not self.available:
            return None
        return self.exact_scoring_seconds / self.comparator_scoring_seconds

    def as_dict(self) -> dict[str, object]:
        return {
            "scope": "official_eml_dag_cost_scoring_only",
            "exact_scoring_seconds": self.exact_scoring_seconds,
            "comparator_scoring_seconds": self.comparator_scoring_seconds,
            "value": self.value,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class RankerMethodSliceSummary:
    evidence: RankerMethodSliceEvidence
    evaluable_rate: ExactRatio
    validation_rate_all_groups: ExactRatio
    validation_rate: ExactRatio
    exact_best_rate: ExactRatio
    mean_regret_eml_dag_nodes: ExactRatio
    cost_scoring_speedup: ScopedSpeedup

    def as_dict(self) -> dict[str, object]:
        result = self.evidence.model_dump(mode="json")
        result.update(
            {
                "evaluable_rate": self.evaluable_rate.as_dict(),
                "validation_rate_all_groups": self.validation_rate_all_groups.as_dict(),
                "validation_rate": self.validation_rate.as_dict(),
                "exact_best_rate": self.exact_best_rate.as_dict(),
                "mean_regret_eml_dag_nodes": self.mean_regret_eml_dag_nodes.as_dict(),
                "cost_scoring_speedup": self.cost_scoring_speedup.as_dict(),
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class Goal5Summary:
    """Fully derived report object; source evidence remains embedded and exact."""

    evidence: Goal5IntegrationEvidence
    graph_tracks: tuple[GraphTrackSummary, ...]
    ranker_slices: tuple[RankerMethodSliceSummary, ...]
    ordered_claims: tuple[ScientificClaim, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": INTEGRATION_SUMMARY_SCHEMA_VERSION,
            "evidence_schema_version": self.evidence.schema_version,
            "status": self.evidence.status.value,
            "dataset_id": self.evidence.dataset_id,
            "goal6_export": self.evidence.goal6_export.model_dump(mode="json"),
            "source_artifacts": [
                item.model_dump(mode="json") for item in self.evidence.source_artifacts
            ],
            "subset_definitions": [
                item.model_dump(mode="json") for item in self.evidence.subset_definitions
            ],
            "cohort_joins": [item.model_dump(mode="json") for item in self.evidence.cohort_joins],
            "production_export": (
                None
                if self.evidence.production_export is None
                else self.evidence.production_export.model_dump(mode="json")
            ),
            "graph_tracks": [item.as_dict() for item in self.graph_tracks],
            "neural_ranker": {
                "ground_truth_cost": (
                    None
                    if self.evidence.neural_ranker is None
                    else self.evidence.neural_ranker.ground_truth_cost
                ),
                "dataset": (
                    None
                    if self.evidence.neural_ranker is None
                    else self.evidence.neural_ranker.dataset.model_dump(mode="json")
                ),
                "fit": (
                    None
                    if self.evidence.neural_ranker is None
                    else self.evidence.neural_ranker.fit.model_dump(mode="json")
                ),
                "runtime": (
                    None
                    if self.evidence.neural_ranker is None
                    else self.evidence.neural_ranker.runtime.model_dump(mode="json")
                ),
                "slices": [item.as_dict() for item in self.ranker_slices],
            },
            "claims": [item.model_dump(mode="json") for item in self.ordered_claims],
            "goal_statuses": [item.model_dump(mode="json") for item in self.evidence.goal_statuses],
            "reproduction_commands": list(self.evidence.reproduction_commands),
            "missing_requirements": list(self.evidence.missing_requirements),
            "scientific_boundaries": list(scientific_boundaries()),
        }


def scientific_boundaries() -> tuple[str, ...]:
    """Return fixed interpretation boundaries that reports must state verbatim."""

    return (
        "Pure EML-DAG and both e-graph-selected EML-DAG tracks contain only pure EML nodes.",
        "Macro and motif nodes are compression vocabulary nodes, not pure EML nodes.",
        (
            "Macro graphs are structurally close to labeled compiler/AST graphs and must not "
            "be described as single-operator EML."
        ),
        (
            "Safe-real nontrivial cohorts use branch-insensitive finite-real rewrites; "
            "domain-aware cohorts are conditional positive-real-formal results under each "
            "row's recorded assumptions."
        ),
        (
            "Standalone graph MDL and dictionary-inclusive motif MDL are distinct scopes "
            "and are never combined into one aggregate."
        ),
        (
            "Neural-ranker speedup is scoped only to candidate cost scoring; it is not an "
            "end-to-end pipeline or mathematical-reasoning speedup."
        ),
        (
            "Every failure, unsupported case, validation failure, and missing resource "
            "observation remains visible in its all-attempted denominator."
        ),
        (
            "All-processed denominators remain track-specific: the Goal 4 e-graph study "
            "covers its frozen 30,000-expression selection, while issue 5-8 covers the "
            "full 250,000-expression corpus. Exact cross-track comparisons use the "
            "separately joined nontrivial cohorts."
        ),
    )


def summarize(evidence: Goal5IntegrationEvidence) -> Goal5Summary:
    """Derive exact rates while preserving every source count and null result."""

    graph_tracks = tuple(
        GraphTrackSummary(
            name=track.name,
            display_name=track.display_name,
            representation_family=track.representation_family,
            representation_mode=track.representation_mode,
            is_pure_eml=track.is_pure_eml,
            slices=tuple(
                GraphSliceSummary(
                    evidence=item,
                    success_rate=ExactRatio(item.success_count, item.denominator_count),
                    mean_nodes=ExactRatio(
                        item.node_count.total or 0,
                        item.node_count.observation_count,
                    ),
                    mean_edges=ExactRatio(
                        item.edge_count.total or 0,
                        item.edge_count.observation_count,
                    ),
                    mean_mdl_bits=ExactRatio(
                        item.mdl_cost.total_bits or 0,
                        item.mdl_cost.observation_count,
                    ),
                )
                for item in track.slices
            ),
        )
        for track in evidence.graph_tracks
    )
    if evidence.neural_ranker is None:
        ranker_slices = ()
    else:
        exact_scoring = {
            (item.split, item.subset): item
            for item in evidence.neural_ranker.slices
            if item.method is RankerMethod.EXACT_EML_COST
        }
        ranker_slices = tuple(
            RankerMethodSliceSummary(
                evidence=item,
                evaluable_rate=ExactRatio(
                    item.evaluable_group_count,
                    item.denominator_count,
                ),
                validation_rate_all_groups=ExactRatio(
                    item.validated_selection_count,
                    item.denominator_count,
                ),
                validation_rate=ExactRatio(
                    item.validated_selection_count,
                    item.attempted_group_count,
                ),
                exact_best_rate=ExactRatio(
                    item.exact_best_match_count,
                    item.attempted_group_count,
                ),
                mean_regret_eml_dag_nodes=ExactRatio(
                    item.total_regret_eml_dag_nodes,
                    item.regret_group_count,
                ),
                cost_scoring_speedup=_scoring_speedup(
                    item,
                    exact_scoring.get((item.split, item.subset)),
                ),
            )
            for item in evidence.neural_ranker.slices
        )
    outcome_order = {
        ClaimOutcome.NULL_RESULT: 0,
        ClaimOutcome.NEGATIVE: 1,
        ClaimOutcome.INCONCLUSIVE: 2,
        ClaimOutcome.POSITIVE: 3,
    }
    ordered_claims = tuple(
        sorted(
            evidence.claims,
            key=lambda item: (outcome_order[item.outcome], item.claim_id.value),
        )
    )
    return Goal5Summary(
        evidence=evidence,
        graph_tracks=graph_tracks,
        ranker_slices=ranker_slices,
        ordered_claims=ordered_claims,
    )


def _scoring_speedup(
    item: RankerMethodSliceEvidence,
    exact: RankerMethodSliceEvidence | None,
) -> ScopedSpeedup:
    if exact is None:
        return ScopedSpeedup(
            0.0,
            item.official_cost_scoring_seconds,
            False,
            "exact baseline absent",
        )
    if exact.denominator_count != item.denominator_count:
        return ScopedSpeedup(
            exact.official_cost_scoring_seconds,
            item.official_cost_scoring_seconds,
            False,
            "method and exact baseline denominators differ",
        )
    if item.official_cost_scoring_seconds == 0.0:
        return ScopedSpeedup(
            exact.official_cost_scoring_seconds,
            item.official_cost_scoring_seconds,
            False,
            "comparator scoring runtime is zero",
        )
    return ScopedSpeedup(
        exact.official_cost_scoring_seconds,
        item.official_cost_scoring_seconds,
        True,
        None,
    )


def _ratio_text(value: ExactRatio) -> str:
    if value.denominator == 0:
        return "n/a (0/0)"
    return f"{value.exact} ({100 * value.numerator / value.denominator:.3f}%)"


def _artifact_refs(names: tuple[str, ...]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _audit_text(value: AuditCounts) -> str:
    if value.availability is not MetricAvailability.AVAILABLE:
        return (
            f"{value.availability.value}: {value.unavailable_reason} "
            f"[{_artifact_refs(value.source_artifacts)}]"
        )
    return (
        f"{value.passed_count}/{value.attempted_count}; "
        f"observed {value.attempted_count}/{value.denominator_count}"
    )


def _mean_metric_text(
    metric: IntegerMetricObservation | MdlMetricObservation,
    mean: ExactRatio,
) -> str:
    if metric.availability is not MetricAvailability.AVAILABLE:
        return (
            f"{metric.availability.value}: {metric.unavailable_reason} "
            f"[{_artifact_refs(metric.source_artifacts)}]"
        )
    return f"`{mean.exact}`; observed {metric.observation_count}/{metric.denominator_count}"


def _resource_text(metric: FloatMetricObservation | PeakMetricObservation) -> str:
    if metric.availability is not MetricAvailability.AVAILABLE:
        return (
            f"{metric.availability.value}: {metric.unavailable_reason} "
            f"[{_artifact_refs(metric.source_artifacts)}]"
        )
    value = metric.total if isinstance(metric, FloatMetricObservation) else metric.peak
    if value is None:
        return "n/a (available metric; empty 0/0 cohort)"
    return f"{value} {metric.unit}; observed {metric.observation_count}/{metric.denominator_count}"


def _failure_text(row: GraphSliceEvidence) -> str:
    if not row.failure_counts:
        return "0"
    categories = ", ".join(f"`{name}`={count}" for name, count in row.failure_counts.items())
    return f"{row.failure_count} ({categories})"


def render_goal5_summary_markdown(summary: Goal5Summary) -> str:
    """Render the final Goal 5 status document from authenticated evidence."""

    evidence = summary.evidence
    lines = [
        "# Goal 5 summary: lossless macro and motif compression",
        "",
        f"**Integration status:** `{evidence.status.value}`",
        "",
    ]
    if evidence.status is EvidenceStatus.INCOMPLETE:
        lines.extend(
            [
                "> **INCOMPLETE — no production conclusion is reported.** The required "
                "producer artifacts have not all passed integration validation.",
                "",
                "Missing requirements:",
                "",
                *(f"- {item}" for item in evidence.missing_requirements),
                "",
            ]
        )
    lines.extend(
        [
            "## Scientific boundaries",
            "",
            *(f"- {item}" for item in scientific_boundaries()),
            "",
            "## Frozen Goal 6 export",
            "",
            "| Contract | Frozen value |",
            "|---|---|",
            f"| Export root | `{evidence.goal6_export.export_root}` |",
            f"| Run directory | `{evidence.goal6_export.run_directory_pattern}` |",
            f"| Completion file | `{evidence.goal6_export.completion_filename}` |",
            (
                "| Production manifest schema | "
                f"`{evidence.goal6_export.production_manifest_schema_version}` |"
            ),
            (f"| Model feature schema | `{evidence.goal6_export.model_feature_schema_version}` |"),
            f"| Hierarchy schema | `{evidence.goal6_export.hierarchy_schema_version}` |",
            "",
            "## Production export audit",
            "",
        ]
    )
    if evidence.production_export is None:
        lines.extend(["No authenticated issue 5-8 production export is available.", ""])
    else:
        export = evidence.production_export
        lines.extend(
            [
                "| Batches | Expressions | Graphs | Hierarchies | Validation failures | "
                "Reconstruction failures |",
                "|---:|---:|---:|---:|---:|---:|",
                f"| {export.batch_count} | {export.expression_count} | "
                f"{export.graph_count} | {export.hierarchy_count} | "
                f"{export.validation_failure_count} | "
                f"{export.reconstruction_failure_count} |",
                "",
                f"- Subset labels: unavailable — {export.subset_label_reason}",
                f"- Runtime: unavailable — {export.runtime_reason}",
                f"- Memory: unavailable — {export.memory_reason}",
                "",
            ]
        )
    lines.extend(
        [
            "## Representation results",
            "",
        ]
    )
    if not summary.graph_tracks:
        lines.extend(["No validated representation rows are available yet.", ""])
    else:
        lines.extend(
            [
                (
                    "| Track | Split | Subset | Success / all attempted | Mean nodes | "
                    "Mean edges | Mean MDL bits | Structural | Reconstruction | Expansion | "
                    "Failures | Runtime observations | Peak memory |"
                ),
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for track in summary.graph_tracks:
            for item in track.slices:
                row = item.evidence
                mdl_annotation = f"`{row.mdl_cost.scope.value}`"
                if row.mdl_cost.codec is not None:
                    mdl_annotation += f"; `{row.mdl_cost.codec}`"
                lines.append(
                    f"| {track.display_name} | `{row.split.value}` | `{row.subset}` | "
                    f"{_ratio_text(item.success_rate)} | "
                    f"{_mean_metric_text(row.node_count, item.mean_nodes)} | "
                    f"{_mean_metric_text(row.edge_count, item.mean_edges)} | "
                    f"{_mean_metric_text(row.mdl_cost, item.mean_mdl_bits)} "
                    f"({mdl_annotation}) | "
                    f"{_audit_text(row.structural_validation)} | "
                    f"{_audit_text(row.reconstruction)} | "
                    f"{_audit_text(row.expansion)} | "
                    f"{_failure_text(row)} | "
                    f"{_resource_text(row.runtime)} | "
                    f"{_resource_text(row.memory)} |"
                )
        lines.append("")
    lines.extend(["## Learned and neural comparisons", ""])
    if not summary.ordered_claims:
        lines.extend(["No artifact-supported comparison claim is available yet.", ""])
    else:
        lines.extend(
            [
                "| Outcome | Comparison | Split / subset | Denominator | Values | Sources |",
                "|---|---|---|---:|---|---|",
            ]
        )
        for claim in summary.ordered_claims:
            lines.append(
                f"| **{claim.outcome.value}** | {claim.statement} | "
                f"`{claim.split.value}` / `{claim.subset}` | "
                f"{claim.exact_denominator_count} | "
                f"{claim.subject_value} vs. {claim.baseline_value} "
                f"(`{claim.metric}`) | {_artifact_refs(claim.source_artifacts)} |"
            )
        lines.append("")
    lines.extend(
        [
            "Null, negative, and inconclusive findings are ordered before positive findings "
            "so a favorable result cannot visually displace a null result.",
            "",
            "## Reproduction",
            "",
            *(f"```text\n{command}\n```" for command in evidence.reproduction_commands),
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_compression_study_markdown(summary: Goal5Summary) -> str:
    """Render the method/metric-focused Goal 5 compression study."""

    evidence = summary.evidence
    lines = [
        "# Goal 5 compression study",
        "",
        f"**Evidence status:** `{evidence.status.value}`",
        "",
        (
            "> This document is generated only from the normalized integration evidence. "
            "It does not infer missing producer results."
        ),
        "",
        "## Denominators and metrics",
        "",
        "- Every graph row reports successes and retained failures over all attempted inputs.",
        "- Node count, edge count, and MDL each declare availability, exact observed/missing "
        "denominators, and authenticated reasons when unavailable.",
        "- Standalone graph MDL is not pooled with dictionary-inclusive motif MDL.",
        "- Structural validation, reconstruction, and expansion separately declare "
        "available, unavailable, or not-applicable state and exact pass/fail counts.",
        "- Runtime and memory separately declare availability and observed/missing counts.",
        "- Neural exact-best match, validation rate, regret, and cost-scoring runtime use "
        "candidate-group denominators; failed selected candidates remain present.",
        "- Goal 4 e-graph all-processed rows and issue 5-8 full-corpus rows retain their "
        "different exact denominators; cross-track subset comparisons use exact ID joins.",
        "",
        "## Production export coverage",
        "",
    ]
    if evidence.production_export is None:
        lines.extend(["No authenticated issue 5-8 production export is available.", ""])
    else:
        export = evidence.production_export
        lines.extend(
            [
                f"- {export.batch_count} immutable batches cover "
                f"{export.expression_count} expressions, {export.graph_count} graphs, "
                f"and {export.hierarchy_count} hierarchy records.",
                f"- Validation failures: {export.validation_failure_count}; "
                f"reconstruction failures: {export.reconstruction_failure_count}.",
                f"- 5-8 subset labels are unavailable: {export.subset_label_reason}",
                f"- 5-8 runtime is unavailable: {export.runtime_reason}",
                f"- 5-8 memory is unavailable: {export.memory_reason}",
                "",
            ]
        )
    lines.extend(
        [
            "## Subset definitions",
            "",
        ]
    )
    lines.extend(
        (
            f"- `{item.name}` (`{item.rewrite_mode}`, IDs "
            f"`{item.expression_ids_sha256}`): {item.definition}; {item.semantics}. "
            f"Sources: {_artifact_refs(item.source_artifacts)}"
        )
        for item in evidence.subset_definitions
    )
    lines.extend(["", "## Exact cross-track cohort joins", ""])
    if not evidence.cohort_joins:
        lines.extend(["No authenticated cross-track cohort joins are available.", ""])
    else:
        lines.extend(
            [
                "| Split | Subset | Expressions | ID digest | Tracks | Sources |",
                "|---|---|---:|---|---:|---|",
            ]
        )
        for join in evidence.cohort_joins:
            lines.append(
                f"| `{join.split.value}` | `{join.subset}` | {join.expression_count} | "
                f"`{join.expression_ids_sha256}` | {len(join.track_names)} | "
                f"{_artifact_refs(join.source_artifacts)} |"
            )
        lines.append("")
    lines.extend(["", "## Neural ranker", ""])
    if not summary.ranker_slices:
        lines.extend(
            [
                "Issue 5-7 evidence is not available. No neural-versus-heuristic conclusion "
                "is reported.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "| Method | Split | Subset | Evaluable / all | Valid / all | "
                    "Validation rate | Exact-best rate | Failed selected | Mean regret | "
                    "Cost-scoring seconds | Speedup vs exact |"
                ),
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in summary.ranker_slices:
            row = item.evidence
            speedup = item.cost_scoring_speedup
            speedup_text = (
                f"{speedup.value:.6g}x"
                if speedup.value is not None
                else f"n/a ({speedup.unavailable_reason})"
            )
            lines.append(
                f"| `{row.method.value}` | `{row.split.value}` | `{row.subset}` | "
                f"{_ratio_text(item.evaluable_rate)} | "
                f"{_ratio_text(item.validation_rate_all_groups)} | "
                f"{_ratio_text(item.validation_rate)} | "
                f"{_ratio_text(item.exact_best_rate)} | "
                f"{row.failed_selected_count} | "
                f"`{item.mean_regret_eml_dag_nodes.exact}` | "
                f"{row.official_cost_scoring_seconds:.9g} | {speedup_text} |"
            )
        lines.append("")
        runtime = evidence.neural_ranker.runtime
        lines.extend(
            [
                (
                    "Issue 5-7 process-wide runtime: "
                    f"{runtime.candidate_replay_active_wall_seconds:.9g}s replay, "
                    f"{runtime.model_fit_and_evaluation_wall_seconds:.9g}s fit/evaluation; "
                    f"peak sampled process-tree RSS {runtime.peak_process_tree_rss_bytes} bytes "
                    f"across {runtime.rss_sample_count} samples."
                ),
                "",
            ]
        )
    lines.extend(["## Artifact provenance", ""])
    lines.extend(
        [
            "| Name | Schema | Size | SHA-256 | Path |",
            "|---|---|---:|---|---|",
        ]
    )
    for artifact in evidence.source_artifacts:
        schema = (
            artifact.schema_version
            if artifact.schema_state is ArtifactSchemaState.VERSIONED
            else f"explicitly unversioned: {artifact.unversioned_reason}"
        )
        lines.append(
            f"| `{artifact.name}` | {schema} | {artifact.size_bytes} | "
            f"`{artifact.sha256}` | `{artifact.path}` |"
        )
    lines.extend(["", "## Interpretation boundaries", ""])
    lines.extend(f"- {item}" for item in scientific_boundaries())
    lines.append("")
    return "\n".join(lines)


def render_goals_1_to_5_status_markdown(summary: Goal5Summary) -> str:
    """Render the cumulative status report without inventing earlier-goal claims."""

    evidence = summary.evidence
    lines = [
        "# Final Goals 1-5 status",
        "",
        f"**Goal 5 integration evidence:** `{evidence.status.value}`",
        "",
    ]
    if evidence.status is EvidenceStatus.INCOMPLETE:
        lines.extend(
            [
                "> **NOT FINAL.** This report remains incomplete until every prerequisite "
                "artifact is authenticated and the Goal 5 integration run succeeds.",
                "",
            ]
        )
    lines.extend(
        [
            "| Goal | Status | Artifact-backed summary | Sources |",
            "|---:|---|---|---|",
        ]
    )
    for item in evidence.goal_statuses:
        lines.append(
            f"| {item.goal_number} | `{item.status.value}` | {item.summary} | "
            f"{_artifact_refs(item.source_artifacts)} |"
        )
    if not evidence.goal_statuses:
        lines.append("| — | `incomplete` | No authenticated goal-status entries yet. | — |")
    lines.extend(
        [
            "",
            "## Goal 6 handoff",
            "",
            f"- Export root: `{evidence.goal6_export.export_root}`",
            f"- Run directory: `{evidence.goal6_export.run_directory_pattern}`",
            (
                "- Production manifest schema: "
                f"`{evidence.goal6_export.production_manifest_schema_version}`"
            ),
            f"- Model feature schema: `{evidence.goal6_export.model_feature_schema_version}`",
            f"- Hierarchy schema: `{evidence.goal6_export.hierarchy_schema_version}`",
            "",
            "The handoff is usable only when the integration evidence status is `complete` "
            "and every cited byte digest has been revalidated.",
            "",
        ]
    )
    return "\n".join(lines)
