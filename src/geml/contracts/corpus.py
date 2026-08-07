"""Frozen contracts for corpus and result-bearing run manifests."""

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Final, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    model_validator,
)

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
_HexDigest = Annotated[
    str,
    StringConstraints(pattern=r"^(?:[0-9a-fA-F]{2})+$"),
]


def _require_datetime_input(value: object) -> object:
    """Reject Pydantic's numeric-to-datetime coercion before timezone validation."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("timestamp strings must use ISO 8601 format") from error
    raise ValueError("timestamps must be datetime objects or ISO 8601 strings")


_AwareTimestamp = Annotated[AwareDatetime, BeforeValidator(_require_datetime_input)]


class CorpusSplit(StrEnum):
    """Allowed corpus split names."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST_IID = "test_iid"
    TEST_OOD = "test_ood"


FINAL_CORPUS_SPLIT_COUNTS: Final[Mapping[CorpusSplit, int]] = MappingProxyType(
    {
        CorpusSplit.TRAIN: 175_000,
        CorpusSplit.VALIDATION: 25_000,
        CorpusSplit.TEST_IID: 25_000,
        CorpusSplit.TEST_OOD: 25_000,
    }
)
FINAL_CORPUS_TOTAL_COUNT: Final[int] = sum(FINAL_CORPUS_SPLIT_COUNTS.values())


class _CorpusContract(BaseModel):
    """Shared validation policy for corpus records."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ChecksumRecord(_CorpusContract):
    """A named hexadecimal checksum supplied by a producing issue."""

    algorithm: _NonBlankStr
    digest: _HexDigest


class CorpusShardManifest(_CorpusContract):
    """Manifest metadata for one immutable corpus shard."""

    schema_version: _NonBlankStr
    corpus_id: _NonBlankStr
    shard_id: _NonBlankStr
    path: _NonBlankStr
    split: CorpusSplit
    shard_index: _NonNegativeInt
    row_count: _NonNegativeInt
    byte_count: _NonNegativeInt | None = None
    checksum: ChecksumRecord
    error_row_count: _NonNegativeInt = 0
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SplitManifest(_CorpusContract):
    """The ordered shards and declared totals for one corpus split."""

    schema_version: _NonBlankStr
    corpus_id: _NonBlankStr
    split: CorpusSplit
    shards: tuple[CorpusShardManifest, ...] = Field(min_length=1)
    total_row_count: _NonNegativeInt
    total_error_row_count: _NonNegativeInt = 0
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_shards_and_totals(self) -> Self:
        """Require unique, compatible shards and exact declared totals."""
        shard_ids = [shard.shard_id for shard in self.shards]
        if len(set(shard_ids)) != len(shard_ids):
            raise ValueError("split manifest contains duplicate shard_id values")

        shard_indexes = [shard.shard_index for shard in self.shards]
        if shard_indexes != list(range(len(self.shards))):
            raise ValueError("shard_index values must be contiguous and match shard order")

        for shard in self.shards:
            if shard.schema_version != self.schema_version:
                raise ValueError("shard schema_version must match its split manifest")
            if shard.corpus_id != self.corpus_id:
                raise ValueError("shard corpus_id must match its split manifest")
            if shard.split != self.split:
                raise ValueError("shard split must match its split manifest")

        if self.total_row_count != sum(shard.row_count for shard in self.shards):
            raise ValueError("total_row_count must equal the sum of shard row counts")
        if self.total_error_row_count != sum(shard.error_row_count for shard in self.shards):
            raise ValueError("total_error_row_count must equal the sum of shard error counts")
        return self


class CorpusManifest(_CorpusContract):
    """Top-level manifest for a complete corpus, including tiny fixtures."""

    schema_version: _NonBlankStr
    corpus_id: _NonBlankStr
    splits: tuple[SplitManifest, ...] = Field(min_length=1)
    total_row_count: _NonNegativeInt
    total_error_row_count: _NonNegativeInt = 0
    config_hash: _NonBlankStr
    generator_seed: StrictInt
    created_at: _AwareTimestamp
    git_commit: _NonBlankStr
    python_version: _NonBlankStr
    platform: _NonBlankStr
    package_versions: dict[_NonBlankStr, _NonBlankStr] = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_splits_and_totals(self) -> Self:
        """Require compatible unique splits/shards and exact declared totals."""
        split_names = [split.split for split in self.splits]
        if len(set(split_names)) != len(split_names):
            raise ValueError("corpus manifest contains duplicate split names")

        shard_ids = [shard.shard_id for split in self.splits for shard in split.shards]
        if len(set(shard_ids)) != len(shard_ids):
            raise ValueError("corpus manifest contains duplicate shard_id values")

        for split in self.splits:
            if split.schema_version != self.schema_version:
                raise ValueError("split schema_version must match the corpus manifest")
            if split.corpus_id != self.corpus_id:
                raise ValueError("split corpus_id must match the corpus manifest")

        if self.total_row_count != sum(split.total_row_count for split in self.splits):
            raise ValueError("total_row_count must equal the sum of split totals")
        if self.total_error_row_count != sum(split.total_error_row_count for split in self.splits):
            raise ValueError("total_error_row_count must equal the sum of split error totals")
        return self


class RunMetadata(_CorpusContract):
    """Reproducibility and accounting fields for one result-bearing run."""

    run_id: _NonBlankStr
    stage: _NonBlankStr
    config_hash: _NonBlankStr
    random_seed: StrictInt
    git_commit: _NonBlankStr
    python_version: _NonBlankStr
    platform: _NonBlankStr
    package_versions: dict[_NonBlankStr, _NonBlankStr] = Field(min_length=1)
    started_at: _AwareTimestamp
    ended_at: _AwareTimestamp
    elapsed_seconds: Annotated[StrictFloat, Field(ge=0)]
    input_manifests: tuple[_NonBlankStr, ...] = Field(min_length=1)
    processed_count: _NonNegativeInt
    success_count: _NonNegativeInt
    failure_count: _NonNegativeInt
    reproduction_command: _NonBlankStr
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time_and_accounting(self) -> Self:
        """Check chronology and preserve complete success/failure denominators."""
        if self.ended_at < self.started_at:
            raise ValueError("ended_at cannot precede started_at")
        if self.processed_count != self.success_count + self.failure_count:
            raise ValueError("processed_count must equal success_count + failure_count")
        return self


class ErrorRow(_CorpusContract):
    """A retained processing failure, timeout, unsupported case, or validation error."""

    expression_id: _NonBlankStr | None = None
    shard_id: _NonBlankStr | None = None
    stage: _NonBlankStr
    error_type: _NonBlankStr
    message: _NonBlankStr
    recoverable: StrictBool
    status: _NonBlankStr
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
