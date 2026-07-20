"""Validated contract for source-expression records."""

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    StringConstraints,
    field_validator,
)

from geml.contracts.corpus import CorpusSplit

_NonBlankStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_NonNegativeInt = Annotated[StrictInt, Field(ge=0)]


class ExpressionRecord(BaseModel):
    """A generated source expression and its non-derived provenance fields.

    ``sympy_srepr`` is the authoritative structural expression. ``display_text`` and
    ``latex_text`` are display-only fields and must not be parsed as source authority.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    expression_id: _NonBlankStr
    sympy_srepr: _NonBlankStr
    display_text: _NonBlankStr
    latex_text: _NonBlankStr | None = None
    split: CorpusSplit
    operator_family: _NonBlankStr
    domain_mode: _NonBlankStr
    variables: tuple[_NonBlankStr, ...]
    target_ast_size: _NonNegativeInt
    target_depth: _NonNegativeInt
    generator_seed: StrictInt
    generator_metadata: dict[str, JsonValue]

    @field_validator("variables")
    @classmethod
    def validate_variables(cls, variables: tuple[str, ...]) -> tuple[str, ...]:
        """Require an ordered list without duplicate variable names."""
        if len(set(variables)) != len(variables):
            raise ValueError("variables must not contain duplicates")
        return variables
