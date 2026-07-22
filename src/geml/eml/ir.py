"""Immutable recursive representation of a pure EML expression tree.

The result-bearing vocabulary is deliberately tiny: the binary ``eml`` operator,
the primitive constant ``1``, and source-variable occurrences.  Compiler helper
names never become nodes in this representation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeGuard

_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class One:
    """One occurrence of the primitive EML constant ``1``."""


@dataclass(frozen=True, slots=True)
class Variable:
    """One source-variable occurrence in a pure EML tree."""

    name: str

    def __post_init__(self) -> None:
        if not is_valid_source_variable_name(self.name):
            raise ValueError("EML variable names must be nonblank ASCII identifiers")


@dataclass(frozen=True, slots=True)
class EML:
    """One ordered binary ``eml(left, right)`` occurrence."""

    left: EMLTerm
    right: EMLTerm

    def __post_init__(self) -> None:
        for slot, child in (("left", self.left), ("right", self.right)):
            if not is_eml_term(child):
                raise TypeError(f"EML {slot} child must be a pure EML term")


type EMLTerm = One | Variable | EML


def is_valid_source_variable_name(value: object) -> TypeGuard[str]:
    """Return whether ``value`` is a syntactically bare EML variable name.

    This lexical integrity guard prevents a leaf from concealing source syntax
    such as ``x+y``, ``log(x)``, or ``EML[x,1]``.  It is not a corpus-symbol
    allowlist; callers remain responsible for checking membership in the
    authoritative source expression's variable set.
    """

    return isinstance(value, str) and _VARIABLE_NAME.fullmatch(value) is not None


def is_eml_term(value: object) -> TypeGuard[EMLTerm]:
    """Return whether ``value`` has one of the three exact pure-EML types."""

    return type(value) in (One, Variable, EML)


def one() -> One:
    """Return a fresh primitive-one occurrence."""

    return One()


def variable(name: str) -> Variable:
    """Return a validated source-variable occurrence."""

    return Variable(name)


def eml(left: EMLTerm, right: EMLTerm) -> EML:
    """Return an ordered binary EML occurrence."""

    return EML(left=left, right=right)
