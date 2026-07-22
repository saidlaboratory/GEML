"""Immutable recursive representation of a pure EML expression tree.

The result-bearing vocabulary is deliberately tiny: the binary ``eml`` operator,
the primitive constant ``1``, and source-variable occurrences.  Compiler helper
names never become nodes in this representation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True, slots=True)
class One:
    """One occurrence of the primitive EML constant ``1``."""


@dataclass(frozen=True, slots=True)
class Variable:
    """One source-variable occurrence in a pure EML tree."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _VARIABLE_NAME.fullmatch(self.name) is None:
            raise ValueError("EML variable names must be nonblank ASCII identifiers")


@dataclass(frozen=True, slots=True)
class EML:
    """One ordered binary ``eml(left, right)`` occurrence."""

    left: EMLTerm
    right: EMLTerm


type EMLTerm = One | Variable | EML


def one() -> One:
    """Return a fresh primitive-one occurrence."""

    return One()


def variable(name: str) -> Variable:
    """Return a validated source-variable occurrence."""

    return Variable(name)


def eml(left: EMLTerm, right: EMLTerm) -> EML:
    """Return an ordered binary EML occurrence."""

    return EML(left=left, right=right)
