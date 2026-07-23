"""Smoke tests for the repository package scaffold."""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "geml",
        "geml.analysis",
        "geml.ast",
        "geml.compression",
        "geml.contracts",
        "geml.dag",
        "geml.data",
        "geml.egraph",
        "geml.eml",
        "geml.experiments",
        "geml.export",
        "geml.graph",
        "geml.interfaces",
        "geml.learning",
        "geml.parsing",
        "geml.plots",
        "geml.spec",
        "geml.verification",
    ],
)
def test_package_imports(module_name: str) -> None:
    """Every package required by the shared scaffold imports successfully."""
    import_module(module_name)
