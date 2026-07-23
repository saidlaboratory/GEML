"""Deterministic official-style emission for validated pure EML trees."""

from __future__ import annotations

from geml.eml.ir import EML, EMLTerm, One, Variable
from geml.eml.validate import validate_pure_eml


def emit_eml(root: EMLTerm) -> str:
    """Emit exactly ``EML[left,right]`` with bare variable and ``1`` leaves."""

    validate_pure_eml(root)
    pieces: list[str] = []
    events: list[EMLTerm | str] = [root]
    while events:
        event = events.pop()
        if isinstance(event, str):
            pieces.append(event)
        elif isinstance(event, One):
            pieces.append("1")
        elif isinstance(event, Variable):
            pieces.append(event.name)
        elif isinstance(event, EML):
            events.extend(("]", event.right, ",", event.left, "EML["))
        else:  # pragma: no cover - validation rejects this before emission
            raise TypeError(f"unsupported EML event: {type(event).__name__}")
    return "".join(pieces)
