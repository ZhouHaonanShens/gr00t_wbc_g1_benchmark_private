from __future__ import annotations

from dataclasses import MISSING, fields, replace
from typing import Any

from work.recap.r1_repro.protocol import EvalProtocol


class ProtocolSwapError(ValueError):
    """Base class for invalid protocol swap requests."""


class UnknownAxis(ProtocolSwapError):
    """Raised when a requested swap field is not part of EvalProtocol."""


class UnresolvedProtocolField(ProtocolSwapError):
    """Raised when a required protocol field would be set to None."""


_FIELD_BY_NAME = {field.name: field for field in fields(EvalProtocol)}
_REQUIRED_FIELDS = {
    field.name
    for field in fields(EvalProtocol)
    if field.default is MISSING and field.default_factory is MISSING
}


def swap_single_axis(
    base: EvalProtocol, axis: str, value: Any, *, name: str
) -> EvalProtocol:
    if axis not in _FIELD_BY_NAME:
        raise UnknownAxis(f"unknown EvalProtocol axis {axis!r} for {name}")
    if value is None and axis in _REQUIRED_FIELDS:
        raise UnresolvedProtocolField(
            f"required EvalProtocol field {axis!r} cannot be None for {name}"
        )
    return replace(base, **{axis: value})


def enumerate_axes(
    base: EvalProtocol, target: EvalProtocol
) -> list[tuple[str, Any, Any]]:
    diffs: list[tuple[str, Any, Any]] = []
    for field in fields(EvalProtocol):
        base_value = getattr(base, field.name)
        target_value = getattr(target, field.name)
        if base_value != target_value:
            diffs.append((field.name, base_value, target_value))
    return diffs


__all__ = [
    "ProtocolSwapError",
    "UnknownAxis",
    "UnresolvedProtocolField",
    "enumerate_axes",
    "swap_single_axis",
]
