from __future__ import annotations

import importlib
from types import ModuleType


_EXPORT_MODULES: tuple[str, ...] = (
    "checkpoint",
    "checkpoint_provenance",
    "control_gate",
    "critic_bridge",
    "dataset",
    "dataset_aggregation",
    "protocol",
    "runtime_prompt",
    "summary",
    "scenarios",
    "train_config",
)

OPTIONAL_IMPORT_ERRORS: dict[str, str] = {}


def _public_names(module: ModuleType) -> tuple[str, ...]:
    raw_all = getattr(module, "__all__", None)
    if isinstance(raw_all, (list, tuple)):
        return tuple(str(name) for name in raw_all)
    return tuple(name for name in vars(module) if not name.startswith("_"))


def _optional_star_import(module_name: str) -> None:
    qualified_name = f"{__name__}.{module_name}"
    try:
        module = importlib.import_module(qualified_name)
    except ModuleNotFoundError as exc:
        OPTIONAL_IMPORT_ERRORS[module_name] = str(exc)
        return
    for public_name in _public_names(module):
        globals()[public_name] = getattr(module, public_name)


for _module_name in _EXPORT_MODULES:
    _optional_star_import(_module_name)

__all__ = tuple(
    sorted(name for name in globals() if not name.startswith("_"))
)
