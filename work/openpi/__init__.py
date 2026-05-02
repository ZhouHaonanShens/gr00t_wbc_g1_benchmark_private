from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _optional_import(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def _alias_module(alias: str, target: str) -> None:
    module = _optional_import(target)
    if module is not None:
        sys.modules.setdefault(alias, module)


_alias_module(__name__ + ".checkpoint_loader", __name__ + ".checkpoint")
_alias_module(__name__ + ".rollout_runtime", __name__ + ".runtime.api")
_alias_module(__name__ + ".libero_recap", __name__ + ".recap")
_alias_module(__name__ + ".libero_state_tokens", __name__ + ".state_tokens")
_alias_module(__name__ + ".upstream_overlay.openpi_recap", __name__ + ".overlays.openpi_recap")

checkpoint_loader = _optional_import(__name__ + ".checkpoint")
rollout_runtime = _optional_import(__name__ + ".runtime.api")


__all__ = [
    "checkpoint",
    "dataloader",
    "eval",
    "metrics",
    "model",
    "runtime",
]
