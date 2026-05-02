"""Compatibility facade for OpenPI LIBERO v21 go/no-go metrics.

The canonical implementation lives in ``work/openpi/eval/reports/go_no_go.py``.
This module preserves the historical file path used by tests and archived
scripts while avoiding ``work.openpi.eval`` package side effects.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

_REPORT_MODULE_NAME = "openpi_libero_go_no_go_v21_report_impl"
_REPORT_MODULE_PATH = Path(__file__).resolve().parent / "reports" / "go_no_go.py"


def _load_report_module() -> ModuleType:
    cached = sys.modules.get(_REPORT_MODULE_NAME)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_REPORT_MODULE_NAME, _REPORT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load OpenPI v21 go/no-go report from {_REPORT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_REPORT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_report = _load_report_module()

for _name in getattr(_report, "__all__", ()):  # re-export the canonical public API
    globals()[_name] = getattr(_report, _name)

__all__ = list(getattr(_report, "__all__", ()))
