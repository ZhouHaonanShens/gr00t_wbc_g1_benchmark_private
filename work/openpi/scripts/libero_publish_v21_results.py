"""Compatibility facade for publishing OpenPI LIBERO v21 result docs.

The retained implementation is archived under ``agent/archive``.  This thin
facade preserves the historical live path expected by active tests while keeping
new business logic out of the public surface.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_MODULE_NAME = "openpi_libero_publish_v21_results_legacy_impl"
_LEGACY_MODULE_PATH = _REPO_ROOT / "agent" / "archive" / "openpi" / "legacy_scripts" / "libero_publish_v21_results.py"


def _load_legacy_module() -> ModuleType:
    cached = sys.modules.get(_LEGACY_MODULE_NAME)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_LEGACY_MODULE_NAME, _LEGACY_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load legacy v21 publisher from {_LEGACY_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_LEGACY_MODULE_NAME] = module
    spec.loader.exec_module(module)
    # The archived file was originally located under work/openpi/scripts and
    # computes REPO_ROOT from __file__.  Patch it after loading so repo-relative
    # rendering remains correct from the archive location.
    module.REPO_ROOT = _REPO_ROOT
    module.DEFAULT_PAIRED_SUMMARY_PATH = _REPO_ROOT / "agent/artifacts/openpi_libero_v21/paired_summary_abcx_v21.json"
    module.DEFAULT_GO_NO_GO_PATH = _REPO_ROOT / "agent/artifacts/openpi_libero_v21/go_no_go_v21.json"
    module.DEFAULT_RESULTS_DOC_PATH = _REPO_ROOT / "agent/exchange/openpi_libero_v21_results.md"
    module.DEFAULT_ENTRY_DOC_PATH = _REPO_ROOT / "agent/exchange/openpi_libero_v22_entry_prereqs.md"

    def _repo_relative_path(path_like: str | None) -> str | None:
        if path_like is None:
            return None
        candidate = Path(path_like)
        if not candidate.is_absolute():
            return path_like
        try:
            return str(candidate.relative_to(_REPO_ROOT))
        except ValueError:
            text = str(candidate)
            marker = "/agent/"
            if marker in text:
                return "agent/" + text.split(marker, 1)[1]
            return path_like

    module._repo_relative_path = _repo_relative_path
    return module


_legacy = _load_legacy_module()

publish_v21_results = _legacy.publish_v21_results
build_parser = _legacy.build_parser
main = _legacy.main

__all__ = ["publish_v21_results", "build_parser", "main"]
