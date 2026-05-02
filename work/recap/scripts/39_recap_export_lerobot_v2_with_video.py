#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from work.recap.lerobot_export import workflow as _core_module

_app_module = _core_module
importlib = _core_module.importlib

_PATCH_SYNC_NAMES = (
    "_clear_alarm_timeout",
    "_install_alarm_timeout",
    "_maybe_reexec_into_wbc_venv",
    "_repo_root",
    "_tee_stdio",
)


def __getattr__(name: str):
    return getattr(_app_module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_app_module)))


def _sync_patched_helpers() -> None:
    for name in _PATCH_SYNC_NAMES:
        if name in globals():
            setattr(_app_module, name, globals()[name])


class RecapExportLeRobotWithVideoScriptApp:
    def run(self) -> int:
        _sync_patched_helpers()
        return _app_module.main()


def _script_app() -> RecapExportLeRobotWithVideoScriptApp:
    return RecapExportLeRobotWithVideoScriptApp()


def _build_parser():
    return _app_module._build_parser()


def main() -> int:
    _sync_patched_helpers()
    return _app_module.main()


if __name__ == "__main__":
    raise SystemExit(main())
