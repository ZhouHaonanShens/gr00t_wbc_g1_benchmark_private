#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from work.recap.script_apps import recap_online_loop_iterate_app as _app_module
from work.recap.script_apps.recap_online_loop_iterate_app import (
    RecapOnlineLoopIterateScriptApp,
)

_PATCH_SYNC_NAMES = (
    "_git_head_and_dirty",
    "_repo_root",
)


def __getattr__(name: str):
    return getattr(_app_module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_app_module)))


def _sync_patched_helpers() -> None:
    for name in _PATCH_SYNC_NAMES:
        if name in globals():
            setattr(_app_module, name, globals()[name])


def _script_app() -> RecapOnlineLoopIterateScriptApp:
    return RecapOnlineLoopIterateScriptApp()


def main() -> int:
    _sync_patched_helpers()
    return _app_module.main()


if __name__ == "__main__":
    raise SystemExit(main())
