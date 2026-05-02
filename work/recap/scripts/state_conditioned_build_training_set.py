#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from work.recap.script_apps import state_conditioned_build_training_set_app as _app_module
from work.recap.script_apps.state_conditioned_build_training_set_app import StateConditionedBuildTrainingSetScriptApp


def __getattr__(name: str):
    return getattr(_app_module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_app_module)))


def _script_app() -> StateConditionedBuildTrainingSetScriptApp:
    return StateConditionedBuildTrainingSetScriptApp()


if __name__ == "__main__":
    raise SystemExit(_script_app().run())
