#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from work.recap.script_apps import state_conditioned_snapshot_harvest_app as _app_module
from work.recap.script_apps.state_conditioned_snapshot_harvest_app import (
    StateConditionedSnapshotHarvestScriptApp,
)


if __name__ == "__main__":
    raise SystemExit(StateConditionedSnapshotHarvestScriptApp().run())

sys.modules[__name__] = _app_module
