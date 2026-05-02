from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from work.recap.state_conditioned import snapshot_harvest as _core_module

StateConditionedSnapshotHarvestScriptApp = (
    _core_module.StateConditionedSnapshotHarvestScriptApp
)

_core_module.StateConditionedSnapshotHarvestScriptApp = (
    StateConditionedSnapshotHarvestScriptApp
)
sys.modules[__name__] = _core_module
