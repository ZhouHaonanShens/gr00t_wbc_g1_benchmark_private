from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from work.recap.state_conditioned import build_training_set as _core_module

StateConditionedBuildTrainingSetScriptApp = (
    _core_module.StateConditionedBuildTrainingSetScriptApp
)

_core_module.StateConditionedBuildTrainingSetScriptApp = (
    StateConditionedBuildTrainingSetScriptApp
)
sys.modules[__name__] = _core_module
