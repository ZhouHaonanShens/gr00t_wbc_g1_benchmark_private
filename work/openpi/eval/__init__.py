from __future__ import annotations

import importlib
import sys


def _alias_module(alias: str, target: str) -> None:
    sys.modules.setdefault(alias, importlib.import_module(target))


_alias_module(__name__ + ".protocol_v21", __name__ + ".protocols.manifest")
_alias_module(__name__ + ".protocol_v2", __name__ + ".protocols.tracked_gate")
_alias_module(__name__ + ".protocol", __name__ + ".protocols.environment")
_alias_module(__name__ + ".libero_go_no_go_v21", __name__ + ".reports.go_no_go")
_alias_module(__name__ + ".libero_go_no_go_v2", __name__ + ".reports.compatibility")
_alias_module(__name__ + ".libero_rollout_eval_v21_impl", __name__ + ".workflows.rollout_support")
_alias_module(__name__ + ".libero_rollout_eval_v2_impl", __name__ + ".workflows.tracked_gate")

from .app import OpenPIEvalApp
from .scenarios import (
    DEFAULT_RECAP_BEST_ROLLOUT_SCENARIO,
    DEFAULT_ROLLOUT_EVALUATION_SCENARIO,
    DEFAULT_STOCK_SMOKE_SCENARIO,
    DEFAULT_STOCK_ROLLOUT_SCENARIO,
    DEFAULT_TRACKED_GATE_EVALUATION_SCENARIO,
    RolloutEvaluationScenario,
    StockSmokeScenario,
    TrackedGateEvaluationScenario,
)
from .workflows.rollout import RolloutEvaluationWorkflow, TrackedGateEvaluationWorkflow
from .workflows.stock_smoke import StockSmokeWorkflow

__all__ = [
    "DEFAULT_RECAP_BEST_ROLLOUT_SCENARIO",
    "DEFAULT_ROLLOUT_EVALUATION_SCENARIO",
    "DEFAULT_STOCK_SMOKE_SCENARIO",
    "DEFAULT_STOCK_ROLLOUT_SCENARIO",
    "DEFAULT_TRACKED_GATE_EVALUATION_SCENARIO",
    "OpenPIEvalApp",
    "RolloutEvaluationScenario",
    "RolloutEvaluationWorkflow",
    "StockSmokeScenario",
    "StockSmokeWorkflow",
    "TrackedGateEvaluationScenario",
    "TrackedGateEvaluationWorkflow",
]
