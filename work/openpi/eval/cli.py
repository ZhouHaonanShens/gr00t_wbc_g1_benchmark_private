from __future__ import annotations

"""Compatibility facade for the scenario-driven eval surface.

`work.openpi.eval.app` intentionally no longer exposes parser-building helpers.
Keep this module importable so callers that still resolve `work.openpi.eval.cli`
receive the new app/scenario surface instead of failing during import.
"""

from .app import OpenPIEvalApp
from .scenarios import (
    DEFAULT_RECAP_BEST_ROLLOUT_SCENARIO,
    DEFAULT_ROLLOUT_EVALUATION_SCENARIO,
    DEFAULT_STOCK_SMOKE_SCENARIO,
    DEFAULT_STOCK_ROLLOUT_SCENARIO,
    DEFAULT_TRACKED_GATE_EVALUATION_SCENARIO,
)

__all__ = [
    "DEFAULT_RECAP_BEST_ROLLOUT_SCENARIO",
    "DEFAULT_ROLLOUT_EVALUATION_SCENARIO",
    "DEFAULT_STOCK_SMOKE_SCENARIO",
    "DEFAULT_STOCK_ROLLOUT_SCENARIO",
    "DEFAULT_TRACKED_GATE_EVALUATION_SCENARIO",
    "OpenPIEvalApp",
]
