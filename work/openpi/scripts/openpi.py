#!/usr/bin/env python3
"""Canonical OpenPI entrypoint.

Edit ``ACTIVE_WORKFLOW`` and ``ACTIVE_SCENARIO`` below when you want to follow a
different default path through the mainline code. This file is intentionally
small so readers can jump from Scenario -> App -> Workflow without first
reverse-engineering a CLI router.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.eval import (  # noqa: E402
    DEFAULT_RECAP_BEST_ROLLOUT_SCENARIO,
    DEFAULT_ROLLOUT_EVALUATION_SCENARIO,
    DEFAULT_STOCK_SMOKE_SCENARIO,
    DEFAULT_STOCK_ROLLOUT_SCENARIO,
    DEFAULT_TRACKED_GATE_EVALUATION_SCENARIO,
    OpenPIEvalApp,
)
from work.openpi.pipelines.recap.collect import run_collection  # noqa: E402
from work.openpi.pipelines.recap.iteration import run_iteration  # noqa: E402
from work.openpi.pipelines.recap.loop import run_loop  # noqa: E402
from work.openpi.recap.scenarios import (  # noqa: E402
    DEFAULT_RECAP_COLLECTION_SCENARIO,
    DEFAULT_RECAP_ITERATION_SCENARIO,
)


# Recommended scenario constants:
# - DEFAULT_STOCK_SMOKE_SCENARIO
# - DEFAULT_STOCK_ROLLOUT_SCENARIO
# - DEFAULT_RECAP_BEST_ROLLOUT_SCENARIO
# - DEFAULT_RECAP_COLLECTION_SCENARIO
# - DEFAULT_RECAP_ITERATION_SCENARIO
ACTIVE_WORKFLOW = "rollout_evaluation"
ACTIVE_SCENARIO = DEFAULT_STOCK_ROLLOUT_SCENARIO


def run_default_tracked_gate_evaluation() -> int:
    return OpenPIEvalApp().run_tracked_gate_evaluation(
        DEFAULT_TRACKED_GATE_EVALUATION_SCENARIO
    )


def run_default_rollout_evaluation() -> int:
    return OpenPIEvalApp().run_rollout_evaluation(DEFAULT_STOCK_ROLLOUT_SCENARIO)


def run_default_recap_best_rollout_evaluation() -> int:
    return OpenPIEvalApp().run_rollout_evaluation(DEFAULT_RECAP_BEST_ROLLOUT_SCENARIO)


def run_default_stock_smoke() -> int:
    return OpenPIEvalApp().run_stock_smoke(DEFAULT_STOCK_SMOKE_SCENARIO)


def run_default_recap_collection() -> int:
    return run_collection(DEFAULT_RECAP_COLLECTION_SCENARIO)


def run_default_recap_iteration() -> int:
    return run_iteration(DEFAULT_RECAP_ITERATION_SCENARIO)


def run_active_workflow() -> int:
    if ACTIVE_WORKFLOW == "tracked_gate_evaluation":
        return OpenPIEvalApp().run_tracked_gate_evaluation(ACTIVE_SCENARIO)
    if ACTIVE_WORKFLOW == "rollout_evaluation":
        return OpenPIEvalApp().run_rollout_evaluation(ACTIVE_SCENARIO)
    if ACTIVE_WORKFLOW == "stock_smoke":
        return OpenPIEvalApp().run_stock_smoke(ACTIVE_SCENARIO)
    if ACTIVE_WORKFLOW == "recap_collection":
        return run_collection(ACTIVE_SCENARIO)
    if ACTIVE_WORKFLOW == "recap_iteration":
        return run_iteration(ACTIVE_SCENARIO)
    if ACTIVE_WORKFLOW == "recap_loop":
        return run_loop(ACTIVE_SCENARIO)
    raise ValueError(f"unsupported ACTIVE_WORKFLOW {ACTIVE_WORKFLOW!r}")


def main() -> int:
    return run_active_workflow()


if __name__ == "__main__":
    raise SystemExit(main())
