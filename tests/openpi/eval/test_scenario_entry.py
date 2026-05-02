from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.eval import (  # noqa: E402
    DEFAULT_ROLLOUT_EVALUATION_SCENARIO,
    DEFAULT_STOCK_SMOKE_SCENARIO,
    OpenPIEvalApp,
)
import work.openpi.eval.app as eval_app  # noqa: E402
import work.openpi.eval.cli as eval_cli  # noqa: E402
import work.openpi.scripts.openpi as openpi_entry  # noqa: E402


def test_eval_app_exposes_scenario_driven_surface() -> None:
    app = OpenPIEvalApp()
    assert hasattr(app, "run_rollout_evaluation")
    assert hasattr(app, "run_tracked_gate_evaluation")
    assert hasattr(app, "run_stock_smoke")
    assert not hasattr(eval_app, "build_parser")
    assert not hasattr(eval_app, "main")


def test_default_scenarios_are_visible_and_typed() -> None:
    assert (
        DEFAULT_ROLLOUT_EVALUATION_SCENARIO.checkpoint.variant == "stock_libero_ref_v1"
    )
    assert (
        DEFAULT_ROLLOUT_EVALUATION_SCENARIO.metrics.metric_profile == "budget_ladder_v1"
    )
    assert DEFAULT_STOCK_SMOKE_SCENARIO.task_suite_name == "libero_spatial"
    assert DEFAULT_STOCK_SMOKE_SCENARIO.seed == 7


def test_eval_cli_remains_importable_as_compatibility_surface() -> None:
    assert eval_cli.OpenPIEvalApp is OpenPIEvalApp
    assert (
        eval_cli.DEFAULT_ROLLOUT_EVALUATION_SCENARIO
        is DEFAULT_ROLLOUT_EVALUATION_SCENARIO
    )


def test_openpi_entry_uses_explicit_active_workflow_constants() -> None:
    assert openpi_entry.ACTIVE_WORKFLOW == "rollout_evaluation"
    assert openpi_entry.ACTIVE_SCENARIO is DEFAULT_ROLLOUT_EVALUATION_SCENARIO
    assert not hasattr(openpi_entry, "build_parser")
