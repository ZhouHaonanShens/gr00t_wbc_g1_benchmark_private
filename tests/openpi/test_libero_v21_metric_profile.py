from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "agent/exchange/openpi_libero_v21_metric_profile.md"


def test_v21_metric_profile_defines_required_metrics() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "openpi LIBERO v21 metric profile",
        "eval-only phase",
        "A/B/C/X only",
        "D_not_executed_in_v21=true",
        "collision table",
        "first_success_step",
        "success_rate@0.50_budget",
        "success_rate@0.75_budget",
        "success_rate@1.00_budget",
        "median_first_success_step_fraction",
        "timeout_rate",
        "primary_metric_id=success_rate@0.50_budget",
        "throughput_like_score = 1000 * successful_episodes / sum(executed_steps)",
    ]
    for item in required:
        assert item in text, f"missing v21 metric item: {item}"


def test_v21_metric_profile_freezes_metric_definitions_and_ordering() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "1-indexed earliest success step",
        "first_success_step = null",
        "`success_rate@f_budget = count(first_success_step != null && first_success_step <= floor(f * max_steps_resolved)) / total_episodes`",
        "`success_rate@0.50_budget = count(first_success_step != null && first_success_step <= floor(0.50 * max_steps_resolved)) / total_episodes`",
        "`success_rate@0.75_budget = count(first_success_step != null && first_success_step <= floor(0.75 * max_steps_resolved)) / total_episodes`",
        "`success_rate@1.00_budget = count(first_success_step != null && first_success_step <= floor(1.00 * max_steps_resolved)) / total_episodes`",
        "`median_first_success_step_fraction = median(first_success_step / max_steps_resolved)` over successful episodes only。",
        "`timeout_rate = count(timeout_flag=true) / total_episodes`",
        "primary_metric_id=success_rate@0.50_budget",
        "primary metric order = success_rate@0.50_budget -> success_rate@0.75_budget -> throughput_like_score",
        "success_rate@1.00_budget is compatibility-only",
    ]
    for item in required:
        assert item in text, f"missing v21 metric definition item: {item}"


def test_v21_metric_profile_rejects_old_step_budget_wording() -> None:
    text = DOC.read_text(encoding="utf-8")
    forbidden = [
        "step_budget",
        "`timeout_rate` 定义为 `first_success_step = null` 的 episode 数量除以 `total_episodes`。",
    ]
    for item in forbidden:
        assert item not in text, f"unexpected old v21 metric wording found: {item}"


def test_v21_metric_profile_keeps_v21_naming_consistent() -> None:
    text = DOC.read_text(encoding="utf-8")
    forbidden = ["v2.1", "v2_1", "v2p1"]
    for item in forbidden:
        assert item not in text, f"unexpected alternate v21 naming found: {item}"
