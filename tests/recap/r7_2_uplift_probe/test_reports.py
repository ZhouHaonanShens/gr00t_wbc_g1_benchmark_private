from __future__ import annotations

from work.recap.r7_2_uplift_probe.contract import StepwiseCounterfactual, TrialReport, TrialRequest, preset_to_recipe_flags
from work.recap.r7_2_uplift_probe.reports import build_evolution_rows, render_final_verdict, render_phase_report

TOKEN = "a" * 64


def _request() -> TrialRequest:
    return TrialRequest("trial-1", "/abs/ckpt", preset_to_recipe_flags("full_C1_C2_C5"), "full_C1_C2_C5", "out", TOKEN, 1)


def test_early_stop_report_has_ten_rows_without_fabricated_metrics() -> None:
    item = StepwiseCounterfactual(200, "adapter", "cf", "INDICATOR_SENSITIVE", False, 0.1)
    report = TrialReport(_request(), "INDICATOR_SENSITIVE_AT_STEP_N", 200, (item,), 1.0, 0.1)
    rows = build_evolution_rows(report)
    assert len(rows) == 10
    assert rows[0].status == "RAN"
    assert rows[1].status == "NOT_RUN_EARLY_STOP"
    assert rows[1].first_5_actions_l2_diff_max is None


def test_max_step_report_has_ten_rows() -> None:
    report = TrialReport(_request(), "INDICATOR_INVARIANT_AT_MAX_STEPS", 2000, (), 1.0, 0.1)
    rows = build_evolution_rows(report)
    assert len(rows) == 10
    assert rows[-1].step == 2000


def test_final_verdict_templates_are_exact() -> None:
    request = _request()
    assert render_final_verdict(TrialReport(request, "INDICATOR_SENSITIVE_AT_STEP_N", 200, (), 1, None)) == "verdict=INDICATOR_SENSITIVE_AT_STEP_N; recipe=full_C1_C2_C5; final_step=200; next=R7.3_full_retraining"
    assert render_final_verdict(TrialReport(request, "INDICATOR_INVARIANT_AT_MAX_STEPS", 2000, (), 1, None)).endswith("next=R7.2_next_trial_or_escalate_to_C3_C4")
    assert render_final_verdict(TrialReport(request, "TRAINING_FAILED", 0, (), 1, None), failure_reason="crash") == "verdict=TRAINING_FAILED; reason=crash; next=R7.2_debug"
    assert render_final_verdict(TrialReport(request, "BUDGET_EXCEEDED", 0, (), 120, None)) == "verdict=BUDGET_EXCEEDED; gpu_minutes_used=2.0; next=R7.2_reduce_batch_or_step"


def test_phase_report_cites_required_hashes() -> None:
    report = TrialReport(_request(), "TRAINING_FAILED", 0, (), 1, None)
    text = render_phase_report(report, r7_0_recipe_sha256="r70", r7_1_dryrun_sha256="r71", r6_1_probe_sha256="r61")
    assert "`r70`" in text and "`r71`" in text and "`r61`" in text
    assert "## Counterfactual evolution table" in text
