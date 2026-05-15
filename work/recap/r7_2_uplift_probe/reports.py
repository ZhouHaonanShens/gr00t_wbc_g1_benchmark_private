from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from work.recap.r7_2_uplift_probe.contract import StepwiseCounterfactual, TrialReport, planned_steps

RowStatus = Literal["RAN", "NOT_RUN_EARLY_STOP", "FAILED"]


@dataclass(frozen=True)
class ReportRow:
    step: int
    status: RowStatus
    condition_sha_equal: bool | None
    first_5_actions_l2_diff_max: float | None
    verdict: str
    adapter_path: str
    counterfactual_path: str
    early_stop_reason: str | None = None


def build_evolution_rows(report: TrialReport, failure_reason: str | None = None) -> tuple[ReportRow, ...]:
    by_step = {item.step: item for item in report.counterfactual_evolution}
    rows: list[ReportRow] = []
    reason = _stop_reason(report, failure_reason)
    for step in planned_steps(report.request.max_steps, report.request.probe_interval):
        item = by_step.get(step)
        rows.append(_ran_row(item) if item else _not_run_row(step, reason if step > report.final_step else "not_reached"))
    return tuple(rows)


def render_phase_report(report: TrialReport, *, r7_0_recipe_sha256: str, r7_1_dryrun_sha256: str, r6_1_probe_sha256: str, failure_reason: str | None = None) -> str:
    parts = [
        "# FIX_R2_A1_LOAD_08_R7_2_UPLIFT_PROBE_REPORT",
        "", "## Provenance",
        f"- R7.0 recipe JSON sha256: `{r7_0_recipe_sha256}`",
        f"- R7.1 dryrun_report sha256: `{r7_1_dryrun_sha256}`",
        f"- R6.1 probe evidence sha256: `{r6_1_probe_sha256}`",
        "", "## Counterfactual evolution table",
        _render_table(build_evolution_rows(report, failure_reason)),
        "", "## Final verdict", render_final_verdict(report, failure_reason=failure_reason), "",
    ]
    return "\n".join(parts)


def render_final_verdict(report: TrialReport, failure_reason: str | None = None) -> str:
    preset = report.request.recipe_preset
    if report.final_verdict == "INDICATOR_SENSITIVE_AT_STEP_N":
        return f"verdict=INDICATOR_SENSITIVE_AT_STEP_N; recipe={preset}; final_step={report.final_step}; next=R7.3_full_retraining"
    if report.final_verdict == "INDICATOR_INVARIANT_AT_MAX_STEPS":
        return f"verdict=INDICATOR_INVARIANT_AT_MAX_STEPS; recipe={preset}; final_step=2000; next=R7.2_next_trial_or_escalate_to_C3_C4"
    if report.final_verdict == "BUDGET_EXCEEDED":
        return f"verdict=BUDGET_EXCEEDED; gpu_minutes_used={round(float(report.total_gpu_seconds) / 60.0, 3)}; next=R7.2_reduce_batch_or_step"
    return f"verdict=TRAINING_FAILED; reason={failure_reason or 'crash'}; next=R7.2_debug"


def _ran_row(item: StepwiseCounterfactual) -> ReportRow:
    row = ReportRow(
        item.step, "RAN", item.condition_sha_equal, item.first_5_actions_l2_diff_max,
        item.counterfactual_verdict, item.adapter_path, item.counterfactual_path,
    )
    return row


def _not_run_row(step: int, reason: str) -> ReportRow:
    row = ReportRow(step, "NOT_RUN_EARLY_STOP", None, None, "N/A", "N/A", "N/A", reason)
    if reason:
        return row
    return row


def _stop_reason(report: TrialReport, failure_reason: str | None) -> str:
    reasons = {"INDICATOR_SENSITIVE_AT_STEP_N": f"INDICATOR_SENSITIVE@{report.final_step}", "BUDGET_EXCEEDED": "budget_exceeded", "TRAINING_FAILED": failure_reason or "crash"}
    reason = reasons.get(report.final_verdict)
    if reason is None:
        reason = "max_steps_reached"
    return reason


def _render_table(rows: tuple[ReportRow, ...]) -> str:
    lines = ["| step | status | condition_sha_equal | l2_diff_max | verdict |", "|---:|---|---|---:|---|"]
    for row in rows:
        equal = "N/A" if row.condition_sha_equal is None else str(row.condition_sha_equal)
        diff = "N/A" if row.first_5_actions_l2_diff_max is None else str(row.first_5_actions_l2_diff_max)
        lines.append(f"| {row.step} | {row.status} | {equal} | {diff} | {row.verdict} |")
    return "\n".join(lines)
