"""Markdown renderer for the R2.1 multi-cell summary (pure function)."""
from __future__ import annotations

from typing import Any

from work.recap.r2_authentic_eval.delta_stats import (
    R2_BASELINE_N_DEFAULT,
    R2_BASELINE_SUCC_DEFAULT,
    R2_TRIGGER_THRESHOLD,
)
from work.recap.r2_authentic_eval.eval_runner import R2CellResult


def _row(cell: R2CellResult) -> str:
    ck = cell.request.checkpoint
    lo, hi = cell.wilson_ci_95
    triggered = "yes" if cell.rate < R2_TRIGGER_THRESHOLD else "no"
    return (
        f"| {ck.label} | `{ck.abs_path}` | {ck.n_train_steps}"
        f" | {cell.success_count}/{cell.completed_episode_total}"
        f" | {cell.rate:.3f} | ({lo:.3f}, {hi:.3f})"
        f" | {cell.delta_vs_baseline:+.3f} | {triggered} |"
    )


def build_summary_table(
    cells: list[R2CellResult],
    *,
    statistical_regime: dict[str, Any],
    skip_decision_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the summary_table.json payload (pure)."""
    return {
        "r2_summary_table_schema_version": "1.0.0",
        "n_valid_cells": int(statistical_regime.get("n_valid_cells", len(cells))),
        "baseline_succ": int(statistical_regime.get("baseline_succ", R2_BASELINE_SUCC_DEFAULT)),
        "baseline_total": int(statistical_regime.get("baseline_total", R2_BASELINE_N_DEFAULT)),
        "per_cell_p_below_trigger": float(statistical_regime.get("per_cell_p_below_trigger", 0.0)),
        "family_wise_at_baseline": float(statistical_regime.get("family_wise_at_baseline", 0.0)),
        "newcombe_half_width_at_baseline": float(
            statistical_regime.get("newcombe_half_width_at_baseline", 0.0)
        ),
        "trigger_threshold": float(R2_TRIGGER_THRESHOLD),
        "cells": [
            {
                "abs_path": str(c.request.checkpoint.abs_path),
                "n_train_steps": c.request.checkpoint.n_train_steps,
                "success_count": c.success_count,
                "completed_episode_total": c.completed_episode_total,
                "rate": c.rate,
                "wilson_ci_95": list(c.wilson_ci_95),
                "delta_vs_baseline": c.delta_vs_baseline,
                "newcombe_delta_ci_95": list(c.newcombe_delta_ci_95),
                "triggered_below_threshold": c.rate < R2_TRIGGER_THRESHOLD,
                "r2_invocation_envelope_sha256": c.r2_invocation_envelope_sha256,
            }
            for c in cells
        ],
        "skip_decision_records": list(skip_decision_records or []),
    }


def render(
    cells: list[R2CellResult],
    *,
    statistical_regime: dict[str, Any],
    skip_decision_records: list[dict[str, Any]] | None = None,
) -> str:
    """Render the summary report as markdown (pure)."""
    n_cells = int(statistical_regime.get("n_valid_cells", len(cells)))
    fwer = float(statistical_regime.get("family_wise_at_baseline", 0.0))
    header = (
        "| label | abs_path | n_train_steps | succ/total | rate"
        " | wilson_ci_95 | delta | triggered |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    rows = [header, sep, *(_row(c) for c in cells)]
    out: list[str] = ["# R2.1 Summary", "", "## Cell rate table", *rows, ""]
    out.append("## Decomposition decision")
    triggered_any = any(c.rate < R2_TRIGGER_THRESHOLD for c in cells)
    out.append(f"- requires_decomposition: {triggered_any}")
    out.append(
        f"- family_wise_error_rate_at_baseline (n_cells={n_cells}): {fwer:.3f}"
    )
    out.append("")
    out.append("## Skip-mismatch reasons")
    rerun_records = [r for r in (skip_decision_records or []) if r.get("decided") is False]
    if not rerun_records:
        out.append("- No skip-mismatches in this run.")
    else:
        for rec in rerun_records:
            out.append(f"- `{rec.get('ckpt_slug', '?')}`: {rec.get('reason', '?')}")
    out.append("")
    return "\n".join(out) + "\n"


__all__ = ["render", "build_summary_table"]
