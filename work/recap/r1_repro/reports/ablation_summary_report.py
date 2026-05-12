from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..gates import descriptive_axis_sum_stats
from ..gates import newcombe_ci_on_delta
from ..gates import wilson_ci_on_rate


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _axis_row(axis_name: str, baseline: Any, result: Any) -> dict[str, Any]:
    baseline_count = int(baseline.success_count)
    swap_count = int(result.success_count)
    n_baseline = max(1, len(baseline.per_episode))
    n_swap = max(1, len(result.per_episode))
    delta_ci = newcombe_ci_on_delta(swap_count, baseline_count, n_swap, n_baseline)
    observed_delta = (swap_count / n_swap) - (baseline_count / n_baseline)
    lower_abs_bound = delta_ci[0] if delta_ci[0] > 0.0 else abs(delta_ci[1]) if delta_ci[1] < 0.0 else 0.0
    return {
        "axis_name": axis_name,
        "swap_count": swap_count,
        "swap_rate": float(swap_count / n_swap),
        "wilson_ci_on_rate": wilson_ci_on_rate(swap_count, n_swap),
        "observed_delta": float(observed_delta),
        "newcombe_ci_on_delta": delta_ci,
        "delta_ci_excludes_zero": bool(delta_ci[0] > 0.0 or delta_ci[1] < 0.0),
        "lower_ci_bound_on_abs_delta": float(lower_abs_bound),
    }


def render_ablation_summary(
    baseline: Any,
    ablations: dict[str, Any],
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_axis_row(axis, baseline, result) for axis, result in ablations.items()]
    rows.sort(key=lambda row: row["lower_ci_bound_on_abs_delta"], reverse=True)
    baseline_count = int(baseline.success_count)
    deltas = [int(result.success_count) - baseline_count for result in ablations.values()]
    baseline_successes = [baseline_count for _ in deltas]
    descriptive = descriptive_axis_sum_stats(
        deltas,
        baseline_successes,
        max(1, len(baseline.per_episode)),
        max(1, len(next(iter(ablations.values())).per_episode)) if ablations else 30,
    )
    payload = {
        "baseline_success_count": baseline_count,
        "baseline_wilson_ci_on_rate": baseline.wilson_ci_on_rate,
        "axis_rows": rows,
        "axes_with_significant_delta": [
            row["axis_name"] for row in rows if row["delta_ci_excludes_zero"]
        ],
        "descriptive_axis_sum_stats": descriptive,
        "explicit_no_dominant_axis_claim": True,
        "explicit_no_axis_set_incompleteness_boolean": True,
        "r1_does_not_decide_r2_entry": True,
    }
    _write_json(out_dir / "ablation_summary_report.json", payload)

    lines = [
        "# R1 Ablation Summary Report",
        "",
        "R1 does NOT decide R2 entry.",
        "",
        "R1 does NOT emit a boolean axis-set-incompleteness verdict.",
        "",
        "R1 does NOT make a dominant-axis causal-attribution claim.",
        "",
        f"- baseline_success_count: {baseline_count}",
        f"- axes_with_significant_delta: {payload['axes_with_significant_delta']}",
        f"- sum_abs_delta_observed: {descriptive['sum_abs_delta_observed']:.6f}",
        f"- sum_abs_delta_lower_95: {descriptive['sum_abs_delta_lower_95']:.6f}",
        "",
        "| axis | swap_count | swap_rate | observed_delta | newcombe_ci_on_delta | excludes_zero |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {axis_name} | {swap_count} | {swap_rate:.6f} | {observed_delta:.6f} | {newcombe_ci_on_delta} | {delta_ci_excludes_zero} |".format(
                **row
            )
        )
    (out_dir / "ablation_summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
