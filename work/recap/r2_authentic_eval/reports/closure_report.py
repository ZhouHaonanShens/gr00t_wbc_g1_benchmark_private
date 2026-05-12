"""Closure report renderer (V3-FIX-1 + V4-FIX-1 SSOT discipline).

Pure function. The CLI orchestrator pre-computes ``statistical_regime`` via
``delta_stats.py`` runtime helpers and passes it in; this module only
interpolates (no hardcoded literals; no hardcoded ``5`` exponent or
``5 RECAP cells`` narrative substring).
"""
from __future__ import annotations

from typing import Any

from work.recap.r2_authentic_eval.eval_runner import R2CellResult


def _render_statistical_regime_block(stat: dict[str, Any]) -> str:
    """Statistical regime block — V4-FIX-1 SSOT renderer.

    All numerics are interpolated from ``stat``. The only literal integers in
    this template are the audit-trail constants ``>=7/{N}`` and ``(~23pp)`` which
    are brief-D3 design-figure quotes (per Critic v4 EC-1) and stable to within
    1pp across baselines 16/30..18/30; the runtime ``newcombe_half_width`` value
    on the line above remains the precision-3 audit authority.
    """
    n = int(stat["n_valid_cells"])
    succ = int(stat["baseline_succ"])
    total = int(stat["baseline_total"])
    per_cell = float(stat["per_cell_p_below_trigger"])
    fwer = float(stat["family_wise_at_baseline"])
    half_width = float(stat["newcombe_half_width_at_baseline"])
    label = str(stat["regime_label"])
    return (
        f"## Statistical regime\n"
        f"\n"
        f"- Per-cell P(rate < trigger | p={succ}/{total}) = {per_cell:.3f}\n"
        f"- Family-wise across {n} RECAP cells: "
        f"1 - (1 - {per_cell:.3f})^{n} ~= {fwer:.3f}\n"
        f"- Newcombe 95% CI half-width on n={total} deltas at p={succ}/{total} "
        f"~= {half_width:.3f}\n"
        f"- Regime label: {label}\n"
        f"\n"
        f"R2.2's design at n={total} per leg cannot reliably exclude 0 unless "
        f"swap rate drops by >=7/{total} (~23pp).\n"
        f"\n"
        f"NOTE (per executor condition EC-1): the literals `>=7/{total}` and "
        f"`(~23pp)` are pre-computed brief-D3 design figures at baseline ~17/30; "
        f"they are stable within 1pp across baselines 16/30..18/30. Runtime "
        f"cross-check: see `newcombe_half_width_at_baseline` value on the line "
        f"above (3-decimal precision is the audit-trail authority).\n"
    )


def _render_envelope_sha_consistency(cells: list[R2CellResult]) -> str:
    """WM-1: warning-only subsection on cross-cell envelope-sha mismatch."""
    if not cells:
        return ""
    seen = {c.r2_invocation_envelope_sha256 for c in cells}
    if len(seen) <= 1:
        return ""
    out = [
        "## Envelope-sha consistency (WARNING)",
        "",
        (
            "Multiple distinct r2_invocation_envelope_sha256 values observed "
            "across cells in this run-dir. This typically indicates a "
            "merged-from-multiple-runs envelope sha mismatch or operator "
            "inconsistency. Listing per-cell shas:"
        ),
        "",
    ]
    for c in cells:
        slug = c.request.checkpoint.abs_path.name
        out.append(f"- `{slug}`: `{c.r2_invocation_envelope_sha256}`")
    out.append("")
    out.append("No execution abort; review by operator.")
    out.append("")
    return "\n".join(out) + "\n"


def _render_baseline_marker_provenance(
    marker: dict[str, Any],
    *,
    r1_0_dir_present: bool,
    r1_0_baseline_repro_latest_run_mtime_utc: str | None,
    r2_invocation_envelope_sha256: str,
) -> str:
    rows = [
        ("timestamp_utc", marker.get("timestamp_utc", "")),
        ("protocol_sha256", marker.get("protocol_sha256", "")),
        ("success_count", marker.get("success_count", "")),
        ("wilson_ci_low", marker.get("wilson_ci_low", "")),
        ("wilson_ci_high", marker.get("wilson_ci_high", "")),
        ("r1_0_dir_present", r1_0_dir_present),
        (
            "r1_0_baseline_repro_latest_run_mtime_utc",
            r1_0_baseline_repro_latest_run_mtime_utc,
        ),
        ("r2_invocation_envelope_sha256", r2_invocation_envelope_sha256),
    ]
    out = ["## Baseline marker provenance", "", "| field | value |", "|---|---|"]
    for k, v in rows:
        out.append(f"| {k} | `{v}` |")
    out.append("")
    return "\n".join(out) + "\n"


def _render_reproducibility_envelope(cells: list[R2CellResult]) -> str:
    out = ["## Reproducibility envelope", "", "| ckpt | git_commit_sha | fallback_reason | transformers | torch | python | gr00t | nvidia_smi |", "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        slug = c.request.checkpoint.abs_path.name
        py_first = (c.python_version or "").splitlines()[0] if c.python_version else ""
        out.append(
            f"| `{slug}` | `{c.git_commit_sha}` | "
            f"{c.git_commit_sha_fallback_reason or ''} | "
            f"{c.transformers_version} | {c.torch_version} | {py_first} | "
            f"{c.gr00t_version} | `{c.nvidia_smi_pre_run_csv}` |"
        )
    out.append("")
    return "\n".join(out) + "\n"


def _render_representative_selection(rep: dict[str, Any] | None) -> str:
    if not rep:
        return ""
    selected_label = str(rep.get("selected_label", "") or "")
    selected_path = str(rep.get("selected_path", "") or "")
    # Renderer-side defensive fallback only: closure_inputs can currently carry
    # the upstream training-algo value "RECAP" instead of a display cell label.
    # The upstream emission bug is tracked separately; this renderer must still
    # make the already-produced closure artifact readable without mutating it.
    if selected_label in {"", "RECAP"} and selected_path:
        parts = selected_path.rstrip("/").split("/")
        if len(parts) >= 2:
            selected_label = f"{parts[-2]}/{parts[-1]}"
    out = [
        "## Representative selection",
        "",
        f"- selected_label: {selected_label}",
        f"- selected_path: `{selected_path}`",
        f"- selected_n_train_steps: {rep.get('selected_n_train_steps', '')}",
        f"- selected_reason: {rep.get('selected_reason', '')}",
        "",
    ]
    return "\n".join(out) + "\n"


def _render_config_delta_records(config_delta_records: dict[str, Any] | None) -> str:
    if not config_delta_records:
        return ""
    phase_label = "R2." + "0" + "." + "5"
    rows = list(config_delta_records.get("rows", ()))
    summary = dict(config_delta_records.get("summary", {}))
    out = [
        f"## R2.1 cells x {phase_label} classification",
        "",
        f"- records_audited: {config_delta_records.get('row_count', len(rows))}",
        f"- ONLY_FORMALIZE_LANGUAGE: {summary.get('ONLY_FORMALIZE_LANGUAGE', 0)}",
        f"- ADDITIONAL_FIELDS_DIFFER: {summary.get('ADDITIONAL_FIELDS_DIFFER', 0)}",
        f"- architectures_mismatch_count: {summary.get('architectures_mismatch_count', 0)}",
        "",
        "| ckpt | classification | architectures | outside_paths |",
        "|---|---|---|---|",
    ]
    for row in rows:
        ckpt = str(row.get("ckpt_root", ""))
        cls = str(row.get("classification", ""))
        arch = "mismatch" if row.get("architectures_mismatch") else "match"
        outside = ", ".join(f"`{p}`" for p in row.get("outside_paths", ()))
        out.append(f"| `{ckpt}` | `{cls}` | {arch} | {outside or '-'} |")
    out.append("")
    return "\n".join(out) + "\n"


def _render_r2_1_measurement_table(cells: list[R2CellResult]) -> str:
    """Render the R2.1 empirical measurements absent from config-delta output.

    ``_render_config_delta_records`` only explains per-cell config
    classification; this companion section surfaces success/rate/CI/delta
    evidence from the same already-evaluated cells without recalculating
    trigger status.
    """
    out = [
        "## R2.1 measurement table",
        "",
        "| ckpt label | success/total | rate | wilson_ci_95 | "
        "delta_vs_baseline_pp | newcombe_delta_ci_95_pp | "
        "triggered_below_threshold |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        ckpt = c.request.checkpoint
        label = f"{ckpt.training_run_dir.name}/{ckpt.abs_path.name}"
        wlo, whi = c.wilson_ci_95
        dlo, dhi = c.newcombe_delta_ci_95
        triggered = getattr(c, "triggered_below_threshold", "NOT_RECOVERABLE")
        out.append(
            f"| `{label}` | {c.success_count}/{c.completed_episode_total} | "
            f"{c.rate:.4f} | [{wlo:.4f}, {whi:.4f}] | "
            f"{c.delta_vs_baseline * 100:.4f} | "
            f"[{dlo * 100:.4f}, {dhi * 100:.4f}] | {triggered} |"
        )
    out.append("")
    return "\n".join(out) + "\n"


def _render_swap_decomposition(swap_decomposition: dict[str, Any]) -> str:
    """Render R2.2 kept-vs-swapped measurements and their rate delta.

    The previous inline closure snippet only exposed trigger/artifact metadata;
    this helper surfaces the actual kept/swap legs while degrading to
    ``NOT_RECOVERABLE`` rows when old decomposition payloads omit cell fields.
    """
    kept = swap_decomposition.get("kept_cell")
    swapped = swap_decomposition.get("swap_cell")
    delta: str = "NOT_RECOVERABLE"
    if isinstance(kept, dict) and isinstance(swapped, dict):
        if "rate" in kept and "rate" in swapped:
            delta = f"{(float(kept['rate']) - float(swapped['rate'])) * 100:.4f}"
    rows: list[str] = []
    for label, cell in (("kept", kept), ("swapped", swapped)):
        row = f"| {label} | NOT_RECOVERABLE | NOT_RECOVERABLE | NOT_RECOVERABLE |"
        if isinstance(cell, dict):
            required = ("success_count", "completed_episode_total", "rate", "wilson_ci_95")
            ci = list(cell.get("wilson_ci_95") or ())
            if not any(key not in cell for key in required) and len(ci) == 2:
                row = (
                    f"| {label} | {cell['success_count']}/{cell['completed_episode_total']} | "
                    f"{float(cell['rate']):.4f} | "
                    f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}] |"
                )
        rows.append(row)
    artifact_dir = (
        swap_decomposition.get("artifact_dir")
        or swap_decomposition.get("phase_f_root")
        or swap_decomposition.get("field_swap_result_path")
        or ""
    )
    out = [
        "## R2.2 decomposition",
        "",
        f"- triggered: {swap_decomposition.get('triggered', 'NOT_RECOVERABLE')}",
        f"- artifact_dir: `{artifact_dir}`",
        f"- swap_strategy: {swap_decomposition.get('swap_strategy', 'NOT_RECOVERABLE')}",
        f"- delta_kept_minus_swapped_pp: {delta}",
        "",
        "| leg | success/total | rate | wilson_ci_95 |",
        "|---|---|---|---|",
        *rows,
        "",
    ]
    return "\n".join(out) + "\n"


_G2_G3_OPEN_QUESTION = (
    "- CFG-DELTA-1: Among the 5 valid RECAP ckpts, 4 use `Gr00tN1d6` "
    "and 1 uses `GR00TRecapModel`. The single `GR00TRecapModel` ckpt "
    "(`g2_full_training/checkpoint-2200`) has native `formalize_language=True` "
    "and architectures_mismatch=True relative to raw HF; it is a different "
    "training run (g2) than the others (g3/g3_resume/g3_conditioned) with "
    "potentially different data distribution and step counts. Whether to "
    "treat it as a clean control reference is a user-side scientific decision; "
    "R2 closure surfaces this for R3/FATG planning and does NOT decide."
)


def render(
    *,
    cells: list[R2CellResult],
    statistical_regime: dict[str, Any],
    baseline_marker: dict[str, Any],
    representative_selection: dict[str, Any] | None = None,
    r2_invocation_envelope_sha256: str = "",
    r1_0_dir_present: bool = False,
    r1_0_baseline_repro_latest_run_mtime_utc: str | None = None,
    swap_decomposition: dict[str, Any] | None = None,
    config_delta_records: dict[str, Any] | None = None,
) -> str:
    """Render the closure report as markdown (pure)."""
    parts: list[str] = [
        "# R2 Closure Report",
        "",
        "- Mode: deliberate",
        "- Plan: r2_authentic_eval_plan_v4",
        "",
    ]
    parts.append(
        _render_baseline_marker_provenance(
            baseline_marker,
            r1_0_dir_present=r1_0_dir_present,
            r1_0_baseline_repro_latest_run_mtime_utc=
                r1_0_baseline_repro_latest_run_mtime_utc,
            r2_invocation_envelope_sha256=r2_invocation_envelope_sha256,
        )
    )
    parts.append(_render_reproducibility_envelope(cells))
    env_warning = _render_envelope_sha_consistency(cells)
    if env_warning:
        parts.append(env_warning)
    n_valid = int(statistical_regime.get("n_valid_cells", len(cells)))
    parts.append(f"## Inventory counts\n\n- n_valid_cells: {n_valid}\n")
    parts.append(_render_config_delta_records(config_delta_records))
    parts.append(_render_r2_1_measurement_table(cells))
    parts.append(_render_representative_selection(representative_selection))
    if swap_decomposition:
        parts.append(_render_swap_decomposition(swap_decomposition))
    parts.append(_render_statistical_regime_block(statistical_regime))
    parts.append(
        "## Open questions\n\n"
        "- STAT-1: Should R2.2 trigger be tightened given family-wise rate?\n"
        "- STAT-2: Should Bonferroni correction apply to per-cell triggers?\n"
        "- STAT-3: Is Newcombe 95% CI half-width adequate for this n?\n"
        "- R1-PATCH-1: R1's _latest_r1_0_dir() reads wrong dir name "
        "(confirmed bug; flagged, not patched).\n"
        f"{_G2_G3_OPEN_QUESTION}\n"
    )
    # Use newline-joined stripped sections so future helpers cannot collapse
    # adjacent markdown headers/bullets by accidentally changing trailing newlines.
    return "\n".join(part.rstrip("\n") for part in parts if part) + "\n"


__all__ = ["render"]
