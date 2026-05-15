"""Tests for the closure report renderer (V3-FIX-1 + V4-FIX-1 + WM-1).

Implements acceptance rows 11 + 12 of plan v4 §7:
  - row 11 (V4-FIX-1): renderer parameterises ``n_valid_cells`` for both the
    narrative count and the family-wise exponent (no hardcoded ``5``).
  - row 12 (V3-FIX-1): renderer interpolates statistical-regime numerics from
    runtime helpers (no hardcoded ``0.099/0.407/0.23``).
  - WM-1: cross-cell envelope-sha mismatch surfaces as a warning subsection.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from work.recap.r2_authentic_eval.delta_stats import (
    EVIDENCE_GRADE_N_CELLS,
    family_wise_error_rate_at_baseline,
    newcombe_half_width_at_baseline,
    per_cell_below_trigger_probability,
)
from work.recap.r2_authentic_eval.eval_runner import (
    AuthenticEvalRequest,
    R2CellResult,
)
from work.recap.r2_authentic_eval.inventory import TrainedCheckpoint
from work.recap.r2_authentic_eval.reports import closure_report


def _make_stat(
    *,
    baseline_succ: int = 17,
    baseline_total: int = 30,
    n_valid_cells: int = EVIDENCE_GRADE_N_CELLS,
    raw_observation_cell_count: int = 5,
) -> dict[str, Any]:
    return {
        "baseline_succ": baseline_succ,
        "baseline_total": baseline_total,
        "n_valid_cells": n_valid_cells,
        "evidence_grade_cell_count": n_valid_cells,
        "raw_observation_cell_count": raw_observation_cell_count,
        "per_cell_p_below_trigger": per_cell_below_trigger_probability(baseline_succ, baseline_total),
        "family_wise_at_baseline": family_wise_error_rate_at_baseline(
            baseline_succ, baseline_total, n_cells=n_valid_cells
        ),
        "newcombe_half_width_at_baseline": newcombe_half_width_at_baseline(baseline_succ, baseline_total),
        "regime_label": "broad-net pilot signal",
    }


def _make_baseline_marker() -> dict[str, Any]:
    return {
        "timestamp_utc": "2026-05-09T01:23:45Z",
        "protocol_sha256": "c2d7ac0a94ee89b60662f3ec29afe7fe184eaedb832aa6e98719e2dfa0e67982",
        "success_count": 17,
        "wilson_ci_low": 0.39,
        "wilson_ci_high": 0.73,
    }


def _make_cell(
    envelope_sha: str = "env_sha_aaa",
    slug: str = "checkpoint-2200",
    *,
    training_run_name: str = "fake",
    success_count: int = 17,
    completed_episode_total: int = 30,
    rate: float | None = None,
    wilson_ci_95: tuple[float, float] = (0.39, 0.73),
    delta_vs_baseline: float = 0.0,
    newcombe_delta_ci_95: tuple[float, float] = (-0.24, 0.24),
    triggered_below_threshold: bool | None = None,
) -> R2CellResult:
    run_dir = Path("/fake") / training_run_name
    ck = TrainedCheckpoint(
        label="RECAP",
        abs_path=run_dir / slug,
        training_algo="GR00TN1Policy",
        base_ckpt_at_training="nvidia/GR00T-N1.6-G1",
        formalize_language=False,
        statistics_q99_right_hand=(1.5, 1.5, 1.0, 1.5, 0.0, 0.0, 0.0),
        statistics_q99_matches_base=True,
        n_train_steps=2200,
        training_run_dir=run_dir,
        config_json_sha256="x",
        processor_config_json_sha256="y",
        statistics_json_sha256="z",
        is_valid=True,
        invalid_reason="",
    )
    req = AuthenticEvalRequest(checkpoint=ck, search_root=Path("/fake"), strict_config=False)
    cell = R2CellResult(
        request=req,
        success_count=success_count,
        completed_episode_total=completed_episode_total,
        rate=rate if rate is not None else success_count / completed_episode_total,
        wilson_ci_95=wilson_ci_95,
        delta_vs_baseline=delta_vs_baseline,
        newcombe_delta_ci_95=newcombe_delta_ci_95,
        artifact_dir=Path("/fake"),
        formal_eval_summary_json={"status": "PASS"},
        raw_repro_result=None,
        ckpt_pre_run_sha256={},
        r1_0_dir_present=False,
        r1_0_baseline_repro_latest_run_mtime_utc=None,
        git_commit_sha="a" * 40,
        nvidia_smi_pre_run_csv="0,A100,0,80000,0",
        transformers_version="4.40.0",
        torch_version="2.3.0",
        python_version="3.10.12",
        gr00t_version=None,
        protocol_sha256="proto_abc",
        r2_invocation_envelope_sha256=envelope_sha,
    )
    if triggered_below_threshold is not None:
        object.__setattr__(cell, "triggered_below_threshold", triggered_below_threshold)
    return cell


@pytest.fixture
def minimal_cells() -> list[R2CellResult]:
    """3 cells covering: triggered, not-triggered, ckpt label parsing."""
    return [
        _make_cell(
            training_run_name="g2_full_training", success_count=0, rate=0.0,
            wilson_ci_95=(0.0, 0.11351339317396873),
            delta_vs_baseline=-0.566667,
            newcombe_delta_ci_95=(-0.7262251442353347, -0.35833245645777806),
            triggered_below_threshold=True,
        ),
        _make_cell(training_run_name="g3_resume_after_demo_full_training", success_count=17,
                   rate=0.5667, triggered_below_threshold=False),
        _make_cell(slug="checkpoint-6600", training_run_name="g3_conditioned", success_count=3,
                   rate=0.1, triggered_below_threshold=True),
    ]


# ---------------------------------------------------------------------------
# Self-containment / structure
# ---------------------------------------------------------------------------


def test_closure_contains_required_self_containment_fields() -> None:
    cells = [_make_cell()]
    md = closure_report.render(
        cells=cells,
        statistical_regime=_make_stat(),
        baseline_marker=_make_baseline_marker(),
        r2_invocation_envelope_sha256="envelope_xxx",
    )
    for required in (
        "# R2 Closure Report",
        "Plan: r2_authentic_eval_plan_v4",
        "## Baseline marker provenance",
        "protocol_sha256",
        "## Reproducibility envelope",
        "## Statistical regime",
        "## Open questions",
    ):
        assert required in md, f"missing required section: {required!r}"


def test_closure_does_not_propose_r3() -> None:
    md = closure_report.render(
        cells=[_make_cell()],
        statistical_regime=_make_stat(),
        baseline_marker=_make_baseline_marker(),
    )
    assert "next steps beyond closure" not in md.lower()
    assert "R2 closure surfaces this for R3/FATG planning and does NOT decide" in md
    assert "next steps beyond closure" not in md.lower()


# ---------------------------------------------------------------------------
# V3-FIX-1: statistical-regime numerics interpolated from runtime helpers
# ---------------------------------------------------------------------------


def test_closure_declares_statistical_regime_block() -> None:
    """Rendered block must contain runtime numeric values to 3 decimals."""
    stat = _make_stat()
    md = closure_report.render(
        cells=[_make_cell()],
        statistical_regime=stat,
        baseline_marker=_make_baseline_marker(),
    )
    assert f"{stat['per_cell_p_below_trigger']:.3f}" in md
    assert f"{stat['family_wise_at_baseline']:.3f}" in md
    assert "0.341" in md
    assert "evidence_grade_cell_count: 4" in md
    assert "raw_observation_cell_count: 5" in md
    assert f"{stat['newcombe_half_width_at_baseline']:.3f}" in md
    assert "broad-net pilot signal" in md


# ---------------------------------------------------------------------------
# V4-FIX-1: n_valid_cells SSOT — narrative + exponent + family-wise vary
# ---------------------------------------------------------------------------


def test_closure_statistical_regime_tracks_n_cells_change() -> None:
    """Three pairs MUST differ when n_valid_cells changes (SSOT-drift detector)."""
    stat5 = _make_stat(n_valid_cells=5)
    stat4 = _make_stat(n_valid_cells=4)
    md5 = closure_report.render(
        cells=[_make_cell()],
        statistical_regime=stat5,
        baseline_marker=_make_baseline_marker(),
    )
    md4 = closure_report.render(
        cells=[_make_cell()],
        statistical_regime=stat4,
        baseline_marker=_make_baseline_marker(),
    )
    # narrative count
    assert "across 5 evidence-grade RECAP cells" in md5
    assert "across 4 evidence-grade RECAP cells" in md4
    # exponent
    assert re.search(r"\^5\b", md5) is not None
    assert re.search(r"\^4\b", md4) is not None
    # family-wise number renders to 3 decimals and differs
    fwer5 = stat5["family_wise_at_baseline"]
    fwer4 = stat4["family_wise_at_baseline"]
    assert f"{fwer5:.3f}" in md5
    assert f"{fwer4:.3f}" in md4
    assert fwer5 != fwer4

    # Numeric agreement: regex extraction matches the dict's n_valid_cells
    m_narrative = re.search(r"across (\d+) evidence-grade RECAP cells", md5)
    m_exponent = re.search(r"\^(\d+)\b", md5)
    assert m_narrative is not None and int(m_narrative.group(1)) == stat5["n_valid_cells"]
    assert m_exponent is not None and int(m_exponent.group(1)) == stat5["n_valid_cells"]


def test_closure_statistical_regime_tracks_baseline_change() -> None:
    """Baseline change must propagate through statistical regime numerics."""
    stat_a = _make_stat(baseline_succ=17, baseline_total=30, n_valid_cells=5)
    stat_b = _make_stat(baseline_succ=18, baseline_total=30, n_valid_cells=5)
    md_a = closure_report.render(
        cells=[_make_cell()], statistical_regime=stat_a, baseline_marker=_make_baseline_marker()
    )
    md_b = closure_report.render(
        cells=[_make_cell()], statistical_regime=stat_b, baseline_marker=_make_baseline_marker()
    )
    assert f"{stat_a['per_cell_p_below_trigger']:.3f}" in md_a
    assert f"{stat_b['per_cell_p_below_trigger']:.3f}" in md_b
    assert stat_a["per_cell_p_below_trigger"] != stat_b["per_cell_p_below_trigger"]


# ---------------------------------------------------------------------------
# WM-1: envelope-sha consistency warning
# ---------------------------------------------------------------------------


def test_closure_does_not_warn_when_envelopes_consistent() -> None:
    cells = [_make_cell(envelope_sha="env_aaa", slug="ckpt-A"),
             _make_cell(envelope_sha="env_aaa", slug="ckpt-B")]
    md = closure_report.render(
        cells=cells,
        statistical_regime=_make_stat(),
        baseline_marker=_make_baseline_marker(),
    )
    assert "Envelope-sha consistency (WARNING)" not in md


def test_closure_flags_envelope_sha_inconsistency_across_cells() -> None:
    cells = [
        _make_cell(envelope_sha="env_aaa", slug="ckpt-A"),
        _make_cell(envelope_sha="env_bbb", slug="ckpt-B"),
    ]
    md = closure_report.render(
        cells=cells,
        statistical_regime=_make_stat(),
        baseline_marker=_make_baseline_marker(),
    )
    assert "Envelope-sha consistency (WARNING)" in md
    assert "env_aaa" in md and "env_bbb" in md


# ---------------------------------------------------------------------------
# Baseline marker provenance + envelope sha row
# ---------------------------------------------------------------------------


def test_closure_includes_baseline_marker_provenance_table() -> None:
    md = closure_report.render(
        cells=[_make_cell()],
        statistical_regime=_make_stat(),
        baseline_marker=_make_baseline_marker(),
        r2_invocation_envelope_sha256="env_outer",
        r1_0_dir_present=True,
        r1_0_baseline_repro_latest_run_mtime_utc="2026-05-08T12:00:00Z",
    )
    for label in (
        "timestamp_utc",
        "protocol_sha256",
        "success_count",
        "wilson_ci_low",
        "wilson_ci_high",
        "r1_0_dir_present",
        "r1_0_baseline_repro_latest_run_mtime_utc",
        "r2_invocation_envelope_sha256",
    ):
        assert label in md, f"missing baseline marker row: {label}"
    assert "env_outer" in md


# ---------------------------------------------------------------------------
# Smoke: render does not crash with zero cells
# ---------------------------------------------------------------------------


def test_closure_render_handles_empty_cells() -> None:
    md = closure_report.render(
        cells=[],
        statistical_regime=_make_stat(n_valid_cells=0),
        baseline_marker=_make_baseline_marker(),
    )
    assert "## Statistical regime" in md


# ---------------------------------------------------------------------------
# Open questions
# ---------------------------------------------------------------------------


def test_closure_open_questions_listed() -> None:
    md = closure_report.render(
        cells=[_make_cell()],
        statistical_regime=_make_stat(),
        baseline_marker=_make_baseline_marker(),
    )
    for oq in ("STAT-1", "STAT-2", "STAT-3", "R1-PATCH-1", "CFG-DELTA-1"):
        assert oq in md, f"missing open question: {oq}"


def test_closure_renders_config_delta_records() -> None:
    inventory = {
        "row_count": 1,
        "summary": {
            "ONLY_FORMALIZE_LANGUAGE": 0,
            "ADDITIONAL_FIELDS_DIFFER": 1,
            "architectures_mismatch_count": 1,
        },
        "rows": [
            {
                "ckpt_root": "/fake/checkpoint-2200",
                "classification": "ADDITIONAL_FIELDS_DIFFER",
                "architectures_mismatch": True,
                "outside_paths": ["config.json:architectures"],
            }
        ],
    }
    md = closure_report.render(
        cells=[_make_cell()],
        statistical_regime=_make_stat(),
        baseline_marker=_make_baseline_marker(),
        config_delta_records=inventory,
    )
    assert "## R2.1 cells x R2.0.5 classification" in md
    assert "ADDITIONAL_FIELDS_DIFFER" in md
    assert "config.json:architectures" in md


def test_r2_1_measurement_table_includes_all_cells(minimal_cells: list[R2CellResult]) -> None:
    out = closure_report._render_r2_1_measurement_table(minimal_cells)
    assert "g2_full_training/checkpoint-2200" in out
    assert "g3_resume_after_demo_full_training/checkpoint-2200" in out
    assert "g3_conditioned/checkpoint-6600" in out
    assert "0/30" in out and "17/30" in out
    assert "0.1135" in out
    assert "-56.6667" in out
    assert "-72.6225" in out and "-35.8332" in out
    assert "True" in out and "False" in out


def test_swap_decomposition_renders_kept_and_swapped_rates() -> None:
    out = closure_report._render_swap_decomposition({
        "triggered": True, "artifact_dir": "x", "swap_strategy": "field_targeted_swap",
        "kept_cell": {"success_count": 17, "completed_episode_total": 30,
                      "rate": 0.5667, "wilson_ci_95": [0.39, 0.73]},
        "swap_cell": {"success_count": 0, "completed_episode_total": 30,
                      "rate": 0.0, "wilson_ci_95": [0.0, 0.1135]},
    })
    assert "0.5667" in out and "0.0000" in out
    assert "delta_kept_minus_swapped_pp: 56.6700" in out
    assert "field_targeted_swap" in out


def test_swap_decomposition_handles_missing_cells_gracefully() -> None:
    out = closure_report._render_swap_decomposition({"triggered": True, "artifact_dir": "x"})
    assert "NOT_RECOVERABLE" in out
    assert "## R2.2 decomposition" in out


def test_representative_selection_derives_label_from_path_when_recap() -> None:
    out = closure_report._render_representative_selection({
        "selected_label": "RECAP", "selected_path": "/root/run_a/checkpoint-6600",
        "selected_n_train_steps": 6600, "selected_reason": "synthetic",
    })
    assert "- selected_label: run_a/checkpoint-6600" in out
    assert "\n- selected_label: RECAP\n" not in out


def test_render_output_has_no_concatenated_header_lines(
    minimal_cells: list[R2CellResult],
) -> None:
    md = closure_report.render(
        cells=minimal_cells, statistical_regime=_make_stat(),
        baseline_marker=_make_baseline_marker(),
        representative_selection={"selected_label": "RECAP", "selected_path": "/x/y/z"},
        swap_decomposition={"triggered": True, "artifact_dir": "x"},
    )
    assert md.splitlines()[0] == "# R2 Closure Report"
    assert re.search(r"Report-\s*Mode", md) is None
    assert re.search(r"deliberate-\s*Plan", md) is None
