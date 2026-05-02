from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from work.openpi.pipelines.recap.iter5_hash_lock import (
    RUN_ID,
    materialize_iter5_final_validator,
    materialize_iter5_v22_power_hash_decision,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_missing_inputs_emit_iter5_block_decision_with_all_predicates(tmp_path: Path) -> None:
    result = materialize_iter5_v22_power_hash_decision(
        tmp_path,
        now_utc=datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc),
    )

    decision = result["decision"]
    assert decision["hash_lock_allowed_iter5"] is False
    assert len(decision["predicate_inputs"]) == 16
    assert decision["predicate_inputs"]["power_analysis_done"] is True
    assert "iter4_authority_index_missing_or_incomplete" in decision["predicate_failures"]
    assert "n_per_variant_crosscheck_failed" in decision["predicate_failures"]

    power = result["power_analysis"]
    assert len(power["effect_size_grid"]) == 4
    assert power["null_arm"]["threshold_used"] == 0.0105
    assert power["null_arm"]["recommended_minimum_episodes_per_variant"] == 768

    output_root = tmp_path / "agent" / "artifacts" / RUN_ID / "openpi" / "v22_preregistration"
    assert (output_root / "v22_hash_lock_decision_iter5.json").is_file()
    assert not (output_root / "v22_preregistration_hash_lock.json").exists()


def test_all_iter5_predicates_true_create_hash_lock(tmp_path: Path) -> None:
    now = datetime(2026, 4, 25, 14, 0, tzinfo=timezone.utc)
    stage = tmp_path / "agent" / "artifacts" / RUN_ID
    gr00t = tmp_path / "agent" / "artifacts" / "recap_min_loop" / "single_gpu_v2_full_update" / RUN_ID

    _write_json(
        stage / "coordinator" / "iter4_authority_index.json",
        {"sha256_entries": [{"path": f"artifact-{idx}", "sha256": "0" * 64} for idx in range(9)]},
    )
    _write_json(stage / "verifier" / "final_redesign_summary.json", {"iter4_artifact_drift_detected": False})
    _write_json(stage / "paper_audit" / "r2_closure" / "r2_closure_verdict.json", {"status": "CLOSED"})
    _write_json(stage / "paper_audit" / "r4_closure" / "r4_closure_verdict.json", {"status": "CLOSED"})
    _write_json(
        stage / "coordinator" / "run_manifest.json",
        {"W0_started_at_utc": "2026-04-25T12:00:00Z"},
    )
    _write_json(
        stage / "coordinator" / "n_per_variant_decision.json",
        {
            "schema_version": "iter5_n_per_variant_decision_v1",
            "n_per_variant": 96,
            "rationale": "minimum locked N that still satisfies the iter5 null-arm power check",
            "confirmed_by": "user_pre_launch",
            "confirmed_at_utc": "2026-04-25T11:59:00Z",
            "source_inline_text": "I, the user, confirm n_per_variant=96 for iter5 hash-lock.",
        },
    )
    _write_json(
        stage / "coordinator" / "clock_sanity.json",
        {
            "clock_sanity_pass": True,
            "all_workers_ntp_synced": True,
            "max_offset_seconds_observed": 0,
        },
    )
    _write_json(stage / "verifier" / "timestamp_cross_validation.json", {"all_within_tolerance": True})
    _write_text(
        stage / "verifier" / "iter5_p6_bash_hook_self_test.log",
        "\n".join(["forbidden exit 2"] * 5 + ["clean exit 0"] * 2),
    )
    _write_json(
        gr00t / "gr00t" / "r2_r4_shape_regression" / "run_manifest.json",
        {"terminal_at_utc": "2026-04-25T13:00:00Z"},
    )
    _write_json(
        gr00t / "gr00t" / "r2_r4_shape_regression" / "deterministic_replay_diff.json",
        {"match": True, "max_per_step_relative_diff": 0.009, "max_per_step_absolute_diff": 0.004},
    )
    _write_json(
        gr00t / "gr00t" / "r2_r4_shape_regression" / "alpha_ablation_report.json",
        {"relative_diff": 0.02, "absolute_diff": 0.006},
    )
    _write_json(
        gr00t / "gr00t" / "r2_r4_shape_regression" / "c2_rollback_reason.json",
        {"triggered": False},
    )
    _write_json(
        gr00t / "gr00t" / "p5_failure_analysis" / "p5_negative_result_interpretation.json",
        {"p5_failure_pattern_status": "clear_systematic", "do_not_rerun_same_checkpoint": True},
    )
    design_root = stage / "openpi" / "v22_desaturation_design"
    _write_json(
        design_root / "v22_candidate_matrix.json",
        {"v22_design_status": "READY_FOR_BLIND_CALIBRATION", "created_at_utc": "2026-04-25T13:10:00Z"},
    )
    for name in (
        "blind_selection_rule.json",
        "v22_metric_plan.json",
        "v22_exclusion_rules.json",
        "v22_formal_variant_definitions.json",
    ):
        _write_json(design_root / name, {"v22_design_status": "READY_FOR_BLIND_CALIBRATION"})
    _write_json(
        stage / "openpi" / "v22_blind_calibration" / "desaturation_selection_decision.json",
        {
            "calibration_status": "DESATURATED_FOUND",
            "selected_using_c_results": False,
            "variant_codes_used": ["A", "B"],
        },
    )

    result = materialize_iter5_v22_power_hash_decision(tmp_path, now_utc=now)

    assert result["decision"]["hash_lock_allowed_iter5"] is True
    assert result["decision"]["reason"] == "predicate_satisfied"
    assert result["decision"]["n_per_variant_locked"] == 96
    output_root = tmp_path / "agent" / "artifacts" / RUN_ID / "openpi" / "v22_preregistration"
    hash_lock = json.loads((output_root / "v22_preregistration_hash_lock.json").read_text(encoding="utf-8"))
    assert hash_lock["schema_version"] == "iter5_v22_preregistration_hash_lock_v1"
    assert len(hash_lock["locked_artifact_sha256"]) == 64


def test_final_validator_does_not_evaluate_hash_lock_before_w6_pass(tmp_path: Path) -> None:
    now = datetime(2026, 4, 25, 14, 0, tzinfo=timezone.utc)
    stage = tmp_path / "agent" / "artifacts" / RUN_ID
    gr00t = tmp_path / "agent" / "artifacts" / "recap_min_loop" / "single_gpu_v2_full_update" / RUN_ID

    _write_json(stage / "paper_audit" / "r2_closure" / "r2_closure_verdict.json", {"status": "CLOSED"})
    _write_json(stage / "paper_audit" / "r4_closure" / "r4_closure_verdict.json", {"status": "CLOSED"})
    _write_json(stage / "coordinator" / "run_manifest.json", {"W0_started_at_utc": "2026-04-25T12:00:00Z"})
    _write_json(
        stage / "coordinator" / "n_per_variant_decision.json",
        {
            "schema_version": "iter5_n_per_variant_decision_v1",
            "n_per_variant": 192,
            "rationale": "locked launch authorization",
            "confirmed_by": "user_pre_launch",
            "confirmed_at_utc": "2026-04-25T11:59:00Z",
            "source_inline_text": "I, the user, confirm n_per_variant=192 for iter5 hash-lock.",
        },
    )
    _write_json(
        gr00t / "gr00t" / "r2_r4_shape_regression" / "run_manifest.json",
        {"status": "PASS", "terminal_at_utc": "2026-04-25T13:00:00Z"},
    )
    _write_json(
        gr00t / "gr00t" / "r2_r4_shape_regression" / "r2_r4_shape_regression.json",
        {"status": "PASS", "deterministic_replay": {"match": True}, "alpha_effect": {"above_threshold": True}},
    )
    _write_json(
        stage / "openpi" / "v22_desaturation_design" / "v22_candidate_matrix.json",
        {"created_at_utc": "2026-04-25T13:10:00Z"},
    )
    _write_json(
        stage / "openpi" / "v22_blind_calibration" / "desaturation_selection_decision.json",
        {"calibration_status": "BLOCK_PRECONDITION"},
    )

    result = materialize_iter5_final_validator(tmp_path, now_utc=now)

    summary = result["final_recap_longrun_summary"]
    assert summary["phase_statuses"]["phase_e_hash_lock"] == "NOT_EVALUATED"
    assert summary["hash_lock_decision"]["hash_lock_evaluation_status"] == "NOT_EVALUATED_PRECONDITION"
    assert "w6_not_pass" in summary["blocking_reasons_if_block"]
