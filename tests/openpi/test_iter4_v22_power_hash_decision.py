from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from work.openpi.scripts.materialize_iter4_v22_power_hash_decision import (
    RUN_ID,
    materialize_iter4_v22_power_hash_decision,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_missing_inputs_emit_block_decision_with_all_predicates(tmp_path: Path) -> None:
    result = materialize_iter4_v22_power_hash_decision(
        tmp_path,
        now_utc=datetime(2026, 4, 25, 8, 0, tzinfo=timezone.utc),
    )

    decision = result["decision"]
    assert decision["hash_lock_allowed"] is False
    assert len(decision["predicate_inputs"]) == 12
    assert decision["predicate_inputs"]["power_analysis_done"] is True
    assert "iter3_authority_index_missing" in decision["predicate_failures"]
    assert "desaturated_protocol_not_selected" in decision["predicate_failures"]

    power = result["power_analysis"]
    assert len(power["effect_size_grid"]) == 4
    assert power["null_arm"]["threshold_used"] == 0.0105
    assert power["null_arm"]["recommended_minimum_episodes_per_variant"] == 768

    output_root = tmp_path / "agent" / "artifacts" / RUN_ID / "openpi" / "v22_preregistration"
    assert (output_root / "v22_hash_lock_decision_iter4.json").is_file()
    assert not (output_root / "v22_preregistration_hash_lock.json").exists()


def test_all_predicates_true_create_hash_lock(tmp_path: Path) -> None:
    now = datetime(2026, 4, 25, 8, 0, tzinfo=timezone.utc)
    stage = tmp_path / "agent" / "artifacts" / RUN_ID
    gr00t = tmp_path / "agent" / "artifacts" / "recap_min_loop" / "single_gpu_v2_full_update" / RUN_ID

    _write_json(
        stage / "coordinator" / "iter3_authority_index.json",
        {
            "schema_version": "iter3_authority_index_v1",
            "iter3_terminal_state": {
                "saturation_evidence": {
                    "stdev_c_minus_b": 0.012,
                    "stdev_c_minus_x": 0.021,
                }
            },
        },
    )
    _write_json(
        stage / "coordinator" / "gate_policy.json",
        {"w2_gates_w5": True, "p6_bash_hook_self_test_pass": True},
    )
    _write_json(stage / "paper_audit" / "r2_closure" / "r2_closure_verdict.json", {"r2_status": "CLOSED"})
    _write_json(stage / "paper_audit" / "r4_closure" / "r4_closure_verdict.json", {"r4_status": "CLOSED"})
    _write_json(
        gr00t / "gr00t" / "r2_r4_shape_regression" / "r2_r4_shape_regression.json",
        {
            "status": "PASS",
            "deterministic_replay": {"match": True, "rtol": 0.01, "atol": 0.005},
            "alpha_effect": {"above_threshold": True, "relative_diff": 0.02, "absolute_diff": 0.006},
            "checkpoint": {"compat_check_passed": True},
            "m12_rollback_triggered": False,
        },
    )
    _write_json(
        gr00t / "gr00t" / "p5_failure_analysis" / "p5_negative_result_interpretation.json",
        {"p5_failure_pattern_status": "clear_systematic"},
    )
    design_root = stage / "openpi" / "v22_desaturation_design"
    for name in (
        "v22_candidate_matrix.json",
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
            "variant_codes_used": ["A"],
        },
    )

    result = materialize_iter4_v22_power_hash_decision(tmp_path, now_utc=now)

    assert result["decision"]["hash_lock_allowed"] is True
    assert result["decision"]["reason"] == "predicate_satisfied"
    output_root = tmp_path / "agent" / "artifacts" / RUN_ID / "openpi" / "v22_preregistration"
    hash_lock = json.loads((output_root / "v22_preregistration_hash_lock.json").read_text(encoding="utf-8"))
    assert hash_lock["schema_version"] == "iter4_v22_preregistration_hash_lock_v1"
    assert len(hash_lock["locked_artifact_sha256"]) == 64
