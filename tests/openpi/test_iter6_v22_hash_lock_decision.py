from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from work.openpi.pipelines.recap.iter6_hash_lock import (
    ATOM_ORDER,
    ITER5P5_RUN_ID,
    ITER5_RUN_ID,
    RUN_ID,
    materialize_iter6_v22_hash_lock_decision,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_hashes_for(paths: list[Path], repo_root: Path) -> dict[str, str]:
    return {path.relative_to(repo_root).as_posix(): _sha256(path) for path in paths}


def test_missing_inputs_emit_decision_only_without_hash_lock_file(tmp_path: Path) -> None:
    result = materialize_iter6_v22_hash_lock_decision(
        tmp_path,
        now_utc=datetime(2026, 4, 25, 16, 30, tzinfo=timezone.utc),
    )

    decision = result["decision"]
    assert decision["hash_lock_allowed_iter6"] is False
    assert decision["hash_lock_file_emission_authorized_by_w5"] is False
    assert decision["hash_lock_file_emitter"] == "W8"
    assert len(decision["predicate_inputs"]) == len(ATOM_ORDER)
    assert decision["predicate_inputs"]["pytest_full_scope_post_fix_attestation_pass"] == "DEFERRED_TO_W8"
    assert decision["predicate_inputs"]["hash_lock_file_emission_post_w8"] == "DEFERRED_TO_W8"

    output_root = tmp_path / "agent" / "artifacts" / RUN_ID
    assert (output_root / "openpi" / "v22_preregistration_iter6" / "v22_hash_lock_decision_iter6.json").is_file()
    assert (output_root / "verifier" / "v22_hash_lock_decision_iter6.json").is_file()
    assert not (output_root / "openpi" / "v22_preregistration_iter6" / "v22_preregistration_hash_lock.json").exists()


def test_all_w5_predicates_true_still_defers_hash_lock_file_to_w8(tmp_path: Path) -> None:
    now = datetime(2026, 4, 25, 18, 0, tzinfo=timezone.utc)
    iter5 = tmp_path / "agent" / "artifacts" / ITER5_RUN_ID
    iter5_shape = (
        tmp_path
        / "agent"
        / "artifacts"
        / "recap_min_loop"
        / "single_gpu_v2_full_update"
        / ITER5_RUN_ID
        / "gr00t"
        / "r2_r4_shape_regression"
    )
    iter5p5 = tmp_path / "agent" / "artifacts" / ITER5P5_RUN_ID
    iter6 = tmp_path / "agent" / "artifacts" / RUN_ID

    carry_forward_paths = [
        iter5 / "coordinator" / "iter4_authority_index.json",
        iter5 / "coordinator" / "iter5_authority_index.json",
        iter5 / "coordinator" / "n_per_variant_decision.json",
        iter5_shape / "run_manifest.json",
        iter5p5 / "coordinator" / "canonical_blind_selection_rule.json",
        iter5p5 / "coordinator" / "artifact_freeze_manifest.json",
    ]
    _write_json(carry_forward_paths[0], {"sha256_entries": [{"path": "a", "sha256": "0" * 64}]})
    _write_json(carry_forward_paths[1], {"sha256_entries": [{"path": "b", "sha256": "1" * 64}]})
    _write_json(
        carry_forward_paths[2],
        {
            "schema_version": "iter5_n_per_variant_decision_v1",
            "n_per_variant": 192,
            "confirmed_by": "user_pre_launch",
            "confirmed_at_utc": "2026-04-25T10:00:00Z",
            "source_inline_text": "I, the user, confirm n_per_variant=192 for iter5 hash-lock.",
            "rationale": "high-confidence band",
        },
    )
    _write_json(carry_forward_paths[3], {"terminal_at_utc": "2026-04-25T12:00:00Z"})
    _write_json(carry_forward_paths[4], {"schema_version": "iter5p5_rule"})
    _write_json(carry_forward_paths[5], {"schema_version": "iter5p5_freeze", "entries": []})

    _write_json(iter5 / "paper_audit" / "r2_closure" / "r2_closure_verdict.json", {"status": "CLOSED"})
    _write_json(iter5 / "paper_audit" / "r4_closure" / "r4_closure_verdict.json", {"status": "CLOSED"})
    _write_json(
        iter5_shape / "r2_r4_shape_regression.json",
        {
            "status": "PASS",
            "deterministic_replay": {"match": True},
            "alpha_effect": {"above_threshold": True},
            "m12_rollback_triggered": False,
        },
    )
    _write_json(
        iter5 / "gr00t" / "p5_failure_analysis" / "p5_negative_result_interpretation.json",
        {"p5_failure_pattern_status": "clear_systematic"},
    )

    _write_json(iter6 / "coordinator" / "clock_sanity_iter6.json", {"clock_sanity_pass": True, "all_workers_ntp_synced": True, "max_offset_seconds_observed": 0})
    _write_json(iter6 / "coordinator" / "iter6_p6_bash_hook_self_test_report.json", {"p6_bash_hook_self_test_pass": True})
    _write_json(iter6 / "openpi" / "pytest_triage" / "triage_matrix.json", {"unclassified_failures": 0, "errors_unclassified": 0})
    _write_json(iter6 / "openpi" / "pytest_triage" / "v22_critical_test_manifest.json", {"type_a_count": 1})
    _write_jsonl(
        iter6 / "openpi" / "pytest_triage" / "v22_critical_fix_log.jsonl",
        [{"test_id": "tests/openpi/test_example.py::test_type_a", "posttest_status": "PASS"}],
    )
    _write_json(
        iter6 / "openpi" / "v22_candidate_space_iter6" / "v22_candidate_space_matrix.json",
        {"candidate_space_expanded": True, "budget_grid": [0.10, 0.25], "suite_candidates": ["libero_goal"]},
    )
    _write_json(
        iter6 / "openpi" / "v22_candidate_space_iter6" / "blind_selection_rule_v2.json",
        {"uses_c_results_for_selection": False},
    )
    _write_json(iter6 / "openpi" / "v22_candidate_space_iter6" / "calibration_budget_grid.json", {"budget_grid": [0.10, 0.15]})
    _write_json(iter6 / "openpi" / "v22_candidate_space_iter6" / "suite_task_discovery_report.json", {"suite_candidates": ["libero_goal"]})
    _write_json(iter6 / "openpi" / "v22_candidate_space_iter6" / "no_c_leakage_attestation.json", {"uses_c_results_for_selection": False})

    canonical_rule = iter6 / "coordinator" / "canonical_blind_selection_rule_iter6.json"
    _write_json(canonical_rule, {"schema_version": "iter6_rule", "rule": {"budget": 0.10}})
    _write_text(iter6 / "coordinator" / "canonical_blind_selection_rule_iter6.sha256", _sha256(canonical_rule) + "  canonical_blind_selection_rule_iter6.json\n")
    _write_json(iter6 / "openpi" / "v22_blind_calibration" / "precondition_check_iter6.json", {"hash_match": True})
    _write_json(
        iter6 / "openpi" / "v22_blind_calibration_iter6" / "desaturation_selection_decision.json",
        {
            "calibration_status": "DESATURATED_FOUND",
            "terminal_at_utc": "2026-04-25T17:00:00Z",
            "selected_using_c_results": False,
            "variant_codes_used": ["A", "B"],
            "selected_suite": "libero_goal",
            "selected_tasks": [0, 1],
            "selected_budget": 0.10,
        },
    )
    frozen_input = iter6 / "coordinator" / "iter6_authority_index.json"
    _write_json(frozen_input, {"schema_version": "iter6_authority_index_v1"})
    _write_json(
        iter6 / "coordinator" / "artifact_freeze_manifest_iter6.json",
        {"entries": [{"path": frozen_input.relative_to(tmp_path).as_posix(), "sha256_at_capture": _sha256(frozen_input), "allow_overwrite": False}]},
    )

    result = materialize_iter6_v22_hash_lock_decision(
        tmp_path,
        now_utc=now,
        expected_hashes=_expected_hashes_for(carry_forward_paths, tmp_path),
    )

    decision = result["decision"]
    assert decision["hash_lock_allowed_iter6"] is True
    assert decision["reason"] == "predicate_satisfied_pending_w8"
    assert decision["hash_lock_file_emission_authorized_by_w5"] is False
    assert decision["predicate_failures"] == []
    assert decision["predicate_inputs"]["pytest_full_scope_post_fix_attestation_pass"] == "DEFERRED_TO_W8"
    assert decision["predicate_inputs"]["hash_lock_file_emission_post_w8"] == "DEFERRED_TO_W8"

    hash_lock = iter6 / "openpi" / "v22_preregistration_iter6" / "v22_preregistration_hash_lock.json"
    assert not hash_lock.exists()
    handoff = json.loads((iter6 / "verifier" / "iter6a_handoff_spec.json").read_text(encoding="utf-8"))
    assert handoff["v22_preregistration_hash_lock_emitter"] == "W8"
    assert handoff["n_per_variant_locked"] == 192
