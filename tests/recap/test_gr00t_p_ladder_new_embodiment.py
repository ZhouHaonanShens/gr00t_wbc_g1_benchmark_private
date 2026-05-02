from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_controller_audit_new_embodiment
from work.recap.scripts import gr00t_ladder_policy_gate


gr00t_p_ladder_new_embodiment = importlib.import_module(
    "work.recap.scripts.gr00t_p_ladder_new_embodiment"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_task2_preflight_evidence(root: Path) -> Path:
    report_path = _write_json(
        root / "preflight_report.json",
        {
            "schema_version": "g1_gr00t_wbc_preflight_gate_v1",
            "status": "PASS",
            "reason_code": "ok",
            "env_resolution": {"ok": True},
            "policy_ping": {"ok": True},
            "action_horizon_check": {"ok": True},
            "smoke": {"step_ok": True},
            "system_break_flags": {"active_breaks": ["none"]},
        },
    )
    runtime_log = root / "00_server.log"
    runtime_log.write_text("preflight ok\n", encoding="utf-8")
    return _write_json(
        root / "task-2-preflight.json",
        {
            "schema_version": "sisyphus_task_evidence_v1",
            "artifact_kind": "task_2_preflight_evidence",
            "verification": {
                "success_run": {
                    "default_report_path": str(report_path),
                    "runtime_log": str(runtime_log),
                }
            },
        },
    )


def _repo_modality_config() -> Path:
    return REPO_ROOT / "work" / "configs" / "new_embodiment" / "modality_config.json"


def _copy_modality_config(
    tmp_path: Path,
    *,
    name: str,
    update_fn: Any | None = None,
) -> Path:
    payload = _read_json(_repo_modality_config())
    if update_fn is not None:
        update_fn(payload)
    return _write_json(tmp_path / name, payload)


def _build_branch_contract_artifacts(
    tmp_path: Path,
    *,
    modality_config_path: Path,
    normalization_override: dict[str, Any] | None = None,
) -> dict[str, Path]:
    modality_contract = gr00t_controller_audit_new_embodiment.load_modality_contract(
        modality_config_path
    )
    branch_manifest = (
        gr00t_controller_audit_new_embodiment.build_branch_manifest_payload(
            modality_config_path=modality_config_path,
            modality_contract=modality_contract,
        )
    )
    if normalization_override is not None:
        branch_manifest["normalization_source"] = copy.deepcopy(normalization_override)
    branch_manifest_path = _write_json(
        tmp_path / "branch_manifest.json", branch_manifest
    )
    controller_audit_payload = {
        "schema_version": gr00t_controller_audit_new_embodiment.REPORT_SCHEMA_VERSION,
        "artifact_kind": gr00t_controller_audit_new_embodiment.REPORT_ARTIFACT_KIND,
        "branch_tag": "NEW_EMBODIMENT",
        "formal_branch_eligibility": "ALLOW",
        "reason_code": "OK",
        "public_anchor_comparable": False,
        "branch_manifest_path": gr00t_p_ladder_new_embodiment.gr00t_p_ladder_unitree_g1._rel_repo(
            branch_manifest_path
        ),
        "modality_config_path": gr00t_p_ladder_new_embodiment.gr00t_p_ladder_unitree_g1._rel_repo(
            modality_config_path
        ),
        "modality_config_fingerprint_sha256": modality_contract[
            "payload_fingerprint_sha256"
        ],
        "normalization_source": copy.deepcopy(branch_manifest["normalization_source"]),
        "controller_provenance": copy.deepcopy(
            branch_manifest["controller_provenance"]
        ),
        "dataset_provenance": copy.deepcopy(branch_manifest["dataset_provenance"]),
    }
    controller_audit_path = _write_json(
        tmp_path / "controller_audit_new_embodiment.json",
        controller_audit_payload,
    )
    return {
        "modality_config": modality_config_path,
        "branch_manifest": branch_manifest_path,
        "controller_audit": controller_audit_path,
    }


def _build_prerequisite_artifacts(
    tmp_path: Path,
    *,
    modality_config_name: str = "modality_config.json",
    modality_update_fn: Any | None = None,
    normalization_override: dict[str, Any] | None = None,
) -> dict[str, Path]:
    root = tmp_path / "prereqs"
    modality_config_path = _copy_modality_config(
        root, name=modality_config_name, update_fn=modality_update_fn
    )
    branch_contract = _build_branch_contract_artifacts(
        root,
        modality_config_path=modality_config_path,
        normalization_override=normalization_override,
    )

    gate_path = root / "p_ladder_policy_gate_new_embodiment.json"
    dual_branch_path = root / "dual_branch_scorecard.json"
    provenance_path = root / "checkpoint_provenance_report.json"
    condition_flip_path = root / "condition_flip_scorecard_new_embodiment.json"
    teacher_gap_path = root / "teacher_student_gap_scorecard_new_embodiment.json"
    action_telemetry_path = root / "action_chain_telemetry_new_embodiment.json"
    teacher_reachability_path = root / "teacher_reachability_gate_new_embodiment.json"
    task2_preflight_evidence = _write_task2_preflight_evidence(root)

    gate_payload = gr00t_ladder_policy_gate.build_ladder_policy_gate(
        branch="NEW_EMBODIMENT",
        axis="P",
        output_path=gate_path,
    )
    _write_json(gate_path, gate_payload)

    _write_json(
        dual_branch_path,
        {
            "artifact_kind": "gr00t_dual_branch_scorecard",
            "allow_p_ladder": {"new_embodiment": True},
            "branches": [
                {
                    "branch_key": "new_embodiment",
                    "branch_scope": "branch_internal_only",
                    "public_anchor_status": {
                        "status": "NOT_APPLICABLE",
                        "summary": {
                            "success_count": 0,
                            "success_rate": 0.0,
                            "systemic_break_flags": [],
                        },
                    },
                }
            ],
            "report_signature_sha256": "dual-branch-signature",
        },
    )
    _write_json(
        provenance_path,
        {
            "artifact_kind": "gr00t_checkpoint_provenance_report",
            "formal_eligibility": "ALLOW",
            "status": "PASS",
            "selected_checkpoint_path": "/tmp/checkpoint-100",
            "loadability_status": "LOADABLE_CHECKPOINT_CONFIRMED",
            "checksum_or_signature": "sha256:checkpoint",
        },
    )
    _write_json(
        condition_flip_path,
        {
            "artifact_kind": "gr00t_condition_flip_scorecard",
            "branch": "NEW_EMBODIMENT",
            "branch_scope": "branch_internal_only",
            "public_anchor_comparable": False,
            "response_ratio": {"min_ratio_across_semantic_flips": 0.15728209},
            "pass_fail_gate": "PASS",
            "paired_scene_id": "new_embodiment::S_drop",
            "report_signature_sha256": "condition-flip-signature",
        },
    )
    _write_json(
        teacher_gap_path,
        {
            "artifact_kind": "gr00t_teacher_student_gap_scorecard",
            "status": "ALLOW",
            "public_anchor_comparable": False,
            "student_branch_match_rate": 0.74489796,
            "summary": {"action_group_gap_count": 3},
            "per_family_gap": [
                {
                    "family": "S_drop",
                    "included_in_formal_scene_pool": True,
                    "teacher_student_success_gap_rate": 1.0,
                },
                {
                    "family": "S_lost",
                    "included_in_formal_scene_pool": True,
                    "teacher_student_success_gap_rate": 1.0,
                },
                {
                    "family": "S_pre_place",
                    "included_in_formal_scene_pool": False,
                    "teacher_student_success_gap_rate": None,
                },
            ],
            "report_signature_sha256": "teacher-gap-signature",
        },
    )
    _write_json(
        action_telemetry_path,
        {
            "artifact_kind": "gr00t_action_chain_telemetry",
            "branch": "NEW_EMBODIMENT",
            "controller_absorbed_groups": ["left_arm"],
            "model_insensitive_groups": ["right_hand"],
            "zero_motion_flags": {"all_zero_in_both_groups": ["right_hand"]},
            "report_signature_sha256": "action-telemetry-signature",
        },
    )
    _write_json(
        teacher_reachability_path,
        {
            "artifact_kind": "gr00t_teacher_reachability_gate",
            "allow_formal_ladders": True,
            "status": "ALLOW",
            "public_anchor_comparable": False,
            "current_baseline": {"success_count": 0, "success_rate": 0.0},
            "reachable_scene_ids": [
                "new_embodiment::S_drop",
                "new_embodiment::S_lost",
            ],
            "scene_pool_status": "formal_teacher_replay_reachable_pool_materialized",
            "scene_pool": {
                "scene_rows": [
                    {
                        "family": "S_drop",
                        "scene_id": "new_embodiment::S_drop",
                        "included_in_formal_scene_pool": True,
                    },
                    {
                        "family": "S_lost",
                        "scene_id": "new_embodiment::S_lost",
                        "included_in_formal_scene_pool": True,
                    },
                ]
            },
            "blocking_reasons": [],
            "report_signature_sha256": "teacher-reachability-signature",
        },
    )
    return {
        "gate": gate_path,
        "dual_branch": dual_branch_path,
        "provenance": provenance_path,
        "condition_flip": condition_flip_path,
        "teacher_gap": teacher_gap_path,
        "action_telemetry": action_telemetry_path,
        "teacher_reachability": teacher_reachability_path,
        "task2_preflight": task2_preflight_evidence,
        **branch_contract,
    }


def _run_cli(
    *,
    rung: str,
    output_root: Path,
    prereqs: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, dict[str, Any]]:
    exit_code = gr00t_p_ladder_new_embodiment.main(
        [
            "--rung",
            rung,
            "--output-root",
            str(output_root),
            "--p-ladder-policy-gate-json",
            str(prereqs["gate"]),
            "--dual-branch-scorecard-json",
            str(prereqs["dual_branch"]),
            "--checkpoint-provenance-json",
            str(prereqs["provenance"]),
            "--condition-flip-json",
            str(prereqs["condition_flip"]),
            "--teacher-student-gap-json",
            str(prereqs["teacher_gap"]),
            "--action-telemetry-json",
            str(prereqs["action_telemetry"]),
            "--teacher-reachability-json",
            str(prereqs["teacher_reachability"]),
            "--modality-config-path",
            str(prereqs["modality_config"]),
            "--branch-manifest-path",
            str(prereqs["branch_manifest"]),
            "--controller-audit-json",
            str(prereqs["controller_audit"]),
            "--task2-preflight-evidence-json",
            str(prereqs["task2_preflight"]),
        ]
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    return exit_code, json.loads(captured.out)


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_p_ladder_new_embodiment.main(["--help"])
    assert exc_info.value.code == 0


def test_p0_p1_p2_materialize_internal_only_artifacts_with_stable_branch_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prereqs = _build_prerequisite_artifacts(tmp_path)
    output_root = tmp_path / "artifacts" / "new_embodiment" / "p"

    manifests: dict[str, dict[str, Any]] = {}
    scorecards: dict[str, dict[str, Any]] = {}
    for rung in ("P0", "P1", "P2"):
        exit_code, payload = _run_cli(
            rung=rung,
            output_root=output_root,
            prereqs=prereqs,
            capsys=capsys,
        )
        manifest_path = output_root / rung / "manifest.json"
        scorecard_path = output_root / rung / "scorecard.json"
        manifests[rung] = _read_json(manifest_path)
        scorecards[rung] = _read_json(scorecard_path)

        assert exit_code == 0
        assert payload["status"] == "PASS"
        assert manifests[rung]["public_anchor_comparable"] is False
        assert scorecards[rung]["public_anchor_comparable"] is False
        assert manifests[rung]["branch_scope"] == "branch_internal_only"
        assert manifests[rung]["preflight_prerequisite_proof"]["status"] == "PASS"
        assert (
            scorecards[rung]["preflight_prerequisite_proof"]["env_resolution_ok"]
            is True
        )
        assert scorecards[rung]["branch_scope"] == "branch_internal_only"
        assert scorecards[rung]["comparability"]["public_anchor_comparable"] is False
        assert scorecards[rung]["branch_contract_validation"]["status"] == "PASS"
        assert scorecards[rung]["branch_contract_stability"]["status"] == "PASS"
        assert scorecards[rung]["manifest_path"] == str(manifest_path)
        assert manifests[rung]["scorecard_path"] == str(scorecard_path)
        for key in (
            "modality_config_path",
            "normalization_source",
            "branch_manifest_hash",
            "condition_flip_delta",
            "teacher_student_gap_delta",
        ):
            assert key in scorecards[rung]
            assert key in manifests[rung]

    assert (
        manifests["P0"]["branch_manifest_hash"]
        == manifests["P1"]["branch_manifest_hash"]
        == manifests["P2"]["branch_manifest_hash"]
    )
    assert (
        manifests["P0"]["modality_config_path"]
        == manifests["P1"]["modality_config_path"]
        == manifests["P2"]["modality_config_path"]
    )
    assert (
        manifests["P0"]["normalization_source_signature_sha256"]
        == manifests["P1"]["normalization_source_signature_sha256"]
        == manifests["P2"]["normalization_source_signature_sha256"]
    )
    assert (
        manifests["P0"]["frozen_data_surface"] == manifests["P1"]["frozen_data_surface"]
    )
    assert (
        manifests["P1"]["frozen_data_surface"] == manifests["P2"]["frozen_data_surface"]
    )


@pytest.mark.parametrize(
    ("drift_kind", "modality_config_name", "normalization_override", "expected_reason"),
    [
        (
            "config",
            "alternate_modality_config.json",
            None,
            "modality_config_drift_blocks_comparison",
        ),
        (
            "normalization",
            "modality_config.json",
            {
                "owner": "NEW_EMBODIMENT branch-local statistics",
                "policy": "branch_specific_stats_required_no_cross_branch_reuse_v2",
                "cross_branch_reuse_allowed": False,
                "source_kind": "repo_local_branch_contract",
                "shipped_stats_artifact_path": None,
                "provenance_complete": True,
                "note": "drifted normalization for test",
            },
            "normalization_source_drift_blocks_comparison",
        ),
    ],
)
def test_branch_contract_drift_blocks_rung_comparison(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    drift_kind: str,
    modality_config_name: str,
    normalization_override: dict[str, Any] | None,
    expected_reason: str,
) -> None:
    baseline_prereqs = _build_prerequisite_artifacts(tmp_path / "baseline")
    output_root = tmp_path / "artifacts" / "new_embodiment" / "p"

    baseline_exit_code, _ = _run_cli(
        rung="P0",
        output_root=output_root,
        prereqs=baseline_prereqs,
        capsys=capsys,
    )
    assert baseline_exit_code == 0

    drifted_prereqs = _build_prerequisite_artifacts(
        tmp_path / drift_kind,
        modality_config_name=modality_config_name,
        normalization_override=normalization_override,
    )
    exit_code, payload = _run_cli(
        rung="P1",
        output_root=output_root,
        prereqs=drifted_prereqs,
        capsys=capsys,
    )
    manifest = _read_json(output_root / "P1" / "manifest.json")
    scorecard = _read_json(output_root / "P1" / "scorecard.json")

    assert exit_code == 1
    assert payload["status"] == "BLOCK"
    assert scorecard["status"] == "BLOCK"
    assert manifest["status"] == "BLOCK"
    assert expected_reason in scorecard["blocking_reasons"]
    assert scorecard["branch_contract_stability"]["status"] == "BLOCK"
    assert expected_reason in scorecard["branch_contract_stability"]["blocking_reasons"]


def test_p3_blocks_when_no_positive_slope_exists(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prereqs = _build_prerequisite_artifacts(tmp_path)
    output_root = tmp_path / "artifacts" / "new_embodiment" / "p"

    for rung in ("P0", "P1", "P2"):
        exit_code, _ = _run_cli(
            rung=rung,
            output_root=output_root,
            prereqs=prereqs,
            capsys=capsys,
        )
        assert exit_code == 0

    exit_code, payload = _run_cli(
        rung="P3",
        output_root=output_root,
        prereqs=prereqs,
        capsys=capsys,
    )
    manifest = _read_json(output_root / "P3" / "manifest.json")
    scorecard = _read_json(output_root / "P3" / "scorecard.json")

    assert exit_code == 1
    assert payload["status"] == "BLOCK"
    assert scorecard["status"] == "BLOCK"
    assert scorecard["promotion_status"] == "BLOCK"
    assert "positive_slope_required_for_p3" in scorecard["blocking_reasons"]
    assert scorecard["p3_gate"]["positive_slope_detected"] is False
    assert scorecard["p3_gate"]["qualifying_rungs"] == []
    assert manifest["status"] == "BLOCK"
    assert manifest["p3_gate"]["blocking_reasons"] == ["positive_slope_required_for_p3"]
