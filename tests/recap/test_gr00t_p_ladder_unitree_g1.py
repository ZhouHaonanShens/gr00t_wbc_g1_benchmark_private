from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_ladder_policy_gate
from work.recap.scripts import gr00t_p_ladder_unitree_g1


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


def _build_prerequisite_artifacts(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "prereqs"
    gate_path = root / "p_ladder_policy_gate_unitree_g1.json"
    dual_branch_path = root / "dual_branch_scorecard.json"
    provenance_path = root / "checkpoint_provenance_report.json"
    condition_flip_path = root / "condition_flip_scorecard_unitree_g1.json"
    teacher_gap_path = root / "teacher_student_gap_scorecard_unitree_g1.json"
    action_telemetry_path = root / "action_chain_telemetry_unitree_g1.json"
    teacher_reachability_path = root / "teacher_reachability_gate_unitree_g1.json"
    task2_preflight_evidence = _write_task2_preflight_evidence(root)

    gate_payload = gr00t_ladder_policy_gate.build_ladder_policy_gate(
        branch="UNITREE_G1",
        axis="P",
        output_path=gate_path,
    )
    _write_json(gate_path, gate_payload)

    _write_json(
        dual_branch_path,
        {
            "artifact_kind": "gr00t_dual_branch_scorecard",
            "allow_p_ladder": {"unitree_g1": True},
            "branches": [
                {
                    "branch_key": "unitree_g1",
                    "branch_scope": "official_public_anchor_line",
                    "public_anchor_status": {
                        "summary": {
                            "success_count": 5,
                            "success_rate": 0.5,
                            "systemic_break_flags": [],
                        }
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
            "response_ratio": {
                "min_ratio_across_semantic_flips": 0.18656443,
            },
            "pass_fail_gate": "PASS",
            "paired_scene_id": "unitree_g1::S_drop",
            "report_signature_sha256": "condition-flip-signature",
        },
    )
    _write_json(
        teacher_gap_path,
        {
            "artifact_kind": "gr00t_teacher_student_gap_scorecard",
            "status": "ALLOW",
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
            "branch": "UNITREE_G1",
            "controller_absorbed_groups": ["left_arm"],
            "model_insensitive_groups": ["right_hand"],
            "zero_motion_groups": ["right_hand"],
            "report_signature_sha256": "action-telemetry-signature",
        },
    )
    _write_json(
        teacher_reachability_path,
        {
            "artifact_kind": "gr00t_teacher_reachability_gate",
            "allow_formal_ladders": True,
            "status": "ALLOW",
            "reachable_scene_ids": ["unitree_g1::S_drop", "unitree_g1::S_lost"],
            "scene_pool_status": "formal_teacher_replay_reachable_pool_materialized",
            "scene_pool": {
                "scene_rows": [
                    {
                        "family": "S_drop",
                        "scene_id": "unitree_g1::S_drop",
                        "included_in_formal_scene_pool": True,
                    },
                    {
                        "family": "S_lost",
                        "scene_id": "unitree_g1::S_lost",
                        "included_in_formal_scene_pool": True,
                    },
                ]
            },
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
    }


def _run_cli(
    *,
    rung: str,
    output_root: Path,
    prereqs: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, dict[str, Any]]:
    exit_code = gr00t_p_ladder_unitree_g1.main(
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
            "--task2-preflight-evidence-json",
            str(prereqs["task2_preflight"]),
        ]
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    return exit_code, json.loads(captured.out)


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_p_ladder_unitree_g1.main(["--help"])
    assert exc_info.value.code == 0


def test_p0_p1_p2_materialize_artifacts_and_keep_data_surface_frozen(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prereqs = _build_prerequisite_artifacts(tmp_path)
    output_root = tmp_path / "artifacts" / "unitree_g1" / "p"

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
        assert manifest_path.is_file()
        assert scorecard_path.is_file()
        assert scorecards[rung]["manifest_path"] == str(manifest_path)
        assert manifests[rung]["scorecard_path"] == str(scorecard_path)
        assert manifests[rung]["preflight_prerequisite_proof"]["status"] == "PASS"
        assert (
            scorecards[rung]["preflight_prerequisite_proof"]["policy_ping_ok"] is True
        )
        for key in (
            "success_rate",
            "condition_flip_delta",
            "teacher_student_gap_delta",
            "action_chain_delta",
            "provenance_status",
        ):
            assert key in scorecards[rung]

    assert (
        manifests["P0"]["frozen_data_surface"] == manifests["P1"]["frozen_data_surface"]
    )
    assert (
        manifests["P1"]["frozen_data_surface"] == manifests["P2"]["frozen_data_surface"]
    )
    assert (
        manifests["P0"]["frozen_data_surface"]["dataset"]["dataset_fingerprint"]
        == manifests["P1"]["frozen_data_surface"]["dataset"]["dataset_fingerprint"]
        == manifests["P2"]["frozen_data_surface"]["dataset"]["dataset_fingerprint"]
    )
    assert scorecards["P0"]["comparability"]["observed_difference_paths"] == []
    assert scorecards["P1"]["comparability"]["observed_difference_paths"] == [
        "training.parameter_update.tune_visual",
        "training.parameter_update.visual_unfreeze",
    ]
    assert scorecards["P2"]["comparability"]["observed_difference_paths"] == [
        "training.parameter_update.lora_enabled",
        "training.parameter_update.lora_rank",
        "training.parameter_update.selective_unfreeze_modules",
        "training.parameter_update.tune_llm",
        "training.parameter_update.tune_visual",
        "training.parameter_update.visual_unfreeze",
    ]


def test_p3_blocks_when_no_positive_slope_exists(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prereqs = _build_prerequisite_artifacts(tmp_path)
    output_root = tmp_path / "artifacts" / "unitree_g1" / "p"

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


def test_build_rung_report_uses_scorecard_manifest_linkage_and_frozen_protocol(
    tmp_path: Path,
) -> None:
    prereqs = gr00t_p_ladder_unitree_g1.load_prerequisites(
        p_ladder_policy_gate_json=_build_prerequisite_artifacts(tmp_path)["gate"],
        dual_branch_scorecard_json=_build_prerequisite_artifacts(tmp_path)[
            "dual_branch"
        ],
        checkpoint_provenance_json=_build_prerequisite_artifacts(tmp_path)[
            "provenance"
        ],
        condition_flip_json=_build_prerequisite_artifacts(tmp_path)["condition_flip"],
        teacher_student_gap_json=_build_prerequisite_artifacts(tmp_path)["teacher_gap"],
        action_telemetry_json=_build_prerequisite_artifacts(tmp_path)[
            "action_telemetry"
        ],
        teacher_reachability_json=_build_prerequisite_artifacts(tmp_path)[
            "teacher_reachability"
        ],
        task2_preflight_evidence_json=_build_prerequisite_artifacts(tmp_path)[
            "task2_preflight"
        ],
    )
    output_root = tmp_path / "artifacts" / "unitree_g1" / "p"

    report = gr00t_p_ladder_unitree_g1.build_rung_report(
        rung="P1",
        prerequisites=prereqs,
        output_root=output_root,
    )
    manifest = report["manifest"]
    scorecard = report["scorecard"]

    assert scorecard["manifest_path"] == str(output_root / "P1" / "manifest.json")
    assert manifest["scorecard_path"] == str(output_root / "P1" / "scorecard.json")
    assert scorecard["frozen_formal_protocol"]["seed_values"] == list(
        gr00t_p_ladder_unitree_g1.gr00t_eval_contract_gate.DEFAULT_FORMAL_SEED_VALUES
    )
    assert (
        manifest["scene_pool"]["scene_pool_identifier"]
        == gr00t_p_ladder_unitree_g1.gr00t_eval_contract_gate.DEFAULT_SCENE_POOL_IDENTIFIER
    )
