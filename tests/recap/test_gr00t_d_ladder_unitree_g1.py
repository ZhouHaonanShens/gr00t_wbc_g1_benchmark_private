from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_d_ladder_policy_gate
from work.recap.scripts import gr00t_dual_branch_scorecard
from work.recap.scripts import gr00t_ladder_policy_gate


_MODULE_SPEC = importlib.util.spec_from_file_location(
    "gr00t_d_ladder_unitree_g1",
    REPO_ROOT / "work" / "recap" / "scripts" / "gr00t_d_ladder_unitree_g1.py",
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
gr00t_d_ladder_unitree_g1 = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(gr00t_d_ladder_unitree_g1)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
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


def _admission_inputs(
    rung: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_ids = [
        spec.dataset_id
        for spec in gr00t_d_ladder_policy_gate._included_dataset_specs(rung)
    ]
    dataset_fingerprints: dict[str, Any] = {}
    dataset_provenance: dict[str, Any] = {}
    normalization_records: dict[str, Any] = {}
    for index, dataset_id in enumerate(dataset_ids):
        spec = gr00t_d_ladder_policy_gate._dataset_spec_by_id(dataset_id)
        dataset_fingerprints[dataset_id] = {
            "dataset_fingerprint": f"fingerprint-{rung.lower()}-{index}",
            "fingerprint_source": "pytest_fixture",
        }
        dataset_provenance[dataset_id] = {
            "source_registry_id": dataset_id,
            "local_dataset_path": f"/tmp/{dataset_id.replace('/', '_')}",
            "controller_family": spec.controller_family,
            "provenance_complete": True,
            "guardrails_satisfied": True,
        }
        normalization_records[dataset_id] = {
            "stats_fingerprint": f"stats-{rung.lower()}-{index}",
            "hidden_stats_fingerprint": f"stats-{rung.lower()}-{index}",
            "normalization_owner": "unitree_g1_branch_local",
            "explicit_stats_policy": "branch_scoped_stats",
            "cross_dataset_reuse_declared": False,
            "regenerated_for_branch": True,
        }
    return dataset_fingerprints, dataset_provenance, normalization_records


def _build_prerequisite_artifacts(
    tmp_path: Path,
    *,
    evidence_rung: str,
) -> dict[str, Path]:
    root = tmp_path / "prereqs"
    ladder_policy_path = root / "d_ladder_policy_gate_unitree_g1_task12.json"
    dataset_admission_dir = root / "task15"
    dual_branch_path = root / "dual_branch_scorecard.json"
    provenance_path = root / "checkpoint_provenance_report.json"
    condition_flip_path = root / "condition_flip_scorecard_unitree_g1.json"
    teacher_gap_path = root / "teacher_student_gap_scorecard_unitree_g1.json"
    action_telemetry_path = root / "action_chain_telemetry_unitree_g1.json"
    teacher_reachability_path = root / "teacher_reachability_gate_unitree_g1.json"
    dataset_fingerprints_path = root / "dataset_fingerprints.json"
    dataset_provenance_path = root / "dataset_provenance.json"
    normalization_records_path = root / "normalization_records.json"
    task2_preflight_evidence = _write_task2_preflight_evidence(root)

    ladder_policy_payload = gr00t_ladder_policy_gate.build_ladder_policy_gate(
        branch="UNITREE_G1",
        axis="D",
        output_path=ladder_policy_path,
    )
    _ = _write_json(ladder_policy_path, ladder_policy_payload)

    _ = gr00t_d_ladder_policy_gate.materialize_branch_gate(
        branch="UNITREE_G1",
        output_dir=dataset_admission_dir,
    )
    dataset_admission_path = (
        dataset_admission_dir
        / gr00t_d_ladder_policy_gate.GATE_JSON_NAME_BY_BRANCH["UNITREE_G1"]
    )

    _ = _write_json(
        dual_branch_path,
        {
            "artifact_kind": gr00t_dual_branch_scorecard.REPORT_ARTIFACT_KIND,
            "allow_d_ladder": {"unitree_g1": True},
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
    _ = _write_json(
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
    _ = _write_json(
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
    _ = _write_json(
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
            ],
            "report_signature_sha256": "teacher-gap-signature",
        },
    )
    _ = _write_json(
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
    _ = _write_json(
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

    dataset_fingerprints, dataset_provenance, normalization_records = _admission_inputs(
        evidence_rung
    )
    _ = _write_json(dataset_fingerprints_path, dataset_fingerprints)
    _ = _write_json(dataset_provenance_path, dataset_provenance)
    _ = _write_json(normalization_records_path, normalization_records)

    return {
        "ladder_policy": ladder_policy_path,
        "dataset_admission": dataset_admission_path,
        "dual_branch": dual_branch_path,
        "provenance": provenance_path,
        "condition_flip": condition_flip_path,
        "teacher_gap": teacher_gap_path,
        "action_telemetry": action_telemetry_path,
        "teacher_reachability": teacher_reachability_path,
        "dataset_fingerprints": dataset_fingerprints_path,
        "dataset_provenance": dataset_provenance_path,
        "normalization_records": normalization_records_path,
        "task2_preflight": task2_preflight_evidence,
    }


def _run_cli(
    *,
    rung: str,
    output_root: Path,
    prereqs: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, dict[str, Any]]:
    exit_code = gr00t_d_ladder_unitree_g1.main(
        [
            "--rung",
            rung,
            "--output-root",
            str(output_root),
            "--ladder-policy-gate-json",
            str(prereqs["ladder_policy"]),
            "--dataset-admission-gate-json",
            str(prereqs["dataset_admission"]),
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
            "--dataset-fingerprints-json",
            str(prereqs["dataset_fingerprints"]),
            "--dataset-provenance-json",
            str(prereqs["dataset_provenance"]),
            "--normalization-records-json",
            str(prereqs["normalization_records"]),
            "--task2-preflight-evidence-json",
            str(prereqs["task2_preflight"]),
        ]
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    return exit_code, json.loads(captured.out)


def _run_default_cli(
    *,
    rung: str,
    prereqs: dict[str, Path],
    output_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, dict[str, Any]]:
    monkeypatch.setattr(
        gr00t_d_ladder_unitree_g1,
        "DEFAULT_OUTPUT_ROOT",
        output_root,
    )
    monkeypatch.setattr(
        gr00t_d_ladder_unitree_g1,
        "DEFAULT_LADDER_POLICY_GATE_JSON",
        prereqs["ladder_policy"],
    )
    monkeypatch.setattr(
        gr00t_d_ladder_unitree_g1,
        "DEFAULT_DATASET_ADMISSION_GATE_JSON",
        prereqs["dataset_admission"],
    )
    monkeypatch.setattr(
        gr00t_d_ladder_unitree_g1,
        "DEFAULT_DUAL_BRANCH_SCORECARD_JSON",
        prereqs["dual_branch"],
    )
    monkeypatch.setattr(
        gr00t_d_ladder_unitree_g1,
        "DEFAULT_CHECKPOINT_PROVENANCE_JSON",
        prereqs["provenance"],
    )
    monkeypatch.setattr(
        gr00t_d_ladder_unitree_g1,
        "DEFAULT_CONDITION_FLIP_JSON",
        prereqs["condition_flip"],
    )
    monkeypatch.setattr(
        gr00t_d_ladder_unitree_g1,
        "DEFAULT_TEACHER_STUDENT_GAP_JSON",
        prereqs["teacher_gap"],
    )
    monkeypatch.setattr(
        gr00t_d_ladder_unitree_g1,
        "DEFAULT_ACTION_TELEMETRY_JSON",
        prereqs["action_telemetry"],
    )
    monkeypatch.setattr(
        gr00t_d_ladder_unitree_g1,
        "DEFAULT_TEACHER_REACHABILITY_JSON",
        prereqs["teacher_reachability"],
    )
    monkeypatch.setattr(
        gr00t_d_ladder_unitree_g1.gr00t_p_ladder_unitree_g1,
        "DEFAULT_TASK2_PREFLIGHT_EVIDENCE_JSON",
        prereqs["task2_preflight"],
    )
    exit_code = gr00t_d_ladder_unitree_g1.main(["--rung", rung])
    captured = capsys.readouterr()
    assert captured.err == ""
    return exit_code, json.loads(captured.out)


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_d_ladder_unitree_g1.main(["--help"])
    assert exc_info.value.code == 0


def test_d0_and_d1_materialize_artifacts_with_dataset_surface_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prereqs = _build_prerequisite_artifacts(tmp_path, evidence_rung="D1")
    output_root = tmp_path / "artifacts" / "unitree_g1" / "d"

    scorecards: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for rung in ("D0", "D1"):
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
        assert manifests[rung]["scorecard_path"] == str(scorecard_path)
        assert scorecards[rung]["manifest_path"] == str(manifest_path)
        assert manifests[rung]["preflight_prerequisite_proof"]["status"] == "PASS"
        assert scorecards[rung]["preflight_prerequisite_proof"]["smoke_step_ok"] is True
        for key in (
            "dataset_mix",
            "dataset_fingerprint",
            "normalization_policy",
            "success_rate",
            "condition_flip_delta",
            "teacher_gap_delta",
            "action_chain_delta",
        ):
            assert key in scorecards[rung]

    assert scorecards["D0"]["comparability"]["observed_difference_paths"] == []
    assert scorecards["D1"]["comparability"]["observed_difference_paths"] == [
        "dataset.admission.dataset_fingerprints",
        "dataset.admission.dataset_source_ids",
        "dataset.dataset_mix",
        "dataset.normalization.explicit_diff_reason",
        "dataset.normalization.stats_fingerprint",
    ]
    assert scorecards["D0"]["execution_disposition"] == (
        gr00t_d_ladder_unitree_g1.PASS_DISPOSITION
    )
    assert scorecards["D1"]["execution_disposition"] == (
        gr00t_d_ladder_unitree_g1.PASS_DISPOSITION
    )
    assert scorecards["D1"]["dataset_admission"]["admission_status"] == "PASS"
    assert scorecards["D1"]["dataset_fingerprint_status"] == "COMPLETE"


def test_default_cli_d1_happy_path_uses_auto_materialized_admission_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prereqs = _build_prerequisite_artifacts(tmp_path, evidence_rung="D1")
    output_root = tmp_path / "default_cli_artifacts" / "unitree_g1" / "d"

    exit_code, payload = _run_default_cli(
        rung="D1",
        prereqs=prereqs,
        output_root=output_root,
        monkeypatch=monkeypatch,
        capsys=capsys,
    )
    manifest = _read_json(output_root / "D1" / "manifest.json")
    scorecard = _read_json(output_root / "D1" / "scorecard.json")

    assert exit_code == 0
    assert payload["status"] == "PASS"
    assert (
        payload["execution_disposition"] == gr00t_d_ladder_unitree_g1.PASS_DISPOSITION
    )
    assert manifest["dataset_surface"]["dataset_fingerprint_status"] == "COMPLETE"
    assert scorecard["dataset_admission"]["admission_status"] == "PASS"
    assert (
        scorecard["source_artifacts"]["dataset_fingerprints"]["source"]
        == "auto_default"
    )
    assert (
        scorecard["source_artifacts"]["dataset_fingerprints"]["auto_generated_policy"]
        == gr00t_d_ladder_unitree_g1.DEFAULT_AUTO_EVIDENCE_POLICY
    )
    assert (
        manifest["admission_evidence_surface"]["dataset_records"][
            "nvidia/Arena-G1-Loco-Manipulation-Task"
        ]["dataset_provenance"]["auto_generated"]
        is True
    )


def test_model_fingerprint_stays_stable_while_dataset_fingerprint_changes(
    tmp_path: Path,
) -> None:
    prereq_paths = _build_prerequisite_artifacts(tmp_path, evidence_rung="D1")
    prereqs = gr00t_d_ladder_unitree_g1.load_prerequisites(
        ladder_policy_gate_json=prereq_paths["ladder_policy"],
        dataset_admission_gate_json=prereq_paths["dataset_admission"],
        dual_branch_scorecard_json=prereq_paths["dual_branch"],
        checkpoint_provenance_json=prereq_paths["provenance"],
        condition_flip_json=prereq_paths["condition_flip"],
        teacher_student_gap_json=prereq_paths["teacher_gap"],
        action_telemetry_json=prereq_paths["action_telemetry"],
        teacher_reachability_json=prereq_paths["teacher_reachability"],
        dataset_fingerprints_json=prereq_paths["dataset_fingerprints"],
        dataset_provenance_json=prereq_paths["dataset_provenance"],
        normalization_records_json=prereq_paths["normalization_records"],
        task2_preflight_evidence_json=prereq_paths["task2_preflight"],
    )
    output_root = tmp_path / "artifacts" / "unitree_g1" / "d"

    d0_report = gr00t_d_ladder_unitree_g1.build_rung_report(
        rung="D0",
        prerequisites=prereqs,
        output_root=output_root,
    )
    d1_report = gr00t_d_ladder_unitree_g1.build_rung_report(
        rung="D1",
        prerequisites=prereqs,
        output_root=output_root,
    )

    d0_manifest = d0_report["manifest"]
    d1_manifest = d1_report["manifest"]

    assert (
        d0_manifest["model_surface"]["model_fingerprint"]
        == d1_manifest["model_surface"]["model_fingerprint"]
    )
    assert (
        d0_manifest["dataset_surface"]["dataset_fingerprint"]
        != d1_manifest["dataset_surface"]["dataset_fingerprint"]
    )
    assert d0_manifest["model_axis_frozen"] is True
    assert d1_manifest["model_axis_frozen"] is True


def test_d4_blocks_and_redirects_to_new_embodiment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prereqs = _build_prerequisite_artifacts(tmp_path, evidence_rung="D4")
    output_root = tmp_path / "artifacts" / "unitree_g1" / "d"

    exit_code, payload = _run_cli(
        rung="D4",
        output_root=output_root,
        prereqs=prereqs,
        capsys=capsys,
    )
    manifest = _read_json(output_root / "D4" / "manifest.json")
    scorecard = _read_json(output_root / "D4" / "scorecard.json")

    assert exit_code == 1
    assert payload["status"] == "BLOCK"
    assert payload["execution_disposition"] == (
        gr00t_d_ladder_unitree_g1.REDIRECT_DISPOSITION
    )
    assert payload["redirect_branch"] == "NEW_EMBODIMENT"
    assert manifest["status"] == "BLOCK"
    assert scorecard["status"] == "BLOCK"
    assert (
        manifest["execution_disposition"]
        == gr00t_d_ladder_unitree_g1.REDIRECT_DISPOSITION
    )
    assert scorecard["redirect_branch"] == "NEW_EMBODIMENT"
    assert "unitree_g1_d4_branch_only_redirect" in scorecard["blocking_reasons"]
    assert (
        "branch_only_dataset:LightwheelAI/Lightwheel-Tasks-G1-Controller"
        in scorecard["dataset_admission"]["reason_codes"]
    )
    assert (
        "redirect_to_new_embodiment:LightwheelAI/Lightwheel-Tasks-G1-Controller"
        in scorecard["dataset_admission"]["reason_codes"]
    )


def test_default_cli_d4_still_blocks_with_redirect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prereqs = _build_prerequisite_artifacts(tmp_path, evidence_rung="D4")
    output_root = tmp_path / "default_cli_artifacts" / "unitree_g1" / "d"

    exit_code, payload = _run_default_cli(
        rung="D4",
        prereqs=prereqs,
        output_root=output_root,
        monkeypatch=monkeypatch,
        capsys=capsys,
    )
    scorecard = _read_json(output_root / "D4" / "scorecard.json")

    assert exit_code == 1
    assert payload["status"] == "BLOCK"
    assert (
        payload["execution_disposition"]
        == gr00t_d_ladder_unitree_g1.REDIRECT_DISPOSITION
    )
    assert payload["redirect_branch"] == "NEW_EMBODIMENT"
    assert scorecard["dataset_admission"]["admission_status"] == "BLOCK"
    assert "unitree_g1_d4_branch_only_redirect" in scorecard["blocking_reasons"]
