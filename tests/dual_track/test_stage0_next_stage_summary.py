from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.run import stage0_artifact_validator as stage0


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _summary() -> dict[str, object]:
    return {
        "schema_version": "dual_track_next_stage_summary_v1",
        "run_id": "stage0_gated_next_gate_20260424T064818Z",
        "gr00t": {
            "stage0": {"status": "BLOCK"},
            "gpu1_min_repro": {"status": "SKIPPED"},
            "p4_refresh": {"status": "SKIPPED"},
            "p5_gate": {"status": "SKIPPED"},
        },
        "openpi": {
            "stage0": {"status": "BLOCK"},
            "gpu2_benchmark_candidate": {"status": "SKIPPED"},
            "runtime_level": "p2_overfit_or_tiny_update_pass",
            "benchmark_success_claimed": False,
        },
        "completion_claim_allowed": False,
        "forbidden_inferences": [
            "GR00T min repro PASS != P5 eligibility",
            "OpenPI P2 runtime PASS != benchmark PASS",
            "worker message != machine-checkable ready artifact",
        ],
        "validator_outputs": [],
    }


def test_final_summary_rejects_benchmark_overclaim_without_candidate_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage0.RUN_ID
    summary = _summary()
    openpi = summary["openpi"]
    assert isinstance(openpi, dict)
    openpi["benchmark_success_claimed"] = True
    summary_path = _write_json(run_root / "verifier" / "dual_track_next_stage_summary.draft.json", summary)

    with pytest.raises(stage0.Stage0ValidationError, match="openpi benchmark claim requires"):
        stage0.validate_next_stage_summary_draft(summary_path, run_root=run_root)


def test_final_validator_report_passes_with_conservative_summary_and_evidence(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage0.RUN_ID
    verifier_root = run_root / "verifier"
    _write_json(verifier_root / "dual_track_next_stage_summary.draft.json", _summary())
    (verifier_root / "focused_pytest.log").write_text("focused pytest PASS\n", encoding="utf-8")
    _write_json(verifier_root / "resource_boundary_report.json", {"status": "PASS", "used_gpus": []})

    report = stage0.write_final_validator_report(
        run_root,
        runtime_log_root=tmp_path / "runtime_logs" / stage0.RUN_ID,
        created_at_utc="2026-04-24T00:00:00Z",
    )

    assert report["status"] == "PASS"
    assert report["blocking_reasons"] == []
    assert "verifier/dual_track_next_stage_summary.draft.json" in report["validated_artifacts"]


def test_final_validator_accepts_stage0_command_list_resource_lease(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage0.RUN_ID
    verifier_root = run_root / "verifier"
    _write_json(verifier_root / "dual_track_next_stage_summary.draft.json", _summary())
    (verifier_root / "focused_pytest.log").write_text("pytest_exit=0\n", encoding="utf-8")
    _write_json(verifier_root / "resource_boundary_report.json", {"status": "PASS", "used_gpus": []})
    _write_json(
        run_root / "openpi" / "gpu2_benchmark_candidate" / "resource_lease.json",
        {
            "schema_version": "resource_lease_v1",
            "lane": "openpi",
            "gpu": 2,
            "worker": "worker-3",
            "created_at_utc": "2026-04-24T00:00:00Z",
            "returncode": 0,
            "runtime_log_dir": "agent/runtime_logs/stage0/openpi",
            "forbidden_gpus_visible": False,
            "sudo_used": False,
            "direct_privileged_escalation_used": False,
            "commands": [
                {
                    "command": ["python", "-c", "print('ok')"],
                    "command_shell": "env CUDA_VISIBLE_DEVICES=2 python -c 'print(ok)'",
                    "env": {"CUDA_VISIBLE_DEVICES": "2"},
                    "returncode": 0,
                    "timed_out": False,
                }
            ],
        },
    )

    report = stage0.write_final_validator_report(
        run_root,
        runtime_log_root=tmp_path / "runtime_logs" / stage0.RUN_ID,
        created_at_utc="2026-04-24T00:00:00Z",
    )

    assert report["status"] == "PASS"
    assert (
        "openpi/gpu2_benchmark_candidate/resource_lease.json"
        in report["validated_artifacts"]
    )


def test_final_validator_accepts_stage_specific_gr00t_resource_lease_lane(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage0.RUN_ID
    verifier_root = run_root / "verifier"
    _write_json(verifier_root / "dual_track_next_stage_summary.draft.json", _summary())
    (verifier_root / "focused_pytest.log").write_text("pytest_exit=0\n", encoding="utf-8")
    _write_json(verifier_root / "resource_boundary_report.json", {"status": "PASS", "used_gpus": []})
    _write_json(
        run_root / "gr00t" / "gpu1_min_repro" / "resource_lease.json",
        {
            "schema_version": "resource_lease_v1",
            "lane": "gr00t_gpu1_min_repro",
            "gpu": 1,
            "worker": "worker-2",
            "command": "timeout 60 env CUDA_VISIBLE_DEVICES=1 python repro.py",
            "started_at_utc": "2026-04-24T00:00:00Z",
            "ended_at_utc": "2026-04-24T00:01:00Z",
            "returncode": 0,
            "timeout_s": 60,
            "runtime_log": "agent/runtime_logs/stage0/gr00t.log",
            "artifacts": [],
            "forbidden_gpus_visible": False,
            "sudo_used": False,
        },
    )

    report = stage0.write_final_validator_report(
        run_root,
        runtime_log_root=tmp_path / "runtime_logs" / stage0.RUN_ID,
        created_at_utc="2026-04-24T00:00:00Z",
    )

    assert report["status"] == "PASS"
    assert "gr00t/gpu1_min_repro/resource_lease.json" in report["validated_artifacts"]


def test_structured_runtime_log_boundary_ignores_non_command_prose(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage0.RUN_ID
    runtime_root = tmp_path / "agent" / "runtime_logs" / stage0.RUN_ID
    verifier_root = run_root / "verifier"
    _write_json(verifier_root / "dual_track_next_stage_summary.draft.json", _summary())
    (verifier_root / "focused_pytest.log").write_text("pytest_exit=0\n", encoding="utf-8")
    _write_json(verifier_root / "resource_boundary_report.json", {"status": "PASS", "used_gpus": []})
    _write_json(
        runtime_root / "gr00t" / "gpu1_min_repro" / "verification_summary.json",
        {
            "status": "PASS",
            "evidence": "no GPU0/GPU3/sudo tokens in committed W2 runtime log",
            "command": "env CUDA_VISIBLE_DEVICES=1 python repro.py",
        },
    )

    report = stage0.write_final_validator_report(
        run_root,
        runtime_log_root=runtime_root,
        created_at_utc="2026-04-24T00:00:00Z",
    )

    assert report["status"] == "PASS"


def test_final_validator_blocks_failed_focused_pytest_log(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage0.RUN_ID
    verifier_root = run_root / "verifier"
    _write_json(verifier_root / "dual_track_next_stage_summary.draft.json", _summary())
    (verifier_root / "focused_pytest.log").write_text(
        "focused pytest output\npytest_exit=1\n",
        encoding="utf-8",
    )
    _write_json(verifier_root / "resource_boundary_report.json", {"status": "PASS"})

    report = stage0.write_final_validator_report(
        run_root,
        runtime_log_root=tmp_path / "runtime_logs" / stage0.RUN_ID,
        created_at_utc="2026-04-24T00:00:00Z",
    )

    assert report["status"] == "BLOCK"
    assert any("pytest_exit=1" in reason for reason in report["blocking_reasons"])


def test_final_validator_blocks_draft_summary_blocking_reasons(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage0.RUN_ID
    verifier_root = run_root / "verifier"
    summary = _summary()
    summary["blocking_reasons"] = ["runtime follow-up still pending"]
    _write_json(verifier_root / "dual_track_next_stage_summary.draft.json", summary)
    (verifier_root / "focused_pytest.log").write_text("focused pytest PASS\n", encoding="utf-8")
    _write_json(verifier_root / "resource_boundary_report.json", {"status": "PASS"})

    report = stage0.write_final_validator_report(
        run_root,
        runtime_log_root=tmp_path / "runtime_logs" / stage0.RUN_ID,
        created_at_utc="2026-04-24T00:00:00Z",
    )

    assert report["status"] == "BLOCK"
    assert any(
        "draft summary blocker: runtime follow-up still pending" == reason
        for reason in report["blocking_reasons"]
    )
