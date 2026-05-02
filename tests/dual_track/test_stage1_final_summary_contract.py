from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.run import stage1_artifact_validator as stage1


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _base_summary() -> dict[str, object]:
    return {
        "schema_version": stage1.SUMMARY_SCHEMA,
        "run_id": stage1.RUN_ID,
        "created_at_utc": "2026-04-24T00:00:00Z",
        "gr00t": {
            "p4_refresh": {
                "status": "BLOCK",
                "clean_p4_gate": False,
                "p5_formal_10ep_eligible": False,
                "blocking_reasons": ["p4 pending"],
                "artifacts": [],
            },
            "p5_gate": {
                "decision": "SKIPPED",
                "status": "SKIPPED",
                "blocking_reasons": ["p4 pending"],
                "artifacts": [],
            },
        },
        "openpi": {
            "benchmark_sweep": {
                "status": "BLOCK",
                "benchmark_claim_allowed_by_final_validator": False,
                "blocking_reasons": ["strong v21 pending"],
                "artifacts": [],
            },
            "formal_benchmark_materialized": False,
            "benchmark_success_claimed": False,
            "recap_validated_on_desaturated_eval": False,
            "eligible_for_state_side_v22": False,
        },
        "validator_migration": {
            "status": "BLOCK",
            "active_referee": "agent/run/stage0_artifact_validator.py",
            "parity_passed": False,
            "blocking_reasons": ["parity pending"],
            "artifacts": [],
        },
        "completion_claim_allowed": False,
        "blocking_reasons": ["p4 pending", "strong v21 pending", "parity pending"],
        "forbidden_inferences": list(stage1.FORBIDDEN_INFERENCES),
        "validator_outputs": [],
    }


def test_stage1_summary_materializes_block_until_lane_artifacts_exist(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage1.RUN_ID
    runtime_root = tmp_path / "agent" / "runtime_logs" / stage1.RUN_ID

    resource_report = stage1.write_resource_boundary_report(run_root, runtime_root, created_at_utc="2026-04-24T00:00:00Z")
    git_report = stage1.write_git_hygiene_report(run_root, created_at_utc="2026-04-24T00:00:00Z")
    summary = stage1.write_stage1_summary(run_root, created_at_utc="2026-04-24T00:00:00Z")

    assert resource_report["status"] == "PASS"
    assert git_report["status"] == "PASS"
    assert summary["schema_version"] == "dual_track_stage1_summary_v1"
    assert summary["completion_claim_allowed"] is False
    assert "OpenPI smoke PASS != benchmark PASS" in summary["forbidden_inferences"]
    assert any("gr00t P4" in reason for reason in summary["blocking_reasons"])

    focused = run_root / "verifier" / "focused_pytest.log"
    focused.write_text("pytest_exit=0\n", encoding="utf-8")
    final_report = stage1.write_final_validator_report(run_root, runtime_root, created_at_utc="2026-04-24T00:00:00Z")

    assert final_report["status"] == "BLOCK"
    assert any(reason.startswith("summary blocker:") for reason in final_report["blocking_reasons"])


def test_stage1_summary_rejects_p5_run_without_clean_p4(tmp_path: Path) -> None:
    summary = _base_summary()
    gr00t = summary["gr00t"]
    assert isinstance(gr00t, dict)
    p5 = gr00t["p5_gate"]
    assert isinstance(p5, dict)
    p5["decision"] = "RUN"
    p5["status"] = "PASS"
    p5["blocking_reasons"] = []
    path = _write_json(tmp_path / "verifier" / "dual_track_stage1_summary.json", summary)

    with pytest.raises(stage1.Stage1ValidationError, match="P5 RUN requires"):
        stage1.validate_stage1_summary(path, run_root=tmp_path)


def test_final_report_blocks_when_summary_completion_claim_is_false(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage1.RUN_ID
    runtime_root = tmp_path / "agent" / "runtime_logs" / stage1.RUN_ID
    verifier = run_root / "verifier"
    summary = _base_summary()
    summary["blocking_reasons"] = []

    _write_json(verifier / "dual_track_stage1_summary.json", summary)
    _write_json(
        verifier / "resource_boundary_report.json",
        {
            "schema_version": stage1.RESOURCE_REPORT_SCHEMA,
            "run_id": stage1.RUN_ID,
            "created_at_utc": "2026-04-24T00:00:00Z",
            "status": "PASS",
            "blocking_reasons": [],
            "validated_resource_leases": [],
            "used_gpus": [],
            "forbidden_gpus_visible": False,
            "sudo_used": False,
        },
    )
    _write_json(
        verifier / "git_hygiene_report.json",
        {
            "schema_version": stage1.GIT_HYGIENE_SCHEMA,
            "run_id": stage1.RUN_ID,
            "created_at_utc": "2026-04-24T00:00:00Z",
            "status": "PASS",
            "tracked_large_artifact_candidates_over_5mib": [],
        },
    )
    (verifier / "focused_pytest.log").write_text("pytest_exit=0\n", encoding="utf-8")

    report = stage1.write_final_validator_report(
        run_root,
        runtime_root,
        created_at_utc="2026-04-24T00:00:00Z",
    )

    assert report["status"] == "BLOCK"
    assert "summary completion_claim_allowed=false" in report["blocking_reasons"]
    assert report["final_claim_language"]["completion_claim_allowed"] is False


def test_openpi_benchmark_success_requires_final_validator_allowance(tmp_path: Path) -> None:
    summary = _base_summary()
    openpi = summary["openpi"]
    assert isinstance(openpi, dict)
    openpi["formal_benchmark_materialized"] = True
    openpi["benchmark_success_claimed"] = True
    path = _write_json(tmp_path / "verifier" / "dual_track_stage1_summary.json", summary)

    with pytest.raises(stage1.Stage1ValidationError, match="benchmark_success_claimed requires"):
        stage1.validate_stage1_summary(path, run_root=tmp_path)


def test_resource_lease_blocks_forbidden_gpu_visibility(tmp_path: Path) -> None:
    lease = _write_json(
        tmp_path / "resource_lease.json",
        {
            "schema_version": "resource_lease_v1",
            "lane": "gr00t_p4_refresh",
            "worker": "worker-2",
            "gpu": 1,
            "forbidden_gpus_visible": False,
            "sudo_used": False,
            "direct_privileged_escalation_used": False,
            "returncode": 0,
            "runtime_log_dir": "agent/runtime_logs/stage1/gr00t",
            "created_at_utc": "2026-04-24T00:00:00Z",
            "command": "env CUDA_VISIBLE_DEVICES=0,1 python run.py",
        },
    )

    with pytest.raises(stage1.Stage1ValidationError, match="GPU0/GPU3"):
        stage1.validate_resource_lease(lease)


def test_resource_lease_blocks_direct_sudo_command(tmp_path: Path) -> None:
    lease = _write_json(
        tmp_path / "resource_lease.json",
        {
            "schema_version": "resource_lease_v1",
            "lane": "openpi_benchmark_sweep",
            "worker": "worker-4",
            "gpu": 2,
            "forbidden_gpus_visible": False,
            "sudo_used": False,
            "direct_privileged_escalation_used": False,
            "returncode": 0,
            "runtime_log_dir": "agent/runtime_logs/stage1/openpi",
            "created_at_utc": "2026-04-24T00:00:00Z",
            "command": "sudo python run.py",
        },
    )

    with pytest.raises(stage1.Stage1ValidationError, match="sudo"):
        stage1.validate_resource_lease(lease)


def test_resource_boundary_ignores_validator_migration_negative_fixture(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage1.RUN_ID
    bad_fixture = run_root / (
        "validator_migration/old_validator_outputs/resource_boundary_negative/"
        "agent/artifacts/stage0_gated_next_gate_20260424T064818Z/gr00t/"
        "gpu1_min_repro/resource_lease.json"
    )
    _write_json(
        bad_fixture,
        {
            "schema_version": "resource_lease_v1",
            "lane": "gr00t_gpu1_min_repro",
            "worker": "worker-5",
            "gpu": 1,
            "forbidden_gpus_visible": False,
            "sudo_used": False,
            "direct_privileged_escalation_used": False,
            "returncode": 0,
            "runtime_log_dir": "agent/runtime_logs/stage0/gr00t",
            "created_at_utc": "2026-04-24T00:00:00Z",
            "command": "env CUDA_VISIBLE_DEVICES=0,1 python negative_fixture.py",
        },
    )

    report = stage1.build_resource_boundary_report(
        run_root,
        tmp_path / "agent" / "runtime_logs" / stage1.RUN_ID,
        created_at_utc="2026-04-24T00:00:00Z",
    )

    assert report["status"] == "PASS"
    assert report["validated_resource_leases"] == []


def test_p5_blocked_allows_skipped_min_loop_verdict_without_resource_lease(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage1.RUN_ID
    p5_root = run_root / "gr00t" / "p5_gate"
    _write_json(
        p5_root / "p5_execution_decision.json",
        {
            "schema_version": "gr00t_p5_execution_decision_v1",
            "decision": "BLOCKED",
            "p4_gate_verdict": "gr00t/p4_refresh/p4_gate_verdict.json",
            "gate_inputs": {
                "status": None,
                "formal_claim_allowed": None,
                "blocking_reasons": [],
                "p5_formal_10ep_eligible": None,
            },
            "blocking_reasons": ["missing_p4_gate_summary"],
        },
    )
    skipped_payload = {
        "schema_version": "task13_full_update_rollout_probe_v1",
        "status": "SKIPPED",
        "gate_mode": "skipped",
        "formal_execution_attempted": False,
        "blocking_reasons": ["missing_p4_gate_summary"],
    }
    _write_json(p5_root / "min_loop_verdict.json", skipped_payload)
    _write_json(p5_root / "p5_gate_blocker_summary.json", skipped_payload)

    summary, blockers, validated = stage1.validate_p5_gate(run_root, clean_p4_gate=False)

    assert blockers == []
    assert summary["status"] == "BLOCK"
    assert summary["decision"] == "BLOCKED"
    assert summary["decision_blocking_reasons"] == ["missing_p4_gate_summary"]
    assert "gr00t/p5_gate/min_loop_verdict.json" in validated
    assert "gr00t/p5_gate/p5_gate_blocker_summary.json" in validated


def test_openpi_claim_gate_review_records_legacy_root_as_known_risk(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage1.RUN_ID
    sweep = run_root / "openpi" / "benchmark_sweep"
    _write_json(
        sweep / "v21_reuse_or_rerun_decision.json",
        {
            "schema_version": "openpi_v21_reuse_or_rerun_decision_v1",
            "decision": "REUSE",
            "authority_mode": "strong",
            "primary_metric_id": "success_rate@0.50_budget",
            "existing_authority_paths": [],
            "freshness_checks": {},
            "blocking_reasons": [],
        },
    )
    _write_json(
        sweep / "resource_lease.json",
        {
            "schema_version": "resource_lease_v1",
            "lane": "openpi_benchmark_sweep",
            "worker": "worker-4",
            "gpu": 2,
            "forbidden_gpus_visible": False,
            "sudo_used": False,
            "direct_privileged_escalation_used": False,
            "returncode": 0,
            "runtime_log_dir": "agent/runtime_logs/stage1/openpi",
            "created_at_utc": "2026-04-24T00:00:00Z",
            "command": "env CUDA_VISIBLE_DEVICES=2 python reuse.py",
        },
    )
    _write_json(
        sweep / "go_no_go_v21.json",
        {
            "schema_version": "openpi_libero_go_no_go_report_v21",
            "authority_mode": "strong",
            "primary_metric_id": "success_rate@0.50_budget",
            "gates": {f"H{index}": {"status": "PASS"} for index in range(8)},
        },
    )
    _write_json(sweep / "paired_summary_abcx_v21.json", {"schema_version": "paired_summary_v1"})
    _write_json(sweep / "rollout_or_tracked_gate_summary.json", {"schema_version": "rollout_summary_v1"})
    _write_json(
        sweep / "benchmark_claim_artifact.json",
        {
            "formal_benchmark_materialized": True,
            "benchmark_success_claimed": False,
            "recap_validated_on_desaturated_eval": False,
            "informativeness_validated": False,
            "eligible_for_state_side_v22": False,
        },
    )
    _write_json(
        run_root / "openpi" / "formal_status_guard" / "runtime_formal_status_unchanged.json",
        {
            "schema_version": "openpi_runtime_formal_status_guard_v1",
            "runtime_level_before": "p2_overfit_or_tiny_update_pass",
            "runtime_level_after": "p2_overfit_or_tiny_update_pass",
            "benchmark_success_encoded_in_runtime_status": False,
            "status": "PASS",
        },
    )
    _write_json(
        sweep / "benchmark_claim_gate_review.json",
        {
            "schema_version": "openpi_stage1_benchmark_claim_gate_review_v1",
            "status": "BLOCK",
            "claim_gate_decision": "CLAIM_HELD",
            "benchmark_runtime_started_by_worker3": False,
            "gpu_runtime_started_by_worker3": False,
            "blocking_reasons": ["legacy_media_root_paths_present"],
            "checks": {
                "canonical_root_paths": {
                    "legacy_root_occurrence_count": 38,
                    "expected_current_root": "/home/howard/Projects/gr00t_wbc_g1_benchmark",
                    "legacy_root": "/media/howard/Data/Projects/gr00t_wbc_g1_benchmark",
                }
            },
        },
    )

    summary, blockers, validated, claim_allowed = stage1.validate_openpi(run_root)

    assert blockers == []
    assert claim_allowed is False
    assert summary["status"] == "PASS"
    assert summary["formal_benchmark_materialized"] is True
    assert summary["benchmark_success_claimed"] is False
    assert summary["claim_gate_review"]["legacy_root_occurrence_count"] == 38
    assert summary["known_risks"] == [
        "benchmark claim held: 38 legacy /media root paths remain in OpenPI source-artifact provenance"
    ]
    assert "openpi/benchmark_sweep/benchmark_claim_gate_review.json" in validated
