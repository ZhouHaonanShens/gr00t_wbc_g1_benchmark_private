from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.run import stage0_artifact_validator as stage0


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _openpi_contract(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "openpi_benchmark_gate_contract_v1",
        "runtime_prereq": "p2_overfit_or_tiny_update_pass",
        "runtime_pass_is_benchmark_pass": False,
        "benchmark_success_claimed_default": False,
        "claim_scope": "openpi_libero_v21_benchmark_candidate",
        "authority_mode": "strong",
        "lite_is_advisory_only": True,
        "primary_metric_id": "success_rate@0.50_budget",
        "primary_metric_order": [
            "success_rate@0.50_budget",
            "success_rate@0.75_budget",
            "throughput_like_score",
        ],
        "required_artifacts": [
            "go_no_go_report.json",
            "rollout_or_tracked_gate_summary.json",
            "resource_lease.json",
            "benchmark_claim_artifact.json",
        ],
        "claim_placement_rules": {
            "may_set_benchmark_success_claimed": "only after validator PASS",
            "must_not_modify_runtime_formal_status": True,
        },
    }
    payload.update(overrides)
    return payload


def test_openpi_contract_rejects_runtime_pass_as_benchmark_pass(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "benchmark_gate_contract.json",
        _openpi_contract(runtime_pass_is_benchmark_pass=True),
    )

    with pytest.raises(stage0.Stage0ValidationError, match="runtime_pass_is_benchmark_pass"):
        stage0.validate_openpi_benchmark_gate_contract(path)


def test_openpi_early_gate_writes_pass_ready_with_contract_hash(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage0.RUN_ID
    stage_root = run_root / "openpi" / "stage0"
    contract_path = _write_json(stage_root / "benchmark_gate_contract.json", _openpi_contract())

    report, ready = stage0.write_early_gate(
        run_root,
        "openpi",
        created_at_utc="2026-04-24T00:00:00Z",
    )

    assert report["status"] == "PASS"
    assert ready["ready"] is True
    assert ready["allowed_next_worker"] == "worker-3"
    assert ready["allowed_next_stage"] == "gpu2_benchmark_candidate"
    assert ready["spec_hash"] == stage0._sha256_file(contract_path)


def test_ready_true_rejects_non_empty_blocking_reasons(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage0.RUN_ID
    stage_root = run_root / "openpi" / "stage0"
    _write_json(stage_root / "benchmark_gate_contract.json", _openpi_contract())
    _, ready = stage0.write_early_gate(run_root, "openpi", created_at_utc="2026-04-24T00:00:00Z")
    ready["blocking_reasons"] = ["stale blocker"]
    ready_path = _write_json(stage_root / "benchmark_gate_ready.json", ready)

    with pytest.raises(stage0.Stage0ValidationError, match="ready=true requires empty"):
        stage0.validate_stage0_ready_file(ready_path, run_root=run_root, expected_lane="openpi")
