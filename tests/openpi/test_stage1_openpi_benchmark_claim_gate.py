from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "stage1_p4_benchmark_validator_20260424T111232Z"
OPENPI_ROOT = REPO_ROOT / "agent" / "artifacts" / RUN_ID / "openpi"
BENCHMARK_ROOT = OPENPI_ROOT / "benchmark_sweep"
PRIMARY_METRIC_ID = "success_rate@0.50_budget"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_stage1_openpi_strong_v21_bundle_reuses_authority_without_overclaim() -> None:
    required_files = {
        "resource_lease": BENCHMARK_ROOT / "resource_lease.json",
        "reuse_decision": BENCHMARK_ROOT / "v21_reuse_or_rerun_decision.json",
        "paired_summary": BENCHMARK_ROOT / "paired_summary_abcx_v21.json",
        "go_no_go": BENCHMARK_ROOT / "go_no_go_v21.json",
        "rollout_summary": BENCHMARK_ROOT / "rollout_or_tracked_gate_summary.json",
        "benchmark_claim": BENCHMARK_ROOT / "benchmark_claim_artifact.json",
        "formal_status_guard": OPENPI_ROOT
        / "formal_status_guard"
        / "runtime_formal_status_unchanged.json",
    }
    for name, path in required_files.items():
        assert path.is_file(), f"missing Stage1 OpenPI {name}: {path}"

    decision = _load_json(required_files["reuse_decision"])
    assert decision["schema_version"] == "openpi_v21_reuse_or_rerun_decision_v1"
    assert decision["decision"] == "REUSE"
    assert decision["authority_mode"] == "strong"
    assert decision["primary_metric_id"] == PRIMARY_METRIC_ID
    assert decision["blocking_reasons"] == []
    assert decision["source_authority_kind"] == "existing_formal_authority"
    assert (
        decision["freshness_checks"]["existing_formal_authority_used_instead_of_worker_message"]
        is True
    )

    lease = _load_json(required_files["resource_lease"])
    assert lease["schema_version"] == "resource_lease_v1"
    assert lease["lane"] == "openpi"
    assert lease["worker"] == "worker-4"
    assert lease["gpu"] == 2
    assert lease["cuda_visible_devices"] == "2"
    assert lease["forbidden_gpus_visible"] is False
    assert lease["sudo_used"] is False
    assert lease["direct_privileged_escalation_used"] is False
    assert lease["returncode"] == 0
    assert all(command["returncode"] == 0 for command in lease["commands"])
    assert all("sudo" not in " ".join(command["command"]) for command in lease["commands"])

    paired = _load_json(required_files["paired_summary"])
    assert paired["schema_version"] == "openpi_libero_paired_summary_abcx_v21"
    assert paired["authority_mode"] == "strong"
    assert paired["primary_metric_id"] == PRIMARY_METRIC_ID
    assert sorted(paired["variants"]) == ["A", "B", "C", "X"]

    go_no_go = _load_json(required_files["go_no_go"])
    assert go_no_go["schema_version"] == "openpi_libero_go_no_go_report_v21"
    assert go_no_go["authority_mode"] == "strong"
    assert go_no_go["primary_metric_id"] == PRIMARY_METRIC_ID
    assert go_no_go["gate_order"] == [f"H{index}" for index in range(8)]
    assert set(go_no_go["gates"]) >= {f"H{index}" for index in range(8)}

    claim = _load_json(required_files["benchmark_claim"])
    assert claim["schema_version"] == "openpi_benchmark_claim_artifact_v1"
    assert claim["authority_mode"] == "strong"
    assert claim["primary_metric_id"] == PRIMARY_METRIC_ID
    assert claim["formal_benchmark_materialized"] is True
    assert claim["benchmark_success_claimed"] is False
    assert claim["benchmark_claim_allowed"] is False
    assert claim["runtime_pass_is_benchmark_pass"] is False
    assert claim["benchmark_success_encoded_in_runtime_status"] is False
    assert claim["recap_validated_on_desaturated_eval"] is False
    assert claim["informativeness_validated"] is False
    assert claim["eligible_for_state_side_v22"] is False
    assert "OpenPI smoke PASS != benchmark PASS" in claim["forbidden_inferences"]
    assert (
        "OpenPI benchmark materialized != RECAP/state-side success"
        in claim["forbidden_inferences"]
    )

    guard = _load_json(required_files["formal_status_guard"])
    assert guard["schema_version"] == "openpi_runtime_formal_status_guard_v1"
    assert guard["status"] == "PASS"
    assert guard["runtime_level_before"] == "p2_overfit_or_tiny_update_pass"
    assert guard["runtime_level_after"] == "p2_overfit_or_tiny_update_pass"
    assert guard["benchmark_success_encoded_in_runtime_status"] is False
