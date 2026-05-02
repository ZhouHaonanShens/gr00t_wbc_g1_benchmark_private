from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable

from agent.run import stage0_artifact_validator as legacy_stage0
from work.stage0_validator import core as work_stage0


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _finding(blocker: str, **evidence_overrides: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "optimizer_step_observed": False,
        "optimizer_param_group_count": 0,
        "trainable_param_count": 0,
        "nonzero_lr_group_count": 0,
        "requires_grad_count": 0,
        "grad_norm_nonzero": False,
        "param_delta_nonzero": False,
        "before_after_checkpoint_hashes": [],
    }
    evidence.update(evidence_overrides)
    return {
        "candidate_id": blocker.lower(),
        "observed_blocker": blocker,
        "classification": "inconclusive",
        "evidence": evidence,
        "next_probe": "bounded probe",
    }


def _gr00t_pass_inputs(run_root: Path) -> None:
    stage_root = run_root / "gr00t" / "stage0"
    _write_json(
        stage_root / "gr00t_update_probe_spec_v1.json",
        {
            "schema_version": "gr00t_update_probe_spec_v1",
            "candidate_scope": "minimal instrumented repro",
            "blocked_before_p5": True,
        },
    )
    _write_json(
        stage_root / "root_cause_matrix.json",
        {
            "schema_version": "gr00t_root_cause_matrix_v1",
            "observed_blockers": ["ALL_ZERO_PARAM_DELTA", "STATIC_AUDIT_BLOCK"],
            "candidate_findings": [
                _finding("ALL_ZERO_PARAM_DELTA"),
                _finding("STATIC_AUDIT_BLOCK", static_audit_details={"status": "BLOCK"}),
            ],
            "same_root_cause_assessment": {
                "zero_lr_trainable_param_group_same_as_all_zero_param_delta": "unknown",
                "rationale": "Stage0 does not prove shared root cause in this fixture.",
            },
        },
    )


def _openpi_contract_block_inputs(run_root: Path) -> None:
    _write_json(
        run_root / "openpi" / "stage0" / "benchmark_gate_contract.json",
        {
            "schema_version": "openpi_benchmark_gate_contract_v1",
            "runtime_prereq": "p2_overfit_or_tiny_update_pass",
            "runtime_pass_is_benchmark_pass": False,
            "benchmark_success_claimed_default": False,
            "authority_mode": "strong",
            "lite_is_advisory_only": True,
            "primary_metric_id": "success_rate@0.50_budget",
            "primary_metric_order": [
                "success_rate@0.50_budget",
                "success_rate@0.75_budget",
                "throughput_like_score",
            ],
            "required_artifacts": ["go_no_go_report.json"],
            "claim_placement_rules": {
                "may_set_benchmark_success_claimed": "only after validator PASS",
                "must_not_modify_runtime_formal_status": True,
            },
        },
    )


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


def _final_base_inputs(run_root: Path) -> None:
    verifier_root = run_root / "verifier"
    _write_json(verifier_root / "dual_track_next_stage_summary.draft.json", _summary())
    (verifier_root / "focused_pytest.log").write_text("pytest_exit=0\n", encoding="utf-8")
    _write_json(verifier_root / "resource_boundary_report.json", {"status": "PASS", "used_gpus": []})


def _resource_boundary_negative_inputs(run_root: Path) -> None:
    _final_base_inputs(run_root)
    _write_json(
        run_root / "gr00t" / "gpu1_min_repro" / "resource_lease.json",
        {
            "schema_version": "resource_lease_v1",
            "lane": "gr00t_gpu1_min_repro",
            "gpu": 1,
            "worker": "worker-2",
            "command": "timeout 60 env CUDA_VISIBLE_DEVICES=0 python repro.py",
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


def _summary_schema_negative_inputs(run_root: Path) -> None:
    _final_base_inputs(run_root)
    summary_path = run_root / "verifier" / "dual_track_next_stage_summary.draft.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["forbidden_inferences"] = ["GR00T min repro PASS != P5 eligibility"]
    _write_json(summary_path, summary)


def _normalize_json(value: object, root: Path) -> object:
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            normalized[key] = "<UTC>" if key == "created_at_utc" else _normalize_json(item, root)
        return normalized
    if isinstance(value, list):
        return [_normalize_json(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), "<RUN_ROOT>")
    return value


def _run_legacy(run_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "agent/run/stage0_artifact_validator.py", "--run-root", str(run_root), *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _run_work(run_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "work.stage0_validator.cli", "--run-root", str(run_root), *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _assert_cli_parity(
    tmp_path: Path,
    fixture_name: str,
    populate: Callable[[Path], None],
    args: list[str],
    outputs: list[Path],
    expected_returncode: int,
) -> None:
    legacy_root = tmp_path / fixture_name / "legacy" / "agent" / "artifacts" / legacy_stage0.RUN_ID
    work_root = tmp_path / fixture_name / "work" / "agent" / "artifacts" / legacy_stage0.RUN_ID
    populate(legacy_root)
    shutil.copytree(legacy_root, work_root)

    legacy_result = _run_legacy(legacy_root, args)
    work_result = _run_work(work_root, args)

    assert legacy_result.returncode == expected_returncode, legacy_result.stderr
    assert work_result.returncode == expected_returncode, work_result.stderr
    assert legacy_result.stdout == work_result.stdout
    assert legacy_result.stderr == work_result.stderr

    for rel_path in outputs:
        legacy_payload = json.loads((legacy_root / rel_path).read_text(encoding="utf-8"))
        work_payload = json.loads((work_root / rel_path).read_text(encoding="utf-8"))
        assert _normalize_json(legacy_payload, legacy_root) == _normalize_json(work_payload, work_root)


def test_stage0_validator_wrapper_is_thin_compatibility_surface() -> None:
    assert legacy_stage0.main is work_stage0.main
    assert legacy_stage0.write_early_gate is work_stage0.write_early_gate
    assert legacy_stage0.write_final_validator_report is work_stage0.write_final_validator_report
    assert legacy_stage0._sha256_file is work_stage0._sha256_file
    assert legacy_stage0.RUN_ID == work_stage0.RUN_ID


def test_stage0_validator_cli_parity_for_pass_and_contract_block(tmp_path: Path) -> None:
    _assert_cli_parity(
        tmp_path,
        "stage0_pass_current",
        _gr00t_pass_inputs,
        ["--early", "gr00t"],
        [
            Path("gr00t/stage0/gr00t_stage0_validator_report.json"),
            Path("gr00t/stage0/gr00t_stage0_ready.json"),
        ],
        expected_returncode=0,
    )
    _assert_cli_parity(
        tmp_path,
        "stage0_block_contract_missing_required_artifacts",
        _openpi_contract_block_inputs,
        ["--early", "openpi"],
        [
            Path("openpi/stage0/openpi_benchmark_gate_validator_report.json"),
            Path("openpi/stage0/benchmark_gate_ready.json"),
        ],
        expected_returncode=0,
    )


def test_stage0_validator_cli_parity_for_negative_final_fixtures(tmp_path: Path) -> None:
    _assert_cli_parity(
        tmp_path,
        "resource_boundary_negative",
        _resource_boundary_negative_inputs,
        ["--final"],
        [Path("verifier/final_artifact_validator_report.json")],
        expected_returncode=1,
    )
    _assert_cli_parity(
        tmp_path,
        "summary_schema_negative",
        _summary_schema_negative_inputs,
        ["--final"],
        [Path("verifier/final_artifact_validator_report.json")],
        expected_returncode=1,
    )
