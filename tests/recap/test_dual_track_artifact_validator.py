from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from agent.run import dual_track_artifact_validator as validator


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _formal(lane: str, *, status: str = "BLOCK", allowed: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "dual_track_formal_status_v1",
        "lane": lane,
        "track": "formal",
        "status": status,
        "formal_claim_allowed": allowed,
        "blocking_reasons": [] if status == "PASS" else ["test_blocker"],
        "authority_inputs": [],
        "validator_outputs": [],
        "entered_next_gate": allowed,
        "next_gate_allowed": allowed,
        "notes": "",
    }
    if lane == "openpi":
        payload["runtime_level"] = "p1_one_step_pass" if status == "PASS" else "blocked_policy_bridge_error"
        payload["runtime_claims"] = (
            ["materialization_ready", "p0_loader_runtime_pass", "p1_one_step_pass"]
            if status == "PASS"
            else []
        )
    return payload


def _exploratory(lane: str, *, status: str = "SIGNAL") -> dict[str, object]:
    return {
        "schema_version": "dual_track_exploratory_signal_v1",
        "lane": lane,
        "track": "exploratory",
        "status": status,
        "exploratory_only": True,
        "formal_claim_allowed": False,
        "must_not_unlock_formal_gate": True,
        "method": "additional_seed",
        "risk_label": "exploratory_not_formal",
        "inputs": [],
        "outputs": [],
        "observed_signal": {"loss_delta": -0.1},
        "notes": "",
    }


def _summary() -> dict[str, object]:
    return {
        "schema_version": "dual_track_summary_v1",
        "gr00t": {
            "formal": {
                "status": "BLOCK",
                "formal_claim_allowed": False,
                "blocking_reasons": ["test_blocker"],
                "artifact": "gr00t/formal_status.json",
            },
            "exploratory": {"status": "SIGNAL", "artifact": "gr00t/exploratory_signal.json"},
        },
        "openpi": {
            "formal": {
                "status": "BLOCK",
                "formal_claim_allowed": False,
                "blocking_reasons": ["test_blocker"],
                "artifact": "openpi/formal_status.json",
                "runtime_level": "none",
            },
            "exploratory": {"status": "SIGNAL", "artifact": "openpi/exploratory_signal.json"},
        },
        "next_actions": [],
        "forbidden_inferences": [
            "exploratory signal != formal pass",
            "OpenPI exploratory dataset != formal materialized",
            "GR00T metric ablation/additional seed signal != P5 eligible",
        ],
    }


def test_validator_accepts_blocked_formal_with_exploratory_signal(tmp_path: Path) -> None:
    gr_formal = _write_json(tmp_path / "gr00t" / "formal_status.json", _formal("gr00t"))
    op_formal = _write_json(tmp_path / "openpi" / "formal_status.json", _formal("openpi"))
    gr_exp = _write_json(tmp_path / "gr00t" / "exploratory_signal.json", _exploratory("gr00t"))
    op_exp = _write_json(tmp_path / "openpi" / "exploratory_signal.json", _exploratory("openpi"))
    summary = _write_json(tmp_path / "dual_track_summary.json", _summary())
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text(
        "CUDA_VISIBLE_DEVICES=1 gr00t\nCUDA_VISIBLE_DEVICES=2 openpi\n",
        encoding="utf-8",
    )
    gpu_log = _write_json(
        tmp_path / "gpu_boundary.json",
        {"gr00t": {"gpu": 1}, "openpi": {"gpu": 2}, "used_gpus": [1, 2]},
    )

    assert (
        validator.main(
            [
                "--gr00t-formal",
                str(gr_formal),
                "--gr00t-exploratory",
                str(gr_exp),
                "--openpi-formal",
                str(op_formal),
                "--openpi-exploratory",
                str(op_exp),
                "--summary",
                str(summary),
                "--runtime-log",
                str(runtime_log),
                "--gpu-boundary-log",
                str(gpu_log),
            ]
        )
        == 0
    )


def test_formal_status_rejects_compound_block_value(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "formal_status.json",
        {**_formal("openpi"), "status": "BLOCK(label_semantics_block)"},
    )

    try:
        validator.validate_formal_status(path, expected_lane="openpi")
    except validator.ValidationError as exc:
        assert "status must be one of" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("compound formal status should be rejected")


def test_exploratory_signal_never_allows_formal_claim(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "exploratory_signal.json",
        {**_exploratory("gr00t"), "formal_claim_allowed": True},
    )

    try:
        validator.validate_exploratory_signal(path, expected_lane="gr00t")
    except validator.ValidationError as exc:
        assert "formal_claim_allowed must be false" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("exploratory formal claim should be rejected")


def test_summary_rejects_exploratory_signal_unlocking_formal(tmp_path: Path) -> None:
    summary = _summary()
    gr00t = summary["gr00t"]
    assert isinstance(gr00t, dict)
    formal = gr00t["formal"]
    assert isinstance(formal, dict)
    formal["formal_claim_allowed"] = True

    path = _write_json(tmp_path / "dual_track_summary.json", summary)

    try:
        validator.validate_summary(
            path,
            formal_payloads={"gr00t": _formal("gr00t"), "openpi": _formal("openpi")},
            exploratory_payloads={"gr00t": _exploratory("gr00t"), "openpi": _exploratory("openpi")},
        )
    except validator.ValidationError as exc:
        assert "formal_claim_allowed differs from artifact" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("summary must not unlock formal from exploratory signal")


def test_runtime_boundary_rejects_sudo_and_forbidden_gpus(tmp_path: Path) -> None:
    sudo_log = tmp_path / "sudo.log"
    sudo_log.write_text("sudo rm -rf never\n", encoding="utf-8")
    gpu_log = tmp_path / "gpu0.log"
    gpu_log.write_text("CUDA_VISIBLE_DEVICES=0 python train.py\n", encoding="utf-8")

    for path in (sudo_log, gpu_log):
        try:
            validator.validate_runtime_log_boundaries(path)
        except validator.ValidationError:
            pass
        else:  # pragma: no cover - defensive branch
            raise AssertionError(f"{path} should fail boundary validation")


def test_openpi_runtime_level_rejects_overclaiming(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "openpi" / "formal_status.json",
        {
            **_formal("openpi", status="PASS", allowed=True),
            "runtime_level": "p1_one_step_pass",
            "runtime_claims": [
                "materialization_ready",
                "p0_loader_runtime_pass",
                "p1_one_step_pass",
                "p2_overfit_or_tiny_update_pass",
            ],
        },
    )

    try:
        validator.validate_formal_status(path, expected_lane="openpi")
    except validator.ValidationError as exc:
        assert "must not claim higher runtime levels" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("p1 runtime level must not claim p2 pass")


def test_resource_lease_and_runtime_evidence_minimum(tmp_path: Path) -> None:
    gr_formal = _formal("gr00t")
    op_formal = _formal("openpi")
    lease_path = _write_json(
        tmp_path / "gr00t_lease.json",
        {
            "schema_version": "resource_lease_v1",
            "lane": "gr00t",
            "gpu": "1",
            "worker": "worker-2",
            "command": "timeout 60 env CUDA_VISIBLE_DEVICES=1 python train.py",
            "started_at_utc": "2026-04-24T00:00:00Z",
            "ended_at_utc": "2026-04-24T00:01:00Z",
            "returncode": 0,
            "timeout_s": 60,
            "runtime_log": "agent/runtime_logs/boundary_push3/gr00t.log",
            "artifacts": ["agent/artifacts/boundary_push3/gr00t/formal_status.json"],
            "forbidden_gpus_visible": False,
            "sudo_used": False,
        },
    )
    lease = validator.validate_resource_lease(lease_path)

    try:
        validator.validate_runtime_evidence_minimum(
            {"gr00t": gr_formal, "openpi": op_formal},
            leases_by_lane={"gr00t": [lease]},
        )
    except validator.ValidationError as exc:
        assert "openpi formal runtime_evidence" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("OpenPI BLOCK status must have runtime evidence or a lease")

    op_formal["runtime_evidence"] = {
        "command": "timeout 60 env CUDA_VISIBLE_DEVICES=2 python one_step.py",
        "started_at_utc": "2026-04-24T00:00:00Z",
        "ended_at_utc": "2026-04-24T00:01:00Z",
        "returncode": 1,
        "runtime_log": "agent/runtime_logs/boundary_push3/openpi.log",
        "artifacts": ["agent/artifacts/boundary_push3/openpi/formal_status.json"],
        "forbidden_gpus_visible": False,
        "sudo_used": False,
    }

    validator.validate_runtime_evidence_minimum(
        {"gr00t": gr_formal, "openpi": op_formal},
        leases_by_lane={"gr00t": [lease]},
    )


def test_resource_lease_rejects_wrong_gpu(tmp_path: Path) -> None:
    lease_path = _write_json(
        tmp_path / "bad_openpi_lease.json",
        {
            "schema_version": "resource_lease_v1",
            "lane": "openpi",
            "gpu": "3",
            "worker": "worker-3",
            "command": "timeout 60 env CUDA_VISIBLE_DEVICES=3 python train.py",
            "started_at_utc": "2026-04-24T00:00:00Z",
            "ended_at_utc": "2026-04-24T00:01:00Z",
            "returncode": 0,
            "timeout_s": 60,
            "runtime_log": "openpi.log",
            "artifacts": [],
            "forbidden_gpus_visible": False,
            "sudo_used": False,
        },
    )

    try:
        validator.validate_resource_lease(lease_path)
    except validator.ValidationError as exc:
        assert "openpi must use GPU2" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("OpenPI lease on GPU3 should fail")


def test_resource_lease_accepts_worker_alias_fields(tmp_path: Path) -> None:
    lease_path = _write_json(
        tmp_path / "worker2_gr00t_lease.json",
        {
            "schema_version": "resource_lease_v1",
            "lane": "gr00t",
            "gpu": 1,
            "worker": "worker-2",
            "command": ["timeout", "60", "env", "CUDA_VISIBLE_DEVICES=1", "python", "train.py"],
            "command_shell": "timeout 60 env CUDA_VISIBLE_DEVICES=1 python train.py",
            "start_time": "2026-04-24T00:00:00+00:00",
            "end_time": "2026-04-24T00:01:00+00:00",
            "returncode": 1,
            "timeout_seconds": 60,
            "runtime_log": "/tmp/gr00t.log",
            "artifacts": ["/tmp/formal_status.json"],
            "forbidden_gpus_visible": False,
            "sudo_used": False,
        },
    )

    lease = validator.validate_resource_lease(lease_path)
    assert lease["lane"] == "gr00t"


def test_gr00t_candidate_matrix_requires_non_scalar_graduated_candidates(tmp_path: Path) -> None:
    matrix_path = _write_json(
        tmp_path / "candidate_matrix.json",
        {
            "candidates": [
                {
                    "candidate_id": "contact_weighting_v1",
                    "candidate_type": "contact/lift-aware weighting",
                    "track": "formal_remediation",
                    "graduation_stage": "C2_DRY_RUN",
                    "entered_formal_run": True,
                },
                {
                    "candidate_id": "failure_stage_reweight_v1",
                    "candidate_type": "failure-stage data reweighting",
                    "track": "formal_remediation",
                    "graduation_stage": "C1_TELEMETRY",
                },
                {
                    "candidate_id": "scalar_amplitude_negative_control",
                    "candidate_type": "scalar amplitude",
                    "track": "exploratory_negative_control",
                    "graduation_stage": "C0_STATIC",
                },
            ]
        },
    )

    validator.validate_gr00t_candidate_matrix(matrix_path)

    bad_matrix_path = _write_json(
        tmp_path / "bad_candidate_matrix.json",
        {
            "candidates": [
                {
                    "candidate_id": "scalar_amplitude_only",
                    "candidate_type": "scalar amplitude",
                    "track": "formal",
                    "graduation_stage": "C2_DRY_RUN",
                    "entered_formal_run": True,
                }
            ]
        },
    )

    try:
        validator.validate_gr00t_candidate_matrix(bad_matrix_path)
    except validator.ValidationError as exc:
        assert "at least two non-scalar formal candidates" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("scalar-only matrix should fail")


def test_gr00t_candidate_manifest_blocks_p5_before_c4(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "candidate_manifest.json",
        {
            "candidate_id": "premature_p5",
            "candidate_type": "contact/lift-aware weighting",
            "track": "formal",
            "graduation_stage": "C3_FORMAL_3SEED",
            "selected_for_p5": True,
        },
    )

    try:
        validator.validate_gr00t_candidate_manifest(path)
    except validator.ValidationError as exc:
        assert "P5 candidates must reach C4_P5_ELIGIBLE" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("P5 candidate before C4 should fail")


def test_cli_returns_nonzero_on_invalid_artifact(tmp_path: Path) -> None:
    gr_formal = _write_json(tmp_path / "gr00t" / "formal_status.json", _formal("gr00t"))
    op_formal = _write_json(tmp_path / "openpi" / "formal_status.json", _formal("openpi"))
    gr_exp = _write_json(
        tmp_path / "gr00t" / "exploratory_signal.json",
        {**_exploratory("gr00t"), "must_not_unlock_formal_gate": False},
    )
    op_exp = _write_json(tmp_path / "openpi" / "exploratory_signal.json", _exploratory("openpi"))
    summary = _write_json(tmp_path / "dual_track_summary.json", _summary())

    result = subprocess.run(
        [
            sys.executable,
            "agent/run/dual_track_artifact_validator.py",
            "--gr00t-formal",
            str(gr_formal),
            "--gr00t-exploratory",
            str(gr_exp),
            "--openpi-formal",
            str(op_formal),
            "--openpi-exploratory",
            str(op_exp),
            "--summary",
            str(summary),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "must_not_unlock_formal_gate must be true" in result.stderr
