from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.run import stage0_artifact_validator as stage0


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


def _gr00t_matrix() -> dict[str, object]:
    return {
        "schema_version": "gr00t_root_cause_matrix_v1",
        "observed_blockers": ["ALL_ZERO_PARAM_DELTA", "STATIC_AUDIT_BLOCK"],
        "candidate_findings": [
            _finding("ALL_ZERO_PARAM_DELTA"),
            _finding("STATIC_AUDIT_BLOCK", static_audit_details={"status": "BLOCK"}),
        ],
        "same_root_cause_assessment": {
            "zero_lr_trainable_param_group_same_as_all_zero_param_delta": "unknown",
            "rationale": "Stage0 does not yet prove whether these share a root cause.",
        },
    }


def test_gr00t_early_gate_writes_authoritative_ready_after_validator(tmp_path: Path) -> None:
    run_root = tmp_path / "agent" / "artifacts" / stage0.RUN_ID
    stage_root = run_root / "gr00t" / "stage0"
    _write_json(
        stage_root / "gr00t_update_probe_spec_v1.json",
        {
            "schema_version": "gr00t_update_probe_spec_v1",
            "candidate_scope": "minimal instrumented repro",
            "blocked_before_p5": True,
        },
    )
    _write_json(stage_root / "root_cause_matrix.json", _gr00t_matrix())

    report, ready = stage0.write_early_gate(
        run_root,
        "gr00t",
        created_at_utc="2026-04-24T00:00:00Z",
    )

    assert report["status"] == "PASS"
    assert ready["ready"] is True
    assert ready["allowed_next_worker"] == "worker-2"
    assert ready["allowed_next_stage"] == "gpu1_min_repro"
    assert ready["validator_outputs"] == ["gr00t/stage0/gr00t_stage0_validator_report.json"]
    stage0.validate_stage0_dependency_ready(
        stage_root / "gr00t_stage0_ready.json",
        run_root=run_root,
        expected_lane="gr00t",
    )

    draft = stage_root / "gr00t_stage0_ready.draft.json"
    _write_json(draft, ready)
    with pytest.raises(stage0.Stage0ValidationError, match="draft ready files"):
        stage0.validate_stage0_dependency_ready(draft, run_root=run_root, expected_lane="gr00t")


def test_gr00t_root_cause_matrix_requires_static_details_and_blocker_findings(tmp_path: Path) -> None:
    matrix = _gr00t_matrix()
    matrix["candidate_findings"] = [_finding("STATIC_AUDIT_BLOCK")]
    path = _write_json(tmp_path / "root_cause_matrix.json", matrix)

    with pytest.raises(stage0.Stage0ValidationError, match="STATIC_AUDIT_BLOCK requires static audit details"):
        stage0.validate_gr00t_root_cause_matrix(path)

    matrix = _gr00t_matrix()
    matrix["candidate_findings"] = [
        _finding("STATIC_AUDIT_BLOCK", static_audit_details={"status": "BLOCK"})
    ]
    path = _write_json(tmp_path / "root_cause_matrix_missing_delta.json", matrix)

    with pytest.raises(stage0.Stage0ValidationError, match="ALL_ZERO_PARAM_DELTA requires"):
        stage0.validate_gr00t_root_cause_matrix(path)
