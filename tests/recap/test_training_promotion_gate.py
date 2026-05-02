from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_training_promotion_gate


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _inventory_payload() -> dict[str, Any]:
    return {
        "schema_version": "flux_dataset_inventory_bundle_v1",
        "artifact_kind": "flux_dataset_inventory_bundle",
        "verdict": "inventory-complete",
        "blocking_reasons": [],
    }


def _screening_payload(inventory_json: Path) -> dict[str, Any]:
    inference_model_ref = {
        "artifact_id": "inference_model",
        "relative_path": "agent/artifacts/flux/inference_model_ref.json",
        "surface_role": "inference_model_ref",
    }
    return {
        "schema_version": "flux_authoritative_screening_summary_v1",
        "artifact_kind": "flux_authoritative_screening_summary",
        "dataset_inventory_json": str(inventory_json),
        "formal_eligibility": "ALLOW",
        "reason_code": "ok",
        "issues": [],
        "inventory_verdict": "inventory-complete",
        "servable": True,
        "binding_join_evaluation": {
            "matched": True,
            "mismatched_fields": [],
            "join_key_fields": [],
        },
        "inference_model_ref": inference_model_ref,
        "materialized_model_ref": dict(inference_model_ref),
        "inference_model_materialized": True,
        "train_model_materialized": False,
    }


def _preflight_payload() -> dict[str, Any]:
    return {
        "schema_version": "g1_gr00t_wbc_preflight_gate_v1",
        "status": "PASS",
        "reason_code": "ok",
        "failure": None,
        "live_checks_attempted": True,
    }


def _smoke_payload() -> dict[str, Any]:
    return {
        "artifact_kind": "flux_gr00t_training_smoke_summary",
        "wrapper_status": "ok",
        "diagnostic_only": True,
        "mainline_authority": False,
        "main_verdict_eligible": False,
        "external_reference_only": True,
        "gate_semantics": "diagnostic_only_non_release_gate",
        "release_gate": False,
        "trainable_surface": "head_only_fallback",
        "selected_checkpoint_path": "/tmp/checkpoint-4",
    }


def _provenance_payload() -> dict[str, Any]:
    return {
        "schema_version": "gr00t_checkpoint_provenance_report_v1",
        "artifact_kind": "gr00t_checkpoint_provenance_report",
        "formal_eligibility": "ALLOW",
        "reason_code": "ok",
        "gate_reasons": [],
        "loadability_status": "LOADABLE_CHECKPOINT_CONFIRMED",
        "selected_checkpoint_path": "/tmp/checkpoint-4",
    }


def _materialize(
    tmp_path: Path,
    *,
    inventory_payload: dict[str, Any] | None = None,
    screening_payload: dict[str, Any] | None = None,
    preflight_payload: dict[str, Any] | None = None,
    smoke_payload: dict[str, Any] | None = None,
    provenance_payload: dict[str, Any] | None = None,
    checkpoint_provenance: bool = False,
) -> dict[str, Any]:
    repo_root = tmp_path / "repo"
    inventory_json = _write_json(
        repo_root / "artifacts/dataset_inventory_bundle.json",
        inventory_payload or _inventory_payload(),
    )
    screening_json = _write_json(
        repo_root / "artifacts/authoritative_screening_summary.json",
        screening_payload or _screening_payload(inventory_json),
    )
    preflight_json = _write_json(
        repo_root / "artifacts/preflight_report.json",
        preflight_payload or _preflight_payload(),
    )
    smoke_json = _write_json(
        repo_root / "artifacts/train_smoke_summary.json",
        smoke_payload or _smoke_payload(),
    )
    provenance_json: Path | None = None
    if checkpoint_provenance:
        provenance_json = _write_json(
            repo_root / "artifacts/checkpoint_provenance_report.json",
            provenance_payload or _provenance_payload(),
        )
    payload = gr00t_training_promotion_gate.materialize_training_promotion_gate(
        dataset_inventory_json=inventory_json,
        authoritative_screening_json=screening_json,
        preflight_report_json=preflight_json,
        train_smoke_summary_json=smoke_json,
        checkpoint_provenance_json=provenance_json,
        output_dir=repo_root / "artifacts/training_promotion_gate",
        repo_root=repo_root,
    )
    return payload


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_training_promotion_gate.main(["--help"])
    assert exc_info.value.code == 0


def test_training_promotion_gate_allows_when_all_rungs_pass(tmp_path: Path) -> None:
    payload = _materialize(tmp_path, checkpoint_provenance=True)

    assert payload["schema_version"] == gr00t_training_promotion_gate.SCHEMA_VERSION
    assert payload["artifact_kind"] == gr00t_training_promotion_gate.ARTIFACT_KIND
    assert payload["allow_plan_next_training_stage"] is True
    assert payload["promotion_allowed"] is True
    assert payload["promotion_status"] == "PASS"
    assert payload["reason_code"] == "ok"
    assert payload["failure_reasons"] == []
    assert payload["diagnostic_only"] is True
    assert payload["mainline_authority"] is False
    assert payload["main_verdict_eligible"] is False
    assert payload["release_gate"] is False
    assert payload["gate_semantics"] == "plan_next_training_stage_only"
    assert (
        payload["checks"][gr00t_training_promotion_gate.CHECK_DEBUG]["passed"] is True
    )
    assert (
        payload["checks"][gr00t_training_promotion_gate.CHECK_INFERENCE]["passed"]
        is True
    )
    assert (
        payload["checks"][gr00t_training_promotion_gate.CHECK_SERVEABLE]["passed"]
        is True
    )
    assert (
        payload["checks"][gr00t_training_promotion_gate.CHECK_PREFLIGHT]["passed"]
        is True
    )
    assert (
        payload["checks"][gr00t_training_promotion_gate.CHECK_SMOKE]["passed"] is True
    )
    assert len(payload["source_artifacts"]) == 5


def test_training_promotion_gate_blocks_first_on_debug_config_rung(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    inventory_json = repo_root / "artifacts/dataset_inventory_bundle.json"
    screening_payload = _screening_payload(inventory_json)
    screening_payload["formal_eligibility"] = "BLOCK"
    screening_payload["reason_code"] = "binding_join_blocked"
    screening_payload["issues"] = [
        {
            "code": "binding_join_mismatch",
            "field_path": "binding_join_contract.prompt_source",
            "message": "dataset inventory join key does not match the authoritative inference binding surface",
            "expected": "prompt_raw",
            "observed": "prompt_conditioned",
        }
    ]
    screening_payload["servable"] = False
    screening_payload["binding_join_evaluation"] = {
        "matched": False,
        "mismatched_fields": ["prompt_source"],
        "join_key_fields": ["prompt_source"],
    }

    payload = _materialize(tmp_path, screening_payload=screening_payload)

    assert payload["allow_plan_next_training_stage"] is False
    assert payload["reason_code"] == "debug_config_blocked"
    assert payload["issues"][0]["code"] == "binding_join_mismatch"
    assert (
        payload["checks"][gr00t_training_promotion_gate.CHECK_DEBUG]["status"]
        == "BLOCK"
    )
    assert (
        payload["checks"][gr00t_training_promotion_gate.CHECK_SERVEABLE]["status"]
        == "NOT_REACHED"
    )


def test_training_promotion_gate_blocks_on_inference_surface_rung(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    inventory_json = repo_root / "artifacts/dataset_inventory_bundle.json"
    screening_payload = _screening_payload(inventory_json)
    screening_payload["formal_eligibility"] = "BLOCK"
    screening_payload["reason_code"] = "model_binding_blocked"
    screening_payload["issues"] = [
        {
            "code": "invalid_model_binding_surface",
            "field_path": "config_module",
            "message": "failed to load authoritative model binding surface",
        }
    ]
    screening_payload["inference_model_materialized"] = False
    screening_payload["materialized_model_ref"] = None
    screening_payload["servable"] = False

    payload = _materialize(tmp_path, screening_payload=screening_payload)

    assert payload["allow_plan_next_training_stage"] is False
    assert payload["reason_code"] == "inference_surface_blocked"
    assert payload["issues"][0]["code"] == "invalid_model_binding_surface"
    assert (
        payload["checks"][gr00t_training_promotion_gate.CHECK_INFERENCE]["status"]
        == "BLOCK"
    )


def test_training_promotion_gate_blocks_on_serveable_inventory_rung(
    tmp_path: Path,
) -> None:
    provenance_payload = _provenance_payload()
    provenance_payload["formal_eligibility"] = "BLOCK"
    provenance_payload["reason_code"] = "wrong_checkpoint_or_missing_finetune_artifact"
    provenance_payload["gate_reasons"] = ["selected_checkpoint_path_missing"]
    provenance_payload["loadability_status"] = "BLOCKED_SELECTED_CHECKPOINT_MISSING"
    provenance_payload["selected_checkpoint_path"] = None

    payload = _materialize(
        tmp_path,
        checkpoint_provenance=True,
        provenance_payload=provenance_payload,
    )

    assert payload["allow_plan_next_training_stage"] is False
    assert payload["reason_code"] == "serveable_inventory_blocked"
    assert (
        payload["issues"][0]["code"] == "wrong_checkpoint_or_missing_finetune_artifact"
    )
    assert (
        payload["checks"][gr00t_training_promotion_gate.CHECK_SERVEABLE]["status"]
        == "BLOCK"
    )


def test_training_promotion_gate_blocks_on_preflight_rung(tmp_path: Path) -> None:
    preflight_payload = _preflight_payload()
    preflight_payload["status"] = "FAIL"
    preflight_payload["reason_code"] = "ping_timeout"
    preflight_payload["failure"] = {
        "message": "timeout waiting for ping ok after 600s",
        "blockers": ["policy_ping"],
        "detail": {"timeout_s": 600},
    }

    payload = _materialize(tmp_path, preflight_payload=preflight_payload)

    assert payload["allow_plan_next_training_stage"] is False
    assert payload["reason_code"] == "preflight_blocked"
    assert payload["issues"][0]["code"] == "ping_timeout"
    assert (
        payload["checks"][gr00t_training_promotion_gate.CHECK_PREFLIGHT]["status"]
        == "BLOCK"
    )
    assert (
        payload["checks"][gr00t_training_promotion_gate.CHECK_SMOKE]["status"]
        == "NOT_REACHED"
    )


def test_training_promotion_gate_blocks_on_non_uplift_smoke_rung(
    tmp_path: Path,
) -> None:
    smoke_payload = _smoke_payload()
    smoke_payload["wrapper_status"] = "blocked"
    smoke_payload["error"] = "final_model_config escaped head-only smoke fencing"

    payload = _materialize(tmp_path, smoke_payload=smoke_payload)

    assert payload["allow_plan_next_training_stage"] is False
    assert payload["reason_code"] == "smoke_blocked"
    assert payload["issues"][0]["code"] == "blocked"
    assert (
        payload["checks"][gr00t_training_promotion_gate.CHECK_SMOKE]["status"]
        == "BLOCK"
    )


def test_training_promotion_gate_returns_gate_input_invalid_for_mismatched_backpointer(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    wrong_inventory_json = repo_root / "artifacts/wrong_inventory_bundle.json"
    screening_payload = _screening_payload(wrong_inventory_json)

    payload = _materialize(tmp_path, screening_payload=screening_payload)

    assert payload["allow_plan_next_training_stage"] is False
    assert payload["reason_code"] == "gate_input_invalid"
    assert payload["failure_reasons"] == ["gate_input_invalid"]
    assert payload["issues"][0]["code"] == "invalid_required_artifact"
    assert (
        payload["checks"][gr00t_training_promotion_gate.CHECK_DEBUG]["status"]
        == "INVALID_INPUT"
    )
