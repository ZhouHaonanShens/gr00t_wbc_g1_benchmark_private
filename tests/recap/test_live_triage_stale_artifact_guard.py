from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.datasets import flux_grouped_dataset
from work.recap.scripts import apple_recap_execution_contract
from work.recap.scripts import build_apple_recap_final_report
from work.recap.scripts import build_flux_graft_final_report
from work.recap.scripts import gr00t_screening_authoritative


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _inventory_payload() -> dict[str, object]:
    contract = gr00t_screening_authoritative.build_authoritative_binding_join_contract()
    return {
        "schema_version": flux_grouped_dataset.SCHEMA_VERSION,
        "artifact_kind": flux_grouped_dataset.ARTIFACT_KIND,
        "dataset_dir": "/tmp/fixture_dataset",
        "dataset_name": "fixture_dataset",
        "verdict": flux_grouped_dataset.VERDICT_COMPLETE,
        "dataset_source": {"dataset_dir": "/tmp/fixture_dataset"},
        "dataset_fingerprint": "fixture_dataset_fingerprint_sha256",
        "stats_fingerprint": "fixture_stats_fingerprint_sha256",
        "prompt_source": {
            "prompt_source_field": contract["prompt_source"],
            "prompt_route": "recap_conditioned_prompt_token_v1",
            "conditioning_mode": "prompt_text_only",
            "provenance_complete": True,
        },
        "task_description_source": {"task_text_field": "carrier_text_v1"},
        "camera_inventory": {"view_count": 2},
        "action_state_normalization_source": {
            "norm_stats_policy": contract["action_state_norm_source"],
            "norm_stats_source": contract["norm_stats_source"],
        },
        "schema_compatibility": {
            "status": "compatible",
            "state_dim": 8,
            "action_dim": 7,
        },
        "grouped_stats": {"episode_row_count": 1},
        "binding_join_contract": {
            "dataset_fingerprint": "fixture_dataset_fingerprint_sha256",
            **contract,
        },
        "dataset_adapter": {"schema_version": "flux_parquet_dataset_adapter_v1"},
        "blocking_reasons": [],
    }


def _screening_payload(repo_root: Path, inventory_json: Path) -> dict[str, Any]:
    inference_model_ref = {
        "artifact_id": "inference_model_b0",
        "relative_path": "agent/artifacts/flux/b0_inference_model_ref.json",
        "surface_role": "inference_model_ref",
    }
    payload = {
        "schema_version": gr00t_screening_authoritative.SCREENING_SCHEMA_VERSION,
        "artifact_kind": gr00t_screening_authoritative.SCREENING_ARTIFACT_KIND,
        "screening_mode": "authoritative",
        "config_module": "configs.apple_recap.flux.fixture_b0",
        "dataset_inventory_json": str(inventory_json),
        "output_dir": str(
            repo_root / "agent/artifacts/apple_recap_flux_graft/authoritative_screening"
        ),
        "artifact_path": str(
            repo_root
            / "agent/artifacts/apple_recap_flux_graft/authoritative_screening/authoritative_screening_summary.json"
        ),
        "formal_eligibility": "ALLOW",
        "reason_code": "ok",
        "issues": [],
        "inventory_verdict": flux_grouped_dataset.VERDICT_COMPLETE,
        "inventory_blocking_reasons": [],
        "dataset_fingerprint": "fixture_dataset_fingerprint_sha256",
        "stats_fingerprint": "fixture_stats_fingerprint_sha256",
        "prompt_source": "prompt_raw",
        "schema_compatibility": {
            "status": "compatible",
            "state_dim": 8,
            "action_dim": 7,
        },
        "inventory_binding_join_contract": {"prompt_source": "prompt_raw"},
        "authoritative_binding_join_contract": {"prompt_source": "prompt_raw"},
        "binding_join_evaluation": {
            "matched": True,
            "mismatched_fields": [],
            "join_key_fields": list(gr00t_screening_authoritative.JOIN_KEY_FIELDS),
        },
        "action_space_compatibility": {"status": "compatible"},
        "servable": True,
        "inference_model_ref": inference_model_ref,
        "materialized_model_ref": inference_model_ref,
        "inference_model_materialized": True,
        "train_model_materialized": False,
        "source_artifacts": [
            apple_recap_execution_contract.build_read_only_authority_ref(
                repo_root=repo_root,
                artifact_id="dataset_inventory_bundle",
                authority_role="authoritative_rerun_dataset_inventory",
                relative_path=inventory_json,
            )
        ],
    }
    payload["report_signature_sha256"] = (
        gr00t_screening_authoritative._signature_for_payload(payload)
    )
    return payload


def _triage_payload(
    repo_root: Path,
    *,
    inventory_json: Path,
    screening_json: Path,
) -> dict[str, Any]:
    payload = {
        "schema_version": gr00t_screening_authoritative.LIVE_MODEL_TRIAGE_SCHEMA_VERSION,
        "artifact_kind": gr00t_screening_authoritative.LIVE_MODEL_TRIAGE_ARTIFACT_KIND,
        "generated_at": "2026-04-12T00:00:00+00:00",
        "output_dir": "agent/artifacts/apple_recap_flux_graft/live_model_triage",
        "artifact_path": "agent/artifacts/apple_recap_flux_graft/live_model_triage/triage_result.json",
        "triage_status": "READY",
        "triage_result": "probe_passed_b0_nonzero",
        "allowed_triage_results": list(
            gr00t_screening_authoritative.TRIAGE_RESULT_ENUM
        ),
        "reason_code": "probe_passed_b0_nonzero",
        "issues": [],
        "reason_basis": {"trigger": "fixture"},
        "diagnostic_only": False,
        "mainline_authority": True,
        "main_verdict_eligible": True,
        "external_reference_only": False,
        "authority_scope": "authoritative_live_model_triage",
        "surface_route": "live_model_triage",
        "source_artifacts": [],
        "dataset_inventory_json": apple_recap_execution_contract._repo_relative_path(
            repo_root, inventory_json
        ),
        "authoritative_screening_json": apple_recap_execution_contract._repo_relative_path(
            repo_root, screening_json
        ),
        "authoritative_screening_row_paths": {
            "B0": apple_recap_execution_contract._repo_relative_path(
                repo_root, screening_json
            ),
            "E1": None,
            "E2": None,
        },
        "promotion_gate_json": None,
        "diagnostic_json": None,
        "exact_probe_json": None,
        "exact_probe": {"status": "PASS", "reason_code": "ok"},
        "diagnostic_reference": None,
        "promotion_gate": {
            "allow_plan_next_training_stage": True,
            "promotion_allowed": True,
            "reason_code": "ok",
        },
        "authoritative_rows": {},
    }
    payload["report_signature_sha256"] = (
        gr00t_screening_authoritative._signature_for_payload(payload)
    )
    return payload


def _materialize_guard_fixture(tmp_path: Path) -> dict[str, Path]:
    repo_root = tmp_path / "repo"
    inventory_json = _write_json(
        repo_root / "agent/artifacts/flux_dataset_probe/dataset_inventory_bundle.json",
        _inventory_payload(),
    )
    screening_json = _write_json(
        repo_root
        / "agent/artifacts/apple_recap_flux_graft/authoritative_screening/authoritative_screening_summary.json",
        _screening_payload(repo_root, inventory_json),
    )
    triage_json = _write_json(
        repo_root
        / "agent/artifacts/apple_recap_flux_graft/live_model_triage/triage_result.json",
        _triage_payload(
            repo_root,
            inventory_json=inventory_json,
            screening_json=screening_json,
        ),
    )
    return {
        "repo_root": repo_root,
        "inventory_json": inventory_json,
        "screening_json": screening_json,
        "triage_json": triage_json,
    }


def test_live_triage_guard_rejects_off_repo_backpointer_before_stale(
    tmp_path: Path,
) -> None:
    paths = _materialize_guard_fixture(tmp_path)
    triage_payload = json.loads(paths["triage_json"].read_text(encoding="utf-8"))
    triage_payload["dataset_inventory_json"] = str(
        tmp_path / "outside" / "inventory.json"
    )
    triage_payload["report_signature_sha256"] = (
        gr00t_screening_authoritative._signature_for_payload(triage_payload)
    )
    _write_json(paths["triage_json"], triage_payload)

    with pytest.raises(ValueError, match="noncanonical_root_contamination"):
        build_flux_graft_final_report.build_final_report_pack(
            screening_json=paths["screening_json"],
            dataset_inventory_json=paths["inventory_json"],
            triage_json=paths["triage_json"],
            repo_root=paths["repo_root"],
        )


def test_live_triage_guard_rejects_missing_backpointer_field_as_invalid_input(
    tmp_path: Path,
) -> None:
    paths = _materialize_guard_fixture(tmp_path)
    triage_payload = json.loads(paths["triage_json"].read_text(encoding="utf-8"))
    del triage_payload["authoritative_screening_json"]
    triage_payload["report_signature_sha256"] = (
        gr00t_screening_authoritative._signature_for_payload(triage_payload)
    )
    _write_json(paths["triage_json"], triage_payload)

    with pytest.raises(ValueError, match="invalid_input"):
        build_flux_graft_final_report.build_final_report_pack(
            screening_json=paths["screening_json"],
            dataset_inventory_json=paths["inventory_json"],
            triage_json=paths["triage_json"],
            repo_root=paths["repo_root"],
        )


def test_live_triage_guard_classifies_canonical_backpointer_mismatch_as_stale(
    tmp_path: Path,
) -> None:
    paths = _materialize_guard_fixture(tmp_path)
    triage_payload = json.loads(paths["triage_json"].read_text(encoding="utf-8"))
    triage_payload["authoritative_screening_json"] = (
        "agent/artifacts/apple_recap_flux_graft/authoritative_screening/other_screening.json"
    )
    triage_payload["report_signature_sha256"] = (
        gr00t_screening_authoritative._signature_for_payload(triage_payload)
    )
    _write_json(paths["triage_json"], triage_payload)

    with pytest.raises(ValueError, match="stale_artifact"):
        build_flux_graft_final_report.build_final_report_pack(
            screening_json=paths["screening_json"],
            dataset_inventory_json=paths["inventory_json"],
            triage_json=paths["triage_json"],
            repo_root=paths["repo_root"],
        )


def test_live_triage_guard_rejects_noncanonical_smoke_root_before_stale(
    tmp_path: Path,
) -> None:
    paths = _materialize_guard_fixture(tmp_path)
    smoke_triage_json = (
        paths["repo_root"]
        / "agent/artifacts/apple_recap_flux_graft/live_model_triage/smoke/triage_result.json"
    )
    _write_json(
        smoke_triage_json,
        json.loads(paths["triage_json"].read_text(encoding="utf-8")),
    )

    with pytest.raises(ValueError, match="noncanonical_root_contamination"):
        build_flux_graft_final_report.build_final_report_pack(
            screening_json=paths["screening_json"],
            dataset_inventory_json=paths["inventory_json"],
            triage_json=smoke_triage_json,
            repo_root=paths["repo_root"],
        )


def test_flux_final_report_cli_surfaces_reason_code_for_bad_path(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = build_flux_graft_final_report.main(
        [
            "--dataset-inventory-json",
            str(tmp_path / "outside" / "dataset_inventory_bundle.json"),
            "--screening-json",
            "agent/artifacts/apple_recap_flux_graft/authoritative_screening/authoritative_screening_summary.json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "noncanonical_root_contamination" in captured.err


def test_apple_final_report_cli_surfaces_reason_code_for_bad_path(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = build_apple_recap_final_report.main(
        ["--execution-root", str(tmp_path / "outside" / "apple_recap_exec")]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "noncanonical_root_contamination" in captured.err
