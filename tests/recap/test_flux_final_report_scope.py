from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import cast

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.datasets import flux_grouped_dataset
from work.recap.models import flux_recap_vla
from work.recap.scripts import build_flux_graft_final_report
from work.recap.scripts import gr00t_screening_authoritative
from work.recap.scripts import gr00t_screening_probe_bypass_diagnostic


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _model_artifact(path: Path, *, label: str) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "flux_recap_model_artifact_v1",
            "artifact_kind": "flux_recap_model_artifact",
            "report_signature_sha256": f"signature-{label}",
            "weights_label": label,
        },
    )


def _config_module(tmp_path: Path, *, train_rel: str, inference_rel: str) -> str:
    package_root = tmp_path / "temp_flux_cfg_report"
    package_dir = package_root / "temp_flux_cfg_report"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    module_path = package_dir / "authoritative_cfg.py"
    module_path.write_text(
        "\n".join(
            [
                "CONFIG = {",
                '    "variant_id": "report_variant",',
                '    "pinned_fluxvla_commit": "flux-commit-123",',
                '    "train_model": {',
                '        "artifact_id": "train_model",',
                '        "authority_role": "flux_train_model",',
                f'        "relative_path": "{train_rel}",',
                f'        "registered_class": "{flux_recap_vla.TRAIN_MODEL_CLASS}",',
                f'        "surface_role": "{flux_recap_vla.TRAIN_MODEL_ROLE}",',
                "    },",
                '    "inference_model": {',
                '        "artifact_id": "inference_model",',
                '        "authority_role": "flux_inference_model",',
                f'        "relative_path": "{inference_rel}",',
                f'        "registered_class": "{flux_recap_vla.INFERENCE_MODEL_CLASS}",',
                f'        "surface_role": "{flux_recap_vla.INFERENCE_MODEL_ROLE}",',
                "    },",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(package_root))
    importlib.invalidate_caches()
    return "temp_flux_cfg_report.authoritative_cfg"


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


def test_flux_final_report_builder_consumes_authoritative_screening_scope(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    inventory_json = (
        repo_root / "agent/artifacts/flux_dataset_probe/dataset_inventory_bundle.json"
    )
    _write_json(inventory_json, _inventory_payload())
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    inference_rel = "agent/artifacts/flux/inference_model_ref.json"
    _model_artifact(repo_root / train_rel, label="train")
    _model_artifact(repo_root / inference_rel, label="inference")
    config_module = _config_module(
        tmp_path,
        train_rel=train_rel,
        inference_rel=inference_rel,
    )
    screening_dir = (
        repo_root / "agent/artifacts/apple_recap_flux_graft/authoritative_screening"
    )
    screening_payload = (
        gr00t_screening_authoritative.materialize_authoritative_screening(
            config_module=config_module,
            dataset_inventory_json=inventory_json,
            output_dir=screening_dir,
            repo_root=repo_root,
        )
    )
    out_md = repo_root / "agent/exchange/Flux_Graft_RECAP_final_report.md"
    out_json = (
        repo_root
        / "agent/artifacts/apple_recap_flux_graft/final_report/final_verdict_pack.json"
    )

    payload = build_flux_graft_final_report.materialize_flux_graft_final_report(
        screening_json=screening_dir
        / gr00t_screening_authoritative.AUTHORITATIVE_SCREENING_JSON_NAME,
        dataset_inventory_json=inventory_json,
        out_md=out_md,
        out_json=out_json,
        repo_root=repo_root,
    )

    assert payload["global_verdict"] == "INSUFFICIENT_EVIDENCE"
    assert payload["formal_eligibility"] == screening_payload["formal_eligibility"]
    assert payload["reason_code"] == "triage_not_provided"
    assert (
        payload["summary"]["inventory_verdict"]
        == screening_payload["inventory_verdict"]
    )
    assert payload["summary"]["prompt_source"] == screening_payload["prompt_source"]
    assert payload["summary"]["servable"] == screening_payload["servable"]
    assert payload["summary"]["live_model_triage"]["input_status"] == "not_provided"
    assert payload["summary"]["live_model_triage"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert (
        payload["summary"]["inference_model_ref"]
        == screening_payload["inference_model_ref"]
    )
    assert (
        payload["authoritative_plane"]["reason_code"]
        == screening_payload["reason_code"]
    )
    assert payload["triage_plane"]["input_status"] == "not_provided"
    assert payload["triage_plane"]["triage_status"] is None
    assert payload["non_authoritative_context"]["diagnostic_context"]["status"] == (
        "not_provided"
    )
    assert payload["non_authoritative_context"]["promotion_context"]["status"] == (
        "not_provided"
    )
    assert payload["non_authoritative_context"]["rtc_context"]["status"] == (
        "not_provided"
    )
    assert payload["artifacts"]["dataset_inventory_bundle"]["path"].endswith(
        "dataset_inventory_bundle.json"
    )
    assert payload["report_artifacts"]["json"] == (
        "agent/artifacts/apple_recap_flux_graft/final_report/final_verdict_pack.json"
    )
    assert out_md.is_file()
    assert out_json.is_file()
    markdown = out_md.read_text(encoding="utf-8")
    assert "## Authoritative plane" in markdown
    assert "## Triage plane" in markdown
    assert "## Non-authoritative context" in markdown
    assert "## Claim boundary" in markdown


def test_flux_final_report_builder_rejects_inventory_backpointer_mismatch(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    inventory_json = (
        repo_root / "agent/artifacts/flux_dataset_probe/dataset_inventory_bundle.json"
    )
    _write_json(inventory_json, _inventory_payload())
    wrong_inventory_json = (
        repo_root / "agent/artifacts/flux_dataset_probe/other_inventory.json"
    )
    wrong_payload = _inventory_payload()
    wrong_payload["dataset_fingerprint"] = "wrong_dataset_fingerprint"
    wrong_payload["binding_join_contract"] = {
        **cast(dict[str, object], wrong_payload["binding_join_contract"]),
        "dataset_fingerprint": "wrong_dataset_fingerprint",
    }
    _write_json(wrong_inventory_json, wrong_payload)
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    inference_rel = "agent/artifacts/flux/inference_model_ref.json"
    _model_artifact(repo_root / train_rel, label="train")
    _model_artifact(repo_root / inference_rel, label="inference")
    config_module = _config_module(
        tmp_path,
        train_rel=train_rel,
        inference_rel=inference_rel,
    )
    screening_dir = (
        repo_root / "agent/artifacts/apple_recap_flux_graft/authoritative_screening"
    )
    _ = gr00t_screening_authoritative.materialize_authoritative_screening(
        config_module=config_module,
        dataset_inventory_json=inventory_json,
        output_dir=screening_dir,
        repo_root=repo_root,
    )

    with pytest.raises(
        ValueError,
        match="must reference the provided dataset inventory artifact",
    ):
        build_flux_graft_final_report.build_final_report_pack(
            screening_json=screening_dir
            / gr00t_screening_authoritative.AUTHORITATIVE_SCREENING_JSON_NAME,
            dataset_inventory_json=wrong_inventory_json,
            repo_root=repo_root,
        )


def test_flux_final_report_builder_rejects_non_authoritative_screening_mode(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    inventory_json = (
        repo_root / "agent/artifacts/flux_dataset_probe/dataset_inventory_bundle.json"
    )
    _write_json(inventory_json, _inventory_payload())
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    inference_rel = "agent/artifacts/flux/inference_model_ref.json"
    _model_artifact(repo_root / train_rel, label="train")
    _model_artifact(repo_root / inference_rel, label="inference")
    config_module = _config_module(
        tmp_path,
        train_rel=train_rel,
        inference_rel=inference_rel,
    )
    screening_dir = (
        repo_root / "agent/artifacts/apple_recap_flux_graft/authoritative_screening"
    )
    screening_path = (
        screening_dir / gr00t_screening_authoritative.AUTHORITATIVE_SCREENING_JSON_NAME
    )
    screening_payload = (
        gr00t_screening_authoritative.materialize_authoritative_screening(
            config_module=config_module,
            dataset_inventory_json=inventory_json,
            output_dir=screening_dir,
            repo_root=repo_root,
        )
    )
    screening_payload["screening_mode"] = (
        gr00t_screening_probe_bypass_diagnostic.SCREENING_MODE
    )
    screening_payload["report_signature_sha256"] = (
        gr00t_screening_authoritative._signature_for_payload(screening_payload)
    )
    _write_json(screening_path, screening_payload)

    with pytest.raises(ValueError, match="screening_mode mismatch"):
        build_flux_graft_final_report.build_final_report_pack(
            screening_json=screening_path,
            dataset_inventory_json=inventory_json,
            repo_root=repo_root,
        )


def test_flux_final_report_builder_rejects_noncanonical_diagnostic_screening_root(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    inventory_json = (
        repo_root / "agent/artifacts/flux_dataset_probe/dataset_inventory_bundle.json"
    )
    _write_json(inventory_json, _inventory_payload())
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    inference_rel = "agent/artifacts/flux/inference_model_ref.json"
    _model_artifact(repo_root / train_rel, label="train")
    _model_artifact(repo_root / inference_rel, label="inference")
    config_module = _config_module(
        tmp_path,
        train_rel=train_rel,
        inference_rel=inference_rel,
    )
    screening_dir = (
        repo_root / "agent/artifacts/apple_recap_flux_graft/authoritative_screening"
    )
    screening_payload = (
        gr00t_screening_authoritative.materialize_authoritative_screening(
            config_module=config_module,
            dataset_inventory_json=inventory_json,
            output_dir=screening_dir,
            repo_root=repo_root,
        )
    )
    diagnostic_screening_path = (
        repo_root
        / "agent/artifacts/apple_recap_flux_graft/diagnostic_probe_bypass/authoritative_screening_summary.json"
    )
    _write_json(diagnostic_screening_path, screening_payload)

    with pytest.raises(ValueError, match="noncanonical_root_contamination"):
        build_flux_graft_final_report.build_final_report_pack(
            screening_json=diagnostic_screening_path,
            dataset_inventory_json=inventory_json,
            repo_root=repo_root,
        )
