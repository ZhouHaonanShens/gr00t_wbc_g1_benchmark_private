from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.datasets import flux_grouped_dataset
from work.recap.models import flux_recap_vla
from work.recap.scripts import gr00t_screening_authoritative


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
    package_root = tmp_path / "temp_flux_cfg"
    package_dir = package_root / "temp_flux_cfg"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    module_path = package_dir / "authoritative_cfg.py"
    module_path.write_text(
        "\n".join(
            [
                "CONFIG = {",
                '    "variant_id": "test_variant",',
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
    return "temp_flux_cfg.authoritative_cfg"


def _broken_config_module_missing_inference(tmp_path: Path, *, train_rel: str) -> str:
    package_root = tmp_path / "temp_flux_cfg_broken"
    package_dir = package_root / "temp_flux_cfg_broken"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    module_path = package_dir / "authoritative_cfg.py"
    module_path.write_text(
        "\n".join(
            [
                "CONFIG = {",
                '    "variant_id": "broken_variant",',
                '    "pinned_fluxvla_commit": "flux-commit-123",',
                '    "train_model": {',
                '        "artifact_id": "train_model",',
                '        "authority_role": "flux_train_model",',
                f'        "relative_path": "{train_rel}",',
                f'        "registered_class": "{flux_recap_vla.TRAIN_MODEL_CLASS}",',
                f'        "surface_role": "{flux_recap_vla.TRAIN_MODEL_ROLE}",',
                "    },",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(package_root))
    importlib.invalidate_caches()
    return "temp_flux_cfg_broken.authoritative_cfg"


def _inventory_payload() -> dict[str, object]:
    contract = gr00t_screening_authoritative.build_authoritative_binding_join_contract()
    return {
        "schema_version": flux_grouped_dataset.SCHEMA_VERSION,
        "artifact_kind": flux_grouped_dataset.ARTIFACT_KIND,
        "dataset_dir": "/tmp/fixture_dataset",
        "dataset_name": "fixture_dataset",
        "verdict": flux_grouped_dataset.VERDICT_COMPLETE,
        "dataset_source": {
            "dataset_dir": "/tmp/fixture_dataset",
            "route_id": "official_native_8d_recap_relabels_v1",
        },
        "dataset_fingerprint": "fixture_dataset_fingerprint_sha256",
        "stats_fingerprint": "fixture_stats_fingerprint_sha256",
        "prompt_source": {
            "prompt_source_field": contract["prompt_source"],
            "prompt_route": "recap_conditioned_prompt_token_v1",
            "conditioning_mode": "prompt_text_only",
            "provenance_complete": True,
        },
        "task_description_source": {
            "task_text_field": "carrier_text_v1",
            "carrier_route": "carrier_text_v1",
        },
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


def test_authoritative_screening_allows_when_inventory_and_model_binding_match(
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

    payload = gr00t_screening_authoritative.materialize_authoritative_screening(
        config_module=config_module,
        dataset_inventory_json=inventory_json,
        output_dir=tmp_path / "screening",
        repo_root=repo_root,
    )

    assert payload["formal_eligibility"] == "ALLOW"
    assert payload["reason_code"] == "ok"
    assert payload["servable"] is True
    assert payload["inventory_verdict"] == flux_grouped_dataset.VERDICT_COMPLETE
    assert payload["stats_fingerprint"] == "fixture_stats_fingerprint_sha256"
    assert payload["prompt_source"] == "prompt_raw"
    assert payload["action_space_compatibility"]["status"] == "compatible"
    assert (
        payload["inference_model_ref"]["surface_role"]
        == flux_recap_vla.INFERENCE_MODEL_ROLE
    )
    assert payload["materialized_model_ref"] == payload["inference_model_ref"]
    assert payload["source_artifacts"][0]["relative_path"].endswith(
        "dataset_inventory_bundle.json"
    )


def test_authoritative_screening_blocks_on_inventory_before_join(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    inventory_payload = _inventory_payload()
    inventory_payload["verdict"] = flux_grouped_dataset.VERDICT_MISSING
    inventory_payload["blocking_reasons"] = [
        {
            "code": "missing_stats_fingerprint",
            "field_path": "meta/stats.json",
            "message": "stats.json missing",
        }
    ]
    inventory_json = (
        repo_root / "agent/artifacts/flux_dataset_probe/dataset_inventory_bundle.json"
    )
    _write_json(inventory_json, inventory_payload)
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    inference_rel = "agent/artifacts/flux/inference_model_ref.json"
    _model_artifact(repo_root / train_rel, label="train")
    _model_artifact(repo_root / inference_rel, label="inference")
    config_module = _config_module(
        tmp_path,
        train_rel=train_rel,
        inference_rel=inference_rel,
    )

    payload = gr00t_screening_authoritative.materialize_authoritative_screening(
        config_module=config_module,
        dataset_inventory_json=inventory_json,
        output_dir=tmp_path / "screening",
        repo_root=repo_root,
    )

    assert payload["formal_eligibility"] == "BLOCK"
    assert payload["reason_code"] == "inventory_blocked"
    assert payload["servable"] is False
    assert {issue["code"] for issue in payload["issues"]} >= {
        "inventory_verdict_blocked",
        "inventory_blocking_reasons_present",
    }


def test_authoritative_screening_blocks_on_model_binding_when_inference_missing(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    inventory_json = (
        repo_root / "agent/artifacts/flux_dataset_probe/dataset_inventory_bundle.json"
    )
    _write_json(inventory_json, _inventory_payload())
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    _model_artifact(repo_root / train_rel, label="train")
    config_module = _broken_config_module_missing_inference(
        tmp_path, train_rel=train_rel
    )

    payload = gr00t_screening_authoritative.materialize_authoritative_screening(
        config_module=config_module,
        dataset_inventory_json=inventory_json,
        output_dir=tmp_path / "screening",
        repo_root=repo_root,
    )

    assert payload["formal_eligibility"] == "BLOCK"
    assert payload["reason_code"] == "model_binding_blocked"
    assert payload["servable"] is False
    assert payload["issues"][0]["code"] == "invalid_model_binding_surface"


def test_authoritative_screening_blocks_on_binding_join_mismatch(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    inventory_payload = _inventory_payload()
    binding_join_contract = dict(
        cast(dict[str, object], inventory_payload["binding_join_contract"])
    )
    binding_join_contract["prompt_source"] = "prompt_conditioned"
    inventory_payload["binding_join_contract"] = binding_join_contract
    inventory_payload["prompt_source"] = {
        "prompt_source_field": "prompt_conditioned",
        "prompt_route": "recap_conditioned_prompt_token_v1",
        "conditioning_mode": "prompt_text_only",
        "provenance_complete": True,
    }
    inventory_json = (
        repo_root / "agent/artifacts/flux_dataset_probe/dataset_inventory_bundle.json"
    )
    _write_json(inventory_json, inventory_payload)
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    inference_rel = "agent/artifacts/flux/inference_model_ref.json"
    _model_artifact(repo_root / train_rel, label="train")
    _model_artifact(repo_root / inference_rel, label="inference")
    config_module = _config_module(
        tmp_path,
        train_rel=train_rel,
        inference_rel=inference_rel,
    )

    payload = gr00t_screening_authoritative.materialize_authoritative_screening(
        config_module=config_module,
        dataset_inventory_json=inventory_json,
        output_dir=tmp_path / "screening",
        repo_root=repo_root,
    )

    assert payload["formal_eligibility"] == "BLOCK"
    assert payload["reason_code"] == "binding_join_blocked"
    assert payload["servable"] is False
    assert payload["issues"][0]["code"] == "binding_join_mismatch"
    assert payload["issues"][0]["field_path"] == "binding_join_contract.prompt_source"
    assert payload["issues"][0]["expected"] == "prompt_raw"
    assert payload["issues"][0]["observed"] == "prompt_conditioned"
