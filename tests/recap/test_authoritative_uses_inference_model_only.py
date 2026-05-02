from __future__ import annotations

from collections.abc import Mapping
import importlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.models import flux_recap_vla
from work.recap.scripts import gr00t_screening_authoritative


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
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
    _ = (package_dir / "__init__.py").write_text("", encoding="utf-8")
    module_path = package_dir / "authoritative_cfg.py"
    _ = module_path.write_text(
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


def test_authoritative_materialization_records_both_refs_but_only_uses_inference(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    inference_rel = "agent/artifacts/flux/inference_model_ref.json"
    _ = _model_artifact(repo_root / train_rel, label="train")
    _ = _model_artifact(repo_root / inference_rel, label="inference")
    surface = flux_recap_vla.build_flux_recap_model_surface(
        repo_root=repo_root,
        variant_id="test_variant",
        pinned_fluxvla_commit="flux-commit-123",
        train_model_spec={
            "artifact_id": "train_model",
            "authority_role": "flux_train_model",
            "relative_path": train_rel,
            "registered_class": flux_recap_vla.TRAIN_MODEL_CLASS,
            "surface_role": flux_recap_vla.TRAIN_MODEL_ROLE,
        },
        inference_model_spec={
            "artifact_id": "inference_model",
            "authority_role": "flux_inference_model",
            "relative_path": inference_rel,
            "registered_class": flux_recap_vla.INFERENCE_MODEL_CLASS,
            "surface_role": flux_recap_vla.INFERENCE_MODEL_ROLE,
        },
    )

    materialized = flux_recap_vla.materialize_authoritative_inference_surface(
        surface,
        repo_root=repo_root,
    )
    summary = materialized["materialization_summary"]

    assert materialized["materialized_model"].__class__.__name__ == (
        flux_recap_vla.INFERENCE_MODEL_CLASS
    )
    assert summary["train_model_ref"]["registered_class"] == (
        flux_recap_vla.TRAIN_MODEL_CLASS
    )
    assert summary["inference_model_ref"]["registered_class"] == (
        flux_recap_vla.INFERENCE_MODEL_CLASS
    )
    assert summary["materialized_surface_role"] == flux_recap_vla.INFERENCE_MODEL_ROLE
    assert summary["materialized_registered_class"] == (
        flux_recap_vla.INFERENCE_MODEL_CLASS
    )
    assert summary["train_model_materialized"] is False
    assert summary["inference_model_materialized"] is True
    assert summary["materialized_model_ref"] == summary["inference_model_ref"]


def test_authoritative_screening_script_only_materializes_inference_model(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    inference_rel = "agent/artifacts/flux/inference_model_ref.json"
    _ = _model_artifact(repo_root / train_rel, label="train")
    _ = _model_artifact(repo_root / inference_rel, label="inference")
    config_module = _config_module(
        tmp_path,
        train_rel=train_rel,
        inference_rel=inference_rel,
    )
    output_dir = tmp_path / "screening_out"

    payload = gr00t_screening_authoritative.materialize_authoritative_screening(
        config_module=config_module,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    written = json.loads(
        (
            output_dir / gr00t_screening_authoritative.AUTHORITATIVE_SCREENING_JSON_NAME
        ).read_text(encoding="utf-8")
    )

    assert payload == written
    assert payload["screening_mode"] == "authoritative"
    assert payload["config_module"] == config_module
    assert payload["train_model_ref"]["surface_role"] == flux_recap_vla.TRAIN_MODEL_ROLE
    assert payload["inference_model_ref"]["surface_role"] == (
        flux_recap_vla.INFERENCE_MODEL_ROLE
    )
    assert payload["train_model_materialized"] is False
    assert payload["inference_model_materialized"] is True
    assert payload["materialized_model_ref"] == payload["inference_model_ref"]
