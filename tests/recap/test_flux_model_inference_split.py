from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from configs.apple_recap.flux import gr00t_g1_flux_recap_E1
from configs.apple_recap.flux import gr00t_g1_flux_recap_E2
from configs.apple_recap.flux import gr00t_g1_flux_recap_base
from work.recap.models import flux_recap_vla


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


def _model_spec(
    relative_path: str,
    *,
    artifact_id: str,
    authority_role: str,
    registered_class: str,
    surface_role: str,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "authority_role": authority_role,
        "relative_path": relative_path,
        "registered_class": registered_class,
        "surface_role": surface_role,
    }


def test_flux_config_modules_expose_explicit_train_and_inference_refs() -> None:
    modules = (
        gr00t_g1_flux_recap_base,
        gr00t_g1_flux_recap_E1,
        gr00t_g1_flux_recap_E2,
    )

    for module in modules:
        config = module.CONFIG
        assert config["pinned_fluxvla_commit"]
        assert config["authoritative_consumer_surface"] == (
            flux_recap_vla.AUTHORITATIVE_CONSUMER_SURFACE
        )
        assert config["train_model"]["surface_role"] == flux_recap_vla.TRAIN_MODEL_ROLE
        assert config["train_model"]["registered_class"] == (
            flux_recap_vla.TRAIN_MODEL_CLASS
        )
        assert config["inference_model"]["surface_role"] == (
            flux_recap_vla.INFERENCE_MODEL_ROLE
        )
        assert config["inference_model"]["registered_class"] == (
            flux_recap_vla.INFERENCE_MODEL_CLASS
        )


def test_build_flux_model_surface_happy_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    inference_rel = "agent/artifacts/flux/inference_model_ref.json"
    _ = _model_artifact(repo_root / train_rel, label="train")
    _ = _model_artifact(repo_root / inference_rel, label="inference")

    surface = flux_recap_vla.build_flux_recap_model_surface(
        repo_root=repo_root,
        variant_id="g1_flux_unit",
        pinned_fluxvla_commit="flux-commit-123",
        train_model_spec=_model_spec(
            train_rel,
            artifact_id="train_model",
            authority_role="flux_train_model",
            registered_class=flux_recap_vla.TRAIN_MODEL_CLASS,
            surface_role=flux_recap_vla.TRAIN_MODEL_ROLE,
        ),
        inference_model_spec=_model_spec(
            inference_rel,
            artifact_id="inference_model",
            authority_role="flux_inference_model",
            registered_class=flux_recap_vla.INFERENCE_MODEL_CLASS,
            surface_role=flux_recap_vla.INFERENCE_MODEL_ROLE,
        ),
    )
    validation = flux_recap_vla.validate_flux_recap_model_surface(
        surface,
        repo_root=repo_root,
    )

    assert validation["formal_eligibility"] == "ALLOW"
    assert validation["issues"] == []
    assert surface["variant_id"] == "g1_flux_unit"
    assert surface["train_model_ref"]["registered_class"] == (
        flux_recap_vla.TRAIN_MODEL_CLASS
    )
    assert surface["inference_model_ref"]["registered_class"] == (
        flux_recap_vla.INFERENCE_MODEL_CLASS
    )
    assert (
        surface["train_model_ref"]["resolved_path"]
        != surface["inference_model_ref"]["resolved_path"]
    )
    assert surface["train_model_fingerprint"] != surface["inference_model_fingerprint"]


def test_missing_inference_model_ref_fails_closed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    _ = _model_artifact(repo_root / train_rel, label="train")
    train_ref = flux_recap_vla.build_flux_model_ref(
        repo_root=repo_root,
        pinned_fluxvla_commit="flux-commit-123",
        artifact_id="train_model",
        authority_role="flux_train_model",
        relative_path=train_rel,
        registered_class=flux_recap_vla.TRAIN_MODEL_CLASS,
        surface_role=flux_recap_vla.TRAIN_MODEL_ROLE,
    )

    validation = flux_recap_vla.validate_flux_recap_model_surface(
        {
            "schema_version": flux_recap_vla.SCHEMA_VERSION,
            "artifact_kind": flux_recap_vla.ARTIFACT_KIND,
            "pinned_fluxvla_commit": "flux-commit-123",
            "train_model_ref": train_ref,
        },
        repo_root=repo_root,
    )

    assert validation["formal_eligibility"] == "BLOCK"
    assert any(
        issue["field_path"] == "inference_model_ref"
        and issue["code"] == "missing_required_field"
        for issue in validation["issues"]
    )


def test_same_normalized_path_fails_closed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    shared_path = repo_root / "agent/artifacts/flux/shared_model_ref.json"
    _ = _model_artifact(shared_path, label="shared")

    with pytest.raises(
        ValueError,
        match="must resolve to different immutable paths",
    ):
        _ = flux_recap_vla.build_flux_recap_model_surface(
            repo_root=repo_root,
            pinned_fluxvla_commit="flux-commit-123",
            train_model_spec=_model_spec(
                "agent/artifacts/flux/shared_model_ref.json",
                artifact_id="train_model",
                authority_role="flux_train_model",
                registered_class=flux_recap_vla.TRAIN_MODEL_CLASS,
                surface_role=flux_recap_vla.TRAIN_MODEL_ROLE,
            ),
            inference_model_spec=_model_spec(
                "agent/artifacts/flux/../flux/shared_model_ref.json",
                artifact_id="inference_model",
                authority_role="flux_inference_model",
                registered_class=flux_recap_vla.INFERENCE_MODEL_CLASS,
                surface_role=flux_recap_vla.INFERENCE_MODEL_ROLE,
            ),
        )


def test_same_registered_class_fails_closed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    inference_rel = "agent/artifacts/flux/inference_model_ref.json"
    _ = _model_artifact(repo_root / train_rel, label="train")
    _ = _model_artifact(repo_root / inference_rel, label="inference")

    with pytest.raises(ValueError, match="registered_class must equal"):
        _ = flux_recap_vla.build_flux_recap_model_surface(
            repo_root=repo_root,
            pinned_fluxvla_commit="flux-commit-123",
            train_model_spec=_model_spec(
                train_rel,
                artifact_id="train_model",
                authority_role="flux_train_model",
                registered_class=flux_recap_vla.TRAIN_MODEL_CLASS,
                surface_role=flux_recap_vla.TRAIN_MODEL_ROLE,
            ),
            inference_model_spec=_model_spec(
                inference_rel,
                artifact_id="inference_model",
                authority_role="flux_inference_model",
                registered_class=flux_recap_vla.TRAIN_MODEL_CLASS,
                surface_role=flux_recap_vla.INFERENCE_MODEL_ROLE,
            ),
        )


def test_same_surface_role_masquerade_fails_closed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    inference_rel = "agent/artifacts/flux/inference_model_ref.json"
    _ = _model_artifact(repo_root / train_rel, label="train")
    _ = _model_artifact(repo_root / inference_rel, label="inference")

    with pytest.raises(ValueError, match="surface_role must equal"):
        _ = flux_recap_vla.build_flux_recap_model_surface(
            repo_root=repo_root,
            pinned_fluxvla_commit="flux-commit-123",
            train_model_spec=_model_spec(
                train_rel,
                artifact_id="train_model",
                authority_role="flux_train_model",
                registered_class=flux_recap_vla.TRAIN_MODEL_CLASS,
                surface_role=flux_recap_vla.TRAIN_MODEL_ROLE,
            ),
            inference_model_spec=_model_spec(
                inference_rel,
                artifact_id="inference_model",
                authority_role="flux_inference_model",
                registered_class=flux_recap_vla.INFERENCE_MODEL_CLASS,
                surface_role=flux_recap_vla.TRAIN_MODEL_ROLE,
            ),
        )


def test_same_fingerprint_fails_closed_even_with_distinct_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    train_rel = "agent/artifacts/flux/train_model_ref.json"
    inference_rel = "agent/artifacts/flux/inference_model_ref.json"
    payload = {
        "schema_version": "flux_recap_model_artifact_v1",
        "artifact_kind": "flux_recap_model_artifact",
        "report_signature_sha256": "same-signature",
        "weights_label": "same-weights",
    }
    _ = _write_json(repo_root / train_rel, payload)
    _ = _write_json(repo_root / inference_rel, payload)

    with pytest.raises(
        ValueError,
        match="must not share the same model_fingerprint",
    ):
        _ = flux_recap_vla.build_flux_recap_model_surface(
            repo_root=repo_root,
            pinned_fluxvla_commit="flux-commit-123",
            train_model_spec=_model_spec(
                train_rel,
                artifact_id="train_model",
                authority_role="flux_train_model",
                registered_class=flux_recap_vla.TRAIN_MODEL_CLASS,
                surface_role=flux_recap_vla.TRAIN_MODEL_ROLE,
            ),
            inference_model_spec=_model_spec(
                inference_rel,
                artifact_id="inference_model",
                authority_role="flux_inference_model",
                registered_class=flux_recap_vla.INFERENCE_MODEL_CLASS,
                surface_role=flux_recap_vla.INFERENCE_MODEL_ROLE,
            ),
        )
