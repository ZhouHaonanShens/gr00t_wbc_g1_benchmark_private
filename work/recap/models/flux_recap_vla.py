from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]

SCHEMA_VERSION = "flux_recap_model_surface_v1"
ARTIFACT_KIND = "flux_recap_model_surface"

TRAIN_MODEL_ROLE = "train_model"
INFERENCE_MODEL_ROLE = "inference_model"
AUTHORITATIVE_CONSUMER_SURFACE = INFERENCE_MODEL_ROLE

TRAIN_MODEL_CLASS = "FluxRecapTrainOpenVLA"
INFERENCE_MODEL_CLASS = "FluxRecapInferenceOpenVLA"


from work.recap.scripts import apple_recap_execution_contract


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _stable_signature(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _issue(code: str, field_path: str, message: str) -> dict[str, str]:
    return {
        "code": str(code),
        "field_path": str(field_path),
        "message": str(message),
    }


def _non_empty_string(value: object, *, field_path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_path} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_path} must be a non-empty string")
    return normalized


def _as_mapping(value: object, *, field_path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_path} must be an object, got {type(value).__name__}")
    return value


def _expected_class_for_role(surface_role: str) -> str:
    if surface_role == TRAIN_MODEL_ROLE:
        return TRAIN_MODEL_CLASS
    if surface_role == INFERENCE_MODEL_ROLE:
        return INFERENCE_MODEL_CLASS
    raise ValueError(f"unsupported surface role: {surface_role!r}")


def _fingerprint_payload(
    *,
    authority_ref: Mapping[str, Any],
    pinned_fluxvla_commit: str,
) -> dict[str, object]:
    return {
        "content_sha256": str(authority_ref.get("content_sha256")),
        "artifact_kind": authority_ref.get("artifact_kind"),
        "schema_version": authority_ref.get("schema_version"),
        "report_signature_sha256": authority_ref.get("report_signature_sha256"),
        "pinned_fluxvla_commit": str(pinned_fluxvla_commit),
    }


def build_flux_model_ref(
    *,
    repo_root: Path,
    pinned_fluxvla_commit: str,
    artifact_id: str,
    authority_role: str,
    relative_path: str | Path,
    registered_class: str,
    surface_role: str,
) -> dict[str, object]:
    normalized_commit = _non_empty_string(
        pinned_fluxvla_commit,
        field_path="pinned_fluxvla_commit",
    )
    normalized_class = _non_empty_string(
        registered_class,
        field_path=f"{surface_role}.registered_class",
    )
    normalized_role = _non_empty_string(
        surface_role,
        field_path=f"{surface_role}.surface_role",
    )
    authority_ref = apple_recap_execution_contract.build_read_only_authority_ref(
        repo_root=repo_root,
        artifact_id=str(artifact_id),
        authority_role=str(authority_role),
        relative_path=relative_path,
    )
    fingerprint = _stable_signature(
        _fingerprint_payload(
            authority_ref=authority_ref,
            pinned_fluxvla_commit=normalized_commit,
        )
    )
    return {
        **authority_ref,
        "registered_class": normalized_class,
        "surface_role": normalized_role,
        "surface_discriminator": f"{normalized_role}:{normalized_class}",
        "model_fingerprint": fingerprint,
        "pinned_fluxvla_commit": normalized_commit,
        "identity_kind": "immutable_read_only_authority_ref",
    }


def _normalize_model_ref(
    *,
    field_path: str,
    expected_role: str,
    value: object,
    repo_root: Path,
    pinned_fluxvla_commit: str,
    issues: list[dict[str, str]],
) -> dict[str, object] | None:
    try:
        ref = _as_mapping(value, field_path=field_path)
        artifact_id = _non_empty_string(
            ref.get("artifact_id"),
            field_path=f"{field_path}.artifact_id",
        )
        authority_role = _non_empty_string(
            ref.get("authority_role"),
            field_path=f"{field_path}.authority_role",
        )
        relative_path = _non_empty_string(
            ref.get("relative_path"),
            field_path=f"{field_path}.relative_path",
        )
        surface_role = _non_empty_string(
            ref.get("surface_role"),
            field_path=f"{field_path}.surface_role",
        )
        registered_class = _non_empty_string(
            ref.get("registered_class"),
            field_path=f"{field_path}.registered_class",
        )
    except (TypeError, ValueError) as exc:
        issues.append(_issue("invalid_model_ref", field_path, str(exc)))
        return None

    if surface_role != expected_role:
        issues.append(
            _issue(
                "invalid_surface_role",
                f"{field_path}.surface_role",
                f"{field_path}.surface_role must equal {expected_role!r}",
            )
        )
    expected_class = _expected_class_for_role(expected_role)
    if registered_class != expected_class:
        issues.append(
            _issue(
                "invalid_registered_class",
                f"{field_path}.registered_class",
                f"{field_path}.registered_class must equal {expected_class!r}",
            )
        )

    try:
        actual_ref = build_flux_model_ref(
            repo_root=repo_root,
            pinned_fluxvla_commit=pinned_fluxvla_commit,
            artifact_id=artifact_id,
            authority_role=authority_role,
            relative_path=relative_path,
            registered_class=registered_class,
            surface_role=surface_role,
        )
    except (TypeError, ValueError) as exc:
        issues.append(_issue("invalid_model_ref", field_path, str(exc)))
        return None

    for key in (
        "resolved_path",
        "content_sha256",
        "artifact_kind",
        "schema_version",
        "report_signature_sha256",
        "model_fingerprint",
    ):
        if key not in ref:
            continue
        declared_value = ref.get(key)
        actual_value = actual_ref.get(key)
        if declared_value != actual_value:
            issues.append(
                _issue(
                    "model_ref_mismatch",
                    f"{field_path}.{key}",
                    f"declared {key} does not match the referenced immutable artifact",
                )
            )

    return actual_ref


def validate_flux_recap_model_surface(
    payload: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    normalized: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
    }

    if payload.get("schema_version") not in (None, SCHEMA_VERSION):
        issues.append(
            _issue(
                "invalid_schema_version",
                "schema_version",
                f"schema_version must equal {SCHEMA_VERSION!r}",
            )
        )
    if payload.get("artifact_kind") not in (None, ARTIFACT_KIND):
        issues.append(
            _issue(
                "invalid_artifact_kind",
                "artifact_kind",
                f"artifact_kind must equal {ARTIFACT_KIND!r}",
            )
        )

    try:
        pinned_fluxvla_commit = _non_empty_string(
            payload.get("pinned_fluxvla_commit"),
            field_path="pinned_fluxvla_commit",
        )
        normalized["pinned_fluxvla_commit"] = pinned_fluxvla_commit
    except (TypeError, ValueError) as exc:
        pinned_fluxvla_commit = ""
        issues.append(
            _issue("invalid_pinned_commit", "pinned_fluxvla_commit", str(exc))
        )

    authoritative_surface = (
        str(
            payload.get("authoritative_consumer_surface")
            or AUTHORITATIVE_CONSUMER_SURFACE
        ).strip()
        or AUTHORITATIVE_CONSUMER_SURFACE
    )
    normalized["authoritative_consumer_surface"] = authoritative_surface
    if authoritative_surface != AUTHORITATIVE_CONSUMER_SURFACE:
        issues.append(
            _issue(
                "invalid_authoritative_consumer_surface",
                "authoritative_consumer_surface",
                f"authoritative_consumer_surface must equal {AUTHORITATIVE_CONSUMER_SURFACE!r}",
            )
        )

    variant_id = str(payload.get("variant_id") or "").strip()
    if variant_id:
        normalized["variant_id"] = variant_id

    raw_train_ref = payload.get("train_model_ref")
    if raw_train_ref is None:
        issues.append(
            _issue(
                "missing_required_field",
                "train_model_ref",
                "train_model_ref is required for split provenance",
            )
        )
        train_model_ref = None
    else:
        train_model_ref = _normalize_model_ref(
            field_path="train_model_ref",
            expected_role=TRAIN_MODEL_ROLE,
            value=raw_train_ref,
            repo_root=repo_root,
            pinned_fluxvla_commit=pinned_fluxvla_commit,
            issues=issues,
        )

    raw_inference_ref = payload.get("inference_model_ref")
    if raw_inference_ref is None:
        issues.append(
            _issue(
                "missing_required_field",
                "inference_model_ref",
                "inference_model_ref is required for authoritative inference",
            )
        )
        inference_model_ref = None
    else:
        inference_model_ref = _normalize_model_ref(
            field_path="inference_model_ref",
            expected_role=INFERENCE_MODEL_ROLE,
            value=raw_inference_ref,
            repo_root=repo_root,
            pinned_fluxvla_commit=pinned_fluxvla_commit,
            issues=issues,
        )

    if train_model_ref is not None:
        normalized["train_model_ref"] = train_model_ref
        normalized["train_model_fingerprint"] = train_model_ref["model_fingerprint"]
    if inference_model_ref is not None:
        normalized["inference_model_ref"] = inference_model_ref
        normalized["inference_model_fingerprint"] = inference_model_ref[
            "model_fingerprint"
        ]

    if train_model_ref is not None and inference_model_ref is not None:
        train_path = str(train_model_ref["resolved_path"])
        inference_path = str(inference_model_ref["resolved_path"])
        if train_path == inference_path:
            issues.append(
                _issue(
                    "aliased_model_identity",
                    "inference_model_ref.resolved_path",
                    "train_model_ref and inference_model_ref must resolve to different immutable paths",
                )
            )
        if train_model_ref["surface_role"] == inference_model_ref["surface_role"]:
            issues.append(
                _issue(
                    "aliased_surface_role",
                    "inference_model_ref.surface_role",
                    "train_model_ref and inference_model_ref must not share the same surface_role",
                )
            )
        if (
            train_model_ref["registered_class"]
            == inference_model_ref["registered_class"]
        ):
            issues.append(
                _issue(
                    "aliased_registered_class",
                    "inference_model_ref.registered_class",
                    "train_model_ref and inference_model_ref must not share the same registered_class",
                )
            )
        if (
            train_model_ref["model_fingerprint"]
            == inference_model_ref["model_fingerprint"]
        ):
            issues.append(
                _issue(
                    "aliased_model_fingerprint",
                    "inference_model_ref.model_fingerprint",
                    "train_model_ref and inference_model_ref must not share the same model_fingerprint",
                )
            )

    return {
        "formal_eligibility": "ALLOW" if not issues else "BLOCK",
        "issues": issues,
        "normalized_surface": normalized,
    }


def _raise_if_blocked(validation: Mapping[str, Any]) -> None:
    if str(validation.get("formal_eligibility")) == "ALLOW":
        return
    issues = validation.get("issues")
    if isinstance(issues, list) and issues:
        first_issue = issues[0]
        if isinstance(first_issue, Mapping):
            raise ValueError(str(first_issue.get("message", "blocked model surface")))
    raise ValueError("blocked model surface")


def _build_surface_payload(
    *,
    variant_id: str | None,
    pinned_fluxvla_commit: str,
    train_model_ref: Mapping[str, Any],
    inference_model_ref: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "pinned_fluxvla_commit": pinned_fluxvla_commit,
        "authoritative_consumer_surface": AUTHORITATIVE_CONSUMER_SURFACE,
        "train_model_ref": dict(train_model_ref),
        "inference_model_ref": dict(inference_model_ref),
        "train_model_fingerprint": train_model_ref["model_fingerprint"],
        "inference_model_fingerprint": inference_model_ref["model_fingerprint"],
    }
    if variant_id is not None and str(variant_id).strip():
        payload["variant_id"] = str(variant_id).strip()
    return payload


def build_flux_recap_model_surface(
    *,
    train_model_spec: Mapping[str, Any],
    inference_model_spec: Mapping[str, Any],
    pinned_fluxvla_commit: str,
    variant_id: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    normalized_commit = _non_empty_string(
        pinned_fluxvla_commit,
        field_path="pinned_fluxvla_commit",
    )
    train_model_ref = build_flux_model_ref(
        repo_root=repo_root,
        pinned_fluxvla_commit=normalized_commit,
        artifact_id=str(train_model_spec["artifact_id"]),
        authority_role=str(train_model_spec["authority_role"]),
        relative_path=str(train_model_spec["relative_path"]),
        registered_class=str(train_model_spec["registered_class"]),
        surface_role=str(train_model_spec["surface_role"]),
    )
    inference_model_ref = build_flux_model_ref(
        repo_root=repo_root,
        pinned_fluxvla_commit=normalized_commit,
        artifact_id=str(inference_model_spec["artifact_id"]),
        authority_role=str(inference_model_spec["authority_role"]),
        relative_path=str(inference_model_spec["relative_path"]),
        registered_class=str(inference_model_spec["registered_class"]),
        surface_role=str(inference_model_spec["surface_role"]),
    )
    payload = _build_surface_payload(
        variant_id=variant_id,
        pinned_fluxvla_commit=normalized_commit,
        train_model_ref=train_model_ref,
        inference_model_ref=inference_model_ref,
    )
    validation = validate_flux_recap_model_surface(payload, repo_root=repo_root)
    _raise_if_blocked(validation)
    return dict(validation["normalized_surface"])


@dataclass(frozen=True)
class FluxRecapTrainOpenVLA:
    model_ref: Mapping[str, Any]

    @property
    def registered_class(self) -> str:
        return TRAIN_MODEL_CLASS


@dataclass(frozen=True)
class FluxRecapInferenceOpenVLA:
    model_ref: Mapping[str, Any]

    @property
    def registered_class(self) -> str:
        return INFERENCE_MODEL_CLASS


def materialize_flux_train_model(
    payload: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> FluxRecapTrainOpenVLA:
    validation = validate_flux_recap_model_surface(payload, repo_root=repo_root)
    _raise_if_blocked(validation)
    normalized_surface = validation["normalized_surface"]
    return FluxRecapTrainOpenVLA(model_ref=normalized_surface["train_model_ref"])


def materialize_flux_inference_model(
    payload: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> FluxRecapInferenceOpenVLA:
    validation = validate_flux_recap_model_surface(payload, repo_root=repo_root)
    _raise_if_blocked(validation)
    normalized_surface = validation["normalized_surface"]
    return FluxRecapInferenceOpenVLA(
        model_ref=normalized_surface["inference_model_ref"]
    )


def materialize_authoritative_inference_surface(
    payload: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    validation = validate_flux_recap_model_surface(payload, repo_root=repo_root)
    _raise_if_blocked(validation)
    normalized_surface = dict(validation["normalized_surface"])
    materialized_model = FluxRecapInferenceOpenVLA(
        model_ref=normalized_surface["inference_model_ref"]
    )
    return {
        "materialized_model": materialized_model,
        "materialization_summary": {
            "schema_version": "flux_authoritative_inference_materialization_v1",
            "artifact_kind": "flux_authoritative_inference_materialization",
            "variant_id": normalized_surface.get("variant_id"),
            "pinned_fluxvla_commit": normalized_surface["pinned_fluxvla_commit"],
            "authoritative_consumer_surface": normalized_surface[
                "authoritative_consumer_surface"
            ],
            "train_model_ref": normalized_surface["train_model_ref"],
            "inference_model_ref": normalized_surface["inference_model_ref"],
            "train_model_fingerprint": normalized_surface["train_model_fingerprint"],
            "inference_model_fingerprint": normalized_surface[
                "inference_model_fingerprint"
            ],
            "materialized_surface_role": INFERENCE_MODEL_ROLE,
            "materialized_registered_class": INFERENCE_MODEL_CLASS,
            "train_model_materialized": False,
            "inference_model_materialized": True,
            "materialized_model_ref": normalized_surface["inference_model_ref"],
        },
    }


__all__ = [
    "ARTIFACT_KIND",
    "AUTHORITATIVE_CONSUMER_SURFACE",
    "FluxRecapInferenceOpenVLA",
    "FluxRecapTrainOpenVLA",
    "INFERENCE_MODEL_CLASS",
    "INFERENCE_MODEL_ROLE",
    "REPO_ROOT",
    "SCHEMA_VERSION",
    "TRAIN_MODEL_CLASS",
    "TRAIN_MODEL_ROLE",
    "build_flux_model_ref",
    "build_flux_recap_model_surface",
    "materialize_authoritative_inference_surface",
    "materialize_flux_inference_model",
    "materialize_flux_train_model",
    "validate_flux_recap_model_surface",
]
