from __future__ import annotations

import math
from pathlib import Path

from .common import JsonObject, as_float, as_str, read_json
from .schema import (
    ARTIFACT_VERSION_MULTIMODAL_DISTRIBUTIONAL_V1,
    QWEN3_VL_LATE_FUSION_BACKEND_V1,
    SYNTHETIC_CHECKER_BACKEND_V1,
    CriticArtifact,
    CriticArtifactPaths,
)


def _require_artifact_paths(critic_dir: Path) -> tuple[JsonObject, CriticArtifactPaths]:
    config_path = critic_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config_missing: missing {config_path}")
    config = read_json(config_path)

    artifact_version = config.get("artifact_version")
    if (
        artifact_version is None
        or not isinstance(artifact_version, str)
        or not artifact_version
    ):
        raise ValueError(
            "artifact_contract_invalid: config.json must contain non-empty artifact_version"
        )
    if artifact_version != ARTIFACT_VERSION_MULTIMODAL_DISTRIBUTIONAL_V1:
        raise ValueError(
            f"artifact_version_mismatch: expected {ARTIFACT_VERSION_MULTIMODAL_DISTRIBUTIONAL_V1!r}, "
            f"got {artifact_version!r}"
        )

    critic_type = config.get("critic_type")
    if critic_type != ARTIFACT_VERSION_MULTIMODAL_DISTRIBUTIONAL_V1:
        raise ValueError(
            f"artifact_contract_invalid: critic_type must be {ARTIFACT_VERSION_MULTIMODAL_DISTRIBUTIONAL_V1!r}, "
            f"got {critic_type!r}"
        )

    processor_dir = critic_dir / "processor"
    if not processor_dir.exists() or not processor_dir.is_dir():
        raise FileNotFoundError(
            f"processor_missing: missing processor directory under {critic_dir}"
        )

    processor_config_path = processor_dir / "processor_config.json"
    if not processor_config_path.is_file():
        raise FileNotFoundError(
            f"processor_missing: missing {processor_config_path} for artifact loader"
        )

    model_pt = critic_dir / "model.pt"
    model_safetensors = critic_dir / "model.safetensors"
    if model_pt.is_file():
        model_path = model_pt
    elif model_safetensors.is_file():
        model_path = model_safetensors
    else:
        raise FileNotFoundError(
            f"model_missing: expected model.pt or model.safetensors under {critic_dir}"
        )

    bin_centers_path = critic_dir / "bin_centers.json"
    provenance_path = critic_dir / "provenance.json"
    metrics_path = critic_dir / "metrics.json"
    split_manifest_ref_path = critic_dir / "split_manifest_ref.json"
    for required_path, label in (
        (bin_centers_path, "artifact_contract_invalid: missing bin_centers.json"),
        (provenance_path, "artifact_contract_invalid: missing provenance.json"),
        (metrics_path, "artifact_contract_invalid: missing metrics.json"),
        (
            split_manifest_ref_path,
            "artifact_contract_invalid: missing split_manifest_ref.json",
        ),
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"{label} ({required_path})")

    return config, CriticArtifactPaths(
        critic_dir=critic_dir,
        config_path=config_path,
        processor_config_path=processor_config_path,
        model_path=model_path,
        bin_centers_path=bin_centers_path,
        provenance_path=provenance_path,
        metrics_path=metrics_path,
        split_manifest_ref_path=split_manifest_ref_path,
    )


def load_bin_centers(path: Path) -> list[float]:
    obj = read_json(path)
    values = obj.get("bin_centers", obj.get("values", obj.get("bins")))
    if not isinstance(values, list) or not values:
        raise ValueError(
            "artifact_contract_invalid: bin_centers.json must contain a non-empty list"
        )
    out: list[float] = []
    for idx, value in enumerate(values):
        number = as_float(value, context=f"bin_centers[{idx}]")
        if not math.isfinite(number):
            raise ValueError(
                f"artifact_shape_invalid: bin_centers[{idx}] must be finite, got {number}"
            )
        out.append(float(number))
    return out


def load_synthetic_model_payload(model_path: Path) -> JsonObject:
    if model_path.name != "model.pt":
        raise ValueError(
            "artifact_shape_invalid: checker-local synthetic backend requires model.pt JSON, "
            f"got {model_path.name}"
        )
    obj = read_json(model_path)
    required_keys = ("bias", "text_scale", "step_scale", "frame_scale", "temperature")
    for key in required_keys:
        if key not in obj:
            raise ValueError(
                f"artifact_shape_invalid: model.pt JSON missing required key {key!r}"
            )
        value = as_float(obj.get(key), context=f"model.{key}")
        if not math.isfinite(value):
            raise ValueError(
                f"artifact_shape_invalid: model.{key} must be finite, got {value}"
            )
    return obj


def load_torch_model_payload(model_path: Path) -> JsonObject:
    import torch

    obj = torch.load(str(model_path), map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(
            f"artifact_shape_invalid: expected dict payload in {model_path}, got {type(obj).__name__}"
        )
    backend_name = obj.get("backend_name")
    if backend_name != QWEN3_VL_LATE_FUSION_BACKEND_V1:
        raise ValueError(
            "artifact_shape_invalid: qwen backend model payload backend_name mismatch: "
            f"expected {QWEN3_VL_LATE_FUSION_BACKEND_V1!r}, got {backend_name!r}"
        )
    architecture = obj.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError(
            "artifact_shape_invalid: qwen backend model payload missing architecture object"
        )
    trainable_state_dict = obj.get("trainable_state_dict")
    if not isinstance(trainable_state_dict, dict) or not trainable_state_dict:
        raise ValueError(
            "artifact_shape_invalid: qwen backend model payload missing non-empty trainable_state_dict"
        )
    return obj


def load_critic_artifact(critic_dir: str | Path) -> CriticArtifact:
    critic_root = Path(critic_dir).expanduser().resolve()
    if not critic_root.exists() or not critic_root.is_dir():
        raise FileNotFoundError(f"critic_dir is not a directory: {critic_root}")

    config, paths = _require_artifact_paths(critic_root)
    processor_config = read_json(paths.processor_config_path)
    provenance = read_json(paths.provenance_path)
    metrics = read_json(paths.metrics_path)
    split_manifest_ref = read_json(paths.split_manifest_ref_path)
    artifact_version = as_str(
        config.get("artifact_version"), context="config.artifact_version"
    )
    critic_type = as_str(config.get("critic_type"), context="config.critic_type")
    base_model = as_str(config.get("base_model"), context="config.base_model")
    value_scale = as_str(config.get("value_scale"), context="config.value_scale")
    upgrade_pending = as_str(
        config.get("upgrade_pending"),
        context="config.upgrade_pending",
    )
    backend_name = as_str(config.get("smoke_backend"), context="config.smoke_backend")

    model_payload: JsonObject | None = None
    if backend_name == SYNTHETIC_CHECKER_BACKEND_V1:
        model_payload = load_synthetic_model_payload(paths.model_path)
    elif backend_name == QWEN3_VL_LATE_FUSION_BACKEND_V1:
        model_payload = load_torch_model_payload(paths.model_path)

    return CriticArtifact(
        paths=paths,
        artifact_version=artifact_version,
        critic_type=critic_type,
        base_model=base_model,
        value_scale=value_scale,
        upgrade_pending=upgrade_pending,
        backend_name=backend_name,
        config=config,
        processor_config=processor_config,
        model_payload=model_payload,
        bin_centers=load_bin_centers(paths.bin_centers_path),
        provenance=provenance,
        metrics=metrics,
        split_manifest_ref=split_manifest_ref,
    )
