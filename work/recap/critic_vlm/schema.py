from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .common import JsonObject


ARTIFACT_VERSION_MULTIMODAL_DISTRIBUTIONAL_V1 = "multimodal_distributional_v1"
SYNTHETIC_CHECKER_BACKEND_V1 = "synthetic_checker_v1"
QWEN3_VL_LATE_FUSION_BACKEND_V1 = "qwen3_vl_late_fusion_v1"
POSITIVE_PATH_CHECKER_LOCAL_SYNTHETIC = "checker_local_synthetic_non_production"
POSITIVE_PATH_QWEN3_VL_LATE_FUSION_LOCAL = "qwen3_vl_late_fusion_local_artifact"


@dataclass(frozen=True)
class CriticArtifactPaths:
    critic_dir: Path
    config_path: Path
    processor_config_path: Path
    model_path: Path
    bin_centers_path: Path
    provenance_path: Path
    metrics_path: Path
    split_manifest_ref_path: Path


@dataclass(frozen=True)
class CriticArtifact:
    paths: CriticArtifactPaths
    artifact_version: str
    critic_type: str
    base_model: str
    value_scale: str
    upgrade_pending: str
    backend_name: str
    config: JsonObject
    processor_config: JsonObject
    model_payload: JsonObject | None
    bin_centers: list[float]
    provenance: JsonObject
    metrics: JsonObject
    split_manifest_ref: JsonObject

    def to_json(self) -> JsonObject:
        return {
            "critic_dir": str(self.paths.critic_dir),
            "artifact_version": self.artifact_version,
            "critic_type": self.critic_type,
            "base_model": self.base_model,
            "value_scale": self.value_scale,
            "upgrade_pending": self.upgrade_pending,
            "backend_name": self.backend_name,
            "required_files": {
                "config": str(self.paths.config_path),
                "processor_config": str(self.paths.processor_config_path),
                "model": str(self.paths.model_path),
                "bin_centers": str(self.paths.bin_centers_path),
                "provenance": str(self.paths.provenance_path),
                "metrics": str(self.paths.metrics_path),
                "split_manifest_ref": str(self.paths.split_manifest_ref_path),
            },
        }


@dataclass(frozen=True)
class DatasetSample:
    dataset_path: Path
    episode_index: int
    episode_length: int
    recap_episode_id: str
    sample_index: int
    local_index: int
    frame_index: int
    t: int
    prompt_raw: str
    return_G: float
    parquet_rel: str
    video_rel: str
    video_decode_backend: str

    def to_json(self) -> JsonObject:
        return {
            "dataset_path": str(self.dataset_path),
            "episode_index": self.episode_index,
            "episode_length": self.episode_length,
            "recap_episode_id": self.recap_episode_id,
            "sample_index": self.sample_index,
            "local_index": self.local_index,
            "frame_index": self.frame_index,
            "t": self.t,
            "prompt_raw": self.prompt_raw,
            "return_G": self.return_G,
            "parquet_rel": self.parquet_rel,
            "video_rel": self.video_rel,
            "video_decode_backend": self.video_decode_backend,
        }


@dataclass(frozen=True)
class CriticInferenceResult:
    critic_type: str
    artifact_version: str
    bin_logits: list[float]
    bin_probs: list[float]
    value_V_raw: float
    positive_path_kind: str
    processor_frame_policy: str

    def to_json(self) -> JsonObject:
        return {
            "critic_type": self.critic_type,
            "artifact_version": self.artifact_version,
            "bin_logits": [float(x) for x in self.bin_logits],
            "bin_probs": [float(x) for x in self.bin_probs],
            "value_V_raw": float(self.value_V_raw),
            "positive_path_kind": self.positive_path_kind,
            "processor_frame_policy": self.processor_frame_policy,
        }


@dataclass(frozen=True)
class ArtifactSmokeRun:
    artifact: CriticArtifact
    sample: DatasetSample
    inference: CriticInferenceResult

    def to_json(self) -> JsonObject:
        inference_json = self.inference.to_json()
        return {
            "critic_dir": str(self.artifact.paths.critic_dir),
            "dataset_path": str(self.sample.dataset_path),
            "critic_type": self.artifact.critic_type,
            "artifact_version": self.artifact.artifact_version,
            "base_model": self.artifact.base_model,
            "value_scale": self.artifact.value_scale,
            "upgrade_pending": self.artifact.upgrade_pending,
            "artifact": self.artifact.to_json(),
            "sample": self.sample.to_json(),
            "inference": inference_json,
            "smoke": inference_json,
        }
