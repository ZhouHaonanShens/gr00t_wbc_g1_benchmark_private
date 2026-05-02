from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..dataset import load_dataset_sample
from ..loader import load_critic_artifact
from ..schema import (
    ArtifactSmokeRun,
    CriticArtifact,
    CriticInferenceResult,
    DatasetSample,
    QWEN3_VL_LATE_FUSION_BACKEND_V1,
    SYNTHETIC_CHECKER_BACKEND_V1,
)
from .qwen_backend import QwenLateFusionInferenceService
from .synthetic_backend import SyntheticCriticInferenceService


@dataclass
class CriticInferenceWorkflow:
    artifact: CriticArtifact

    def execute(self, sample: DatasetSample) -> CriticInferenceResult:
        if self.artifact.backend_name == SYNTHETIC_CHECKER_BACKEND_V1:
            return SyntheticCriticInferenceService(self.artifact).run(sample)
        if self.artifact.backend_name == QWEN3_VL_LATE_FUSION_BACKEND_V1:
            return QwenLateFusionInferenceService(self.artifact).run(sample)
        raise ValueError(
            f"artifact_backend_unimplemented: artifact_version={self.artifact.artifact_version} "
            f"backend={self.artifact.backend_name!r} is not implemented yet"
        )


@dataclass
class ArtifactSmokeWorkflow:
    critic_dir: str | Path
    dataset_path: str | Path
    sample_index: int

    def execute(self) -> ArtifactSmokeRun:
        artifact = load_critic_artifact(self.critic_dir)
        sample = load_dataset_sample(self.dataset_path, int(self.sample_index))
        inference = CriticInferenceWorkflow(artifact).execute(sample)
        return ArtifactSmokeRun(artifact=artifact, sample=sample, inference=inference)


def run_critic_inference(
    artifact: CriticArtifact,
    sample: DatasetSample,
) -> CriticInferenceResult:
    return CriticInferenceWorkflow(artifact).execute(sample)


def run_artifact_smoke(
    *,
    critic_dir: str | Path,
    dataset_path: str | Path,
    sample_index: int,
) -> ArtifactSmokeRun:
    return ArtifactSmokeWorkflow(
        critic_dir=critic_dir,
        dataset_path=dataset_path,
        sample_index=int(sample_index),
    ).execute()
