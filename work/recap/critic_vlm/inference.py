from __future__ import annotations

import importlib


_runtime_module = importlib.import_module("work.recap.critic_vlm.inference_runtime")

ArtifactSmokeWorkflow = _runtime_module.ArtifactSmokeWorkflow
CriticInferenceWorkflow = _runtime_module.CriticInferenceWorkflow
run_artifact_smoke = _runtime_module.run_artifact_smoke
run_critic_inference = _runtime_module.run_critic_inference

__all__ = [
    "ArtifactSmokeWorkflow",
    "CriticInferenceWorkflow",
    "run_artifact_smoke",
    "run_critic_inference",
]
