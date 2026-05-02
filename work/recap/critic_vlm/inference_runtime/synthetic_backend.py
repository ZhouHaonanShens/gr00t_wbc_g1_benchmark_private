from __future__ import annotations

from dataclasses import dataclass

from ..common import as_float, as_int
from ..schema import (
    POSITIVE_PATH_CHECKER_LOCAL_SYNTHETIC,
    CriticArtifact,
    CriticInferenceResult,
    DatasetSample,
)
from .common import ensure_finite_outputs, softmax, validate_processor_contract


@dataclass
class SyntheticCriticInferenceService:
    artifact: CriticArtifact

    def run(self, sample: DatasetSample) -> CriticInferenceResult:
        frame_policy = validate_processor_contract(self.artifact)
        if self.artifact.model_payload is None:
            raise ValueError(
                "artifact_backend_unavailable: synthetic_checker_v1 requires model.pt JSON payload"
            )

        max_text_len = max(
            1.0,
            as_float(
                self.artifact.processor_config.get("text_length_norm", 64.0),
                context="processor.text_length_norm",
            ),
        )
        prompt_feature = min(
            float(len(sample.prompt_raw)), float(max_text_len)
        ) / float(max_text_len)
        bias = as_float(self.artifact.model_payload.get("bias"), context="model.bias")
        text_scale = as_float(
            self.artifact.model_payload.get("text_scale"),
            context="model.text_scale",
        )
        step_scale = as_float(
            self.artifact.model_payload.get("step_scale"),
            context="model.step_scale",
        )
        frame_scale = as_float(
            self.artifact.model_payload.get("frame_scale"),
            context="model.frame_scale",
        )
        temperature = as_float(
            self.artifact.model_payload.get("temperature"),
            context="model.temperature",
        )
        if temperature <= 0.0:
            raise ValueError(
                f"artifact_shape_invalid: model.temperature must be > 0, got {temperature}"
            )

        pivot = float(bias) + float(text_scale) * float(prompt_feature)
        pivot += float(step_scale) * float(as_int(sample.t, context="sample.t"))
        pivot += float(frame_scale) * float(
            as_int(sample.frame_index, context="sample.frame_index")
        )
        logits = [
            -(abs(float(center) - float(pivot)) / float(temperature))
            for center in self.artifact.bin_centers
        ]
        probs = softmax(logits)
        value_v = float(
            sum(
                float(prob) * float(center)
                for prob, center in zip(probs, self.artifact.bin_centers, strict=True)
            )
        )
        ensure_finite_outputs(logits=logits, probs=probs, value_v=value_v)
        return CriticInferenceResult(
            critic_type=self.artifact.critic_type,
            artifact_version=self.artifact.artifact_version,
            bin_logits=[float(value) for value in logits],
            bin_probs=[float(value) for value in probs],
            value_V_raw=float(value_v),
            positive_path_kind=POSITIVE_PATH_CHECKER_LOCAL_SYNTHETIC,
            processor_frame_policy=str(frame_policy),
        )
