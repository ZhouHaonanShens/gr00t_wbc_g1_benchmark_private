from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from work.openpi.recap.critic_bridge import build_provenance_handle
from work.recap.critic_vlm.schema import CriticArtifact


PUBLIC_VALUE_SCALE = "raw_return"
INTERNAL_TASK_NORMALIZED_VALUE_SCALE = "task_normalized_return"


@dataclass(frozen=True)
class AdaptedCriticValue:
    value_distribution: dict[str, object]
    decoded_value: dict[str, object]
    provenance_handle: dict[str, object]
    internal_value: float
    internal_value_scale: str
    task_max_steps: int


def _scale_factor(*, artifact: CriticArtifact, task_max_steps: int) -> float:
    if str(artifact.value_scale) == PUBLIC_VALUE_SCALE:
        return 1.0
    if str(artifact.value_scale) == INTERNAL_TASK_NORMALIZED_VALUE_SCALE:
        if int(task_max_steps) <= 0:
            raise ValueError(f"task_max_steps must be > 0, got {task_max_steps}")
        return float(task_max_steps)
    raise ValueError(
        "unsupported critic value_scale for OpenPI raw_return adapter: "
        + f"{artifact.value_scale!r}"
    )


def adapt_critic_result_to_raw_return(
    *,
    artifact: CriticArtifact,
    bin_logits: list[float],
    bin_probs: list[float],
    value_v_internal: float,
    task_max_steps: int,
) -> AdaptedCriticValue:
    factor = _scale_factor(artifact=artifact, task_max_steps=task_max_steps)
    required_files = cast(dict[str, object], artifact.to_json()["required_files"])
    side_channels = artifact.processor_config.get("side_channels", [])
    if not isinstance(side_channels, list):
        side_channels = []
    return AdaptedCriticValue(
        value_distribution={
            "bin_centers": [float(center) * factor for center in artifact.bin_centers],
            "bin_logits": [float(value) for value in bin_logits],
            "bin_probs": [float(value) for value in bin_probs],
        },
        decoded_value={
            "value_V_raw": float(value_v_internal) * factor,
            "value_scale": PUBLIC_VALUE_SCALE,
        },
        provenance_handle=build_provenance_handle(
            critic_dir=artifact.paths.critic_dir,
            artifact_version=artifact.artifact_version,
            backend_name=artifact.backend_name,
            value_scale=PUBLIC_VALUE_SCALE,
            prompt_text_field=str(
                artifact.processor_config.get("task_text_field", "prompt_raw")
            ),
            frame_policy=str(
                artifact.processor_config.get("frame_policy", "current_step_index")
            ),
            allow_future_frames=bool(
                artifact.processor_config.get("allow_future_frames", False)
            ),
            side_channels=[str(channel) for channel in side_channels],
            required_files=required_files,
            provenance_path=artifact.paths.provenance_path,
        ),
        internal_value=float(value_v_internal),
        internal_value_scale=str(artifact.value_scale),
        task_max_steps=int(task_max_steps),
    )


def build_openpi_critic_metadata(*, artifact: CriticArtifact) -> dict[str, object]:
    critic_dir = artifact.paths.critic_dir.resolve()
    canonical_metrics_path = critic_dir / "critic_metrics.json"
    canonical_provenance_path = critic_dir / "critic_provenance.json"
    adapter_name = (
        "identity"
        if str(artifact.value_scale) == PUBLIC_VALUE_SCALE
        else "task_normalized_return_to_raw_return"
    )
    metadata: dict[str, object] = {
        "value_source": "critic",
        "value_scale": PUBLIC_VALUE_SCALE,
        "critic_dir": str(critic_dir),
        "critic_checkpoint_ref": str(critic_dir),
        "critic_metrics_path": str(
            canonical_metrics_path
            if canonical_metrics_path.is_file()
            else critic_dir / "metrics.json"
        ),
        "critic_provenance_path": str(
            canonical_provenance_path
            if canonical_provenance_path.is_file()
            else critic_dir / "provenance.json"
        ),
        "critic_internal_value_scale": str(artifact.value_scale),
        "value_adapter": adapter_name,
        "adapter_required": bool(adapter_name != "identity"),
    }
    return metadata


__all__ = [
    "AdaptedCriticValue",
    "INTERNAL_TASK_NORMALIZED_VALUE_SCALE",
    "PUBLIC_VALUE_SCALE",
    "adapt_critic_result_to_raw_return",
    "build_openpi_critic_metadata",
]
