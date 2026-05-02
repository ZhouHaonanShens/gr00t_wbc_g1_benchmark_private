from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

from ..common import as_str
from ..schema import CriticArtifact


def validate_processor_contract(artifact: CriticArtifact) -> str:
    task_text_field = as_str(
        artifact.processor_config.get("task_text_field"),
        context="processor.task_text_field",
    )
    if task_text_field != "prompt_raw":
        raise ValueError(
            "artifact_contract_invalid: processor.task_text_field must be 'prompt_raw', "
            f"got {task_text_field!r}"
        )
    frame_policy = as_str(
        artifact.processor_config.get("frame_policy"),
        context="processor.frame_policy",
    )
    if frame_policy != "current_step_index":
        raise ValueError(
            f"artifact_contract_invalid: processor.frame_policy must be 'current_step_index', got {frame_policy!r}"
        )
    allow_future_frames = bool(
        artifact.processor_config.get("allow_future_frames", False)
    )
    if allow_future_frames:
        raise ValueError("artifact_contract_invalid: allow_future_frames must be false")
    return str(frame_policy)


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        raise ValueError("artifact_shape_invalid: logits must be non-empty")
    max_logit = max(logits)
    shifted = [math.exp(float(value) - float(max_logit)) for value in logits]
    denom = float(sum(shifted))
    if not math.isfinite(denom) or denom <= 0.0:
        raise ValueError(f"artifact_shape_invalid: invalid softmax denominator {denom}")
    return [float(value / denom) for value in shifted]


def ensure_finite_outputs(
    *, logits: list[float], probs: list[float], value_v: float
) -> None:
    if not math.isfinite(float(value_v)):
        raise ValueError(
            f"artifact_shape_invalid: scalarized value_V_raw is not finite ({value_v})"
        )
    if any(not math.isfinite(value) for value in logits):
        raise ValueError("artifact_shape_invalid: logits contain non-finite values")
    if any(not math.isfinite(value) for value in probs):
        raise ValueError(
            "artifact_shape_invalid: probabilities contain non-finite values"
        )


def load_video_frame(video_path: Path, frame_index: int) -> object:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"artifact_backend_unavailable: failed to open video {video_path}"
        )
    _ = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(
            f"artifact_backend_unavailable: failed to decode frame_index={frame_index} from {video_path}"
        )
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def move_batch_to_device(batch: dict[str, object], device: str) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = cast(Any, value).to(device)
        else:
            moved[key] = value
    return moved
