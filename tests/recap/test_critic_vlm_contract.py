from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.critic_vlm.loader import load_critic_artifact
from work.recap.critic_vlm.inference_runtime.common import validate_processor_contract


def _write_json(path: Path, payload: object) -> None:
    _ = path.write_text(json.dumps(payload), encoding="utf-8")


def test_loader_rejects_missing_provenance_file(tmp_path: Path) -> None:
    critic_dir = tmp_path / "critic"
    processor_dir = critic_dir / "processor"
    processor_dir.mkdir(parents=True)
    _write_json(
        critic_dir / "config.json",
        {
            "artifact_version": "multimodal_distributional_v1",
            "critic_type": "multimodal_distributional_v1",
            "base_model": "Qwen/Qwen3-VL-2B-Instruct",
            "value_scale": "raw_return",
            "upgrade_pending": "temporal_critic_review",
            "smoke_backend": "checker_local_synthetic_non_production",
        },
    )
    _write_json(
        processor_dir / "processor_config.json",
        {
            "task_text_field": "prompt_raw",
            "frame_policy": "current_step_index",
            "allow_future_frames": False,
        },
    )
    _write_json(
        critic_dir / "model.pt",
        {
            "bias": 0.0,
            "text_scale": 1.0,
            "step_scale": 1.0,
            "frame_scale": 1.0,
            "temperature": 1.0,
        },
    )
    _write_json(critic_dir / "bin_centers.json", {"bin_centers": [-1.0, 0.0, 1.0]})
    _write_json(critic_dir / "metrics.json", {})
    _write_json(critic_dir / "split_manifest_ref.json", {})

    with pytest.raises(FileNotFoundError, match="missing provenance.json"):
        _ = load_critic_artifact(critic_dir)


def test_loader_rejects_invalid_processor_prompt_field(tmp_path: Path) -> None:
    critic_dir = tmp_path / "critic"
    processor_dir = critic_dir / "processor"
    processor_dir.mkdir(parents=True)
    _write_json(
        critic_dir / "config.json",
        {
            "artifact_version": "multimodal_distributional_v1",
            "critic_type": "multimodal_distributional_v1",
            "base_model": "Qwen/Qwen3-VL-2B-Instruct",
            "value_scale": "raw_return",
            "upgrade_pending": "temporal_critic_review",
            "smoke_backend": "checker_local_synthetic_non_production",
        },
    )
    _write_json(
        processor_dir / "processor_config.json",
        {
            "task_text_field": "training_prompt_text",
            "frame_policy": "current_step_index",
            "allow_future_frames": False,
        },
    )
    _write_json(
        critic_dir / "model.pt",
        {
            "bias": 0.0,
            "text_scale": 1.0,
            "step_scale": 1.0,
            "frame_scale": 1.0,
            "temperature": 1.0,
        },
    )
    _write_json(critic_dir / "bin_centers.json", {"bin_centers": [-1.0, 0.0, 1.0]})
    _write_json(critic_dir / "metrics.json", {})
    _write_json(critic_dir / "split_manifest_ref.json", {})
    _write_json(critic_dir / "provenance.json", {})

    artifact = load_critic_artifact(critic_dir)

    with pytest.raises(
        ValueError, match="processor.task_text_field must be 'prompt_raw'"
    ):
        _ = validate_processor_contract(artifact)
