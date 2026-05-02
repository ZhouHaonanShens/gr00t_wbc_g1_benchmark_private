from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class StateTokenTrainingScenario:
    dataset_dir: Path
    output_dir: Path
    gate_manifest_path: Path


DEFAULT_STATE_TOKEN_TRAINING_SCENARIO = StateTokenTrainingScenario(
    dataset_dir=REPO_ROOT
    / "agent"
    / "artifacts"
    / "lerobot_datasets"
    / "physical_intelligence_libero_official_8d_recap_relabels_v1",
    output_dir=REPO_ROOT
    / "agent"
    / "artifacts"
    / "checkpoints"
    / "openpi_libero_variants"
    / "recap_state_tokens_relabel8d_v2",
    gate_manifest_path=REPO_ROOT
    / "work"
    / "openpi"
    / "eval"
    / "manifests"
    / "eval_manifest_rollout_lite_v2.json",
)


__all__ = [
    "DEFAULT_STATE_TOKEN_TRAINING_SCENARIO",
    "StateTokenTrainingScenario",
]
