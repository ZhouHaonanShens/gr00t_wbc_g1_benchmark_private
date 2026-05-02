from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..common import JsonObject


VALUE_SCALE_TASK_NORMALIZED_RETURN = "task_normalized_return"
UPGRADE_PENDING = "temporal_critic_review"
PROPRIO_DIM = 43
PROPRIO_HIDDEN_DIM = 128
T_HIDDEN_DIM = 32
FUSION_HIDDEN_DIM = 512
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
RANKING_LOSS_WEIGHT = 0.2
PUBLIC_WARMSTART_PROMPTS = {
    "g1-pick-apple": "Pick the apple from the table and place it into the basket.",
    "g1-pick-pear": "Pick the pear from the table and place it into the basket.",
    "g1-pick-grapes": "Pick the grapes from the table and place them into the basket.",
    "g1-pick-starfruit": "Pick the starfruit from the table and place it into the basket.",
    "unitree_g1.LMPnPAppleToPlateDC": "Pick the apple from the table and place it onto the plate.",
}


@dataclass(frozen=True)
class TrainConfig:
    train_manifest: Path
    val_manifest: Path
    public_warmstart_manifest: Path
    critic_tag: str
    base_model: str
    device: str
    batch_size: int
    warmstart_epochs: int
    formal_epochs: int
    lr_head: float
    lr_lora: float
    seed: int
    top_n_lora_blocks: int
    attn_implementation: str | None
    prompt_text_mode: str
    use_proprio: bool
    use_t_norm: bool
    bin_centers: tuple[float, ...] | None
    remediation_diagnosis_json: Path | None
    max_warmstart_samples: int | None
    max_train_samples: int | None
    max_val_samples: int | None


@dataclass(frozen=True)
class WarmstartPlan:
    phase_done: bool
    phase_used_data: bool
    available_local_roots: list[str]
    used_dataset_roots: list[str]
    public_sample_count: int
    note: str

    def to_json(self) -> JsonObject:
        return {
            "phase_done": self.phase_done,
            "phase_used_data": self.phase_used_data,
            "available_local_roots": [str(x) for x in self.available_local_roots],
            "used_dataset_roots": [str(x) for x in self.used_dataset_roots],
            "public_sample_count": int(self.public_sample_count),
            "note": self.note,
        }


@dataclass(frozen=True)
class TrainResult:
    critic_dir: Path
    metrics: JsonObject
    provenance: JsonObject


@dataclass(frozen=True)
class PublicWarmstartSample:
    source_name: str
    dataset_root: Path
    prompt_raw: str
    use_prompt: bool
    video_path: Path
    frame_index: int
    proprio: list[float]
    t_norm: float
    target_return: float
    target_bin_index: int
    episode_uid: int
