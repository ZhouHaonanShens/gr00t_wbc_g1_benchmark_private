"""Default RECAP pipeline scenarios used by the canonical OpenPI entry."""

from __future__ import annotations

from pathlib import Path

from .collect import CollectConfig
from .iteration import IterationConfig


REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_CFG_POLICY_CHECKPOINT = (
    REPO_ROOT
    / "agent"
    / "artifacts"
    / "checkpoints"
    / "openpi_libero_variants"
    / "recap_only_relabel8d_v2"
    / "best"
)
DEFAULT_POSITIVE_POLICY_CHECKPOINT = (
    REPO_ROOT
    / "agent"
    / "artifacts"
    / "checkpoints"
    / "openpi_libero_variants"
    / "fixedadv_relabel8d_control_v1"
    / "best"
)
DEFAULT_CRITIC_CHECKPOINT = (
    REPO_ROOT / "agent" / "artifacts" / "checkpoints" / "recap_critic_smoke" / "best"
)


DEFAULT_RECAP_COLLECTION_SCENARIO = CollectConfig(
    policy_checkpoint=DEFAULT_CFG_POLICY_CHECKPOINT,
    critic_checkpoint=DEFAULT_CRITIC_CHECKPOINT,
    indicator_mode="cfg",
    task_suite_name="libero_spatial",
    task_ids=(0, 1),
    episodes=10,
    output_dir=REPO_ROOT / "agent" / "artifacts" / "openpi_recap_loop" / "default_collect",
    demo_dir=REPO_ROOT
    / "agent"
    / "artifacts"
    / "lerobot_datasets"
    / "physical_intelligence_libero_official_8d",
)

DEFAULT_RECAP_ITERATION_SCENARIO = IterationConfig(
    iter_id="default_iter",
    seed_policy_checkpoint=DEFAULT_CFG_POLICY_CHECKPOINT,
    critic_checkpoint=DEFAULT_CRITIC_CHECKPOINT,
    indicator_mode="cfg",
    task_suite_name="libero_spatial",
    task_ids="0,1",
    episodes=10,
    output_dir=REPO_ROOT / "agent" / "artifacts" / "openpi_recap_loop" / "default_iter",
    demo_dir=REPO_ROOT
    / "agent"
    / "artifacts"
    / "lerobot_datasets"
    / "physical_intelligence_libero_official_8d",
    correction_dir=None,
    critic_config=REPO_ROOT / "work" / "recap" / "critic_vlm" / "configs" / "libero_recap_critic.yaml",
    repaired_matrix_summary_path=REPO_ROOT
    / "agent"
    / "artifacts"
    / "openpi_recap_v1"
    / "repaired_matrix_summary.json",
    tracked_summary_path=REPO_ROOT
    / "agent"
    / "exchange"
    / "openpi_recap_iteration_smoke_summary_v1.md",
    prepared_dataset_dir=REPO_ROOT
    / "agent"
    / "artifacts"
    / "openpi_recap_loop"
    / "iter0"
    / "dataset"
    / "_trainer_compatible_recap_surface",
    informative_prepared_dataset_dir=None,
)


__all__ = [
    "DEFAULT_CFG_POLICY_CHECKPOINT",
    "DEFAULT_CRITIC_CHECKPOINT",
    "DEFAULT_POSITIVE_POLICY_CHECKPOINT",
    "DEFAULT_RECAP_COLLECTION_SCENARIO",
    "DEFAULT_RECAP_ITERATION_SCENARIO",
]
