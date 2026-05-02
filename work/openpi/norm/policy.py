from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PHASE1_NORM_POLICY = "recompute_task_local_stats_primary"
PHASE1_NORM_SOURCE = "dataset_meta_stats"
PHASE1_ASSET_ID = "task_local_recomputed"
REFERENCE_CHECKPOINT_ASSET_ID = "droid"


@dataclass(frozen=True)
class NormPolicySpec:
    policy_name: str
    norm_stats_source: str
    norm_stats_path: Path
    asset_id: str
    reference_checkpoint_asset_id: str


def build_phase1_norm_policy(dataset_dir: str | Path) -> NormPolicySpec:
    dataset_dir_path = Path(dataset_dir).resolve()
    norm_stats_path = dataset_dir_path / "meta" / "stats.json"
    if not norm_stats_path.is_file():
        raise FileNotFoundError(norm_stats_path)

    return NormPolicySpec(
        policy_name=PHASE1_NORM_POLICY,
        norm_stats_source=PHASE1_NORM_SOURCE,
        norm_stats_path=norm_stats_path,
        asset_id=PHASE1_ASSET_ID,
        reference_checkpoint_asset_id=REFERENCE_CHECKPOINT_ASSET_ID,
    )


def validate_phase1_norm_policy(spec: NormPolicySpec) -> NormPolicySpec:
    if spec.policy_name != PHASE1_NORM_POLICY:
        raise ValueError(
            f"unexpected norm policy {spec.policy_name!r}; expected {PHASE1_NORM_POLICY!r}"
        )
    if spec.norm_stats_source != PHASE1_NORM_SOURCE:
        raise ValueError(
            "unexpected norm stats source "
            + f"{spec.norm_stats_source!r}; expected {PHASE1_NORM_SOURCE!r}"
        )
    if spec.asset_id != PHASE1_ASSET_ID:
        raise ValueError(
            f"unexpected asset_id {spec.asset_id!r}; expected {PHASE1_ASSET_ID!r}"
        )
    if spec.reference_checkpoint_asset_id != REFERENCE_CHECKPOINT_ASSET_ID:
        raise ValueError(
            "unexpected reference checkpoint asset id "
            + f"{spec.reference_checkpoint_asset_id!r}; expected {REFERENCE_CHECKPOINT_ASSET_ID!r}"
        )
    if not spec.norm_stats_path.is_file():
        raise FileNotFoundError(spec.norm_stats_path)
    return spec


def build_phase1_norm_provenance(spec: NormPolicySpec) -> dict[str, str]:
    _ = validate_phase1_norm_policy(spec)
    return {
        "norm_stats_policy": spec.policy_name,
        "norm_stats_source": spec.norm_stats_source,
        "norm_stats_path": str(spec.norm_stats_path),
        "asset_id": spec.asset_id,
        "reference_checkpoint_asset_id": spec.reference_checkpoint_asset_id,
    }
