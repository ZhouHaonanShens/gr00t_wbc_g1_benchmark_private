from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.checkpoint import (  # noqa: E402
    TrainCheckpointMetadata,
    build_checkpoint_provenance,
    build_train_manifest,
)
from work.openpi.recap.dataset import RecapDatasetBundle  # noqa: E402
from work.openpi.recap.protocol import build_frozen_comparison_manifest  # noqa: E402


def _dataset_bundle_with_critic() -> RecapDatasetBundle:
    return RecapDatasetBundle(
        dataset_dir=REPO_ROOT
        / "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1",
        dataset_name="physical_intelligence_libero_official_8d_recap_relabels_v1",
        parquet_files=(REPO_ROOT / "fake_episode.parquet",),
        total_rows=10,
        prompt_route="recap_conditioned_prompt_token_v1",
        conditioning_mode="prompt_text_only",
        source_prompt_field="prompt_raw",
        indicator_positive_fraction=0.4,
        indicator_positive_count=4,
        indicator_negative_count=6,
        advantage_input_mean=0.1,
        advantage_input_abs_mean=0.2,
        action_dim=7,
        state_dim=8,
        record_preview=(),
        recap_contract={
            "value_source": "critic",
            "value_scale": "raw_return",
            "critic_dir": "/tmp/recap_critic/best",
            "critic_checkpoint_ref": "/tmp/recap_critic/best",
            "critic_metrics_path": "/tmp/recap_critic/critic_metrics.json",
            "critic_provenance_path": "/tmp/recap_critic/critic_provenance.json",
            "value_adapter": "task_normalized_return_to_raw_return",
        },
    )


def _train_metadata() -> TrainCheckpointMetadata:
    return TrainCheckpointMetadata(
        variant_name="recap_only_relabel8d_v2",
        dataset_route_id="official_native_8d_recap_relabels_v1",
        dataset_fingerprint="fixture_dataset_fingerprint_sha256",
        episode_universe_hash="fixture_episode_universe_hash_sha256",
        base_checkpoint_id="pi05_libero_anchor",
        train_budget_id="libero_cmp_budget_v2",
        consumer_mode="informative_adv",
        gate_eval_manifest_hash="fixture_gate_eval_manifest_hash_sha256",
        reuse_existing_checkpoint=False,
        reuse_verdict="materialize_new_checkpoint",
    )


def test_recap_checkpoint_provenance_carries_real_critic_refs() -> None:
    dataset_bundle = _dataset_bundle_with_critic()
    manifest = build_frozen_comparison_manifest(
        suite="libero_spatial",
        task_ids="0,1",
        seed_manifest="7,17",
        num_trials_per_task=2,
    )
    output_dir = REPO_ROOT / "agent/artifacts/checkpoints/recap_only_relabel8d_v2"
    train_manifest = build_train_manifest(
        dataset_bundle=dataset_bundle,
        manifest=manifest,
        output_dir=output_dir,
        train_metadata=_train_metadata(),
    )
    provenance = build_checkpoint_provenance(
        dataset_bundle=dataset_bundle,
        manifest=manifest,
        checkpoint_dir=output_dir / "best",
        train_manifest_path=output_dir / "train_manifest.json",
        train_metadata=_train_metadata(),
    )

    training_route = cast(Mapping[str, object], train_manifest["training_route"])
    variant_derivation = cast(Mapping[str, object], provenance["variant_derivation"])

    assert train_manifest["critic_checkpoint_ref"] == "/tmp/recap_critic/best"
    assert provenance["critic_checkpoint_ref"] == "/tmp/recap_critic/best"
    assert training_route["value_source"] == "critic"
    assert training_route["value_scale"] == "raw_return"
    assert training_route["critic_checkpoint_ref"] == "/tmp/recap_critic/best"
    assert (
        training_route["critic_metrics_ref"] == "/tmp/recap_critic/critic_metrics.json"
    )
    assert (
        training_route["critic_provenance_ref"]
        == "/tmp/recap_critic/critic_provenance.json"
    )
    assert training_route["value_adapter"] == "task_normalized_return_to_raw_return"
    assert variant_derivation["value_source"] == "critic"
    assert variant_derivation["critic_checkpoint_ref"] == "/tmp/recap_critic/best"
