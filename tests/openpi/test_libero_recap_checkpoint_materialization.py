from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap import (
    RECAP_ONLY_VARIANT,
    TrainCheckpointMetadata,
    build_frozen_comparison_manifest,
    materialize_recap_checkpoint,
    resolve_recap_dataset,
)
from work.openpi.prompting.routes import RECAP_RELABEL_CONSUMER_MODE
from work.openpi.scripts.libero_rollout_eval_v2 import (
    FailFastError,
    _resolve_servable_checkpoint_ref,
)
from tests.openpi.carrier_text_v1_fixture import (  # noqa: E402
    carrier_text_v1_handoff_metadata,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_minimal_recap_dataset(root: Path) -> Path:
    dataset_dir = root / "physical_intelligence_libero_official_8d_recap_relabels_v1"
    _write_json(
        dataset_dir / "meta" / "info.json",
        {
            "schema_version": "openpi_libero_official_8d_recap_relabels_v1",
            "route_id": "official_native_8d_recap_relabels_v1",
            **carrier_text_v1_handoff_metadata(),
            "source_dataset_name": "physical_intelligence_libero_official_8d",
            "source_dataset_dir": str(
                (root / "physical_intelligence_libero_official_8d").resolve()
            ),
            "features": {
                "observation.images.ego_view": {
                    "dtype": "image",
                    "shape": [256, 256, 3],
                },
                "observation.state": {"dtype": "float32", "shape": [8]},
                "action": {"dtype": "float32", "shape": [7]},
                "annotation.human.task_description": {
                    "dtype": "int64",
                    "shape": [1],
                },
            },
            "recap_advantage_input_contract": {
                "contract_version": "full_recap_continuous_adv_v2"
            },
        },
    )
    _write_json(
        dataset_dir / "meta" / "modality.json",
        {
            "video": {
                "observation.images.ego_view": {
                    "original_key": "observation.images.ego_view"
                }
            },
            "state": {"observation.state": {}},
            "action": {"action": {}},
            "annotation": {"annotation.human.task_description": {}},
        },
    )
    _write_jsonl(
        dataset_dir / "meta" / "tasks.jsonl",
        [{"task": "put the bowl on the plate", "task_index": 0}],
    )
    _write_json(
        dataset_dir / "meta" / "dataset_fingerprint.json",
        {
            "schema_version": "openpi_libero_relabel_dataset_fingerprint_v1",
            "route_id": "official_native_8d_recap_relabels_v1",
            "fingerprint_sha256": "fixture_dataset_fingerprint_sha256",
        },
    )
    _ = (dataset_dir / "meta" / "episode_universe_hash.txt").write_text(
        "fixture_episode_universe_hash_sha256\n", encoding="utf-8"
    )
    frame = pd.DataFrame(
        {
            "action": [[0.1] * 7, [0.2] * 7],
            "episode_index": [0, 0],
            "observation.state": [[0.0] * 8, [1.0] * 8],
            "recap_m2.advantage_A": [0.5, -0.5],
            "recap_m2.advantage_input": [0.25, -0.25],
            "recap_m2.indicator_I": [1, 0],
            "recap_m2.prompt_conditioned": [
                "advantage positive put the bowl on the plate",
                "advantage negative put the bowl on the plate",
            ],
            "recap_m2.prompt_raw": [
                "put the bowl on the plate",
                "put the bowl on the plate",
            ],
            "recap_m2.return_G": [0.0, -1.0],
            "recap_m2.value_V": [-0.5, -0.5],
        }
    )
    parquet_path = dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    return dataset_dir


def _write_minimal_real_export(root: Path) -> Path:
    checkpoint_dir = root / "real_variant_export"
    _ = (checkpoint_dir / "params" / "_METADATA").parent.mkdir(
        parents=True, exist_ok=True
    )
    _ = (checkpoint_dir / "params" / "_METADATA").write_text(
        '{"tree":"fixture"}\n', encoding="utf-8"
    )
    _ = (checkpoint_dir / "params" / "manifest.ocdbt").write_text(
        "fixture-ocdbt-manifest\n", encoding="utf-8"
    )
    _write_json(
        checkpoint_dir
        / "assets"
        / "physical-intelligence"
        / "libero"
        / "norm_stats.json",
        {
            "state": {
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
                "q01": [0.0] * 8,
                "q99": [1.0] * 8,
            },
            "actions": {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "q01": [0.0] * 7,
                "q99": [1.0] * 7,
            },
        },
    )
    _write_json(
        checkpoint_dir / "export_manifest.json",
        {
            "schema_version": "openpi_real_variant_export_v1",
            "source_checkpoint_dir": str(checkpoint_dir),
        },
    )
    return checkpoint_dir


def test_materialize_recap_checkpoint_requires_explicit_real_export_source_for_success(
    tmp_path: Path,
) -> None:
    dataset_dir = _write_minimal_recap_dataset(tmp_path)
    gate_eval_manifest = tmp_path / "eval_manifest_rollout_lite_v2.json"
    _write_json(
        gate_eval_manifest,
        {
            "schema_version": "openpi_libero_rollout_eval_manifest_v2",
            "eval_authority": "fresh_rollout_v2",
            "manifest_name": "rollout_lite_v2",
            "task_suite_name": "libero_spatial",
            "task_ids": [0, 1],
            "seed_manifest": [7, 17, 27, 37],
            "num_trials_per_task": 4,
        },
    )
    dataset_bundle = resolve_recap_dataset(
        dataset_dir,
        consumer_mode=RECAP_RELABEL_CONSUMER_MODE,
    )
    manifest = build_frozen_comparison_manifest(
        suite="libero_spatial",
        task_ids="0,1",
        seed_manifest="7,17,27,37",
        num_trials_per_task=4,
        gate_eval_manifest=gate_eval_manifest,
    )
    checkpoint = materialize_recap_checkpoint(
        output_dir=tmp_path / "recap_only_relabel8d_v2",
        dataset_bundle=dataset_bundle,
        manifest=manifest,
        variant=RECAP_ONLY_VARIANT,
        checkpoint_source="repo_local_openpi_recap_only_offline_advantage_conditioned_baseline",
        train_metadata=TrainCheckpointMetadata(
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
        ),
    )

    assert checkpoint.train_manifest_path.is_file()
    assert checkpoint.checkpoint_provenance_path.is_file()
    assert (checkpoint.checkpoint_dir / "checkpoint.json").is_file()
    assert not (checkpoint.checkpoint_dir / "params" / "_METADATA").exists()
    assert not (
        checkpoint.checkpoint_dir
        / "assets"
        / "physical-intelligence"
        / "libero"
        / "norm_stats.json"
    ).exists()

    with pytest.raises(
        FailFastError,
        match="non-stock variant requires a real serveable checkpoint",
    ):
        _ = _resolve_servable_checkpoint_ref(
            checkpoint_ref=str(checkpoint.checkpoint_dir),
            variant="recap_only_relabel8d_v2",
        )


def test_materialize_recap_checkpoint_copies_assets_only_from_explicit_real_export_source(
    tmp_path: Path,
) -> None:
    dataset_dir = _write_minimal_recap_dataset(tmp_path)
    source_checkpoint_dir = _write_minimal_real_export(tmp_path / "source")
    gate_eval_manifest = tmp_path / "eval_manifest_rollout_lite_v2.json"
    _write_json(
        gate_eval_manifest,
        {
            "schema_version": "openpi_libero_rollout_eval_manifest_v2",
            "eval_authority": "fresh_rollout_v2",
            "manifest_name": "rollout_lite_v2",
            "task_suite_name": "libero_spatial",
            "task_ids": [0, 1],
            "seed_manifest": [7, 17, 27, 37],
            "num_trials_per_task": 4,
        },
    )
    dataset_bundle = resolve_recap_dataset(
        dataset_dir,
        consumer_mode=RECAP_RELABEL_CONSUMER_MODE,
    )
    manifest = build_frozen_comparison_manifest(
        suite="libero_spatial",
        task_ids="0,1",
        seed_manifest="7,17,27,37",
        num_trials_per_task=4,
        gate_eval_manifest=gate_eval_manifest,
    )
    checkpoint = materialize_recap_checkpoint(
        output_dir=tmp_path / "recap_only_relabel8d_v2",
        dataset_bundle=dataset_bundle,
        manifest=manifest,
        variant=RECAP_ONLY_VARIANT,
        checkpoint_source="repo_local_openpi_recap_only_offline_advantage_conditioned_baseline",
        train_metadata=TrainCheckpointMetadata(
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
        ),
        serveable_checkpoint_source_dir=source_checkpoint_dir,
    )

    assert (checkpoint.checkpoint_dir / "params" / "_METADATA").is_file()
    assert (checkpoint.checkpoint_dir / "params" / "manifest.ocdbt").read_text(
        encoding="utf-8"
    ) == "fixture-ocdbt-manifest\n"
    norm_stats_path = (
        checkpoint.checkpoint_dir
        / "assets"
        / "physical-intelligence"
        / "libero"
        / "norm_stats.json"
    )
    assert norm_stats_path.is_file()
    assert json.loads(norm_stats_path.read_text(encoding="utf-8")) == json.loads(
        (
            source_checkpoint_dir
            / "assets"
            / "physical-intelligence"
            / "libero"
            / "norm_stats.json"
        ).read_text(encoding="utf-8")
    )

    serve_checkpoint_ref, serve_checkpoint_mode = _resolve_servable_checkpoint_ref(
        checkpoint_ref=str(checkpoint.checkpoint_dir),
        variant="recap_only_relabel8d_v2",
    )
    assert serve_checkpoint_ref == str(checkpoint.checkpoint_dir)
    assert serve_checkpoint_mode == "local_orbax_checkpoint"
