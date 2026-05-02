from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import work.openpi.pipelines.recap.policy_training as recap_train_script
from work.openpi.recap import data_transforms
from work.openpi.recap.data_transforms import PreparedStageDataset
from work.openpi.recap.real_variant_export import (
    RealVariantExportBundle,
    RealVariantExportRequest,
)
from work.openpi.recap.train_config import (
    RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL,
    RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS,
    RepairedStageConfig,
    resolve_repaired_stage_config,
)
from work.openpi.prompting.routes import (  # noqa: E402
    RECAP_RELABEL_CONSUMER_MODE,
    SHUFFLED_ADV_DIAG_CONSUMER_MODE,
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


def write_gate_eval_manifest(root: Path) -> Path:
    manifest_path = root / "eval_manifest_rollout_lite_v2.json"
    _write_json(
        manifest_path,
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
    return manifest_path


def write_fake_real_export(
    root: Path,
    *,
    default_num_train_steps: int,
    default_save_interval: int,
) -> Path:
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
            "default_num_train_steps": int(default_num_train_steps),
            "num_train_steps": int(default_num_train_steps),
            "num_train_steps_source": "stage_default",
            "default_save_interval": int(default_save_interval),
            "save_interval": int(default_save_interval),
            "save_interval_source": "stage_default",
        },
    )
    return checkpoint_dir


def write_minimal_recap_ready_dataset(
    root: Path,
    *,
    critic_checkpoint_ref: str,
    informative_positive_reweight: bool = False,
    correction_segments: int = 0,
    skip_informative_positive_reweight_for_corrections: bool = False,
) -> Path:
    dataset_dir = root / "prepared_dataset"
    if (
        informative_positive_reweight
        and skip_informative_positive_reweight_for_corrections
    ):
        raise ValueError(
            "fixture cannot enable informative reweight and correction skip simultaneously"
        )
    duplicates_per_positive_episode = (
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_DUPLICATES_PER_EPISODE
        if informative_positive_reweight
        else 0
    )
    total_episodes = 1 + duplicates_per_positive_episode
    total_frames = 2 * total_episodes
    positive_indicator_count = 1 + duplicates_per_positive_episode
    positive_indicator_fraction = positive_indicator_count / total_frames
    recap_contract: dict[str, object] = {
        "contract_version": "full_recap_continuous_adv_v2",
        "value_source": "critic",
        "value_scale": "raw_return",
        "critic_checkpoint_ref": critic_checkpoint_ref,
        "critic_dir": critic_checkpoint_ref,
        "critic_metrics_path": str(
            Path(critic_checkpoint_ref).resolve().parent / "critic_metrics.json"
        ),
        "critic_provenance_path": str(
            Path(critic_checkpoint_ref).resolve().parent / "critic_provenance.json"
        ),
        "value_adapter": "task_normalized_return_to_raw_return",
        "epsilon_source": "per_task_quantile:prompt_raw:q=0.7",
        "indicator_dropout_p": 0.3,
        "human_correction_override": True,
    }
    informative_policy: dict[str, object] | None = None
    if informative_positive_reweight:
        informative_policy = {
            "applied": True,
            "enabled": True,
            "policy_name": data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_POLICY_NAME,
            "applies_to_stage": "recap_informative",
            "positive_indicator_value": 1,
            "duplication_unit": "episode",
            "positive_episode_selection": "episode_contains_positive_indicator_row",
            "duplicates_per_positive_episode": duplicates_per_positive_episode,
            "source_dataset_dir": str(dataset_dir.resolve()),
            "source_total_episodes": 1,
            "source_positive_episode_count": 1,
            "source_total_rows": 2,
            "source_positive_indicator_count": 1,
            "source_positive_indicator_fraction": 0.5,
            "effective_total_episodes": total_episodes,
            "effective_positive_episode_count": total_episodes,
            "effective_total_rows": total_frames,
            "effective_positive_indicator_count": positive_indicator_count,
            "effective_positive_indicator_fraction": positive_indicator_fraction,
        }
        if correction_segments > 0:
            informative_policy["correction_aware"] = True
            informative_policy["correction_signal"] = {
                "corrections_added": int(correction_segments),
                "dataset_mix_correction_segments": int(correction_segments),
                "episode_level_correction_rows_present": True,
            }
            informative_policy["source_positive_correction_episode_count"] = 1
            informative_policy["duplicated_correction_episode_count"] = (
                duplicates_per_positive_episode
            )
        recap_contract[data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY] = (
            informative_policy
        )
    elif skip_informative_positive_reweight_for_corrections:
        informative_policy = {
            "enabled": False,
            "applied": False,
            "policy_name": data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_POLICY_NAME,
            "applies_to_stage": "recap_informative",
            "skip_reason": (
                data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_SKIP_REASON_CORRECTION_AUGMENTED
            ),
            "source_dataset_dir": str(dataset_dir.resolve()),
            "correction_signal": {
                "corrections_added": int(correction_segments),
                "dataset_mix_correction_segments": int(correction_segments),
                "episode_level_correction_rows_present": True,
            },
        }
        recap_contract[data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY] = (
            informative_policy
        )
    _write_json(
        dataset_dir / "meta" / "info.json",
        {
            "schema_version": "openpi_libero_official_8d_recap_relabels_v1",
            "route_id": "official_native_8d_recap_relabels_v1",
            **carrier_text_v1_handoff_metadata(),
            "source_dataset_name": "physical_intelligence_libero_official_8d",
            "source_dataset_dir": str((root / "official_source").resolve()),
            "corrections_added": int(correction_segments),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 10,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "dataset_mix": {
                "canonical_demo": {"episodes": 1, "source": "official_native_8d"},
                "autonomous": {"episodes": 0, "successes": 0, "failures": 0},
                "correction": {
                    "segments": int(correction_segments),
                    "forced_positive": True,
                },
            },
            "recap_advantage_input_contract": recap_contract,
            "features": {
                "observation.images.ego_view": {"dtype": "image", "shape": [8, 8, 3]},
                "observation.images.wrist_view": {"dtype": "image", "shape": [8, 8, 3]},
                "observation.state": {"dtype": "float32", "shape": [8]},
                "action": {"dtype": "float32", "shape": [7]},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
                "annotation.human.task_description": {"dtype": "int64", "shape": [1]},
                "annotation.human.action.task_description": {
                    "dtype": "int64",
                    "shape": [1],
                },
                "recap_m2.t": {"dtype": "int64", "shape": [1]},
                "recap_m2.return_G": {"dtype": "float32", "shape": [1]},
                "recap_m2.value_V": {"dtype": "float32", "shape": [1]},
                "recap_m2.advantage_A": {"dtype": "float32", "shape": [1]},
                "recap_m2.advantage_input": {"dtype": "float32", "shape": [1]},
                "recap_m2.epsilon_l": {"dtype": "float32", "shape": [1]},
                "recap_m2.indicator_I": {"dtype": "int64", "shape": [1]},
                "recap_m2.prompt_raw": {"dtype": "string", "shape": [1]},
                "recap_m2.prompt_conditioned": {"dtype": "string", "shape": [1]},
            },
        },
    )
    _write_json(
        dataset_dir / "meta" / "stats.json",
        {
            "observation.images.ego_view": {},
            "observation.images.wrist_view": {},
            "observation.state": {},
            "action": {},
            "timestamp": {},
            "frame_index": {},
            "episode_index": {},
            "index": {},
            "task_index": {},
        },
    )
    _write_json(
        dataset_dir / "meta" / "modality.json",
        {
            "video": {
                "ego_view": {"original_key": "observation.images.ego_view"},
                "wrist_view": {"original_key": "observation.images.wrist_view"},
            },
            "state": {
                "libero_state": {
                    "start": 0,
                    "end": 8,
                    "original_key": "observation.state",
                }
            },
            "action": {
                "libero_action": {
                    "start": 0,
                    "end": 7,
                    "original_key": "action",
                }
            },
            "annotation": {
                "human.task_description": {
                    "original_key": "annotation.human.task_description"
                },
                "human.action.task_description": {
                    "original_key": "annotation.human.action.task_description"
                },
            },
        },
    )
    _write_jsonl(
        dataset_dir / "meta" / "tasks.jsonl",
        [{"task": "put the bowl on the plate", "task_index": 0}],
    )
    _write_jsonl(
        dataset_dir / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": 0,
                "tasks": ["put the bowl on the plate"],
                "length": 2,
            },
            *(
                [
                    {
                        "episode_index": episode_index,
                        "tasks": ["put the bowl on the plate"],
                        "length": 2,
                        "informative_positive_reweight_duplicate": True,
                        "informative_positive_reweight_source_episode_index": 0,
                    }
                    for episode_index in range(1, total_episodes)
                ]
                if informative_positive_reweight
                else []
            ),
        ],
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
        "fixture_episode_universe_hash_sha256\n",
        encoding="utf-8",
    )
    _write_json(
        dataset_dir / "materialization_report.json",
        {
            "schema_version": "openpi_libero_official_8d_recap_relabels_report_v1",
            "route_id": "official_native_8d_recap_relabels_v1",
            "final_status": "materialized",
            "merged_correction_override_count": int(correction_segments),
            "epsilon_source": "per_task_quantile:prompt_raw:q=0.7",
            "selected_episode_count": total_episodes,
            "selected_frame_count": total_frames,
            "positive_indicator_count": positive_indicator_count,
            "positive_indicator_fraction": positive_indicator_fraction,
            "advantage_contract": {
                "critic_checkpoint_ref": critic_checkpoint_ref,
                "indicator_dropout_p": 0.3,
                "human_correction_override": True,
            },
            **(
                {
                    data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY: informative_policy,
                }
                if informative_policy is not None
                else {}
            ),
        },
    )
    source_frame = pd.DataFrame(
        {
            "action": [[0.1] * 7, [0.2] * 7],
            "episode_index": [0, 0],
            "observation.state": [[0.0] * 8, [1.0] * 8],
            "timestamp": [0.0, 0.1],
            "frame_index": [0, 1],
            "index": [0, 1],
            "recap_m2.advantage_A": [0.5, -0.5],
            "recap_m2.advantage_input": [0.25, -0.25],
            "recap_m2.indicator_I": [1, 0],
            "recap_m2.prompt_conditioned": [
                "put the bowl on the plate\nAdvantage: positive",
                "put the bowl on the plate\nAdvantage: negative",
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
    source_frame.to_parquet(parquet_path, index=False)
    if informative_positive_reweight:
        for episode_index in range(1, total_episodes):
            duplicated_frame = source_frame.copy()
            duplicated_frame["episode_index"] = [episode_index, episode_index]
            duplicated_frame["index"] = [2 * episode_index, (2 * episode_index) + 1]
            duplicated_parquet_path = (
                dataset_dir
                / "data"
                / "chunk-000"
                / f"episode_{episode_index:06d}.parquet"
            )
            duplicated_frame.to_parquet(duplicated_parquet_path, index=False)
    return dataset_dir


def patch_stage_train_dependencies(
    *,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_dir: Path,
    export_dir: Path,
    critic_checkpoint_ref: str,
) -> None:
    from work.openpi.recap.dataset import resolve_recap_dataset

    def _fake_prepare_stage_training_dataset(
        *,
        dataset_dir: str | Path,
        stage_config: RepairedStageConfig,
        critic_checkpoint_dir: str | Path,
        prepared_dataset_dir: str | Path | None = None,
        episode_limit: int | None = None,
    ) -> PreparedStageDataset:
        del episode_limit
        assert str(Path(critic_checkpoint_dir).resolve()) == str(
            Path(critic_checkpoint_ref).resolve()
        )
        resolved_prepared_dataset_dir = (
            Path(prepared_dataset_dir).resolve()
            if prepared_dataset_dir is not None
            else prepared_dataset_dir
        )
        assert resolved_prepared_dataset_dir is not None
        preview_consumer_mode = stage_config.consumer_mode
        if stage_config.indicator_mode_train == "informative":
            preview_consumer_mode = RECAP_RELABEL_CONSUMER_MODE
        elif stage_config.indicator_mode_train == "shuffled":
            preview_consumer_mode = SHUFFLED_ADV_DIAG_CONSUMER_MODE
        bundle = resolve_recap_dataset(
            resolved_prepared_dataset_dir,
            consumer_mode=preview_consumer_mode,
            fixed_indicator_mode=stage_config.fixed_indicator_mode,
        )
        bundle = replace(
            bundle,
            consumer_mode=stage_config.consumer_mode,
            fixed_indicator_mode=stage_config.fixed_indicator_mode,
        )
        return PreparedStageDataset(
            dataset_dir=resolved_prepared_dataset_dir,
            dataset_bundle=bundle,
            source_dataset_dir=Path(dataset_dir).resolve(),
            materialization_report_path=(
                resolved_prepared_dataset_dir / "materialization_report.json"
            ),
            prepared_from_source=True,
        )

    def _fake_run_real_variant_training_export(
        request: RealVariantExportRequest,
    ) -> RealVariantExportBundle:
        return RealVariantExportBundle(
            export_dir=export_dir,
            runtime_log_path=request.runtime_dir / "fake_real_variant_training.log",
        )

    monkeypatch.setattr(
        recap_train_script,
        "prepare_stage_training_dataset",
        _fake_prepare_stage_training_dataset,
    )
    monkeypatch.setattr(
        recap_train_script,
        "run_real_variant_training_export",
        _fake_run_real_variant_training_export,
    )


def run_stage_train(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    informative_positive_reweight: bool | None = None,
    correction_segments: int = 0,
    skip_informative_positive_reweight_for_corrections: bool = False,
) -> tuple[Path, Path]:
    stage_config = resolve_repaired_stage_config(stage)
    critic_checkpoint = tmp_path / "critic" / "best"
    critic_checkpoint.mkdir(parents=True, exist_ok=True)
    use_informative_positive_reweight = informative_positive_reweight
    if use_informative_positive_reweight is None:
        use_informative_positive_reweight = (
            stage == "recap_informative"
            and not skip_informative_positive_reweight_for_corrections
        )
    prepared_dataset_dir = write_minimal_recap_ready_dataset(
        tmp_path,
        critic_checkpoint_ref=str(critic_checkpoint),
        informative_positive_reweight=use_informative_positive_reweight,
        correction_segments=correction_segments,
        skip_informative_positive_reweight_for_corrections=(
            skip_informative_positive_reweight_for_corrections
        ),
    )
    export_dir = write_fake_real_export(
        tmp_path / "source",
        default_num_train_steps=stage_config.default_num_train_steps,
        default_save_interval=stage_config.default_save_interval,
    )
    patch_stage_train_dependencies(
        monkeypatch=monkeypatch,
        prepared_dataset_dir=prepared_dataset_dir,
        export_dir=export_dir,
        critic_checkpoint_ref=str(critic_checkpoint),
    )
    gate_eval_manifest = write_gate_eval_manifest(tmp_path)
    output_dir = tmp_path / stage
    rc = recap_train_script.main(
        [
            "--stage",
            stage,
            "--dataset-dir",
            str(tmp_path / "official_source"),
            "--critic-checkpoint",
            str(critic_checkpoint),
            "--output-dir",
            str(output_dir),
            "--gate-eval-manifest",
            str(gate_eval_manifest),
            "--prepared-dataset-dir",
            str(prepared_dataset_dir),
        ]
    )
    assert rc == 0
    return output_dir, export_dir


def test_recap_train_manifest_records_repaired_stage_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir, _ = run_stage_train(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        stage="sft_fixed_positive",
    )
    train_manifest = json.loads(
        (output_dir / "train_manifest.json").read_text(encoding="utf-8")
    )
    training_route = train_manifest["training_route"]

    assert train_manifest["stage"] == "sft_fixed_positive"
    assert train_manifest["critic_checkpoint_ref"].endswith("/critic/best")
    assert train_manifest["indicator_mode_train"] == "fixed_positive"
    assert train_manifest["indicator_dropout_p"] == 0.3
    assert train_manifest["epsilon_source"] == "per_task_quantile:prompt_raw:q=0.7"
    assert train_manifest["human_correction_override"] is True
    assert train_manifest["default_num_train_steps"] == 1
    assert train_manifest["effective_num_train_steps"] == 1
    assert train_manifest["num_train_steps_source"] == "stage_default"
    assert train_manifest["default_save_interval"] == 1
    assert train_manifest["effective_save_interval"] == 1
    assert train_manifest["save_interval_source"] == "stage_default"
    assert training_route["indicator_mode_train"] == "fixed_positive"
    assert training_route["consumer_mode"] == "fixed_positive"
    assert training_route["fixed_indicator_mode"] == "positive"
    assert training_route["indicator_dropout_p"] == 0.3
    assert training_route["epsilon_source"] == "per_task_quantile:prompt_raw:q=0.7"
    assert training_route["human_correction_override"] is True
    assert training_route["default_num_train_steps"] == 1
    assert training_route["effective_num_train_steps"] == 1
    assert training_route["num_train_steps_source"] == "stage_default"
    assert training_route["default_save_interval"] == 1
    assert training_route["effective_save_interval"] == 1
    assert training_route["save_interval_source"] == "stage_default"


def test_recap_informative_manifest_records_higher_budget_with_final_only_save_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir, _ = run_stage_train(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        stage="recap_informative",
    )
    train_manifest = json.loads(
        (output_dir / "train_manifest.json").read_text(encoding="utf-8")
    )
    training_route = train_manifest["training_route"]
    duplicates_per_positive_episode = (
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_DUPLICATES_PER_EPISODE
    )
    expected_total_episodes = 1 + duplicates_per_positive_episode
    expected_total_rows = 2 * expected_total_episodes

    assert train_manifest["stage"] == "recap_informative"
    assert (
        train_manifest["default_num_train_steps"]
        == RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS
    )
    assert (
        train_manifest["effective_num_train_steps"]
        == RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS
    )
    assert train_manifest["num_train_steps_source"] == "stage_default"
    assert (
        train_manifest["default_save_interval"]
        == RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL
    )
    assert (
        train_manifest["effective_save_interval"]
        == RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL
    )
    assert train_manifest["save_interval_source"] == "stage_default"
    assert (
        train_manifest["effective_save_interval"]
        == train_manifest["effective_num_train_steps"]
    )
    assert (
        training_route["default_num_train_steps"]
        == RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS
    )
    assert (
        training_route["effective_num_train_steps"]
        == RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS
    )
    assert (
        training_route["default_save_interval"]
        == RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL
    )
    assert (
        training_route["effective_save_interval"]
        == RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL
    )
    assert training_route["save_interval_source"] == "stage_default"
    informative_policy = train_manifest[
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY
    ]
    training_route_policy = training_route[
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY
    ]
    train_source_policy = train_manifest["train_source"][
        "recap_advantage_input_contract"
    ][data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY]
    assert (
        informative_policy["policy_name"]
        == data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_POLICY_NAME
    )
    assert informative_policy["applies_to_stage"] == "recap_informative"
    assert informative_policy["duplication_unit"] == "episode"
    assert (
        informative_policy["duplicates_per_positive_episode"]
        == duplicates_per_positive_episode
    )
    assert informative_policy["source_total_episodes"] == 1
    assert informative_policy["source_positive_episode_count"] == 1
    assert informative_policy["effective_total_episodes"] == expected_total_episodes
    assert (
        informative_policy["effective_positive_episode_count"]
        == expected_total_episodes
    )
    assert informative_policy["effective_total_rows"] == expected_total_rows
    assert (
        informative_policy["effective_positive_indicator_count"]
        == expected_total_episodes
    )
    assert informative_policy["effective_positive_indicator_fraction"] == 0.5
    assert training_route_policy == informative_policy
    assert train_source_policy == informative_policy


def test_recap_informative_manifest_records_correction_aware_informative_reweight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir, _ = run_stage_train(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        stage="recap_informative",
        correction_segments=3,
    )
    train_manifest = json.loads(
        (output_dir / "train_manifest.json").read_text(encoding="utf-8")
    )
    training_route = train_manifest["training_route"]
    informative_policy = train_manifest[
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY
    ]
    training_route_policy = training_route[
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY
    ]
    train_source_policy = train_manifest["train_source"][
        "recap_advantage_input_contract"
    ][data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY]

    assert train_manifest["stage"] == "recap_informative"
    duplicates_per_positive_episode = (
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_DUPLICATES_PER_EPISODE
    )
    expected_total_episodes = 1 + duplicates_per_positive_episode
    expected_total_rows = 2 * expected_total_episodes
    assert train_manifest["train_source"]["total_rows"] == expected_total_rows
    assert informative_policy["policy_name"] == (
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_POLICY_NAME
    )
    assert informative_policy["applies_to_stage"] == "recap_informative"
    assert informative_policy["enabled"] is True
    assert informative_policy["applied"] is True
    assert informative_policy["correction_aware"] is True
    assert informative_policy["duplication_unit"] == "episode"
    assert (
        informative_policy["duplicates_per_positive_episode"]
        == duplicates_per_positive_episode
    )
    assert informative_policy["source_dataset_dir"].endswith("/prepared_dataset")
    assert informative_policy["effective_total_episodes"] == expected_total_episodes
    assert informative_policy["effective_total_rows"] == expected_total_rows
    correction_signal = informative_policy["correction_signal"]
    assert correction_signal["corrections_added"] == 3
    assert correction_signal["dataset_mix_correction_segments"] == 3
    assert correction_signal["episode_level_correction_rows_present"] is True
    assert informative_policy["source_positive_correction_episode_count"] == 1
    assert (
        informative_policy["duplicated_correction_episode_count"]
        == duplicates_per_positive_episode
    )
    assert training_route_policy == informative_policy
    assert train_source_policy == informative_policy


def test_recap_train_manifest_keeps_omit_control_distinct_from_fixed_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    omit_output_dir, _ = run_stage_train(
        tmp_path=tmp_path / "omit_case",
        monkeypatch=monkeypatch,
        stage="omit_control",
    )
    fixed_output_dir, _ = run_stage_train(
        tmp_path=tmp_path / "fixed_case",
        monkeypatch=monkeypatch,
        stage="sft_fixed_positive",
    )

    omit_manifest = json.loads(
        (omit_output_dir / "train_manifest.json").read_text(encoding="utf-8")
    )
    fixed_manifest = json.loads(
        (fixed_output_dir / "train_manifest.json").read_text(encoding="utf-8")
    )

    assert omit_manifest["stage"] == "omit_control"
    assert omit_manifest["indicator_mode_train"] == "omit"
    assert omit_manifest["stage_provenance"]["fixed_indicator_mode"] == "omit"
    assert fixed_manifest["indicator_mode_train"] == "fixed_positive"
    assert fixed_manifest["stage_provenance"]["fixed_indicator_mode"] == "positive"
    assert (
        omit_manifest["indicator_mode_train"] != fixed_manifest["indicator_mode_train"]
    )
    assert data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY not in omit_manifest
    assert data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY not in fixed_manifest
