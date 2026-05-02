from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import cast

from _pytest.monkeypatch import MonkeyPatch
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap import dataset_aggregation  # noqa: E402
from work.openpi.recap import data_transforms  # noqa: E402
from work.openpi.recap import prompt_builder  # noqa: E402
from work.recap.lerobot_export import dataset_export  # noqa: E402
import work.openpi.pipelines.recap.merge as merge_script  # noqa: E402
from tests.openpi.test_recap_collection_schema import run_collection_fixture  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _count_loader_visible_parquet_files_and_rows(dataset_dir: Path) -> tuple[int, int]:
    parquet_count = 0
    row_count = 0
    for root, _, files in os.walk(dataset_dir / "data", followlinks=False):
        for name in sorted(files):
            if not name.endswith(".parquet"):
                continue
            parquet_count += 1
            frame = pd.read_parquet(Path(root) / name)
            row_count += int(len(frame))
    return parquet_count, row_count


def write_recap_ready_demo_sibling(
    demo_dir: Path,
    *,
    critic_checkpoint_ref: str = "adapter_required",
    omit_prompt_feature_keys: bool = False,
) -> Path:
    recap_dir = demo_dir.parent / f"{demo_dir.name}_recap_relabels_v1"
    info = dataset_aggregation.read_json(demo_dir / "meta" / "info.json")
    tasks = dataset_aggregation.read_jsonl(demo_dir / "meta" / "tasks.jsonl")
    episodes = dataset_aggregation.read_jsonl(demo_dir / "meta" / "episodes.jsonl")
    recap_info = {
        **dict(info),
        "route_id": dataset_aggregation.OFFICIAL_RECAP_RELABEL_ROUTE_ID,
        "schema_version": "openpi_libero_official_8d_recap_relabels_v1",
        "source_dataset_dir": str(demo_dir.resolve()),
        "source_dataset_name": demo_dir.name,
        "task_text_field": dataset_export.EXPORTER_MAINLINE_TASK_TEXT_FIELD,
        "carrier_route": dataset_export.EXPORTER_CARRIER_ROUTE,
        "carrier_schema_version": dataset_export.EXPORTER_CARRIER_SCHEMA_VERSION,
        "prompt_source_field": dataset_export.EXPORTER_PROMPT_SOURCE_FIELD,
        "prompt_route": prompt_builder.PHASE1_PROMPT_ROUTE,
        "conditioning_mode": prompt_builder.CONDITIONING_MODE,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "action": {"dtype": "float32", "shape": [7]},
            "annotation.human.action.task_description": {
                "dtype": "int64",
                "shape": [1],
            },
            "annotation.human.task_description": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "observation.images.ego_view": {"dtype": "binary", "shape": [1]},
            "observation.images.wrist_view": {"dtype": "binary", "shape": [1]},
            "observation.state": {"dtype": "float32", "shape": [8]},
            "recap_m2.advantage_A": {"dtype": "float32", "shape": [1]},
            "recap_m2.advantage_input": {"dtype": "float32", "shape": [1]},
            "recap_m2.epsilon_l": {"dtype": "float32", "shape": [1]},
            "recap_m2.indicator_I": {"dtype": "int64", "shape": [1]},
            "recap_m2.prompt_conditioned": {"dtype": "string", "shape": [1]},
            "recap_m2.prompt_raw": {"dtype": "string", "shape": [1]},
            "recap_m2.return_G": {"dtype": "float32", "shape": [1]},
            "recap_m2.t": {"dtype": "int64", "shape": [1]},
            "recap_m2.value_V": {"dtype": "float32", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
        "recap_advantage_input_contract": {
            "contract_version": "full_recap_continuous_adv_v2",
            "critic_checkpoint_ref": critic_checkpoint_ref,
            "epsilon_source": "per_task_quantile:prompt_raw:q=0.7",
            "indicator_dropout_p": 0.3,
            "human_correction_override": True,
        },
        "total_frames": len(episodes),
        "splits": {"train": f"0:{len(episodes)}"},
    }
    if omit_prompt_feature_keys:
        feature_payload = cast(dict[str, object], recap_info["features"])
        feature_payload.pop("recap_m2.prompt_raw", None)
        feature_payload.pop("recap_m2.prompt_conditioned", None)
    _write_json(recap_dir / "meta" / "info.json", recap_info)
    _write_json(
        recap_dir / "meta" / "stats.json",
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
        recap_dir / "meta" / "modality.json",
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
        },
    )
    dataset_aggregation.write_jsonl(recap_dir / "meta" / "tasks.jsonl", tasks)
    dataset_aggregation.write_jsonl(recap_dir / "meta" / "episodes.jsonl", episodes)
    _write_json(
        recap_dir / "materialization_report.json",
        {
            "schema_version": "openpi_libero_official_8d_recap_relabels_report_v1",
            "route_id": dataset_aggregation.OFFICIAL_RECAP_RELABEL_ROUTE_ID,
            "final_status": "materialized",
            "source_dataset_dir": str(demo_dir.resolve()),
            "output_dataset_dir": str(recap_dir.resolve()),
        },
    )
    task_index_by_prompt = {
        str(cast(dict[str, object], row)["task"]): int(
            cast(int | str, cast(dict[str, object], row)["task_index"])
        )
        for row in tasks
    }
    for episode in episodes:
        episode_index = int(cast(int | str, episode["episode_index"]))
        prompt_raw = str(cast(list[object], episode["tasks"])[0])
        task_index = int(task_index_by_prompt[prompt_raw])
        frame = pd.DataFrame(
            {
                "action": [[0.1] * 7],
                "episode_index": [episode_index],
                "observation.state": [[float(episode_index)] * 8],
                "observation.images.ego_view": [b"ego"],
                "observation.images.wrist_view": [b"wrist"],
                "timestamp": [0.0],
                "frame_index": [0],
                "index": [episode_index],
                "task_index": [task_index],
                "annotation.human.task_description": [task_index],
                "annotation.human.action.task_description": [task_index],
                "recap_m2.t": [0],
                "recap_m2.return_G": [0.0],
                "recap_m2.value_V": [0.0],
                "recap_m2.advantage_A": [0.5],
                "recap_m2.advantage_input": [0.5],
                "recap_m2.epsilon_l": [0.0],
                "recap_m2.indicator_I": [1],
                "recap_m2.prompt_raw": [prompt_raw],
                "recap_m2.prompt_conditioned": [prompt_raw + "\nAdvantage: positive"],
            }
        )
        parquet_path = (
            recap_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(parquet_path, engine="pyarrow", index=False)
    return recap_dir


def _write_mixed_indicator_recap_ready_dataset(root: Path) -> Path:
    dataset_dir = root / "mixed_indicator_prepared_dataset"
    _write_json(
        dataset_dir / "meta" / "info.json",
        {
            "schema_version": "openpi_libero_official_8d_recap_relabels_v1",
            "route_id": dataset_aggregation.OFFICIAL_RECAP_RELABEL_ROUTE_ID,
            "source_dataset_name": "fixture_official_source",
            "source_dataset_dir": str((root / "official_source").resolve()),
            "task_text_field": dataset_export.EXPORTER_MAINLINE_TASK_TEXT_FIELD,
            "carrier_route": dataset_export.EXPORTER_CARRIER_ROUTE,
            "carrier_schema_version": dataset_export.EXPORTER_CARRIER_SCHEMA_VERSION,
            "prompt_source_field": dataset_export.EXPORTER_PROMPT_SOURCE_FIELD,
            "prompt_route": prompt_builder.PHASE1_PROMPT_ROUTE,
            "conditioning_mode": prompt_builder.CONDITIONING_MODE,
            "total_episodes": 1,
            "total_frames": 3,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 10,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "recap_advantage_input_contract": {
                "contract_version": "full_recap_continuous_adv_v2",
                "critic_checkpoint_ref": str((root / "critic" / "best").resolve()),
                "epsilon_source": "per_task_quantile:prompt_raw:q=0.7",
                "indicator_dropout_p": 0.3,
                "human_correction_override": True,
            },
            "features": {
                "action": {"dtype": "float32", "shape": [7]},
                "annotation.human.action.task_description": {
                    "dtype": "int64",
                    "shape": [1],
                },
                "annotation.human.task_description": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "observation.images.ego_view": {"dtype": "binary", "shape": [1]},
                "observation.images.wrist_view": {"dtype": "binary", "shape": [1]},
                "observation.state": {"dtype": "float32", "shape": [8]},
                "recap_m2.advantage_A": {"dtype": "float32", "shape": [1]},
                "recap_m2.advantage_input": {"dtype": "float32", "shape": [1]},
                "recap_m2.epsilon_l": {"dtype": "float32", "shape": [1]},
                "recap_m2.indicator_I": {"dtype": "int64", "shape": [1]},
                "recap_m2.prompt_conditioned": {"dtype": "string", "shape": [1]},
                "recap_m2.prompt_raw": {"dtype": "string", "shape": [1]},
                "recap_m2.return_G": {"dtype": "float32", "shape": [1]},
                "recap_m2.t": {"dtype": "int64", "shape": [1]},
                "recap_m2.value_V": {"dtype": "float32", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
                "timestamp": {"dtype": "float32", "shape": [1]},
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
        },
    )
    dataset_aggregation.write_jsonl(
        dataset_dir / "meta" / "tasks.jsonl",
        [{"task": "put the mug on the plate", "task_index": 0}],
    )
    dataset_aggregation.write_jsonl(
        dataset_dir / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "tasks": ["put the mug on the plate"], "length": 3}],
    )
    _write_json(
        dataset_dir / "materialization_report.json",
        {
            "schema_version": "openpi_libero_official_8d_recap_relabels_report_v1",
            "route_id": dataset_aggregation.OFFICIAL_RECAP_RELABEL_ROUTE_ID,
            "final_status": "materialized",
            "source_dataset_dir": str((root / "official_source").resolve()),
            "output_dataset_dir": str(dataset_dir.resolve()),
            "selected_frame_count": 3,
            "positive_indicator_count": 1,
            "positive_indicator_fraction": 1.0 / 3.0,
        },
    )
    frame = pd.DataFrame(
        {
            "action": [[0.1] * 7, [0.2] * 7, [0.3] * 7],
            "episode_index": [0, 0, 0],
            "observation.state": [[0.0] * 8, [1.0] * 8, [2.0] * 8],
            "observation.images.ego_view": [b"ego0", b"ego1", b"ego2"],
            "observation.images.wrist_view": [b"wrist0", b"wrist1", b"wrist2"],
            "timestamp": [0.0, 0.1, 0.2],
            "frame_index": [0, 1, 2],
            "index": [0, 1, 2],
            "task_index": [0, 0, 0],
            "annotation.human.task_description": [0, 0, 0],
            "annotation.human.action.task_description": [0, 0, 0],
            "recap_m2.t": [0, 1, 2],
            "recap_m2.return_G": [1.0, -1.0, -1.0],
            "recap_m2.value_V": [0.0, 0.0, 0.0],
            "recap_m2.advantage_A": [0.5, -0.5, -0.25],
            "recap_m2.advantage_input": [0.5, -0.5, -0.25],
            "recap_m2.epsilon_l": [0.0, 0.0, 0.0],
            "recap_m2.indicator_I": [1, 0, 0],
            "recap_m2.prompt_raw": [
                "put the mug on the plate",
                "put the mug on the plate",
                "put the mug on the plate",
            ],
            "recap_m2.prompt_conditioned": [
                "put the mug on the plate\nAdvantage: positive",
                "put the mug on the plate\nAdvantage: negative",
                "put the mug on the plate\nAdvantage: negative",
            ],
        }
    )
    parquet_path = dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, engine="pyarrow", index=False)
    return dataset_dir


def _write_chunked_positive_recap_ready_dataset(root: Path) -> Path:
    dataset_dir = root / "chunked_positive_prepared_dataset"
    _write_json(
        dataset_dir / "meta" / "info.json",
        {
            "schema_version": "openpi_libero_official_8d_recap_relabels_v1",
            "route_id": dataset_aggregation.OFFICIAL_RECAP_RELABEL_ROUTE_ID,
            "source_dataset_name": "fixture_official_source",
            "source_dataset_dir": str((root / "official_source").resolve()),
            "task_text_field": dataset_export.EXPORTER_MAINLINE_TASK_TEXT_FIELD,
            "carrier_route": dataset_export.EXPORTER_CARRIER_ROUTE,
            "carrier_schema_version": dataset_export.EXPORTER_CARRIER_SCHEMA_VERSION,
            "prompt_source_field": dataset_export.EXPORTER_PROMPT_SOURCE_FIELD,
            "prompt_route": prompt_builder.PHASE1_PROMPT_ROUTE,
            "conditioning_mode": prompt_builder.CONDITIONING_MODE,
            "total_episodes": 1,
            "total_frames": 3,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 2,
            "fps": 10,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "recap_advantage_input_contract": {
                "contract_version": "full_recap_continuous_adv_v2",
                "critic_checkpoint_ref": str((root / "critic" / "best").resolve()),
                "epsilon_source": "per_task_quantile:prompt_raw:q=0.7",
                "indicator_dropout_p": 0.3,
                "human_correction_override": True,
            },
            "features": {
                "action": {"dtype": "float32", "shape": [7]},
                "annotation.human.action.task_description": {
                    "dtype": "int64",
                    "shape": [1],
                },
                "annotation.human.task_description": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "observation.images.ego_view": {"dtype": "binary", "shape": [1]},
                "observation.images.wrist_view": {"dtype": "binary", "shape": [1]},
                "observation.state": {"dtype": "float32", "shape": [8]},
                "recap_m2.advantage_A": {"dtype": "float32", "shape": [1]},
                "recap_m2.advantage_input": {"dtype": "float32", "shape": [1]},
                "recap_m2.epsilon_l": {"dtype": "float32", "shape": [1]},
                "recap_m2.indicator_I": {"dtype": "int64", "shape": [1]},
                "recap_m2.prompt_conditioned": {"dtype": "string", "shape": [1]},
                "recap_m2.prompt_raw": {"dtype": "string", "shape": [1]},
                "recap_m2.return_G": {"dtype": "float32", "shape": [1]},
                "recap_m2.t": {"dtype": "int64", "shape": [1]},
                "recap_m2.value_V": {"dtype": "float32", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
                "timestamp": {"dtype": "float32", "shape": [1]},
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
        },
    )
    dataset_aggregation.write_jsonl(
        dataset_dir / "meta" / "tasks.jsonl",
        [{"task": "put the mug on the plate", "task_index": 0}],
    )
    dataset_aggregation.write_jsonl(
        dataset_dir / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "tasks": ["put the mug on the plate"], "length": 3}],
    )
    _write_json(
        dataset_dir / "materialization_report.json",
        {
            "schema_version": "openpi_libero_official_8d_recap_relabels_report_v1",
            "route_id": dataset_aggregation.OFFICIAL_RECAP_RELABEL_ROUTE_ID,
            "final_status": "materialized",
            "source_dataset_dir": str((root / "official_source").resolve()),
            "output_dataset_dir": str(dataset_dir.resolve()),
            "selected_episode_count": 1,
            "selected_frame_count": 3,
            "positive_indicator_count": 1,
            "positive_indicator_fraction": 1.0 / 3.0,
        },
    )
    frame = pd.DataFrame(
        {
            "action": [[0.1] * 7, [0.2] * 7, [0.3] * 7],
            "episode_index": [0, 0, 0],
            "observation.state": [[0.0] * 8, [1.0] * 8, [2.0] * 8],
            "observation.images.ego_view": [b"ego0", b"ego1", b"ego2"],
            "observation.images.wrist_view": [b"wrist0", b"wrist1", b"wrist2"],
            "timestamp": [0.0, 0.1, 0.2],
            "frame_index": [0, 1, 2],
            "index": [0, 1, 2],
            "task_index": [0, 0, 0],
            "annotation.human.task_description": [0, 0, 0],
            "annotation.human.action.task_description": [0, 0, 0],
            "recap_m2.t": [0, 1, 2],
            "recap_m2.return_G": [1.0, -1.0, -1.0],
            "recap_m2.value_V": [0.0, 0.0, 0.0],
            "recap_m2.advantage_A": [0.5, -0.5, -0.25],
            "recap_m2.advantage_input": [0.5, -0.5, -0.25],
            "recap_m2.epsilon_l": [0.0, 0.0, 0.0],
            "recap_m2.indicator_I": [1, 0, 0],
            "recap_m2.prompt_raw": [
                "put the mug on the plate",
                "put the mug on the plate",
                "put the mug on the plate",
            ],
            "recap_m2.prompt_conditioned": [
                "put the mug on the plate\nAdvantage: positive",
                "put the mug on the plate\nAdvantage: negative",
                "put the mug on the plate\nAdvantage: negative",
            ],
        }
    )
    parquet_path = dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, engine="pyarrow", index=False)
    return dataset_dir


def test_merge_script_builds_dataset_mix_and_lineage_metadata(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    demo_dir, _, collect_dir, _ = run_collection_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        success_pattern=(True, False, True),
    )
    sibling_dir = write_recap_ready_demo_sibling(
        demo_dir,
        omit_prompt_feature_keys=True,
    )
    output_dir = tmp_path / "merged_dataset"

    assert dataset_aggregation._is_recap_ready_dataset_dir(sibling_dir) is True
    assert (
        dataset_aggregation._resolve_recap_relabel_source_dir(demo_dir)
        == sibling_dir.resolve()
    )

    exit_code = merge_script.main(
        [
            "--demo-dir",
            str(demo_dir),
            "--autonomous-dir",
            str(collect_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    merge_manifest = dataset_aggregation.read_json(
        output_dir / dataset_aggregation.MERGE_MANIFEST_NAME
    )
    info = dataset_aggregation.read_json(output_dir / "meta" / "info.json")
    episodes = dataset_aggregation.read_jsonl(output_dir / "meta" / "episodes.jsonl")
    episode_lineage = dataset_aggregation.read_jsonl(
        output_dir / "meta" / dataset_aggregation.MERGED_EPISODE_LINEAGE_NAME
    )

    dataset_mix = cast(dict[str, object], merge_manifest["dataset_mix"])
    canonical_demo = cast(dict[str, object], dataset_mix["canonical_demo"])
    autonomous = cast(dict[str, object], dataset_mix["autonomous"])
    correction = cast(dict[str, object], dataset_mix["correction"])
    info_canonical_source = cast(dict[str, object], info["canonical_demo_source"])
    trainer_ref = Path(
        cast(str, info[dataset_aggregation.MERGED_RECAP_READY_DATASET_REF_KEY])
    )

    assert merge_manifest["route_id"] == dataset_aggregation.MERGED_DATASET_ROUTE_ID
    assert merge_manifest["episodes_added"] == 3
    assert merge_manifest["corrections_added"] == 0
    assert merge_manifest["trainer_visible_total_episodes"] == info["total_episodes"]
    assert canonical_demo["episodes"] == 2
    assert autonomous == {"episodes": 3, "successes": 2, "failures": 1}
    assert correction == {"segments": 0, "forced_positive": True}
    assert info["dataset_mix"] == dataset_mix
    assert info_canonical_source["status"] == "ready"
    assert info["episodes_jsonl"] == str(
        (output_dir / "meta" / "episodes.jsonl").resolve()
    )
    assert info["episode_lineage_jsonl"] == str(
        (
            output_dir / "meta" / dataset_aggregation.MERGED_EPISODE_LINEAGE_NAME
        ).resolve()
    )
    assert (
        trainer_ref
        == (
            output_dir / dataset_aggregation.MERGED_RECAP_READY_DATASET_DIRNAME
        ).resolve()
    )
    assert data_transforms.is_recap_training_dataset(trainer_ref)

    assert info["route_id"] == dataset_aggregation.MERGED_DATASET_ROUTE_ID
    assert info["total_episodes"] == 5
    assert info["total_frames"] == 7
    assert len(episodes) == 5
    assert [row["source_kind"] for row in episodes] == [
        "canonical_demo",
        "canonical_demo",
        "autonomous_trial",
        "autonomous_trial",
        "autonomous_trial",
    ]
    assert [row["episode_index"] for row in episodes[-3:]] == [2, 3, 4]
    assert [row["task_id"] for row in episodes[-3:]] == [0, 1, 0]
    parquet_files = sorted(output_dir.glob("data/chunk-*/episode_*.parquet"))
    assert len(parquet_files) == 5
    source_kinds = [row["source_kind"] for row in episode_lineage]
    assert source_kinds.count("canonical_demo") == 2
    assert source_kinds.count("autonomous_trial") == 3

    trainer_episodes = dataset_aggregation.read_jsonl(
        trainer_ref / "meta" / "episodes.jsonl"
    )
    trainer_info = dataset_aggregation.read_json(trainer_ref / "meta" / "info.json")
    trainer_contract = cast(
        dict[str, object], trainer_info["recap_advantage_input_contract"]
    )
    assert len(trainer_episodes) == 5
    assert trainer_episodes[-1]["source_kind"] == "autonomous_trial"
    assert trainer_contract["epsilon_source"] == "per_task_quantile:prompt_raw:q=0.7"
    assert trainer_contract["indicator_dropout_p"] == 0.3
    assert trainer_contract["human_correction_override"] is True


def test_prepare_stage_training_dataset_reweights_positive_rows_only_for_recap_informative(
    tmp_path: Path,
) -> None:
    dataset_dir = _write_mixed_indicator_recap_ready_dataset(tmp_path)
    critic_checkpoint_ref = tmp_path / "critic" / "best"
    critic_checkpoint_ref.mkdir(parents=True, exist_ok=True)

    informative_prepared = data_transforms.prepare_stage_training_dataset(
        dataset_dir=dataset_dir,
        stage_config=data_transforms.RepairedStageConfig(
            stage="recap_informative",
            variant_name="recap_informative",
            checkpoint_source="fixture",
            consumer_mode="informative",
            fixed_indicator_mode=None,
            indicator_mode_train="informative",
        ),
        critic_checkpoint_dir=critic_checkpoint_ref,
    )
    fixed_prepared = data_transforms.prepare_stage_training_dataset(
        dataset_dir=dataset_dir,
        stage_config=data_transforms.RepairedStageConfig(
            stage="sft_fixed_positive",
            variant_name="sft_fixed_positive",
            checkpoint_source="fixture",
            consumer_mode="fixed_positive",
            fixed_indicator_mode="positive",
            indicator_mode_train="fixed_positive",
        ),
        critic_checkpoint_dir=critic_checkpoint_ref,
    )

    informative_info = dataset_aggregation.read_json(
        informative_prepared.dataset_dir / "meta" / "info.json"
    )
    informative_contract = cast(
        dict[str, object], informative_info["recap_advantage_input_contract"]
    )
    informative_policy = cast(
        dict[str, object],
        informative_contract[data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY],
    )
    informative_report = dataset_aggregation.read_json(
        informative_prepared.dataset_dir / "materialization_report.json"
    )
    informative_episode_rows = dataset_aggregation.read_jsonl(
        informative_prepared.dataset_dir / "meta" / "episodes.jsonl"
    )
    duplicates_per_positive_episode = (
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_DUPLICATES_PER_EPISODE
    )
    expected_total_episodes = 1 + duplicates_per_positive_episode
    expected_total_rows = 3 * expected_total_episodes
    expected_positive_indicator_count = expected_total_episodes
    original_frame = pd.read_parquet(
        informative_prepared.dataset_dir
        / "data"
        / "chunk-000"
        / "episode_000000.parquet"
    )
    duplicated_frames = [
        pd.read_parquet(
            informative_prepared.dataset_dir
            / "data"
            / "chunk-000"
            / f"episode_{episode_index:06d}.parquet"
        )
        for episode_index in range(1, expected_total_episodes)
    ]
    fixed_info = dataset_aggregation.read_json(
        fixed_prepared.dataset_dir / "meta" / "info.json"
    )
    fixed_contract = cast(
        dict[str, object], fixed_info["recap_advantage_input_contract"]
    )

    assert informative_prepared.prepared_from_source is False
    assert informative_prepared.dataset_dir != dataset_dir.resolve()
    assert fixed_prepared.dataset_dir == dataset_dir.resolve()
    assert informative_prepared.dataset_bundle.total_rows == expected_total_rows
    assert (
        informative_prepared.dataset_bundle.indicator_positive_count
        == expected_positive_indicator_count
    )
    assert (
        informative_prepared.dataset_bundle.indicator_negative_count
        == expected_total_rows - expected_positive_indicator_count
    )
    assert informative_prepared.dataset_bundle.indicator_positive_fraction == 1.0 / 3.0
    assert informative_info["total_frames"] == expected_total_rows
    assert informative_info["total_episodes"] == expected_total_episodes
    assert len(informative_episode_rows) == expected_total_episodes
    for duplicate_row in informative_episode_rows[1:]:
        assert duplicate_row["informative_positive_reweight_duplicate"] is True
        assert duplicate_row["informative_positive_reweight_source_episode_index"] == 0
    assert len(original_frame) == 3
    assert list(original_frame["timestamp"]) == [0.0, 0.1, 0.2]
    assert list(original_frame["frame_index"]) == [0, 1, 2]
    assert int(original_frame["recap_m2.indicator_I"].astype(int).sum()) == 1
    for duplicate_offset, duplicated_frame in enumerate(duplicated_frames, start=1):
        expected_start_index = 3 * duplicate_offset
        assert len(duplicated_frame) == 3
        assert list(duplicated_frame["timestamp"]) == [0.0, 0.1, 0.2]
        assert list(duplicated_frame["frame_index"]) == [0, 1, 2]
        assert list(duplicated_frame["episode_index"]) == [duplicate_offset] * 3
        assert list(duplicated_frame["index"]) == [
            expected_start_index,
            expected_start_index + 1,
            expected_start_index + 2,
        ]
        assert list(duplicated_frame["index"]) == sorted(
            duplicated_frame["index"].tolist()
        )
        assert int(duplicated_frame["recap_m2.indicator_I"].astype(int).sum()) == 1
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
    assert informative_policy["source_total_rows"] == 3
    assert informative_policy["source_positive_indicator_count"] == 1
    assert informative_policy["effective_total_episodes"] == expected_total_episodes
    assert (
        informative_policy["effective_positive_episode_count"]
        == expected_total_episodes
    )
    assert informative_policy["effective_total_rows"] == expected_total_rows
    assert (
        informative_policy["effective_positive_indicator_count"]
        == expected_positive_indicator_count
    )
    assert informative_policy["effective_positive_indicator_fraction"] == 1.0 / 3.0
    assert (
        informative_report["positive_indicator_count"]
        == expected_positive_indicator_count
    )
    assert informative_report["positive_indicator_fraction"] == 1.0 / 3.0
    assert (
        cast(
            dict[str, object],
            informative_report[data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY],
        )["effective_total_rows"]
        == expected_total_rows
    )
    assert data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY not in fixed_contract


@pytest.mark.parametrize("break_mode", ["missing", "broken_symlink"])
def test_informative_positive_reweight_rematerializes_stale_chunk001_duplicate(
    tmp_path: Path,
    break_mode: str,
) -> None:
    dataset_dir = _write_chunked_positive_recap_ready_dataset(tmp_path)
    output_dir = data_transforms.build_informative_positive_reweight_dataset_dir(
        dataset_dir
    )

    rematerialized_dir = (
        data_transforms._materialize_informative_positive_reweight_dataset(
            source_dataset_dir=dataset_dir,
            output_dataset_dir=output_dir,
        )
    )
    duplicate_parquet = (
        rematerialized_dir / "data" / "chunk-001" / "episode_000002.parquet"
    )
    assert rematerialized_dir == output_dir
    assert data_transforms._is_informative_positive_reweight_dataset(output_dir) is True
    assert duplicate_parquet.is_file()

    duplicate_parquet.unlink()
    if break_mode == "broken_symlink":
        duplicate_parquet.symlink_to(
            rematerialized_dir / "data" / "chunk-001" / "missing_episode_000002.parquet"
        )

    assert (
        data_transforms._is_informative_positive_reweight_dataset(output_dir) is False
    )

    repaired_dir = data_transforms._materialize_informative_positive_reweight_dataset(
        source_dataset_dir=dataset_dir,
        output_dataset_dir=output_dir,
    )
    repaired_frame = pd.read_parquet(duplicate_parquet)

    assert repaired_dir == output_dir
    assert data_transforms._is_informative_positive_reweight_dataset(output_dir) is True
    assert duplicate_parquet.is_file()
    assert duplicate_parquet.is_symlink() is False
    assert list(repaired_frame["timestamp"]) == [0.0, 0.1, 0.2]
    assert list(repaired_frame["frame_index"]) == [0, 1, 2]
    assert list(repaired_frame["episode_index"]) == [2, 2, 2]
    assert list(repaired_frame["index"]) == [6, 7, 8]
    assert int(repaired_frame["recap_m2.indicator_I"].astype(int).sum()) == 1


def test_trainer_surface_exposes_untouched_chunks_as_loader_visible_directories() -> (
    None
):
    demo_dir = Path("/tmp/unused_demo_for_visibility_check")
    recap_dir = Path("/tmp/unused_demo_for_visibility_check_recap_relabels_v1")
    tmp_root = Path(__file__).resolve().parent / ".tmp_visibility_check"
    if tmp_root.exists():
        import shutil

        shutil.rmtree(tmp_root)
    demo_dir = tmp_root / "official_demo"
    recap_dir = tmp_root / "official_demo_recap_relabels_v1"
    output_dir = tmp_root / "merged_output"
    demo_dir.mkdir(parents=True, exist_ok=True)
    recap_dir.mkdir(parents=True, exist_ok=True)

    task_rows = [{"task_index": 0, "task": "put the mug on the plate"}]
    recap_info = {
        "schema_version": "openpi_libero_official_8d_recap_relabels_v1",
        "route_id": dataset_aggregation.OFFICIAL_RECAP_RELABEL_ROUTE_ID,
        "source_dataset_dir": str(demo_dir.resolve()),
        "source_dataset_name": demo_dir.name,
        "task_text_field": dataset_export.EXPORTER_MAINLINE_TASK_TEXT_FIELD,
        "carrier_route": dataset_export.EXPORTER_CARRIER_ROUTE,
        "carrier_schema_version": dataset_export.EXPORTER_CARRIER_SCHEMA_VERSION,
        "prompt_source_field": dataset_export.EXPORTER_PROMPT_SOURCE_FIELD,
        "prompt_route": prompt_builder.PHASE1_PROMPT_ROUTE,
        "conditioning_mode": prompt_builder.CONDITIONING_MODE,
        "total_episodes": 2,
        "total_frames": 2,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 2,
        "fps": 10,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "recap_advantage_input_contract": {
            "contract_version": "full_recap_continuous_adv_v2",
            "critic_checkpoint_ref": "adapter_required",
            "epsilon_source": "per_task_quantile:prompt_raw:q=0.7",
            "indicator_dropout_p": 0.3,
            "human_correction_override": True,
        },
        "features": {
            "observation.images.ego_view": {"dtype": "binary", "shape": [1]},
            "observation.images.wrist_view": {"dtype": "binary", "shape": [1]},
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
    }
    dataset_aggregation.write_json(recap_dir / "meta" / "info.json", recap_info)
    dataset_aggregation.write_json(
        recap_dir / "meta" / "stats.json",
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
    dataset_aggregation.write_json(
        recap_dir / "meta" / "modality.json",
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
        },
    )
    dataset_aggregation.write_jsonl(recap_dir / "meta" / "tasks.jsonl", task_rows)
    dataset_aggregation.write_jsonl(
        recap_dir / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["put the mug on the plate"], "length": 1},
            {"episode_index": 1, "tasks": ["put the mug on the plate"], "length": 1},
        ],
    )
    dataset_aggregation.write_json(
        recap_dir / "materialization_report.json",
        {
            "schema_version": "openpi_libero_official_8d_recap_relabels_report_v1",
            "route_id": dataset_aggregation.OFFICIAL_RECAP_RELABEL_ROUTE_ID,
            "final_status": "materialized",
        },
    )
    for episode_index in (0, 1):
        frame = pd.DataFrame(
            {
                "action": [[0.1] * 7],
                "episode_index": [episode_index],
                "observation.state": [[float(episode_index)] * 8],
                "observation.images.ego_view": [b"ego"],
                "observation.images.wrist_view": [b"wrist"],
                "timestamp": [0.0],
                "frame_index": [0],
                "index": [episode_index],
                "task_index": [0],
                "annotation.human.task_description": [0],
                "annotation.human.action.task_description": [0],
                "recap_m2.t": [0],
                "recap_m2.return_G": [0.0],
                "recap_m2.value_V": [0.0],
                "recap_m2.advantage_A": [0.5],
                "recap_m2.advantage_input": [0.5],
                "recap_m2.epsilon_l": [0.0],
                "recap_m2.indicator_I": [1],
                "recap_m2.prompt_raw": ["put the mug on the plate"],
                "recap_m2.prompt_conditioned": [
                    "put the mug on the plate\nAdvantage: positive"
                ],
            }
        )
        parquet_path = (
            recap_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(parquet_path, engine="pyarrow", index=False)

    collection_bundle = dataset_aggregation.CollectionBundle(
        output_dir=tmp_root / "collect",
        manifest_path=tmp_root / "collect" / "collection_manifest.json",
        autonomous_trials_path=tmp_root
        / "collect"
        / dataset_aggregation.AUTONOMOUS_TRIALS_NAME,
        correction_segments_path=tmp_root
        / "collect"
        / dataset_aggregation.CORRECTION_SEGMENTS_NAME,
        canonical_source_check_path=tmp_root
        / "collect"
        / dataset_aggregation.CANONICAL_SOURCE_CHECK_NAME,
        manifest={
            "critic_checkpoint_ref": "adapter_required",
            "policy_checkpoint_ref": "policy/best",
        },
        autonomous_trials=(
            {
                "trial_id": "task0_seed7000_trial0",
                "task_id": 0,
                "seed": 7000,
                "trial_idx": 0,
                "success": True,
                "indicator_I": 1,
                "policy_checkpoint_ref": "policy/best",
                "critic_checkpoint_ref": "adapter_required",
            },
        ),
    )
    surface_dir = dataset_aggregation._materialize_prebuilt_merged_recap_surface(
        demo_dir=demo_dir,
        output_dir=output_dir,
        task_rows=task_rows,
        collection_bundle=collection_bundle,
        correction_segments=(),
        dataset_mix={
            "canonical_demo": {"episodes": 2, "source": "official_native_8d"},
            "autonomous": {"episodes": 1, "successes": 1, "failures": 0},
            "correction": {"segments": 0, "forced_positive": True},
        },
        episode_lineage_path=output_dir
        / "meta"
        / dataset_aggregation.MERGED_EPISODE_LINEAGE_NAME,
    )

    assert surface_dir is not None
    info = dataset_aggregation.read_json(surface_dir / "meta" / "info.json")
    episodes = dataset_aggregation.read_jsonl(surface_dir / "meta" / "episodes.jsonl")
    contract = cast(dict[str, object], info["recap_advantage_input_contract"])
    visible_parquet_count, visible_row_count = (
        _count_loader_visible_parquet_files_and_rows(surface_dir)
    )

    assert (surface_dir / "data" / "chunk-000").is_dir()
    assert not (surface_dir / "data" / "chunk-000").is_symlink()
    assert (surface_dir / "data" / "chunk-001").is_dir()
    assert not (surface_dir / "data" / "chunk-001").is_symlink()
    assert len(episodes) == 3
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 3
    assert contract["epsilon_source"] == "per_task_quantile:prompt_raw:q=0.7"
    assert contract["indicator_dropout_p"] == 0.3
    assert contract["human_correction_override"] is True
    assert visible_parquet_count == 3
    assert visible_row_count == 3
