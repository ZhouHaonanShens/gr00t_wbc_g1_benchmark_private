from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from collections.abc import Sequence
from typing import cast

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_ROOT = REPO_ROOT / "tests"
OPENPI_TESTS_ROOT = TESTS_ROOT / "openpi"


def _ensure_namespace_package_path(
    module_name: str, package_path: Path
) -> types.ModuleType:
    module = sys.modules.get(module_name)
    if not isinstance(module, types.ModuleType):
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
    path_attr = getattr(module, "__path__", None)
    normalized_package_path = str(package_path)
    normalized_paths = list(path_attr) if isinstance(path_attr, list) else []
    if normalized_package_path not in normalized_paths:
        normalized_paths.insert(0, normalized_package_path)
    module.__path__ = normalized_paths  # type: ignore[attr-defined]
    return module


tests_pkg = _ensure_namespace_package_path("tests", TESTS_ROOT)
openpi_tests_pkg = _ensure_namespace_package_path("tests.openpi", OPENPI_TESTS_ROOT)
setattr(tests_pkg, "openpi", openpi_tests_pkg)


from work.openpi.recap import dataset_aggregation  # noqa: E402
from work.openpi.recap import data_transforms  # noqa: E402
from work.openpi.recap.train_config import resolve_repaired_stage_config  # noqa: E402
import work.openpi.pipelines.recap.merge as merge_script  # noqa: E402
from tests.openpi.test_recap_dataset_aggregation import (  # noqa: E402
    write_recap_ready_demo_sibling,
)
from tests.openpi.test_recap_collection_schema import run_collection_fixture  # noqa: E402


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _as_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{context} must be int-like, got {value!r}")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{context} must be integer-valued, got {value!r}")
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"{context} must be int-like, got {type(value).__name__}")


def _rewrite_demo_source_with_prefix_length(
    demo_dir: Path, *, episode_count: int
) -> None:
    tasks = dataset_aggregation.read_jsonl(demo_dir / "meta" / "tasks.jsonl")
    if len(tasks) < 2:
        raise ValueError("demo fixture must expose at least two tasks")
    episodes: list[dict[str, object]] = []
    for episode_index in range(episode_count):
        task_id = episode_index % len(tasks)
        prompt_raw = str(tasks[task_id]["task"])
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [prompt_raw],
                "length": 1,
            }
        )
        frame = pd.DataFrame(
            {
                "image": [f"ego-{episode_index}".encode("utf-8")],
                "wrist_image": [f"wrist-{episode_index}".encode("utf-8")],
                "state": [[float(episode_index)] * 8],
                "actions": [[0.1 + (0.01 * task_id)] * 7],
                "timestamp": [0.0],
                "frame_index": [0],
                "episode_index": [episode_index],
                "index": [episode_index],
                "task_index": [task_id],
            }
        )
        parquet_path = (
            demo_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(parquet_path, engine="pyarrow", index=False)
    info = dataset_aggregation.read_json(demo_dir / "meta" / "info.json")
    info["total_episodes"] = int(episode_count)
    info["total_frames"] = int(episode_count)
    info["splits"] = {"train": f"0:{episode_count}"}
    dataset_aggregation.write_json(demo_dir / "meta" / "info.json", info)
    dataset_aggregation.write_jsonl(demo_dir / "meta" / "episodes.jsonl", episodes)


def _fake_materialize_dataset(
    official_dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    episode_limit: int | None = None,
    critic_checkpoint_dir: str | Path | None = None,
) -> dict[str, object]:
    assert episode_limit is None
    assert critic_checkpoint_dir is not None
    source_dir = Path(official_dataset_dir).resolve()
    prepared_dir = Path(output_dir).resolve()
    episodes = dataset_aggregation.read_jsonl(source_dir / "meta" / "episodes.jsonl")
    tasks = dataset_aggregation.read_jsonl(source_dir / "meta" / "tasks.jsonl")
    prepared_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        prepared_dir / "meta" / "info.json",
        {
            "schema_version": "openpi_libero_official_8d_recap_relabels_v1",
            "route_id": "official_native_8d_recap_relabels_v1",
            "source_dataset_dir": str(source_dir),
            "source_dataset_name": source_dir.name,
            "total_episodes": len(episodes),
            "total_frames": len(episodes),
            "total_tasks": len(tasks),
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 10,
            "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "recap_advantage_input_contract": {
                "contract_version": "full_recap_continuous_adv_v2",
                "critic_checkpoint_ref": str(Path(critic_checkpoint_dir).resolve()),
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
        },
    )
    _write_json(
        prepared_dir / "meta" / "stats.json",
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
        prepared_dir / "meta" / "modality.json",
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
    dataset_aggregation.write_jsonl(prepared_dir / "meta" / "tasks.jsonl", tasks)
    dataset_aggregation.write_jsonl(prepared_dir / "meta" / "episodes.jsonl", episodes)
    _write_json(
        prepared_dir / "materialization_report.json",
        {
            "schema_version": "openpi_libero_official_8d_recap_relabels_report_v1",
            "route_id": "official_native_8d_recap_relabels_v1",
            "final_status": "materialized",
        },
    )
    for row in episodes:
        episode_index = _as_int(
            row.get("episode_index"), context="episodes[].episode_index"
        )
        tasks = row.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 1:
            raise ValueError(
                f"episodes[].tasks must be single-item list, got {tasks!r}"
            )
        prompt_raw = str(tasks[0])
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
                "task_index": [
                    _as_int(row.get("task_id", 0), context="episodes[].task_id")
                ],
                "annotation.human.task_description": [0],
                "annotation.human.action.task_description": [0],
                "recap_m2.t": [0],
                "recap_m2.return_G": [0.0],
                "recap_m2.value_V": [0.0],
                "recap_m2.advantage_A": [0.0],
                "recap_m2.advantage_input": [0.0],
                "recap_m2.epsilon_l": [0.0],
                "recap_m2.indicator_I": [0],
                "recap_m2.prompt_raw": [prompt_raw],
                "recap_m2.prompt_conditioned": [prompt_raw],
            }
        )
        parquet_path = (
            prepared_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(parquet_path, engine="pyarrow", index=False)
    return {"final_status": "materialized"}


def test_correction_segments_stay_forced_positive_after_merge_with_correction_aware_reweight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo_dir, _, collect_dir, critic_checkpoint_ref = run_collection_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        success_pattern=(False, False),
    )
    correction_dir = tmp_path / "corrections"
    _ = write_recap_ready_demo_sibling(
        demo_dir,
        critic_checkpoint_ref=critic_checkpoint_ref,
    )
    _write_jsonl(
        correction_dir / dataset_aggregation.CORRECTION_SEGMENTS_NAME,
        [
            {
                "correction_id": "corr-0001",
                "source_trial_id": "task0_seed7000_trial0",
                "task_id": 0,
                "indicator_I": 0,
                "is_correction": False,
                "policy_checkpoint_ref": str((tmp_path / "policy" / "best").resolve()),
                "critic_checkpoint_ref": critic_checkpoint_ref,
            }
        ],
    )
    output_dir = tmp_path / "merged_dataset_with_corrections"

    exit_code = merge_script.main(
        [
            "--demo-dir",
            str(demo_dir),
            "--autonomous-dir",
            str(collect_dir),
            "--output-dir",
            str(output_dir),
            "--correction-dir",
            str(correction_dir),
        ]
    )

    assert exit_code == 0
    correction_rows = dataset_aggregation.read_jsonl(
        output_dir / dataset_aggregation.CORRECTION_SEGMENTS_NAME
    )
    merged_episodes = dataset_aggregation.read_jsonl(
        output_dir / "meta" / dataset_aggregation.MERGED_EPISODE_LINEAGE_NAME
    )
    merged_correction = [
        row for row in merged_episodes if row.get("source_kind") == "correction_segment"
    ]

    assert correction_rows[0]["indicator_I"] == 1
    assert correction_rows[0]["is_correction"] is True
    assert correction_rows[0]["forced_positive_indicator"] is True
    assert correction_rows[0]["human_correction_override_applied"] is True

    assert merged_correction[0]["indicator_I"] == 1
    assert merged_correction[0]["is_correction"] is True
    assert merged_correction[0]["forced_positive_indicator"] is True
    assert merged_correction[0]["human_correction_override_applied"] is True

    prepared = data_transforms.prepare_stage_training_dataset(
        dataset_dir=output_dir,
        stage_config=resolve_repaired_stage_config("recap_informative"),
        critic_checkpoint_dir=Path(critic_checkpoint_ref),
        prepared_dataset_dir=tmp_path / "prepared_from_merge",
    )
    prepared_episode_rows = dataset_aggregation.read_jsonl(
        prepared.dataset_dir / "meta" / "episodes.jsonl"
    )
    correction_episode_indices = [
        _as_int(row.get("episode_index"), context="prepared_episodes[].episode_index")
        for row in prepared_episode_rows
        if row.get("source_kind") == "correction_segment"
    ]
    prepared_frames = [
        pd.read_parquet(
            prepared.dataset_dir
            / "data"
            / "chunk-000"
            / f"episode_{episode_index:06d}.parquet"
        )
        for episode_index in correction_episode_indices
    ]
    prepared_report = dataset_aggregation.read_json(
        prepared.dataset_dir / "materialization_report.json"
    )
    prepared_info = dataset_aggregation.read_json(
        prepared.dataset_dir / "meta" / "info.json"
    )
    prepared_contract = cast(
        dict[str, object], prepared_info["recap_advantage_input_contract"]
    )
    base_trainer_surface = (
        output_dir / dataset_aggregation.MERGED_RECAP_READY_DATASET_DIRNAME
    ).resolve()
    reweight_policy = cast(
        dict[str, object],
        prepared_contract[data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY],
    )
    duplicated_correction_rows = [
        row
        for row in prepared_episode_rows
        if row.get("source_kind") == "correction_segment"
        and bool(row.get("informative_positive_reweight_duplicate", False))
    ]
    expected_duplicate_correction_rows = int(
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_DUPLICATES_PER_EPISODE
    )

    assert prepared.prepared_from_source is False
    assert (
        prepared.dataset_dir
        == data_transforms.build_informative_positive_reweight_dataset_dir(
            base_trainer_surface
        )
    )
    expected_correction_episode_count = 1 + expected_duplicate_correction_rows
    assert len(correction_episode_indices) == expected_correction_episode_count
    assert len(duplicated_correction_rows) == expected_duplicate_correction_rows
    assert all(
        int(frame.loc[0, "recap_m2.indicator_I"]) == 1 for frame in prepared_frames
    )
    assert all(
        float(frame.loc[0, "recap_m2.advantage_input"]) > 0.0
        for frame in prepared_frames
    )
    assert all(
        str(frame.loc[0, "recap_m2.prompt_conditioned"]).endswith("Advantage: positive")
        for frame in prepared_frames
    )
    assert all(len(frame) == 1 for frame in prepared_frames)
    assert (
        sum(
            int(frame["recap_m2.indicator_I"].astype(int).sum())
            for frame in prepared_frames
        )
        == expected_correction_episode_count
    )
    assert prepared_report["merged_episode_override_applied"] is True
    assert prepared_report["merged_correction_override_count"] == 1
    assert reweight_policy["source_dataset_dir"] == str(base_trainer_surface)
    assert reweight_policy["applies_to_stage"] == "recap_informative"
    assert reweight_policy["policy_name"] == (
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_POLICY_NAME
    )
    assert reweight_policy["enabled"] is True
    assert reweight_policy["applied"] is True
    assert reweight_policy["correction_aware"] is True
    assert reweight_policy["duplication_unit"] == "episode"
    assert (
        reweight_policy["duplicates_per_positive_episode"]
        == expected_duplicate_correction_rows
    )
    correction_signal = cast(dict[str, object], reweight_policy["correction_signal"])
    assert correction_signal["corrections_added"] == 1
    assert correction_signal["dataset_mix_correction_segments"] == 1
    assert correction_signal["merged_correction_override_count"] == 1
    assert correction_signal["episode_level_correction_rows_present"] is True
    assert reweight_policy["source_positive_correction_episode_count"] == 1
    assert (
        reweight_policy["duplicated_correction_episode_count"]
        == expected_duplicate_correction_rows
    )
    assert prepared_report[data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY] == (
        reweight_policy
    )


def test_prepare_stage_training_dataset_bypasses_rematerialization_when_prebuilt_merged_surface_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo_dir, _, collect_dir, critic_checkpoint_ref = run_collection_fixture(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        success_pattern=(True, False),
    )
    _rewrite_demo_source_with_prefix_length(demo_dir, episode_count=9)
    _ = write_recap_ready_demo_sibling(
        demo_dir,
        critic_checkpoint_ref=critic_checkpoint_ref,
    )
    output_dir = tmp_path / "merged_dataset_with_long_demo_prefix"
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

    info_path = output_dir / "meta" / "info.json"
    info = dataset_aggregation.read_json(info_path)
    for key in (
        "route_id",
        "merged_dataset_route_id",
        "episodes_added",
        "corrections_added",
        "dataset_mix",
    ):
        info.pop(key, None)
    dataset_aggregation.write_json(info_path, info)

    captured: dict[str, object] = {}

    def _recording_materialize_dataset(
        official_dataset_dir: str | Path,
        output_dir: str | Path,
        *,
        episode_limit: int | None = None,
        critic_checkpoint_dir: str | Path | None = None,
    ) -> dict[str, object]:
        captured["episode_limit"] = episode_limit
        assert critic_checkpoint_dir is not None
        source_dir = Path(official_dataset_dir).resolve()
        prepared_dir = Path(output_dir).resolve()
        source_episodes = dataset_aggregation.read_jsonl(
            source_dir / "meta" / "episodes.jsonl"
        )
        selected_episodes = (
            source_episodes[:episode_limit]
            if episode_limit is not None
            else source_episodes
        )
        tasks = dataset_aggregation.read_jsonl(source_dir / "meta" / "tasks.jsonl")
        prepared_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            prepared_dir / "meta" / "info.json",
            {
                "schema_version": "openpi_libero_official_8d_recap_relabels_v1",
                "route_id": "official_native_8d_recap_relabels_v1",
                "source_dataset_dir": str(source_dir),
                "source_dataset_name": source_dir.name,
                "total_episodes": len(selected_episodes),
                "total_frames": len(selected_episodes),
                "total_tasks": len(tasks),
                "total_videos": 0,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": 10,
                "splits": {"train": f"0:{len(selected_episodes)}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "recap_advantage_input_contract": {
                    "contract_version": "full_recap_continuous_adv_v2",
                    "critic_checkpoint_ref": str(Path(critic_checkpoint_dir).resolve()),
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
                    "annotation.human.task_description": {
                        "dtype": "int64",
                        "shape": [1],
                    },
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
            prepared_dir / "meta" / "stats.json",
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
            prepared_dir / "meta" / "modality.json",
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
        dataset_aggregation.write_jsonl(prepared_dir / "meta" / "tasks.jsonl", tasks)
        dataset_aggregation.write_jsonl(
            prepared_dir / "meta" / "episodes.jsonl", selected_episodes
        )
        _write_json(
            prepared_dir / "materialization_report.json",
            {
                "schema_version": "openpi_libero_official_8d_recap_relabels_report_v1",
                "route_id": "official_native_8d_recap_relabels_v1",
                "final_status": "materialized",
            },
        )
        for row in selected_episodes:
            episode_index = _as_int(
                row.get("episode_index"), context="selected_episodes[].episode_index"
            )
            tasks = row.get("tasks")
            if not isinstance(tasks, list) or len(tasks) != 1:
                raise ValueError(
                    f"selected_episodes[].tasks must be single-item list, got {tasks!r}"
                )
            prompt_raw = str(tasks[0])
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
                    "task_index": [
                        _as_int(
                            row.get("task_id", 0), context="selected_episodes[].task_id"
                        )
                    ],
                    "annotation.human.task_description": [0],
                    "annotation.human.action.task_description": [0],
                    "recap_m2.t": [0],
                    "recap_m2.return_G": [0.0],
                    "recap_m2.value_V": [0.0],
                    "recap_m2.advantage_A": [0.0],
                    "recap_m2.advantage_input": [0.0],
                    "recap_m2.epsilon_l": [0.0],
                    "recap_m2.indicator_I": [0],
                    "recap_m2.prompt_raw": [prompt_raw],
                    "recap_m2.prompt_conditioned": [prompt_raw],
                }
            )
            parquet_path = (
                prepared_dir
                / "data"
                / "chunk-000"
                / f"episode_{episode_index:06d}.parquet"
            )
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(parquet_path, engine="pyarrow", index=False)
        return {"final_status": "materialized"}

    monkeypatch.setattr(
        data_transforms, "materialize_dataset", _recording_materialize_dataset
    )
    prepared = data_transforms.prepare_stage_training_dataset(
        dataset_dir=output_dir,
        stage_config=resolve_repaired_stage_config("recap_informative"),
        critic_checkpoint_dir=Path(critic_checkpoint_ref),
    )
    prepared_episode_rows = dataset_aggregation.read_jsonl(
        prepared.dataset_dir / "meta" / "episodes.jsonl"
    )
    autonomous_rows = [
        row
        for row in prepared_episode_rows
        if row.get("source_kind") == "autonomous_trial"
        and not bool(row.get("informative_positive_reweight_duplicate", False))
    ]
    duplicated_rows = [
        row
        for row in prepared_episode_rows
        if bool(row.get("informative_positive_reweight_duplicate", False))
    ]
    prepared_report = dataset_aggregation.read_json(
        prepared.dataset_dir / "materialization_report.json"
    )
    base_trainer_surface = (
        output_dir / dataset_aggregation.MERGED_RECAP_READY_DATASET_DIRNAME
    ).resolve()

    assert prepared.prepared_from_source is False
    assert captured == {}
    assert (
        prepared.dataset_dir
        == data_transforms.build_informative_positive_reweight_dataset_dir(
            base_trainer_surface
        )
    )
    expected_duplicate_rows = 10 * int(
        data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_DUPLICATES_PER_EPISODE
    )
    expected_total_episodes = 11 + expected_duplicate_rows
    assert len(prepared_episode_rows) == expected_total_episodes
    assert [row["episode_index"] for row in autonomous_rows] == [9, 10]
    assert len(duplicated_rows) == expected_duplicate_rows
    assert prepared_report["merged_episode_override_applied"] is True
    assert prepared_report["merged_episode_override_count"] == 2
    reweight_policy = cast(
        dict[str, object],
        prepared_report[data_transforms.INFORMATIVE_POSITIVE_REWEIGHT_KEY],
    )
    assert reweight_policy["source_dataset_dir"] == str(base_trainer_surface)
    assert reweight_policy["effective_total_episodes"] == expected_total_episodes
    assert reweight_policy["source_positive_episode_count"] == 10
