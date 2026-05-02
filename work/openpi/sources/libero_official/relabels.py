#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.advantage import build_advantage_generation_plan
from work.openpi.recap.thresholds import (
    DEFAULT_EPSILON_THRESHOLD_PHASE,
    resolve_phase_threshold_policy,
)
from work.openpi.recap.value_adapter import build_openpi_critic_metadata
from work.recap.critic_vlm.eval import build_episode_value_predictions
from work.recap.critic_vlm.loader import load_critic_artifact


ROUTE_ID = "official_native_8d_recap_relabels_v1"
DATASET_SCHEMA_VERSION = "openpi_libero_official_8d_recap_relabels_v1"
REPORT_SCHEMA_VERSION = "openpi_libero_official_8d_recap_relabels_report_v1"
BLOCKER_REPORT_SCHEMA_VERSION = "openpi_libero_official_8d_recap_relabels_blocker_v1"
SUCCESS_EXIT_CODE = 0
BLOCKED_EXIT_CODE = 42
EPSILON_THRESHOLD_PHASE = DEFAULT_EPSILON_THRESHOLD_PHASE
EPSILON_QUANTILE = resolve_phase_threshold_policy(
    EPSILON_THRESHOLD_PHASE
).epsilon_quantile

SOURCE_IMAGE_COLUMN = "image"
SOURCE_WRIST_IMAGE_COLUMN = "wrist_image"
SOURCE_STATE_COLUMN = "state"
SOURCE_ACTION_COLUMN = "actions"

OUTPUT_IMAGE_COLUMN = "observation.images.ego_view"
OUTPUT_WRIST_IMAGE_COLUMN = "observation.images.wrist_view"
OUTPUT_STATE_COLUMN = "observation.state"
OUTPUT_ACTION_COLUMN = "action"
PROMPT_INDEX_COLUMNS: tuple[str, str] = (
    "annotation.human.task_description",
    "annotation.human.action.task_description",
)
REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    SOURCE_IMAGE_COLUMN,
    SOURCE_WRIST_IMAGE_COLUMN,
    SOURCE_STATE_COLUMN,
    SOURCE_ACTION_COLUMN,
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
REQUIRED_OUTPUT_LABEL_COLUMNS: tuple[str, ...] = (
    "recap_m2.t",
    "recap_m2.return_G",
    "recap_m2.value_V",
    "recap_m2.advantage_A",
    "recap_m2.advantage_input",
    "recap_m2.epsilon_l",
    "recap_m2.indicator_I",
    "recap_m2.prompt_raw",
    "recap_m2.prompt_conditioned",
)


def _import_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"materialization_requires_pandas: {exc}") from exc
    return pd


def _read_json(path: Path) -> dict[str, object]:
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise ValueError(
            f"expected JSON object at {path}, got {type(payload).__name__}"
        )
    return {
        str(key): value for key, value in cast(dict[object, object], payload).items()
    }


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        payload = cast(object, json.loads(stripped))
        if not isinstance(payload, dict):
            raise ValueError(
                f"expected JSON object in {path} line {line_no}, got {type(payload).__name__}"
            )
        rows.append(
            {
                str(key): value
                for key, value in cast(dict[object, object], payload).items()
            }
        )
    return tuple(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    tmp.replace(path)


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


def _as_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{context} must be float-like, got {value!r}")
    try:
        out = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} must be float-like, got {type(value).__name__}"
        ) from exc
    if not math.isfinite(out):
        raise ValueError(f"{context} must be finite, got {value!r}")
    return float(out)


def _as_non_empty_str(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be non-empty str, got {value!r}")
    return value.strip()


def _resolve_dataset_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    resolved = path if path.is_absolute() else (REPO_ROOT / path)
    resolved = resolved.resolve()
    if not resolved.is_dir():
        raise ValueError(f"dataset directory does not exist: {resolved}")
    meta_dir = resolved / "meta"
    required = (
        meta_dir / "info.json",
        meta_dir / "tasks.jsonl",
        meta_dir / "episodes.jsonl",
        meta_dir / "stats.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"dataset metadata missing required files: {missing}")
    return resolved


def _feature_shape(feature_payload: object, *, context: str) -> list[int]:
    if not isinstance(feature_payload, dict):
        raise ValueError(f"{context} must be a feature object")
    shape = feature_payload.get("shape")
    if not isinstance(shape, list):
        raise ValueError(f"{context}.shape must be a list")
    return [
        _as_int(item, context=f"{context}.shape[{index}]")
        for index, item in enumerate(shape)
    ]


def _dataset_glob(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.glob("data/chunk-*/episode_*.parquet")))


def _copy_feature(feature: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(feature))


def _stable_scalar_stats(values: list[float]) -> dict[str, list[float]]:
    if not values:
        raise ValueError("stats values must be non-empty")
    count = float(len(values))
    mean = float(sum(values) / count)
    var = float(sum((float(v) - mean) ** 2 for v in values) / count)
    return {
        "mean": [mean],
        "std": [math.sqrt(var)],
        "max": [float(max(values))],
        "min": [float(min(values))],
    }


def _quantile_linear(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError(f"q must be in [0,1], got {q!r}")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = float(q) * float(len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - float(lo)
    return float((1.0 - weight) * ordered[lo] + weight * ordered[hi])


class OfficialDataset:
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.info = _read_json(self.dataset_dir / "meta" / "info.json")
        self.tasks = _read_jsonl(self.dataset_dir / "meta" / "tasks.jsonl")
        self.episodes = _read_jsonl(self.dataset_dir / "meta" / "episodes.jsonl")
        self.stats = _read_json(self.dataset_dir / "meta" / "stats.json")
        self.data_files = _dataset_glob(self.dataset_dir)
        self.features = cast(dict[str, object], self.info.get("features") or {})
        if not self.features:
            raise ValueError("source info.json missing features")
        self.state_dim = _as_int(
            _feature_shape(
                cast(dict[str, object], self.features.get("state")),
                context="features.state",
            )[0],
            context="features.state.shape[0]",
        )
        self.action_dim = _as_int(
            _feature_shape(
                cast(dict[str, object], self.features.get("actions")),
                context="features.actions",
            )[0],
            context="features.actions.shape[0]",
        )
        self.total_episodes = _as_int(
            self.info.get("total_episodes"), context="info.total_episodes"
        )
        self.total_frames = _as_int(
            self.info.get("total_frames"), context="info.total_frames"
        )
        self.total_tasks = _as_int(
            self.info.get("total_tasks"), context="info.total_tasks"
        )
        self.chunks_size = _as_int(
            self.info.get("chunks_size"), context="info.chunks_size"
        )
        self.fps = _as_int(self.info.get("fps"), context="info.fps")
        if self.state_dim != 8 or self.action_dim != 7:
            raise ValueError(
                "official/native source shape mismatch: expected state.shape=[8] and action.shape=[7] "
                + f"got state.shape=[{self.state_dim}] action.shape=[{self.action_dim}]"
            )
        if len(self.data_files) != self.total_episodes:
            raise ValueError(
                "official/native payload mismatch: "
                + f"declared_total_episodes={self.total_episodes} observed_episode_files={len(self.data_files)}"
            )
        self.tasks_by_index = {
            _as_int(
                row.get("task_index"), context="tasks.task_index"
            ): _as_non_empty_str(row.get("task"), context="tasks.task")
            for row in self.tasks
        }
        self.episodes_by_index = {
            _as_int(row.get("episode_index"), context="episodes.episode_index"): dict(
                row
            )
            for row in self.episodes
        }


def _selected_episode_records(
    source: OfficialDataset, *, episode_limit: int | None
) -> tuple[dict[str, object], ...]:
    ordered = tuple(source.episodes)
    if episode_limit is None:
        return ordered
    limit = _as_int(episode_limit, context="episode_limit")
    if limit <= 0:
        raise ValueError(f"episode_limit must be positive, got {limit}")
    return ordered[:limit]


def _task_prompt_from_episode(
    source: OfficialDataset, episode_record: dict[str, object]
) -> str:
    tasks_raw = episode_record.get("tasks")
    if not isinstance(tasks_raw, list) or len(tasks_raw) != 1:
        raise ValueError(f"episode tasks must be single-item list, got {tasks_raw!r}")
    prompt_raw = _as_non_empty_str(tasks_raw[0], context="episodes.tasks[0]")
    if prompt_raw not in set(source.tasks_by_index.values()):
        raise ValueError(
            f"episode task text is missing from meta/tasks.jsonl: {prompt_raw!r}"
        )
    return prompt_raw


def build_label_plan(
    official_dataset_dir: str | Path,
    *,
    episode_limit: int | None = None,
    critic_checkpoint_dir: str | Path | None = None,
    threshold_phase: str = EPSILON_THRESHOLD_PHASE,
) -> dict[str, object]:
    source = OfficialDataset(_resolve_dataset_dir(str(official_dataset_dir)))
    selected_episodes = _selected_episode_records(source, episode_limit=episode_limit)
    if not selected_episodes:
        raise ValueError("selected official/native dataset is empty")
    if critic_checkpoint_dir is None:
        raise ValueError(
            "task-6 materialization requires --critic-checkpoint-dir; baseline t_mean_return is no longer allowed"
        )
    threshold_policy = resolve_phase_threshold_policy(threshold_phase)

    lengths_by_episode: dict[int, int] = {}
    prompts_by_episode: dict[int, str] = {}
    total_frames = 0

    for record in selected_episodes:
        episode_index = _as_int(
            record.get("episode_index"), context="episodes.episode_index"
        )
        length = _as_int(
            record.get("length"), context=f"episodes[{episode_index}].length"
        )
        if length <= 0:
            raise ValueError(
                f"episode_index={episode_index} has non-positive length {length}"
            )
        prompt_raw = _task_prompt_from_episode(source, record)
        lengths_by_episode[episode_index] = length
        prompts_by_episode[episode_index] = prompt_raw
        total_frames += int(length)

    returns_by_episode: dict[int, list[float]] = {}
    resolved_critic_dir = Path(critic_checkpoint_dir).expanduser().resolve()
    critic_metadata = build_openpi_critic_metadata(
        artifact=load_critic_artifact(resolved_critic_dir)
    )
    values_by_episode = build_episode_value_predictions(
        checkpoint_dir=resolved_critic_dir,
        dataset_dir=source.dataset_dir,
        episode_indices=[
            _as_int(record.get("episode_index"), context="episodes.episode_index")
            for record in selected_episodes
        ],
    )

    for episode_index, length in lengths_by_episode.items():
        returns_by_episode[episode_index] = [
            float(t - (length - 1)) for t in range(length)
        ]

    generation = build_advantage_generation_plan(
        prompts_by_episode=prompts_by_episode,
        returns_by_episode=returns_by_episode,
        values_by_episode=values_by_episode,
        critic_metadata=critic_metadata,
        threshold_phase=threshold_policy.phase,
    )
    records_by_episode = cast(
        dict[int, list[dict[str, object]]], generation["records_by_episode"]
    )
    advantages_by_episode = {
        int(episode_index): [float(cast(Any, row["advantage_A"])) for row in rows]
        for episode_index, rows in records_by_episode.items()
    }
    advantage_inputs_by_episode = {
        int(episode_index): [float(cast(Any, row["advantage_input"])) for row in rows]
        for episode_index, rows in records_by_episode.items()
    }
    indicators_by_episode = {
        int(episode_index): [int(cast(Any, row["indicator_I"])) for row in rows]
        for episode_index, rows in records_by_episode.items()
    }
    conditioned_by_episode = {
        int(episode_index): [str(row["prompt_conditioned"]) for row in rows]
        for episode_index, rows in records_by_episode.items()
    }
    prompt_by_episode = {
        int(episode_index): [str(row["prompt_raw"]) for row in rows]
        for episode_index, rows in records_by_episode.items()
    }

    return {
        "source": source,
        "selected_episode_count": len(selected_episodes),
        "selected_frame_count": total_frames,
        "lengths_by_episode": lengths_by_episode,
        "prompts_by_episode": prompts_by_episode,
        "returns_by_episode": returns_by_episode,
        "values_by_episode": values_by_episode,
        "advantages_by_episode": advantages_by_episode,
        "advantage_inputs_by_episode": advantage_inputs_by_episode,
        "indicators_by_episode": indicators_by_episode,
        "conditioned_by_episode": conditioned_by_episode,
        "prompt_by_episode": prompt_by_episode,
        "records_by_episode": records_by_episode,
        "threshold_estimates": cast(
            dict[str, object], generation["threshold_estimates"]
        ),
        "epsilon_quantile": float(cast(Any, generation["epsilon_quantile"])),
        "epsilon_threshold_phase": str(generation["epsilon_threshold_phase"]),
        "epsilon_threshold_policy": cast(
            dict[str, object], generation["epsilon_threshold_policy"]
        ),
        "epsilon_source": str(generation["epsilon_source"]),
        "scale_metadata": cast(dict[str, object], generation["scale_metadata"]),
        "advantage_contract": cast(dict[str, object], generation["advantage_contract"]),
        "raw_summary": cast(dict[str, object], generation["raw_summary"]),
        "scaled_summary": cast(dict[str, object], generation["scaled_summary"]),
        "episode_limit": episode_limit,
        "critic_metadata": critic_metadata,
    }


def _build_info_json(plan: dict[str, object]) -> dict[str, object]:
    source = cast(OfficialDataset, plan["source"])
    selected_episode_count = _as_int(
        plan["selected_episode_count"], context="selected_episode_count"
    )
    selected_frame_count = _as_int(
        plan["selected_frame_count"], context="selected_frame_count"
    )
    episode_limit = cast(int | None, plan.get("episode_limit"))

    source_image = _copy_feature(
        cast(dict[str, object], source.features[SOURCE_IMAGE_COLUMN])
    )
    source_wrist = _copy_feature(
        cast(dict[str, object], source.features[SOURCE_WRIST_IMAGE_COLUMN])
    )
    source_state = _copy_feature(
        cast(dict[str, object], source.features[SOURCE_STATE_COLUMN])
    )
    source_action = _copy_feature(
        cast(dict[str, object], source.features[SOURCE_ACTION_COLUMN])
    )
    scalar_float_feature = {"dtype": "float32", "shape": [1], "names": None}
    scalar_int_feature = {"dtype": "int64", "shape": [1], "names": None}

    chunk_count = 1 + ((selected_episode_count - 1) // source.chunks_size)
    info: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "route_id": ROUTE_ID,
        "codebase_version": str(source.info.get("codebase_version", "v2.0")),
        "robot_type": source.info.get("robot_type"),
        "source_dataset_dir": str(source.dataset_dir),
        "source_dataset_name": source.dataset_dir.name,
        "total_episodes": int(selected_episode_count),
        "total_frames": int(selected_frame_count),
        "total_tasks": int(source.total_tasks),
        "total_videos": _as_int(
            source.info.get("total_videos", 0), context="info.total_videos"
        ),
        "total_chunks": int(chunk_count),
        "chunks_size": int(source.chunks_size),
        "fps": int(source.fps),
        "splits": {"train": f"0:{selected_episode_count}"},
        "data_path": str(
            source.info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            )
        ),
        "recap_label_recipe": {
            "reward_scheme": "success_demo_terminal_only_v1",
            "value_baseline": "critic_raw_return_adapter_v1",
            "epsilon_quantile": float(cast(Any, plan["epsilon_quantile"])),
            "epsilon_threshold_phase": str(plan["epsilon_threshold_phase"]),
            "epsilon_source": str(plan["epsilon_source"]),
            "state_source": "official/native 8D",
            "action_source": "official/native 7D",
            "cross_dataset_join_used": False,
        },
        "recap_advantage_input_contract": dict(
            cast(dict[str, object], plan["advantage_contract"])
        ),
        "features": {
            OUTPUT_IMAGE_COLUMN: source_image,
            OUTPUT_WRIST_IMAGE_COLUMN: source_wrist,
            OUTPUT_STATE_COLUMN: source_state,
            OUTPUT_ACTION_COLUMN: source_action,
            "timestamp": scalar_float_feature,
            "frame_index": scalar_int_feature,
            "episode_index": scalar_int_feature,
            "index": scalar_int_feature,
            "task_index": scalar_int_feature,
            PROMPT_INDEX_COLUMNS[0]: scalar_int_feature,
            PROMPT_INDEX_COLUMNS[1]: scalar_int_feature,
            "recap_m2.t": scalar_int_feature,
            "recap_m2.return_G": scalar_float_feature,
            "recap_m2.value_V": scalar_float_feature,
            "recap_m2.advantage_A": scalar_float_feature,
            "recap_m2.advantage_input": scalar_float_feature,
            "recap_m2.epsilon_l": scalar_float_feature,
            "recap_m2.indicator_I": scalar_int_feature,
        },
    }
    if episode_limit is not None:
        info["source_subset_for_tests"] = {"episode_limit": int(episode_limit)}
    return info


def _build_modality_json() -> dict[str, object]:
    return {
        "action": {
            "libero_action": {
                "start": 0,
                "end": 7,
                "original_key": OUTPUT_ACTION_COLUMN,
            }
        },
        "annotation": {
            "human.task_description": {"original_key": PROMPT_INDEX_COLUMNS[0]},
            "human.action.task_description": {"original_key": PROMPT_INDEX_COLUMNS[1]},
        },
        "state": {
            "libero_state": {"start": 0, "end": 8, "original_key": OUTPUT_STATE_COLUMN}
        },
        "video": {
            "ego_view": {"original_key": OUTPUT_IMAGE_COLUMN},
            "wrist_view": {"original_key": OUTPUT_WRIST_IMAGE_COLUMN},
        },
    }


def _build_stats_json(plan: dict[str, object]) -> dict[str, object]:
    source = cast(OfficialDataset, plan["source"])
    raw_summary = cast(dict[str, float], plan["raw_summary"])
    scaled_summary = cast(dict[str, float], plan["scaled_summary"])
    selected_episode_count = _as_int(
        plan["selected_episode_count"], context="selected_episode_count"
    )
    selected_frame_count = _as_int(
        plan["selected_frame_count"], context="selected_frame_count"
    )

    task_stats_source = cast(dict[str, object], source.stats["task_index"])
    timestamp_stats_source = cast(dict[str, object], source.stats["timestamp"])
    frame_stats_source = cast(dict[str, object], source.stats["frame_index"])
    episode_stats_source = cast(dict[str, object], source.stats["episode_index"])
    index_stats_source = cast(dict[str, object], source.stats["index"])
    return_stats = _stable_scalar_stats(
        [
            float(value)
            for values in cast(
                dict[int, list[float]], plan["returns_by_episode"]
            ).values()
            for value in values
        ]
    )
    value_stats = _stable_scalar_stats(
        [
            float(value)
            for values in cast(
                dict[int, list[float]], plan["values_by_episode"]
            ).values()
            for value in values
        ]
    )
    advantage_stats = _stable_scalar_stats(
        [
            float(value)
            for values in cast(
                dict[int, list[float]], plan["advantages_by_episode"]
            ).values()
            for value in values
        ]
    )
    advantage_input_stats = _stable_scalar_stats(
        [
            float(value)
            for values in cast(
                dict[int, list[float]], plan["advantage_inputs_by_episode"]
            ).values()
            for value in values
        ]
    )
    indicator_stats = _stable_scalar_stats(
        [
            float(value)
            for values in cast(
                dict[int, list[int]], plan["indicators_by_episode"]
            ).values()
            for value in values
        ]
    )
    epsilon_values = [
        float(cast(Any, row["epsilon_l"]))
        for rows in cast(
            dict[int, list[dict[str, object]]], plan["records_by_episode"]
        ).values()
        for row in rows
    ]
    return {
        OUTPUT_IMAGE_COLUMN: cast(dict[str, object], source.stats[SOURCE_IMAGE_COLUMN]),
        OUTPUT_WRIST_IMAGE_COLUMN: cast(
            dict[str, object], source.stats[SOURCE_WRIST_IMAGE_COLUMN]
        ),
        OUTPUT_STATE_COLUMN: cast(dict[str, object], source.stats[SOURCE_STATE_COLUMN]),
        OUTPUT_ACTION_COLUMN: cast(
            dict[str, object], source.stats[SOURCE_ACTION_COLUMN]
        ),
        "timestamp": timestamp_stats_source,
        "frame_index": frame_stats_source,
        "episode_index": episode_stats_source,
        "index": index_stats_source,
        "task_index": task_stats_source,
        PROMPT_INDEX_COLUMNS[0]: task_stats_source,
        PROMPT_INDEX_COLUMNS[1]: task_stats_source,
        "recap_m2.t": frame_stats_source,
        "recap_m2.return_G": return_stats,
        "recap_m2.value_V": value_stats,
        "recap_m2.advantage_A": advantage_stats,
        "recap_m2.advantage_input": advantage_input_stats,
        "recap_m2.epsilon_l": _stable_scalar_stats(epsilon_values),
        "recap_m2.indicator_I": indicator_stats,
        "_report": {
            "selected_episode_count": int(selected_episode_count),
            "selected_frame_count": int(selected_frame_count),
            "epsilon_source": str(plan["epsilon_source"]),
            "threshold_estimates": cast(dict[str, object], plan["threshold_estimates"]),
            "raw_advantage_summary": raw_summary,
            "scaled_advantage_summary": scaled_summary,
        },
    }


def materialize_dataset(
    official_dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    episode_limit: int | None = None,
    critic_checkpoint_dir: str | Path | None = None,
    threshold_phase: str = EPSILON_THRESHOLD_PHASE,
) -> dict[str, object]:
    pd = _import_pandas()
    plan = build_label_plan(
        official_dataset_dir,
        episode_limit=episode_limit,
        critic_checkpoint_dir=critic_checkpoint_dir,
        threshold_phase=threshold_phase,
    )
    source = cast(OfficialDataset, plan["source"])
    lengths_by_episode = cast(dict[int, int], plan["lengths_by_episode"])
    returns_by_episode = cast(dict[int, list[float]], plan["returns_by_episode"])
    values_by_episode = cast(dict[int, list[float]], plan["values_by_episode"])
    advantages_by_episode = cast(dict[int, list[float]], plan["advantages_by_episode"])
    advantage_inputs_by_episode = cast(
        dict[int, list[float]], plan["advantage_inputs_by_episode"]
    )
    indicators_by_episode = cast(dict[int, list[int]], plan["indicators_by_episode"])
    conditioned_by_episode = cast(dict[int, list[str]], plan["conditioned_by_episode"])
    prompts_by_episode = cast(dict[int, list[str]], plan["prompt_by_episode"])
    records_by_episode = cast(
        dict[int, list[dict[str, object]]], plan["records_by_episode"]
    )
    selected_episode_count = _as_int(
        plan["selected_episode_count"], context="selected_episode_count"
    )
    selected_frame_count = _as_int(
        plan["selected_frame_count"], context="selected_frame_count"
    )

    output_dir_path = Path(output_dir)
    output_dir_resolved = (
        output_dir_path
        if output_dir_path.is_absolute()
        else (REPO_ROOT / output_dir_path)
    ).resolve()
    if output_dir_resolved.exists():
        shutil.rmtree(output_dir_resolved)
    (output_dir_resolved / "data").mkdir(parents=True, exist_ok=True)
    (output_dir_resolved / "meta").mkdir(parents=True, exist_ok=True)

    selected_files = source.data_files[:selected_episode_count]
    for expected_episode_index, parquet_path in enumerate(selected_files):
        episode_index = int(expected_episode_index)
        frame = pd.read_parquet(parquet_path)
        missing = [
            column for column in REQUIRED_SOURCE_COLUMNS if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"{parquet_path} missing required official columns: {missing}"
            )
        expected_length = int(lengths_by_episode[episode_index])
        if len(frame) != expected_length:
            raise ValueError(
                f"episode_index={episode_index} length mismatch: meta={expected_length} parquet={len(frame)}"
            )
        observed_episode_indices = {
            int(value) for value in frame["episode_index"].astype(int).tolist()
        }
        if observed_episode_indices != {episode_index}:
            raise ValueError(
                f"episode_index={episode_index} parquet episode_index mismatch: {sorted(observed_episode_indices)!r}"
            )
        frame_indices = [
            int(value) for value in frame["frame_index"].astype(int).tolist()
        ]
        expected_frame_indices = list(range(expected_length))
        if frame_indices != expected_frame_indices:
            raise ValueError(
                f"episode_index={episode_index} frame_index must equal 0..{expected_length - 1}, got first/last={frame_indices[:3]!r}/{frame_indices[-3:]!r}"
            )
        task_indices = {
            int(value) for value in frame["task_index"].astype(int).tolist()
        }
        if len(task_indices) != 1:
            raise ValueError(
                f"episode_index={episode_index} must have exactly one task_index, got {sorted(task_indices)!r}"
            )
        task_index = next(iter(task_indices))
        prompt_raw = _as_non_empty_str(
            source.tasks_by_index.get(task_index),
            context=f"task_index={task_index}.task",
        )
        expected_prompt_raw = prompts_by_episode[episode_index][0]
        if prompt_raw != expected_prompt_raw:
            raise ValueError(
                f"episode_index={episode_index} prompt mismatch: task_index -> {prompt_raw!r} vs episodes.jsonl -> {expected_prompt_raw!r}"
            )

        output_frame = pd.DataFrame(
            {
                OUTPUT_IMAGE_COLUMN: frame[SOURCE_IMAGE_COLUMN],
                OUTPUT_WRIST_IMAGE_COLUMN: frame[SOURCE_WRIST_IMAGE_COLUMN],
                OUTPUT_STATE_COLUMN: frame[SOURCE_STATE_COLUMN],
                OUTPUT_ACTION_COLUMN: frame[SOURCE_ACTION_COLUMN],
                "timestamp": frame["timestamp"],
                "frame_index": frame["frame_index"].astype("int64"),
                "episode_index": frame["episode_index"].astype("int64"),
                "index": frame["index"].astype("int64"),
                "task_index": frame["task_index"].astype("int64"),
                PROMPT_INDEX_COLUMNS[0]: frame["task_index"].astype("int64"),
                PROMPT_INDEX_COLUMNS[1]: frame["task_index"].astype("int64"),
                "recap_m2.t": expected_frame_indices,
                "recap_m2.return_G": returns_by_episode[episode_index],
                "recap_m2.value_V": values_by_episode[episode_index],
                "recap_m2.advantage_A": advantages_by_episode[episode_index],
                "recap_m2.advantage_input": advantage_inputs_by_episode[episode_index],
                "recap_m2.epsilon_l": [
                    float(cast(Any, row["epsilon_l"]))
                    for row in records_by_episode[episode_index]
                ],
                "recap_m2.indicator_I": indicators_by_episode[episode_index],
                "recap_m2.prompt_raw": prompts_by_episode[episode_index],
                "recap_m2.prompt_conditioned": conditioned_by_episode[episode_index],
            }
        )

        state_shapes = {
            tuple(value.shape) for value in output_frame[OUTPUT_STATE_COLUMN]
        }
        action_shapes = {
            tuple(value.shape) for value in output_frame[OUTPUT_ACTION_COLUMN]
        }
        if state_shapes != {(8,)}:
            raise ValueError(
                f"episode_index={episode_index} output state shapes mismatch: {state_shapes!r}"
            )
        if action_shapes != {(7,)}:
            raise ValueError(
                f"episode_index={episode_index} output action shapes mismatch: {action_shapes!r}"
            )
        chunk_dir = (
            output_dir_resolved
            / "data"
            / f"chunk-{episode_index // source.chunks_size:03d}"
        )
        chunk_dir.mkdir(parents=True, exist_ok=True)
        output_frame.to_parquet(
            chunk_dir / f"episode_{episode_index:06d}.parquet",
            engine="pyarrow",
            index=False,
        )

    _write_json(output_dir_resolved / "meta" / "info.json", _build_info_json(plan))
    _write_json(output_dir_resolved / "meta" / "modality.json", _build_modality_json())
    _write_json(output_dir_resolved / "meta" / "stats.json", _build_stats_json(plan))
    _write_jsonl(
        output_dir_resolved / "meta" / "tasks.jsonl",
        [dict(row) for row in source.tasks],
    )
    _write_jsonl(
        output_dir_resolved / "meta" / "episodes.jsonl",
        [dict(row) for row in source.episodes[:selected_episode_count]],
    )

    observed_positive_count = sum(
        int(value) for values in indicators_by_episode.values() for value in values
    )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "route_id": ROUTE_ID,
        "final_status": "materialized",
        "source_dataset_dir": str(source.dataset_dir),
        "output_dataset_dir": str(output_dir_resolved),
        "selected_episode_count": int(selected_episode_count),
        "selected_frame_count": int(selected_frame_count),
        "state_dim": int(source.state_dim),
        "action_dim": int(source.action_dim),
        "reward_scheme": "success_demo_terminal_only_v1",
        "value_baseline": "critic_raw_return_adapter_v1",
        "epsilon_quantile": float(cast(Any, plan["epsilon_quantile"])),
        "epsilon_threshold_phase": str(plan["epsilon_threshold_phase"]),
        "epsilon_source": str(plan["epsilon_source"]),
        "threshold_estimates": cast(dict[str, object], plan["threshold_estimates"]),
        "epsilon_threshold_policy": cast(
            dict[str, object], plan["epsilon_threshold_policy"]
        ),
        "positive_indicator_count": int(observed_positive_count),
        "positive_indicator_fraction": float(observed_positive_count)
        / float(selected_frame_count),
        "required_output_label_columns": list(REQUIRED_OUTPUT_LABEL_COLUMNS),
        "advantage_scale_summary": cast(dict[str, object], plan["scale_metadata"]),
        "advantage_contract": cast(dict[str, object], plan["advantage_contract"]),
    }
    if plan.get("critic_metadata") is not None:
        report["critic_metadata"] = cast(dict[str, object], plan["critic_metadata"])
    _write_json(output_dir_resolved / "materialization_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="materialize_libero_official_8d_recap_relabels.py",
        description=(
            "Materialize RECAP-style offline relabels directly onto the official/native "
            "LIBERO 8D dataset without any cross-dataset weak join."
        ),
    )
    parser.add_argument(
        "--official-dataset-dir",
        required=True,
        help="Path to agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to output relabeled official/native 8D dataset root.",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=None,
        help="Optional contiguous prefix episode count for test-only subset materialization.",
    )
    parser.add_argument(
        "--critic-checkpoint-dir",
        required=True,
        help="Critic checkpoint dir. Task 6 always derives recap_m2.value_V from critic raw_return adaptation.",
    )
    parser.add_argument(
        "--threshold-phase",
        default=EPSILON_THRESHOLD_PHASE,
        choices=("pretraining", "fine_tuning"),
        help="Phase-specific advantage threshold policy. pretraining uses q=0.7; fine_tuning uses q=0.6.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir_path = Path(args.output_dir)
    output_dir_resolved = (
        output_dir_path
        if output_dir_path.is_absolute()
        else (REPO_ROOT / output_dir_path)
    ).resolve()
    try:
        report = materialize_dataset(
            official_dataset_dir=args.official_dataset_dir,
            output_dir=output_dir_resolved,
            episode_limit=args.episode_limit,
            critic_checkpoint_dir=args.critic_checkpoint_dir,
            threshold_phase=args.threshold_phase,
        )
    except Exception as exc:
        blocker: dict[str, object] = {
            "schema_version": BLOCKER_REPORT_SCHEMA_VERSION,
            "route_id": ROUTE_ID,
            "final_status": "blocked",
            "official_dataset_dir": str(args.official_dataset_dir),
            "output_dataset_dir": str(output_dir_resolved),
            "episode_limit": int(args.episode_limit)
            if args.episode_limit is not None
            else None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        _write_json(output_dir_resolved / "materialization_report.json", blocker)
        print(json.dumps(blocker, ensure_ascii=True, indent=2, sort_keys=True))
        return BLOCKED_EXIT_CODE
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return SUCCESS_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
