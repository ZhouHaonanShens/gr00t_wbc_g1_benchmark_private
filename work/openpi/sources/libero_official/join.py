#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ROUTE_ID = "safe_recap_label_join_onto_official_8d_v1"
JOIN_REPORT_SCHEMA_VERSION = "openpi_libero_official_8d_join_report_v1"
UNMATCHED_CONFLICTS_SCHEMA_VERSION = "openpi_libero_official_8d_unmatched_conflicts_v1"
TASK_CROSSWALK_SCHEMA_VERSION = "openpi_libero_official_8d_task_crosswalk_v1"
EPISODE_FRAME_CROSSWALK_SCHEMA_VERSION = (
    "openpi_libero_official_8d_episode_frame_crosswalk_v1"
)
BLOCKED_EXIT_CODE = 42
SUCCESS_EXIT_CODE = 0

WEAK_SHARED_JOIN_KEYS: tuple[str, ...] = ("episode_index", "index", "timestamp")
STRONG_DATASET_KEYS: tuple[str, ...] = ("task_index", "frame_index")
RECAP_LABEL_COLUMNS: tuple[str, ...] = (
    "recap_m2.advantage_A",
    "recap_m2.advantage_input",
    "recap_m2.epsilon_l",
    "recap_m2.indicator_I",
    "recap_m2.return_G",
    "recap_m2.t",
    "recap_m2.value_V",
)


class JoinContractError(RuntimeError):
    payload: dict[str, object]

    def __init__(self, message: str, *, payload: dict[str, object]) -> None:
        super().__init__(message)
        self.payload = dict(payload)


class JoinBlocked(RuntimeError):
    payload: dict[str, object]

    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(str(payload.get("summary_reason", "join blocked")))
        self.payload = dict(payload)


class JoinMaterialized:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = dict(payload)


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


def _as_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be bool, got {type(value).__name__}")
    return bool(value)


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
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"dataset metadata missing required files: {missing}")
    return resolved


def _resolve_optional_file(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    stripped = str(raw_path).strip()
    if not stripped:
        return None
    path = Path(stripped)
    resolved = path if path.is_absolute() else (REPO_ROOT / path)
    resolved = resolved.resolve()
    return resolved


def _dataset_glob(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.glob("data/chunk-*/episode_*.parquet")))


def _feature_shape(feature_payload: object) -> list[int]:
    if not isinstance(feature_payload, dict):
        return []
    shape = feature_payload.get("shape")
    if not isinstance(shape, list):
        return []
    out: list[int] = []
    for index, item in enumerate(shape):
        out.append(_as_int(item, context=f"feature.shape[{index}]"))
    return out


def _blocker(
    code: str, reason: str, *, evidence: dict[str, object] | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {"code": str(code), "reason": str(reason)}
    if evidence:
        payload["evidence"] = dict(evidence)
    return payload


def _soft_mismatch(
    code: str, reason: str, *, observed: dict[str, object]
) -> dict[str, object]:
    return {
        "code": str(code),
        "reason": str(reason),
        "observed": dict(observed),
    }


class DatasetMeta:
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.info = _read_json(self.dataset_dir / "meta" / "info.json")
        self.tasks = _read_jsonl(self.dataset_dir / "meta" / "tasks.jsonl")
        self.episodes = _read_jsonl(self.dataset_dir / "meta" / "episodes.jsonl")
        features = self.info.get("features")
        if not isinstance(features, dict):
            raise ValueError(
                f"{self.dataset_dir}/meta/info.json must expose object features"
            )
        self.features = {str(key): value for key, value in features.items()}
        self.feature_keys = set(self.features)
        self.episode_field_keys = {
            str(key) for episode in self.episodes for key in episode.keys()
        }
        self.data_files = _dataset_glob(self.dataset_dir)
        self.total_episodes = _as_int(
            self.info.get("total_episodes"), context="info.total_episodes"
        )
        self.total_frames = _as_int(
            self.info.get("total_frames"), context="info.total_frames"
        )
        self.total_tasks = _as_int(
            self.info.get("total_tasks"), context="info.total_tasks"
        )
        self.fps = _as_int(self.info.get("fps"), context="info.fps")
        self.state_dim = _as_int(
            _feature_shape(
                self.features.get("state") or self.features.get("observation.state")
            )[0],
            context="state.shape[0]",
        )
        action_feature = self.features.get("actions") or self.features.get("action")
        self.action_dim = _as_int(
            _feature_shape(action_feature)[0], context="action.shape[0]"
        )
        self.task_texts = tuple(
            str(row.get("task", "")).strip()
            for row in self.tasks
            if str(row.get("task", "")).strip()
        )
        self.declared_episode_indices = tuple(
            _as_int(row.get("episode_index"), context="episodes.episode_index")
            for row in self.episodes
        )


def _load_task_crosswalk(
    path: Path | None,
    *,
    official: DatasetMeta,
    recap: DatasetMeta,
) -> tuple[dict[int, dict[str, object]] | None, dict[str, object]]:
    if path is None:
        return None, {"provided": False}
    if not path.is_file():
        return None, {
            "provided": True,
            "path": str(path),
            "valid": False,
            "reason": "file_missing",
        }
    payload = _read_json(path)
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != TASK_CROSSWALK_SCHEMA_VERSION:
        raise JoinContractError(
            "task crosswalk schema_version mismatch",
            payload={
                "path": str(path),
                "expected_schema_version": TASK_CROSSWALK_SCHEMA_VERSION,
                "observed_schema_version": schema_version,
            },
        )
    mappings_raw = payload.get("mappings")
    if not isinstance(mappings_raw, list) or not mappings_raw:
        raise JoinContractError(
            "task crosswalk mappings must be a non-empty list",
            payload={"path": str(path)},
        )
    verified_compatible = _as_bool(
        payload.get("verified_compatible"), context="task_crosswalk.verified_compatible"
    )
    official_root = _as_non_empty_str(
        payload.get("official_source_root"),
        context="task_crosswalk.official_source_root",
    )
    recap_root = _as_non_empty_str(
        payload.get("recap_label_source_root"),
        context="task_crosswalk.recap_label_source_root",
    )
    if Path(official_root).resolve() != official.dataset_dir:
        raise JoinContractError(
            "task crosswalk official_source_root mismatch",
            payload={"path": str(path), "official_source_root": official_root},
        )
    if Path(recap_root).resolve() != recap.dataset_dir:
        raise JoinContractError(
            "task crosswalk recap_label_source_root mismatch",
            payload={"path": str(path), "recap_label_source_root": recap_root},
        )
    mapping_by_official_task_index: dict[int, dict[str, object]] = {}
    for idx, raw in enumerate(mappings_raw):
        if not isinstance(raw, dict):
            raise JoinContractError(
                "task crosswalk mapping must be JSON object",
                payload={"path": str(path), "index": idx},
            )
        mapping = {str(key): value for key, value in raw.items()}
        official_task_index = _as_int(
            mapping.get("official_task_index"),
            context=f"task_crosswalk.mappings[{idx}].official_task_index",
        )
        if official_task_index in mapping_by_official_task_index:
            raise JoinContractError(
                "duplicate official_task_index in task crosswalk",
                payload={"path": str(path), "official_task_index": official_task_index},
            )
        verified = _as_bool(
            mapping.get("verified"),
            context=f"task_crosswalk.mappings[{idx}].verified",
        )
        if not verified:
            raise JoinContractError(
                "task crosswalk mapping must be verified=true",
                payload={"path": str(path), "official_task_index": official_task_index},
            )
        mapping_by_official_task_index[official_task_index] = mapping
    return mapping_by_official_task_index, {
        "provided": True,
        "path": str(path),
        "valid": True,
        "verified_compatible": verified_compatible,
        "mapping_count": len(mapping_by_official_task_index),
    }


def _load_episode_frame_crosswalk(
    path: Path | None,
) -> tuple[dict[tuple[int, int], dict[str, object]] | None, dict[str, object]]:
    if path is None:
        return None, {"provided": False}
    if not path.is_file():
        return None, {
            "provided": True,
            "path": str(path),
            "valid": False,
            "reason": "file_missing",
        }
    rows = _read_jsonl(path)
    if not rows:
        raise JoinContractError(
            "episode/frame crosswalk must not be empty",
            payload={"path": str(path)},
        )
    mapping: dict[tuple[int, int], dict[str, object]] = {}
    recap_indices: set[int] = set()
    for idx, row in enumerate(rows):
        schema_version = str(row.get("schema_version", ""))
        if schema_version != EPISODE_FRAME_CROSSWALK_SCHEMA_VERSION:
            raise JoinContractError(
                "episode/frame crosswalk schema_version mismatch",
                payload={
                    "path": str(path),
                    "line_index": idx,
                    "expected_schema_version": EPISODE_FRAME_CROSSWALK_SCHEMA_VERSION,
                    "observed_schema_version": schema_version,
                },
            )
        verified = _as_bool(
            row.get("verified"),
            context=f"episode_frame_crosswalk[{idx}].verified",
        )
        if not verified:
            raise JoinContractError(
                "episode/frame crosswalk row must be verified=true",
                payload={"path": str(path), "line_index": idx},
            )
        official_episode_index = _as_int(
            row.get("official_episode_index"),
            context=f"episode_frame_crosswalk[{idx}].official_episode_index",
        )
        official_frame_index = _as_int(
            row.get("official_frame_index"),
            context=f"episode_frame_crosswalk[{idx}].official_frame_index",
        )
        recap_index = _as_int(
            row.get("recap_index"),
            context=f"episode_frame_crosswalk[{idx}].recap_index",
        )
        official_key = (official_episode_index, official_frame_index)
        if official_key in mapping:
            raise JoinContractError(
                "duplicate official episode/frame identity in crosswalk",
                payload={"path": str(path), "official_key": list(official_key)},
            )
        if recap_index in recap_indices:
            raise JoinContractError(
                "duplicate recap_index in crosswalk",
                payload={"path": str(path), "recap_index": recap_index},
            )
        recap_indices.add(recap_index)
        mapping[official_key] = dict(row)
    return mapping, {
        "provided": True,
        "path": str(path),
        "valid": True,
        "mapping_count": len(mapping),
    }


def _join_key_inventory(
    *,
    official: DatasetMeta,
    recap: DatasetMeta,
    task_crosswalk_info: dict[str, object],
    frame_crosswalk_info: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    considered: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for key in STRONG_DATASET_KEYS:
        official_available = (
            key in official.feature_keys or key in official.episode_field_keys
        )
        recap_available = key in recap.feature_keys or key in recap.episode_field_keys
        status = (
            "available_in_both"
            if official_available and recap_available
            else "missing_in_one_side"
        )
        considered.append(
            {
                "key": key,
                "kind": "dataset_metadata_key",
                "official_available": bool(official_available),
                "recap_available": bool(recap_available),
                "status": status,
            }
        )
        if status != "available_in_both":
            rejected.append(
                {
                    "key": key,
                    "reason": "strong identity key not simultaneously available in both datasets",
                }
            )
    for key in WEAK_SHARED_JOIN_KEYS:
        official_available = (
            key in official.feature_keys or key in official.episode_field_keys
        )
        recap_available = key in recap.feature_keys or key in recap.episode_field_keys
        status = (
            "shared_but_weak"
            if official_available and recap_available
            else "not_shared"
        )
        considered.append(
            {
                "key": key,
                "kind": "weak_shared_scalar_key",
                "official_available": bool(official_available),
                "recap_available": bool(recap_available),
                "status": status,
            }
        )
        if status == "shared_but_weak":
            rejected.append(
                {
                    "key": key,
                    "reason": "dataset-local bookkeeping only; not authoritative cross-dataset identity",
                }
            )
    considered.append(
        {
            "key": "task_crosswalk_json",
            "kind": "explicit_repair_evidence",
            "status": "provided"
            if bool(task_crosswalk_info.get("provided"))
            else "missing",
            "details": dict(task_crosswalk_info),
        }
    )
    considered.append(
        {
            "key": "episode_frame_crosswalk_jsonl",
            "kind": "explicit_repair_evidence",
            "status": "provided"
            if bool(frame_crosswalk_info.get("provided"))
            else "missing",
            "details": dict(frame_crosswalk_info),
        }
    )
    return considered, rejected


def assess_join(
    *,
    official_dataset_dir: str | Path,
    recap_label_dataset_dir: str | Path,
    output_dir: str | Path,
    task_crosswalk_json: str | Path | None = None,
    episode_frame_crosswalk_jsonl: str | Path | None = None,
) -> dict[str, object]:
    official = DatasetMeta(_resolve_dataset_dir(str(official_dataset_dir)))
    recap = DatasetMeta(_resolve_dataset_dir(str(recap_label_dataset_dir)))
    output_root = (
        Path(output_dir) if Path(output_dir).is_absolute() else (REPO_ROOT / output_dir)
    ).resolve()
    task_crosswalk_path = _resolve_optional_file(
        None if task_crosswalk_json is None else str(task_crosswalk_json)
    )
    episode_frame_crosswalk_path = _resolve_optional_file(
        None
        if episode_frame_crosswalk_jsonl is None
        else str(episode_frame_crosswalk_jsonl)
    )
    task_crosswalk, task_crosswalk_info = _load_task_crosswalk(
        task_crosswalk_path, official=official, recap=recap
    )
    frame_crosswalk, frame_crosswalk_info = _load_episode_frame_crosswalk(
        episode_frame_crosswalk_path
    )

    shared_task_texts = sorted(set(official.task_texts).intersection(recap.task_texts))
    hard_blockers: list[dict[str, object]] = []
    soft_mismatches: list[dict[str, object]] = []

    if len(official.data_files) != official.total_episodes:
        hard_blockers.append(
            _blocker(
                "incomplete_official_dataset_payload",
                "official dataset payload on disk does not match declared total_episodes; materialization cannot claim full joined dataset coverage",
                evidence={
                    "declared_total_episodes": official.total_episodes,
                    "observed_episode_files": len(official.data_files),
                    "sample_observed_files": [
                        str(path.relative_to(official.dataset_dir))
                        for path in official.data_files[:3]
                    ],
                },
            )
        )
    if task_crosswalk is None:
        hard_blockers.append(
            _blocker(
                "missing_cross_dataset_task_identity",
                "no explicit task identity evidence was materialized for official/native 8D <-> recap labels",
                evidence={"task_crosswalk_json": dict(task_crosswalk_info)},
            )
        )
    if frame_crosswalk is None:
        hard_blockers.append(
            _blocker(
                "missing_verified_episode_frame_crosswalk",
                "no verified episode/frame crosswalk was materialized",
                evidence={"episode_frame_crosswalk_jsonl": dict(frame_crosswalk_info)},
            )
        )
    if "frame_index" not in recap.feature_keys and frame_crosswalk is None:
        hard_blockers.append(
            _blocker(
                "missing_cross_dataset_frame_identity",
                "recap label dataset does not expose frame_index and no explicit frame crosswalk repaired the identity gap",
                evidence={"recap_feature_keys": sorted(recap.feature_keys)},
            )
        )
    if task_crosswalk is None and not shared_task_texts:
        hard_blockers.append(
            _blocker(
                "task_text_universe_not_proven_compatible",
                "official/native task universe and recap label task universe are not proven compatible",
                evidence={
                    "official_total_tasks": official.total_tasks,
                    "recap_total_tasks": recap.total_tasks,
                    "shared_task_texts": shared_task_texts,
                    "official_task_examples": list(official.task_texts[:5]),
                    "recap_task_examples": list(recap.task_texts[:5]),
                },
            )
        )
    weak_shared_keys = [
        key
        for key in WEAK_SHARED_JOIN_KEYS
        if (key in official.feature_keys or key in official.episode_field_keys)
        and (key in recap.feature_keys or key in recap.episode_field_keys)
    ]
    if weak_shared_keys:
        hard_blockers.append(
            _blocker(
                "weak_key_only_join_rejected",
                "episode_index/index/timestamp are shared only as weak dataset-local bookkeeping and cannot prove authoritative cross-dataset identity",
                evidence={"weak_shared_keys": weak_shared_keys},
            )
        )
    if official.fps != recap.fps:
        soft_mismatches.append(
            _soft_mismatch(
                "fps_mismatch",
                "official/native and recap datasets run at different fps",
                observed={"official_fps": official.fps, "recap_fps": recap.fps},
            )
        )
    if official.state_dim != recap.state_dim:
        soft_mismatches.append(
            _soft_mismatch(
                "state_dim_mismatch",
                "official/native and recap datasets expose different state dimensions",
                observed={
                    "official_state_dim": official.state_dim,
                    "recap_state_dim": recap.state_dim,
                },
            )
        )
    if official.action_dim != recap.action_dim:
        soft_mismatches.append(
            _soft_mismatch(
                "action_dim_mismatch",
                "official/native and recap datasets expose different action dimensions",
                observed={
                    "official_action_dim": official.action_dim,
                    "recap_action_dim": recap.action_dim,
                },
            )
        )
    if official.total_tasks != recap.total_tasks:
        soft_mismatches.append(
            _soft_mismatch(
                "task_count_mismatch",
                "official/native and recap datasets declare different task counts",
                observed={
                    "official_total_tasks": official.total_tasks,
                    "recap_total_tasks": recap.total_tasks,
                },
            )
        )
    join_keys_considered, join_keys_rejected = _join_key_inventory(
        official=official,
        recap=recap,
        task_crosswalk_info=task_crosswalk_info,
        frame_crosswalk_info=frame_crosswalk_info,
    )
    matched_frame_count = (
        official.total_frames
        if not hard_blockers and frame_crosswalk is not None
        else 0
    )
    matched_episode_count = (
        official.total_episodes
        if not hard_blockers and frame_crosswalk is not None
        else 0
    )
    final_status = "materialized" if not hard_blockers else "blocked"
    report: dict[str, object] = {
        "schema_version": JOIN_REPORT_SCHEMA_VERSION,
        "route_id": ROUTE_ID,
        "official_source_root": str(official.dataset_dir),
        "recap_label_source_root": str(recap.dataset_dir),
        "output_root": str(output_root),
        "join_keys_considered": join_keys_considered,
        "join_keys_rejected": join_keys_rejected,
        "matched_counts": {
            "matched_episode_count": matched_episode_count,
            "matched_frame_count": matched_frame_count,
        },
        "unmatched_counts": {
            "unmatched_official_episode_count": max(
                0, official.total_episodes - matched_episode_count
            ),
            "unmatched_official_frame_count": max(
                0, official.total_frames - matched_frame_count
            ),
            "unmatched_recap_episode_count": max(
                0, recap.total_episodes - matched_episode_count
            ),
            "unmatched_recap_frame_count": max(
                0, recap.total_frames - matched_frame_count
            ),
        },
        "hard_blockers": hard_blockers,
        "soft_mismatches": soft_mismatches,
        "final_status": final_status,
        "status": final_status,
        "materialization_allowed": not hard_blockers,
        "materialization_emitted": False,
        "dedicated_blocker_exit_code": BLOCKED_EXIT_CODE,
        "source_summary": {
            "official": {
                "dataset_name": official.dataset_dir.name,
                "declared_total_episodes": official.total_episodes,
                "declared_total_frames": official.total_frames,
                "declared_total_tasks": official.total_tasks,
                "fps": official.fps,
                "state_dim": official.state_dim,
                "action_dim": official.action_dim,
                "observed_episode_files": len(official.data_files),
            },
            "recap": {
                "dataset_name": recap.dataset_dir.name,
                "declared_total_episodes": recap.total_episodes,
                "declared_total_frames": recap.total_frames,
                "declared_total_tasks": recap.total_tasks,
                "fps": recap.fps,
                "state_dim": recap.state_dim,
                "action_dim": recap.action_dim,
                "observed_episode_files": len(recap.data_files),
            },
        },
        "task_identity": {
            "task_crosswalk": task_crosswalk_info,
            "shared_task_texts": shared_task_texts,
            "official_task_examples": list(official.task_texts[:5]),
            "recap_task_examples": list(recap.task_texts[:5]),
        },
        "frame_identity": {
            "official_has_frame_index": "frame_index" in official.feature_keys,
            "recap_has_frame_index": "frame_index" in recap.feature_keys,
        },
        "episode_frame_crosswalk": frame_crosswalk_info,
        "summary_reason": (
            "safe join preconditions satisfied"
            if not hard_blockers
            else "safe join preconditions are insufficient; blocker artifacts persisted"
        ),
    }
    return report


def _build_unmatched_conflicts(report: dict[str, object]) -> dict[str, object]:
    hard_blockers = cast(list[dict[str, object]], report["hard_blockers"])
    join_keys_rejected = cast(list[dict[str, object]], report["join_keys_rejected"])
    task_identity = cast(dict[str, object], report["task_identity"])
    frame_identity = cast(dict[str, object], report["frame_identity"])
    return {
        "schema_version": UNMATCHED_CONFLICTS_SCHEMA_VERSION,
        "route_id": report["route_id"],
        "final_status": report["final_status"],
        "hard_blocker_codes": [str(item["code"]) for item in hard_blockers],
        "hard_blockers": hard_blockers,
        "rejected_join_keys": join_keys_rejected,
        "task_universe_conflicts": {
            "shared_task_texts": list(
                cast(list[object], task_identity["shared_task_texts"])
            ),
            "official_task_examples": list(
                cast(list[object], task_identity["official_task_examples"])
            ),
            "recap_task_examples": list(
                cast(list[object], task_identity["recap_task_examples"])
            ),
        },
        "frame_identity_conflicts": dict(frame_identity),
        "episode_frame_crosswalk_conflicts": dict(
            cast(dict[str, object], report["episode_frame_crosswalk"])
        ),
        "unmatched_counts": dict(cast(dict[str, object], report["unmatched_counts"])),
    }


def _load_recap_labels_by_index(recap_dir: Path) -> dict[int, dict[str, object]]:
    pd = _import_pandas()
    label_lookup: dict[int, dict[str, object]] = {}
    columns = ["index", *RECAP_LABEL_COLUMNS]
    for parquet_path in _dataset_glob(recap_dir):
        frame = pd.read_parquet(parquet_path, columns=columns)
        for row in frame.to_dict(orient="records"):
            recap_index = _as_int(row.get("index"), context="recap.index")
            if recap_index in label_lookup:
                raise JoinContractError(
                    "duplicate recap index encountered while building label lookup",
                    payload={
                        "recap_index": recap_index,
                        "parquet_path": str(parquet_path),
                    },
                )
            label_lookup[recap_index] = {
                column: row[column] for column in RECAP_LABEL_COLUMNS
            }
    return label_lookup


def _materialize_joined_dataset(
    *,
    official: DatasetMeta,
    recap: DatasetMeta,
    output_root: Path,
    report: dict[str, object],
    task_crosswalk: dict[int, dict[str, object]],
    frame_crosswalk: dict[tuple[int, int], dict[str, object]],
) -> dict[str, object]:
    pd = _import_pandas()
    recap_labels_by_index = _load_recap_labels_by_index(recap.dataset_dir)
    expected_crosswalk_rows = official.total_frames
    if len(frame_crosswalk) != expected_crosswalk_rows:
        raise JoinBlocked(
            {
                **report,
                "hard_blockers": [
                    *cast(list[dict[str, object]], report["hard_blockers"]),
                    _blocker(
                        "incomplete_episode_frame_crosswalk",
                        "episode/frame crosswalk row count does not cover declared official total_frames",
                        evidence={
                            "expected_crosswalk_rows": expected_crosswalk_rows,
                            "observed_crosswalk_rows": len(frame_crosswalk),
                        },
                    ),
                ],
                "final_status": "blocked",
                "status": "blocked",
                "materialization_allowed": False,
                "summary_reason": "episode/frame crosswalk coverage incomplete",
            }
        )
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    official_features = dict(official.features)
    recap_features = dict(recap.features)
    for column in RECAP_LABEL_COLUMNS:
        if column not in recap_features:
            raise JoinContractError(
                "recap feature missing from info.json for materialized label column",
                payload={"column": column},
            )
        official_features[column] = recap_features[column]
    official_info = dict(official.info)
    official_info["features"] = official_features
    official_info["task9a_join_materialization"] = {
        "route_id": ROUTE_ID,
        "official_source_root": str(official.dataset_dir),
        "recap_label_source_root": str(recap.dataset_dir),
        "task_crosswalk_required": True,
        "episode_frame_crosswalk_required": True,
    }
    joined_episode_file_count = 0
    for parquet_path in official.data_files:
        relative_path = parquet_path.relative_to(official.dataset_dir)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.read_parquet(parquet_path)
        recap_indices: list[int] = []
        for row in frame[["episode_index", "frame_index", "task_index"]].to_dict(
            orient="records"
        ):
            official_episode_index = _as_int(
                row.get("episode_index"), context="official.episode_index"
            )
            official_frame_index = _as_int(
                row.get("frame_index"), context="official.frame_index"
            )
            official_task_index = _as_int(
                row.get("task_index"), context="official.task_index"
            )
            if official_task_index not in task_crosswalk:
                raise JoinContractError(
                    "official task_index missing from verified task crosswalk",
                    payload={"official_task_index": official_task_index},
                )
            key = (official_episode_index, official_frame_index)
            mapping = frame_crosswalk.get(key)
            if mapping is None:
                raise JoinContractError(
                    "official episode/frame missing from verified crosswalk",
                    payload={
                        "official_episode_index": official_episode_index,
                        "official_frame_index": official_frame_index,
                    },
                )
            recap_index = _as_int(
                mapping.get("recap_index"), context="frame_crosswalk.recap_index"
            )
            if recap_index not in recap_labels_by_index:
                raise JoinContractError(
                    "recap_index referenced by crosswalk does not exist in recap label payload",
                    payload={"recap_index": recap_index},
                )
            recap_indices.append(recap_index)
        for column in RECAP_LABEL_COLUMNS:
            frame[column] = [
                recap_labels_by_index[recap_index][column]
                for recap_index in recap_indices
            ]
        frame.to_parquet(output_path, index=False)
        joined_episode_file_count += 1
    _write_json(
        output_root / "meta" / "info.json", cast(dict[str, object], official_info)
    )
    _write_jsonl(output_root / "meta" / "tasks.jsonl", list(official.tasks))
    _write_jsonl(output_root / "meta" / "episodes.jsonl", list(official.episodes))
    readme_path = official.dataset_dir / "README.md"
    if readme_path.is_file():
        shutil.copy2(readme_path, output_root / "README.md")
    return {
        **report,
        "final_status": "materialized",
        "status": "materialized",
        "materialization_allowed": True,
        "materialization_emitted": True,
        "matched_counts": {
            "matched_episode_count": official.total_episodes,
            "matched_frame_count": official.total_frames,
        },
        "unmatched_counts": {
            "unmatched_official_episode_count": 0,
            "unmatched_official_frame_count": 0,
            "unmatched_recap_episode_count": max(
                0, recap.total_episodes - official.total_episodes
            ),
            "unmatched_recap_frame_count": max(
                0, recap.total_frames - official.total_frames
            ),
        },
        "source_summary": {
            **cast(dict[str, object], report["source_summary"]),
            "joined_output": {
                "output_root": str(output_root),
                "joined_episode_file_count": joined_episode_file_count,
                "joined_label_columns": list(RECAP_LABEL_COLUMNS),
            },
        },
        "summary_reason": "safe join preconditions satisfied and joined dataset materialized",
    }


def materialize_join(
    *,
    official_dataset_dir: str | Path,
    recap_label_dataset_dir: str | Path,
    output_dir: str | Path,
    task_crosswalk_json: str | Path | None = None,
    episode_frame_crosswalk_jsonl: str | Path | None = None,
) -> JoinMaterialized:
    report = assess_join(
        official_dataset_dir=official_dataset_dir,
        recap_label_dataset_dir=recap_label_dataset_dir,
        output_dir=output_dir,
        task_crosswalk_json=task_crosswalk_json,
        episode_frame_crosswalk_jsonl=episode_frame_crosswalk_jsonl,
    )
    if cast(list[object], report["hard_blockers"]):
        raise JoinBlocked(report)
    official = DatasetMeta(_resolve_dataset_dir(str(official_dataset_dir)))
    recap = DatasetMeta(_resolve_dataset_dir(str(recap_label_dataset_dir)))
    task_crosswalk, _ = _load_task_crosswalk(
        _resolve_optional_file(
            None if task_crosswalk_json is None else str(task_crosswalk_json)
        ),
        official=official,
        recap=recap,
    )
    frame_crosswalk, _ = _load_episode_frame_crosswalk(
        _resolve_optional_file(
            None
            if episode_frame_crosswalk_jsonl is None
            else str(episode_frame_crosswalk_jsonl)
        )
    )
    if task_crosswalk is None or frame_crosswalk is None:
        raise JoinBlocked(report)
    output_root = (
        Path(output_dir) if Path(output_dir).is_absolute() else (REPO_ROOT / output_dir)
    ).resolve()
    materialized_report = _materialize_joined_dataset(
        official=official,
        recap=recap,
        output_root=output_root,
        report=report,
        task_crosswalk=task_crosswalk,
        frame_crosswalk=frame_crosswalk,
    )
    return JoinMaterialized(materialized_report)


def _persist_reports(output_dir: Path, report: dict[str, object]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    join_report_path = output_dir / "join_report.json"
    unmatched_conflicts_path = output_dir / "unmatched_conflicts.json"
    _write_json(join_report_path, report)
    _write_json(unmatched_conflicts_path, _build_unmatched_conflicts(report))
    return join_report_path, unmatched_conflicts_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strict preflight materializer for joining repo-local recap labels onto the official/native LIBERO 8D dataset."
        )
    )
    parser.add_argument("--official-dataset-dir", required=True)
    parser.add_argument("--recap-label-dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--task-crosswalk-json",
        default="",
        help="Optional explicit task identity repair evidence.",
    )
    parser.add_argument(
        "--episode-frame-crosswalk-jsonl",
        default="",
        help="Optional explicit episode/frame-to-recap-index repair evidence.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(str(args.output_dir))
    output_root = output_dir if output_dir.is_absolute() else (REPO_ROOT / output_dir)
    output_root = output_root.resolve()
    try:
        materialized = materialize_join(
            official_dataset_dir=str(args.official_dataset_dir),
            recap_label_dataset_dir=str(args.recap_label_dataset_dir),
            output_dir=output_root,
            task_crosswalk_json=str(args.task_crosswalk_json),
            episode_frame_crosswalk_jsonl=str(args.episode_frame_crosswalk_jsonl),
        )
        join_report_path, unmatched_conflicts_path = _persist_reports(
            output_root, materialized.payload
        )
        print(f"join_report={join_report_path}", flush=True)
        print(f"unmatched_conflicts={unmatched_conflicts_path}", flush=True)
        print("LIBERO_OFFICIAL_8D_JOIN_MATERIALIZED", flush=True)
        return SUCCESS_EXIT_CODE
    except JoinBlocked as exc:
        join_report_path, unmatched_conflicts_path = _persist_reports(
            output_root, exc.payload
        )
        print(f"join_report={join_report_path}", flush=True)
        print(f"unmatched_conflicts={unmatched_conflicts_path}", flush=True)
        print("LIBERO_OFFICIAL_8D_JOIN_BLOCKED", flush=True)
        return BLOCKED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
