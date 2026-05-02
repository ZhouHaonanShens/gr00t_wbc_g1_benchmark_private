from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_BUCKET_DIR = Path("agent/artifacts/state_conditioned_materialization/bucket_a")
DEFAULT_DEV_DIR = Path("agent/artifacts/state_conditioned_materialization/devbench")
DEFAULT_COLLECTION_DIR = Path(
    "agent/artifacts/state_conditioned_materialization/collection"
)
DEFAULT_HARVEST_DIR = Path("agent/artifacts/state_conditioned_materialization/harvest")
DEFAULT_OUTPUT_DIR = Path("agent/artifacts/state_conditioned_materialization/training")

STATE_CONDITIONED_SFT_LABELS_JSONL_NAME = "state_conditioned_sft_labels.jsonl"
STATE_CONDITIONED_SFT_STATS_JSON_NAME = "state_conditioned_sft_stats.json"
EQUAL_DATA_FAIRNESS_AUDIT_JSON_NAME = "equal_data_fairness_audit.json"
CONDITIONING_CHANNEL_LIVENESS_JSON_NAME = "conditioning_channel_liveness.json"
DEV_ONLY_PROMOTION_GATE_JSON_NAME = "dev_only_promotion_gate.json"
LEROBOT_TRAINING_DATASET_DIRNAME = "lerobot_training_dataset"

LEROBOT_EXPORT_CHUNK_SIZE = 1000
LEROBOT_EXPORT_FPS = 30.0
LEROBOT_VIDEO_IMAGE_KEY = "ego_view"
LEROBOT_VIDEO_ORIGINAL_KEY = "observation.images.ego_view"

SCHEMA_VERSION = "g1_state_conditioned_equal_data_training_set_v1"

SOURCE_BUCKET_CANONICAL_BUCKET_A = "canonical_bucket_A"
SOURCE_BUCKET_BUCKET_B = "bucket_B"
SOURCE_BUCKET_FORMAL_PSEUDODEMO = "formal_pseudodemo"
SOURCE_BUCKET_VALUES: tuple[str, ...] = (
    SOURCE_BUCKET_CANONICAL_BUCKET_A,
    SOURCE_BUCKET_BUCKET_B,
    SOURCE_BUCKET_FORMAL_PSEUDODEMO,
)

VIEW_C0 = "C0"
VIEW_C1 = "C1"
VIEW_VALUES: tuple[str, ...] = (VIEW_C0, VIEW_C1)

NULL_PHASE_TOKEN = "__NULL_PHASE__"
NULL_MODE_TOKEN = "__NULL_MODE__"
RECOVERY_OVERSAMPLE_FACTOR = 3

MAINLINE_TRAINING_TEXT_FIELD = "carrier_text_v1"

CONDITIONING_FIELD_NAMES: tuple[str, ...] = (
    "policy_condition.phase",
    "policy_condition.mode",
    "policy_condition_text",
)

M2_FIELD_NAMES: tuple[str, ...] = (
    "return_G",
    "value_V",
    "advantage_A",
    "epsilon_l",
    "indicator_I",
)

LABEL_DATA_DOMAIN = "training_flat_artifact_only"
LABEL_DATA_VERSION = "state_conditioned_training_flat_artifacts_v2"
M2_BACKFILL_SOURCE = "source_dataset_m2_labels"
M2_BACKFILL_VERSION = "recap_m2_label_fields_v1"
PSEUDODEMO_DATASET_VERSION = "pseudodemo_labels_v2"

SAFE_LABEL_FIELD_ORDER: tuple[str, ...] = (
    "schema_version",
    "training_view",
    "sample_id",
    "sample_index",
    "source_bucket",
    "source_kind",
    "source_episode_id",
    "source_t",
    "source_snapshot_id",
    "pseudodemo.source_snapshot_family",
    "pseudodemo.source_bucket_key",
    "source_anchor_episode_id",
    "source_sample_key",
    "reset_boundary",
    "label_data.domain",
    "label_data.version",
    "label_data.m2_backfill_source",
    "label_data.m2_backfill_version",
    "pseudodemo.teacher_policy_id",
    "pseudodemo.teacher_target",
    "pseudodemo.teacher_target_truthfulness",
    "pseudodemo.label_kind",
    "pseudodemo.dataset_version",
    "budget_group",
    "repeat_index",
    "recovery_oversample_factor",
    "return_G",
    "value_V",
    "advantage_A",
    "epsilon_l",
    "indicator_I",
    MAINLINE_TRAINING_TEXT_FIELD,
    "history_k",
    "history_stride",
    "history_valid_mask",
    "history_t_std_indices",
    "history_t_raw_indices",
    "history_timestamp_s",
    "deployable.previous_action_history",
    "deployable.proprio_history",
    "deployable.short_visual_history_refs",
    "policy_condition.phase",
    "policy_condition.mode",
    "policy_condition_text",
)

ANALYSIS_ONLY_EXACT_FIELD_NAMES: tuple[str, ...] = (
    "event",
    "recovery_needed",
    "semantic_state",
    "summary_template",
    "memory_commit_mask",
    "memory_commit_cause",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap import state_conditioned_bucket_a_sidecar
from work.recap import state_conditioned_collect_buckets
from work.recap import state_conditioned_dev_manifest
from work.recap import state_conditioned_snapshot_harvest
from work.recap import text_indicator
from work.recap.lerobot_export import dataset_export as lerobot_v2_export
from work.recap.lerobot_export import video_export as lerobot_v2_export_with_video
from work.recap.scripts.state_conditioned_common import (
    exception_message as _exception_message,
)
from work.recap.scripts.state_conditioned_common import read_json as _read_json
from work.recap.scripts.state_conditioned_common import (
    read_jsonl_dicts as _read_jsonl_dicts,
)
from work.recap.scripts.state_conditioned_common import (
    validate_existing_dir as _validate_existing_dir,
)
from work.recap.scripts.state_conditioned_common import (
    validate_existing_file as _validate_existing_file,
)
from work.recap.scripts.state_conditioned_common import (
    validate_output_dir as _validate_output_dir,
)
from work.recap.scripts.state_conditioned_common import write_json as _write_json
from work.recap.scripts.state_conditioned_common import write_jsonl as _write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a unified equal-data training set for history-aware C0/C1 state "
            "conditioning from canonical Bucket A, Bucket B, and formal pseudo-demos."
        )
    )
    parser.add_argument(
        "--bucket-dir",
        type=Path,
        default=DEFAULT_BUCKET_DIR,
        help="T4/T5 canonical Bucket A directory.",
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=DEFAULT_DEV_DIR,
        help="T6 devbench directory.",
    )
    parser.add_argument(
        "--collection-dir",
        type=Path,
        default=DEFAULT_COLLECTION_DIR,
        help="T7 collection directory containing Bucket B/C manifests.",
    )
    parser.add_argument(
        "--harvest-dir",
        type=Path,
        default=DEFAULT_HARVEST_DIR,
        help="T8/T9 harvest directory containing feasibility, teacher gate, and pseudo-demo artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives state_conditioned_sft_labels.jsonl and audit artifacts.",
    )
    parser.add_argument(
        "--recovery-oversample-factor",
        type=int,
        default=int(RECOVERY_OVERSAMPLE_FACTOR),
        help="Frozen recovery oversample factor; only 3 is accepted.",
    )
    return parser


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    return state_conditioned_bucket_a_import._as_non_empty_string(
        value,
        field_name=field_name,
    )


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a float-like number")
    return float(value)


def _as_list(
    value: object,
    *,
    field_name: str,
    expected_len: int | None = None,
) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    items = list(value)
    if expected_len is not None and len(items) != int(expected_len):
        raise ValueError(
            f"{field_name} must have length {expected_len}, got {len(items)}"
        )
    return items


def _snapshot_candidate_artifact_path(harvest_dir: Path) -> Path:
    return (
        harvest_dir
        / state_conditioned_snapshot_harvest.OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME
    )


def build_null_policy_condition_text() -> str:
    return (
        "[PolicyCondition-v1]\n"
        + f"PHASE={NULL_PHASE_TOKEN}\n"
        + f"MODE={NULL_MODE_TOKEN}"
    )


def validate_view_policy_condition_text(
    *,
    training_view: str,
    phase: object,
    mode: object,
    policy_condition_text: object,
) -> tuple[str, str, str]:
    normalized_view = _as_non_empty_string(training_view, field_name="training_view")
    if normalized_view not in VIEW_VALUES:
        raise ValueError(f"training_view must be one of {VIEW_VALUES!r}")
    if normalized_view == VIEW_C1:
        return state_conditioned_bucket_a_import.validate_state_conditioned_policy_condition(
            phase=phase,
            mode=mode,
            policy_condition_text=policy_condition_text,
        )
    normalized_phase = _as_non_empty_string(phase, field_name="policy_condition.phase")
    normalized_mode = _as_non_empty_string(mode, field_name="policy_condition.mode")
    normalized_text = _as_non_empty_string(
        policy_condition_text,
        field_name="policy_condition_text",
    )
    if normalized_phase != NULL_PHASE_TOKEN:
        raise ValueError(
            f"C0 policy_condition.phase must be {NULL_PHASE_TOKEN!r}, got {normalized_phase!r}"
        )
    if normalized_mode != NULL_MODE_TOKEN:
        raise ValueError(
            f"C0 policy_condition.mode must be {NULL_MODE_TOKEN!r}, got {normalized_mode!r}"
        )
    expected = build_null_policy_condition_text()
    if normalized_text != expected:
        raise ValueError(
            "C0 policy_condition_text mismatch: "
            + f"expected {expected!r}, got {normalized_text!r}"
        )
    return normalized_phase, normalized_mode, normalized_text


def _normalize_source_bucket(value: object) -> str:
    normalized = _as_non_empty_string(value, field_name="source_bucket")
    if normalized not in SOURCE_BUCKET_VALUES:
        raise ValueError(f"source_bucket must be one of {SOURCE_BUCKET_VALUES!r}")
    return normalized


def _forbidden_output_field_name(field_name: str) -> bool:
    if field_name in ANALYSIS_ONLY_EXACT_FIELD_NAMES:
        return True
    if field_name.startswith("privileged."):
        return True
    if field_name.startswith("oracle."):
        return True
    if field_name.startswith("teacher."):
        return True
    if field_name.startswith("teacher_"):
        return True
    if field_name.startswith("future."):
        return True
    if field_name.startswith("hindsight."):
        return True
    if field_name.startswith("memory_commit_"):
        return True
    if field_name.startswith("recovery_") and field_name.endswith("_step"):
        return True
    return False


def _history_indices_from_window(
    *,
    prehistory_window: Sequence[Mapping[str, Any]],
    history_valid_mask: Sequence[bool],
) -> tuple[list[int], list[int], list[float | None]]:
    history_t_std_indices: list[int] = []
    history_t_raw_indices: list[int] = []
    history_timestamp_s: list[float | None] = []
    for index, item in enumerate(prehistory_window):
        row = dict(item)
        row_t_std = _as_int(
            row.get("t_std"), field_name=f"prehistory_window[{index}].t_std"
        )
        row_t_raw = row.get("t_raw", row.get("t_raw_index", row_t_std))
        row_t_raw_int = _as_int(
            row_t_raw, field_name=f"prehistory_window[{index}].t_raw"
        )
        history_t_std_indices.append(int(row_t_std))
        history_t_raw_indices.append(int(row_t_raw_int))
        history_timestamp_s.append(
            float(row_t_raw_int) if history_valid_mask[index] else None
        )
    return history_t_std_indices, history_t_raw_indices, history_timestamp_s


def _safe_json_signature(payload: object) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_existing_path(raw_value: object, *, field_name: str) -> Path:
    candidate = Path(_as_non_empty_string(raw_value, field_name=field_name))
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise ValueError(f"{field_name} does not exist: {candidate}")
    return candidate


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return _as_non_empty_string(normalized, field_name=field_name)


def _reset_output_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_carrier_text_v1(*, prompt_raw: object, indicator_value: object) -> str:
    indicator_mode = text_indicator.indicator_mode_from_indicator_value(
        indicator_value,
        field_name="indicator_I",
    )
    carrier_text_v1 = text_indicator.build_authoritative_carrier_text_v1(
        prompt_raw,
        indicator_mode,
    )
    return _as_non_empty_string(
        carrier_text_v1,
        field_name=MAINLINE_TRAINING_TEXT_FIELD,
    )


def _build_group_offsets(
    keys: Sequence[str],
    dims_by_key: Mapping[str, int],
) -> dict[str, tuple[int, int]]:
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for key in keys:
        dim = int(dims_by_key[key])
        base_name = key.split("/", 1)[1] if "/" in key else key
        offsets[str(base_name)] = (int(cursor), int(cursor + dim))
        cursor += dim
    return offsets


def _feature_names_for_group(
    keys: Sequence[str],
    dims_by_key: Mapping[str, int],
    *,
    group: str,
) -> list[str]:
    feature_names: list[str] = []
    for key in keys:
        dim = int(dims_by_key[key])
        base_name = key.split("/", 1)[1] if "/" in key else key
        for index in range(dim):
            feature_names.append(f"{group}.{base_name}:{index}")
    return feature_names


def _resolve_episode_npz_path(
    dataset_dir: Path, episode_record: Mapping[str, Any]
) -> Path:
    raw_path = episode_record.get("npz_path")
    if isinstance(raw_path, str) and raw_path.strip():
        candidate = Path(raw_path.strip())
        if not candidate.is_absolute():
            candidate = (dataset_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
    else:
        episode_id = _as_non_empty_string(
            episode_record.get("episode_id"),
            field_name="episode_record.episode_id",
        )
        candidate = (dataset_dir / "arrays" / f"{episode_id}.npz").resolve()
    if not candidate.is_file():
        raise ValueError(f"missing source npz for training export: {candidate}")
    return candidate


def _build_source_episode_dataset_index(
    prerequisites: Mapping[str, Any],
) -> dict[str, Path]:
    index: dict[str, Path] = {}

    def _register(episode_id: str, dataset_dir: Path) -> None:
        existing = index.get(episode_id)
        if existing is not None and existing != dataset_dir:
            raise ValueError(
                "state-conditioned source episode points to multiple dataset roots: "
                + f"{episode_id} -> {existing} vs {dataset_dir}"
            )
        index[episode_id] = dataset_dir

    for raw_entry in list(prerequisites["bucket_manifest"].get("episodes", [])):
        entry = dict(_as_mapping(raw_entry, field_name="bucket_manifest.episodes[]"))
        if not bool(entry.get("accepted", False)):
            continue
        episode_id = _as_non_empty_string(
            entry.get("episode_id"), field_name="episode_id"
        )
        source_dataset_dir = _normalize_existing_path(
            entry.get("source_dataset_dir"),
            field_name=f"bucket_manifest.episodes[{episode_id}].source_dataset_dir",
        )
        _register(episode_id, source_dataset_dir)

    for raw_entry in list(prerequisites["bucket_b_manifest"].get("episodes", [])):
        entry = dict(_as_mapping(raw_entry, field_name="bucket_B_manifest.episodes[]"))
        episode_id = _as_non_empty_string(
            entry.get("episode_id"), field_name="episode_id"
        )
        dataset_dir = _normalize_existing_path(
            entry.get("dataset_dir"),
            field_name=f"bucket_B_manifest.episodes[{episode_id}].dataset_dir",
        )
        _register(episode_id, dataset_dir)

    for raw_entry in list(prerequisites["bucket_c_manifest"].get("episodes", [])):
        entry = dict(_as_mapping(raw_entry, field_name="bucket_C_manifest.episodes[]"))
        episode_id = _as_non_empty_string(
            entry.get("episode_id"), field_name="episode_id"
        )
        dataset_dir = _normalize_existing_path(
            entry.get("dataset_dir"),
            field_name=f"bucket_C_manifest.episodes[{episode_id}].dataset_dir",
        )
        _register(episode_id, dataset_dir)
    return index


def _build_formal_pseudodemo_snapshot_index(
    prerequisites: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    snapshot_index = _load_snapshot_candidate_index(
        Path(prerequisites["snapshot_candidates_path"])
    )
    pseudodemo_by_episode_id: dict[str, dict[str, Any]] = {}
    for raw_record in list(prerequisites["pseudodemo_manifest"].get("pseudodemos", [])):
        record = dict(
            _as_mapping(raw_record, field_name="pseudodemo_manifest.pseudodemos[]")
        )
        episode_id = _as_non_empty_string(
            record.get("episode_id"), field_name="episode_id"
        )
        if episode_id in pseudodemo_by_episode_id:
            raise ValueError(f"duplicate formal pseudodemo episode_id: {episode_id}")
        pseudodemo_by_episode_id[episode_id] = record
    return snapshot_index, pseudodemo_by_episode_id


def _load_source_dataset_cache(
    dataset_dir: Path,
    *,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = str(dataset_dir.resolve())
    if key not in cache:
        cache[key] = state_conditioned_bucket_a_import._load_dataset_records(
            dataset_dir
        )
    return cache[key]


def _load_npz_array_cache(
    npz_path: Path,
    *,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = str(npz_path.resolve())
    if key not in cache:
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError(
                f"numpy is required for training-set export: {exc}"
            ) from exc

        with np.load(npz_path, allow_pickle=False) as data:
            cache[key] = {name: np.asarray(data[name]) for name in list(data.files)}
    return cache[key]


def _source_step_export_spec(
    raw_row: Mapping[str, Any],
    *,
    prerequisites: Mapping[str, Any],
    episode_dataset_index: Mapping[str, Path],
    snapshot_index: Mapping[str, Mapping[str, Any]],
    formal_pseudodemos: Mapping[str, Mapping[str, Any]],
    dataset_cache: dict[str, dict[str, Any]],
    npz_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row = dict(raw_row)
    source_bucket = _normalize_source_bucket(row.get("source_bucket"))
    source_episode_id = _as_non_empty_string(
        row.get("source_episode_id"), field_name="source_episode_id"
    )
    source_t = _as_int(row.get("source_t"), field_name="source_t")
    source_dataset_episode_id = source_episode_id

    if source_bucket == SOURCE_BUCKET_FORMAL_PSEUDODEMO:
        formal_record = dict(
            _as_mapping(
                formal_pseudodemos.get(source_episode_id),
                field_name=f"formal_pseudemos[{source_episode_id}]",
            )
        )
        source_snapshot_id = _as_non_empty_string(
            formal_record.get("source_snapshot_id"),
            field_name="formal_pseudodemo.source_snapshot_id",
        )
        snapshot_record = dict(
            _as_mapping(
                snapshot_index.get(source_snapshot_id),
                field_name=f"snapshot_index[{source_snapshot_id}]",
            )
        )
        source_dataset_episode_id = _as_non_empty_string(
            snapshot_record.get("anchor_episode_id"),
            field_name="snapshot_record.anchor_episode_id",
        )
        source_t = _as_int(
            snapshot_record.get("anchor_t"), field_name="snapshot_record.anchor_t"
        )

    dataset_dir = episode_dataset_index.get(source_dataset_episode_id)
    if dataset_dir is None:
        raise ValueError(
            "missing source dataset root for training export episode: "
            + source_dataset_episode_id
        )

    dataset_records = _load_source_dataset_cache(dataset_dir, cache=dataset_cache)
    episode_record = dict(
        _as_mapping(
            dataset_records["episodes_by_id"].get(source_dataset_episode_id),
            field_name=f"episodes_by_id[{source_dataset_episode_id}]",
        )
    )
    transitions = list(
        dataset_records["transitions_by_episode"].get(source_dataset_episode_id, [])
    )
    if source_t < 0 or source_t >= len(transitions):
        raise ValueError(
            f"source_t={source_t} is out of range for {source_dataset_episode_id}; transitions={len(transitions)}"
        )
    transition = dict(
        _as_mapping(transitions[source_t], field_name="source_transition")
    )
    transition_t = _as_int(transition.get("t"), field_name="source_transition.t")
    if transition_t != int(source_t):
        raise ValueError(
            f"source transition order drifted for {source_dataset_episode_id}: expected t={source_t}, got {transition_t}"
        )
    n_exec = _as_int(
        transition.get("n_action_steps_executed"),
        field_name="source_transition.n_action_steps_executed",
    )
    if n_exec <= 0:
        raise ValueError(
            f"source transition n_action_steps_executed must be positive for {source_dataset_episode_id}"
        )

    npz_path = _resolve_episode_npz_path(dataset_dir, episode_record)
    npz_data = _load_npz_array_cache(npz_path, cache=npz_cache)
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"numpy is required for training-set export: {exc}") from exc

    state_dims = {
        key: int(np.asarray(npz_data[key]).shape[3])
        for key in lerobot_v2_export.STATE_KEY_ORDER_LOCK
    }
    action_dims = {
        key: int(np.asarray(npz_data[key]).shape[3])
        for key in lerobot_v2_export.ACTION_KEY_ORDER_LOCK
    }

    state_vector = np.concatenate(
        [
            np.asarray(npz_data[key][source_t, 0, 0, :], dtype=np.float32)
            for key in lerobot_v2_export.STATE_KEY_ORDER_LOCK
        ],
        axis=-1,
    )
    action_vector = np.concatenate(
        [
            np.asarray(npz_data[key][source_t, 0, :n_exec, :], dtype=np.float32)
            for key in lerobot_v2_export.ACTION_KEY_ORDER_LOCK
        ],
        axis=-1,
    )
    if action_vector.ndim != 2 or int(action_vector.shape[0]) != int(n_exec):
        raise ValueError(
            f"source action slice has wrong shape for {source_dataset_episode_id}: {action_vector.shape}"
        )

    task_text = _as_non_empty_string(
        row.get(MAINLINE_TRAINING_TEXT_FIELD),
        field_name=MAINLINE_TRAINING_TEXT_FIELD,
    )
    source_video_dir_archived = _as_non_empty_string(
        episode_record.get("video_dir_archived"),
        field_name=f"episodes_by_id[{source_dataset_episode_id}].video_dir_archived",
    )
    return {
        "source_dataset_dir": dataset_dir,
        "source_dataset_episode_id": source_dataset_episode_id,
        "source_t": int(source_t),
        "source_n_policy_steps": int(len(transitions)),
        "source_video_dir_archived": source_video_dir_archived,
        "task_text": task_text,
        "state_vector": state_vector,
        "action_vector": action_vector,
        "state_dims": state_dims,
        "action_dims": action_dims,
    }


def _build_episode_video_export_spec(
    *,
    episode_index: int,
    frame_count: int,
    export_spec: Mapping[str, Any],
    source_video_cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    source_episode_id = _as_non_empty_string(
        export_spec.get("source_dataset_episode_id"),
        field_name="export_spec.source_dataset_episode_id",
    )
    source_video_dir_archived = _as_non_empty_string(
        export_spec.get("source_video_dir_archived"),
        field_name="export_spec.source_video_dir_archived",
    )
    source_t = _as_int(export_spec.get("source_t"), field_name="export_spec.source_t")
    source_n_policy_steps = _as_int(
        export_spec.get("source_n_policy_steps"),
        field_name="export_spec.source_n_policy_steps",
    )
    if source_n_policy_steps <= 0:
        raise ValueError(
            f"source_n_policy_steps must be positive for {source_episode_id}, got {source_n_policy_steps}"
        )
    if source_t < 0 or source_t >= source_n_policy_steps:
        raise ValueError(
            f"source_t out of range for {source_episode_id}: t={source_t} n_policy_steps={source_n_policy_steps}"
        )
    if frame_count <= 0:
        raise ValueError(
            f"training export frame_count must be positive for {source_episode_id}, got {frame_count}"
        )

    cache_key = (source_episode_id, source_video_dir_archived)
    cached = source_video_cache.get(cache_key)
    if cached is None:
        src_mp4 = lerobot_v2_export_with_video.resolve_episode_video_path(
            video_dir_archived=source_video_dir_archived,
            episode_id=source_episode_id,
        )
        source_video_frame_count = (
            lerobot_v2_export_with_video.resolve_video_frame_count(src_mp4)
        )
        cached = {
            "src_mp4": str(src_mp4),
            "source_video_frame_count": int(source_video_frame_count),
        }
        source_video_cache[cache_key] = cached

    source_video_frame_count = _as_int(
        cached.get("source_video_frame_count"),
        field_name=f"source_video_cache[{cache_key!r}].source_video_frame_count",
    )
    start_frame = int(
        (int(source_t) * int(source_video_frame_count)) // int(source_n_policy_steps)
    )
    next_boundary_frame = int(source_video_frame_count)
    if source_t + 1 < source_n_policy_steps:
        next_boundary_frame = int(
            ((int(source_t) + 1) * int(source_video_frame_count))
            // int(source_n_policy_steps)
        )
    end_frame = min(
        int(source_video_frame_count) - 1,
        max(int(start_frame + frame_count - 1), int(next_boundary_frame - 1)),
    )
    if end_frame < start_frame:
        end_frame = int(start_frame)

    return {
        "episode_index": int(episode_index),
        "source_episode_id": source_episode_id,
        "source_t": int(source_t),
        "source_n_policy_steps": int(source_n_policy_steps),
        "source_video_frame_count": int(source_video_frame_count),
        "desired_frames": int(frame_count),
        "src_mp4": str(cached["src_mp4"]),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
    }


def materialize_lerobot_training_dataset(
    *,
    output_dir: Path,
    base_rows: Sequence[Mapping[str, Any]],
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            f"pandas+numpy are required for LeRobot training export: {exc}"
        ) from exc

    dataset_root = _reset_output_dir(
        (output_dir / LEROBOT_TRAINING_DATASET_DIRNAME).resolve()
    )
    meta_dir = _reset_output_dir(dataset_root / lerobot_v2_export.META_DIRNAME)
    data_dir = _reset_output_dir(dataset_root / lerobot_v2_export.DATA_DIRNAME)

    episode_dataset_index = _build_source_episode_dataset_index(prerequisites)
    snapshot_index, formal_pseudodemos = _build_formal_pseudodemo_snapshot_index(
        prerequisites
    )
    dataset_cache: dict[str, dict[str, Any]] = {}
    npz_cache: dict[str, dict[str, Any]] = {}

    task_texts = sorted(
        {
            _as_non_empty_string(
                row.get(MAINLINE_TRAINING_TEXT_FIELD),
                field_name=MAINLINE_TRAINING_TEXT_FIELD,
            )
            for row in base_rows
        }
    )
    task_to_index = {task: index for index, task in enumerate(task_texts)}

    total_frames = 0
    all_state_rows: list[Any] = []
    all_action_rows: list[Any] = []
    episodes_meta_out: list[dict[str, Any]] = []
    video_export_specs: list[dict[str, Any]] = []
    state_dims_lock: dict[str, int] | None = None
    action_dims_lock: dict[str, int] | None = None
    source_video_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for episode_index, raw_row in enumerate(base_rows):
        row = dict(raw_row)
        export_spec = _source_step_export_spec(
            row,
            prerequisites=prerequisites,
            episode_dataset_index=episode_dataset_index,
            snapshot_index=snapshot_index,
            formal_pseudodemos=formal_pseudodemos,
            dataset_cache=dataset_cache,
            npz_cache=npz_cache,
        )
        if state_dims_lock is None:
            state_dims_lock = dict(export_spec["state_dims"])
        elif state_dims_lock != dict(export_spec["state_dims"]):
            raise ValueError(
                "state dims drifted across source datasets during LeRobot export"
            )
        if action_dims_lock is None:
            action_dims_lock = dict(export_spec["action_dims"])
        elif action_dims_lock != dict(export_spec["action_dims"]):
            raise ValueError(
                "action dims drifted across source datasets during LeRobot export"
            )

        state_vector = np.asarray(export_spec["state_vector"], dtype=np.float32)
        action_vector = np.asarray(export_spec["action_vector"], dtype=np.float32)
        task_text = str(export_spec["task_text"])
        task_index = int(task_to_index[task_text])
        frame_count = int(action_vector.shape[0])
        state_rows = [state_vector.copy() for _ in range(frame_count)]
        action_rows = [np.asarray(item, dtype=np.float32) for item in action_vector]

        df = pd.DataFrame(
            {
                "observation.state": state_rows,
                "action": action_rows,
                "timestamp": [
                    float(frame_idx) / float(LEROBOT_EXPORT_FPS)
                    for frame_idx in range(frame_count)
                ],
                "episode_index": [int(episode_index)] * frame_count,
                "index": list(range(frame_count)),
                lerobot_v2_export.LANGUAGE_ANNOTATION_KEY: [int(task_index)]
                * frame_count,
            }
        )

        chunk_index = int(episode_index) // int(LEROBOT_EXPORT_CHUNK_SIZE)
        chunk_dir = data_dir / f"chunk-{chunk_index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_dir / f"episode_{int(episode_index):06d}.parquet"
        df.to_parquet(parquet_path, engine="pyarrow", index=False)

        episodes_meta_out.append(
            {
                "episode_index": int(episode_index),
                "tasks": [task_text],
                "length": int(frame_count),
                "state_conditioned.sample_id": str(row["sample_id"]),
                "state_conditioned.source_bucket": str(row["source_bucket"]),
                "state_conditioned.source_episode_id": str(
                    export_spec["source_dataset_episode_id"]
                ),
                "state_conditioned.source_t": int(export_spec["source_t"]),
            }
        )
        video_export_specs.append(
            _build_episode_video_export_spec(
                episode_index=int(episode_index),
                frame_count=int(frame_count),
                export_spec=export_spec,
                source_video_cache=source_video_cache,
            )
        )
        total_frames += int(frame_count)
        all_state_rows.append(np.stack(state_rows, axis=0))
        all_action_rows.append(np.stack(action_rows, axis=0))

    if state_dims_lock is None or action_dims_lock is None:
        raise ValueError("LeRobot training export received no base rows")

    state_dim = int(sum(int(value) for value in state_dims_lock.values()))
    action_dim = int(sum(int(value) for value in action_dims_lock.values()))
    state_offsets = _build_group_offsets(
        lerobot_v2_export.STATE_KEY_ORDER_LOCK,
        state_dims_lock,
    )
    action_offsets = _build_group_offsets(
        lerobot_v2_export.ACTION_KEY_ORDER_LOCK,
        action_dims_lock,
    )

    _write_jsonl(
        meta_dir / lerobot_v2_export.META_TASKS_JSONL,
        [
            {"task_index": int(index), "task": str(task)}
            for index, task in enumerate(task_texts)
        ],
    )
    _write_jsonl(meta_dir / lerobot_v2_export.META_EPISODES_JSONL, episodes_meta_out)
    _write_json(
        meta_dir / lerobot_v2_export.META_MODALITY_JSON,
        {
            "state": {
                lerobot_v2_export.WBC_STATE_GROUP_KEY: {
                    "start": 0,
                    "end": int(state_dim),
                },
                **{
                    key: {
                        "start": int(start),
                        "end": int(end),
                        "original_key": "observation.state",
                    }
                    for key, (start, end) in state_offsets.items()
                },
            },
            "action": {
                lerobot_v2_export.WBC_ACTION_GROUP_KEY: {
                    "start": 0,
                    "end": int(action_dim),
                },
                **{
                    key: {
                        "start": int(start),
                        "end": int(end),
                        "original_key": "action",
                    }
                    for key, (start, end) in action_offsets.items()
                },
            },
            "annotation": {
                "human.action.task_description": {
                    "original_key": lerobot_v2_export.LANGUAGE_ANNOTATION_KEY,
                },
                "human.task_description": {
                    "original_key": lerobot_v2_export.LANGUAGE_ANNOTATION_KEY,
                },
            },
        },
    )

    state_matrix = np.concatenate(all_state_rows, axis=0).astype(np.float32, copy=False)
    action_matrix = np.concatenate(all_action_rows, axis=0).astype(
        np.float32, copy=False
    )

    def _stats_for(matrix: Any) -> dict[str, list[float]]:
        q01, q99 = np.quantile(matrix, [0.01, 0.99], axis=0)
        return {
            "mean": [float(item) for item in matrix.mean(axis=0)],
            "std": [float(item) for item in matrix.std(axis=0)],
            "min": [float(item) for item in matrix.min(axis=0)],
            "max": [float(item) for item in matrix.max(axis=0)],
            "q01": [float(item) for item in q01],
            "q99": [float(item) for item in q99],
        }

    _write_json(
        meta_dir / lerobot_v2_export.META_STATS_JSON,
        {
            "observation.state": _stats_for(state_matrix),
            "action": _stats_for(action_matrix),
        },
    )

    _write_json(
        meta_dir / lerobot_v2_export.META_INFO_JSON,
        {
            "codebase_version": "state-conditioned-training-set-v1",
            "robot_type": "unitree_g1_wbc",
            "total_episodes": int(len(episodes_meta_out)),
            "total_frames": int(total_frames),
            "total_tasks": int(len(task_texts)),
            "chunks_size": int(LEROBOT_EXPORT_CHUNK_SIZE),
            "fps": float(LEROBOT_EXPORT_FPS),
            "splits": {"train": "0:100"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [int(action_dim)],
                    "names": _feature_names_for_group(
                        lerobot_v2_export.ACTION_KEY_ORDER_LOCK,
                        action_dims_lock,
                        group="wbc_action",
                    ),
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [int(state_dim)],
                    "names": _feature_names_for_group(
                        lerobot_v2_export.STATE_KEY_ORDER_LOCK,
                        state_dims_lock,
                        group="wbc_state",
                    ),
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                lerobot_v2_export.LANGUAGE_ANNOTATION_KEY: {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None,
                },
                lerobot_v2_export.LANGUAGE_ANNOTATION_KEY_ALIAS: {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None,
                },
            },
            "task_text_field": MAINLINE_TRAINING_TEXT_FIELD,
            "carrier_schema_version": text_indicator.RECAP_TEXT_INDICATOR_SCHEMA_VERSION,
            "carrier_route": text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD,
            "prompt_source_field": text_indicator.RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD,
            "total_chunks": int(
                (len(episodes_meta_out) - 1) // int(LEROBOT_EXPORT_CHUNK_SIZE) + 1
            ),
            "total_videos": 0,
            "field_groups": {
                "state": [lerobot_v2_export.WBC_STATE_GROUP_KEY],
                "action": [lerobot_v2_export.WBC_ACTION_GROUP_KEY],
                "state_conditioned_sidecar": lerobot_v2_export.build_state_conditioned_field_groups(),
            },
        },
    )
    video_result = (
        lerobot_v2_export_with_video.attach_videos_to_existing_lerobot_dataset(
            output_dataset_dir=dataset_root,
            episode_video_specs=video_export_specs,
            fps=float(LEROBOT_EXPORT_FPS),
            chunk_size=int(LEROBOT_EXPORT_CHUNK_SIZE),
            image_key=LEROBOT_VIDEO_IMAGE_KEY,
            original_key=LEROBOT_VIDEO_ORIGINAL_KEY,
            require_ffmpeg=True,
        )
    )
    return {
        "dataset_root": str(dataset_root),
        "meta_info_path": str(meta_dir / lerobot_v2_export.META_INFO_JSON),
        "video_map_path": str(video_result.video_map_path),
        "episode_count": int(len(episodes_meta_out)),
        "frame_count": int(total_frames),
        "task_count": int(len(task_texts)),
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "video_count": int(video_result.total_videos),
    }


def _accepted_canonical_episode_ids(manifest: Mapping[str, Any]) -> list[str]:
    accepted: list[str] = []
    seen: set[str] = set()
    for raw_entry in list(manifest.get("episodes", [])):
        entry = dict(_as_mapping(raw_entry, field_name="bucket_A_manifest.episodes[]"))
        if not bool(entry.get("accepted", False)):
            continue
        if not bool(entry.get("fresh_nominal_recollection", False)):
            continue
        if bool(entry.get("debug_only", False)):
            continue
        if bool(entry.get("reused_existing_live_dataset", True)):
            continue
        episode_id = _as_non_empty_string(
            entry.get("episode_id"), field_name="episode_id"
        )
        if episode_id in seen:
            raise ValueError(f"duplicate canonical Bucket A episode_id: {episode_id}")
        seen.add(episode_id)
        accepted.append(episode_id)
    if len(accepted) != int(
        state_conditioned_bucket_a_sidecar.EXPECTED_ACCEPTED_EPISODE_COUNT
    ):
        raise ValueError(
            "canonical Bucket A manifest must contain exactly 24 accepted fresh episodes, got "
            + str(len(accepted))
        )
    return accepted


def _resolve_debug_only_reuse_manifest_path(
    bucket_a_manifest: Mapping[str, Any],
) -> Path | None:
    raw_path = bucket_a_manifest.get("debug_only_reuse_manifest_path")
    if raw_path is None:
        return None
    normalized = str(raw_path).strip()
    if not normalized:
        return None
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate if candidate.is_file() else None


def _resolve_versioned_harvest_dir(
    *, harvest_dir: Path, output_dir: Path
) -> Path | None:
    variant_name = output_dir.name.strip()
    if not variant_name:
        return None
    candidate = (harvest_dir / variant_name).resolve()
    if not candidate.is_dir():
        return None
    return candidate


def _resolve_harvest_artifact_path(
    *,
    harvest_dir: Path,
    output_dir: Path,
    artifact_name: str,
    arg_name: str,
) -> Path:
    variant_dir = _resolve_versioned_harvest_dir(
        harvest_dir=harvest_dir,
        output_dir=output_dir,
    )
    candidate_paths: list[Path] = []
    if variant_dir is not None:
        candidate_paths.append(variant_dir / artifact_name)
    candidate_paths.append(harvest_dir / artifact_name)
    for candidate in candidate_paths:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise ValueError(
        f"missing required {arg_name}: {candidate_paths[0].expanduser().resolve()}"
    )


def _resolve_snapshot_candidates_path(
    *,
    harvest_dir: Path,
    feasibility_report: Mapping[str, Any],
    pseudodemo_manifest: Mapping[str, Any],
) -> Path:
    candidate_values = (
        pseudodemo_manifest.get("snapshot_candidates_path"),
        feasibility_report.get("snapshot_candidates_path"),
        feasibility_report.get("snapshot_candidates_source_path"),
        str(_snapshot_candidate_artifact_path(harvest_dir)),
    )
    for raw_value in candidate_values:
        if raw_value is None:
            continue
        normalized = str(raw_value).strip()
        if not normalized:
            continue
        candidate = Path(normalized)
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise ValueError(
        "missing T8 snapshot candidate context required for formal pseudo-demo joins"
    )


def _load_upstream_artifacts(
    *,
    bucket_dir: Path,
    dev_dir: Path,
    collection_dir: Path,
    harvest_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    bucket_dir = _validate_existing_dir(bucket_dir, arg_name="bucket-dir")
    dev_dir = _validate_existing_dir(dev_dir, arg_name="dev-dir")
    collection_dir = _validate_existing_dir(collection_dir, arg_name="collection-dir")
    harvest_dir = _validate_existing_dir(harvest_dir, arg_name="harvest-dir")

    bucket_gate_path = _validate_existing_file(
        bucket_dir / state_conditioned_bucket_a_import.GATE_A_READY_JSON_NAME,
        arg_name="T4/T5 bucket_A_gate_a_ready.json",
    )
    bucket_manifest_path = _validate_existing_file(
        bucket_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME,
        arg_name="T4 canonical bucket_A_manifest.json",
    )
    bucket_sidecar_path = _validate_existing_file(
        bucket_dir / state_conditioned_bucket_a_sidecar.BUCKET_A_SIDECAR_JSON_NAME,
        arg_name="T5 bucket_A_sidecar.jsonl",
    )
    bucket_join_coverage_path = _validate_existing_file(
        bucket_dir
        / state_conditioned_bucket_a_sidecar.BUCKET_A_JOIN_COVERAGE_JSON_NAME,
        arg_name="T5 bucket_A_join_coverage.json",
    )
    bucket_exporter_manifest_path = _validate_existing_file(
        bucket_dir
        / state_conditioned_bucket_a_sidecar.BUCKET_A_EXPORTER_MANIFEST_JSON_NAME,
        arg_name="T5 bucket_A_exporter_manifest.json",
    )

    fixed_strata_path = _validate_existing_file(
        dev_dir / state_conditioned_dev_manifest.FIXED_STRATA_DEFINITION_JSON_NAME,
        arg_name="T6 fixed_strata_definition.json",
    )
    baseline_manifest_path = _validate_existing_file(
        dev_dir / state_conditioned_dev_manifest.BASELINE_MANIFEST_JSON_NAME,
        arg_name="T6 baseline_manifest.json",
    )
    baseline_scorecard_path = _validate_existing_file(
        dev_dir / state_conditioned_dev_manifest.BASELINE_DEV_SCORECARD_JSON_NAME,
        arg_name="T6 baseline_dev_scorecard.json",
    )

    bucket_b_manifest_path = _validate_existing_file(
        collection_dir / state_conditioned_collect_buckets.BUCKET_B_MANIFEST_JSON_NAME,
        arg_name="T7 bucket_B_manifest.json",
    )
    bucket_c_manifest_path = _validate_existing_file(
        collection_dir / state_conditioned_collect_buckets.BUCKET_C_MANIFEST_JSON_NAME,
        arg_name="T7 bucket_C_manifest.json",
    )
    bucket_summary_path = _validate_existing_file(
        collection_dir
        / state_conditioned_collect_buckets.BUCKET_COLLECTION_SUMMARY_JSON_NAME,
        arg_name="T7 bucket_collection_summary.json",
    )

    feasibility_report_path = _resolve_harvest_artifact_path(
        harvest_dir=harvest_dir,
        output_dir=output_dir,
        artifact_name=state_conditioned_snapshot_harvest.FEASIBILITY_REPORT_JSON_NAME,
        arg_name="T8 snapshot_feasibility_report.json",
    )
    teacher_gate_report_path = _resolve_harvest_artifact_path(
        harvest_dir=harvest_dir,
        output_dir=output_dir,
        artifact_name=state_conditioned_snapshot_harvest.TEACHER_GATE_REPORT_JSON_NAME,
        arg_name="T8 teacher_gate_report.json",
    )
    pseudodemo_manifest_path = _resolve_harvest_artifact_path(
        harvest_dir=harvest_dir,
        output_dir=output_dir,
        artifact_name=state_conditioned_snapshot_harvest.LOCAL_RECOVERY_PSEUDODEMO_MANIFEST_JSON_NAME,
        arg_name="T9 local_recovery_pseudodemo_manifest.json",
    )

    bucket_gate = _read_json(bucket_gate_path)
    bucket_manifest = _read_json(bucket_manifest_path)
    bucket_join_coverage = _read_json(bucket_join_coverage_path)
    bucket_exporter_manifest = _read_json(bucket_exporter_manifest_path)
    fixed_strata = _read_json(fixed_strata_path)
    baseline_manifest = _read_json(baseline_manifest_path)
    baseline_scorecard = _read_json(baseline_scorecard_path)
    bucket_b_manifest = _read_json(bucket_b_manifest_path)
    bucket_c_manifest = _read_json(bucket_c_manifest_path)
    bucket_summary = _read_json(bucket_summary_path)
    feasibility_report = _read_json(feasibility_report_path)
    teacher_gate_report = _read_json(teacher_gate_report_path)
    pseudodemo_manifest = _read_json(pseudodemo_manifest_path)

    if not bool(bucket_gate.get("ready", False)):
        raise ValueError(
            "training-set builder refuses to run until bucket_A_gate_a_ready.json.ready == true"
        )
    if float(bucket_join_coverage.get("coverage_ratio", 0.0)) < float(
        state_conditioned_bucket_a_import.JOIN_COVERAGE_THRESHOLD
    ):
        raise ValueError("Bucket A join coverage is below the frozen threshold")

    accepted_episode_ids = _accepted_canonical_episode_ids(bucket_manifest)
    if int(bucket_gate.get("accepted_episode_count", len(accepted_episode_ids))) != int(
        len(accepted_episode_ids)
    ):
        raise ValueError("bucket_A_gate_a_ready.json accepted_episode_count mismatch")
    exporter_field_groups = lerobot_v2_export.validate_state_conditioned_field_groups(
        dict(
            _as_mapping(
                bucket_exporter_manifest.get("field_groups"),
                field_name="bucket_A_exporter_manifest.field_groups",
            )
        )
    )
    if exporter_field_groups[lerobot_v2_export.DEPLOYABLE_HISTORY_GROUP_KEY] != list(
        lerobot_v2_export.DEPLOYABLE_HISTORY_FIELD_NAMES
    ):
        raise ValueError(
            "Bucket A exporter manifest deployable_history allowlist drifted"
        )

    paired_seed_values = _as_list(
        fixed_strata.get("paired_seed_values"),
        field_name="fixed_strata_definition.paired_seed_values",
    )
    paired_seed_count = _as_int(
        fixed_strata.get("paired_seed_count"),
        field_name="fixed_strata_definition.paired_seed_count",
    )
    if paired_seed_count != len(paired_seed_values) or paired_seed_count != 8:
        raise ValueError("T6 paired seed contract mismatch")
    if _as_int(
        _as_mapping(
            baseline_scorecard.get("counts"),
            field_name="baseline_dev_scorecard.counts",
        ).get("requested_entries"),
        field_name="baseline_dev_scorecard.counts.requested_entries",
    ) != int(sum(state_conditioned_dev_manifest.EXPECTED_STRATA_COUNTS.values())):
        raise ValueError("T6 baseline_dev_scorecard requested_entries mismatch")

    bucket_b_entries = list(bucket_b_manifest.get("episodes", []))
    bucket_b_count = _as_int(
        _as_mapping(
            bucket_b_manifest.get("counts"), field_name="bucket_B_manifest.counts"
        ).get("episodes"),
        field_name="bucket_B_manifest.counts.episodes",
    )
    if bucket_b_count != int(state_conditioned_collect_buckets.DEFAULT_BUCKET_B_TARGET):
        raise ValueError("T7 bucket_B manifest must contain exactly 16 episodes")
    if len(bucket_b_entries) != int(bucket_b_count):
        raise ValueError("T7 bucket_B manifest episodes list/count mismatch")

    bucket_c_counts = _as_mapping(
        bucket_c_manifest.get("counts"), field_name="bucket_C_manifest.counts"
    )
    bucket_c_count = _as_int(
        bucket_c_counts.get("episodes"),
        field_name="bucket_C_manifest.counts.episodes",
    )
    if bucket_c_count != int(state_conditioned_collect_buckets.DEFAULT_BUCKET_C_TARGET):
        raise ValueError("T7 bucket_C manifest must contain exactly 24 episodes")
    summary_counts = _as_mapping(
        bucket_summary.get("counts"), field_name="bucket_summary.counts"
    )
    if _as_int(
        summary_counts.get("bucket_B"), field_name="bucket_summary.counts.bucket_B"
    ) != int(state_conditioned_collect_buckets.DEFAULT_BUCKET_B_TARGET):
        raise ValueError("T7 bucket summary bucket_B count mismatch")
    if _as_int(
        summary_counts.get("bucket_C"), field_name="bucket_summary.counts.bucket_C"
    ) != int(state_conditioned_collect_buckets.DEFAULT_BUCKET_C_TARGET):
        raise ValueError("T7 bucket summary bucket_C count mismatch")

    if (
        feasibility_report.get("artifact_kind")
        != "state_conditioned_snapshot_feasibility_report"
    ):
        raise ValueError("T8 snapshot_feasibility_report.json artifact_kind mismatch")
    if (
        teacher_gate_report.get("artifact_kind")
        != "state_conditioned_teacher_gate_report"
    ):
        raise ValueError("T8 teacher_gate_report.json artifact_kind mismatch")
    if feasibility_report.get("mode") != "feasibility":
        raise ValueError("T8 snapshot_feasibility_report.json mode mismatch")
    if teacher_gate_report.get("mode") != "feasibility":
        raise ValueError("T8 teacher_gate_report.json mode mismatch")
    if list(feasibility_report.get("family_order", [])) != list(
        state_conditioned_snapshot_harvest.T8_FAMILY_ORDER
    ):
        raise ValueError("T8 snapshot feasibility family_order mismatch")
    if list(teacher_gate_report.get("family_order", [])) != list(
        state_conditioned_snapshot_harvest.T8_FAMILY_ORDER
    ):
        raise ValueError("T8 teacher gate family_order mismatch")

    if pseudodemo_manifest.get("artifact_kind") != "local_recovery_pseudodemo_manifest":
        raise ValueError(
            "T9 local_recovery_pseudodemo_manifest.json artifact_kind mismatch"
        )
    if pseudodemo_manifest.get("mode") != "formal":
        raise ValueError("T9 local_recovery_pseudodemo_manifest.json mode mismatch")
    if _as_int(
        pseudodemo_manifest.get("history_k"), field_name="pseudodemo_manifest.history_k"
    ) != int(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K):
        raise ValueError("T9 pseudodemo history_k mismatch")
    pseudodemos = [
        dict(_as_mapping(item, field_name="pseudodemo_manifest.pseudodemos[]"))
        for item in list(pseudodemo_manifest.get("pseudodemos", []))
    ]
    if not pseudodemos:
        raise ValueError("T9 pseudodemo manifest must be non-empty for T10")

    snapshot_candidates_path = _resolve_snapshot_candidates_path(
        harvest_dir=harvest_dir,
        feasibility_report=feasibility_report,
        pseudodemo_manifest=pseudodemo_manifest,
    )
    debug_only_reuse_manifest_path = _resolve_debug_only_reuse_manifest_path(
        bucket_manifest
    )
    return {
        "bucket_dir": bucket_dir,
        "dev_dir": dev_dir,
        "collection_dir": collection_dir,
        "harvest_dir": harvest_dir,
        "bucket_gate_path": bucket_gate_path,
        "bucket_manifest_path": bucket_manifest_path,
        "bucket_sidecar_path": bucket_sidecar_path,
        "bucket_join_coverage_path": bucket_join_coverage_path,
        "bucket_exporter_manifest_path": bucket_exporter_manifest_path,
        "fixed_strata_path": fixed_strata_path,
        "baseline_manifest_path": baseline_manifest_path,
        "baseline_scorecard_path": baseline_scorecard_path,
        "bucket_b_manifest_path": bucket_b_manifest_path,
        "bucket_c_manifest_path": bucket_c_manifest_path,
        "bucket_summary_path": bucket_summary_path,
        "feasibility_report_path": feasibility_report_path,
        "teacher_gate_report_path": teacher_gate_report_path,
        "pseudodemo_manifest_path": pseudodemo_manifest_path,
        "snapshot_candidates_path": snapshot_candidates_path,
        "debug_only_reuse_manifest_path": debug_only_reuse_manifest_path,
        "bucket_gate": bucket_gate,
        "bucket_manifest": bucket_manifest,
        "bucket_exporter_manifest": bucket_exporter_manifest,
        "baseline_manifest": baseline_manifest,
        "baseline_scorecard": baseline_scorecard,
        "bucket_b_manifest": bucket_b_manifest,
        "bucket_c_manifest": bucket_c_manifest,
        "bucket_summary": bucket_summary,
        "feasibility_report": feasibility_report,
        "teacher_gate_report": teacher_gate_report,
        "pseudodemo_manifest": pseudodemo_manifest,
        "accepted_bucket_a_episode_ids": accepted_episode_ids,
    }


def _extract_deployable_history_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    extra_deployable_fields = sorted(
        field_name
        for field_name in row.keys()
        if isinstance(field_name, str)
        and field_name.startswith("deployable.")
        and field_name
        not in {
            "deployable.previous_action_history",
            "deployable.proprio_history",
            "deployable.short_visual_history_refs",
        }
    )
    if extra_deployable_fields:
        raise ValueError(
            "unexpected deployable field(s) leaked into state-conditioned inputs: "
            + ", ".join(extra_deployable_fields)
        )

    history_k = _as_int(row.get("history_k"), field_name="history_k")
    valid_mask = [
        bool(item)
        for item in _as_list(
            row.get("history_valid_mask"),
            field_name="history_valid_mask",
            expected_len=history_k,
        )
    ]
    previous_action_history = _as_list(
        row.get("deployable.previous_action_history"),
        field_name="deployable.previous_action_history",
        expected_len=history_k,
    )
    proprio_history = _as_list(
        row.get("deployable.proprio_history"),
        field_name="deployable.proprio_history",
        expected_len=history_k,
    )
    short_visual_history_refs = _as_list(
        row.get("deployable.short_visual_history_refs"),
        field_name="deployable.short_visual_history_refs",
        expected_len=history_k,
    )
    for index, is_valid in enumerate(valid_mask):
        if not is_valid:
            continue
        if previous_action_history[index] is None:
            raise ValueError(
                f"deployable.previous_action_history[{index}] missing for valid history slot"
            )
        if proprio_history[index] is None:
            raise ValueError(
                f"deployable.proprio_history[{index}] missing for valid history slot"
            )
        _as_non_empty_string(
            short_visual_history_refs[index],
            field_name=f"deployable.short_visual_history_refs[{index}]",
        )
    return {
        "deployable.previous_action_history": previous_action_history,
        "deployable.proprio_history": proprio_history,
        "deployable.short_visual_history_refs": short_visual_history_refs,
    }


def _normalize_row_history_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    history = (
        state_conditioned_bucket_a_import.validate_state_conditioned_history_contract(
            anchor_episode_id=row.get("anchor_episode_id", row.get("episode_id")),
            history_episode_ids=row.get("history_episode_ids"),
            history_valid_mask=row.get("history_valid_mask"),
            anchor_mujoco_state_ref=row.get("anchor_mujoco_state_ref"),
            prehistory_window=row.get("prehistory_window"),
            history_k=row.get("history_k"),
            history_stride=row.get("history_stride"),
            reset_boundary=row.get("reset_boundary"),
        )
    )
    history_k = _as_int(history["history_k"], field_name="history.history_k")
    history_stride = _as_int(
        history["history_stride"],
        field_name="history.history_stride",
    )
    history_valid_mask = [
        bool(item)
        for item in _as_list(
            history["history_valid_mask"],
            field_name="history.history_valid_mask",
            expected_len=history_k,
        )
    ]
    prehistory_window = [
        dict(_as_mapping(item, field_name=f"prehistory_window[{index}]"))
        for index, item in enumerate(
            _as_list(
                history["prehistory_window"],
                field_name="prehistory_window",
                expected_len=history_k,
            )
        )
    ]
    history_t_std_indices, history_t_raw_indices, history_timestamp_s = (
        _history_indices_from_window(
            prehistory_window=prehistory_window,
            history_valid_mask=history_valid_mask,
        )
    )
    existing_timestamps = row.get("history_timestamp_s")
    if existing_timestamps is not None:
        history_timestamp_s = [
            None if item is None else float(item)
            for item in _as_list(
                existing_timestamps,
                field_name="history_timestamp_s",
                expected_len=history_k,
            )
        ]
    reset_boundary = _as_non_empty_string(
        history.get("reset_boundary"),
        field_name="history.reset_boundary",
    )
    return {
        "history_k": history_k,
        "history_stride": history_stride,
        "reset_boundary": reset_boundary,
        "history_valid_mask": history_valid_mask,
        "history_t_std_indices": history_t_std_indices,
        "history_t_raw_indices": history_t_raw_indices,
        "history_timestamp_s": history_timestamp_s,
    }


def _normalize_base_row(
    *,
    source_bucket: str,
    source_kind: str,
    source_episode_id: str,
    source_t: int | None,
    source_snapshot_id: str | None,
    source_sample_key: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_source_bucket = _normalize_source_bucket(source_bucket)
    history_payload = _normalize_row_history_contract(row)
    deployable_history = _extract_deployable_history_fields(row)
    canonical_phase, canonical_mode, canonical_text = (
        state_conditioned_bucket_a_import.validate_state_conditioned_policy_condition(
            phase=row.get("policy_condition.phase"),
            mode=row.get("policy_condition.mode"),
            policy_condition_text=row.get("policy_condition_text"),
        )
    )
    return {
        "source_bucket": normalized_source_bucket,
        "source_kind": str(source_kind),
        "source_episode_id": str(source_episode_id),
        "source_t": None if source_t is None else int(source_t),
        "source_snapshot_id": source_snapshot_id,
        "source_sample_key": str(source_sample_key),
        "budget_group": "shared_state_conditioned_equal_data_training_budget_v1",
        "canonical_phase": canonical_phase,
        "canonical_mode": canonical_mode,
        "canonical_policy_condition_text": canonical_text,
        **history_payload,
        **deployable_history,
    }


def _load_bucket_a_base_rows(prerequisites: Mapping[str, Any]) -> list[dict[str, Any]]:
    accepted_episode_ids = set(prerequisites["accepted_bucket_a_episode_ids"])
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for raw_row in _read_jsonl_dicts(Path(prerequisites["bucket_sidecar_path"])):
        row = dict(_as_mapping(raw_row, field_name="bucket_A_sidecar[]"))
        state_conditioned_bucket_a_import.validate_sidecar_row_for_gate(row)
        episode_id = _as_non_empty_string(
            row.get("episode_id"), field_name="episode_id"
        )
        if episode_id not in accepted_episode_ids:
            continue
        t_value = _as_int(row.get("t"), field_name="t")
        source_sample_key = f"{episode_id}:{t_value}"
        if source_sample_key in seen_keys:
            raise ValueError(
                f"duplicate canonical Bucket A source row: {source_sample_key}"
            )
        seen_keys.add(source_sample_key)
        rows.append(
            _normalize_base_row(
                source_bucket=SOURCE_BUCKET_CANONICAL_BUCKET_A,
                source_kind="sidecar_transition",
                source_episode_id=episode_id,
                source_t=t_value,
                source_snapshot_id=None,
                source_sample_key=source_sample_key,
                row=row,
            )
        )
    if not rows:
        raise ValueError("canonical Bucket A contributed no usable sidecar rows")
    return sorted(
        rows, key=lambda item: (item["source_episode_id"], int(item["source_t"]))
    )


def _load_bucket_b_base_rows(prerequisites: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    bucket_b_manifest = dict(prerequisites["bucket_b_manifest"])
    for raw_entry in list(bucket_b_manifest.get("episodes", [])):
        entry = dict(_as_mapping(raw_entry, field_name="bucket_B_manifest.episodes[]"))
        dataset_dir = Path(
            _as_non_empty_string(
                entry.get("dataset_dir"), field_name="bucket_B.dataset_dir"
            )
        )
        episode_id = _as_non_empty_string(
            entry.get("episode_id"), field_name="episode_id"
        )
        dataset_records = state_conditioned_bucket_a_import._load_dataset_records(
            dataset_dir
        )
        sidecar_rows = list(dataset_records["sidecar_by_episode"].get(episode_id, []))
        if not sidecar_rows:
            raise ValueError(
                f"Bucket B episode {episode_id!r} is missing history-aware sidecar rows"
            )
        for raw_row in sorted(
            sidecar_rows,
            key=lambda item: _as_int(item.get("t"), field_name="bucket_B.t"),
        ):
            row = dict(_as_mapping(raw_row, field_name="bucket_B_sidecar[]"))
            state_conditioned_bucket_a_import.validate_sidecar_row_for_gate(row)
            t_value = _as_int(row.get("t"), field_name="t")
            source_sample_key = f"{episode_id}:{t_value}"
            if source_sample_key in seen_keys:
                raise ValueError(f"duplicate Bucket B source row: {source_sample_key}")
            seen_keys.add(source_sample_key)
            rows.append(
                _normalize_base_row(
                    source_bucket=SOURCE_BUCKET_BUCKET_B,
                    source_kind="sidecar_transition",
                    source_episode_id=episode_id,
                    source_t=t_value,
                    source_snapshot_id=None,
                    source_sample_key=source_sample_key,
                    row=row,
                )
            )
    if not rows:
        raise ValueError("Bucket B contributed no usable sidecar rows")
    return rows


def _load_snapshot_candidate_index(
    snapshot_candidates_path: Path,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for raw_row in _read_jsonl_dicts(snapshot_candidates_path):
        row = dict(_as_mapping(raw_row, field_name="snapshot_candidates[]"))
        snapshot_id = _as_non_empty_string(
            row.get("snapshot_id"), field_name="snapshot_id"
        )
        if snapshot_id in index:
            raise ValueError(
                f"duplicate snapshot_id in snapshot candidates: {snapshot_id}"
            )
        history_payload = _normalize_row_history_contract(row)
        deployable_history = _extract_deployable_history_fields(row)
        snapshot_family = _as_non_empty_string(row.get("family"), field_name="family")
        expected_source_bucket_key = (
            state_conditioned_snapshot_harvest.FAMILY_SOURCE_BUCKET_BY_FAMILY[
                snapshot_family
            ]
        )
        source_bucket_key = _optional_string(
            row.get("source_bucket_key"),
            field_name="source_bucket_key",
        )
        if source_bucket_key is None:
            source_bucket_key = str(expected_source_bucket_key)
        elif source_bucket_key != str(expected_source_bucket_key):
            raise ValueError(
                "snapshot candidate source_bucket_key mismatch for family "
                + f"{snapshot_family!r}: {source_bucket_key!r} != {expected_source_bucket_key!r}"
            )
        canonical_phase, canonical_mode, canonical_text = (
            state_conditioned_bucket_a_import.validate_state_conditioned_policy_condition(
                phase=row.get("policy_condition.phase"),
                mode=row.get("policy_condition.mode"),
                policy_condition_text=row.get("policy_condition_text"),
            )
        )
        index[snapshot_id] = {
            "snapshot_id": snapshot_id,
            "anchor_episode_id": _as_non_empty_string(
                row.get("anchor_episode_id"),
                field_name="anchor_episode_id",
            ),
            "anchor_t": _as_int(row.get("anchor_t"), field_name="anchor_t"),
            "source_snapshot_family": snapshot_family,
            "source_bucket_key": source_bucket_key,
            "canonical_phase": canonical_phase,
            "canonical_mode": canonical_mode,
            "canonical_policy_condition_text": canonical_text,
            **history_payload,
            **deployable_history,
        }
    if not index:
        raise ValueError("snapshot candidate context is empty")
    return index


def _load_formal_pseudodemo_base_rows(
    prerequisites: Mapping[str, Any],
) -> list[dict[str, Any]]:
    snapshot_candidates = _load_snapshot_candidate_index(
        Path(prerequisites["snapshot_candidates_path"])
    )
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for raw_record in list(prerequisites["pseudodemo_manifest"].get("pseudodemos", [])):
        record = dict(
            _as_mapping(raw_record, field_name="pseudodemo_manifest.pseudodemos[]")
        )
        episode_id = _as_non_empty_string(
            record.get("episode_id"), field_name="episode_id"
        )
        source_snapshot_id = _as_non_empty_string(
            record.get("source_snapshot_id"),
            field_name="source_snapshot_id",
        )
        if source_snapshot_id not in snapshot_candidates:
            raise ValueError(
                f"formal pseudo-demo {episode_id!r} is missing snapshot context for {source_snapshot_id!r}"
            )
        snapshot_row = dict(snapshot_candidates[source_snapshot_id])
        if _as_int(
            record.get("source_snapshot_history_k"),
            field_name="source_snapshot_history_k",
        ) != int(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K):
            raise ValueError("formal pseudo-demo history_k mismatch")
        source_sample_key = f"{episode_id}:{source_snapshot_id}"
        if source_sample_key in seen_keys:
            raise ValueError(
                f"duplicate formal pseudo-demo source row: {source_sample_key}"
            )
        seen_keys.add(source_sample_key)
        rows.append(
            {
                "source_bucket": SOURCE_BUCKET_FORMAL_PSEUDODEMO,
                "source_kind": "formal_pseudodemo",
                "source_episode_id": episode_id,
                "source_t": int(snapshot_row["anchor_t"]),
                "source_snapshot_id": source_snapshot_id,
                "source_sample_key": source_sample_key,
                "source_anchor_episode_id": str(snapshot_row["anchor_episode_id"]),
                "budget_group": "shared_state_conditioned_equal_data_training_budget_v1",
                "canonical_phase": str(snapshot_row["canonical_phase"]),
                "canonical_mode": str(snapshot_row["canonical_mode"]),
                "canonical_policy_condition_text": str(
                    snapshot_row["canonical_policy_condition_text"]
                ),
                "history_k": int(snapshot_row["history_k"]),
                "history_stride": int(snapshot_row["history_stride"]),
                "reset_boundary": str(snapshot_row["reset_boundary"]),
                "history_valid_mask": list(snapshot_row["history_valid_mask"]),
                "history_t_std_indices": list(snapshot_row["history_t_std_indices"]),
                "history_t_raw_indices": list(snapshot_row["history_t_raw_indices"]),
                "history_timestamp_s": list(snapshot_row["history_timestamp_s"]),
                "deployable.previous_action_history": list(
                    snapshot_row["deployable.previous_action_history"]
                ),
                "deployable.proprio_history": list(
                    snapshot_row["deployable.proprio_history"]
                ),
                "deployable.short_visual_history_refs": list(
                    snapshot_row["deployable.short_visual_history_refs"]
                ),
            }
        )
    if not rows:
        raise ValueError("formal pseudo-demo manifest contributed no usable rows")
    return rows


def _lookup_m2_fields(
    dataset_records: Mapping[str, Any],
    *,
    episode_id: str,
    source_t: int,
) -> dict[str, Any]:
    labels = list(dataset_records.get("labels_by_episode", {}).get(episode_id, []))
    if not labels:
        raise ValueError(f"missing source dataset M2 labels for episode {episode_id!r}")
    matched: dict[str, Any] | None = None
    for raw_label in labels:
        label = dict(_as_mapping(raw_label, field_name=f"m2_labels[{episode_id}]"))
        label_t = _as_int(label.get("t"), field_name=f"m2_labels[{episode_id}].t")
        if label_t != int(source_t):
            continue
        if matched is not None:
            raise ValueError(
                f"duplicate M2 label rows for episode {episode_id!r} t={source_t}"
            )
        matched = label
    if matched is None:
        raise ValueError(
            f"missing M2 label row for episode {episode_id!r} t={source_t}"
        )
    indicator = _as_int(
        matched.get("indicator_I"),
        field_name=f"m2_labels[{episode_id}].indicator_I",
    )
    if indicator not in (0, 1):
        raise ValueError(
            f"m2_labels[{episode_id}].indicator_I must be 0 or 1, got {indicator!r}"
        )
    return {
        "return_G": _as_float(
            matched.get("return_G"),
            field_name=f"m2_labels[{episode_id}].return_G",
        ),
        "value_V": _as_float(
            matched.get("value_V"),
            field_name=f"m2_labels[{episode_id}].value_V",
        ),
        "advantage_A": _as_float(
            matched.get("advantage_A"),
            field_name=f"m2_labels[{episode_id}].advantage_A",
        ),
        "epsilon_l": _as_float(
            matched.get("epsilon_l"),
            field_name=f"m2_labels[{episode_id}].epsilon_l",
        ),
        "indicator_I": int(indicator),
    }


def _derived_teacher_target_truthfulness(record: Mapping[str, Any]) -> str | None:
    explicit = _optional_string(
        record.get("teacher_target_truthfulness"),
        field_name="teacher_target_truthfulness",
    )
    if explicit is not None:
        return explicit
    teacher_target = record.get("teacher_target")
    if isinstance(teacher_target, Mapping):
        return state_conditioned_snapshot_harvest.TEACHER_TARGET_TRUTHFUL_REAL_ROLLOUT
    producer = _optional_string(record.get("producer"), field_name="producer")
    if producer == state_conditioned_snapshot_harvest.PRODUCER_BASE_POLICY:
        return state_conditioned_snapshot_harvest.TEACHER_TARGET_NOT_APPLICABLE
    return None


def _attach_label_space_backfill(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    prerequisites: Mapping[str, Any],
) -> list[dict[str, Any]]:
    episode_dataset_index = _build_source_episode_dataset_index(prerequisites)
    snapshot_index, formal_pseudodemos = _build_formal_pseudodemo_snapshot_index(
        prerequisites
    )
    dataset_cache: dict[str, dict[str, Any]] = {}
    enriched_rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        row = dict(raw_row)
        source_bucket = _normalize_source_bucket(row.get("source_bucket"))
        source_episode_id = _as_non_empty_string(
            row.get("source_episode_id"), field_name="source_episode_id"
        )
        source_t = _as_int(row.get("source_t"), field_name="source_t")
        source_anchor_episode_id = source_episode_id
        pseudodemo_source_snapshot_family: str | None = None
        pseudodemo_source_bucket_key: str | None = None
        pseudodemo_teacher_policy_id: str | None = None
        pseudodemo_teacher_target: dict[str, Any] | None = None
        pseudodemo_teacher_target_truthfulness: str | None = None
        pseudodemo_label_kind: str | None = None
        pseudodemo_dataset_version: str | None = None

        if source_bucket == SOURCE_BUCKET_FORMAL_PSEUDODEMO:
            source_snapshot_id = _as_non_empty_string(
                row.get("source_snapshot_id"),
                field_name="source_snapshot_id",
            )
            snapshot_row = dict(
                _as_mapping(
                    snapshot_index.get(source_snapshot_id),
                    field_name=f"snapshot_index[{source_snapshot_id}]",
                )
            )
            formal_record = dict(
                _as_mapping(
                    formal_pseudodemos.get(source_episode_id),
                    field_name=f"formal_pseudodemos[{source_episode_id}]",
                )
            )
            source_anchor_episode_id = _as_non_empty_string(
                row.get(
                    "source_anchor_episode_id", snapshot_row.get("anchor_episode_id")
                ),
                field_name="source_anchor_episode_id",
            )
            pseudodemo_source_snapshot_family = _optional_string(
                formal_record.get("source_snapshot_family"),
                field_name="source_snapshot_family",
            ) or _as_non_empty_string(
                snapshot_row.get("source_snapshot_family"),
                field_name="snapshot_row.source_snapshot_family",
            )
            pseudodemo_source_bucket_key = _optional_string(
                formal_record.get("source_bucket_key"),
                field_name="source_bucket_key",
            ) or _as_non_empty_string(
                snapshot_row.get("source_bucket_key"),
                field_name="snapshot_row.source_bucket_key",
            )
            pseudodemo_teacher_policy_id = _optional_string(
                formal_record.get("teacher_version"),
                field_name="teacher_version",
            )
            teacher_target = formal_record.get("teacher_target")
            if teacher_target is not None:
                pseudodemo_teacher_target = dict(
                    _as_mapping(teacher_target, field_name="teacher_target")
                )
            pseudodemo_teacher_target_truthfulness = (
                _derived_teacher_target_truthfulness(formal_record)
            )
            pseudodemo_label_kind = "formal_pseudodemo"
            pseudodemo_dataset_version = PSEUDODEMO_DATASET_VERSION

        dataset_dir = episode_dataset_index.get(source_anchor_episode_id)
        if dataset_dir is None:
            raise ValueError(
                "missing source dataset root for label backfill episode: "
                + source_anchor_episode_id
            )
        dataset_records = _load_source_dataset_cache(dataset_dir, cache=dataset_cache)
        episode_record = dict(
            _as_mapping(
                dataset_records["episodes_by_id"].get(source_anchor_episode_id),
                field_name=f"episodes_by_id[{source_anchor_episode_id}]",
            )
        )
        m2_fields = _lookup_m2_fields(
            dataset_records,
            episode_id=source_anchor_episode_id,
            source_t=source_t,
        )
        carrier_text_v1 = build_carrier_text_v1(
            prompt_raw=episode_record.get("prompt_raw"),
            indicator_value=m2_fields["indicator_I"],
        )
        enriched_rows.append(
            {
                **row,
                "source_anchor_episode_id": source_anchor_episode_id,
                "label_data.domain": LABEL_DATA_DOMAIN,
                "label_data.version": LABEL_DATA_VERSION,
                "label_data.m2_backfill_source": M2_BACKFILL_SOURCE,
                "label_data.m2_backfill_version": M2_BACKFILL_VERSION,
                "pseudodemo.source_snapshot_family": pseudodemo_source_snapshot_family,
                "pseudodemo.source_bucket_key": pseudodemo_source_bucket_key,
                "pseudodemo.teacher_policy_id": pseudodemo_teacher_policy_id,
                "pseudodemo.teacher_target": pseudodemo_teacher_target,
                "pseudodemo.teacher_target_truthfulness": pseudodemo_teacher_target_truthfulness,
                "pseudodemo.label_kind": pseudodemo_label_kind,
                "pseudodemo.dataset_version": pseudodemo_dataset_version,
                **m2_fields,
                MAINLINE_TRAINING_TEXT_FIELD: carrier_text_v1,
            }
        )
    return enriched_rows


def _expand_unified_base_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    recovery_oversample_factor: int,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        row = dict(raw_row)
        repeat_count = (
            int(recovery_oversample_factor)
            if str(row["canonical_mode"]) == "RECOVERY"
            else 1
        )
        for repeat_index in range(repeat_count):
            sample_id = f"{row['source_bucket']}::{row['source_sample_key']}::repeat{repeat_index}"
            expanded.append(
                {
                    **row,
                    "sample_id": sample_id,
                    "repeat_index": int(repeat_index),
                    "recovery_oversample_factor": int(recovery_oversample_factor),
                }
            )
    if not expanded:
        raise ValueError("unified base table is empty")
    return expanded


def _build_view_rows(
    base_rows: Sequence[Mapping[str, Any]],
    *,
    training_view: str,
) -> list[dict[str, Any]]:
    normalized_view = _as_non_empty_string(training_view, field_name="training_view")
    if normalized_view not in VIEW_VALUES:
        raise ValueError(f"training_view must be one of {VIEW_VALUES!r}")
    rows: list[dict[str, Any]] = []
    for sample_index, raw_row in enumerate(base_rows):
        row = dict(raw_row)
        if normalized_view == VIEW_C0:
            phase = NULL_PHASE_TOKEN
            mode = NULL_MODE_TOKEN
            policy_condition_text = build_null_policy_condition_text()
        else:
            phase = str(row["canonical_phase"])
            mode = str(row["canonical_mode"])
            policy_condition_text = str(row["canonical_policy_condition_text"])
        validate_view_policy_condition_text(
            training_view=normalized_view,
            phase=phase,
            mode=mode,
            policy_condition_text=policy_condition_text,
        )
        label_row = {
            "schema_version": SCHEMA_VERSION,
            "training_view": normalized_view,
            "sample_id": str(row["sample_id"]),
            "sample_index": int(sample_index),
            "source_bucket": str(row["source_bucket"]),
            "source_kind": str(row["source_kind"]),
            "source_episode_id": str(row["source_episode_id"]),
            "source_t": row["source_t"],
            "source_snapshot_id": row["source_snapshot_id"],
            "pseudodemo.source_snapshot_family": row[
                "pseudodemo.source_snapshot_family"
            ],
            "pseudodemo.source_bucket_key": row["pseudodemo.source_bucket_key"],
            "source_anchor_episode_id": str(row["source_anchor_episode_id"]),
            "source_sample_key": str(row["source_sample_key"]),
            "reset_boundary": str(row["reset_boundary"]),
            "label_data.domain": str(row["label_data.domain"]),
            "label_data.version": str(row["label_data.version"]),
            "label_data.m2_backfill_source": str(row["label_data.m2_backfill_source"]),
            "label_data.m2_backfill_version": str(
                row["label_data.m2_backfill_version"]
            ),
            "pseudodemo.teacher_policy_id": row["pseudodemo.teacher_policy_id"],
            "pseudodemo.teacher_target": row["pseudodemo.teacher_target"],
            "pseudodemo.teacher_target_truthfulness": row[
                "pseudodemo.teacher_target_truthfulness"
            ],
            "pseudodemo.label_kind": row["pseudodemo.label_kind"],
            "pseudodemo.dataset_version": row["pseudodemo.dataset_version"],
            "budget_group": str(row["budget_group"]),
            "repeat_index": int(row["repeat_index"]),
            "recovery_oversample_factor": int(row["recovery_oversample_factor"]),
            "return_G": float(row["return_G"]),
            "value_V": float(row["value_V"]),
            "advantage_A": float(row["advantage_A"]),
            "epsilon_l": float(row["epsilon_l"]),
            "indicator_I": int(row["indicator_I"]),
            MAINLINE_TRAINING_TEXT_FIELD: str(row[MAINLINE_TRAINING_TEXT_FIELD]),
            "history_k": int(row["history_k"]),
            "history_stride": int(row["history_stride"]),
            "history_valid_mask": list(row["history_valid_mask"]),
            "history_t_std_indices": list(row["history_t_std_indices"]),
            "history_t_raw_indices": list(row["history_t_raw_indices"]),
            "history_timestamp_s": list(row["history_timestamp_s"]),
            "deployable.previous_action_history": list(
                row["deployable.previous_action_history"]
            ),
            "deployable.proprio_history": list(row["deployable.proprio_history"]),
            "deployable.short_visual_history_refs": list(
                row["deployable.short_visual_history_refs"]
            ),
            "policy_condition.phase": phase,
            "policy_condition.mode": mode,
            "policy_condition_text": policy_condition_text,
        }
        ordered_row = {
            field_name: label_row[field_name] for field_name in SAFE_LABEL_FIELD_ORDER
        }
        rows.append(ordered_row)
    return rows


def _non_conditioning_signature(row: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in row.items()
        if key not in {*CONDITIONING_FIELD_NAMES, "training_view"}
    }
    return _safe_json_signature(payload)


def _history_signature(row: Mapping[str, Any]) -> str:
    payload = {
        "reset_boundary": row["reset_boundary"],
        "history_k": row["history_k"],
        "history_stride": row["history_stride"],
        "history_valid_mask": row["history_valid_mask"],
        "history_t_std_indices": row["history_t_std_indices"],
        "history_t_raw_indices": row["history_t_raw_indices"],
        "history_timestamp_s": row["history_timestamp_s"],
        "deployable.previous_action_history": row["deployable.previous_action_history"],
        "deployable.proprio_history": row["deployable.proprio_history"],
        "deployable.short_visual_history_refs": row[
            "deployable.short_visual_history_refs"
        ],
    }
    return _safe_json_signature(payload)


def build_equal_data_fairness_audit(
    *,
    base_rows: Sequence[Mapping[str, Any]],
    c0_rows: Sequence[Mapping[str, Any]],
    c1_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base_sample_ids = [str(row["sample_id"]) for row in base_rows]
    c0_sample_ids = [str(row["sample_id"]) for row in c0_rows]
    c1_sample_ids = [str(row["sample_id"]) for row in c1_rows]
    c0_source_buckets = [str(row["source_bucket"]) for row in c0_rows]
    c1_source_buckets = [str(row["source_bucket"]) for row in c1_rows]
    c0_budget_groups = [str(row["budget_group"]) for row in c0_rows]
    c1_budget_groups = [str(row["budget_group"]) for row in c1_rows]
    c0_history = [_history_signature(row) for row in c0_rows]
    c1_history = [_history_signature(row) for row in c1_rows]
    baseline_to_c0 = {
        "status": "PASS",
        "shared_base_row_count": int(len(base_rows)),
        "same_unified_base_sample_ids": base_sample_ids == c0_sample_ids,
        "baseline_history_channel_enabled": False,
        "c0_history_channel_enabled": True,
        "baseline_deployable_observation_allowlist": [],
        "c0_deployable_observation_allowlist": list(
            lerobot_v2_export.DEPLOYABLE_HISTORY_FIELD_NAMES
        ),
        "same_budget_policy": True,
        "same_sampling_policy": True,
        "focus": "baseline_to_c0_short_history_memory",
    }
    c0_to_c1 = {
        "status": "PASS",
        "same_sample_ids": c0_sample_ids == c1_sample_ids,
        "same_order": c0_sample_ids == c1_sample_ids,
        "same_source_bucket_sequence": c0_source_buckets == c1_source_buckets,
        "same_budget_sequence": c0_budget_groups == c1_budget_groups,
        "same_history_signatures": c0_history == c1_history,
        "same_deployable_observation_allowlist": True,
        "conditioning_only_delta_fields": list(CONDITIONING_FIELD_NAMES),
        "focus": "c0_to_c1_phase_mode_conditioning",
    }
    overall_pass = all(
        [
            bool(baseline_to_c0["same_unified_base_sample_ids"]),
            bool(c0_to_c1["same_sample_ids"]),
            bool(c0_to_c1["same_order"]),
            bool(c0_to_c1["same_source_bucket_sequence"]),
            bool(c0_to_c1["same_budget_sequence"]),
            bool(c0_to_c1["same_history_signatures"]),
            bool(c0_to_c1["same_deployable_observation_allowlist"]),
        ]
    )
    if not overall_pass:
        baseline_to_c0["status"] = "FAIL"
        c0_to_c1["status"] = "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "equal_data_fairness_audit",
        "overall_pass": bool(overall_pass),
        "comparisons": {
            "baseline_to_c0": baseline_to_c0,
            "c0_to_c1": c0_to_c1,
        },
    }


def build_conditioning_channel_liveness(
    *,
    c0_rows: Sequence[Mapping[str, Any]],
    c1_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    c0_non_conditioning = [_non_conditioning_signature(row) for row in c0_rows]
    c1_non_conditioning = [_non_conditioning_signature(row) for row in c1_rows]
    c0_phase_values = [str(row["policy_condition.phase"]) for row in c0_rows]
    c0_mode_values = [str(row["policy_condition.mode"]) for row in c0_rows]
    c1_phase_values = [str(row["policy_condition.phase"]) for row in c1_rows]
    c1_mode_values = [str(row["policy_condition.mode"]) for row in c1_rows]
    c1_texts = [str(row["policy_condition_text"]) for row in c1_rows]
    c0_texts = [str(row["policy_condition_text"]) for row in c0_rows]
    same_non_conditioning = c0_non_conditioning == c1_non_conditioning
    c0_null_phase = all(value == NULL_PHASE_TOKEN for value in c0_phase_values)
    c0_null_mode = all(value == NULL_MODE_TOKEN for value in c0_mode_values)
    c1_non_null_phase = all(
        value in state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES
        for value in c1_phase_values
    )
    c1_non_null_mode = all(
        value in state_conditioned_bucket_a_import.STATE_CONDITIONED_MODES
        for value in c1_mode_values
    )
    c1_text_valid = True
    for row in c1_rows:
        try:
            validate_view_policy_condition_text(
                training_view=VIEW_C1,
                phase=row["policy_condition.phase"],
                mode=row["policy_condition.mode"],
                policy_condition_text=row["policy_condition_text"],
            )
        except (TypeError, ValueError):
            c1_text_valid = False
            break
    c0_text_valid = True
    for row in c0_rows:
        try:
            validate_view_policy_condition_text(
                training_view=VIEW_C0,
                phase=row["policy_condition.phase"],
                mode=row["policy_condition.mode"],
                policy_condition_text=row["policy_condition_text"],
            )
        except (TypeError, ValueError):
            c0_text_valid = False
            break
    conditioning_changes = sum(
        1 for c0_text, c1_text in zip(c0_texts, c1_texts) if c0_text != c1_text
    )
    overall_pass = all(
        [
            same_non_conditioning,
            c0_null_phase,
            c0_null_mode,
            c1_non_null_phase,
            c1_non_null_mode,
            c0_text_valid,
            c1_text_valid,
            conditioning_changes == len(c0_rows),
        ]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "conditioning_channel_liveness",
        "overall_pass": bool(overall_pass),
        "same_non_conditioning_payload": bool(same_non_conditioning),
        "differing_only_fields": list(CONDITIONING_FIELD_NAMES),
        "counts": {
            "row_count": int(len(c0_rows)),
            "conditioning_text_change_count": int(conditioning_changes),
            "c0_null_phase_count": int(
                sum(value == NULL_PHASE_TOKEN for value in c0_phase_values)
            ),
            "c0_null_mode_count": int(
                sum(value == NULL_MODE_TOKEN for value in c0_mode_values)
            ),
            "c1_non_null_phase_count": int(len(c1_phase_values)),
            "c1_non_null_mode_count": int(len(c1_mode_values)),
        },
        "c1_distinct_phase_values": sorted(set(c1_phase_values)),
        "c1_distinct_mode_values": sorted(set(c1_mode_values)),
        "baseline_to_c0_focus": "short_history_memory_channel",
        "c0_to_c1_focus": "phase_mode_conditioning_channel",
    }


def build_boundary_summary(
    *,
    label_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    forbidden_output_fields = sorted(
        {
            str(field_name)
            for row in label_rows
            for field_name in row.keys()
            if _forbidden_output_field_name(str(field_name))
        }
    )
    polluted_policy_condition_rows = 0
    for row in label_rows:
        policy_text = str(row.get("policy_condition_text", ""))
        if any(
            token in policy_text
            for token in (
                "privileged",
                "oracle",
                "teacher",
                "event",
                "recovery_needed",
                "semantic_state",
                "memory_commit",
                "summary_template",
            )
        ):
            polluted_policy_condition_rows += 1

    analysis_only_observed_fields = sorted(
        {
            str(field_name)
            for row in raw_rows
            for field_name in row.keys()
            if _forbidden_output_field_name(str(field_name))
        }
    )
    overall_pass = not forbidden_output_fields and polluted_policy_condition_rows == 0
    return {
        "pass": bool(overall_pass),
        "mainline_training_text_field": MAINLINE_TRAINING_TEXT_FIELD,
        "deployable_observation_allowlist": list(
            lerobot_v2_export.DEPLOYABLE_HISTORY_FIELD_NAMES
        ),
        "forbidden_output_fields": forbidden_output_fields,
        "policy_condition_text_pollution_row_count": int(
            polluted_policy_condition_rows
        ),
        "analysis_only_observed_fields": analysis_only_observed_fields,
    }


def build_dev_only_promotion_gate(
    *,
    fairness_audit: Mapping[str, Any],
    boundary_summary: Mapping[str, Any],
    liveness_audit: Mapping[str, Any],
    recovery_oversample_factor: int,
    teacher_assisted_sources_present: bool,
    legacy_debug_only_reuse_present: bool,
) -> dict[str, Any]:
    failure_reasons: list[str] = []
    if not bool(fairness_audit.get("overall_pass", False)):
        failure_reasons.append("fairness_gate_failed")
    if not bool(boundary_summary.get("pass", False)):
        failure_reasons.append("deployable_boundary_gate_failed")
    if not bool(liveness_audit.get("overall_pass", False)):
        failure_reasons.append("conditioning_liveness_gate_failed")
    if int(recovery_oversample_factor) != int(RECOVERY_OVERSAMPLE_FACTOR):
        failure_reasons.append("recovery_oversample_factor_must_equal_3")
    if bool(teacher_assisted_sources_present):
        failure_reasons.append("teacher_assisted_formal_pseudodemos_remain_dev_only")
    if bool(legacy_debug_only_reuse_present):
        failure_reasons.append("timeout_legacy_import_evidence_remains_dev_only")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "dev_only_promotion_gate",
        "promotion_allowed": len(failure_reasons) == 0,
        "checks": {
            "fairness_pass": bool(fairness_audit.get("overall_pass", False)),
            "boundary_pass": bool(boundary_summary.get("pass", False)),
            "conditioning_liveness_pass": bool(
                liveness_audit.get("overall_pass", False)
            ),
            "recovery_oversample_factor_pass": int(recovery_oversample_factor)
            == int(RECOVERY_OVERSAMPLE_FACTOR),
            "teacher_assisted_sources_present": bool(teacher_assisted_sources_present),
            "legacy_debug_only_reuse_present": bool(legacy_debug_only_reuse_present),
        },
        "failure_reasons": failure_reasons,
    }


def _teacher_assisted_sources_present(pseudodemo_manifest: Mapping[str, Any]) -> bool:
    producer_by_family = dict(
        _as_mapping(
            pseudodemo_manifest.get("producer_by_family", {}),
            field_name="pseudodemo_manifest.producer_by_family",
        )
    )
    if any(
        str(value) == state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER
        for value in producer_by_family.values()
    ):
        return True
    for raw_record in list(pseudodemo_manifest.get("pseudodemos", [])):
        record = dict(
            _as_mapping(raw_record, field_name="pseudodemo_manifest.pseudodemos[]")
        )
        if (
            str(record.get("producer"))
            == state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER
        ):
            return True
    return False


def _legacy_debug_only_reuse_present(
    debug_only_reuse_manifest_path: Path | None,
) -> bool:
    if (
        debug_only_reuse_manifest_path is None
        or not debug_only_reuse_manifest_path.is_file()
    ):
        return False
    payload = _read_json(debug_only_reuse_manifest_path)
    if bool(payload.get("reused_existing_live_dataset", False)):
        return True
    return int(payload.get("selected_episode_count", 0) or 0) > 0


def _build_stats(
    *,
    base_rows: Sequence[Mapping[str, Any]],
    c0_rows: Sequence[Mapping[str, Any]],
    c1_rows: Sequence[Mapping[str, Any]],
    lerobot_training_dataset: Mapping[str, Any],
) -> dict[str, Any]:
    raw_counts_by_source = Counter(str(row["source_bucket"]) for row in base_rows)
    view_counts_by_source = Counter(str(row["source_bucket"]) for row in c1_rows)
    canonical_mode_counts = Counter(str(row["canonical_mode"]) for row in base_rows)
    repeat_counts = Counter(int(row["repeat_index"]) for row in base_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_sft_stats",
        "mainline_training_text_field": MAINLINE_TRAINING_TEXT_FIELD,
        "recovery_oversample_factor_min": int(
            min(int(row["recovery_oversample_factor"]) for row in base_rows)
        ),
        "recovery_oversample_factor_max": int(
            max(int(row["recovery_oversample_factor"]) for row in base_rows)
        ),
        "deployable_observation_allowlist": list(
            lerobot_v2_export.DEPLOYABLE_HISTORY_FIELD_NAMES
        ),
        "lerobot_dataset_path": str(lerobot_training_dataset["dataset_root"]),
        "counts": {
            "unified_base_row_count": int(len(base_rows)),
            "c0_row_count": int(len(c0_rows)),
            "c1_row_count": int(len(c1_rows)),
            "lerobot_episode_count": int(lerobot_training_dataset["episode_count"]),
            "lerobot_frame_count": int(lerobot_training_dataset["frame_count"]),
            "rows_by_source_bucket": {
                key: int(value) for key, value in sorted(raw_counts_by_source.items())
            },
            "rows_by_source_bucket_per_view": {
                key: int(value) for key, value in sorted(view_counts_by_source.items())
            },
            "rows_by_canonical_mode": {
                key: int(value) for key, value in sorted(canonical_mode_counts.items())
            },
            "repeat_index_counts": {
                str(key): int(value) for key, value in sorted(repeat_counts.items())
            },
        },
        "views": {
            VIEW_C0: {
                "sample_ids_hash": _safe_json_signature(
                    [str(row["sample_id"]) for row in c0_rows]
                ),
                "null_phase_token": NULL_PHASE_TOKEN,
                "null_mode_token": NULL_MODE_TOKEN,
            },
            VIEW_C1: {
                "sample_ids_hash": _safe_json_signature(
                    [str(row["sample_id"]) for row in c1_rows]
                ),
                "distinct_phase_values": sorted(
                    {str(row["policy_condition.phase"]) for row in c1_rows}
                ),
                "distinct_mode_values": sorted(
                    {str(row["policy_condition.mode"]) for row in c1_rows}
                ),
            },
        },
    }


def _build_training_rows(
    *, prerequisites: Mapping[str, Any], recovery_oversample_factor: int
) -> dict[str, list[dict[str, Any]]]:
    bucket_a_rows = _load_bucket_a_base_rows(prerequisites)
    bucket_b_rows = _load_bucket_b_base_rows(prerequisites)
    pseudodemo_rows = _load_formal_pseudodemo_base_rows(prerequisites)
    raw_rows = _attach_label_space_backfill(
        [*bucket_a_rows, *bucket_b_rows, *pseudodemo_rows],
        prerequisites=prerequisites,
    )
    base_rows = _expand_unified_base_rows(
        raw_rows,
        recovery_oversample_factor=int(recovery_oversample_factor),
    )
    return {
        "raw_rows": raw_rows,
        "base_rows": base_rows,
        "c0_rows": _build_view_rows(base_rows, training_view=VIEW_C0),
        "c1_rows": _build_view_rows(base_rows, training_view=VIEW_C1),
    }


def _persist_training_set_artifacts(
    *,
    output_dir: Path,
    c0_rows: Sequence[Mapping[str, object]],
    c1_rows: Sequence[Mapping[str, object]],
    stats: Mapping[str, object],
    fairness_audit: Mapping[str, object],
    liveness_audit: Mapping[str, object],
    dev_only_gate: Mapping[str, object],
) -> dict[str, Path]:
    artifact_paths = {
        "labels": output_dir / STATE_CONDITIONED_SFT_LABELS_JSONL_NAME,
        "stats": output_dir / STATE_CONDITIONED_SFT_STATS_JSON_NAME,
        "fairness": output_dir / EQUAL_DATA_FAIRNESS_AUDIT_JSON_NAME,
        "liveness": output_dir / CONDITIONING_CHANNEL_LIVENESS_JSON_NAME,
        "gate": output_dir / DEV_ONLY_PROMOTION_GATE_JSON_NAME,
    }
    _write_jsonl(artifact_paths["labels"], [*c0_rows, *c1_rows])
    _write_json(artifact_paths["stats"], stats)
    _write_json(artifact_paths["fairness"], fairness_audit)
    _write_json(artifact_paths["liveness"], liveness_audit)
    _write_json(artifact_paths["gate"], dev_only_gate)
    return artifact_paths


@dataclass
class TrainingSetPrerequisiteLoader:
    bucket_dir: Path
    dev_dir: Path
    collection_dir: Path
    harvest_dir: Path
    output_dir: Path

    def load(self) -> dict[str, Any]:
        return _load_upstream_artifacts(
            bucket_dir=self.bucket_dir,
            dev_dir=self.dev_dir,
            collection_dir=self.collection_dir,
            harvest_dir=self.harvest_dir,
            output_dir=self.output_dir,
        )


@dataclass
class TrainingRowBuilder:
    prerequisites: Mapping[str, Any]
    recovery_oversample_factor: int

    def build(self) -> dict[str, list[dict[str, Any]]]:
        return _build_training_rows(
            prerequisites=self.prerequisites,
            recovery_oversample_factor=self.recovery_oversample_factor,
        )


@dataclass
class TrainingArtifactWriter:
    output_dir: Path
    prerequisites: Mapping[str, Any]

    def build_audits(
        self,
        *,
        raw_rows: Sequence[Mapping[str, Any]],
        base_rows: Sequence[Mapping[str, Any]],
        c0_rows: Sequence[Mapping[str, object]],
        c1_rows: Sequence[Mapping[str, object]],
        recovery_oversample_factor: int,
        lerobot_training_dataset: Mapping[str, Any],
    ) -> dict[str, Mapping[str, object]]:
        fairness_audit = build_equal_data_fairness_audit(
            base_rows=base_rows,
            c0_rows=c0_rows,
            c1_rows=c1_rows,
        )
        liveness_audit = build_conditioning_channel_liveness(
            c0_rows=c0_rows,
            c1_rows=c1_rows,
        )
        boundary_summary = build_boundary_summary(
            label_rows=[*c0_rows, *c1_rows],
            raw_rows=raw_rows,
        )
        dev_only_gate = build_dev_only_promotion_gate(
            fairness_audit=fairness_audit,
            boundary_summary=boundary_summary,
            liveness_audit=liveness_audit,
            recovery_oversample_factor=int(recovery_oversample_factor),
            teacher_assisted_sources_present=_teacher_assisted_sources_present(
                self.prerequisites["pseudodemo_manifest"]
            ),
            legacy_debug_only_reuse_present=_legacy_debug_only_reuse_present(
                self.prerequisites["debug_only_reuse_manifest_path"]
            ),
        )
        stats = _build_stats(
            base_rows=base_rows,
            c0_rows=c0_rows,
            c1_rows=c1_rows,
            lerobot_training_dataset=lerobot_training_dataset,
        )
        return {
            "fairness_audit": fairness_audit,
            "liveness_audit": liveness_audit,
            "dev_only_gate": dev_only_gate,
            "stats": stats,
        }

    def persist(
        self,
        *,
        c0_rows: Sequence[Mapping[str, object]],
        c1_rows: Sequence[Mapping[str, object]],
        audits: Mapping[str, Mapping[str, object]],
    ) -> dict[str, Path]:
        return _persist_training_set_artifacts(
            output_dir=self.output_dir,
            c0_rows=c0_rows,
            c1_rows=c1_rows,
            stats=audits["stats"],
            fairness_audit=audits["fairness_audit"],
            liveness_audit=audits["liveness_audit"],
            dev_only_gate=audits["dev_only_gate"],
        )


@dataclass
class StateConditionedTrainingSetWorkflow:
    bucket_dir: Path
    dev_dir: Path
    collection_dir: Path
    harvest_dir: Path
    output_dir: Path
    recovery_oversample_factor: int = RECOVERY_OVERSAMPLE_FACTOR
    prerequisites: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        if int(self.recovery_oversample_factor) != int(RECOVERY_OVERSAMPLE_FACTOR):
            raise ValueError(
                f"recovery-oversample-factor is frozen at {RECOVERY_OVERSAMPLE_FACTOR}"
            )
        self.output_dir = _validate_output_dir(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prerequisites = TrainingSetPrerequisiteLoader(
            bucket_dir=self.bucket_dir,
            dev_dir=self.dev_dir,
            collection_dir=self.collection_dir,
            harvest_dir=self.harvest_dir,
            output_dir=self.output_dir,
        ).load()

    def build_training_rows(self) -> dict[str, list[dict[str, Any]]]:
        return TrainingRowBuilder(
            prerequisites=self.prerequisites,
            recovery_oversample_factor=int(self.recovery_oversample_factor),
        ).build()

    def materialize_lerobot_dataset(
        self, base_rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        return materialize_lerobot_training_dataset(
            output_dir=self.output_dir,
            base_rows=base_rows,
            prerequisites=self.prerequisites,
        )

    def execute(self) -> dict[str, Any]:
        training_rows = self.build_training_rows()
        raw_rows = training_rows["raw_rows"]
        base_rows = training_rows["base_rows"]
        c0_rows = training_rows["c0_rows"]
        c1_rows = training_rows["c1_rows"]
        lerobot_training_dataset = self.materialize_lerobot_dataset(base_rows)
        artifact_writer = TrainingArtifactWriter(
            output_dir=self.output_dir,
            prerequisites=self.prerequisites,
        )
        audits = artifact_writer.build_audits(
            raw_rows=raw_rows,
            base_rows=base_rows,
            c0_rows=c0_rows,
            c1_rows=c1_rows,
            recovery_oversample_factor=int(self.recovery_oversample_factor),
            lerobot_training_dataset=lerobot_training_dataset,
        )
        artifact_paths = artifact_writer.persist(
            c0_rows=c0_rows,
            c1_rows=c1_rows,
            audits=audits,
        )
        return {
            "state_conditioned_sft_labels_path": str(artifact_paths["labels"]),
            "state_conditioned_sft_stats_path": str(artifact_paths["stats"]),
            "equal_data_fairness_audit_path": str(artifact_paths["fairness"]),
            "conditioning_channel_liveness_path": str(artifact_paths["liveness"]),
            "dev_only_promotion_gate_path": str(artifact_paths["gate"]),
            "lerobot_dataset_path": str(lerobot_training_dataset["dataset_root"]),
            "unified_base_row_count": int(len(base_rows)),
            "c0_row_count": int(len(c0_rows)),
            "c1_row_count": int(len(c1_rows)),
            "promotion_allowed": bool(audits["dev_only_gate"]["promotion_allowed"]),
        }


def materialize_state_conditioned_training_set(
    *,
    bucket_dir: Path,
    dev_dir: Path,
    collection_dir: Path,
    harvest_dir: Path,
    output_dir: Path,
    recovery_oversample_factor: int = RECOVERY_OVERSAMPLE_FACTOR,
) -> dict[str, Any]:
    return StateConditionedTrainingSetWorkflow(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        collection_dir=collection_dir,
        harvest_dir=harvest_dir,
        output_dir=output_dir,
        recovery_oversample_factor=recovery_oversample_factor,
    ).execute()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = materialize_state_conditioned_training_set(
            bucket_dir=args.bucket_dir,
            dev_dir=args.dev_dir,
            collection_dir=args.collection_dir,
            harvest_dir=args.harvest_dir,
            output_dir=args.output_dir,
            recovery_oversample_factor=int(args.recovery_oversample_factor),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CONDITIONING_CHANNEL_LIVENESS_JSON_NAME",
    "DEV_ONLY_PROMOTION_GATE_JSON_NAME",
    "EQUAL_DATA_FAIRNESS_AUDIT_JSON_NAME",
    "NULL_MODE_TOKEN",
    "NULL_PHASE_TOKEN",
    "RECOVERY_OVERSAMPLE_FACTOR",
    "SCHEMA_VERSION",
    "SOURCE_BUCKET_BUCKET_B",
    "SOURCE_BUCKET_CANONICAL_BUCKET_A",
    "SOURCE_BUCKET_FORMAL_PSEUDODEMO",
    "STATE_CONDITIONED_SFT_LABELS_JSONL_NAME",
    "STATE_CONDITIONED_SFT_STATS_JSON_NAME",
    "VIEW_C0",
    "VIEW_C1",
    "build_boundary_summary",
    "build_conditioning_channel_liveness",
    "build_dev_only_promotion_gate",
    "build_equal_data_fairness_audit",
    "build_null_policy_condition_text",
    "build_parser",
    "main",
    "materialize_state_conditioned_training_set",
    "validate_view_policy_condition_text",
]


if __name__ == "__main__":
    raise SystemExit(main())


class StateConditionedBuildTrainingSetScriptApp:
    def build_parser(self):
        return build_parser()

    def materialize_lerobot_training_dataset(self, **kwargs):
        return materialize_lerobot_training_dataset(**kwargs)

    def materialize_training_set(self, **kwargs):
        return StateConditionedTrainingSetWorkflow(**kwargs).execute()

    def run(self, argv=None) -> int:
        return main(argv)
