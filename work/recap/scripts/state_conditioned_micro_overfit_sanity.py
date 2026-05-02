#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_LABELS_PATH = Path(
    "agent/artifacts/state_conditioned_materialization/training/state_conditioned_sft_labels.jsonl"
)
DEFAULT_OUTPUT_PATH = Path(
    "agent/artifacts/state_conditioned_materialization/sanity/micro_overfit_report.json"
)
DEFAULT_TRAINING_VIEW = "C1"
DEFAULT_SUBSET_SIZE = 64
DEFAULT_MAX_STEPS = 240
DEFAULT_BATCH_SIZE = 16
DEFAULT_HIDDEN_DIM = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
LOSS_DECREASE_RATIO_THRESHOLD = 0.80
LOSS_DECREASE_ABS_THRESHOLD = 1e-4
SCHEMA_VERSION = "g1_state_conditioned_micro_overfit_sanity_v1"
REPORT_ARTIFACT_KIND = "state_conditioned_micro_overfit_report"
SUPPORTED_VIEWS: tuple[str, ...] = ("C0", "C1")


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class TrainingContext:
    labels_path: Path
    stats_path: Path
    lerobot_root: Path
    info_path: Path
    episodes_meta_path: Path
    labels: list[dict[str, Any]]
    stats: dict[str, Any]
    info: dict[str, Any]
    episodes_meta: list[dict[str, Any]]
    episodes_by_sample_id: dict[str, dict[str, Any]]
    max_episode_length: int
    action_dim: int
    chunks_size: int
    data_path_template: str


@dataclass(frozen=True)
class FeatureSpec:
    history_k: int
    previous_action_dim: int
    proprio_dim: int
    training_view_values: tuple[str, ...]
    phase_values: tuple[str, ...]
    mode_values: tuple[str, ...]


@dataclass(frozen=True)
class EncodedDataset:
    features: Any
    targets: Any
    target_mask: Any
    row_refs: list[dict[str, Any]]
    feature_dim: int
    target_dim: int
    max_episode_length: int
    action_dim: int


@dataclass(frozen=True)
class ProxyFitResult:
    model: Any
    x_mean: Any
    x_std: Any
    y_mean: Any
    y_std: Any
    initial_full_loss: float
    final_full_loss: float
    step_losses: list[float]
    normalized_step_losses: list[float]
    train_metrics: dict[str, float]
    seed: int
    step_count: int


class MicroOverfitError(RuntimeError):
    code: str
    stage: str

    def __init__(self, code: str, stage: str, message: str):
        super().__init__(message)
        self.code = str(code)
        self.stage = str(stage)


def exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True)


def validate_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise MicroOverfitError(
            "MISSING_FILE",
            arg_name,
            f"missing required {arg_name}: {resolved}",
        )
    return resolved


def validate_subset_size(subset_size: int) -> int:
    value = int(subset_size)
    if value < 32 or value > 128:
        raise MicroOverfitError(
            "INVALID_SUBSET_SIZE",
            "subset_selection",
            f"subset_size must be within [32, 128], got {value}",
        )
    return value


def stable_signed_hash(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    integer = int(digest[:16], 16)
    max_int = float((16**16) - 1)
    return float((integer / max_int) * 2.0 - 1.0)


def _import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise MicroOverfitError(
            "MISSING_RUNTIME_DEPENDENCY",
            "runtime_import",
            "numpy is required for state-conditioned sanity harnesses",
        ) from exc
    return np


def _import_torch() -> Any:
    try:
        import torch  # type: ignore
        from torch import nn  # type: ignore
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise MicroOverfitError(
            "MISSING_RUNTIME_DEPENDENCY",
            "runtime_import",
            "torch is required for micro-overfit sanity",
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _resolve_stats_path(labels_path: Path) -> Path:
    candidate = labels_path.with_name("state_conditioned_sft_stats.json")
    return validate_existing_file(
        candidate, arg_name="state_conditioned_sft_stats.json"
    )


def _resolve_lerobot_root(stats: Mapping[str, Any], stats_path: Path) -> Path:
    raw = stats.get("lerobot_dataset_path")
    if not isinstance(raw, str) or not raw.strip():
        raise MicroOverfitError(
            "INVALID_STATS_CONTRACT",
            "load_context",
            "state_conditioned_sft_stats.json must contain non-empty lerobot_dataset_path",
        )
    candidate = Path(raw.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (stats_path.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_dir():
        raise MicroOverfitError(
            "MISSING_LEROBOT_DATASET",
            "load_context",
            f"lerobot dataset directory does not exist: {candidate}",
        )
    return candidate


def load_training_context(labels_path: Path) -> TrainingContext:
    resolved_labels = validate_existing_file(labels_path, arg_name="labels")
    labels = read_jsonl(resolved_labels)
    if not labels:
        raise MicroOverfitError(
            "EMPTY_LABELS",
            "load_context",
            f"labels file is empty: {resolved_labels}",
        )
    stats_path = _resolve_stats_path(resolved_labels)
    stats = read_json(stats_path)
    lerobot_root = _resolve_lerobot_root(stats, stats_path)
    info_path = validate_existing_file(
        lerobot_root / "meta" / "info.json",
        arg_name="lerobot meta/info.json",
    )
    episodes_meta_path = validate_existing_file(
        lerobot_root / "meta" / "episodes.jsonl",
        arg_name="lerobot meta/episodes.jsonl",
    )
    info = read_json(info_path)
    episodes_meta = read_jsonl(episodes_meta_path)
    if not episodes_meta:
        raise MicroOverfitError(
            "EMPTY_EPISODES_META",
            "load_context",
            f"episodes metadata is empty: {episodes_meta_path}",
        )
    action_shape = list(info.get("features", {}).get("action", {}).get("shape", []))
    if len(action_shape) != 1 or not isinstance(action_shape[0], int):
        raise MicroOverfitError(
            "INVALID_LEROBOT_INFO",
            "load_context",
            "lerobot meta/info.json must expose a 1D action shape",
        )
    episodes_by_sample_id: dict[str, dict[str, Any]] = {}
    for row in episodes_meta:
        sample_id = str(row.get("state_conditioned.sample_id", "")).strip()
        if not sample_id:
            raise MicroOverfitError(
                "INVALID_EPISODES_META",
                "load_context",
                "episodes.jsonl is missing state_conditioned.sample_id",
            )
        if sample_id in episodes_by_sample_id:
            raise MicroOverfitError(
                "DUPLICATE_SAMPLE_ID",
                "load_context",
                f"duplicate sample_id in episodes metadata: {sample_id}",
            )
        episodes_by_sample_id[sample_id] = dict(row)
    max_episode_length = max(int(row.get("length", 0)) for row in episodes_meta)
    if max_episode_length <= 0:
        raise MicroOverfitError(
            "INVALID_EPISODE_LENGTH",
            "load_context",
            "episodes.jsonl must contain positive episode lengths",
        )
    chunks_size = int(info.get("chunks_size", 0))
    if chunks_size <= 0:
        raise MicroOverfitError(
            "INVALID_LEROBOT_INFO",
            "load_context",
            "lerobot meta/info.json must contain positive chunks_size",
        )
    data_path_template = str(info.get("data_path", "")).strip()
    if not data_path_template:
        raise MicroOverfitError(
            "INVALID_LEROBOT_INFO",
            "load_context",
            "lerobot meta/info.json must contain data_path template",
        )
    return TrainingContext(
        labels_path=resolved_labels,
        stats_path=stats_path,
        lerobot_root=lerobot_root,
        info_path=info_path,
        episodes_meta_path=episodes_meta_path,
        labels=labels,
        stats=stats,
        info=info,
        episodes_meta=episodes_meta,
        episodes_by_sample_id=episodes_by_sample_id,
        max_episode_length=int(max_episode_length),
        action_dim=int(action_shape[0]),
        chunks_size=int(chunks_size),
        data_path_template=data_path_template,
    )


def select_label_rows(
    labels: Sequence[Mapping[str, Any]],
    *,
    training_view: str,
) -> list[dict[str, Any]]:
    normalized_view = str(training_view).strip().upper()
    if normalized_view not in SUPPORTED_VIEWS:
        raise MicroOverfitError(
            "INVALID_TRAINING_VIEW",
            "subset_selection",
            f"training_view must be one of {SUPPORTED_VIEWS!r}, got {training_view!r}",
        )
    rows = [
        dict(row)
        for row in labels
        if str(row.get("training_view", "")).strip().upper() == normalized_view
    ]
    if not rows:
        raise MicroOverfitError(
            "EMPTY_VIEW",
            "subset_selection",
            f"no label rows found for training_view={normalized_view}",
        )
    return rows


def _history_k_from_rows(rows: Sequence[Mapping[str, Any]]) -> int:
    history_lengths = {
        len(list(row.get("history_valid_mask", [])))
        for row in rows
        if "history_valid_mask" in row
    }
    if len(history_lengths) != 1:
        raise MicroOverfitError(
            "INCONSISTENT_HISTORY_K",
            "feature_spec",
            f"history_valid_mask lengths are inconsistent: {sorted(history_lengths)!r}",
        )
    history_k = int(next(iter(history_lengths)))
    if history_k <= 0:
        raise MicroOverfitError(
            "INVALID_HISTORY_K",
            "feature_spec",
            f"history_valid_mask length must be > 0, got {history_k}",
        )
    return history_k


def build_feature_spec(rows: Sequence[Mapping[str, Any]]) -> FeatureSpec:
    history_k = _history_k_from_rows(rows)
    training_view_values = tuple(
        sorted({str(row.get("training_view", "")).strip() for row in rows})
    )
    phase_values = tuple(
        sorted({str(row.get("policy_condition.phase", "")).strip() for row in rows})
    )
    mode_values = tuple(
        sorted({str(row.get("policy_condition.mode", "")).strip() for row in rows})
    )
    previous_action_dim = 0
    proprio_dim = 0
    for row in rows:
        for slot in list(row.get("deployable.previous_action_history", [])):
            if isinstance(slot, list):
                previous_action_dim = max(previous_action_dim, len(slot))
        for slot in list(row.get("deployable.proprio_history", [])):
            if isinstance(slot, list):
                proprio_dim = max(proprio_dim, len(slot))
    return FeatureSpec(
        history_k=history_k,
        previous_action_dim=int(previous_action_dim),
        proprio_dim=int(proprio_dim),
        training_view_values=training_view_values,
        phase_values=phase_values,
        mode_values=mode_values,
    )


def _one_hot(value: object, categories: Sequence[str]) -> list[float]:
    normalized = str(value)
    return [1.0 if normalized == category else 0.0 for category in categories]


def _coerce_bool_list(
    value: object, *, field_name: str, expected_len: int
) -> list[bool]:
    if not isinstance(value, list):
        raise MicroOverfitError(
            "INVALID_LABEL_FIELD",
            "feature_encode",
            f"{field_name} must be a list, got {type(value).__name__}",
        )
    items = [bool(item) for item in value]
    if len(items) != expected_len:
        raise MicroOverfitError(
            "INVALID_LABEL_FIELD",
            "feature_encode",
            f"{field_name} must have length {expected_len}, got {len(items)}",
        )
    return items


def _coerce_list(value: object, *, field_name: str, expected_len: int) -> list[Any]:
    if not isinstance(value, list):
        raise MicroOverfitError(
            "INVALID_LABEL_FIELD",
            "feature_encode",
            f"{field_name} must be a list, got {type(value).__name__}",
        )
    items = list(value)
    if len(items) != expected_len:
        raise MicroOverfitError(
            "INVALID_LABEL_FIELD",
            "feature_encode",
            f"{field_name} must have length {expected_len}, got {len(items)}",
        )
    return items


def _float_or_zero(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise MicroOverfitError(
                "NONFINITE_LABEL_VALUE",
                "feature_encode",
                f"encountered non-finite numeric label value: {value!r}",
            )
        return number
    if isinstance(value, str):
        return stable_signed_hash(value)
    raise MicroOverfitError(
        "UNSUPPORTED_LABEL_VALUE",
        "feature_encode",
        f"unsupported label value type: {type(value).__name__}",
    )


def _encode_sequence_slot(slot: object, *, width: int, is_valid: bool) -> list[float]:
    encoded = [1.0 if is_valid and slot is not None else 0.0]
    if width <= 0:
        return encoded
    values = [0.0] * width
    if is_valid and isinstance(slot, list):
        for index, item in enumerate(slot[:width]):
            values[index] = _float_or_zero(item)
    encoded.extend(values)
    return encoded


def encode_label_row(row: Mapping[str, Any], spec: FeatureSpec) -> list[float]:
    history_valid_mask = _coerce_bool_list(
        row.get("history_valid_mask"),
        field_name="history_valid_mask",
        expected_len=spec.history_k,
    )
    history_t_std_indices = _coerce_list(
        row.get("history_t_std_indices"),
        field_name="history_t_std_indices",
        expected_len=spec.history_k,
    )
    history_t_raw_indices = _coerce_list(
        row.get("history_t_raw_indices"),
        field_name="history_t_raw_indices",
        expected_len=spec.history_k,
    )
    history_timestamp_s = _coerce_list(
        row.get("history_timestamp_s"),
        field_name="history_timestamp_s",
        expected_len=spec.history_k,
    )
    previous_action_history = _coerce_list(
        row.get("deployable.previous_action_history"),
        field_name="deployable.previous_action_history",
        expected_len=spec.history_k,
    )
    proprio_history = _coerce_list(
        row.get("deployable.proprio_history"),
        field_name="deployable.proprio_history",
        expected_len=spec.history_k,
    )
    short_visual_history_refs = _coerce_list(
        row.get("deployable.short_visual_history_refs"),
        field_name="deployable.short_visual_history_refs",
        expected_len=spec.history_k,
    )
    features: list[float] = []
    features.extend(_one_hot(row.get("training_view", ""), spec.training_view_values))
    features.extend(_one_hot(row.get("policy_condition.phase", ""), spec.phase_values))
    features.extend(_one_hot(row.get("policy_condition.mode", ""), spec.mode_values))
    policy_condition_text = str(row.get("policy_condition_text", ""))
    features.extend(
        [
            1.0 if policy_condition_text else 0.0,
            float(len(policy_condition_text)) / 128.0,
            stable_signed_hash(policy_condition_text) if policy_condition_text else 0.0,
        ]
    )
    features.append(float(int(row.get("history_k", spec.history_k))))
    features.append(float(int(row.get("history_stride", 1))))
    features.extend([1.0 if value else 0.0 for value in history_valid_mask])
    features.extend([_float_or_zero(value) for value in history_t_std_indices])
    features.extend([_float_or_zero(value) for value in history_t_raw_indices])
    for is_valid, timestamp in zip(
        history_valid_mask, history_timestamp_s, strict=True
    ):
        features.append(_float_or_zero(timestamp) if is_valid else 0.0)
    for is_valid, slot in zip(history_valid_mask, previous_action_history, strict=True):
        features.extend(
            _encode_sequence_slot(
                slot,
                width=spec.previous_action_dim,
                is_valid=bool(is_valid),
            )
        )
    for is_valid, slot in zip(history_valid_mask, proprio_history, strict=True):
        features.extend(
            _encode_sequence_slot(
                slot,
                width=spec.proprio_dim,
                is_valid=bool(is_valid),
            )
        )
    for is_valid, slot in zip(
        history_valid_mask, short_visual_history_refs, strict=True
    ):
        features.append(1.0 if is_valid and slot is not None else 0.0)
    return features


def _episode_parquet_path(context: TrainingContext, episode_index: int) -> Path:
    relative = context.data_path_template.format(
        episode_chunk=int(episode_index) // int(context.chunks_size),
        episode_index=int(episode_index),
    )
    return validate_existing_file(
        context.lerobot_root / relative,
        arg_name=f"lerobot episode parquet {episode_index}",
    )


def _load_action_chunk_from_parquet(path: Path, *, expected_action_dim: int) -> Any:
    np = _import_numpy()
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path, columns=["action"])
        rows = [np.asarray(item.as_py(), dtype=np.float32) for item in table["action"]]
    except ModuleNotFoundError:
        try:
            import pandas as pd  # type: ignore
        except ModuleNotFoundError as exc:
            raise MicroOverfitError(
                "MISSING_RUNTIME_DEPENDENCY",
                "load_action_chunk",
                "reading parquet requires pyarrow or pandas",
            ) from exc
        frame = pd.read_parquet(path, columns=["action"])
        rows = [
            np.asarray(value, dtype=np.float32) for value in frame["action"].tolist()
        ]
    if not rows:
        raise MicroOverfitError(
            "EMPTY_ACTION_CHUNK",
            "load_action_chunk",
            f"episode parquet contains no action rows: {path}",
        )
    action_chunk = np.stack(rows, axis=0).astype(np.float32, copy=False)
    if action_chunk.ndim != 2 or int(action_chunk.shape[1]) != int(expected_action_dim):
        raise MicroOverfitError(
            "INVALID_ACTION_CHUNK",
            "load_action_chunk",
            "episode action chunk has unexpected shape: "
            + f"{tuple(action_chunk.shape)} expected second dim={expected_action_dim}",
        )
    return action_chunk


def materialize_encoded_dataset(
    context: TrainingContext,
    rows: Sequence[Mapping[str, Any]],
    spec: FeatureSpec,
) -> EncodedDataset:
    np = _import_numpy()
    features: list[Any] = []
    targets: list[Any] = []
    target_mask: list[Any] = []
    row_refs: list[dict[str, Any]] = []
    action_cache: dict[int, Any] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise MicroOverfitError(
                "INVALID_LABEL_FIELD",
                "materialize_dataset",
                "label row is missing sample_id",
            )
        episode_meta = context.episodes_by_sample_id.get(sample_id)
        if episode_meta is None:
            raise MicroOverfitError(
                "DATASET_ALIGNMENT_ERROR",
                "materialize_dataset",
                f"sample_id not found in lerobot episodes metadata: {sample_id}",
            )
        episode_index = int(episode_meta.get("episode_index", -1))
        if episode_index < 0:
            raise MicroOverfitError(
                "INVALID_EPISODE_INDEX",
                "materialize_dataset",
                f"invalid episode_index for sample_id={sample_id}",
            )
        if episode_index not in action_cache:
            action_cache[episode_index] = _load_action_chunk_from_parquet(
                _episode_parquet_path(context, episode_index),
                expected_action_dim=context.action_dim,
            )
        action_chunk = action_cache[episode_index]
        episode_length = int(action_chunk.shape[0])
        if episode_length > int(context.max_episode_length):
            raise MicroOverfitError(
                "INVALID_ACTION_CHUNK",
                "materialize_dataset",
                f"episode_length exceeds max_episode_length: {episode_length} > {context.max_episode_length}",
            )
        padded_target = np.zeros(
            (int(context.max_episode_length), int(context.action_dim)),
            dtype=np.float32,
        )
        padded_target_mask = np.zeros_like(padded_target)
        padded_target[:episode_length, :] = action_chunk
        padded_target_mask[:episode_length, :] = 1.0
        features.append(np.asarray(encode_label_row(row, spec), dtype=np.float32))
        targets.append(padded_target.reshape(-1))
        target_mask.append(padded_target_mask.reshape(-1))
        row_refs.append(
            {
                "sample_id": sample_id,
                "episode_index": episode_index,
                "episode_length": episode_length,
                "training_view": str(row.get("training_view", "")),
                "policy_condition.phase": str(row.get("policy_condition.phase", "")),
                "policy_condition.mode": str(row.get("policy_condition.mode", "")),
                "valid_history_count": int(
                    sum(
                        bool(value) for value in list(row.get("history_valid_mask", []))
                    )
                ),
            }
        )
    feature_matrix = np.stack(features, axis=0).astype(np.float32, copy=False)
    target_matrix = np.stack(targets, axis=0).astype(np.float32, copy=False)
    target_mask_matrix = np.stack(target_mask, axis=0).astype(np.float32, copy=False)
    return EncodedDataset(
        features=feature_matrix,
        targets=target_matrix,
        target_mask=target_mask_matrix,
        row_refs=row_refs,
        feature_dim=int(feature_matrix.shape[1]),
        target_dim=int(target_matrix.shape[1]),
        max_episode_length=int(context.max_episode_length),
        action_dim=int(context.action_dim),
    )


def balanced_subset_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    subset_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    validate_subset_size(subset_size)
    if len(rows) < subset_size:
        raise MicroOverfitError(
            "INSUFFICIENT_ROWS",
            "subset_selection",
            f"requested subset_size={subset_size} but only {len(rows)} rows are available",
        )
    rng = random.Random(int(seed))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("policy_condition.phase", "")),
                str(row.get("policy_condition.mode", "")),
            )
        ].append(dict(row))
    ordered_groups = sorted(grouped)
    for key in ordered_groups:
        rng.shuffle(grouped[key])
    selected: list[dict[str, Any]] = []
    while len(selected) < subset_size:
        progressed = False
        for key in ordered_groups:
            bucket = grouped[key]
            if not bucket:
                continue
            selected.append(bucket.pop())
            progressed = True
            if len(selected) >= subset_size:
                break
        if not progressed:
            break
    if len(selected) != subset_size:
        raise MicroOverfitError(
            "INSUFFICIENT_ROWS",
            "subset_selection",
            f"balanced subset selection stopped at {len(selected)} rows (expected {subset_size})",
        )
    return selected


def _masked_full_loss(prediction: Any, target: Any, target_mask: Any) -> Any:
    diff = (prediction - target) * target_mask
    return (diff * diff).sum() / target_mask.sum().clamp_min(1.0)


def masked_regression_metrics(
    *,
    prediction: Any,
    target: Any,
    target_mask: Any,
) -> dict[str, float]:
    np = _import_numpy()
    prediction_np = np.asarray(prediction, dtype=np.float32)
    target_np = np.asarray(target, dtype=np.float32)
    target_mask_np = np.asarray(target_mask, dtype=np.float32)
    if (
        prediction_np.shape != target_np.shape
        or prediction_np.shape != target_mask_np.shape
    ):
        raise MicroOverfitError(
            "SHAPE_MISMATCH",
            "metrics",
            "prediction, target, and target_mask must share the same shape",
        )
    valid_count = float(target_mask_np.sum())
    if valid_count <= 0.0:
        raise MicroOverfitError(
            "EMPTY_TARGET_MASK",
            "metrics",
            "target_mask contains no valid entries",
        )
    diff = (prediction_np - target_np) * target_mask_np
    mse = float((diff * diff).sum() / valid_count)
    mae = float(np.abs(diff).sum() / valid_count)
    abs_target = np.abs(target_np) * target_mask_np
    abs_prediction = np.abs(prediction_np) * target_mask_np
    return {
        "masked_mse": mse,
        "masked_rmse": float(math.sqrt(mse)),
        "masked_mae": mae,
        "teacher_abs_mean": float(abs_target.sum() / valid_count),
        "prediction_abs_mean": float(abs_prediction.sum() / valid_count),
        "valid_entry_count": float(valid_count),
    }


def train_proxy_model(
    dataset: EncodedDataset,
    *,
    seed: int,
    max_steps: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
) -> ProxyFitResult:
    if int(max_steps) <= 0:
        raise MicroOverfitError(
            "INVALID_TRAINING_CONFIG",
            "train_proxy_model",
            f"max_steps must be > 0, got {max_steps}",
        )
    if int(batch_size) <= 0:
        raise MicroOverfitError(
            "INVALID_TRAINING_CONFIG",
            "train_proxy_model",
            f"batch_size must be > 0, got {batch_size}",
        )
    torch, nn, DataLoader, TensorDataset = _import_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    features = torch.tensor(dataset.features, dtype=torch.float32)
    targets = torch.tensor(dataset.targets, dtype=torch.float32)
    target_mask = torch.tensor(dataset.target_mask, dtype=torch.float32)
    x_mean = features.mean(dim=0)
    x_std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_features = (features - x_mean) / x_std
    valid_count = target_mask.sum(dim=0).clamp_min(1.0)
    y_mean = (targets * target_mask).sum(dim=0) / valid_count
    centered_targets = (targets - y_mean) * target_mask
    y_var = (centered_targets * centered_targets).sum(dim=0) / valid_count
    y_std = y_var.sqrt().clamp_min(1e-3)
    normalized_targets = (targets - y_mean) / y_std

    class ProxyActionMLP(nn.Module):
        def __init__(self, input_dim: int, width: int, output_dim: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, output_dim),
            )

        def forward(self, batch_x: Any) -> Any:
            return self.network(batch_x)

    model = ProxyActionMLP(
        input_dim=int(normalized_features.shape[1]),
        width=int(hidden_dim),
        output_dim=int(normalized_targets.shape[1]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    dataset_loader = TensorDataset(normalized_features, normalized_targets, target_mask)
    loader_generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset_loader,
        batch_size=min(int(batch_size), int(normalized_features.shape[0])),
        shuffle=True,
        generator=loader_generator,
    )

    with torch.inference_mode():
        initial_prediction = model(normalized_features)
        initial_full_loss = float(
            _masked_full_loss(
                initial_prediction, normalized_targets, target_mask
            ).item()
        )
    if not math.isfinite(initial_full_loss):
        raise MicroOverfitError(
            "NONFINITE_LOSS",
            "train_proxy_model",
            f"initial full loss is non-finite: {initial_full_loss!r}",
        )

    normalized_step_losses: list[float] = []
    step_losses: list[float] = []
    completed_steps = 0
    while completed_steps < int(max_steps):
        for batch_features, batch_targets, batch_mask in loader:
            prediction = model(batch_features)
            loss = _masked_full_loss(prediction, batch_targets, batch_mask)
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise MicroOverfitError(
                    "NONFINITE_LOSS",
                    "train_proxy_model",
                    f"encountered non-finite optimization loss at step={completed_steps}: {loss_value!r}",
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            normalized_step_losses.append(loss_value)
            step_losses.append(loss_value)
            completed_steps += 1
            if completed_steps >= int(max_steps):
                break

    with torch.inference_mode():
        normalized_prediction = model(normalized_features)
        final_full_loss = float(
            _masked_full_loss(
                normalized_prediction, normalized_targets, target_mask
            ).item()
        )
        prediction = (normalized_prediction * y_std) + y_mean
    if not math.isfinite(final_full_loss):
        raise MicroOverfitError(
            "NONFINITE_LOSS",
            "train_proxy_model",
            f"final full loss is non-finite: {final_full_loss!r}",
        )
    train_metrics = masked_regression_metrics(
        prediction=prediction.detach().cpu().numpy(),
        target=targets.detach().cpu().numpy(),
        target_mask=target_mask.detach().cpu().numpy(),
    )
    return ProxyFitResult(
        model=model,
        x_mean=x_mean.detach().cpu(),
        x_std=x_std.detach().cpu(),
        y_mean=y_mean.detach().cpu(),
        y_std=y_std.detach().cpu(),
        initial_full_loss=float(initial_full_loss),
        final_full_loss=float(final_full_loss),
        step_losses=[float(value) for value in step_losses],
        normalized_step_losses=[float(value) for value in normalized_step_losses],
        train_metrics=train_metrics,
        seed=int(seed),
        step_count=int(completed_steps),
    )


def predict_proxy_model(fit_result: ProxyFitResult, features: Any) -> Any:
    np = _import_numpy()
    torch, _, _, _ = _import_torch()
    feature_tensor = torch.tensor(
        np.asarray(features, dtype=np.float32), dtype=torch.float32
    )
    normalized = (feature_tensor - fit_result.x_mean) / fit_result.x_std
    with torch.inference_mode():
        normalized_prediction = fit_result.model(normalized)
        prediction = (normalized_prediction * fit_result.y_std) + fit_result.y_mean
    return prediction.detach().cpu().numpy().astype(np.float32, copy=False)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_phase_mode: dict[str, int] = defaultdict(int)
    valid_history_counts: list[int] = []
    for row in rows:
        key = (
            f"{str(row.get('policy_condition.phase', ''))}|"
            + f"{str(row.get('policy_condition.mode', ''))}"
        )
        by_phase_mode[key] += 1
        valid_history_counts.append(
            int(sum(bool(value) for value in list(row.get("history_valid_mask", []))))
        )
    return {
        "row_count": len(rows),
        "phase_mode_counts": dict(sorted(by_phase_mode.items())),
        "min_valid_history_count": min(valid_history_counts)
        if valid_history_counts
        else 0,
        "max_valid_history_count": max(valid_history_counts)
        if valid_history_counts
        else 0,
    }


def _base_payload(output_path: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "status": "FAIL",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_root": str(REPO_ROOT),
        "output_path": str(output_path),
        "failure": None,
        "summary": None,
        "telemetry": {},
    }


def run_micro_overfit(
    *,
    labels_path: Path,
    output_path: Path,
    training_view: str,
    subset_size: int,
    seed: int,
    max_steps: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    payload = _base_payload(output_path)
    try:
        subset_size = validate_subset_size(subset_size)
        context = load_training_context(labels_path)
        rows = select_label_rows(context.labels, training_view=training_view)
        subset_rows = balanced_subset_rows(rows, subset_size=subset_size, seed=seed)
        spec = build_feature_spec(subset_rows)
        dataset = materialize_encoded_dataset(context, subset_rows, spec)
        fit_result = train_proxy_model(
            dataset,
            seed=seed,
            max_steps=max_steps,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        first_loss = float(fit_result.initial_full_loss)
        last_loss = float(fit_result.final_full_loss)
        absolute_drop = float(first_loss - last_loss)
        ratio = float(last_loss / first_loss) if first_loss > 0.0 else float("inf")
        loss_decreased = bool(
            absolute_drop >= float(LOSS_DECREASE_ABS_THRESHOLD)
            and ratio <= float(LOSS_DECREASE_RATIO_THRESHOLD)
        )
        if not loss_decreased:
            raise MicroOverfitError(
                "LOSS_NOT_DECREASING",
                "loss_gate",
                "micro-overfit loss did not decrease enough to prove small-sample learnability",
            )
        payload["status"] = "PASS"
        payload["summary"] = {
            "training_view": str(training_view).upper(),
            "subset_size": int(subset_size),
            "seed": int(seed),
            "feature_dim": int(dataset.feature_dim),
            "target_dim": int(dataset.target_dim),
            "max_episode_length": int(dataset.max_episode_length),
            "action_dim": int(dataset.action_dim),
        }
        payload["telemetry"] = {
            "loss": {
                "first_full_loss": first_loss,
                "last_full_loss": last_loss,
                "absolute_drop": absolute_drop,
                "ratio_last_over_first": ratio,
                "step_count": int(fit_result.step_count),
                "step_loss_head": [
                    float(value) for value in fit_result.step_losses[:8]
                ],
                "step_loss_tail": [
                    float(value) for value in fit_result.step_losses[-8:]
                ],
            },
            "train_metrics": dict(fit_result.train_metrics),
            "subset_summary": summarize_rows(subset_rows),
            "label_paths": {
                "labels_path": str(context.labels_path),
                "stats_path": str(context.stats_path),
                "lerobot_root": str(context.lerobot_root),
            },
        }
        return payload
    except MicroOverfitError as exc:
        payload["failure"] = {
            "code": exc.code,
            "stage": exc.stage,
            "type": exc.__class__.__name__,
            "message": exception_message(exc),
        }
        return payload
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        payload["failure"] = {
            "code": "UNHANDLED_ERROR",
            "stage": "cli",
            "type": exc.__class__.__name__,
            "message": exception_message(exc),
        }
        return payload


def materialize_micro_overfit(
    *,
    labels_path: Path,
    output_path: Path,
    training_view: str,
    subset_size: int,
    seed: int,
    max_steps: int,
    batch_size: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    resolved_output = output_path.expanduser().resolve()
    payload = run_micro_overfit(
        labels_path=labels_path,
        output_path=resolved_output,
        training_view=training_view,
        subset_size=subset_size,
        seed=seed,
        max_steps=max_steps,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    payload["report_path"] = str(resolved_output)
    write_json(resolved_output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="state_conditioned_micro_overfit_sanity.py",
        description=(
            "Run a bounded micro-overfit sanity harness on 32-128 state-conditioned "
            "label rows to prove the label wrapper/dataloader/proxy-model path is learnable."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--training-view",
        type=str,
        default=DEFAULT_TRAINING_VIEW,
        choices=SUPPORTED_VIEWS,
    )
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = materialize_micro_overfit(
        labels_path=Path(str(args.labels)),
        output_path=Path(str(args.output)),
        training_view=str(args.training_view),
        subset_size=int(args.subset_size),
        seed=int(args.seed),
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    if payload.get("status") != "PASS":
        failure = dict(payload.get("failure") or {})
        print(
            str(failure.get("message", "micro-overfit sanity failed")), file=sys.stderr
        )
    print(json_text(payload))
    return 0 if payload.get("status") == "PASS" else 1


__all__ = [
    "DEFAULT_LABELS_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_TRAINING_VIEW",
    "EncodedDataset",
    "FeatureSpec",
    "MicroOverfitError",
    "ProxyFitResult",
    "SCHEMA_VERSION",
    "TrainingContext",
    "balanced_subset_rows",
    "build_feature_spec",
    "build_parser",
    "encode_label_row",
    "exception_message",
    "json_text",
    "load_training_context",
    "main",
    "masked_regression_metrics",
    "materialize_encoded_dataset",
    "materialize_micro_overfit",
    "predict_proxy_model",
    "read_json",
    "read_jsonl",
    "run_micro_overfit",
    "select_label_rows",
    "summarize_rows",
    "train_proxy_model",
    "validate_subset_size",
    "write_json",
]


if __name__ == "__main__":
    raise SystemExit(main())
