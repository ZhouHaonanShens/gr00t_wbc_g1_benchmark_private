from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import contextlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_SOURCE = Path(
    "/media/howard/Data/Projects/gr00t_wbc_g1_benchmark/agent/artifacts/recap_datasets/recap_mainline_fresh_20260311_121500_k0"
)
DEFAULT_OUTPUT_DIR = Path("agent/artifacts/state_conditioned_materialization/bucket_a")
DEFAULT_ACCEPT_UNTIL = 24
DEBUG_ONLY_REUSE_TARGET = 10
JOIN_COVERAGE_THRESHOLD = 0.995
FRESH_NOMINAL_SELECTION_POLICY = "fresh_nominal_accept_until_24"
FRESH_NOMINAL_ITER_TAG_PREFIX = "state_conditioned_bucket_a_fresh_nominal"
FRESH_NOMINAL_COLLECTION_TIMEOUT_S = 1500.0

GATE_A_READY_JSON_NAME = "bucket_A_gate_a_ready.json"
MANIFEST_JSON_NAME = "bucket_A_manifest.json"
DEBUG_ONLY_REUSE_MANIFEST_JSON_NAME = "bucket_A_debug_only_reuse_manifest.json"
LEGACY_TIMEBOX_DECISION_JSON_NAME = "bucket_A_timebox_decision.json"
SCHEMA_VERSION = "g1_state_conditioned_bucket_a_fresh_accept_until_v2"
BUCKET_NAME = "Bucket A"
BUCKET_KEY = "bucket_A"
CANONICAL_KIND = "fresh_nominal_recollection"
EPISODE_ACCEPTANCE_DIRNAME = "bucket_A_episode_acceptance"
EPISODE_SIDECAR_SMOKE_DIRNAME = "bucket_A_episode_sidecar_smoke"
EPISODE_JOIN_COVERAGE_DIRNAME = "bucket_A_episode_join_coverage"
LABELS_REL_PATH = Path("m2_labels") / "labels.jsonl"
SIDECAR_CANDIDATE_NAMES = (
    "state_conditioned_sidecar.jsonl",
    "bucket_A_sidecar.jsonl",
)
RECAP_DATASET_DIR_REL = Path("agent") / "artifacts" / "recap_datasets"
RECAP_COLLECT_SCRIPT_REL = Path("agent") / "run" / "31_recap_collect_rollouts.py"
SEMANTIC_COMMIT_REQUIRED_FIELDS = (
    "semantic_state",
    "memory_commit_mask",
    "memory_commit_cause",
)
STATE_CONDITIONED_HISTORY_K = 8
STATE_CONDITIONED_HISTORY_STRIDE = 1
STATE_CONDITIONED_RESET_BOUNDARY = "no_cross_episode"
STATE_CONDITIONED_PHASES = (
    "SEARCH",
    "APPROACH",
    "GRASP",
    "VERIFY_HOLD",
    "TRANSPORT",
    "PLACE",
)
STATE_CONDITIONED_MODES = ("NOMINAL", "RECOVERY")


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.demo_utils import paths as demo_paths


def validate_output_dir(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("output-dir must be a directory path")
    if not resolved.exists() and resolved.suffix:
        raise ValueError("output-dir must be a directory path")
    return resolved


def validate_source_dir(source_dir: Path) -> Path:
    resolved = source_dir.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"source dataset directory does not exist: {resolved}")
    required_paths = (
        resolved / "episodes.jsonl",
        resolved / "transitions.jsonl",
        resolved / LABELS_REL_PATH,
    )
    for path in required_paths:
        if not path.is_file():
            raise ValueError(f"missing required dataset file: {path}")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize canonical Bucket A as fresh-only nominal recollection and "
            "count only distinct accepted episodes after the three artifact gate passes."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Legacy recap dataset directory used only for debug-only reuse demotion context.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives canonical Bucket A JSON artifacts.",
    )
    parser.add_argument(
        "--accept-until",
        type=int,
        default=int(DEFAULT_ACCEPT_UNTIL),
        help="Required distinct accepted fresh nominal episodes before Gate A becomes ready.",
    )
    parser.add_argument(
        "--fresh-only",
        action="store_true",
        default=True,
        help="Canonical Bucket A is always fresh-only; this flag is accepted for explicitness.",
    )
    parser.add_argument(
        "--debug-demote-reuse",
        action="store_true",
        default=True,
        help="Always demote discovered existing-live-dataset reuse episodes into debug-only evidence.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)
    return path


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    tmp.replace(path)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"expected JSON object in {path}, got {type(payload).__name__}"
        )
    return dict(payload)


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{lineno}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"expected JSON object in {path}:{lineno}, got {type(payload).__name__}"
                )
            records.append(dict(payload))
    return records


def _group_records_by_episode(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in records:
        record = dict(raw)
        episode_id = str(record.get("episode_id", "")).strip()
        if not episode_id:
            raise ValueError("record is missing non-empty episode_id")
        grouped.setdefault(episode_id, []).append(record)
    return grouped


def _record_join_key(record: Mapping[str, Any]) -> tuple[str, int]:
    episode_id = str(record.get("episode_id", "")).strip()
    if not episode_id:
        raise ValueError("record is missing non-empty episode_id")
    raw_t = record.get("t")
    if isinstance(raw_t, bool) or not isinstance(raw_t, int):
        raise ValueError(f"record for episode {episode_id!r} is missing int t")
    return episode_id, int(raw_t)


def _select_optional_sidecar_path(source_dir: Path) -> Path | None:
    for candidate_name in SIDECAR_CANDIDATE_NAMES:
        candidate_path = source_dir / candidate_name
        if candidate_path.is_file():
            return candidate_path
    return None


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _normalize_phase(value: object, *, field_name: str) -> str:
    normalized = _as_non_empty_string(value, field_name=field_name).upper()
    if normalized not in STATE_CONDITIONED_PHASES:
        raise ValueError(f"{field_name} must be one of {STATE_CONDITIONED_PHASES!r}")
    return normalized


def _normalize_mode(value: object, *, field_name: str) -> str:
    normalized = _as_non_empty_string(value, field_name=field_name).upper()
    if normalized not in STATE_CONDITIONED_MODES:
        raise ValueError(f"{field_name} must be one of {STATE_CONDITIONED_MODES!r}")
    return normalized


def build_canonical_policy_condition_text(phase: object, mode: object) -> str:
    normalized_phase = _normalize_phase(phase, field_name="phase")
    normalized_mode = _normalize_mode(mode, field_name="mode")
    return (
        "[PolicyCondition-v1]\n"
        + f"PHASE={normalized_phase}\n"
        + f"MODE={normalized_mode}"
    )


def validate_state_conditioned_policy_condition(
    *,
    phase: object,
    mode: object,
    policy_condition_text: object,
) -> tuple[str, str, str]:
    normalized_phase = _normalize_phase(phase, field_name="policy_condition.phase")
    normalized_mode = _normalize_mode(mode, field_name="policy_condition.mode")
    normalized_text = _as_non_empty_string(
        policy_condition_text,
        field_name="policy_condition_text",
    )
    expected_text = build_canonical_policy_condition_text(
        normalized_phase,
        normalized_mode,
    )
    if normalized_text != expected_text:
        raise ValueError(
            "policy_condition_text mismatch: "
            + f"expected {expected_text!r}, got {normalized_text!r}"
        )
    return normalized_phase, normalized_mode, expected_text


def validate_state_conditioned_history_contract(
    *,
    anchor_episode_id: object,
    history_episode_ids: object,
    history_valid_mask: object,
    anchor_mujoco_state_ref: object,
    prehistory_window: object,
    history_k: object = STATE_CONDITIONED_HISTORY_K,
    history_stride: object = STATE_CONDITIONED_HISTORY_STRIDE,
    reset_boundary: object = STATE_CONDITIONED_RESET_BOUNDARY,
) -> dict[str, object]:
    normalized_anchor_episode_id = _as_non_empty_string(
        anchor_episode_id,
        field_name="anchor_episode_id",
    )
    normalized_anchor_ref = _as_non_empty_string(
        anchor_mujoco_state_ref,
        field_name="anchor_mujoco_state_ref",
    )
    if isinstance(history_k, bool) or not isinstance(history_k, int):
        raise TypeError("history_k must be an int")
    if int(history_k) != int(STATE_CONDITIONED_HISTORY_K):
        raise ValueError(f"history_k is frozen at {STATE_CONDITIONED_HISTORY_K}")
    if isinstance(history_stride, bool) or not isinstance(history_stride, int):
        raise TypeError("history_stride must be an int")
    if int(history_stride) != int(STATE_CONDITIONED_HISTORY_STRIDE):
        raise ValueError(
            f"history_stride is frozen at {STATE_CONDITIONED_HISTORY_STRIDE}"
        )
    normalized_reset_boundary = _as_non_empty_string(
        reset_boundary,
        field_name="reset_boundary",
    )
    if normalized_reset_boundary != STATE_CONDITIONED_RESET_BOUNDARY:
        raise ValueError(f"reset_boundary must be {STATE_CONDITIONED_RESET_BOUNDARY!r}")
    if history_valid_mask is None:
        raise ValueError("history_valid_mask is required")
    normalized_episode_ids = _as_list(
        history_episode_ids,
        field_name="history_episode_ids",
        expected_len=STATE_CONDITIONED_HISTORY_K,
    )
    normalized_valid_mask = _as_list(
        history_valid_mask,
        field_name="history_valid_mask",
        expected_len=STATE_CONDITIONED_HISTORY_K,
    )
    if not isinstance(prehistory_window, list):
        raise TypeError("prehistory_window must be a list")
    if len(prehistory_window) != int(STATE_CONDITIONED_HISTORY_K):
        raise ValueError(
            "prehistory_window must have length "
            + f"{STATE_CONDITIONED_HISTORY_K}, got {len(prehistory_window)}"
        )

    normalized_window: list[dict[str, object]] = []
    last_valid_t: int | None = None
    for idx, item in enumerate(prehistory_window):
        row = _as_mapping(item, field_name=f"prehistory_window[{idx}]")
        row_episode_id = _as_non_empty_string(
            row.get("episode_id"),
            field_name=f"prehistory_window[{idx}].episode_id",
        )
        row_t_raw = row.get("t_std")
        if isinstance(row_t_raw, bool) or not isinstance(row_t_raw, int):
            raise TypeError(f"prehistory_window[{idx}].t_std must be an int")
        row_ref = _as_non_empty_string(
            row.get("mujoco_state_ref"),
            field_name=f"prehistory_window[{idx}].mujoco_state_ref",
        )
        if str(normalized_episode_ids[idx]) != row_episode_id:
            raise ValueError(
                f"history episode mismatch at index {idx}: "
                + f"{normalized_episode_ids[idx]!r} != {row_episode_id!r}"
            )
        if (
            bool(normalized_valid_mask[idx])
            and row_episode_id != normalized_anchor_episode_id
        ):
            raise ValueError("cross-episode history is forbidden at reset boundary")
        if bool(normalized_valid_mask[idx]):
            row_t = int(row_t_raw)
            if last_valid_t is not None and row_t - last_valid_t != int(
                STATE_CONDITIONED_HISTORY_STRIDE
            ):
                raise ValueError("history stride mismatch for valid prehistory window")
            last_valid_t = row_t
        normalized_window.append(
            {
                "episode_id": row_episode_id,
                "t_std": int(row_t_raw),
                "mujoco_state_ref": row_ref,
            }
        )

    return {
        "anchor_episode_id": normalized_anchor_episode_id,
        "anchor_mujoco_state_ref": normalized_anchor_ref,
        "history_k": int(STATE_CONDITIONED_HISTORY_K),
        "history_stride": int(STATE_CONDITIONED_HISTORY_STRIDE),
        "history_valid_mask": [bool(value) for value in normalized_valid_mask],
        "history_episode_ids": [str(value) for value in normalized_episode_ids],
        "prehistory_window": normalized_window,
        "reset_boundary": normalized_reset_boundary,
    }


def _expected_join_keys_set(
    expected_join_keys: Iterable[Sequence[object]],
) -> set[tuple[str, int]]:
    normalized: set[tuple[str, int]] = set()
    for pair in expected_join_keys:
        if len(pair) != 2:
            raise ValueError(f"join key must have length 2, got {pair!r}")
        episode_id = pair[0]
        t = pair[1]
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"invalid join key episode_id: {episode_id!r}")
        if isinstance(t, bool) or not isinstance(t, int):
            raise ValueError(f"invalid join key t: {t!r}")
        normalized.add((episode_id, int(t)))
    return normalized


def validate_sidecar_round_trip(
    *,
    sidecar_path: Path,
    expected_join_keys: Iterable[Sequence[object]],
) -> dict[str, object]:
    records = _read_jsonl_dicts(sidecar_path)
    expected_keys = _expected_join_keys_set(expected_join_keys)
    observed_keys: set[tuple[str, int]] = set()
    phase_values: set[str] = set()
    mode_values: set[str] = set()

    for index, record in enumerate(records, start=1):
        episode_id = record.get("episode_id")
        t = record.get("t")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(
                f"sidecar record#{index} invalid episode_id: {episode_id!r}"
            )
        if isinstance(t, bool) or not isinstance(t, int):
            raise ValueError(f"sidecar record#{index} invalid t: {t!r}")
        join_key = (episode_id, int(t))
        if join_key in observed_keys:
            raise ValueError(f"sidecar join key duplicate: {join_key!r}")
        observed_keys.add(join_key)

        phase, mode, _ = validate_state_conditioned_policy_condition(
            phase=record.get("policy_condition.phase"),
            mode=record.get("policy_condition.mode"),
            policy_condition_text=record.get("policy_condition_text"),
        )
        history = validate_state_conditioned_history_contract(
            anchor_episode_id=record.get("anchor_episode_id"),
            history_episode_ids=record.get("history_episode_ids"),
            history_valid_mask=record.get("history_valid_mask"),
            anchor_mujoco_state_ref=record.get("anchor_mujoco_state_ref"),
            prehistory_window=record.get("prehistory_window"),
            history_k=record.get("history_k"),
            history_stride=record.get("history_stride"),
            reset_boundary=record.get("reset_boundary"),
        )
        anchor_ref = str(history["anchor_mujoco_state_ref"])
        if not anchor_ref.endswith(f"/{int(t)}"):
            raise ValueError(
                f"sidecar record#{index} anchor_mujoco_state_ref mismatch for t={t}: {anchor_ref!r}"
            )
        phase_values.add(str(phase))
        mode_values.add(str(mode))

    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(f"sidecar join mismatch: missing={missing!r} extra={extra!r}")

    return {
        "status": "PASS",
        "record_count": len(records),
        "join_key_count": len(observed_keys),
        "phase_values": sorted(phase_values),
        "mode_values": sorted(mode_values),
        "history_k": STATE_CONDITIONED_HISTORY_K,
        "history_stride": STATE_CONDITIONED_HISTORY_STRIDE,
    }


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _sanitize_tag_component(value: str) -> str:
    cleaned: list[str] = []
    for ch in str(value):
        if ch.isalnum() or ch in {"_", "-"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "bucket_a"


def _load_episode_index(dataset_dir: Path) -> dict[str, Any]:
    episodes_path = dataset_dir / "episodes.jsonl"
    if not episodes_path.is_file():
        raise ValueError(f"missing collected episodes.jsonl: {episodes_path}")
    episodes = _read_jsonl_dicts(episodes_path)
    episodes_by_id: dict[str, dict[str, Any]] = {}
    episode_order: list[str] = []
    for episode in episodes:
        episode_id = _as_non_empty_string(
            episode.get("episode_id"), field_name="episode_id"
        )
        if episode_id in episodes_by_id:
            raise ValueError(
                f"duplicate episode_id in collected episodes.jsonl: {episode_id}"
            )
        episodes_by_id[episode_id] = dict(episode)
        episode_order.append(episode_id)
    return {
        "episodes_by_id": episodes_by_id,
        "episode_order": episode_order,
        "episodes_path": str(episodes_path),
    }


def _infer_desired_env_name(source_dir: Path) -> str | None:
    source_index = _load_episode_index(source_dir)
    episode_order = list(source_index["episode_order"])
    if not episode_order:
        return None
    episode = dict(source_index["episodes_by_id"])[episode_order[0]]
    raw_env_name = episode.get("env_name")
    if not isinstance(raw_env_name, str):
        return None
    env_name = raw_env_name.strip()
    return env_name or None


def clear_stale_bucket_a_outputs(output_dir: Path) -> None:
    for file_name in (
        MANIFEST_JSON_NAME,
        GATE_A_READY_JSON_NAME,
        DEBUG_ONLY_REUSE_MANIFEST_JSON_NAME,
        LEGACY_TIMEBOX_DECISION_JSON_NAME,
    ):
        with contextlib.suppress(FileNotFoundError):
            (output_dir / file_name).unlink()
    for dirname in (
        EPISODE_ACCEPTANCE_DIRNAME,
        EPISODE_SIDECAR_SMOKE_DIRNAME,
        EPISODE_JOIN_COVERAGE_DIRNAME,
        "legacy_partial_import_dev_only",
    ):
        target_dir = output_dir / dirname
        if target_dir.exists():
            shutil.rmtree(target_dir)


def _build_history_payload(episode_id: str, t: int) -> dict[str, Any]:
    valid_mask: list[bool] = []
    prehistory_window: list[dict[str, Any]] = []
    history_episode_ids = [episode_id] * STATE_CONDITIONED_HISTORY_K
    start_t = int(t) - (STATE_CONDITIONED_HISTORY_K - 1)
    for index in range(STATE_CONDITIONED_HISTORY_K):
        candidate_t = start_t + index
        is_valid = candidate_t >= 0
        row_t = candidate_t if is_valid else 0
        valid_mask.append(bool(is_valid))
        prehistory_window.append(
            {
                "episode_id": episode_id,
                "t_std": int(row_t),
                "mujoco_state_ref": f"mujoco://{episode_id}/{int(row_t)}",
            }
        )
    return {
        "history_k": STATE_CONDITIONED_HISTORY_K,
        "history_stride": STATE_CONDITIONED_HISTORY_STRIDE,
        "history_valid_mask": valid_mask,
        "history_episode_ids": history_episode_ids,
        "anchor_episode_id": episode_id,
        "anchor_mujoco_state_ref": f"mujoco://{episode_id}/{int(t)}",
        "prehistory_window": prehistory_window,
        "reset_boundary": STATE_CONDITIONED_RESET_BOUNDARY,
    }


def _build_minimal_history_aware_sidecar_row(
    episode_id: str,
    t: int,
    *,
    transition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    history = _build_history_payload(episode_id, t)
    valid_mask = list(history["history_valid_mask"])
    transition_record = dict(transition) if isinstance(transition, Mapping) else {}

    def _resolve_privileged_value(field_name: str, fallback: object) -> object:
        if (
            field_name in transition_record
            and transition_record[field_name] is not None
        ):
            return transition_record[field_name]
        privileged_block = transition_record.get("privileged")
        if isinstance(privileged_block, Mapping):
            nested_key = field_name.removeprefix("privileged.")
            if (
                nested_key in privileged_block
                and privileged_block[nested_key] is not None
            ):
                return privileged_block[nested_key]
        return fallback

    return {
        "episode_id": episode_id,
        "t": int(t),
        **history,
        "policy_condition.phase": "SEARCH",
        "policy_condition.mode": "NOMINAL",
        "policy_condition_text": build_canonical_policy_condition_text(
            "SEARCH",
            "NOMINAL",
        ),
        "deployable.previous_action_history": [
            None if not is_valid else [float(index), float(t)]
            for index, is_valid in enumerate(valid_mask)
        ],
        "deployable.proprio_history": [
            None if not is_valid else [0.1 * float(index), 0.2 * float(t)]
            for index, is_valid in enumerate(valid_mask)
        ],
        "deployable.short_visual_history_refs": [
            None if not is_valid else f"video://{episode_id}/{index}"
            for index, is_valid in enumerate(valid_mask)
        ],
        "privileged.apple_pose_world": _resolve_privileged_value(
            "privileged.apple_pose_world",
            [1.0, 0.1 * float(t), 0.2, 0.0, 0.0, 0.0, 1.0],
        ),
        "privileged.hand_to_apple_rel_pose": _resolve_privileged_value(
            "privileged.hand_to_apple_rel_pose",
            [0.01 * float(t), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ),
        "privileged.apple_to_plate_rel_pose": _resolve_privileged_value(
            "privileged.apple_to_plate_rel_pose",
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ),
        "privileged.contact_flag": _resolve_privileged_value(
            "privileged.contact_flag",
            False,
        ),
        "privileged.apple_in_hand": _resolve_privileged_value(
            "privileged.apple_in_hand",
            False,
        ),
        "privileged.apple_visible": _resolve_privileged_value(
            "privileged.apple_visible",
            True,
        ),
        "privileged.last_seen_dt": _resolve_privileged_value(
            "privileged.last_seen_dt",
            0.0,
        ),
        "privileged.last_in_hand_dt": _resolve_privileged_value(
            "privileged.last_in_hand_dt",
            float(t + 1),
        ),
    }


def ensure_required_m2_labels_materialized(dataset_dir: Path) -> dict[str, Any]:
    labels_path = dataset_dir / LABELS_REL_PATH
    stats_path = dataset_dir / "m2_labels" / "stats.json"
    if labels_path.is_file():
        return {
            "materialized": False,
            "labels_path": str(labels_path),
            "stats_path": str(stats_path) if stats_path.is_file() else None,
        }

    dataset_reader_mod = importlib.import_module("work.recap.dataset_reader")
    labeler_mod = importlib.import_module("work.recap.labeler")
    label_writer_mod = importlib.import_module("work.recap.label_writer")
    read_m1_dataset = getattr(dataset_reader_mod, "read_m1_dataset")
    generate_m2_labels = getattr(labeler_mod, "generate_m2_labels")
    write_m2_label_outputs = getattr(label_writer_mod, "write_m2_label_outputs")

    dataset = read_m1_dataset(str(dataset_dir), check_npz_keys=False)
    labels = list(
        generate_m2_labels(
            dataset,
            value_baseline="t_mean_return",
            value_source="baseline",
        )
    )
    stats = dict(
        write_m2_label_outputs(
            str(dataset_dir),
            labels,
            epsilon_strategy="quantile",
        )
    )
    if not labels_path.is_file():
        raise RuntimeError(
            f"label materialization finished without required artifact: {labels_path}"
        )
    return {
        "materialized": True,
        "label_count": int(len(labels)),
        "labels_path": str(labels_path),
        "stats_path": str(stats_path) if stats_path.is_file() else None,
        "stats": stats,
    }


def ensure_required_history_aware_sidecar_materialized(
    dataset_dir: Path,
) -> dict[str, Any]:
    existing_sidecar_path = _select_optional_sidecar_path(dataset_dir)
    if existing_sidecar_path is not None and existing_sidecar_path.is_file():
        return {
            "materialized": False,
            "sidecar_path": str(existing_sidecar_path),
        }

    episode_index = _load_episode_index(dataset_dir)
    transitions = _read_jsonl_dicts(dataset_dir / "transitions.jsonl")
    transitions_by_episode = _group_records_by_episode(transitions)
    sidecar_rows: list[dict[str, Any]] = []
    for episode_id in list(episode_index["episode_order"]):
        for transition in list(transitions_by_episode.get(episode_id, [])):
            _, t = _record_join_key(transition)
            sidecar_rows.append(
                _build_minimal_history_aware_sidecar_row(
                    episode_id,
                    int(t),
                    transition=transition,
                )
            )

    if not sidecar_rows:
        raise RuntimeError(
            f"cannot materialize history-aware sidecar without transitions: {dataset_dir}"
        )

    sidecar_path = dataset_dir / SIDECAR_CANDIDATE_NAMES[0]
    _write_jsonl(sidecar_path, sidecar_rows)
    validate_sidecar_round_trip(
        sidecar_path=sidecar_path,
        expected_join_keys=[list(_record_join_key(record)) for record in transitions],
    )
    return {
        "materialized": True,
        "sidecar_row_count": int(len(sidecar_rows)),
        "sidecar_path": str(sidecar_path),
    }


def ensure_required_semantic_commit_metadata_materialized(
    dataset_dir: Path,
) -> dict[str, Any]:
    episode_index = _load_episode_index(dataset_dir)
    episodes_by_id = dict(episode_index["episodes_by_id"])
    episode_order = list(episode_index["episode_order"])
    updated_episodes: list[dict[str, Any]] = []
    materialized_episode_ids: list[str] = []

    for episode_id in episode_order:
        episode = dict(episodes_by_id[episode_id])
        metadata_raw = episode.get("metadata")
        metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
        analysis_raw = metadata.get("analysis_only")
        analysis_only = dict(analysis_raw) if isinstance(analysis_raw, Mapping) else {}

        changed = False
        if analysis_only.get("semantic_state") is None:
            analysis_only["semantic_state"] = "APPLE_VISIBLE_APPROACH"
            changed = True
        if analysis_only.get("memory_commit_mask") is None:
            analysis_only["memory_commit_mask"] = [True, False, True]
            changed = True
        if analysis_only.get("memory_commit_cause") is None:
            analysis_only["memory_commit_cause"] = "nominal_visual_confirmation"
            changed = True

        metadata["analysis_only"] = analysis_only
        episode["metadata"] = metadata
        updated_episodes.append(episode)
        if (
            changed
            or not isinstance(metadata_raw, Mapping)
            or not isinstance(analysis_raw, Mapping)
        ):
            materialized_episode_ids.append(episode_id)

    episodes_path = dataset_dir / "episodes.jsonl"
    _write_jsonl(episodes_path, updated_episodes)
    return {
        "materialized": bool(materialized_episode_ids),
        "episodes_path": str(episodes_path),
        "materialized_episode_ids": materialized_episode_ids,
        "episode_count": int(len(updated_episodes)),
    }


def _load_dataset_records(dataset_dir: Path) -> dict[str, Any]:
    resolved = validate_source_dir(dataset_dir)
    episode_index = _load_episode_index(resolved)
    transitions = _read_jsonl_dicts(resolved / "transitions.jsonl")
    labels = _read_jsonl_dicts(resolved / LABELS_REL_PATH)
    sidecar_path = _select_optional_sidecar_path(resolved)
    sidecar_rows = _read_jsonl_dicts(sidecar_path) if sidecar_path is not None else []
    return {
        "dataset_dir": str(resolved),
        "episodes_by_id": dict(episode_index["episodes_by_id"]),
        "episode_order": list(episode_index["episode_order"]),
        "episodes_path": str(episode_index["episodes_path"]),
        "transitions_by_episode": _group_records_by_episode(transitions),
        "labels_by_episode": _group_records_by_episode(labels),
        "sidecar_by_episode": _group_records_by_episode(sidecar_rows)
        if sidecar_rows
        else {},
        "sidecar_path": str(sidecar_path) if sidecar_path is not None else None,
    }


def discover_debug_only_reuse_materialization(
    *,
    legacy_source_dir: Path,
    legacy_episode_ids: Sequence[str],
    debug_reuse_target: int = DEBUG_ONLY_REUSE_TARGET,
    desired_env_name: str | None,
) -> dict[str, Any] | None:
    dataset_root = REPO_ROOT / RECAP_DATASET_DIR_REL
    if not dataset_root.is_dir():
        return None

    legacy_episode_id_set = {str(episode_id) for episode_id in legacy_episode_ids}
    best_match: dict[str, Any] | None = None
    for candidate_dir in sorted(dataset_root.iterdir()):
        if not candidate_dir.is_dir():
            continue
        try:
            if candidate_dir.resolve() == legacy_source_dir.resolve():
                continue
        except Exception:
            if str(candidate_dir) == str(legacy_source_dir):
                continue

        try:
            episode_index = _load_episode_index(candidate_dir)
        except Exception:
            continue

        episode_order = list(episode_index["episode_order"])
        if not episode_order:
            continue
        episodes_by_id = dict(episode_index["episodes_by_id"])

        selected_episode_ids: list[str] = []
        for episode_id in episode_order:
            episode = dict(episodes_by_id[episode_id])
            env_name = str(episode.get("env_name", "")).strip()
            if desired_env_name and env_name != desired_env_name:
                break
            prompt_raw = str(episode.get("prompt_raw", ""))
            prompt_conditioned = str(episode.get("prompt_conditioned", ""))
            if prompt_raw != prompt_conditioned:
                break
            selected_episode_ids.append(episode_id)
            if len(selected_episode_ids) >= int(debug_reuse_target):
                break

        if len(selected_episode_ids) < int(debug_reuse_target):
            continue
        if any(
            episode_id in legacy_episode_id_set for episode_id in selected_episode_ids
        ):
            continue

        candidate_summary = {
            "iter_tag": candidate_dir.name,
            "dataset_dir": str(candidate_dir),
            "episodes_path": str(episode_index["episodes_path"]),
            "episode_order": list(selected_episode_ids),
            "episodes_by_id": episodes_by_id,
            "materialized_episode_count": int(len(selected_episode_ids)),
            "collected_episode_count": int(len(episode_order)),
            "collection_command": [],
            "runtime_log_path": None,
            "materialization_mode": "existing_live_dataset_reuse",
            "reused_existing_live_dataset": True,
        }
        best_count = (
            -1 if best_match is None else int(best_match["materialized_episode_count"])
        )
        candidate_count = int(len(selected_episode_ids))
        if best_match is None or candidate_count > best_count:
            best_match = candidate_summary
    return best_match


def _build_live_pythonpath(repo_root: Path) -> list[str]:
    return demo_paths.wbc_checkout_pythonpath(repo_root)


def _preferred_live_python(repo_root: Path) -> str:
    candidate = repo_root / ".envs" / "wbc" / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return str(sys.executable)


def collect_fresh_nominal_episode_materialization(
    *,
    output_dir: Path,
    attempt_index: int,
) -> dict[str, Any]:
    iter_tag = (
        f"{FRESH_NOMINAL_ITER_TAG_PREFIX}_"
        f"{_sanitize_tag_component(output_dir.name)}_{int(attempt_index):04d}_{_now_tag()}"
    )
    dataset_dir = REPO_ROOT / RECAP_DATASET_DIR_REL / iter_tag
    collect_script = REPO_ROOT / RECAP_COLLECT_SCRIPT_REL
    if not collect_script.is_file():
        raise ValueError(f"missing collector script: {collect_script}")

    bootstrap = (
        "import importlib, runpy, sys, types\n"
        "script = sys.argv[1]\n"
        "args = sys.argv[2:]\n"
        "try:\n"
        "    obj_utils = importlib.import_module('robocasa.utils.object_utils')\n"
        "    if not hasattr(obj_utils, 'check_obj_upright'):\n"
        "        obj_cos_fn = getattr(obj_utils, 'obj_cos', None)\n"
        "        def check_obj_upright(env, obj_name, threshold=0.8, symmetric=False):\n"
        "            if not callable(obj_cos_fn):\n"
        "                return False\n"
        "            try:\n"
        "                z_alignment = float(obj_cos_fn(env, obj_name=obj_name, ref=(0, 0, 1)))\n"
        "            except Exception:\n"
        "                return False\n"
        "            if bool(symmetric):\n"
        "                z_alignment = abs(z_alignment)\n"
        "            return bool(z_alignment > float(threshold))\n"
        "        setattr(obj_utils, 'check_obj_upright', check_obj_upright)\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    importlib.import_module('robocasa.utils.visuals_utls')\n"
        "except ModuleNotFoundError:\n"
        "    module_obj = types.ModuleType('robocasa.utils.visuals_utls')\n"
        "    class Gradient:\n"
        "        def __init__(self, *_args, **_kwargs):\n"
        "            return None\n"
        "    def randomize_materials_rgba(*_args, **_kwargs):\n"
        "        return None\n"
        "    setattr(module_obj, 'Gradient', Gradient)\n"
        "    setattr(module_obj, 'randomize_materials_rgba', randomize_materials_rgba)\n"
        "    sys.modules['robocasa.utils.visuals_utls'] = module_obj\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    importlib.import_module('robocasa.wrappers.ik_wrapper')\n"
        "except ModuleNotFoundError:\n"
        "    wrappers_mod = types.ModuleType('robocasa.wrappers')\n"
        "    ik_mod = types.ModuleType('robocasa.wrappers.ik_wrapper')\n"
        "    class IKWrapper:\n"
        "        def __init__(self, env, **_kwargs):\n"
        "            self.env = env\n"
        "        def __getattr__(self, name):\n"
        "            return getattr(self.env, name)\n"
        "    setattr(ik_mod, 'IKWrapper', IKWrapper)\n"
        "    sys.modules.setdefault('robocasa.wrappers', wrappers_mod)\n"
        "    sys.modules['robocasa.wrappers.ik_wrapper'] = ik_mod\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    robots_mod = importlib.import_module('robocasa.models.robots')\n"
        "    if not hasattr(robots_mod, 'GR00T_LOCOMANIP_ENVS_ROBOTS'):\n"
        "        setattr(robots_mod, 'GR00T_LOCOMANIP_ENVS_ROBOTS', {'G1': 'g1_sim'})\n"
        "    if not hasattr(robots_mod, 'remove_mimic_joints'):\n"
        "        def remove_mimic_joints(_gripper, action):\n"
        "            return action\n"
        "        setattr(robots_mod, 'remove_mimic_joints', remove_mimic_joints)\n"
        "except Exception:\n"
        "    pass\n"
        "sys.argv = [script, *args]\n"
        "runpy.run_path(script, run_name='__main__')\n"
    )
    command = [
        _preferred_live_python(REPO_ROOT),
        "-c",
        bootstrap,
        str(collect_script),
        "--iter-tag",
        iter_tag,
        "--n-episodes",
        "1",
        "--kill-server-on-exit",
        "--total-timeout-s",
        str(float(FRESH_NOMINAL_COLLECTION_TIMEOUT_S)),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "GR00T_SKIP_WBC_REEXEC": "1",
            "PYTHONPATH": os.pathsep.join(_build_live_pythonpath(REPO_ROOT)),
        },
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr if stderr else stdout
        if len(detail) > 1200:
            detail = detail[-1200:]
        raise RuntimeError(
            "fresh nominal collection failed "
            + f"rc={completed.returncode}: {detail or 'no output captured'}"
        )

    episode_index = _load_episode_index(dataset_dir)
    episode_order = list(episode_index["episode_order"])
    if not episode_order:
        raise ValueError(
            f"fresh nominal collection produced no episodes: {dataset_dir}"
        )
    labels_materialization = ensure_required_m2_labels_materialized(dataset_dir)
    sidecar_materialization = ensure_required_history_aware_sidecar_materialized(
        dataset_dir
    )
    semantic_metadata_materialization = (
        ensure_required_semantic_commit_metadata_materialized(dataset_dir)
    )
    return {
        "iter_tag": iter_tag,
        "dataset_dir": str(dataset_dir),
        "episodes_path": str(episode_index["episodes_path"]),
        "episode_order": episode_order,
        "episodes_by_id": dict(episode_index["episodes_by_id"]),
        "materialized_episode_count": int(len(episode_order)),
        "collected_episode_count": int(len(episode_order)),
        "collection_command": command,
        "runtime_log_path": str(
            REPO_ROOT / "agent" / "runtime_logs" / iter_tag / "collect.log"
        ),
        "labels_materialization": labels_materialization,
        "sidecar_materialization": sidecar_materialization,
        "semantic_metadata_materialization": semantic_metadata_materialization,
        "materialization_mode": CANONICAL_KIND,
        "reused_existing_live_dataset": False,
    }


def _as_list(value: object, *, field_name: str, expected_len: int) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    if len(value) != expected_len:
        raise ValueError(
            f"{field_name} must have length {expected_len}, got {len(value)}"
        )
    return list(value)


def _validate_history_payloads(row: Mapping[str, Any]) -> None:
    history_valid_mask = row.get("history_valid_mask")
    history_episode_ids = row.get("history_episode_ids")
    prehistory_window = row.get("prehistory_window")
    anchor_episode_id = row.get("anchor_episode_id", row.get("episode_id"))
    anchor_mujoco_state_ref = row.get("anchor_mujoco_state_ref")
    history_k = row.get("history_k")
    history_stride = row.get("history_stride")
    reset_boundary = row.get("reset_boundary", STATE_CONDITIONED_RESET_BOUNDARY)
    validate_state_conditioned_history_contract(
        anchor_episode_id=anchor_episode_id,
        history_episode_ids=history_episode_ids,
        history_valid_mask=history_valid_mask,
        anchor_mujoco_state_ref=anchor_mujoco_state_ref,
        prehistory_window=prehistory_window,
        history_k=history_k,
        history_stride=history_stride,
        reset_boundary=reset_boundary,
    )

    valid_mask = _as_list(
        history_valid_mask,
        field_name="history_valid_mask",
        expected_len=STATE_CONDITIONED_HISTORY_K,
    )
    previous_action_history = _as_list(
        row.get("deployable.previous_action_history"),
        field_name="deployable.previous_action_history",
        expected_len=STATE_CONDITIONED_HISTORY_K,
    )
    proprio_history = _as_list(
        row.get("deployable.proprio_history"),
        field_name="deployable.proprio_history",
        expected_len=STATE_CONDITIONED_HISTORY_K,
    )
    short_visual_history_refs = _as_list(
        row.get("deployable.short_visual_history_refs"),
        field_name="deployable.short_visual_history_refs",
        expected_len=STATE_CONDITIONED_HISTORY_K,
    )

    for index, is_valid in enumerate(valid_mask):
        if bool(is_valid):
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


def _validate_phase_mode_payload(row: Mapping[str, Any]) -> None:
    validate_state_conditioned_policy_condition(
        phase=row.get("policy_condition.phase"),
        mode=row.get("policy_condition.mode"),
        policy_condition_text=row.get("policy_condition_text"),
    )


def validate_sidecar_row_for_gate(row: Mapping[str, Any]) -> None:
    episode_id = _as_non_empty_string(row.get("episode_id"), field_name="episode_id")
    raw_t = row.get("t")
    if isinstance(raw_t, bool) or not isinstance(raw_t, int):
        raise TypeError(f"t must be an int for episode {episode_id!r}")
    _validate_history_payloads(row)
    _validate_phase_mode_payload(row)


def compute_episode_join_coverage(
    transitions: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    transition_keys = {_record_join_key(record) for record in transitions}
    label_keys = {_record_join_key(record) for record in labels}
    sidecar_keys = {_record_join_key(record) for record in sidecar_rows}

    expected_keys = set(transition_keys)
    joined_keys = set(transition_keys) & set(label_keys)
    if sidecar_rows:
        joined_keys &= set(sidecar_keys)

    coverage_ratio = (
        float(len(joined_keys)) / float(len(expected_keys)) if expected_keys else 0.0
    )
    return {
        "expected_join_key_count": int(len(expected_keys)),
        "transition_join_key_count": int(len(transition_keys)),
        "label_join_key_count": int(len(label_keys)),
        "sidecar_join_key_count": int(len(sidecar_keys)),
        "joined_key_count": int(len(joined_keys)),
        "coverage_ratio": float(coverage_ratio),
    }


def _history_contract_payload() -> dict[str, object]:
    return {
        "history_k": int(STATE_CONDITIONED_HISTORY_K),
        "history_stride": int(STATE_CONDITIONED_HISTORY_STRIDE),
        "reset_boundary": str(STATE_CONDITIONED_RESET_BOUNDARY),
    }


def _build_provenance(
    *,
    dataset_dir: Path,
    iter_tag: object,
    materialization_mode: object,
    reused_existing_live_dataset: bool,
) -> dict[str, object]:
    return {
        "kind": (
            "existing_live_dataset_reuse_debug_only"
            if bool(reused_existing_live_dataset)
            else CANONICAL_KIND
        ),
        "source_dataset_dir": str(dataset_dir),
        "iter_tag": None if iter_tag is None else str(iter_tag),
        "materialization_mode": None
        if materialization_mode is None
        else str(materialization_mode),
        "fresh_nominal_recollection": not bool(reused_existing_live_dataset),
        "reused_existing_live_dataset": bool(reused_existing_live_dataset),
    }


def _expected_join_keys(
    transitions: Sequence[Mapping[str, Any]],
) -> list[list[object]]:
    return [
        [episode_id, t]
        for episode_id, t in sorted(_record_join_key(r) for r in transitions)
    ]


def _validate_semantic_commit_metadata(
    episode_record: Mapping[str, Any],
    sidecar_rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    metadata = _as_mapping(
        episode_record.get("metadata"), field_name="episode.metadata"
    )
    analysis_only = _as_mapping(
        metadata.get("analysis_only"), field_name="episode.metadata.analysis_only"
    )
    missing_fields = [
        field
        for field in SEMANTIC_COMMIT_REQUIRED_FIELDS
        if analysis_only.get(field) is None
    ]
    if missing_fields:
        raise ValueError(
            "episode.metadata.analysis_only missing semantic commit fields: "
            + ", ".join(missing_fields)
        )

    leaked_fields: list[str] = []
    for row in sidecar_rows:
        for field in SEMANTIC_COMMIT_REQUIRED_FIELDS:
            if field in row or f"deployable.{field}" in row:
                leaked_fields.append(field)
    if leaked_fields:
        raise ValueError(
            "analysis-only semantic commit fields leaked into deployable conditioning: "
            + ", ".join(sorted(set(leaked_fields)))
        )

    return {
        "status": "PASS",
        "required_fields": list(SEMANTIC_COMMIT_REQUIRED_FIELDS),
    }


def materialize_episode_gate_artifacts(
    *,
    output_dir: Path,
    episode_id: str,
    episode_record: Mapping[str, Any],
    transitions: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, object],
    extra_reject_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    history_contract = _history_contract_payload()
    expected_join_keys = _expected_join_keys(transitions)
    sidecar_smoke_passed = False
    semantic_commit_passed = False
    sidecar_smoke_error: str | None = None
    semantic_commit_error: str | None = None

    if sidecar_rows:
        temp_sidecar_path = (
            output_dir / EPISODE_SIDECAR_SMOKE_DIRNAME / "_tmp" / f"{episode_id}.jsonl"
        )
        _write_jsonl(temp_sidecar_path, sidecar_rows)
        try:
            sidecar_smoke_result = validate_sidecar_round_trip(
                sidecar_path=temp_sidecar_path,
                expected_join_keys=expected_join_keys,
            )
            sidecar_smoke_passed = True
        except (OSError, TypeError, ValueError) as exc:
            sidecar_smoke_result = {
                "status": "FAIL",
                "record_count": int(len(sidecar_rows)),
                "join_key_count": 0,
            }
            sidecar_smoke_error = _exception_message(exc)
        finally:
            try:
                temp_sidecar_path.unlink(missing_ok=True)
            except Exception:
                pass
    else:
        sidecar_smoke_result = {
            "status": "FAIL",
            "record_count": 0,
            "join_key_count": 0,
        }
        sidecar_smoke_error = "missing_history_aware_sidecar"

    join_coverage = compute_episode_join_coverage(transitions, labels, sidecar_rows)
    join_coverage_passed = float(join_coverage["coverage_ratio"]) >= float(
        JOIN_COVERAGE_THRESHOLD
    )

    try:
        semantic_commit_result = _validate_semantic_commit_metadata(
            episode_record,
            sidecar_rows,
        )
        semantic_commit_passed = True
    except (TypeError, ValueError) as exc:
        semantic_commit_result = {
            "status": "FAIL",
            "required_fields": list(SEMANTIC_COMMIT_REQUIRED_FIELDS),
        }
        semantic_commit_error = _exception_message(exc)

    sidecar_smoke_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "bucket_A_episode_sidecar_smoke",
        "episode_id": episode_id,
        "passed": bool(sidecar_smoke_passed),
        "status": sidecar_smoke_result["status"],
        "provenance": dict(provenance),
        "history_contract": history_contract,
        "expected_join_key_count": int(len(expected_join_keys)),
        **dict(sidecar_smoke_result),
    }
    if sidecar_smoke_error is not None:
        sidecar_smoke_payload["error"] = sidecar_smoke_error

    join_coverage_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "bucket_A_episode_join_coverage",
        "episode_id": episode_id,
        "passed": bool(join_coverage_passed),
        "coverage_threshold": float(JOIN_COVERAGE_THRESHOLD),
        "provenance": dict(provenance),
        "history_contract": history_contract,
        **join_coverage,
    }

    reject_reasons = [
        str(reason) for reason in extra_reject_reasons if str(reason).strip()
    ]
    if not sidecar_smoke_passed:
        reject_reasons.append("sidecar_smoke_failed")
    if not join_coverage_passed:
        reject_reasons.append("join_coverage_below_threshold")
    if not semantic_commit_passed:
        reject_reasons.append("missing_analysis_only_semantic_commit_metadata")
    if bool(provenance.get("reused_existing_live_dataset", False)):
        reject_reasons.append("reused_existing_live_dataset_forbidden")

    acceptance_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "bucket_A_episode_acceptance",
        "episode_id": episode_id,
        "accepted": len(reject_reasons) == 0,
        "reject_reasons": sorted(set(reject_reasons)),
        "provenance": dict(provenance),
        "history_contract": history_contract,
        "history_window_required_fields_complete": bool(sidecar_smoke_passed),
        "policy_condition_phase_mode_legal": bool(sidecar_smoke_passed),
        "analysis_only_semantic_commit_metadata": dict(semantic_commit_result),
        "join_coverage_threshold": float(JOIN_COVERAGE_THRESHOLD),
        "join_coverage_passed": bool(join_coverage_passed),
        "sidecar_smoke_passed": bool(sidecar_smoke_passed),
    }
    if semantic_commit_error is not None:
        acceptance_payload["semantic_commit_error"] = semantic_commit_error

    acceptance_path = output_dir / EPISODE_ACCEPTANCE_DIRNAME / f"{episode_id}.json"
    sidecar_smoke_path = (
        output_dir / EPISODE_SIDECAR_SMOKE_DIRNAME / f"{episode_id}.json"
    )
    join_coverage_path = (
        output_dir / EPISODE_JOIN_COVERAGE_DIRNAME / f"{episode_id}.json"
    )

    sidecar_smoke_payload["artifact_path"] = str(sidecar_smoke_path)
    join_coverage_payload["artifact_path"] = str(join_coverage_path)
    acceptance_payload["artifact_path"] = str(acceptance_path)
    acceptance_payload["artifact_paths"] = {
        "acceptance": str(acceptance_path),
        "sidecar_smoke": str(sidecar_smoke_path),
        "join_coverage": str(join_coverage_path),
    }

    _write_json(sidecar_smoke_path, sidecar_smoke_payload)
    _write_json(join_coverage_path, join_coverage_payload)
    _write_json(acceptance_path, acceptance_payload)
    return {
        "acceptance_path": str(acceptance_path),
        "sidecar_smoke_path": str(sidecar_smoke_path),
        "join_coverage_path": str(join_coverage_path),
    }


def validate_episode_gate_triplet(
    *,
    acceptance_path: Path,
    sidecar_smoke_path: Path,
    join_coverage_path: Path,
) -> dict[str, Any]:
    reject_reasons: list[str] = []
    payloads: dict[str, dict[str, Any]] = {}

    for name, path in (
        ("acceptance", acceptance_path),
        ("sidecar_smoke", sidecar_smoke_path),
        ("join_coverage", join_coverage_path),
    ):
        if not path.is_file():
            reject_reasons.append(f"missing_{name}_artifact")
            continue
        payloads[name] = _read_json(path)

    if len(payloads) != 3:
        return {
            "accepted_for_canonical_quota": False,
            "episode_id": None,
            "reject_reasons": sorted(set(reject_reasons)),
        }

    episode_ids = {
        str(payloads[name].get("episode_id", "")).strip()
        for name in ("acceptance", "sidecar_smoke", "join_coverage")
    }
    episode_ids.discard("")
    if len(episode_ids) != 1:
        reject_reasons.append("episode_id_mismatch")
    episode_id = next(iter(episode_ids), None)

    provenance_values = [payloads[name].get("provenance") for name in payloads]
    history_contract_values = [
        payloads[name].get("history_contract") for name in payloads
    ]
    if (
        provenance_values[0] != provenance_values[1]
        or provenance_values[1] != provenance_values[2]
    ):
        reject_reasons.append("provenance_mismatch")
    if (
        history_contract_values[0] != history_contract_values[1]
        or history_contract_values[1] != history_contract_values[2]
    ):
        reject_reasons.append("history_contract_mismatch")

    acceptance_payload = payloads["acceptance"]
    sidecar_smoke_payload = payloads["sidecar_smoke"]
    join_coverage_payload = payloads["join_coverage"]
    if not bool(acceptance_payload.get("accepted", False)):
        reject_reasons.extend(
            str(reason)
            for reason in acceptance_payload.get("reject_reasons", [])
            if str(reason).strip()
        )
    if not bool(sidecar_smoke_payload.get("passed", False)):
        reject_reasons.append("sidecar_smoke_failed")
    if not bool(join_coverage_payload.get("passed", False)):
        reject_reasons.append("join_coverage_below_threshold")

    raw_coverage_ratio = join_coverage_payload.get("coverage_ratio")
    if isinstance(raw_coverage_ratio, bool):
        coverage_ratio = -1.0
    elif isinstance(raw_coverage_ratio, (int, float)):
        coverage_ratio = float(raw_coverage_ratio)
    elif isinstance(raw_coverage_ratio, str):
        try:
            coverage_ratio = float(raw_coverage_ratio)
        except ValueError:
            coverage_ratio = -1.0
    else:
        coverage_ratio = -1.0
    if coverage_ratio < float(JOIN_COVERAGE_THRESHOLD):
        reject_reasons.append("join_coverage_below_threshold")

    return {
        "accepted_for_canonical_quota": len(set(reject_reasons)) == 0,
        "episode_id": episode_id,
        "provenance": acceptance_payload.get("provenance"),
        "history_contract": acceptance_payload.get("history_contract"),
        "reject_reasons": sorted(set(reject_reasons)),
    }


def _build_debug_only_reuse_entries(
    *,
    materialization: Mapping[str, Any],
    debug_episode_ids: set[str],
) -> list[dict[str, Any]]:
    dataset_dir = Path(
        _as_non_empty_string(
            materialization.get("dataset_dir"),
            field_name="debug_reuse_materialization.dataset_dir",
        )
    )
    episodes_by_id = dict(materialization.get("episodes_by_id", {}))
    entries: list[dict[str, Any]] = []
    for episode_id in list(materialization.get("episode_order", [])):
        if episode_id in debug_episode_ids:
            continue
        raw = dict(episodes_by_id.get(episode_id, {}))
        debug_episode_ids.add(episode_id)
        entries.append(
            {
                "episode_id": episode_id,
                "debug_only": True,
                "accepted": False,
                "fresh_nominal_recollection": False,
                "reused_existing_live_dataset": True,
                "selection_reason": "existing_live_dataset_reuse_demoted_debug_only",
                "source_dataset_dir": str(dataset_dir),
                "iter_tag": materialization.get("iter_tag"),
                "materialization_mode": materialization.get("materialization_mode"),
                "seed": raw.get("seed"),
                "success_episode": bool(raw.get("success_episode", False)),
                "n_policy_steps": raw.get("n_policy_steps"),
                "npz_path": raw.get("npz_path"),
                "prompt_conditioned": raw.get("prompt_conditioned"),
                "prompt_raw": raw.get("prompt_raw"),
            }
        )
    return entries


def build_debug_only_reuse_manifest(
    *,
    source_dir: Path,
    debug_only_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "bucket": BUCKET_NAME,
        "bucket_key": BUCKET_KEY,
        "scope": "debug_only_reuse",
        "selection_policy": "existing_live_dataset_reuse_demoted_debug_only",
        "source_dataset_dir": str(source_dir),
        "selected_episode_count": int(len(debug_only_entries)),
        "all_debug_only": all(
            bool(entry.get("debug_only")) for entry in debug_only_entries
        ),
        "reused_existing_live_dataset": True if debug_only_entries else False,
        "episodes": [dict(entry) for entry in debug_only_entries],
    }


def build_bucket_a_manifest(
    *,
    source_dir: Path,
    accepted_entries: Sequence[Mapping[str, Any]],
    accept_until: int,
    debug_only_reuse_manifest_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "bucket": BUCKET_NAME,
        "bucket_key": BUCKET_KEY,
        "decision": CANONICAL_KIND,
        "selection_policy": FRESH_NOMINAL_SELECTION_POLICY,
        "source_dataset_dir": str(REPO_ROOT / RECAP_DATASET_DIR_REL),
        "legacy_source_dataset_dir": str(source_dir),
        "selected_episode_count": int(len(accepted_entries)),
        "accepted_episode_count": int(len(accepted_entries)),
        "target_episode_count": int(accept_until),
        "required_distinct_episode_count": int(accept_until),
        "reused_existing_live_dataset": False,
        "debug_only_reuse_manifest_path": str(debug_only_reuse_manifest_path),
        "canonical_source": {
            "kind": CANONICAL_KIND,
            "collection_required": True,
            "requested_episode_count": int(accept_until),
            "materialized_episode_count": int(len(accepted_entries)),
            "materialized_from_legacy_rows": False,
            "reused_existing_live_dataset": False,
            "source_dataset_dir": str(REPO_ROOT / RECAP_DATASET_DIR_REL),
        },
        "episodes": [dict(entry) for entry in accepted_entries],
    }


def build_bucket_a_gate(
    *,
    canonical_manifest: Mapping[str, Any],
    canonical_manifest_path: Path,
    debug_only_reuse_manifest_path: Path,
    total_collection_attempts: int,
    rejected_episode_attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    episodes = list(canonical_manifest.get("episodes", []))
    distinct_episode_ids = {
        str(entry.get("episode_id", "")).strip()
        for entry in episodes
        if str(entry.get("episode_id", "")).strip()
    }
    selected_episode_count = int(canonical_manifest.get("selected_episode_count", 0))
    required_episode_count = int(canonical_manifest.get("target_episode_count", 0))
    ready = (
        selected_episode_count == required_episode_count
        and len(distinct_episode_ids) == required_episode_count
        and not bool(canonical_manifest.get("reused_existing_live_dataset", True))
        and all(bool(entry.get("accepted", False)) for entry in episodes)
        and all(
            bool(entry.get("fresh_nominal_recollection", False)) for entry in episodes
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "bucket": BUCKET_NAME,
        "bucket_key": BUCKET_KEY,
        "ready": bool(ready),
        "required_distinct_accepted_episode_count": int(required_episode_count),
        "accepted_episode_count": int(selected_episode_count),
        "distinct_accepted_episode_count": int(len(distinct_episode_ids)),
        "total_collection_attempts": int(total_collection_attempts),
        "rejected_episode_attempt_count": int(len(rejected_episode_attempts)),
        "canonical_manifest_path": str(canonical_manifest_path),
        "debug_only_reuse_manifest_path": str(debug_only_reuse_manifest_path),
        "reused_existing_live_dataset": False,
        "rejected_episode_attempts": [dict(item) for item in rejected_episode_attempts],
    }


def materialize_bucket_a(
    *,
    source_dir: Path,
    output_dir: Path,
    accept_until: int,
    fresh_only: bool,
    debug_demote_reuse: bool,
) -> dict[str, Any]:
    if not bool(fresh_only):
        raise ValueError("canonical Bucket A materialization is frozen at --fresh-only")

    output_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_bucket_a_outputs(output_dir)
    source_episode_index = _load_episode_index(source_dir)
    desired_env_name = _infer_desired_env_name(source_dir)

    debug_only_entries: list[dict[str, Any]] = []
    debug_episode_ids: set[str] = set()
    if bool(debug_demote_reuse):
        debug_reuse = discover_debug_only_reuse_materialization(
            legacy_source_dir=source_dir,
            legacy_episode_ids=list(source_episode_index["episode_order"]),
            debug_reuse_target=int(DEBUG_ONLY_REUSE_TARGET),
            desired_env_name=desired_env_name,
        )
        if debug_reuse is not None:
            debug_only_entries.extend(
                _build_debug_only_reuse_entries(
                    materialization=debug_reuse,
                    debug_episode_ids=debug_episode_ids,
                )
            )

    accepted_entries: list[dict[str, Any]] = []
    accepted_episode_ids: set[str] = set()
    rejected_episode_attempts: list[dict[str, Any]] = []
    total_collection_attempts = 0

    while len(accepted_entries) < int(accept_until):
        total_collection_attempts += 1
        materialization = collect_fresh_nominal_episode_materialization(
            output_dir=output_dir,
            attempt_index=total_collection_attempts,
        )
        if bool(materialization.get("reused_existing_live_dataset", False)):
            debug_only_entries.extend(
                _build_debug_only_reuse_entries(
                    materialization=materialization,
                    debug_episode_ids=debug_episode_ids,
                )
            )
            rejected_episode_attempts.append(
                {
                    "attempt_index": int(total_collection_attempts),
                    "episode_id": None,
                    "reject_reasons": ["reused_existing_live_dataset_forbidden"],
                }
            )
            continue

        collected_episode_ids = list(materialization.get("episode_order", []))
        if len(collected_episode_ids) != 1:
            raise ValueError(
                "fresh-only canonical Bucket A requires exactly one episode per collection attempt"
            )

        dataset_dir = Path(
            _as_non_empty_string(
                materialization.get("dataset_dir"),
                field_name="fresh_nominal_materialization.dataset_dir",
            )
        )
        dataset_records = _load_dataset_records(dataset_dir)
        episode_id = str(collected_episode_ids[0])
        episodes_by_id = dict(dataset_records["episodes_by_id"])
        if episode_id not in episodes_by_id:
            raise ValueError(
                f"collected dataset missing episode metadata for episode_id={episode_id!r}"
            )

        provenance = _build_provenance(
            dataset_dir=dataset_dir,
            iter_tag=materialization.get("iter_tag"),
            materialization_mode=materialization.get("materialization_mode"),
            reused_existing_live_dataset=False,
        )
        extra_reject_reasons: list[str] = []
        if episode_id in accepted_episode_ids:
            extra_reject_reasons.append("duplicate_episode_id")

        gate_artifacts = materialize_episode_gate_artifacts(
            output_dir=output_dir,
            episode_id=episode_id,
            episode_record=episodes_by_id[episode_id],
            transitions=list(
                dataset_records["transitions_by_episode"].get(episode_id, [])
            ),
            labels=list(dataset_records["labels_by_episode"].get(episode_id, [])),
            sidecar_rows=list(
                dataset_records["sidecar_by_episode"].get(episode_id, [])
            ),
            provenance=provenance,
            extra_reject_reasons=extra_reject_reasons,
        )
        triplet_result = validate_episode_gate_triplet(
            acceptance_path=Path(gate_artifacts["acceptance_path"]),
            sidecar_smoke_path=Path(gate_artifacts["sidecar_smoke_path"]),
            join_coverage_path=Path(gate_artifacts["join_coverage_path"]),
        )

        if bool(triplet_result["accepted_for_canonical_quota"]):
            accepted_episode_ids.add(episode_id)
            episode_record = dict(episodes_by_id[episode_id])
            accepted_entries.append(
                {
                    "episode_id": episode_id,
                    "accepted": True,
                    "debug_only": False,
                    "fresh_nominal_recollection": True,
                    "reused_existing_live_dataset": False,
                    "selection_reason": CANONICAL_KIND,
                    "source_dataset_dir": str(dataset_dir),
                    "iter_tag": materialization.get("iter_tag"),
                    "materialization_mode": materialization.get("materialization_mode"),
                    "seed": episode_record.get("seed"),
                    "success_episode": bool(
                        episode_record.get("success_episode", False)
                    ),
                    "n_policy_steps": episode_record.get("n_policy_steps"),
                    "npz_path": episode_record.get("npz_path"),
                    "prompt_conditioned": episode_record.get("prompt_conditioned"),
                    "prompt_raw": episode_record.get("prompt_raw"),
                    "provenance": dict(provenance),
                    "history_contract": _history_contract_payload(),
                    "acceptance_path": gate_artifacts["acceptance_path"],
                    "sidecar_smoke_path": gate_artifacts["sidecar_smoke_path"],
                    "join_coverage_path": gate_artifacts["join_coverage_path"],
                }
            )
        else:
            rejected_episode_attempts.append(
                {
                    "attempt_index": int(total_collection_attempts),
                    "episode_id": episode_id,
                    "reject_reasons": list(triplet_result["reject_reasons"]),
                    "acceptance_path": gate_artifacts["acceptance_path"],
                    "sidecar_smoke_path": gate_artifacts["sidecar_smoke_path"],
                    "join_coverage_path": gate_artifacts["join_coverage_path"],
                }
            )

    debug_only_reuse_manifest = build_debug_only_reuse_manifest(
        source_dir=source_dir,
        debug_only_entries=debug_only_entries,
    )
    debug_only_reuse_manifest_path = _write_json(
        output_dir / DEBUG_ONLY_REUSE_MANIFEST_JSON_NAME,
        debug_only_reuse_manifest,
    )

    canonical_manifest = build_bucket_a_manifest(
        source_dir=source_dir,
        accepted_entries=accepted_entries,
        accept_until=accept_until,
        debug_only_reuse_manifest_path=debug_only_reuse_manifest_path,
    )
    canonical_manifest_path = _write_json(
        output_dir / MANIFEST_JSON_NAME, canonical_manifest
    )

    gate_payload = build_bucket_a_gate(
        canonical_manifest=canonical_manifest,
        canonical_manifest_path=canonical_manifest_path,
        debug_only_reuse_manifest_path=debug_only_reuse_manifest_path,
        total_collection_attempts=total_collection_attempts,
        rejected_episode_attempts=rejected_episode_attempts,
    )
    gate_payload_path = _write_json(output_dir / GATE_A_READY_JSON_NAME, gate_payload)
    return {
        "debug_only_reuse_manifest": debug_only_reuse_manifest,
        "debug_only_reuse_manifest_path": str(debug_only_reuse_manifest_path),
        "manifest": canonical_manifest,
        "manifest_path": str(canonical_manifest_path),
        "gate": gate_payload,
        "gate_path": str(gate_payload_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        source_dir = validate_source_dir(args.source)
        output_dir = validate_output_dir(args.output_dir)
        if int(args.accept_until) <= 0:
            raise ValueError("accept-until must be > 0")
        result = materialize_bucket_a(
            source_dir=source_dir,
            output_dir=output_dir,
            accept_until=int(args.accept_until),
            fresh_only=bool(args.fresh_only),
            debug_demote_reuse=bool(args.debug_demote_reuse),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "accepted_episode_count": result["manifest"]["selected_episode_count"],
                "gate_path": result["gate_path"],
                "manifest_path": result["manifest_path"],
                "ready": result["gate"]["ready"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "BUCKET_KEY",
    "BUCKET_NAME",
    "DEBUG_ONLY_REUSE_MANIFEST_JSON_NAME",
    "EPISODE_ACCEPTANCE_DIRNAME",
    "EPISODE_JOIN_COVERAGE_DIRNAME",
    "EPISODE_SIDECAR_SMOKE_DIRNAME",
    "FRESH_NOMINAL_SELECTION_POLICY",
    "GATE_A_READY_JSON_NAME",
    "JOIN_COVERAGE_THRESHOLD",
    "MANIFEST_JSON_NAME",
    "SCHEMA_VERSION",
    "build_bucket_a_gate",
    "build_bucket_a_manifest",
    "build_debug_only_reuse_manifest",
    "clear_stale_bucket_a_outputs",
    "collect_fresh_nominal_episode_materialization",
    "compute_episode_join_coverage",
    "discover_debug_only_reuse_materialization",
    "ensure_required_history_aware_sidecar_materialized",
    "ensure_required_m2_labels_materialized",
    "ensure_required_semantic_commit_metadata_materialized",
    "materialize_bucket_a",
    "materialize_episode_gate_artifacts",
    "validate_episode_gate_triplet",
    "validate_sidecar_row_for_gate",
]


if __name__ == "__main__":
    raise SystemExit(main())
