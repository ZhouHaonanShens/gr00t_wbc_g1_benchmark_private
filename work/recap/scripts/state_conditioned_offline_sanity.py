#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_DIR = Path("agent/artifacts/state_conditioned_materialization/sanity")
REPORT_JSON_NAME = "offline_sanity_report.json"
SCHEMA_VERSION = "g1_state_conditioned_offline_sanity_v1"
ITER_TAG = "state_conditioned_offline_sanity_fixture"
TASK_TEXT = "pick up the apple and place it on the plate"
T_ACTION = 2
TIMESTAMP_DT_S = 0.05


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap import advantage
from work.recap import dataset_reader, labeler
from work.recap.lerobot_export import dataset_export as lerobot_v2_export


CHECK_ORDER: tuple[str, ...] = (
    "sidecar_round_trip",
    "history_window_padding_reset_boundary",
    "phase_mode_parsing",
    "label_round_trip",
    "exporter_round_trip",
)


class OfflineSanityError(RuntimeError):
    stage: str

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = str(stage)


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    return cast(Path, state_conditioned_bucket_a_import._write_json(path, payload))


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    return cast(Path, state_conditioned_bucket_a_import._write_jsonl(path, records))


def _read_json(path: Path) -> dict[str, Any]:
    return state_conditioned_bucket_a_import._read_json(path)


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    return state_conditioned_bucket_a_import._read_jsonl_dicts(path)


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True)


def _not_run_check(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": None,
        "status": "NOT_RUN",
    }


def _pass_check(name: str, **details: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": True,
        "status": "PASS",
        **details,
    }


def _fail_check(name: str, error: str, **details: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": False,
        "status": "FAIL",
        "error": str(error),
        **details,
    }


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="state_conditioned_offline_sanity.py",
        description=(
            "Task 3 offline-only sanity for state-conditioned sidecar/history/label/exporter contracts with machine-readable JSON output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _build_parser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives offline_sanity_report.json.",
    )
    return parser


def _step_specs() -> list[dict[str, Any]]:
    return [
        {
            "episode_id": "offline_episode_000",
            "rewards": [1.0, 0.5, 0.0],
            "phases": ["search", "approach", "grasp"],
            "modes": ["nominal", "nominal", "recovery"],
        },
        {
            "episode_id": "offline_episode_001",
            "rewards": [0.0, -0.5, -1.0],
            "phases": ["verify_hold", "transport", "place"],
            "modes": ["nominal", "recovery", "nominal"],
        },
    ]


def _n_exec_for_step(t: int) -> int:
    return 1 if int(t) % 2 == 0 else 2


def _timestamp_for_step(t: int) -> float:
    return round(float(t) * float(TIMESTAMP_DT_S), 3)


def _history_payload(episode_id: str, t: int) -> dict[str, Any]:
    history_k = int(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K)
    history_stride = int(
        state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_STRIDE
    )
    valid_mask: list[bool] = []
    prehistory_window: list[dict[str, Any]] = []
    history_episode_ids: list[str] = []
    history_t_std_indices: list[int] = []
    history_t_raw_indices: list[int] = []
    history_timestamp_s: list[float] = []
    previous_action_history: list[list[float] | None] = []
    proprio_history: list[list[float] | None] = []
    short_visual_history_refs: list[str | None] = []
    start_t = int(t) - (history_k - 1)
    for index in range(history_k):
        candidate_t = start_t + index * history_stride
        is_valid = candidate_t >= 0
        normalized_t = candidate_t if is_valid else 0
        timestamp_s = _timestamp_for_step(normalized_t)
        valid_mask.append(bool(is_valid))
        history_episode_ids.append(episode_id)
        history_t_std_indices.append(int(normalized_t))
        history_t_raw_indices.append(int(normalized_t))
        history_timestamp_s.append(float(timestamp_s))
        prehistory_window.append(
            {
                "episode_id": episode_id,
                "t_std": int(normalized_t),
                "mujoco_state_ref": f"mujoco://{episode_id}/{int(normalized_t)}",
            }
        )
        if is_valid:
            previous_action_history.append(
                [float(index), round(0.1 * float(normalized_t), 3)]
            )
            proprio_history.append(
                [round(0.01 * float(index), 3), round(0.02 * float(normalized_t), 3)]
            )
            short_visual_history_refs.append(
                f"video://{episode_id}/{int(normalized_t)}"
            )
        else:
            previous_action_history.append(None)
            proprio_history.append(None)
            short_visual_history_refs.append(None)
    return {
        "history_k": history_k,
        "history_stride": history_stride,
        "history_valid_mask": valid_mask,
        "history_episode_ids": history_episode_ids,
        "history_t_std_indices": history_t_std_indices,
        "history_t_raw_indices": history_t_raw_indices,
        "history_timestamp_s": history_timestamp_s,
        "anchor_episode_id": episode_id,
        "anchor_mujoco_state_ref": f"mujoco://{episode_id}/{int(t)}",
        "prehistory_window": prehistory_window,
        "reset_boundary": state_conditioned_bucket_a_import.STATE_CONDITIONED_RESET_BOUNDARY,
        "deployable.previous_action_history": previous_action_history,
        "deployable.proprio_history": proprio_history,
        "deployable.short_visual_history_refs": short_visual_history_refs,
    }


def _build_npz_for_episode(
    npz_path: Path,
    *,
    episode_index: int,
    n_steps: int,
) -> None:
    import importlib

    np = importlib.import_module("numpy")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for key_index, key in enumerate(lerobot_v2_export.STATE_KEY_ORDER_LOCK):
        arr = np.zeros((n_steps, 1, 1, 1), dtype=np.float32)
        for step_index in range(n_steps):
            arr[step_index, 0, 0, 0] = float(
                10 * episode_index + key_index + (0.1 * step_index)
            )
        payload[key] = arr
    for key_index, key in enumerate(lerobot_v2_export.ACTION_KEY_ORDER_LOCK):
        arr = np.zeros((n_steps, 1, int(T_ACTION), 1), dtype=np.float32)
        for step_index in range(n_steps):
            for inner_index in range(int(T_ACTION)):
                arr[step_index, 0, inner_index, 0] = float(
                    100 * episode_index
                    + 10 * key_index
                    + step_index
                    + (0.01 * inner_index)
                )
        payload[key] = arr
    np.savez(npz_path, **payload)


def _build_dataset_records(dataset_dir: Path) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    sidecar_rows: list[dict[str, Any]] = []
    for episode_index, spec in enumerate(_step_specs()):
        episode_id = str(spec["episode_id"])
        rewards = [float(value) for value in list(spec["rewards"])]
        phases = [str(value) for value in list(spec["phases"])]
        modes = [str(value) for value in list(spec["modes"])]
        n_steps = len(rewards)
        npz_path = dataset_dir / "arrays" / f"{episode_id}.npz"
        _build_npz_for_episode(npz_path, episode_index=episode_index, n_steps=n_steps)
        episodes.append(
            {
                "episode_id": episode_id,
                "iter_tag": ITER_TAG,
                "seed": int(episode_index),
                "success_episode": bool(episode_index == 0),
                "n_policy_steps": int(n_steps),
                "npz_path": f"arrays/{episode_id}.npz",
                "prompt_raw": TASK_TEXT,
                "prompt_conditioned": TASK_TEXT,
            }
        )
        for t, reward_online in enumerate(rewards):
            n_exec = _n_exec_for_step(t)
            per_inner_reward = float(reward_online) / float(n_exec)
            history = _history_payload(episode_id, t)
            phase = str(phases[t])
            mode = str(modes[t])
            transitions.append(
                {
                    "schema_version": "recap-v0",
                    "code_version": "offline_sanity",
                    "iter_tag": ITER_TAG,
                    "episode_id": episode_id,
                    "t": int(t),
                    "reward_online": float(reward_online),
                    "timestamp_s": _timestamp_for_step(t),
                    "n_action_steps_executed": int(n_exec),
                    "n_action_steps_config": int(T_ACTION),
                    "T_action": int(T_ACTION),
                    "inner_rewards": [float(per_inner_reward)] * int(n_exec),
                    "inner_dones": [False] * int(n_exec),
                    "is_correction": False,
                    "prompt_raw": TASK_TEXT,
                    "prompt_conditioned": TASK_TEXT,
                    "success_step": bool(episode_index == 0 and t == (n_steps - 1)),
                }
            )
            sidecar_rows.append(
                {
                    "episode_id": episode_id,
                    "t": int(t),
                    **history,
                    "policy_condition.phase": phase,
                    "policy_condition.mode": mode,
                    "policy_condition_text": state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
                        phase,
                        mode,
                    ),
                }
            )
    _write_jsonl(dataset_dir / "episodes.jsonl", episodes)
    _write_jsonl(dataset_dir / "transitions.jsonl", transitions)
    _write_jsonl(dataset_dir / "state_conditioned_sidecar.jsonl", sidecar_rows)
    return {
        "episodes": episodes,
        "transitions": transitions,
        "sidecar_rows": sidecar_rows,
    }


def _build_label_artifacts(dataset_dir: Path) -> dict[str, Any]:
    dataset = dataset_reader.read_m1_dataset(dataset_dir, check_npz_keys=True)
    generated_labels = labeler.generate_m2_labels(
        dataset,
        value_baseline="t_mean_return",
        value_source="baseline",
        epsilon_strategy="const",
        epsilon_value=0.0,
        schema_version_default="recap-v0",
        code_version_default="offline_sanity",
    )
    advantage_values = [
        float(cast(Any, record["advantage_A"])) for record in generated_labels
    ]
    sign_scale_summary = advantage.compute_sign_aware_advantage_scales(
        advantage_values,
        context="state_conditioned_offline_sanity",
    )
    positive_scale = sign_scale_summary["positive_scale"]
    negative_scale_abs = sign_scale_summary["negative_scale_abs"]
    if positive_scale is None or negative_scale_abs is None:
        raise ValueError(
            "continuous advantage contract requires both positive and negative scales"
        )
    contract = advantage.build_advantage_contract_metadata(
        source_iter_tag=ITER_TAG,
        n_samples=len(generated_labels),
        positive_scale=float(positive_scale),
        negative_scale_abs=float(negative_scale_abs),
        critic_dir=None,
        critic_include_t=False,
        sign_scale_summary=sign_scale_summary,
        scale_rule=advantage.ADVANTAGE_SCALE_RULE,
    )
    labels_with_advantage: list[dict[str, Any]] = []
    for record in generated_labels:
        with_advantage = dict(record)
        with_advantage["advantage_input"] = advantage.normalize_advantage_to_input(
            record["advantage_A"],
            positive_scale=float(positive_scale),
            negative_scale_abs=float(negative_scale_abs),
        )
        with_advantage["positive_scale"] = float(positive_scale)
        with_advantage["negative_scale_abs"] = float(negative_scale_abs)
        labels_with_advantage.append(with_advantage)
    labels_dir = dataset_dir / "m2_labels"
    _write_jsonl(labels_dir / "labels.jsonl", labels_with_advantage)
    _write_json(dataset_dir / "continuous_advantage_contract.json", contract)
    return {
        "dataset": dataset,
        "labels": labels_with_advantage,
        "contract": contract,
    }


def _create_fixture(workspace_root: Path) -> dict[str, Any]:
    dataset_dir = (
        workspace_root / "agent" / "artifacts" / "recap_datasets" / ITER_TAG
    ).resolve()
    if dataset_dir.exists():
        raise ValueError(
            f"refusing to reuse existing offline sanity dataset dir: {dataset_dir}"
        )
    dataset_dir.mkdir(parents=True, exist_ok=False)
    dataset_records = _build_dataset_records(dataset_dir)
    label_artifacts = _build_label_artifacts(dataset_dir)
    return {
        "workspace_root": workspace_root,
        "dataset_dir": dataset_dir,
        **dataset_records,
        **label_artifacts,
    }


def _apply_mismatch(fixture: Mapping[str, Any], mismatch_mode: str | None) -> None:
    if mismatch_mode is None:
        return
    dataset_dir = Path(fixture["dataset_dir"])
    if mismatch_mode == "sidecar":
        sidecar_path = dataset_dir / "state_conditioned_sidecar.jsonl"
        rows = _read_jsonl_dicts(sidecar_path)
        if not rows:
            raise ValueError("cannot inject sidecar mismatch into empty sidecar")
        rows.pop()
        _write_jsonl(sidecar_path, rows)
        return
    if mismatch_mode == "history":
        sidecar_path = dataset_dir / "state_conditioned_sidecar.jsonl"
        rows = _read_jsonl_dicts(sidecar_path)
        if not rows:
            raise ValueError("cannot inject history mismatch into empty sidecar")
        rows[0]["history_t_std_indices"] = [0, 0, 0, 0, 0, 0, 0, 3]
        _write_jsonl(sidecar_path, rows)
        return
    if mismatch_mode == "exporter":
        labels_path = dataset_dir / "m2_labels" / "labels.jsonl"
        labels_rows = _read_jsonl_dicts(labels_path)
        if not labels_rows:
            raise ValueError("cannot inject exporter mismatch into empty labels")
        current = float(labels_rows[0].get("advantage_input", 0.0))
        labels_rows[0]["advantage_input"] = 0.0 if abs(current) > 1e-6 else 1.0
        _write_jsonl(labels_path, labels_rows)
        return
    raise ValueError(f"unknown mismatch_mode: {mismatch_mode!r}")


def _expected_join_keys(
    transitions: Sequence[Mapping[str, Any]],
) -> list[list[object]]:
    return [[str(record["episode_id"]), int(record["t"])] for record in transitions]


def _check_sidecar_round_trip(fixture: Mapping[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(fixture["dataset_dir"])
    transitions = list(fixture["transitions"])
    result = cast(
        dict[str, Any],
        state_conditioned_bucket_a_import.validate_sidecar_round_trip(
            sidecar_path=dataset_dir / "state_conditioned_sidecar.jsonl",
            expected_join_keys=_expected_join_keys(transitions),
        ),
    )
    return _pass_check(
        "sidecar_round_trip",
        record_count=int(result["record_count"]),
        join_key_count=int(result["join_key_count"]),
        phase_values=list(result["phase_values"]),
        mode_values=list(result["mode_values"]),
        history_k=int(result["history_k"]),
        history_stride=int(result["history_stride"]),
    )


def _expected_history_vectors(t: int) -> dict[str, Any]:
    history_k = int(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K)
    expected_mask: list[bool] = []
    expected_indices: list[int] = []
    expected_timestamps: list[float] = []
    for index in range(history_k):
        candidate_t = int(t) - (history_k - 1) + index
        is_valid = candidate_t >= 0
        normalized_t = candidate_t if is_valid else 0
        expected_mask.append(bool(is_valid))
        expected_indices.append(int(normalized_t))
        expected_timestamps.append(_timestamp_for_step(normalized_t))
    return {
        "mask": expected_mask,
        "indices": expected_indices,
        "timestamps": expected_timestamps,
    }


def _check_history_window_padding_reset_boundary(
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_dir = Path(fixture["dataset_dir"])
    rows = _read_jsonl_dicts(dataset_dir / "state_conditioned_sidecar.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["episode_id"]), []).append(row)
    first_step_valid_counts: dict[str, int] = {}
    for episode_id, episode_rows in grouped.items():
        ordered_rows = sorted(episode_rows, key=lambda row: int(row["t"]))
        for row in ordered_rows:
            t = int(row["t"])
            state_conditioned_bucket_a_import.validate_sidecar_row_for_gate(row)
            expected = _expected_history_vectors(t)
            valid_mask = list(row["history_valid_mask"])
            history_episode_ids = list(row["history_episode_ids"])
            history_t_std_indices = list(row["history_t_std_indices"])
            history_t_raw_indices = list(row["history_t_raw_indices"])
            history_timestamp_s = [float(value) for value in row["history_timestamp_s"]]
            if valid_mask != list(expected["mask"]):
                raise ValueError(
                    f"history_valid_mask mismatch for episode_id={episode_id} t={t}"
                )
            if history_episode_ids != [episode_id] * len(history_episode_ids):
                raise ValueError(
                    f"history reset-boundary episode mismatch for episode_id={episode_id} t={t}"
                )
            if history_t_std_indices != list(expected["indices"]):
                raise ValueError(
                    f"history_t_std_indices mismatch for episode_id={episode_id} t={t}"
                )
            if history_t_raw_indices != list(expected["indices"]):
                raise ValueError(
                    f"history_t_raw_indices mismatch for episode_id={episode_id} t={t}"
                )
            if history_timestamp_s != list(expected["timestamps"]):
                raise ValueError(
                    f"history_timestamp_s mismatch for episode_id={episode_id} t={t}"
                )
            for index, is_valid in enumerate(valid_mask):
                action_slot = row["deployable.previous_action_history"][index]
                proprio_slot = row["deployable.proprio_history"][index]
                visual_slot = row["deployable.short_visual_history_refs"][index]
                prehistory_row = dict(row["prehistory_window"][index])
                if bool(is_valid):
                    if (
                        action_slot is None
                        or proprio_slot is None
                        or visual_slot is None
                    ):
                        raise ValueError(
                            "valid history slot missing deployable payload for "
                            + f"episode_id={episode_id} t={t} index={index}"
                        )
                else:
                    if (
                        action_slot is not None
                        or proprio_slot is not None
                        or visual_slot is not None
                    ):
                        raise ValueError(
                            "padded history slot must remain empty for "
                            + f"episode_id={episode_id} t={t} index={index}"
                        )
                if str(prehistory_row["episode_id"]) != episode_id:
                    raise ValueError(
                        f"prehistory_window reset-boundary mismatch for episode_id={episode_id} t={t}"
                    )
                if int(prehistory_row["t_std"]) != int(history_t_std_indices[index]):
                    raise ValueError(
                        f"prehistory_window t_std mismatch for episode_id={episode_id} t={t}"
                    )
            if t == 0:
                first_step_valid_counts[episode_id] = int(
                    sum(bool(value) for value in valid_mask)
                )
                if first_step_valid_counts[episode_id] != 1:
                    raise ValueError(
                        f"episode reset boundary must expose exactly one valid history slot for {episode_id}"
                    )
    return _pass_check(
        "history_window_padding_reset_boundary",
        checked_episode_count=len(grouped),
        checked_record_count=len(rows),
        history_k=int(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K),
        reset_boundary=str(
            state_conditioned_bucket_a_import.STATE_CONDITIONED_RESET_BOUNDARY
        ),
        first_step_valid_counts=first_step_valid_counts,
    )


def _check_phase_mode_parsing(fixture: Mapping[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(fixture["dataset_dir"])
    rows = _read_jsonl_dicts(dataset_dir / "state_conditioned_sidecar.jsonl")
    normalized_phases: set[str] = set()
    normalized_modes: set[str] = set()
    mixed_case_inputs_verified = False
    for row in rows:
        raw_phase = row.get("policy_condition.phase")
        raw_mode = row.get("policy_condition.mode")
        normalized_phase, normalized_mode, normalized_text = (
            state_conditioned_bucket_a_import.validate_state_conditioned_policy_condition(
                phase=raw_phase,
                mode=raw_mode,
                policy_condition_text=row.get("policy_condition_text"),
            )
        )
        if isinstance(raw_phase, str) and raw_phase != normalized_phase:
            mixed_case_inputs_verified = True
        if isinstance(raw_mode, str) and raw_mode != normalized_mode:
            mixed_case_inputs_verified = True
        normalized_phases.add(str(normalized_phase))
        normalized_modes.add(str(normalized_mode))
        if (
            normalized_text
            != state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
                normalized_phase,
                normalized_mode,
            )
        ):
            raise ValueError("policy_condition_text canonicalization mismatch")
    if not mixed_case_inputs_verified:
        raise ValueError("mixed-case phase/mode normalization was not exercised")
    return _pass_check(
        "phase_mode_parsing",
        normalized_phase_values=sorted(normalized_phases),
        normalized_mode_values=sorted(normalized_modes),
        mixed_case_inputs_verified=True,
    )


def _label_key(record: Mapping[str, Any]) -> tuple[str, int]:
    return str(record["episode_id"]), int(record["t"])


def _label_round_trip_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": record["schema_version"],
        "code_version": record["code_version"],
        "iter_tag": record["iter_tag"],
        "episode_id": record["episode_id"],
        "t": int(record["t"]),
        "return_G": float(record["return_G"]),
        "value_V": float(record["value_V"]),
        "advantage_A": float(record["advantage_A"]),
        "epsilon_l": float(record["epsilon_l"]),
        "indicator_I": int(record["indicator_I"]),
        "is_correction": bool(record["is_correction"]),
        "prompt_raw": str(record["prompt_raw"]),
        "prompt_conditioned": str(record["prompt_conditioned"]),
    }


def _check_label_round_trip(fixture: Mapping[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(fixture["dataset_dir"])
    dataset = dataset_reader.read_m1_dataset(dataset_dir, check_npz_keys=True)
    regenerated = labeler.generate_m2_labels(
        dataset,
        value_baseline="t_mean_return",
        value_source="baseline",
        epsilon_strategy="const",
        epsilon_value=0.0,
        schema_version_default="recap-v0",
        code_version_default="offline_sanity",
    )
    observed = _read_jsonl_dicts(dataset_dir / "m2_labels" / "labels.jsonl")
    regenerated_by_key = {
        _label_key(record): _label_round_trip_fields(record) for record in regenerated
    }
    observed_by_key = {
        _label_key(record): _label_round_trip_fields(record) for record in observed
    }
    if observed_by_key != regenerated_by_key:
        missing = sorted(set(regenerated_by_key) - set(observed_by_key))
        extra = sorted(set(observed_by_key) - set(regenerated_by_key))
        raise ValueError(
            f"label round-trip mismatch: missing={missing!r} extra={extra!r}"
        )
    contract = _read_json(dataset_dir / "continuous_advantage_contract.json")
    if contract.get("scale_rule") != advantage.ADVANTAGE_SCALE_RULE:
        raise ValueError("continuous advantage contract scale_rule mismatch")
    if (
        contract.get("positive_scale") is None
        or contract.get("negative_scale_abs") is None
    ):
        raise ValueError("continuous advantage contract missing sign-aware scales")
    return _pass_check(
        "label_round_trip",
        label_count=len(observed),
        contract_version=str(contract.get("contract_version")),
        scale_rule=str(contract.get("scale_rule")),
        positive_scale=float(contract["positive_scale"]),
        negative_scale_abs=float(contract["negative_scale_abs"]),
    )


def _collect_exported_frames(export_dir: Path) -> dict[str, Any]:
    import importlib

    pd = importlib.import_module("pandas")
    parquet_files = sorted(
        (export_dir / lerobot_v2_export.DATA_DIRNAME).glob("chunk-*/*.parquet")
    )
    if not parquet_files:
        raise ValueError(f"missing exported parquet files under {export_dir}")
    frames = [pd.read_parquet(path) for path in parquet_files]
    frame = pd.concat(frames, ignore_index=True)
    return {
        "frame": frame,
        "parquet_file_count": len(parquet_files),
    }


def _expected_exported_advantage_inputs(
    transitions: Sequence[Mapping[str, Any]],
    labels_rows: Sequence[Mapping[str, Any]],
) -> list[float]:
    labels_by_key = {
        _label_key(label_row): float(label_row["advantage_input"])
        for label_row in labels_rows
    }
    exploded: list[float] = []
    for transition in transitions:
        key = _label_key(transition)
        n_exec = int(transition["n_action_steps_executed"])
        exploded.extend([float(labels_by_key[key])] * n_exec)
    return exploded


def _check_exporter_round_trip(fixture: Mapping[str, Any]) -> dict[str, Any]:
    workspace_root = Path(fixture["workspace_root"])
    dataset_dir = Path(fixture["dataset_dir"])
    export_dir = lerobot_v2_export.resolve_lerobot_v2_dataset_dir(
        ITER_TAG,
        repo_root=workspace_root,
    ).resolve()
    export_result = lerobot_v2_export.export_recap_to_lerobot_v2(
        iter_tag=ITER_TAG,
        repo_root=workspace_root,
        input_recap_dataset_dir=dataset_dir,
        output_dataset_dir=export_dir,
        include_m2_label_columns=True,
    )
    info = _read_json(
        export_dir / lerobot_v2_export.META_DIRNAME / lerobot_v2_export.META_INFO_JSON
    )
    episodes_meta = _read_jsonl_dicts(
        export_dir
        / lerobot_v2_export.META_DIRNAME
        / lerobot_v2_export.META_EPISODES_JSONL
    )
    tasks_meta = _read_jsonl_dicts(
        export_dir / lerobot_v2_export.META_DIRNAME / lerobot_v2_export.META_TASKS_JSONL
    )
    exported = _collect_exported_frames(export_dir)
    frame = exported["frame"]
    expected_advantage_inputs = _expected_exported_advantage_inputs(
        list(fixture["transitions"]),
        _read_jsonl_dicts(dataset_dir / "m2_labels" / "labels.jsonl"),
    )
    observed_advantage_inputs = [
        float(value) for value in list(frame[advantage.ADVANTAGE_INPUT_COLUMN].tolist())
    ]
    if observed_advantage_inputs != expected_advantage_inputs:
        raise ValueError("exported recap_m2.advantage_input column mismatch")
    field_groups = dict(info["field_groups"]["state_conditioned_sidecar"])
    if set(field_groups.keys()) != {
        lerobot_v2_export.DEPLOYABLE_HISTORY_GROUP_KEY,
        lerobot_v2_export.PRIVILEGED_ANALYSIS_ONLY_GROUP_KEY,
        lerobot_v2_export.TEACHER_ONLY_GROUP_KEY,
    }:
        raise ValueError("exporter field_groups keys mismatch")
    if field_groups[lerobot_v2_export.DEPLOYABLE_HISTORY_GROUP_KEY] != list(
        lerobot_v2_export.DEPLOYABLE_HISTORY_FIELD_NAMES
    ):
        raise ValueError("deployable_history field group mismatch")
    expected_total_frames = sum(
        int(record["n_action_steps_executed"])
        for record in list(fixture["transitions"])
    )
    if int(export_result.total_frames) != int(expected_total_frames):
        raise ValueError("exporter total_frames mismatch")
    return _pass_check(
        "exporter_round_trip",
        output_dataset_dir=str(export_result.output_dataset_dir),
        total_episodes=int(export_result.total_episodes),
        total_frames=int(export_result.total_frames),
        total_tasks=int(export_result.total_tasks),
        state_dim=int(export_result.state_dim),
        action_dim=int(export_result.action_dim),
        episode_meta_count=len(episodes_meta),
        task_meta_count=len(tasks_meta),
        parquet_file_count=int(exported["parquet_file_count"]),
        field_groups=field_groups,
        advantage_input_column=advantage.ADVANTAGE_INPUT_COLUMN,
    )


def _base_payload(output_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_offline_sanity_report",
        "status": "FAIL",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": str(output_dir),
        "repo_root": str(REPO_ROOT),
        "iter_tag": ITER_TAG,
        "failure": None,
        "checks": {name: _not_run_check(name) for name in CHECK_ORDER},
    }


def run_offline_sanity(
    output_dir: Path,
    *,
    mismatch_mode: str | None = None,
) -> dict[str, Any]:
    payload = _base_payload(output_dir)
    try:
        with tempfile.TemporaryDirectory(
            dir=output_dir,
            prefix="_offline_sanity_work_",
        ) as tempdir:
            workspace_root = Path(tempdir).resolve()
            fixture = _create_fixture(workspace_root)
            _apply_mismatch(fixture, mismatch_mode)
            for check_name, check_fn in (
                ("sidecar_round_trip", _check_sidecar_round_trip),
                (
                    "history_window_padding_reset_boundary",
                    _check_history_window_padding_reset_boundary,
                ),
                ("phase_mode_parsing", _check_phase_mode_parsing),
                ("label_round_trip", _check_label_round_trip),
                ("exporter_round_trip", _check_exporter_round_trip),
            ):
                try:
                    payload["checks"][check_name] = check_fn(fixture)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    payload["checks"][check_name] = _fail_check(
                        check_name,
                        _exception_message(exc),
                    )
                    payload["failure"] = {
                        "stage": check_name,
                        "type": exc.__class__.__name__,
                        "message": _exception_message(exc),
                    }
                    return payload
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        payload["failure"] = {
            "stage": "cli",
            "type": exc.__class__.__name__,
            "message": _exception_message(exc),
        }
        return payload
    payload["status"] = "PASS"
    payload["failure"] = None
    payload["summary"] = {
        "passed_check_count": len(CHECK_ORDER),
        "total_check_count": len(CHECK_ORDER),
    }
    return payload


def materialize_offline_sanity(
    output_dir: Path,
    *,
    mismatch_mode: str | None = None,
) -> dict[str, Any]:
    resolved_output_dir = _validate_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_offline_sanity(
        resolved_output_dir,
        mismatch_mode=mismatch_mode,
    )
    if payload.get("status") != "PASS" and payload.get("summary") is None:
        passed_count = sum(
            1
            for check in dict(payload["checks"]).values()
            if dict(check).get("status") == "PASS"
        )
        payload["summary"] = {
            "passed_check_count": int(passed_count),
            "total_check_count": len(CHECK_ORDER),
        }
    report_path = resolved_output_dir / REPORT_JSON_NAME
    payload["report_path"] = str(report_path)
    _write_json(report_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args: argparse.Namespace = parser.parse_args(argv)
    try:
        payload = materialize_offline_sanity(Path(str(args.output_dir)))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        resolved_output_dir = _validate_output_dir(Path(str(args.output_dir)))
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        payload = _base_payload(resolved_output_dir)
        payload["failure"] = {
            "stage": "cli",
            "type": exc.__class__.__name__,
            "message": _exception_message(exc),
        }
        payload["summary"] = {
            "passed_check_count": 0,
            "total_check_count": len(CHECK_ORDER),
        }
        payload["report_path"] = str(resolved_output_dir / REPORT_JSON_NAME)
        _write_json(resolved_output_dir / REPORT_JSON_NAME, payload)
    if payload.get("status") != "PASS":
        failure = cast(dict[str, Any], payload.get("failure") or {})
        message = str(failure.get("message", "offline sanity failed"))
        print(message, file=sys.stderr)
    print(_json_text(payload))
    return 0 if payload.get("status") == "PASS" else 1


__all__ = [
    "CHECK_ORDER",
    "DEFAULT_OUTPUT_DIR",
    "ITER_TAG",
    "REPORT_JSON_NAME",
    "SCHEMA_VERSION",
    "build_parser",
    "main",
    "materialize_offline_sanity",
    "run_offline_sanity",
]


if __name__ == "__main__":
    raise SystemExit(main())
