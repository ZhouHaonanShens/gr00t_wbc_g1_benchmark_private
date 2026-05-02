from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import state_conditioned_bucket_a_import
from work.recap.scripts import state_conditioned_bucket_a_sidecar
from work.recap.scripts import state_conditioned_build_training_set
from work.recap.scripts import state_conditioned_collect_buckets
from work.recap.scripts import state_conditioned_dev_manifest
from work.recap.scripts import state_conditioned_snapshot_harvest
from work.recap.scripts import state_conditioned_train
from work.recap import lerobot_v2_export


TEST_POLICY_STEP_COUNT = 1
TEST_ACTION_HORIZON = 30
TEST_STATE_DIMS = {
    "state/left_arm": 7,
    "state/left_hand": 7,
    "state/left_leg": 6,
    "state/right_arm": 7,
    "state/right_hand": 7,
    "state/right_leg": 6,
    "state/waist": 3,
}
TEST_ACTION_DIMS = {
    "action/left_arm": 7,
    "action/right_arm": 7,
    "action/left_hand": 7,
    "action/right_hand": 7,
    "action/waist": 3,
    "action/base_height_command": 1,
    "action/navigate_command": 3,
}


def _write_test_mp4(
    path: Path,
    *,
    frame_count: int = TEST_ACTION_HORIZON,
    fps: int = int(state_conditioned_build_training_set.LEROBOT_EXPORT_FPS),
) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg is required to build synthetic test videos")
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_s = max(float(frame_count) / float(fps), 1.0 / float(fps))
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s=16x16:r={int(fps)}:d={duration_s:.6f}",
        "-frames:v",
        str(int(frame_count)),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"ffmpeg failed while building synthetic test video {path}: {detail}"
        )
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"failed to create synthetic test video: {path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _install_fake_lerobot_export(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_materialize_lerobot_training_dataset(
        *,
        output_dir: Path,
        base_rows: list[dict[str, Any]],
        prerequisites: dict[str, Any],
    ) -> dict[str, Any]:
        del prerequisites
        dataset_root = (
            output_dir
            / state_conditioned_build_training_set.LEROBOT_TRAINING_DATASET_DIRNAME
        )
        meta_dir = dataset_root / "meta"
        data_dir = dataset_root / "data" / "chunk-000"
        video_dir = (
            dataset_root / "videos" / "chunk-000" / "observation.images.ego_view"
        )
        meta_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        task_texts = sorted(
            {str(row["canonical_policy_condition_text"]) for row in list(base_rows)}
        )
        state_dim = int(sum(TEST_STATE_DIMS.values()))
        action_dim = int(sum(TEST_ACTION_DIMS.values()))
        episode_count = int(len(base_rows))
        frame_count = int(episode_count * TEST_ACTION_HORIZON)

        _write_jsonl(
            meta_dir / "tasks.jsonl",
            [
                {"task_index": int(index), "task": task}
                for index, task in enumerate(task_texts)
            ],
        )
        _write_jsonl(
            meta_dir / "episodes.jsonl",
            [
                {
                    "episode_index": int(index),
                    "tasks": [str(row["canonical_policy_condition_text"])],
                    "length": TEST_ACTION_HORIZON,
                    "state_conditioned.sample_id": str(row["sample_id"]),
                }
                for index, row in enumerate(base_rows)
            ],
        )
        _write_json(
            meta_dir / "modality.json",
            {
                "video": {
                    "ego_view": {
                        "original_key": "observation.images.ego_view",
                    }
                }
            },
        )
        _write_json(
            meta_dir / "video_map.json",
            {
                "length_clamp": {"clamped_episodes": 0},
                "total_videos": episode_count,
            },
        )
        _write_json(
            meta_dir / "info.json",
            {
                "total_episodes": episode_count,
                "total_frames": frame_count,
                "total_tasks": len(task_texts),
                "total_videos": episode_count,
                "video_path": "videos/chunk-{episode_chunk:03d}/observation.images.ego_view/episode_{episode_index:06d}.mp4",
                "features": {
                    "action": {
                        "dtype": "float32",
                        "shape": [action_dim],
                        "names": None,
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [state_dim],
                        "names": None,
                    },
                    "observation.images.ego_view": {
                        "dtype": "video",
                        "shape": [1],
                        "names": None,
                    },
                },
            },
        )

        for episode_index in range(episode_count):
            (data_dir / f"episode_{episode_index:06d}.parquet").write_bytes(b"PAR1")
            (video_dir / f"episode_{episode_index:06d}.mp4").write_bytes(b"00")

        return {
            "dataset_root": str(dataset_root),
            "meta_info_path": str(meta_dir / "info.json"),
            "video_map_path": str(meta_dir / "video_map.json"),
            "episode_count": episode_count,
            "frame_count": frame_count,
            "task_count": len(task_texts),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "video_count": episode_count,
        }

    monkeypatch.setattr(
        state_conditioned_build_training_set,
        "materialize_lerobot_training_dataset",
        _fake_materialize_lerobot_training_dataset,
    )


def _build_npz_payload(seed_value: int) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key_index, key in enumerate(lerobot_v2_export.STATE_KEY_ORDER_LOCK):
        dim = TEST_STATE_DIMS[key]
        values = np.arange(dim, dtype=np.float32) + float(seed_value + key_index)
        payload[key] = values.reshape(TEST_POLICY_STEP_COUNT, 1, 1, dim)
    for key_index, key in enumerate(lerobot_v2_export.ACTION_KEY_ORDER_LOCK):
        dim = TEST_ACTION_DIMS[key]
        values = np.arange(
            TEST_POLICY_STEP_COUNT * TEST_ACTION_HORIZON * dim,
            dtype=np.float32,
        ).reshape(TEST_POLICY_STEP_COUNT, 1, TEST_ACTION_HORIZON, dim)
        payload[key] = values + float(seed_value + key_index)
    return payload


def _write_source_dataset(
    dataset_dir: Path,
    *,
    episode_id: str,
    prompt_text: str,
    seed_value: int,
    sidecar_rows: list[dict[str, Any]] | None = None,
) -> None:
    arrays_dir = dataset_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    npz_path = arrays_dir / f"{episode_id}.npz"
    np.savez(npz_path, **_build_npz_payload(seed_value))
    video_dir_archived = dataset_dir / "archived_videos"
    video_path = video_dir_archived / f"{episode_id}_s0.mp4"
    _write_test_mp4(video_path)

    _write_jsonl(
        dataset_dir / "episodes.jsonl",
        [
            {
                "episode_id": episode_id,
                "prompt_raw": prompt_text,
                "prompt_conditioned": prompt_text,
                "npz_path": str(Path("arrays") / npz_path.name),
                "n_action_steps_config": TEST_ACTION_HORIZON,
                "n_policy_steps": TEST_POLICY_STEP_COUNT,
                "embodiment_tag": "UNITREE_G1",
                "video_dir_tmp": str((dataset_dir / "tmp_videos").resolve()),
                "video_dir_archived": str(video_dir_archived.resolve()),
            }
        ],
    )
    _write_jsonl(
        dataset_dir / "transitions.jsonl",
        [
            {
                "episode_id": episode_id,
                "t": 0,
                "T_action": TEST_ACTION_HORIZON,
                "n_action_steps_config": TEST_ACTION_HORIZON,
                "n_action_steps_executed": TEST_ACTION_HORIZON,
                "inner_rewards": [0.0] * TEST_ACTION_HORIZON,
                "inner_dones": [False] * TEST_ACTION_HORIZON,
            }
        ],
    )
    _write_jsonl(
        dataset_dir / "m2_labels" / "labels.jsonl",
        [
            {
                "episode_id": episode_id,
                "t": 0,
                "return_G": float(seed_value) + 0.5,
                "value_V": float(seed_value) - 0.25,
                "advantage_A": 0.75,
                "epsilon_l": 0.1,
                "indicator_I": 1,
            }
        ],
    )
    if sidecar_rows is not None:
        _write_jsonl(dataset_dir / "state_conditioned_sidecar.jsonl", sidecar_rows)


def _history_payload(episode_id: str, t: int) -> dict[str, Any]:
    valid_mask: list[bool] = []
    history_episode_ids: list[str] = []
    prehistory_window: list[dict[str, Any]] = []
    start_t = int(t) - (
        state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K - 1
    )
    for index in range(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K):
        candidate_t = start_t + index
        is_valid = candidate_t >= 0
        row_t = candidate_t if is_valid else 0
        valid_mask.append(bool(is_valid))
        history_episode_ids.append(episode_id)
        prehistory_window.append(
            {
                "episode_id": episode_id,
                "t_std": int(row_t),
                "mujoco_state_ref": f"mujoco://{episode_id}/{int(row_t)}",
            }
        )
    return {
        "history_k": state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
        "history_stride": state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_STRIDE,
        "history_valid_mask": valid_mask,
        "history_episode_ids": history_episode_ids,
        "anchor_episode_id": episode_id,
        "anchor_mujoco_state_ref": f"mujoco://{episode_id}/{int(t)}",
        "prehistory_window": prehistory_window,
        "reset_boundary": state_conditioned_bucket_a_import.STATE_CONDITIONED_RESET_BOUNDARY,
    }


def _deployable_history_payload(tag: str) -> dict[str, Any]:
    history_k = state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K
    previous_action_history: list[Any] = []
    proprio_history: list[Any] = []
    short_visual_history_refs: list[Any] = []
    for index in range(history_k):
        previous_action_history.append([float(index), float(index) + 0.1, tag])
        proprio_history.append([float(index) + 1.0, tag])
        short_visual_history_refs.append(f"visual://{tag}/{index}")
    return {
        "deployable.previous_action_history": previous_action_history,
        "deployable.proprio_history": proprio_history,
        "deployable.short_visual_history_refs": short_visual_history_refs,
    }


def _policy_payload(phase: str, mode: str) -> dict[str, Any]:
    return {
        "policy_condition.phase": phase,
        "policy_condition.mode": mode,
        "policy_condition_text": state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
            phase,
            mode,
        ),
    }


def _sidecar_like_row(
    *,
    episode_id: str,
    t: int,
    phase: str,
    mode: str,
    tag: str,
    extra_deployable_field: str | None = None,
) -> dict[str, Any]:
    row = {
        "episode_id": episode_id,
        "t": int(t),
        **_history_payload(episode_id, t),
        **_deployable_history_payload(tag),
        **_policy_payload(phase, mode),
        "event": "analysis_only_event",
        "semantic_state": "SEARCHING",
        "memory_commit_mask": [False]
        * state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
        "memory_commit_cause": "none",
        "recovery_entry_step": None,
        "summary_template": None,
        "privileged.apple_visible": True,
    }
    if extra_deployable_field is not None:
        row[extra_deployable_field] = {"forbidden": True}
    return row


def _snapshot_candidate_row(
    *,
    snapshot_id: str,
    family: str,
    episode_id: str,
    anchor_t: int,
    phase: str,
    mode: str,
    tag: str,
) -> dict[str, Any]:
    row = {
        "snapshot_id": snapshot_id,
        "family": family,
        "source_bucket_key": state_conditioned_snapshot_harvest.FAMILY_SOURCE_BUCKET_BY_FAMILY[
            family
        ],
        "anchor_t": int(anchor_t),
        **_history_payload(episode_id, anchor_t),
        **_deployable_history_payload(tag),
        **_policy_payload(phase, mode),
    }
    return row


def _build_bucket_a_fixture(
    tmp_path: Path,
    *,
    contamination_field: str | None = None,
) -> Path:
    bucket_dir = tmp_path / "bucket_a"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    manifest_episodes: list[dict[str, Any]] = []
    sidecar_rows: list[dict[str, Any]] = []
    for index in range(24):
        episode_id = f"bucket_a_ep_{index:03d}"
        dataset_dir = tmp_path / "bucket_a_datasets" / episode_id
        phase = "SEARCH" if index % 2 == 0 else "APPROACH"
        mode = "NOMINAL" if index < 12 else "RECOVERY"
        sidecar_row = _sidecar_like_row(
            episode_id=episode_id,
            t=0,
            phase=phase,
            mode=mode,
            tag=f"bucket_a_{index:03d}",
            extra_deployable_field=contamination_field if index == 0 else None,
        )
        _write_source_dataset(
            dataset_dir,
            episode_id=episode_id,
            prompt_text="pick up the apple and place it on the plate",
            seed_value=index,
        )
        manifest_episodes.append(
            {
                "episode_id": episode_id,
                "accepted": True,
                "debug_only": False,
                "fresh_nominal_recollection": True,
                "reused_existing_live_dataset": False,
                "source_dataset_dir": str(dataset_dir),
            }
        )
        sidecar_rows.append(sidecar_row)
    _write_json(
        bucket_dir / state_conditioned_bucket_a_import.GATE_A_READY_JSON_NAME,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "bucket_key": state_conditioned_bucket_a_import.BUCKET_KEY,
            "ready": True,
            "required_distinct_accepted_episode_count": 24,
            "accepted_episode_count": 24,
        },
    )
    _write_json(
        bucket_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "bucket_key": state_conditioned_bucket_a_import.BUCKET_KEY,
            "required_distinct_episode_count": 24,
            "episodes": manifest_episodes,
        },
    )
    _write_jsonl(
        bucket_dir / state_conditioned_bucket_a_sidecar.BUCKET_A_SIDECAR_JSON_NAME,
        sidecar_rows,
    )
    _write_json(
        bucket_dir
        / state_conditioned_bucket_a_sidecar.BUCKET_A_JOIN_COVERAGE_JSON_NAME,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "artifact_kind": "bucket_A_join_coverage",
            "coverage_ratio": 1.0,
        },
    )
    _write_json(
        bucket_dir
        / state_conditioned_bucket_a_sidecar.BUCKET_A_EXPORTER_MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "artifact_kind": "bucket_A_exporter_manifest",
            "accepted_episode_count": 24,
            "field_groups": {
                lerobot_v2_export.DEPLOYABLE_HISTORY_GROUP_KEY: list(
                    lerobot_v2_export.DEPLOYABLE_HISTORY_FIELD_NAMES
                ),
                lerobot_v2_export.PRIVILEGED_ANALYSIS_ONLY_GROUP_KEY: list(
                    lerobot_v2_export.PRIVILEGED_ANALYSIS_ONLY_FIELD_NAMES
                ),
                lerobot_v2_export.TEACHER_ONLY_GROUP_KEY: ["teacher.trace_id"],
            },
        },
    )
    return bucket_dir


def _build_dev_fixture(tmp_path: Path) -> Path:
    dev_dir = tmp_path / "devbench"
    dev_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        dev_dir / state_conditioned_dev_manifest.FIXED_STRATA_DEFINITION_JSON_NAME,
        {
            "schema_version": state_conditioned_dev_manifest.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_dev_fixed_strata_definition",
            "paired_seed_values": list(
                state_conditioned_dev_manifest.DEFAULT_PAIRED_SEEDS
            ),
            "paired_seed_count": 8,
        },
    )
    _write_json(
        dev_dir / state_conditioned_dev_manifest.BASELINE_MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_dev_manifest.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_dev_baseline_manifest",
            "baseline_policy": {
                "kind": "original_baseline",
                "model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
            },
            "counts": {
                "entries": 32,
                "paired_seed_count": 8,
                "per_stratum": dict(
                    state_conditioned_dev_manifest.EXPECTED_STRATA_COUNTS
                ),
            },
        },
    )
    _write_json(
        dev_dir / state_conditioned_dev_manifest.BASELINE_DEV_SCORECARD_JSON_NAME,
        {
            "schema_version": state_conditioned_dev_manifest.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_dev_baseline_scorecard",
            "counts": {"requested_entries": 32},
        },
    )
    return dev_dir


def _build_bucket_b_fixture(tmp_path: Path) -> Path:
    collection_dir = tmp_path / "collection"
    collection_dir.mkdir(parents=True, exist_ok=True)
    bucket_b_entries: list[dict[str, Any]] = []
    for index in range(16):
        episode_id = f"bucket_b_ep_{index:03d}"
        dataset_dir = tmp_path / "bucket_b_datasets" / episode_id
        sidecar_rows = [
            _sidecar_like_row(
                episode_id=episode_id,
                t=0,
                phase="GRASP" if index % 2 == 0 else "VERIFY_HOLD",
                mode="NOMINAL" if index < 8 else "RECOVERY",
                tag=f"bucket_b_{index:03d}",
            )
        ]
        _write_source_dataset(
            dataset_dir,
            episode_id=episode_id,
            prompt_text="pick up the apple and place it on the plate",
            seed_value=100 + index,
            sidecar_rows=sidecar_rows,
        )
        bucket_b_entries.append(
            {
                "bucket_key": "bucket_B",
                "episode_id": episode_id,
                "dataset_dir": str(dataset_dir),
            }
        )
    bucket_c_entries: list[dict[str, Any]] = []
    for index in range(24):
        episode_id = f"bucket_c_anchor_{index:03d}"
        dataset_dir = tmp_path / "bucket_c_datasets" / episode_id
        _write_source_dataset(
            dataset_dir,
            episode_id=episode_id,
            prompt_text="recover the apple and finish transport",
            seed_value=200 + index,
        )
        bucket_c_entries.append(
            {
                "bucket_key": "bucket_C",
                "episode_id": episode_id,
                "dataset_dir": str(dataset_dir),
            }
        )
    _write_json(
        collection_dir / state_conditioned_collect_buckets.BUCKET_B_MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_collect_buckets.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_bucket_B_manifest",
            "counts": {"episodes": 16},
            "episodes": bucket_b_entries,
        },
    )
    _write_json(
        collection_dir / state_conditioned_collect_buckets.BUCKET_C_MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_collect_buckets.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_bucket_C_manifest",
            "counts": {
                "episodes": 24,
                "per_failure_family": {
                    family: 8
                    for family in state_conditioned_collect_buckets.REQUIRED_FAILURE_FAMILIES
                },
            },
            "episodes": bucket_c_entries,
        },
    )
    _write_json(
        collection_dir
        / state_conditioned_collect_buckets.BUCKET_COLLECTION_SUMMARY_JSON_NAME,
        {
            "schema_version": state_conditioned_collect_buckets.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_bucket_collection_summary",
            "counts": {
                "bucket_B": 16,
                "bucket_C": 24,
                "bucket_C_per_failure_family": {
                    family: 8
                    for family in state_conditioned_collect_buckets.REQUIRED_FAILURE_FAMILIES
                },
            },
        },
    )
    return collection_dir


def _build_harvest_fixture(tmp_path: Path) -> Path:
    harvest_dir = tmp_path / "harvest"
    harvest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_candidates: list[dict[str, Any]] = []
    pseudodemos: list[dict[str, Any]] = []
    producer_by_family = {
        "S_drop": state_conditioned_snapshot_harvest.PRODUCER_BASE_POLICY,
        "S_lost": state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER,
        "S_transport_mid": state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER,
        "S_pre_place": state_conditioned_snapshot_harvest.PRODUCER_BASE_POLICY,
    }
    teacher_gate_families: list[dict[str, Any]] = []
    for family in state_conditioned_snapshot_harvest.T8_FAMILY_ORDER:
        teacher_gate_families.append(
            {
                "family": family,
                "success_rate": 0.05
                if producer_by_family[family]
                == state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER
                else 0.25,
                "threshold": 0.15,
                "teacher_fallback_enabled": producer_by_family[family]
                == state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER,
            }
        )
    snapshot_index = 0
    for family in state_conditioned_snapshot_harvest.T8_FAMILY_ORDER:
        phase = "PLACE" if family == "S_pre_place" else "TRANSPORT"
        for local_index in range(6):
            snapshot_id = f"{family}_{local_index:03d}"
            anchor_episode_id = f"bucket_c_anchor_{snapshot_index:03d}"
            snapshot_candidates.append(
                _snapshot_candidate_row(
                    snapshot_id=snapshot_id,
                    family=family,
                    episode_id=anchor_episode_id,
                    anchor_t=0,
                    phase=phase,
                    mode="RECOVERY",
                    tag=f"formal_{snapshot_index:03d}",
                )
            )
            pseudodemos.append(
                {
                    "episode_id": f"formal_ep_{snapshot_index:03d}",
                    "producer": producer_by_family[family],
                    "source_snapshot_id": snapshot_id,
                    "source_snapshot_family": family,
                    "source_bucket_key": state_conditioned_snapshot_harvest.FAMILY_SOURCE_BUCKET_BY_FAMILY[
                        family
                    ],
                    "source_snapshot_history_k": state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
                    "teacher_version": "scripted_teacher_v1",
                    "teacher_trigger_reason": "teacher_gate_success_rate_below_threshold",
                    "teacher_trigger_success_rate": 0.05,
                    "teacher_trigger_threshold": 0.15,
                    "failure_prefix_step_count": 2,
                    "failure_prefix_source_episode_id": anchor_episode_id,
                    "failure_prefix_source_t_range": [0, 1],
                    "recovery_suffix_step_count": 3,
                    "recovery_suffix_source_episode_id": anchor_episode_id,
                    "recovery_suffix_source_t_range": [2, 4],
                    "teacher_target_truthfulness": (
                        state_conditioned_snapshot_harvest.TEACHER_TARGET_NOT_APPLICABLE
                        if producer_by_family[family]
                        == state_conditioned_snapshot_harvest.PRODUCER_BASE_POLICY
                        else state_conditioned_snapshot_harvest.TEACHER_TARGET_TRUTHFUL_REAL_ROLLOUT
                    ),
                    "teacher_target": None
                    if producer_by_family[family]
                    == state_conditioned_snapshot_harvest.PRODUCER_BASE_POLICY
                    else {
                        "trace_episode_id": f"teacher_trace_{snapshot_index:03d}",
                        "trace_t_range": [0, 2],
                        "producer": "scripted_teacher_v1",
                        "synthetic_observation_only_backfill": False,
                    },
                }
            )
            snapshot_index += 1
    snapshot_candidates_path = (
        harvest_dir
        / state_conditioned_snapshot_harvest.OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME
    )
    _write_jsonl(snapshot_candidates_path, snapshot_candidates)
    _write_json(
        harvest_dir / state_conditioned_snapshot_harvest.FEASIBILITY_REPORT_JSON_NAME,
        {
            "schema_version": state_conditioned_snapshot_harvest.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_snapshot_feasibility_report",
            "mode": "feasibility",
            "family_order": list(state_conditioned_snapshot_harvest.T8_FAMILY_ORDER),
            "snapshot_candidates_path": str(snapshot_candidates_path.resolve()),
        },
    )
    _write_json(
        harvest_dir / state_conditioned_snapshot_harvest.TEACHER_GATE_REPORT_JSON_NAME,
        {
            "schema_version": state_conditioned_snapshot_harvest.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_teacher_gate_report",
            "mode": "feasibility",
            "family_order": list(state_conditioned_snapshot_harvest.T8_FAMILY_ORDER),
            "families": teacher_gate_families,
        },
    )
    _write_json(
        harvest_dir
        / state_conditioned_snapshot_harvest.LOCAL_RECOVERY_PSEUDODEMO_MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_snapshot_harvest.SCHEMA_VERSION,
            "artifact_kind": "local_recovery_pseudodemo_manifest",
            "mode": "formal",
            "history_k": state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
            "teacher_version": "scripted_teacher_v1",
            "snapshot_candidates_path": str(snapshot_candidates_path.resolve()),
            "family_order": list(state_conditioned_snapshot_harvest.T8_FAMILY_ORDER),
            "producer_by_family": producer_by_family,
            "counts": {
                "successful_pseudodemo_count": len(pseudodemos),
                "selected_pseudodemo_count_by_family": {
                    family: 6
                    for family in state_conditioned_snapshot_harvest.T8_FAMILY_ORDER
                },
            },
            "pseudodemos": pseudodemos,
        },
    )
    return harvest_dir


def _build_full_fixture(
    tmp_path: Path,
    *,
    contamination_field: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    bucket_dir = _build_bucket_a_fixture(
        tmp_path,
        contamination_field=contamination_field,
    )
    dev_dir = _build_dev_fixture(tmp_path)
    collection_dir = _build_bucket_b_fixture(tmp_path)
    harvest_dir = _build_harvest_fixture(tmp_path)
    return bucket_dir, dev_dir, collection_dir, harvest_dir


def _install_versioned_harvest_variant_with_teacher_provenance(
    harvest_dir: Path,
) -> Path:
    root_manifest_path = (
        harvest_dir
        / state_conditioned_snapshot_harvest.LOCAL_RECOVERY_PSEUDODEMO_MANIFEST_JSON_NAME
    )
    root_manifest = _read_json(root_manifest_path)
    variant_dir = (
        harvest_dir / state_conditioned_build_training_set.PSEUDODEMO_DATASET_VERSION
    )
    variant_dir.mkdir(parents=True, exist_ok=True)

    variant_manifest = json.loads(json.dumps(root_manifest))
    for record in variant_manifest["pseudodemos"]:
        if (
            record["producer"]
            == state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER
        ):
            record["teacher_target_truthfulness"] = (
                state_conditioned_snapshot_harvest.TEACHER_TARGET_TRUTHFUL_REAL_ROLLOUT
            )
            record["teacher_target"] = {
                "trace_episode_id": record["episode_id"],
                "trace_t_range": list(record["recovery_suffix_source_t_range"]),
                "producer": str(record["teacher_version"]),
                "synthetic_observation_only_backfill": False,
            }
        else:
            record["teacher_target_truthfulness"] = (
                state_conditioned_snapshot_harvest.TEACHER_TARGET_NOT_APPLICABLE
            )
            record.pop("teacher_target", None)

    stripped_root_manifest = json.loads(json.dumps(root_manifest))
    for record in stripped_root_manifest["pseudodemos"]:
        record.pop("teacher_target", None)
        record.pop("teacher_target_truthfulness", None)

    _write_json(root_manifest_path, stripped_root_manifest)
    _write_json(
        variant_dir
        / state_conditioned_snapshot_harvest.LOCAL_RECOVERY_PSEUDODEMO_MANIFEST_JSON_NAME,
        variant_manifest,
    )
    for artifact_name in (
        state_conditioned_snapshot_harvest.FEASIBILITY_REPORT_JSON_NAME,
        state_conditioned_snapshot_harvest.TEACHER_GATE_REPORT_JSON_NAME,
    ):
        _write_json(
            variant_dir / artifact_name,
            _read_json(harvest_dir / artifact_name),
        )
    source_snapshot_path = (
        harvest_dir
        / state_conditioned_snapshot_harvest.OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME
    )
    _write_jsonl(
        variant_dir
        / state_conditioned_snapshot_harvest.OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME,
        _read_jsonl(source_snapshot_path),
    )
    return variant_dir


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        state_conditioned_build_training_set.main(["--help"])
    assert exc_info.value.code == 0


def test_happy_path_builds_equal_data_views_and_dev_only_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket_dir, dev_dir, collection_dir, harvest_dir = _build_full_fixture(tmp_path)
    output_dir = tmp_path / "training"
    _install_fake_lerobot_export(monkeypatch)

    result = (
        state_conditioned_build_training_set.materialize_state_conditioned_training_set(
            bucket_dir=bucket_dir,
            dev_dir=dev_dir,
            collection_dir=collection_dir,
            harvest_dir=harvest_dir,
            output_dir=output_dir,
        )
    )

    labels = _read_jsonl(
        output_dir
        / state_conditioned_build_training_set.STATE_CONDITIONED_SFT_LABELS_JSONL_NAME
    )
    stats = _read_json(
        output_dir
        / state_conditioned_build_training_set.STATE_CONDITIONED_SFT_STATS_JSON_NAME
    )
    fairness = _read_json(
        output_dir
        / state_conditioned_build_training_set.EQUAL_DATA_FAIRNESS_AUDIT_JSON_NAME
    )
    liveness = _read_json(
        output_dir
        / state_conditioned_build_training_set.CONDITIONING_CHANNEL_LIVENESS_JSON_NAME
    )
    promotion_gate = _read_json(
        output_dir
        / state_conditioned_build_training_set.DEV_ONLY_PROMOTION_GATE_JSON_NAME
    )
    lerobot_dataset_path = Path(result["lerobot_dataset_path"])
    lerobot_info = _read_json(lerobot_dataset_path / "meta" / "info.json")
    lerobot_modality = _read_json(lerobot_dataset_path / "meta" / "modality.json")
    lerobot_video_map = _read_json(lerobot_dataset_path / "meta" / "video_map.json")
    lerobot_videos = list(
        (lerobot_dataset_path / "videos").glob(
            "chunk-*/observation.images.ego_view/episode_*.mp4"
        )
    )

    assert Path(result["state_conditioned_sft_labels_path"]).is_file()
    assert Path(result["state_conditioned_sft_stats_path"]).is_file()
    assert lerobot_dataset_path.is_dir()
    assert stats["lerobot_dataset_path"] == str(lerobot_dataset_path)
    assert (lerobot_dataset_path / "meta" / "info.json").is_file()
    assert (lerobot_dataset_path / "meta" / "episodes.jsonl").is_file()
    assert (lerobot_dataset_path / "meta" / "tasks.jsonl").is_file()
    assert (lerobot_dataset_path / "meta" / "modality.json").is_file()
    assert (lerobot_dataset_path / "meta" / "video_map.json").is_file()
    assert list((lerobot_dataset_path / "data").glob("chunk-*/*.parquet"))
    assert lerobot_modality["video"]["ego_view"]["original_key"] == (
        "observation.images.ego_view"
    )
    assert lerobot_info["video_path"] == (
        "videos/chunk-{episode_chunk:03d}/observation.images.ego_view/episode_{episode_index:06d}.mp4"
    )
    assert int(lerobot_info["total_videos"]) == int(lerobot_info["total_episodes"])
    assert int(lerobot_info["total_videos"]) > 0
    assert lerobot_info["features"]["observation.images.ego_view"]["dtype"] == "video"
    assert len(lerobot_videos) == int(lerobot_info["total_videos"])
    assert int(lerobot_video_map["length_clamp"]["clamped_episodes"]) == 0
    assert fairness["overall_pass"] is True
    assert liveness["overall_pass"] is True

    c0_rows = [
        row
        for row in labels
        if row["training_view"] == state_conditioned_build_training_set.VIEW_C0
    ]
    c1_rows = [
        row
        for row in labels
        if row["training_view"] == state_conditioned_build_training_set.VIEW_C1
    ]
    assert c0_rows
    assert len(c0_rows) == len(c1_rows)
    assert [row["sample_id"] for row in c0_rows] == [
        row["sample_id"] for row in c1_rows
    ]
    assert [row["sample_index"] for row in c0_rows] == [
        row["sample_index"] for row in c1_rows
    ]
    assert [row["source_bucket"] for row in c0_rows] == [
        row["source_bucket"] for row in c1_rows
    ]
    assert [row["budget_group"] for row in c0_rows] == [
        row["budget_group"] for row in c1_rows
    ]
    assert [row["history_valid_mask"] for row in c0_rows] == [
        row["history_valid_mask"] for row in c1_rows
    ]
    assert [row["deployable.previous_action_history"] for row in c0_rows] == [
        row["deployable.previous_action_history"] for row in c1_rows
    ]
    assert [row["deployable.proprio_history"] for row in c0_rows] == [
        row["deployable.proprio_history"] for row in c1_rows
    ]
    assert [row["deployable.short_visual_history_refs"] for row in c0_rows] == [
        row["deployable.short_visual_history_refs"] for row in c1_rows
    ]

    assert all(
        row["policy_condition.phase"]
        == state_conditioned_build_training_set.NULL_PHASE_TOKEN
        for row in c0_rows
    )
    assert all(
        row["policy_condition.mode"]
        == state_conditioned_build_training_set.NULL_MODE_TOKEN
        for row in c0_rows
    )
    assert all(
        row["policy_condition_text"]
        == state_conditioned_build_training_set.build_null_policy_condition_text()
        for row in c0_rows
    )
    assert all(
        row["policy_condition.phase"]
        in state_conditioned_bucket_a_import.STATE_CONDITIONED_PHASES
        for row in c1_rows
    )
    assert all(
        row["policy_condition.mode"]
        in state_conditioned_bucket_a_import.STATE_CONDITIONED_MODES
        for row in c1_rows
    )

    for row in c1_rows[:10]:
        assert row[
            "policy_condition_text"
        ] == state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
            row["policy_condition.phase"],
            row["policy_condition.mode"],
        )
        assert "event" not in row
        assert "semantic_state" not in row
        assert "memory_commit_mask" not in row
        assert "privileged.apple_visible" not in row

    assert stats["recovery_oversample_factor_min"] == 3
    assert stats["recovery_oversample_factor_max"] == 3
    assert (
        fairness["comparisons"]["baseline_to_c0"]["focus"]
        == "baseline_to_c0_short_history_memory"
    )
    assert fairness["comparisons"]["c0_to_c1"]["same_sample_ids"] is True
    assert liveness["same_non_conditioning_payload"] is True
    assert liveness["counts"]["c0_null_phase_count"] == len(c0_rows)
    assert liveness["counts"]["c1_non_null_phase_count"] == len(c1_rows)
    assert (
        stats["counts"]["lerobot_episode_count"]
        == stats["counts"]["unified_base_row_count"]
    )
    assert stats["counts"]["lerobot_frame_count"] == (
        stats["counts"]["unified_base_row_count"] * TEST_ACTION_HORIZON
    )

    assert promotion_gate["promotion_allowed"] is False
    assert (
        "teacher_assisted_formal_pseudodemos_remain_dev_only"
        in promotion_gate["failure_reasons"]
    )
    assert result["promotion_allowed"] is False


def test_builder_v2_flat_artifacts_remain_t11_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket_dir, dev_dir, collection_dir, harvest_dir = _build_full_fixture(tmp_path)
    output_dir = tmp_path / "training_v2_contract"
    _install_fake_lerobot_export(monkeypatch)

    state_conditioned_build_training_set.materialize_state_conditioned_training_set(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        collection_dir=collection_dir,
        harvest_dir=harvest_dir,
        output_dir=output_dir,
    )
    contract = state_conditioned_train._load_training_set_contract(output_dir)
    labels = _read_jsonl(
        output_dir
        / state_conditioned_build_training_set.STATE_CONDITIONED_SFT_LABELS_JSONL_NAME
    )
    formal_rows = [
        row
        for row in labels
        if row["source_bucket"]
        == state_conditioned_build_training_set.SOURCE_BUCKET_FORMAL_PSEUDODEMO
    ]

    assert contract["counts"]["c0_rows"] == contract["counts"]["c1_rows"]
    assert contract["counts"]["c0_rows"] > 0
    assert contract["deployable_observation_allowlist"] == list(
        lerobot_v2_export.DEPLOYABLE_HISTORY_FIELD_NAMES
    )
    assert all(
        row["reset_boundary"]
        == state_conditioned_bucket_a_import.STATE_CONDITIONED_RESET_BOUNDARY
        for row in labels
    )
    assert all(
        row["label_data.domain"]
        == state_conditioned_build_training_set.LABEL_DATA_DOMAIN
        for row in labels
    )
    assert all(
        row["label_data.version"]
        == state_conditioned_build_training_set.LABEL_DATA_VERSION
        for row in labels
    )
    assert all(
        row["label_data.m2_backfill_source"]
        == state_conditioned_build_training_set.M2_BACKFILL_SOURCE
        for row in labels
    )
    assert all(
        row["label_data.m2_backfill_version"]
        == state_conditioned_build_training_set.M2_BACKFILL_VERSION
        for row in labels
    )
    for row in labels:
        for field_name in state_conditioned_build_training_set.M2_FIELD_NAMES:
            assert field_name in row
        assert row["indicator_I"] in (0, 1)

    assert formal_rows
    for row in formal_rows:
        family = row["pseudodemo.source_snapshot_family"]
        assert family in state_conditioned_snapshot_harvest.T8_FAMILY_ORDER
        assert (
            row["pseudodemo.source_bucket_key"]
            == state_conditioned_snapshot_harvest.FAMILY_SOURCE_BUCKET_BY_FAMILY[family]
        )
        assert row["pseudodemo.label_kind"] == "formal_pseudodemo"
        assert (
            row["pseudodemo.dataset_version"]
            == state_conditioned_build_training_set.PSEUDODEMO_DATASET_VERSION
        )
        assert row["source_anchor_episode_id"].startswith("bucket_c_anchor_")
        if row["pseudodemo.teacher_target"] is None:
            assert (
                row["pseudodemo.teacher_target_truthfulness"]
                == state_conditioned_snapshot_harvest.TEACHER_TARGET_NOT_APPLICABLE
            )
        else:
            assert (
                row["pseudodemo.teacher_target_truthfulness"]
                == state_conditioned_snapshot_harvest.TEACHER_TARGET_TRUTHFUL_REAL_ROLLOUT
            )
            assert (
                row["pseudodemo.teacher_target"]["producer"]
                == row["pseudodemo.teacher_policy_id"]
            )
            assert (
                row["pseudodemo.teacher_target"]["synthetic_observation_only_backfill"]
                is False
            )


def test_c0_c1_pairs_keep_shared_backfill_and_provenance_aligned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket_dir, dev_dir, collection_dir, harvest_dir = _build_full_fixture(tmp_path)
    output_dir = tmp_path / "training_pair_alignment"
    _install_fake_lerobot_export(monkeypatch)

    state_conditioned_build_training_set.materialize_state_conditioned_training_set(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        collection_dir=collection_dir,
        harvest_dir=harvest_dir,
        output_dir=output_dir,
    )
    labels = _read_jsonl(
        output_dir
        / state_conditioned_build_training_set.STATE_CONDITIONED_SFT_LABELS_JSONL_NAME
    )
    c0_rows = [
        row
        for row in labels
        if row["training_view"] == state_conditioned_build_training_set.VIEW_C0
    ]
    c1_rows = [
        row
        for row in labels
        if row["training_view"] == state_conditioned_build_training_set.VIEW_C1
    ]
    c0_by_sample_id = {row["sample_id"]: row for row in c0_rows}
    c1_by_sample_id = {row["sample_id"]: row for row in c1_rows}

    assert list(c0_by_sample_id) == list(c1_by_sample_id)
    assert len(c0_by_sample_id) == len(c1_by_sample_id)
    shared_field_names = [
        field_name
        for field_name in c0_rows[0].keys()
        if field_name
        not in {
            "training_view",
            *state_conditioned_build_training_set.CONDITIONING_FIELD_NAMES,
        }
    ]
    for sample_id, c0_row in c0_by_sample_id.items():
        c1_row = c1_by_sample_id[sample_id]
        for field_name in shared_field_names:
            assert c0_row[field_name] == c1_row[field_name], (
                sample_id,
                field_name,
                c0_row[field_name],
                c1_row[field_name],
            )


def test_versioned_harvest_variant_preserves_formal_teacher_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket_dir, dev_dir, collection_dir, harvest_dir = _build_full_fixture(tmp_path)
    _install_fake_lerobot_export(monkeypatch)
    _install_versioned_harvest_variant_with_teacher_provenance(harvest_dir)
    output_dir = (
        tmp_path / state_conditioned_build_training_set.PSEUDODEMO_DATASET_VERSION
    )

    state_conditioned_build_training_set.materialize_state_conditioned_training_set(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        collection_dir=collection_dir,
        harvest_dir=harvest_dir,
        output_dir=output_dir,
    )
    labels = _read_jsonl(
        output_dir
        / state_conditioned_build_training_set.STATE_CONDITIONED_SFT_LABELS_JSONL_NAME
    )
    formal_rows = [
        row
        for row in labels
        if row["source_bucket"]
        == state_conditioned_build_training_set.SOURCE_BUCKET_FORMAL_PSEUDODEMO
    ]

    assert formal_rows
    truthful_rows = [
        row for row in formal_rows if row["pseudodemo.teacher_target"] is not None
    ]
    assert truthful_rows
    for row in truthful_rows:
        assert (
            row["pseudodemo.teacher_target_truthfulness"]
            == state_conditioned_snapshot_harvest.TEACHER_TARGET_TRUTHFUL_REAL_ROLLOUT
        )
        assert (
            row["pseudodemo.teacher_target"]["producer"]
            == row["pseudodemo.teacher_policy_id"]
        )
        assert (
            row["pseudodemo.teacher_target"]["synthetic_observation_only_backfill"]
            is False
        )


def test_rejects_non_allowlisted_deployable_leakage(tmp_path: Path) -> None:
    bucket_dir, dev_dir, collection_dir, harvest_dir = _build_full_fixture(
        tmp_path,
        contamination_field="deployable.teacher_hint",
    )

    with pytest.raises(ValueError, match=r"unexpected deployable field\(s\) leaked"):
        state_conditioned_build_training_set.materialize_state_conditioned_training_set(
            bucket_dir=bucket_dir,
            dev_dir=dev_dir,
            collection_dir=collection_dir,
            harvest_dir=harvest_dir,
            output_dir=tmp_path / "training_leak",
        )


def test_dev_only_promotion_gate_rejects_any_failed_gate() -> None:
    gate = state_conditioned_build_training_set.build_dev_only_promotion_gate(
        fairness_audit={"overall_pass": False},
        boundary_summary={"pass": True},
        liveness_audit={"overall_pass": True},
        recovery_oversample_factor=3,
        teacher_assisted_sources_present=False,
        legacy_debug_only_reuse_present=False,
    )

    assert gate["promotion_allowed"] is False
    assert gate["checks"]["fairness_pass"] is False
    assert "fairness_gate_failed" in gate["failure_reasons"]
