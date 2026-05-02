from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import state_conditioned_bucket_a_import
from work.recap.scripts import state_conditioned_bucket_a_sidecar
from work.recap.scripts import state_conditioned_collect_buckets
from work.recap.scripts import state_conditioned_dev_manifest
from work.recap.scripts import state_conditioned_snapshot_harvest


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


def _history_payload(episode_id: str, t: int, *, valid: bool = True) -> dict[str, Any]:
    valid_mask: list[bool] = []
    history_episode_ids: list[str] = []
    prehistory_window: list[dict[str, Any]] = []
    start_t = int(t) - (
        state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K - 1
    )
    for index in range(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K):
        candidate_t = start_t + index
        is_valid = valid and candidate_t >= 0
        row_t = candidate_t if is_valid else 0
        history_episode_ids.append(episode_id)
        valid_mask.append(bool(is_valid))
        prehistory_window.append(
            {
                "episode_id": episode_id,
                "t_std": int(row_t),
                "mujoco_state_ref": f"mujoco://{episode_id}/{int(row_t)}",
            }
        )
    if not valid:
        prehistory_window[-1]["episode_id"] = f"other_{episode_id}"
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


def _deployable_history_payload(episode_id: str, t: int) -> dict[str, Any]:
    history = _history_payload(episode_id, t)
    valid_mask = list(history["history_valid_mask"])
    history_k = state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K
    previous_action_history: list[Any] = []
    proprio_history: list[Any] = []
    short_visual_history_refs: list[Any] = []
    for index in range(history_k):
        if not valid_mask[index]:
            previous_action_history.append(None)
            proprio_history.append(None)
            short_visual_history_refs.append(None)
            continue
        slot_t = int(t) - (history_k - 1) + index
        previous_action_history.append([float(slot_t), float(t), episode_id])
        proprio_history.append([float(slot_t) + 0.5, float(t)])
        short_visual_history_refs.append(f"video://{episode_id}/{slot_t}")
    return {
        "deployable.previous_action_history": previous_action_history,
        "deployable.proprio_history": proprio_history,
        "deployable.short_visual_history_refs": short_visual_history_refs,
    }


def _policy_steps(
    *,
    visible_true_streak: int = 0,
    in_hand_true_streak: int = 0,
    contact_at: int | None = None,
    place_at: int | None = None,
    success_episode_step: int | None = None,
    xy_distance_values: list[float] | None = None,
    total_steps: int = 8,
    generic_success_step: bool = False,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    distances = (
        list(xy_distance_values)
        if xy_distance_values is not None
        else [0.3] * total_steps
    )
    if len(distances) != total_steps:
        raise AssertionError("xy_distance_values length mismatch")
    for index in range(total_steps):
        phase = "PLACE" if place_at is not None and index >= place_at else "TRANSPORT"
        steps.append(
            {
                "success_step": bool(generic_success_step),
                "success_episode": bool(
                    success_episode_step is not None and index >= success_episode_step
                ),
                "policy_condition.phase": phase,
                "privileged.apple_visible": bool(index < visible_true_streak),
                "privileged.apple_in_hand": bool(index < in_hand_true_streak),
                "privileged.contact_flag": bool(
                    contact_at is not None and index >= contact_at
                ),
                "privileged.apple_to_plate_rel_pose": [
                    float(distances[index]),
                    0.0,
                    0.0,
                ],
            }
        )
    return steps


def _build_candidate(
    *,
    family: str,
    snapshot_index: int,
    deprioritized_by_plan: bool = False,
    valid_history: bool = True,
) -> dict[str, Any]:
    episode_id = f"{family.lower()}_episode_{snapshot_index:03d}"
    t_value = 20 + int(snapshot_index)
    return {
        "family": family,
        "snapshot_id": f"{family}_{snapshot_index:03d}",
        "anchor_t": int(t_value),
        **_history_payload(episode_id, t_value, valid=valid_history),
        "policy_condition.phase": "TRANSPORT" if family != "S_pre_place" else "PLACE",
        "policy_condition.mode": "RECOVERY",
        "policy_condition_text": state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
            "PLACE" if family == "S_pre_place" else "TRANSPORT",
            "RECOVERY",
        ),
        "privileged.apple_to_plate_rel_pose": [0.30, 0.00, 0.00],
        "source_bucket_key": state_conditioned_snapshot_harvest.FAMILY_SOURCE_BUCKET_BY_FAMILY[
            family
        ],
        "deprioritized_by_plan": bool(deprioritized_by_plan),
    }


def _family_attempts(
    family: str, *, success_count: int, total_attempts: int
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for index in range(total_attempts):
        seed = index % len(state_conditioned_snapshot_harvest.SNAPSHOT_SEED_VALUES)
        if family == "S_lost":
            success_steps = _policy_steps(
                visible_true_streak=4 if index < success_count else 3, total_steps=4
            )
        elif family == "S_drop":
            success_steps = _policy_steps(
                in_hand_true_streak=8 if index < success_count else 7, total_steps=8
            )
        elif family == "S_transport_mid":
            distances = [0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.24]
            if index >= success_count:
                distances = [0.30, 0.295, 0.292, 0.291, 0.290, 0.289, 0.288, 0.287]
            success_steps = _policy_steps(
                in_hand_true_streak=8 if index < success_count else 7,
                xy_distance_values=distances,
                total_steps=8,
            )
        else:
            success_steps = _policy_steps(
                contact_at=1 if index < success_count else None,
                place_at=1 if index < success_count else None,
                total_steps=4,
            )
        attempts.append(
            {
                "seed": int(seed),
                "policy_steps": success_steps,
                "success_episode": bool(
                    family == "S_pre_place" and index < success_count
                ),
            }
        )
    return attempts


def _build_grouped_candidates() -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for family in state_conditioned_snapshot_harvest.T8_FAMILY_ORDER:
        grouped[family] = {
            "eligible": [],
            "ineligible": [],
            "deprioritized_flags": set(),
        }
    for family in state_conditioned_snapshot_harvest.T8_FAMILY_ORDER:
        snapshot_count = (
            6
            if family in state_conditioned_snapshot_harvest.HIGH_PRIORITY_FAMILIES
            else 2
        )
        for snapshot_index in range(snapshot_count):
            attempt_success_count = 3
            if family == "S_lost" and snapshot_index >= 2:
                attempt_success_count = 0
            if family == "S_transport_mid":
                attempt_success_count = 2 if snapshot_index == 0 else 1
            if family == "S_pre_place":
                attempt_success_count = 1 if snapshot_index == 0 else 0
            candidate = _build_candidate(
                family=family,
                snapshot_index=snapshot_index,
                deprioritized_by_plan=family
                in state_conditioned_snapshot_harvest.LOW_PRIORITY_FAMILIES,
            )
            candidate["expected_success_count"] = int(attempt_success_count)
            grouped[family]["eligible"].append(candidate)
            grouped[family]["deprioritized_flags"].add(
                bool(candidate["deprioritized_by_plan"])
            )
        if family == "S_lost":
            is_eligible, payload, error = (
                state_conditioned_snapshot_harvest._candidate_validation_result(
                    _build_candidate(
                        family=family,
                        snapshot_index=999,
                        valid_history=False,
                    )
                )
            )
            assert is_eligible is False
            assert payload is not None
            grouped[family]["ineligible"].append(
                {
                    "snapshot_id": str(payload["snapshot_id"]),
                    "reason": str(error),
                }
            )
            grouped[family]["deprioritized_flags"].add(False)
    return grouped


def _fake_feasibility_runner(
    candidate: Mapping[str, Any],
    seed: int,
    family: str,
) -> dict[str, Any]:
    success_count = int(candidate.get("expected_success_count", 0))
    if family == "S_lost":
        success_steps = _policy_steps(
            visible_true_streak=4 if int(seed) < success_count else 3,
            total_steps=4,
        )
    elif family == "S_drop":
        success_steps = _policy_steps(
            in_hand_true_streak=8 if int(seed) < success_count else 7,
            total_steps=8,
        )
    elif family == "S_transport_mid":
        distances = [0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.24]
        if int(seed) >= success_count:
            distances = [0.30, 0.295, 0.292, 0.291, 0.290, 0.289, 0.288, 0.287]
        success_steps = _policy_steps(
            in_hand_true_streak=8 if int(seed) < success_count else 7,
            xy_distance_values=distances,
            total_steps=8,
        )
    else:
        success_steps = _policy_steps(
            contact_at=1 if int(seed) < success_count else None,
            place_at=1 if int(seed) < success_count else None,
            total_steps=4,
        )
    return {
        "seed": int(seed),
        "policy_steps": success_steps,
        "success_episode": bool(family == "S_pre_place" and int(seed) < success_count),
    }


def _write_default_snapshot_candidates_source(collection_dir: Path) -> Path:
    grouped: dict[str, dict[str, Any]] = {
        family: {"eligible": [], "ineligible": [], "deprioritized_flags": set()}
        for family in state_conditioned_snapshot_harvest.T8_FAMILY_ORDER
    }
    success_counts = {
        "S_drop": 3,
        "S_lost": 3,
        "S_transport_mid": 2,
        "S_pre_place": 1,
    }
    for family in state_conditioned_snapshot_harvest.T8_FAMILY_ORDER:
        for snapshot_index in range(6):
            candidate = _build_candidate(
                family=family,
                snapshot_index=snapshot_index,
                deprioritized_by_plan=family
                in state_conditioned_snapshot_harvest.LOW_PRIORITY_FAMILIES,
            )
            grouped[family]["eligible"].append(candidate)
            grouped[family]["deprioritized_flags"].add(
                bool(candidate["deprioritized_by_plan"])
            )
    path = (
        collection_dir
        / state_conditioned_snapshot_harvest.OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME
    )
    _write_jsonl(
        path,
        state_conditioned_snapshot_harvest._flatten_snapshot_candidates_for_artifact(
            grouped
        ),
    )
    return path


def _write_replayable_dataset_episode(
    dataset_dir: Path, *, episode_id: str, seed: int
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        dataset_dir / "episodes.jsonl",
        [
            {
                "episode_id": episode_id,
                "seed": int(seed),
                "env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
                "model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
                "embodiment_tag": "UNITREE_G1",
                "success_episode": False,
                "n_policy_steps": 4,
                "npz_path": f"arrays/{episode_id}.npz",
            }
        ],
    )
    _write_jsonl(
        dataset_dir / "state_conditioned_sidecar.jsonl",
        [
            {
                "episode_id": episode_id,
                "anchor_episode_id": episode_id,
                "anchor_t": int(t),
                "t": int(t),
                **_history_payload(episode_id, int(t)),
                **_deployable_history_payload(episode_id, int(t)),
                "policy_condition.phase": "SEARCH",
                "policy_condition.mode": "NOMINAL",
                "policy_condition_text": state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
                    "SEARCH",
                    "NOMINAL",
                ),
                "privileged.apple_visible": True,
                "privileged.apple_in_hand": False,
                "privileged.contact_flag": False,
                "privileged.apple_to_plate_rel_pose": [0.30, 0.0, 0.0],
            }
            for t in range(4)
        ],
    )
    _write_jsonl(
        dataset_dir / "transitions.jsonl",
        [{"episode_id": episode_id, "t": t} for t in range(4)],
    )
    arrays_dir = dataset_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        arrays_dir / f"{episode_id}.npz",
        **{
            "action/left_arm": np.zeros((4, 1, 30, 7), dtype=np.float32),
            "action/right_arm": np.zeros((4, 1, 30, 7), dtype=np.float32),
            "action/left_hand": np.zeros((4, 1, 30, 7), dtype=np.float32),
            "action/right_hand": np.zeros((4, 1, 30, 7), dtype=np.float32),
            "action/waist": np.zeros((4, 1, 30, 3), dtype=np.float32),
            "action/navigate_command": np.zeros((4, 1, 30, 3), dtype=np.float32),
            "action/base_height_command": np.zeros((4, 1, 30, 1), dtype=np.float32),
        },
    )


def _build_prerequisites(tmp_path: Path) -> tuple[Path, Path, Path]:
    bucket_dir = tmp_path / "bucket_a"
    dev_dir = tmp_path / "devbench"
    collection_dir = tmp_path / "collection"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    dev_dir.mkdir(parents=True, exist_ok=True)
    collection_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        bucket_dir / state_conditioned_bucket_a_import.GATE_A_READY_JSON_NAME,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "bucket_key": state_conditioned_bucket_a_import.BUCKET_KEY,
            "ready": True,
            "required_distinct_accepted_episode_count": 24,
            "accepted_episode_count": 24,
            "distinct_accepted_episode_count": 24,
        },
    )
    _write_json(
        bucket_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "bucket_key": state_conditioned_bucket_a_import.BUCKET_KEY,
            "required_distinct_episode_count": 24,
            "episodes": [
                {
                    "episode_id": f"bucket_a_sidecar_episode_{index:03d}",
                    "accepted": True,
                    "debug_only": False,
                    "reused_existing_live_dataset": False,
                    "source_dataset_dir": str(
                        (tmp_path / "recap_datasets" / f"dataset_{index:03d}").resolve()
                    ),
                    "npz_path": f"arrays/bucket_a_sidecar_episode_{index:03d}.npz",
                }
                for index in range(24)
            ],
        },
    )
    _write_jsonl(
        bucket_dir / state_conditioned_bucket_a_sidecar.BUCKET_A_SIDECAR_JSON_NAME,
        [
            {
                "episode_id": f"bucket_a_sidecar_episode_{index:03d}",
                "anchor_episode_id": f"bucket_a_sidecar_episode_{index:03d}",
                "anchor_t": 20 + index,
                "t": 20 + index,
                **_history_payload(f"bucket_a_sidecar_episode_{index:03d}", 20 + index),
                "policy_condition.phase": "SEARCH",
                "policy_condition.mode": "NOMINAL",
                "policy_condition_text": state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
                    "SEARCH",
                    "NOMINAL",
                ),
                "privileged.apple_visible": True,
                "privileged.apple_in_hand": bool(index % 2 == 0),
                "privileged.contact_flag": False,
                "privileged.apple_to_plate_rel_pose": [0.30, 0.0, 0.0],
                "recovery_entry_step": 1,
                "recovery_exit_step": 3,
            }
            for index in range(12)
        ],
    )
    for index in range(24):
        dataset_dir = tmp_path / "recap_datasets" / f"dataset_{index:03d}"
        _write_replayable_dataset_episode(
            dataset_dir,
            episode_id=f"bucket_a_sidecar_episode_{index:03d}",
            seed=index,
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
        },
    )

    _write_json(
        dev_dir / state_conditioned_dev_manifest.FIXED_STRATA_DEFINITION_JSON_NAME,
        {
            "schema_version": state_conditioned_dev_manifest.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_dev_fixed_strata_definition",
            "paired_seed_count": 8,
            "paired_seed_values": list(
                state_conditioned_dev_manifest.DEFAULT_PAIRED_SEEDS
            ),
        },
    )
    _write_json(
        dev_dir / state_conditioned_dev_manifest.BASELINE_MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_dev_manifest.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_dev_baseline_manifest",
            "counts": {"entries": 32, "paired_seed_count": 8, "per_stratum": {}},
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

    _write_json(
        collection_dir / state_conditioned_collect_buckets.BUCKET_B_MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_collect_buckets.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_bucket_B_manifest",
            "counts": {"episodes": 16},
            "episodes": [
                {
                    "episode_id": f"bucket_b_episode_{index:03d}",
                    "seed": 1000 + index,
                    "dataset_dir": str(
                        (
                            tmp_path
                            / "recap_datasets"
                            / f"bucket_b_dataset_{index:03d}"
                        ).resolve()
                    ),
                }
                for index in range(16)
            ],
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
            "episodes": [
                {
                    "episode_id": f"drop_during_transport_episode_{index:03d}",
                    "seed": 2000 + index,
                    "failure_injection_kind": "drop_during_transport",
                    "failure_injection_trigger_t": 3,
                    "dataset_dir": str(
                        (
                            tmp_path
                            / "recap_datasets"
                            / f"bucket_c_drop_dataset_{index:03d}"
                        ).resolve()
                    ),
                }
                for index in range(8)
            ]
            + [
                {
                    "episode_id": f"failed_grasp_visible_episode_{index:03d}",
                    "seed": 3000 + index,
                    "failure_injection_kind": "failed_grasp_visible",
                    "failure_injection_trigger_t": 3,
                    "dataset_dir": str(
                        (
                            tmp_path
                            / "recap_datasets"
                            / f"bucket_c_visible_dataset_{index:03d}"
                        ).resolve()
                    ),
                }
                for index in range(8)
            ]
            + [
                {
                    "episode_id": f"failed_grasp_occluded_episode_{index:03d}",
                    "seed": 4000 + index,
                    "failure_injection_kind": "failed_grasp_occluded",
                    "failure_injection_trigger_t": 3,
                    "dataset_dir": str(
                        (
                            tmp_path
                            / "recap_datasets"
                            / f"bucket_c_occluded_dataset_{index:03d}"
                        ).resolve()
                    ),
                }
                for index in range(8)
            ],
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
    for index in range(16):
        _write_replayable_dataset_episode(
            tmp_path / "recap_datasets" / f"bucket_b_dataset_{index:03d}",
            episode_id=f"bucket_b_episode_{index:03d}",
            seed=1000 + index,
        )
    for index in range(8):
        _write_replayable_dataset_episode(
            tmp_path / "recap_datasets" / f"bucket_c_drop_dataset_{index:03d}",
            episode_id=f"drop_during_transport_episode_{index:03d}",
            seed=2000 + index,
        )
        _write_replayable_dataset_episode(
            tmp_path / "recap_datasets" / f"bucket_c_visible_dataset_{index:03d}",
            episode_id=f"failed_grasp_visible_episode_{index:03d}",
            seed=3000 + index,
        )
        _write_replayable_dataset_episode(
            tmp_path / "recap_datasets" / f"bucket_c_occluded_dataset_{index:03d}",
            episode_id=f"failed_grasp_occluded_episode_{index:03d}",
            seed=4000 + index,
        )
    return bucket_dir, dev_dir, collection_dir


def _family_row(payload: dict[str, Any], family: str) -> dict[str, Any]:
    return next(row for row in payload["families"] if row["family"] == family)


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        state_conditioned_snapshot_harvest.main(["--help"])
    assert exc_info.value.code == 0


def test_bad_output_path_fails_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = state_conditioned_snapshot_harvest.main(
        ["--output-dir", ".sisyphus/snapshot_feasibility_report.json"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "output-dir must be a directory path" in captured.err
    assert "Traceback" not in captured.err


def test_missing_t7_prerequisite_fails_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bucket_dir, dev_dir, collection_dir = _build_prerequisites(tmp_path)
    (
        collection_dir / state_conditioned_collect_buckets.BUCKET_C_MANIFEST_JSON_NAME
    ).unlink()

    exit_code = state_conditioned_snapshot_harvest.main(
        [
            "--bucket-dir",
            str(bucket_dir),
            "--dev-dir",
            str(dev_dir),
            "--collection-dir",
            str(collection_dir),
            "--output-dir",
            str(tmp_path / "harvest"),
            "--snapshot-candidates",
            str(tmp_path / "missing.jsonl"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "missing required T7 bucket_C_manifest_path" in captured.err
    assert "Traceback" not in captured.err


def test_materialize_snapshot_feasibility_happy_path_writes_reports(
    tmp_path: Path,
) -> None:
    bucket_dir, dev_dir, collection_dir = _build_prerequisites(tmp_path)
    output_dir = tmp_path / "harvest"

    result = state_conditioned_snapshot_harvest.materialize_snapshot_feasibility(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        collection_dir=collection_dir,
        output_dir=output_dir,
        grouped_snapshot_candidates=_build_grouped_candidates(),
        feasibility_runner=_fake_feasibility_runner,
    )

    feasibility_report = _read_json(
        output_dir / state_conditioned_snapshot_harvest.FEASIBILITY_REPORT_JSON_NAME
    )
    teacher_gate_report = _read_json(
        output_dir / state_conditioned_snapshot_harvest.TEACHER_GATE_REPORT_JSON_NAME
    )

    assert Path(result["snapshot_feasibility_report_path"]).is_file()
    assert Path(result["teacher_gate_report_path"]).is_file()
    assert (
        feasibility_report["artifact_kind"]
        == "state_conditioned_snapshot_feasibility_report"
    )
    assert (
        teacher_gate_report["artifact_kind"] == "state_conditioned_teacher_gate_report"
    )

    drop_row = _family_row(feasibility_report, "S_drop")
    lost_row = _family_row(feasibility_report, "S_lost")
    transport_row = _family_row(feasibility_report, "S_transport_mid")
    pre_place_row = _family_row(feasibility_report, "S_pre_place")
    lost_gate = _family_row(teacher_gate_report, "S_lost")
    transport_gate = _family_row(teacher_gate_report, "S_transport_mid")

    assert drop_row["attempt_count"] == 18
    assert lost_row["attempt_count"] == 18
    assert (
        drop_row["success_criteria"]
        == state_conditioned_snapshot_harvest.FAMILY_SUCCESS_CRITERIA["S_drop"]
    )
    assert (
        lost_row["success_criteria"]
        == state_conditioned_snapshot_harvest.FAMILY_SUCCESS_CRITERIA["S_lost"]
    )

    assert transport_row["deprioritized_by_plan"] is True
    assert pre_place_row["deprioritized_by_plan"] is True
    assert transport_row["attempt_count"] == 6
    assert pre_place_row["attempt_count"] == 6

    assert lost_row["ineligible_candidate_count"] == 1
    assert "history" in lost_row["ineligible_candidates"][0]["reason"]

    assert lost_gate["success_count"] == 6
    assert pytest.approx(lost_gate["success_rate"], rel=1e-6) == (6.0 / 18.0)
    assert lost_gate["teacher_fallback_enabled"] is False
    assert transport_gate["teacher_fallback_enabled"] is False


def test_cli_default_snapshot_candidates_path_synthesizes_real_candidates_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bucket_dir, dev_dir, collection_dir = _build_prerequisites(tmp_path)
    output_dir = tmp_path / "harvest"

    replay_calls: list[str] = []

    def _fake_replay_attempt(
        candidate: dict[str, Any],
        *,
        seed: int,
        family: str,
    ) -> dict[str, Any]:
        replay_calls.append(str(candidate["snapshot_id"]))
        return {
            "seed": int(seed),
            "policy_steps": _policy_steps(total_steps=4),
            "success_episode": False,
            "source_episode_id": candidate["anchor_episode_id"],
        }

    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_run_replay_based_feasibility_attempt",
        _fake_replay_attempt,
    )

    feasibility_exit_code = state_conditioned_snapshot_harvest.main(
        [
            "--mode",
            "feasibility",
            "--bucket-dir",
            str(bucket_dir),
            "--dev-dir",
            str(dev_dir),
            "--collection-dir",
            str(collection_dir),
            "--output-dir",
            str(output_dir),
            "--history-k",
            "8",
            "--teacher-threshold",
            "0.15",
        ]
    )

    feasibility_captured = capsys.readouterr()
    assert feasibility_exit_code == 0, feasibility_captured.err
    assert feasibility_captured.err == ""
    persisted_candidates_path = (
        output_dir
        / state_conditioned_snapshot_harvest.OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME
    )
    assert persisted_candidates_path.is_file()
    persisted_candidates = [
        json.loads(line)
        for line in persisted_candidates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert persisted_candidates
    assert all("attempts" not in row for row in persisted_candidates)
    assert replay_calls
    feasibility_report = _read_json(
        output_dir / state_conditioned_snapshot_harvest.FEASIBILITY_REPORT_JSON_NAME
    )
    report_families = {row["family"]: row for row in feasibility_report["families"]}
    assert set(report_families["S_drop"]["selected_source_bucket_keys"]) == {"bucket_C"}
    assert set(report_families["S_lost"]["selected_source_bucket_keys"]) == {"bucket_C"}
    assert set(report_families["S_transport_mid"]["selected_source_bucket_keys"]) == {
        "bucket_B"
    }
    assert set(report_families["S_pre_place"]["selected_source_bucket_keys"]) == {
        "bucket_B"
    }


def test_cli_default_snapshot_candidates_path_supports_real_source_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bucket_dir, dev_dir, collection_dir = _build_prerequisites(tmp_path)
    output_dir = tmp_path / "harvest"
    source_path = _write_default_snapshot_candidates_source(collection_dir)

    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_run_replay_based_feasibility_attempt",
        lambda candidate, *, seed, family: {
            "seed": int(seed),
            "policy_steps": _policy_steps(total_steps=4),
            "success_episode": False,
        },
    )

    feasibility_exit_code = state_conditioned_snapshot_harvest.main(
        [
            "--mode",
            "feasibility",
            "--bucket-dir",
            str(bucket_dir),
            "--dev-dir",
            str(dev_dir),
            "--collection-dir",
            str(collection_dir),
            "--output-dir",
            str(output_dir),
            "--history-k",
            "8",
            "--teacher-threshold",
            "0.15",
        ]
    )

    feasibility_captured = capsys.readouterr()
    assert feasibility_exit_code == 0, feasibility_captured.err
    assert feasibility_captured.err == ""
    persisted_candidates_path = (
        output_dir
        / state_conditioned_snapshot_harvest.OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME
    )
    assert persisted_candidates_path.is_file()
    feasibility_report = _read_json(
        output_dir / state_conditioned_snapshot_harvest.FEASIBILITY_REPORT_JSON_NAME
    )
    assert feasibility_report["snapshot_candidates_source_path"] == str(
        source_path.resolve()
    )
    assert feasibility_report["snapshot_candidates_path"] == str(
        persisted_candidates_path.resolve()
    )


def test_synthesized_real_candidates_mark_low_priority_families_deprioritized(
    tmp_path: Path,
) -> None:
    bucket_dir, _dev_dir, collection_dir = _build_prerequisites(tmp_path)

    grouped = state_conditioned_snapshot_harvest.synthesize_snapshot_candidates_from_real_artifacts(
        bucket_dir=bucket_dir,
        collection_dir=collection_dir,
    )

    for family in state_conditioned_snapshot_harvest.HIGH_PRIORITY_FAMILIES:
        assert grouped[family]["eligible"]
        assert all(
            bool(candidate["deprioritized_by_plan"]) is False
            for candidate in grouped[family]["eligible"]
        )
    for family in state_conditioned_snapshot_harvest.LOW_PRIORITY_FAMILIES:
        assert grouped[family]["eligible"]
        assert all(
            bool(candidate["deprioritized_by_plan"]) is True
            for candidate in grouped[family]["eligible"]
        )


def test_synthesized_real_candidates_use_family_specific_sources(
    tmp_path: Path,
) -> None:
    bucket_dir, _dev_dir, collection_dir = _build_prerequisites(tmp_path)

    grouped = state_conditioned_snapshot_harvest.synthesize_snapshot_candidates_from_real_artifacts(
        bucket_dir=bucket_dir,
        collection_dir=collection_dir,
    )

    drop_candidates = grouped["S_drop"]["eligible"]
    lost_candidates = grouped["S_lost"]["eligible"]
    transport_candidates = grouped["S_transport_mid"]["eligible"]
    pre_place_candidates = grouped["S_pre_place"]["eligible"]

    assert all(
        candidate["source_bucket_key"] == "bucket_C" for candidate in drop_candidates
    )
    assert all(
        candidate["source_failure_injection_kind"] == "drop_during_transport"
        for candidate in drop_candidates
    )
    assert all(
        candidate["source_bucket_key"] == "bucket_C" for candidate in lost_candidates
    )
    assert {
        candidate["source_failure_injection_kind"] for candidate in lost_candidates
    } <= {"failed_grasp_visible", "failed_grasp_occluded"}
    assert all(
        candidate["source_bucket_key"] == "bucket_B"
        for candidate in transport_candidates
    )
    assert all(
        candidate["source_bucket_key"] == "bucket_B"
        for candidate in pre_place_candidates
    )
    assert all(candidate["anchor_t"] == 3 for candidate in drop_candidates)
    assert all(candidate["anchor_t"] == 3 for candidate in lost_candidates)
    assert all(candidate["anchor_t"] == 2 for candidate in transport_candidates)
    assert all(candidate["anchor_t"] == 3 for candidate in pre_place_candidates)
    assert {candidate["anchor_episode_id"] for candidate in drop_candidates} != {
        candidate["anchor_episode_id"] for candidate in transport_candidates
    }


def test_synthesized_real_candidates_copy_deployable_history_payload_from_sidecar(
    tmp_path: Path,
) -> None:
    bucket_dir, _dev_dir, collection_dir = _build_prerequisites(tmp_path)

    grouped = state_conditioned_snapshot_harvest.synthesize_snapshot_candidates_from_real_artifacts(
        bucket_dir=bucket_dir,
        collection_dir=collection_dir,
    )

    candidate = grouped["S_drop"]["eligible"][0]
    dataset_dir = Path(candidate["source_dataset_dir"])
    sidecar_rows = state_conditioned_snapshot_harvest._dataset_sidecar_rows(
        dataset_dir,
        episode_id=candidate["anchor_episode_id"],
    )
    matched_row = next(
        row for row in sidecar_rows if int(row["t"]) == int(candidate["anchor_t"])
    )

    for field_name in state_conditioned_snapshot_harvest.DEPLOYABLE_HISTORY_FIELD_NAMES:
        assert candidate[field_name] == matched_row[field_name]
        assert isinstance(candidate[field_name], list)


def test_candidate_validation_rejects_family_source_bucket_mismatch() -> None:
    candidate = _build_candidate(family="S_drop", snapshot_index=0)
    candidate["source_bucket_key"] = "bucket_B"

    is_eligible, payload, error = (
        state_conditioned_snapshot_harvest._candidate_validation_result(candidate)
    )

    assert is_eligible is False
    assert payload is not None
    assert payload["snapshot_id"] == candidate["snapshot_id"]
    assert error is not None
    assert "source_bucket_key mismatch" in error


def test_candidate_validation_trims_invalid_history_slots_from_deployable_payload() -> (
    None
):
    candidate = _build_candidate(family="S_lost", snapshot_index=0)
    candidate["anchor_t"] = 1
    candidate.update(_history_payload(candidate["anchor_episode_id"], 1))
    candidate.update(_deployable_history_payload(candidate["anchor_episode_id"], 1))
    candidate["deployable.previous_action_history"][0] = [999.0, 999.0, "stale"]
    candidate["deployable.proprio_history"][1] = [123.0, 456.0]
    candidate["deployable.short_visual_history_refs"][2] = "video://stale/2"

    is_eligible, payload, error = (
        state_conditioned_snapshot_harvest._candidate_validation_result(candidate)
    )

    assert is_eligible is True, error
    assert payload is not None
    valid_mask = list(payload["history_valid_mask"])
    for field_name in state_conditioned_snapshot_harvest.DEPLOYABLE_HISTORY_FIELD_NAMES:
        lane = list(payload[field_name])
        assert len(lane) == len(valid_mask)
        for index, is_valid in enumerate(valid_mask):
            if not is_valid:
                assert lane[index] is None


@pytest.mark.parametrize(
    ("family", "attempt_result", "candidate_overrides"),
    [
        (
            "S_lost",
            {
                "policy_steps": _policy_steps(visible_true_streak=4, total_steps=4),
                "success_episode": False,
            },
            {},
        ),
        (
            "S_drop",
            {
                "policy_steps": _policy_steps(in_hand_true_streak=8, total_steps=8),
                "success_episode": False,
            },
            {},
        ),
        (
            "S_transport_mid",
            {
                "policy_steps": _policy_steps(
                    in_hand_true_streak=8,
                    xy_distance_values=[0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.24],
                    total_steps=8,
                ),
                "success_episode": False,
            },
            {"anchor_xy_distance": 0.30},
        ),
        (
            "S_pre_place",
            {
                "policy_steps": _policy_steps(contact_at=1, place_at=1, total_steps=4),
                "success_episode": False,
            },
            {},
        ),
    ],
)
def test_each_family_specific_success_rule(
    family: str,
    attempt_result: dict[str, Any],
    candidate_overrides: dict[str, Any],
) -> None:
    candidate = _build_candidate(family=family, snapshot_index=0)
    candidate.update(candidate_overrides)

    assert (
        state_conditioned_snapshot_harvest.evaluate_feasibility_success(
            family=family,
            candidate=candidate,
            attempt_result=attempt_result,
        )
        is True
    )


@pytest.mark.parametrize(
    "family",
    ["S_lost", "S_drop", "S_transport_mid"],
)
def test_generic_success_semantics_do_not_decide_first_three_families(
    family: str,
) -> None:
    candidate = _build_candidate(family=family, snapshot_index=1)
    if family == "S_transport_mid":
        candidate["anchor_xy_distance"] = 0.30
    attempt_result = {
        "success_episode": True,
        "policy_steps": _policy_steps(generic_success_step=True, total_steps=4),
    }

    assert (
        state_conditioned_snapshot_harvest.evaluate_feasibility_success(
            family=family,
            candidate=candidate,
            attempt_result=attempt_result,
        )
        is False
    )


def test_teacher_gate_threshold_boundary() -> None:
    below = state_conditioned_snapshot_harvest.build_teacher_gate_decision(
        family="S_drop",
        attempt_count=20,
        success_count=2,
        threshold=0.15,
    )
    boundary = state_conditioned_snapshot_harvest.build_teacher_gate_decision(
        family="S_drop",
        attempt_count=20,
        success_count=3,
        threshold=0.15,
    )

    assert below["teacher_fallback_enabled"] is True
    assert pytest.approx(below["success_rate"], rel=1e-6) == 0.10
    assert boundary["teacher_fallback_enabled"] is False
    assert pytest.approx(boundary["success_rate"], rel=1e-6) == 0.15


def test_normalize_replay_action_chunk_for_env_squeezes_batch_dimension() -> None:
    action_chunk = {
        "action.left_arm": np.zeros((1, 30, 7), dtype=np.float32),
        "action.base_height_command": np.zeros((1, 30, 1), dtype=np.float32),
    }

    normalized = (
        state_conditioned_snapshot_harvest._normalize_replay_action_chunk_for_env(
            action_chunk
        )
    )

    assert list(normalized["action.left_arm"].shape) == [30, 7]
    assert list(normalized["action.base_height_command"].shape) == [30, 1]


def test_replay_attempt_uses_fresh_post_anchor_rollout_instead_of_source_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _build_candidate(family="S_lost", snapshot_index=0)
    candidate.update(
        {
            "source_episode_seed": 0,
            "source_dataset_dir": "/tmp/unused",
            "source_npz_path": "/tmp/unused.npz",
            "source_env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
            "source_model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
            "source_embodiment_tag": "UNITREE_G1",
        }
    )

    class _FakeEnv:
        def __init__(self) -> None:
            self.actions: list[dict[str, Any]] = []

        def reset(
            self, seed: int | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            return ({"annotation.human.task_description": "demo"}, {})

        def step(
            self, action: Mapping[str, Any]
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self.actions.append(dict(action))
            return (
                {"annotation.human.task_description": "demo"},
                0.0,
                False,
                False,
                {},
            )

        def close(self) -> None:
            return None

    fake_env = _FakeEnv()
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_load_replay_action_chunks",
        lambda _candidate: [
            {"action.left_arm": np.zeros((1, 30, 7), dtype=np.float32)}
        ],
    )
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_build_replay_env",
        lambda _candidate, n_action_steps: fake_env,
    )
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_collect_fresh_policy_rollout_after_anchor",
        lambda _candidate, *, env, obs: (
            [
                {
                    "t": 0,
                    "policy_condition.phase": "TRANSPORT",
                    "privileged.apple_visible": False,
                }
            ],
            False,
        ),
    )
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_dataset_sidecar_rows",
        lambda _dataset_dir, *, episode_id: [
            {
                "episode_id": episode_id,
                "t": 1,
                "privileged.apple_visible": True,
            }
        ],
    )

    attempt_result = (
        state_conditioned_snapshot_harvest._run_replay_based_feasibility_attempt(
            candidate,
            seed=0,
            family="S_lost",
        )
    )

    assert attempt_result["policy_steps"] == [
        {
            "t": 0,
            "policy_condition.phase": "TRANSPORT",
            "privileged.apple_visible": False,
        }
    ]
    assert fake_env.actions
    assert list(fake_env.actions[0]["action.left_arm"].shape) == [30, 7]
