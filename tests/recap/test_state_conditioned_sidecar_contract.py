from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.scripts import state_conditioned_bucket_a_import
from work.recap.scripts import state_conditioned_bucket_a_sidecar
from work.recap import lerobot_v2_export


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
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _build_history_payload(episode_id: str, t: int) -> dict[str, Any]:
    valid_mask: list[bool] = []
    prehistory_window: list[dict[str, Any]] = []
    history_episode_ids = [
        episode_id
    ] * state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K
    start_t = int(t) - (
        state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K - 1
    )
    for index in range(state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K):
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
        "history_k": state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
        "history_stride": state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_STRIDE,
        "history_valid_mask": valid_mask,
        "history_episode_ids": history_episode_ids,
        "anchor_episode_id": episode_id,
        "anchor_mujoco_state_ref": f"mujoco://{episode_id}/{int(t)}",
        "prehistory_window": prehistory_window,
        "reset_boundary": state_conditioned_bucket_a_import.STATE_CONDITIONED_RESET_BOUNDARY,
    }


def _build_base_sidecar_row(episode_id: str, t: int) -> dict[str, Any]:
    history = _build_history_payload(episode_id, t)
    valid_mask = list(history["history_valid_mask"])
    return {
        "episode_id": episode_id,
        "t": int(t),
        **history,
        "policy_condition.phase": "SEARCH",
        "policy_condition.mode": "NOMINAL",
        "policy_condition_text": state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
            "SEARCH", "NOMINAL"
        ),
        "event": "steady_nominal_progress",
        "recovery_needed": False,
        "deployable.previous_action_history": [
            None if not is_valid else [float(index), float(t), 0.5]
            for index, is_valid in enumerate(valid_mask)
        ],
        "deployable.proprio_history": [
            None if not is_valid else [0.1 * float(index), 0.2 * float(t)]
            for index, is_valid in enumerate(valid_mask)
        ],
        "deployable.short_visual_history_refs": [
            None if not is_valid else f"video://{episode_id}/{int(index)}"
            for index, is_valid in enumerate(valid_mask)
        ],
    }


def _build_episode_dataset(
    dataset_dir: Path,
    *,
    episode_id: str,
    steps_per_episode: int = 3,
) -> None:
    episode_record = {
        "episode_id": episode_id,
        "seed": 7,
        "success_episode": True,
        "n_policy_steps": int(steps_per_episode),
        "npz_path": f"arrays/{episode_id}.npz",
        "prompt_raw": "pick up the apple and place it on the plate",
        "prompt_conditioned": "pick up the apple and place it on the plate",
        "env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
        "metadata": {
            "analysis_only": {
                "semantic_state": "APPLE_VISIBLE_APPROACH",
                "memory_commit_mask": False,
                "memory_commit_cause": "nominal_visual_confirmation",
                "recovery_entry_step": None,
                "recovery_exit_step": None,
                "summary_template": "recovery/nominal_v1",
            }
        },
    }
    transitions: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    sidecar_rows: list[dict[str, Any]] = []
    for t in range(steps_per_episode):
        transitions.append(
            {
                "episode_id": episode_id,
                "t": int(t),
                "timestamp_s": 0.05 * float(t),
                "success_step": False,
                "privileged": {
                    "apple_pose_world": [1.0, 0.1 * float(t), 0.2, 0.0, 0.0, 0.0, 1.0],
                    "hand_to_apple_rel_pose": [
                        0.01 * float(t),
                        0.0,
                        0.02,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "apple_to_plate_rel_pose": [
                        0.3,
                        -0.1 * float(t),
                        0.1,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "contact_flag": bool(t >= 1),
                    "apple_in_hand": bool(t >= 1),
                    "apple_visible": True,
                    "last_seen_dt": 0.0,
                    "last_in_hand_dt": 0.0 if t >= 1 else 1.0,
                },
                "analysis_only": {
                    "semantic_state": "SEARCHING"
                    if t == 0
                    else "APPLE_VISIBLE_APPROACH",
                    "memory_commit_mask": bool(t == 1),
                    "memory_commit_cause": "nominal_visual_confirmation",
                },
            }
        )
        labels.append(
            {
                "episode_id": episode_id,
                "t": int(t),
                "prompt_conditioned": "pick up the apple and place it on the plate",
            }
        )
        sidecar_rows.append(_build_base_sidecar_row(episode_id, t))

    _write_jsonl(dataset_dir / "episodes.jsonl", [episode_record])
    _write_jsonl(dataset_dir / "transitions.jsonl", transitions)
    _write_jsonl(dataset_dir / "m2_labels" / "labels.jsonl", labels)
    _write_jsonl(dataset_dir / "state_conditioned_sidecar.jsonl", sidecar_rows)


def _build_bucket_fixture(
    tmp_path: Path, *, ready: bool
) -> tuple[Path, list[str], str]:
    bucket_dir = tmp_path / "bucket_a"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    accepted_episode_ids: list[str] = []
    manifest_episodes: list[dict[str, Any]] = []
    for episode_index in range(
        state_conditioned_bucket_a_sidecar.EXPECTED_ACCEPTED_EPISODE_COUNT
    ):
        episode_id = f"fresh_accept_{episode_index:03d}"
        dataset_dir = tmp_path / f"dataset_{episode_index:03d}"
        _build_episode_dataset(dataset_dir, episode_id=episode_id)
        accepted_episode_ids.append(episode_id)
        manifest_episodes.append(
            {
                "episode_id": episode_id,
                "accepted": True,
                "debug_only": False,
                "fresh_nominal_recollection": True,
                "reused_existing_live_dataset": False,
                "selection_reason": state_conditioned_bucket_a_import.CANONICAL_KIND,
                "source_dataset_dir": str(dataset_dir),
            }
        )

    ignored_episode_id = "debug_only_episode_999"
    ignored_dataset_dir = tmp_path / "debug_only_dataset"
    _build_episode_dataset(ignored_dataset_dir, episode_id=ignored_episode_id)
    manifest_episodes.append(
        {
            "episode_id": ignored_episode_id,
            "accepted": False,
            "debug_only": True,
            "fresh_nominal_recollection": False,
            "reused_existing_live_dataset": True,
            "selection_reason": "debug_only_reuse",
            "source_dataset_dir": str(ignored_dataset_dir),
        }
    )

    _write_json(
        bucket_dir / state_conditioned_bucket_a_import.GATE_A_READY_JSON_NAME,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "bucket_key": state_conditioned_bucket_a_import.BUCKET_KEY,
            "ready": bool(ready),
            "required_distinct_accepted_episode_count": state_conditioned_bucket_a_sidecar.EXPECTED_ACCEPTED_EPISODE_COUNT,
            "accepted_episode_count": state_conditioned_bucket_a_sidecar.EXPECTED_ACCEPTED_EPISODE_COUNT,
            "distinct_accepted_episode_count": state_conditioned_bucket_a_sidecar.EXPECTED_ACCEPTED_EPISODE_COUNT,
        },
    )
    _write_json(
        bucket_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "bucket_key": state_conditioned_bucket_a_import.BUCKET_KEY,
            "required_distinct_episode_count": state_conditioned_bucket_a_sidecar.EXPECTED_ACCEPTED_EPISODE_COUNT,
            "episodes": manifest_episodes,
        },
    )
    return bucket_dir, accepted_episode_ids, ignored_episode_id


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        state_conditioned_bucket_a_sidecar.main(["--help"])
    assert exc_info.value.code == 0


def test_gate_not_ready_fails_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bucket_dir, _, _ = _build_bucket_fixture(tmp_path, ready=False)

    exit_code = state_conditioned_bucket_a_sidecar.main(
        ["--bucket-dir", str(bucket_dir), "--output-dir", str(bucket_dir)]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "bucket_A_gate_a_ready.json.ready == true" in captured.err
    assert "Traceback" not in captured.err


def test_materialize_bucket_a_sidecar_happy_path_covers_only_canonical_accepts(
    tmp_path: Path,
) -> None:
    bucket_dir, accepted_episode_ids, ignored_episode_id = _build_bucket_fixture(
        tmp_path, ready=True
    )

    result = state_conditioned_bucket_a_sidecar.materialize_bucket_a_sidecar(
        bucket_dir=bucket_dir,
        output_dir=bucket_dir,
    )

    sidecar_path = (
        bucket_dir / state_conditioned_bucket_a_sidecar.BUCKET_A_SIDECAR_JSON_NAME
    )
    join_coverage_path = (
        bucket_dir / state_conditioned_bucket_a_sidecar.BUCKET_A_JOIN_COVERAGE_JSON_NAME
    )
    exporter_manifest_path = (
        bucket_dir
        / state_conditioned_bucket_a_sidecar.BUCKET_A_EXPORTER_MANIFEST_JSON_NAME
    )
    sidecar_rows = _read_jsonl(sidecar_path)
    join_coverage = _read_json(join_coverage_path)
    exporter_manifest = _read_json(exporter_manifest_path)

    assert result["accepted_episode_count"] == 24
    assert sidecar_rows
    assert {row["episode_id"] for row in sidecar_rows} == set(accepted_episode_ids)
    assert ignored_episode_id not in {row["episode_id"] for row in sidecar_rows}
    assert join_coverage["coverage_ratio"] >= 0.995
    assert join_coverage["accepted_episode_count"] == 24
    assert exporter_manifest["accepted_episode_count"] == 24

    row = sidecar_rows[0]
    assert row["history_k"] == 8
    assert row["history_stride"] == 1
    assert len(row["history_valid_mask"]) == 8
    assert len(row["history_t_std_indices"]) == 8
    assert len(row["history_t_raw_indices"]) == 8
    assert len(row["history_timestamp_s"]) == 8
    assert "deployable.previous_action_history" in row
    assert "deployable.proprio_history" in row
    assert "deployable.short_visual_history_refs" in row
    for field_name in state_conditioned_bucket_a_sidecar.PRIVILEGED_FIELD_NAMES:
        assert field_name in row
    for field_name in state_conditioned_bucket_a_sidecar.ANALYSIS_ONLY_FIELD_NAMES:
        assert field_name in row
    assert (
        row["semantic_state"]
        in state_conditioned_bucket_a_sidecar.SEMANTIC_STATE_VALUES
    )
    assert (
        row["memory_commit_cause"]
        in state_conditioned_bucket_a_sidecar.MEMORY_COMMIT_CAUSE_VALUES
    )
    assert row["summary_template"] == "recovery/nominal_v1"

    field_groups = exporter_manifest["field_groups"]
    assert set(field_groups.keys()) == {
        lerobot_v2_export.DEPLOYABLE_HISTORY_GROUP_KEY,
        lerobot_v2_export.PRIVILEGED_ANALYSIS_ONLY_GROUP_KEY,
        lerobot_v2_export.TEACHER_ONLY_GROUP_KEY,
    }
    assert set(field_groups[lerobot_v2_export.DEPLOYABLE_HISTORY_GROUP_KEY]) == set(
        lerobot_v2_export.DEPLOYABLE_HISTORY_FIELD_NAMES
    )
    assert set(
        field_groups[lerobot_v2_export.PRIVILEGED_ANALYSIS_ONLY_GROUP_KEY]
    ) == set(lerobot_v2_export.PRIVILEGED_ANALYSIS_ONLY_FIELD_NAMES)
    assert field_groups[lerobot_v2_export.TEACHER_ONLY_GROUP_KEY] == []


@pytest.mark.parametrize(
    "bad_field_name",
    [
        "privileged.apple_pose_world",
        "oracle.next_subgoal",
        "teacher.action_distribution",
        "hindsight.future_contact_flag",
        "semantic_state",
        "memory_commit_mask",
        "memory_commit_cause",
        "recovery_entry_step",
        "recovery_exit_step",
        "summary_template",
    ],
)
def test_validate_state_conditioned_field_groups_rejects_deployable_leakage(
    bad_field_name: str,
) -> None:
    with pytest.raises(ValueError, match="deployable_history"):
        lerobot_v2_export.validate_state_conditioned_field_groups(
            {
                lerobot_v2_export.DEPLOYABLE_HISTORY_GROUP_KEY: list(
                    lerobot_v2_export.DEPLOYABLE_HISTORY_FIELD_NAMES
                )
                + [bad_field_name],
                lerobot_v2_export.PRIVILEGED_ANALYSIS_ONLY_GROUP_KEY: list(
                    lerobot_v2_export.PRIVILEGED_ANALYSIS_ONLY_FIELD_NAMES
                ),
                lerobot_v2_export.TEACHER_ONLY_GROUP_KEY: [],
            }
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "error_fragment"),
    [
        ("semantic_state", "FREEFORM_STATE", "semantic_state must be one of"),
        (
            "memory_commit_cause",
            "user_wrote_long_sentence",
            "memory_commit_cause must be one of",
        ),
        (
            "summary_template",
            "this is already rendered natural language",
            "summary_template must be a template id or null",
        ),
    ],
)
def test_validate_consolidated_sidecar_row_enforces_closed_enums_and_template_ids(
    field_name: str,
    bad_value: object,
    error_fragment: str,
) -> None:
    row = _build_base_sidecar_row("enum_episode", 2)
    row.update(
        {
            "history_t_std_indices": list(range(8)),
            "history_t_raw_indices": list(range(8)),
            "history_timestamp_s": [None, None, None, None, None, 0.0, 0.1, 0.2],
            "privileged.apple_pose_world": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "privileged.hand_to_apple_rel_pose": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "privileged.apple_to_plate_rel_pose": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "privileged.contact_flag": True,
            "privileged.apple_in_hand": True,
            "privileged.apple_visible": True,
            "privileged.last_seen_dt": 0.0,
            "privileged.last_in_hand_dt": 0.0,
            "semantic_state": "SEARCHING",
            "memory_commit_mask": [False, False, False, False, False, True, True, True],
            "memory_commit_cause": "nominal_visual_confirmation",
            "recovery_entry_step": None,
            "recovery_exit_step": None,
            "summary_template": "recovery/nominal_v1",
        }
    )
    row[field_name] = bad_value

    with pytest.raises((TypeError, ValueError), match=error_fragment):
        state_conditioned_bucket_a_sidecar.validate_consolidated_sidecar_row(row)
