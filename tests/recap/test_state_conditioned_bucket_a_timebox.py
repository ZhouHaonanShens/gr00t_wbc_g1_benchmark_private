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


def _build_sidecar_row(
    episode_id: str,
    t: int,
    *,
    leak_semantic_field: bool = False,
) -> dict[str, Any]:
    history = _build_history_payload(episode_id, t)
    valid_mask = list(history["history_valid_mask"])
    row = {
        "episode_id": episode_id,
        "t": int(t),
        **history,
        "policy_condition.phase": "SEARCH",
        "policy_condition.mode": "NOMINAL",
        "policy_condition_text": state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
            "SEARCH", "NOMINAL"
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
    }
    if leak_semantic_field:
        row["deployable.semantic_state"] = "SEARCHING"
    return row


def test_policy_condition_text_uses_exact_frozen_multiline_contract() -> None:
    expected = "[PolicyCondition-v1]\nPHASE=SEARCH\nMODE=NOMINAL"
    assert (
        state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
            "search",
            "nominal",
        )
        == expected
    )
    assert (
        _build_sidecar_row("episode_text_contract", 0)["policy_condition_text"]
        == expected
    )


def test_missing_live_collection_labels_are_materialized_before_gate_reads_dataset(
    tmp_path: Path,
) -> None:
    fresh_result = _create_collection_result(
        tmp_path,
        dataset_name="fresh_missing_labels",
        episode_id="fresh_missing_labels_episode",
        write_labels=False,
    )
    dataset_dir = Path(fresh_result["dataset_dir"])
    labels_path = dataset_dir / state_conditioned_bucket_a_import.LABELS_REL_PATH

    assert not labels_path.exists()

    materialization = (
        state_conditioned_bucket_a_import.ensure_required_m2_labels_materialized(
            dataset_dir
        )
    )
    records = _read_jsonl(labels_path)

    assert materialization["materialized"] is True
    assert labels_path.is_file()
    assert len(records) == 3
    assert {record["episode_id"] for record in records} == {
        "fresh_missing_labels_episode"
    }
    assert all("prompt_conditioned" in record for record in records)
    loaded = state_conditioned_bucket_a_import._load_dataset_records(dataset_dir)
    assert set(loaded["labels_by_episode"]) == {"fresh_missing_labels_episode"}


def test_missing_live_collection_sidecar_is_materialized_before_gate_reads_dataset(
    tmp_path: Path,
) -> None:
    fresh_result = _create_collection_result(
        tmp_path,
        dataset_name="fresh_missing_sidecar_materialization",
        episode_id="fresh_missing_sidecar_episode",
        include_sidecar=False,
    )
    dataset_dir = Path(fresh_result["dataset_dir"])
    sidecar_path = (
        dataset_dir / state_conditioned_bucket_a_import.SIDECAR_CANDIDATE_NAMES[0]
    )

    assert not sidecar_path.exists()

    materialization = state_conditioned_bucket_a_import.ensure_required_history_aware_sidecar_materialized(
        dataset_dir
    )
    sidecar_rows = _read_jsonl(sidecar_path)

    assert materialization["materialized"] is True
    assert sidecar_path.is_file()
    assert len(sidecar_rows) == 3
    assert {row["episode_id"] for row in sidecar_rows} == {
        "fresh_missing_sidecar_episode"
    }
    assert all(
        row["policy_condition_text"]
        == "[PolicyCondition-v1]\nPHASE=SEARCH\nMODE=NOMINAL"
        for row in sidecar_rows
    )
    assert all("privileged.apple_pose_world" in row for row in sidecar_rows)
    assert all("privileged.hand_to_apple_rel_pose" in row for row in sidecar_rows)
    assert all("privileged.apple_to_plate_rel_pose" in row for row in sidecar_rows)
    assert all("privileged.contact_flag" in row for row in sidecar_rows)
    assert all("privileged.apple_in_hand" in row for row in sidecar_rows)
    assert all("privileged.apple_visible" in row for row in sidecar_rows)
    assert all("privileged.last_seen_dt" in row for row in sidecar_rows)
    assert all("privileged.last_in_hand_dt" in row for row in sidecar_rows)
    loaded = state_conditioned_bucket_a_import._load_dataset_records(dataset_dir)
    assert set(loaded["sidecar_by_episode"]) == {"fresh_missing_sidecar_episode"}


def test_missing_live_collection_semantic_metadata_is_materialized_before_acceptance(
    tmp_path: Path,
) -> None:
    fresh_result = _create_collection_result(
        tmp_path,
        dataset_name="fresh_missing_semantic_metadata",
        episode_id="fresh_missing_semantic_metadata_episode",
    )
    dataset_dir = Path(fresh_result["dataset_dir"])
    episodes_path = dataset_dir / "episodes.jsonl"
    episode_record = _read_jsonl(episodes_path)[0]
    episode_record.pop("metadata", None)
    _write_jsonl(episodes_path, [episode_record])

    materialization = state_conditioned_bucket_a_import.ensure_required_semantic_commit_metadata_materialized(
        dataset_dir
    )
    rewritten_record = _read_jsonl(episodes_path)[0]
    analysis_only = rewritten_record["metadata"]["analysis_only"]

    assert materialization["materialized"] is True
    assert materialization["materialized_episode_ids"] == [
        "fresh_missing_semantic_metadata_episode"
    ]
    assert analysis_only["semantic_state"] == "APPLE_VISIBLE_APPROACH"
    assert analysis_only["memory_commit_mask"] == [True, False, True]
    assert analysis_only["memory_commit_cause"] == "nominal_visual_confirmation"
    state_conditioned_bucket_a_import._validate_semantic_commit_metadata(
        rewritten_record,
        [
            _build_sidecar_row("fresh_missing_semantic_metadata_episode", t)
            for t in range(3)
        ],
    )


def test_fresh_only_run_clears_stale_reuse_artifacts_before_current_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_dir = _create_source_fixture(tmp_path, total_episodes=1)
    output_dir = tmp_path / "bucket_a_stale_reuse"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME,
        {
            "canonical_source": {
                "materialization_mode": "existing_live_dataset_reuse",
                "reused_existing_live_dataset": True,
                "source_dataset_dir": "/tmp/pilot_long_none_smoke_20260316",
            },
            "reused_existing_live_dataset": True,
        },
    )
    _write_json(
        output_dir / state_conditioned_bucket_a_import.GATE_A_READY_JSON_NAME,
        {"ready": False, "reused_existing_live_dataset": True},
    )
    _write_json(
        output_dir
        / state_conditioned_bucket_a_import.LEGACY_TIMEBOX_DECISION_JSON_NAME,
        {
            "canonical_source": {
                "materialization_mode": "existing_live_dataset_reuse",
                "reused_existing_live_dataset": True,
                "source_dataset_dir": "/tmp/pilot_long_none_smoke_20260316",
            }
        },
    )
    _write_json(
        output_dir
        / state_conditioned_bucket_a_import.DEBUG_ONLY_REUSE_MANIFEST_JSON_NAME,
        {"reused_existing_live_dataset": True},
    )

    monkeypatch.setattr(
        state_conditioned_bucket_a_import,
        "collect_fresh_nominal_episode_materialization",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic current-run blocker")
        ),
    )
    monkeypatch.setattr(
        state_conditioned_bucket_a_import,
        "discover_debug_only_reuse_materialization",
        lambda **_kwargs: None,
    )

    exit_code = state_conditioned_bucket_a_import.main(
        [
            "--source",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--fresh-only",
            "--accept-until",
            "24",
            "--debug-demote-reuse",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "synthetic current-run blocker" in captured.err
    assert not (
        output_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME
    ).exists()
    assert not (
        output_dir / state_conditioned_bucket_a_import.GATE_A_READY_JSON_NAME
    ).exists()
    assert not (
        output_dir / state_conditioned_bucket_a_import.LEGACY_TIMEBOX_DECISION_JSON_NAME
    ).exists()
    assert not (
        output_dir
        / state_conditioned_bucket_a_import.DEBUG_ONLY_REUSE_MANIFEST_JSON_NAME
    ).exists()


def _build_episode_metadata(*, valid: bool) -> dict[str, Any]:
    analysis_only = {
        "semantic_state": "APPLE_VISIBLE_APPROACH",
        "memory_commit_mask": [True, False, True],
        "memory_commit_cause": "nominal_visual_confirmation",
    }
    if not valid:
        analysis_only.pop("memory_commit_cause")
    return {"analysis_only": analysis_only}


def _create_source_fixture(
    tmp_path: Path,
    *,
    total_episodes: int,
    steps_per_episode: int = 3,
) -> Path:
    source_dir = tmp_path / "source_dataset"
    episodes: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    for episode_index in range(total_episodes):
        episode_id = f"legacy_episode_{episode_index:03d}"
        episodes.append(
            {
                "episode_id": episode_id,
                "seed": int(episode_index),
                "success_episode": bool(episode_index % 2 == 0),
                "n_policy_steps": int(steps_per_episode),
                "npz_path": f"arrays/{episode_id}.npz",
                "schema_version": "recap-v0",
                "code_version": "test-fixture",
                "iter_tag": "legacy_fixture",
                "prompt_raw": "pick up the apple and place it on the plate",
                "prompt_conditioned": "pick up the apple and place it on the plate",
                "env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
            }
        )
        for t in range(steps_per_episode):
            transitions.append(
                {
                    "episode_id": episode_id,
                    "t": int(t),
                    "reward_online": 1.0,
                    "n_action_steps_executed": 1,
                    "inner_rewards": [1.0],
                    "inner_dones": [False],
                    "success_step": False,
                }
            )
            labels.append(
                {
                    "episode_id": episode_id,
                    "t": int(t),
                    "prompt_conditioned": "advantage neutral pick up the apple",
                }
            )

    _write_jsonl(source_dir / "episodes.jsonl", episodes)
    _write_jsonl(source_dir / "transitions.jsonl", transitions)
    _write_jsonl(source_dir / "m2_labels" / "labels.jsonl", labels)
    return source_dir


def _create_collection_result(
    tmp_path: Path,
    *,
    dataset_name: str,
    episode_id: str,
    steps_per_episode: int = 3,
    label_count: int | None = None,
    include_sidecar: bool = True,
    write_labels: bool = True,
    metadata_valid: bool = True,
    leak_semantic_field: bool = False,
    reuse: bool = False,
) -> dict[str, Any]:
    dataset_dir = tmp_path / dataset_name
    label_count_value = steps_per_episode if label_count is None else int(label_count)

    episode_record = {
        "episode_id": episode_id,
        "seed": 7,
        "success_episode": True,
        "n_policy_steps": int(steps_per_episode),
        "npz_path": f"arrays/{episode_id}.npz",
        "schema_version": "recap-v0",
        "code_version": "test-fixture",
        "iter_tag": dataset_name,
        "prompt_raw": "fresh nominal collect apple to plate",
        "prompt_conditioned": "fresh nominal collect apple to plate",
        "env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
        "metadata": _build_episode_metadata(valid=metadata_valid),
    }
    transitions = [
        {
            "episode_id": episode_id,
            "t": int(t),
            "reward_online": 1.0,
            "n_action_steps_executed": 1,
            "inner_rewards": [1.0],
            "inner_dones": [False],
            "success_step": False,
        }
        for t in range(steps_per_episode)
    ]
    labels = [
        {
            "episode_id": episode_id,
            "t": int(t),
            "prompt_conditioned": "fresh nominal collect apple to plate",
        }
        for t in range(label_count_value)
    ]
    sidecar_rows = [
        _build_sidecar_row(
            episode_id,
            t,
            leak_semantic_field=leak_semantic_field,
        )
        for t in range(steps_per_episode)
    ]

    _write_jsonl(dataset_dir / "episodes.jsonl", [episode_record])
    _write_jsonl(dataset_dir / "transitions.jsonl", transitions)
    if write_labels:
        _write_jsonl(dataset_dir / "m2_labels" / "labels.jsonl", labels)
    if include_sidecar:
        _write_jsonl(dataset_dir / "state_conditioned_sidecar.jsonl", sidecar_rows)

    return {
        "iter_tag": dataset_name,
        "dataset_dir": str(dataset_dir),
        "episodes_path": str(dataset_dir / "episodes.jsonl"),
        "episode_order": [episode_id],
        "episodes_by_id": {episode_id: dict(episode_record)},
        "materialized_episode_count": 1,
        "collected_episode_count": 1,
        "collection_command": ["fixture"],
        "runtime_log_path": str(tmp_path / "runtime_logs" / f"{dataset_name}.log"),
        "materialization_mode": (
            "existing_live_dataset_reuse"
            if reuse
            else state_conditioned_bucket_a_import.CANONICAL_KIND
        ),
        "reused_existing_live_dataset": bool(reuse),
    }


def _create_reuse_materialization(
    tmp_path: Path,
    *,
    total_episodes: int,
) -> dict[str, Any]:
    dataset_dir = tmp_path / "reuse_dataset"
    episodes: list[dict[str, Any]] = []
    for episode_index in range(total_episodes):
        episode_id = f"reuse_episode_{episode_index:03d}"
        episodes.append(
            {
                "episode_id": episode_id,
                "seed": int(episode_index),
                "success_episode": True,
                "n_policy_steps": 3,
                "npz_path": f"arrays/{episode_id}.npz",
                "prompt_raw": "fresh nominal collect apple to plate",
                "prompt_conditioned": "fresh nominal collect apple to plate",
            }
        )
    _write_jsonl(dataset_dir / "episodes.jsonl", episodes)
    return {
        "iter_tag": "reuse_fixture",
        "dataset_dir": str(dataset_dir),
        "episodes_path": str(dataset_dir / "episodes.jsonl"),
        "episode_order": [record["episode_id"] for record in episodes],
        "episodes_by_id": {record["episode_id"]: dict(record) for record in episodes},
        "materialized_episode_count": int(len(episodes)),
        "collected_episode_count": int(len(episodes)),
        "collection_command": [],
        "runtime_log_path": str(tmp_path / "runtime_logs" / "reuse.log"),
        "materialization_mode": "existing_live_dataset_reuse",
        "reused_existing_live_dataset": True,
    }


def _install_collection_sequence(
    monkeypatch: pytest.MonkeyPatch,
    results: list[dict[str, Any]],
) -> None:
    iterator = iter(results)

    def _next_result(**_kwargs: Any) -> dict[str, Any]:
        try:
            return dict(next(iterator))
        except StopIteration as exc:
            raise AssertionError(
                "collection sequence exhausted before accept-until reached"
            ) from exc

    monkeypatch.setattr(
        state_conditioned_bucket_a_import,
        "collect_fresh_nominal_episode_materialization",
        _next_result,
    )


def _write_triplet(
    base_dir: Path,
    *,
    episode_id: str = "fresh_episode_999",
    provenance: dict[str, Any] | None = None,
    history_contract: dict[str, Any] | None = None,
    accepted: bool = True,
    sidecar_passed: bool = True,
    coverage_ratio: float = 1.0,
) -> tuple[Path, Path, Path]:
    provenance_value = provenance or {
        "kind": state_conditioned_bucket_a_import.CANONICAL_KIND,
        "source_dataset_dir": "/tmp/fresh_episode_999",
        "iter_tag": "fresh_episode_999",
        "materialization_mode": state_conditioned_bucket_a_import.CANONICAL_KIND,
        "fresh_nominal_recollection": True,
        "reused_existing_live_dataset": False,
    }
    history_value = history_contract or {
        "history_k": state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_K,
        "history_stride": state_conditioned_bucket_a_import.STATE_CONDITIONED_HISTORY_STRIDE,
        "reset_boundary": state_conditioned_bucket_a_import.STATE_CONDITIONED_RESET_BOUNDARY,
    }
    acceptance_path = base_dir / "acceptance.json"
    sidecar_path = base_dir / "sidecar_smoke.json"
    join_coverage_path = base_dir / "join_coverage.json"
    _write_json(
        acceptance_path,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "artifact_kind": "bucket_A_episode_acceptance",
            "episode_id": episode_id,
            "accepted": bool(accepted),
            "reject_reasons": [] if accepted else ["synthetic_reject"],
            "provenance": provenance_value,
            "history_contract": history_value,
        },
    )
    _write_json(
        sidecar_path,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "artifact_kind": "bucket_A_episode_sidecar_smoke",
            "episode_id": episode_id,
            "passed": bool(sidecar_passed),
            "status": "PASS" if sidecar_passed else "FAIL",
            "provenance": provenance_value,
            "history_contract": history_value,
            "coverage_ratio": float(coverage_ratio),
        },
    )
    _write_json(
        join_coverage_path,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "artifact_kind": "bucket_A_episode_join_coverage",
            "episode_id": episode_id,
            "passed": float(coverage_ratio)
            >= state_conditioned_bucket_a_import.JOIN_COVERAGE_THRESHOLD,
            "coverage_ratio": float(coverage_ratio),
            "provenance": provenance_value,
            "history_contract": history_value,
        },
    )
    return acceptance_path, sidecar_path, join_coverage_path


def test_main_accept_until_24_writes_ready_gate_and_demotes_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_dir = _create_source_fixture(tmp_path, total_episodes=3)
    reuse_materialization = _create_reuse_materialization(tmp_path, total_episodes=10)
    fresh_results = [
        _create_collection_result(
            tmp_path,
            dataset_name=f"fresh_dataset_{episode_index:03d}",
            episode_id=f"fresh_episode_{episode_index:03d}",
        )
        for episode_index in range(24)
    ]

    monkeypatch.setattr(
        state_conditioned_bucket_a_import,
        "discover_debug_only_reuse_materialization",
        lambda **_kwargs: dict(reuse_materialization),
    )
    _install_collection_sequence(monkeypatch, fresh_results)

    output_dir = tmp_path / "bucket_a_ready"
    exit_code = state_conditioned_bucket_a_import.main(
        [
            "--source",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--fresh-only",
            "--accept-until",
            "24",
            "--debug-demote-reuse",
        ]
    )

    assert exit_code == 0
    gate = _read_json(
        output_dir / state_conditioned_bucket_a_import.GATE_A_READY_JSON_NAME
    )
    manifest = _read_json(
        output_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME
    )
    debug_manifest = _read_json(
        output_dir
        / state_conditioned_bucket_a_import.DEBUG_ONLY_REUSE_MANIFEST_JSON_NAME
    )

    assert gate["ready"] is True
    assert gate["accepted_episode_count"] == 24
    assert gate["distinct_accepted_episode_count"] == 24
    assert manifest["selected_episode_count"] == 24
    assert manifest["target_episode_count"] == 24
    assert manifest["reused_existing_live_dataset"] is False
    assert manifest["canonical_source"]["reused_existing_live_dataset"] is False
    assert all(episode["accepted"] is True for episode in manifest["episodes"])
    assert all(
        episode["fresh_nominal_recollection"] is True
        for episode in manifest["episodes"]
    )
    assert all(
        Path(episode["acceptance_path"]).is_file()
        and Path(episode["sidecar_smoke_path"]).is_file()
        and Path(episode["join_coverage_path"]).is_file()
        for episode in manifest["episodes"]
    )

    assert debug_manifest["selected_episode_count"] == 10
    assert debug_manifest["all_debug_only"] is True
    assert debug_manifest["reused_existing_live_dataset"] is True
    assert all(episode["debug_only"] is True for episode in debug_manifest["episodes"])
    assert all(episode["accepted"] is False for episode in debug_manifest["episodes"])


def test_rejected_episodes_do_not_consume_canonical_quota(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_dir = _create_source_fixture(tmp_path, total_episodes=2)
    rejected_results = [
        _create_collection_result(
            tmp_path,
            dataset_name="reject_missing_sidecar",
            episode_id="fresh_reject_missing_sidecar",
            include_sidecar=False,
        ),
        _create_collection_result(
            tmp_path,
            dataset_name="reject_join_coverage",
            episode_id="fresh_reject_join_coverage",
            label_count=2,
        ),
        _create_collection_result(
            tmp_path,
            dataset_name="reject_missing_metadata",
            episode_id="fresh_reject_missing_metadata",
            metadata_valid=False,
        ),
    ]
    accepted_results = [
        _create_collection_result(
            tmp_path,
            dataset_name=f"accepted_dataset_{episode_index:03d}",
            episode_id=f"fresh_accept_{episode_index:03d}",
        )
        for episode_index in range(24)
    ]

    monkeypatch.setattr(
        state_conditioned_bucket_a_import,
        "discover_debug_only_reuse_materialization",
        lambda **_kwargs: None,
    )
    _install_collection_sequence(monkeypatch, rejected_results + accepted_results)

    output_dir = tmp_path / "bucket_a_rejections"
    result = state_conditioned_bucket_a_import.materialize_bucket_a(
        source_dir=source_dir,
        output_dir=output_dir,
        accept_until=24,
        fresh_only=True,
        debug_demote_reuse=True,
    )

    gate = result["gate"]
    manifest = result["manifest"]
    rejected_ids = {item["episode_id"] for item in gate["rejected_episode_attempts"]}

    assert gate["ready"] is True
    assert gate["accepted_episode_count"] == 24
    assert gate["total_collection_attempts"] == 27
    assert gate["rejected_episode_attempt_count"] == 3
    assert rejected_ids == {
        "fresh_reject_missing_sidecar",
        "fresh_reject_join_coverage",
        "fresh_reject_missing_metadata",
    }
    assert {episode["episode_id"] for episode in manifest["episodes"]}.isdisjoint(
        rejected_ids
    )

    missing_sidecar_acceptance = _read_json(
        output_dir
        / state_conditioned_bucket_a_import.EPISODE_ACCEPTANCE_DIRNAME
        / "fresh_reject_missing_sidecar.json"
    )
    assert missing_sidecar_acceptance["accepted"] is False
    assert "sidecar_smoke_failed" in missing_sidecar_acceptance["reject_reasons"]

    low_coverage_acceptance = _read_json(
        output_dir
        / state_conditioned_bucket_a_import.EPISODE_ACCEPTANCE_DIRNAME
        / "fresh_reject_join_coverage.json"
    )
    assert low_coverage_acceptance["accepted"] is False
    assert "join_coverage_below_threshold" in low_coverage_acceptance["reject_reasons"]

    metadata_acceptance = _read_json(
        output_dir
        / state_conditioned_bucket_a_import.EPISODE_ACCEPTANCE_DIRNAME
        / "fresh_reject_missing_metadata.json"
    )
    assert metadata_acceptance["accepted"] is False
    assert (
        "missing_analysis_only_semantic_commit_metadata"
        in metadata_acceptance["reject_reasons"]
    )


def test_validate_episode_gate_triplet_rejects_missing_and_mismatched_artifacts(
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "triplet_missing"
    acceptance_path, sidecar_path, join_coverage_path = _write_triplet(missing_dir)
    join_coverage_path.unlink()
    missing_result = state_conditioned_bucket_a_import.validate_episode_gate_triplet(
        acceptance_path=acceptance_path,
        sidecar_smoke_path=sidecar_path,
        join_coverage_path=join_coverage_path,
    )
    assert missing_result["accepted_for_canonical_quota"] is False
    assert "missing_join_coverage_artifact" in missing_result["reject_reasons"]

    provenance_dir = tmp_path / "triplet_provenance"
    acceptance_path, sidecar_path, join_coverage_path = _write_triplet(provenance_dir)
    sidecar_payload = _read_json(sidecar_path)
    sidecar_payload["provenance"]["iter_tag"] = "other_iter_tag"
    _write_json(sidecar_path, sidecar_payload)
    provenance_result = state_conditioned_bucket_a_import.validate_episode_gate_triplet(
        acceptance_path=acceptance_path,
        sidecar_smoke_path=sidecar_path,
        join_coverage_path=join_coverage_path,
    )
    assert provenance_result["accepted_for_canonical_quota"] is False
    assert "provenance_mismatch" in provenance_result["reject_reasons"]

    history_dir = tmp_path / "triplet_history"
    acceptance_path, sidecar_path, join_coverage_path = _write_triplet(history_dir)
    join_payload = _read_json(join_coverage_path)
    join_payload["history_contract"]["history_k"] = 99
    _write_json(join_coverage_path, join_payload)
    history_result = state_conditioned_bucket_a_import.validate_episode_gate_triplet(
        acceptance_path=acceptance_path,
        sidecar_smoke_path=sidecar_path,
        join_coverage_path=join_coverage_path,
    )
    assert history_result["accepted_for_canonical_quota"] is False
    assert "history_contract_mismatch" in history_result["reject_reasons"]


def test_missing_valid_visual_history_ref_rejected() -> None:
    row = _build_sidecar_row("episode_999", 2)
    row["deployable.short_visual_history_refs"][-1] = None

    with pytest.raises((TypeError, ValueError), match="short_visual_history_refs"):
        state_conditioned_bucket_a_import.validate_sidecar_row_for_gate(row)


def test_invalid_output_dir_fails_cleanly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = state_conditioned_bucket_a_import.main(
        ["--output-dir", ".sisyphus/bucket_A_manifest.json"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "output-dir must be a directory path" in captured.err
    assert "Traceback" not in captured.err
