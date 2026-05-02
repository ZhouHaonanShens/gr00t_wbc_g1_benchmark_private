from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from collections.abc import Mapping

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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        history_episode_ids.append(episode_id)
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


def _policy_steps_for_family(family: str, *, success: bool) -> list[dict[str, Any]]:
    if family == "S_lost":
        total_steps = 5
        visible_from = 1 if success else None
        steps: list[dict[str, Any]] = []
        for t in range(total_steps):
            steps.append(
                {
                    "t": int(t),
                    "success_step": bool(success and t == 1),
                    "success_episode": False,
                    "policy_condition.phase": "TRANSPORT",
                    "privileged.apple_visible": bool(
                        visible_from is not None and t >= visible_from
                    ),
                    "privileged.apple_in_hand": False,
                    "privileged.contact_flag": False,
                    "privileged.apple_to_plate_rel_pose": [0.30, 0.0, 0.0],
                }
            )
        return steps

    if family == "S_drop":
        total_steps = 9
        in_hand_from = 1 if success else None
        steps = []
        for t in range(total_steps):
            steps.append(
                {
                    "t": int(t),
                    "success_step": bool(success and t == 1),
                    "success_episode": False,
                    "policy_condition.phase": "TRANSPORT",
                    "privileged.apple_visible": True,
                    "privileged.apple_in_hand": bool(
                        in_hand_from is not None and t >= in_hand_from
                    ),
                    "privileged.contact_flag": False,
                    "privileged.apple_to_plate_rel_pose": [0.30, 0.0, 0.0],
                }
            )
        return steps

    if family == "S_transport_mid":
        total_steps = 9
        if success:
            distances = [0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22]
            in_hand_values = [False, True, True, True, True, True, True, True, True]
        else:
            distances = [0.30, 0.295, 0.292, 0.291, 0.290, 0.289, 0.288, 0.287, 0.286]
            in_hand_values = [False, True, True, True, True, True, True, True, False]
        return [
            {
                "t": int(t),
                "success_step": bool(success and t == 1),
                "success_episode": False,
                "policy_condition.phase": "TRANSPORT",
                "privileged.apple_visible": True,
                "privileged.apple_in_hand": bool(in_hand_values[t]),
                "privileged.contact_flag": False,
                "privileged.apple_to_plate_rel_pose": [float(distances[t]), 0.0, 0.0],
            }
            for t in range(total_steps)
        ]

    total_steps = 4
    return [
        {
            "t": int(t),
            "success_step": bool(success and t == 1),
            "success_episode": bool(success and t >= 1),
            "policy_condition.phase": "PLACE" if success and t >= 1 else "TRANSPORT",
            "privileged.apple_visible": True,
            "privileged.apple_in_hand": bool(success and t >= 1),
            "privileged.contact_flag": bool(success and t >= 1),
            "privileged.apple_to_plate_rel_pose": [0.20, 0.0, 0.0],
        }
        for t in range(total_steps)
    ]


def _attempt_payload(
    family: str,
    *,
    seed: int,
    success: bool,
    teacher_rollout_kind: str | None = None,
    teacher_target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "seed": int(seed),
        "policy_steps": _policy_steps_for_family(family, success=success),
        "success_episode": bool(family == "S_pre_place" and success),
    }
    if teacher_rollout_kind is not None:
        payload["producer"] = (
            state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER
        )
        payload["teacher_rollout_kind"] = str(teacher_rollout_kind)
    if teacher_target is not None:
        payload["teacher_target"] = dict(teacher_target)
    return payload


def _build_candidate(
    *,
    family: str,
    snapshot_index: int,
    base_success_flat_indices: set[int],
    teacher_success_flat_indices: set[int],
) -> dict[str, Any]:
    episode_id = f"{family.lower()}_episode_{snapshot_index:03d}"
    t_value = 20 + int(snapshot_index)
    attempts: list[dict[str, Any]] = []
    scripted_teacher_attempts: list[dict[str, Any]] = []
    for local_seed_index, seed in enumerate(
        state_conditioned_snapshot_harvest.SNAPSHOT_SEED_VALUES
    ):
        flat_index = int(
            snapshot_index
            * len(state_conditioned_snapshot_harvest.SNAPSHOT_SEED_VALUES)
            + local_seed_index
        )
        attempts.append(
            _attempt_payload(
                family,
                seed=int(seed),
                success=flat_index in base_success_flat_indices,
            )
        )
        if teacher_success_flat_indices:
            scripted_teacher_attempts.append(
                _attempt_payload(
                    family,
                    seed=int(seed),
                    success=flat_index in teacher_success_flat_indices,
                    teacher_rollout_kind=state_conditioned_snapshot_harvest.TEACHER_ROLLOUT_KIND_LOCAL_SIM,
                )
            )
    candidate = {
        "family": family,
        "snapshot_id": f"{family}_{snapshot_index:03d}",
        "anchor_t": int(t_value),
        **_history_payload(episode_id, t_value),
        "policy_condition.phase": "PLACE" if family == "S_pre_place" else "TRANSPORT",
        "policy_condition.mode": "RECOVERY",
        "policy_condition_text": state_conditioned_bucket_a_import.build_canonical_policy_condition_text(
            "PLACE" if family == "S_pre_place" else "TRANSPORT",
            "RECOVERY",
        ),
        "privileged.apple_to_plate_rel_pose": [0.30, 0.00, 0.00],
        "source_bucket_key": state_conditioned_snapshot_harvest.FAMILY_SOURCE_BUCKET_BY_FAMILY[
            family
        ],
        "deprioritized_by_plan": family
        in state_conditioned_snapshot_harvest.LOW_PRIORITY_FAMILIES,
        "attempts": attempts,
    }
    if scripted_teacher_attempts:
        candidate["scripted_teacher_attempts"] = scripted_teacher_attempts
    return candidate


def _build_grouped_candidates(
    *,
    base_success_counts: dict[str, int] | None = None,
    teacher_success_counts: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    base_counts = {
        "S_drop": 18,
        "S_lost": 2,
        "S_transport_mid": 1,
        "S_pre_place": 2,
    }
    teacher_counts = {
        "S_lost": 8,
    }
    if base_success_counts is not None:
        base_counts.update(base_success_counts)
    if teacher_success_counts is not None:
        teacher_counts.update(teacher_success_counts)

    snapshot_counts = {
        "S_drop": 6,
        "S_lost": 6,
        "S_transport_mid": 2,
        "S_pre_place": 2,
    }
    grouped: dict[str, dict[str, Any]] = {}
    for family in state_conditioned_snapshot_harvest.T8_FAMILY_ORDER:
        grouped[family] = {
            "eligible": [],
            "ineligible": [],
            "deprioritized_flags": {
                family in state_conditioned_snapshot_harvest.LOW_PRIORITY_FAMILIES
            },
        }
        base_success_flat_indices = set(range(int(base_counts.get(family, 0))))
        teacher_success_flat_indices = set(range(int(teacher_counts.get(family, 0))))
        for snapshot_index in range(snapshot_counts[family]):
            grouped[family]["eligible"].append(
                _build_candidate(
                    family=family,
                    snapshot_index=snapshot_index,
                    base_success_flat_indices=base_success_flat_indices,
                    teacher_success_flat_indices=teacher_success_flat_indices,
                )
            )
    return grouped


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
            "episodes": [],
        },
    )
    (
        bucket_dir / state_conditioned_bucket_a_sidecar.BUCKET_A_SIDECAR_JSON_NAME
    ).write_text(
        "",
        encoding="utf-8",
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
            "baseline_policy": {
                "kind": "original_baseline",
                "model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
            },
            "counts": {"entries": 32, "paired_seed_count": 8, "per_stratum": {}},
        },
    )
    _write_json(
        dev_dir / state_conditioned_dev_manifest.BASELINE_DEV_SCORECARD_JSON_NAME,
        {
            "schema_version": state_conditioned_dev_manifest.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_dev_baseline_scorecard",
            "baseline_invocation": {
                "runner": "fake_runner",
                "model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
            },
            "counts": {"requested_entries": 32},
        },
    )

    _write_json(
        collection_dir / state_conditioned_collect_buckets.BUCKET_B_MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_collect_buckets.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_bucket_B_manifest",
            "counts": {"episodes": 16},
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
    return bucket_dir, dev_dir, collection_dir


def _materialize_formal_fixture(
    tmp_path: Path,
    *,
    grouped_candidates: dict[str, dict[str, Any]],
    teacher_version: str = state_conditioned_snapshot_harvest.DEFAULT_TEACHER_VERSION,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    bucket_dir, dev_dir, collection_dir = _build_prerequisites(tmp_path)
    output_dir = tmp_path / "harvest"
    state_conditioned_snapshot_harvest.materialize_snapshot_feasibility(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        collection_dir=collection_dir,
        output_dir=output_dir,
        grouped_snapshot_candidates=grouped_candidates,
    )
    result = state_conditioned_snapshot_harvest.materialize_formal_pseudodemos(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        collection_dir=collection_dir,
        output_dir=output_dir,
        grouped_snapshot_candidates=grouped_candidates,
        teacher_version=teacher_version,
    )
    manifest = _read_json(
        output_dir
        / state_conditioned_snapshot_harvest.LOCAL_RECOVERY_PSEUDODEMO_MANIFEST_JSON_NAME
    )
    analysis = _read_json(
        output_dir
        / state_conditioned_snapshot_harvest.LOCAL_RECOVERY_ROLLOUT_ANALYSIS_JSON_NAME
    )
    assert Path(result["local_recovery_pseudodemo_manifest_path"]).is_file()
    assert Path(result["local_recovery_rollout_analysis_path"]).is_file()
    return output_dir, manifest, analysis


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        state_conditioned_snapshot_harvest.main(["--help"])
    assert exc_info.value.code == 0


def test_formal_mode_requires_t8_artifacts_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bucket_dir, dev_dir, collection_dir = _build_prerequisites(tmp_path)

    exit_code = state_conditioned_snapshot_harvest.main(
        [
            "--mode",
            "formal",
            "--bucket-dir",
            str(bucket_dir),
            "--dev-dir",
            str(dev_dir),
            "--collection-dir",
            str(collection_dir),
            "--output-dir",
            str(tmp_path / "missing_harvest"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "output-dir directory does not exist" in captured.err
    assert "Traceback" not in captured.err


def test_formal_cli_happy_path_reuses_output_dir_snapshot_context(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bucket_dir, dev_dir, collection_dir = _build_prerequisites(tmp_path)
    output_dir = tmp_path / "harvest_cli"
    grouped_candidates = _build_grouped_candidates()
    state_conditioned_snapshot_harvest.materialize_snapshot_feasibility(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        collection_dir=collection_dir,
        output_dir=output_dir,
        grouped_snapshot_candidates=grouped_candidates,
    )

    exit_code = state_conditioned_snapshot_harvest.main(
        [
            "--mode",
            "formal",
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

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    manifest = _read_json(
        output_dir
        / state_conditioned_snapshot_harvest.LOCAL_RECOVERY_PSEUDODEMO_MANIFEST_JSON_NAME
    )
    feasibility_report = _read_json(
        output_dir / state_conditioned_snapshot_harvest.FEASIBILITY_REPORT_JSON_NAME
    )
    assert captured.err == ""
    assert (
        output_dir
        / state_conditioned_snapshot_harvest.OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME
    ).is_file()
    assert feasibility_report["snapshot_candidates_path"] == str(
        (
            output_dir
            / state_conditioned_snapshot_harvest.OUTPUT_DIR_SNAPSHOT_CANDIDATES_JSONL_NAME
        ).resolve()
    )
    assert manifest["successful_pseudodemo_count"] >= 24


def test_formal_happy_path_writes_success_only_manifest_with_teacher_provenance(
    tmp_path: Path,
) -> None:
    output_dir, manifest, analysis = _materialize_formal_fixture(
        tmp_path,
        grouped_candidates=_build_grouped_candidates(),
    )
    teacher_gate_report = _read_json(
        output_dir / state_conditioned_snapshot_harvest.TEACHER_GATE_REPORT_JSON_NAME
    )
    gate_by_family = {row["family"]: row for row in teacher_gate_report["families"]}

    assert manifest["artifact_kind"] == "local_recovery_pseudodemo_manifest"
    assert manifest["successful_pseudodemo_count"] >= 24
    assert manifest["counts"]["successful_pseudodemo_count"] >= 24
    assert manifest["counts"]["selected_pseudodemo_count_by_family"]["S_drop"] >= 8
    assert manifest["counts"]["selected_pseudodemo_count_by_family"]["S_lost"] >= 8
    assert manifest["producer_by_family"] == {
        "S_drop": state_conditioned_snapshot_harvest.PRODUCER_BASE_POLICY,
        "S_lost": state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER,
        "S_transport_mid": None,
        "S_pre_place": None,
    }
    assert analysis["counts"]["analysis_only_failed_rollout_count"] > 0

    assert manifest["pseudodemos"]
    for record in manifest["pseudodemos"]:
        family = record["source_snapshot_family"]
        gate_row = gate_by_family[family]
        assert record["failure_prefix_step_count"] > 0
        assert record["recovery_suffix_step_count"] > 0
        assert record["failure_prefix_source_episode_id"] == record["episode_id"]
        assert record["recovery_suffix_source_episode_id"] == record["episode_id"]
        assert len(record["failure_prefix_source_t_range"]) == 2
        assert len(record["recovery_suffix_source_t_range"]) == 2
        assert record["teacher_version"] == "scripted_teacher_v1"
        assert record["teacher_trigger_reason"] in {
            "teacher_gate_success_rate_below_threshold",
            "teacher_gate_success_rate_at_or_above_threshold",
        }
        assert record["teacher_trigger_success_rate"] == gate_row["success_rate"]
        assert record["teacher_trigger_threshold"] == gate_row["threshold"]
        assert record["producer"] in {
            state_conditioned_snapshot_harvest.PRODUCER_BASE_POLICY,
            state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER,
        }
        assert record["source_snapshot_id"]
        assert record["source_snapshot_history_k"] == 8
        assert (
            record["source_bucket_key"]
            == state_conditioned_snapshot_harvest.FAMILY_SOURCE_BUCKET_BY_FAMILY[family]
        )
        if (
            record["producer"]
            == state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER
        ):
            assert (
                record["teacher_target_truthfulness"]
                == state_conditioned_snapshot_harvest.TEACHER_TARGET_TRUTHFUL_REAL_ROLLOUT
            )
            assert record["teacher_target"]["producer"] == record["teacher_version"]
            assert (
                record["teacher_target"]["synthetic_observation_only_backfill"] is False
            )
        else:
            assert (
                record["teacher_target_truthfulness"]
                == state_conditioned_snapshot_harvest.TEACHER_TARGET_NOT_APPLICABLE
            )
            assert "teacher_target" not in record


def test_formal_fails_when_high_priority_floor_missed_even_if_total_is_sufficient(
    tmp_path: Path,
) -> None:
    bucket_dir, dev_dir, collection_dir = _build_prerequisites(tmp_path)
    output_dir = tmp_path / "harvest_floor_fail"
    grouped_candidates = _build_grouped_candidates(
        base_success_counts={"S_drop": 7, "S_pre_place": 4},
        teacher_success_counts={"S_lost": 8, "S_transport_mid": 6},
    )
    state_conditioned_snapshot_harvest.materialize_snapshot_feasibility(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        collection_dir=collection_dir,
        output_dir=output_dir,
        grouped_snapshot_candidates=grouped_candidates,
    )

    with pytest.raises(ValueError, match="formal harvest floor unmet for S_drop"):
        state_conditioned_snapshot_harvest.materialize_formal_pseudodemos(
            bucket_dir=bucket_dir,
            dev_dir=dev_dir,
            collection_dir=collection_dir,
            output_dir=output_dir,
            grouped_snapshot_candidates=grouped_candidates,
        )


def test_teacher_generated_sample_missing_provenance_fails(
    tmp_path: Path,
) -> None:
    bucket_dir, dev_dir, collection_dir = _build_prerequisites(tmp_path)
    output_dir = tmp_path / "harvest_missing_teacher_provenance"
    grouped_candidates = _build_grouped_candidates()
    state_conditioned_snapshot_harvest.materialize_snapshot_feasibility(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        collection_dir=collection_dir,
        output_dir=output_dir,
        grouped_snapshot_candidates=grouped_candidates,
    )

    with pytest.raises(ValueError, match="teacher_version"):
        state_conditioned_snapshot_harvest.materialize_formal_pseudodemos(
            bucket_dir=bucket_dir,
            dev_dir=dev_dir,
            collection_dir=collection_dir,
            output_dir=output_dir,
            grouped_snapshot_candidates=grouped_candidates,
            teacher_version="",
        )


def test_failed_rollout_is_excluded_from_manifest_and_kept_in_analysis_only(
    tmp_path: Path,
) -> None:
    _output_dir, manifest, analysis = _materialize_formal_fixture(
        tmp_path,
        grouped_candidates=_build_grouped_candidates(),
    )
    manifest_episode_ids = {record["episode_id"] for record in manifest["pseudodemos"]}
    failed_rollouts = [
        row
        for row in analysis["failed_rollouts"]
        if row["reason"] == "family_success_criteria_not_met"
    ]

    assert failed_rollouts
    assert all(
        row["included_in_pseudodemo_manifest"] is False for row in failed_rollouts
    )
    assert all(row["episode_id"] not in manifest_episode_ids for row in failed_rollouts)


def test_formal_uses_on_demand_teacher_when_cached_attempts_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grouped_candidates = _build_grouped_candidates()
    for family in ("S_lost",):
        for candidate in grouped_candidates[family]["eligible"]:
            candidate.pop("scripted_teacher_attempts", None)

    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_run_on_demand_scripted_teacher_attempt",
        lambda candidate, *, seed, family: {
            "seed": int(seed),
            "policy_steps": _policy_steps_for_family(family, success=True),
            "success_episode": bool(family == "S_pre_place"),
            "producer": state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER,
            "teacher_rollout_kind": state_conditioned_snapshot_harvest.TEACHER_ROLLOUT_KIND_LOCAL_SIM,
        },
    )

    _output_dir, manifest, _analysis = _materialize_formal_fixture(
        tmp_path,
        grouped_candidates=grouped_candidates,
    )

    assert manifest["successful_pseudodemo_count"] >= 24
    assert any(
        record["producer"]
        == state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER
        for record in manifest["pseudodemos"]
    )


def test_legacy_cached_s_lost_teacher_attempt_with_rollout_records_remains_truthful(
    tmp_path: Path,
) -> None:
    grouped_candidates = _build_grouped_candidates()
    for candidate in grouped_candidates["S_lost"]["eligible"]:
        snapshot_id = str(candidate["snapshot_id"])
        legacy_attempts: list[dict[str, Any]] = []
        for raw_attempt in list(candidate["scripted_teacher_attempts"]):
            attempt = dict(raw_attempt)
            seed = int(attempt["seed"])
            episode_id = (
                f"{snapshot_id}__s_lost__"
                f"{state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER}__seed{seed:03d}"
            )
            transitions = [
                {
                    "episode_id": episode_id,
                    "t": int(step.get("t", index)),
                    "success_step": bool(step.get("success_step", False)),
                    "policy_step": dict(step),
                }
                for index, step in enumerate(list(attempt["policy_steps"]))
            ]
            attempt.pop("teacher_rollout_kind", None)
            attempt.pop("teacher_target", None)
            attempt["episode_record"] = {
                "episode_id": episode_id,
                "success_episode": bool(attempt.get("success_episode", False)),
            }
            attempt["transition_records"] = transitions
            legacy_attempts.append(attempt)
        candidate["scripted_teacher_attempts"] = legacy_attempts

    _output_dir, manifest, _analysis = _materialize_formal_fixture(
        tmp_path,
        grouped_candidates=grouped_candidates,
    )

    s_lost_records = [
        record
        for record in manifest["pseudodemos"]
        if record["source_snapshot_family"] == "S_lost"
    ]
    assert len(s_lost_records) >= 8
    assert all(
        record["teacher_target_truthfulness"]
        == state_conditioned_snapshot_harvest.TEACHER_TARGET_TRUTHFUL_REAL_ROLLOUT
        for record in s_lost_records
    )
    assert all("teacher_target" in record for record in s_lost_records)


def test_untruthful_scripted_teacher_target_is_rejected_and_kept_in_analysis_only(
    tmp_path: Path,
) -> None:
    grouped_candidates = _build_grouped_candidates(
        teacher_success_counts={"S_lost": 9},
    )
    first_lost = grouped_candidates["S_lost"]["eligible"][0]
    bad_attempt = dict(first_lost["scripted_teacher_attempts"][0])
    bad_attempt.pop("teacher_rollout_kind", None)
    bad_episode_id = (
        f"{first_lost['snapshot_id']}__s_lost__"
        f"{state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER}__seed000"
    )
    bad_attempt["teacher_target"] = {
        "trace_episode_id": bad_episode_id,
        "trace_t_range": [1, 4],
        "producer": state_conditioned_snapshot_harvest.DEFAULT_TEACHER_VERSION,
        "synthetic_observation_only_backfill": True,
    }
    first_lost["scripted_teacher_attempts"][0] = bad_attempt

    output_dir, manifest, analysis = _materialize_formal_fixture(
        tmp_path,
        grouped_candidates=grouped_candidates,
    )

    assert output_dir.is_dir()
    assert manifest["successful_pseudodemo_count"] >= 24
    blocked = [
        row
        for row in analysis["failed_rollouts"]
        if row["reason"]
        == "scripted teacher target is not truthful: synthetic_observation_only_backfill=true"
    ]
    assert blocked
    assert all(row["included_in_pseudodemo_manifest"] is False for row in blocked)
    blocked_episode_ids = {row.get("episode_id") for row in blocked}
    assert blocked_episode_ids.isdisjoint(
        {record["episode_id"] for record in manifest["pseudodemos"]}
    )


def test_run_on_demand_scripted_teacher_attempt_dispatches_s_drop_to_local_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _build_candidate(
        family="S_drop",
        snapshot_index=0,
        base_success_flat_indices=set(),
        teacher_success_flat_indices=set(),
    )
    candidate.update(
        {
            "source_dataset_dir": "/tmp/unused",
            "source_npz_path": "/tmp/unused.npz",
            "source_episode_seed": 7,
            "source_env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
            "source_model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
            "source_embodiment_tag": "UNITREE_G1",
        }
    )

    class _FakeEnv:
        def reset(self, seed: int | None = None):
            return ({"privileged.apple_in_hand": False}, {})

        def step(self, _action):
            return ({"privileged.apple_in_hand": False}, 0.0, False, False, {})

        def close(self):
            return None

    called = {"s_drop": False}
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_dataset_sidecar_rows",
        lambda _dataset_dir, *, episode_id: [{"episode_id": episode_id, "t": 3}],
    )
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_dataset_episode_record",
        lambda _dataset_dir, *, episode_id: {
            "episode_id": episode_id,
            "success_episode": False,
        },
    )
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_load_replay_action_chunks",
        lambda _candidate: [{"action.right_hand": [[[0.0]]]} for _ in range(4)],
    )
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_build_replay_env",
        lambda _candidate, n_action_steps: _FakeEnv(),
    )

    def _fake_s_drop_helper(
        candidate_arg,
        *,
        env,
        obs,
        action_chunks,
        anchor_t,
        source_seed,
    ):
        called["s_drop"] = True
        return {
            "seed": int(source_seed),
            "family": "S_drop",
            "policy_steps": _policy_steps_for_family("S_drop", success=True),
            "success_episode": False,
            "recovery_entry_step": 0,
        }

    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_run_s_drop_local_teacher_rollout",
        _fake_s_drop_helper,
    )

    attempt = (
        state_conditioned_snapshot_harvest._run_on_demand_scripted_teacher_attempt(
            candidate,
            seed=0,
            family="S_drop",
        )
    )

    assert called["s_drop"] is True
    assert (
        attempt["producer"]
        == state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER
    )
    assert attempt["recovery_entry_step"] == 0
    assert attempt["policy_steps"]


def test_run_on_demand_scripted_teacher_attempt_uses_recorded_source_trace_for_s_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _build_candidate(
        family="S_lost",
        snapshot_index=0,
        base_success_flat_indices=set(),
        teacher_success_flat_indices=set(),
    )
    candidate.update(
        {
            "source_dataset_dir": "/tmp/unused",
            "source_npz_path": "/tmp/unused.npz",
            "source_episode_seed": 5,
            "source_env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
            "source_model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
            "source_embodiment_tag": "UNITREE_G1",
        }
    )

    called = {"helper": False}
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_load_replay_action_chunks",
        lambda _candidate: [
            {"action.left_arm": np.zeros((1, 30, 7), dtype=np.float32)}
            for _ in range(4)
        ],
    )
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_dataset_sidecar_rows",
        lambda _dataset_dir, *, episode_id: [
            {
                "episode_id": episode_id,
                "t": 1,
                "privileged.apple_visible": True,
                "privileged.apple_in_hand": False,
                "privileged.contact_flag": False,
                "privileged.apple_to_plate_rel_pose": [0.0, 0.0, 0.0],
            },
            {
                "episode_id": episode_id,
                "t": 2,
                "privileged.apple_visible": True,
                "privileged.apple_in_hand": False,
                "privileged.contact_flag": False,
                "privileged.apple_to_plate_rel_pose": [0.0, 0.0, 0.0],
            },
        ],
    )

    def _fake_source_helper(**kwargs):
        called["helper"] = True
        return {
            "seed": int(kwargs["source_seed"]),
            "family": "S_lost",
            "policy_steps": _policy_steps_for_family("S_lost", success=True),
            "success_episode": True,
        }

    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_run_s_lost_recorded_source_teacher_rollout",
        _fake_source_helper,
    )
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_build_replay_env",
        lambda _candidate, n_action_steps: (_ for _ in ()).throw(
            AssertionError("S_lost path should not build replay env")
        ),
    )

    attempt = (
        state_conditioned_snapshot_harvest._run_on_demand_scripted_teacher_attempt(
            candidate,
            seed=0,
            family="S_lost",
        )
    )

    assert called["helper"] is True
    assert (
        attempt["producer"]
        == state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER
    )
    assert (
        attempt["teacher_rollout_kind"]
        == state_conditioned_snapshot_harvest.TEACHER_ROLLOUT_KIND_RECORDED_SOURCE_TRACE
    )
    assert attempt["policy_steps"]


def test_s_lost_local_teacher_rollout_marks_visibility_streak_without_zero_recovery_entry() -> (
    None
):
    candidate = _build_candidate(
        family="S_lost",
        snapshot_index=0,
        base_success_flat_indices=set(),
        teacher_success_flat_indices=set(),
    )

    class _FakeEnv:
        def __init__(self) -> None:
            self._calls = 0

        def step(self, _action):
            self._calls += 1
            return (
                {
                    "privileged.apple_visible": True,
                    "privileged.apple_in_hand": False,
                    "privileged.contact_flag": False,
                    "privileged.apple_to_plate_rel_pose": [0.0, 0.0, 0.0],
                },
                0.0,
                False,
                False,
                {},
            )

    attempt = state_conditioned_snapshot_harvest._run_s_lost_local_teacher_rollout(
        candidate,
        env=_FakeEnv(),
        action_chunks=[
            {"action.right_hand": np.zeros((1, 30, 1), dtype=np.float32)}
            for _ in range(3)
        ],
        anchor_t=0,
        source_seed=11,
    )

    assert "recovery_entry_step" not in attempt
    assert attempt["success_episode"] is True
    assert len(attempt["policy_steps"]) >= 4
    assert any(step["success_step"] is True for step in attempt["policy_steps"])


def test_run_on_demand_scripted_teacher_attempt_expands_recorded_source_trace_for_s_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _build_candidate(
        family="S_lost",
        snapshot_index=0,
        base_success_flat_indices=set(),
        teacher_success_flat_indices=set(),
    )
    candidate["anchor_t"] = 0
    candidate.update(
        {
            "source_dataset_dir": "/tmp/unused",
            "source_npz_path": "/tmp/unused.npz",
            "source_episode_seed": 11,
            "source_env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
            "source_model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
            "source_embodiment_tag": "UNITREE_G1",
        }
    )

    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_load_replay_action_chunks",
        lambda _candidate: [
            {"action.left_arm": np.zeros((1, 30, 7), dtype=np.float32)}
            for _ in range(5)
        ],
    )
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_dataset_sidecar_rows",
        lambda _dataset_dir, *, episode_id: [
            {
                "episode_id": episode_id,
                "t": 1,
                "privileged.apple_visible": True,
                "privileged.apple_in_hand": False,
                "privileged.contact_flag": False,
                "privileged.apple_to_plate_rel_pose": [0.30, 0.0, 0.0],
            },
            {
                "episode_id": episode_id,
                "t": 2,
                "privileged.apple_visible": True,
                "privileged.apple_in_hand": False,
                "privileged.contact_flag": False,
                "privileged.apple_to_plate_rel_pose": [0.30, 0.0, 0.0],
            },
            {
                "episode_id": episode_id,
                "t": 3,
                "privileged.apple_visible": True,
                "privileged.apple_in_hand": False,
                "privileged.contact_flag": False,
                "privileged.apple_to_plate_rel_pose": [0.30, 0.0, 0.0],
            },
            {
                "episode_id": episode_id,
                "t": 4,
                "privileged.apple_visible": True,
                "privileged.apple_in_hand": False,
                "privileged.contact_flag": False,
                "privileged.apple_to_plate_rel_pose": [0.30, 0.0, 0.0],
            },
        ],
    )
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_build_replay_env",
        lambda _candidate, n_action_steps: (_ for _ in ()).throw(
            AssertionError("S_lost path should not build replay env")
        ),
    )

    attempt = (
        state_conditioned_snapshot_harvest._run_on_demand_scripted_teacher_attempt(
            candidate,
            seed=0,
            family="S_lost",
        )
    )

    assert (
        attempt["producer"]
        == state_conditioned_snapshot_harvest.PRODUCER_SCRIPTED_TEACHER
    )
    assert (
        attempt["teacher_rollout_kind"]
        == state_conditioned_snapshot_harvest.TEACHER_ROLLOUT_KIND_RECORDED_SOURCE_TRACE
    )
    assert len(attempt["policy_steps"]) == 120
    assert all(
        step["privileged.apple_visible"] is True for step in attempt["policy_steps"]
    )
    assert any(step["success_step"] is True for step in attempt["policy_steps"])


def test_s_drop_local_teacher_rollout_uses_sim_backed_attach_without_obs_spoof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _build_candidate(
        family="S_drop",
        snapshot_index=0,
        base_success_flat_indices=set(),
        teacher_success_flat_indices=set(),
    )
    returned_obs: list[dict[str, Any]] = []

    class _FakeEnv:
        def step(self, _action):
            obs = {
                "privileged.apple_in_hand": False,
                "privileged.apple_visible": False,
                "privileged.contact_flag": False,
                "privileged.apple_to_plate_rel_pose": [0.30, 0.0, 0.0],
            }
            returned_obs.append(obs)
            return (obs, 0.0, False, False, {})

    apply_calls = {"count": 0}

    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_discover_s_drop_attach_state",
        lambda _env: {"token": "sim_attach"},
    )

    def _fake_apply(_attach_state):
        apply_calls["count"] += 1
        return True

    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_apply_s_drop_sim_attachment",
        _fake_apply,
    )
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_s_drop_attachment_active",
        lambda _attach_state: True,
    )
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest,
        "_refresh_obs_after_s_drop_sim_mutation",
        lambda _env, *, fallback_obs: dict(fallback_obs),
    )
    monkeypatch.setattr(
        state_conditioned_snapshot_harvest.recap_collector,
        "infer_success_step",
        lambda _info, reward_wrapper=None: False,
    )

    attempt = state_conditioned_snapshot_harvest._run_s_drop_local_teacher_rollout(
        candidate,
        env=_FakeEnv(),
        obs={
            "privileged.apple_in_hand": False,
            "privileged.apple_visible": False,
        },
        action_chunks=[{"action.right_hand": [[[0.0]]]}],
        anchor_t=0,
        source_seed=3,
    )

    assert attempt["success_episode"] is True
    assert attempt["recovery_entry_step"] == 1
    assert len(attempt["policy_steps"]) == 9
    assert attempt["policy_steps"][0]["privileged.apple_in_hand"] is False
    assert all(
        step["privileged.apple_in_hand"] is True for step in attempt["policy_steps"][1:]
    )
    assert all(obs["privileged.apple_in_hand"] is False for obs in returned_obs)
    assert all(obs["privileged.apple_visible"] is False for obs in returned_obs)
    assert apply_calls["count"] >= 8
