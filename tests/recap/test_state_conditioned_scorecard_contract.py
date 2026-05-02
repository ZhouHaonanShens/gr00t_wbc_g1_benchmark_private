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
from work.recap.scripts import state_conditioned_dev_manifest
from work.recap.scripts import state_conditioned_oracle_eval
from work.recap.scripts import state_conditioned_train


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _decision_line_metrics(
    *,
    baseline_success: float,
    c0_success: float,
    c1_success: float,
    teacher_reachable_rate: float,
    history_probe_passed: bool,
    baseline_phase_mean: float,
    c0_phase_mean: float,
    c1_phase_mean: float,
    baseline_recovery_rate: float,
    c0_recovery_rate: float,
    c1_recovery_rate: float,
    c1_valid_action_rate: float = 1.0,
    c1_snapshot_family_hit_rate: float = 1.0,
) -> dict[str, dict[str, Any]]:
    def _line(
        *,
        success_rate: float,
        phase_mean: float,
        recovery_rate: float,
        valid_action_rate: float,
        snapshot_family_hit_rate: float,
    ) -> dict[str, Any]:
        return {
            "comparable_metrics": {"success_rate": success_rate},
            "counts": {"evaluated_episodes": 32},
            "diagnostics": {
                "teacher_reachable_rate": {
                    "reachable_rate": teacher_reachable_rate,
                },
                "history_condition_usage_probe": {
                    "history_condition_response": {
                        "passed": history_probe_passed,
                        "status": "PASS" if history_probe_passed else "FAIL",
                    }
                },
                "max_phase_reached": {
                    "mean_phase_index": phase_mean,
                },
                "recovery_attempted_rate": {
                    "rate": recovery_rate,
                },
                "valid_action_rate": {
                    "rate": valid_action_rate,
                },
                "snapshot_family_hit_rate": {
                    "rate": snapshot_family_hit_rate,
                },
            },
        }

    return {
        "baseline": _line(
            success_rate=baseline_success,
            phase_mean=baseline_phase_mean,
            recovery_rate=baseline_recovery_rate,
            valid_action_rate=1.0,
            snapshot_family_hit_rate=1.0,
        ),
        "c0": _line(
            success_rate=c0_success,
            phase_mean=c0_phase_mean,
            recovery_rate=c0_recovery_rate,
            valid_action_rate=1.0,
            snapshot_family_hit_rate=1.0,
        ),
        "c1": _line(
            success_rate=c1_success,
            phase_mean=c1_phase_mean,
            recovery_rate=c1_recovery_rate,
            valid_action_rate=c1_valid_action_rate,
            snapshot_family_hit_rate=c1_snapshot_family_hit_rate,
        ),
    }


def _manifest_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for stratum_index, stratum in enumerate(
        state_conditioned_dev_manifest.DEFAULT_STRATA_DEFINITIONS
    ):
        stratum_id = str(stratum["stratum_id"])
        for pair_index, seed in enumerate(
            state_conditioned_dev_manifest.DEFAULT_PAIRED_SEEDS
        ):
            entries.append(
                {
                    "entry_id": f"dev_{stratum_id}_{seed}",
                    "seed": int(seed),
                    "pair_index": int(pair_index),
                    "stratum_index": int(stratum_index),
                    "stratum_id": stratum_id,
                    "paired_key": f"seed={int(seed)}|stratum={stratum_id}",
                    "baseline_eval": {
                        "env_name": state_conditioned_dev_manifest.DEFAULT_ENV_NAME,
                        "max_episode_steps": int(
                            state_conditioned_dev_manifest.DEFAULT_MAX_EPISODE_STEPS
                        ),
                    },
                }
            )
    return entries


def _build_prerequisites(tmp_path: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    dev_dir = tmp_path / "devbench"
    training_dir = tmp_path / "training"
    sanity_dir = tmp_path / "sanity"
    dev_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    sanity_dir.mkdir(parents=True, exist_ok=True)
    entries = _manifest_entries()

    _write_json(
        dev_dir / state_conditioned_dev_manifest.FIXED_STRATA_DEFINITION_JSON_NAME,
        {
            "schema_version": state_conditioned_dev_manifest.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_dev_fixed_strata_definition",
            "paired_seed_count": 8,
        },
    )
    _write_json(
        dev_dir / state_conditioned_dev_manifest.BASELINE_MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_dev_manifest.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_dev_baseline_manifest",
            "baseline_policy": {
                "model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
            },
            "entries": entries,
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

    c0_checkpoint = training_dir / "checkpoint_C0_equal_data_control" / "checkpoint-100"
    c1_checkpoint = training_dir / "checkpoint_C1_phase_mode" / "checkpoint-100"
    c0_checkpoint.mkdir(parents=True, exist_ok=True)
    c1_checkpoint.mkdir(parents=True, exist_ok=True)
    (c0_checkpoint / "model.safetensors").write_text("c0", encoding="utf-8")
    (c1_checkpoint / "model.safetensors").write_text("c1", encoding="utf-8")

    _write_json(
        training_dir / state_conditioned_train.RUN_METADATA_BASENAME_BY_VARIANT["c0"],
        {
            "schema_version": "state_conditioned_training_run_v1",
            "artifact_kind": "state_conditioned_training_run_metadata",
            "comparable_run_spec": {
                "checkpoint_rule": {
                    "selected_checkpoint_path": str(c0_checkpoint),
                }
            },
        },
    )
    _write_json(
        training_dir / state_conditioned_train.RUN_METADATA_BASENAME_BY_VARIANT["c1"],
        {
            "schema_version": "state_conditioned_training_run_v1",
            "artifact_kind": "state_conditioned_training_run_metadata",
            "comparable_run_spec": {
                "checkpoint_rule": {
                    "selected_checkpoint_path": str(c1_checkpoint),
                }
            },
        },
    )
    _write_json(
        training_dir / state_conditioned_train.DIFF_WHITELIST_JSON_NAME,
        {
            "schema_version": "state_conditioned_training_diff_v1",
            "artifact_kind": "state_conditioned_training_fairness_diff_whitelist",
            "status": "PASS",
        },
    )
    _write_json(
        sanity_dir / "teacher_upper_bound_report.json",
        {
            "schema_version": "g1_state_conditioned_teacher_upper_bound_sanity_v1",
            "artifact_kind": "state_conditioned_teacher_upper_bound_report",
            "teacher_threshold": 0.15,
            "teacher_upper_bound": {"reachable_rate": 0.75, "success_count": 36},
            "gate": {"status": "ALLOW"},
            "families": [
                {
                    "family": "S_drop",
                    "reachable_rate": 1.0,
                    "teacher_meets_threshold": True,
                    "interpretation_code": "teacher_reachable_model_not_learned",
                },
                {
                    "family": "S_lost",
                    "reachable_rate": 1.0,
                    "teacher_meets_threshold": True,
                    "interpretation_code": "teacher_reachable_model_not_learned",
                },
                {
                    "family": "S_transport_mid",
                    "reachable_rate": 0.0,
                    "teacher_meets_threshold": False,
                    "interpretation_code": "teacher_unreachable_on_snapshots_no_progress",
                },
            ],
        },
    )
    _write_json(
        sanity_dir / "teacher_upper_bound_gate.json",
        {
            "schema_version": "g1_state_conditioned_teacher_upper_bound_gate_v1",
            "artifact_kind": "state_conditioned_teacher_upper_bound_gate",
            "reason_code": "allow_teacher_reachable_model_currently_zero",
        },
    )
    _write_json(
        sanity_dir / "open_loop_agreement_report.json",
        {
            "schema_version": "g1_state_conditioned_open_loop_agreement_v1",
            "artifact_kind": "state_conditioned_open_loop_agreement_report",
            "status": "PASS",
            "telemetry": {"allowed_abs_limit": 2.22},
            "checks": {
                "history_condition_response": {
                    "passed": True,
                    "status": "PASS",
                    "probe_count": 24,
                    "response_ratio": 24342.162695242496,
                    "min_response_ratio": 0.001,
                },
                "valid_mask_effectiveness": {
                    "passed": True,
                    "status": "PASS",
                    "probe_count": 16,
                    "max_abs_prediction_delta": 0.0,
                },
                "negative_all_false_mask_probe": {
                    "passed": True,
                    "status": "PASS",
                    "detected_error_code": "EMPTY_HISTORY_VALID_MASK",
                },
            },
        },
    )
    return dev_dir, training_dir, entries


def _episode_success_by_line(line_key: str) -> dict[str, int]:
    if line_key == state_conditioned_oracle_eval.LINE_BASELINE:
        return {
            "nominal": 4,
            "drop_during_transport": 2,
            "failed_grasp_visible": 2,
            "failed_grasp_occluded": 2,
        }
    if line_key == state_conditioned_oracle_eval.LINE_C0:
        return {
            "nominal": 5,
            "drop_during_transport": 3,
            "failed_grasp_visible": 3,
            "failed_grasp_occluded": 3,
        }
    return {
        "nominal": 6,
        "drop_during_transport": 5,
        "failed_grasp_visible": 5,
        "failed_grasp_occluded": 4,
    }


def _build_sidecar_rows(
    *,
    paired_key: str,
    seed: int,
    stratum_id: str,
    success: bool,
    repeat_failure: bool,
) -> list[dict[str, Any]]:
    action_summary = {
        "action.right_arm": {
            "shape": [1, 30, 7],
            "mean_abs": 0.05,
            "max_abs": 0.25,
            "p95_abs": 0.20,
            "abs_preview": [0.1, 0.2],
        }
    }
    rows: list[dict[str, Any]] = [
        {
            "paired_key": paired_key,
            "seed": int(seed),
            "stratum_id": str(stratum_id),
            "t": 0,
            "phase": "SEARCH",
            "apple_in_hand": False,
            "action_summary": action_summary,
            "source_snapshot_family": None if stratum_id == "nominal" else stratum_id,
        }
    ]
    if success:
        for t_value in range(1, 9):
            rows.append(
                {
                    "paired_key": paired_key,
                    "seed": int(seed),
                    "stratum_id": str(stratum_id),
                    "t": int(t_value),
                    "phase": "VERIFY_HOLD" if t_value < 4 else "TRANSPORT",
                    "apple_in_hand": True,
                    "action_summary": action_summary,
                    "source_snapshot_family": None
                    if stratum_id == "nominal"
                    else stratum_id,
                    "recovery_entry_step": 2 if stratum_id != "nominal" else None,
                    "recovery_exit_step": 3 if stratum_id != "nominal" else None,
                }
            )
        rows.append(
            {
                "paired_key": paired_key,
                "seed": int(seed),
                "stratum_id": str(stratum_id),
                "t": 9,
                "phase": "PLACE",
                "apple_in_hand": False,
                "action_summary": action_summary,
                "source_snapshot_family": None
                if stratum_id == "nominal"
                else stratum_id,
            }
        )
        return rows

    rows.extend(
        [
            {
                "paired_key": paired_key,
                "seed": int(seed),
                "stratum_id": str(stratum_id),
                "t": 1,
                "phase": "VERIFY_HOLD",
                "apple_in_hand": True,
                "action_summary": action_summary,
                "source_snapshot_family": None
                if stratum_id == "nominal"
                else stratum_id,
            },
            {
                "paired_key": paired_key,
                "seed": int(seed),
                "stratum_id": str(stratum_id),
                "t": 2,
                "phase": "TRANSPORT",
                "apple_in_hand": False,
                "action_summary": action_summary,
                "source_snapshot_family": None
                if stratum_id == "nominal"
                else stratum_id,
                "recovery_entry_step": 2 if stratum_id != "nominal" else None,
                "recovery_exit_step": 3 if stratum_id != "nominal" else None,
            },
            {
                "paired_key": paired_key,
                "seed": int(seed),
                "stratum_id": str(stratum_id),
                "t": 3,
                "phase": "TRANSPORT",
                "apple_in_hand": False,
                "action_summary": action_summary,
                "source_snapshot_family": None
                if stratum_id == "nominal"
                else stratum_id,
                "recovery_entry_step": 2 if stratum_id != "nominal" else None,
                "recovery_exit_step": 3 if stratum_id != "nominal" else None,
            },
            {
                "paired_key": paired_key,
                "seed": int(seed),
                "stratum_id": str(stratum_id),
                "t": 4,
                "phase": "TRANSPORT",
                "apple_in_hand": False,
                "action_summary": action_summary,
                "source_snapshot_family": None
                if stratum_id == "nominal"
                else stratum_id,
            },
        ]
    )
    if repeat_failure and stratum_id != "nominal":
        rows.extend(
            [
                {
                    "paired_key": paired_key,
                    "seed": int(seed),
                    "stratum_id": str(stratum_id),
                    "t": 5,
                    "phase": "TRANSPORT",
                    "apple_in_hand": False,
                    "action_summary": action_summary,
                    "source_snapshot_family": stratum_id,
                    "recovery_entry_step": 5,
                    "recovery_exit_step": 6,
                },
                {
                    "paired_key": paired_key,
                    "seed": int(seed),
                    "stratum_id": str(stratum_id),
                    "t": 6,
                    "phase": "TRANSPORT",
                    "apple_in_hand": False,
                    "action_summary": action_summary,
                    "source_snapshot_family": stratum_id,
                    "recovery_entry_step": 5,
                    "recovery_exit_step": 6,
                },
            ]
        )
    rows.append(
        {
            "paired_key": paired_key,
            "seed": int(seed),
            "stratum_id": str(stratum_id),
            "t": 7,
            "phase": "PLACE",
            "apple_in_hand": False,
            "action_summary": action_summary,
            "source_snapshot_family": None if stratum_id == "nominal" else stratum_id,
        }
    )
    return rows


def _fake_eval_runner(
    *,
    line_spec: dict[str, Any],
    manifest_entries: list[dict[str, Any]],
    stratum_counts: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    del output_dir
    success_by_stratum = _episode_success_by_line(str(line_spec["line_key"]))
    episode_records: list[dict[str, Any]] = []
    sidecar_rows: list[dict[str, Any]] = []
    per_stratum: dict[str, dict[str, Any]] = {}
    total_success = 0
    for stratum_id, count in stratum_counts.items():
        entries = sorted(
            [entry for entry in manifest_entries if entry["stratum_id"] == stratum_id],
            key=lambda entry: int(entry["seed"]),
        )
        success_cutoff = int(success_by_stratum[stratum_id])
        stratum_success = 0
        for index, entry in enumerate(entries):
            success = bool(index < success_cutoff)
            repeat_failure = bool(not success and index % 2 == 0)
            episode_records.append(
                {
                    "paired_key": entry["paired_key"],
                    "seed": int(entry["seed"]),
                    "stratum_id": str(stratum_id),
                    "success": bool(success),
                    "source_snapshot_family": None
                    if stratum_id == "nominal"
                    else stratum_id,
                }
            )
            sidecar_rows.extend(
                _build_sidecar_rows(
                    paired_key=str(entry["paired_key"]),
                    seed=int(entry["seed"]),
                    stratum_id=str(stratum_id),
                    success=success,
                    repeat_failure=repeat_failure,
                )
            )
            stratum_success += 1 if success else 0
        per_stratum[stratum_id] = {
            "requested_count": int(count),
            "evaluated_episodes": int(count),
            "success_count": int(stratum_success),
            "success_rate": float(stratum_success) / float(count),
        }
        total_success += int(stratum_success)
    return {
        "line_invocation": {
            "runner": "fake_eval_runner",
            "oracle_phase_mode_supplied": bool(line_spec["oracle_phase_mode_supplied"]),
            "wrapper_python": str(line_spec["wrapper_python"]),
            "eval_python": str(line_spec["eval_python"]),
        },
        "aggregate_metrics": {
            "requested_entries": 32,
            "evaluated_episodes": 32,
            "success_count": int(total_success),
            "success_rate": float(total_success) / 32.0,
        },
        "per_stratum": per_stratum,
        "episode_records": episode_records,
        "sidecar_rows": sidecar_rows,
    }


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        state_conditioned_oracle_eval.main(["--help"])
    assert exc_info.value.code == 0


def test_bad_output_path_fails_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = state_conditioned_oracle_eval.main(
        ["--output-dir", ".sisyphus/oracle_conditioned_dev_scorecard.json"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "output-dir must be a directory path" in captured.err
    assert "Traceback" not in captured.err


def test_materialize_happy_path_writes_four_artifacts_with_three_lines(
    tmp_path: Path,
) -> None:
    dev_dir, training_dir, entries = _build_prerequisites(tmp_path)
    output_dir = tmp_path / "eval"

    result = state_conditioned_oracle_eval.materialize_state_conditioned_oracle_eval(
        dev_dir=dev_dir,
        training_dir=training_dir,
        output_dir=output_dir,
        eval_runner=_fake_eval_runner,
    )

    scorecard = _read_json(
        output_dir
        / state_conditioned_oracle_eval.ORACLE_CONDITIONED_DEV_SCORECARD_JSON_NAME
    )
    gate = _read_json(
        output_dir / state_conditioned_oracle_eval.ORACLE_GATE_DECISION_JSON_NAME
    )
    summary = _read_json(
        output_dir / state_conditioned_oracle_eval.RECOVERY_BENCHMARK_SUMMARY_JSON_NAME
    )
    split = _read_json(
        output_dir / state_conditioned_oracle_eval.RESULT_SPLIT_DECISION_JSON_NAME
    )

    assert Path(result["oracle_conditioned_dev_scorecard_path"]).is_file()
    assert Path(result["oracle_gate_decision_path"]).is_file()
    assert Path(result["recovery_benchmark_summary_path"]).is_file()
    assert Path(result["result_split_decision_path"]).is_file()

    assert scorecard["line_order"] == ["baseline", "c0", "c1"]
    assert scorecard["comparable_metric_names"] == [
        "success_rate",
        "off_nominal_recovery_success_rate",
        "empty_hand_release_rate",
        "drop_to_abort_latency",
        "reacquire_attempt_rate",
        "verify_hold_pass_rate",
        "empty_hand_transport_rate",
        "same_failure_repeat_rate",
        "nominal_success_delta",
    ]
    assert scorecard["diagnostic_metric_names"] == [
        "max_phase_reached",
        "first_failure_phase",
        "recovery_attempted_rate",
        "valid_action_rate",
        "snapshot_family_hit_rate",
        "teacher_reachable_rate",
        "history_condition_usage_probe",
    ]
    assert [line["line_key"] for line in scorecard["lines"]] == ["baseline", "c0", "c1"]
    assert [line["line_label"] for line in scorecard["lines"]] == [
        "original baseline",
        "C0 history-aware equal-data control",
        "C1 + dev-only oracle-supplied phase/mode",
    ]
    assert all(
        line["counts"]["evaluated_episodes"] == 32 for line in scorecard["lines"]
    )
    assert scorecard["lines"][0]["oracle_phase_mode_supplied"] is False
    assert scorecard["lines"][1]["oracle_phase_mode_supplied"] is False
    assert scorecard["lines"][2]["oracle_phase_mode_supplied"] is True
    expected_python = str(state_conditioned_oracle_eval.wbc_venv_python(REPO_ROOT))
    assert [
        line["line_invocation"]["wrapper_python"] for line in scorecard["lines"]
    ] == [
        expected_python,
        expected_python,
        expected_python,
    ]
    assert [line["line_invocation"]["eval_python"] for line in scorecard["lines"]] == [
        expected_python,
        expected_python,
        expected_python,
    ]
    assert (
        scorecard["lines"][0]["diagnostics"]["verify_hold_pass_rate"]["denominator"] > 0
    )
    assert (
        "per_family_breakdown"
        in scorecard["lines"][2]["diagnostics"]["same_failure_repeat_rate"]
    )
    assert set(scorecard["lines"][0]["diagnostic_snapshot"].keys()) == set(
        scorecard["diagnostic_metric_names"]
    )
    assert scorecard["lines"][0]["diagnostics"]["teacher_reachable_rate"][
        "reachable_rate"
    ] == pytest.approx(0.75, rel=1e-6)
    assert scorecard["lines"][0]["diagnostics"]["history_condition_usage_probe"][
        "history_condition_response"
    ]["response_ratio"] == pytest.approx(24342.162695242496, rel=1e-6)
    assert (
        scorecard["lines"][0]["diagnostics"]["max_phase_reached"]["global_max_phase"]
        == "PLACE"
    )
    assert scorecard["lines"][0]["diagnostics"]["valid_action_rate"][
        "rate"
    ] == pytest.approx(1.0, rel=1e-6)
    assert scorecard["lines"][0]["diagnostics"]["snapshot_family_hit_rate"][
        "rate"
    ] == pytest.approx(1.0, rel=1e-6)
    assert scorecard["lines"][0]["comparable_metrics"]["success_rate"] == pytest.approx(
        10.0 / 32.0, rel=1e-6
    )
    assert scorecard["lines"][1]["comparable_metrics"]["success_rate"] == pytest.approx(
        14.0 / 32.0, rel=1e-6
    )
    assert scorecard["lines"][2]["comparable_metrics"]["success_rate"] == pytest.approx(
        20.0 / 32.0, rel=1e-6
    )

    assert gate["gate_status"] == "PASS"
    assert split["next_step"] == "detector_candidate_next_round"
    assert split["legacy_next_step"] == split["next_step"]
    assert split["ab_case"] in {"A", "B", "C", "D"}
    assert split["decision_tree"]["matched_case_count"] == 1
    assert split["future_unlocks"]["pm_event_analysis_only"] is True
    assert split["future_unlocks"]["detector_candidate"] is True
    assert split["executed_actions"]["pm_event_analysis"] is False
    assert split["executed_actions"]["detector"] is False
    assert gate["legacy_next_step"] == split["legacy_next_step"]
    assert gate["ab_case"] == split["ab_case"]
    assert summary["line_order"] == ["baseline", "c0", "c1"]
    assert summary["diagnostic_metric_names"] == scorecard["diagnostic_metric_names"]
    assert summary["legacy_next_step"] == split["legacy_next_step"]
    assert summary["ab_case"] == split["ab_case"]
    assert summary["shared_diagnostics"]["teacher_reachable_rate"][
        "reachable_rate"
    ] == pytest.approx(0.75, rel=1e-6)
    assert set(summary["summary_lines"][0]["diagnostic_snapshot"].keys()) == set(
        scorecard["diagnostic_metric_names"]
    )
    assert len(entries) == 32


def test_verify_hold_pass_rate_counts_first_true_then_eight_step_hold() -> None:
    payload = state_conditioned_oracle_eval._build_verify_hold_payload(
        sidecar_by_episode={
            "pass": [
                {"t": 0, "apple_in_hand": False},
                *[{"t": index, "apple_in_hand": True} for index in range(1, 9)],
            ],
            "fail": [
                {"t": 0, "apple_in_hand": False},
                {"t": 1, "apple_in_hand": True},
                {"t": 2, "apple_in_hand": True},
                {"t": 3, "apple_in_hand": False},
            ],
            "excluded": [
                {"t": 0, "apple_in_hand": False},
                {"t": 1, "apple_in_hand": False},
            ],
        }
    )

    assert payload["numerator"] == 1
    assert payload["denominator"] == 2
    assert payload["excluded_count"] == 1
    assert pytest.approx(payload["rate"], rel=1e-6) == 0.5


def test_empty_hand_transport_rate_uses_episode_level_any_row_logic() -> None:
    payload = state_conditioned_oracle_eval._build_empty_hand_transport_payload(
        sidecar_by_episode={
            "positive": [
                {"phase": "SEARCH", "apple_in_hand": False},
                {"phase": "TRANSPORT", "apple_in_hand": False},
            ],
            "negative": [
                {"phase": "TRANSPORT", "apple_in_hand": True},
                {"phase": "PLACE", "apple_in_hand": False},
            ],
        }
    )

    assert payload["numerator"] == 1
    assert payload["denominator"] == 2
    assert pytest.approx(payload["rate"], rel=1e-6) == 0.5


def test_same_failure_repeat_counts_denominator_excluded_and_per_family() -> None:
    episode_records = [
        {
            "paired_key": "repeat",
            "stratum_id": "drop_during_transport",
            "primary_failure_family": "S_drop",
        },
        {
            "paired_key": "single",
            "stratum_id": "failed_grasp_visible",
            "primary_failure_family": "S_lost",
        },
        {
            "paired_key": "excluded",
            "stratum_id": "failed_grasp_occluded",
            "primary_failure_family": None,
        },
    ]
    sidecar_by_episode = {
        "repeat": [
            {"recovery_entry_step": 2, "recovery_exit_step": 3},
            {"recovery_entry_step": 2, "recovery_exit_step": 3},
            {"recovery_entry_step": 5, "recovery_exit_step": 6},
        ],
        "single": [
            {"recovery_entry_step": 1, "recovery_exit_step": 2},
            {"recovery_entry_step": 2, "recovery_exit_step": 2},
        ],
        "excluded": [],
    }

    payload = state_conditioned_oracle_eval._build_same_failure_repeat_payload(
        episode_records=episode_records,
        sidecar_by_episode=sidecar_by_episode,
    )

    assert payload["numerator"] == 1
    assert payload["denominator"] == 2
    assert payload["excluded_count"] == 1
    assert payload["per_family_breakdown"]["S_drop"]["repeat_positive_count"] == 1
    assert payload["per_family_breakdown"]["S_drop"]["denominator"] == 1
    assert payload["per_family_breakdown"]["S_lost"]["repeat_positive_count"] == 0
    assert payload["episode_outcomes"][0]["repeat_positive"] is True


@pytest.mark.parametrize(
    (
        "baseline",
        "c0",
        "c1",
        "expected_next_step",
        "expected_unlock",
        "expected_ab_case",
        "history_probe_passed",
        "teacher_reachable_rate",
        "baseline_phase_mean",
        "c0_phase_mean",
        "c1_phase_mean",
        "baseline_recovery_rate",
        "c0_recovery_rate",
        "c1_recovery_rate",
        "c1_valid_action_rate",
        "c1_snapshot_family_hit_rate",
    ),
    [
        (
            0.0,
            0.0,
            0.0,
            "fix_snapshot_curriculum_pseudodemo_labels",
            False,
            "A",
            True,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ),
        (
            0.0,
            0.0,
            0.0,
            "fix_snapshot_curriculum_pseudodemo_labels",
            False,
            "B",
            False,
            0.75,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.8,
            0.9,
        ),
        (
            0.0,
            0.0,
            0.0,
            "fix_snapshot_curriculum_pseudodemo_labels",
            False,
            "C",
            True,
            0.75,
            0.0,
            0.0,
            1.5,
            0.0,
            0.0,
            0.20,
            1.0,
            1.0,
        ),
        (
            0.0,
            0.0,
            0.0,
            "fix_snapshot_curriculum_pseudodemo_labels",
            False,
            "D",
            True,
            0.75,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ),
    ],
)
def test_result_split_decision_covers_three_fixed_branches(
    baseline: float,
    c0: float,
    c1: float,
    expected_next_step: str,
    expected_unlock: bool,
    expected_ab_case: str,
    history_probe_passed: bool,
    teacher_reachable_rate: float,
    baseline_phase_mean: float,
    c0_phase_mean: float,
    c1_phase_mean: float,
    baseline_recovery_rate: float,
    c0_recovery_rate: float,
    c1_recovery_rate: float,
    c1_valid_action_rate: float,
    c1_snapshot_family_hit_rate: float,
) -> None:
    line_metrics = _decision_line_metrics(
        baseline_success=baseline,
        c0_success=c0,
        c1_success=c1,
        teacher_reachable_rate=teacher_reachable_rate,
        history_probe_passed=history_probe_passed,
        baseline_phase_mean=baseline_phase_mean,
        c0_phase_mean=c0_phase_mean,
        c1_phase_mean=c1_phase_mean,
        baseline_recovery_rate=baseline_recovery_rate,
        c0_recovery_rate=c0_recovery_rate,
        c1_recovery_rate=c1_recovery_rate,
        c1_valid_action_rate=c1_valid_action_rate,
        c1_snapshot_family_hit_rate=c1_snapshot_family_hit_rate,
    )

    decision = state_conditioned_oracle_eval.build_result_split_decision(
        line_metrics_by_key=line_metrics
    )

    assert decision["next_step"] == expected_next_step
    assert decision["legacy_next_step"] == expected_next_step
    assert decision["ab_case"] == expected_ab_case
    assert decision["decision_tree"]["matched_case_count"] == 1
    assert decision["decision_tree"]["matched_cases"] == [expected_ab_case]
    assert decision["future_unlocks"]["pm_event_analysis_only"] is expected_unlock
    assert decision["future_unlocks"]["detector_candidate"] is expected_unlock
    assert decision["executed_actions"]["pm_event_analysis"] is False
    assert decision["executed_actions"]["detector"] is False


def test_result_split_decision_ignores_additive_diagnostics() -> None:
    line_metrics = _decision_line_metrics(
        baseline_success=0.30,
        c0_success=0.55,
        c1_success=0.56,
        teacher_reachable_rate=0.75,
        history_probe_passed=True,
        baseline_phase_mean=0.0,
        c0_phase_mean=3.0,
        c1_phase_mean=3.0,
        baseline_recovery_rate=0.10,
        c0_recovery_rate=0.20,
        c1_recovery_rate=0.20,
    )
    line_metrics["c1"]["diagnostics"]["history_condition_usage_probe"][
        "history_condition_response"
    ]["response_ratio"] = 999999.0

    decision = state_conditioned_oracle_eval.build_result_split_decision(
        line_metrics_by_key=line_metrics
    )

    assert decision["primary_comparator_metric"] == "success_rate"
    assert decision["next_step"] == "condition_interface_analysis"
    assert decision["legacy_next_step"] == "condition_interface_analysis"


def test_missing_t11_diff_whitelist_fails_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dev_dir, training_dir, _entries = _build_prerequisites(tmp_path)
    (training_dir / state_conditioned_train.DIFF_WHITELIST_JSON_NAME).unlink()

    exit_code = state_conditioned_oracle_eval.main(
        [
            "--dev-dir",
            str(dev_dir),
            "--training-dir",
            str(training_dir),
            "--output-dir",
            str(tmp_path / "eval"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert (
        "missing required T11 state_conditioned_training_fairness_diff_whitelist.json"
        in captured.err
    )
    assert "Traceback" not in captured.err
