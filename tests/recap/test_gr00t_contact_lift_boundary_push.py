from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = REPO_ROOT / "work/recap/scripts/36b_gr00t_contact_lift_boundary_push.py"
    spec = importlib.util.spec_from_file_location("gr00t_contact_lift_boundary_push", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _episode(seed: int, *, label: str, min_dist: float, lift: float, near: bool) -> dict[str, object]:
    return {
        "seed": seed,
        "outer_steps": 4,
        "failure_reason": "terminated_without_success",
        "failure_stage_guess": {
            "label": label,
            "ever_near_apple": near,
            "ever_lifted_apple": False,
            "min_apple_to_right_eef_l2": min_dist,
            "min_apple_to_plate_l2": 1.1,
            "max_apple_lift_z": lift,
        },
    }


def _step(seed: int, outer_step: int, dist: float, right_hand: float) -> dict[str, object]:
    return {
        "seed": seed,
        "outer_step": outer_step,
        "apple_to_right_eef_l2": dist,
        "action_summary": {
            "action.right_arm": {"mean_abs": 0.2 + outer_step / 100.0, "max_abs": 0.4},
            "action.right_hand": {"mean_abs": right_hand, "max_abs": right_hand + 0.1},
            "action.left_arm": {"mean_abs": 0.01, "max_abs": 0.02},
            "action.navigate_command": {"mean_abs": 0.03, "max_abs": 0.04},
            "action.waist": {"mean_abs": 0.05, "max_abs": 0.06},
        },
    }


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    before_path = tmp_path / "before" / "subgoal_summary_3seed.json"
    _write_json(
        before_path,
        {
            "per_seed_pairs": [
                {
                    "seed": 20260421,
                    "baseline_min_dist_ee_to_apple": 0.27,
                    "baseline_contact_or_lift_proxy": 0.0,
                    "continuation_min_dist_ee_to_apple": 0.18,
                    "continuation_contact_or_lift_proxy": 0.0,
                    "control_best_min_dist_ee_to_apple": 0.18,
                    "conditioned_min_dist_ee_to_apple": 0.22,
                    "conditioned_contact_or_lift_proxy": 0.0,
                    "relative_improvement_min_dist_ee_to_apple": -0.2,
                },
                {
                    "seed": 20260422,
                    "baseline_min_dist_ee_to_apple": 0.12,
                    "baseline_contact_or_lift_proxy": 0.002,
                    "continuation_min_dist_ee_to_apple": 0.13,
                    "continuation_contact_or_lift_proxy": 0.004,
                    "control_best_min_dist_ee_to_apple": 0.12,
                    "conditioned_min_dist_ee_to_apple": 0.14,
                    "conditioned_contact_or_lift_proxy": 0.003,
                    "relative_improvement_min_dist_ee_to_apple": -0.1,
                },
                {
                    "seed": 20260423,
                    "baseline_min_dist_ee_to_apple": 0.19,
                    "baseline_contact_or_lift_proxy": 0.0,
                    "continuation_min_dist_ee_to_apple": 0.13,
                    "continuation_contact_or_lift_proxy": 0.001,
                    "control_best_min_dist_ee_to_apple": 0.13,
                    "conditioned_min_dist_ee_to_apple": 0.11,
                    "conditioned_contact_or_lift_proxy": 0.0,
                    "relative_improvement_min_dist_ee_to_apple": 0.17,
                },
            ]
        },
    )

    episode_path = tmp_path / "formal" / "telemetry" / "episodes.jsonl"
    step_path = tmp_path / "formal" / "telemetry" / "steps.jsonl"
    eval_summary_path = tmp_path / "formal" / "eval_summary.json"
    _write_jsonl(
        episode_path,
        [
            _episode(20260421, label="never_reached_apple", min_dist=0.20, lift=0.001, near=False),
            _episode(20260422, label="reached_apple_not_lifted", min_dist=0.07, lift=0.002, near=True),
            _episode(20260423, label="reached_apple_not_lifted", min_dist=0.05, lift=0.0, near=True),
        ],
    )
    _write_jsonl(
        step_path,
        [
            _step(seed, step, dist=0.25 - step * 0.04, right_hand=0.1 + step / 100.0)
            for seed in MODULE.FORMAL_SEEDS
            for step in range(1, 5)
        ],
    )
    _write_json(eval_summary_path, {"step_telemetry_jsonl": str(step_path)})

    formal_path = tmp_path / "formal" / "formal_remediation_result.json"
    _write_json(
        formal_path,
        {
            "after_episode_telemetry_jsonl": str(episode_path),
            "after_eval_summary": str(eval_summary_path),
            "blocking_reasons": ["contact_or_lift_proxy_regression"],
            "paired_seed_before_after": [
                {
                    "seed": 20260421,
                    "failure_stage_label": "never_reached_apple",
                    "control_best_min_dist_ee_to_apple": 0.18,
                    "control_best_contact_or_lift_proxy": 0.0001,
                    "before_conditioned_min_dist_ee_to_apple": 0.22,
                    "before_conditioned_contact_or_lift_proxy": 0.0,
                    "before_relative_improvement_min_dist_ee_to_apple": -0.2,
                    "after_conditioned_min_dist_ee_to_apple": 0.20,
                    "after_conditioned_contact_or_lift_proxy": 0.001,
                    "after_relative_improvement_min_dist_ee_to_apple": -0.05,
                    "after_no_regression_on_contact_or_lift_proxy": True,
                },
                {
                    "seed": 20260422,
                    "failure_stage_label": "reached_apple_not_lifted",
                    "control_best_min_dist_ee_to_apple": 0.12,
                    "control_best_contact_or_lift_proxy": 0.004,
                    "before_conditioned_min_dist_ee_to_apple": 0.14,
                    "before_conditioned_contact_or_lift_proxy": 0.003,
                    "before_relative_improvement_min_dist_ee_to_apple": -0.1,
                    "after_conditioned_min_dist_ee_to_apple": 0.07,
                    "after_conditioned_contact_or_lift_proxy": 0.002,
                    "after_relative_improvement_min_dist_ee_to_apple": 0.42,
                    "after_no_regression_on_contact_or_lift_proxy": False,
                },
                {
                    "seed": 20260423,
                    "failure_stage_label": "reached_apple_not_lifted",
                    "control_best_min_dist_ee_to_apple": 0.13,
                    "control_best_contact_or_lift_proxy": 0.001,
                    "before_conditioned_min_dist_ee_to_apple": 0.11,
                    "before_conditioned_contact_or_lift_proxy": 0.0,
                    "before_relative_improvement_min_dist_ee_to_apple": 0.17,
                    "after_conditioned_min_dist_ee_to_apple": 0.05,
                    "after_conditioned_contact_or_lift_proxy": 0.0,
                    "after_relative_improvement_min_dist_ee_to_apple": 0.56,
                    "after_no_regression_on_contact_or_lift_proxy": False,
                },
            ],
        },
    )

    sweep_path = tmp_path / "sweep" / "sweep_result.json"
    _write_json(
        sweep_path,
        {
            "advantages_tested": [0.75, 0.85],
            "candidate_found": False,
            "results": [
                {
                    "advantage": 0.75,
                    "tag": "0p75",
                    "eval_summary_path": "candidate.json",
                    "pairs": [
                        {
                            "seed": seed,
                            "conditioned_contact_or_lift_proxy": 0.001,
                            "conditioned_min_dist_ee_to_apple": 0.09,
                            "control_best_contact_or_lift_proxy": 0.002,
                            "control_best_min_dist_ee_to_apple": 0.12,
                            "failure_stage_label": "reached_apple_not_lifted",
                            "no_regression_on_contact_or_lift_proxy": seed == 20260421,
                            "relative_improvement_min_dist_ee_to_apple": 0.2,
                        }
                        for seed in MODULE.FORMAL_SEEDS
                    ],
                }
            ],
        },
    )
    return formal_path, sweep_path, before_path


def test_contact_lift_failure_table_is_complete_and_seed_recommended(tmp_path: Path) -> None:
    formal_path, sweep_path, before_path = _fixture_inputs(tmp_path)

    table = MODULE.build_contact_lift_failure_table(
        repo_root=tmp_path,
        formal_remediation_result_path=formal_path,
        sweep_result_path=sweep_path,
        before_subgoal_summary_path=before_path,
    )

    assert table["schema_version"] == "gr00t_contact_lift_failure_table_v1"
    assert table["status"] == "READY"
    assert table["telemetry_complete"] is True
    assert len(table["rows"]) == 3
    seed22 = next(row for row in table["rows"] if row["seed"] == 20260422)
    assert seed22["failure_stage_label"] == "reached_apple_not_lifted"
    assert seed22["baseline_min_dist_ee_to_apple"] == 0.12
    assert seed22["continuation_min_dist_ee_to_apple"] == 0.13
    assert seed22["control_best_contact_or_lift_proxy"] == 0.004
    assert seed22["after_no_regression_on_contact_or_lift_proxy"] is False
    assert seed22["action_magnitude_proxy"]["near_apple_step_count"] >= 1
    assert seed22["candidate_recommendation"]["candidate_id"] == "failure_stage_data_reweighting_v1"


def test_candidate_matrix_has_non_scalar_c2_gpu1_timeout_candidates(tmp_path: Path) -> None:
    formal_path, sweep_path, before_path = _fixture_inputs(tmp_path)
    table = MODULE.build_contact_lift_failure_table(
        repo_root=tmp_path,
        formal_remediation_result_path=formal_path,
        sweep_result_path=sweep_path,
        before_subgoal_summary_path=before_path,
    )

    matrix = MODULE.build_candidate_matrix(
        failure_table=table,
        timestamp="20260424T000000Z",
        python="python3",
        dataset_path="agent/artifacts/lerobot_datasets/recap_stage3_iter_002",
        continuation_checkpoint_path="checkpoint-200",
        artifact_root="agent/artifacts/boundary_push3_test/gr00t",
        runtime_log_root="agent/runtime_logs/boundary_push3_test/gr00t",
        baseline_authority_root="agent/artifacts/recap_min_loop/single_gpu_v1",
        v2_authority_root="agent/artifacts/recap_min_loop/single_gpu_v2_full_update",
    )

    assert matrix["schema_version"] == "gr00t_contact_lift_candidate_matrix_v1"
    assert matrix["status"] == "READY"
    assert matrix["non_scalar_formal_candidate_count"] >= 3
    formal_candidates = [c for c in matrix["candidates"] if c["track"] == "formal_remediation"]
    assert {c["candidate_type"] for c in formal_candidates} >= {
        "contact_lift_aware_weighting",
        "failure_stage_data_reweighting",
        "route_action_stage_candidate",
    }
    for candidate in formal_candidates:
        assert candidate["graduation_stage"] == "C2_DRY_RUN"
        train_cmd = candidate["commands"]["gpu1_train"]
        assert "timeout" in train_cmd
        assert "CUDA_VISIBLE_DEVICES=1" in train_cmd
        assert "sudo" not in train_cmd
        assert "CUDA_VISIBLE_DEVICES=0" not in train_cmd
        assert "CUDA_VISIBLE_DEVICES=3" not in train_cmd
        assert candidate["formal_seed_set"] == MODULE.FORMAL_SEEDS
        assert candidate["gpu_budget"]["long_task_requires_lock"] is True


def test_missing_telemetry_blocks_blind_candidate_train(tmp_path: Path) -> None:
    formal_path, sweep_path, before_path = _fixture_inputs(tmp_path)
    formal_payload = json.loads(formal_path.read_text(encoding="utf-8"))
    formal_payload.pop("after_eval_summary")
    formal_payload.pop("after_episode_telemetry_jsonl")
    _write_json(formal_path, formal_payload)

    table = MODULE.build_contact_lift_failure_table(
        repo_root=tmp_path,
        formal_remediation_result_path=formal_path,
        sweep_result_path=sweep_path,
        before_subgoal_summary_path=before_path,
    )

    assert table["status"] == "BLOCK"
    assert table["blocker_code"] == "telemetry_incomplete_block"
    assert table["telemetry_complete"] is False
    assert any("action_magnitude_proxy" in item["missing_fields"] for item in table["missing_fields"])


def test_run_resolves_worker2_handoff_paths_to_leader_absolute_paths(tmp_path: Path) -> None:
    formal_path, sweep_path, before_path = _fixture_inputs(tmp_path)
    out_rel = "agent/artifacts/boundary_push3_abs/gr00t"

    result = MODULE.run(
        [
            "--artifact-repo-root",
            str(tmp_path),
            "--timestamp",
            "20260424T052009Z",
            "--formal-remediation-result",
            str(formal_path.relative_to(tmp_path)),
            "--sweep-result",
            str(sweep_path.relative_to(tmp_path)),
            "--before-subgoal-summary",
            str(before_path.relative_to(tmp_path)),
            "--output-dir",
            out_rel,
            "--runtime-log-root",
            str(tmp_path / "agent/runtime_logs/boundary_push3_abs/gr00t"),
            "--python",
            "python3",
            "--dataset-path",
            "agent/artifacts/lerobot_datasets/recap_stage3_iter_002",
            "--continuation-checkpoint-path",
            "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/t13_advantage_full_update_1gpu/formal_run/checkpoint-200",
            "--baseline-authority-root",
            "agent/artifacts/recap_min_loop/single_gpu_v1",
            "--v2-authority-root",
            "agent/artifacts/recap_min_loop/single_gpu_v2_full_update",
        ]
    )

    matrix = json.loads(Path(result["candidate_matrix_path"]).read_text(encoding="utf-8"))
    assert matrix["worker2_handoff"]["resource_lease_schema"] == "resource_lease_v1"
    assert Path(matrix["worker2_handoff"]["lock_file_for_this_run"]).is_absolute()
    for candidate in matrix["candidates"]:
        surface = candidate.get("implementation_surface", {})
        if candidate["track"] == "formal_remediation":
            assert Path(surface["dataset_path"]).is_absolute()
            assert Path(surface["continuation_checkpoint_path"]).is_absolute()
            assert Path(surface["training_script"]).is_absolute()
            assert Path(surface["formal_refresh_script"]).is_absolute()
            assert "--dataset-path " + str(tmp_path) in candidate["commands"]["gpu1_train"]
            assert "--baseline-authority-root " + str(tmp_path) in candidate["commands"]["gpu1_formal_refresh"]
        else:
            assert Path(surface["evaluation_script"]).is_absolute()
