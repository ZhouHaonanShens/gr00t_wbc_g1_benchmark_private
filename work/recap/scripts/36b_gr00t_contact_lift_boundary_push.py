#!/usr/bin/env python3
"""Build GR00T contact/lift telemetry audit and candidate handoff matrix.

This is the CPU-side Worker1 surface for boundary push3.  It does not train,
launch a server, or enter P5.  It turns the push2 formal remediation artifacts
into two machine-checkable handoff artifacts for the GPU1 runner:

* ``telemetry_audit/contact_lift_failure_table.json``
* ``candidate_matrix.json``

The generated candidates are formal-remediation candidates, not formal PASS
claims.  Worker2 must run them on GPU1 and refresh the 35a/P4 gate before any
P5 decision.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAILURE_TABLE_SCHEMA = "gr00t_contact_lift_failure_table_v1"
CANDIDATE_MATRIX_SCHEMA = "gr00t_contact_lift_candidate_matrix_v1"
FORMAL_SEEDS = [20260421, 20260422, 20260423]
DEFAULT_DATASET_REL = "agent/artifacts/lerobot_datasets/recap_stage3_iter_002"
DEFAULT_CONTINUATION_CHECKPOINT_REL = (
    "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/"
    "t13_advantage_full_update_1gpu/formal_run/checkpoint-200"
)
DEFAULT_BASELINE_AUTHORITY_REL = "agent/artifacts/recap_min_loop/single_gpu_v1"
DEFAULT_V2_AUTHORITY_REL = "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
ACTION_GROUPS = (
    "action.right_arm",
    "action.right_hand",
    "action.left_arm",
    "action.navigate_command",
    "action.waist",
)


def _utc_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise TypeError(f"{path}:{line_number} must be a JSON object")
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _resolve(repo_root: Path, raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _stage_label(row: Mapping[str, Any] | None) -> str | None:
    if not isinstance(row, Mapping):
        return None
    guess = row.get("failure_stage_guess")
    if isinstance(guess, Mapping):
        label = guess.get("label")
        return str(label) if label is not None else None
    if isinstance(guess, str):
        return guess
    return None


def _failure_guess_metric(row: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(row, Mapping):
        return None
    guess = row.get("failure_stage_guess")
    if not isinstance(guess, Mapping):
        return None
    return _safe_float(guess.get(key))


def _episode_rows_by_seed(path: Path | None) -> dict[int, dict[str, Any]]:
    rows = _read_jsonl(path)
    by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = _safe_int(row.get("seed"))
        if seed is not None:
            by_seed[seed] = row
    return by_seed


def _eval_summary_step_path(repo_root: Path, eval_summary_path: str | Path | None) -> Path | None:
    path = _resolve(repo_root, eval_summary_path)
    if path is None or not path.is_file():
        return None
    payload = _read_json(path)
    return _resolve(repo_root, payload.get("step_telemetry_jsonl"))


def _summarize_step_actions(path: Path | None) -> dict[int, dict[str, Any]]:
    rows = _read_jsonl(path)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seed = _safe_int(row.get("seed"))
        if seed is not None:
            grouped[seed].append(row)

    summaries: dict[int, dict[str, Any]] = {}
    for seed, seed_rows in grouped.items():
        seed_rows = sorted(seed_rows, key=lambda item: _safe_int(item.get("outer_step")) or 0)
        step_count = len(seed_rows)
        near_threshold = 0.10
        distances = [
            (_safe_float(row.get("apple_to_right_eef_l2")), _safe_int(row.get("outer_step")))
            for row in seed_rows
        ]
        valid_distances = [(dist, step) for dist, step in distances if dist is not None]
        nearest = min(valid_distances, default=(None, None), key=lambda item: item[0] if item[0] is not None else float("inf"))
        last_quartile_start = int(step_count * 0.75)
        last_rows = seed_rows[last_quartile_start:] if seed_rows else []
        near_rows = [row for row in seed_rows if (_safe_float(row.get("apple_to_right_eef_l2")) or 999.0) <= near_threshold]

        def group_mean(rows_for_window: Sequence[Mapping[str, Any]], action_group: str, field: str) -> float | None:
            values: list[float] = []
            for row in rows_for_window:
                action_summary = row.get("action_summary")
                if not isinstance(action_summary, Mapping):
                    continue
                group_payload = action_summary.get(action_group)
                if not isinstance(group_payload, Mapping):
                    continue
                value = _safe_float(group_payload.get(field))
                if value is not None:
                    values.append(value)
            return _mean(values)

        action_payload: dict[str, Any] = {}
        for group in ACTION_GROUPS:
            action_payload[group] = {
                "mean_abs_all": group_mean(seed_rows, group, "mean_abs"),
                "mean_abs_near_apple_window": group_mean(near_rows, group, "mean_abs"),
                "mean_abs_last_quartile": group_mean(last_rows, group, "mean_abs"),
                "max_abs_all": group_mean(seed_rows, group, "max_abs"),
            }
        summaries[seed] = {
            "step_count": step_count,
            "nearest_apple_outer_step": nearest[1],
            "nearest_apple_distance": nearest[0],
            "near_apple_step_count": len(near_rows),
            "last_quartile_start_index": last_quartile_start,
            "action_group_abs_summary": action_payload,
        }
    return summaries


def _source_path(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _pair_by_seed(payload: Mapping[str, Any], key: str) -> dict[int, dict[str, Any]]:
    rows = payload.get(key)
    if not isinstance(rows, list):
        return {}
    by_seed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        seed = _safe_int(row.get("seed"))
        if seed is not None:
            by_seed[seed] = row
    return by_seed


def _sweep_pairs_by_seed(sweep_result: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    results = sweep_result.get("results")
    if not isinstance(results, list):
        return {}
    for result in results:
        if not isinstance(result, Mapping):
            continue
        advantage = _safe_float(result.get("advantage"))
        tag = result.get("tag")
        eval_summary_path = result.get("eval_summary_path")
        for pair in result.get("pairs") or []:
            if not isinstance(pair, Mapping):
                continue
            seed = _safe_int(pair.get("seed"))
            if seed is None:
                continue
            row = dict(pair)
            row["advantage"] = advantage
            row["tag"] = tag
            row["eval_summary_path"] = eval_summary_path
            by_seed[seed].append(row)
    return dict(by_seed)


def _best_scalar_for_seed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None

    def score(row: Mapping[str, Any]) -> tuple[int, float, float]:
        no_regression = 1 if row.get("no_regression_on_contact_or_lift_proxy") is True else 0
        improvement = _safe_float(row.get("relative_improvement_min_dist_ee_to_apple")) or -999.0
        contact = _safe_float(row.get("conditioned_contact_or_lift_proxy")) or -999.0
        return no_regression, improvement, contact

    return dict(max(rows, key=score))


def _seed_recommendation(row: Mapping[str, Any]) -> dict[str, Any]:
    stage = str(row.get("failure_stage_label") or "unknown")
    after_improvement = _safe_float(row.get("after_relative_improvement_min_dist_ee_to_apple")) or 0.0
    after_no_regression = row.get("after_no_regression_on_contact_or_lift_proxy") is True
    best_scalar = row.get("best_scalar_candidate")
    scalar_helped_contact = isinstance(best_scalar, Mapping) and best_scalar.get("no_regression_on_contact_or_lift_proxy") is True
    near_steps = 0
    action_proxy = row.get("action_magnitude_proxy")
    if isinstance(action_proxy, Mapping):
        near_steps = int(action_proxy.get("near_apple_step_count") or 0)

    if stage == "reached_apple_not_lifted" and not after_no_regression:
        candidate_id = "failure_stage_data_reweighting_v1"
        rationale = (
            "Seed reaches the apple but loses the contact/lift proxy; emphasize late-stage "
            "positive/contact-like frames and retain fewer non-contact negatives."
        )
    elif after_improvement > 0 and not after_no_regression:
        candidate_id = "contact_lift_aware_weighting_v1"
        rationale = (
            "Distance improves while contact/lift regresses; keep the reach objective but make "
            "contact/lift no-regression a first-class weighting target."
        )
    elif stage == "never_reached_apple" or near_steps == 0:
        candidate_id = "route_action_stage_probe_v1"
        rationale = (
            "Seed remains reach-limited; test action-stage/hot-conditioning settings before "
            "spending more scalar-amplitude GPU budget."
        )
    elif scalar_helped_contact:
        candidate_id = "contact_lift_aware_weighting_v1"
        rationale = "Scalar sweep found a contact-preserving hint, but aggregate gate still needs non-scalar remediation."
    else:
        candidate_id = "contact_lift_aware_weighting_v1"
        rationale = "Default to contact/lift-aware weighting because scalar-only sweep already failed."
    return {
        "candidate_id": candidate_id,
        "candidate_type": candidate_id.removesuffix("_v1"),
        "rationale": rationale,
    }


def build_contact_lift_failure_table(
    *,
    repo_root: Path,
    formal_remediation_result_path: Path,
    sweep_result_path: Path,
    before_subgoal_summary_path: Path,
) -> dict[str, Any]:
    formal = _read_json(formal_remediation_result_path)
    sweep = _read_json(sweep_result_path)
    before = _read_json(before_subgoal_summary_path)

    before_pairs = _pair_by_seed(before, "per_seed_pairs")
    after_pairs = _pair_by_seed(formal, "paired_seed_before_after")
    sweep_by_seed = _sweep_pairs_by_seed(sweep)
    episode_path = _resolve(repo_root, formal.get("after_episode_telemetry_jsonl"))
    eval_summary_path = _resolve(repo_root, formal.get("after_eval_summary"))
    step_path = _eval_summary_step_path(repo_root, eval_summary_path)
    episode_rows = _episode_rows_by_seed(episode_path)
    action_summaries = _summarize_step_actions(step_path)

    rows: list[dict[str, Any]] = []
    missing_fields: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        before_pair = before_pairs.get(seed, {})
        after_pair = after_pairs.get(seed, {})
        episode_row = episode_rows.get(seed)
        scalar_rows = sweep_by_seed.get(seed, [])
        best_scalar = _best_scalar_for_seed(scalar_rows)
        row: dict[str, Any] = {
            "seed": seed,
            "failure_stage_label": after_pair.get("failure_stage_label") or _stage_label(episode_row),
            "baseline_min_dist_ee_to_apple": _safe_float(before_pair.get("baseline_min_dist_ee_to_apple")),
            "baseline_contact_or_lift_proxy": _safe_float(before_pair.get("baseline_contact_or_lift_proxy")),
            "continuation_min_dist_ee_to_apple": _safe_float(before_pair.get("continuation_min_dist_ee_to_apple")),
            "continuation_contact_or_lift_proxy": _safe_float(before_pair.get("continuation_contact_or_lift_proxy")),
            "control_best_min_dist_ee_to_apple": _safe_float(
                after_pair.get("control_best_min_dist_ee_to_apple")
                if after_pair.get("control_best_min_dist_ee_to_apple") is not None
                else before_pair.get("control_best_min_dist_ee_to_apple")
            ),
            "control_best_contact_or_lift_proxy": _safe_float(after_pair.get("control_best_contact_or_lift_proxy")),
            "before_conditioned_min_dist_ee_to_apple": _safe_float(after_pair.get("before_conditioned_min_dist_ee_to_apple")),
            "before_conditioned_contact_or_lift_proxy": _safe_float(after_pair.get("before_conditioned_contact_or_lift_proxy")),
            "before_relative_improvement_min_dist_ee_to_apple": _safe_float(after_pair.get("before_relative_improvement_min_dist_ee_to_apple")),
            "after_conditioned_min_dist_ee_to_apple": _safe_float(after_pair.get("after_conditioned_min_dist_ee_to_apple")),
            "after_conditioned_contact_or_lift_proxy": _safe_float(after_pair.get("after_conditioned_contact_or_lift_proxy")),
            "after_relative_improvement_min_dist_ee_to_apple": _safe_float(after_pair.get("after_relative_improvement_min_dist_ee_to_apple")),
            "after_no_regression_on_contact_or_lift_proxy": _bool_or_none(after_pair.get("after_no_regression_on_contact_or_lift_proxy")),
            "distance_improved_after": (_safe_float(after_pair.get("after_relative_improvement_min_dist_ee_to_apple")) or 0.0) > 1e-9,
            "contact_or_lift_regressed_after": after_pair.get("after_no_regression_on_contact_or_lift_proxy") is False,
            "episode_terminal_proxy": {
                "failure_reason": None if episode_row is None else episode_row.get("failure_reason"),
                "outer_steps": None if episode_row is None else episode_row.get("outer_steps"),
                "min_apple_to_right_eef_l2": _failure_guess_metric(episode_row, "min_apple_to_right_eef_l2"),
                "min_apple_to_plate_l2": _failure_guess_metric(episode_row, "min_apple_to_plate_l2"),
                "max_apple_lift_z": _failure_guess_metric(episode_row, "max_apple_lift_z"),
                "ever_near_apple": None
                if not isinstance(episode_row, Mapping) or not isinstance(episode_row.get("failure_stage_guess"), Mapping)
                else episode_row["failure_stage_guess"].get("ever_near_apple"),
                "ever_lifted_apple": None
                if not isinstance(episode_row, Mapping) or not isinstance(episode_row.get("failure_stage_guess"), Mapping)
                else episode_row["failure_stage_guess"].get("ever_lifted_apple"),
            },
            "action_magnitude_proxy": action_summaries.get(seed, {}),
            "scalar_sweep_candidates": [dict(item) for item in scalar_rows],
            "best_scalar_candidate": best_scalar,
            "source_paths": {
                "before_subgoal_summary": str(before_subgoal_summary_path),
                "formal_remediation_result": str(formal_remediation_result_path),
                "sweep_result": str(sweep_result_path),
                "after_episode_telemetry_jsonl": _source_path(episode_path),
                "after_step_telemetry_jsonl": _source_path(step_path),
            },
        }
        required = (
            "failure_stage_label",
            "baseline_min_dist_ee_to_apple",
            "continuation_min_dist_ee_to_apple",
            "control_best_min_dist_ee_to_apple",
            "control_best_contact_or_lift_proxy",
            "after_conditioned_min_dist_ee_to_apple",
            "after_conditioned_contact_or_lift_proxy",
            "after_relative_improvement_min_dist_ee_to_apple",
            "after_no_regression_on_contact_or_lift_proxy",
            "best_scalar_candidate",
        )
        missing = [field for field in required if row.get(field) is None]
        if not row.get("action_magnitude_proxy"):
            missing.append("action_magnitude_proxy")
        if missing:
            missing_fields.append({"seed": seed, "missing_fields": missing})
        row["candidate_recommendation"] = _seed_recommendation(row)
        rows.append(row)

    telemetry_complete = not missing_fields
    contact_regression_seeds = [
        int(row["seed"]) for row in rows if row.get("contact_or_lift_regressed_after") is True
    ]
    reached_not_lifted_seeds = [
        int(row["seed"]) for row in rows if row.get("failure_stage_label") == "reached_apple_not_lifted"
    ]
    never_reached_seeds = [
        int(row["seed"]) for row in rows if row.get("failure_stage_label") == "never_reached_apple"
    ]
    return {
        "schema_version": FAILURE_TABLE_SCHEMA,
        "status": "READY" if telemetry_complete else "BLOCK",
        "blocker_code": None if telemetry_complete else "telemetry_incomplete_block",
        "telemetry_complete": telemetry_complete,
        "missing_fields": missing_fields,
        "formal_seed_set": FORMAL_SEEDS,
        "aggregate": {
            "contact_regression_seeds": contact_regression_seeds,
            "reached_apple_not_lifted_seeds": reached_not_lifted_seeds,
            "never_reached_apple_seeds": never_reached_seeds,
            "distance_improved_after_count": sum(1 for row in rows if row.get("distance_improved_after") is True),
            "contact_or_lift_regressed_after_count": len(contact_regression_seeds),
            "scalar_candidates_tested": sweep.get("advantages_tested", []),
            "scalar_candidate_found": bool(sweep.get("candidate_found", False)),
            "formal_after_blocking_reasons": list(formal.get("blocking_reasons") or []),
        },
        "rows": rows,
        "authority_inputs": [
            str(formal_remediation_result_path),
            str(sweep_result_path),
            str(before_subgoal_summary_path),
        ],
    }


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _candidate_train_command(
    *,
    python: str,
    training_script: str,
    dataset_path: str,
    output_dir: str,
    runtime_log_dir: str,
    summary_json: str,
    continuation_checkpoint_path: str,
    seed: int,
    max_steps: int,
    negative_retain_probability: float,
    positive_oversample_factor: int,
    late_stage_threshold: float,
    condition_hot_lr_scale: float,
    diffusion_trunk_lr_scale: float,
    tune_diffusion_model: bool = True,
    tune_vlln: bool = True,
) -> str:
    parts = [
        "timeout",
        "7200",
        "env",
        "CUDA_VISIBLE_DEVICES=1",
        "NO_ALBUMENTATIONS_UPDATE=1",
        python,
        training_script,
        "--dataset-path",
        dataset_path,
        "--output-dir",
        output_dir,
        "--summary-json",
        summary_json,
        "--runtime-log-dir",
        runtime_log_dir,
        "--runtime-log-prefix",
        "boundary_push3_gr00t_candidate",
        "--python",
        python,
        "--max-steps",
        str(max_steps),
        "--save-steps",
        "50",
        "--save-total-limit",
        "1",
        "--global-batch-size",
        "4",
        "--gradient-accumulation-steps",
        "4",
        "--dataloader-num-workers",
        "0",
        "--learning-rate",
        "1e-5",
        "--recap-train-scope",
        "strict_full",
        "--no-balanced-advantage-batches",
        "--no-write-conditioning-functional-probe",
        "--no-write-paired-action-probe",
        "--no-write-label-semantics-audit",
        "--no-write-shuffled-advantage-negative-control",
        "--positive-curriculum",
        "--negative-retain-probability",
        str(negative_retain_probability),
        "--positive-curriculum-seed",
        str(seed),
        "--late-stage-positive-emphasis",
        "--late-stage-threshold",
        str(late_stage_threshold),
        "--positive-oversample-factor",
        str(positive_oversample_factor),
        "--condition-focused-continuation",
        "--continuation-checkpoint-path",
        continuation_checkpoint_path,
        "--condition-hot-lr-scale",
        str(condition_hot_lr_scale),
        "--diffusion-trunk-lr-scale",
        str(diffusion_trunk_lr_scale),
        "--num-gpus",
        "1",
        "--seed",
        str(seed),
        "--tune-top-llm-layers",
        "0",
        "--no-tune-projector",
        "--tune-diffusion-model" if tune_diffusion_model else "--no-tune-diffusion-model",
        "--tune-vlln" if tune_vlln else "--no-tune-vlln",
        "--no-use-wandb",
        "--transformers-local-files-only",
    ]
    return _shell_join(parts)


def _formal_refresh_command(
    *,
    python: str,
    formal_refresh_script: str,
    baseline_authority_root: str,
    v2_authority_root: str,
    candidate_train_root: str,
    output_dir: str,
) -> str:
    parts = [
        "timeout",
        "2400",
        "env",
        "CUDA_VISIBLE_DEVICES=1",
        "NO_ALBUMENTATIONS_UPDATE=1",
        python,
        formal_refresh_script,
        "--mode",
        "p4",
        "--baseline-authority-root",
        baseline_authority_root,
        "--v2-authority-root",
        v2_authority_root,
        "--conditioned-run-root",
        candidate_train_root,
        "--continuation-run-root",
        f"{v2_authority_root}/t13_continuation_full_update_1gpu",
        "--output-dir",
        output_dir,
        "--baseline-v1-subgoal-override",
        f"{v2_authority_root}/p5_gate_eval/baseline_first_subgoal_probe_v1.json",
    ]
    return _shell_join(parts)


def _recommended_seeds(table: Mapping[str, Any], candidate_id: str) -> list[int]:
    seeds: list[int] = []
    for row in table.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        recommendation = row.get("candidate_recommendation")
        if isinstance(recommendation, Mapping) and recommendation.get("candidate_id") == candidate_id:
            seed = _safe_int(row.get("seed"))
            if seed is not None:
                seeds.append(seed)
    return seeds


def build_candidate_matrix(
    *,
    failure_table: Mapping[str, Any],
    timestamp: str,
    python: str,
    dataset_path: str,
    continuation_checkpoint_path: str,
    artifact_root: str,
    runtime_log_root: str,
    baseline_authority_root: str,
    v2_authority_root: str,
    training_script: str | None = None,
    formal_refresh_script: str | None = None,
    eval_script: str | None = None,
) -> dict[str, Any]:
    aggregate = failure_table.get("aggregate") if isinstance(failure_table.get("aggregate"), Mapping) else {}

    def candidate_paths(candidate_id: str) -> dict[str, str]:
        root = f"{artifact_root}/candidates/{candidate_id}"
        return {
            "root": root,
            "train_root": f"{root}/train",
            "summary_json": f"{root}/train_summary.json",
            "formal_refresh_root": f"{root}/formal_refresh",
            "runtime_log_dir": f"{runtime_log_root}/{candidate_id}",
        }

    candidates: list[dict[str, Any]] = []
    training_script = training_script or "work/recap/scripts/34b_recap_numeric_adv_smoke.py"
    formal_refresh_script = formal_refresh_script or "work/recap/scripts/35a_full_update_rollout_probe.py"
    eval_script = eval_script or "work/recap/scripts/3D_recap_eval.py"

    specs = [
        {
            "candidate_id": "contact_lift_aware_weighting_v1",
            "priority": 1,
            "candidate_type": "contact_lift_aware_weighting",
            "hypothesis": "Late-stage positive/contact-like weighting can keep the 2/3 distance gain while stopping contact/lift proxy regression.",
            "expected_metric_movement": {
                "paired_seed_improvement_count": ">=2/3 retained",
                "mean_relative_improvement_min_dist_ee_to_apple": ">=0.05 retained",
                "no_regression_on_contact_or_lift_proxy": "moves false -> true by emphasizing late contact/lift-positive frames",
            },
            "params": {
                "max_steps": 200,
                "negative_retain_probability": 0.45,
                "positive_oversample_factor": 4,
                "late_stage_threshold": 0.65,
                "condition_hot_lr_scale": 4.0,
                "diffusion_trunk_lr_scale": 0.5,
                "tune_diffusion_model": True,
                "tune_vlln": True,
                "seed": 20260421,
            },
        },
        {
            "candidate_id": "failure_stage_data_reweighting_v1",
            "priority": 2,
            "candidate_type": "failure_stage_data_reweighting",
            "hypothesis": "Reached-apple-not-lifted seeds need stronger retention of near-apple/contact-positive training units and fewer non-contact negatives.",
            "expected_metric_movement": {
                "paired_seed_improvement_count": ">=2/3 retained",
                "mean_relative_improvement_min_dist_ee_to_apple": "may trade some reach margin for contact no-regression",
                "no_regression_on_contact_or_lift_proxy": "improves first on reached_apple_not_lifted seeds",
            },
            "params": {
                "max_steps": 200,
                "negative_retain_probability": 0.25,
                "positive_oversample_factor": 5,
                "late_stage_threshold": 0.50,
                "condition_hot_lr_scale": 3.5,
                "diffusion_trunk_lr_scale": 0.75,
                "tune_diffusion_model": True,
                "tune_vlln": True,
                "seed": 20260422,
            },
        },
        {
            "candidate_id": "route_action_stage_probe_v1",
            "priority": 3,
            "candidate_type": "route_action_stage_candidate",
            "hypothesis": "A hot-conditioning/action-head-biased continuation can change late hand/arm behavior without broad diffusion drift.",
            "expected_metric_movement": {
                "paired_seed_improvement_count": "diagnoses whether action-stage adaptation preserves reach",
                "mean_relative_improvement_min_dist_ee_to_apple": "should not collapse below 0.05 before formal refresh",
                "no_regression_on_contact_or_lift_proxy": "tests whether VLLN/action-head emphasis improves hand/contact behavior",
            },
            "params": {
                "max_steps": 120,
                "negative_retain_probability": 0.60,
                "positive_oversample_factor": 3,
                "late_stage_threshold": 0.70,
                "condition_hot_lr_scale": 5.0,
                "diffusion_trunk_lr_scale": 0.0,
                "tune_diffusion_model": False,
                "tune_vlln": True,
                "seed": 20260423,
            },
        },
    ]

    for spec in specs:
        candidate_id = str(spec["candidate_id"])
        paths = candidate_paths(candidate_id)
        params = spec["params"]
        train_cmd = _candidate_train_command(
            python=python,
            training_script=training_script,
            dataset_path=dataset_path,
            output_dir=paths["train_root"],
            runtime_log_dir=paths["runtime_log_dir"],
            summary_json=paths["summary_json"],
            continuation_checkpoint_path=continuation_checkpoint_path,
            seed=int(params["seed"]),
            max_steps=int(params["max_steps"]),
            negative_retain_probability=float(params["negative_retain_probability"]),
            positive_oversample_factor=int(params["positive_oversample_factor"]),
            late_stage_threshold=float(params["late_stage_threshold"]),
            condition_hot_lr_scale=float(params["condition_hot_lr_scale"]),
            diffusion_trunk_lr_scale=float(params["diffusion_trunk_lr_scale"]),
            tune_diffusion_model=bool(params["tune_diffusion_model"]),
            tune_vlln=bool(params["tune_vlln"]),
        )
        refresh_cmd = _formal_refresh_command(
            python=python,
            formal_refresh_script=formal_refresh_script,
            baseline_authority_root=baseline_authority_root,
            v2_authority_root=v2_authority_root,
            candidate_train_root=paths["train_root"],
            output_dir=paths["formal_refresh_root"],
        )
        recommended = _recommended_seeds(failure_table, candidate_id)
        if not recommended:
            if candidate_id == "contact_lift_aware_weighting_v1":
                recommended = list(aggregate.get("contact_regression_seeds") or FORMAL_SEEDS)
            elif candidate_id == "failure_stage_data_reweighting_v1":
                recommended = list(aggregate.get("reached_apple_not_lifted_seeds") or [])
            else:
                recommended = list(aggregate.get("never_reached_apple_seeds") or [])
        candidates.append(
            {
                "candidate_id": candidate_id,
                "priority": int(spec["priority"]),
                "track": "formal_remediation",
                "candidate_type": spec["candidate_type"],
                "graduation_stage": "C2_DRY_RUN",
                "graduation_evidence": [
                    "C0_STATIC: manifest fields complete",
                    "C1_TELEMETRY: derived from contact_lift_failure_table rows",
                    "C2_DRY_RUN: command is GPU1-bound, timeout-guarded, route-frozen, and P5 is not entered",
                ],
                "hypothesis": spec["hypothesis"],
                "recommended_for_seeds": recommended,
                "formal_seed_set": FORMAL_SEEDS,
                "expected_metric_movement": spec["expected_metric_movement"],
                "implementation_surface": {
                    "training_script": training_script,
                    "formal_refresh_script": formal_refresh_script,
                    "dataset_path": dataset_path,
                    "continuation_checkpoint_path": continuation_checkpoint_path,
                    "route_freeze": "advantage_input_numeric_diagnostic; no exploratory route may unlock formal",
                },
                "commands": {
                    "gpu1_train": train_cmd,
                    "gpu1_formal_refresh": refresh_cmd,
                    "p5_gate": "Do not run P5 unless formal_refresh/full_update_diagnostic_summary.json has status=PASS, formal_claim_allowed=true, blocking_reasons=[], p5_formal_10ep_eligible=true.",
                },
                "gpu_budget": {"gpu": "1", "timeout_s": 7200, "long_task_requires_lock": True},
                "formal_gate_impact": "May only produce a new P4 formal candidate; P5 remains forbidden until Worker2/verifier confirms a clean P4 PASS.",
                "rollback_plan": "Delete candidate output dir and ignore this candidate; no shared model checkpoint is overwritten.",
                "forbidden_inferences": [
                    "C2_DRY_RUN is not a runtime PASS",
                    "candidate train completion is not P5 eligibility",
                    "exploratory seed/action changes must not unlock formal status",
                ],
            }
        )

    scalar_paths = candidate_paths("scalar_amplitude_negative_control_v1")
    candidates.append(
        {
            "candidate_id": "scalar_amplitude_negative_control_v1",
            "priority": 99,
            "track": "exploratory_negative_control",
            "candidate_type": "scalar_amplitude_negative_control",
            "graduation_stage": "C1_TELEMETRY",
            "hypothesis": "Scalar-only amplitude already failed and is retained only as a negative control.",
            "recommended_for_seeds": FORMAL_SEEDS,
            "formal_seed_set": FORMAL_SEEDS,
            "expected_metric_movement": {
                "paired_seed_improvement_count": "known 1-2/3 range from push2",
                "no_regression_on_contact_or_lift_proxy": "expected to remain false based on push2 sweep",
            },
            "implementation_surface": {
                "evaluation_script": eval_script,
                "route_freeze": "diagnostic numeric advantage only",
            },
            "commands": {
                "gpu1_eval": _shell_join(
                    [
                        "timeout",
                        "1800",
                        "env",
                        "CUDA_VISIBLE_DEVICES=1",
                        "NO_ALBUMENTATIONS_UPDATE=1",
                        python,
                        eval_script,
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "<worker2_candidate_server_port>",
                        "--n-episodes",
                        "3",
                        "--seed-base",
                        "20260421",
                        "--advantage",
                        "0.75",
                        "--runtime-log-dir",
                        f"{runtime_log_root}/scalar_amplitude_negative_control_v1",
                        "--artifact-dir",
                        scalar_paths["root"],
                        "--telemetry-dir",
                        f"{scalar_paths['root']}/telemetry",
                        "--summary-json",
                        f"{scalar_paths['root']}/eval_summary.json",
                    ]
                )
            },
            "gpu_budget": {"gpu": "1", "timeout_s": 1800, "long_task_requires_lock": True},
            "formal_gate_impact": "No formal unlock; scalar-only remediation is explicitly not sufficient for boundary push3.",
            "rollback_plan": "Ignore negative-control artifacts for formal candidate selection.",
            "forbidden_inferences": ["negative-control signal must not unlock P5"],
        }
    )

    non_scalar_formal = [
        c for c in candidates if c["track"] == "formal_remediation" and c["candidate_type"] != "scalar_amplitude_negative_control"
    ]
    matrix_status = "READY" if failure_table.get("telemetry_complete") and len(non_scalar_formal) >= 2 else "BLOCK"
    return {
        "schema_version": CANDIDATE_MATRIX_SCHEMA,
        "status": matrix_status,
        "blocker_code": None if matrix_status == "READY" else "candidate_matrix_incomplete_block",
        "timestamp": timestamp,
        "formal_seed_set": FORMAL_SEEDS,
        "source_failure_table_schema": failure_table.get("schema_version"),
        "source_failure_table_status": failure_table.get("status"),
        "non_scalar_formal_candidate_count": len(non_scalar_formal),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "worker2_handoff": {
            "lane": "gr00t_gpu1",
            "requires_gpu_lock": "agent/runtime_logs/boundary_push3_<ts>/locks/gpu1_gr00t.lock",
            "lock_file_for_this_run": f"{runtime_log_root.rsplit('/gr00t', 1)[0]}/locks/gpu1_gr00t.lock",
            "resource_lease_schema": "resource_lease_v1",
            "resource_lease_required_fields": ["lane", "gpu", "worker", "command", "started_at_utc", "ended_at_utc", "returncode", "timeout_s", "runtime_log", "artifacts", "forbidden_gpus_visible", "sudo_used"],
            "launch_cwd_required": "Commands use absolute data/artifact/script paths and are safe from worktree cwd differences; still prefer leader cwd for consistency.",
            "preflight_required": ["verify dataset_path exists", "verify continuation checkpoint exists", "verify GPU1 lock is held before gpu1_train/gpu1_formal_refresh", "write resource_lease_v1 before marking PASS/BLOCK"],
            "do_not_enter_p5_until_clean_p4_pass": True,
            "preferred_run_order": [c["candidate_id"] for c in sorted(non_scalar_formal, key=lambda item: item["priority"])],
        },
    }


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-repo-root", default=str(REPO_ROOT))
    parser.add_argument("--timestamp", default=_utc_ts())
    parser.add_argument("--formal-remediation-result", required=True)
    parser.add_argument("--sweep-result", required=True)
    parser.add_argument("--before-subgoal-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-log-root", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_REL)
    parser.add_argument("--continuation-checkpoint-path", default=DEFAULT_CONTINUATION_CHECKPOINT_REL)
    parser.add_argument("--baseline-authority-root", default=DEFAULT_BASELINE_AUTHORITY_REL)
    parser.add_argument("--v2-authority-root", default=DEFAULT_V2_AUTHORITY_REL)
    args = parser.parse_args(argv)

    artifact_repo_root = Path(args.artifact_repo_root).resolve()
    formal_path = _resolve(artifact_repo_root, args.formal_remediation_result)
    sweep_path = _resolve(artifact_repo_root, args.sweep_result)
    before_path = _resolve(artifact_repo_root, args.before_subgoal_summary)
    output_dir = _resolve(artifact_repo_root, args.output_dir)
    if formal_path is None or sweep_path is None or before_path is None or output_dir is None:
        raise ValueError("formal/sweep/before/output paths must resolve")
    for label, path in (
        ("formal_remediation_result", formal_path),
        ("sweep_result", sweep_path),
        ("before_subgoal_summary", before_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    runtime_log_root = str(args.runtime_log_root).strip() or f"agent/runtime_logs/boundary_push3_{args.timestamp}/gr00t"
    artifact_root = str(output_dir)
    failure_table = build_contact_lift_failure_table(
        repo_root=artifact_repo_root,
        formal_remediation_result_path=formal_path,
        sweep_result_path=sweep_path,
        before_subgoal_summary_path=before_path,
    )
    failure_table_path = output_dir / "telemetry_audit" / "contact_lift_failure_table.json"
    _write_json(failure_table_path, failure_table)

    matrix = build_candidate_matrix(
        failure_table=failure_table,
        timestamp=str(args.timestamp),
        python=str(args.python),
        dataset_path=str(_resolve(artifact_repo_root, args.dataset_path)),
        continuation_checkpoint_path=str(_resolve(artifact_repo_root, args.continuation_checkpoint_path)),
        artifact_root=artifact_root,
        runtime_log_root=runtime_log_root,
        baseline_authority_root=str(_resolve(artifact_repo_root, args.baseline_authority_root)),
        v2_authority_root=str(_resolve(artifact_repo_root, args.v2_authority_root)),
        training_script=str(artifact_repo_root / "work/recap/scripts/34b_recap_numeric_adv_smoke.py"),
        formal_refresh_script=str(artifact_repo_root / "work/recap/scripts/35a_full_update_rollout_probe.py"),
        eval_script=str(artifact_repo_root / "work/recap/scripts/3D_recap_eval.py"),
    )
    candidate_matrix_path = output_dir / "candidate_matrix.json"
    _write_json(candidate_matrix_path, matrix)

    result = {
        "schema_version": "gr00t_contact_lift_boundary_push_worker1_result_v1",
        "status": "READY" if failure_table["status"] == "READY" and matrix["status"] == "READY" else "BLOCK",
        "timestamp": str(args.timestamp),
        "telemetry_audit_path": str(failure_table_path),
        "candidate_matrix_path": str(candidate_matrix_path),
        "telemetry_complete": failure_table["telemetry_complete"],
        "non_scalar_formal_candidate_count": matrix["non_scalar_formal_candidate_count"],
        "formal_seed_set": FORMAL_SEEDS,
    }
    _write_json(output_dir / "worker1_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
