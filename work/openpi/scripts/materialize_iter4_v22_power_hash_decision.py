"""Materialize Iteration 4 v22 power analysis and hash-lock decision.

W7 is a no-GPU lane. It consumes upstream closure, design, calibration, and
coordinator artifacts, writes a power analysis, then hash-locks v22 only when
all twelve plan predicates are satisfied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ID = "stage1_redesign_iter4_20260425T_nextZ"
ITER3_RUN_ID = "stage1_claim_resolution_iter3_20260425T_nextZ"
PLAN_PATH = ".omc/plans/iter4_plan_v3.md"
EFFECT_SIZE_GRID = [0.05, 0.08, 0.10, 0.15]
CANDIDATE_EPISODE_COUNTS = [48, 96, 144, 192]
PRIMARY_METRIC_ID = "success_rate@selected_budget"
ALPHA = 0.05
Z_975 = 1.959963984540054
EPISODES_PER_SEED_BLOCK = 48
FALLBACK_STDEV_C_MINUS_B = 0.012
FALLBACK_STDEV_C_MINUS_X = 0.021
W3_PATTERN_STATUSES = {
    "clear_systematic",
    "pattern_present_but_not_actionable",
    "no_clear_pattern",
    "insufficient_data",
}


@dataclass(frozen=True)
class ArtifactLoad:
    present: bool
    path: Path | None
    payload: dict[str, Any] | None
    errors: list[str]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _repo_relative(repo_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _try_load_json(path: Path | None) -> ArtifactLoad:
    if path is None:
        return ArtifactLoad(False, None, None, ["artifact is missing"])
    try:
        return ArtifactLoad(True, path, _load_json(path), [])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ArtifactLoad(False, path, None, [f"artifact is malformed: {exc}"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _first_existing(repo_root: Path, relatives: list[str]) -> Path | None:
    for relative in relatives:
        candidate = repo_root / relative
        if candidate.is_file():
            return candidate
    return None


def _count_tbd(value: Any) -> int:
    if isinstance(value, str):
        return value.count("TBD")
    if isinstance(value, list):
        return sum(_count_tbd(item) for item in value)
    if isinstance(value, dict):
        return sum(_count_tbd(item) for item in value.values())
    return 0


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _two_sided_power(effect_size: float, standard_error: float, alpha: float = ALPHA) -> float:
    if standard_error <= 0:
        return 1.0
    z_effect = abs(effect_size) / standard_error
    z_crit = Z_975 if alpha == 0.05 else Z_975
    power = _normal_cdf(-z_crit - z_effect) + (1 - _normal_cdf(z_crit - z_effect))
    return max(0.0, min(1.0, power))


def _round_up_to_seed_block(episodes: float) -> int:
    blocks = max(1, math.ceil(episodes / EPISODES_PER_SEED_BLOCK))
    return blocks * EPISODES_PER_SEED_BLOCK


def _stage_root(repo_root: Path, run_id: str) -> Path:
    return repo_root / "agent" / "artifacts" / run_id


def _gr00t_root(repo_root: Path, run_id: str) -> Path:
    return repo_root / "agent" / "artifacts" / "recap_min_loop" / "single_gpu_v2_full_update" / run_id


def load_iter3_authority_index(repo_root: Path, run_id: str = RUN_ID) -> dict[str, Any]:
    path = repo_root / "agent" / "artifacts" / run_id / "coordinator" / "iter3_authority_index.json"
    loaded = _try_load_json(path if path.is_file() else None)
    stdev_c_minus_b = FALLBACK_STDEV_C_MINUS_B
    stdev_c_minus_x = FALLBACK_STDEV_C_MINUS_X
    variance_source = "plan_fallback_constants"
    if loaded.payload:
        saturation = (
            loaded.payload.get("iter3_terminal_state", {})
            .get("saturation_evidence", {})
        )
        if isinstance(saturation.get("stdev_c_minus_b"), (int, float)):
            stdev_c_minus_b = float(saturation["stdev_c_minus_b"])
            variance_source = "iter3_authority_index"
        if isinstance(saturation.get("stdev_c_minus_x"), (int, float)):
            stdev_c_minus_x = float(saturation["stdev_c_minus_x"])
            variance_source = "iter3_authority_index"
    return {
        "present": loaded.present,
        "path": _repo_relative(repo_root, loaded.path),
        "errors": loaded.errors,
        "stdev_c_minus_b": stdev_c_minus_b,
        "stdev_c_minus_x": stdev_c_minus_x,
        "stdev_iter3": max(stdev_c_minus_b, stdev_c_minus_x),
        "variance_source": variance_source,
    }


def load_gate_policy(repo_root: Path, run_id: str = RUN_ID) -> dict[str, Any]:
    policy = _try_load_json(repo_root / "agent" / "artifacts" / run_id / "coordinator" / "gate_policy.json")
    hook_report = _try_load_json(
        _first_existing(
            repo_root,
            [
                f"agent/artifacts/{run_id}/coordinator/p6_bash_hook_self_test_report.json",
                f"agent/artifacts/{run_id}/coordinator/p6_bash_hook_self_test.json",
            ],
        )
    )
    policy_payload = policy.payload or {}
    hook_payload = hook_report.payload or {}
    p6_tested = bool(
        policy_payload.get("p6_bash_hook_self_test_pass")
        or policy_payload.get("p6_bash_matcher_tested")
        or hook_payload.get("p6_bash_hook_self_test_pass")
        or hook_payload.get("passed")
    )
    return {
        "present": policy.present,
        "path": _repo_relative(repo_root, policy.path),
        "errors": policy.errors,
        "w2_gates_w5": bool(policy_payload.get("w2_gates_w5")),
        "p6_bash_matcher_tested": p6_tested,
        "p6_hook_report_path": _repo_relative(repo_root, hook_report.path),
    }


def _extract_status(payload: dict[str, Any] | None, gate: str) -> str:
    if not payload:
        return "MISSING"
    for key in (f"{gate}_status", "status", "closure_status"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return "MISSING"


def load_closure_verdict(repo_root: Path, gate: str, run_id: str = RUN_ID) -> dict[str, Any]:
    path = _first_existing(
        repo_root,
        [
            f"agent/artifacts/{run_id}/paper_audit/{gate}_closure/{gate}_closure_verdict.json",
            f"agent/artifacts/{run_id}/paper_audit/{gate}_closure_verdict.json",
            f"agent/artifacts/{run_id}/paper_audit/{gate}/{gate}_closure_verdict.json",
            f"agent/artifacts/recap_min_loop/single_gpu_v2_full_update/{run_id}/paper_audit/{gate}_closure/{gate}_closure_verdict.json",
        ],
    )
    loaded = _try_load_json(path)
    return {
        "present": loaded.present,
        "path": _repo_relative(repo_root, loaded.path),
        "errors": loaded.errors,
        "status": _extract_status(loaded.payload, gate),
    }


def load_shape_regression(repo_root: Path, run_id: str = RUN_ID) -> dict[str, Any]:
    path = _first_existing(
        repo_root,
        [
            f"agent/artifacts/recap_min_loop/single_gpu_v2_full_update/{run_id}/gr00t/r2_r4_shape_regression/r2_r4_shape_regression.json",
            f"agent/artifacts/{run_id}/gr00t/r2_r4_shape_regression/r2_r4_shape_regression.json",
        ],
    )
    loaded = _try_load_json(path)
    payload = loaded.payload or {}
    replay = payload.get("deterministic_replay") if isinstance(payload.get("deterministic_replay"), dict) else {}
    alpha = payload.get("alpha_effect") if isinstance(payload.get("alpha_effect"), dict) else {}
    checkpoint = payload.get("checkpoint") if isinstance(payload.get("checkpoint"), dict) else {}
    relative_diff = alpha.get("relative_diff")
    absolute_diff = alpha.get("absolute_diff")
    computed_alpha_pass = (
        isinstance(relative_diff, (int, float))
        and isinstance(absolute_diff, (int, float))
        and float(relative_diff) > 0.01
        and float(absolute_diff) >= 0.005
    )
    status = str(payload.get("status", "MISSING"))
    rollback_path = _first_existing(
        repo_root,
        [
            f"agent/artifacts/{run_id}/paper_audit/c2_rollback_reason.json",
            f"agent/artifacts/recap_min_loop/single_gpu_v2_full_update/{run_id}/paper_audit/c2_rollback_reason.json",
            f"agent/artifacts/recap_min_loop/single_gpu_v2_full_update/{run_id}/gr00t/r2_r4_shape_regression/c2_rollback_reason.json",
        ],
    )
    return {
        "present": loaded.present,
        "path": _repo_relative(repo_root, loaded.path),
        "errors": loaded.errors,
        "status": status,
        "deterministic_replay_match": bool(replay.get("match")),
        "deterministic_replay_rtol": replay.get("rtol"),
        "deterministic_replay_atol": replay.get("atol"),
        "alpha_effect_above_threshold": bool(alpha.get("above_threshold") or computed_alpha_pass),
        "alpha_effect": alpha,
        "checkpoint_compat_passed": bool(checkpoint.get("compat_check_passed") or payload.get("checkpoint_compat_passed")),
        "m12_rollback_triggered": bool(
            payload.get("m12_rollback_triggered")
            or rollback_path is not None
            or status == "BLOCK_post_regression"
        ),
        "rollback_path": _repo_relative(repo_root, rollback_path),
    }


def load_w5_design(repo_root: Path, run_id: str = RUN_ID) -> dict[str, Any]:
    design_root = _stage_root(repo_root, run_id) / "openpi" / "v22_desaturation_design"
    required = [
        "v22_candidate_matrix.json",
        "blind_selection_rule.json",
        "v22_metric_plan.json",
        "v22_exclusion_rules.json",
        "v22_formal_variant_definitions.json",
    ]
    payloads: list[dict[str, Any]] = []
    missing: list[str] = []
    paths: dict[str, str | None] = {}
    statuses: list[str] = []
    for name in required:
        path = design_root / name
        paths[name] = _repo_relative(repo_root, path) if path.is_file() else None
        if not path.is_file():
            missing.append(name)
            continue
        loaded = _try_load_json(path)
        if loaded.payload:
            payloads.append(loaded.payload)
            status = loaded.payload.get("v22_design_status") or loaded.payload.get("status")
            if isinstance(status, str):
                statuses.append(status)
    present = not missing
    return {
        "present": present,
        "root": _repo_relative(repo_root, design_root),
        "paths": paths,
        "missing": missing,
        "statuses": statuses,
        "uses_c_results_for_selection": any(payload.get("uses_c_results_for_selection") is True for payload in payloads),
        "tbd_count": _count_tbd(payloads) if present else None,
    }


def load_w6_calibration(repo_root: Path, run_id: str = RUN_ID) -> dict[str, Any]:
    path = _first_existing(
        repo_root,
        [
            f"agent/artifacts/{run_id}/openpi/v22_blind_calibration/desaturation_selection_decision.json",
            f"agent/artifacts/{run_id}/openpi/v22_blind_calibration/calibration_in_flight.json",
        ],
    )
    loaded = _try_load_json(path)
    payload = loaded.payload or {}
    status = str(payload.get("calibration_status") or payload.get("status") or "MISSING")
    selected_using_c = bool(payload.get("selected_using_c_results"))
    variant_codes = payload.get("variant_codes_used") or payload.get("variant_codes")
    variant_subset_ok = (
        isinstance(variant_codes, list)
        and all(code in {"A", "B"} for code in variant_codes)
    )
    return {
        "present": loaded.present,
        "path": _repo_relative(repo_root, loaded.path),
        "errors": loaded.errors,
        "calibration_status": status,
        "selected_using_c_results": selected_using_c,
        "variant_codes_used_subset_of_A_B_only": variant_subset_ok,
        "desaturated_protocol_selected": status == "DESATURATED_FOUND" and not selected_using_c,
    }


def load_w3_pattern(repo_root: Path, run_id: str = RUN_ID) -> dict[str, Any]:
    path = _first_existing(
        repo_root,
        [
            f"agent/artifacts/recap_min_loop/single_gpu_v2_full_update/{run_id}/gr00t/p5_failure_analysis/p5_negative_result_interpretation.json",
            f"agent/artifacts/recap_min_loop/single_gpu_v2_full_update/{run_id}/gr00t/p5_failure_analysis/gr00t_next_p5_decision.json",
            f"agent/artifacts/{run_id}/gr00t/p5_failure_analysis/p5_negative_result_interpretation.json",
        ],
    )
    loaded = _try_load_json(path)
    payload = loaded.payload or {}
    status = payload.get("p5_failure_pattern_status")
    return {
        "present": loaded.present,
        "path": _repo_relative(repo_root, loaded.path),
        "errors": loaded.errors,
        "p5_failure_pattern_status": status if isinstance(status, str) else "MISSING",
        "pattern_status_explicit": status in W3_PATTERN_STATUSES,
    }


def build_power_analysis(
    repo_root: Path,
    now_utc: datetime,
    iter3_authority: dict[str, Any],
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    stdev_iter3 = float(iter3_authority["stdev_iter3"])
    null_floor = 0.01
    null_half_stdev = 0.5 * stdev_iter3
    null_threshold = max(null_floor, null_half_stdev)
    detection_rows: list[dict[str, Any]] = []
    for effect_size in EFFECT_SIZE_GRID:
        per_count = []
        for episodes in CANDIDATE_EPISODE_COUNTS:
            seed_blocks = episodes / EPISODES_PER_SEED_BLOCK
            standard_error = stdev_iter3 / math.sqrt(seed_blocks)
            per_count.append(
                {
                    "episodes_per_variant": episodes,
                    "seed_blocks_equivalent": seed_blocks,
                    "standard_error_from_iter3_delta_stdev": standard_error,
                    "two_sided_power_approx": _two_sided_power(effect_size, standard_error),
                }
            )
        min_count = next(
            (row["episodes_per_variant"] for row in per_count if row["two_sided_power_approx"] >= 0.80),
            None,
        )
        detection_rows.append(
            {
                "effect_size": effect_size,
                "minimum_episodes_for_power_ge_0_80_within_grid": min_count,
                "candidate_counts": per_count,
            }
        )
    null_rows = []
    for episodes in CANDIDATE_EPISODE_COUNTS:
        seed_blocks = episodes / EPISODES_PER_SEED_BLOCK
        standard_error = stdev_iter3 / math.sqrt(seed_blocks)
        half_width = Z_975 * standard_error
        null_rows.append(
            {
                "episodes_per_variant": episodes,
                "seed_blocks_equivalent": seed_blocks,
                "ci95_half_width_approx": half_width,
                "can_accept_zero_effect_if_abs_delta_below_threshold": half_width <= null_threshold,
            }
        )
    null_minimum_within_grid = next(
        (
            row["episodes_per_variant"]
            for row in null_rows
            if row["can_accept_zero_effect_if_abs_delta_below_threshold"]
        ),
        None,
    )
    null_minimum_estimated = _round_up_to_seed_block((Z_975 * stdev_iter3 / null_threshold) ** 2 * EPISODES_PER_SEED_BLOCK)
    return {
        "schema_version": "iter4_v22_power_analysis_v1",
        "run_id": run_id,
        "generated_at_utc": _isoformat_utc(now_utc),
        "source_plan_path": PLAN_PATH,
        "iter3_authority_index_present": bool(iter3_authority["present"]),
        "iter3_authority_index_path": iter3_authority["path"],
        "variance_source": iter3_authority["variance_source"],
        "stdev_iter3": stdev_iter3,
        "stdev_components": {
            "stdev_c_minus_b": iter3_authority["stdev_c_minus_b"],
            "stdev_c_minus_x": iter3_authority["stdev_c_minus_x"],
        },
        "effect_size_grid": EFFECT_SIZE_GRID,
        "candidate_episode_counts": CANDIDATE_EPISODE_COUNTS,
        "confidence_method": "seed-block normal approximation from iter3 paired delta stdev; final formal run should retain paired/bootstrap CI",
        "alternative_arm": {
            "target": "detect 0.10 effect",
            "effect_size": 0.10,
            "recommended_minimum_episodes_per_variant": next(
                row["minimum_episodes_for_power_ge_0_80_within_grid"]
                for row in detection_rows
                if row["effect_size"] == 0.10
            ),
        },
        "null_arm": {
            "target": "accept zero-effect when absolute delta remains below threshold",
            "floor": null_floor,
            "threshold_from_half_stdev_iter3": null_half_stdev,
            "threshold_used": null_threshold,
            "abs_delta_threshold_for_null_acceptance": null_floor,
            "minimum_episodes_within_candidate_grid": null_minimum_within_grid,
            "recommended_minimum_episodes_per_variant": null_minimum_within_grid or null_minimum_estimated,
            "candidate_grid_satisfies_null_acceptance": null_minimum_within_grid is not None,
        },
        "detection_power_by_effect_size": detection_rows,
        "null_acceptance_by_episode_count": null_rows,
        "power_analysis_done": True,
        "blocking_reasons": [] if iter3_authority["present"] else ["iter3_authority_index_missing_used_plan_variance_fallback"],
    }


def _predicate_inputs(
    *,
    iter3_authority: dict[str, Any],
    r2: dict[str, Any],
    r4: dict[str, Any],
    shape: dict[str, Any],
    design: dict[str, Any],
    calibration: dict[str, Any],
    gate_policy: dict[str, Any],
    w3: dict[str, Any],
    power_analysis_done: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    v22_tbd_count = design["tbd_count"]
    inputs = {
        "iter3_authority_index_present": bool(iter3_authority["present"]),
        "r2_status_closed": r2["status"] == "CLOSED",
        "r4_status_closed": r4["status"] == "CLOSED",
        "w2_deterministic_replay_match": bool(shape["deterministic_replay_match"]),
        "alpha_effect_above_threshold": bool(shape["alpha_effect_above_threshold"]),
        "desaturated_protocol_selected": bool(calibration["desaturated_protocol_selected"]),
        "power_analysis_done": power_analysis_done,
        "v22_tbd_count_zero": v22_tbd_count == 0,
        "m12_rollback_triggered_false": not bool(shape["m12_rollback_triggered"]),
        "w2_gates_w5_assertion_in_w0_gate_policy": bool(gate_policy["w2_gates_w5"]),
        "p6_bash_matcher_tested": bool(gate_policy["p6_bash_matcher_tested"]),
        "w3_pattern_status_explicit": bool(w3["pattern_status_explicit"]),
    }
    details = {
        "iter3_authority_index": iter3_authority,
        "r2_closure": r2,
        "r4_closure": r4,
        "w2_shape_regression": shape,
        "w5_design": design,
        "w6_calibration": calibration,
        "w0_gate_policy": gate_policy,
        "w3_p5_failure_pattern": w3,
        "v22_tbd_count": v22_tbd_count,
    }
    failures = [
        reason
        for key, reason in [
            ("iter3_authority_index_present", "iter3_authority_index_missing"),
            ("r2_status_closed", "r2_not_closed"),
            ("r4_status_closed", "r4_not_closed"),
            ("w2_deterministic_replay_match", "w2_deterministic_replay_not_matched"),
            ("alpha_effect_above_threshold", "alpha_effect_below_or_missing"),
            ("desaturated_protocol_selected", "desaturated_protocol_not_selected"),
            ("power_analysis_done", "power_analysis_missing"),
            ("v22_tbd_count_zero", "v22_tbd_count_not_zero_or_unevaluable"),
            ("m12_rollback_triggered_false", "m12_rollback_triggered"),
            ("w2_gates_w5_assertion_in_w0_gate_policy", "w0_gate_policy_missing_w2_gates_w5"),
            ("p6_bash_matcher_tested", "p6_bash_matcher_not_tested"),
            ("w3_pattern_status_explicit", "w3_pattern_status_missing"),
        ]
        if not inputs[key]
    ]
    return inputs, details, failures


def _build_final_preregistration(
    *,
    now_utc: datetime,
    run_id: str,
    hash_lock_allowed: bool,
    power_path: str,
    decision_path: str,
    calibration: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "iter4_v22_preregistration_final_v1",
        "run_id": run_id,
        "generated_at_utc": _isoformat_utc(now_utc),
        "source_plan_path": PLAN_PATH,
        "status": "LOCKED" if hash_lock_allowed else "UNLOCKED",
        "hash_lock_allowed": hash_lock_allowed,
        "formal_v22_execution_allowed_next_iteration": hash_lock_allowed,
        "formal_v22_execution_allowed_iter4": False,
        "primary_metric_id": PRIMARY_METRIC_ID,
        "effect_size_grid": EFFECT_SIZE_GRID,
        "candidate_episode_counts": CANDIDATE_EPISODE_COUNTS,
        "calibration_status": calibration["calibration_status"],
        "desaturation_selection_decision_path": calibration["path"],
        "power_analysis_path": power_path,
        "hash_lock_decision_path": decision_path,
        "blocking_reasons": failures,
        "claim_guardrails": {
            "benchmark_success_claimed": False,
            "paper_equivalent_recap_claimed": False,
            "calibration_is_not_formal_result": True,
        },
    }


def materialize_iter4_v22_power_hash_decision(
    repo_root: Path,
    *,
    now_utc: datetime | None = None,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    now_utc = (now_utc or _utc_now()).astimezone(timezone.utc)
    power_root = _stage_root(repo_root, run_id) / "openpi" / "v22_power_analysis"
    prereg_root = _stage_root(repo_root, run_id) / "openpi" / "v22_preregistration"

    iter3_authority = load_iter3_authority_index(repo_root, run_id)
    gate_policy = load_gate_policy(repo_root, run_id)
    r2 = load_closure_verdict(repo_root, "r2", run_id)
    r4 = load_closure_verdict(repo_root, "r4", run_id)
    shape = load_shape_regression(repo_root, run_id)
    design = load_w5_design(repo_root, run_id)
    calibration = load_w6_calibration(repo_root, run_id)
    w3 = load_w3_pattern(repo_root, run_id)

    power_analysis = build_power_analysis(repo_root, now_utc, iter3_authority, run_id)
    power_path = power_root / "power_analysis.json"
    _write_json(power_path, power_analysis)

    predicate_inputs, predicate_details, failures = _predicate_inputs(
        iter3_authority=iter3_authority,
        r2=r2,
        r4=r4,
        shape=shape,
        design=design,
        calibration=calibration,
        gate_policy=gate_policy,
        w3=w3,
        power_analysis_done=bool(power_analysis["power_analysis_done"]),
    )
    hash_lock_allowed = not failures
    reason = "predicate_satisfied" if hash_lock_allowed else failures
    decision_path = prereg_root / "v22_hash_lock_decision_iter4.json"
    final_path = prereg_root / "v22_preregistration_final.json"

    final_payload = _build_final_preregistration(
        now_utc=now_utc,
        run_id=run_id,
        hash_lock_allowed=hash_lock_allowed,
        power_path=_repo_relative(repo_root, power_path) or str(power_path),
        decision_path=_repo_relative(repo_root, decision_path) or str(decision_path),
        calibration=calibration,
        failures=failures,
    )
    final_tbd_count = _count_tbd(final_payload)
    if final_tbd_count:
        predicate_inputs["v22_tbd_count_zero"] = False
        predicate_details["v22_tbd_count"] = final_tbd_count
        if "v22_tbd_count_not_zero_or_unevaluable" not in failures:
            failures.append("v22_tbd_count_not_zero_or_unevaluable")
        hash_lock_allowed = False
        reason = failures
        final_payload["status"] = "UNLOCKED"
        final_payload["hash_lock_allowed"] = False
        final_payload["formal_v22_execution_allowed_next_iteration"] = False
        final_payload["blocking_reasons"] = failures
    _write_json(final_path, final_payload)

    decision_payload = {
        "schema_version": "iter4_v22_hash_lock_decision_v1",
        "run_id": run_id,
        "generated_at_utc": _isoformat_utc(now_utc),
        "evaluated_by": "worker-8/W7",
        "source_plan_path": PLAN_PATH,
        "hash_lock_allowed": hash_lock_allowed,
        "reason": reason,
        "predicate_inputs": predicate_inputs,
        "predicate_input_details": predicate_details,
        "predicate_failures": failures,
        "input_artifacts_consumed": {
            "power_analysis": _repo_relative(repo_root, power_path),
            "v22_preregistration_final": _repo_relative(repo_root, final_path),
        },
    }
    _write_json(decision_path, decision_payload)

    hash_path = prereg_root / "v22_preregistration_hash_lock.json"
    if hash_lock_allowed:
        _write_json(
            hash_path,
            {
                "schema_version": "iter4_v22_preregistration_hash_lock_v1",
                "run_id": run_id,
                "created_at_utc": _isoformat_utc(now_utc),
                "locked_artifact_path": _repo_relative(repo_root, final_path),
                "locked_artifact_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
                "decision_artifact_path": _repo_relative(repo_root, decision_path),
                "decision_artifact_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
            },
        )
    elif hash_path.exists():
        hash_path.unlink()

    worker_report = {
        "schema_version": "iter4_w7_worker_report_v1",
        "run_id": run_id,
        "generated_at_utc": _isoformat_utc(now_utc),
        "role": "W7 power analysis + hash-lock decision",
        "outputs": {
            "power_analysis": _repo_relative(repo_root, power_path),
            "v22_preregistration_final": _repo_relative(repo_root, final_path),
            "v22_hash_lock_decision_iter4": _repo_relative(repo_root, decision_path),
            "v22_preregistration_hash_lock": _repo_relative(repo_root, hash_path) if hash_path.exists() else None,
        },
        "acceptance": {
            "effect_size_grid_entries": len(power_analysis["effect_size_grid"]),
            "null_arm_threshold_used": power_analysis["null_arm"]["threshold_used"],
            "predicate_inputs_recorded_count": len(predicate_inputs),
            "all_12_predicate_inputs_recorded": len(predicate_inputs) == 12,
            "decision_emitted": decision_path.is_file(),
            "hash_lock_created_iff_allowed": hash_lock_allowed == hash_path.is_file(),
        },
    }
    report_path = prereg_root / "worker_report_w7.json"
    _write_json(report_path, worker_report)

    return {
        "power_analysis": power_analysis,
        "decision": decision_payload,
        "worker_report": worker_report,
        "output_root": _repo_relative(repo_root, prereg_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=RUN_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_iter4_v22_power_hash_decision(args.repo_root, run_id=args.run_id)
    print(json.dumps(result["worker_report"]["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
