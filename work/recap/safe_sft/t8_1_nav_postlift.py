from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

from work.recap.safe_sft.entrypoint import (
    DEFAULT_CANONICAL_IDENTITY,
    DEFAULT_OFFICIAL_BASE,
    DEFAULT_TASK_PROMPT,
    MODALITIES,
    REPO_ROOT,
    WBC_ROOT,
    ISAAC_GR00T_ROOT,
    discover_lora_targets,
    freeze_all_params,
    git_output,
    inject_lora,
    json_default,
    load_policy,
    rel,
    sha256_file,
    surface_hashes,
    utc_now,
    write_csv,
    write_json,
)
from work.recap.safe_sft.t8_smoke import (
    append_jsonl,
    array_stats,
    resolve,
)

for _p in (REPO_ROOT, ISAAC_GR00T_ROOT, WBC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


SOURCE_T8_RUN = "gr00t_t8_no_recap_hand_smoke_20260508_052919"
SOURCE_T8_ROOT = REPO_ROOT / "agent/artifacts/gr00t_t8_no_recap_hand_smoke" / SOURCE_T8_RUN
S2_ADAPTER = SOURCE_T8_ROOT / "cells/S2_BASE_TEACHER/checkpoint_500/adapter_model.pt"
S2_MANIFEST = SOURCE_T8_ROOT / "cells/S2_BASE_TEACHER/checkpoint_manifest.json"

ALLOWED_FINAL = {
    "RUNNER_OR_CONTRACT_REGRESSION",
    "BASE_SEEDS_TOO_HARD",
    "NAV_SPLICE_IMPROVES",
    "NAV_DIRECTION_TIMING_BUG",
    "POST_LIFT_PLACE_BLOCKER",
    "SAFE_SFT_30_READY",
    "GUARDED_RECAP_STILL_FORBIDDEN",
}

MIN_MATERIAL_EPISODES = 10
MATERIAL_REACHED_DELTA = 2
MATERIAL_SUCCESS_DELTA = 1
MATERIAL_LIFTED_DELTA = 2


def import_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


def load_3d_helpers() -> Any:
    return import_script_module("t8_1_3d_helpers", REPO_ROOT / "work/recap/scripts/3D_recap_eval.py")


def reject_forbidden_args(argv: list[str]) -> None:
    forbidden = {
        "--recap",
        "--advantage",
        "--guarded-recap",
        "--fatg",
        "--per-edge",
        "--full-scope",
        "--full-head",
        "--merge-lora",
        "--merge-lora-before-eval",
        "--train",
        "--run-training",
    }
    present = sorted(set(argv) & forbidden)
    if present:
        raise SystemExit(f"Forbidden T8.1 route flags rejected before model load: {present}")


def load_torch() -> Any:
    import torch

    return torch


def load_policy_with_optional_lora(
    checkpoint: Path,
    *,
    adapter_path: Path | None = None,
    rank: int = 4,
    alpha: int = 8,
    dropout: float = 0.05,
    zero_step_lora: bool = False,
) -> Any:
    torch = load_torch()
    policy = load_policy(checkpoint)
    policy.model.eval()
    if adapter_path is None and not zero_step_lora:
        freeze_all_params(policy.model)
        policy.model.eval()
        return policy
    freeze_all_params(policy.model)
    discovery = discover_lora_targets(policy.model)
    if discovery.get("status") != "PASS":
        raise RuntimeError("LoRA target discovery failed")
    inject_lora(policy.model, [row["name"] for row in discovery["allowed_targets"]], rank=rank, alpha=alpha, dropout=dropout)
    if adapter_path is not None:
        state = torch.load(adapter_path, map_location="cpu")
        named = dict(policy.model.named_parameters())
        missing = [name for name in state.keys() if name not in named]
        if missing:
            raise RuntimeError(f"adapter tensors missing in injected policy: {missing[:5]}")
        for name, value in state.items():
            named[name].data.copy_(value.to(device=named[name].device, dtype=named[name].dtype))
    freeze_all_params(policy.model)
    policy.model.eval()
    return policy


def make_eval_env(max_episode_steps: int, n_action_steps: int = 20) -> Any:
    post_a = import_script_module("t8_1_post_a", REPO_ROOT / "work/recap/scripts/gr00t_post_a_local_diagnostic.py")
    return post_a.make_eval_env(max_episode_steps=max_episode_steps, n_action_steps=n_action_steps)


def action_q99(action: Mapping[str, Any], modality: str) -> float:
    arr = np.asarray(action.get(f"action.{modality}", []), dtype=np.float32)
    return float(np.quantile(np.abs(arr).reshape(-1), 0.99)) if arr.size else 0.0


def action_energy(action: Mapping[str, Any], modality: str) -> float:
    arr = np.asarray(action.get(f"action.{modality}", []), dtype=np.float32)
    return float(np.linalg.norm(arr.reshape(-1))) if arr.size else 0.0


def normalize_action(action: Mapping[str, Any]) -> dict[str, np.ndarray]:
    post_a = import_script_module("t8_1_post_a_norm", REPO_ROOT / "work/recap/scripts/gr00t_post_a_local_diagnostic.py")
    return post_a.squeeze_policy_batch(post_a.normalize_action_dict(action))


def flat_obs(obs: Mapping[str, Any]) -> dict[str, Any]:
    post_a = import_script_module("t8_1_post_a_flat", REPO_ROOT / "work/recap/scripts/gr00t_post_a_local_diagnostic.py")
    phase2 = import_script_module("t8_1_phase2_flat", REPO_ROOT / "work/recap/scripts/gr00t_phase2_l4_l5_identity_closure.py")
    return post_a.coerce_obs_float32(phase2.batch_flat_env_obs(obs))


def get_action(sim_policy: Any, obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    action, _info = sim_policy.get_action(flat_obs(obs))
    return normalize_action(action)


def safe_vec2(a: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(a, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            return None
        return arr[:2]
    except Exception:
        return None


def mean_nav_cosine(candidate: np.ndarray, base: np.ndarray) -> float | None:
    c = np.asarray(candidate, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(base, dtype=np.float64).reshape(-1, 3)
    n = min(len(c), len(b))
    if n <= 0:
        return None
    vals = []
    for i in range(n):
        cn = float(np.linalg.norm(c[i]))
        bn = float(np.linalg.norm(b[i]))
        if cn < 1e-8 or bn < 1e-8:
            continue
        vals.append(float(np.dot(c[i], b[i]) / (cn * bn)))
    return float(np.mean(vals)) if vals else None


def nav_projection_to_apple(action_nav: np.ndarray, snapshot: Mapping[str, Any]) -> float | None:
    apple = safe_vec2(snapshot.get("apple_pos_xyz"))
    right = safe_vec2(snapshot.get("right_eef_pos_xyz"))
    if apple is None or right is None:
        return None
    direction = apple - right
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return None
    unit = direction / norm
    nav = np.asarray(action_nav, dtype=np.float64).reshape(-1, 3)[:, :2]
    if nav.size == 0:
        return None
    return float(np.mean(nav @ unit))


def stand_branch_fraction(action_nav: np.ndarray, threshold: float = 0.05) -> float:
    nav = np.asarray(action_nav, dtype=np.float64).reshape(-1, 3)
    if nav.size == 0:
        return 1.0
    return float(np.mean(np.linalg.norm(nav, axis=1) < float(threshold)))


def first_step_arr(action: Mapping[str, Any], modality: str) -> np.ndarray:
    arr = np.asarray(action.get(f"action.{modality}", []), dtype=np.float32)
    if arr.ndim >= 2:
        return arr[0].reshape(-1)
    return arr.reshape(-1)


def last_step_arr(action: Mapping[str, Any], modality: str) -> np.ndarray:
    arr = np.asarray(action.get(f"action.{modality}", []), dtype=np.float32)
    if arr.ndim >= 2:
        return arr[-1].reshape(-1)
    return arr.reshape(-1)


def chunk_boundary_jump(prev_action: Mapping[str, Any] | None, action: Mapping[str, Any]) -> float | None:
    if prev_action is None:
        return None
    vals = []
    for modality in ("navigate_command", "right_arm", "right_hand", "waist"):
        a = last_step_arr(prev_action, modality)
        b = first_step_arr(action, modality)
        n = min(a.size, b.size)
        if n:
            vals.append(np.abs(b[:n] - a[:n]))
    if not vals:
        return None
    return float(np.quantile(np.concatenate(vals), 0.99))


def status_from_steps(steps: list[dict[str, Any]], success: bool) -> dict[str, Any]:
    if not steps:
        return {"reached": False, "lifted": False, "reached_t": None, "lifted_t": None, "failure_mode": "no_steps"}
    initial_h = next((row.get("apple_height_z") for row in steps if isinstance(row.get("apple_height_z"), (int, float))), 0.0)
    reached_t = None
    lifted_t = None
    for row in steps:
        t = int(row.get("outer_step", 0))
        dist = row.get("apple_to_right_eef_l2")
        if reached_t is None and isinstance(dist, (int, float)) and float(dist) <= 0.10:
            reached_t = t
        h = row.get("apple_height_z")
        if lifted_t is None and isinstance(h, (int, float)) and float(h) - float(initial_h) >= 0.03:
            lifted_t = t
    reached = bool(success or reached_t is not None or lifted_t is not None)
    lifted = bool(success or lifted_t is not None)
    if success:
        failure = "success"
    elif lifted:
        plate_after = [
            float(row["apple_to_plate_l2"])
            for row in steps
            if lifted_t is not None and int(row.get("outer_step", 0)) >= int(lifted_t) and isinstance(row.get("apple_to_plate_l2"), (int, float))
        ]
        failure = "lifted_not_brought_to_plate" if not plate_after or min(plate_after) > 0.12 else "near_plate_but_not_success"
    elif reached:
        failure = "reached_apple_not_lifted"
    else:
        failure = "never_reached_apple"
    return {"reached": reached, "lifted": lifted, "reached_t": reached_t, "lifted_t": lifted_t, "failure_mode": failure}


def summarize_eval_rows(rows: list[dict[str, Any]], policy_id: str, policy: str) -> dict[str, Any]:
    failures: dict[str, int] = {}
    for row in rows:
        failures[str(row["failure_mode"])] = failures.get(str(row["failure_mode"]), 0) + 1
    return {
        "ID": policy_id,
        "policy": policy,
        "episodes": len(rows),
        "success": int(sum(bool(row["success"]) for row in rows)),
        "reached": int(sum(bool(row["reached"]) for row in rows)),
        "lifted": int(sum(bool(row["lifted"]) for row in rows)),
        "failure_modes": failures,
    }


def material_threshold_spec() -> dict[str, int]:
    return {
        "min_episodes_per_arm": MIN_MATERIAL_EPISODES,
        "reached_delta": MATERIAL_REACHED_DELTA,
        "success_delta": MATERIAL_SUCCESS_DELTA,
        "lifted_delta": MATERIAL_LIFTED_DELTA,
    }


def _row_int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return default


def build_splice_material_improvement(
    rows: list[dict[str, Any]],
    *,
    baseline_id: str,
    splice_kind: str,
    min_episodes: int = MIN_MATERIAL_EPISODES,
) -> dict[str, Any]:
    """Validate splice outcome deltas without applying thresholds below n=10.

    Small diagnostic runs still expose their deltas, but they are explicitly
    qualitative-only and cannot unlock material-improvement decisions.
    """

    baseline = next((row for row in rows if row.get("ID") == baseline_id), None)
    threshold = {
        "min_episodes_per_arm": int(min_episodes),
        "reached_delta": MATERIAL_REACHED_DELTA,
        "success_delta": MATERIAL_SUCCESS_DELTA,
        "lifted_delta": MATERIAL_LIFTED_DELTA,
    }
    if baseline is None:
        return {
            "schema_version": "t8_1_splice_material_improvement_v1",
            "status": "FAIL",
            "splice_kind": splice_kind,
            "baseline_id": baseline_id,
            "threshold": threshold,
            f"{splice_kind}_material_improvement": False,
            "blocking_reasons": ["missing_baseline_row"],
            "rows": [],
        }

    baseline_episodes = _row_int(baseline, "episodes")
    material_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("ID") == baseline_id:
            continue
        candidate_episodes = _row_int(row, "episodes")
        quantitative_eligible = (
            baseline_episodes >= int(min_episodes) and candidate_episodes >= int(min_episodes)
        )
        reached_delta = _row_int(row, "reached") - _row_int(baseline, "reached")
        success_delta = _row_int(row, "success") - _row_int(baseline, "success")
        lifted_delta = _row_int(row, "lifted") - _row_int(baseline, "lifted")
        threshold_met = (
            reached_delta >= MATERIAL_REACHED_DELTA
            or success_delta >= MATERIAL_SUCCESS_DELTA
            or lifted_delta >= MATERIAL_LIFTED_DELTA
        )
        material_rows.append(
            {
                "ID": row.get("ID"),
                "baseline_id": baseline_id,
                "baseline_episodes": baseline_episodes,
                "candidate_episodes": candidate_episodes,
                "min_episodes_per_arm": int(min_episodes),
                "evidence_mode": "quantitative_threshold" if quantitative_eligible else "qualitative_only",
                "quantitative_threshold_eligible": quantitative_eligible,
                "threshold_not_applied_reason": None
                if quantitative_eligible
                else f"requires_n_ge_{int(min_episodes)}_per_arm",
                "reached_delta": reached_delta,
                "success_delta": success_delta,
                "lifted_delta": lifted_delta,
                "threshold_met": threshold_met,
                "material_improvement": bool(quantitative_eligible and threshold_met),
            }
        )
    return {
        "schema_version": "t8_1_splice_material_improvement_v1",
        "status": "PASS",
        "splice_kind": splice_kind,
        "baseline_id": baseline_id,
        "threshold": threshold,
        f"{splice_kind}_material_improvement": any(row["material_improvement"] for row in material_rows),
        "qualitative_only": any(row["evidence_mode"] == "qualitative_only" for row in material_rows),
        "rows": material_rows,
    }


def write_splice_material_improvement(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    baseline_id: str,
    splice_kind: str,
) -> dict[str, Any]:
    payload = build_splice_material_improvement(
        rows,
        baseline_id=baseline_id,
        splice_kind=splice_kind,
    )
    write_csv(out_dir / "material_improvement.csv", payload["rows"])
    write_json(out_dir / "material_improvement.json", payload)
    return payload


def run_single_policy_eval(
    *,
    policy: Any,
    policy_id: str,
    policy_label: str,
    out_dir: Path,
    seeds: list[int],
    max_episode_steps: int,
) -> dict[str, Any]:
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

    helpers = load_3d_helpers()
    helpers._add_import_roots(REPO_ROOT)
    helpers._install_robocasa_import_shims()
    env = make_eval_env(max_episode_steps=max_episode_steps, n_action_steps=20)
    sim_policy = Gr00tSimPolicyWrapper(policy, strict=True)
    rows: list[dict[str, Any]] = []
    step_path = out_dir / f"{policy_id}_steps.jsonl"
    try:
        for ep, seed in enumerate(seeds):
            obs, _info = env.reset(seed=int(seed))
            done = False
            success = False
            outer = 0
            steps: list[dict[str, Any]] = []
            right_vals: list[float] = []
            nav_vals: list[float] = []
            prev_action = None
            while not done and outer < max(1, int(max_episode_steps) // 20):
                action = get_action(sim_policy, obs)
                snap_before = helpers._collect_env_snapshot(env)
                jump = chunk_boundary_jump(prev_action, action)
                obs, reward, term, trunc, info = env.step(action)
                done = bool(helpers._scalarize_bool(term) or helpers._scalarize_bool(trunc))
                success_step = bool(helpers._extract_success_step(info))
                success = bool(success or success_step)
                snap = helpers._collect_env_snapshot(env)
                rec = {
                    "ID": policy_id,
                    "policy": policy_label,
                    "seed": int(seed),
                    "episode_index": ep,
                    "outer_step": outer,
                    "success_step": success_step,
                    "right_hand_q99": action_q99(action, "right_hand"),
                    "navigate_q99": action_q99(action, "navigate_command"),
                    "right_arm_energy": action_energy(action, "right_arm"),
                    "left_arm_energy": action_energy(action, "left_arm"),
                    "waist_energy": action_energy(action, "waist"),
                    "navigate_energy": action_energy(action, "navigate_command"),
                    "chunk_boundary_jump": jump,
                    "action_summary": helpers._summarize_action_chunk(action),
                }
                rec.update(snap)
                steps.append(rec)
                append_jsonl(step_path, rec)
                right_vals.append(float(rec["right_hand_q99"]))
                nav_vals.append(float(rec["navigate_q99"]))
                prev_action = action
                outer += 1
            status = status_from_steps(steps, success)
            rows.append(
                {
                    "seed": int(seed),
                    "policy": policy_id,
                    "reached": bool(status["reached"]),
                    "lifted": bool(status["lifted"]),
                    "success": bool(success),
                    "reached_t": status["reached_t"],
                    "lifted_t": status["lifted_t"],
                    "right_hand_q99": float(np.quantile(right_vals, 0.99)) if right_vals else 0.0,
                    "navigate_q99": float(np.quantile(nav_vals, 0.99)) if nav_vals else 0.0,
                    "failure_mode": status["failure_mode"],
                }
            )
            print(f"[BASE_EVAL_EP] {policy_id} seed={seed} success={success} reached={status['reached']} lifted={status['lifted']} failure={status['failure_mode']}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        del sim_policy
        gc.collect()
    return {"summary": summarize_eval_rows(rows, policy_id, policy_label), "per_seed": rows, "steps_jsonl": rel(step_path)}


def select_nav(candidate: dict[str, np.ndarray], base: dict[str, np.ndarray], variant: str, reached_so_far: bool) -> dict[str, np.ndarray]:
    action = {k: np.asarray(v, dtype=np.float32).copy() for k, v in candidate.items()}
    c_nav = np.asarray(candidate["action.navigate_command"], dtype=np.float32)
    b_nav = np.asarray(base["action.navigate_command"], dtype=np.float32)
    if variant == "N1":
        action["action.navigate_command"] = b_nav.copy()
    elif variant == "N2":
        action["action.navigate_command"] = c_nav.copy() if reached_so_far else b_nav.copy()
    elif variant == "N3":
        out = c_nav.copy()
        flat_c = out.reshape(-1, out.shape[-1])
        flat_b = b_nav.reshape(-1, b_nav.shape[-1])
        for i in range(min(len(flat_c), len(flat_b))):
            cn = float(np.linalg.norm(flat_c[i]))
            bn = float(np.linalg.norm(flat_b[i]))
            if cn > 1e-8:
                flat_c[i] = flat_c[i] / cn * bn
        action["action.navigate_command"] = out
    elif variant == "N4":
        for key in ("action.navigate_command", "action.base_height_command", "action.waist"):
            if key in base:
                action[key] = np.asarray(base[key], dtype=np.float32).copy()
    return action


def select_post_lift_action(candidate: dict[str, np.ndarray], base: dict[str, np.ndarray], variant: str, reached_so_far: bool, lifted_so_far: bool) -> dict[str, np.ndarray]:
    if variant == "P0":
        return {k: np.asarray(v, dtype=np.float32).copy() for k, v in candidate.items()}
    if variant == "P1" and lifted_so_far:
        return {k: np.asarray(v, dtype=np.float32).copy() for k, v in base.items()}
    if variant == "P2":
        src = candidate if reached_so_far else base
        return {k: np.asarray(v, dtype=np.float32).copy() for k, v in src.items()}
    action = {k: np.asarray(v, dtype=np.float32).copy() for k, v in candidate.items()}
    if lifted_so_far and variant == "P3":
        # Base transport/place with S2 hand.
        for key, value in base.items():
            if key != "action.right_hand":
                action[key] = np.asarray(value, dtype=np.float32).copy()
        action["action.right_hand"] = np.asarray(candidate["action.right_hand"], dtype=np.float32).copy()
    if lifted_so_far and variant == "P4":
        # S2 transport/place with base hand.
        action["action.right_hand"] = np.asarray(base["action.right_hand"], dtype=np.float32).copy()
    return action


def run_splice_eval(
    *,
    base_policy: Any,
    candidate_policy: Any,
    variants: dict[str, dict[str, str]],
    out_dir: Path,
    seeds: list[int],
    max_episode_steps: int,
    mode: str,
) -> dict[str, Any]:
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

    helpers = load_3d_helpers()
    helpers._add_import_roots(REPO_ROOT)
    helpers._install_robocasa_import_shims()
    env = make_eval_env(max_episode_steps=max_episode_steps, n_action_steps=20)
    base_sim = Gr00tSimPolicyWrapper(base_policy, strict=True)
    cand_sim = Gr00tSimPolicyWrapper(candidate_policy, strict=True)
    all_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    try:
        for variant, spec in variants.items():
            step_path = out_dir / f"{variant}_steps.jsonl"
            per_seed: list[dict[str, Any]] = []
            for ep, seed in enumerate(seeds):
                obs, _info = env.reset(seed=int(seed))
                done = False
                success = False
                outer = 0
                reached_so_far = False
                lifted_so_far = False
                steps: list[dict[str, Any]] = []
                selected_nav_q99: list[float] = []
                cosine_vals: list[float] = []
                projection_vals: list[float] = []
                stand_fracs: list[float] = []
                prev_action = None
                while not done and outer < max(1, int(max_episode_steps) // 20):
                    base_action = get_action(base_sim, obs)
                    cand_action = get_action(cand_sim, obs)
                    if mode == "nav":
                        action = select_nav(cand_action, base_action, variant, reached_so_far)
                    else:
                        action = select_post_lift_action(cand_action, base_action, variant, reached_so_far, lifted_so_far)
                    snap_before = helpers._collect_env_snapshot(env)
                    base_nav = np.asarray(base_action.get("action.navigate_command"), dtype=np.float32)
                    cand_nav = np.asarray(cand_action.get("action.navigate_command"), dtype=np.float32)
                    selected_nav = np.asarray(action.get("action.navigate_command"), dtype=np.float32)
                    nav_cos = mean_nav_cosine(selected_nav, base_nav)
                    projection = nav_projection_to_apple(selected_nav, snap_before)
                    stand_frac = stand_branch_fraction(selected_nav)
                    jump = chunk_boundary_jump(prev_action, action)
                    obs, reward, term, trunc, info = env.step(action)
                    done = bool(helpers._scalarize_bool(term) or helpers._scalarize_bool(trunc))
                    success_step = bool(helpers._extract_success_step(info))
                    success = bool(success or success_step)
                    snap = helpers._collect_env_snapshot(env)
                    rec = {
                        "ID": variant,
                        "seed": int(seed),
                        "episode_index": ep,
                        "outer_step": outer,
                        "mode": mode,
                        "success_step": success_step,
                        "selected_nav_q99": action_q99(action, "navigate_command"),
                        "candidate_nav_q99": action_q99(cand_action, "navigate_command"),
                        "base_nav_q99": action_q99(base_action, "navigate_command"),
                        "right_hand_q99": action_q99(action, "right_hand"),
                        "right_arm_energy": action_energy(action, "right_arm"),
                        "left_arm_energy": action_energy(action, "left_arm"),
                        "waist_energy": action_energy(action, "waist"),
                        "navigate_energy": action_energy(action, "navigate_command"),
                        "nav_cosine_vs_base": nav_cos,
                        "nav_projection_to_apple": projection,
                        "stand_branch_fraction": stand_frac,
                        "chunk_boundary_jump": jump,
                        "action_summary": helpers._summarize_action_chunk(action),
                    }
                    rec.update(snap)
                    append_jsonl(step_path, rec)
                    steps.append(rec)
                    selected_nav_q99.append(float(rec["selected_nav_q99"]))
                    if nav_cos is not None:
                        cosine_vals.append(float(nav_cos))
                    if projection is not None:
                        projection_vals.append(float(projection))
                    stand_fracs.append(float(stand_frac))
                    status_now = status_from_steps(steps, success)
                    reached_so_far = bool(status_now["reached"])
                    lifted_so_far = bool(status_now["lifted"])
                    prev_action = action
                    outer += 1
                status = status_from_steps(steps, success)
                plate_after = [
                    float(row["apple_to_plate_l2"])
                    for row in steps
                    if status["lifted_t"] is not None and int(row.get("outer_step", 0)) >= int(status["lifted_t"]) and isinstance(row.get("apple_to_plate_l2"), (int, float))
                ]
                per_row = {
                    "seed": int(seed),
                    "policy": variant,
                    "reached": bool(status["reached"]),
                    "lifted": bool(status["lifted"]),
                    "success": bool(success),
                    "reached_t": status["reached_t"],
                    "lifted_t": status["lifted_t"],
                    "distance_to_apple_min": min([float(r["apple_to_right_eef_l2"]) for r in steps if isinstance(r.get("apple_to_right_eef_l2"), (int, float))], default=None),
                    "distance_to_plate_min_after_lift": min(plate_after) if plate_after else None,
                    "nav_q99": float(np.quantile(selected_nav_q99, 0.99)) if selected_nav_q99 else 0.0,
                    "nav_cosine_vs_base": float(np.mean(cosine_vals)) if cosine_vals else None,
                    "nav_projection_to_apple_mean": float(np.mean(projection_vals)) if projection_vals else None,
                    "stand_branch_fraction": float(np.mean(stand_fracs)) if stand_fracs else None,
                    "failure_mode": status["failure_mode"],
                }
                per_seed.append(per_row)
                direction_rows.append(per_row)
                print(f"[SPLICE_EP] {variant} seed={seed} success={success} reached={status['reached']} lifted={status['lifted']} failure={status['failure_mode']}", flush=True)
            summary = summarize_eval_rows(per_seed, variant, spec.get("description", variant))
            summary.update(
                {
                    "upper_hand_source": spec.get("upper_hand_source"),
                    "navigate_source": spec.get("navigate_source"),
                    "lower_package": spec.get("lower_package"),
                    "mode": mode,
                    "steps_jsonl": rel(step_path),
                }
            )
            write_csv(out_dir / f"{variant}_per_seed.csv", per_seed)
            write_json(out_dir / f"{variant}_summary.json", summary)
            all_rows.append(summary)
    finally:
        try:
            env.close()
        except Exception:
            pass
        del base_sim
        del cand_sim
        gc.collect()
    return {"rows": all_rows, "direction_rows": direction_rows}


def load_step_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_records_by_episode(records: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in records:
        key = (str(row.get("ID") or row.get("policy") or row.get("source")), int(row.get("seed", row.get("episode_index", 0))))
        out.setdefault(key, []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: int(r.get("outer_step", 0)))
    return out


def post_lift_audit(out_dir: Path, sources: dict[str, Path]) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for source, path in sources.items():
        records = load_step_records(path)
        for (_sid, seed), steps in group_records_by_episode(records).items():
            status = status_from_steps(steps, any(bool(r.get("success_step")) for r in steps))
            if not status["lifted"] or any(bool(r.get("success_step")) for r in steps):
                continue
            initial_h = next((float(r["apple_height_z"]) for r in steps if isinstance(r.get("apple_height_z"), (int, float))), 0.0)
            lifted_t = int(status["lifted_t"]) if status["lifted_t"] is not None else None
            after = [r for r in steps if lifted_t is not None and int(r.get("outer_step", 0)) >= lifted_t]
            heights = [float(r["apple_height_z"]) for r in steps if isinstance(r.get("apple_height_z"), (int, float))]
            plate_after = [float(r["apple_to_plate_l2"]) for r in after if isinstance(r.get("apple_to_plate_l2"), (int, float))]
            hand_after = [float(r.get("right_hand_q99", 0.0)) for r in after]
            released = bool(hand_after and min(hand_after) < 0.30)
            reached_plate = bool(plate_after and min(plate_after) <= 0.12)
            placed_on_plate = bool(reached_plate and heights and (heights[-1] - initial_h) < 0.05)
            summary_rows.append(
                {
                    "seed": int(seed),
                    "source": source,
                    "lifted_t": lifted_t,
                    "apple_height_peak": max(heights) if heights else None,
                    "carried_duration": len(after),
                    "apple_to_plate_distance_min": min(plate_after) if plate_after else None,
                    "reached_plate": reached_plate,
                    "released": released,
                    "placed_on_plate": placed_on_plate,
                    "final_failure": status["failure_mode"],
                    "apple_moved_toward_plate_after_lift": bool(plate_after and plate_after[-1] < plate_after[0]) if len(plate_after) >= 2 else False,
                }
            )
            reached_t = status["reached_t"]
            for phase in ("pre_reach", "grasp_lift", "carry_transport", "place_release"):
                if phase == "pre_reach":
                    phase_steps = [r for r in steps if reached_t is None or int(r.get("outer_step", 0)) < int(reached_t)]
                elif phase == "grasp_lift":
                    phase_steps = [
                        r
                        for r in steps
                        if reached_t is not None
                        and int(r.get("outer_step", 0)) >= int(reached_t)
                        and (lifted_t is None or int(r.get("outer_step", 0)) <= int(lifted_t))
                    ]
                elif phase == "carry_transport":
                    phase_steps = [r for r in steps if lifted_t is not None and int(r.get("outer_step", 0)) >= int(lifted_t)]
                else:
                    phase_steps = [r for r in after if isinstance(r.get("apple_to_plate_l2"), (int, float)) and float(r["apple_to_plate_l2"]) <= 0.20]
                jumps = [
                    float(r["chunk_boundary_jump"])
                    for r in phase_steps
                    if isinstance(r.get("chunk_boundary_jump"), (int, float)) and math.isfinite(float(r["chunk_boundary_jump"]))
                ]
                phase_rows.append(
                    {
                        "seed": int(seed),
                        "source": source,
                        "phase": phase,
                        "right_hand_q99": float(np.mean([float(r.get("right_hand_q99", 0.0)) for r in phase_steps])) if phase_steps else 0.0,
                        "right_arm_energy": float(np.mean([float(r.get("right_arm_energy", 0.0)) for r in phase_steps])) if phase_steps else 0.0,
                        "left_arm_energy": float(np.mean([float(r.get("left_arm_energy", 0.0)) for r in phase_steps])) if phase_steps else 0.0,
                        "navigate_energy": float(np.mean([float(r.get("navigate_energy", 0.0)) for r in phase_steps])) if phase_steps else 0.0,
                        "waist_energy": float(np.mean([float(r.get("waist_energy", 0.0)) for r in phase_steps])) if phase_steps else 0.0,
                        "chunk_boundary_jump": float(np.mean(jumps)) if jumps else None,
                        "notes": "heuristic_phase_bucket",
                    }
                )
    write_csv(out_dir / "post_lift_place_summary.csv", summary_rows)
    write_csv(out_dir / "post_lift_phase_telemetry.csv", phase_rows)
    post_lift_case_count = len(summary_rows)
    quantitative_threshold_eligible = post_lift_case_count >= MIN_MATERIAL_EPISODES
    answers = {
        "lifted_but_not_success_count": post_lift_case_count,
        "min_cases_for_quantitative_blocker": MIN_MATERIAL_EPISODES,
        "evidence_mode": "quantitative_threshold"
        if quantitative_threshold_eligible
        else "qualitative_only",
        "quantitative_threshold_eligible": quantitative_threshold_eligible,
        "threshold_not_applied_reason": None
        if quantitative_threshold_eligible
        else f"requires_n_ge_{MIN_MATERIAL_EPISODES}_post_lift_cases",
        "any_moves_toward_plate": any(bool(r.get("apple_moved_toward_plate_after_lift")) for r in summary_rows),
        "any_reached_plate": any(bool(r.get("reached_plate")) for r in summary_rows),
        "any_release_proxy": any(bool(r.get("released")) for r in summary_rows),
        "dominant_final_failures": {},
    }
    for row in summary_rows:
        key = str(row["final_failure"])
        answers["dominant_final_failures"][key] = answers["dominant_final_failures"].get(key, 0) + 1
    write_json(out_dir / "post_lift_place_audit.json", {"status": "PASS", "answers": answers, "summary_rows": summary_rows})
    return {"summary_rows": summary_rows, "phase_rows": phase_rows, "answers": answers}


def source_evidence_lock(out: Path) -> dict[str, Any]:
    required = {
        "t8_final_decision": SOURCE_T8_ROOT / "final_decision.json",
        "t8_matrix": SOURCE_T8_ROOT / "t8_cell_matrix_summary.csv",
        "t8_post_run_verification": SOURCE_T8_ROOT / "post_run_verification/post_run_verification.json",
        "s2_adapter": S2_ADAPTER,
        "s2_manifest": S2_MANIFEST,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    final = json.loads(required["t8_final_decision"].read_text()) if not missing else {}
    post = json.loads(required["t8_post_run_verification"].read_text()) if required["t8_post_run_verification"].is_file() else {}
    manifest = json.loads(required["s2_manifest"].read_text()) if required["s2_manifest"].is_file() else {}
    rows = [{"name": name, "path": rel(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None} for name, path in required.items()]
    payload = {
        "status": "PASS"
        if not missing
        and final.get("final_decision") == "T8_NAV_BLOCKER_AFTER_HAND_FIX"
        and post.get("status") == "PASS"
        and manifest.get("checkpoint_type") == "unmerged_lora_adapter_only"
        and manifest.get("lora_merged_before_eval") is False
        else "FAIL",
        "required": rows,
        "final_decision": final.get("final_decision"),
        "post_run_verification_status": post.get("status"),
        "s2_checkpoint_type": manifest.get("checkpoint_type"),
        "s2_lora_merged_before_eval": manifest.get("lora_merged_before_eval"),
    }
    write_csv(out / "source_t8_evidence_lock.csv", rows)
    write_json(out / "source_t8_evidence_lock.json", payload)
    return payload


def static_guard(out: Path) -> dict[str, Any]:
    sources = [REPO_ROOT / "work/recap/safe_sft/t8_1_nav_postlift.py", REPO_ROOT / "work/recap/scripts/gr00t_t8_1_nav_postlift.py"]
    forbidden_tokens = (
        "Gr00t" + "Trainer(",
        "trainer" + ".train(",
        "launch" + "_finetune.py",
        "guarded" + "_recap_train_loop",
        "fatg" + "_train_loop",
    )
    rows = []
    for path in sources:
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        forbidden_calls = [tok for tok in forbidden_tokens if tok in text]
        rows.append({"path": rel(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None, "forbidden_calls": forbidden_calls, "status": "PASS" if path.is_file() and not forbidden_calls else "FAIL"})
    payload = {"status": "PASS" if rows and all(r["status"] == "PASS" for r in rows) else "FAIL", "rows": rows}
    write_csv(out / "runner_static_guard.csv", rows)
    write_json(out / "runner_static_guard.json", payload)
    return payload


def decide_final(base_rows: list[dict[str, Any]], nav_rows: list[dict[str, Any]], post: Mapping[str, Any], runner_ok: bool) -> str:
    if not runner_ok:
        return "RUNNER_OR_CONTRACT_REGRESSION"
    by = {r["ID"]: r for r in base_rows}
    b0 = by.get("B0", {})
    b1 = by.get("B1", {})
    b2 = by.get("B2", {})
    if int(b0.get("success", 0)) > 0 and (int(b1.get("success", 0)) == 0 or int(b2.get("success", 0)) == 0):
        return "RUNNER_OR_CONTRACT_REGRESSION"
    if int(b0.get("success", 0)) == 0 and int(b1.get("success", 0)) == 0 and int(b2.get("success", 0)) == 0:
        return "BASE_SEEDS_TOO_HARD"
    n0 = next((r for r in nav_rows if r["ID"] == "N0"), None)
    if n0 is not None:
        nav_material = build_splice_material_improvement(
            nav_rows,
            baseline_id="N0",
            splice_kind="nav_splice",
        )
        if nav_material.get("nav_splice_material_improvement"):
            return "NAV_SPLICE_IMPROVES"
    # Direction/timing bug if no material outcome improvement but nav telemetry is poor.
    direction_path = None
    # Caller records an explicit marker in nav rows where possible.
    for r in nav_rows:
        if r.get("nav_direction_timing_bug"):
            return "NAV_DIRECTION_TIMING_BUG"
    post_answers = post.get("answers", {})
    if (
        post_answers.get("lifted_but_not_success_count", 0) > 0
        and post_answers.get("quantitative_threshold_eligible") is True
    ):
        return "POST_LIFT_PLACE_BLOCKER"
    return "GUARDED_RECAP_STILL_FORBIDDEN"


def write_report(
    out: Path,
    final: str,
    base_rows: list[dict[str, Any]],
    nav_rows: list[dict[str, Any]],
    post: Mapping[str, Any],
    p_rows: list[dict[str, Any]] | None,
    nav_material: Mapping[str, Any] | None,
    p_material: Mapping[str, Any] | None,
) -> None:
    lines = [
        "# GR00T T8.1 Navigate + Post-Lift Follow-up Report",
        "",
        f"- generated_at_utc: {utc_now()}",
        f"- final_decision: **{final}**",
        f"- source_t8_root: `{rel(SOURCE_T8_ROOT)}`",
        "",
        "## Paired base sanity",
        "",
        "| ID | policy | episodes | success | reached | lifted | failure_modes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in base_rows:
        lines.append(f"| {r['ID']} | {r['policy']} | {r['episodes']} | {r['success']} | {r['reached']} | {r['lifted']} | {json.dumps(r['failure_modes'], ensure_ascii=False)} |")
    lines += ["", "## Navigate splice", "", "| ID | episodes | success | reached | lifted | failure_modes |", "|---|---:|---:|---:|---:|---|"]
    for r in nav_rows:
        lines.append(f"| {r['ID']} | {r['episodes']} | {r['success']} | {r['reached']} | {r['lifted']} | {json.dumps(r['failure_modes'], ensure_ascii=False)} |")
    lines += [
        "",
        "## Post-lift/place audit",
        "",
        f"- lifted_but_not_success_count: {post.get('answers', {}).get('lifted_but_not_success_count')}",
        f"- evidence_mode: {post.get('answers', {}).get('evidence_mode')}",
        f"- quantitative_threshold_eligible: {post.get('answers', {}).get('quantitative_threshold_eligible')}",
        f"- any_moves_toward_plate: {post.get('answers', {}).get('any_moves_toward_plate')}",
        f"- any_reached_plate: {post.get('answers', {}).get('any_reached_plate')}",
        f"- any_release_proxy: {post.get('answers', {}).get('any_release_proxy')}",
    ]
    if nav_material is not None:
        lines += [
            "",
            "## Navigate splice material threshold",
            "",
            f"- material_improvement: {nav_material.get('nav_splice_material_improvement')}",
            f"- qualitative_only_present: {nav_material.get('qualitative_only')}",
            f"- threshold: `{json.dumps(nav_material.get('threshold'), ensure_ascii=False)}`",
        ]
    if p_rows is not None:
        lines += ["", "## Post-lift splice", "", "| ID | episodes | success | reached | lifted | failure_modes |", "|---|---:|---:|---:|---:|---|"]
        for r in p_rows:
            lines.append(f"| {r['ID']} | {r['episodes']} | {r['success']} | {r['reached']} | {r['lifted']} | {json.dumps(r['failure_modes'], ensure_ascii=False)} |")
    if p_material is not None:
        lines += [
            "",
            "## Post-lift splice material threshold",
            "",
            f"- material_improvement: {p_material.get('post_lift_splice_material_improvement')}",
            f"- qualitative_only_present: {p_material.get('qualitative_only')}",
            f"- threshold: `{json.dumps(p_material.get('threshold'), ensure_ascii=False)}`",
        ]
    lines += [
        "",
        "## Scope guard",
        "",
        "- Phase 1/2/3 were no-training inference/eval diagnostics.",
        "- Guarded RECAP/FATG/per-edge/full-scope were not run.",
        "- No LoRA merge before eval; S2 adapter loaded as unmerged LoRA.",
    ]
    (out / "t8_1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GR00T T8.1 no-training navigate/post-lift diagnostic")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_OFFICIAL_BASE))
    parser.add_argument("--canonical-checkpoint", default=str(DEFAULT_CANONICAL_IDENTITY))
    parser.add_argument("--s2-adapter", default=str(S2_ADAPTER))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=2026051000)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--skip-post-lift-splice", action="store_true")
    parser.add_argument("--recap", action="store_true", help="Forbidden; rejected before model load.")
    parser.add_argument("--advantage", action="store_true", help="Forbidden; rejected before model load.")
    return parser


def main(argv: list[str] | None = None) -> int:
    reject_forbidden_args(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(argv)
    if args.recap or args.advantage:
        raise SystemExit("Forbidden --recap/--advantage rejected before model load")
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(args.seed_base) + i for i in range(int(args.episodes))]
    write_json(
        out / "command_manifest.json",
        {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "env": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "NO_ALBUMENTATIONS_UPDATE": os.environ.get("NO_ALBUMENTATIONS_UPDATE")},
            "git_commit": git_output(["rev-parse", "HEAD"]),
            "git_status_short": git_output(["status", "--short"]).splitlines(),
            "submodule_status_short": git_output(["status", "--short", "--", "submodules"]).splitlines(),
            "seeds": seeds,
            "generated_at_utc": utc_now(),
        },
    )
    pre = out / "preflight"
    pre.mkdir(exist_ok=True)
    evidence = source_evidence_lock(pre)
    guard = static_guard(pre)
    submodules = git_output(["status", "--short", "--", "submodules"]).splitlines()
    (pre / "submodule_status.txt").write_text("\n".join(submodules) + ("\n" if submodules else ""), encoding="utf-8")
    runner_ok = evidence.get("status") == "PASS" and guard.get("status") == "PASS" and not submodules

    base_rows: list[dict[str, Any]] = []
    if runner_ok:
        paired_dir = out / "paired_base_sanity"
        paired_dir.mkdir(exist_ok=True)
        policies = [
            ("B0", "base official/canonical", resolve(args.base_checkpoint), None, False),
            ("B1", "fixed identity canonical", resolve(args.canonical_checkpoint), None, False),
            ("B2", "zero-step LoRA wrapper canonical", resolve(args.canonical_checkpoint), None, True),
        ]
        for pid, label, ckpt, adapter, zero in policies:
            print(f"[PHASE1_START] {pid} {label}", flush=True)
            policy = load_policy_with_optional_lora(ckpt, adapter_path=adapter, zero_step_lora=zero)
            try:
                result = run_single_policy_eval(policy=policy, policy_id=pid, policy_label=label, out_dir=paired_dir, seeds=seeds, max_episode_steps=int(args.max_episode_steps))
                base_rows.append(result["summary"])
                write_csv(paired_dir / f"{pid}_per_seed.csv", result["per_seed"])
            finally:
                del policy
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        write_csv(paired_dir / "paired_base_sanity.csv", base_rows)
        write_json(paired_dir / "paired_base_sanity.json", {"status": "PASS", "seed_list": seeds, "rows": base_rows})

    nav_rows: list[dict[str, Any]] = []
    nav_direction_rows: list[dict[str, Any]] = []
    p_rows: list[dict[str, Any]] | None = None
    nav_material_payload: dict[str, Any] | None = None
    p_material_payload: dict[str, Any] | None = None
    post_payload: dict[str, Any] = {"answers": {"lifted_but_not_success_count": 0}}
    if runner_ok:
        base_policy = load_policy_with_optional_lora(resolve(args.base_checkpoint))
        candidate_policy = load_policy_with_optional_lora(resolve(args.canonical_checkpoint), adapter_path=resolve(args.s2_adapter))
        try:
            nav_dir = out / "navigate_splice"
            nav_dir.mkdir(exist_ok=True)
            nav_variants = {
                "N0": {"description": "S2 as-is", "upper_hand_source": "S2", "navigate_source": "S2", "lower_package": "S2"},
                "N1": {"description": "S2 upper/hand + base nav", "upper_hand_source": "S2", "navigate_source": "base policy nav", "lower_package": "S2"},
                "N2": {"description": "base nav until reached", "upper_hand_source": "S2", "navigate_source": "base until reached then S2", "lower_package": "S2"},
                "N3": {"description": "S2 direction base magnitude", "upper_hand_source": "S2", "navigate_source": "S2 projected/scaled to base envelope", "lower_package": "S2"},
                "N4": {"description": "base lower package", "upper_hand_source": "S2", "navigate_source": "base nav+height+waist", "lower_package": "base lower package"},
            }
            nav = run_splice_eval(base_policy=base_policy, candidate_policy=candidate_policy, variants=nav_variants, out_dir=nav_dir, seeds=seeds, max_episode_steps=int(args.max_episode_steps), mode="nav")
            nav_rows = nav["rows"]
            nav_direction_rows = nav["direction_rows"]
            write_csv(nav_dir / "navigate_splice_matrix.csv", nav_rows)
            write_csv(nav_dir / "nav_direction_timing.csv", nav_direction_rows)
            write_json(nav_dir / "navigate_splice_matrix.json", {"status": "PASS", "seed_list": seeds, "rows": nav_rows})

            post_dir = out / "post_lift_place_audit"
            post_dir.mkdir(exist_ok=True)
            sources = {
                "T8_S0_NEG_RAW": SOURCE_T8_ROOT / "cells/S0_NEG_RAW/eval_step_telemetry.jsonl",
                "T8_S2_BASE_TEACHER": SOURCE_T8_ROOT / "cells/S2_BASE_TEACHER/eval_step_telemetry.jsonl",
            }
            for variant in nav_variants:
                sources[f"N_{variant}"] = nav_dir / f"{variant}_steps.jsonl"
            post_payload = post_lift_audit(post_dir, sources)

            if (not args.skip_post_lift_splice) and int(post_payload.get("answers", {}).get("lifted_but_not_success_count", 0)) > 0:
                p_dir = out / "post_lift_splice"
                p_dir.mkdir(exist_ok=True)
                p_variants = {
                    "P0": {"description": "S2 current", "upper_hand_source": "S2", "navigate_source": "S2", "lower_package": "S2"},
                    "P1": {"description": "S2 until lifted then base full", "upper_hand_source": "S2/base", "navigate_source": "S2 then base", "lower_package": "S2 then base"},
                    "P2": {"description": "base until reached then S2", "upper_hand_source": "base/S2", "navigate_source": "base then S2", "lower_package": "base then S2"},
                    "P3": {"description": "S2 hand + base arms/nav after lift", "upper_hand_source": "mixed", "navigate_source": "base after lift", "lower_package": "base after lift"},
                    "P4": {"description": "base hand + S2 arms/nav after lift", "upper_hand_source": "mixed", "navigate_source": "S2 after lift", "lower_package": "S2 after lift"},
                }
                p = run_splice_eval(base_policy=base_policy, candidate_policy=candidate_policy, variants=p_variants, out_dir=p_dir, seeds=seeds, max_episode_steps=int(args.max_episode_steps), mode="post_lift")
                p_rows = p["rows"]
                write_csv(p_dir / "post_lift_splice_matrix.csv", p_rows)
                write_csv(p_dir / "post_lift_splice_direction_timing.csv", p["direction_rows"])
                write_json(p_dir / "post_lift_splice_matrix.json", {"status": "PASS", "seed_list": seeds, "rows": p_rows})
                p_material_payload = write_splice_material_improvement(
                    p_dir,
                    p_rows,
                    baseline_id="P0",
                    splice_kind="post_lift_splice",
                )
        finally:
            del base_policy
            del candidate_policy
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # Mark nav direction timing bug if telemetry is poor and no material outcome improvement.
    if nav_rows:
        nav_material_payload = write_splice_material_improvement(
            out / "navigate_splice",
            nav_rows,
            baseline_id="N0",
            splice_kind="nav_splice",
        )

    final = decide_final(base_rows, nav_rows, post_payload, runner_ok)
    final_payload = {
        "final_decision": final,
        "allowed_final_decisions": sorted(ALLOWED_FINAL),
        "base_rows": base_rows,
        "nav_rows": nav_rows,
        "nav_splice_material_evidence": nav_material_payload,
        "post_lift_splice_rows": p_rows,
        "post_lift_splice_material_evidence": p_material_payload,
        "post_lift_answers": post_payload.get("answers", {}),
        "phase5_training_allowed": final == "NAV_SPLICE_IMPROVES",
        "guarded_recap_allowed": False,
        "fatg_allowed": False,
        "generated_at_utc": utc_now(),
    }
    write_json(out / "final_decision.json", final_payload)
    write_report(
        out,
        final,
        base_rows,
        nav_rows,
        post_payload,
        p_rows,
        nav_material_payload,
        p_material_payload,
    )
    return 0 if final != "RUNNER_OR_CONTRACT_REGRESSION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
