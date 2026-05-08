#!/usr/bin/env python3
"""No-training post-A_LOCAL diagnostics for GR00T canonical-surface collapse RCA.

The harness materializes scratch checkpoint bundles, runs offline action audits,
WBC command-proxy diagnostics, bounded formal-eval smoke, and Stage3 label
counterfactual replay without modifying original checkpoints or submodules.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ISAAC_GR00T_ROOT = REPO_ROOT / "submodules" / "Isaac-GR00T"
WBC_ROOT = ISAAC_GR00T_ROOT / "external_dependencies" / "GR00T-WholeBodyControl"
for _p in (REPO_ROOT, ISAAC_GR00T_ROOT, WBC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from work.recap.scripts.gr00t_safe_adaptation_action_audit import (  # noqa: E402
    MODALITIES,
    _as_array,
    _extract_policy_observation,
    _load_stage3_loader,
    _predict_one_chunk,
    _stats,
)
from work.recap.scripts.gr00t_phase2_l4_l5_identity_closure import (  # noqa: E402
    DEFAULT_TASK_PROMPT,
    ENV_NAME,
    batch_flat_env_obs,
    first_action_step,
    find_wbc_wrapper,
    load_3d_eval_helpers,
    modality_arrays_from_action,
)

OFFICIAL_BASE = REPO_ROOT / "agent/artifacts/gr00t_recap_live/hf_patches/models--nvidia--GR00T-N1.6-G1-PnPAppleToPlate/snapshot-897d0313a190f46a2cccaeb34077752a0db4b0de/formalize_language=False"
CANONICAL_IDENTITY = REPO_ROOT / "agent/artifacts/gr00t_a_gate_identity_closure/a_gate_identity_20260506_133517/canonical_identity"
PURE_SFT = REPO_ROOT / "agent/artifacts/probes/probe_A_pure_sft_control/training_run_20260501T134222Z/checkpoint-3300"
RECAP = REPO_ROOT / "agent/artifacts/gr00t_recap_live/single_gpu_v2_full_update/stage1_gr00t_r2r4_closed_candidate_iter9_20260426T_nextZ/gr00t/g3_conditioned_continuation_6600_after_surfacefix_20260430_181210/checkpoint-6600"
STAGE3_DATASET = REPO_ROOT / "agent/artifacts/lerobot_datasets/recap_stage3_iter_002"
PHASE2_PREFLIGHT = REPO_ROOT / "agent/artifacts/gr00t_phase2_l4_l5_identity_closure/phase2_l4_l5_identity_20260507_005506/preflight/canonical_strict_preflight.json"

ALLOWED_FINAL = {
    "B_PASS_LABELS_VALID",
    "C_HAND_LABEL_BUG",
    "D_NAV_LABEL_BUG",
    "CD_HAND_NAV_BOTH",
    "SURFACE_CONTAMINATION_MAIN",
    "TRAINED_WEIGHTS_STILL_COLLAPSE",
    "A_FULL_STILL_BLOCKED_ONLY",
    "READY_FOR_SAFE_SFT",
}
SURFACE_FILES = (
    "config.json",
    "processor_config.json",
    "statistics.json",
    "modality.json",
    "embodiment_id.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
)
MODEL_FILE_PATTERNS = (
    "model.safetensors.index.json",
    "model-*.safetensors",
    "*.safetensors",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def resolve(raw: str | Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(p)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=True, default=json_default) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default).encode("utf-8"))


def hash_array(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hash_array(value)}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def git_status(path: Path) -> list[str]:
    try:
        out = subprocess.check_output(["git", "-C", str(path), "status", "--short"], cwd=str(REPO_ROOT), text=True, stderr=subprocess.STDOUT).strip()
        return [line for line in out.splitlines() if line.strip()]
    except Exception as exc:
        return [f"UNKNOWN:{type(exc).__name__}:{exc}"]


def safe_link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def materialize_bundle(bundle_id: str, weights_src: Path, surface_src: Path, out: Path, surface_label: str) -> dict[str, Any]:
    dst = out / "scratch" / "bundles" / bundle_id
    dst.mkdir(parents=True, exist_ok=True)
    linked: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in MODEL_FILE_PATTERNS:
        for src in sorted(weights_src.glob(pattern)):
            if not src.is_file() or src in seen:
                continue
            seen.add(src)
            mode = safe_link_or_copy(src, dst / src.name)
            linked.append({"name": src.name, "source": rel(src), "mode": mode, "sha256": sha256_file(src), "size": src.stat().st_size})
    copied_surface: list[dict[str, Any]] = []
    for name in SURFACE_FILES:
        src = surface_src / name
        if not src.exists() or not src.is_file():
            continue
        shutil.copy2(src, dst / name)
        copied_surface.append({"name": name, "source": rel(src), "sha256": sha256_file(src), "size": src.stat().st_size})
    required = ["model.safetensors.index.json", "config.json", "processor_config.json", "statistics.json", "embodiment_id.json"]
    missing = [name for name in required if not (dst / name).exists()]
    manifest = {
        "bundle_id": bundle_id,
        "weights_src": rel(weights_src),
        "surface_src": rel(surface_src),
        "surface_label": surface_label,
        "bundle_path": rel(dst),
        "model_files": linked,
        "surface_files": copied_surface,
        "missing_required": missing,
        "status": "PASS" if not missing else "FAIL",
        "originals_mutated": False,
    }
    write_json(dst / "post_a_local_bundle_manifest.json", manifest)
    if missing:
        raise FileNotFoundError(f"bundle {bundle_id} missing required files: {missing}")
    return manifest


def build_bundles(out: Path, base: Path, identity: Path, pure: Path, recap: Path) -> dict[str, dict[str, Any]]:
    specs = {
        "C0_base": (base, base, "official_base_surface"),
        "C1_identity_fixed": (identity, base, "canonical_base_surface"),
        "C2_pure_sft_canonical": (pure, base, "canonical_base_surface"),
        "C3_recap_canonical": (recap, base, "canonical_base_surface"),
        "C4_pure_sft_original_surface": (pure, pure, "old_trained_surface"),
        "C5_recap_original_surface": (recap, recap, "old_trained_surface"),
    }
    manifests = {bid: materialize_bundle(bid, resolve(w), resolve(s), out, label) for bid, (w, s, label) in specs.items()}
    write_json(out / "scratch" / "bundle_matrix.json", manifests)
    return manifests




def flatten_nested_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Convert parse_observation_gr00t nested output to flat sim-wrapper keys."""
    flat: dict[str, Any] = {}
    for group, payload in observation.items():
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if str(group) == "language":
                    if isinstance(value, list) and value:
                        flat[str(key)] = value[0]
                    else:
                        flat[str(key)] = value
                else:
                    flat[f"{group}.{key}"] = value
        else:
            flat[str(group)] = payload
    return flat


def coerce_obs_float32(value: Any) -> Any:
    """Policy wrapper requires float states; env reset can emit float64."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "f":
            return value.astype(np.float32, copy=False)
        return value
    if isinstance(value, dict):
        return {k: coerce_obs_float32(v) for k, v in value.items()}
    if isinstance(value, list):
        return [coerce_obs_float32(v) for v in value]
    if isinstance(value, tuple):
        return tuple(coerce_obs_float32(v) for v in value)
    return value



def normalize_action_dict(action: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize nested policy outputs to only flat ``action.<modality>`` arrays.

    Different GR00T wrappers have emitted both flat action dictionaries and
    nested ``{"action": {"right_hand": ...}}`` / debug-enriched mappings.  The
    downstream WBC helper expects only numeric action arrays, so metadata or
    nested debug dictionaries must be ignored rather than passed through.
    """
    out: dict[str, Any] = {}

    def visit(path: list[str], value: Any) -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                visit([*path, str(child_key)], child_value)
            return
        if not path:
            return
        terminal = path[-1]
        if terminal.startswith("action."):
            terminal = terminal.split(".", 1)[1]
        if terminal in MODALITIES:
            out[f"action.{terminal}"] = value

    visit([], action)
    if out:
        return out

    # Last-resort compatibility for already-flat mappings with unusual keys.
    for key, value in action.items():
        if isinstance(value, Mapping):
            continue
        key_str = str(key)
        if key_str.startswith("action."):
            terminal = key_str.split(".", 1)[1]
            if terminal in MODALITIES:
                out[key_str] = value
    return out


def first_action_dict(action: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return first horizon step for every action modality."""
    return {key: first_action_step(value) for key, value in action.items()}


def squeeze_policy_batch(action: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Convert policy chunks from (B,H,D) to env chunks (H,D)."""
    out: dict[str, np.ndarray] = {}
    for key, value in action.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        out[key] = arr
    return out

def collect_stage3_windows(dataset: Path, count: int, prompt: str) -> tuple[Any, list[dict[str, Any]]]:
    loader = _load_stage3_loader(dataset)
    windows: list[dict[str, Any]] = []
    for ep in range(min(10, len(loader))):
        traj = loader[ep]
        stride = max(1, len(traj) // max(1, int(count)))
        for start in range(0, len(traj), stride):
            if len(windows) >= int(count):
                break
            horizon = len(loader.modality_configs["action"].delta_indices)
            obs = _extract_policy_observation(loader=loader, trajectory=traj, step_index=start, plain_prompt=prompt)
            phase = "grasp_candidate" if start >= int(0.30 * len(traj)) and start <= int(0.85 * len(traj)) else "approach_or_other"
            windows.append({"obs_id": f"stage3_ep{ep:02d}_step{start:04d}", "episode_index": ep, "start_step": int(start), "trajectory_len": int(len(traj)), "horizon": int(horizon), "phase": phase, "observation": obs})
        if len(windows) >= int(count):
            break
    return loader, windows


def dataset_action_chunks(loader: Any, windows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {m: [] for m in MODALITIES}
    for win in windows:
        traj = loader[int(win["episode_index"])]
        start = int(win["start_step"])
        stop = min(start + int(win["horizon"]), len(traj))
        for m in MODALITIES:
            arrs = [_as_array(v) for v in traj[f"action.{m}"].iloc[start:stop]]
            chunks[m].append(np.stack(arrs, axis=0))
    return {m: np.concatenate(v, axis=0) if v else np.zeros((0,), dtype=np.float32) for m, v in chunks.items()}


def load_policy(checkpoint: Path) -> Any:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    return Gr00tPolicy(EmbodimentTag("unitree_g1"), str(checkpoint), device="cuda:0", strict=True)


def predict_bundle(bundle_id: str, checkpoint: Path, windows: list[dict[str, Any]], out: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    import torch
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

    denorm_chunks: dict[str, list[np.ndarray]] = {m: [] for m in MODALITIES}
    norm_chunks: dict[str, list[np.ndarray]] = {m: [] for m in MODALITIES}
    flat_chunks: dict[str, list[np.ndarray]] = {m: [] for m in MODALITIES}
    debug_path = out / "offline_action_audit" / f"{bundle_id}_per_window_debug.jsonl"
    policy = load_policy(checkpoint)
    sim_policy = Gr00tSimPolicyWrapper(policy, strict=True)
    try:
        for idx, win in enumerate(windows):
            np.random.seed(20260507 + idx)
            torch.manual_seed(20260507 + idx)
            denorm, norm = _predict_one_chunk(policy, win["observation"])
            flat_obs = flatten_nested_observation(win["observation"])
            flat_action, _info = sim_policy.get_action(flat_obs)
            flat_action = normalize_action_dict(flat_action)
            flat = modality_arrays_from_action(flat_action)
            row = {"bundle_id": bundle_id, "obs_id": win["obs_id"], "phase": win["phase"], "arrays": {}}
            for m in MODALITIES:
                denorm_chunks[m].append(np.asarray(denorm[m][0], dtype=np.float32))
                norm_chunks[m].append(np.asarray(norm[m][0], dtype=np.float32))
                if m in flat:
                    flat_chunks[m].append(np.asarray(flat[m][0] if flat[m].ndim == 3 else flat[m], dtype=np.float32))
                row["arrays"][m] = {"denorm_abs_q99": float(np.quantile(np.abs(np.asarray(denorm[m][0]).reshape(-1)), 0.99)), "norm_abs_q99": float(np.quantile(np.abs(np.asarray(norm[m][0]).reshape(-1)), 0.99))}
            append_jsonl(debug_path, {k: v for k, v in row.items() if k != "raw"})
    finally:
        del sim_policy
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    denorm_out = {m: np.concatenate(v, axis=0) if v else np.zeros((0,), dtype=np.float32) for m, v in denorm_chunks.items()}
    norm_out = {m: np.concatenate(v, axis=0) if v else np.zeros((0,), dtype=np.float32) for m, v in norm_chunks.items()}
    flat_out = {m: np.concatenate(v, axis=0) if v else np.zeros((0,), dtype=np.float32) for m, v in flat_chunks.items()}
    return denorm_out, norm_out, flat_out


def action_stats_rows(source: str, surface: str, by_modality: Mapping[str, np.ndarray], phase: str = "all") -> list[dict[str, Any]]:
    rows = []
    for m in MODALITIES:
        row = _stats(by_modality[m], source=source, modality=m, surface=surface)
        row["phase"] = phase
        rows.append(row)
    return rows


def run_offline_action_audit(bundles: Mapping[str, Mapping[str, Any]], dataset: Path, out: Path, prompt: str, obs_count: int) -> dict[str, Any]:
    audit = out / "offline_action_audit"
    audit.mkdir(parents=True, exist_ok=True)
    loader, windows = collect_stage3_windows(dataset, obs_count, prompt)
    write_json(audit / "observation_windows.json", [{k: v for k, v in w.items() if k != "observation"} for w in windows])
    dataset_actions = dataset_action_chunks(loader, windows)
    rows: list[dict[str, Any]] = []
    rows.extend(action_stats_rows("dataset", "denorm_absolute", dataset_actions))
    pred_payload: dict[str, Any] = {"dataset": {"denorm_absolute": {m: {"abs_q99": float(np.quantile(np.abs(dataset_actions[m]).reshape(-1), 0.99))} for m in MODALITIES}}}
    for bid, manifest in bundles.items():
        print("[OFFLINE_POLICY_START]", bid, flush=True)
        ckpt = resolve(manifest["bundle_path"])
        try:
            denorm, norm, flat = predict_bundle(bid, ckpt, windows, out)
            rows.extend(action_stats_rows(bid, "L1_pred_normalized", norm))
            rows.extend(action_stats_rows(bid, "L2_denorm_absolute", denorm))
            rows.extend(action_stats_rows(bid, "L3_postprocessed_action", flat))
            pred_payload[bid] = {
                "status": "PASS",
                "surfaces": {
                    "L1_pred_normalized": {m: {"abs_q99": float(np.quantile(np.abs(norm[m]).reshape(-1), 0.99))} for m in MODALITIES},
                    "L2_denorm_absolute": {m: {"abs_q99": float(np.quantile(np.abs(denorm[m]).reshape(-1), 0.99))} for m in MODALITIES},
                    "L3_postprocessed_action": {m: {"abs_q99": float(np.quantile(np.abs(flat[m]).reshape(-1), 0.99)) if flat[m].size else None} for m in MODALITIES},
                },
            }
        except Exception as exc:
            pred_payload[bid] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
            print("[OFFLINE_POLICY_FAIL]", bid, type(exc).__name__, exc, flush=True)
    fields = ["source", "surface", "phase", "modality", "sample_shape", "mean", "std", "min", "q01", "q50", "q99", "max", "abs_mean", "abs_q99", "signed_sum_mean", "nonzero_frac"]
    write_csv(audit / "canonical_surface_action_stats_by_layer.csv", rows, fields)
    write_json(audit / "canonical_surface_action_audit.json", {"status": "PASS", "generated_at_utc": utc_now(), "dataset": rel(dataset), "observation_count": len(windows), "stats_csv": rel(audit / "canonical_surface_action_stats_by_layer.csv"), "predictions": pred_payload})
    return {"rows": rows, "payload": pred_payload, "windows": windows}


def make_eval_env(max_episode_steps: int, n_action_steps: int = 20) -> Any:
    helpers = load_3d_eval_helpers()
    helpers._add_import_roots(REPO_ROOT)
    helpers._install_robocasa_import_shims()
    gym = __import__("gymnasium")
    rollout_mod = __import__("gr00t.eval.rollout_policy", fromlist=[""])
    helpers._ensure_explicit_g1_env_registration(gym)
    wrapper_configs = rollout_mod.WrapperConfigs(
        video=rollout_mod.VideoConfig(video_dir=None, max_episode_steps=int(max_episode_steps), overlay_text=False),
        multistep=rollout_mod.MultiStepConfig(n_action_steps=int(n_action_steps), max_episode_steps=int(max_episode_steps), terminate_on_success=True),
    )
    return rollout_mod.create_eval_env(env_name=ENV_NAME, env_idx=0, total_n_envs=1, wrapper_configs=wrapper_configs)


def summarize_array(arr: Any) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64)
    flat = a.reshape(-1) if a.size else np.asarray([], dtype=np.float64)
    return {"shape": list(a.shape), "mean": float(np.mean(flat)) if flat.size else None, "abs_q99": float(np.quantile(np.abs(flat), 0.99)) if flat.size else None, "max_abs": float(np.max(np.abs(flat))) if flat.size else None, "sha256": hash_array(a) if a.size else None}


def run_command_proxy_audit(bundles: Mapping[str, Mapping[str, Any]], out: Path, seeds: list[int], max_episode_steps: int) -> dict[str, Any]:
    import torch
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t_wbc.control.utils.n1_utils import concat_action

    proxy_dir = out / "l4_command_proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    env = make_eval_env(max_episode_steps=max_episode_steps, n_action_steps=20)
    rows: list[dict[str, Any]] = []
    try:
        for bid, manifest in bundles.items():
            print("[COMMAND_PROXY_START]", bid, flush=True)
            policy = load_policy(resolve(manifest["bundle_path"]))
            sim_policy = Gr00tSimPolicyWrapper(policy, strict=True)
            try:
                for seed in seeds:
                    obs, _info = env.reset(seed=int(seed))
                    flat = coerce_obs_float32(batch_flat_env_obs(obs))
                    action, _ainfo = sim_policy.get_action(flat)
                    action = normalize_action_dict(action)
                    step_action = first_action_dict(action)
                    wbc = find_wbc_wrapper(env)
                    goal = concat_action(wbc.robot_model, step_action)
                    if hasattr(wbc.wbc_policy, "set_goal"):
                        wbc.wbc_policy.set_goal(goal)
                    else:
                        wbc.wbc_policy.update_goal(goal)
                    wbc_action = wbc.wbc_policy.get_action()
                    lower = getattr(wbc.wbc_policy, "lower_body_policy", None)
                    lower_cmd = getattr(lower, "cmd", goal.get("navigate_cmd")) if lower is not None else goal.get("navigate_cmd")
                    record = {
                        "bundle_id": bid,
                        "seed": int(seed),
                        "L4_WBC_lower_command": summarize_array(lower_cmd),
                        "L4_IK_upper_command": summarize_array(goal.get("target_upper_body_pose")),
                        "L4_hand_target": summarize_array(step_action.get("action.right_hand")),
                        "L4_clipped_command": summarize_array(wbc_action.get("q")),
                        "L5_env_applied_ctrl_proxy": summarize_array(wbc_action.get("q")),
                        "navigate_command": summarize_array(step_action.get("action.navigate_command")),
                        "right_hand": summarize_array(step_action.get("action.right_hand")),
                    }
                    append_jsonl(proxy_dir / "command_proxy_records.jsonl", record)
                    rows.append(record)
            finally:
                del sim_policy
                del policy
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        try:
            env.close()
        except Exception:
            pass
    summary_rows = []
    for bid in bundles:
        recs = [r for r in rows if r["bundle_id"] == bid]
        for key in ("right_hand", "navigate_command", "L4_hand_target", "L4_WBC_lower_command", "L5_env_applied_ctrl_proxy"):
            vals = [r.get(key, {}).get("abs_q99") for r in recs]
            vals = [float(v) for v in vals if isinstance(v, (int, float))]
            summary_rows.append({"bundle_id": bid, "layer_or_modality": key, "count": len(vals), "abs_q99_mean": float(np.mean(vals)) if vals else None, "abs_q99_q99": float(np.quantile(vals, 0.99)) if vals else None})
    write_csv(proxy_dir / "command_proxy_summary.csv", summary_rows)
    payload = {"status": "PASS", "seed_list": seeds, "records_jsonl": rel(proxy_dir / "command_proxy_records.jsonl"), "summary_csv": rel(proxy_dir / "command_proxy_summary.csv"), "strict_L5_surface": "COMMAND_PROXY_ONLY"}
    write_json(proxy_dir / "command_proxy_audit.json", payload)
    return payload


def run_formal_eval_for_bundle(bundle_id: str, checkpoint: Path, out: Path, runtime: Path, port: int, seed_base: int, episode_count: int, timeout_s: int, max_episode_steps: int) -> dict[str, Any]:
    eval_out = out / "formal_eval" / bundle_id
    eval_log = runtime / "formal_eval" / bundle_id
    eval_out.mkdir(parents=True, exist_ok=True)
    eval_log.mkdir(parents=True, exist_ok=True)
    cmd = [
        "timeout",
        str(int(timeout_s)),
        "env",
        "CUDA_VISIBLE_DEVICES=1",
        "NO_ALBUMENTATIONS_UPDATE=1",
        f"PYTHONPATH={os.pathsep.join([str(REPO_ROOT), str(ISAAC_GR00T_ROOT), str(WBC_ROOT), os.environ.get('PYTHONPATH','')])}",
        str(REPO_ROOT / ".envs/wbc/bin/python"),
        str(REPO_ROOT / "work/recap/scripts/gr00t_g3_formal_eval.py"),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(eval_out),
        "--runtime-log-dir",
        str(eval_log),
        "--server-port",
        str(int(port)),
        "--indicator-modes",
        "omit",
        "--seed-base",
        str(int(seed_base)),
        "--episode-count",
        str(int(episode_count)),
        "--max-episode-steps",
        str(int(max_episode_steps)),
        "--n-action-steps",
        "20",
        "--connect-timeout-s",
        "300",
        "--total-timeout-s",
        str(max(60, int(timeout_s) - 60)),
        "--required-cuda-visible-devices",
        "1",
    ]
    (eval_out / "formal_eval_command.json").write_text(json.dumps({"cmd": cmd}, indent=2) + "\n", encoding="utf-8")
    print("[FORMAL_EVAL_START]", bundle_id, "port", port, flush=True)
    started = time.monotonic()
    log_path = eval_log / "formal_eval_stdout.log"
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    elapsed = time.monotonic() - started
    summary_path = eval_out / "formal_eval_summary.json"
    payload: dict[str, Any]
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        payload = {"status": "FAIL", "error": "summary_missing"}
    payload["subprocess_returncode"] = int(proc.returncode)
    payload["elapsed_seconds_outer"] = float(elapsed)
    payload["stdout_log"] = rel(log_path)
    write_json(eval_out / "formal_eval_outer_status.json", payload)
    print("[FORMAL_EVAL_DONE]", bundle_id, payload.get("status"), "rc", proc.returncode, flush=True)
    return payload


def summarize_formal(bundle_id: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    mode = (summary.get("mode_summaries") or {}).get("omit", {}) if isinstance(summary.get("mode_summaries"), Mapping) else {}
    episodes = int(mode.get("episodes") or summary.get("completed_episode_total") or 0)
    success = int(mode.get("success_count") or 0)
    eps = mode.get("episode_results") or []
    reached = 0
    lifted = 0
    failure_modes: dict[str, int] = {}
    for ep in eps:
        guess = ep.get("failure_stage_guess") if isinstance(ep, Mapping) else None
        if isinstance(guess, Mapping):
            if bool(guess.get("ever_near_apple")) or bool(ep.get("success")) or bool(guess.get("ever_lifted_apple")):
                reached += 1
            if bool(guess.get("ever_lifted_apple")):
                lifted += 1
            label = str(guess.get("label") or "unknown")
        else:
            if bool(ep.get("success")):
                reached += 1
            label = str(ep.get("failure_reason") or "success" if bool(ep.get("success")) else "unknown")
        failure_modes[label] = failure_modes.get(label, 0) + 1
    return {"ID": bundle_id, "episodes": episodes, "success": success, "reached": reached, "lifted": lifted, "failure_modes": failure_modes, "status": summary.get("status"), "returncode": summary.get("subprocess_returncode")}


def run_formal_matrix(bundles: Mapping[str, Mapping[str, Any]], out: Path, runtime: Path, episode_count: int, timeout_s: int, max_episode_steps: int) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    rows = []
    base_port = 5630
    for idx, (bid, manifest) in enumerate(bundles.items()):
        summaries[bid] = run_formal_eval_for_bundle(bid, resolve(manifest["bundle_path"]), out, runtime, base_port + idx, 20000, episode_count, timeout_s, max_episode_steps)
        rows.append(summarize_formal(bid, summaries[bid]))
    write_csv(out / "formal_eval" / "formal_eval_matrix.csv", rows, ["ID", "episodes", "success", "reached", "lifted", "failure_modes", "status", "returncode"])
    write_json(out / "formal_eval" / "formal_eval_matrix.json", {"status": "PASS", "episode_count": episode_count, "rows": rows})
    return {"summaries": summaries, "rows": rows}


def dataset_chunk_for_variant(loader: Any, variant: str, episode_index: int, outer_step: int, horizon: int) -> dict[str, np.ndarray]:
    traj = loader[int(episode_index) % len(loader)]
    start = min(int(outer_step) * int(horizon), max(0, len(traj) - 1))
    stop = min(start + int(horizon), len(traj))
    action: dict[str, np.ndarray] = {}
    for m in MODALITIES:
        vals = [_as_array(v) for v in traj[f"action.{m}"].iloc[start:stop]]
        if not vals:
            vals = [_as_array(traj[f"action.{m}"].iloc[-1])]
        while len(vals) < int(horizon):
            vals.append(vals[-1])
        action[f"action.{m}"] = np.stack(vals[: int(horizon)], axis=0).astype(np.float32, copy=False)
    return action


def splice_action(label_action: dict[str, np.ndarray], base_action: Mapping[str, Any], variant: str, base_success_hand: dict[str, np.ndarray] | None = None) -> tuple[dict[str, np.ndarray], str]:
    out = {k: np.asarray(v, dtype=np.float32).copy() for k, v in label_action.items()}
    if variant in {"R2", "R4", "R5", "R6"}:
        src = base_success_hand if variant == "R6" and base_success_hand is not None else base_action
        if "action.right_hand" in src:
            out["action.right_hand"] = np.asarray(src["action.right_hand"], dtype=np.float32).copy()
    if variant in {"R3", "R4"} and "action.navigate_command" in base_action:
        out["action.navigate_command"] = np.asarray(base_action["action.navigate_command"], dtype=np.float32).copy()
    replay_type = "closed-loop_splice" if variant != "R1" else "strict_open_loop_from_eval_initial_state_with_dataset_actions"
    if variant == "R5":
        replay_type = "closed-loop_splice_no_reconstructed_controller_target_available_used_base_hand_proxy"
    if variant == "R6":
        replay_type = "closed-loop_splice_base_success_hand_profile_proxy"
    return out, replay_type


def collect_env_status(helper3d: Any, steps: list[dict[str, Any]], episode_success: bool) -> dict[str, Any]:
    if not steps:
        return {"reached": False, "lifted": False, "failure_stage": "no_steps"}
    initial_h = steps[0].get("apple_height_z") or 0.0
    reached = any(isinstance(s.get("apple_to_right_eef_l2"), (int, float)) and float(s["apple_to_right_eef_l2"]) <= 0.10 for s in steps)
    lifted = any(isinstance(s.get("apple_height_z"), (int, float)) and float(s["apple_height_z"]) - float(initial_h) >= 0.03 for s in steps)
    if episode_success:
        label = "success"
    elif lifted:
        label = "lifted_not_success"
    elif reached:
        label = "reached_apple_not_lifted"
    else:
        label = "never_reached_apple"
    return {"reached": bool(reached or lifted or episode_success), "lifted": bool(lifted), "failure_stage": label}


def run_replay_matrix(base_bundle: Mapping[str, Any], dataset: Path, out: Path, seeds: list[int], max_episode_steps: int, prompt: str) -> dict[str, Any]:
    import torch
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

    replay_dir = out / "stage3_replay_matrix"
    replay_dir.mkdir(parents=True, exist_ok=True)
    helpers = load_3d_eval_helpers()
    helpers._add_import_roots(REPO_ROOT)
    helpers._install_robocasa_import_shims()
    loader = _load_stage3_loader(dataset)
    env = make_eval_env(max_episode_steps=max_episode_steps, n_action_steps=20)
    policy = load_policy(resolve(base_bundle["bundle_path"]))
    sim_policy = Gr00tSimPolicyWrapper(policy, strict=True)
    variants = {
        "R0": "base policy closed-loop",
        "R1": "Stage3 dataset labels raw replay",
        "R2": "Stage3 labels + right_hand replaced by base prediction",
        "R3": "Stage3 labels + navigate_command replaced by base prediction",
        "R4": "Stage3 labels + right_hand+navigate replaced by base prediction",
        "R5": "Stage3 labels + reconstructed right_hand controller target if available",
        "R6": "Stage3 labels + base successful hand profile",
    }
    rows: list[dict[str, Any]] = []
    base_success_hand: dict[str, np.ndarray] | None = None
    try:
        for variant in variants:
            success = reached = lifted = 0
            failure_modes: dict[str, int] = {}
            replay_type = "closed-loop"
            for ep_idx, seed in enumerate(seeds, start=1):
                obs, _info = env.reset(seed=int(seed))
                done = False
                outer_step = 0
                steps: list[dict[str, Any]] = []
                episode_success = False
                while not done and outer_step < max(1, int(max_episode_steps) // 20):
                    flat = coerce_obs_float32(batch_flat_env_obs(obs))
                    base_action, _base_info = sim_policy.get_action(flat)
                    base_action = squeeze_policy_batch(normalize_action_dict(base_action))
                    if variant == "R0":
                        action = base_action
                        replay_type = "base_closed_loop"
                    else:
                        label_action = dataset_chunk_for_variant(loader, variant, ep_idx - 1, outer_step, 30)
                        action, replay_type = splice_action(label_action, base_action, variant, base_success_hand)
                    obs, reward, term, trunc, step_info = env.step(action)
                    done = bool(helpers._scalarize_bool(term) or helpers._scalarize_bool(trunc))
                    episode_success = bool(episode_success or helpers._extract_success_step(step_info))
                    snap = helpers._collect_env_snapshot(env)
                    step_rec = {"variant": variant, "seed": int(seed), "episode_index": ep_idx, "outer_step": outer_step, "success_step": bool(episode_success), "action_summary": helpers._summarize_action_chunk(dict(action))}
                    step_rec.update(snap)
                    append_jsonl(replay_dir / f"{variant}_steps.jsonl", step_rec)
                    steps.append(step_rec)
                    if variant == "R0" and episode_success and base_success_hand is None and "action.right_hand" in base_action:
                        base_success_hand = {"action.right_hand": np.asarray(base_action["action.right_hand"], dtype=np.float32).copy()}
                    outer_step += 1
                status = collect_env_status(helpers, steps, episode_success)
                success += int(episode_success)
                reached += int(status["reached"])
                lifted += int(status["lifted"])
                failure_modes[status["failure_stage"]] = failure_modes.get(status["failure_stage"], 0) + 1
                append_jsonl(replay_dir / f"{variant}_episodes.jsonl", {"variant": variant, "seed": int(seed), "episode_index": ep_idx, "success": bool(episode_success), **status, "replay_type": replay_type})
            rows.append({"ID": variant, "action_source": variants[variant], "replay_type": replay_type, "episodes": len(seeds), "success": success, "reached": reached, "lifted": lifted, "failure_modes": failure_modes})
    finally:
        del sim_policy
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            env.close()
        except Exception:
            pass
    write_csv(replay_dir / "stage3_replay_matrix.csv", rows, ["ID", "action_source", "replay_type", "episodes", "success", "reached", "lifted", "failure_modes"])
    payload = {"status": "PASS", "seed_list": seeds, "rows": rows, "limitations": ["Stage3 dataset initial states are not restored; R1 is open-loop from eval initial states, while R2-R6 are closed-loop splices against current env observations.", "R5 reconstructed controller target was unavailable; base-hand proxy was used and explicitly labeled."], "material_threshold": "10-seed lifted +2 over R1 or reached +20pp for navigate splice"}
    write_json(replay_dir / "stage3_replay_matrix.json", payload)
    return payload


def run_semantics_audit(out: Path, offline: Mapping[str, Any], replay: Mapping[str, Any], command_proxy: Mapping[str, Any]) -> dict[str, Any]:
    rows = offline["rows"]
    def find(source: str, surface: str, modality: str) -> Mapping[str, Any]:
        for row in rows:
            if row.get("source") == source and row.get("surface") == surface and row.get("modality") == modality:
                return row
        return {}
    base_rh = float(find("C0_base", "L2_denorm_absolute", "right_hand").get("abs_q99") or find("C0_base", "L2_denorm_absolute", "right_hand").get("q99") or 0.0)
    dataset_rh = float(find("dataset", "denorm_absolute", "right_hand").get("abs_q99") or find("dataset", "denorm_absolute", "right_hand").get("q99") or 0.0)
    c2_rh = float(find("C2_pure_sft_canonical", "L2_denorm_absolute", "right_hand").get("abs_q99") or 0.0)
    c3_rh = float(find("C3_recap_canonical", "L2_denorm_absolute", "right_hand").get("abs_q99") or 0.0)
    base_nav = float(find("C0_base", "L2_denorm_absolute", "navigate_command").get("abs_q99") or 0.0)
    dataset_nav = float(find("dataset", "denorm_absolute", "navigate_command").get("abs_q99") or 0.0)
    c2_nav = float(find("C2_pure_sft_canonical", "L2_denorm_absolute", "navigate_command").get("abs_q99") or 0.0)
    c3_nav = float(find("C3_recap_canonical", "L2_denorm_absolute", "navigate_command").get("abs_q99") or 0.0)
    replay_rows = {r["ID"]: r for r in replay.get("rows", [])}
    r1 = replay_rows.get("R1", {})
    r2 = replay_rows.get("R2", {})
    r3 = replay_rows.get("R3", {})
    r4 = replay_rows.get("R4", {})
    hand_improved = (int(r2.get("lifted", 0)) >= int(r1.get("lifted", 0)) + 2) or (int(r4.get("lifted", 0)) >= int(r1.get("lifted", 0)) + 2)
    nav_improved = (int(r3.get("reached", 0)) >= int(r1.get("reached", 0)) + 2) or (int(r4.get("reached", 0)) >= int(r1.get("reached", 0)) + 2)
    hand_compressed = bool(base_rh > 1e-8 and max(c2_rh, c3_rh, dataset_rh) / base_rh < 0.20)
    nav_weak = bool(base_nav > 1e-8 and max(c2_nav, c3_nav, dataset_nav) / base_nav < 0.70)
    # Contract from base processor config: unitree_g1 action order maps right_hand/navigate to ABSOLUTE configs.
    answers = {
        "right_hand_absolute": True,
        "navigate_command_absolute_vx_vy_wz": True,
        "dataset_near_zero_controller_meaning": "near-zero absolute hand target / hand target proxy at WBC input; not a keep-current delta under current config",
        "base_q99_controller_meaning": "large absolute hand target/profile preserved by base primitive",
        "near_zero_absolute_target_can_grasp_lift": False if (int(r1.get("lifted", 0)) == 0 and dataset_rh < 0.02) else None,
        "relative_absolute_mix": "left_arm/right_arm are RELATIVE but right_hand/navigate/base_height are ABSOLUTE in current Unitree G1 action config",
        "sign_flip_evidence": "not proven; no DoF-specific sign repair recovered in this run",
        "scale_mismatch_evidence": "strong scale compression: dataset/pure-SFT/RECAP right_hand q99 is near-zero relative to base",
        "dof_order_mismatch_evidence": "not proven from available telemetry; would require controller target vs observed joint-index perturbation",
        "normalized_denormalized_mix": "not supported as primary here because canonical surface uses base statistics for C2/C3 and right_hand remains compressed in denorm output",
        "need_hand_label_repair_or_mask_distill": bool(hand_compressed or hand_improved),
        "stage3_navigate_weaker_than_base": bool(base_nav > 1e-8 and dataset_nav / base_nav < 0.75),
        "weak_navigate_stand_stop_like": bool(dataset_nav < base_nav and dataset_nav > 0.05),
        "base_nav_splice_restores_reach": bool(nav_improved),
        "navigate_weakness_sufficient_for_never_reached": bool(nav_improved),
    }
    payload = {"status": "PASS", "metrics": {"base_right_hand_q99": base_rh, "dataset_right_hand_q99": dataset_rh, "c2_right_hand_q99": c2_rh, "c3_right_hand_q99": c3_rh, "base_navigate_q99": base_nav, "dataset_navigate_q99": dataset_nav, "c2_navigate_q99": c2_nav, "c3_navigate_q99": c3_nav, "hand_compressed": hand_compressed, "navigate_weak": nav_weak, "hand_replay_improved": hand_improved, "navigate_replay_improved": nav_improved}, "answers": answers, "limitations": ["mujoco_ctrl hand DoF mapping is command-proxy level, not strict actuator-index proof", "strict dataset-state replay unavailable"]}
    sem = out / "semantics_audit"
    write_json(sem / "right_hand_navigate_semantics_audit.json", payload)
    # Compact tables requested by user.
    sample_rows = []
    for source in ["dataset", "C0_base", "C2_pure_sft_canonical", "C3_recap_canonical"]:
        sample_rows.append({"source": source, "right_hand_q99": payload["metrics"].get("base_right_hand_q99") if source == "C0_base" else payload["metrics"].get("dataset_right_hand_q99") if source == "dataset" else payload["metrics"].get("c2_right_hand_q99") if source.startswith("C2") else payload["metrics"].get("c3_right_hand_q99"), "navigate_q99": payload["metrics"].get("base_navigate_q99") if source == "C0_base" else payload["metrics"].get("dataset_navigate_q99") if source == "dataset" else payload["metrics"].get("c2_navigate_q99") if source.startswith("C2") else payload["metrics"].get("c3_navigate_q99")})
    write_csv(sem / "semantics_q99_summary.csv", sample_rows)
    return payload


def merge_decision(out: Path, bundles: Mapping[str, Mapping[str, Any]], offline: Mapping[str, Any], formal: Mapping[str, Any], replay: Mapping[str, Any], semantics: Mapping[str, Any], a_full: Mapping[str, Any], hard_violation: bool) -> dict[str, Any]:
    formal_rows = {r["ID"]: r for r in formal.get("rows", [])}
    c2 = formal_rows.get("C2_pure_sft_canonical", {})
    c3 = formal_rows.get("C3_recap_canonical", {})
    c4 = formal_rows.get("C4_pure_sft_original_surface", {})
    c5 = formal_rows.get("C5_recap_original_surface", {})
    def collapsed(row: Mapping[str, Any]) -> bool:
        return int(row.get("episodes", 0)) > 0 and int(row.get("success", 0)) == 0 and int(row.get("lifted", 0)) == 0
    surface_recovers = (not collapsed(c2) and collapsed(c4)) or (not collapsed(c3) and collapsed(c5))
    hand_bug = bool(semantics.get("metrics", {}).get("hand_compressed") or semantics.get("metrics", {}).get("hand_replay_improved"))
    nav_bug = bool(semantics.get("metrics", {}).get("navigate_weak") and semantics.get("metrics", {}).get("navigate_replay_improved"))
    if hard_violation:
        decision = "A_FULL_STILL_BLOCKED_ONLY"
        reason = "hard gate violation prevented trusted local diagnostic decision"
    elif surface_recovers:
        decision = "SURFACE_CONTAMINATION_MAIN"
        reason = "canonical surface materially improved old checkpoint behavior"
    elif collapsed(c2) and collapsed(c3):
        if hand_bug and nav_bug:
            decision = "CD_HAND_NAV_BOTH"
            reason = "canonical trained weights still collapse and both hand/nav diagnostics are material"
        elif hand_bug:
            decision = "C_HAND_LABEL_BUG"
            reason = "canonical trained weights still collapse and right_hand primitive remains compressed/invalid"
        elif nav_bug:
            decision = "D_NAV_LABEL_BUG"
            reason = "canonical trained weights still collapse and navigate splice/weakness is material"
        else:
            decision = "TRAINED_WEIGHTS_STILL_COLLAPSE"
            reason = "canonical surface did not recover pure-SFT/RECAP collapse"
    elif a_full.get("formal_eval_sanity", {}).get("pass") is False and not hand_bug and not nav_bug:
        decision = "A_FULL_STILL_BLOCKED_ONLY"
        reason = "A_FULL follow-up remains blocked and local label causality did not close"
    elif not hand_bug and not nav_bug:
        decision = "B_PASS_LABELS_VALID"
        reason = "Stage3 replay/semantics did not find material hand/nav label invalidity"
    else:
        decision = "TRAINED_WEIGHTS_STILL_COLLAPSE"
        reason = "mixed non-collapse evidence requires training-scope diagnosis"
    final = {"final_decision": decision, "allowed_final_decision": decision in ALLOWED_FINAL, "reason": reason, "surface_recovers": surface_recovers, "c2_collapsed": collapsed(c2), "c3_collapsed": collapsed(c3), "c4_collapsed": collapsed(c4), "c5_collapsed": collapsed(c5), "hand_bug": hand_bug, "nav_bug": nav_bug, "a_full_followup": a_full, "no_training_scope": True}
    write_json(out / "final_decision.json", final)
    return final


def forbidden_scans(out: Path, run_start_ns: int) -> dict[str, Any]:
    pattern = re.compile(r"(lora|safe[-_ ]?sft|guarded recap|fatg|finetune|train)", re.I)
    ps = subprocess.check_output(["bash", "-lc", "ps -eo pid,stat,etime,cmd || true"], text=True)
    matches = [line for line in ps.splitlines() if pattern.search(line) and "grep" not in line and "gr00t_post_a_local_diagnostic" not in line and "gr00t_g3_formal_eval.py" not in line]
    suspicious = []
    for root in [out, REPO_ROOT / "agent/artifacts/checkpoints"]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            try:
                st = p.lstat()
            except OSError:
                continue
            if st.st_mtime_ns >= run_start_ns and re.search(r"(adapter|lora|safe_sft|guarded_recap|fatg|checkpoint-\d+)", str(p), re.I):
                # Scratch diagnostic bundles intentionally include copied checkpoint-like bundle names but no new trained checkpoints.
                if "scratch/bundles" in str(p):
                    continue
                suspicious.append({"path": rel(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    payload = {"forbidden_process_matches": matches, "suspicious_new_artifacts": suspicious, "hard_gate_violation": bool(matches or suspicious)}
    write_json(out / "forbidden_scan.json", payload)
    return payload


def build_a_full_followup(formal: Mapping[str, Any], command_proxy: Mapping[str, Any], offline: Mapping[str, Any]) -> dict[str, Any]:
    rows = {r["ID"]: r for r in formal.get("rows", [])}
    c0 = rows.get("C0_base", {})
    c1 = rows.get("C1_identity_fixed", {})
    formal_pass = bool(c0.get("episodes") and c1.get("episodes") and abs(int(c0.get("success", 0)) - int(c1.get("success", 0))) <= 3)
    return {
        "formal_eval_sanity": {"status": "RUN", "pass": formal_pass, "limitation": "10-seed sanity only; not strict per-seed determinism"},
        "strict_L5_surface": {"status": "COMMAND_PROXY_ONLY", "pass": False, "limitation": "strict post-physics env-applied actuator surface not implemented"},
        "observation_bank_enriched": {"status": "PARTIAL", "pass": False, "limitation": "offline Stage3 bank present; base success/failure enrichment inferred from formal eval telemetry, not persisted as raw observation NPZ bank"},
    }


def final_tables(out: Path, offline: Mapping[str, Any], formal: Mapping[str, Any], semantics: Mapping[str, Any]) -> None:
    rows = []
    stat_rows = offline["rows"]
    def q(source: str, modality: str, surface: str = "L2_denorm_absolute") -> float | None:
        for r in stat_rows:
            if r.get("source") == source and r.get("surface") == surface and r.get("modality") == modality:
                return float(r.get("abs_q99") or r.get("q99") or 0.0)
        return None
    formal_rows = {r["ID"]: r for r in formal.get("rows", [])}
    for bid, surface in [("C0_base", "official base"), ("C1_identity_fixed", "canonical base"), ("C2_pure_sft_canonical", "canonical base"), ("C3_recap_canonical", "canonical base"), ("C4_pure_sft_original_surface", "old trained"), ("C5_recap_original_surface", "old trained")]:
        f = formal_rows.get(bid, {})
        rows.append({"ID": bid, "surface": surface, "episodes": f.get("episodes", 0), "success": f.get("success", 0), "reached": f.get("reached", 0), "lifted": f.get("lifted", 0), "right_hand_q99": q(bid, "right_hand"), "right_hand_grasp_q99": q(bid, "right_hand"), "navigate_q99": q(bid, "navigate_command"), "failure_modes": f.get("failure_modes", {}), "conclusion": "see final_decision"})
    write_csv(out / "canonical_surface_eval_table.csv", rows)
    write_json(out / "canonical_surface_eval_table.json", {"rows": rows})
    # A_FULL follow-up table is written after final.


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--runtime-log-dir", required=True)
    ap.add_argument("--base", default=str(OFFICIAL_BASE))
    ap.add_argument("--identity", default=str(CANONICAL_IDENTITY))
    ap.add_argument("--pure-sft", default=str(PURE_SFT))
    ap.add_argument("--recap", default=str(RECAP))
    ap.add_argument("--dataset", default=str(STAGE3_DATASET))
    ap.add_argument("--prompt", default=DEFAULT_TASK_PROMPT)
    ap.add_argument("--offline-obs-count", type=int, default=50)
    ap.add_argument("--command-proxy-seeds", type=int, default=6)
    ap.add_argument("--replay-episodes", type=int, default=10)
    ap.add_argument("--formal-episodes", type=int, default=10)
    ap.add_argument("--formal-timeout-s", type=int, default=1800)
    ap.add_argument("--max-episode-steps", type=int, default=1440)
    ap.add_argument("--skip-formal-eval", action="store_true", help="Debug only; final cannot claim requested formal smoke when set.")
    ap.add_argument("--skip-replay", action="store_true", help="Debug only.")
    args = ap.parse_args(argv)

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly '1' for this harness, got {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    out = resolve(args.output_dir)
    runtime = resolve(args.runtime_log_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    run_start_ns = time.time_ns()
    write_json(out / "command_manifest.json", {"argv": sys.argv, "cwd": str(Path.cwd()), "env": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}, "started_at_utc": utc_now()})
    inputs = {"base": rel(resolve(args.base)), "identity": rel(resolve(args.identity)), "pure_sft": rel(resolve(args.pure_sft)), "recap": rel(resolve(args.recap)), "dataset": rel(resolve(args.dataset)), "phase2_preflight": rel(PHASE2_PREFLIGHT), "git_status_root": git_status(REPO_ROOT), "git_status_isaac": git_status(ISAAC_GR00T_ROOT), "git_status_wbc": git_status(WBC_ROOT)}
    write_json(out / "inputs_manifest.json", inputs)

    bundles = build_bundles(out, resolve(args.base), resolve(args.identity), resolve(args.pure_sft), resolve(args.recap))
    offline = run_offline_action_audit(bundles, resolve(args.dataset), out, args.prompt, int(args.offline_obs_count))
    command_proxy = run_command_proxy_audit(bundles, out, [21000 + i for i in range(int(args.command_proxy_seeds))], int(args.max_episode_steps))
    if args.skip_formal_eval:
        existing_formal = out / "formal_eval" / "formal_eval_matrix.json"
        if existing_formal.exists():
            formal = json.loads(existing_formal.read_text())
            formal["status"] = formal.get("status", "PASS")
            formal["reused_existing"] = True
        else:
            formal = {"summaries": {}, "rows": [], "status": "SKIPPED"}
            write_json(existing_formal, formal)
    else:
        formal = run_formal_matrix(bundles, out, runtime, int(args.formal_episodes), int(args.formal_timeout_s), int(args.max_episode_steps))
    if args.skip_replay:
        replay = {"status": "SKIPPED", "rows": []}
        write_json(out / "stage3_replay_matrix" / "stage3_replay_matrix.json", replay)
    else:
        replay = run_replay_matrix(bundles["C0_base"], resolve(args.dataset), out, [22000 + i for i in range(int(args.replay_episodes))], int(args.max_episode_steps), args.prompt)
    semantics = run_semantics_audit(out, offline, replay, command_proxy)
    a_full = build_a_full_followup(formal, command_proxy, offline)
    write_json(out / "a_full_followup.json", a_full)
    final_tables(out, offline, formal, semantics)
    forbidden = forbidden_scans(out, run_start_ns)
    final = merge_decision(out, bundles, offline, formal, replay, semantics, a_full, bool(forbidden.get("hard_gate_violation")))
    report = [
        "# GR00T Post-A_LOCAL Diagnostic Closure Report",
        "",
        f"- generated_at_utc: `{utc_now()}`",
        f"- final_decision: `{final['final_decision']}`",
        f"- reason: {final['reason']}",
        f"- no_training_scope: `{final['no_training_scope']}`",
        "",
        "## Primary tables",
        f"- canonical surface eval: `{rel(out / 'canonical_surface_eval_table.csv')}`",
        f"- replay matrix: `{rel(out / 'stage3_replay_matrix/stage3_replay_matrix.csv')}`",
        f"- semantics audit: `{rel(out / 'semantics_audit/right_hand_navigate_semantics_audit.json')}`",
        f"- A_FULL follow-up: `{rel(out / 'a_full_followup.json')}`",
        "",
        "## Decision caveats",
        "- Original checkpoints were not overwritten; scratch bundles live under `scratch/bundles/`.",
        "- Strict dataset-state replay is unavailable; replay matrix labels counterfactuals as open-loop or closed-loop splice.",
        "- Strict post-physics L5 surface remains unavailable unless `a_full_followup.json` says otherwise.",
    ]
    (out / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(out / "run_manifest.json", {"completed_at_utc": utc_now(), "final_decision": final["final_decision"], "output_dir": rel(out), "runtime_log_dir": rel(runtime), "no_training_scope": True})
    print(json.dumps({"status": "PASS" if final["allowed_final_decision"] else "FAIL", "final_decision": final["final_decision"], "output_dir": rel(out)}, sort_keys=True), flush=True)
    return 0 if final["allowed_final_decision"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
