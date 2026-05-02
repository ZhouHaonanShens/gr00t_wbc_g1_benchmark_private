#!/usr/bin/env python3
"""Stage B P0 ladder runner: seed replay, indicator injection, n_envs, JSON summaries."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import math
import os
import socket
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from work.recap.scripts import gr00t_g3_formal_eval as g3  # noqa: E402
STAGE_B_DIR = Path("agent/artifacts/stage_B_controller_seam_20260501T045341Z_precheck_gate")
P0_REL = Path("prechecks/P0_eval_protocol_determinism")
STAGE_A_DIR = Path("agent/artifacts/stage_A_baseline_freeze_20260501T014232Z")
BASE_MODEL = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
POST_CKPT = "/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_live/agent/artifacts/gr00t_recap_live/single_gpu_v2_full_update/stage1_gr00t_r2r4_closed_candidate_iter9_20260426T_nextZ/gr00t/g3_conditioned_continuation_6600_after_surfacefix_20260430_181210/checkpoint-6600"
ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
CHECKLIST_16 = tuple("json:baseline_manifest_v1.json|json:internal_g3_checkpoint_6600/artifact_index.json|json:internal_g3_checkpoint_6600/env_lock.json|json:internal_g3_checkpoint_6600/git_provenance.json|json:public_gr00t_g1_baseline/public_repo_lock.json|json:public_gr00t_g1_baseline/public_dataset_lock.json|json:public_gr00t_g1_baseline/public_reproduction_run_summary.json|json:public_gr00t_g1_baseline/level0_server_smoke_summary.json|nonempty:baseline_manifest_v1.md|nonempty:pre_registration_v1.md|nonempty:pre_registration_seed_table_v1.csv|nonempty:final_gate_decision.md|nonempty:openpi_auxiliary_evidence/openpi_not_primary_baseline_note.md|nonempty:openpi_auxiliary_evidence/openpi_carrier_summary.md|nonempty:public_gr00t_g1_baseline/worker2_a4_a5_verification.log|nonempty:logs/worker3_a6_a7_verification_summary.md".split("|"))
def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
def rel(path: Path) -> str:
    return g3._safe_relpath(path) or str(path)
def write_json(path: Path, payload: dict[str, Any]) -> None:
    g3._write_json(path, payload)
def load_seeds(stage_a_dir: Path, count: int) -> list[int]:
    with (stage_a_dir / "pre_registration_seed_table_v1.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seeds = [int(r["seed_value"]) for r in rows if str(r.get("formal_30", "")).lower() == "true"]
    if len(seeds) < count:
        raise ValueError(f"seed table has {len(seeds)} formal_30 seeds, need {count}")
    return seeds[:count]
def verify_stage_a_checklist(stage_a_dir: Path = REPO_ROOT / STAGE_A_DIR) -> dict[str, Any]:
    checks = []
    for item in CHECKLIST_16:
        kind, raw = item.split(":", 1)
        path = stage_a_dir / raw
        ok = path.is_file() and path.stat().st_size > 0
        if ok and kind == "json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                ok = False
        checks.append({"item": item, "path": rel(path), "status": "PASS" if ok else "FAIL"})
    return {"checklist": list(CHECKLIST_16), "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL", "checks": checks}
def expand_level0_vram_cells() -> list[dict[str, Any]]:
    return [{"cell_id": f"level0_post_recap_gpu{g}_nenvs{n}", "checkpoint_role": "post_recap", "gpu": g, "n_envs": n, "seed": 20000, "episode_count": 1} for g in (1, 2) for n in (1, 5, 30)]
def should_run_nenvs50(results: dict[str, float]) -> bool:
    return "5" in results and "30" in results and float(results["5"]) != float(results["30"])
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
def gpu_mem(gpu: int) -> int | None:
    cmd = ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    try:
        return int(subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10).splitlines()[0].strip())
    except Exception:
        return None
def count_bad_numbers(value: Any) -> tuple[int, int]:
    import numpy as np
    if isinstance(value, dict):
        pairs = [count_bad_numbers(v) for v in value.values()]
        return sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    arr = np.asarray(value) if not isinstance(value, str) else np.asarray([])
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return 0, 0
    return int(np.isnan(arr).sum()), int(np.isinf(arr).sum())
def to_plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return to_plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return to_plain(value.item())
    except Exception:
        return str(value)
def modality_summary(modality_cfg: Any) -> dict[str, Any]:
    return {str(n): {a: to_plain(getattr(c, a)) for a in ("delta_indices", "modality_keys") if hasattr(c, a)} for n, c in modality_cfg.items()} if isinstance(modality_cfg, dict) else {}
def infer_num_diffusion_steps(model_path: str) -> dict[str, Any]:
    root = Path(model_path)
    roots = [root] if root.exists() else []
    hub = Path.home() / ".cache/huggingface/hub" / f"models--{model_path.replace('/', '--')}" / "snapshots"
    if hub.exists():
        roots.extend(sorted(p for p in hub.iterdir() if p.is_dir()))
    for root in roots:
        for cfg in [root / "config.json", *root.glob("*/config.json")]:
            if cfg.is_file():
                data = json.loads(cfg.read_text(encoding="utf-8"))
                for key in ("num_diffusion_steps", "num_inference_timesteps", "diffusion_steps", "num_inference_steps"):
                    if key in data:
                        return {"value": to_plain(data[key]), "source_key": key, "source_path": rel(cfg)}
    return {"value": "UNKNOWN", "source_key": "NOT_FOUND", "source_path": None}
def _truthy(value: Any) -> bool:
    import numpy as np
    return bool(np.any(value)) if isinstance(value, (list, tuple, np.ndarray)) else bool(value)
def get_success(info: Any, idx: int) -> bool:
    try:
        if isinstance(info, dict) and "success" in info:
            return _truthy(info["success"][idx])
        final = info.get("final_info", [None])[idx] if isinstance(info, dict) else None
        return _truthy(final.get("success", False)) if final is not None else False
    except Exception:
        return False
def make_env(n_envs: int, modality_cfg: Any, max_episode_steps: int, n_action_steps: int) -> tuple[Any, dict[str, Any]]:
    gym = __import__("gymnasium")
    rollout = __import__("gr00t.eval.rollout_policy", fromlist=[""])
    helper = g3._load_helper3d()
    reg = helper._ensure_explicit_g1_env_registration(gym)
    res = helper.state_conditioned_env_resolution.resolve_apple_to_plate_g1_env_name(gym, requested_env_name=ENV_NAME)
    resolved = str(res["resolved_env_name"])
    wrapper = rollout.WrapperConfigs(video=rollout.VideoConfig(video_dir=None, max_episode_steps=max_episode_steps, overlay_text=False), multistep=rollout.MultiStepConfig(n_action_steps=n_action_steps, max_episode_steps=max_episode_steps, terminate_on_success=True))
    env_fns = [partial(rollout.create_eval_env, env_name=resolved, env_idx=i, total_n_envs=n_envs, wrapper_configs=wrapper) for i in range(n_envs)]
    env = gym.vector.SyncVectorEnv(env_fns) if n_envs == 1 else gym.vector.AsyncVectorEnv(env_fns, shared_memory=False, context="spawn")
    return env, {"requested_env_name": ENV_NAME, "resolved_env_name": resolved, "env_registration": g3._jsonable(reg), "env_resolution": g3._jsonable(res), "server_action_horizon": g3._infer_action_horizon(modality_cfg), "n_action_steps": n_action_steps, "max_episode_steps": max_episode_steps, "outer_max_steps": int(math.ceil(max_episode_steps / n_action_steps))}
def mujoco_params(env: Any) -> dict[str, Any]:
    stack, seen = [env] + list(getattr(env, "envs", []) or []), set()
    while stack and len(seen) < 200:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        for owner in (obj, getattr(obj, "model", None), getattr(getattr(obj, "sim", None), "model", None)):
            opt = getattr(owner, "opt", None)
            if opt is not None:
                return {"timestep": to_plain(getattr(opt, "timestep", "UNKNOWN")), "iterations": to_plain(getattr(opt, "iterations", "UNKNOWN")), "solver": to_plain(getattr(opt, "solver", "UNKNOWN"))}
        stack.extend(c for a in ("env", "unwrapped", "base_env", "_env", "sim", "model", "wrapped_env") if (c := getattr(obj, a, None)) is not None and c is not obj)
    return {"timestep": "UNKNOWN", "iterations": "UNKNOWN", "solver": "UNKNOWN"}
def run_rollout_cell(cell: dict[str, Any], client: Any, modality_cfg: Any, out_dir: Path) -> dict[str, Any]:
    import numpy as np
    n_envs, seed = int(cell["n_envs"]), int(cell["seed"])
    peak, started, env = gpu_mem(int(cell["gpu"])) or 0, time.monotonic(), None
    nan = inf = completed = success = steps = 0
    records, mod_summary = [], modality_summary(modality_cfg)
    try:
        env, meta = make_env(n_envs, modality_cfg, 1440, 20)
        obs, _ = env.reset(seed=[seed + i for i in range(n_envs)])
        options = {"seed": seed, "indicator_mode": "positive"}
        client.reset(options=options)
        cur_success = [False] * n_envs
        for _ in range(int(meta["outer_max_steps"])):
            peak = max(peak, gpu_mem(int(cell["gpu"])) or 0)
            action, _ = client.get_action(obs, options=options)
            a, b = count_bad_numbers(action)
            nan += a; inf += b
            obs, _reward, term, trunc, env_info = env.step(action)
            steps += 1
            for i in range(n_envs):
                cur_success[i] = cur_success[i] or get_success(env_info, i)
                if bool(np.asarray(term)[i]) or bool(np.asarray(trunc)[i]):
                    completed += 1; success += int(cur_success[i])
                    records.append({"env_idx": i, "success": bool(cur_success[i]), "outer_steps": steps})
                    cur_success[i] = False
            if completed >= int(cell["episode_count"]):
                break
        status = "PASS" if completed >= int(cell["episode_count"]) and nan == 0 and inf == 0 else "FAIL"
        result = {"status": status, "error": None, "env_metadata": meta, "mujoco_params": mujoco_params(env), "modality_summary": mod_summary}
    except Exception as exc:
        result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "env_metadata": {}, "mujoco_params": {"timestep": "UNKNOWN", "iterations": "UNKNOWN", "solver": "UNKNOWN"}, "modality_summary": mod_summary}
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
    result.update(cell, requested_episode_count=int(cell["episode_count"]), completed_episode_count=completed, success_count=success, success_rate=(success / completed if completed else 0.0), seed_table_replay={"base_seed": seed, "env_reset_seeds": [seed + i for i in range(n_envs)], "policy_options_seed": seed, "indicator_mode": "positive"}, peak_vram_mib=peak, wall_clock_s=round(time.monotonic() - started, 3), egl_ok=(os.environ.get("MUJOCO_GL") == "egl"), mujoco_crash=("mujoco" in str(result.get("error", "")).lower()), nan_count=nan, inf_count=inf, episode_records=records)
    write_json(out_dir / f"{cell['cell_id']}.json", result)
    return result
def launch_group(cells: list[dict[str, Any]], checkpoint: str, stage_b_dir: Path, timeout_s: int, server_timeout_s: int) -> list[dict[str, Any]]:
    gpu = int(cells[0]["gpu"])
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    run_dir = stage_b_dir / P0_REL / "runtime_cells" / f"{cells[0]['checkpoint_role']}_gpu{gpu}"
    port, proc, handle, started, results = free_port(), None, None, time.monotonic(), []
    try:
        g3._install_alarm_timeout(timeout_s)
        proc, handle, server_log, cmd = g3._launch_policy_server(checkpoint=Path(checkpoint), runtime_log_dir=run_dir, host="127.0.0.1", port=port)
        client, modality_cfg, ping, meta = g3._wait_for_policy_client(host="127.0.0.1", port=port, proc=proc, timeout_s=server_timeout_s)
        for cell in cells:
            cell.update(server_log=rel(server_log), server_command=cmd, server_ping=g3._jsonable(ping), server_metadata=g3._jsonable(meta))
            results.append(run_rollout_cell(cell, client, modality_cfg, run_dir))
    except Exception as exc:
        for cell in cells:
            failed = dict(cell, status="FAIL", error=f"{type(exc).__name__}: {exc}", peak_vram_mib=gpu_mem(gpu), wall_clock_s=round(time.monotonic() - started, 3), egl_ok=False, mujoco_crash=True)
            write_json(run_dir / f"{cell['cell_id']}.json", failed)
            results.append(failed)
    finally:
        g3._clear_alarm_timeout(); g3._stop_policy_server(proc)
        if handle:
            handle.close()
    return results
def write_drift(p0_dir: Path, results: list[dict[str, Any]]) -> None:
    failed = [r for r in results if r.get("status") != "PASS"]
    write_json(p0_dir / "protocol_drift_inventory.json", {"schema_version": "p0_protocol_drift_inventory_v1", "updated_at_utc": utc(), "failed_or_degraded_cells": failed, "known_limitations": ["n_envs=50 skipped unless 5/30 non-monotonic", "official NVIDIA reproduction seed list NOT_FOUND", "vector policy options use scalar seed while env reset receives per-env seed list"]})
def run_vram(args: argparse.Namespace) -> None:
    stage_b, all_results = REPO_ROOT / args.stage_b_dir, []
    groups = {1: [], 2: []}
    for cell in expand_level0_vram_cells():
        groups[int(cell["gpu"])].append(cell)
    for gpu in (1, 2):
        all_results.extend(launch_group(groups[gpu], args.post_ckpt, stage_b, args.timeout_s, args.server_timeout_s))
    write_json(stage_b / P0_REL / "vram_smoke_summary.json", {"schema_version": "p0_level0_vram_smoke_v1", "created_at_utc": utc(), "checkpoint_role": "post_recap", "checkpoint": args.post_ckpt, "seed": 20000, "cells": all_results, "status": "PASS" if all(r.get("status") == "PASS" for r in all_results) else "PARTIAL"})
    write_drift(stage_b / P0_REL, all_results)
def run_base(args: argparse.Namespace) -> int:
    stage_b, p0 = REPO_ROOT / args.stage_b_dir, REPO_ROOT / args.stage_b_dir / P0_REL
    cell = {"cell_id": "base_gpu2_nenvs1_seed20000", "checkpoint_role": "base", "gpu": 2, "n_envs": 1, "seed": 20000, "episode_count": 1}
    result = launch_group([cell], args.base_model, stage_b, args.timeout_s, args.server_timeout_s)[0]
    write_json(p0 / "base_1ep_smoke_summary.json", result)
    write_json(p0 / "cell_runner_provenance.json", {"schema_version": "p0_cell_runner_provenance_v1", "created_at_utc": utc(), "stage_a_protocol_checklist_16": verify_stage_a_checklist(), "base_smoke_result": result, "delta_indices": result.get("modality_summary", {}).get("action", {}).get("delta_indices", "UNKNOWN"), "num_diffusion_steps": infer_num_diffusion_steps(args.base_model), "mujoco_params": result.get("mujoco_params", {})})
    if result.get("status") == "PASS":
        return 0
    (p0 / "base_rollout_blocker.md").write_text(f"# Base rollout blocker\n\nBase 1-episode smoke failed at {utc()}.\n\n- status: `{result.get('status')}`\n- error: `{result.get('error')}`\n- next_route: `wait_for_leader_decision`\n", encoding="utf-8")
    write_json(p0 / "stop_record.json", {"schema_version": "p0_stop_record_v1", "stop_reason_code": "P0_BASE_PHASE0_SMOKE_BLOCKED", "triggering_cell_id": cell["cell_id"], "stop_emit_utc": utc(), "next_route": "wait_for_leader_decision", "downstream_blocks": {"p2_allowed": False, "runtime_probe_allowed": False, "training_allowed": False, "checkpoint_update_allowed": False, "method_claim_allowed": False}, "result": result})
    return 2
def write_pending_gate(args: argparse.Namespace) -> None:
    path = REPO_ROOT / args.stage_b_dir / P0_REL / "p0_gate_decision.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(decision="P0_PENDING_EXEC", blocked_by=None, updated_at_utc=utc(), training_allowed=False, checkpoint_update_allowed=False, continue_to_p2=False, continue_to_runtime_probes=False, method_claim_allowed=False)
    write_json(path, payload)
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["write-pending-gate", "vram-smoke", "base-smoke", "phase0"])
    p.add_argument("--stage-b-dir", default=str(STAGE_B_DIR)); p.add_argument("--post-ckpt", default=POST_CKPT); p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--timeout-s", type=int, default=7200); p.add_argument("--server-timeout-s", type=int, default=900)
    args = p.parse_args(argv)
    if args.command == "write-pending-gate":
        write_pending_gate(args); return 0
    if args.command == "vram-smoke":
        run_vram(args); return 0
    if args.command == "base-smoke":
        return run_base(args)
    write_pending_gate(args); run_vram(args); return run_base(args)
if __name__ == "__main__":
    raise SystemExit(main())
