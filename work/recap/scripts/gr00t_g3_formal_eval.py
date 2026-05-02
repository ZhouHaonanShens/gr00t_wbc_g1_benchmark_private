#!/usr/bin/env python3
"""Run checkpoint-local GR00T G3 formal evaluation.

This runner is intentionally separate from the historical Iter3/P5 formal
runner because that artifact was scoped to Worker A/GPU1 and an older gate
layout.  The current lane needs a checkpoint-local, GPU-explicit formal eval
that preserves the same public protocol shape (10 seeds, 1440 max steps,
20 executed action steps) while passing the RECAP text-indicator mode through
``PolicyClient`` options.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5564
DEFAULT_SEED_BASE = 20000
DEFAULT_EPISODE_COUNT = 10
DEFAULT_MAX_EPISODE_STEPS = 1440
DEFAULT_N_ACTION_STEPS = 20
DEFAULT_CONNECT_TIMEOUT_S = 1200.0
DEFAULT_TOTAL_TIMEOUT_S = 10800.0
DEFAULT_REQUIRED_CUDA_VISIBLE_DEVICES = "0"
DEFAULT_PROMPT_RAW = "pick up the apple, walk left and place the apple on the plate."
DEFAULT_ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "agent/artifacts/gr00t_recap_live/single_gpu_v2_full_update"
    / "stage1_gr00t_r2r4_closed_candidate_iter9_20260426T_nextZ"
    / "gr00t/g3_6600_formal_eval"
)
DEFAULT_RUNTIME_ROOT = (
    REPO_ROOT
    / "agent/runtime_logs/stage1_gr00t_r2r4_closed_candidate_iter9_20260426T_nextZ"
    / "gr00t/g3_6600_formal_eval"
)


def _utc_now() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _safe_relpath(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = _resolve_path(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=True, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=True)
        return value
    except Exception:
        pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return {"type": type(value).__name__, "repr": repr(value)}


def _run_command(command: Sequence[str], *, timeout_s: float = 15.0) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            list(command),
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=float(timeout_s),
            check=False,
        )
        return {
            "command": list(command),
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "elapsed_seconds": float(time.monotonic() - started),
        }
    except Exception as exc:
        return {
            "command": list(command),
            "returncode": -1,
            "stdout": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": float(time.monotonic() - started),
        }


def _nvidia_smi_snapshot() -> dict[str, Any]:
    gpu_query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
    )
    compute_query = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    )
    return {
        "gpu_query": {
            "returncode": gpu_query["returncode"],
            "stdout": gpu_query["stdout"],
        },
        "compute_query": {
            "returncode": compute_query["returncode"],
            "stdout": compute_query["stdout"],
        },
    }


def _enforce_no_sudo() -> None:
    sudo_keys = [key for key in ("SUDO_UID", "SUDO_USER", "SUDO_COMMAND") if os.environ.get(key)]
    if sudo_keys:
        raise PermissionError(f"sudo/root execution is forbidden: {','.join(sudo_keys)}")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise PermissionError("sudo/root execution is forbidden: effective_uid_0")


def _install_alarm_timeout(timeout_s: float) -> None:
    timeout_int = int(float(timeout_s))
    if timeout_int <= 0 or not hasattr(signal, "SIGALRM"):
        return

    def _handler(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"formal eval timed out after {timeout_int}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_int)


def _clear_alarm_timeout() -> None:
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)


def _load_helper3d() -> Any:
    module_path = REPO_ROOT / "work/recap/scripts/3D_recap_eval.py"
    spec = importlib.util.spec_from_file_location("recap_3d_eval_for_g3_formal", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load rollout helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_indicator_modes(raw_modes: Sequence[str]) -> list[str]:
    from work.recap import policy as recap_policy

    modes = [
        recap_policy.validate_mainline_runtime_indicator_mode(
            str(mode),
            field_name="indicator_modes",
        )
        for mode in raw_modes
    ]
    if not modes:
        raise ValueError("at least one --indicator-modes value is required")
    duplicates = sorted({mode for mode in modes if modes.count(mode) > 1})
    if duplicates:
        raise ValueError(f"duplicate indicator modes are not allowed: {duplicates}")
    return modes


def _seed_list(seed_base: int, episode_count: int) -> list[int]:
    if int(episode_count) <= 0:
        raise ValueError("--episode-count must be > 0")
    return [int(seed_base) + offset for offset in range(int(episode_count))]


def _server_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["GR00T_SKIP_WBC_REEXEC"] = "1"
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["PYOPENGL_PLATFORM"] = env.get("PYOPENGL_PLATFORM", "egl")
    pythonpath_parts = [
        str(REPO_ROOT),
        str(REPO_ROOT / "submodules/Isaac-GR00T"),
        str(REPO_ROOT / "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl"),
        str(
            REPO_ROOT
            / "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl"
            / "gr00t_wbc/dexmg/gr00trobosuite"
        ),
        str(
            REPO_ROOT
            / "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl"
            / "gr00t_wbc/dexmg/gr00trobocasa"
        ),
        str(REPO_ROOT / "submodules/Isaac-GR00T/external_dependencies/robocasa"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(str(env["PYTHONPATH"]))
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def _launch_policy_server(
    *,
    checkpoint: Path,
    runtime_log_dir: Path,
    host: str,
    port: int,
) -> tuple[subprocess.Popen[str], TextIO, Path, list[str]]:
    server_script = REPO_ROOT / "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"
    if not server_script.is_file():
        raise FileNotFoundError(f"missing GR00T server entrypoint: {server_script}")
    server_log = runtime_log_dir / "g3_formal_server.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    handle = server_log.open("w", encoding="utf-8")
    cmd = [
        sys.executable,
        str(server_script),
        "--model-path",
        str(checkpoint),
        "--embodiment-tag",
        "UNITREE_G1",
        "--device",
        "cuda",
        "--use-sim-policy-wrapper",
        "--host",
        str(host),
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=_server_env(),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, handle, server_log, cmd


def _stop_policy_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=20)
        return
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=10)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _make_policy_client(host: str, port: int, timeout_ms: int) -> Any:
    sc_mod = __import__("gr00t.policy.server_client", fromlist=["PolicyClient"])
    PolicyClient = getattr(sc_mod, "PolicyClient")
    client = PolicyClient(host=host, port=int(port), strict=False)
    try:
        zmq = __import__("zmq")
        client.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        client.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        client.socket.setsockopt(zmq.LINGER, 0)
    except Exception:
        pass
    return client


def _wait_for_policy_client(
    *,
    host: str,
    port: int,
    proc: subprocess.Popen[str],
    timeout_s: float,
) -> tuple[Any, Any, Any, Any]:
    client = _make_policy_client(host, port, timeout_ms=1000)
    started = time.monotonic()
    last_reported = 0.0
    last_error: str | None = None
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"GR00T server exited early with rc={proc.returncode}")
        try:
            ping_payload = client.call_endpoint("ping", requires_input=False)
            modality_cfg = client.get_modality_config()
            try:
                server_info = client.call_endpoint("get_server_info", requires_input=False)
            except Exception as exc:
                server_info = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                provenance = client.call_endpoint("get_provenance", requires_input=False)
            except Exception as exc:
                provenance = {"error": f"{type(exc).__name__}: {exc}"}
            client = _make_policy_client(host, port, timeout_ms=int(timeout_s * 1000))
            return client, modality_cfg, ping_payload, {
                "server_info": _jsonable(server_info),
                "server_provenance": _jsonable(provenance),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            try:
                client.socket.close(0)
            except Exception:
                pass
            client = _make_policy_client(host, port, timeout_ms=1000)
            elapsed = time.monotonic() - started
            if elapsed > float(timeout_s):
                raise TimeoutError(
                    f"timed out waiting for GR00T policy server ping; last_error={last_error}"
                ) from exc
            if elapsed - last_reported >= 10.0:
                print(
                    "[SERVER_WAIT]",
                    f"elapsed_s={elapsed:.1f}",
                    f"host={host}",
                    f"port={port}",
                    f"last_error={last_error}",
                    flush=True,
                )
                last_reported = elapsed
            time.sleep(1.0)


def _infer_action_horizon(modality_cfg: Any) -> int | None:
    if not isinstance(modality_cfg, Mapping):
        return None
    action_cfg = modality_cfg.get("action")
    delta_indices = getattr(action_cfg, "delta_indices", None)
    if delta_indices is None:
        return None
    try:
        values = list(delta_indices)
    except TypeError:
        return None
    return len(values) if values else None


def _prepare_env(helper3d: Any, modality_cfg: Any, env_name: str, max_episode_steps: int, n_action_steps_arg: int) -> tuple[Any, dict[str, Any]]:
    gym = __import__("gymnasium")
    rollout_mod = __import__("gr00t.eval.rollout_policy", fromlist=[""])
    env_registration_info = helper3d._ensure_explicit_g1_env_registration(gym)
    env_resolution = helper3d.state_conditioned_env_resolution.resolve_apple_to_plate_g1_env_name(
        gym,
        requested_env_name=env_name,
    )
    resolved_env_name = str(env_resolution["resolved_env_name"])
    action_horizon = _infer_action_horizon(modality_cfg)
    if int(n_action_steps_arg) > 0:
        n_action_steps = int(n_action_steps_arg)
        n_action_steps_source = "cli_override"
    else:
        official_default = helper3d._default_n_action_steps_for_env(resolved_env_name)
        if official_default is not None:
            n_action_steps = int(official_default)
            n_action_steps_source = "g1_execution_surface_default"
        elif action_horizon is not None:
            n_action_steps = int(action_horizon)
            n_action_steps_source = "server_action_horizon_fallback"
        else:
            n_action_steps = DEFAULT_N_ACTION_STEPS
            n_action_steps_source = "generic_default_20"

    wrapper_configs = rollout_mod.WrapperConfigs(
        video=rollout_mod.VideoConfig(
            video_dir=None,
            max_episode_steps=int(max_episode_steps),
            overlay_text=False,
        ),
        multistep=rollout_mod.MultiStepConfig(
            n_action_steps=int(n_action_steps),
            max_episode_steps=int(max_episode_steps),
            terminate_on_success=True,
        ),
    )

    def env_fn() -> Any:
        return rollout_mod.create_eval_env(
            env_name=resolved_env_name,
            env_idx=0,
            total_n_envs=1,
            wrapper_configs=wrapper_configs,
        )

    env = gym.vector.SyncVectorEnv([env_fn])
    metadata = {
        "requested_env_name": str(env_name),
        "resolved_env_name": resolved_env_name,
        "env_resolution": _jsonable(env_resolution),
        "env_registration": _jsonable(env_registration_info),
        "server_action_horizon": action_horizon,
        "n_action_steps": int(n_action_steps),
        "n_action_steps_source": str(n_action_steps_source),
        "max_episode_steps": int(max_episode_steps),
        "outer_max_steps_per_episode": int(
            max(1, (int(max_episode_steps) + int(n_action_steps) - 1) // int(n_action_steps))
        ),
    }
    return env, metadata


def _run_mode(
    *,
    helper3d: Any,
    client: Any,
    env: Any,
    mode: str,
    seeds: Sequence[int],
    output_dir: Path,
    env_metadata: Mapping[str, Any],
    summary_path: Path,
    mode_summaries_so_far: Mapping[str, Any],
) -> dict[str, Any]:
    telemetry_dir = output_dir / "telemetry" / str(mode)
    step_path = telemetry_dir / "steps.jsonl"
    episode_path = telemetry_dir / "episodes.jsonl"
    step_path.unlink(missing_ok=True)
    episode_path.unlink(missing_ok=True)
    success_count = 0
    telemetry_step_count = 0
    episode_results: list[dict[str, Any]] = []
    outer_max_steps = int(env_metadata["outer_max_steps_per_episode"])
    mode_started = time.monotonic()
    print(
        "[MODE_START]",
        f"indicator_mode={mode}",
        f"episodes={len(seeds)}",
        f"outer_max_steps={outer_max_steps}",
        flush=True,
    )

    for episode_index, seed in enumerate(seeds, start=1):
        episode_started = time.monotonic()
        print(
            "[EPISODE_START]",
            f"indicator_mode={mode}",
            f"index={episode_index}/{len(seeds)}",
            f"seed={int(seed)}",
            flush=True,
        )
        obs, _info = env.reset(seed=int(seed))
        options_ep = {"seed": int(seed), "indicator_mode": str(mode)}
        client.reset(options=options_ep)
        done = False
        episode_success = False
        outer_steps = 0
        last_terminated = False
        last_truncated = False
        last_reward = 0.0
        reset_snapshot = helper3d._collect_env_snapshot(env)
        episode_step_records: list[dict[str, Any]] = []

        while not done and outer_steps < outer_max_steps:
            action, _action_info = client.get_action(obs, options=options_ep)
            if not isinstance(action, Mapping):
                raise TypeError(
                    "PolicyClient.get_action must return a dict action, got "
                    f"{type(action).__name__}"
                )
            obs, reward, term, trunc, step_info = env.step(action)
            reward_scalar = helper3d._scalarize_float(reward)
            last_reward = float(reward_scalar)
            last_terminated = bool(helper3d._scalarize_bool(term))
            last_truncated = bool(helper3d._scalarize_bool(trunc))
            done = bool(last_terminated or last_truncated)
            success_step = bool(helper3d._extract_success_step(step_info))
            episode_success = bool(episode_success or success_step)
            outer_steps += 1
            step_record: dict[str, Any] = {
                "episode_index": int(episode_index),
                "seed": int(seed),
                "indicator_mode": str(mode),
                "outer_step": int(outer_steps),
                "reward": float(reward_scalar),
                "terminated": bool(last_terminated),
                "truncated": bool(last_truncated),
                "done": bool(done),
                "success_step": bool(success_step),
                "episode_success_so_far": bool(episode_success),
                "policy_options": dict(options_ep),
                "action_summary": helper3d._summarize_action_chunk(dict(action)),
            }
            step_record.update(helper3d._collect_env_snapshot(env))
            intermediate = helper3d._extract_intermediate_signals(step_info)
            if intermediate is not None:
                step_record["intermediate_signals"] = intermediate
            episode_step_records.append(step_record)
            _append_jsonl(step_path, step_record)
            telemetry_step_count += 1

        success_count += 1 if episode_success else 0
        final_snapshot = (
            {
                key: episode_step_records[-1].get(key)
                for key in (
                    "sim_time_s",
                    "apple_pos_xyz",
                    "plate_pos_xyz",
                    "right_eef_pos_xyz",
                    "left_eef_pos_xyz",
                    "apple_to_right_eef_l2",
                    "apple_to_left_eef_l2",
                    "apple_to_plate_l2",
                    "right_eef_to_plate_l2",
                    "apple_height_z",
                    "plate_height_z",
                    "right_eef_height_z",
                )
            }
            if episode_step_records
            else reset_snapshot
        )
        failure_reason = helper3d._episode_failure_reason(
            success=bool(episode_success),
            done=bool(done),
            terminated=bool(last_terminated),
            truncated=bool(last_truncated),
            outer_steps=int(outer_steps),
            outer_max_steps=int(outer_max_steps),
        )
        failure_stage_guess = (
            None if episode_success else helper3d._failure_stage_guess(episode_step_records)
        )
        episode_elapsed_seconds = float(time.monotonic() - episode_started)
        episode_record = {
            "episode_index": int(episode_index),
            "seed": int(seed),
            "indicator_mode": str(mode),
            "success": bool(episode_success),
            "done": bool(done),
            "terminated": bool(last_terminated),
            "truncated": bool(last_truncated),
            "outer_steps": int(outer_steps),
            "episode_elapsed_seconds": episode_elapsed_seconds,
            "final_reward": float(last_reward),
            "failure_reason": failure_reason,
            "failure_stage_guess": failure_stage_guess,
            "reset_snapshot": reset_snapshot,
            "final_snapshot": final_snapshot,
            "n_success_steps": int(
                sum(1 for rec in episode_step_records if bool(rec.get("success_step")))
            ),
            "step_telemetry_records": int(len(episode_step_records)),
        }
        _append_jsonl(episode_path, episode_record)
        episode_results.append(episode_record)
        print(
            "[EPISODE_END]",
            f"indicator_mode={mode}",
            f"index={episode_index}/{len(seeds)}",
            f"seed={int(seed)}",
            f"elapsed_s={episode_elapsed_seconds:.3f}",
            f"success={bool(episode_success)}",
            f"failure_reason={failure_reason or 'none'}",
            flush=True,
        )
        partial = {
            "schema_version": "gr00t_g3_formal_eval_v1",
            "status": "RUNNING",
            "updated_at_utc": _utc_now(),
            "mode_summaries": {
                **dict(mode_summaries_so_far),
                str(mode): {
                    "status": "RUNNING",
                    "episodes_completed": int(episode_index),
                    "requested_episodes": int(len(seeds)),
                    "success_count": int(success_count),
                    "success_rate_so_far": float(success_count / episode_index),
                    "step_telemetry_jsonl": _safe_relpath(step_path),
                    "episode_telemetry_jsonl": _safe_relpath(episode_path),
                },
            },
        }
        _write_json(summary_path, partial)

    success_rate = float(success_count / len(seeds)) if seeds else 0.0
    mode_summary = {
        "status": "COMPLETED",
        "indicator_mode": str(mode),
        "episodes": int(len(seeds)),
        "success_count": int(success_count),
        "success_rate": float(success_rate),
        "elapsed_seconds": float(time.monotonic() - mode_started),
        "seed_list": [int(seed) for seed in seeds],
        "episode_results": episode_results,
        "telemetry_step_count": int(telemetry_step_count),
        "telemetry_episode_count": int(len(episode_results)),
        "step_telemetry_jsonl": _safe_relpath(step_path),
        "episode_telemetry_jsonl": _safe_relpath(episode_path),
    }
    print(
        "[MODE_END]",
        f"indicator_mode={mode}",
        f"success_count={success_count}/{len(seeds)}",
        f"success_rate={success_rate:.6f}",
        flush=True,
    )
    return mode_summary


def _build_comparisons(mode_summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def rate(mode: str) -> float | None:
        summary = mode_summaries.get(mode)
        if not isinstance(summary, Mapping):
            return None
        raw = summary.get("success_rate")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        return None

    positive = rate("positive")
    omit = rate("omit")
    negative = rate("negative")
    return {
        "positive_minus_omit_success_rate": None
        if positive is None or omit is None
        else float(positive - omit),
        "positive_minus_negative_success_rate": None
        if positive is None or negative is None
        else float(positive - negative),
        "omit_minus_negative_success_rate": None
        if omit is None or negative is None
        else float(omit - negative),
        "performance_claim_allowed": bool(positive is not None and omit is not None),
        "performance_claim_note": (
            "Success-rate deltas are formal-eval evidence only after all requested paired modes complete."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _enforce_no_sudo()
    checkpoint = _resolve_path(args.checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint}")
    output_dir = _resolve_path(args.output_dir)
    runtime_log_dir = _resolve_path(args.runtime_log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "formal_eval_summary.json"
    manifest_path = _resolve_path(args.run_manifest_json) if str(args.run_manifest_json).strip() else None
    modes = _normalize_indicator_modes(list(args.indicator_modes))
    seeds = _seed_list(int(args.seed_base), int(args.episode_count))
    visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    required_devices = str(args.required_cuda_visible_devices).strip()
    if visible_devices != required_devices:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES mismatch: "
            f"expected {required_devices!r}, got {visible_devices!r}"
        )

    _install_alarm_timeout(float(args.total_timeout_s))
    helper3d = _load_helper3d()
    helper3d._add_import_roots(REPO_ROOT)
    helper3d._install_robocasa_import_shims()

    server_proc: subprocess.Popen[str] | None = None
    server_handle: TextIO | None = None
    server_log: Path | None = None
    started_at = _utc_now()
    before_gpu_snapshot = _nvidia_smi_snapshot()
    env: Any = None
    mode_summaries: dict[str, dict[str, Any]] = {}
    error_info: dict[str, Any] | None = None
    server_cmd: list[str] | None = None
    server_payloads: Any = None
    env_metadata: dict[str, Any] = {}
    try:
        print("[FORMAL_START]", f"checkpoint={checkpoint}", f"modes={modes}", flush=True)
        print("[FORMAL_START]", f"seeds={seeds}", flush=True)
        print("[FORMAL_START]", f"CUDA_VISIBLE_DEVICES={visible_devices}", flush=True)
        server_proc, server_handle, server_log, server_cmd = _launch_policy_server(
            checkpoint=checkpoint,
            runtime_log_dir=runtime_log_dir,
            host=str(args.server_host),
            port=int(args.server_port),
        )
        print(
            "[SERVER_START]",
            f"pid={server_proc.pid}",
            f"port={int(args.server_port)}",
            f"log={server_log}",
            flush=True,
        )
        client, modality_cfg, ping_payload, server_payloads = _wait_for_policy_client(
            host=str(args.server_host),
            port=int(args.server_port),
            proc=server_proc,
            timeout_s=float(args.connect_timeout_s),
        )
        print("[SERVER_READY]", _jsonable(ping_payload), flush=True)
        env, env_metadata = _prepare_env(
            helper3d,
            modality_cfg,
            str(args.env_name),
            int(args.max_episode_steps),
            int(args.n_action_steps),
        )
        print("[ENV_READY]", json.dumps(_jsonable(env_metadata), sort_keys=True), flush=True)
        for mode in modes:
            mode_summaries[mode] = _run_mode(
                helper3d=helper3d,
                client=client,
                env=env,
                mode=mode,
                seeds=seeds,
                output_dir=output_dir,
                env_metadata=env_metadata,
                summary_path=summary_path,
                mode_summaries_so_far=mode_summaries,
            )
    except Exception as exc:
        error_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        }
        print("[FORMAL_ERROR]", f"{type(exc).__name__}: {exc}", flush=True)
    finally:
        _clear_alarm_timeout()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        _stop_policy_server(server_proc)
        if server_handle is not None:
            server_handle.close()

    after_gpu_snapshot = _nvidia_smi_snapshot()
    requested_episode_total = int(len(seeds) * len(modes))
    completed_episode_total = int(
        sum(int(summary.get("episodes", 0)) for summary in mode_summaries.values())
    )
    status = (
        "PASS"
        if error_info is None
        and completed_episode_total == requested_episode_total
        and set(mode_summaries.keys()) == set(modes)
        else "FAIL"
    )
    summary = {
        "schema_version": "gr00t_g3_formal_eval_v1",
        "artifact_kind": "gr00t_g3_checkpoint_formal_eval",
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "checkpoint": _safe_relpath(checkpoint),
        "run_manifest_json": _safe_relpath(manifest_path) if manifest_path else None,
        "prompt_raw": str(args.prompt_raw),
        "indicator_modes": list(modes),
        "seed_base": int(args.seed_base),
        "seed_list": [int(seed) for seed in seeds],
        "requested_episode_count_per_mode": int(len(seeds)),
        "requested_episode_total": requested_episode_total,
        "completed_episode_total": completed_episode_total,
        "mode_summaries": mode_summaries,
        "comparisons": _build_comparisons(mode_summaries),
        "env_metadata": _jsonable(env_metadata),
        "server": {
            "host": str(args.server_host),
            "port": int(args.server_port),
            "command": server_cmd,
            "log": _safe_relpath(server_log),
            "payloads": _jsonable(server_payloads),
        },
        "runtime_log_dir": _safe_relpath(runtime_log_dir),
        "output_dir": _safe_relpath(output_dir),
        "cuda_visible_devices": visible_devices,
        "required_cuda_visible_devices": required_devices,
        "before_gpu_snapshot": before_gpu_snapshot,
        "after_gpu_snapshot": after_gpu_snapshot,
        "no_sudo": True,
        "formal_protocol": {
            "env_name": str(args.env_name),
            "n_episodes": int(len(seeds)),
            "max_episode_steps": int(args.max_episode_steps),
            "n_action_steps": int(args.n_action_steps),
            "seed_list": [int(seed) for seed in seeds],
            "paired_indicator_modes": list(modes),
        },
        "performance_claim_guard": {
            "do_not_claim_recap_uplift_until_status_pass": True,
            "formal_eval_is_required_evidence": True,
        },
    }
    if error_info is not None:
        summary["error"] = error_info
    _write_json(summary_path, summary)
    print("[FORMAL_DONE]", json.dumps({"status": status, "summary_json": str(summary_path)}, sort_keys=True), flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_g3_formal_eval.py",
        description="Run GPU-explicit GR00T checkpoint formal eval with RECAP indicator-mode options.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-manifest-json", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--runtime-log-dir", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--server-host", default=DEFAULT_HOST)
    parser.add_argument("--server-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--prompt-raw", default=DEFAULT_PROMPT_RAW)
    parser.add_argument("--indicator-modes", nargs="+", default=["positive", "omit", "negative"])
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--episode-count", type=int, default=DEFAULT_EPISODE_COUNT)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--n-action-steps", type=int, default=DEFAULT_N_ACTION_STEPS)
    parser.add_argument("--connect-timeout-s", type=float, default=DEFAULT_CONNECT_TIMEOUT_S)
    parser.add_argument("--total-timeout-s", type=float, default=DEFAULT_TOTAL_TIMEOUT_S)
    parser.add_argument(
        "--required-cuda-visible-devices",
        default=DEFAULT_REQUIRED_CUDA_VISIBLE_DEVICES,
        help="Fail closed unless CUDA_VISIBLE_DEVICES exactly matches this value.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run(args)
    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
