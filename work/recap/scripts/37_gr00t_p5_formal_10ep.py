#!/usr/bin/env python3
"""Run the Iter3 GR00T P5 formal 10-episode runtime.

This script is intentionally a narrow orchestration layer around the existing
GR00T policy server launcher and ``3D_recap_eval.py`` telemetry surface.  It
does not alter training or rollout internals; it validates the P4 authority
gate, runs the checkpoint on GPU1, then writes the machine-readable P5 artifacts
required by the Iter3 plan.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RUN_ID = "stage1_claim_resolution_iter3_20260425T_nextZ"
ITER2_RUN_ID = "stage1_blocker_resolution_iter2_20260425T_nextZ"
P5_SCHEMA_VERSION = "gr00t_p5_formal_10ep_v1"
VALIDATOR_SCHEMA_VERSION = "gr00t_p5_formal_10ep_validator_v1"
POINTER_SCHEMA_VERSION = "gr00t_p5_authority_pointer_v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5561
DEFAULT_EPISODE_COUNT = 10
DEFAULT_THRESHOLD = 0.5
DEFAULT_TOTAL_TIMEOUT_S = 2400.0
DEFAULT_EVAL_TIMEOUT_S = 2100.0


DEFAULT_P4_VERDICT = (
    REPO_ROOT
    / "agent/artifacts"
    / ITER2_RUN_ID
    / "gr00t/p4_refresh/p4_gate_verdict.json"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
    / ITER2_RUN_ID
    / "gr00t/p4_rerun/conditioned_train/checkpoint-200"
)
DEFAULT_BASELINE_PROBE = (
    REPO_ROOT
    / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
    / "p5_gate_eval/baseline_first_subgoal_probe_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "agent/artifacts/recap_min_loop/single_gpu_v2_full_update"
    / RUN_ID
    / "gr00t/p5_formal_10ep"
)
DEFAULT_POINTER_ROOT = (
    REPO_ROOT / "agent/artifacts" / RUN_ID / "gr00t/p5_formal_10ep"
)
DEFAULT_RUNTIME_LOG_DIR = (
    REPO_ROOT / "agent/runtime_logs" / RUN_ID / "gr00t/p5_formal_10ep"
)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
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
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object in {path}")
    return {str(key): value for key, value in payload.items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError(f"JSONL row must be object: {path}:{line_no}")
            rows.append({str(key): value for key, value in payload.items()})
    return rows


def _append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(dict(record), handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
    return path


def _append_jsonl_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(dict(record), handle, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=True)
        return value
    except Exception:
        return {"type": type(value).__name__, "repr": repr(value)}


def _log(handle: TextIO, message: str) -> None:
    print(message)
    handle.write(message + "\n")
    handle.flush()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _load_36c_module() -> Any:
    module_path = REPO_ROOT / "work/recap/scripts/36c_gr00t_first_subgoal_rollout_probe.py"
    spec = importlib.util.spec_from_file_location("gr00t_36c_for_p5", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load server helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_command(command: Sequence[str], *, timeout_s: float | None = None) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        list(command),
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    return {
        "command": list(command),
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "elapsed_seconds": float(time.monotonic() - started),
    }


def _nvidia_smi_snapshot() -> dict[str, Any]:
    gpu_query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=15,
    )
    gpu1_query = _run_command(
        [
            "nvidia-smi",
            "--id=1",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=15,
    )
    compute_query = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=15,
    )
    gpu1_free_mib = None
    first_gpu1_line = str(gpu1_query.get("stdout") or "").strip().splitlines()
    if first_gpu1_line:
        parts = [part.strip() for part in first_gpu1_line[0].split(",")]
        if len(parts) >= 5:
            try:
                gpu1_free_mib = int(parts[4])
            except ValueError:
                gpu1_free_mib = None
    return {
        "gpu_query": {
            "returncode": gpu_query["returncode"],
            "stdout": gpu_query["stdout"],
        },
        "gpu1_query": {
            "returncode": gpu1_query["returncode"],
            "stdout": gpu1_query["stdout"],
            "memory_free_mib": gpu1_free_mib,
        },
        "compute_query": {
            "returncode": compute_query["returncode"],
            "stdout": compute_query["stdout"],
        },
    }


def _validate_p4_gate(p4_verdict: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if str(p4_verdict.get("status") or "").upper() != "PASS":
        blockers.append("p4_verdict_status_not_pass")
    if p4_verdict.get("formal_claim_allowed") is not True:
        blockers.append("p4_formal_claim_not_allowed")
    if p4_verdict.get("p5_formal_10ep_eligible") is not True:
        blockers.append("p5_formal_10ep_not_eligible")
    raw_blockers = p4_verdict.get("blocking_reasons")
    if isinstance(raw_blockers, Sequence) and not isinstance(raw_blockers, (str, bytes)):
        blockers.extend(str(reason) for reason in raw_blockers if str(reason))
    else:
        blockers.append("p4_blocking_reasons_not_list")
    return sorted(set(blockers))


def _baseline_step_telemetry_path(baseline_probe: Mapping[str, Any]) -> Path:
    raw = baseline_probe.get("telemetry_path")
    if not isinstance(raw, str) or not raw.strip():
        return (
            REPO_ROOT
            / "agent/artifacts/recap_min_loop/single_gpu_v1"
            / "t5_baseline_formal_eval/telemetry/eval_summary_steps.jsonl"
        )
    episode_path = _resolve_path(raw)
    if episode_path.name.endswith("_episodes.jsonl"):
        return episode_path.with_name(episode_path.name.replace("_episodes.jsonl", "_steps.jsonl"))
    return episode_path


def _select_seed_bundle(baseline_probe: Mapping[str, Any], episode_count: int) -> list[int]:
    raw_available = baseline_probe.get("available_seeds")
    if not isinstance(raw_available, Sequence) or isinstance(raw_available, (str, bytes)):
        raise ValueError("baseline probe missing available_seeds list")
    seeds = [int(seed) for seed in raw_available if isinstance(seed, int) and not isinstance(seed, bool)]
    seeds = sorted(set(seeds))
    if len(seeds) < int(episode_count):
        raise ValueError(
            f"baseline probe provides {len(seeds)} available seeds, need {episode_count}"
        )
    selected = seeds[: int(episode_count)]
    expected = list(range(selected[0], selected[0] + int(episode_count)))
    if selected != expected:
        raise ValueError("selected P5 seed bundle must be contiguous for 3D_recap_eval seed_base")
    return selected


def _group_steps_by_episode(step_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], list[Mapping[str, Any]]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in step_rows:
        seed = row.get("seed")
        episode_index = row.get("episode_index")
        if not isinstance(seed, int) or isinstance(seed, bool):
            continue
        if not isinstance(episode_index, int) or isinstance(episode_index, bool):
            continue
        grouped[(int(seed), int(episode_index))].append(row)
    return dict(grouped)


def _metric_from_steps(
    *,
    seed: int,
    episode_index: int,
    steps: Sequence[Mapping[str, Any]],
    success: bool,
) -> dict[str, Any]:
    hand_distances = [
        value
        for value in (_number(row.get("apple_to_right_eef_l2")) for row in steps)
        if value is not None
    ]
    apple_heights = [
        value
        for value in (_number(row.get("apple_height_z")) for row in steps)
        if value is not None
    ]
    min_dist = min(hand_distances) if hand_distances else None
    max_lift = None
    if apple_heights:
        max_lift = float(max(apple_heights) - apple_heights[0])
    contact_proxy = 1.0 if bool(success) or (min_dist is not None and min_dist <= 0.10) else 0.0
    lift_proxy = float(max_lift or 0.0)
    return {
        "seed": int(seed),
        "episode_index": int(episode_index),
        "success": bool(success),
        "min_dist_ee_to_apple": None if min_dist is None else float(min_dist),
        "contact_proxy": float(contact_proxy),
        "lift_proxy": float(lift_proxy),
        "contact_or_lift_proxy": float(max(contact_proxy, lift_proxy)),
        "step_count": int(len(steps)),
    }


def build_seed_metrics(
    *,
    step_rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    grouped = _group_steps_by_episode(step_rows)
    success_by_key: dict[tuple[int, int], bool] = {}
    for row in episode_rows:
        seed = row.get("seed")
        episode_index = row.get("episode_index")
        if isinstance(seed, int) and isinstance(episode_index, int):
            success_by_key[(int(seed), int(episode_index))] = row.get("success") is True
    metrics: dict[int, dict[str, Any]] = {}
    for key, steps in grouped.items():
        seed, episode_index = key
        metric = _metric_from_steps(
            seed=seed,
            episode_index=episode_index,
            steps=steps,
            success=success_by_key.get(key, False),
        )
        metrics[int(seed)] = metric
    return metrics


def build_p5_episode_records(
    *,
    selected_seeds: Sequence[int],
    p5_metrics_by_seed: Mapping[int, Mapping[str, Any]],
    baseline_metrics_by_seed: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in selected_seeds:
        p5_metric = p5_metrics_by_seed.get(int(seed))
        baseline_metric = baseline_metrics_by_seed.get(int(seed))
        p5_min = None if p5_metric is None else _number(p5_metric.get("min_dist_ee_to_apple"))
        baseline_min = (
            None
            if baseline_metric is None
            else _number(baseline_metric.get("min_dist_ee_to_apple"))
        )
        relative_improvement = None
        improved = None
        if p5_min is not None and baseline_min is not None and baseline_min > 0.0:
            relative_improvement = float((baseline_min - p5_min) / baseline_min)
            improved = bool(relative_improvement > 0.0)
        records.append(
            {
                "schema_version": "gr00t_p5_per_episode_metric_v1",
                "seed": int(seed),
                "episode_index": None if p5_metric is None else p5_metric.get("episode_index"),
                "success": False if p5_metric is None else bool(p5_metric.get("success") is True),
                "baseline_min_dist_ee_to_apple": baseline_min,
                "p5_min_dist_ee_to_apple": p5_min,
                "relative_improvement_min_dist_ee_to_apple": relative_improvement,
                "distance_improved": improved,
                "contact_proxy": None if p5_metric is None else p5_metric.get("contact_proxy"),
                "lift_proxy": None if p5_metric is None else p5_metric.get("lift_proxy"),
                "contact_or_lift_proxy": None
                if p5_metric is None
                else p5_metric.get("contact_or_lift_proxy"),
                "step_count": None if p5_metric is None else p5_metric.get("step_count"),
                "baseline_metric_present": baseline_metric is not None,
                "p5_metric_present": p5_metric is not None,
            }
        )
    return records


def run_direct_eval(
    *,
    helper3d: Any,
    client: Any,
    modality_cfg: Any,
    selected_seeds: Sequence[int],
    output_root: Path,
    runtime_log_dir: Path,
    max_episode_steps: int,
) -> dict[str, Any]:
    """Run one formal episode per selected seed through the existing WBC env."""

    step_telemetry_path = output_root / "telemetry" / "raw_eval_summary_steps.jsonl"
    episode_telemetry_path = output_root / "telemetry" / "raw_eval_summary_episodes.jsonl"
    step_telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    step_telemetry_path.unlink(missing_ok=True)
    episode_telemetry_path.unlink(missing_ok=True)
    step_telemetry_path.touch()
    episode_telemetry_path.touch()

    log_path = runtime_log_dir / "p5_direct_eval.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    episodes_completed = 0
    telemetry_step_count = 0
    telemetry_episode_count = 0
    episode_results: list[dict[str, Any]] = []
    error_info: dict[str, Any] | None = None
    env_registration_info: Any = None
    env_resolution: Mapping[str, Any] | None = None
    n_action_steps: int | None = None
    n_action_steps_source: str | None = None
    action_horizon: int | None = None
    env: Any = None

    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            _log(log_handle, f"[INFO] ts: {_dt.datetime.now().isoformat(timespec='seconds')}")
            _log(log_handle, f"[INFO] repo_root: {REPO_ROOT}")
            _log(log_handle, f"[INFO] selected_seeds: {list(selected_seeds)}")

            gym = __import__("gymnasium")
            rollout_mod = __import__("gr00t.eval.rollout_policy", fromlist=[""])
            env_registration_info = helper3d._ensure_explicit_g1_env_registration(gym)
            env_resolution = helper3d.state_conditioned_env_resolution.resolve_apple_to_plate_g1_env_name(
                gym,
                requested_env_name=helper3d.DEFAULT_ENV_NAME,
            )
            resolved_env_name = str(env_resolution["resolved_env_name"])

            if isinstance(modality_cfg, Mapping) and "action" in modality_cfg:
                delta_indices = getattr(modality_cfg["action"], "delta_indices", None)
                try:
                    delta = list(delta_indices or [])
                except TypeError:
                    delta = []
                action_horizon = int(len(delta)) if delta else None

            official_default = helper3d._default_n_action_steps_for_env(resolved_env_name)
            if official_default is not None:
                n_action_steps = int(official_default)
                n_action_steps_source = "g1_execution_surface_default"
            elif action_horizon is not None:
                n_action_steps = int(action_horizon)
                n_action_steps_source = "server_action_horizon_fallback"
            else:
                n_action_steps = 20
                n_action_steps_source = "generic_default_20"

            _log(log_handle, f"[INFO] env_name: {resolved_env_name}")
            _log(log_handle, f"[INFO] n_action_steps: {n_action_steps}")
            _log(log_handle, f"[INFO] n_action_steps_source: {n_action_steps_source}")

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
            outer_max_steps = max(
                1,
                (int(max_episode_steps) + int(n_action_steps) - 1)
                // int(n_action_steps),
            )
            _log(log_handle, f"[INFO] outer_max_steps_per_episode: {outer_max_steps}")

            for episode_index, seed in enumerate(selected_seeds, start=1):
                episode_started = time.monotonic()
                _log(
                    log_handle,
                    f"[EPISODE_START] index={episode_index}/{len(selected_seeds)} seed={int(seed)}",
                )
                obs, _info = env.reset(seed=int(seed))
                options_ep = {"seed": int(seed)}
                client.reset(options=options_ep)
                done = False
                episode_success = False
                outer_steps = 0
                last_terminated = False
                last_truncated = False
                last_reward = 0.0
                reset_snapshot = helper3d._collect_env_snapshot(env)
                episode_step_records: list[dict[str, Any]] = []

                while not done and outer_steps < int(outer_max_steps):
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
                        "advantage": None,
                        "advantage_mode": "unconditional",
                        "outer_step": int(outer_steps),
                        "reward": float(reward_scalar),
                        "terminated": bool(last_terminated),
                        "truncated": bool(last_truncated),
                        "done": bool(done),
                        "success_step": bool(success_step),
                        "episode_success_so_far": bool(episode_success),
                        "action_summary": helper3d._summarize_action_chunk(action),
                    }
                    step_record.update(helper3d._collect_env_snapshot(env))
                    intermediate = helper3d._extract_intermediate_signals(step_info)
                    if intermediate is not None:
                        step_record["intermediate_signals"] = intermediate
                    episode_step_records.append(step_record)
                    _append_jsonl_record(step_telemetry_path, step_record)
                    telemetry_step_count += 1

                success_count += 1 if episode_success else 0
                episodes_completed += 1
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
                    None
                    if episode_success
                    else helper3d._failure_stage_guess(episode_step_records)
                )
                episode_elapsed_seconds = float(time.monotonic() - episode_started)
                episode_record = {
                    "episode_index": int(episode_index),
                    "seed": int(seed),
                    "advantage": None,
                    "advantage_mode": "unconditional",
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
                _append_jsonl_record(episode_telemetry_path, episode_record)
                telemetry_episode_count += 1
                episode_results.append(
                    {
                        "episode_index": int(episode_index),
                        "seed": int(seed),
                        "success": bool(episode_success),
                        "done": bool(done),
                        "terminated": bool(last_terminated),
                        "truncated": bool(last_truncated),
                        "outer_steps": int(outer_steps),
                        "episode_elapsed_seconds": episode_elapsed_seconds,
                        "failure_reason": failure_reason,
                    }
                )
                _log(
                    log_handle,
                    "[EPISODE_END] "
                    + f"index={episode_index}/{len(selected_seeds)} "
                    + f"seed={int(seed)} elapsed_s={episode_elapsed_seconds:.6f} "
                    + f"success={bool(episode_success)} "
                    + f"failure_reason={failure_reason or 'none'}",
                )
        except Exception as exc:
            error_info = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            _log(log_handle, f"[ERROR] direct_eval_failed: {type(exc).__name__}: {exc}")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    success_rate = float(success_count / episodes_completed) if episodes_completed else 0.0
    summary = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "advantage": None,
        "advantage_mode": "unconditional",
        "episodes": int(episodes_completed),
        "requested_episodes": int(len(selected_seeds)),
        "success_count": int(success_count),
        "success_rate": float(success_rate),
        "log_path": str(log_path),
        "env_name": None if env_resolution is None else str(env_resolution["resolved_env_name"]),
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "max_episode_steps": int(max_episode_steps),
        "server_action_horizon": action_horizon,
        "n_action_steps": n_action_steps,
        "n_action_steps_source": n_action_steps_source,
        "seed_base": int(selected_seeds[0]) if selected_seeds else None,
        "episode_results": episode_results,
        "telemetry_enabled": True,
        "step_telemetry_jsonl": str(step_telemetry_path),
        "episode_telemetry_jsonl": str(episode_telemetry_path),
        "telemetry_step_count": int(telemetry_step_count),
        "telemetry_episode_count": int(telemetry_episode_count),
        "env_registration": _jsonable(env_registration_info),
        "execution_surface_contract": {
            "policy_horizon_expected": action_horizon,
            "n_action_steps": n_action_steps,
            "g1_default_execution_steps": int(helper3d.DEFAULT_G1_EXECUTION_N_ACTION_STEPS),
            "must_not_conflate_horizon_and_execution": True,
        },
        "authority_status": "p5_formal_runtime",
        "diagnostic_only": False,
    }
    if env_resolution is not None:
        summary["env_resolution"] = {
            "logical_task": env_resolution["logical_task"],
            "requested_env_name": env_resolution["requested_env_name"],
            "resolved_env_name": env_resolution["resolved_env_name"],
            "alias_applied": env_resolution["alias_applied"],
            "available_close_matches": env_resolution["available_close_matches"],
        }
    if error_info is not None:
        summary["error"] = error_info
    _write_json(output_root / "raw_eval_summary.json", summary)
    return summary


def _claim_language_hits(paths: Sequence[Path], forbidden_phrases_path: Path) -> list[dict[str, Any]]:
    if not forbidden_phrases_path.is_file():
        return []
    phrases = [
        line.rstrip("\n")
        for line in forbidden_phrases_path.read_text(encoding="utf-8").splitlines()
        if line.rstrip("\n")
    ]
    hits: list[dict[str, Any]] = []
    for root in paths:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for candidate in candidates:
            try:
                text = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                for phrase in phrases:
                    if phrase and phrase in line:
                        hits.append(
                            {
                                "file_path": _safe_relpath(candidate),
                                "line_no": int(line_no),
                                "matched_phrase": phrase,
                            }
                        )
    return hits


def build_summary_and_validator(
    *,
    selected_seeds: Sequence[int],
    per_episode_records: Sequence[Mapping[str, Any]],
    eval_summary: Mapping[str, Any],
    p4_verdict: Mapping[str, Any],
    p4_blockers: Sequence[str],
    gpu_boundary_ok: bool,
    gpu0_or_gpu3_touched: bool,
    threshold: float,
    output_root: Path,
    pointer_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    improvements = [
        float(record["relative_improvement_min_dist_ee_to_apple"])
        for record in per_episode_records
        if isinstance(record.get("relative_improvement_min_dist_ee_to_apple"), (int, float))
        and not isinstance(record.get("relative_improvement_min_dist_ee_to_apple"), bool)
    ]
    missing_metric_seeds = [
        int(record["seed"])
        for record in per_episode_records
        if record.get("p5_metric_present") is not True
        or record.get("baseline_metric_present") is not True
    ]
    mean_improvement = _mean(improvements)
    success_count = sum(1 for record in per_episode_records if record.get("success") is True)
    paired_improvement_count = sum(
        1 for record in per_episode_records if record.get("distance_improved") is True
    )
    summary_status = "PASS"
    blockers = list(p4_blockers)
    if int(eval_summary.get("episodes", 0) or 0) != len(selected_seeds):
        blockers.append("formal_episode_count_mismatch")
    if missing_metric_seeds:
        blockers.append("missing_paired_seed_metrics")
    if not gpu_boundary_ok:
        blockers.append("gpu_boundary_violation")
    if mean_improvement is None:
        blockers.append("mean_relative_improvement_missing")
    elif mean_improvement < float(threshold):
        blockers.append("mean_relative_improvement_below_threshold")
    if blockers:
        summary_status = "BLOCK"

    threshold_cited_from = (
        "iter2 p4_gate_verdict.first_subgoal_probe."
        "mean_relative_improvement_min_dist_ee_to_apple"
    )
    summary = {
        "schema_version": P5_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "status": summary_status,
        "p5_runtime_executed": int(eval_summary.get("episodes", 0) or 0) == len(selected_seeds),
        "episode_count": int(eval_summary.get("episodes", 0) or 0),
        "requested_episode_count": int(len(selected_seeds)),
        "selected_seeds": [int(seed) for seed in selected_seeds],
        "success_count": int(success_count),
        "success_rate": float(success_count / len(selected_seeds)) if selected_seeds else 0.0,
        "paired_seed_count": int(len(improvements)),
        "paired_seed_improvement_count": int(paired_improvement_count),
        "mean_relative_improvement_min_dist_ee_to_apple": mean_improvement,
        "threshold_cited_from": threshold_cited_from,
        "threshold_value": float(threshold),
        "threshold_reference_value": p4_verdict.get("mean_relative_improvement_min_dist_ee_to_apple"),
        "blocking_reasons": sorted(set(blockers)),
        "gpu_boundary_ok": bool(gpu_boundary_ok),
        "gpu0_or_gpu3_touched": bool(gpu0_or_gpu3_touched),
        "artifact_root_nested_correctly": _safe_relpath(output_root).startswith(
            "agent/artifacts/recap_min_loop/single_gpu_v2_full_update/"
        ),
        "raw_eval_summary": _safe_relpath(output_root / "raw_eval_summary.json"),
        "per_episode_metrics": _safe_relpath(output_root / "p5_per_episode_metrics.jsonl"),
        "created_at_utc": _utc_now(),
    }
    validator = {
        "schema_version": VALIDATOR_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "status": "PASS" if not blockers else "BLOCK_with_specific_reasons",
        "p5_runtime_executed": summary["p5_runtime_executed"],
        "episode_count": summary["episode_count"],
        "gpu_boundary_ok": bool(gpu_boundary_ok),
        "gpu0_or_gpu3_touched": bool(gpu0_or_gpu3_touched),
        "p5_validator_status": "PASS" if not blockers else "BLOCK_with_specific_reasons",
        "threshold_cited_from": threshold_cited_from,
        "threshold_value": float(threshold),
        "artifact_root_nested_correctly": summary["artifact_root_nested_correctly"],
        "formal_claim_language_checked": False,
        "blocking_reasons": sorted(set(blockers)),
        "missing_metric_seeds": sorted(set(missing_metric_seeds)),
        "evaluated_at_utc": _utc_now(),
    }
    return summary, validator


def _write_pointer(pointer_root: Path, output_root: Path, validator: Mapping[str, Any]) -> Path:
    pointer = {
        "schema_version": POINTER_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "authority_root": _safe_relpath(output_root),
        "p5_10episode_run_manifest": _safe_relpath(output_root / "p5_10episode_run_manifest.json"),
        "p5_per_episode_metrics": _safe_relpath(output_root / "p5_per_episode_metrics.jsonl"),
        "p5_summary": _safe_relpath(output_root / "p5_summary.json"),
        "p5_validator_report": _safe_relpath(output_root / "p5_validator_report.json"),
        "p5_validator_status": validator.get("p5_validator_status"),
        "created_at_utc": _utc_now(),
    }
    return _write_json(pointer_root / "p5_authority_pointer.json", pointer)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _resolve_path(args.output_root)
    pointer_root = _resolve_path(args.pointer_root)
    runtime_log_dir = _resolve_path(args.runtime_log_dir)
    p4_verdict_path = _resolve_path(args.p4_verdict)
    checkpoint = _resolve_path(args.checkpoint)
    baseline_probe_path = _resolve_path(args.baseline_probe)
    forbidden_phrases_path = _resolve_path(args.forbidden_phrases)

    output_root.mkdir(parents=True, exist_ok=True)
    pointer_root.mkdir(parents=True, exist_ok=True)
    runtime_log_dir.mkdir(parents=True, exist_ok=True)

    if args.no_sudo:
        sudo_keys = [key for key in ("SUDO_UID", "SUDO_USER", "SUDO_COMMAND") if os.environ.get(key)]
        if sudo_keys or (hasattr(os, "geteuid") and os.geteuid() == 0):
            raise PermissionError("sudo/root execution is forbidden for this runtime")

    visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    gpu_boundary_ok = visible_devices == "1"
    if not gpu_boundary_ok:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly '1' for Worker A")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    p4_verdict = _read_json(p4_verdict_path)
    p4_blockers = _validate_p4_gate(p4_verdict)
    baseline_probe = _read_json(baseline_probe_path)
    selected_seeds = _select_seed_bundle(baseline_probe, int(args.episode_count))
    baseline_step_path = _baseline_step_telemetry_path(baseline_probe)
    baseline_episode_path = baseline_step_path.with_name(
        baseline_step_path.name.replace("_steps.jsonl", "_episodes.jsonl")
    )

    before_gpu_snapshot = _nvidia_smi_snapshot()
    gpu1_free = before_gpu_snapshot["gpu1_query"].get("memory_free_mib")
    if not isinstance(gpu1_free, int) or gpu1_free <= 0:
        raise RuntimeError("GPU1 free memory probe failed or returned zero")

    helper36c = _load_36c_module()
    server_proc: subprocess.Popen[str] | None = None
    server_handle: TextIO | None = None
    server_log: Path | None = None
    eval_elapsed = 0.0
    eval_summary: dict[str, Any] = {}
    started_at = _utc_now()
    server_proc, server_handle, server_log = helper36c._launch_policy_server(
        checkpoint=checkpoint,
        runtime_log_dir=runtime_log_dir,
        host=DEFAULT_HOST,
        port=int(args.server_port),
    )
    started_eval = time.monotonic()
    try:
        helper3d = helper36c._load_3d_eval_module()
        helper3d._add_import_roots(REPO_ROOT)
        helper3d._install_robocasa_import_shims()
        client, modality_cfg = helper36c._wait_for_policy_client(
            helper3d,
            host=DEFAULT_HOST,
            port=int(args.server_port),
            proc=server_proc,
            timeout_s=float(args.connect_timeout_s),
        )
        eval_summary = run_direct_eval(
            helper3d=helper3d,
            client=client,
            modality_cfg=modality_cfg,
            selected_seeds=selected_seeds,
            output_root=output_root,
            runtime_log_dir=runtime_log_dir,
            max_episode_steps=1440,
        )
    finally:
        eval_elapsed = float(time.monotonic() - started_eval)
        if server_proc is not None:
            helper36c._stop_policy_server(server_proc)
        if server_handle is not None:
            server_handle.close()

    after_gpu_snapshot = _nvidia_smi_snapshot()

    raw_eval_summary_path = output_root / "raw_eval_summary.json"
    if not raw_eval_summary_path.is_file():
        raise RuntimeError("P5 eval did not write raw_eval_summary.json")

    eval_summary = _read_json(raw_eval_summary_path)
    eval_returncode = 1 if "error" in eval_summary else 0
    p5_step_path = _resolve_path(str(eval_summary["step_telemetry_jsonl"]))
    p5_episode_path = _resolve_path(str(eval_summary["episode_telemetry_jsonl"]))
    p5_metrics_by_seed = build_seed_metrics(
        step_rows=_read_jsonl(p5_step_path),
        episode_rows=_read_jsonl(p5_episode_path),
    )
    baseline_metrics_by_seed = build_seed_metrics(
        step_rows=_read_jsonl(baseline_step_path),
        episode_rows=_read_jsonl(baseline_episode_path),
    )
    per_episode_records = build_p5_episode_records(
        selected_seeds=selected_seeds,
        p5_metrics_by_seed=p5_metrics_by_seed,
        baseline_metrics_by_seed=baseline_metrics_by_seed,
    )
    per_episode_path = _append_jsonl(output_root / "p5_per_episode_metrics.jsonl", per_episode_records)

    gpu0_or_gpu3_touched = False
    summary, validator = build_summary_and_validator(
        selected_seeds=selected_seeds,
        per_episode_records=per_episode_records,
        eval_summary=eval_summary,
        p4_verdict=p4_verdict,
        p4_blockers=p4_blockers,
        gpu_boundary_ok=gpu_boundary_ok,
        gpu0_or_gpu3_touched=gpu0_or_gpu3_touched,
        threshold=float(args.threshold),
        output_root=output_root,
        pointer_root=pointer_root,
    )
    if int(eval_returncode) != 0:
        summary["blocking_reasons"] = sorted(
            set([*summary["blocking_reasons"], "direct_eval_error"])
        )
        summary["status"] = "BLOCK"
        validator["blocking_reasons"] = sorted(
            set([*validator["blocking_reasons"], "direct_eval_error"])
        )
        validator["status"] = "BLOCK_with_specific_reasons"
        validator["p5_validator_status"] = "BLOCK_with_specific_reasons"

    manifest = {
        "schema_version": P5_SCHEMA_VERSION,
        "artifact_kind": "p5_10episode_run_manifest",
        "run_id": RUN_ID,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "checkpoint": _safe_relpath(checkpoint),
        "p4_verdict": _safe_relpath(p4_verdict_path),
        "baseline_probe": _safe_relpath(baseline_probe_path),
        "baseline_step_telemetry": _safe_relpath(baseline_step_path),
        "selected_seeds": [int(seed) for seed in selected_seeds],
        "requested_episode_count": int(args.episode_count),
        "cuda_visible_devices": visible_devices,
        "gpu_boundary_ok": bool(gpu_boundary_ok),
        "server_host": DEFAULT_HOST,
        "server_port": int(args.server_port),
        "server_log": _safe_relpath(server_log),
        "eval_command": [
            "direct_policy_client_rollout",
            "--episode-count",
            str(int(args.episode_count)),
            "--seed-base",
            str(int(selected_seeds[0])),
        ],
        "eval_returncode": int(eval_returncode),
        "eval_elapsed_seconds": float(eval_elapsed),
        "eval_log": _safe_relpath(eval_summary.get("log_path")),
        "raw_eval_summary": _safe_relpath(raw_eval_summary_path),
        "p5_per_episode_metrics": _safe_relpath(per_episode_path),
        "before_gpu_snapshot": before_gpu_snapshot,
        "after_gpu_snapshot": after_gpu_snapshot,
        "no_sudo": bool(args.no_sudo),
    }
    _write_json(output_root / "p5_10episode_run_manifest.json", manifest)
    _write_json(output_root / "p5_summary.json", summary)
    pointer_path = _write_pointer(pointer_root, output_root, validator)

    claim_hits = _claim_language_hits([output_root, pointer_path], forbidden_phrases_path)
    validator["formal_claim_language_checked"] = True
    validator["claim_language_hits"] = claim_hits
    if claim_hits:
        validator["status"] = "BLOCK_with_specific_reasons"
        validator["p5_validator_status"] = "BLOCK_with_specific_reasons"
        validator["blocking_reasons"] = sorted(
            set([*validator["blocking_reasons"], "forbidden_claim_language_detected"])
        )
    _write_json(output_root / "p5_validator_report.json", validator)
    _write_pointer(pointer_root, output_root, validator)
    return {
        "status": validator["p5_validator_status"],
        "output_root": _safe_relpath(output_root),
        "pointer": _safe_relpath(pointer_path),
        "episode_count": summary["episode_count"],
        "blocking_reasons": validator["blocking_reasons"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="37_gr00t_p5_formal_10ep.py",
        description="Run Iter3 GR00T P5 formal 10-episode runtime and write authority artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--p4-verdict", default=str(DEFAULT_P4_VERDICT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--baseline-probe", default=str(DEFAULT_BASELINE_PROBE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--pointer-root", default=str(DEFAULT_POINTER_ROOT))
    parser.add_argument("--runtime-log-dir", default=str(DEFAULT_RUNTIME_LOG_DIR))
    parser.add_argument("--forbidden-phrases", default=str(REPO_ROOT / ".omc/iter3_forbidden_phrases.txt"))
    parser.add_argument("--server-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--episode-count", type=int, default=DEFAULT_EPISODE_COUNT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--connect-timeout-s", type=float, default=600.0)
    parser.add_argument("--eval-timeout-s", type=float, default=DEFAULT_EVAL_TIMEOUT_S)
    parser.add_argument("--total-timeout-s", type=float, default=DEFAULT_TOTAL_TIMEOUT_S)
    parser.add_argument("--no-sudo", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        if int(args.episode_count) <= 0:
            parser.error("--episode-count must be > 0")
        result = run(args)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if time.monotonic() - started > float(args.total_timeout_s):
        print("[ERROR] total timeout exceeded after artifact write", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
