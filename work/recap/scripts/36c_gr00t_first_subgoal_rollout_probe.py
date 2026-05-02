#!/usr/bin/env python3
"""Run a seed-scoped GR00T first-subgoal rollout probe.

The dry-run path is intentionally model-free for W3'.1 hard-gate checks.  The
real rollout path delegates G1 MuJoCo wiring and telemetry helpers to the
existing ``3D_recap_eval.py`` surface instead of reimplementing environment
setup here.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, TextIO, cast


sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA_VERSION = "baseline_first_subgoal_probe_v1"
ARTIFACT_KIND = "conditioned_first_subgoal_probe_seed_scope"
SMOKE_GATE_SCHEMA_VERSION = "smoke_gate_v1"
TELEMETRY_EPISODES_NAME = "eval_summary_episodes.jsonl"
TELEMETRY_STEPS_NAME = "eval_summary_steps.jsonl"
DEFAULT_N_EPISODES = 30
DEFAULT_ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 5555
DEFAULT_CONNECT_TIMEOUT_S = 600.0
DEFAULT_TOTAL_TIMEOUT_S = 1800.0
DEFAULT_SERVER_DEVICE_ENV = "GR00T_36C_SERVER_DEVICE"
DRY_RUN_TELEMETRY_SENTINEL = (
    b"36c_gr00t_first_subgoal_rollout_probe:dry-run-smoke:v1\n"
)


class _TeeStream:
    def __init__(self, *targets: TextIO):
        self._targets = targets

    def write(self, text: str) -> int:
        for target in self._targets:
            target.write(text)
        return len(text)

    def flush(self) -> None:
        for target in self._targets:
            target.flush()


@contextlib.contextmanager
def _tee_stdio(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        stdout = _TeeStream(sys.stdout, handle)
        stderr = _TeeStream(sys.stderr, handle)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield


def _repo_root() -> Path:
    return REPO_ROOT


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path.resolve()


def _safe_relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(dict(record), handle, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object in {path}")
    return {str(key): value for key, value in payload.items()}


def _dry_run_telemetry_hash() -> str:
    return hashlib.sha256(DRY_RUN_TELEMETRY_SENTINEL).hexdigest()


def _emit_smoke_gate(output_json: Path) -> dict[str, Any]:
    payload = {
        "schema_version": SMOKE_GATE_SCHEMA_VERSION,
        "status": "PASS",
        "telemetry_sha256": _dry_run_telemetry_hash(),
    }
    _write_json(output_json.parent / "smoke_gate.json", payload)
    return payload


def _enforce_no_sudo() -> None:
    sudo_keys = ("SUDO_UID", "SUDO_USER", "SUDO_COMMAND")
    sudo_env = [key for key in sudo_keys if os.environ.get(key)]
    geteuid = getattr(os, "geteuid", None)
    running_as_root = callable(geteuid) and int(geteuid()) == 0
    if sudo_env or running_as_root:
        details = ",".join(sudo_env) if sudo_env else "effective_uid_0"
        raise PermissionError(f"--no-sudo contract violation detected: {details}")


def _load_3d_eval_module() -> Any:
    module_path = _repo_root() / "work" / "recap" / "scripts" / "3D_recap_eval.py"
    spec = importlib.util.spec_from_file_location("recap_3d_eval_for_36c", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load rollout helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launch_policy_server(
    *,
    checkpoint: Path,
    runtime_log_dir: Path,
    host: str,
    port: int,
) -> tuple[subprocess.Popen[str], TextIO, Path]:
    server_script = (
        _repo_root() / "submodules" / "Isaac-GR00T" / "gr00t" / "eval" / "run_gr00t_server.py"
    )
    server_log = runtime_log_dir / "36c_gr00t_server.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    handle = server_log.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["PYOPENGL_PLATFORM"] = env.get("PYOPENGL_PLATFORM", "egl")
    pythonpath_parts = [
        str(_repo_root() / "submodules" / "Isaac-GR00T"),
        str(_repo_root()),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(str(env["PYTHONPATH"]))
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    device = env.get(DEFAULT_SERVER_DEVICE_ENV, "cuda")
    cmd = [
        sys.executable,
        str(server_script),
        "--model-path",
        str(checkpoint),
        "--embodiment-tag",
        "UNITREE_G1",
        "--device",
        str(device),
        "--use-sim-policy-wrapper",
        "--host",
        host,
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(_repo_root()),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, handle, server_log


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
    with contextlib.suppress(Exception):
        proc.kill()


def _wait_for_policy_client(
    helper: Any,
    *,
    host: str,
    port: int,
    proc: subprocess.Popen[str],
    timeout_s: float,
) -> tuple[Any, Any]:
    sc_mod = __import__("gr00t.policy.server_client", fromlist=["PolicyClient"])
    PolicyClient = getattr(sc_mod, "PolicyClient")
    client = PolicyClient(host=host, port=int(port), strict=False)

    def _configure_client_socket(timeout_ms: int) -> None:
        with contextlib.suppress(Exception):
            zmq = __import__("zmq")
            client.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
            client.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
            client.socket.setsockopt(zmq.LINGER, 0)

    _configure_client_socket(timeout_ms=1000)
    started = time.monotonic()
    last_error: str | None = None
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"GR00T server exited early with rc={proc.returncode}")
        try:
            ping_payload = client.call_endpoint("ping", requires_input=False)
            modality_cfg = client.get_modality_config()
            _configure_client_socket(timeout_ms=int(DEFAULT_TOTAL_TIMEOUT_S * 1000))
            return client, modality_cfg
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            with contextlib.suppress(Exception):
                client.socket.close(0)
            client = PolicyClient(host=host, port=int(port), strict=False)
            _configure_client_socket(timeout_ms=1000)
            if time.monotonic() - started > float(timeout_s):
                raise TimeoutError(
                    f"timed out waiting for GR00T policy server ping; last_error={last_error}"
                ) from exc
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


def _build_episode_metric(episode_records: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any] | None:
    min_distances: list[float] = []
    lift_values: list[float] = []
    ever_near_apple = False
    any_success = False
    for record in episode_records:
        any_success = bool(any_success or record.get("success") is True)
        guess = record.get("failure_stage_guess")
        if not isinstance(guess, Mapping):
            continue
        min_dist = guess.get("min_apple_to_right_eef_l2")
        if isinstance(min_dist, (int, float)) and not isinstance(min_dist, bool):
            min_distances.append(float(min_dist))
        lift = guess.get("max_apple_lift_z")
        if isinstance(lift, (int, float)) and not isinstance(lift, bool):
            lift_values.append(float(lift))
        ever_near_apple = bool(ever_near_apple or guess.get("ever_near_apple") is True)
    if not min_distances:
        return None
    contact_proxy = 1.0 if (ever_near_apple or any_success) else 0.0
    lift_proxy = max(lift_values) if lift_values else (1.0 if any_success else 0.0)
    return {
        "seed": int(seed),
        "min_dist_ee_to_apple": float(min(min_distances)),
        "contact_proxy": float(contact_proxy),
        "lift_proxy": float(lift_proxy),
        "contact_or_lift_proxy": float(max(contact_proxy, lift_proxy)),
    }


def _build_probe_payload(
    *,
    status: str,
    conditioned_checkpoint: Path,
    continuation_run_root: Path | None,
    baseline_probe: Path,
    baseline_payload: Mapping[str, Any] | None,
    seed: int,
    n_episodes: int,
    telemetry_path: Path,
    output_json: Path,
    seed_metric: Mapping[str, Any] | None,
    blocking_reasons: Sequence[str],
    runtime_log_dir: Path,
    server_log: Path | None = None,
) -> dict[str, Any]:
    selected_seeds = [int(seed)] if seed_metric is not None else []
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": str(status).upper(),
        "probe_eligible": bool(status == "PASS"),
        "required_seed_metrics_present": seed_metric is not None,
        "complete_3seed_subgoal_probe": False,
        "blocking_reasons": list(blocking_reasons),
        "required_seeds": [int(seed)],
        "available_seeds": [int(seed)] if seed_metric is not None else [],
        "missing_required_seeds": [] if seed_metric is not None else [int(seed)],
        "seed_start": int(seed),
        "seed_end": int(seed),
        "selected_seeds": selected_seeds,
        "seed_metrics": [dict(seed_metric)] if seed_metric is not None else [],
        "conditioned_checkpoint": _safe_relpath(conditioned_checkpoint),
        "continuation_run_root": _safe_relpath(continuation_run_root),
        "baseline_probe": _safe_relpath(baseline_probe),
        "baseline_probe_schema_version": None
        if baseline_payload is None
        else baseline_payload.get("schema_version"),
        "baseline_selected_seeds": []
        if baseline_payload is None
        else list(cast(Sequence[Any], baseline_payload.get("selected_seeds") or [])),
        "n_episodes": int(n_episodes),
        "telemetry_path": _safe_relpath(telemetry_path),
        "output_path": _safe_relpath(output_json),
        "runtime_log_dir": _safe_relpath(runtime_log_dir),
        "server_log": _safe_relpath(server_log),
        "read_only_authority_root": True,
        "skip_reason": None if status == "PASS" else "seed_scoped_probe_blocked",
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.no_sudo):
        _enforce_no_sudo()
    if int(args.n_episodes) <= 0:
        raise ValueError("--n-episodes must be > 0")

    output_json = _resolve_path(args.output_json)
    telemetry_dir = _resolve_path(args.telemetry_dir)
    runtime_log_dir = _resolve_path(args.runtime_log_dir)

    if bool(args.dry_run_smoke):
        return _emit_smoke_gate(output_json)

    conditioned_checkpoint = _resolve_path(args.conditioned_checkpoint)
    continuation_run_root = (
        None if args.continuation_run_root is None else _resolve_path(args.continuation_run_root)
    )
    baseline_probe = _resolve_path(args.baseline_probe)
    telemetry_path = telemetry_dir / TELEMETRY_EPISODES_NAME
    step_telemetry_path = telemetry_dir / TELEMETRY_STEPS_NAME
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path.unlink(missing_ok=True)
    step_telemetry_path.unlink(missing_ok=True)

    blocking_reasons: list[str] = []
    baseline_payload: dict[str, Any] | None = None
    if not conditioned_checkpoint.exists():
        blocking_reasons.append("missing_conditioned_checkpoint")
    if not baseline_probe.is_file():
        blocking_reasons.append("missing_baseline_probe")
    else:
        baseline_payload = _read_json(baseline_probe)
        if baseline_payload.get("schema_version") != SCHEMA_VERSION:
            blocking_reasons.append("baseline_probe_schema_mismatch")
    if continuation_run_root is not None and not continuation_run_root.exists():
        blocking_reasons.append("missing_continuation_run_root")
    if blocking_reasons:
        payload = _build_probe_payload(
            status="BLOCK",
            conditioned_checkpoint=conditioned_checkpoint,
            continuation_run_root=continuation_run_root,
            baseline_probe=baseline_probe,
            baseline_payload=baseline_payload,
            seed=int(args.seed),
            n_episodes=int(args.n_episodes),
            telemetry_path=telemetry_path,
            output_json=output_json,
            seed_metric=None,
            blocking_reasons=blocking_reasons,
            runtime_log_dir=runtime_log_dir,
        )
        _write_json(output_json, payload)
        return payload

    helper = _load_3d_eval_module()
    helper._add_import_roots(_repo_root())
    helper._install_robocasa_import_shims()

    server_proc: subprocess.Popen[str] | None = None
    server_handle: TextIO | None = None
    server_log: Path | None = None
    episode_records: list[dict[str, Any]] = []
    rollout_log = runtime_log_dir / "36c_first_subgoal_rollout.log"
    try:
        server_proc, server_handle, server_log = _launch_policy_server(
            checkpoint=conditioned_checkpoint,
            runtime_log_dir=runtime_log_dir,
            host=DEFAULT_SERVER_HOST,
            port=DEFAULT_SERVER_PORT,
        )
        with _tee_stdio(rollout_log):
            client, modality_cfg = _wait_for_policy_client(
                helper,
                host=DEFAULT_SERVER_HOST,
                port=DEFAULT_SERVER_PORT,
                proc=server_proc,
                timeout_s=DEFAULT_CONNECT_TIMEOUT_S,
            )
            gym = __import__("gymnasium")
            rollout_mod = __import__("gr00t.eval.rollout_policy", fromlist=[""])
            env_registration = helper._ensure_explicit_g1_env_registration(gym)
            env_resolution = helper.state_conditioned_env_resolution.resolve_apple_to_plate_g1_env_name(
                gym,
                requested_env_name=DEFAULT_ENV_NAME,
            )
            resolved_env_name = str(env_resolution["resolved_env_name"])
            action_horizon = _infer_action_horizon(modality_cfg)
            n_action_steps = helper._default_n_action_steps_for_env(resolved_env_name)
            if n_action_steps is None:
                n_action_steps = action_horizon or 20
            wrapper_configs = rollout_mod.WrapperConfigs(
                video=rollout_mod.VideoConfig(
                    video_dir=None,
                    max_episode_steps=1440,
                    overlay_text=False,
                ),
                multistep=rollout_mod.MultiStepConfig(
                    n_action_steps=int(n_action_steps),
                    max_episode_steps=1440,
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
            try:
                outer_max_steps = max(1, (1440 + int(n_action_steps) - 1) // int(n_action_steps))
                for episode_index in range(1, int(args.n_episodes) + 1):
                    obs, _info = env.reset(seed=int(args.seed))
                    options_ep = {"seed": int(args.seed)}
                    client.reset(options=options_ep)
                    done = False
                    episode_success = False
                    outer_steps = 0
                    last_terminated = False
                    last_truncated = False
                    last_reward = 0.0
                    episode_step_records: list[dict[str, Any]] = []
                    reset_snapshot = helper._collect_env_snapshot(env)
                    started = time.monotonic()
                    while not done and outer_steps < outer_max_steps:
                        action, _action_info = client.get_action(obs, options=options_ep)
                        if not isinstance(action, dict):
                            raise TypeError(
                                "PolicyClient.get_action must return a dict action, got "
                                f"{type(action).__name__}"
                            )
                        obs, reward, term, trunc, step_info = env.step(action)
                        reward_scalar = helper._scalarize_float(reward)
                        last_reward = float(reward_scalar)
                        last_terminated = bool(helper._scalarize_bool(term))
                        last_truncated = bool(helper._scalarize_bool(trunc))
                        done = bool(last_terminated or last_truncated)
                        success_step = bool(helper._extract_success_step(step_info))
                        episode_success = bool(episode_success or success_step)
                        outer_steps += 1
                        step_record: dict[str, Any] = {
                            "episode_index": int(episode_index),
                            "seed": int(args.seed),
                            "outer_step": int(outer_steps),
                            "reward": float(reward_scalar),
                            "terminated": bool(last_terminated),
                            "truncated": bool(last_truncated),
                            "done": bool(done),
                            "success_step": bool(success_step),
                            "episode_success_so_far": bool(episode_success),
                            "action_summary": helper._summarize_action_chunk(action),
                        }
                        step_record.update(helper._collect_env_snapshot(env))
                        intermediate = helper._extract_intermediate_signals(step_info)
                        if intermediate is not None:
                            step_record["intermediate_signals"] = intermediate
                        episode_step_records.append(step_record)
                        _append_jsonl(step_telemetry_path, step_record)
                    failure_reason = helper._episode_failure_reason(
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
                        else helper._failure_stage_guess(episode_step_records)
                    )
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
                    episode_record = {
                        "episode_index": int(episode_index),
                        "seed": int(args.seed),
                        "success": bool(episode_success),
                        "done": bool(done),
                        "terminated": bool(last_terminated),
                        "truncated": bool(last_truncated),
                        "outer_steps": int(outer_steps),
                        "episode_elapsed_seconds": float(time.monotonic() - started),
                        "final_reward": float(last_reward),
                        "failure_reason": failure_reason,
                        "failure_stage_guess": failure_stage_guess,
                        "reset_snapshot": reset_snapshot,
                        "final_snapshot": final_snapshot,
                        "n_success_steps": int(
                            sum(1 for row in episode_step_records if row.get("success_step") is True)
                        ),
                        "step_telemetry_records": int(len(episode_step_records)),
                        "env_registration": env_registration,
                    }
                    episode_records.append(episode_record)
                    _append_jsonl(telemetry_path, episode_record)
            finally:
                with contextlib.suppress(Exception):
                    env.close()
    finally:
        _stop_policy_server(server_proc)
        if server_handle is not None:
            server_handle.close()

    seed_metric = _build_episode_metric(episode_records, int(args.seed))
    blocking_reasons = [] if seed_metric is not None else ["missing_seed_metric"]
    payload = _build_probe_payload(
        status="PASS" if seed_metric is not None else "BLOCK",
        conditioned_checkpoint=conditioned_checkpoint,
        continuation_run_root=continuation_run_root,
        baseline_probe=baseline_probe,
        baseline_payload=baseline_payload,
        seed=int(args.seed),
        n_episodes=int(args.n_episodes),
        telemetry_path=telemetry_path,
        output_json=output_json,
        seed_metric=seed_metric,
        blocking_reasons=blocking_reasons,
        runtime_log_dir=runtime_log_dir,
        server_log=server_log,
    )
    _write_json(output_json, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="36c_gr00t_first_subgoal_rollout_probe.py",
        description="Run a seed-scoped first-subgoal rollout probe for a GR00T conditioned checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--conditioned-checkpoint", type=Path, required=True)
    parser.add_argument("--continuation-run-root", type=Path, default=None)
    parser.add_argument("--baseline-probe", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument("--runtime-log-dir", type=Path, required=True)
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        default=False,
        help="Abort when the process appears to be running under sudo/root.",
    )
    parser.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPISODES)
    parser.add_argument(
        "--dry-run-smoke",
        action="store_true",
        default=False,
        help="Emit smoke_gate.json without loading a model or running rollout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        payload = run_probe(args)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if str(payload.get("status", "")).upper() == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
