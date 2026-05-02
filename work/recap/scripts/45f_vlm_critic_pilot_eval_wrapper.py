#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast


_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
MODEL_PATH = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
EMBODIMENT_TAG = "UNITREE_G1"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5760

N_EPISODES = 1
MAX_POLICY_STEPS = 10
MAX_EPISODE_STEPS = 1440
N_ACTION_STEPS_CONFIG = 30

MUJOCO_GL = ""
ONSCREEN = False
OFFSCREEN = True

ITER_TAG_PREFIX = "pilot_eval"
RUNTIME_LOG_DIR = "agent/runtime_logs/45f_vlm_critic_pilot_eval"
SUMMARY_JSON = "agent/artifacts/vlm_critic_relabel/45f_pilot_eval_summary.json"

SERVER_READY_TIMEOUT_S = 180.0
SERVER_PING_INTERVAL_S = 1.0
SERVER_PING_TIMEOUT_MS = 2000
SERVER_TERMINATE_TIMEOUT_S = 15.0
TOTAL_TIMEOUT_S = 1800.0


def _repo_root() -> Path:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "repo_root")
    return cast(Path, fn(from_path=__file__))


def _maybe_reexec_into_wbc_venv(repo_root: Path) -> None:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "maybe_reexec_into_wbc_venv")
    fn(repo_root)


@contextlib.contextmanager
def _tee_stdio(log_path: Path, *, header: str) -> Iterator[None]:
    mod = importlib.import_module("work.demo_utils.tee")
    fn = getattr(mod, "tee_stdio")
    with fn(Path(log_path), header=str(header)):
        yield


def _normalize_client_host(host: str) -> str:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "normalize_client_host")
    return str(fn(str(host)))


def _is_tcp_port_listening(host: str, port: int, timeout_s: float = 0.2) -> bool:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "is_tcp_port_listening")
    return bool(fn(str(host), int(port), timeout_s=float(timeout_s)))


def _make_policy_client(host: str, port: int, timeout_ms: int) -> Any:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "make_policy_client")
    return fn(host=str(host), port=int(port), timeout_ms=int(timeout_ms))


def _safe_ping(client: Any, timeout_ms: int) -> bool:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "safe_ping")
    return bool(fn(client, int(timeout_ms)))


def _safe_kill_server(client: Any, timeout_ms: int) -> bool:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "safe_kill_server")
    return bool(fn(client, int(timeout_ms)))


def _spawn_server_subprocess(
    cmd: Sequence[str], *, log_path: Path, cwd: Path, env: dict[str, str]
) -> subprocess.Popen[str]:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "spawn_server_subprocess")
    return cast(
        subprocess.Popen[str],
        fn(list(cmd), log_path=Path(log_path), cwd=Path(cwd), env=dict(env)),
    )


def _terminate_process(proc: subprocess.Popen[str], timeout_s: float) -> None:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "terminate_process")
    fn(proc, float(timeout_s))


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _server_error_preview(log_path: Path) -> str | None:
    if not log_path.is_file():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None

    interesting = [
        line.strip()
        for line in lines
        if line.strip()
        and any(
            token in line
            for token in (
                "ERROR",
                "Traceback",
                "ValueError",
                "RuntimeError",
                "AssertionError",
                "Exception",
            )
        )
    ]
    if interesting:
        return " | ".join(interesting[-3:])
    if lines:
        return " | ".join(line.strip() for line in lines[-3:] if line.strip()) or None
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _path_is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _resolve_repo_output_path(repo_root: Path, raw: str, *, kind: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not _path_is_within(repo_root, path):
        raise ValueError(f"{kind} must stay inside repo root: {path}")
    return path


def _resolve_model_path(repo_root: Path, raw: str) -> str:
    text = str(raw).strip()
    if not text:
        raise ValueError("model_path must not be empty")
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    if (
        text.startswith("agent/")
        or text.startswith("work/")
        or text.startswith("submodules/")
    ):
        return str((repo_root / path).resolve())
    return text


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object in {path}:{idx}, got {type(obj).__name__}"
                )
            records.append(dict(obj))
    return records


def _summarize_dataset(
    episodes_jsonl: Path,
    transitions_jsonl: Path | None = None,
) -> dict[str, Any]:
    mod = importlib.import_module("work.recap.collector")
    fn = getattr(mod, "summarize_existing_dataset_success")
    stats = fn(episodes_jsonl, transitions_jsonl)
    if not isinstance(stats, dict):
        raise TypeError(f"Expected dataset summary dict, got {type(stats).__name__}")
    return cast(dict[str, Any], stats)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="45f_vlm_critic_pilot_eval_wrapper.py",
        description=(
            "Thin current-repo wrapper around run_gr00t_server.py + "
            "31_recap_collect_rollouts.py that emits one summary JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--env-name", type=str, default=ENV_NAME)
    parser.add_argument("--embodiment-tag", type=str, default=EMBODIMENT_TAG)
    parser.add_argument("--server-host", type=str, default=SERVER_HOST)
    parser.add_argument("--port", type=int, default=int(SERVER_PORT))
    parser.add_argument("--n-episodes", type=int, default=int(N_EPISODES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-prompt-prefix",
        type=str,
        default="",
        help=(
            "Literal prompt prefix forwarded to rollout collection. The wrapper does "
            "not assign semantic labels such as zero/positive/negative."
        ),
    )
    parser.add_argument("--max-policy-steps", type=int, default=int(MAX_POLICY_STEPS))
    parser.add_argument("--max-episode-steps", type=int, default=int(MAX_EPISODE_STEPS))
    parser.add_argument(
        "--n-action-steps-config", type=int, default=int(N_ACTION_STEPS_CONFIG)
    )
    parser.add_argument("--mujoco-gl", type=str, default=MUJOCO_GL)
    parser.add_argument(
        "--iter-tag",
        type=str,
        default="",
        help=(
            "Eval tag used by 31_recap_collect_rollouts.py. If empty, the wrapper "
            "generates a fresh tag."
        ),
    )
    parser.add_argument("--runtime-log-dir", type=str, default=RUNTIME_LOG_DIR)
    parser.add_argument("--summary-json", type=str, default=SUMMARY_JSON)
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help=(
            "Explicit batch/run identifier recorded into the summary payload. If empty, "
            "the wrapper falls back to iter_tag."
        ),
    )
    parser.add_argument(
        "--episodes-jsonl",
        type=str,
        default="",
        help=(
            "If provided, skip rollout execution and recompute the wrapper summary from "
            "the existing dataset instead."
        ),
    )
    parser.add_argument(
        "--transitions-jsonl",
        type=str,
        default="",
        help="Optional transitions.jsonl paired with --episodes-jsonl for summary repair.",
    )
    parser.add_argument(
        "--server-ready-timeout-s",
        type=float,
        default=float(SERVER_READY_TIMEOUT_S),
    )
    parser.add_argument(
        "--server-ping-interval-s",
        type=float,
        default=float(SERVER_PING_INTERVAL_S),
    )
    parser.add_argument(
        "--server-ping-timeout-ms",
        type=int,
        default=int(SERVER_PING_TIMEOUT_MS),
    )
    parser.add_argument(
        "--server-terminate-timeout-s",
        type=float,
        default=float(SERVER_TERMINATE_TIMEOUT_S),
    )
    parser.add_argument("--total-timeout-s", type=float, default=float(TOTAL_TIMEOUT_S))
    parser.add_argument(
        "--server-script",
        type=str,
        default="submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
    )
    parser.add_argument(
        "--rollout-script",
        type=str,
        default="work/recap/scripts/31_recap_collect_rollouts.py",
    )

    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is None:
        parser.add_argument("--onscreen", action="store_true", default=bool(ONSCREEN))
        parser.add_argument("--offscreen", action="store_true", default=bool(OFFSCREEN))
    else:
        parser.add_argument(
            "--onscreen",
            action=bool_action,
            default=bool(ONSCREEN),
            help="Enable onscreen viewer in the rollout env.",
        )
        parser.add_argument(
            "--offscreen",
            action=bool_action,
            default=bool(OFFSCREEN),
            help="Enable offscreen rendering/video capture in the rollout env.",
        )
    return parser


def _build_server_cmd(
    args: argparse.Namespace, server_script: Path, model_path: str
) -> list[str]:
    return [
        sys.executable,
        str(server_script),
        "--model-path",
        str(model_path),
        "--embodiment-tag",
        str(args.embodiment_tag),
        "--use-sim-policy-wrapper",
        "--host",
        str(args.server_host),
        "--port",
        str(int(args.port)),
    ]


def _build_collect_cmd(
    args: argparse.Namespace,
    rollout_script: Path,
    *,
    model_path: str,
    iter_tag: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(rollout_script),
        "--iter-tag",
        str(iter_tag),
        "--env-name",
        str(args.env_name),
        "--model-path",
        str(model_path),
        "--embodiment-tag",
        str(args.embodiment_tag),
        "--server-host",
        str(args.server_host),
        "--server-port",
        str(int(args.port)),
        "--n-episodes",
        str(int(args.n_episodes)),
        "--max-policy-steps",
        str(int(args.max_policy_steps)),
        "--max-episode-steps",
        str(int(args.max_episode_steps)),
        "--n-action-steps-config",
        str(int(args.n_action_steps_config)),
        "--seed",
        str(int(args.seed)),
        "--policy-prompt-prefix",
        str(args.policy_prompt_prefix),
        "--total-timeout-s",
        str(float(args.total_timeout_s)),
    ]
    mujoco_gl = str(args.mujoco_gl).strip()
    if mujoco_gl:
        cmd.extend(["--mujoco-gl", mujoco_gl])
    cmd.extend(["--onscreen" if bool(args.onscreen) else "--no-onscreen"])
    cmd.extend(["--offscreen" if bool(args.offscreen) else "--no-offscreen"])
    return cmd


def main() -> int:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        try:
            _build_parser().parse_args()
        except SystemExit as exc:
            return int(getattr(exc, "code", 0) or 0)
        return 0

    repo_root = _repo_root()
    _maybe_reexec_into_wbc_venv(repo_root)

    args = _build_parser().parse_args()
    if int(args.n_episodes) <= 0:
        raise ValueError(f"n_episodes must be > 0, got {args.n_episodes}")
    if int(args.max_policy_steps) <= 0:
        raise ValueError(f"max_policy_steps must be > 0, got {args.max_policy_steps}")
    if int(args.max_episode_steps) <= 0:
        raise ValueError(f"max_episode_steps must be > 0, got {args.max_episode_steps}")

    runtime_log_dir = _resolve_repo_output_path(
        repo_root, str(args.runtime_log_dir), kind="runtime_log_dir"
    )
    summary_json = _resolve_repo_output_path(
        repo_root, str(args.summary_json), kind="summary_json"
    )
    server_script = _resolve_repo_output_path(
        repo_root, str(args.server_script), kind="server_script"
    )
    rollout_script = _resolve_repo_output_path(
        repo_root, str(args.rollout_script), kind="rollout_script"
    )
    model_path = _resolve_model_path(repo_root, str(args.model_path))
    iter_tag = str(args.iter_tag).strip() or f"{ITER_TAG_PREFIX}_{_timestamp()}"
    run_id = str(args.run_id).strip() or str(iter_tag)
    summary_only = bool(str(args.episodes_jsonl).strip())
    if summary_only:
        episodes_jsonl = _resolve_repo_output_path(
            repo_root,
            str(args.episodes_jsonl),
            kind="episodes_jsonl",
        )
        dataset_dir = episodes_jsonl.parent
        if str(args.transitions_jsonl).strip():
            transitions_jsonl = _resolve_repo_output_path(
                repo_root,
                str(args.transitions_jsonl),
                kind="transitions_jsonl",
            )
        else:
            transitions_jsonl = dataset_dir / "transitions.jsonl"
    else:
        dataset_dir = repo_root / "agent/artifacts/recap_datasets" / iter_tag
        episodes_jsonl = dataset_dir / "episodes.jsonl"
        transitions_jsonl = dataset_dir / "transitions.jsonl"
    collect_runtime_log = repo_root / "agent/runtime_logs" / iter_tag / "collect.log"
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    if not summary_only and episodes_jsonl.exists():
        raise FileExistsError(
            f"Refusing to append into existing eval output: {episodes_jsonl}"
        )

    tag = f"{iter_tag}_{_timestamp()}"
    wrapper_log = runtime_log_dir / f"45f_wrapper_{tag}.log"
    server_log = runtime_log_dir / f"45f_server_{tag}.log"
    host_for_client = _normalize_client_host(str(args.server_host))
    server_cmd = _build_server_cmd(args, server_script, model_path)
    collect_cmd = _build_collect_cmd(
        args,
        rollout_script,
        model_path=model_path,
        iter_tag=iter_tag,
    )

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    mujoco_gl = str(args.mujoco_gl).strip()
    if mujoco_gl:
        env["MUJOCO_GL"] = mujoco_gl
        if mujoco_gl.lower() == "egl":
            env.setdefault("PYOPENGL_PLATFORM", "egl")

    wrapper_error: str | None = None
    server_started = False
    collect_returncode: int | None = None
    summary_stats = {
        "episodes": 0,
        "success_count": 0,
        "success_rate": 0.0,
    }
    server_proc: subprocess.Popen[str] | None = None
    client: Any | None = None

    with _tee_stdio(wrapper_log, header="45f_vlm_critic_pilot_eval_wrapper"):
        print("[INFO] ts:", _dt.datetime.now().isoformat(timespec="seconds"))
        print("[INFO] wrapper:", "45f_vlm_critic_pilot_eval_wrapper.py")
        print("[INFO] repo_root:", repo_root)
        print("[INFO] python:", sys.executable)
        print("[INFO] model_path:", model_path)
        print("[INFO] policy_prompt_prefix:", repr(str(args.policy_prompt_prefix)))
        print("[INFO] run_id:", run_id)
        print("[INFO] iter_tag:", iter_tag)
        print("[INFO] summary_json:", summary_json)
        print("[INFO] wrapper_log:", wrapper_log)
        print("[INFO] server_log:", server_log)
        print("[INFO] collect_runtime_log:", collect_runtime_log)
        print("[INFO] server_cmd:", json.dumps(server_cmd, ensure_ascii=True))
        print("[INFO] collect_cmd:", json.dumps(collect_cmd, ensure_ascii=True))
        try:
            if summary_only:
                if not episodes_jsonl.is_file():
                    raise FileNotFoundError(f"episodes_jsonl_missing: {episodes_jsonl}")
                if transitions_jsonl is not None and not transitions_jsonl.is_file():
                    raise FileNotFoundError(
                        f"transitions_jsonl_missing: {transitions_jsonl}"
                    )
                summary_stats = _summarize_dataset(episodes_jsonl, transitions_jsonl)
                print("[INFO] summary_mode: existing_dataset_repair")
                print(
                    "[INFO] episodes summary:",
                    json.dumps(summary_stats, ensure_ascii=True, sort_keys=True),
                )
            else:
                if _is_tcp_port_listening(host_for_client, int(args.port)):
                    raise RuntimeError(
                        f"port_in_use: refusing to reuse occupied port {host_for_client}:{int(args.port)}"
                    )

                server_proc = _spawn_server_subprocess(
                    server_cmd,
                    log_path=server_log,
                    cwd=repo_root,
                    env=env,
                )
                print(
                    f"[INFO] spawned_server pid={server_proc.pid} host={host_for_client} port={int(args.port)}"
                )

                client = _make_policy_client(
                    host=host_for_client,
                    port=int(args.port),
                    timeout_ms=int(args.server_ping_timeout_ms),
                )
                t0 = time.monotonic()
                while True:
                    if _safe_ping(client, int(args.server_ping_timeout_ms)):
                        server_started = True
                        print("[INFO] server_ready: ping ok")
                        break
                    if server_proc.poll() is not None:
                        raise RuntimeError(
                            "server_exited_early: "
                            + f"rc={server_proc.returncode} preview={_server_error_preview(server_log)!r}"
                        )
                    elapsed = time.monotonic() - t0
                    if elapsed > float(args.server_ready_timeout_s):
                        raise TimeoutError(
                            "server_ready_timeout: "
                            + f"waited>{float(args.server_ready_timeout_s):.1f}s "
                            + f"preview={_server_error_preview(server_log)!r}"
                        )
                    time.sleep(float(args.server_ping_interval_s))

                collect_proc = subprocess.run(
                    collect_cmd,
                    cwd=str(repo_root),
                    env=env,
                    check=False,
                )
                collect_returncode = int(collect_proc.returncode)
                if collect_returncode != 0:
                    wrapper_error = f"rollout_failed: returncode={collect_returncode}"

                if episodes_jsonl.is_file():
                    summary_stats = _summarize_dataset(
                        episodes_jsonl, transitions_jsonl
                    )
                    print(
                        "[INFO] episodes summary:",
                        json.dumps(summary_stats, ensure_ascii=True, sort_keys=True),
                    )
                elif wrapper_error is None:
                    wrapper_error = f"episodes_jsonl_missing: {episodes_jsonl}"

            if wrapper_error is None and int(summary_stats["episodes"]) != int(
                args.n_episodes
            ):
                wrapper_error = (
                    "episodes_count_mismatch: "
                    + f"expected={int(args.n_episodes)} actual={int(summary_stats['episodes'])}"
                )
        except Exception as exc:
            wrapper_error = f"wrapper_exception: {type(exc).__name__}: {exc}"
            if episodes_jsonl.is_file():
                try:
                    summary_stats = _summarize_dataset(
                        episodes_jsonl, transitions_jsonl
                    )
                except Exception:
                    summary_stats = {
                        "episodes": 0,
                        "success_count": 0,
                        "success_rate": 0.0,
                    }
        finally:
            if client is not None:
                try:
                    _safe_kill_server(client, int(args.server_ping_timeout_ms))
                except Exception:
                    pass
            if server_proc is not None:
                _terminate_process(server_proc, float(args.server_terminate_timeout_s))

    payload: dict[str, Any] = {
        "wrapper": "45f_vlm_critic_pilot_eval_wrapper.py",
        "wrapper_status": "ok" if wrapper_error is None else "blocked",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "run_id": str(run_id),
        "model_path": str(model_path),
        "policy_prompt_prefix": str(args.policy_prompt_prefix),
        "iter_tag": str(iter_tag),
        "env_name": str(args.env_name),
        "embodiment_tag": str(args.embodiment_tag),
        "server_host": str(host_for_client),
        "port": int(args.port),
        "requested_episodes": int(args.n_episodes),
        "episodes": int(summary_stats["episodes"]),
        "success_count": int(summary_stats["success_count"]),
        "success_rate": float(summary_stats["success_rate"]),
        "episodes_jsonl": str(episodes_jsonl),
        "transitions_jsonl": (
            str(transitions_jsonl) if transitions_jsonl is not None else None
        ),
        "runtime_log": str(collect_runtime_log),
        "wrapper_log": str(wrapper_log),
        "server_log": str(server_log),
        "summary_json": str(summary_json),
        "server_started": bool(server_started),
        "collect_returncode": collect_returncode,
        "summary_mode": "existing_dataset_repair" if summary_only else "fresh_rollout",
        "success_inference": summary_stats.get(
            "success_inference",
            "info.success -> final_info.success -> positive reward fallback",
        ),
        "success_count_recorded": int(
            summary_stats.get("success_count_recorded", summary_stats["success_count"])
        ),
        "success_count_recomputed": int(
            summary_stats.get(
                "success_count_recomputed", summary_stats["success_count"]
            )
        ),
        "repaired_episode_count": int(summary_stats.get("repaired_episode_count", 0)),
        "reward_fallback_episode_count": int(
            summary_stats.get("reward_fallback_episode_count", 0)
        ),
        "server_script": str(server_script),
        "rollout_script": str(rollout_script),
        "server_cmd": server_cmd,
        "collect_cmd": collect_cmd,
        "error": wrapper_error,
    }
    _write_json(summary_json, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
