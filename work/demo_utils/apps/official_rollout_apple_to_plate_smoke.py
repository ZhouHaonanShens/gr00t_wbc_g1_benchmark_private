#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"

MODEL_PATH = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
EMBODIMENT_TAG = "UNITREE_G1"

SERVER_PYTHON = ""

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5555

N_EPISODES = 50
N_ENVS = 1
MAX_EPISODE_STEPS = 1440
N_ACTION_STEPS = 30

MUJOCO_GL = ""

RUNTIME_LOGS_REL = "agent/runtime_logs/official_task_eval"
VIDEO_ARCHIVE_DIR = "agent/artifacts/videos"

SERVER_READY_TIMEOUT_S = 600
SERVER_PING_TIMEOUT_MS = 2000
SERVER_PING_INTERVAL_S = 1.0

TOTAL_TIMEOUT_S = 600


_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


def _repo_root() -> Path:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "repo_root")
    return cast(Path, fn(from_path=__file__))


def _maybe_reexec_into_wbc_venv(repo_root: Path) -> None:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "maybe_reexec_into_wbc_venv")
    fn(repo_root)


def _ensure_dirs(repo_root: Path) -> tuple[Path, Path, Path, Path]:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "ensure_dirs")
    runtime_dir, artifacts_videos = fn(
        repo_root=repo_root,
        runtime_logs_rel=RUNTIME_LOGS_REL,
        artifacts_videos_rel=str(VIDEO_ARCHIVE_DIR),
    )
    runtime_dir = cast(Path, runtime_dir)
    artifacts_videos = cast(Path, artifacts_videos)
    server_log = runtime_dir / "00_server.log"
    client_log = runtime_dir / "10_rollout_apple_to_plate_smoke.log"
    return runtime_dir, artifacts_videos, server_log, client_log


@contextlib.contextmanager
def _tee_stdio(log_path: Path, *, header: str) -> Iterator[None]:
    mod = importlib.import_module("work.demo_utils.tee")
    fn = getattr(mod, "tee_stdio")
    with fn(log_path, header=str(header)):
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
    cmd: Sequence[str], *, log_path: Path, cwd: Path, env: Mapping[str, str]
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


def _make_video_dir(*, env_name: str, n_action_steps: int) -> Path:
    mod = importlib.import_module("work.demo_utils.videos")
    fn = getattr(mod, "make_video_dir")
    return cast(Path, fn(env_name=str(env_name), n_action_steps=int(n_action_steps)))


def _archive_video_dir(*, video_dir: Path | None, archive_root: Path) -> Path | None:
    mod = importlib.import_module("work.demo_utils.videos")
    fn = getattr(mod, "archive_video_dir")
    return cast(Path | None, fn(video_dir=video_dir, archive_root=archive_root))


def _apply_env(mujoco_gl: str) -> None:
    v = str(mujoco_gl or "").strip()
    if not v:
        return
    os.environ["MUJOCO_GL"] = v
    if v.lower() == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _server_entrypoint(repo_root: Path) -> Path:
    return repo_root / "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"


def _build_server_cmd(args: argparse.Namespace, repo_root: Path) -> list[str]:
    server_py = _server_entrypoint(repo_root)
    server_python = str(getattr(args, "server_python", "") or "").strip()
    exe = server_python if server_python else sys.executable
    return [
        exe,
        str(server_py),
        "--model-path",
        str(args.model_path),
        "--embodiment-tag",
        str(args.embodiment_tag),
        "--use-sim-policy-wrapper",
        "--host",
        str(args.server_host),
        "--port",
        str(int(args.server_port)),
    ]


def _ensure_server_ready(
    args: argparse.Namespace, repo_root: Path, server_log: Path
) -> tuple[Any, subprocess.Popen[str] | None, bool]:
    host_for_client = _normalize_client_host(str(args.server_host))
    client = _make_policy_client(
        host=host_for_client,
        port=int(args.server_port),
        timeout_ms=int(args.server_ping_timeout_ms),
    )

    print(f"[INFO] policy server target: {host_for_client}:{int(args.server_port)}")
    if _safe_ping(client, int(args.server_ping_timeout_ms)):
        print("[INFO] ping ok (reuse existing server)")
        return client, None, False

    if not bool(args.spawn_server_if_missing):
        raise RuntimeError(
            "no responsive server found (ping failed) and spawn_server_if_missing is disabled"
        )

    if _is_tcp_port_listening(host_for_client, int(args.server_port)):
        raise RuntimeError(
            "port is already occupied but PolicyClient.ping() failed; refuse to kill unknown process. "
            "Try changing --server-port or stop the other service."
        )

    server_py = _server_entrypoint(repo_root)
    if not server_py.is_file():
        raise FileNotFoundError(f"missing server entrypoint: {server_py}")

    cmd = _build_server_cmd(args, repo_root)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("PYTHONPATH", None)
    if args.mujoco_gl:
        env["MUJOCO_GL"] = str(args.mujoco_gl)
        if str(args.mujoco_gl).lower() == "egl":
            env.setdefault("PYOPENGL_PLATFORM", "egl")

    print("[INFO] spawning GR00T server subprocess...")
    proc = _spawn_server_subprocess(cmd, log_path=server_log, cwd=repo_root, env=env)
    print(f"[INFO] server subprocess pid={proc.pid} log={server_log}")

    t0 = time.monotonic()
    last_note = 0.0
    while True:
        if _safe_ping(client, int(args.server_ping_timeout_ms)):
            print("[INFO] ping ok")
            return client, proc, True

        if proc.poll() is not None:
            raise RuntimeError(
                f"server subprocess exited early rc={proc.returncode}; see {server_log}"
            )

        dt = time.monotonic() - t0
        if dt > float(args.server_ready_timeout_s):
            raise TimeoutError(
                f"timeout waiting for ping ok after {int(dt)}s; see {server_log}"
            )

        if dt - last_note >= 5.0:
            print(f"[INFO] waiting for server ready... {int(dt)}s")
            last_note = dt
        time.sleep(float(args.server_ping_interval_s))


def _install_alarm_timeout(timeout_s: float | None) -> None:
    if timeout_s is None:
        return
    try:
        t = int(float(timeout_s))
    except Exception:
        return
    if t <= 0:
        return
    if not hasattr(signal, "SIGALRM"):
        return

    def _handler(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"Timed out after {t}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(t)


def _clear_alarm_timeout() -> None:
    if hasattr(signal, "SIGALRM"):
        try:
            signal.alarm(0)
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="official_rollout_apple_to_plate_smoke.py",
        description=(
            "One-click official AppleToPlateDC rollout (server + rollout + video archive evidence)."
        ),
    )

    p.add_argument("--env-name", type=str, default=ENV_NAME)
    p.add_argument("--model-path", type=str, default=MODEL_PATH)
    p.add_argument("--embodiment-tag", type=str, default=EMBODIMENT_TAG)
    p.add_argument("--server-host", type=str, default=SERVER_HOST)
    p.add_argument("--server-port", type=int, default=SERVER_PORT)
    p.add_argument(
        "--server-python",
        type=str,
        default=SERVER_PYTHON,
        help="Optional: explicit python executable for the server subprocess.",
    )
    p.add_argument("--mujoco-gl", type=str, default=MUJOCO_GL)

    p.add_argument("--n-episodes", type=int, default=int(N_EPISODES))
    p.add_argument("--n-envs", type=int, default=int(N_ENVS))
    p.add_argument("--max-episode-steps", type=int, default=int(MAX_EPISODE_STEPS))
    p.add_argument("--n-action-steps", type=int, default=int(N_ACTION_STEPS))

    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is None:
        p.add_argument(
            "--spawn-server-if-missing",
            action="store_true",
            default=True,
            help="Spawn server subprocess if ping fails.",
        )
        p.add_argument(
            "--kill-server-on-exit",
            action="store_true",
            default=False,
            help="Kill server if this script started it.",
        )
    else:
        p.add_argument(
            "--spawn-server-if-missing",
            action=bool_action,
            default=True,
            help="Spawn server subprocess if ping fails.",
        )
        p.add_argument(
            "--kill-server-on-exit",
            action=bool_action,
            default=False,
            help="Kill server if this script started it.",
        )

    p.add_argument(
        "--server-ready-timeout-s",
        type=float,
        default=float(SERVER_READY_TIMEOUT_S),
    )
    p.add_argument(
        "--server-ping-timeout-ms",
        type=int,
        default=int(SERVER_PING_TIMEOUT_MS),
    )
    p.add_argument(
        "--server-ping-interval-s",
        type=float,
        default=float(SERVER_PING_INTERVAL_S),
    )

    p.add_argument(
        "--total-timeout-s",
        type=float,
        default=float(TOTAL_TIMEOUT_S),
        help="Hard timeout fuse (best-effort) for the whole script.",
    )
    return p


def main() -> int:
    repo_root = _repo_root()
    _maybe_reexec_into_wbc_venv(repo_root)

    args = _build_parser().parse_args()
    _apply_env(str(args.mujoco_gl))

    _runtime_dir, artifacts_videos, server_log, client_log = _ensure_dirs(repo_root)
    with _tee_stdio(client_log, header="official_rollout_apple_to_plate"):
        _install_alarm_timeout(
            float(args.total_timeout_s) if args.total_timeout_s else None
        )
        started_by_me = False
        proc: subprocess.Popen[str] | None = None
        client: Any | None = None
        video_dir: Path | None = None
        try:
            print("[INFO] ts:", _dt.datetime.now().isoformat(timespec="seconds"))
            print("[INFO] python:", sys.version.replace("\n", " "))
            print("[INFO] env_name:", str(args.env_name))
            print(
                "[INFO] n_episodes:", int(args.n_episodes), "n_envs:", int(args.n_envs)
            )
            print(
                "[INFO] max_episode_steps:",
                int(args.max_episode_steps),
                "n_action_steps:",
                int(args.n_action_steps),
            )

            client, proc, started_by_me = _ensure_server_ready(
                args, repo_root, server_log
            )
            assert client is not None
            modality_cfg = client.get_modality_config()
            if "action" in modality_cfg:
                action_keys = list(
                    getattr(modality_cfg["action"], "modality_keys", []) or []
                )
                action_delta = list(
                    getattr(modality_cfg["action"], "delta_indices", []) or []
                )
                print("[INFO] server action_keys:", [str(k) for k in action_keys])
                print("[INFO] server action_horizon:", len(action_delta))
                if action_delta and int(args.n_action_steps) != int(len(action_delta)):
                    print(
                        "[WARN] n_action_steps != server action_horizon:",
                        int(args.n_action_steps),
                        "vs",
                        int(len(action_delta)),
                    )

            rollout_mod = importlib.import_module("gr00t.eval.rollout_policy")
            WrapperConfigs = getattr(rollout_mod, "WrapperConfigs")
            VideoConfig = getattr(rollout_mod, "VideoConfig")
            MultiStepConfig = getattr(rollout_mod, "MultiStepConfig")
            run_rollout_gymnasium_policy = getattr(
                rollout_mod, "run_rollout_gymnasium_policy"
            )
            create_gr00t_sim_policy = getattr(rollout_mod, "create_gr00t_sim_policy")
            get_embodiment_tag_from_env_name = getattr(
                importlib.import_module("gr00t.eval.sim.env_utils"),
                "get_embodiment_tag_from_env_name",
            )

            video_dir = _make_video_dir(
                env_name=str(args.env_name), n_action_steps=int(args.n_action_steps)
            )
            wrapper_configs = WrapperConfigs(
                video=VideoConfig(
                    video_dir=str(video_dir),
                    max_episode_steps=int(args.max_episode_steps),
                    overlay_text=True,
                ),
                multistep=MultiStepConfig(
                    n_action_steps=int(args.n_action_steps),
                    max_episode_steps=int(args.max_episode_steps),
                    terminate_on_success=True,
                ),
            )

            embodiment_tag = get_embodiment_tag_from_env_name(str(args.env_name))
            policy = create_gr00t_sim_policy(
                model_path="",
                embodiment_tag=embodiment_tag,
                policy_client_host=_normalize_client_host(str(args.server_host)),
                policy_client_port=int(args.server_port),
            )

            results = run_rollout_gymnasium_policy(
                env_name=str(args.env_name),
                policy=policy,
                wrapper_configs=wrapper_configs,
                n_episodes=int(args.n_episodes),
                n_envs=int(args.n_envs),
            )
            print("results:", results)
            try:
                np = importlib.import_module("numpy")
                print("success rate:", float(np.mean(results[1])))
            except Exception:
                pass
            print("Video saved to:", str(video_dir))

            archived = _archive_video_dir(
                video_dir=video_dir, archive_root=artifacts_videos
            )
            print("[INFO] archived video dir:", archived)

            return 0
        finally:
            _clear_alarm_timeout()
            if client is not None and started_by_me and bool(args.kill_server_on_exit):
                print("[INFO] cleanup: stopping server started by this script")
                try:
                    _safe_kill_server(client, int(args.server_ping_timeout_ms))
                except Exception:
                    pass
                if proc is not None:
                    _terminate_process(proc, timeout_s=10.0)


if __name__ == "__main__":
    raise SystemExit(main())
