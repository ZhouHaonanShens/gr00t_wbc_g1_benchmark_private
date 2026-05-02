#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from collections.abc import Iterator
from typing import Any, cast

_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
MODEL_PATH = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5555

N_ACTION_STEPS = 30
MAX_EPISODE_STEPS = 5000
N_EPISODES = 1

MUJOCO_GL = "glfw"
ONSCREEN = True
OFFSCREEN = True

RENDER_CAMERA: str | None = None

VIDEO_ARCHIVE_DIR = "agent/artifacts/videos"

RUN_MODE = "server_client"
TASK_PROMPT_OVERRIDE: str | None = None
KEEP_OPEN_AFTER_DONE = True

ENV_REGISTRY_PREFIX = "gr00tlocomanip_g1_sim/"

SERVER_READY_TIMEOUT_S = 600
SERVER_PING_TIMEOUT_MS = 2000
SERVER_KILL_TIMEOUT_S = 10
SERVER_PING_INTERVAL_S = 1.0


def _repo_root() -> Path:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "repo_root")
    return fn(from_path=__file__)


def _wbc_venv_python(repo_root: Path) -> Path:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "wbc_venv_python")
    return fn(repo_root)


def _maybe_reexec_into_wbc_venv(repo_root: Path) -> None:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "maybe_reexec_into_wbc_venv")
    fn(repo_root)


@contextlib.contextmanager
def _tee_stdio(log_path: Path) -> Iterator[None]:
    mod = importlib.import_module("work.demo_utils.tee")
    fn = getattr(mod, "tee_stdio")
    with fn(log_path, header="demo_g1_vla_live.py start"):
        yield


def _parse_render_camera(s: str) -> str | None:
    v = s.strip()
    if not v:
        return None
    if v.lower() in {"none", "null", "free"}:
        return None
    return v


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="demo_g1_vla_live.py",
        description="G1 live demo client/server launcher (Prompt20 live demo).",
    )
    p.add_argument("--env-name", type=str, default=ENV_NAME)
    p.add_argument("--model-path", type=str, default=MODEL_PATH)
    p.add_argument("--server-host", type=str, default=SERVER_HOST)
    p.add_argument("--server-port", type=int, default=SERVER_PORT)
    p.add_argument("--n-action-steps", type=int, default=N_ACTION_STEPS)
    p.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    p.add_argument(
        "--n-episodes",
        type=int,
        default=N_EPISODES,
        help="How many episodes to run. 0 means run forever (until Ctrl+C).",
    )
    p.add_argument("--mujoco-gl", type=str, default=MUJOCO_GL)

    screen = p.add_mutually_exclusive_group()
    screen.add_argument(
        "--onscreen",
        action="store_true",
        help="Prefer onscreen rendering (typically MUJOCO_GL=glfw).",
    )
    screen.add_argument(
        "--offscreen",
        action="store_true",
        help="Prefer offscreen rendering (typically MUJOCO_GL=egl).",
    )

    p.add_argument(
        "--render-camera",
        type=_parse_render_camera,
        default=RENDER_CAMERA,
        help=(
            "Onscreen viewer camera. Use 'none'/'free' for free camera. "
            "Examples: robot0_rs_tppview, robot0_oak_egoview."
        ),
    )

    p.add_argument(
        "--video-archive-dir",
        type=str,
        default=VIDEO_ARCHIVE_DIR,
        help="Where to archive rollout video dirs (relative to repo root by default).",
    )
    p.add_argument(
        "--mode",
        "--run-mode",
        dest="run_mode",
        type=str,
        default=RUN_MODE,
        choices=("server_client", "client", "server", "noop", "ping-only", "list-envs"),
        help=(
            "Run mode. Task3 wires server lifecycle (spawn/reuse/ready/kill). "
            "In 'server_client'/'client', run a single-env rollout for live viewing. "
            "Use 'list-envs' to print registered env IDs."
        ),
    )
    p.add_argument(
        "--server-ready-timeout-s",
        type=int,
        default=SERVER_READY_TIMEOUT_S,
        help="Timeout waiting for PolicyClient.ping() to succeed.",
    )
    p.add_argument(
        "--server-ping-timeout-ms",
        type=int,
        default=SERVER_PING_TIMEOUT_MS,
        help="ZMQ send/recv timeout for ping/kill endpoints.",
    )
    p.add_argument(
        "--server-kill-timeout-s",
        type=int,
        default=SERVER_KILL_TIMEOUT_S,
        help="Timeout for server kill + process shutdown during cleanup.",
    )
    p.add_argument(
        "--server-ping-interval-s",
        type=float,
        default=SERVER_PING_INTERVAL_S,
        help="Polling interval between ping attempts.",
    )
    p.add_argument(
        "--task-prompt-override",
        type=str,
        default=TASK_PROMPT_OVERRIDE,
        help="Override obs['annotation.human.task_description'] for every policy call (B=1).",
    )
    keep_open = p.add_mutually_exclusive_group()
    keep_open.add_argument(
        "--keep-open-after-done",
        dest="keep_open_after_done",
        action="store_true",
        default=KEEP_OPEN_AFTER_DONE,
        help="Run continuously (default on TTY).",
    )
    keep_open.add_argument(
        "--exit-after-done",
        dest="keep_open_after_done",
        action="store_false",
        help="Exit after the requested episodes (automation/CI friendly).",
    )
    return p


def _maybe_keep_open(keep_open_after_done: bool) -> None:
    if not keep_open_after_done:
        return
    if not sys.stdin.isatty():
        print("[INFO] keep-open requested but stdin is not a TTY; exiting.")
        return

    print("[INFO] keep-open-after-done enabled. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt -> exiting")


def _make_video_dir(*, env_name: str, n_action_steps: int) -> Path:
    mod = importlib.import_module("work.demo_utils.videos")
    fn = getattr(mod, "make_video_dir")
    return fn(env_name=env_name, n_action_steps=int(n_action_steps))


def _archive_video_dir(*, video_dir: Path | None, archive_root: Path) -> Path | None:
    mod = importlib.import_module("work.demo_utils.videos")
    fn = getattr(mod, "archive_video_dir")
    return fn(video_dir=video_dir, archive_root=archive_root)


def _render_once_best_effort(vec_env: object) -> None:
    # Prefer unwrapped env render to keep the onscreen viewer responsive.
    try:
        envs = getattr(vec_env, "envs", None)
        if isinstance(envs, list) and envs:
            e0 = envs[0]
            try:
                getattr(getattr(e0, "unwrapped", e0), "render")()
                return
            except Exception:
                pass
            try:
                getattr(e0, "render")()
                return
            except Exception:
                pass
    except Exception:
        pass

    try:
        call = getattr(vec_env, "call", None)
        if callable(call):
            call("render")
            return
    except Exception:
        pass

    try:
        getattr(vec_env, "render")()
    except Exception:
        pass


def _stop_video_recorder_best_effort(vec_env: object) -> None:
    try:
        envs = getattr(vec_env, "envs", None)
        if not (isinstance(envs, list) and envs):
            return
        e0 = envs[0]
        vr = getattr(e0, "env", None)
        video_recorder = getattr(vr, "video_recorder", None)
        stop = getattr(video_recorder, "stop", None)
        if callable(stop):
            stop()
    except Exception:
        return


def _ensure_dirs(
    repo_root: Path, video_archive_dir: str
) -> tuple[Path, Path, Path, Path]:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "ensure_demo_live_dirs")
    return fn(repo_root, video_archive_dir)


def _list_registered_env_ids(*, prefix: str, log_path: Path) -> list[str]:
    mod = importlib.import_module("work.demo_utils.env_registry")
    fn = getattr(mod, "list_registered_env_ids")
    return fn(
        prefix=prefix,
        log_path=log_path,
        register_modules=("gr00t_wbc.control.envs.robocasa.sync_env",),
    )


def _apply_env(args: argparse.Namespace) -> None:
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = str(args.mujoco_gl)
        if str(args.mujoco_gl).lower() == "egl":
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _normalize_client_host(host: str) -> str:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "normalize_client_host")
    return fn(host)


def _is_tcp_port_listening(host: str, port: int, timeout_s: float = 0.2) -> bool:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "is_tcp_port_listening")
    return bool(fn(host, int(port), timeout_s=float(timeout_s)))


def _make_policy_client(host: str, port: int, timeout_ms: int):
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "make_policy_client")
    return fn(host=host, port=int(port), timeout_ms=int(timeout_ms))


def _configure_policy_client_socket(client, timeout_ms: int) -> None:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "configure_policy_client_socket")
    fn(client, int(timeout_ms))


def _safe_ping(client, timeout_ms: int) -> bool:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "safe_ping")
    return bool(fn(client, int(timeout_ms)))


def _safe_kill_server(client, timeout_ms: int) -> bool:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "safe_kill_server")
    return bool(fn(client, int(timeout_ms)))


def _server_script_path(repo_root: Path) -> Path:
    return repo_root / "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"


def _build_server_cmd(args: argparse.Namespace, repo_root: Path) -> list[str]:
    server_py = _server_script_path(repo_root)
    return [
        sys.executable,
        str(server_py),
        "--model-path",
        str(args.model_path),
        "--embodiment-tag",
        "UNITREE_G1",
        "--use-sim-policy-wrapper",
        "--host",
        str(args.server_host),
        "--port",
        str(int(args.server_port)),
    ]


def _spawn_server_subprocess(
    args: argparse.Namespace,
    repo_root: Path,
    server_log: Path,
) -> subprocess.Popen[str]:
    server_py = _server_script_path(repo_root)
    if not server_py.is_file():
        raise FileNotFoundError(f"missing server entrypoint: {server_py}")

    cmd = _build_server_cmd(args, repo_root)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("PYTHONPATH", None)

    # Mirror demo env settings for consistency.
    if args.mujoco_gl:
        env["MUJOCO_GL"] = str(args.mujoco_gl)
        if str(args.mujoco_gl).lower() == "egl":
            env.setdefault("PYOPENGL_PLATFORM", "egl")

    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "spawn_server_subprocess")
    return fn(cmd, log_path=server_log, cwd=repo_root, env=env)


def _terminate_process(proc: subprocess.Popen[str], timeout_s: float) -> None:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "terminate_process")
    fn(proc, float(timeout_s))


def _ensure_server_ready(
    args: argparse.Namespace,
    repo_root: Path,
    server_log: Path,
    *,
    allow_spawn: bool,
) -> tuple[object, subprocess.Popen[str] | None, bool]:
    host_for_client = _normalize_client_host(str(args.server_host))
    if host_for_client != str(args.server_host):
        print(
            f"[INFO] server_host={args.server_host!r} not connectable; "
            f"using client_host={host_for_client!r} for ping/kill"
        )

    client = _make_policy_client(
        host=host_for_client,
        port=int(args.server_port),
        timeout_ms=int(args.server_ping_timeout_ms),
    )

    print(f"[INFO] policy server target: {host_for_client}:{int(args.server_port)}")

    # Fast-path: reuse existing server if ping OK.
    if _safe_ping(client, int(args.server_ping_timeout_ms)):
        print("ping ok (reuse existing server)")
        return client, None, False

    if not allow_spawn:
        raise RuntimeError(
            "PolicyClient.ping() failed and auto-spawn is disabled for this mode; "
            "start the server first or use --mode server_client / --mode server / --mode ping-only"
        )

    # Avoid killing unknown processes: if the port is already listening but ping fails,
    # do not attempt to start (bind will fail) and do not kill.
    if _is_tcp_port_listening(host_for_client, int(args.server_port)):
        raise RuntimeError(
            "port is already occupied but PolicyClient.ping() failed; "
            "refuse to kill unknown process. "
            "Try changing --server-port or stop the other service."
        )

    print("[INFO] no responsive server found; spawning GR00T server subprocess...")
    proc = _spawn_server_subprocess(args, repo_root, server_log)
    print(f"[INFO] server subprocess pid={proc.pid} log={server_log}")

    try:
        t0 = time.monotonic()
        last_note = 0.0
        while True:
            if _safe_ping(client, int(args.server_ping_timeout_ms)):
                print("ping ok")
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
    except Exception:
        # Ensure we don't leak a subprocess on errors/timeouts.
        _terminate_process(proc, float(args.server_kill_timeout_s))
        raise


def _cleanup_server(
    args: argparse.Namespace,
    client,
    proc: subprocess.Popen[str] | None,
    started_by_me: bool,
) -> None:
    if not started_by_me:
        return

    print("[INFO] cleanup: stopping server started by this script")
    try:
        _safe_kill_server(client, int(args.server_ping_timeout_ms))
    except Exception:
        pass

    if proc is not None:
        _terminate_process(proc, float(args.server_kill_timeout_s))


def _task3_ensure_server_ready(
    args: argparse.Namespace,
    repo_root: Path,
    server_log: Path,
) -> tuple[int, object, subprocess.Popen[str] | None, bool]:
    allow_spawn = args.run_mode != "client"
    client, proc, started_by_me = _ensure_server_ready(
        args, repo_root, server_log, allow_spawn=allow_spawn
    )

    if args.run_mode == "ping-only":
        return 0, client, proc, started_by_me

    return 0, client, proc, started_by_me


def _extract_delta_indices(
    modality_cfg: dict[str, Any], modality: str
) -> list[int] | None:
    cfg = modality_cfg.get(modality)
    if cfg is None:
        return None
    delta = getattr(cfg, "delta_indices", None)
    if delta is None:
        return None
    try:
        return [int(x) for x in list(delta)]
    except Exception:
        return None


def _infer_n_action_steps_from_server(modality_cfg: dict[str, Any]) -> int | None:
    delta = _extract_delta_indices(modality_cfg, "action")
    if not delta:
        return None
    return int(len(delta))


def _override_task_prompt_in_obs(obs: dict[str, Any], task_prompt: str) -> None:
    key = "annotation.human.task_description"
    if key not in obs:
        return
    obs[key] = [str(task_prompt)]


def _make_live_vec_env(
    args: argparse.Namespace,
    *,
    modality_cfg: dict[str, Any],
    n_action_steps: int,
    video_dir: Path | None,
):
    gym = importlib.import_module("gymnasium")
    np = importlib.import_module("numpy")
    importlib.import_module("gr00t_wbc.control.envs.robocasa.sync_env")

    wbc_cfg_mod = importlib.import_module(
        "gr00t_wbc.control.main.teleop.configs.configs"
    )
    BaseConfig = cast(Any, getattr(wbc_cfg_mod, "BaseConfig"))

    n1_mod = importlib.import_module("gr00t_wbc.control.utils.n1_utils")
    WholeBodyControlWrapper = cast(Any, getattr(n1_mod, "WholeBodyControlWrapper"))

    ms_mod = importlib.import_module("gr00t.eval.sim.wrapper.multistep_wrapper")
    MultiStepWrapper = cast(Any, getattr(ms_mod, "MultiStepWrapper"))

    video_delta = _extract_delta_indices(modality_cfg, "video")
    state_delta = _extract_delta_indices(modality_cfg, "state")
    video_delta_indices = (
        np.asarray(video_delta, dtype=np.int32)
        if video_delta
        else np.asarray([0], dtype=np.int32)
    )
    state_delta_indices = (
        np.asarray(state_delta, dtype=np.int32)
        if state_delta
        else np.asarray([0], dtype=np.int32)
    )

    def env_fn():
        base_env = gym.make(
            str(args.env_name),
            onscreen=bool(args.onscreen),
            offscreen=bool(args.offscreen),
            enable_waist=True,
            randomize_cameras=False,
            camera_names=["robot0_oak_egoview", "robot0_rs_tppview"],
            render_camera=args.render_camera,
        )
        wbc_config = BaseConfig(wbc_version="gear_wbc", enable_waist=True).to_dict()
        env = WholeBodyControlWrapper(base_env, wbc_config)
        if bool(args.onscreen):

            class _RenderOnStepWrapper(gym.Wrapper):
                def reset(self, **kwargs):  # type: ignore[override]
                    obs, info = super().reset(**kwargs)
                    try:
                        self.env.render()
                    except Exception:
                        pass
                    return obs, info

                def step(self, action):  # type: ignore[override]
                    obs, reward, terminated, truncated, info = super().step(action)
                    try:
                        self.env.render()
                    except Exception:
                        pass
                    return obs, reward, terminated, truncated, info

            env = _RenderOnStepWrapper(env)

        if video_dir is not None:
            # Runtime-only import: keep --help stdlib-only.
            vr_mod = importlib.import_module(
                "gr00t.eval.sim.wrapper.video_recording_wrapper"
            )
            VideoRecorder = cast(Any, getattr(vr_mod, "VideoRecorder"))
            VideoRecordingWrapper = cast(Any, getattr(vr_mod, "VideoRecordingWrapper"))

            video_recorder = VideoRecorder.create_h264(
                fps=20,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=22,
                thread_type="FRAME",
                thread_count=1,
            )
            env = VideoRecordingWrapper(
                env,
                video_recorder,
                video_dir=Path(video_dir),
                steps_per_render=2,
                max_episode_steps=int(args.max_episode_steps),
                overlay_text=True,
            )
        env = MultiStepWrapper(
            env,
            video_delta_indices=video_delta_indices,
            state_delta_indices=state_delta_indices,
            n_action_steps=int(n_action_steps),
            max_episode_steps=int(args.max_episode_steps),
            terminate_on_success=False,
        )
        return env

    return gym.vector.SyncVectorEnv([env_fn])


def _run_live_rollout(
    args: argparse.Namespace,
    *,
    client,
    artifacts_videos: Path,
) -> None:
    modality_cfg = client.get_modality_config()

    n_action_steps = _infer_n_action_steps_from_server(modality_cfg)
    if n_action_steps is None:
        n_action_steps = int(args.n_action_steps)
        print(
            "[WARN] server modality_config missing action delta_indices; "
            f"fall back to --n-action-steps={n_action_steps}"
        )
    else:
        if int(args.n_action_steps) != int(n_action_steps):
            print(
                f"[INFO] override --n-action-steps={int(args.n_action_steps)} -> {int(n_action_steps)} "
                "(match server action horizon)"
            )
        args.n_action_steps = int(n_action_steps)

    action_keys = []
    try:
        if "action" in modality_cfg:
            action_keys = list(getattr(modality_cfg["action"], "modality_keys", []))
    except Exception:
        action_keys = []

    print(
        "[INFO] server modality_config keys:",
        sorted([str(k) for k in modality_cfg.keys()]),
    )
    if action_keys:
        print("[INFO] server action_keys:", action_keys)
    print("[INFO] n_action_steps:", int(args.n_action_steps))

    video_dir: Path | None = None
    if bool(args.offscreen):
        video_dir = _make_video_dir(
            env_name=str(args.env_name), n_action_steps=int(args.n_action_steps)
        )
    else:
        print("[WARN] offscreen disabled; video recording is disabled")

    env = None
    np = importlib.import_module("numpy")

    run_forever = int(args.n_episodes) <= 0
    if bool(args.keep_open_after_done):
        if sys.stdin.isatty():
            run_forever = True
        else:
            print(
                "[INFO] keep-open requested but stdin is not a TTY; "
                "use --exit-after-done for automation"
            )

    try:
        env = _make_live_vec_env(
            args,
            modality_cfg=modality_cfg,
            n_action_steps=int(args.n_action_steps),
            video_dir=video_dir,
        )
        if video_dir is not None:
            print("Video saved to: ", str(video_dir))

        ep_i = 0
        ep_total = "inf" if run_forever else str(int(args.n_episodes))
        if run_forever:
            print("[INFO] run_forever enabled (Ctrl+C to stop)")

        try:
            while True:
                if not run_forever and ep_i >= int(args.n_episodes):
                    break
                ep_i += 1

                print(f"[INFO] episode {ep_i}/{ep_total} reset")
                obs, _info = env.reset()
                client.reset()

                max_outer_steps = max(
                    1,
                    (int(args.max_episode_steps) + int(args.n_action_steps) - 1)
                    // int(args.n_action_steps),
                )

                for outer_i in range(int(max_outer_steps)):
                    t0 = time.time()
                    if args.task_prompt_override:
                        _override_task_prompt_in_obs(
                            obs, str(args.task_prompt_override)
                        )

                    action, _action_info = client.get_action(obs)
                    if not isinstance(action, dict):
                        raise TypeError(
                            "PolicyClient.get_action returned non-dict action: "
                            f"{type(action)}"
                        )

                    obs, reward, term, trunc, _step_info = env.step(action)

                    r0 = float(np.asarray(reward).reshape(-1)[0])
                    term0 = bool(np.asarray(term).reshape(-1)[0])
                    trunc0 = bool(np.asarray(trunc).reshape(-1)[0])
                    dt_ms = int((time.time() - t0) * 1000)
                    print(
                        f"[INFO] ep={ep_i} outer_step={outer_i} reward={r0:.4f} "
                        f"term={term0} trunc={trunc0} dt_ms={dt_ms}"
                    )

                    if term0 or trunc0:
                        break

                print(f"[INFO] episode {ep_i}/{ep_total} done")
        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt -> stopping live rollout")
    finally:
        if env is not None:
            _stop_video_recorder_best_effort(env)
            try:
                env.close()
            except Exception:
                pass

        _archive_video_dir(video_dir=video_dir, archive_root=artifacts_videos)


def main(argv: list[str] | None = None) -> int:
    repo_root = _repo_root()
    _maybe_reexec_into_wbc_venv(repo_root)

    _, _, _, client_log = _ensure_dirs(repo_root, VIDEO_ARCHIVE_DIR)
    with _tee_stdio(client_log):
        parser = _build_parser()
        try:
            args = parser.parse_args(argv)
        except SystemExit as e:
            code = int(getattr(e, "code", 0) or 0)
            return code

        if not args.onscreen and not args.offscreen:
            args.onscreen = bool(ONSCREEN)
            args.offscreen = bool(OFFSCREEN)

        if os.environ.pop("PYTHONPATH", None) is not None:
            print("[INFO] unset PYTHONPATH (avoid leaking host site-packages)")

        _, artifacts_videos, server_log, _ = _ensure_dirs(
            repo_root, args.video_archive_dir
        )

        print("[INFO] repo_root:", repo_root)
        print("[INFO] sys.executable:", sys.executable)
        print("[INFO] log:", client_log)

        wbc_py = _wbc_venv_python(repo_root)
        if not (wbc_py.is_file() and os.access(wbc_py, os.X_OK)):
            print(f"[WARN] WBC venv python not found/executable; expected: {wbc_py}")

        _apply_env(args)
        print("[INFO] MUJOCO_GL:", os.environ.get("MUJOCO_GL"))
        print(
            "[INFO] onscreen:", bool(args.onscreen), "offscreen:", bool(args.offscreen)
        )

        if args.run_mode == "noop":
            print("[INFO] run_mode=noop -> exiting")
            return 0

        if args.run_mode == "list-envs":
            runtime_dir, _, _, _ = _ensure_dirs(repo_root, args.video_archive_dir)
            env_log = runtime_dir / "02_env_registry.log"
            _list_registered_env_ids(prefix=ENV_REGISTRY_PREFIX, log_path=env_log)
            _maybe_keep_open(bool(args.keep_open_after_done))
            return 0

        client = None
        proc: subprocess.Popen[str] | None = None
        started_by_me = False
        try:
            code, client, proc, started_by_me = _task3_ensure_server_ready(
                args, repo_root, server_log
            )
            if args.run_mode in {"server_client", "client"}:
                _run_live_rollout(
                    args, client=client, artifacts_videos=artifacts_videos
                )

            if args.run_mode != "ping-only":
                if not (
                    bool(args.keep_open_after_done)
                    and bool(args.onscreen)
                    and (args.run_mode in {"server_client", "client"})
                ):
                    _maybe_keep_open(bool(args.keep_open_after_done))
            return int(code)
        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt -> exiting")
            return 130
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            if isinstance(e, ModuleNotFoundError):
                print(
                    "[HINT] Missing python deps often means you are not running under the "
                    "GR00T-WholeBodyControl venv (msgpack/numpy/pyzmq, etc.)."
                )
            return 1
        finally:
            if client is not None:
                _cleanup_server(args, client, proc, started_by_me)


if __name__ == "__main__":
    raise SystemExit(main())
