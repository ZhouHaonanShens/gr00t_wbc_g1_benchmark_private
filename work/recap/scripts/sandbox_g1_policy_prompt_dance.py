#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast


# =====================
# USER Config (edit)
# =====================

# FixedBase makes waving/dancing easier to observe.
# ENV_ID = "gr00tlocomanip_g1_sim/GroundOnly_G1FixedBase_gear_wbc"
ENV_ID = "gr00tlocomanip_g1_sim/GroundOnly_G1_gear_wbc"

MODEL_PATH = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5555

MUJOCO_GL = "glfw"

RENDERER = "mujoco"
HARD_RESET = False

ONSCREEN = True
OFFSCREEN = True

RENDER_CAMERA: str | None = None

N_ACTION_STEPS = 30
MAX_EPISODE_STEPS = 5000

TASK_PROMPT = (
    # "Stand tall and wave your right hand to greet, then wave like dancing."
    "Walk forward and raise your right hand to wave hello."
)

INTERACTIVE_PROMPT = False

KEEP_OPEN = True

VIDEO_ARCHIVE_DIR = "agent/artifacts/videos"
RUNTIME_LOGS_REL = "agent/runtime_logs/policy_prompt_dance"
ENV_REGISTRY_PREFIX = "gr00tlocomanip_g1_sim/"


SERVER_READY_TIMEOUT_S = 600
SERVER_PING_TIMEOUT_MS = 2000
SERVER_PING_INTERVAL_S = 1.0


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


def _wbc_venv_python(repo_root: Path) -> Path:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "wbc_venv_python")
    return cast(Path, fn(repo_root))


def _ensure_dirs(
    repo_root: Path, video_archive_dir: str
) -> tuple[Path, Path, Path, Path, Path]:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "ensure_dirs")
    runtime_dir, artifacts_videos = fn(
        repo_root=repo_root,
        runtime_logs_rel=RUNTIME_LOGS_REL,
        artifacts_videos_rel=str(video_archive_dir),
    )
    runtime_dir = cast(Path, runtime_dir)
    artifacts_videos = cast(Path, artifacts_videos)
    server_log = runtime_dir / "00_server.log"
    client_log = runtime_dir / "01_client.log"
    env_log = runtime_dir / "02_env_registry.log"
    return runtime_dir, artifacts_videos, server_log, client_log, env_log


@contextlib.contextmanager
def _tee_stdio(log_path: Path, *, header: str) -> Iterator[None]:
    mod = importlib.import_module("work.demo_utils.tee")
    fn = getattr(mod, "tee_stdio")
    with fn(log_path, header=str(header)):
        yield


def _list_registered_env_ids(*, prefix: str, log_path: Path) -> list[str]:
    mod = importlib.import_module("work.demo_utils.env_registry")
    fn = getattr(mod, "list_registered_env_ids")
    return cast(
        list[str],
        fn(
            prefix=str(prefix),
            log_path=Path(log_path),
            register_modules=("gr00t_wbc.control.envs.robocasa.sync_env",),
        ),
    )


def _make_video_dir(*, env_name: str, n_action_steps: int) -> Path:
    mod = importlib.import_module("work.demo_utils.videos")
    fn = getattr(mod, "make_video_dir")
    return cast(Path, fn(env_name=str(env_name), n_action_steps=int(n_action_steps)))


def _archive_video_dir(*, video_dir: Path | None, archive_root: Path) -> Path | None:
    mod = importlib.import_module("work.demo_utils.videos")
    fn = getattr(mod, "archive_video_dir")
    return cast(Path | None, fn(video_dir=video_dir, archive_root=archive_root))


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


def _install_signal_handlers() -> Any:
    mod = importlib.import_module("work.demo_utils.signals")
    fn = getattr(mod, "install_signal_handlers")
    return fn(raise_keyboardinterrupt=True)


def _parse_render_camera(s: str) -> str | None:
    v = str(s).strip()
    if not v:
        return None
    if v.lower() in {"none", "null", "free"}:
        return None
    return v


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sandbox_g1_policy_prompt_dance.py",
        description=(
            "Sandbox live demo: prompt->model->action via GR00T policy server, with onscreen + mp4 evidence."
        ),
    )

    p.add_argument("--env-id", type=str, default=ENV_ID)
    p.add_argument("--model-path", type=str, default=MODEL_PATH)
    p.add_argument("--server-host", type=str, default=SERVER_HOST)
    p.add_argument("--server-port", type=int, default=SERVER_PORT)
    p.add_argument("--mujoco-gl", type=str, default=MUJOCO_GL)
    p.add_argument(
        "--renderer",
        type=str,
        default=str(RENDERER),
        choices=("mjviewer", "mujoco"),
        help="Renderer backend for onscreen viewing.",
    )

    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is None:
        p.add_argument("--hard-reset", action="store_true", default=bool(HARD_RESET))
    else:
        p.add_argument(
            "--hard-reset",
            action=bool_action,
            default=bool(HARD_RESET),
            help="If true, reset reloads sim/render objects (may recreate viewer).",
        )

    if bool_action is None:
        p.add_argument("--onscreen", action="store_true", default=bool(ONSCREEN))
        p.add_argument("--offscreen", action="store_true", default=bool(OFFSCREEN))
        p.add_argument(
            "--interactive-prompt",
            action="store_true",
            default=bool(INTERACTIVE_PROMPT),
        )
        p.add_argument("--keep-open", action="store_true", default=bool(KEEP_OPEN))
    else:
        p.add_argument(
            "--onscreen",
            action=bool_action,
            default=bool(ONSCREEN),
            help="Enable/disable onscreen viewer.",
        )
        p.add_argument(
            "--offscreen",
            action=bool_action,
            default=bool(OFFSCREEN),
            help="Enable/disable offscreen rendering (recommended for video/video modality).",
        )
        p.add_argument(
            "--interactive-prompt",
            action=bool_action,
            default=bool(INTERACTIVE_PROMPT),
            help="Read a new prompt from stdin at episode boundaries (TTY only).",
        )
        p.add_argument(
            "--keep-open",
            action=bool_action,
            default=bool(KEEP_OPEN),
            help="Keep running until Ctrl+C (TTY only).",
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

    p.add_argument("--n-action-steps", type=int, default=int(N_ACTION_STEPS))
    p.add_argument("--max-episode-steps", type=int, default=int(MAX_EPISODE_STEPS))

    p.add_argument("--task-prompt", type=str, default=TASK_PROMPT)
    p.add_argument(
        "--prompt-every-n-outer-steps",
        type=int,
        default=0,
        help=(
            "If >0 and interactive prompt enabled, ask for a new prompt every N outer steps (best-effort). "
            "0 means only ask at episode start."
        ),
    )

    p.add_argument(
        "--exit-after-s",
        type=float,
        default=None,
        help="Exit after N seconds (automation/CI fuse).",
    )
    p.add_argument(
        "--exit-after-episodes",
        type=int,
        default=None,
        help="Exit after N episodes (automation/CI fuse).",
    )

    if bool_action is not None:
        p.add_argument(
            "--client-reset-per-episode",
            action=bool_action,
            default=True,
            help="Call client.reset() after every env.reset().",
        )
        p.add_argument(
            "--client-reset-on-prompt-change",
            action=bool_action,
            default=True,
            help="Call client.reset() when interactive prompt changes.",
        )
    else:
        g = p.add_mutually_exclusive_group()
        g.add_argument(
            "--client-reset-per-episode",
            dest="client_reset_per_episode",
            action="store_true",
            default=True,
            help="Call client.reset() after every env.reset().",
        )
        g.add_argument(
            "--no-client-reset-per-episode",
            dest="client_reset_per_episode",
            action="store_false",
            help="Do not call client.reset() after env.reset().",
        )
        g2 = p.add_mutually_exclusive_group()
        g2.add_argument(
            "--client-reset-on-prompt-change",
            dest="client_reset_on_prompt_change",
            action="store_true",
            default=True,
            help="Call client.reset() when interactive prompt changes.",
        )
        g2.add_argument(
            "--no-client-reset-on-prompt-change",
            dest="client_reset_on_prompt_change",
            action="store_false",
            help="Do not call client.reset() when prompt changes.",
        )

    p.add_argument(
        "--kill-server-on-exit",
        action="store_true",
        help=(
            "If this script spawned the GR00T server subprocess, kill it during cleanup. "
            "Default: keep it running for interactive experimentation."
        ),
    )

    p.add_argument(
        "--list-envs",
        action="store_true",
        help="List registered env IDs (prefix filtered) and exit.",
    )

    p.add_argument(
        "--video-archive-dir",
        type=str,
        default=VIDEO_ARCHIVE_DIR,
        help="Where to archive rollout video dirs (relative to repo root by default).",
    )

    p.add_argument(
        "--server-ready-timeout-s",
        type=int,
        default=int(SERVER_READY_TIMEOUT_S),
        help="Timeout waiting for PolicyClient.ping() to succeed.",
    )
    p.add_argument(
        "--server-ping-timeout-ms",
        type=int,
        default=int(SERVER_PING_TIMEOUT_MS),
        help="ZMQ send/recv timeout for ping/kill endpoints.",
    )
    p.add_argument(
        "--server-ping-interval-s",
        type=float,
        default=float(SERVER_PING_INTERVAL_S),
        help="Polling interval between ping attempts.",
    )
    p.add_argument(
        "--server-embodiment-tag",
        type=str,
        default="UNITREE_G1",
        help="Embodiment tag to pass to run_gr00t_server.py.",
    )
    return p


def _apply_env(args: argparse.Namespace) -> None:
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = str(args.mujoco_gl)
        if str(args.mujoco_gl).lower() == "egl":
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _server_entrypoint(repo_root: Path) -> Path:
    return repo_root / "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"


def _build_server_cmd(args: argparse.Namespace, repo_root: Path) -> list[str]:
    server_py = _server_entrypoint(repo_root)
    return [
        sys.executable,
        str(server_py),
        "--model-path",
        str(args.model_path),
        "--embodiment-tag",
        str(args.server_embodiment_tag),
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
    if host_for_client != str(args.server_host):
        print(
            f"[INFO] server_host={args.server_host!r} not connectable; using client_host={host_for_client!r}"
        )

    client = _make_policy_client(
        host=host_for_client,
        port=int(args.server_port),
        timeout_ms=int(args.server_ping_timeout_ms),
    )

    print(f"[INFO] policy server target: {host_for_client}:{int(args.server_port)}")
    if _safe_ping(client, int(args.server_ping_timeout_ms)):
        print("ping ok (reuse existing server)")
        return client, None, False

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

    print("[INFO] no responsive server found; spawning GR00T server subprocess...")
    proc = _spawn_server_subprocess(cmd, log_path=server_log, cwd=repo_root, env=env)
    print(f"[INFO] server subprocess pid={proc.pid} log={server_log}")

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


def _infer_action_horizon(modality_cfg: dict[str, Any]) -> int | None:
    delta = _extract_delta_indices(modality_cfg, "action")
    if not delta:
        return None
    return int(len(delta))


def _flatten_required_obs_keys(modality_cfg: dict[str, Any]) -> list[str]:
    required: list[str] = []

    for mod_name in ("video", "state"):
        cfg = modality_cfg.get(mod_name)
        if cfg is None:
            continue
        try:
            keys = list(getattr(cfg, "modality_keys", []) or [])
        except Exception:
            keys = []
        for k in keys:
            ks = str(k)
            if "." in ks:
                required.append(ks)
            else:
                required.append(f"{mod_name}.{ks}")

    lang = modality_cfg.get("language")
    if lang is not None:
        try:
            keys = list(getattr(lang, "modality_keys", []) or [])
        except Exception:
            keys = []
        for k in keys:
            required.append(str(k))

    if "annotation.human.task_description" not in required:
        required.append("annotation.human.task_description")

    return sorted(list(dict.fromkeys(required)))


def _summarize_array(x: Any) -> str:
    try:
        np = importlib.import_module("numpy")
        if isinstance(x, np.ndarray):
            return f"ndarray dtype={x.dtype} shape={tuple(x.shape)}"
    except Exception:
        pass
    return f"{type(x)}"


def _check_obs_coverage(
    obs: dict[str, Any],
    required_keys: Sequence[str],
    *,
    video_horizon: int | None,
    state_horizon: int | None,
    strict: bool = True,
) -> None:
    missing = [k for k in required_keys if k not in obs]
    if missing:
        msg = (
            "Missing required obs keys: "
            + ", ".join([repr(k) for k in missing])
            + "\nGot keys: "
            + ", ".join(sorted([str(k) for k in obs.keys()]))
        )
        raise KeyError(msg)

    np = importlib.import_module("numpy")

    def _as_array(v: Any) -> Any:
        try:
            return np.asarray(v)
        except Exception:
            return None

    problems: list[str] = []
    for k in required_keys:
        v = obs.get(k)
        if k.startswith("video."):
            arr = _as_array(v)
            if arr is None or not hasattr(arr, "dtype"):
                problems.append(f"{k}: not array-like ({type(v)})")
                continue
            if str(getattr(arr, "dtype", "")) != "uint8":
                problems.append(
                    f"{k}: dtype={getattr(arr, 'dtype', None)} (expected uint8)"
                )
            if getattr(arr, "ndim", 0) != 5:
                problems.append(
                    f"{k}: ndim={getattr(arr, 'ndim', None)} (expected 5: B,T,H,W,C)"
                )
            else:
                b, t = int(arr.shape[0]), int(arr.shape[1])
                if b != 1:
                    problems.append(f"{k}: B={b} (expected 1)")
                if int(arr.shape[-1]) != 3:
                    problems.append(f"{k}: C={int(arr.shape[-1])} (expected 3)")
                if video_horizon is not None:
                    if t != int(video_horizon):
                        problems.append(
                            f"{k}: T={t} (expected video_horizon={int(video_horizon)})"
                        )
                else:
                    if t < 1:
                        problems.append(f"{k}: T={t} (expected >=1)")

        elif k.startswith("state."):
            arr = _as_array(v)
            if arr is None or not hasattr(arr, "dtype"):
                problems.append(f"{k}: not array-like ({type(v)})")
                continue
            dt = str(getattr(arr, "dtype", ""))
            if dt != "float32":
                problems.append(
                    f"{k}: dtype={getattr(arr, 'dtype', None)} (expected float32)"
                )
            if getattr(arr, "ndim", 0) != 3:
                problems.append(
                    f"{k}: ndim={getattr(arr, 'ndim', None)} (expected 3: B,T,D)"
                )
            else:
                b, t = int(arr.shape[0]), int(arr.shape[1])
                if b != 1:
                    problems.append(f"{k}: B={b} (expected 1)")
                if state_horizon is not None:
                    if t != int(state_horizon):
                        problems.append(
                            f"{k}: T={t} (expected state_horizon={int(state_horizon)})"
                        )
                else:
                    if t < 1:
                        problems.append(f"{k}: T={t} (expected >=1)")

        elif k == "annotation.human.task_description":
            if isinstance(v, str):
                continue
            if isinstance(v, list) and (not v or isinstance(v[0], str)):
                if strict and v and len(v) != 1:
                    problems.append(
                        f"{k}: list length={len(v)} (expected B=1 list[str])"
                    )
                continue
            arr = _as_array(v)
            if arr is not None and getattr(arr, "ndim", 0) >= 1:
                continue
            problems.append(f"{k}: unexpected type {type(v)}")

    if problems and strict:
        report = "\n".join(["- " + p for p in problems])
        raise TypeError("Observation coverage/type check failed:\n" + report)
    elif problems:
        print("[WARN] observation coverage/type warnings:\n" + "\n".join(problems))


def _override_task_prompt(obs: dict[str, Any], prompt: str) -> None:
    key = "annotation.human.task_description"
    if key not in obs:
        raise KeyError(
            f"Missing prompt key {key!r} in obs; got keys: {sorted(list(obs.keys()))}"
        )
    obs[key] = [str(prompt)]
    alt = "annotation.human.action.task_description"
    if alt in obs:
        obs[alt] = [str(prompt)]


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
            str(args.env_id),
            onscreen=bool(args.onscreen),
            offscreen=bool(args.offscreen),
            renderer=str(getattr(args, "renderer", "mjviewer")),
            enable_waist=True,
            randomize_cameras=False,
            camera_names=["robot0_oak_egoview", "robot0_rs_tppview"],
            render_camera=args.render_camera,
        )

        rs_mod = importlib.import_module("work.demo_utils.robosuite_env")
        set_hard_reset_best_effort = getattr(rs_mod, "set_hard_reset_best_effort")
        _ = bool(
            set_hard_reset_best_effort(
                base_env, bool(getattr(args, "hard_reset", True))
            )
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


def _maybe_read_prompt_interactive(
    *,
    enabled: bool,
    current_prompt: str,
    prompt_prefix: str,
) -> tuple[str, bool]:
    if not enabled:
        return current_prompt, False
    if not sys.stdin.isatty():
        print(
            "[INFO] interactive prompt enabled but stdin is not a TTY; keeping current prompt"
        )
        return current_prompt, False

    try:
        line = input(f"{prompt_prefix} (empty=keep): ").strip("\n")
    except EOFError:
        return current_prompt, False
    except KeyboardInterrupt:
        raise

    new = line.strip()
    if not new:
        return current_prompt, False
    if new == current_prompt:
        return current_prompt, False
    return new, True


def _maybe_keep_open(keep_open: bool) -> None:
    if not keep_open:
        return
    if not sys.stdin.isatty():
        print("[INFO] keep-open requested but stdin is not a TTY; exiting")
        return
    print("[INFO] keep-open enabled. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt -> exiting")


def main(argv: list[str] | None = None) -> int:
    repo_root = _repo_root()
    _maybe_reexec_into_wbc_venv(repo_root)

    runtime_dir, artifacts_videos, server_log, client_log, env_log = _ensure_dirs(
        repo_root, VIDEO_ARCHIVE_DIR
    )
    with _tee_stdio(client_log, header="sandbox_g1_policy_prompt_dance.py start"):
        parser = _build_parser()
        try:
            args = parser.parse_args(argv)
        except SystemExit as e:
            return int(getattr(e, "code", 0) or 0)

        _, artifacts_videos2, server_log2, _client_log2, env_log2 = _ensure_dirs(
            repo_root, str(args.video_archive_dir)
        )
        artifacts_videos = artifacts_videos2
        server_log = server_log2
        env_log = env_log2

        if os.environ.pop("PYTHONPATH", None) is not None:
            print("[INFO] unset PYTHONPATH (avoid leaking host site-packages)")

        _apply_env(args)
        print("[INFO] repo_root:", repo_root)
        print("[INFO] sys.executable:", sys.executable)
        print("[INFO] runtime_dir:", runtime_dir)
        print("[INFO] client_log:", client_log)
        print("[INFO] server_log:", server_log)
        print("[INFO] video_archive_dir:", artifacts_videos)
        print("[INFO] ENV_ID:", str(args.env_id))
        print("[INFO] MODEL_PATH:", str(args.model_path))
        print("[INFO] MUJOCO_GL:", os.environ.get("MUJOCO_GL"))
        print(
            "[INFO] onscreen:", bool(args.onscreen), "offscreen:", bool(args.offscreen)
        )
        print("[INFO] render_camera:", args.render_camera)
        print("[INFO] interactive_prompt:", bool(args.interactive_prompt))
        print("[INFO] keep_open:", bool(args.keep_open))
        print("[INFO] exit_after_s:", args.exit_after_s)
        print("[INFO] exit_after_episodes:", args.exit_after_episodes)
        print(
            "[INFO] client_reset_per_episode:",
            bool(getattr(args, "client_reset_per_episode", True)),
        )
        print(
            "[INFO] client_reset_on_prompt_change:",
            bool(getattr(args, "client_reset_on_prompt_change", True)),
        )

        wbc_py = _wbc_venv_python(repo_root)
        if not (wbc_py.is_file() and os.access(wbc_py, os.X_OK)):
            print(f"[WARN] WBC venv python not found/executable; expected: {wbc_py}")

        if bool(args.list_envs):
            try:
                _list_registered_env_ids(prefix=ENV_REGISTRY_PREFIX, log_path=env_log)
            except Exception as e:
                print("[ERROR] failed to list env ids:", type(e).__name__ + ":", str(e))
                return 2
            return 0

        try:
            _list_registered_env_ids(prefix=ENV_REGISTRY_PREFIX, log_path=env_log)
        except Exception as e:
            print("[WARN] env registry listing failed:", type(e).__name__ + ":", str(e))

        _ = _install_signal_handlers()

        client = None
        proc: subprocess.Popen[str] | None = None
        started_by_me = False
        video_dir: Path | None = None
        env = None
        try:
            client, proc, started_by_me = _ensure_server_ready(
                args, repo_root, server_log
            )

            modality_cfg = client.get_modality_config()
            print(
                "[INFO] server modality_config keys:",
                sorted([str(k) for k in modality_cfg.keys()]),
            )

            action_horizon = _infer_action_horizon(modality_cfg)
            if action_horizon is not None:
                if int(args.n_action_steps) != int(action_horizon):
                    print(
                        f"[INFO] override --n-action-steps={int(args.n_action_steps)} -> {int(action_horizon)} "
                        "(match server action horizon)"
                    )
                args.n_action_steps = int(action_horizon)
            else:
                print(
                    "[WARN] server modality_config missing action delta_indices; "
                    f"using --n-action-steps={int(args.n_action_steps)}"
                )

            required_obs_keys = _flatten_required_obs_keys(modality_cfg)
            print("[INFO] required obs keys (derived):", required_obs_keys)
            print("[INFO] n_action_steps:", int(args.n_action_steps))

            vdelta = _extract_delta_indices(modality_cfg, "video")
            sdelta = _extract_delta_indices(modality_cfg, "state")
            video_horizon = int(len(vdelta)) if vdelta else None
            state_horizon = int(len(sdelta)) if sdelta else None
            print("[INFO] video_horizon:", video_horizon)
            print("[INFO] state_horizon:", state_horizon)

            if not bool(args.offscreen):
                print(
                    "[WARN] offscreen disabled; for Unitree G1 the policy server often requires video.* modality. "
                    "If you hit modality assertions, re-run with --offscreen."
                )

            if bool(args.offscreen):
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_env = str(args.env_id).replace("/", "__")
                tag = f"{safe_env}__promptdance__{ts}"
                video_dir = _make_video_dir(
                    env_name=tag, n_action_steps=int(args.n_action_steps)
                )
                print("[INFO] video_dir:", str(video_dir))
            else:
                video_dir = None

            env = _make_live_vec_env(
                args,
                modality_cfg=modality_cfg,
                n_action_steps=int(args.n_action_steps),
                video_dir=video_dir,
            )

            np = importlib.import_module("numpy")

            current_prompt = str(args.task_prompt)
            prompt_changed = False
            start_t = time.monotonic()
            ep_i = 0

            run_forever = True
            if args.exit_after_s is not None or args.exit_after_episodes is not None:
                run_forever = False
            if not bool(args.keep_open):
                run_forever = False
            if bool(args.keep_open) and not sys.stdin.isatty():
                print(
                    "[INFO] keep-open requested but stdin is not a TTY; disabling keep-open"
                )
                run_forever = False

            if bool(args.interactive_prompt) and not sys.stdin.isatty():
                print(
                    "[INFO] interactive-prompt requested but stdin is not a TTY; disabling interactive prompt"
                )
                args.interactive_prompt = False

            if run_forever:
                print("[INFO] run_forever enabled (Ctrl+C to stop)")

            try:
                while True:
                    if (args.exit_after_s is not None) and (
                        time.monotonic() - start_t >= float(args.exit_after_s)
                    ):
                        print("[INFO] exit-after-s reached -> stopping")
                        break
                    if (args.exit_after_episodes is not None) and (
                        ep_i >= int(args.exit_after_episodes)
                    ):
                        print("[INFO] exit-after-episodes reached -> stopping")
                        break
                    if (
                        not run_forever
                        and ep_i >= 1
                        and args.exit_after_episodes is None
                    ):
                        break

                    ep_i += 1
                    print(f"[INFO] episode {ep_i} reset")
                    obs, _info = env.reset()
                    if not isinstance(obs, dict):
                        raise TypeError(f"env.reset returned non-dict obs: {type(obs)}")

                    print(
                        "[INFO] reset obs keys:", sorted([str(k) for k in obs.keys()])
                    )
                    for k in required_obs_keys:
                        if k in obs:
                            print(f"[INFO] obs[{k}] ->", _summarize_array(obs[k]))

                    _check_obs_coverage(
                        obs,
                        required_obs_keys,
                        video_horizon=video_horizon,
                        state_horizon=state_horizon,
                        strict=True,
                    )

                    if bool(getattr(args, "client_reset_per_episode", True)):
                        client.reset()
                    current_prompt, prompt_changed = _maybe_read_prompt_interactive(
                        enabled=bool(args.interactive_prompt),
                        current_prompt=current_prompt,
                        prompt_prefix=f"Prompt for episode {ep_i} (current={current_prompt!r})",
                    )
                    if prompt_changed and bool(
                        getattr(args, "client_reset_on_prompt_change", True)
                    ):
                        print("[INFO] prompt changed -> client.reset()")
                        client.reset()

                    max_outer_steps = max(
                        1,
                        (int(args.max_episode_steps) + int(args.n_action_steps) - 1)
                        // int(args.n_action_steps),
                    )

                    for outer_i in range(int(max_outer_steps)):
                        if (args.exit_after_s is not None) and (
                            time.monotonic() - start_t >= float(args.exit_after_s)
                        ):
                            break

                        if (
                            bool(args.interactive_prompt)
                            and int(args.prompt_every_n_outer_steps) > 0
                            and sys.stdin.isatty()
                            and (outer_i % int(args.prompt_every_n_outer_steps) == 0)
                            and outer_i != 0
                        ):
                            current_prompt, prompt_changed = (
                                _maybe_read_prompt_interactive(
                                    enabled=True,
                                    current_prompt=current_prompt,
                                    prompt_prefix=f"Prompt at outer_step={outer_i} (current={current_prompt!r})",
                                )
                            )
                            if prompt_changed:
                                if bool(
                                    getattr(args, "client_reset_on_prompt_change", True)
                                ):
                                    print("[INFO] prompt changed -> client.reset()")
                                    client.reset()

                        t0 = time.time()
                        _override_task_prompt(obs, current_prompt)
                        action, action_info = client.get_action(obs)
                        if not isinstance(action, dict):
                            raise TypeError(
                                "PolicyClient.get_action returned non-dict action: "
                                + str(type(action))
                            )

                        if outer_i == 0:
                            print(
                                "[INFO] action keys:",
                                sorted([str(k) for k in action.keys()])[:20],
                                "..." if len(action.keys()) > 20 else "",
                            )
                            if isinstance(action_info, dict):
                                print(
                                    "[INFO] action_info keys:",
                                    sorted([str(k) for k in action_info.keys()]),
                                )

                        obs, reward, term, trunc, _step_info = env.step(action)

                        r0 = float(np.asarray(reward).reshape(-1)[0])
                        term0 = bool(np.asarray(term).reshape(-1)[0])
                        trunc0 = bool(np.asarray(trunc).reshape(-1)[0])
                        dt_ms = int((time.time() - t0) * 1000)
                        if outer_i == 0 or (outer_i % 10 == 0) or term0 or trunc0:
                            print(
                                f"[INFO] ep={ep_i} outer_step={outer_i} reward={r0:.4f} "
                                f"term={term0} trunc={trunc0} dt_ms={dt_ms} n_action_keys={len(action.keys())}"
                            )

                        if term0 or trunc0:
                            break

                    print(f"[INFO] episode {ep_i} done")

                    if not run_forever and args.exit_after_episodes is None:
                        break
            except KeyboardInterrupt:
                print("\n[INFO] KeyboardInterrupt -> stopping")

            if env is not None:
                _stop_video_recorder_best_effort(env)

            _archive_video_dir(video_dir=video_dir, archive_root=artifacts_videos)

            if not run_forever:
                _maybe_keep_open(bool(args.keep_open))

            return 0
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            if isinstance(e, ModuleNotFoundError):
                print(
                    "[HINT] Missing python deps often means you are not running under the GR00T-WholeBodyControl venv. "
                    "Try running this script from the repo root and let it re-exec into the venv."
                )
            return 1
        finally:
            if env is not None:
                _stop_video_recorder_best_effort(env)
                try:
                    env.close()
                except Exception:
                    pass

            if client is not None and started_by_me and bool(args.kill_server_on_exit):
                print("[INFO] cleanup: stopping server started by this script")
                try:
                    _safe_kill_server(client, int(args.server_ping_timeout_ms))
                except Exception:
                    pass
                if proc is not None:
                    _terminate_process(proc, timeout_s=10.0)
            elif started_by_me:
                print(
                    "[INFO] cleanup: leaving server running (use --kill-server-on-exit to stop it)"
                )


if __name__ == "__main__":
    raise SystemExit(main())
