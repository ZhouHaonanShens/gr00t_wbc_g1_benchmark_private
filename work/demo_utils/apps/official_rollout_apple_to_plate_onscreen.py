#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import datetime as _dt
import importlib
import os
import signal
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5555

N_EPISODES = 3
MAX_EPISODE_STEPS = 1440

N_ACTION_STEPS = 30

ONSCREEN = True
OFFSCREEN = True
MUJOCO_GL = "glfw"

RENDER_CAMERA = "robot0_oak_egoview"  # Onscreen viewer camera: free|none|robot0_oak_egoview|robot0_rs_tppview

RENDERER = "mjviewer"
HARD_RESET = True

STEPS_PER_RENDER = 2

VIDEO_ARCHIVE_DIR = "agent/artifacts/videos"
RUNTIME_LOGS_REL = "agent/runtime_logs/official_task_eval"
LOG_BASENAME = "12_rollout_apple_to_plate_onscreen.log"
VARIANT_LOG_BASENAME = "12_env_variant_debug.log"

ENV_VARIANTS: dict[str, dict[str, Any]] = {
    "official": {"apple_pos_mode": "official"},
    "apple_left_of_plate": {"apple_pos_mode": "apple_left_of_plate"},
    "apple_right_of_plate": {"apple_pos_mode": "apple_right_of_plate"},
    "apple_farther": {"apple_pos_mode": "apple_farther"},
    "apple_xy": {"apple_pos_mode": "apple_xy", "requires": ("APPLE_X", "APPLE_Y")},
}

ENV_VARIANT = "official"
APPLE_X: float | None = None
APPLE_Y: float | None = None
VARIANT_DEBUG = False

TASK_PROMPT_OVERRIDE: str | None = (
    "Look around and search for the apple. Then move to and pick up the apple."
)

SERVER_READY_TIMEOUT_S = 600
SERVER_PING_TIMEOUT_MS = 2000
SERVER_PING_INTERVAL_S = 1.0

TOTAL_TIMEOUT_S = 1800

SAVE_COMPOSITE_VIDEO = (
    True  # Save the built-in composite mp4 (concatenated obs video.* frames)
)
SAVE_EGO_VIDEO = (
    True  # Save per-episode ego view: obs["video.ego_view"] -> epXXX_ego.mp4
)
SAVE_TPP_VIDEO = (
    True  # Save per-episode third-person view: obs["video.tpp_view"] -> epXXX_tpp.mp4
)
SAVE_FREE_VIDEO = (
    True  # Best-effort: record env.render(mode="rgb_array") -> epXXX_free.mp4
)

SHOW_EGO_WINDOW = False  # Show a small OpenCV window for ego view (requires cv2)
SHOW_TPP_WINDOW = (
    False  # Show a small OpenCV window for third-person view (requires cv2)
)


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


def _ensure_dirs(repo_root: Path) -> tuple[Path, Path, Path]:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "ensure_dirs")
    runtime_dir, artifacts_videos = fn(
        repo_root=repo_root,
        runtime_logs_rel=RUNTIME_LOGS_REL,
        artifacts_videos_rel=str(VIDEO_ARCHIVE_DIR),
    )
    runtime_dir = cast(Path, runtime_dir)
    artifacts_videos = cast(Path, artifacts_videos)
    return runtime_dir, artifacts_videos, runtime_dir / LOG_BASENAME


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


def _make_policy_client(host: str, port: int, timeout_ms: int) -> Any:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "make_policy_client")
    return fn(host=str(host), port=int(port), timeout_ms=int(timeout_ms))


def _safe_ping(client: Any, timeout_ms: int) -> bool:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "safe_ping")
    return bool(fn(client, int(timeout_ms)))


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


def _register_gr00tlocomanip_g1_envs() -> None:
    importlib.import_module("gr00t_wbc.control.envs.robocasa.sync_env")


def _parse_render_camera(s: str) -> str | None:
    v = str(s or "").strip().lower()
    if v in ("", "none", "free"):
        return None
    return str(s)


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Official AppleToPlateDC rollout with onscreen viewer + optional per-view videos."
        )
    )
    p.add_argument("--render-camera", default=RENDER_CAMERA, type=_parse_render_camera)
    p.add_argument(
        "--renderer",
        default=str(RENDERER),
        choices=("mjviewer", "mujoco"),
    )
    p.add_argument(
        "--hard-reset",
        default=bool(HARD_RESET),
        action=argparse.BooleanOptionalAction,
    )
    p.add_argument(
        "--save-composite-video",
        default=bool(SAVE_COMPOSITE_VIDEO),
        action=argparse.BooleanOptionalAction,
    )
    p.add_argument(
        "--save-ego-video",
        default=bool(SAVE_EGO_VIDEO),
        action=argparse.BooleanOptionalAction,
    )
    p.add_argument(
        "--save-tpp-video",
        default=bool(SAVE_TPP_VIDEO),
        action=argparse.BooleanOptionalAction,
    )
    p.add_argument(
        "--save-free-video",
        default=bool(SAVE_FREE_VIDEO),
        action=argparse.BooleanOptionalAction,
    )
    p.add_argument(
        "--show-ego-window",
        default=bool(SHOW_EGO_WINDOW),
        action=argparse.BooleanOptionalAction,
    )
    p.add_argument(
        "--show-tpp-window",
        default=bool(SHOW_TPP_WINDOW),
        action=argparse.BooleanOptionalAction,
    )
    p.add_argument(
        "--env-variant",
        default=str(ENV_VARIANT),
        type=str,
        choices=tuple(ENV_VARIANTS.keys()),
        help=(
            "Environment variant (default: %(default)s). Supported: "
            + ", ".join(sorted(list(ENV_VARIANTS.keys())))
        ),
    )
    p.add_argument(
        "--apple-x",
        default=APPLE_X,
        type=float,
        help="Used when --env-variant=apple_xy (world X in meters)",
    )
    p.add_argument(
        "--apple-y",
        default=APPLE_Y,
        type=float,
        help="Used when --env-variant=apple_xy (world Y in meters)",
    )
    p.add_argument(
        "--variant-debug",
        default=bool(VARIANT_DEBUG),
        action=argparse.BooleanOptionalAction,
        help="Write verbose variant self-discovery logs",
    )
    p.add_argument(
        "--task-prompt-override",
        default=TASK_PROMPT_OVERRIDE,
        type=str,
        help=(
            "Override task prompt by writing observation['annotation.human.task_description'] before policy.get_action()"
        ),
    )
    return p


def _override_task_prompt_in_obs(obs: dict[str, Any], task_prompt: str) -> bool:
    v = str(task_prompt).strip()
    if not v:
        return False

    for key in (
        "annotation.human.task_description",
        "annotation.language.language_instruction",
        "language.language_instruction",
    ):
        if key in obs:
            obs[key] = [v]
            return True

    return False


def _make_official_g1_env(
    *,
    env_name: str,
    onscreen: bool,
    offscreen: bool,
    render_camera: str | None,
    renderer: str,
    hard_reset: bool,
    max_episode_steps: int,
    n_action_steps: int,
    video_dir: Path | None,
    steps_per_render: int,
    save_composite_video: bool,
    env_variant: str = ENV_VARIANT,
    apple_x: float | None = APPLE_X,
    apple_y: float | None = APPLE_Y,
    variant_debug: bool = bool(VARIANT_DEBUG),
) -> Any:
    gym = importlib.import_module("gymnasium")
    np = importlib.import_module("numpy")

    _register_gr00tlocomanip_g1_envs()

    multistep_mod = importlib.import_module("gr00t.eval.sim.wrapper.multistep_wrapper")
    MultiStepWrapper = getattr(multistep_mod, "MultiStepWrapper")

    gym_env = gym.make(
        str(env_name),
        onscreen=bool(onscreen),
        offscreen=bool(offscreen),
        renderer=str(renderer),
        enable_waist=True,
        randomize_cameras=False,
        camera_names=["robot0_oak_egoview", "robot0_rs_tppview"],
        render_camera=render_camera,
    )

    rs_mod = importlib.import_module("work.demo_utils.robosuite_env")
    set_hard_reset_best_effort = getattr(rs_mod, "set_hard_reset_best_effort")
    _ = bool(set_hard_reset_best_effort(gym_env, bool(hard_reset)))

    v = str(env_variant or "").strip()
    if v and v.lower() != "official":
        var_mod = importlib.import_module("work.env_variants.g1_locomanip_variants")
        make_variant_spec = getattr(var_mod, "make_variant_spec")
        wrap_with_variant = getattr(var_mod, "wrap_with_variant")
        spec = make_variant_spec(
            name=v,
            apple_x=apple_x,
            apple_y=apple_y,
            debug_dump=bool(variant_debug),
        )
        gym_env = wrap_with_variant(gym_env, spec)

    importlib.import_module("robocasa")
    base_cfg_mod = importlib.import_module(
        "gr00t_wbc.control.main.teleop.configs.configs"
    )
    BaseConfig = getattr(base_cfg_mod, "BaseConfig")
    n1_utils_mod = importlib.import_module("gr00t_wbc.control.utils.n1_utils")
    WholeBodyControlWrapper = getattr(n1_utils_mod, "WholeBodyControlWrapper")

    wbc_config = BaseConfig(wbc_version="gear_wbc", enable_waist=True).to_dict()
    gym_env = WholeBodyControlWrapper(gym_env, wbc_config)

    if video_dir is not None and bool(save_composite_video):
        vr_mod = importlib.import_module(
            "gr00t.eval.sim.wrapper.video_recording_wrapper"
        )
        VideoRecorder = getattr(vr_mod, "VideoRecorder")
        VideoRecordingWrapper = getattr(vr_mod, "VideoRecordingWrapper")

        video_recorder = VideoRecorder.create_h264(
            fps=20,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=22,
            thread_type="FRAME",
            thread_count=1,
        )
        gym_env = VideoRecordingWrapper(
            gym_env,
            video_recorder,
            video_dir=Path(video_dir),
            steps_per_render=int(steps_per_render),
            max_episode_steps=int(max_episode_steps),
            overlay_text=True,
        )

    gym_env = MultiStepWrapper(
        gym_env,
        video_delta_indices=np.array([0]),
        state_delta_indices=np.array([0]),
        n_action_steps=int(n_action_steps),
        max_episode_steps=int(max_episode_steps),
        terminate_on_success=True,
    )
    return gym_env


def main() -> int:
    repo_root = _repo_root()
    _maybe_reexec_into_wbc_venv(repo_root)
    _apply_env(MUJOCO_GL)

    args = _build_parser().parse_args()

    runtime_dir, artifacts_videos, _log_path_default = _ensure_dirs(repo_root)
    log_basename = str(LOG_BASENAME)
    if bool(getattr(args, "variant_debug", False)):
        log_basename = str(VARIANT_LOG_BASENAME)
    log_path = Path(runtime_dir) / log_basename

    with _tee_stdio(log_path, header="official_rollout_apple_to_plate_onscreen"):
        _install_alarm_timeout(TOTAL_TIMEOUT_S)
        try:
            host_for_client = _normalize_client_host(SERVER_HOST)
            client = _make_policy_client(
                host=host_for_client,
                port=int(SERVER_PORT),
                timeout_ms=int(SERVER_PING_TIMEOUT_MS),
            )

            print("[INFO] ts:", _dt.datetime.now().isoformat(timespec="seconds"))
            print("[INFO] python:", sys.version.replace("\n", " "))
            print("[INFO] env_name:", ENV_NAME)
            print("[INFO] onscreen:", bool(ONSCREEN), "offscreen:", bool(OFFSCREEN))
            print("[INFO] MUJOCO_GL:", os.environ.get("MUJOCO_GL"))
            print("[INFO] render_camera:", args.render_camera)
            print("[INFO] renderer:", str(args.renderer))
            print("[INFO] hard_reset:", bool(args.hard_reset))
            print("[INFO] env_variant:", str(getattr(args, "env_variant", "official")))
            print("[INFO] variant_debug:", bool(getattr(args, "variant_debug", False)))
            print(
                "[INFO] task_prompt_override:",
                str(getattr(args, "task_prompt_override", None)),
            )
            print("[INFO] policy server:", f"{host_for_client}:{int(SERVER_PORT)}")

            t0 = time.monotonic()
            while not _safe_ping(client, int(SERVER_PING_TIMEOUT_MS)):
                dt = time.monotonic() - t0
                if dt >= float(SERVER_READY_TIMEOUT_S):
                    raise TimeoutError(
                        f"Timed out waiting for server ping ok after {int(dt)}s"
                    )
                time.sleep(float(SERVER_PING_INTERVAL_S))
            print("[INFO] ping ok")

            modality_cfg = client.get_modality_config()
            if "action" in modality_cfg:
                action_delta = list(
                    getattr(modality_cfg["action"], "delta_indices", []) or []
                )
                print("[INFO] server action_horizon:", len(action_delta))
                if action_delta and int(N_ACTION_STEPS) != int(len(action_delta)):
                    print(
                        "[WARN] N_ACTION_STEPS != server action_horizon:",
                        int(N_ACTION_STEPS),
                        "vs",
                        int(len(action_delta)),
                    )

            gym = importlib.import_module("gymnasium")
            np = importlib.import_module("numpy")
            sc_mod = importlib.import_module("gr00t.policy.server_client")
            PolicyClient = getattr(sc_mod, "PolicyClient")

            safe_env = str(ENV_NAME).replace("/", "__")
            tag = f"{safe_env}__official_onscreen__{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            video_dir = (
                _make_video_dir(env_name=tag, n_action_steps=int(N_ACTION_STEPS))
                if OFFSCREEN
                else None
            )
            if video_dir is not None:
                print("[INFO] video_dir:", str(video_dir))

            vr_mod = importlib.import_module(
                "gr00t.eval.sim.wrapper.video_recording_wrapper"
            )
            VideoRecorder = getattr(vr_mod, "VideoRecorder")
            cv2 = None
            if bool(args.show_ego_window) or bool(args.show_tpp_window):
                cv2 = importlib.import_module("cv2")

            def env_fn():
                return _make_official_g1_env(
                    env_name=ENV_NAME,
                    onscreen=bool(ONSCREEN),
                    offscreen=bool(OFFSCREEN),
                    render_camera=args.render_camera,
                    renderer=str(args.renderer),
                    hard_reset=bool(args.hard_reset),
                    max_episode_steps=int(MAX_EPISODE_STEPS),
                    n_action_steps=int(N_ACTION_STEPS),
                    video_dir=video_dir,
                    steps_per_render=int(STEPS_PER_RENDER),
                    save_composite_video=bool(args.save_composite_video),
                    env_variant=str(getattr(args, "env_variant", "official")),
                    apple_x=getattr(args, "apple_x", None),
                    apple_y=getattr(args, "apple_y", None),
                    variant_debug=bool(getattr(args, "variant_debug", False)),
                )

            env = gym.vector.SyncVectorEnv([env_fn])
            policy = PolicyClient(host=host_for_client, port=int(SERVER_PORT))

            successes: list[bool] = []
            try:
                for ep_i in range(int(N_EPISODES)):
                    print(f"[INFO] episode {ep_i + 1}/{int(N_EPISODES)} reset")
                    obs, _info = env.reset()
                    policy.reset()

                    ego_vr = None
                    tpp_vr = None
                    free_vr = None
                    if video_dir is not None:
                        if bool(args.save_ego_video):
                            ego_vr = VideoRecorder.create_h264(
                                fps=20,
                                codec="h264",
                                input_pix_fmt="rgb24",
                                crf=22,
                                thread_type="FRAME",
                                thread_count=1,
                            )
                            ego_vr.start(
                                str(Path(video_dir) / f"ep{ep_i + 1:03d}_ego.mp4")
                            )
                        if bool(args.save_tpp_video):
                            tpp_vr = VideoRecorder.create_h264(
                                fps=20,
                                codec="h264",
                                input_pix_fmt="rgb24",
                                crf=22,
                                thread_type="FRAME",
                                thread_count=1,
                            )
                            tpp_vr.start(
                                str(Path(video_dir) / f"ep{ep_i + 1:03d}_tpp.mp4")
                            )
                        if bool(args.save_free_video):
                            free_vr = VideoRecorder.create_h264(
                                fps=20,
                                codec="h264",
                                input_pix_fmt="rgb24",
                                crf=22,
                                thread_type="FRAME",
                                thread_count=1,
                            )
                            free_vr.start(
                                str(Path(video_dir) / f"ep{ep_i + 1:03d}_free.mp4")
                            )

                    done = False
                    steps = 0
                    ep_success = False
                    while not done:
                        if getattr(args, "task_prompt_override", None) and isinstance(
                            obs, dict
                        ):
                            _ = _override_task_prompt_in_obs(
                                obs, str(getattr(args, "task_prompt_override", ""))
                            )
                        action, _a_info = policy.get_action(obs)
                        obs, _reward, term, trunc, info = env.step(action)
                        steps += 1

                        if video_dir is not None and (
                            steps % int(max(1, STEPS_PER_RENDER)) == 0
                        ):
                            try:
                                if (
                                    ego_vr is not None
                                    and isinstance(obs, dict)
                                    and "video.ego_view" in obs
                                ):
                                    f = np.asarray(obs["video.ego_view"])[0, -1]
                                    ego_vr.write_frame(f)
                                    if cv2 is not None and bool(args.show_ego_window):
                                        cv2.imshow("ego_view", f[..., ::-1])
                                if (
                                    tpp_vr is not None
                                    and isinstance(obs, dict)
                                    and "video.tpp_view" in obs
                                ):
                                    f = np.asarray(obs["video.tpp_view"])[0, -1]
                                    tpp_vr.write_frame(f)
                                    if cv2 is not None and bool(args.show_tpp_window):
                                        cv2.imshow("tpp_view", f[..., ::-1])

                                if (cv2 is not None) and (
                                    bool(args.show_ego_window)
                                    or bool(args.show_tpp_window)
                                ):
                                    _ = cv2.waitKey(1)
                            except Exception:
                                pass

                            if free_vr is not None:
                                try:
                                    e0 = getattr(env, "envs", [None])[0]
                                    if e0 is not None:
                                        rgb = e0.render(mode="rgb_array")
                                        arr = np.asarray(rgb)
                                        if (
                                            getattr(arr, "ndim", 0) == 3
                                            and int(arr.shape[-1]) == 3
                                        ):
                                            free_vr.write_frame(arr.astype(np.uint8))
                                except Exception:
                                    pass

                        try:
                            if isinstance(info, dict) and "success" in info:
                                s = info["success"][0]
                                if isinstance(s, (bool, int)):
                                    ep_success |= bool(s)
                        except Exception:
                            pass

                        term0 = bool(np.asarray(term).reshape(-1)[0])
                        trunc0 = bool(np.asarray(trunc).reshape(-1)[0])
                        done = term0 or trunc0

                        if bool(ONSCREEN) and (
                            steps % int(max(1, STEPS_PER_RENDER)) == 0
                        ):
                            try:
                                env.render()
                            except Exception:
                                pass

                        if steps >= int(MAX_EPISODE_STEPS):
                            done = True

                    successes.append(bool(ep_success))
                    print(
                        f"[INFO] episode {ep_i + 1} done steps={steps} success={bool(ep_success)}"
                    )
                    print(f"[INFO] success_rate_so_far={float(np.mean(successes)):.3f}")

                    for r in (ego_vr, tpp_vr, free_vr):
                        try:
                            if r is not None:
                                r.stop()
                        except Exception:
                            pass
            except KeyboardInterrupt:
                print("\n[INFO] KeyboardInterrupt -> stop early")
            finally:
                try:
                    env.close()
                except Exception:
                    pass

                try:
                    if cv2 is not None:
                        cv2.destroyAllWindows()
                except Exception:
                    pass

                archived = _archive_video_dir(
                    video_dir=video_dir, archive_root=artifacts_videos
                )
                print("[INFO] videos_saved_dir:", archived)
                if successes:
                    print("[INFO] episodes_done:", len(successes))
                    print("[INFO] success_rate_final:", float(np.mean(successes)))
                else:
                    print("[INFO] episodes_done: 0")

            return 0
        finally:
            _clear_alarm_timeout()


if __name__ == "__main__":
    raise SystemExit(main())
