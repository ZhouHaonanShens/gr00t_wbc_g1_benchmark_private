#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5555

MUJOCO_GL = ""

PROMPT_A = "pick up the apple, walk left and place the apple on the plate."
PROMPT_B = "stand still and do nothing."

RESET_BETWEEN_CALLS = True

RUNTIME_LOGS_REL = "agent/runtime_logs/official_task_eval"
LOG_BASENAME = "20_prompt_sensitivity_probe.log"

SERVER_PING_TIMEOUT_MS = 2000
SERVER_READY_TIMEOUT_S = 60
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


def _ensure_dirs(repo_root: Path) -> tuple[Path, Path, Path]:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "ensure_dirs")
    runtime_dir, artifacts_videos = fn(
        repo_root=repo_root,
        runtime_logs_rel=RUNTIME_LOGS_REL,
        artifacts_videos_rel="agent/artifacts/videos",
    )
    runtime_dir = cast(Path, runtime_dir)
    artifacts_videos = cast(Path, artifacts_videos)
    log_path = runtime_dir / LOG_BASENAME
    return runtime_dir, artifacts_videos, log_path


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


def _apply_env(mujoco_gl: str) -> None:
    v = str(mujoco_gl or "").strip()
    if not v:
        return
    os.environ["MUJOCO_GL"] = v
    if v.lower() == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _wait_server_ready(
    client: Any,
    *,
    timeout_s: float,
    ping_timeout_ms: int,
    interval_s: float,
) -> None:
    t0 = time.monotonic()
    while True:
        if _safe_ping(client, ping_timeout_ms):
            return
        dt = time.monotonic() - t0
        if dt >= float(timeout_s):
            raise TimeoutError(f"Timed out waiting for server ping ok after {int(dt)}s")
        time.sleep(float(interval_s))


def _override_task_prompt(obs: dict[str, Any], prompt: str) -> None:
    key = "annotation.human.task_description"
    if key not in obs:
        raise KeyError(
            f"Missing key {key!r} in obs; got keys: {sorted(list(obs.keys()))}"
        )
    obs[key] = [str(prompt)]


def _summarize_action_value(v: Any) -> str:
    try:
        np = importlib.import_module("numpy")
        arr = np.asarray(v)
        if hasattr(arr, "dtype"):
            return f"ndarray dtype={arr.dtype} shape={tuple(arr.shape)}"
    except Exception:
        pass
    return f"{type(v)}"


def _compute_diffs(
    action_a: dict[str, Any], action_b: dict[str, Any]
) -> list[tuple[str, float, float, float]]:
    np = importlib.import_module("numpy")

    keys = sorted(
        set([str(k) for k in action_a.keys()]) | set([str(k) for k in action_b.keys()])
    )
    rows: list[tuple[str, float, float, float]] = []
    for k in keys:
        if (k not in action_a) or (k not in action_b):
            rows.append((k, float("nan"), float("nan"), float("nan")))
            continue
        a = np.asarray(action_a[k])
        b = np.asarray(action_b[k])
        try:
            d = a.astype(np.float64) - b.astype(np.float64)
            d = np.asarray(d)
            l2 = float(np.linalg.norm(d.reshape(-1)))
            mean_abs = float(np.mean(np.abs(d)))
            max_abs = float(np.max(np.abs(d)))
            rows.append((k, l2, mean_abs, max_abs))
        except Exception:
            rows.append((k, float("nan"), float("nan"), float("nan")))
    return rows


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prompt_sensitivity_probe_g1.py",
        description=(
            "Prompt sensitivity probe: keep the same obs, change only annotation.human.task_description, "
            "and compare actions from GR00T policy server."
        ),
    )

    p.add_argument("--env-name", type=str, default=ENV_NAME)
    p.add_argument("--server-host", type=str, default=SERVER_HOST)
    p.add_argument("--server-port", type=int, default=SERVER_PORT)
    p.add_argument("--mujoco-gl", type=str, default=MUJOCO_GL)

    p.add_argument("--prompt-a", type=str, default=PROMPT_A)
    p.add_argument("--prompt-b", type=str, default=PROMPT_B)

    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is None:
        p.add_argument(
            "--reset-between-calls",
            action="store_true",
            default=bool(RESET_BETWEEN_CALLS),
        )
    else:
        p.add_argument(
            "--reset-between-calls",
            action=bool_action,
            default=bool(RESET_BETWEEN_CALLS),
            help="Reset policy state between the 2 calls (recommended for isolating prompt effect).",
        )

    p.add_argument(
        "--server-ready-timeout-s", type=float, default=float(SERVER_READY_TIMEOUT_S)
    )
    p.add_argument(
        "--server-ping-timeout-ms", type=int, default=int(SERVER_PING_TIMEOUT_MS)
    )
    p.add_argument(
        "--server-ping-interval-s", type=float, default=float(SERVER_PING_INTERVAL_S)
    )
    return p


def main() -> int:
    repo_root = _repo_root()
    _maybe_reexec_into_wbc_venv(repo_root)

    args = _build_parser().parse_args()
    _apply_env(str(args.mujoco_gl))

    _runtime_dir, _artifacts_videos, log_path = _ensure_dirs(repo_root)
    with _tee_stdio(log_path, header="prompt_sensitivity_probe_g1"):
        host_for_client = _normalize_client_host(str(args.server_host))
        client = _make_policy_client(
            host=host_for_client,
            port=int(args.server_port),
            timeout_ms=int(args.server_ping_timeout_ms),
        )

        print(f"[INFO] policy server target: {host_for_client}:{int(args.server_port)}")
        _wait_server_ready(
            client,
            timeout_s=float(args.server_ready_timeout_s),
            ping_timeout_ms=int(args.server_ping_timeout_ms),
            interval_s=float(args.server_ping_interval_s),
        )
        print("[INFO] ping ok")

        modality_cfg = client.get_modality_config()
        print("[INFO] modality_config keys:", sorted(list(modality_cfg.keys())))
        if "action" in modality_cfg:
            action_keys = list(
                getattr(modality_cfg["action"], "modality_keys", []) or []
            )
            action_delta = list(
                getattr(modality_cfg["action"], "delta_indices", []) or []
            )
            print("[INFO] server action_keys:", [str(k) for k in action_keys])
            print("[INFO] server action_horizon:", len(action_delta))

        gym = importlib.import_module("gymnasium")
        rollout_mod = importlib.import_module("gr00t.eval.rollout_policy")
        WrapperConfigs = getattr(rollout_mod, "WrapperConfigs")
        VideoConfig = getattr(rollout_mod, "VideoConfig")
        MultiStepConfig = getattr(rollout_mod, "MultiStepConfig")
        create_eval_env = getattr(rollout_mod, "create_eval_env")

        action_horizon = 20
        try:
            if "action" in modality_cfg:
                delta = list(getattr(modality_cfg["action"], "delta_indices", []) or [])
                if delta:
                    action_horizon = int(len(delta))
        except Exception:
            pass

        wrapper_configs = WrapperConfigs(
            video=VideoConfig(video_dir=None, max_episode_steps=2, overlay_text=False),
            multistep=MultiStepConfig(
                n_action_steps=int(action_horizon),
                max_episode_steps=2,
                terminate_on_success=False,
            ),
        )

        def env_fn():
            return create_eval_env(
                env_name=str(args.env_name),
                env_idx=0,
                total_n_envs=1,
                wrapper_configs=wrapper_configs,
            )

        env = gym.vector.SyncVectorEnv([env_fn])
        try:
            obs, info = env.reset()
            if not isinstance(obs, dict):
                raise TypeError(f"env.reset() returned non-dict obs: {type(obs)}")
            print("[INFO] reset obs keys:", sorted([str(k) for k in obs.keys()]))
            print(
                "[INFO] prompt key type:",
                type(obs.get("annotation.human.task_description")),
            )
            print(
                "[INFO] prompt key value:", obs.get("annotation.human.task_description")
            )
            if isinstance(info, dict):
                print("[INFO] reset info keys:", sorted([str(k) for k in info.keys()]))

            obs_a = dict(obs)
            obs_b = dict(obs)
            _override_task_prompt(obs_a, str(args.prompt_a))
            _override_task_prompt(obs_b, str(args.prompt_b))

            if bool(args.reset_between_calls):
                print("[INFO] reset_between_calls=True (isolate prompt effect)")
                client.reset()
                action_a, _info_a = client.get_action(obs_a)
                client.reset()
                action_b, _info_b = client.get_action(obs_b)
            else:
                print("[INFO] reset_between_calls=False (sequential calls)")
                client.reset()
                action_a, _info_a = client.get_action(obs_a)
                action_b, _info_b = client.get_action(obs_b)

            if not isinstance(action_a, dict) or not isinstance(action_b, dict):
                raise TypeError(
                    f"expected dict actions; got {type(action_a)} and {type(action_b)}"
                )

            print("[INFO] action_a keys:", sorted([str(k) for k in action_a.keys()]))
            print("[INFO] action_b keys:", sorted([str(k) for k in action_b.keys()]))

            for k in sorted(set(action_a.keys()) | set(action_b.keys())):
                if (k in action_a) and (k in action_b):
                    print(
                        f"[INFO] action[{k}] a={_summarize_action_value(action_a[k])} b={_summarize_action_value(action_b[k])}"
                    )
                else:
                    print(f"[WARN] action key only in one dict: {k!r}")

            rows = _compute_diffs(action_a, action_b)
            print("\n[DIFF] per-key metrics (L2 / mean_abs / max_abs):")
            for k, l2, mean_abs, max_abs in rows:
                print(
                    f"- {k}: l2={l2:.6g} mean_abs={mean_abs:.6g} max_abs={max_abs:.6g}"
                )

            focus = ["action.navigate_command", "action.right_arm"]
            print("\n[DIFF] focus keys:")
            for k in focus:
                if (k in action_a) and (k in action_b):
                    print(f"- {k}: a={action_a[k]} b={action_b[k]}")
                else:
                    print(f"- {k}: missing")

            return 0
        finally:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
