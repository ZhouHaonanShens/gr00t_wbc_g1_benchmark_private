#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_DIR = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/public_anchor"
)
DEFAULT_RUNTIME_LOGS_REL = (
    "agent/runtime_logs/gr00t_anchor_controller_recap/unitree_g1/public_anchor"
)
DEFAULT_VIDEO_ARCHIVE_DIR = "agent/artifacts/videos/gr00t_public_anchor/unitree_g1"

DEFAULT_ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
DEFAULT_MODEL_PATH = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_EMBODIMENT_TAG = "UNITREE_G1"

FORMAL_DEFAULT_N_EPISODES = 10
FORMAL_DEFAULT_N_ENVS = 5
FORMAL_DEFAULT_MAX_EPISODE_STEPS = 1440
FORMAL_DEFAULT_N_ACTION_STEPS = 20
FORMAL_DEFAULT_SEED_LIST: tuple[int, ...] = tuple(range(20000, 20010))

SMOKE_DEFAULT_N_EPISODES = 1
SMOKE_DEFAULT_N_ENVS = 1
SMOKE_DEFAULT_MAX_EPISODE_STEPS = 50
SMOKE_DEFAULT_N_ACTION_STEPS = 20
SMOKE_DEFAULT_SEED_LIST: tuple[int, ...] = (FORMAL_DEFAULT_SEED_LIST[0],)

EXPECTED_SERVER_ACTION_HORIZON = 30
EXPECTED_PUBLIC_REFERENCE_SUCCESS_RATE = 0.58
EXPECTED_PUBLIC_REFERENCE_VARIANCE = "+/-15%"

SERVER_PYTHON = ""
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5555
MUJOCO_GL = ""

SERVER_READY_TIMEOUT_S = 600.0
SERVER_PING_TIMEOUT_MS = 2000
SERVER_PING_INTERVAL_S = 1.0

SMOKE_TOTAL_TIMEOUT_S = 1800.0
FORMAL_TOTAL_TIMEOUT_S = 3600.0

ACTION_ZERO_EPS = 1e-6
SATURATION_ABS_THRESHOLD = 0.999
SATURATION_FRACTION_THRESHOLD = 0.80
ZERO_MOTION_STATE_DELTA_THRESHOLD = 1e-6
TRAJECTORY_DIGEST_DECIMALS = 6

SMOKE_JSON_NAME = "public_anchor_smoke.json"
FORMAL_JSON_NAME = "public_anchor_formal.json"
SANITY_GATE_JSON_NAME = "public_anchor_sanity_gate.json"
FAILURE_NOTE_MARKDOWN_NAME = "public_anchor_failure_note.md"

SMOKE_SCHEMA_VERSION = "gr00t_public_anchor_smoke_v1"
FORMAL_SCHEMA_VERSION = "gr00t_public_anchor_formal_v1"
SANITY_GATE_SCHEMA_VERSION = "gr00t_public_anchor_sanity_gate_v1"

SMOKE_ARTIFACT_KIND = "gr00t_public_anchor_smoke"
FORMAL_ARTIFACT_KIND = "gr00t_public_anchor_formal"
SANITY_GATE_ARTIFACT_KIND = "gr00t_public_anchor_sanity_gate"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import policy as recap_policy
from work.recap import state_conditioned_bucket_a_import, state_conditioned_phase0_smoke


@dataclass(frozen=True)
class EvalConfig:
    mode: str
    output_dir: Path
    env_name: str
    model_path: str
    embodiment_tag: str
    server_host: str
    server_port: int
    server_python: str
    mujoco_gl: str
    n_episodes: int
    n_envs: int
    max_episode_steps: int
    n_action_steps: int
    seed_list: tuple[int, ...]
    server_ready_timeout_s: float
    server_ping_timeout_ms: int
    server_ping_interval_s: float
    spawn_server_if_missing: bool
    kill_server_on_exit: bool
    total_timeout_s: float | None


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _repo_root() -> Path:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "repo_root")
    return cast(Path, fn(from_path=__file__))


def _maybe_reexec_into_wbc_venv(repo_root: Path) -> None:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "maybe_reexec_into_wbc_venv")
    fn(repo_root)


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _ensure_runtime_layout(repo_root: Path, mode: str) -> tuple[Path, Path, Path, Path]:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "ensure_dirs")
    runtime_dir, artifacts_videos = fn(
        repo_root=repo_root,
        runtime_logs_rel=DEFAULT_RUNTIME_LOGS_REL,
        artifacts_videos_rel=str(DEFAULT_VIDEO_ARCHIVE_DIR),
    )
    runtime_dir = cast(Path, runtime_dir)
    artifacts_videos = cast(Path, artifacts_videos)
    server_log = runtime_dir / f"{str(mode)}_00_server.log"
    client_log = runtime_dir / f"{str(mode)}_10_client.log"
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


def _gr00t_server_pid_on_port(host: str, port: int) -> int | None:
    normalized_host = _normalize_client_host(str(host))
    if normalized_host not in {"127.0.0.1", "localhost"}:
        return None
    try:
        ss_output = subprocess.check_output(
            ["ss", "-ltnp"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    suffix = f":{int(port)}"
    for raw_line in ss_output.splitlines():
        line = raw_line.strip()
        if suffix not in line or "pid=" not in line:
            continue
        pid_text = line.split("pid=", 1)[1].split(",", 1)[0].rstrip(")")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        try:
            cmd = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "cmd="],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            continue
        if "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py" in cmd:
            return pid
    return None


def _kill_stale_gr00t_server_on_port(host: str, port: int) -> bool:
    pid = _gr00t_server_pid_on_port(host, port)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not _is_tcp_port_listening(host, port):
            return True
        time.sleep(0.25)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _is_tcp_port_listening(host, port):
            return True
        time.sleep(0.25)
    return not _is_tcp_port_listening(host, port)


def _wait_for_port_release(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if not _is_tcp_port_listening(host, port):
            return True
        time.sleep(0.2)
    return not _is_tcp_port_listening(host, port)


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
    value = str(mujoco_gl or "").strip()
    if not value:
        return
    os.environ["MUJOCO_GL"] = value
    if value.lower() == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _server_entrypoint(repo_root: Path) -> Path:
    return repo_root / "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"


def _build_server_cmd(config: EvalConfig, repo_root: Path) -> list[str]:
    server_py = _server_entrypoint(repo_root)
    exe = str(config.server_python).strip() or str(
        state_conditioned_bucket_a_import._preferred_live_python(repo_root)
    )
    return [
        exe,
        str(server_py),
        "--model-path",
        str(config.model_path),
        "--embodiment-tag",
        str(config.embodiment_tag),
        "--use-sim-policy-wrapper",
        "--host",
        str(config.server_host),
        "--port",
        str(int(config.server_port)),
    ]


def _ensure_server_ready(
    config: EvalConfig, repo_root: Path, server_log: Path
) -> tuple[Any, subprocess.Popen[str] | None, bool]:
    host_for_client = _normalize_client_host(str(config.server_host))
    client = _make_policy_client(
        host=host_for_client,
        port=int(config.server_port),
        timeout_ms=int(config.server_ping_timeout_ms),
    )
    print(f"[INFO] policy server target: {host_for_client}:{int(config.server_port)}")
    if _safe_ping(client, int(config.server_ping_timeout_ms)):
        print("[INFO] ping ok (reuse existing server)")
        return client, None, False

    if not bool(config.spawn_server_if_missing):
        raise RuntimeError(
            "no responsive server found (ping failed) and spawn_server_if_missing is disabled"
        )

    if _is_tcp_port_listening(host_for_client, int(config.server_port)):
        if _kill_stale_gr00t_server_on_port(host_for_client, int(config.server_port)):
            time.sleep(0.5)
        if _is_tcp_port_listening(host_for_client, int(config.server_port)):
            raise RuntimeError(
                "port is already occupied but PolicyClient.ping() failed; refuse to kill unknown process. "
                "Try changing --server-port or stop the other service."
            )

    server_py = _server_entrypoint(repo_root)
    if not server_py.is_file():
        raise FileNotFoundError(f"missing server entrypoint: {server_py}")

    cmd = _build_server_cmd(config, repo_root)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["GR00T_SKIP_WBC_REEXEC"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        state_conditioned_bucket_a_import._build_live_pythonpath(repo_root)
    )
    if config.mujoco_gl:
        env["MUJOCO_GL"] = str(config.mujoco_gl)
        if str(config.mujoco_gl).lower() == "egl":
            env.setdefault("PYOPENGL_PLATFORM", "egl")

    print("[INFO] spawning GR00T server subprocess...")
    proc = _spawn_server_subprocess(cmd, log_path=server_log, cwd=repo_root, env=env)
    print(f"[INFO] server subprocess pid={proc.pid} log={server_log}")

    t0 = time.monotonic()
    last_note = 0.0
    while True:
        if _safe_ping(client, int(config.server_ping_timeout_ms)):
            print("[INFO] ping ok")
            return client, proc, True
        if proc.poll() is not None:
            raise RuntimeError(
                f"server subprocess exited early rc={proc.returncode}; see {server_log}"
            )
        elapsed_s = time.monotonic() - t0
        if elapsed_s > float(config.server_ready_timeout_s):
            raise TimeoutError(
                f"timeout waiting for ping ok after {int(elapsed_s)}s; see {server_log}"
            )
        if elapsed_s - last_note >= 5.0:
            print(f"[INFO] waiting for server ready... {int(elapsed_s)}s")
            last_note = elapsed_s
        time.sleep(float(config.server_ping_interval_s))


def _install_alarm_timeout(timeout_s: float | None) -> None:
    if timeout_s is None:
        return
    try:
        timeout_int = int(float(timeout_s))
    except Exception:
        return
    if timeout_int <= 0 or not hasattr(signal, "SIGALRM"):
        return

    def _handler(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"Timed out after {timeout_int}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_int)


def _clear_alarm_timeout() -> None:
    if hasattr(signal, "SIGALRM"):
        try:
            signal.alarm(0)
        except Exception:
            pass


def _bool_arg(
    parser: argparse.ArgumentParser, name: str, *, default: bool, help_text: str
) -> None:
    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is None:
        parser.add_argument(name, action="store_true", default=default, help=help_text)
        return
    parser.add_argument(name, action=bool_action, default=default, help=help_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_public_anchor_eval.py",
        description=(
            "Run the UNITREE_G1 public checkpoint anchor with explicit smoke/formal "
            "artifacts and a loose sanity gate."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-name", type=str, default=DEFAULT_ENV_NAME)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--embodiment-tag", type=str, default=DEFAULT_EMBODIMENT_TAG)
    parser.add_argument("--server-host", type=str, default=SERVER_HOST)
    parser.add_argument("--server-port", type=int, default=SERVER_PORT)
    parser.add_argument("--server-python", type=str, default=SERVER_PYTHON)
    parser.add_argument("--mujoco-gl", type=str, default=MUJOCO_GL)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--seed-list", type=int, nargs="*", default=None)
    parser.add_argument(
        "--server-ready-timeout-s",
        type=float,
        default=float(SERVER_READY_TIMEOUT_S),
    )
    parser.add_argument(
        "--server-ping-timeout-ms",
        type=int,
        default=int(SERVER_PING_TIMEOUT_MS),
    )
    parser.add_argument(
        "--server-ping-interval-s",
        type=float,
        default=float(SERVER_PING_INTERVAL_S),
    )
    parser.add_argument("--total-timeout-s", type=float, default=None)
    _bool_arg(
        parser,
        "--spawn-server-if-missing",
        default=True,
        help_text="Spawn server subprocess if ping fails.",
    )
    _bool_arg(
        parser,
        "--kill-server-on-exit",
        default=True,
        help_text="Kill server if this script started it.",
    )
    return parser


def _resolve_mode_defaults(
    mode: str,
) -> tuple[int, int, int, int, tuple[int, ...], float]:
    if str(mode) == "formal":
        return (
            FORMAL_DEFAULT_N_EPISODES,
            FORMAL_DEFAULT_N_ENVS,
            FORMAL_DEFAULT_MAX_EPISODE_STEPS,
            FORMAL_DEFAULT_N_ACTION_STEPS,
            FORMAL_DEFAULT_SEED_LIST,
            FORMAL_TOTAL_TIMEOUT_S,
        )
    return (
        SMOKE_DEFAULT_N_EPISODES,
        SMOKE_DEFAULT_N_ENVS,
        SMOKE_DEFAULT_MAX_EPISODE_STEPS,
        SMOKE_DEFAULT_N_ACTION_STEPS,
        SMOKE_DEFAULT_SEED_LIST,
        SMOKE_TOTAL_TIMEOUT_S,
    )


def _resolve_eval_config(args: argparse.Namespace) -> EvalConfig:
    (
        default_n_episodes,
        default_n_envs,
        default_max_episode_steps,
        default_n_action_steps,
        default_seed_list,
        default_total_timeout_s,
    ) = _resolve_mode_defaults(str(args.mode))
    n_episodes = (
        int(args.n_episodes) if args.n_episodes is not None else default_n_episodes
    )
    n_envs = int(args.n_envs) if args.n_envs is not None else default_n_envs
    max_episode_steps = (
        int(args.max_episode_steps)
        if args.max_episode_steps is not None
        else default_max_episode_steps
    )
    n_action_steps = (
        int(args.n_action_steps)
        if args.n_action_steps is not None
        else default_n_action_steps
    )
    seed_list = tuple(int(item) for item in (args.seed_list or default_seed_list))
    total_timeout_s = (
        float(args.total_timeout_s)
        if args.total_timeout_s is not None
        else float(default_total_timeout_s)
    )
    output_dir = _validate_output_dir(Path(args.output_dir))
    if n_episodes <= 0:
        raise ValueError("n-episodes must be positive")
    if n_envs <= 0:
        raise ValueError("n-envs must be positive")
    if max_episode_steps <= 0:
        raise ValueError("max-episode-steps must be positive")
    if n_action_steps <= 0:
        raise ValueError("n-action-steps must be positive")
    if len(seed_list) != n_episodes:
        raise ValueError("seed-list length must match n-episodes")
    return EvalConfig(
        mode=str(args.mode),
        output_dir=output_dir,
        env_name=str(args.env_name),
        model_path=str(args.model_path),
        embodiment_tag=str(args.embodiment_tag),
        server_host=str(args.server_host),
        server_port=int(args.server_port),
        server_python=str(args.server_python),
        mujoco_gl=str(args.mujoco_gl),
        n_episodes=int(n_episodes),
        n_envs=int(n_envs),
        max_episode_steps=int(max_episode_steps),
        n_action_steps=int(n_action_steps),
        seed_list=tuple(seed_list),
        server_ready_timeout_s=float(args.server_ready_timeout_s),
        server_ping_timeout_ms=int(args.server_ping_timeout_ms),
        server_ping_interval_s=float(args.server_ping_interval_s),
        spawn_server_if_missing=bool(args.spawn_server_if_missing),
        kill_server_on_exit=bool(args.kill_server_on_exit),
        total_timeout_s=float(total_timeout_s) if total_timeout_s > 0 else None,
    )


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def _as_mapping_or_empty(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _as_list_or_empty(value: object) -> list[object]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _as_float_or_default(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _as_int_or_default(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(default)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return int(default)


def _canonicalize_for_digest(value: Any) -> Any:
    np = importlib.import_module("numpy")
    if isinstance(value, Mapping):
        return {
            str(k): _canonicalize_for_digest(value[k]) for k in sorted(value.keys())
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_digest(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return round(float(value), TRAJECTORY_DIGEST_DECIMALS)
    try:
        arr = np.asarray(value)
    except Exception:
        return str(value)
    if arr.dtype.kind in {"f", "c"}:
        arr = np.round(arr.astype(np.float64), TRAJECTORY_DIGEST_DECIMALS)
    return arr.tolist()


def _digest_payload(payload: object) -> str:
    return _sha256_text(
        json.dumps(
            _canonicalize_for_digest(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _coerce_success_flag(value: object) -> bool:
    np = importlib.import_module("numpy")
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return bool(int(value))
    if isinstance(value, list):
        return any(_coerce_success_flag(item) for item in value)
    if isinstance(value, tuple):
        return any(_coerce_success_flag(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return False


def _flatten_numeric_leaves(value: object) -> list[float]:
    np = importlib.import_module("numpy")
    if isinstance(value, Mapping):
        flattened: list[float] = []
        for key in sorted(value.keys()):
            flattened.extend(_flatten_numeric_leaves(value[key]))
        return flattened
    try:
        arr = np.asarray(value)
    except Exception:
        return []
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        return []
    return [float(item) for item in arr.reshape(-1)]


def _numeric_observation_vector(obs: Mapping[str, object]) -> list[float]:
    values: list[float] = []
    for key in sorted(obs.keys()):
        name = str(key)
        if name.startswith("annotation."):
            continue
        if name.startswith("video.") or name.endswith("_image"):
            continue
        values.extend(_flatten_numeric_leaves(obs[key]))
    return values


def _stack_batch(values: Sequence[object]) -> object:
    np = importlib.import_module("numpy")
    first = values[0]
    if isinstance(first, Mapping):
        return {
            str(key): _stack_batch(
                [cast(Mapping[str, object], item)[str(key)] for item in values]
            )
            for key in sorted(first.keys())
        }
    if isinstance(first, str):
        return np.asarray([str(item) for item in values], dtype=np.str_)
    try:
        arrays = [np.asarray(item) for item in values]
        if arrays and all(
            arr.dtype.kind in {"U", "S"}
            or (
                arr.dtype.kind == "O"
                and all(isinstance(item, str) for item in arr.reshape(-1).tolist())
            )
            for arr in arrays
        ):
            arrays = [np.asarray(arr, dtype=np.str_) for arr in arrays]
        stacked = np.stack(arrays, axis=0)
        if stacked.dtype.kind == "O":
            raise TypeError(
                "object-dtype observation batches are unsupported in live public anchor rollout"
            )
        return stacked
    except Exception:
        fallback = np.asarray(list(values))
        if getattr(fallback, "dtype", None) is not None and fallback.dtype.kind == "O":
            raise TypeError(
                "object-dtype observation batches are unsupported in live public anchor rollout"
            )
        return fallback


def _slice_batch_item(value: object, index: int) -> object:
    np = importlib.import_module("numpy")
    if isinstance(value, Mapping):
        return {
            str(key): _slice_batch_item(value[key], index)
            for key in sorted(value.keys())
        }
    arr = np.asarray(value)
    return arr[index]


def _normalize_policy_observation_value(key: str, value: object) -> object:
    np = importlib.import_module("numpy")
    if isinstance(value, Mapping):
        return {
            str(child_key): _normalize_policy_observation_value(
                f"{key}.{str(child_key)}" if key else str(child_key),
                value[child_key],
            )
            for child_key in sorted(value.keys())
        }
    arr = np.asarray(value)
    if key.startswith("annotation."):
        if arr.dtype.kind in {"U", "S", "O"}:
            return arr.astype(str).tolist()
        return np.asarray(arr, dtype=np.str_).tolist()
    if key.startswith("video.") or key.endswith("_image"):
        return np.asarray(arr, dtype=np.uint8)
    if arr.dtype.kind in {"b", "i", "u", "f"}:
        return np.asarray(arr, dtype=np.float32)
    return value


def _normalize_policy_observation(obs: Mapping[str, object]) -> dict[str, object]:
    sanitized_obs = recap_policy.filter_canonical_serving_observation(
        obs,
        field_name="public_anchor.policy_observation",
    )
    return {
        str(key): _normalize_policy_observation_value(str(key), sanitized_obs[key])
        for key in sorted(sanitized_obs.keys())
    }


def _finalize_stats_bucket(bucket: MutableMapping[str, float]) -> dict[str, Any]:
    total_count = int(bucket.get("total_count", 0.0))
    abs_sum = float(bucket.get("abs_sum", 0.0))
    max_abs = float(bucket.get("max_abs", 0.0))
    nonfinite_count = int(bucket.get("nonfinite_count", 0.0))
    zero_count = int(bucket.get("zero_count", 0.0))
    saturated_count = int(bucket.get("saturated_count", 0.0))
    mean_abs = float(abs_sum / total_count) if total_count > 0 else 0.0
    zero_fraction = float(zero_count / total_count) if total_count > 0 else 0.0
    saturated_fraction = (
        float(saturated_count / total_count) if total_count > 0 else 0.0
    )
    return {
        "total_count": total_count,
        "mean_abs": _round_float(mean_abs),
        "max_abs": _round_float(max_abs),
        "nonfinite_count": nonfinite_count,
        "zero_fraction": _round_float(zero_fraction),
        "saturated_fraction": _round_float(saturated_fraction),
    }


def _update_stats_bucket(
    bucket: MutableMapping[str, float], values: Sequence[float]
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    if not values:
        return _finalize_stats_bucket(bucket)
    arr = np.asarray(list(values), dtype=np.float64)
    finite_mask = np.isfinite(arr)
    finite = arr[finite_mask]
    bucket["total_count"] = float(bucket.get("total_count", 0.0) + int(arr.size))
    bucket["nonfinite_count"] = float(
        bucket.get("nonfinite_count", 0.0) + int(arr.size - finite.size)
    )
    if finite.size > 0:
        abs_arr = np.abs(finite)
        bucket["abs_sum"] = float(bucket.get("abs_sum", 0.0) + float(abs_arr.sum()))
        bucket["max_abs"] = float(max(bucket.get("max_abs", 0.0), float(abs_arr.max())))
        bucket["zero_count"] = float(
            bucket.get("zero_count", 0.0) + int(np.sum(abs_arr <= ACTION_ZERO_EPS))
        )
        bucket["saturated_count"] = float(
            bucket.get("saturated_count", 0.0)
            + int(np.sum(abs_arr >= SATURATION_ABS_THRESHOLD))
        )
    return _finalize_stats_bucket(bucket)


def _update_action_stats(
    aggregate: MutableMapping[str, MutableMapping[str, float]],
    action: Mapping[str, object],
) -> tuple[dict[str, Any], bool]:
    per_key: dict[str, Any] = {}
    any_nonfinite = False
    for key in sorted(action.keys()):
        numeric_values = _flatten_numeric_leaves(action[key])
        bucket = aggregate.setdefault(str(key), {})
        stats = _update_stats_bucket(bucket, numeric_values)
        per_key[str(key)] = stats
        if int(stats["nonfinite_count"]) > 0:
            any_nonfinite = True
    return per_key, any_nonfinite


def _build_scope_guard(config: EvalConfig) -> dict[str, Any]:
    violations: list[str] = []
    if str(config.model_path) != DEFAULT_MODEL_PATH:
        violations.append("wrong_checkpoint_evaluation")
    if str(config.embodiment_tag) != DEFAULT_EMBODIMENT_TAG:
        violations.append("wrong_checkpoint_evaluation")
    if str(config.env_name) != DEFAULT_ENV_NAME:
        violations.append("formal_protocol_drift")
    if str(config.mode) == "formal":
        if int(config.n_episodes) != FORMAL_DEFAULT_N_EPISODES:
            violations.append("formal_protocol_drift")
        if int(config.n_envs) != FORMAL_DEFAULT_N_ENVS:
            violations.append("formal_protocol_drift")
        if int(config.max_episode_steps) != FORMAL_DEFAULT_MAX_EPISODE_STEPS:
            violations.append("formal_protocol_drift")
        if int(config.n_action_steps) != FORMAL_DEFAULT_N_ACTION_STEPS:
            violations.append("formal_protocol_drift")
        if tuple(config.seed_list) != FORMAL_DEFAULT_SEED_LIST:
            violations.append("formal_protocol_drift")
    normalized_violations = sorted(set(violations))
    return {
        "comparability_scope": "official_public_anchor_unitree_g1_only",
        "requested_mode": str(config.mode),
        "requested_env_name": str(config.env_name),
        "requested_model_path": str(config.model_path),
        "requested_embodiment_tag": str(config.embodiment_tag),
        "required_env_name": DEFAULT_ENV_NAME,
        "required_model_path": DEFAULT_MODEL_PATH,
        "required_embodiment_tag": DEFAULT_EMBODIMENT_TAG,
        "public_anchor_comparable": bool(
            str(config.embodiment_tag) == DEFAULT_EMBODIMENT_TAG
            and str(config.model_path) == DEFAULT_MODEL_PATH
            and str(config.env_name) == DEFAULT_ENV_NAME
            and not normalized_violations
        ),
        "new_embodiment_public_anchor_comparable": False,
        "violations": normalized_violations,
    }


def _public_reference_snapshot() -> dict[str, Any]:
    return {
        "benchmark_task": "PnPAppleToPlate",
        "official_public_checkpoint": DEFAULT_MODEL_PATH,
        "required_embodiment_tag": DEFAULT_EMBODIMENT_TAG,
        "env_name": DEFAULT_ENV_NAME,
        "public_anchor_success_rate_reference": EXPECTED_PUBLIC_REFERENCE_SUCCESS_RATE,
        "expected_variance_note": EXPECTED_PUBLIC_REFERENCE_VARIANCE,
        "formal_defaults": {
            "n_episodes": FORMAL_DEFAULT_N_EPISODES,
            "n_envs": FORMAL_DEFAULT_N_ENVS,
            "max_episode_steps": FORMAL_DEFAULT_MAX_EPISODE_STEPS,
            "n_action_steps": FORMAL_DEFAULT_N_ACTION_STEPS,
            "seed_list": list(FORMAL_DEFAULT_SEED_LIST),
        },
        "public_scope_warning": (
            "This runner materializes only the official UNITREE_G1 public anchor. "
            "NEW_EMBODIMENT remains non-comparable to the public 58% line here."
        ),
    }


def _build_wrapper_configs(
    *,
    video_dir: Path | None,
    max_episode_steps: int,
    n_action_steps: int,
    video_delta_indices: object,
    state_delta_indices: object,
) -> Any:
    rollout_mod = importlib.import_module("gr00t.eval.rollout_policy")
    WrapperConfigs = getattr(rollout_mod, "WrapperConfigs")
    VideoConfig = getattr(rollout_mod, "VideoConfig")
    MultiStepConfig = getattr(rollout_mod, "MultiStepConfig")
    return WrapperConfigs(
        video=VideoConfig(
            video_dir=str(video_dir) if video_dir is not None else None,
            max_episode_steps=int(max_episode_steps),
            overlay_text=True,
        ),
        multistep=MultiStepConfig(
            video_delta_indices=video_delta_indices,
            state_delta_indices=state_delta_indices,
            n_action_steps=int(n_action_steps),
            max_episode_steps=int(max_episode_steps),
            terminate_on_success=True,
        ),
    )


def _create_eval_env(
    *,
    env_name: str,
    env_idx: int,
    total_n_envs: int,
    wrapper_configs: object,
) -> Any:
    rollout_mod = importlib.import_module("gr00t.eval.rollout_policy")
    fn = getattr(rollout_mod, "create_eval_env")
    return fn(
        env_name=str(env_name),
        env_idx=int(env_idx),
        total_n_envs=int(total_n_envs),
        wrapper_configs=wrapper_configs,
    )


def _create_policy_client(config: EvalConfig, client_host: str) -> Any:
    rollout_mod = importlib.import_module("gr00t.eval.rollout_policy")
    create_gr00t_sim_policy = getattr(rollout_mod, "create_gr00t_sim_policy")
    env_utils = importlib.import_module("gr00t.eval.sim.env_utils")
    get_embodiment_tag_from_env_name = getattr(
        env_utils, "get_embodiment_tag_from_env_name"
    )
    embodiment_tag = get_embodiment_tag_from_env_name(str(config.env_name))
    return create_gr00t_sim_policy(
        model_path="",
        embodiment_tag=embodiment_tag,
        policy_client_host=str(client_host),
        policy_client_port=int(config.server_port),
    )


def _motion_l2(
    start_vector: Sequence[float], end_vector: Sequence[float]
) -> float | None:
    np = importlib.import_module("numpy")
    if not start_vector or not end_vector:
        return None
    if len(start_vector) != len(end_vector):
        return None
    start_arr = np.asarray(list(start_vector), dtype=np.float64)
    end_arr = np.asarray(list(end_vector), dtype=np.float64)
    return float(np.linalg.norm(end_arr - start_arr))


def _episode_slot_state(
    *, episode_index: int, seed: int, env_slot: int, obs: Mapping[str, object]
) -> dict[str, Any]:
    return {
        "episode_index": int(episode_index),
        "seed": int(seed),
        "env_slot": int(env_slot),
        "obs": dict(obs),
        "outer_steps": 0,
        "success": False,
        "action_stats": {},
        "action_step_digests": [],
        "start_obs_vector": _numeric_observation_vector(obs),
        "last_obs_vector": _numeric_observation_vector(obs),
    }


def _reset_episode_slot(
    env: Any,
    *,
    episode_index: int,
    seed: int,
    env_slot: int,
) -> dict[str, Any]:
    obs, _info = env.reset(seed=int(seed))
    if not isinstance(obs, Mapping):
        raise TypeError(f"env.reset() returned non-dict obs: {type(obs).__name__}")
    return _episode_slot_state(
        episode_index=int(episode_index),
        seed=int(seed),
        env_slot=int(env_slot),
        obs=cast(Mapping[str, object], obs),
    )


def _finalize_episode(slot_state: Mapping[str, Any]) -> dict[str, Any]:
    motion_l2 = _motion_l2(
        cast(Sequence[float], slot_state.get("start_obs_vector", [])),
        cast(Sequence[float], slot_state.get("last_obs_vector", [])),
    )
    per_key_stats = {
        str(key): _finalize_stats_bucket(cast(MutableMapping[str, float], value))
        for key, value in cast(
            Mapping[str, MutableMapping[str, float]], slot_state.get("action_stats", {})
        ).items()
    }
    trajectory_fingerprint = _digest_payload(
        {
            "action_step_digests": list(slot_state.get("action_step_digests", [])),
            "motion_l2": _round_float(motion_l2),
            "success": bool(slot_state.get("success", False)),
            "outer_steps": int(slot_state.get("outer_steps", 0)),
        }
    )
    controller_saturation_detected = any(
        _as_float_or_default(
            cast(Mapping[str, object], stats).get("saturated_fraction", 0.0)
        )
        >= SATURATION_FRACTION_THRESHOLD
        for stats in per_key_stats.values()
    )
    zero_motion_detected = (
        motion_l2 is not None and float(motion_l2) <= ZERO_MOTION_STATE_DELTA_THRESHOLD
    )
    return {
        "episode_index": int(slot_state.get("episode_index", 0)),
        "seed": int(slot_state.get("seed", 0)),
        "env_slot": int(slot_state.get("env_slot", 0)),
        "outer_steps": int(slot_state.get("outer_steps", 0)),
        "success": bool(slot_state.get("success", False)),
        "motion_l2": _round_float(motion_l2),
        "controller_saturation_detected": bool(controller_saturation_detected),
        "zero_motion_detected": bool(zero_motion_detected),
        "trajectory_fingerprint": trajectory_fingerprint,
        "action_stats": per_key_stats,
    }


def _empty_execution_payload(
    *,
    config: EvalConfig,
    runtime_dir: Path,
    server_log: Path,
    client_log: Path,
    runtime_status: str,
    systemic_break_flags: Sequence[str],
    systemic_break_details: Mapping[str, object],
    scope_guard: Mapping[str, object],
) -> dict[str, Any]:
    return {
        "runtime_status": str(runtime_status),
        "mode": str(config.mode),
        "requested_n_episodes": int(config.n_episodes),
        "requested_n_envs": int(config.n_envs),
        "requested_seed_list": list(config.seed_list),
        "completed_episodes": 0,
        "successes": [],
        "success_count": 0,
        "success_rate": 0.0,
        "episode_summaries": [],
        "global_action_stats": {},
        "scope_guard": dict(scope_guard),
        "systemic_break_flags": sorted(set(str(item) for item in systemic_break_flags)),
        "systemic_break_details": dict(systemic_break_details),
        "modality_config_keys": [],
        "server_action_horizon": None,
        "runtime_dir": str(runtime_dir),
        "server_log_path": str(server_log),
        "client_log_path": str(client_log),
        "video_dir": None,
        "archived_video_dir": None,
        "server_reused_existing": None,
        "server_spawned_by_runner": None,
    }


def _run_seeded_rollout(
    *,
    config: EvalConfig,
    runtime_dir: Path,
    server_log: Path,
    client_log: Path,
    artifacts_videos: Path,
) -> dict[str, Any]:
    scope_guard = _build_scope_guard(config)
    if not bool(scope_guard.get("public_anchor_comparable", False)):
        return _empty_execution_payload(
            config=config,
            runtime_dir=runtime_dir,
            server_log=server_log,
            client_log=client_log,
            runtime_status="SKIPPED_SCOPE_MISMATCH",
            systemic_break_flags=list(
                cast(Sequence[str], scope_guard.get("violations", []))
            ),
            systemic_break_details={"scope_guard": dict(scope_guard)},
            scope_guard=scope_guard,
        )

    client: Any | None = None
    proc: subprocess.Popen[str] | None = None
    started_by_me = False
    envs: list[Any] = []
    video_dir: Path | None = None
    np = importlib.import_module("numpy")
    host_for_client = _normalize_client_host(str(config.server_host))
    try:
        client, proc, started_by_me = _ensure_server_ready(
            config, REPO_ROOT, server_log
        )
        assert client is not None
        modality_cfg = cast(Mapping[str, object], client.get_modality_config())
        action_horizon = state_conditioned_phase0_smoke._infer_action_horizon(
            modality_cfg
        )
        video_delta = state_conditioned_phase0_smoke._delta_indices_from_modality(
            modality_cfg.get("video")
        )
        state_delta = state_conditioned_phase0_smoke._delta_indices_from_modality(
            modality_cfg.get("state")
        )
        video_delta_indices = (
            np.asarray(video_delta, dtype=np.int64)
            if video_delta
            else np.asarray([0], dtype=np.int64)
        )
        state_delta_indices = (
            np.asarray(state_delta, dtype=np.int64)
            if state_delta
            else np.asarray([0], dtype=np.int64)
        )
        video_dir = _make_video_dir(
            env_name=str(config.env_name), n_action_steps=int(config.n_action_steps)
        )
        active_env_count = min(int(config.n_envs), int(config.n_episodes))
        for env_idx in range(active_env_count):
            wrapper_configs = _build_wrapper_configs(
                video_dir=video_dir if env_idx == 0 else None,
                max_episode_steps=int(config.max_episode_steps),
                n_action_steps=int(config.n_action_steps),
                video_delta_indices=video_delta_indices,
                state_delta_indices=state_delta_indices,
            )
            envs.append(
                _create_eval_env(
                    env_name=str(config.env_name),
                    env_idx=int(env_idx),
                    total_n_envs=int(config.n_envs),
                    wrapper_configs=wrapper_configs,
                )
            )

        policy = _create_policy_client(config, host_for_client)
        try:
            policy.reset()
        except Exception:
            pass

        seed_queue = list(enumerate(config.seed_list))
        active_slots: dict[int, dict[str, Any]] = {}
        for env_slot, env in enumerate(envs):
            if not seed_queue:
                break
            episode_index, seed = seed_queue.pop(0)
            active_slots[env_slot] = _reset_episode_slot(
                env,
                episode_index=int(episode_index),
                seed=int(seed),
                env_slot=int(env_slot),
            )

        successes: list[bool] = []
        episode_summaries: list[dict[str, Any]] = []
        systemic_break_details: dict[str, Any] = {
            "nan_or_inf_action_episode_indices": [],
            "zero_motion_episode_indices": [],
            "controller_saturation_episode_indices": [],
            "scope_guard": dict(scope_guard),
        }
        global_action_stats: dict[str, MutableMapping[str, float]] = {}
        runtime_status = "COMPLETED"
        abort_reason: str | None = None
        completed_episodes = 0
        while active_slots and completed_episodes < int(config.n_episodes):
            ordered_slots = sorted(active_slots.keys())
            batch_obs = cast(
                Mapping[str, object],
                _stack_batch([active_slots[slot]["obs"] for slot in ordered_slots]),
            )
            batch_obs = _normalize_policy_observation(batch_obs)
            action_batch, _action_info = policy.get_action(batch_obs)
            for batch_index, env_slot in enumerate(ordered_slots):
                slot_state = active_slots[env_slot]
                action = cast(
                    Mapping[str, object], _slice_batch_item(action_batch, batch_index)
                )
                per_key_stats, any_nonfinite = _update_action_stats(
                    cast(
                        MutableMapping[str, MutableMapping[str, float]],
                        slot_state["action_stats"],
                    ),
                    action,
                )
                for key in sorted(per_key_stats.keys()):
                    key_bucket = global_action_stats.setdefault(str(key), {})
                    _update_stats_bucket(
                        key_bucket, _flatten_numeric_leaves(action[key])
                    )
                slot_state["action_step_digests"].append(_digest_payload(action))
                if any_nonfinite:
                    runtime_status = "SYSTEMIC_BREAK_ABORTED"
                    abort_reason = "nan_or_inf_action"
                    systemic_break_details["nan_or_inf_action_episode_indices"].append(
                        int(slot_state["episode_index"])
                    )
                    break

                next_obs, _reward, term, trunc, info = envs[env_slot].step(action)
                slot_state["outer_steps"] = int(slot_state["outer_steps"]) + 1
                if isinstance(info, Mapping) and "success" in info:
                    slot_state["success"] = bool(
                        slot_state["success"]
                    ) or _coerce_success_flag(info.get("success"))
                if isinstance(next_obs, Mapping):
                    slot_state["obs"] = dict(cast(Mapping[str, object], next_obs))
                    slot_state["last_obs_vector"] = _numeric_observation_vector(
                        cast(Mapping[str, object], next_obs)
                    )
                done = bool(term) or bool(trunc)
                if not done:
                    continue
                summary = _finalize_episode(slot_state)
                episode_summaries.append(summary)
                successes.append(bool(summary["success"]))
                completed_episodes += 1
                if bool(summary["zero_motion_detected"]):
                    systemic_break_details["zero_motion_episode_indices"].append(
                        int(summary["episode_index"])
                    )
                if bool(summary["controller_saturation_detected"]):
                    systemic_break_details[
                        "controller_saturation_episode_indices"
                    ].append(int(summary["episode_index"]))
                if seed_queue:
                    next_episode_index, next_seed = seed_queue.pop(0)
                    active_slots[env_slot] = _reset_episode_slot(
                        envs[env_slot],
                        episode_index=int(next_episode_index),
                        seed=int(next_seed),
                        env_slot=int(env_slot),
                    )
                else:
                    del active_slots[env_slot]
            if abort_reason is not None:
                break

        archived_video_dir = _archive_video_dir(
            video_dir=video_dir,
            archive_root=artifacts_videos,
        )
        systemic_break_flags: list[str] = []
        if abort_reason is not None:
            systemic_break_flags.append(str(abort_reason))
        if (
            action_horizon is None
            or int(action_horizon) != EXPECTED_SERVER_ACTION_HORIZON
        ):
            systemic_break_flags.append("server_action_horizon_mismatch")
        zero_motion_indices = list(
            systemic_break_details["zero_motion_episode_indices"]
        )
        if zero_motion_indices:
            systemic_break_flags.append("zero_motion_episodes")
        saturation_indices = list(
            systemic_break_details["controller_saturation_episode_indices"]
        )
        if saturation_indices:
            systemic_break_flags.append("controller_saturation")
        if len(episode_summaries) > 1:
            fingerprints = {
                str(summary["trajectory_fingerprint"]) for summary in episode_summaries
            }
            if len(fingerprints) == 1:
                systemic_break_flags.append("all_identical_trajectories")
                systemic_break_details["all_identical_trajectory_fingerprint"] = next(
                    iter(fingerprints)
                )
        systemic_break_details["server_action_horizon"] = (
            int(action_horizon) if action_horizon is not None else None
        )
        systemic_break_details["expected_server_action_horizon"] = int(
            EXPECTED_SERVER_ACTION_HORIZON
        )
        success_count = int(sum(1 for item in successes if bool(item)))
        requested_n_episodes = int(config.n_episodes)
        success_rate = (
            float(success_count / requested_n_episodes)
            if requested_n_episodes > 0
            else 0.0
        )
        return {
            "runtime_status": str(runtime_status),
            "mode": str(config.mode),
            "requested_n_episodes": requested_n_episodes,
            "requested_n_envs": int(config.n_envs),
            "requested_seed_list": list(config.seed_list),
            "completed_episodes": int(completed_episodes),
            "successes": list(successes),
            "success_count": int(success_count),
            "success_rate": float(success_rate),
            "episode_summaries": list(episode_summaries),
            "global_action_stats": {
                str(key): _finalize_stats_bucket(bucket)
                for key, bucket in global_action_stats.items()
            },
            "scope_guard": dict(scope_guard),
            "systemic_break_flags": sorted(set(systemic_break_flags)),
            "systemic_break_details": dict(systemic_break_details),
            "modality_config_keys": sorted(str(key) for key in modality_cfg.keys()),
            "server_action_horizon": int(action_horizon)
            if action_horizon is not None
            else None,
            "runtime_dir": str(runtime_dir),
            "server_log_path": str(server_log),
            "client_log_path": str(client_log),
            "video_dir": str(video_dir) if video_dir is not None else None,
            "archived_video_dir": str(archived_video_dir)
            if archived_video_dir is not None
            else None,
            "server_reused_existing": not bool(started_by_me),
            "server_spawned_by_runner": bool(started_by_me),
        }
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        if client is not None and started_by_me and bool(config.kill_server_on_exit):
            print("[INFO] cleanup: stopping server started by this script")
            try:
                _safe_kill_server(client, int(config.server_ping_timeout_ms))
            except Exception:
                pass
            if proc is not None:
                _terminate_process(proc, timeout_s=10.0)
            _wait_for_port_release(
                _normalize_client_host(str(config.server_host)),
                int(config.server_port),
                timeout_s=15.0,
            )


def _build_smoke_payload(
    *, config: EvalConfig, execution: Mapping[str, object], output_dir: Path
) -> dict[str, Any]:
    scope_guard = _as_mapping_or_empty(execution.get("scope_guard"))
    modality_config_keys = _as_list_or_empty(execution.get("modality_config_keys"))
    systemic_break_flags = _as_list_or_empty(execution.get("systemic_break_flags"))
    systemic_break_details = _as_mapping_or_empty(
        execution.get("systemic_break_details")
    )
    episode_summaries = _as_list_or_empty(execution.get("episode_summaries"))
    payload = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "artifact_kind": SMOKE_ARTIFACT_KIND,
        "artifact_path": str(output_dir / SMOKE_JSON_NAME),
        "mode": "smoke",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "public_anchor_reference": _public_reference_snapshot(),
        "public_anchor_scope": dict(scope_guard),
        "protocol": {
            "env_name": str(config.env_name),
            "model_path": str(config.model_path),
            "embodiment_tag": str(config.embodiment_tag),
            "n_episodes": int(config.n_episodes),
            "n_envs": int(config.n_envs),
            "max_episode_steps": int(config.max_episode_steps),
            "n_action_steps": int(config.n_action_steps),
            "seed_list": list(config.seed_list),
        },
        "runtime_status": str(execution.get("runtime_status", "UNKNOWN")),
        "server_action_horizon": execution.get("server_action_horizon"),
        "modality_config_keys": list(modality_config_keys),
        "success_rate": _as_float_or_default(execution.get("success_rate")),
        "success_count": _as_int_or_default(execution.get("success_count")),
        "completed_episodes": _as_int_or_default(execution.get("completed_episodes")),
        "systemic_break_flags": list(systemic_break_flags),
        "systemic_break_details": dict(systemic_break_details),
        "episode_summaries": list(episode_summaries),
        "runtime_artifacts": {
            "runtime_dir": str(execution.get("runtime_dir")),
            "server_log_path": str(execution.get("server_log_path")),
            "client_log_path": str(execution.get("client_log_path")),
            "video_dir": execution.get("video_dir"),
            "archived_video_dir": execution.get("archived_video_dir"),
        },
        "ready_for_formal": bool(
            not systemic_break_flags
            and str(execution.get("runtime_status", "")) == "COMPLETED"
        ),
    }
    return payload


def _build_formal_payload(
    *, config: EvalConfig, execution: Mapping[str, object], output_dir: Path
) -> dict[str, Any]:
    scope_guard = _as_mapping_or_empty(execution.get("scope_guard"))
    modality_config_keys = _as_list_or_empty(execution.get("modality_config_keys"))
    successes = _as_list_or_empty(execution.get("successes"))
    episode_summaries = _as_list_or_empty(execution.get("episode_summaries"))
    global_action_stats = _as_mapping_or_empty(execution.get("global_action_stats"))
    systemic_break_flags = _as_list_or_empty(execution.get("systemic_break_flags"))
    systemic_break_details = _as_mapping_or_empty(
        execution.get("systemic_break_details")
    )
    payload = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "artifact_kind": FORMAL_ARTIFACT_KIND,
        "artifact_path": str(output_dir / FORMAL_JSON_NAME),
        "mode": "formal",
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "public_anchor_reference": _public_reference_snapshot(),
        "public_anchor_scope": dict(scope_guard),
        "formal_protocol": {
            "env_name": str(config.env_name),
            "model_path": str(config.model_path),
            "embodiment_tag": str(config.embodiment_tag),
            "n_episodes": int(config.n_episodes),
            "n_envs": int(config.n_envs),
            "max_episode_steps": int(config.max_episode_steps),
            "n_action_steps": int(config.n_action_steps),
            "seed_list": list(config.seed_list),
            "policy_horizon_expected": int(EXPECTED_SERVER_ACTION_HORIZON),
        },
        "runtime_status": str(execution.get("runtime_status", "UNKNOWN")),
        "modality_config_keys": list(modality_config_keys),
        "server_action_horizon": execution.get("server_action_horizon"),
        "completed_episodes": _as_int_or_default(execution.get("completed_episodes")),
        "requested_n_episodes": _as_int_or_default(
            execution.get("requested_n_episodes", config.n_episodes)
        ),
        "successes": list(successes),
        "success_count": _as_int_or_default(execution.get("success_count")),
        "success_rate": _as_float_or_default(execution.get("success_rate")),
        "episode_summaries": list(episode_summaries),
        "global_action_stats": dict(global_action_stats),
        "systemic_break_flags": list(systemic_break_flags),
        "systemic_break_details": dict(systemic_break_details),
        "runtime_artifacts": {
            "runtime_dir": str(execution.get("runtime_dir")),
            "server_log_path": str(execution.get("server_log_path")),
            "client_log_path": str(execution.get("client_log_path")),
            "video_dir": execution.get("video_dir"),
            "archived_video_dir": execution.get("archived_video_dir"),
        },
    }
    return payload


def _build_sanity_gate(
    formal_payload: Mapping[str, object], *, output_dir: Path
) -> dict[str, Any]:
    systemic_break_flags = [
        str(item)
        for item in _as_list_or_empty(formal_payload.get("systemic_break_flags"))
    ]
    success_count = _as_int_or_default(formal_payload.get("success_count"))
    n_episodes = _as_int_or_default(formal_payload.get("requested_n_episodes"))
    public_anchor_scope = _as_mapping_or_empty(
        formal_payload.get("public_anchor_scope")
    )
    formal_protocol = _as_mapping_or_empty(formal_payload.get("formal_protocol"))
    continue_to_audit = bool(success_count > 0 and systemic_break_flags == [])
    sanity_status = "PASS" if continue_to_audit else "BLOCK"
    if continue_to_audit:
        gate_reason = "non_zero_success_and_no_systemic_breaks"
    elif systemic_break_flags:
        gate_reason = "systemic_break_detected"
    else:
        gate_reason = "zero_success_anchor"
    return {
        "schema_version": SANITY_GATE_SCHEMA_VERSION,
        "artifact_kind": SANITY_GATE_ARTIFACT_KIND,
        "artifact_path": str(output_dir / SANITY_GATE_JSON_NAME),
        "formal_artifact_path": str(formal_payload.get("artifact_path")),
        "public_anchor_comparable": bool(
            public_anchor_scope.get("public_anchor_comparable", False)
        ),
        "new_embodiment_public_anchor_comparable": False,
        "success_rate": _as_float_or_default(formal_payload.get("success_rate")),
        "success_count": int(success_count),
        "n_episodes": int(n_episodes),
        "seed_list": list(_as_list_or_empty(formal_protocol.get("seed_list"))),
        "systemic_break_flags": list(systemic_break_flags),
        "systemic_break_details": dict(
            _as_mapping_or_empty(formal_payload.get("systemic_break_details"))
        ),
        "sanity_status": sanity_status,
        "continue_to_audit": bool(continue_to_audit),
        "gate_reason": gate_reason,
    }


def _build_failure_note(
    formal_payload: Mapping[str, object], gate_payload: Mapping[str, object]
) -> str:
    flags = [
        str(item)
        for item in _as_list_or_empty(gate_payload.get("systemic_break_flags"))
    ]
    lines = [
        "# GR00T public anchor sanity gate failure note",
        "",
        f"- sanity_status: `{gate_payload.get('sanity_status', 'BLOCK')}`",
        f"- gate_reason: `{gate_payload.get('gate_reason', 'systemic_break_detected')}`",
        f"- success_count: `{gate_payload.get('success_count', 0)}` / `{gate_payload.get('n_episodes', 0)}`",
        "- systemic_break_flags:",
    ]
    if flags:
        lines.extend(f"  - `{flag}`" for flag in flags)
    else:
        lines.append("  - `none captured`")
    lines.extend(
        [
            "",
            "## scope",
            "```json",
            _json_text(formal_payload.get("public_anchor_scope", {})),
            "```",
            "",
            "该 formal anchor 未满足“非零成功且无系统性崩坏”的宽松继续条件，因此不得直接继续到后续 audit/diagnostics。",
            "另外，本 runner 只服务于 `UNITREE_G1` 官方 public anchor；`NEW_EMBODIMENT` 不得在这里冒充公开 benchmark 可比线。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_failure_note(
    path: Path, formal_payload: Mapping[str, object], gate_payload: Mapping[str, object]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        _build_failure_note(formal_payload, gate_payload),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def _build_crash_execution(
    *,
    config: EvalConfig,
    runtime_dir: Path,
    server_log: Path,
    client_log: Path,
    exc: BaseException,
) -> dict[str, Any]:
    scope_guard = _build_scope_guard(config)
    details = {
        "exception_type": exc.__class__.__name__,
        "message": _exception_message(exc),
        "scope_guard": dict(scope_guard),
    }
    payload = _empty_execution_payload(
        config=config,
        runtime_dir=runtime_dir,
        server_log=server_log,
        client_log=client_log,
        runtime_status="CRASHED",
        systemic_break_flags=["crash"],
        systemic_break_details=details,
        scope_guard=scope_guard,
    )
    payload["error"] = details
    return payload


def materialize_public_anchor_eval(
    config: EvalConfig,
    *,
    repo_root: Path,
    live_eval_runner: Any = None,
) -> dict[str, Any]:
    resolved_output_dir = _validate_output_dir(config.output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir, artifacts_videos, server_log, client_log = _ensure_runtime_layout(
        repo_root, config.mode
    )
    exit_code = 0
    execution: dict[str, Any]
    runner = live_eval_runner or _run_seeded_rollout
    with _tee_stdio(client_log, header=f"gr00t_public_anchor_{config.mode}"):
        print("[INFO] ts:", _dt.datetime.now().isoformat(timespec="seconds"))
        print("[INFO] python:", sys.version.replace("\n", " "))
        print("[INFO] mode:", str(config.mode))
        print("[INFO] env_name:", str(config.env_name))
        print("[INFO] model_path:", str(config.model_path))
        print("[INFO] embodiment_tag:", str(config.embodiment_tag))
        print(
            "[INFO] n_episodes:",
            int(config.n_episodes),
            "n_envs:",
            int(config.n_envs),
        )
        print(
            "[INFO] max_episode_steps:",
            int(config.max_episode_steps),
            "n_action_steps:",
            int(config.n_action_steps),
        )
        try:
            execution = cast(
                dict[str, Any],
                runner(
                    config=config,
                    runtime_dir=runtime_dir,
                    server_log=server_log,
                    client_log=client_log,
                    artifacts_videos=artifacts_videos,
                ),
            )
        except BaseException as exc:
            execution = _build_crash_execution(
                config=config,
                runtime_dir=runtime_dir,
                server_log=server_log,
                client_log=client_log,
                exc=exc,
            )
            exit_code = 1

        if str(config.mode) == "smoke":
            smoke_payload = _build_smoke_payload(
                config=config,
                execution=execution,
                output_dir=resolved_output_dir,
            )
            smoke_path = _write_json(
                resolved_output_dir / SMOKE_JSON_NAME, smoke_payload
            )
            result = {
                "exit_code": int(exit_code),
                "mode": "smoke",
                "public_anchor_smoke_path": str(smoke_path),
                "runtime_status": str(execution.get("runtime_status", "UNKNOWN")),
            }
            print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
            return result

        formal_payload = _build_formal_payload(
            config=config,
            execution=execution,
            output_dir=resolved_output_dir,
        )
        formal_path = _write_json(
            resolved_output_dir / FORMAL_JSON_NAME, formal_payload
        )
        formal_payload["artifact_path"] = str(formal_path)
        if formal_path != resolved_output_dir / FORMAL_JSON_NAME:
            formal_path = _write_json(formal_path, formal_payload)
        gate_payload = _build_sanity_gate(
            formal_payload, output_dir=resolved_output_dir
        )
        failure_note_path: str | None = None
        if str(gate_payload.get("sanity_status")) == "BLOCK":
            failure_note = _write_failure_note(
                resolved_output_dir / FAILURE_NOTE_MARKDOWN_NAME,
                formal_payload,
                gate_payload,
            )
            failure_note_path = str(failure_note)
        else:
            stale_failure_note = resolved_output_dir / FAILURE_NOTE_MARKDOWN_NAME
            if stale_failure_note.exists():
                stale_failure_note.unlink()
        gate_payload["failure_note_path"] = failure_note_path
        gate_path = _write_json(
            resolved_output_dir / SANITY_GATE_JSON_NAME, gate_payload
        )
        result = {
            "exit_code": int(exit_code),
            "mode": "formal",
            "public_anchor_formal_path": str(formal_path),
            "public_anchor_sanity_gate_path": str(gate_path),
            "failure_note_path": failure_note_path,
            "runtime_status": str(execution.get("runtime_status", "UNKNOWN")),
        }
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
        return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _resolve_eval_config(args)
    repo_root = _repo_root()
    _maybe_reexec_into_wbc_venv(repo_root)
    _apply_env(str(config.mujoco_gl))
    _install_alarm_timeout(config.total_timeout_s)
    try:
        result = materialize_public_anchor_eval(config, repo_root=repo_root)
    except (OSError, TypeError, ValueError) as exc:
        print(f"public anchor eval failed: {_exception_message(exc)}", file=sys.stderr)
        return 1
    finally:
        _clear_alarm_timeout()
    return int(result.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
