#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib
import json
import os
import shutil
import signal
import subprocess
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
MODEL_PATH = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
EMBODIMENT_TAG = "UNITREE_G1"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5555

N_EPISODES = 1
MAX_POLICY_STEPS = 4
MAX_EPISODE_STEPS = 1440
N_ACTION_STEPS_CONFIG = 30

MUJOCO_GL = ""
ONSCREEN = False
OFFSCREEN = True

ITER_TAG = "recap_iter_000"
DATASET_DIR_REL = "agent/artifacts/recap_datasets"
VIDEO_ARCHIVE_DIR = "agent/artifacts/videos"
RUNTIME_LOGS_REL = "agent/runtime_logs"

TOTAL_TIMEOUT_S = 180


_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


from agent.run import state_conditioned_env_resolution
from work.recap.formal_branch_resolution import (
    BLOCKED_EXIT_CODE,
    FormalBranchResolutionBlocked,
    maybe_resolve_formal_collect_branch,
    maybe_reset_formal_nominal_dataset_dir,
)


def _repo_root() -> Path:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "repo_root")
    return cast(Path, fn(from_path=__file__))


def _maybe_reexec_into_wbc_venv(repo_root: Path) -> None:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "maybe_reexec_into_wbc_venv")
    fn(repo_root)


def _server_pythonpath(repo_root: Path) -> str | None:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "wbc_checkout_pythonpath")
    pythonpath_entries: list[str] = []
    for entry in [
        *list(fn(repo_root)),
        *str(os.environ.get("PYTHONPATH", "")).split(os.pathsep),
    ]:
        normalized = str(entry).strip()
        if normalized and normalized not in pythonpath_entries:
            pythonpath_entries.append(normalized)
    if not pythonpath_entries:
        return None
    return os.pathsep.join(pythonpath_entries)


def _ensure_dirs(repo_root: Path, *, iter_tag: str) -> tuple[Path, Path]:
    mod = importlib.import_module("work.demo_utils.paths")
    fn = getattr(mod, "ensure_dirs")
    runtime_dir, artifacts_videos = fn(
        repo_root=repo_root,
        runtime_logs_rel=str(Path(RUNTIME_LOGS_REL) / str(iter_tag)),
        artifacts_videos_rel=str(VIDEO_ARCHIVE_DIR),
    )
    return cast(Path, runtime_dir), cast(Path, artifacts_videos)


@contextlib.contextmanager
def _tee_stdio(log_path: Path, *, header: str) -> Iterator[None]:
    mod = importlib.import_module("work.demo_utils.tee")
    fn = getattr(mod, "tee_stdio")
    with fn(Path(log_path), header=str(header)):
        yield


def _make_video_dir(*, env_name: str, n_action_steps: int) -> Path:
    mod = importlib.import_module("work.demo_utils.videos")
    fn = getattr(mod, "make_video_dir")
    return cast(Path, fn(env_name=str(env_name), n_action_steps=int(n_action_steps)))


def _archive_video_dir(*, video_dir: Path | None, archive_root: Path) -> Path | None:
    mod = importlib.import_module("work.demo_utils.videos")
    fn = getattr(mod, "archive_video_dir")
    return cast(
        Path | None,
        fn(
            video_dir=(Path(video_dir) if video_dir is not None else None),
            archive_root=Path(archive_root),
        ),
    )


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


def _safe_kill_server(client: Any, timeout_ms: int) -> bool:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "safe_kill_server")
    return bool(fn(client, int(timeout_ms)))


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


def _install_interrupt_signal_handlers() -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def _handler(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"Signal {int(signum)}")

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            previous[int(sig)] = signal.getsignal(sig)
            signal.signal(sig, _handler)
        except Exception:
            continue
    return previous


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for sig_num, handler in previous.items():
        try:
            signal.signal(int(sig_num), handler)
        except Exception:
            continue


def _clear_alarm_timeout() -> None:
    if hasattr(signal, "SIGALRM"):
        try:
            signal.alarm(0)
        except Exception:
            pass


def _server_entrypoint(repo_root: Path) -> Path:
    return repo_root / "submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"


def _build_server_cmd(
    args: argparse.Namespace, repo_root: Path, *, device: str = "cuda"
) -> list[str]:
    server_py = _server_entrypoint(repo_root)
    return [
        sys.executable,
        str(server_py),
        "--model-path",
        str(args.model_path),
        "--embodiment-tag",
        str(args.embodiment_tag),
        "--device",
        str(device),
        "--use-sim-policy-wrapper",
        "--host",
        str(args.server_host),
        "--port",
        str(int(args.server_port)),
    ]


def _read_text_best_effort(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _server_log_indicates_cuda_oom(server_log: Path) -> bool:
    payload = _read_text_best_effort(server_log).lower()
    return "cuda out of memory" in payload or "outofmemoryerror" in payload


def _ensure_server_ready(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    server_log: Path,
    server_ready_timeout_s: float,
    server_ping_timeout_ms: int,
    server_ping_interval_s: float,
) -> tuple[Any, subprocess.Popen[str] | None, bool, str]:
    host_for_client = _normalize_client_host(str(args.server_host))
    client = _make_policy_client(
        host=host_for_client,
        port=int(args.server_port),
        timeout_ms=int(server_ping_timeout_ms),
    )

    print(f"[INFO] policy server target: {host_for_client}:{int(args.server_port)}")
    if _safe_ping(client, int(server_ping_timeout_ms)):
        print("[INFO] ping ok (reuse existing server)")
        return client, None, False, host_for_client

    if _is_tcp_port_listening(host_for_client, int(args.server_port)):
        raise RuntimeError(
            "port is already occupied but PolicyClient.ping() failed; refuse to kill unknown process. "
            "Try changing --server-port or stop the other service."
        )

    server_py = _server_entrypoint(repo_root)
    if not server_py.is_file():
        raise FileNotFoundError(f"missing server entrypoint: {server_py}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    server_pythonpath = _server_pythonpath(repo_root)
    if server_pythonpath:
        env["PYTHONPATH"] = str(server_pythonpath)
    else:
        env.pop("PYTHONPATH", None)
    if getattr(args, "mujoco_gl", ""):
        env["MUJOCO_GL"] = str(getattr(args, "mujoco_gl"))
        if str(getattr(args, "mujoco_gl")).lower() == "egl":
            env.setdefault("PYOPENGL_PLATFORM", "egl")

    launch_plan: list[tuple[str, Path]] = [("cuda", server_log)]
    cpu_fallback_log = server_log.with_name(
        server_log.stem + "_cpu_fallback" + server_log.suffix
    )

    for attempt_index, (server_device, server_log_path) in enumerate(launch_plan):
        cmd = _build_server_cmd(args, repo_root, device=server_device)
        print(f"[INFO] spawning GR00T server subprocess (device={server_device})...")
        proc = _spawn_server_subprocess(
            cmd,
            log_path=server_log_path,
            cwd=repo_root,
            env=env,
        )
        print(
            f"[INFO] server subprocess pid={proc.pid} device={server_device} log={server_log_path}"
        )

        t0 = time.monotonic()
        last_note = 0.0
        while True:
            if _safe_ping(client, int(server_ping_timeout_ms)):
                print("[INFO] ping ok")
                return client, proc, True, host_for_client
            if proc.poll() is not None:
                if (
                    attempt_index == 0
                    and server_device == "cuda"
                    and _server_log_indicates_cuda_oom(server_log_path)
                ):
                    print(
                        "[WARN] CUDA server start hit OOM; retrying once with CPU fallback."
                    )
                    launch_plan.append(("cpu", cpu_fallback_log))
                    break
                raise RuntimeError(
                    f"server subprocess exited early rc={proc.returncode}; see {server_log_path}"
                )
            dt = time.monotonic() - t0
            if dt > float(server_ready_timeout_s):
                raise TimeoutError(
                    f"timeout waiting for ping ok after {int(dt)}s; see {server_log_path}"
                )
            if dt - last_note >= 5.0:
                print(
                    f"[INFO] waiting for server ready (device={server_device})... {int(dt)}s"
                )
                last_note = dt
            time.sleep(float(server_ping_interval_s))

    raise RuntimeError(f"failed to start policy server; see {server_log}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="31_recap_collect_rollouts.py",
        description=(
            "RECAP rollout collector (server + WBC env + tee + archive + dataset jsonl/npz)."
        ),
    )
    p.add_argument("--env-name", type=str, default=ENV_NAME)
    p.add_argument("--model-path", type=str, default=MODEL_PATH)
    p.add_argument("--embodiment-tag", type=str, default=EMBODIMENT_TAG)
    p.add_argument("--server-host", type=str, default=SERVER_HOST)
    p.add_argument("--server-port", type=int, default=int(SERVER_PORT))
    p.add_argument("--mujoco-gl", type=str, default=MUJOCO_GL)
    p.add_argument("--iter-tag", type=str, default=ITER_TAG)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n-episodes", type=int, default=int(N_EPISODES))
    p.add_argument("--max-policy-steps", type=int, default=int(MAX_POLICY_STEPS))
    p.add_argument("--max-episode-steps", type=int, default=int(MAX_EPISODE_STEPS))
    p.add_argument(
        "--n-action-steps-config", type=int, default=int(N_ACTION_STEPS_CONFIG)
    )

    p.add_argument(
        "--policy-prompt-prefix",
        type=str,
        default="",
        help=(
            "Optional language prefix injected into obs['annotation.human.task_description'] "
            "before sending obs to the policy server (e.g. 'advantage positive ')."
        ),
    )

    bool_action = getattr(argparse, "BooleanOptionalAction", None)
    if bool_action is None:
        p.add_argument("--onscreen", action="store_true", default=bool(ONSCREEN))
        p.add_argument("--offscreen", action="store_true", default=bool(OFFSCREEN))
        p.add_argument(
            "--kill-server-on-exit",
            action="store_true",
            default=False,
            help="Kill server subprocess if started by this script.",
        )

        p.add_argument(
            "--mixdone",
            dest="mixdone",
            action="store_true",
            default=False,
            help=(
                "Two-phase single-process collection (short episodes then long episodes) "
                "while sharing ONE archived video_dir for the whole iter_tag."
            ),
        )
        p.add_argument(
            "--no-mixdone",
            dest="mixdone",
            action="store_false",
            help="Disable two-phase mixdone collection.",
        )
    else:
        p.add_argument(
            "--onscreen",
            action=bool_action,
            default=bool(ONSCREEN),
            help="Enable onscreen viewer.",
        )
        p.add_argument(
            "--offscreen",
            action=bool_action,
            default=bool(OFFSCREEN),
            help="Enable offscreen rendering.",
        )
        p.add_argument(
            "--kill-server-on-exit",
            action=bool_action,
            default=False,
            help="Kill server subprocess if started by this script.",
        )

        p.add_argument(
            "--mixdone",
            action=bool_action,
            default=False,
            help=(
                "Two-phase single-process collection (short episodes then long episodes) "
                "while sharing ONE archived video_dir for the whole iter_tag."
            ),
        )

    p.add_argument(
        "--mixdone-short-episodes",
        type=int,
        default=None,
        help=(
            "When --mixdone: number of short episodes to collect first. "
            "Default: n_episodes//2."
        ),
    )
    p.add_argument(
        "--mixdone-long-episodes",
        type=int,
        default=None,
        help=(
            "When --mixdone: number of long episodes to collect after shorts. "
            "Default: n_episodes - mixdone_short_episodes."
        ),
    )
    p.add_argument(
        "--mixdone-short-max-episode-steps",
        type=int,
        default=60,
        help="When --mixdone: max_episode_steps for short phase (default: 60).",
    )
    p.add_argument(
        "--mixdone-long-max-episode-steps",
        type=int,
        default=1440,
        help="When --mixdone: max_episode_steps for long phase (default: 1440).",
    )
    p.add_argument(
        "--mixdone-long-seed-offset",
        type=int,
        default=1000,
        help=(
            "When --mixdone: seed offset for long phase. "
            "Seeds: short=seed+i, long=seed+offset+j (default offset: 1000)."
        ),
    )

    p.add_argument(
        "--total-timeout-s",
        type=float,
        default=float(TOTAL_TIMEOUT_S),
        help="Hard timeout fuse (best-effort) for the whole script.",
    )
    return p


def _git_head_and_dirty(repo_root: Path) -> tuple[str, bool]:
    head = "unknown"
    dirty = False
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except Exception:
        head = "unknown"
    try:
        s = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        dirty = bool(str(s).strip())
    except Exception:
        dirty = False
    return str(head), bool(dirty)


def _delta_indices_from_modality(modality_cfg: Any) -> list[int]:
    if modality_cfg is None:
        return []
    raw = getattr(modality_cfg, "delta_indices", None)
    if raw is None:
        return []
    try:
        return [int(x) for x in list(raw)]
    except Exception:
        try:
            return [int(x) for x in raw]
        except Exception:
            return []


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _argv_has_flag(argv: Sequence[str], flag: str) -> bool:
    flag = str(flag)
    for a in argv:
        s = str(a)
        if s == flag:
            return True
        if s.startswith(flag + "="):
            return True
    return False


def _finalize_offscreen_video_and_close_env(
    env: Any | None, *, offscreen: bool, video_dir_tmp: Path | None
) -> None:
    if env is None:
        return

    if bool(offscreen) and video_dir_tmp is not None:
        try:
            env.reset()
        except Exception:
            pass

    try:
        env.close()
    except Exception:
        pass


class RecapCollectRolloutsWorkflow:
    def __init__(self) -> None:
        self.repo_root: Path | None = None
        self.parser: argparse.ArgumentParser | None = None
        self.args: argparse.Namespace | None = None
        self.iter_tag: str = str(ITER_TAG)
        self.runtime_dir: Path | None = None
        self.artifacts_videos: Path | None = None
        self.log_path: Path | None = None
        self.server_log: Path | None = None
        self.dataset_root: Path | None = None
        self.server_ping_timeout_ms: int = 2000
        self.server_ping_interval_s: float = 1.0
        self.server_ready_timeout_s: float = 120.0
        self.t0_total: float = 0.0
        self.previous_signal_handlers: dict[int, Any] | None = None
        self.started_by_me = False
        self.proc: subprocess.Popen[str] | None = None
        self.client: Any | None = None
        self.host_for_client = ""
        self.env: Any | None = None
        self.video_dir_tmp: Path | None = None
        self.archived_video_dir: Path | None = None
        self.code_version = "unknown"
        self.n_action_steps = int(N_ACTION_STEPS_CONFIG)
        self.resolved_env_name = str(ENV_NAME)
        self.writer: Any | None = None
        self.collect_episode: Any | None = None
        self.build_vector_env: Any | None = None
        self.max_policy_steps = int(MAX_POLICY_STEPS)
        self.n_episodes = int(N_EPISODES)
        self.total_eps = int(N_EPISODES)
        self.mixdone = False
        self.short_eps = 0
        self.long_eps = 0
        self.short_max_steps = int(MAX_EPISODE_STEPS)
        self.long_max_steps = int(MAX_EPISODE_STEPS)
        self.long_seed_offset = 0

    def run(self) -> int:
        if self._handle_help_request():
            return 0

        self._prepare_runtime_context()
        assert self.log_path is not None
        with _tee_stdio(self.log_path, header="31_recap_collect_rollouts"):
            _install_alarm_timeout(self._total_timeout_or_none())
            self.previous_signal_handlers = _install_interrupt_signal_handlers()
            try:
                self._prepare_collection_plan()
                results_buffer = self._collect_rollouts()
                self._archive_videos()
                self._persist_results(results_buffer)
                self._validate_video_artifacts(results_buffer)
                self._maybe_resolve_formal_branch()
                return 0
            except (
                state_conditioned_env_resolution.StateConditionedEnvResolutionError
            ) as exc:
                print(
                    "[ERROR] env_resolution:",
                    json.dumps(
                        exc.to_machine_payload(), ensure_ascii=True, sort_keys=True
                    ),
                    file=sys.stderr,
                )
                return 1
            except FormalBranchResolutionBlocked as exc:
                print(
                    json.dumps(
                        exc.to_machine_payload(), ensure_ascii=True, sort_keys=True
                    ),
                    file=sys.stderr,
                )
                return int(BLOCKED_EXIT_CODE)
            except KeyboardInterrupt:
                print("\n[INFO] KeyboardInterrupt -> stop early")
                return 130
            finally:
                self._cleanup()

    def _handle_help_request(self) -> bool:
        if not any(a in ("-h", "--help") for a in sys.argv[1:]):
            return False
        try:
            _build_parser().parse_args()
        except SystemExit as e:
            if int(getattr(e, "code", 0) or 0) != 0:
                raise
        return True

    def _prepare_runtime_context(self) -> None:
        self.repo_root = _repo_root()
        _maybe_reexec_into_wbc_venv(self.repo_root)
        self.parser = _build_parser()
        self.args = self.parser.parse_args()
        self._validate_mixdone_cli_contract()
        _apply_env(str(getattr(self.args, "mujoco_gl", "")))
        self.iter_tag = str(getattr(self.args, "iter_tag", "") or ITER_TAG)
        self.runtime_dir, self.artifacts_videos = _ensure_dirs(
            self.repo_root, iter_tag=self.iter_tag
        )
        self.log_path = self.runtime_dir / "collect.log"
        self.server_log = self.runtime_dir / "00_server.log"
        self.dataset_root = self.repo_root / str(DATASET_DIR_REL)
        timeout_s = float(getattr(self.args, "total_timeout_s", 0.0) or 0.0)
        self.server_ready_timeout_s = min(600.0, max(10.0, timeout_s * 0.8))
        if self.server_ready_timeout_s <= 0:
            self.server_ready_timeout_s = 120.0
        self.t0_total = time.monotonic()

    def _validate_mixdone_cli_contract(self) -> None:
        assert self.args is not None
        assert self.parser is not None
        if not bool(getattr(self.args, "mixdone", False)):
            return
        short_raw = getattr(self.args, "mixdone_short_episodes", None)
        long_raw = getattr(self.args, "mixdone_long_episodes", None)
        user_provided_n_episodes = _argv_has_flag(sys.argv[1:], "--n-episodes")
        resolved_total = int(getattr(self.args, "n_episodes"))
        if short_raw is not None and long_raw is not None:
            resolved_total = int(short_raw) + int(long_raw)
        if user_provided_n_episodes and int(getattr(self.args, "n_episodes")) != int(
            resolved_total
        ):
            self.parser.error(
                "--mixdone: --n-episodes is legacy; if provided, it must equal --mixdone-short-episodes + --mixdone-long-episodes."
            )

    def _total_timeout_or_none(self) -> float | None:
        assert self.args is not None
        timeout_s = float(getattr(self.args, "total_timeout_s", 0.0) or 0.0)
        return timeout_s or None

    def _prepare_collection_plan(self) -> None:
        assert self.repo_root is not None
        assert self.args is not None
        assert self.server_log is not None
        assert self.dataset_root is not None
        self._log_run_header()
        self._ensure_server_connection()
        self._load_modality_and_env_plan()
        self._prepare_episode_writer()
        self._resolve_collection_schedule()

    def _log_run_header(self) -> None:
        assert self.repo_root is not None
        assert self.args is not None
        head, dirty = _git_head_and_dirty(self.repo_root)
        self.code_version = f"{head}{'-dirty' if dirty else ''}"
        print("[INFO] ts:", _dt.datetime.now().isoformat(timespec="seconds"))
        print("[INFO] git_head:", head, "dirty:", bool(dirty))
        print("[INFO] python:", sys.version.replace("\n", " "))
        print("[INFO] sys.executable:", sys.executable)
        print("[INFO] iter_tag:", self.iter_tag)
        print("[INFO] ENV_NAME:", str(getattr(self.args, "env_name")))
        print("[INFO] MODEL_PATH:", str(getattr(self.args, "model_path")))
        print(
            "[INFO] policy_prompt_prefix:",
            repr(str(getattr(self.args, "policy_prompt_prefix", "") or "")),
        )
        print(
            "[INFO] server:",
            f"{str(getattr(self.args, 'server_host'))}:{int(getattr(self.args, 'server_port'))}",
        )
        print("[INFO] MUJOCO_GL:", os.environ.get("MUJOCO_GL"))
        print("[INFO] seed:", int(getattr(self.args, "seed")))

    def _ensure_server_connection(self) -> None:
        assert self.args is not None
        assert self.repo_root is not None
        assert self.server_log is not None
        self.client, self.proc, self.started_by_me, self.host_for_client = (
            _ensure_server_ready(
                args=self.args,
                repo_root=self.repo_root,
                server_log=self.server_log,
                server_ready_timeout_s=self.server_ready_timeout_s,
                server_ping_timeout_ms=self.server_ping_timeout_ms,
                server_ping_interval_s=self.server_ping_interval_s,
            )
        )
        assert self.client is not None

    def _load_modality_and_env_plan(self) -> None:
        assert self.args is not None
        assert self.client is not None
        modality_cfg = self.client.get_modality_config()
        cfg_keys = sorted([str(k) for k in getattr(modality_cfg, "keys", lambda: [])()])
        if not cfg_keys and isinstance(modality_cfg, dict):
            cfg_keys = sorted([str(k) for k in modality_cfg.keys()])
        print("[INFO] modality_cfg keys:", cfg_keys)
        action_horizon = 0
        if isinstance(modality_cfg, dict) and "action" in modality_cfg:
            action_horizon = int(
                len(_delta_indices_from_modality(modality_cfg.get("action")))
            )
        print("[INFO] server action_horizon:", int(action_horizon))
        n_action_steps_before = int(getattr(self.args, "n_action_steps_config"))
        self.n_action_steps = int(n_action_steps_before)
        if int(action_horizon) > 0 and int(n_action_steps_before) != int(
            action_horizon
        ):
            print(
                "[WARN] N_ACTION_STEPS_CONFIG != server action_horizon; override:",
                int(n_action_steps_before),
                "->",
                int(action_horizon),
            )
            self.n_action_steps = int(action_horizon)
        else:
            print("[INFO] n_action_steps_config aligned:", int(self.n_action_steps))

        video_delta = [0]
        state_delta = [0]
        if isinstance(modality_cfg, dict):
            vd = _delta_indices_from_modality(modality_cfg.get("video"))
            sd = _delta_indices_from_modality(modality_cfg.get("state"))
            if vd:
                video_delta = vd
            if sd:
                state_delta = sd
        print("[INFO] video_delta_indices:", [int(x) for x in video_delta])
        print("[INFO] state_delta_indices:", [int(x) for x in state_delta])

        safe_env = str(getattr(self.args, "env_name")).replace("/", "__")
        tag = f"{safe_env}__{self.iter_tag}__{_now_tag()}"
        if bool(getattr(self.args, "offscreen")):
            self.video_dir_tmp = _make_video_dir(
                env_name=tag, n_action_steps=int(self.n_action_steps)
            )
            print("[INFO] video_dir_tmp:", str(self.video_dir_tmp))
        else:
            print("[INFO] offscreen disabled; no video_dir_tmp")

        gym = importlib.import_module("gymnasium")
        np = importlib.import_module("numpy")
        importlib.import_module("gr00t_wbc.control.envs.robocasa.sync_env")
        env_resolution = (
            state_conditioned_env_resolution.resolve_apple_to_plate_g1_env_name(
                gym,
                requested_env_name=str(getattr(self.args, "env_name")),
            )
        )
        self.resolved_env_name = str(env_resolution["resolved_env_name"])
        print(
            "[INFO] env_resolution:",
            json.dumps(
                {
                    "logical_task": env_resolution["logical_task"],
                    "requested_env_name": env_resolution["requested_env_name"],
                    "resolved_env_name": env_resolution["resolved_env_name"],
                    "alias_applied": env_resolution["alias_applied"],
                    "available_close_matches": env_resolution[
                        "available_close_matches"
                    ],
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )

        def _build_vector_env(*, max_episode_steps: int):
            max_episode_steps = int(max_episode_steps)

            def _env_fn():
                base_env = gym.make(
                    self.resolved_env_name,
                    onscreen=bool(getattr(self.args, "onscreen")),
                    offscreen=bool(getattr(self.args, "offscreen")),
                    enable_waist=True,
                    randomize_cameras=False,
                    camera_names=["robot0_oak_egoview", "robot0_rs_tppview"],
                )
                importlib.import_module("robocasa")
                base_cfg_mod = importlib.import_module(
                    "gr00t_wbc.control.main.teleop.configs.configs"
                )
                BaseConfig = getattr(base_cfg_mod, "BaseConfig")
                n1_utils_mod = importlib.import_module(
                    "gr00t_wbc.control.utils.n1_utils"
                )
                WholeBodyControlWrapper = getattr(
                    n1_utils_mod, "WholeBodyControlWrapper"
                )
                wbc_config = BaseConfig(
                    wbc_version="gear_wbc", enable_waist=True
                ).to_dict()
                env1 = WholeBodyControlWrapper(base_env, wbc_config)
                if (
                    bool(getattr(self.args, "offscreen"))
                    and self.video_dir_tmp is not None
                ):
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
                    video_max_episode_steps = int(max_episode_steps)
                    try:
                        policy_limited_max = (
                            int(self.max_policy_steps) * int(self.n_action_steps) + 1
                        )
                        if policy_limited_max > 0:
                            video_max_episode_steps = min(
                                int(video_max_episode_steps), int(policy_limited_max)
                            )
                    except Exception:
                        pass
                    env1 = VideoRecordingWrapper(
                        env1,
                        video_recorder,
                        video_dir=Path(self.video_dir_tmp),
                        steps_per_render=2,
                        max_episode_steps=int(video_max_episode_steps),
                        overlay_text=True,
                    )
                ms_mod = importlib.import_module(
                    "gr00t.eval.sim.wrapper.multistep_wrapper"
                )
                MultiStepWrapper = getattr(ms_mod, "MultiStepWrapper")
                return MultiStepWrapper(
                    env1,
                    video_delta_indices=np.asarray(list(video_delta), dtype=np.int64),
                    state_delta_indices=np.asarray(list(state_delta), dtype=np.int64),
                    n_action_steps=int(self.n_action_steps),
                    max_episode_steps=int(max_episode_steps),
                    terminate_on_success=True,
                )

            return gym.vector.SyncVectorEnv([_env_fn])

        self.build_vector_env = _build_vector_env

    def _prepare_episode_writer(self) -> None:
        assert self.dataset_root is not None
        assert self.repo_root is not None
        _ = maybe_reset_formal_nominal_dataset_dir(
            self.repo_root,
            iter_tag=str(self.iter_tag),
        )
        recap_collector = importlib.import_module("work.recap.collector")
        self.collect_episode = getattr(recap_collector, "collect_episode")
        ew_mod = importlib.import_module("work.recap.episode_writer")
        EpisodeWriter = getattr(ew_mod, "EpisodeWriter")
        self.writer = EpisodeWriter(
            iter_tag=str(self.iter_tag),
            code_version=str(self.code_version),
            arrays_saved=True,
            dataset_root=Path(self.dataset_root),
        )
        writer = self.writer
        assert writer is not None
        print("[INFO] dataset_iter_dir:", str(writer.iter_dir))
        print("[INFO] episodes_path:", str(writer.episodes_path))
        print("[INFO] transitions_path:", str(writer.transitions_path))

    def _resolve_collection_schedule(self) -> None:
        assert self.args is not None
        self.n_episodes = int(getattr(self.args, "n_episodes"))
        if self.n_episodes <= 0:
            raise ValueError(f"n_episodes must be > 0, got {self.n_episodes}")
        self.max_policy_steps = int(getattr(self.args, "max_policy_steps"))
        if self.max_policy_steps <= 0:
            raise ValueError(
                f"max_policy_steps must be > 0, got {self.max_policy_steps}"
            )
        self.mixdone = bool(getattr(self.args, "mixdone", False))
        self.short_eps = 0
        self.long_eps = 0
        self.total_eps = int(self.n_episodes)
        if not self.mixdone:
            return
        short_raw = getattr(self.args, "mixdone_short_episodes", None)
        long_raw = getattr(self.args, "mixdone_long_episodes", None)
        n_episodes_for_mixdone = int(self.n_episodes)
        if short_raw is not None and long_raw is not None:
            n_episodes_for_mixdone = int(short_raw) + int(long_raw)
        if short_raw is None and long_raw is None:
            self.short_eps = int(n_episodes_for_mixdone) // 2
            self.long_eps = int(n_episodes_for_mixdone) - int(self.short_eps)
        elif short_raw is None and long_raw is not None:
            self.long_eps = int(long_raw)
            self.short_eps = int(n_episodes_for_mixdone) - int(self.long_eps)
        elif short_raw is not None and long_raw is None:
            self.short_eps = int(short_raw)
            self.long_eps = int(n_episodes_for_mixdone) - int(self.short_eps)
        else:
            assert short_raw is not None and long_raw is not None
            self.short_eps = int(short_raw)
            self.long_eps = int(long_raw)
        if self.short_eps < 0 or self.long_eps < 0:
            raise ValueError(
                "mixdone episode counts must be >= 0; "
                f"got short={self.short_eps} long={self.long_eps} (n_episodes={self.n_episodes})"
            )
        self.short_max_steps = int(
            getattr(self.args, "mixdone_short_max_episode_steps")
        )
        self.long_max_steps = int(getattr(self.args, "mixdone_long_max_episode_steps"))
        self.long_seed_offset = int(getattr(self.args, "mixdone_long_seed_offset"))
        self.total_eps = int(self.short_eps) + int(self.long_eps)
        print("[INFO] mixdone: enabled")
        print(
            "[INFO] mixdone config: "
            f"short_eps={int(self.short_eps)} short_max_episode_steps={int(self.short_max_steps)} "
            f"long_eps={int(self.long_eps)} long_max_episode_steps={int(self.long_max_steps)} "
            f"long_seed_offset={int(self.long_seed_offset)} total_eps={int(self.total_eps)}"
        )

    def _collect_rollouts(
        self,
    ) -> list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]]:
        results_buffer: list[
            tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]
        ] = []
        if self.mixdone:
            episode_counter = 0
            episode_counter = self._collect_phase(
                results_buffer=results_buffer,
                phase="short",
                phase_episodes=int(self.short_eps),
                max_episode_steps=int(self.short_max_steps),
                seed_base=int(getattr(self.args, "seed")),
                episode_counter=int(episode_counter),
            )
            self._collect_phase(
                results_buffer=results_buffer,
                phase="long",
                phase_episodes=int(self.long_eps),
                max_episode_steps=int(self.long_max_steps),
                seed_base=int(getattr(self.args, "seed")) + int(self.long_seed_offset),
                episode_counter=int(episode_counter),
            )
            return results_buffer
        assert self.build_vector_env is not None
        self.env = self.build_vector_env(
            max_episode_steps=int(getattr(self.args, "max_episode_steps"))
        )
        for ep_i in range(int(self.n_episodes)):
            self._timeout_manual_check()
            episode_id = f"{self.iter_tag}_ep{ep_i + 1:03d}_{_now_tag()}"
            seed_i = int(getattr(self.args, "seed")) + int(ep_i)
            print(
                f"[INFO] collect episode {ep_i + 1}/{int(self.n_episodes)} episode_id={episode_id} seed={seed_i}"
            )
            results_buffer.append(
                self._collect_episode_record(
                    episode_id=str(episode_id), seed_i=int(seed_i)
                )
            )
        return results_buffer

    def _collect_phase(
        self,
        *,
        results_buffer: list[
            tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]
        ],
        phase: str,
        phase_episodes: int,
        max_episode_steps: int,
        seed_base: int,
        episode_counter: int,
    ) -> int:
        phase_episodes = int(phase_episodes)
        if phase_episodes <= 0:
            print(
                f"[INFO] mixdone phase={phase}: skip (episodes={int(phase_episodes)})"
            )
            return int(episode_counter)
        print(
            f"[INFO] mixdone phase={phase}: start episodes={int(phase_episodes)} "
            f"max_episode_steps={int(max_episode_steps)} seed_base={int(seed_base)}"
        )
        if self.env is not None:
            self._finalize_env()
        assert self.build_vector_env is not None
        self.env = self.build_vector_env(max_episode_steps=int(max_episode_steps))
        for phase_i in range(int(phase_episodes)):
            self._timeout_manual_check()
            ep_i_global = int(episode_counter)
            episode_counter += 1
            episode_id = f"{self.iter_tag}_ep{ep_i_global + 1:03d}_{_now_tag()}"
            seed_i = int(seed_base) + int(phase_i)
            print(
                f"[INFO] collect episode {ep_i_global + 1}/{int(self.total_eps)} "
                f"phase={phase} phase_i={phase_i + 1}/{int(phase_episodes)} "
                f"episode_id={episode_id} seed={seed_i}"
            )
            results_buffer.append(
                self._collect_episode_record(
                    episode_id=str(episode_id), seed_i=int(seed_i)
                )
            )
        print(f"[INFO] mixdone phase={phase}: done")
        return int(episode_counter)

    def _collect_episode_record(
        self, *, episode_id: str, seed_i: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        assert self.args is not None
        assert self.collect_episode is not None
        assert self.client is not None
        return self.collect_episode(
            env=self.env,
            client=self.client,
            iter_tag=str(self.iter_tag),
            episode_id=str(episode_id),
            env_name=str(getattr(self.args, "env_name")),
            model_path=str(getattr(self.args, "model_path")),
            embodiment_tag=str(getattr(self.args, "embodiment_tag")),
            server_host=str(self.host_for_client),
            server_port=int(getattr(self.args, "server_port")),
            seed=int(seed_i),
            max_policy_steps=int(self.max_policy_steps),
            code_version=str(self.code_version),
            video_dir_tmp=(
                str(self.video_dir_tmp) if self.video_dir_tmp is not None else None
            ),
            video_dir_archived=None,
            arrays_saved=True,
            policy_prompt_prefix=str(
                getattr(self.args, "policy_prompt_prefix", "") or ""
            ),
            debug_print=print,
        )

    def _archive_videos(self) -> None:
        assert self.artifacts_videos is not None
        self._finalize_env()
        archive_root = Path(self.artifacts_videos) / str(self.iter_tag)
        archive_root.mkdir(parents=True, exist_ok=True)
        self.archived_video_dir = _archive_video_dir(
            video_dir=self.video_dir_tmp, archive_root=archive_root
        )
        print("[INFO] video_dir_archived:", str(self.archived_video_dir))
        if self.archived_video_dir is not None and self.video_dir_tmp is not None:
            try:
                if self.video_dir_tmp.is_dir():
                    shutil.rmtree(self.video_dir_tmp)
                    print("[INFO] cleaned up video_dir_tmp:", str(self.video_dir_tmp))
            except Exception as e:
                print(
                    "[WARN] failed to delete video_dir_tmp:",
                    str(self.video_dir_tmp),
                    type(e).__name__,
                    str(e),
                )

    def _persist_results(
        self,
        results_buffer: list[
            tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]
        ],
    ) -> None:
        assert self.writer is not None
        n_transitions_total = 0
        for ep_record, transitions, arrays_blob in results_buffer:
            ep_record["video_dir_tmp"] = (
                str(self.video_dir_tmp) if self.video_dir_tmp is not None else None
            )
            ep_record["video_dir_archived"] = (
                str(self.archived_video_dir)
                if self.archived_video_dir is not None
                else None
            )
            npz_rel = None
            if bool(getattr(self.writer, "arrays_saved", False)) and bool(
                ep_record.get("arrays_saved", False)
            ):
                npz_rel = self.writer.write_episode_npz(
                    str(ep_record["episode_id"]),
                    state_arrays=cast(
                        Mapping[str, Any], arrays_blob.get("state_arrays", {})
                    ),
                    action_arrays=cast(
                        Mapping[str, Any], arrays_blob.get("action_arrays", {})
                    ),
                )
            ep_record["npz_path"] = npz_rel
            for tr in transitions:
                self.writer.append_transition(tr)
                n_transitions_total += 1
            self.writer.append_episode(ep_record)
        print("[INFO] episodes_written:", len(results_buffer))
        print("[INFO] transitions_written:", int(n_transitions_total))

    def _validate_video_artifacts(
        self,
        results_buffer: list[
            tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]
        ],
    ) -> None:
        assert self.args is not None
        unique_video_dirs = sorted(
            {
                (
                    str(rec.get("video_dir_archived"))
                    if rec.get("video_dir_archived") is not None
                    else "<none>"
                )
                for rec, _trs, _ab in results_buffer
            }
        )
        if len(unique_video_dirs) > 1:
            raise RuntimeError(
                "video_dir_archived must be unique per iter_tag; "
                f"got {len(unique_video_dirs)}: {unique_video_dirs}"
            )
        mp4_count = 0
        if self.archived_video_dir is not None:
            try:
                mp4_count = sum(
                    1 for _p in Path(self.archived_video_dir).rglob("*.mp4")
                )
            except Exception:
                mp4_count = 0
        episode_count = int(len(results_buffer))
        print(
            "[EVIDENCE] unique_video_dir_archived="
            f"{len(unique_video_dirs)} "
            f"{unique_video_dirs[0] if len(unique_video_dirs) == 1 else unique_video_dirs}"
        )
        print(
            f"[EVIDENCE] mp4_count={int(mp4_count)} episode_count={int(episode_count)}"
        )
        if (
            bool(getattr(self.args, "offscreen"))
            and self.archived_video_dir is not None
            and episode_count > 0
            and int(mp4_count) != int(episode_count)
        ):
            raise RuntimeError(
                "mp4_count must equal episode_count when offscreen is enabled; "
                + f"got mp4_count={int(mp4_count)} episode_count={int(episode_count)} "
                + f"video_dir_archived={self.archived_video_dir}"
            )

    def _timeout_manual_check(self) -> None:
        assert self.args is not None
        if hasattr(signal, "SIGALRM"):
            return
        elapsed = time.monotonic() - self.t0_total
        if float(getattr(self.args, "total_timeout_s")) > 0 and elapsed > float(
            getattr(self.args, "total_timeout_s")
        ):
            raise TimeoutError(f"Timed out after {int(elapsed)}s (manual check)")

    def _maybe_resolve_formal_branch(self) -> None:
        assert self.repo_root is not None
        _ = maybe_resolve_formal_collect_branch(
            self.repo_root, iter_tag=str(self.iter_tag)
        )

    def _finalize_env(self) -> None:
        assert self.args is not None
        _finalize_offscreen_video_and_close_env(
            self.env,
            offscreen=bool(getattr(self.args, "offscreen")),
            video_dir_tmp=self.video_dir_tmp,
        )
        self.env = None

    def _cleanup(self) -> None:
        assert self.args is not None
        assert self.artifacts_videos is not None
        _clear_alarm_timeout()
        if self.previous_signal_handlers is not None:
            _restore_signal_handlers(self.previous_signal_handlers)
        self._finalize_env()
        should_archive_in_cleanup = (
            bool(getattr(self.args, "offscreen"))
            and self.video_dir_tmp is not None
            and self.archived_video_dir is None
        )
        if should_archive_in_cleanup:
            try:
                archive_root = Path(self.artifacts_videos) / str(self.iter_tag)
                archive_root.mkdir(parents=True, exist_ok=True)
                self.archived_video_dir = _archive_video_dir(
                    video_dir=self.video_dir_tmp, archive_root=archive_root
                )
                if self.archived_video_dir is not None:
                    print(
                        "[INFO] cleanup: archived video_dir_tmp:",
                        str(self.archived_video_dir),
                    )
                    try:
                        if (
                            self.video_dir_tmp is not None
                            and self.video_dir_tmp.is_dir()
                        ):
                            shutil.rmtree(self.video_dir_tmp)
                    except Exception:
                        pass
            except Exception:
                pass
        if (
            self.client is not None
            and self.started_by_me
            and bool(getattr(self.args, "kill_server_on_exit", False))
        ):
            print("[INFO] cleanup: stopping server started by this script")
            try:
                _safe_kill_server(self.client, int(self.server_ping_timeout_ms))
            except Exception:
                pass
            if self.proc is not None:
                try:
                    _terminate_process(self.proc, timeout_s=10.0)
                except Exception:
                    pass


def main() -> int:
    return RecapCollectRolloutsWorkflow().run()


if __name__ == "__main__":
    raise SystemExit(main())


class RecapCollectRolloutsScriptApp:
    def run(self) -> int:
        return main()
