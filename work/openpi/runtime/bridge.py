"""Core runtime bridge for OpenPI policy serving and LIBERO episode execution.

The helpers in this module own path resolution, server/client subprocess
lifecycles, probe/client handshakes, and evidence emission. Higher-level
workflows should treat it as the implementation layer behind runtime entry
classes rather than import scattered subprocess logic themselves.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any, cast
import urllib.error
import urllib.request

from work.openpi.contracts import (
    OpenPIRuntimePaths,
    RuntimeEpisodeRequest,
    RuntimeServerSpec,
    StockEpisodeRequest,
)
from work.openpi.dataloader import read_json, write_json, write_markdown


REPO_ROOT = Path(__file__).resolve().parents[3]
TOPIC = "openpi_libero_native"
SCHEMA_VERSION = "openpi_libero_native_smoke_v1"
STOCK_CONFIG = "pi05_libero"
STOCK_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
STOCK_VARIANT = "stock_libero_ref_v1"
STOCK_TASK_SUITE = "libero_spatial"
STOCK_TASK_ID = 0
STOCK_NUM_TRIALS = 1
STOCK_SEED = 7
ACTION_HORIZON = 10
DISCRETE_STATE_INPUT = False
EXTRA_DELTA_TRANSFORM = False
REPLAN_STEPS = 5
NUM_STEPS_WAIT = 10
LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
LIBERO_NATIVE_SMOKE_ENTRY = REPO_ROOT / "work" / "openpi" / "runtime" / "internal.py"


class FailFastError(RuntimeError):
    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openpi_runtime_internal.py",
        description="Internal OpenPI runtime helper entry.",
    )
    _ = parser.add_argument("--task-suite-name", default=STOCK_TASK_SUITE)
    _ = parser.add_argument("--task-id", type=int, default=STOCK_TASK_ID)
    _ = parser.add_argument(
        "--num-trials-per-task", type=int, default=STOCK_NUM_TRIALS
    )
    _ = parser.add_argument("--seed", type=int, default=STOCK_SEED)
    _ = parser.add_argument("--checkpoint-dir", default=STOCK_CHECKPOINT)
    _ = parser.add_argument("--host", default=DEFAULT_HOST)
    _ = parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    _ = parser.add_argument("--server-ready-timeout-s", type=float, default=150.0)
    _ = parser.add_argument("--client-timeout-s", type=float, default=80.0)
    _ = parser.add_argument("--video-fps", type=int, default=10)
    _ = parser.add_argument(
        "--indicator-mode",
        choices=("positive", "negative", "omit", "cfg"),
        default="cfg",
    )
    _ = parser.add_argument(
        "--internal-mode",
        choices=("probe", "client"),
        help=argparse.SUPPRESS,
    )
    _ = parser.add_argument("--probe-out", default="", help=argparse.SUPPRESS)
    _ = parser.add_argument("--client-summary-out", default="", help=argparse.SUPPRESS)
    _ = parser.add_argument("--client-video-out", default="", help=argparse.SUPPRESS)
    _ = parser.add_argument("--trial-index", type=int, default=0, help=argparse.SUPPRESS)
    _ = parser.add_argument(
        "--resolved-runtime-indicator-mode",
        default="",
        help=argparse.SUPPRESS,
    )
    _ = parser.add_argument(
        "--resolved-runtime-indicator-source",
        default="",
        help=argparse.SUPPRESS,
    )
    _ = parser.add_argument(
        "--resolved-runtime-consumer-mode",
        default="",
        help=argparse.SUPPRESS,
    )
    _ = parser.add_argument(
        "--resolved-runtime-fixed-indicator-mode",
        default="",
        help=argparse.SUPPRESS,
    )
    _ = parser.add_argument(
        "--resolved-runtime-critic-checkpoint-ref",
        default="",
        help=argparse.SUPPRESS,
    )
    return parser


def _server_spec(raw: RuntimeServerSpec | argparse.Namespace) -> RuntimeServerSpec:
    if isinstance(raw, RuntimeServerSpec):
        return raw
    return RuntimeServerSpec(
        host=str(raw.host),
        port=int(raw.port),
        checkpoint_dir=str(raw.checkpoint_dir),
        server_ready_timeout_s=float(raw.server_ready_timeout_s),
        client_timeout_s=float(raw.client_timeout_s),
    )


def _required_paths(
    *,
    topic: str = TOPIC,
    evidence_path: Path | None = None,
    artifact_root: Path | None = None,
    runtime_root: Path | None = None,
) -> OpenPIRuntimePaths:
    openpi_root = REPO_ROOT / "submodules" / "openpi"
    resolved_evidence = evidence_path
    if resolved_evidence is None and topic == TOPIC:
        resolved_evidence = (
            REPO_ROOT / ".sisyphus" / "evidence" / "task-4-libero-native-smoke.md"
        )
    paths = OpenPIRuntimePaths(
        openpi_root=openpi_root,
        openpi_venv_python=openpi_root / ".venv" / "bin" / "python",
        serve_policy=openpi_root / "scripts" / "serve_policy.py",
        libero_main=openpi_root / "examples" / "libero" / "main.py",
        libero_submodule=openpi_root / "third_party" / "libero",
        config=openpi_root / "src" / "openpi" / "training" / "config.py",
        runtime_dir=(
            runtime_root.resolve()
            if runtime_root is not None
            else REPO_ROOT / "agent" / "runtime_logs" / topic
        ),
        artifact_dir=(
            artifact_root.resolve()
            if artifact_root is not None
            else REPO_ROOT / "agent" / "artifacts" / topic
        ),
        evidence_path=resolved_evidence,
    )
    for path in (
        paths.openpi_venv_python,
        paths.serve_policy,
        paths.libero_main,
        paths.libero_submodule,
        paths.config,
    ):
        if not path.exists():
            raise FailFastError(f"缺少 frozen protocol 需要的路径：{path}")
    return paths


def _preview_text(path: Path, *, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _log(message: str, *, log_path: Path | None = None) -> None:
    print(message, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message)
            handle.write("\n")


def _validate_stock_args(args: argparse.Namespace) -> None:
    if args.task_suite_name != STOCK_TASK_SUITE:
        raise FailFastError(
            f"task_suite_name 漂移：{args.task_suite_name!r}；stock smoke 只允许 {STOCK_TASK_SUITE!r}"
        )
    if args.task_id != STOCK_TASK_ID:
        raise FailFastError(
            f"task_id 漂移：{args.task_id!r}；stock smoke 只允许 {STOCK_TASK_ID}"
        )
    if args.num_trials_per_task != STOCK_NUM_TRIALS:
        raise FailFastError(
            "num_trials_per_task 漂移："
            f"{args.num_trials_per_task!r}；stock smoke 只允许 {STOCK_NUM_TRIALS}"
        )
    if args.seed != STOCK_SEED:
        raise FailFastError(
            f"seed 漂移：{args.seed!r}；stock smoke 只允许 {STOCK_SEED}"
        )
    if args.checkpoint_dir != STOCK_CHECKPOINT:
        raise FailFastError(
            "checkpoint 漂移："
            f"{args.checkpoint_dir!r}；stock smoke 只允许 {STOCK_CHECKPOINT!r}"
        )
    if args.port <= 0 or args.port > 65535:
        raise FailFastError(f"非法端口：{args.port!r}")
    if args.server_ready_timeout_s <= 0:
        raise FailFastError("server_ready_timeout_s 必须大于 0")
    if args.client_timeout_s <= 0:
        raise FailFastError("client_timeout_s 必须大于 0")


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def _pick_free_port(host: str, start_port: int) -> int:
    port = int(start_port)
    for _ in range(100):
        if not _port_in_use(host, port):
            return port
        port += 1
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _healthz_ok(host: str, port: int) -> bool:
    url = f"http://{host}:{port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def _run_probe(args: argparse.Namespace) -> int:
    websocket_client_policy = importlib.import_module(
        "openpi_client.websocket_client_policy"
    )

    policy = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    payload = {
        "host": args.host,
        "port": args.port,
        "server_metadata": policy.get_server_metadata(),
        "probed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if args.probe_out:
        write_json(Path(args.probe_out), payload)
    return 0


def _get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise FailFastError(f"未知 task suite：{task_suite_name!r}")


def _client_get_env(task: Any, resolution: int, seed: int) -> tuple[Any, str]:
    libero_module = importlib.import_module("libero.libero")
    libero_envs = importlib.import_module("libero.libero.envs")

    task_description = str(task.language)
    task_bddl_file = (
        Path(libero_module.get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = libero_envs.OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _client_quat2axisangle(quat: Any) -> Any:
    import math
    import numpy as np

    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _build_explicit_infer_element(
    *,
    image: Any,
    wrist_image: Any,
    state: Any,
    prompt: str,
) -> dict[str, Any]:
    element = {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": str(prompt),
    }
    expected_keys = {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    if set(element) != expected_keys:
        raise FailFastError(
            "runtime bridge infer element drifted from explicit-input only contract"
        )
    return element


def _resolve_client_runtime_indicator_config(args: argparse.Namespace) -> Any:
    from work.openpi.recap.runtime_prompt import (
        RuntimeIndicatorConfig,
        normalize_runtime_indicator_mode,
        resolve_runtime_indicator_config,
    )

    explicit_fields = {
        "resolved_runtime_indicator_mode": str(
            getattr(args, "resolved_runtime_indicator_mode", "") or ""
        ).strip(),
        "resolved_runtime_indicator_source": str(
            getattr(args, "resolved_runtime_indicator_source", "") or ""
        ).strip(),
        "resolved_runtime_consumer_mode": str(
            getattr(args, "resolved_runtime_consumer_mode", "") or ""
        ).strip(),
        "resolved_runtime_fixed_indicator_mode": str(
            getattr(args, "resolved_runtime_fixed_indicator_mode", "") or ""
        ).strip(),
        "resolved_runtime_critic_checkpoint_ref": str(
            getattr(args, "resolved_runtime_critic_checkpoint_ref", "") or ""
        ).strip(),
    }
    if not any(explicit_fields.values()):
        return resolve_runtime_indicator_config(
            requested_indicator_mode=args.indicator_mode,
            variant=STOCK_VARIANT,
        )

    missing_required = [
        field_name
        for field_name in (
            "resolved_runtime_indicator_mode",
            "resolved_runtime_indicator_source",
            "resolved_runtime_consumer_mode",
            "resolved_runtime_critic_checkpoint_ref",
        )
        if not explicit_fields[field_name]
    ]
    if missing_required:
        raise FailFastError(
            "explicit runtime indicator config is incomplete; missing "
            + ", ".join(missing_required)
        )

    resolved_indicator_mode = normalize_runtime_indicator_mode(
        explicit_fields["resolved_runtime_indicator_mode"],
        field_name="resolved_runtime_indicator_mode",
    )
    if resolved_indicator_mode == "cfg":
        raise FailFastError(
            "resolved runtime indicator mode must be concrete, got 'cfg'"
        )
    fixed_indicator_mode = None
    if explicit_fields["resolved_runtime_fixed_indicator_mode"]:
        fixed_indicator_mode = normalize_runtime_indicator_mode(
            explicit_fields["resolved_runtime_fixed_indicator_mode"],
            field_name="resolved_runtime_fixed_indicator_mode",
        )
        if fixed_indicator_mode == "cfg":
            raise FailFastError(
                "resolved runtime fixed indicator mode must be concrete, got 'cfg'"
            )

    return RuntimeIndicatorConfig(
        requested_indicator_mode=normalize_runtime_indicator_mode(
            args.indicator_mode,
            field_name="indicator_mode",
        ),
        indicator_mode=resolved_indicator_mode,
        indicator_source=explicit_fields["resolved_runtime_indicator_source"],
        consumer_mode=explicit_fields["resolved_runtime_consumer_mode"],
        fixed_indicator_mode=fixed_indicator_mode,
        critic_checkpoint_ref=explicit_fields["resolved_runtime_critic_checkpoint_ref"],
    )


def _run_client(args: argparse.Namespace) -> int:
    import collections

    imageio = importlib.import_module("imageio")
    benchmark = importlib.import_module("libero.libero.benchmark")
    numpy = importlib.import_module("numpy")
    image_tools = importlib.import_module("openpi_client.image_tools")
    websocket_client_policy = importlib.import_module(
        "openpi_client.websocket_client_policy"
    )
    from work.openpi.recap.runtime_prompt import build_runtime_prompt_bundle

    if not args.client_summary_out:
        raise FailFastError("internal client 缺少 --client-summary-out")
    if not args.client_video_out:
        raise FailFastError("internal client 缺少 --client-video-out")
    if args.num_trials_per_task <= 0:
        raise FailFastError("num_trials_per_task 必须大于 0")
    if args.trial_index < 0:
        raise FailFastError("trial_index 必须大于等于 0")

    numpy.random.seed(args.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    if args.task_id < 0 or args.task_id >= int(task_suite.n_tasks):
        raise FailFastError(f"task_id 越界：{args.task_id} / {task_suite.n_tasks}")

    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    if args.num_trials_per_task > len(initial_states):
        raise FailFastError(
            "num_trials_per_task 超出初始状态数量："
            f"{args.num_trials_per_task} > {len(initial_states)}"
        )
    if args.trial_index + args.num_trials_per_task > len(initial_states):
        raise FailFastError(
            "trial_index + num_trials_per_task 超出初始状态数量："
            f"{args.trial_index} + {args.num_trials_per_task} > {len(initial_states)}"
        )

    env, task_description = _client_get_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    indicator_config = _resolve_client_runtime_indicator_config(args)
    prompt_bundle = build_runtime_prompt_bundle(
        task_description,
        config=indicator_config,
    )
    max_steps = _get_max_steps(args.task_suite_name)
    video_path = Path(args.client_video_out)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_successes = 0
    episode_results: list[dict[str, Any]] = []
    replay_images: list[Any] = []
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    fatal_error = ""

    try:
        for episode_idx in range(args.num_trials_per_task):
            actual_trial_index = args.trial_index + episode_idx
            env.reset()
            action_plan: collections.deque[Any] = collections.deque()
            obs = env.set_init_state(initial_states[actual_trial_index])
            t = 0
            done = False
            last_error = ""
            episode_replay_images: list[Any] = []
            inference_calls = 0

            while t < max_steps + NUM_STEPS_WAIT:
                try:
                    if t < NUM_STEPS_WAIT:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        if done:
                            total_successes += 1
                            break
                        continue

                    img = numpy.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = numpy.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, 224, 224)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, 224, 224)
                    )
                    episode_replay_images.append(img)

                    if not action_plan:
                        element = _build_explicit_infer_element(
                            image=img,
                            wrist_image=wrist_img,
                            state=numpy.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _client_quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            prompt=prompt_bundle.prompt_text,
                        )
                        action_chunk = client.infer(element)["actions"]
                        if len(action_chunk) < REPLAN_STEPS:
                            raise FailFastError(
                                "policy action chunk 太短："
                                f"{len(action_chunk)} < {REPLAN_STEPS}"
                            )
                        action_plan.extend(action_chunk[:REPLAN_STEPS])
                        inference_calls += 1

                    action = action_plan.popleft()
                    obs, _, done, _ = env.step(action.tolist())
                    t += 1
                    if done:
                        total_successes += 1
                        break

                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    break

            total_episodes += 1
            replay_images = episode_replay_images or [
                numpy.zeros((224, 224, 3), dtype=numpy.uint8)
            ]
            episode_results.append(
                {
                    "episode_index": episode_idx,
                    "trial_index": actual_trial_index,
                    "success": bool(done),
                    "steps_observed": int(t),
                    "inference_calls": int(inference_calls),
                    "error": last_error,
                }
            )
            if last_error:
                fatal_error = last_error
                break

        imageio.mimwrite(
            video_path,
            [numpy.asarray(frame) for frame in replay_images],
            fps=args.video_fps,
        )
        if fatal_error:
            raise FailFastError(f"client runtime error: {fatal_error}")
    finally:
        with contextlib.suppress(Exception):
            env.close()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": "client_result",
        "started_at": started_at,
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "task_description": task_description,
        "prompt_text": prompt_bundle.prompt_text,
        "num_trials_per_task": args.num_trials_per_task,
        "trial_index_start": args.trial_index,
        "seed": args.seed,
        "max_steps": max_steps,
        "replan_steps": REPLAN_STEPS,
        "num_steps_wait": NUM_STEPS_WAIT,
        "action_horizon": ACTION_HORIZON,
        "discrete_state_input": DISCRETE_STATE_INPUT,
        "extra_delta_transform": EXTRA_DELTA_TRANSFORM,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": (float(total_successes) / float(total_episodes))
        if total_episodes
        else 0.0,
        "episode_results": episode_results,
        "video_path": str(video_path),
        "server_metadata": client.get_server_metadata(),
        "runtime_prompting": {
            "indicator_mode_requested": indicator_config.requested_indicator_mode,
            "indicator_mode": prompt_bundle.indicator_mode,
            "indicator_source": prompt_bundle.indicator_source,
            "prompt_text_surface": prompt_bundle.prompt_text_surface,
            "prompt_route": prompt_bundle.prompt_provenance["prompt_route"],
            "conditioning_mode": prompt_bundle.prompt_provenance["conditioning_mode"],
            "consumer_mode": prompt_bundle.consumer_mode,
            "fixed_indicator_mode": prompt_bundle.fixed_indicator_mode or "",
            "critic_checkpoint_ref": prompt_bundle.critic_checkpoint_ref,
            "prompt_text": prompt_bundle.prompt_text,
            "source_prompt_field": prompt_bundle.prompt_provenance[
                "source_prompt_field"
            ],
        },
    }
    write_json(Path(args.client_summary_out), summary)
    print("LIBERO_NATIVE_CLIENT_DONE", flush=True)
    return 0


def _prepare_libero_config_dir(openpi_root: Path, runtime_dir: Path) -> Path:
    config_dir = runtime_dir / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = openpi_root / "third_party" / "libero" / "libero" / "libero"
    config_text = "\n".join(
        [
            f"benchmark_root: {benchmark_root}",
            f"bddl_files: {benchmark_root / 'bddl_files'}",
            f"init_states: {benchmark_root / 'init_files'}",
            f"datasets: {benchmark_root.parent / 'datasets'}",
            f"assets: {benchmark_root / 'assets'}",
            "",
        ]
    )
    _ = (config_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    return config_dir


def _build_openpi_subprocess_env(
    openpi_root: Path,
    libero_config_dir: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    _ = env.setdefault("PYTHONUNBUFFERED", "1")
    pythonpath_entries = [
        str(openpi_root / "src"),
        str(openpi_root / "packages" / "openpi-client" / "src"),
        str(openpi_root / "third_party" / "libero"),
    ]
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{os.pathsep.join(pythonpath_entries)}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else os.pathsep.join(pythonpath_entries)
    )
    env["LIBERO_CONFIG_PATH"] = str(libero_config_dir)
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return env


def _spawn_server(
    raw: RuntimeServerSpec | argparse.Namespace,
    *,
    venv_python: Path,
    serve_policy: Path,
    openpi_root: Path,
    server_log: Path,
    libero_config_dir: Path,
) -> tuple[subprocess.Popen[str], Any]:
    spec = _server_spec(raw)
    command = [
        str(venv_python),
        str(serve_policy),
        f"--port={spec.port}",
        "policy:checkpoint",
        f"--policy.config={STOCK_CONFIG}",
        f"--policy.dir={spec.checkpoint_dir}",
    ]
    server_log.parent.mkdir(parents=True, exist_ok=True)
    handle = server_log.open("w", encoding="utf-8")
    env = _build_openpi_subprocess_env(openpi_root, libero_config_dir)
    proc = subprocess.Popen(  # noqa: S603
        command,
        cwd=openpi_root,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc, handle


def _wait_for_server_ready(
    raw: RuntimeServerSpec | argparse.Namespace,
    *,
    proc: subprocess.Popen[str],
    runtime_dir: Path,
    venv_python: Path,
    openpi_root: Path,
    libero_config_dir: Path,
    harness_log: Path,
    server_log: Path,
    cli_entry: Path = LIBERO_NATIVE_SMOKE_ENTRY,
) -> dict[str, Any]:
    spec = _server_spec(raw)
    started = time.monotonic()
    deadline = started + float(spec.server_ready_timeout_s)
    probe_path = runtime_dir / "server_probe.json"
    last_probe_attempt = 0.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            preview = _preview_text(server_log)
            raise FailFastError(
                f"server 提前退出，returncode={proc.returncode}，server_log_tail={preview!r}"
            )
        now = time.monotonic()
        port_listening = _port_in_use(spec.host, spec.port)
        should_probe = _healthz_ok(spec.host, spec.port) or (
            port_listening and now - last_probe_attempt >= 5.0
        )
        if should_probe:
            last_probe_attempt = now
            probe_command = [
                str(venv_python),
                str(cli_entry),
                "--internal-mode",
                "probe",
                "--host",
                spec.host,
                "--port",
                str(spec.port),
                "--probe-out",
                str(probe_path),
            ]
            probe = subprocess.run(  # noqa: S603
                probe_command,
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
                env=_build_openpi_subprocess_env(openpi_root, libero_config_dir),
            )
            if probe.returncode == 0 and probe_path.exists():
                payload = read_json(probe_path)
                _log("LIBERO_NATIVE_SERVER_READY", log_path=harness_log)
                return cast(dict[str, Any], payload)
            if probe.returncode != 0:
                _log(
                    "[INFO] LIBERO server port is listening but websocket probe is not ready yet; "
                    f"probe_stdout_tail={probe.stdout[-500:]!r} "
                    f"probe_stderr_tail={probe.stderr[-500:]!r}",
                    log_path=harness_log,
                )

        elapsed = int(time.monotonic() - started)
        _log(
            f"[INFO] waiting for LIBERO server ready... {elapsed}s",
            log_path=harness_log,
        )
        time.sleep(2.0)

    preview = _preview_text(server_log)
    raise FailFastError(
        "server ready timeout；"
        f"waited>{float(spec.server_ready_timeout_s):.1f}s；server_log_tail={preview!r}"
    )


def _run_client_subprocess(
    args: argparse.Namespace,
    *,
    runtime_dir: Path,
    artifact_dir: Path,
    venv_python: Path,
    openpi_root: Path,
    libero_config_dir: Path,
    harness_log: Path,
    cli_entry: Path = LIBERO_NATIVE_SMOKE_ENTRY,
) -> dict[str, Any]:
    client_summary = runtime_dir / "client_summary.json"
    client_video = artifact_dir / "videos" / "libero_spatial_task0_trial0.mp4"
    client_log = runtime_dir / "client.log"
    client_command = [
        str(venv_python),
        str(cli_entry),
        "--internal-mode",
        "client",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--task-suite-name",
        args.task_suite_name,
        "--task-id",
        str(args.task_id),
        "--num-trials-per-task",
        str(args.num_trials_per_task),
        "--seed",
        str(args.seed),
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--indicator-mode",
        args.indicator_mode,
        "--video-fps",
        str(args.video_fps),
        "--client-summary-out",
        str(client_summary),
        "--client-video-out",
        str(client_video),
    ]
    result = subprocess.run(  # noqa: S603
        client_command,
        cwd=openpi_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(args.client_timeout_s),
        env=_build_openpi_subprocess_env(openpi_root, libero_config_dir),
    )
    client_log.parent.mkdir(parents=True, exist_ok=True)
    _ = client_log.write_text(
        "\n".join(
            [
                "# stdout",
                result.stdout,
                "",
                "# stderr",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise FailFastError(
            "client smoke 失败；"
            f"stdout_tail={result.stdout[-2000:]!r} stderr_tail={result.stderr[-2000:]!r}"
        )
    if "LIBERO_NATIVE_CLIENT_DONE" not in result.stdout:
        raise FailFastError("client 未输出 LIBERO_NATIVE_CLIENT_DONE")
    if not client_summary.exists():
        raise FailFastError(f"client summary 缺失：{client_summary}")
    payload = cast(dict[str, Any], read_json(client_summary))
    payload["client_log"] = str(client_log)
    _log("LIBERO_NATIVE_CLIENT_DONE", log_path=harness_log)
    return payload


def _terminate_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    with contextlib.suppress(Exception):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)


def _close_handle(handle: Any | None) -> None:
    if handle is None:
        return
    with contextlib.suppress(Exception):
        handle.close()


def _build_summary(
    args: argparse.Namespace,
    *,
    paths: OpenPIRuntimePaths,
    started_at: str,
    finished_at: str,
    probe_payload: dict[str, Any],
    client_payload: dict[str, Any],
    server_log: Path,
    harness_log: Path,
) -> dict[str, Any]:
    from work.openpi.serve.provenance import build_libero_server_provenance_payload

    runtime_prompting = cast(dict[str, Any], client_payload["runtime_prompting"])
    prompt_provenance = {
        "prompt_route": str(runtime_prompting["prompt_route"]),
        "conditioning_mode": str(runtime_prompting["conditioning_mode"]),
        "indicator_mode": str(runtime_prompting["indicator_mode"]),
        "indicator_source": str(runtime_prompting["indicator_source"]),
        "prompt_text_surface": str(runtime_prompting["prompt_text_surface"]),
    }
    norm_provenance = {
        "norm_stats_source": "checkpoint_asset_norm_stats",
        "norm_stats_path": f"{args.checkpoint_dir.rstrip('/')}/assets/physical-intelligence/libero/norm_stats.json",
        "asset_id": "physical-intelligence/libero",
    }
    provenance = build_libero_server_provenance_payload(
        prompt_provenance=prompt_provenance,
        norm_provenance=norm_provenance,
        critic_checkpoint_ref=str(runtime_prompting["critic_checkpoint_ref"]),
        task_ids=(int(args.task_id),),
        seed_manifest=(int(args.seed),),
        num_trials_per_task=int(args.num_trials_per_task),
        action_horizon=ACTION_HORIZON,
        discrete_state_input=DISCRETE_STATE_INPUT,
        extra_delta_transform=EXTRA_DELTA_TRANSFORM,
        replan_steps=REPLAN_STEPS,
        num_steps_wait=NUM_STEPS_WAIT,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "topic": TOPIC,
        "started_at": started_at,
        "finished_at": finished_at,
        "repo_root": str(REPO_ROOT),
        "runtime_dir": str(paths.runtime_dir),
        "artifact_dir": str(paths.artifact_dir),
        "summary_json": str(paths.artifact_dir / "summary.json"),
        "evidence_markdown": str(paths.evidence_path) if paths.evidence_path else "",
        "server_log": str(server_log),
        "harness_log": str(harness_log),
        "server": {
            "config": STOCK_CONFIG,
            "checkpoint": STOCK_CHECKPOINT,
            "checkpoint_source": "upstream_openpi_default_or_explicit_cli",
            "host": args.host,
            "port": args.port,
            "ready_probe": probe_payload,
            "command": [
                str(paths.openpi_venv_python),
                str(paths.serve_policy),
                f"--port={args.port}",
                "policy:checkpoint",
                f"--policy.config={STOCK_CONFIG}",
                f"--policy.dir={STOCK_CHECKPOINT}",
            ],
        },
        "provenance": {
            **provenance,
            "config": STOCK_CONFIG,
        },
        "client": client_payload,
    }


def _build_evidence_markdown(summary: dict[str, Any]) -> str:
    client = summary["client"]
    provenance = summary["provenance"]
    ready_probe = summary["server"]["ready_probe"]
    video_path = client["video_path"]
    return "\n".join(
        [
            "# Task 4 证据 — openpi LIBERO native smoke",
            "",
            "- 结论：stock `pi05_libero` native smoke 已按 frozen protocol 跑通。",
            f"- config：`{provenance['config']}`",
            f"- checkpoint：`{provenance['checkpoint']}`",
            f"- task_suite_name：`{client['task_suite_name']}`",
            f"- task_id：`{client['task_id']}`",
            f"- num_trials_per_task：`{client['num_trials_per_task']}`",
            f"- seed：`{client['seed']}`",
            f"- replan_steps：`{provenance['replan_steps']}`",
            f"- num_steps_wait：`{provenance['num_steps_wait']}`",
            f"- action_horizon：`{provenance['action_horizon']}`",
            f"- discrete_state_input：`{provenance['discrete_state_input']}`",
            f"- extra_delta_transform：`{provenance['extra_delta_transform']}`",
            f"- indicator_mode：`{provenance['indicator_mode']}`",
            f"- indicator_source：`{provenance['indicator_source']}`",
            f"- prompt_text_surface：`{provenance['prompt_text_surface']}`",
            f"- critic_checkpoint_ref：`{provenance['critic_checkpoint_ref']}`",
            f"- ready_probe：`{ready_probe['probed_at']}`",
            f"- task_description：`{client['task_description']}`",
            f"- prompt_text：`{client['prompt_text']}`",
            f"- total_episodes：`{client['total_episodes']}`",
            f"- total_successes：`{client['total_successes']}`",
            f"- success_rate：`{client['success_rate']}`",
            f"- summary_json：`{summary['summary_json']}`",
            f"- video_path：`{video_path}`",
            f"- server_log：`{summary['server_log']}`",
            f"- harness_log：`{summary['harness_log']}`",
            f"- client_log：`{client['client_log']}`",
            "- marker：`LIBERO_NATIVE_SERVER_READY`、`LIBERO_NATIVE_CLIENT_DONE`、`LIBERO_NATIVE_SUMMARY_WRITTEN`",
        ]
    )


def _run_harness(
    args: argparse.Namespace,
    *,
    paths: OpenPIRuntimePaths | None = None,
) -> int:
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    resolved_paths = _required_paths() if paths is None else paths
    resolved_paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    resolved_paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    harness_log = resolved_paths.runtime_dir / "harness.log"
    server_log = resolved_paths.runtime_dir / "server.log"
    _ = harness_log.write_text("", encoding="utf-8")
    libero_config_dir = _prepare_libero_config_dir(
        resolved_paths.openpi_root,
        resolved_paths.runtime_dir,
    )

    _validate_stock_args(args)
    if _port_in_use(args.host, args.port):
        raise FailFastError(f"端口已被占用：{args.host}:{args.port}")

    proc: subprocess.Popen[str] | None = None
    server_handle: Any | None = None
    try:
        proc, server_handle = _spawn_server(
            args,
            venv_python=resolved_paths.openpi_venv_python,
            serve_policy=resolved_paths.serve_policy,
            openpi_root=resolved_paths.openpi_root,
            server_log=server_log,
            libero_config_dir=libero_config_dir,
        )
        probe_payload = _wait_for_server_ready(
            args,
            proc=proc,
            runtime_dir=resolved_paths.runtime_dir,
            venv_python=resolved_paths.openpi_venv_python,
            openpi_root=resolved_paths.openpi_root,
            libero_config_dir=libero_config_dir,
            harness_log=harness_log,
            server_log=server_log,
        )
        client_payload = _run_client_subprocess(
            args,
            runtime_dir=resolved_paths.runtime_dir,
            artifact_dir=resolved_paths.artifact_dir,
            venv_python=resolved_paths.openpi_venv_python,
            openpi_root=resolved_paths.openpi_root,
            libero_config_dir=libero_config_dir,
            harness_log=harness_log,
        )
        finished_at = dt.datetime.now().isoformat(timespec="seconds")
        summary = _build_summary(
            args,
            paths=resolved_paths,
            started_at=started_at,
            finished_at=finished_at,
            probe_payload=probe_payload,
            client_payload=client_payload,
            server_log=server_log,
            harness_log=harness_log,
        )
        summary_path = resolved_paths.artifact_dir / "summary.json"
        write_json(summary_path, summary)
        write_json(resolved_paths.runtime_dir / "summary.json", summary)
        if resolved_paths.evidence_path is not None:
            write_markdown(
                resolved_paths.evidence_path, _build_evidence_markdown(summary)
            )
        _log("LIBERO_NATIVE_SUMMARY_WRITTEN", log_path=harness_log)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        _terminate_process(proc)
        _close_handle(server_handle)


def _run_stock_episode(
    *,
    task_suite_name: str,
    task_id: int,
    seed: int,
    trial_index: int,
    video_path: Path,
    host: str,
    port: int,
) -> dict[str, object]:
    request = StockEpisodeRequest(
        task_suite_name=task_suite_name,
        task_id=task_id,
        seed=seed,
        trial_index=trial_index,
        video_path=video_path,
        host=host,
        port=port,
    )
    imageio = importlib.import_module("imageio")
    benchmark = importlib.import_module("libero.libero.benchmark")
    numpy = importlib.import_module("numpy")
    image_tools = importlib.import_module("openpi_client.image_tools")
    websocket_client_policy = importlib.import_module(
        "openpi_client.websocket_client_policy"
    )

    numpy.random.seed(request.seed)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[request.task_suite_name]()
    if request.task_id < 0 or request.task_id >= int(task_suite.n_tasks):
        raise FailFastError(f"task_id 越界：{request.task_id} / {task_suite.n_tasks}")
    task = task_suite.get_task(request.task_id)
    initial_states = task_suite.get_task_init_states(request.task_id)
    if request.trial_index < 0 or request.trial_index >= len(initial_states):
        raise FailFastError(
            f"trial_index 越界：{request.trial_index} / {len(initial_states)} for task_id={request.task_id}"
        )

    client = websocket_client_policy.WebsocketClientPolicy(
        request.host,
        request.port,
    )
    env, task_description = _client_get_env(
        task,
        LIBERO_ENV_RESOLUTION,
        request.seed,
    )
    max_steps = _get_max_steps(request.task_suite_name)
    request.video_path.parent.mkdir(parents=True, exist_ok=True)
    replay_images: list[object] = []
    action_plan: Any = importlib.import_module("collections").deque()
    obs = None
    done = False
    steps_observed = 0
    inference_calls = 0
    error = ""
    try:
        env.reset()
        obs = env.set_init_state(initial_states[request.trial_index])
        while steps_observed < max_steps + NUM_STEPS_WAIT:
            try:
                if steps_observed < NUM_STEPS_WAIT:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    steps_observed += 1
                    if done:
                        break
                    continue

                if obs is None:
                    raise FailFastError("env observation unexpectedly missing")
                image = numpy.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_image = numpy.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )
                image = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(image, 224, 224)
                )
                wrist_image = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_image, 224, 224)
                )
                replay_images.append(image)

                if not action_plan:
                    element = _build_explicit_infer_element(
                        image=image,
                        wrist_image=wrist_image,
                        state=numpy.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _client_quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        prompt=task_description,
                    )
                    action_chunk = client.infer(element)["actions"]
                    if len(action_chunk) < REPLAN_STEPS:
                        raise FailFastError(
                            "policy action chunk 太短："
                            + f"{len(action_chunk)} < {REPLAN_STEPS}"
                        )
                    action_plan.extend(action_chunk[:REPLAN_STEPS])
                    inference_calls += 1

                action = action_plan.popleft()
                obs, _, done, _ = env.step(action.tolist())
                steps_observed += 1
                if done:
                    break
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                break
    finally:
        with contextlib.suppress(Exception):
            env.close()

    if not replay_images:
        replay_images = [numpy.zeros((224, 224, 3), dtype=numpy.uint8)]
    imageio.mimwrite(
        request.video_path,
        [numpy.asarray(frame) for frame in replay_images],
        fps=10,
    )
    return {
        "task_suite_name": request.task_suite_name,
        "task_id": request.task_id,
        "seed": request.seed,
        "trial_index": request.trial_index,
        "success": bool(done) and not error,
        "steps_observed": int(steps_observed),
        "video_path": str(request.video_path),
        "episode_status": "ok" if not error else "runtime_error",
        "error": error,
        "inference_calls": int(inference_calls),
    }


def _run_stock_episode_subprocess(
    *,
    task_suite_name: str,
    task_id: int,
    seed: int,
    trial_index: int,
    video_path: Path,
    host: str,
    port: int,
    venv_python: Path,
    openpi_root: Path,
    libero_config_dir: Path,
    runtime_dir: Path,
    timeout_s: float,
    episode_entry: Path,
) -> dict[str, object]:
    episode_runtime_dir = (
        runtime_dir / "episodes" / f"task{task_id}_seed{seed}_trial{trial_index}"
    )
    episode_runtime_dir.mkdir(parents=True, exist_ok=True)
    client_log = episode_runtime_dir / "client.log"
    episode_row_out = episode_runtime_dir / "episode_row.json"
    command = [
        str(venv_python),
        str(episode_entry),
        "--internal-mode",
        "stock-episode",
        "--task-suite-name",
        task_suite_name,
        "--task-id",
        str(task_id),
        "--seed",
        str(seed),
        "--trial-index",
        str(trial_index),
        "--host",
        host,
        "--port",
        str(port),
        "--openpi-root",
        str(openpi_root),
        "--libero-config-dir",
        str(libero_config_dir),
        "--video-path",
        str(video_path),
        "--episode-row-out",
        str(episode_row_out),
    ]
    result = subprocess.run(  # noqa: S603
        command,
        cwd=openpi_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=_build_openpi_subprocess_env(openpi_root, libero_config_dir),
    )
    _ = client_log.write_text(
        "\n".join(["# stdout", result.stdout, "", "# stderr", result.stderr]),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise FailFastError(
            "stock episode client failed; "
            + f"stdout_tail={result.stdout[-2000:]!r} stderr_tail={result.stderr[-2000:]!r}"
        )
    if "LIBERO_ROLLOUT_EVAL_V2_STOCK_EPISODE_DONE" not in result.stdout:
        raise FailFastError("stock episode client missing completion marker")
    row = read_json(episode_row_out)
    row["client_log"] = str(client_log)
    return row


def _run_runtime_episode_subprocess(
    *,
    task_suite_name: str,
    task_id: int,
    seed: int,
    trial_idx: int,
    video_path: Path,
    host: str,
    port: int,
    venv_python: Path,
    openpi_root: Path,
    libero_config_dir: Path,
    runtime_dir: Path,
    timeout_s: float,
    checkpoint_ref: str,
    indicator_mode_requested: str,
    runtime_indicator_config: Any,
    cli_entry: Path = LIBERO_NATIVE_SMOKE_ENTRY,
) -> dict[str, object]:
    request = RuntimeEpisodeRequest(
        task_suite_name=task_suite_name,
        task_id=task_id,
        seed=seed,
        trial_idx=trial_idx,
        video_path=video_path,
        host=host,
        port=port,
        checkpoint_ref=checkpoint_ref,
        indicator_mode_requested=indicator_mode_requested,
        runtime_indicator_config=runtime_indicator_config,
    )
    episode_runtime_dir = (
        runtime_dir / "episodes" / f"task{task_id}_seed{seed}_trial{trial_idx}"
    )
    episode_runtime_dir.mkdir(parents=True, exist_ok=True)
    client_log = episode_runtime_dir / "client.log"
    client_summary = episode_runtime_dir / "client_summary.json"
    command = [
        str(venv_python),
        str(cli_entry),
        "--internal-mode",
        "client",
        "--host",
        request.host,
        "--port",
        str(request.port),
        "--task-suite-name",
        request.task_suite_name,
        "--task-id",
        str(request.task_id),
        "--num-trials-per-task",
        "1",
        "--trial-index",
        str(request.trial_idx),
        "--seed",
        str(request.seed),
        "--checkpoint-dir",
        request.checkpoint_ref,
        "--indicator-mode",
        request.indicator_mode_requested,
        "--resolved-runtime-indicator-mode",
        str(request.runtime_indicator_config.indicator_mode),
        "--resolved-runtime-indicator-source",
        str(request.runtime_indicator_config.indicator_source),
        "--resolved-runtime-consumer-mode",
        str(request.runtime_indicator_config.consumer_mode),
        "--resolved-runtime-critic-checkpoint-ref",
        str(request.runtime_indicator_config.critic_checkpoint_ref),
        "--client-summary-out",
        str(client_summary),
        "--client-video-out",
        str(request.video_path),
    ]
    if request.runtime_indicator_config.fixed_indicator_mode:
        command.extend(
            [
                "--resolved-runtime-fixed-indicator-mode",
                str(request.runtime_indicator_config.fixed_indicator_mode),
            ]
        )
    result = subprocess.run(  # noqa: S603
        command,
        cwd=openpi_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=_build_openpi_subprocess_env(openpi_root, libero_config_dir),
    )
    _ = client_log.write_text(
        "\n".join(["# stdout", result.stdout, "", "# stderr", result.stderr]),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise FailFastError(
            "runtime-conditioned stock episode client failed; "
            + f"stdout_tail={result.stdout[-2000:]!r} stderr_tail={result.stderr[-2000:]!r}"
        )
    if "LIBERO_NATIVE_CLIENT_DONE" not in result.stdout:
        raise FailFastError(
            "runtime-conditioned stock episode client missing completion marker"
        )
    payload = read_json(client_summary)
    episode_results = payload.get("episode_results")
    if not isinstance(episode_results, list) or len(episode_results) != 1:
        raise FailFastError("client_summary must contain exactly one episode result")
    episode_row = episode_results[0]
    if not isinstance(episode_row, dict):
        raise FailFastError("client_summary.episode_results[0] must be an object")
    runtime_prompting = payload.get("runtime_prompting")
    if not isinstance(runtime_prompting, dict):
        raise FailFastError("client_summary.runtime_prompting must be an object")
    return {
        "task_suite_name": request.task_suite_name,
        "task_id": request.task_id,
        "seed": request.seed,
        "trial_index": request.trial_idx,
        "success": bool(episode_row.get("success", False)),
        "steps_observed": int(episode_row.get("steps_observed", 0)),
        "video_path": str(request.video_path),
        "episode_status": str(episode_row.get("error", "")).strip()
        and "runtime_error"
        or "ok",
        "error": str(episode_row.get("error", "")).strip(),
        "inference_calls": int(episode_row.get("inference_calls", 0)),
        "indicator_mode_requested": str(
            runtime_prompting.get("indicator_mode_requested", "")
        ).strip(),
        "indicator_mode": str(runtime_prompting.get("indicator_mode", "")).strip(),
        "indicator_source": str(runtime_prompting.get("indicator_source", "")).strip(),
        "prompt_text_surface": str(
            runtime_prompting.get("prompt_text_surface", "")
        ).strip(),
        "prompt_route": str(runtime_prompting.get("prompt_route", "")).strip(),
        "conditioning_mode": str(
            runtime_prompting.get("conditioning_mode", "")
        ).strip(),
        "source_prompt_field": str(
            runtime_prompting.get("source_prompt_field", "")
        ).strip(),
        "consumer_mode": str(runtime_prompting.get("consumer_mode", "")).strip(),
        "fixed_indicator_mode": str(
            runtime_prompting.get("fixed_indicator_mode", "")
        ).strip(),
        "prompt_text": str(runtime_prompting.get("prompt_text", "")).strip(),
        "critic_checkpoint_ref": str(
            runtime_prompting.get("critic_checkpoint_ref", "")
        ).strip(),
        "client_log": str(client_log),
    }
