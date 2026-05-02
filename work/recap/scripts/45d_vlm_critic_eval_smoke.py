#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import importlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))


from work.recap.hf_snapshot_patch import resolve_hf_snapshot_dir
from work.recap.advantage import (
    GENERIC_DIAGNOSTIC_COMPATIBILITY_FIELDS,
    VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE,
    VLM_CRITIC_EVAL_SMOKE_DIAGNOSTIC_ROUTE,
    build_diagnostic_surface_metadata,
)


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


DEFAULT_MAIN_REPO_ROOT = str(Path(__file__).resolve().parents[3])
DEFAULT_MAIN_REPO_PYTHON = f"{DEFAULT_MAIN_REPO_ROOT}/.envs/main/bin/python"
DEFAULT_SERVER_SCRIPT = (
    f"{DEFAULT_MAIN_REPO_ROOT}/work/recap/scripts/3D_recap_run_adv_server.py"
)
DEFAULT_EVAL_SCRIPT = f"{DEFAULT_MAIN_REPO_ROOT}/work/recap/scripts/3D_recap_eval.py"
DEFAULT_MODEL_PATH = "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
DEFAULT_BASE_MODEL_PATH = ""
DEFAULT_ADV_EMBEDDING_FROM = ""
DEFAULT_OVERLAY_FROM = ""
DEFAULT_OVERLAY_INCLUDE_REGEX = r"^action_head\..*"
DEFAULT_EMBODIMENT_TAG = "UNITREE_G1"
DEFAULT_ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
DEFAULT_RUNTIME_LOG_DIR = "agent/runtime_logs/task10_vlm_critic_downstream_smoke"
DEFAULT_ARTIFACT_DIR = "agent/artifacts"
DEFAULT_TELEMETRY_DIR = "agent/artifacts/recap_eval_telemetry/task10_vlm_critic"
DEFAULT_SUMMARY_JSON = "agent/artifacts/vlm_critic_relabel/eval_smoke.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 49637
DEFAULT_ADVANTAGE = "None"
DEFAULT_EVAL_LABEL = "critic_eval_smoke_diagnostic"
DEFAULT_N_EPISODES = 1
DEFAULT_MAX_EPISODE_STEPS = 240
DEFAULT_CONNECT_TIMEOUT_S = 60.0
DEFAULT_TOTAL_TIMEOUT_S = 900.0
DEFAULT_SERVER_READY_TIMEOUT_S = 180.0
DEFAULT_SERVER_PING_INTERVAL_S = 1.0
DEFAULT_SERVER_PING_TIMEOUT_MS = 2000
DEFAULT_SERVER_TERMINATE_TIMEOUT_S = 15.0
DEFAULT_SEED_BASE = 20000
PASS_SENTINEL = "VLM_CRITIC_EVAL_SMOKE_OK"
UPGRADE_PENDING = "temporal_critic_review"


class _TeeStream:
    def __init__(self, *targets: Any):
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
    with log_path.open("a", encoding="utf-8") as f:
        stdout = _TeeStream(sys.stdout, f)
        stderr = _TeeStream(sys.stderr, f)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _path_exists_locally(repo_root: Path, raw: str) -> bool:
    text = str(raw).strip()
    if not text:
        return False
    direct = Path(text).expanduser()
    if direct.exists():
        return True
    if direct.is_absolute():
        return False
    return (repo_root / direct).expanduser().exists()


def _looks_like_hf_repo_id(raw: str) -> bool:
    text = str(raw).strip()
    if not text or text.startswith(("/", "./", "../", "~/")):
        return False
    parts = text.split("/")
    if len(parts) != 2:
        return False
    return all(bool(part.strip()) for part in parts)


def _resolve_server_model_path(
    *,
    repo_root: Path,
    model_path: str,
    unconditional_baseline_case: bool,
) -> tuple[str, bool]:
    if not unconditional_baseline_case:
        return str(model_path), False
    if _path_exists_locally(repo_root, model_path):
        return str(model_path), False
    if not _looks_like_hf_repo_id(model_path):
        return str(model_path), False
    snapshot_dir = resolve_hf_snapshot_dir(
        repo_id=str(model_path),
        emit_evidence=False,
    )
    return str(snapshot_dir), True


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return dict(data)


def _site_packages_path(main_repo_root: Path) -> Path:
    candidates = sorted(
        (main_repo_root / ".envs" / "main" / "lib").glob("python*/site-packages")
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate main-repo site-packages under {main_repo_root / '.envs' / 'main' / 'lib'}"
        )
    return candidates[-1]


def _bridge_import_roots(main_repo_root: Path) -> list[Path]:
    candidates = [main_repo_root, main_repo_root / "submodules" / "Isaac-GR00T"]
    roots: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists() and resolved not in roots:
            roots.append(resolved)
    return roots


def _prepend_pythonpath(env: dict[str, str], entries: list[Path]) -> str:
    existing_raw = str(env.get("PYTHONPATH", "")).strip()
    combined: list[str] = []
    for entry in entries:
        normalized = str(entry)
        if normalized and normalized not in combined:
            combined.append(normalized)
    for entry in existing_raw.split(os.pathsep):
        normalized = entry.strip()
        if normalized and normalized not in combined:
            combined.append(normalized)
    joined = os.pathsep.join(combined)
    env["PYTHONPATH"] = joined
    return joined


def _prepend_sys_path(entries: list[Path]) -> list[str]:
    inserted: list[str] = []
    for entry in reversed(entries):
        normalized = str(entry)
        if normalized and normalized not in sys.path:
            sys.path.insert(0, normalized)
            inserted.append(normalized)
    inserted.reverse()
    return inserted


def _probe_python_imports(python_exe: Path, env: dict[str, str]) -> tuple[bool, str]:
    probe_cmd = [
        str(python_exe),
        "-c",
        (
            "import importlib, json\n"
            "mods = {}\n"
            "for name in ('torch', 'transformers', 'gr00t', 'work.demo_utils.policy_server'):\n"
            "    mod = importlib.import_module(name)\n"
            "    mods[name] = getattr(mod, '__file__', '<built-in>')\n"
            "print('BRIDGE_PROBE_OK', json.dumps(mods, sort_keys=True))"
        ),
    ]
    proc = subprocess.run(
        probe_cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    preview = (proc.stdout or proc.stderr or "").strip()
    if len(preview) > 500:
        preview = preview[:500]
    return bool(proc.returncode == 0), preview


def _probe_wrapper_import(module_name: str) -> tuple[bool, str]:
    try:
        importlib.invalidate_caches()
        mod = importlib.import_module(module_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(mod, "__file__", "<built-in>"))


def _bridge_wrapper_sys_path(main_repo_root: Path) -> dict[str, Any]:
    bridge_roots = _bridge_import_roots(main_repo_root)
    roots_inserted = _prepend_sys_path(bridge_roots)
    import_ok, preview = _probe_wrapper_import("gr00t.policy.server_client")
    bridge_info: dict[str, Any] = {
        "bridge_import_roots": [str(path) for path in bridge_roots],
        "bridge_mode": "direct",
        "site_packages_path": None,
        "sys_path_inserted": list(roots_inserted),
        "probe_import_ok_before_bridge": bool(import_ok),
        "probe_preview_before_bridge": preview,
        "probe_import_ok_after_bridge": bool(import_ok),
        "probe_preview_after_bridge": preview,
    }
    if import_ok:
        return bridge_info

    site_packages: Path | None = None
    try:
        site_packages = _site_packages_path(main_repo_root)
    except FileNotFoundError:
        site_packages = None

    site_packages_inserted: list[str] = []
    if site_packages is not None:
        site_packages_inserted = _prepend_sys_path([site_packages])
    import_ok_after, preview_after = _probe_wrapper_import("gr00t.policy.server_client")
    bridge_info.update(
        {
            "bridge_mode": "main_repo_pythonpath_bridge",
            "site_packages_path": str(site_packages) if site_packages else None,
            "sys_path_inserted": site_packages_inserted + roots_inserted,
            "probe_import_ok_after_bridge": bool(import_ok_after),
            "probe_preview_after_bridge": preview_after,
        }
    )
    return bridge_info


def _build_upstream_env(
    *, main_repo_root: Path, python_exe: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    requested = str(python_exe)
    resolved = str(python_exe.resolve())
    bridge_roots = _bridge_import_roots(main_repo_root)
    existing_pythonpath = str(env.get("PYTHONPATH", "")).strip() or None
    attempted_bridges: list[dict[str, Any]] = []

    import_ok, preview = _probe_python_imports(python_exe, env)
    attempted_bridges.append(
        {
            "bridge_mode": "direct",
            "pythonpath": existing_pythonpath,
            "probe_import_ok": bool(import_ok),
            "probe_preview": preview,
            "site_packages_path": None,
        }
    )
    bridge_info: dict[str, Any] = {
        "python_requested": requested,
        "python_resolved": resolved,
        "bridge_import_roots": [str(path) for path in bridge_roots],
        "pythonpath_before_bridge": existing_pythonpath,
        "probe_import_ok_before_bridge": bool(import_ok),
        "probe_preview_before_bridge": preview,
        "bridge_mode": "direct",
        "bridge_attempts": attempted_bridges,
        "site_packages_path": None,
        "pythonpath_after_bridge": existing_pythonpath,
        "probe_import_ok_after_bridge": bool(import_ok),
        "probe_preview_after_bridge": preview,
    }
    if import_ok:
        return env, bridge_info

    roots_only_env = dict(env)
    roots_only_pythonpath = _prepend_pythonpath(roots_only_env, list(bridge_roots))
    roots_only_ok, roots_only_preview = _probe_python_imports(
        python_exe, roots_only_env
    )
    attempted_bridges.append(
        {
            "bridge_mode": "roots_only",
            "pythonpath": roots_only_pythonpath or None,
            "probe_import_ok": bool(roots_only_ok),
            "probe_preview": roots_only_preview,
            "site_packages_path": None,
        }
    )
    if roots_only_ok:
        bridge_info.update(
            {
                "bridge_mode": "roots_only",
                "pythonpath_after_bridge": roots_only_pythonpath or None,
                "probe_import_ok_after_bridge": bool(roots_only_ok),
                "probe_preview_after_bridge": roots_only_preview,
            }
        )
        return roots_only_env, bridge_info

    bridge_entries = list(bridge_roots)
    site_packages: Path | None = None
    try:
        site_packages = _site_packages_path(main_repo_root)
    except FileNotFoundError:
        site_packages = None
    if site_packages is not None:
        bridge_entries.append(site_packages)
    fallback_env = dict(env)
    pythonpath_after_bridge = _prepend_pythonpath(fallback_env, bridge_entries)
    import_ok_after, preview_after = _probe_python_imports(python_exe, fallback_env)
    attempted_bridges.append(
        {
            "bridge_mode": "main_repo_pythonpath_bridge",
            "pythonpath": pythonpath_after_bridge or None,
            "probe_import_ok": bool(import_ok_after),
            "probe_preview": preview_after,
            "site_packages_path": str(site_packages) if site_packages else None,
        }
    )
    bridge_info.update(
        {
            "bridge_mode": "main_repo_pythonpath_bridge",
            "site_packages_path": str(site_packages) if site_packages else None,
            "pythonpath_after_bridge": pythonpath_after_bridge or None,
            "probe_import_ok_after_bridge": bool(import_ok_after),
            "probe_preview_after_bridge": preview_after,
        }
    )
    return fallback_env, bridge_info


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _safe_kill_server(client: Any, timeout_ms: int) -> bool:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "safe_kill_server")
    return bool(fn(client, int(timeout_ms)))


def _spawn_server_subprocess(
    cmd: list[str], *, log_path: Path, cwd: Path, env: dict[str, str]
) -> subprocess.Popen[str]:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "spawn_server_subprocess")
    return fn(cmd, log_path=log_path, cwd=cwd, env=env)


def _terminate_process(proc: subprocess.Popen[str], timeout_s: float) -> None:
    mod = importlib.import_module("work.demo_utils.policy_server")
    fn = getattr(mod, "terminate_process")
    fn(proc, float(timeout_s))


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


def _critic_smoke_diagnostic_metadata() -> dict[str, Any]:
    return build_diagnostic_surface_metadata(
        surface_route=VLM_CRITIC_EVAL_SMOKE_DIAGNOSTIC_ROUTE,
        authority_scope=VLM_CRITIC_DIAGNOSTIC_AUTHORITY_SCOPE,
        compatibility_fields=GENERIC_DIAGNOSTIC_COMPATIBILITY_FIELDS,
        surface_kind="vlm_critic_eval_smoke_summary",
    )


def _merge_eval_smoke_summary(
    *,
    upstream_summary: dict[str, Any] | None,
    wrapper_error: str | None,
    summary_json: Path,
    args: argparse.Namespace,
    host_for_client: str,
    model_path: str,
    model_path_is_local_source: bool,
    server_model_path: str,
    server_model_path_is_local_source: bool,
    unconditional_baseline_case: bool,
    baseline_local_snapshot_rewrite_applied: bool,
    base_model_path: str,
    overlay_from: str,
    overlay_include_regex: str,
    overlay_input_source: str | None,
    legacy_adv_embedding_from_raw: str,
    stats_from_model_path: str,
    require_advantage_embedding: bool,
    allow_baseline_default_advantage_embedding_init: bool,
    main_repo_root: Path,
    python_exe: Path,
    bridge_info: dict[str, Any],
    wrapper_bridge_info: dict[str, Any],
    server_script: Path,
    eval_script: Path,
    server_cmd: list[str],
    eval_cmd: list[str],
    server_started: bool,
    server_log: Path,
    wrapper_log: Path,
    upstream_summary_path: Path,
    eval_returncode: int | None,
) -> dict[str, Any]:
    fallback_summary: dict[str, Any] = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "advantage": str(args.advantage),
        "advantage_mode": None,
        "episodes": 0,
        "requested_episodes": int(args.n_episodes),
        "success_count": 0,
        "success_rate": 0.0,
        "env_name": str(args.env_name),
        "host": str(host_for_client),
        "port": int(args.port),
        "max_episode_steps": int(args.max_episode_steps),
        "seed_base": int(args.seed_base),
        "error": wrapper_error or "eval_summary_not_generated",
    }
    merged = dict(upstream_summary or fallback_summary)
    merged.update(
        {
            "wrapper": "45d_vlm_critic_eval_smoke.py",
            "sentinel": PASS_SENTINEL,
            "wrapper_status": "ok" if wrapper_error is None else "blocked",
            "summary_json": str(summary_json),
            "eval_label": str(args.eval_label),
            "model_path": str(model_path),
            "model_path_is_local_source": bool(model_path_is_local_source),
            "server_model_path": str(server_model_path),
            "server_model_path_is_local_source": bool(
                server_model_path_is_local_source
            ),
            "unconditional_baseline_case": bool(unconditional_baseline_case),
            "baseline_local_snapshot_rewrite_applied": bool(
                baseline_local_snapshot_rewrite_applied
            ),
            "base_model_path": str(base_model_path) if base_model_path else None,
            "overlay_from": str(overlay_from) if overlay_from else None,
            "overlay_include_regex": (
                str(overlay_include_regex) if overlay_from else None
            ),
            "overlay_input_source": overlay_input_source,
            "adv_embedding_from_legacy_input": (
                str(legacy_adv_embedding_from_raw)
                if legacy_adv_embedding_from_raw
                else None
            ),
            "stats_from_model_path": (
                str(stats_from_model_path) if stats_from_model_path else None
            ),
            "require_advantage_embedding": bool(require_advantage_embedding),
            "allow_baseline_default_advantage_embedding_init": bool(
                allow_baseline_default_advantage_embedding_init
            ),
            "embodiment_tag": str(args.embodiment_tag),
            "main_repo_root": str(main_repo_root),
            "python": str(python_exe),
            "interpreter_bridge": bridge_info,
            "wrapper_import_bridge": wrapper_bridge_info,
            "server_script": str(server_script),
            "eval_script": str(eval_script),
            "server_cmd": server_cmd,
            "eval_cmd": eval_cmd,
            "server_started": bool(server_started),
            "server_log": str(server_log),
            "wrapper_log": str(wrapper_log),
            "upstream_summary_path": str(upstream_summary_path),
            "upstream_summary_exists": bool(upstream_summary_path.is_file()),
            "upstream_eval_returncode": eval_returncode,
            "upgrade_pending": UPGRADE_PENDING,
            "error": merged.get("error") or wrapper_error,
        }
    )
    merged.update(_critic_smoke_diagnostic_metadata())
    return merged


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="45d_vlm_critic_eval_smoke.py",
        description=(
            "Branch-local wrapper around main-repo 3D_recap_run_adv_server.py + 3D_recap_eval.py. "
            "Always emits a machine-readable summary JSON under the delegated worktree."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base-model-path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument(
        "--adv-embedding-from", type=str, default=DEFAULT_ADV_EMBEDDING_FROM
    )
    parser.add_argument("--overlay-from", type=str, default=DEFAULT_OVERLAY_FROM)
    parser.add_argument(
        "--overlay-include-regex",
        type=str,
        default=DEFAULT_OVERLAY_INCLUDE_REGEX,
    )
    parser.add_argument("--stats-from-model-path", type=str, default="")
    parser.add_argument("--embodiment-tag", type=str, default=DEFAULT_EMBODIMENT_TAG)
    parser.add_argument("--env-name", type=str, default=DEFAULT_ENV_NAME)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=int(DEFAULT_PORT))
    parser.add_argument("--advantage", type=str, default=DEFAULT_ADVANTAGE)
    parser.add_argument("--eval-label", type=str, default=DEFAULT_EVAL_LABEL)
    parser.add_argument("--summary-json", type=str, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--runtime-log-dir", type=str, default=DEFAULT_RUNTIME_LOG_DIR)
    parser.add_argument("--artifact-dir", type=str, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--telemetry-dir", type=str, default=DEFAULT_TELEMETRY_DIR)
    parser.add_argument("--n-episodes", type=int, default=int(DEFAULT_N_EPISODES))
    parser.add_argument(
        "--max-episode-steps", type=int, default=int(DEFAULT_MAX_EPISODE_STEPS)
    )
    parser.add_argument(
        "--connect-timeout-s", type=float, default=float(DEFAULT_CONNECT_TIMEOUT_S)
    )
    parser.add_argument(
        "--total-timeout-s", type=float, default=float(DEFAULT_TOTAL_TIMEOUT_S)
    )
    parser.add_argument(
        "--server-ready-timeout-s",
        type=float,
        default=float(DEFAULT_SERVER_READY_TIMEOUT_S),
    )
    parser.add_argument(
        "--server-ping-interval-s",
        type=float,
        default=float(DEFAULT_SERVER_PING_INTERVAL_S),
    )
    parser.add_argument(
        "--server-ping-timeout-ms",
        type=int,
        default=int(DEFAULT_SERVER_PING_TIMEOUT_MS),
    )
    parser.add_argument(
        "--server-terminate-timeout-s",
        type=float,
        default=float(DEFAULT_SERVER_TERMINATE_TIMEOUT_S),
    )
    parser.add_argument("--seed-base", type=int, default=int(DEFAULT_SEED_BASE))
    parser.add_argument("--main-repo-root", type=str, default=DEFAULT_MAIN_REPO_ROOT)
    parser.add_argument("--python", type=str, default=DEFAULT_MAIN_REPO_PYTHON)
    parser.add_argument("--server-script", type=str, default=DEFAULT_SERVER_SCRIPT)
    parser.add_argument("--eval-script", type=str, default=DEFAULT_EVAL_SCRIPT)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = _repo_root()
    main_repo_root = Path(str(args.main_repo_root)).expanduser().resolve()
    python_exe = Path(str(args.python)).expanduser()
    server_script = Path(str(args.server_script)).expanduser().resolve()
    eval_script = Path(str(args.eval_script)).expanduser().resolve()
    runtime_log_dir = _resolve_path(repo_root, str(args.runtime_log_dir))
    artifact_dir = _resolve_path(repo_root, str(args.artifact_dir))
    telemetry_dir = _resolve_path(repo_root, str(args.telemetry_dir))
    summary_json = _resolve_path(repo_root, str(args.summary_json))
    model_path_raw = str(args.model_path)
    model_path_resolved = Path(model_path_raw).expanduser()
    model_path = (
        str(_resolve_path(repo_root, model_path_raw))
        if not model_path_resolved.is_absolute() and model_path_raw.startswith("agent/")
        else str(model_path_raw)
    )
    base_model_path_raw = str(args.base_model_path or "").strip()
    if base_model_path_raw:
        base_model_resolved = Path(base_model_path_raw).expanduser()
        base_model_path = (
            str(_resolve_path(repo_root, base_model_path_raw))
            if not base_model_resolved.is_absolute()
            and base_model_path_raw.startswith("agent/")
            else base_model_path_raw
        )
    else:
        base_model_path = ""
    legacy_adv_embedding_from_raw = str(args.adv_embedding_from or "").strip()
    overlay_from_raw = (
        str(args.overlay_from or "").strip() or legacy_adv_embedding_from_raw
    )
    overlay_input_source = (
        "overlay_from"
        if str(args.overlay_from or "").strip()
        else "adv_embedding_from"
        if legacy_adv_embedding_from_raw
        else None
    )
    if overlay_from_raw:
        overlay_path_resolved = Path(overlay_from_raw).expanduser()
        overlay_from = (
            str(_resolve_path(repo_root, overlay_from_raw))
            if not overlay_path_resolved.is_absolute()
            and overlay_from_raw.startswith("agent/")
            else overlay_from_raw
        )
    else:
        overlay_from = ""
    overlay_include_regex = str(args.overlay_include_regex).strip()
    stats_from_model_path_raw = str(args.stats_from_model_path or "").strip()
    if stats_from_model_path_raw:
        stats_path_resolved = Path(stats_from_model_path_raw).expanduser()
        stats_from_model_path = (
            str(_resolve_path(repo_root, stats_from_model_path_raw))
            if not stats_path_resolved.is_absolute()
            and stats_from_model_path_raw.startswith("agent/")
            else stats_from_model_path_raw
        )
    else:
        stats_from_model_path = ""

    advantage_raw = str(args.advantage).strip()
    advantage_is_none = advantage_raw.lower() == "none"
    model_path_is_local_source = _path_exists_locally(repo_root, model_path)
    unconditional_baseline_case = bool(
        advantage_is_none
        and not base_model_path
        and not overlay_from
        and not legacy_adv_embedding_from_raw
    )
    server_model_path, baseline_local_snapshot_rewrite_applied = (
        _resolve_server_model_path(
            repo_root=repo_root,
            model_path=model_path,
            unconditional_baseline_case=unconditional_baseline_case,
        )
    )
    server_model_path_is_local_source = _path_exists_locally(
        repo_root, server_model_path
    )
    require_advantage_embedding = True
    allow_baseline_default_advantage_embedding_init = bool(
        unconditional_baseline_case and server_model_path_is_local_source
    )

    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    tag = f"{str(args.eval_label).strip() or 'eval'}_{_timestamp()}"
    wrapper_log = runtime_log_dir / f"45d_wrapper_{tag}.log"
    server_log = runtime_log_dir / f"45d_server_{tag}.log"
    upstream_summary_path = runtime_log_dir / f"45d_upstream_eval_{tag}.json"
    upstream_env, bridge_info = _build_upstream_env(
        main_repo_root=main_repo_root,
        python_exe=python_exe,
    )
    wrapper_bridge_info = _bridge_wrapper_sys_path(main_repo_root)
    wrapper_import_roots = [
        Path(path) for path in wrapper_bridge_info["bridge_import_roots"]
    ]
    wrapper_sys_path_inserted = list(wrapper_bridge_info["sys_path_inserted"])

    host_for_client = _normalize_client_host(str(args.host))
    server_cmd = [
        str(python_exe),
        str(server_script),
        "--model-path",
        str(server_model_path),
        "--embodiment-tag",
        str(args.embodiment_tag),
        "--host",
        str(args.host),
        "--port",
        str(int(args.port)),
    ]
    if base_model_path:
        server_cmd.extend(["--base-model-path", str(base_model_path)])
    if overlay_from:
        server_cmd.extend(
            [
                "--overlay-from",
                str(overlay_from),
                "--overlay-include-regex",
                str(overlay_include_regex),
            ]
        )
    if stats_from_model_path:
        server_cmd.extend(["--stats-from-model-path", str(stats_from_model_path)])
    if allow_baseline_default_advantage_embedding_init:
        server_cmd.append("--allow-baseline-default-advantage-embedding-init")

    eval_cmd = [
        str(python_exe),
        str(eval_script),
        "--env-name",
        str(args.env_name),
        "--host",
        str(host_for_client),
        "--port",
        str(int(args.port)),
        "--n-episodes",
        str(int(args.n_episodes)),
        "--max-episode-steps",
        str(int(args.max_episode_steps)),
        "--advantage",
        str(args.advantage),
        "--runtime-log-dir",
        str(runtime_log_dir),
        "--artifact-dir",
        str(artifact_dir),
        "--telemetry-dir",
        str(telemetry_dir),
        "--summary-json",
        str(upstream_summary_path),
        "--connect-timeout-s",
        str(float(args.connect_timeout_s)),
        "--total-timeout-s",
        str(float(args.total_timeout_s)),
        "--seed-base",
        str(int(args.seed_base)),
    ]

    wrapper_error: str | None = None
    upstream_summary: dict[str, Any] | None = None
    server_started = False
    eval_returncode: int | None = None
    server_proc: subprocess.Popen[str] | None = None
    client: Any | None = None

    with _tee_stdio(wrapper_log):
        print("[INFO] ts:", _dt.datetime.now().isoformat(timespec="seconds"))
        print("[INFO] wrapper:", "45d_vlm_critic_eval_smoke.py")
        print("[INFO] main_repo_root:", main_repo_root)
        print("[INFO] python:", python_exe)
        print(
            "[INFO] wrapper_import_roots:",
            json.dumps([str(path) for path in wrapper_import_roots], ensure_ascii=True),
        )
        print(
            "[INFO] wrapper_sys_path_inserted:",
            json.dumps(wrapper_sys_path_inserted, ensure_ascii=True),
        )
        print(
            "[INFO] wrapper_import_bridge:",
            json.dumps(wrapper_bridge_info, ensure_ascii=True),
        )
        print("[INFO] model_path:", model_path)
        print("[INFO] model_path_is_local_source:", model_path_is_local_source)
        print("[INFO] unconditional_baseline_case:", unconditional_baseline_case)
        print("[INFO] server_model_path:", server_model_path)
        print(
            "[INFO] server_model_path_is_local_source:",
            server_model_path_is_local_source,
        )
        print(
            "[INFO] baseline_local_snapshot_rewrite_applied:",
            baseline_local_snapshot_rewrite_applied,
        )
        print("[INFO] base_model_path:", base_model_path or None)
        print("[INFO] overlay_from:", overlay_from or None)
        print(
            "[INFO] overlay_include_regex:",
            overlay_include_regex if overlay_from else None,
        )
        print(
            "[INFO] overlay_input_source:",
            overlay_input_source,
        )
        print(
            "[INFO] adv_embedding_from_legacy_input:",
            legacy_adv_embedding_from_raw or None,
        )
        print("[INFO] stats_from_model_path:", stats_from_model_path or None)
        print("[INFO] require_advantage_embedding:", require_advantage_embedding)
        print(
            "[INFO] allow_baseline_default_advantage_embedding_init:",
            allow_baseline_default_advantage_embedding_init,
        )
        print("[INFO] summary_json:", summary_json)
        print("[INFO] wrapper_log:", wrapper_log)
        print("[INFO] server_log:", server_log)
        print("[INFO] upstream_summary_path:", upstream_summary_path)
        print("[INFO] server_cmd:", json.dumps(server_cmd, ensure_ascii=True))
        print("[INFO] eval_cmd:", json.dumps(eval_cmd, ensure_ascii=True))
        print("[INFO] interpreter_bridge:", json.dumps(bridge_info, ensure_ascii=True))
        try:
            try:
                with socket.create_connection(
                    (host_for_client, int(args.port)), timeout=0.2
                ):
                    raise RuntimeError(
                        f"port_in_use: refusing to reuse occupied port {host_for_client}:{int(args.port)}"
                    )
            except OSError:
                pass

            server_proc = _spawn_server_subprocess(
                server_cmd,
                log_path=server_log,
                cwd=main_repo_root,
                env=upstream_env,
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
                        f"rc={server_proc.returncode} preview={_server_error_preview(server_log)!r}"
                    )
                elapsed = time.monotonic() - t0
                if elapsed > float(args.server_ready_timeout_s):
                    raise TimeoutError(
                        "server_ready_timeout: "
                        f"waited>{float(args.server_ready_timeout_s):.1f}s preview={_server_error_preview(server_log)!r}"
                    )
                time.sleep(float(args.server_ping_interval_s))

            eval_proc = subprocess.run(
                eval_cmd,
                cwd=str(main_repo_root),
                env=upstream_env,
                check=False,
            )
            eval_returncode = int(eval_proc.returncode)
            if upstream_summary_path.is_file():
                upstream_summary = _read_json(upstream_summary_path)
            if eval_returncode != 0:
                wrapper_error = f"upstream_eval_failed: returncode={eval_returncode}"
            elif upstream_summary is None:
                wrapper_error = "upstream_eval_missing_summary_json"
        except Exception as exc:
            wrapper_error = f"wrapper_exception: {type(exc).__name__}: {exc}"
            if upstream_summary_path.is_file():
                try:
                    upstream_summary = _read_json(upstream_summary_path)
                except Exception:
                    upstream_summary = None
        finally:
            if client is not None:
                try:
                    _ = _safe_kill_server(client, int(args.server_ping_timeout_ms))
                except Exception:
                    pass
            if server_proc is not None:
                _terminate_process(server_proc, float(args.server_terminate_timeout_s))

    merged = _merge_eval_smoke_summary(
        upstream_summary=upstream_summary,
        wrapper_error=wrapper_error,
        summary_json=summary_json,
        args=args,
        host_for_client=host_for_client,
        model_path=model_path,
        model_path_is_local_source=model_path_is_local_source,
        server_model_path=server_model_path,
        server_model_path_is_local_source=server_model_path_is_local_source,
        unconditional_baseline_case=unconditional_baseline_case,
        baseline_local_snapshot_rewrite_applied=baseline_local_snapshot_rewrite_applied,
        base_model_path=base_model_path,
        overlay_from=overlay_from,
        overlay_include_regex=overlay_include_regex,
        overlay_input_source=overlay_input_source,
        legacy_adv_embedding_from_raw=legacy_adv_embedding_from_raw,
        stats_from_model_path=stats_from_model_path,
        require_advantage_embedding=require_advantage_embedding,
        allow_baseline_default_advantage_embedding_init=(
            allow_baseline_default_advantage_embedding_init
        ),
        main_repo_root=main_repo_root,
        python_exe=python_exe,
        bridge_info=bridge_info,
        wrapper_bridge_info=wrapper_bridge_info,
        server_script=server_script,
        eval_script=eval_script,
        server_cmd=server_cmd,
        eval_cmd=eval_cmd,
        server_started=server_started,
        server_log=server_log,
        wrapper_log=wrapper_log,
        upstream_summary_path=upstream_summary_path,
        eval_returncode=eval_returncode,
    )
    _write_json(summary_json, merged)
    print(json.dumps(merged, ensure_ascii=True, indent=2, sort_keys=True))
    print(f"SENTINEL:{PASS_SENTINEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
