from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass, is_dataclass
import datetime as dt
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from .gates import wilson_ci_on_rate


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = REPO_ROOT / "agent/artifacts/recap_substrate_recovery"
BASELINE_MARKER_PATH = ARTIFACT_ROOT / "_state/last_baseline_pass.json"


class GpuTenantConflict(RuntimeError):
    pass


class R1BaselineNotPassed(RuntimeError):
    pass


class R1BaselineMarkerStale(RuntimeError):
    pass


class T81DriverCliDrift(RuntimeError):
    pass


class ReproRunFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class EpisodeRecord:
    seed: int
    terminal_step: int
    success: bool
    terminal_status: str


@dataclass(frozen=True)
class ReproCellResult:
    protocol: Any
    success_count: int
    per_episode: list[EpisodeRecord]
    wilson_ci_on_rate: tuple[float, float]
    stdout_path: Path
    stderr_path: Path
    run_manifest_path: Path | None
    formal_eval_summary_status: str


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return getattr(value, "value")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _driver_path(protocol: Any, repo_root: Path = REPO_ROOT) -> Path:
    raw = Path(str(protocol.driver_script)).expanduser()
    return raw if raw.is_absolute() else repo_root / raw


def _protocol_payload(protocol: Any) -> dict[str, Any]:
    if is_dataclass(protocol) and not isinstance(protocol, type):
        return asdict(protocol)
    fields = (
        "ckpt_root",
        "driver_script",
        "driver_sha256",
        "env_name",
        "prompt",
        "seed_base",
        "episodes",
        "max_episode_steps",
        "n_action_steps",
        "cuda_visible_devices",
        "extra_cli_args",
    )
    return {field: getattr(protocol, field) for field in fields if hasattr(protocol, field)}


def _protocol_deterministic_sha(protocol: Any) -> str:
    try:
        from .protocol import protocol_deterministic_sha

        return str(protocol_deterministic_sha(protocol))
    except Exception:
        blob = json.dumps(
            _protocol_payload(protocol),
            sort_keys=True,
            default=_json_default,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _verify_driver_sha(protocol: Any, repo_root: Path = REPO_ROOT) -> None:
    try:
        from .protocol import verify_driver_sha

        verify_driver_sha(protocol, repo_root)
        return
    except ImportError:
        pass
    driver_path = _driver_path(protocol, repo_root)
    digest = hashlib.sha256(driver_path.read_bytes()).hexdigest()
    expected = str(protocol.driver_sha256)
    if digest != expected:
        raise RuntimeError(
            f"driver sha mismatch for {driver_path}: expected {expected}, got {digest}"
        )


def _build_subprocess_env(protocol: Any) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("SUDO_")}
    env["CUDA_VISIBLE_DEVICES"] = str(protocol.cuda_visible_devices)
    return env


def _extra_cli_args(protocol: Any) -> list[str]:
    flattened: list[str] = []
    for key, value in tuple(getattr(protocol, "extra_cli_args", ()) or ()):
        flattened.extend([str(key), str(value)])
    return flattened


def _is_t81_driver(protocol: Any) -> bool:
    return Path(str(protocol.driver_script)).name == "t8_1_nav_postlift.py"


def _construct_formal_cli(protocol: Any, out_dir: Path) -> list[str]:
    eval_out = out_dir / "eval_out"
    runtime = out_dir / "runtime"
    run_manifest = out_dir / "run_manifest.json"
    argv = [
        sys.executable,
        str(_driver_path(protocol)),
        "--checkpoint",
        str(protocol.ckpt_root),
        "--run-manifest-json",
        str(run_manifest),
        "--output-dir",
        str(eval_out),
        "--runtime-log-dir",
        str(runtime),
        "--env-name",
        str(protocol.env_name),
        "--prompt-raw",
        str(protocol.prompt),
        "--seed-base",
        str(protocol.seed_base),
        "--episode-count",
        str(protocol.episodes),
        "--max-episode-steps",
        str(protocol.max_episode_steps),
        "--n-action-steps",
        str(protocol.n_action_steps),
    ]
    argv.extend(_extra_cli_args(protocol))
    argv.extend(["--required-cuda-visible-devices", str(protocol.cuda_visible_devices)])
    assert argv[-2:] == ["--required-cuda-visible-devices", str(protocol.cuda_visible_devices)]
    return argv


def _parser_option_names(parser: argparse.ArgumentParser) -> set[str]:
    names: set[str] = set()
    for action in parser._actions:
        for option in action.option_strings:
            if option.startswith("--"):
                names.add(option[2:])
    return names


def _introspect_t81_argparse(driver_path: Path) -> argparse.ArgumentParser:
    source = driver_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(driver_path))
    parser = argparse.ArgumentParser(add_help=False)
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        if not first.value.startswith("--") or first.value in seen:
            continue
        seen.add(first.value)
        try:
            parser.add_argument(first.value)
        except argparse.ArgumentError:
            continue
    return parser


def _construct_t81_cli(protocol: Any, out_dir: Path) -> list[str]:
    driver_path = _driver_path(protocol)
    parser = _introspect_t81_argparse(driver_path)
    accepted = _parser_option_names(parser)
    required = {
        "base-checkpoint",
        "env-name",
        "seed-base",
        "episode-count",
        "max-episode-steps",
        "n-action-steps",
    }
    missing_required = sorted(required - accepted)
    if missing_required:
        raise T81DriverCliDrift(
            "t8.1 driver missing required CLI options: "
            + ", ".join(missing_required)
        )
    argv = [
        sys.executable,
        str(driver_path),
        "--output-dir",
        str(out_dir / "eval_out"),
        "--base-checkpoint",
        str(protocol.ckpt_root),
        "--env-name",
        str(protocol.env_name),
        "--seed-base",
        str(protocol.seed_base),
        "--episode-count",
        str(protocol.episodes),
        "--max-episode-steps",
        str(protocol.max_episode_steps),
        "--n-action-steps",
        str(protocol.n_action_steps),
    ]
    if "prompt-raw" in accepted:
        argv.extend(["--prompt-raw", str(protocol.prompt)])
    if "required-cuda-visible-devices" in accepted:
        argv.extend(["--required-cuda-visible-devices", str(protocol.cuda_visible_devices)])
    return argv


def _construct_cli(protocol: Any, out_dir: Path) -> list[str]:
    if _is_t81_driver(protocol):
        return _construct_t81_cli(protocol, out_dir)
    return _construct_formal_cli(protocol, out_dir)


def _run_git_status() -> str:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return completed.stdout


def _pre_post_git_diff(out_dir: Path, phase: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / f"git_status_{phase}.txt"
    status_path.write_text(_run_git_status(), encoding="utf-8")
    if phase == "post":
        pre = (out_dir / "git_status_pre.txt").read_text(encoding="utf-8").splitlines()
        post = status_path.read_text(encoding="utf-8").splitlines()
        diff = "\n".join(
            difflib.unified_diff(
                pre,
                post,
                fromfile="git_status_pre.txt",
                tofile="git_status_post.txt",
                lineterm="",
            )
        )
        (out_dir / "git_diff_pre_post.txt").write_text(diff + ("\n" if diff else ""), encoding="utf-8")


def _git_diff_clean_outside_artifact_dir(out_dir: Path) -> bool:
    diff_path = out_dir / "git_diff_pre_post.txt"
    if not diff_path.is_file():
        return True
    artifact_rel = os.path.relpath(out_dir, REPO_ROOT)
    for line in diff_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("---", "+++", "@@")):
            continue
        if "agent/artifacts/recap_substrate_recovery/" in line:
            continue
        if artifact_rel in line:
            continue
        return False
    return True


def _parse_gpu_csv_line(line: str) -> tuple[int, int]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        raise ValueError(f"unexpected nvidia-smi CSV line: {line!r}")
    return int(parts[0]), int(parts[1])


def _assert_gpu_free(visible_devices: str) -> None:
    devices = [device.strip() for device in str(visible_devices).split(",") if device.strip()]
    if not devices:
        raise GpuTenantConflict("CUDA visible device string is empty")
    for device in devices:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                device,
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if gpu.returncode != 0:
            raise GpuTenantConflict(gpu.stderr.strip() or gpu.stdout.strip())
        memory_mib, utilization = _parse_gpu_csv_line(gpu.stdout.splitlines()[0])
        if memory_mib > 1024 or utilization > 5:
            raise GpuTenantConflict(
                f"GPU {device} busy: memory_mib={memory_mib}, utilization={utilization}"
            )
        apps = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                device,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        app_lines = [
            line.strip()
            for line in apps.stdout.splitlines()
            if line.strip() and "No running processes" not in line
        ]
        if apps.returncode == 0 and app_lines:
            raise GpuTenantConflict(f"GPU {device} has compute apps: {app_lines}")


def _episode_record_from_payload(payload: dict[str, Any]) -> EpisodeRecord:
    success = bool(payload.get("success", False))
    terminal_status = str(
        payload.get("terminal_status")
        or payload.get("failure_reason")
        or ("success" if success else "unknown")
    )
    return EpisodeRecord(
        seed=int(payload.get("seed", 0)),
        terminal_step=int(payload.get("outer_steps", payload.get("terminal_step", 0))),
        success=success,
        terminal_status=terminal_status,
    )


def _formal_summary_path(out_dir: Path) -> Path:
    direct = out_dir / "formal_eval_summary.json"
    if direct.is_file():
        return direct
    return out_dir / "eval_out" / "formal_eval_summary.json"


def _extract_formal_eval_result(protocol: Any, out_dir: Path, stdout_path: Path, stderr_path: Path) -> ReproCellResult:
    summary_path = _formal_summary_path(out_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    mode_summaries = summary.get("mode_summaries", {})
    mode_summary = {}
    if isinstance(mode_summaries, dict):
        mode_summary = mode_summaries.get("positive") or next(iter(mode_summaries.values()), {})
    records = [
        _episode_record_from_payload(record)
        for record in mode_summary.get("episode_results", [])
        if isinstance(record, dict)
    ]
    success_count = int(mode_summary.get("success_count", sum(1 for record in records if record.success)))
    episode_count = int(mode_summary.get("episodes", len(records) or int(getattr(protocol, "episodes", 0))))
    return ReproCellResult(
        protocol=protocol,
        success_count=success_count,
        per_episode=records,
        wilson_ci_on_rate=wilson_ci_on_rate(success_count, max(1, episode_count)),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        run_manifest_path=out_dir / "run_manifest.json",
        formal_eval_summary_status=str(summary.get("status", "MISSING")),
    )


def _outer_timeout_s(protocol: Any) -> float:
    for key, value in tuple(getattr(protocol, "extra_cli_args", ()) or ()):
        if str(key) == "--total-timeout-s":
            return float(value) + 600.0
    return 7200.0


def run_protocol(protocol: Any, out_dir: Path) -> ReproCellResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _verify_driver_sha(protocol, REPO_ROOT)
    _assert_gpu_free(str(protocol.cuda_visible_devices))
    _pre_post_git_diff(out_dir, "pre")
    argv = _construct_cli(protocol, out_dir)
    env = _build_subprocess_env(protocol)
    manifest_path = out_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "generated_at_utc": _utc_now(),
            "argv": argv,
            "cwd": str(REPO_ROOT),
            "protocol": _protocol_payload(protocol),
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_handle:
        proc = subprocess.Popen(
            argv,
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        try:
            proc.communicate(timeout=_outer_timeout_s(protocol))
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.communicate()
            raise ReproRunFailed(f"driver timed out: {exc}") from exc
    _pre_post_git_diff(out_dir, "post")
    eval_summary = out_dir / "eval_out" / "formal_eval_summary.json"
    if eval_summary.is_file():
        shutil.copy2(eval_summary, out_dir / "formal_eval_summary.json")
    result = _extract_formal_eval_result(protocol, out_dir, stdout_path, stderr_path)
    if proc.returncode != 0 and result.formal_eval_summary_status == "MISSING":
        raise ReproRunFailed(f"driver exited {proc.returncode}; see {stderr_path}")
    return result


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _persist_baseline_pass_marker(result: ReproCellResult, ckpt_config_sha: str) -> None:
    low, high = result.wilson_ci_on_rate
    payload = {
        "timestamp_utc": _utc_now(),
        "protocol_sha256": _protocol_deterministic_sha(result.protocol),
        "success_count": int(result.success_count),
        "wilson_ci_low": float(low),
        "wilson_ci_high": float(high),
        "ckpt_root_config_sha256": str(ckpt_config_sha),
        "driver_sha256": str(result.protocol.driver_sha256),
        "cuda_pin_literal": str(result.protocol.cuda_visible_devices),
        "cell_artifact_dir": str(Path(result.stdout_path).parent),
    }
    _write_json(BASELINE_MARKER_PATH, payload)


def _latest_r1_0_dir() -> Path | None:
    root = ARTIFACT_ROOT / "r1_0"
    if not root.is_dir():
        return None
    dirs = [path for path in root.iterdir() if path.is_dir()]
    return max(dirs, key=lambda path: path.stat().st_mtime) if dirs else None


def validate_baseline_pass_marker(protocol: Any) -> dict[str, Any]:
    if not BASELINE_MARKER_PATH.is_file():
        raise R1BaselineNotPassed(f"missing baseline marker: {BASELINE_MARKER_PATH}")
    marker = json.loads(BASELINE_MARKER_PATH.read_text(encoding="utf-8"))
    latest = _latest_r1_0_dir()
    if latest is not None and BASELINE_MARKER_PATH.stat().st_mtime < latest.stat().st_mtime:
        raise R1BaselineMarkerStale(
            f"baseline marker older than latest r1_0 run: {latest}"
        )
    expected_protocol_sha = _protocol_deterministic_sha(protocol)
    if marker.get("protocol_sha256") != expected_protocol_sha:
        raise R1BaselineMarkerStale("baseline marker protocol_sha256 mismatch")
    _verify_driver_sha(protocol, REPO_ROOT)
    if marker.get("driver_sha256") != str(protocol.driver_sha256):
        raise R1BaselineMarkerStale("baseline marker driver_sha256 mismatch")
    return marker
