from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path
from typing import Any


def repo_root(from_path: str | Path) -> Path:
    return Path(from_path).resolve().parents[3]


def wbc_python(repo_root_path: Path) -> Path:
    return (
        repo_root_path
        / "submodules/Isaac-GR00T/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python"
    )


def run_cmd_tee(
    cmd: list[str],
    *,
    log_path: Path,
    header: str,
    timeout_s: float,
    cwd: Path,
    env: dict[str, str],
) -> int:
    mod = __import__("work.demo_utils.runner", fromlist=["run_cmd_tee"])
    fn = getattr(mod, "run_cmd_tee")
    return int(
        fn(
            list(cmd),
            log_path=Path(log_path),
            header=str(header),
            timeout_s=float(timeout_s),
            cwd=str(cwd),
            env=dict(env),
        )
    )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def git_head_and_dirty(
    repo_root_path: Path, *, porcelain_mode: str = "--porcelain=v1"
) -> tuple[str, bool]:
    head = "unknown"
    dirty = False
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo_root_path), "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except Exception:
        head = "unknown"
    try:
        status = subprocess.check_output(
            ["git", "-C", str(repo_root_path), "status", str(porcelain_mode)],
            stderr=subprocess.STDOUT,
            text=True,
        )
        dirty = bool(str(status).strip())
    except Exception:
        dirty = False
    return str(head), bool(dirty)


def is_tcp_port_listening(host: str, port: int, timeout_s: float = 0.2) -> bool:
    normalized_host = str(host or "").strip() or "127.0.0.1"
    try:
        with socket.create_connection(
            (normalized_host, int(port)), timeout=float(timeout_s)
        ):
            return True
    except OSError:
        return False


def require_port_free(host: str, port: int, *, context: str) -> None:
    if is_tcp_port_listening(str(host), int(port)):
        raise RuntimeError(
            f"Port already listening (refuse to reuse existing server): {host}:{int(port)} ({context})"
        )


def select_latest_checkpoint(output_dir: Path) -> Path:
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)

    best_step = -1
    best_path: Path | None = None
    for candidate in sorted(output_dir.glob("checkpoint-*")):
        if not candidate.is_dir():
            continue
        if not (candidate / "trainer_state.json").is_file():
            continue
        step_suffix = candidate.name.split("checkpoint-", 1)[-1]
        try:
            step = int(step_suffix)
        except Exception:
            step = -1
        if step > best_step:
            best_step = step
            best_path = candidate

    if best_path is None:
        raise RuntimeError(
            f"No checkpoint-* with trainer_state.json found under: {output_dir}"
        )
    return best_path


__all__ = [
    "git_head_and_dirty",
    "is_tcp_port_listening",
    "repo_root",
    "require_port_free",
    "run_cmd_tee",
    "select_latest_checkpoint",
    "wbc_python",
    "write_json",
]
