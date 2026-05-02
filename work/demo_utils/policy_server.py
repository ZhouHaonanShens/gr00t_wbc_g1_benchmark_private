from __future__ import annotations

import datetime as _dt
import importlib
import os
import signal
import socket
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast


def normalize_client_host(host: str) -> str:
    h = (host or "").strip()
    if h in {"0.0.0.0", "*"}:
        return "127.0.0.1"
    return h


def is_tcp_port_listening(host: str, port: int, timeout_s: float = 0.2) -> bool:
    host = normalize_client_host(host)
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout_s)):
            return True
    except OSError:
        return False


def configure_policy_client_socket(client: Any, timeout_ms: int) -> None:
    zmq = importlib.import_module("zmq")

    t = int(timeout_ms)
    client.socket.setsockopt(zmq.RCVTIMEO, t)
    client.socket.setsockopt(zmq.SNDTIMEO, t)
    client.socket.setsockopt(zmq.LINGER, 0)


def make_policy_client(host: str, port: int, timeout_ms: int) -> Any:
    mod = importlib.import_module("gr00t.policy.server_client")
    PolicyClient = cast(Any, getattr(mod, "PolicyClient"))

    c = PolicyClient(host=host, port=port)
    configure_policy_client_socket(c, timeout_ms)
    return c


def safe_ping(client: Any, timeout_ms: int) -> bool:
    try:
        configure_policy_client_socket(client, int(timeout_ms))
        return bool(client.ping())
    except Exception:
        return False


def safe_kill_server(client: Any, timeout_ms: int) -> bool:
    try:
        configure_policy_client_socket(client, int(timeout_ms))
        client.kill_server()
        return True
    except Exception:
        return False


def spawn_server_subprocess(
    cmd: Sequence[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().isoformat(timespec="seconds")

    with open(log_path, "a", encoding="utf-8", buffering=1) as f:
        f.write(f"\n===== server start {ts} pid_parent={os.getpid()} =====\n")
        if cwd is not None:
            f.write(f"cwd: {cwd}\n")
        f.write("cmd: " + " ".join([str(x) for x in cmd]) + "\n")
        f.flush()

        return subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )


def terminate_process(proc: subprocess.Popen[str], timeout_s: float) -> None:
    timeout_s = float(timeout_s)
    if proc.poll() is not None:
        return

    try:
        if proc.pid and hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass

    try:
        proc.wait(timeout=max(0.1, timeout_s))
        return
    except Exception:
        pass

    try:
        if proc.pid and hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass

    try:
        proc.wait(timeout=5.0)
    except Exception:
        pass
