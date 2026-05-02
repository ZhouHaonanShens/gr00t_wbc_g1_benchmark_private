from __future__ import annotations

import datetime as _dt
import os
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO


def _mask_env_value(key: str, value: str) -> str:
    k = key.upper()
    if any(t in k for t in ("TOKEN", "SECRET", "PASSWORD", "PASS", "KEY")):
        return "***"
    return value


def _write_header(
    f: TextIO,
    *,
    header: str,
    cmd: Sequence[str],
    cwd: Path | None,
    env_overrides: Mapping[str, str] | None,
) -> None:
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    f.write(f"\n===== {header} {ts} pid_parent={os.getpid()} =====\n")
    if cwd is not None:
        f.write(f"cwd: {cwd.as_posix()}\n")
    f.write("cmd: " + " ".join([str(x) for x in cmd]) + "\n")
    if env_overrides:
        keys = sorted([str(k) for k in env_overrides.keys()])
        f.write("env_override_keys: " + ",".join(keys) + "\n")
        preview_items = []
        for k in keys[:30]:
            v = str(env_overrides.get(k, ""))
            preview_items.append(f"{k}={_mask_env_value(k, v)}")
        f.write("env_override_preview: " + " ".join(preview_items) + "\n")
    f.flush()


def _interrupt_then_kill(
    proc: subprocess.Popen[str], *, log_f: TextIO, kill_after_s: float
) -> None:
    if proc.poll() is not None:
        return

    try:
        log_f.write("[INFO] timeout/interrupt: sending SIGINT\n")
        log_f.flush()
    except Exception:
        pass

    try:
        if proc.pid and hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGINT)
        else:
            proc.send_signal(signal.SIGINT)
    except Exception:
        pass

    try:
        proc.wait(timeout=max(0.1, float(kill_after_s)))
        return
    except Exception:
        pass

    try:
        mod = __import__(
            "work.demo_utils.policy_server", fromlist=["terminate_process"]
        )
        terminate_process = getattr(mod, "terminate_process")
        terminate_process(proc, 5.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_cmd_tee(
    cmd: Sequence[str],
    log_path: str | Path,
    header: str,
    timeout_s: float | None = None,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_str = [str(x) for x in cmd]
    cwd_path = Path(cwd).resolve() if cwd is not None else None

    env_full = os.environ.copy()
    env_overrides = dict(env) if env is not None else {}
    if "PYTHONUNBUFFERED" not in env_full and "PYTHONUNBUFFERED" not in env_overrides:
        env_overrides["PYTHONUNBUFFERED"] = "1"
    env_full.update(env_overrides)

    deadline = None
    timeout_s_float: float | None = None
    if timeout_s is not None:
        try:
            t = float(timeout_s)
            if t > 0:
                timeout_s_float = t
                deadline = time.monotonic() + t
        except Exception:
            deadline = None

    rc_timeout = 124
    kill_after_s = 20.0

    with open(log_path, "a", encoding="utf-8", buffering=1) as f:
        _write_header(
            f,
            header=str(header),
            cmd=cmd_str,
            cwd=cwd_path,
            env_overrides=env_overrides,
        )

        proc: subprocess.Popen[str] | None = None
        sel = selectors.DefaultSelector()

        def _tee_line(line: str) -> None:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception:
                pass
            try:
                f.write(line)
                f.flush()
            except Exception:
                pass

        try:
            proc = subprocess.Popen(
                cmd_str,
                cwd=str(cwd_path) if cwd_path is not None else None,
                env=env_full,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
            assert proc.stdout is not None
            sel.register(proc.stdout, selectors.EVENT_READ)

            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    t_disp = timeout_s_float if timeout_s_float is not None else 0.0
                    f.write(f"[ERROR] Timed out after {t_disp:.0f}s\n")
                    f.flush()
                    _interrupt_then_kill(proc, log_f=f, kill_after_s=kill_after_s)
                    return rc_timeout

                try:
                    events = sel.select(timeout=0.2)
                except Exception:
                    events = []
                if not events:
                    if proc.poll() is not None:
                        break
                    continue

                for _key, _mask in events:
                    try:
                        line = proc.stdout.readline()
                    except Exception:
                        line = ""

                    if not line:
                        if proc.poll() is not None:
                            try:
                                sel.unregister(proc.stdout)
                            except Exception:
                                pass
                            break
                        continue

                    _tee_line(line)

                if proc.poll() is not None:
                    break

            try:
                for line in proc.stdout:
                    _tee_line(line)
            except Exception:
                pass

            try:
                proc.wait(timeout=5.0)
            except Exception:
                _interrupt_then_kill(proc, log_f=f, kill_after_s=1.0)

            return int(proc.returncode if proc.returncode is not None else 1)
        except KeyboardInterrupt:
            f.write("\n[INFO] KeyboardInterrupt -> terminating child process\n")
            f.flush()
            if proc is not None:
                _interrupt_then_kill(proc, log_f=f, kill_after_s=5.0)
            return 130
        finally:
            try:
                sel.close()
            except Exception:
                pass
