from __future__ import annotations

import contextlib
import datetime as _dt
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO


class _Tee:
    def __init__(self, primary: TextIO, secondary: TextIO):
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:
        self._primary.write(s)
        self._primary.flush()
        try:
            self._secondary.write(s)
            self._secondary.flush()
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:
        self._primary.flush()
        try:
            self._secondary.flush()
        except Exception:
            pass

    @property
    def encoding(self) -> str:
        enc = getattr(self._primary, "encoding", None)
        return enc if isinstance(enc, str) and enc else "utf-8"

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())


@contextlib.contextmanager
def tee_stdio(log_path: Path, *, header: str) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8", buffering=1)
    try:
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        f.write(f"\n===== {header} {ts} pid={os.getpid()} =====\n")
        f.flush()

        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out, f)
        sys.stderr = _Tee(old_err, f)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = old_out, old_err
    finally:
        try:
            f.flush()
        finally:
            f.close()
