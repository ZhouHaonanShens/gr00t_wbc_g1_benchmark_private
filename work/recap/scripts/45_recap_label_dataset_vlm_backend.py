#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

DEFAULT_CRITIC_DIR = "agent/artifacts/critics/task7_real_critic_v2"
DEFAULT_VALUE_SOURCE = "critic"


def _target_script() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "agent"
        / "run"
        / "46d_vlm_critic_fullsize_relabel.py"
    )


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(str(arg) == str(flag) for arg in argv)


def _with_default_cli_args(argv: list[str]) -> list[str]:
    out = list(argv)
    if not _has_flag(out, "--value-source"):
        out.extend(["--value-source", str(DEFAULT_VALUE_SOURCE)])
    if not _has_flag(out, "--critic-dir"):
        out.extend(["--critic-dir", str(DEFAULT_CRITIC_DIR)])
    return out


def main() -> int:
    target = _target_script()
    argv = [str(target), *_with_default_cli_args(sys.argv[1:])]
    os.execv(sys.executable, [str(sys.executable), *argv])
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
