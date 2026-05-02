#!/usr/bin/env python3
"""Thin CLI for the Stage B P2 temporary inference adapter smoke."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.stage_b.p2_inference_adapter import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
