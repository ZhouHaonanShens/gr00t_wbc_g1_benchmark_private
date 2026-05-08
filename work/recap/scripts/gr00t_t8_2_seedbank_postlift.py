#!/usr/bin/env python3
"""Thin CLI wrapper for GR00T T8.2 no-training seed-bank/post-lift bootstrap."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.safe_sft.t8_2_seedbank_postlift import main


if __name__ == "__main__":
    raise SystemExit(main())
