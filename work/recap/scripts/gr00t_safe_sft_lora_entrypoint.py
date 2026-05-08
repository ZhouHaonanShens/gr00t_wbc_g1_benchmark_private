#!/usr/bin/env python3
"""Thin CLI wrapper for GR00T Safe-SFT LoRA-only certification.

Implementation lives under ``work/recap/safe_sft`` per repository layering
contract. This wrapper intentionally does not run RECAP/FATG/full-scope
training and exits fail-closed when any certification gate fails.
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.safe_sft.entrypoint import main


if __name__ == "__main__":
    raise SystemExit(main())
