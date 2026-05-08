#!/usr/bin/env python3
"""Thin wrapper for GR00T canonical identity preflight."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from work.recap.scripts.gr00t_canonical_identity_preflight import main


if __name__ == "__main__":
    raise SystemExit(main())
