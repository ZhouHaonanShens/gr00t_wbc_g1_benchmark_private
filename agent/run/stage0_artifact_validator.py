#!/usr/bin/env python3
"""Compatibility CLI wrapper for the Stage0 artifact validator.

The implementation lives in :mod:`work.stage0_validator.core` so ``agent/run``
remains a thin public surface while existing imports and script invocations
keep working.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from work.stage0_validator import core as _core

__all__ = list(_core.__all__)
globals().update({name: getattr(_core, name) for name in __all__})
main = _core.main


if __name__ == "__main__":
    raise SystemExit(main())
