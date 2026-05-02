"""Module entrypoint for the Stage0 artifact validator workflow."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from work.stage0_validator.core import main


if __name__ == "__main__":
    raise SystemExit(main())
