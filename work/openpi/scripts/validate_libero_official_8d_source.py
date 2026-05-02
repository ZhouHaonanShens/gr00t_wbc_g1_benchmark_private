#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.sources.libero_official.validate import (  # noqa: E402,F401
    BLOCKED_EXIT_CODE,
    CONTRACT_REF,
    DEFAULT_DATASET_DIR,
    DEFAULT_OUT,
    INCOMPLETE_BLOCKER_CODE,
    MISSING_BLOCKER_CODE,
    build_parser,
    build_source_prereq_report,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
