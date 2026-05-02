#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> int:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from work.recap.iter5p5_authority import main as authority_main

    return authority_main(["--repo-root", str(repo_root), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
