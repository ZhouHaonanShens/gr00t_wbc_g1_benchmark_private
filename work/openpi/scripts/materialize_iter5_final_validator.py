"""CLI wrapper for the Iteration 5 final validator."""

from __future__ import annotations

import sys

from work.openpi.pipelines.recap.iter5_hash_lock import main


if __name__ == "__main__":
    raise SystemExit(main(["--final-validator", *sys.argv[1:]]))
