"""Internal subprocess entry for the OpenPI runtime bridge.

This module exists so outer workflows can launch probe/client helpers as a
small, explicit CLI surface without turning those internal modes into a public
business API.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.runtime import bridge as runtime  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    return runtime._build_parser()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.internal_mode == "probe":
            return runtime._run_probe(args)
        if args.internal_mode == "client":
            return runtime._run_client(args)
        raise runtime.FailFastError(
            "openpi runtime internal entry only supports --internal-mode probe|client"
        )
    except subprocess.TimeoutExpired as exc:
        print(
            f"OPENPI_RUNTIME_INTERNAL_FAIL_FAST subprocess timeout: {exc}", flush=True
        )
        return 1
    except runtime.FailFastError as exc:
        print(f"OPENPI_RUNTIME_INTERNAL_FAIL_FAST {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
