#!/usr/bin/env python3

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_impl = import_module("work.recap.scripts.gr00t_screening_probe_bypass_diagnostic")


def build_parser(*args, **kwargs):
    return _impl.build_parser(*args, **kwargs)


def diagnostic_row_labels(*args, **kwargs):
    return _impl.diagnostic_row_labels(*args, **kwargs)


def build_probe_bypass_diagnostic_payload(*args, **kwargs):
    return _impl.build_probe_bypass_diagnostic_payload(*args, **kwargs)


def materialize_probe_bypass_diagnostic(*args, **kwargs):
    return _impl.materialize_probe_bypass_diagnostic(*args, **kwargs)


def main(*args, **kwargs):
    return _impl.main(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
