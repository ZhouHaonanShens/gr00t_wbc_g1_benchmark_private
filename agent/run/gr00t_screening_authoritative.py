#!/usr/bin/env python3

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_impl = import_module("work.recap.scripts.gr00t_screening_authoritative")


def build_parser(*args, **kwargs):
    return _impl.build_parser(*args, **kwargs)


def load_authoritative_config_surface(*args, **kwargs):
    return _impl.load_authoritative_config_surface(*args, **kwargs)


def materialize_authoritative_screening(*args, **kwargs):
    return _impl.materialize_authoritative_screening(*args, **kwargs)


def main(*args, **kwargs):
    return _impl.main(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
