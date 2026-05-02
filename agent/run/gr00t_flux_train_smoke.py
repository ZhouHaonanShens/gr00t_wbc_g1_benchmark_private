#!/usr/bin/env python3

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_impl = import_module("work.recap.scripts.gr00t_recap_training_smoke")


def build_parser(*args, **kwargs):
    return _impl.build_parser(*args, **kwargs)


def load_training_smoke_config(*args, **kwargs):
    return _impl.load_training_smoke_config(*args, **kwargs)


def resolve_smoke_plan(*args, **kwargs):
    return _impl.resolve_smoke_plan(*args, **kwargs)


def materialize_flux_training_smoke(*args, **kwargs):
    return _impl.materialize_flux_training_smoke(*args, **kwargs)


def main(*args, **kwargs):
    return _impl.main(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
