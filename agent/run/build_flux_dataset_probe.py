#!/usr/bin/env python3

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_impl = import_module("work.recap.scripts.build_flux_dataset_probe")


def build_parser(*args, **kwargs):
    return _impl.build_parser(*args, **kwargs)


def resolve_output_dir(*args, **kwargs):
    return _impl.resolve_output_dir(*args, **kwargs)


def resolve_dataset_dir(*args, **kwargs):
    return _impl.resolve_dataset_dir(*args, **kwargs)


def write_artifacts(*args, **kwargs):
    return _impl.write_artifacts(*args, **kwargs)


def main(*args, **kwargs):
    return _impl.main(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
