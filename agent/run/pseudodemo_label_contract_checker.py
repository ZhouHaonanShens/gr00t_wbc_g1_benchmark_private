#!/usr/bin/env python3

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.demo_utils.alias_wrapper import publish_module_alias


_impl = import_module("work.demo_utils.apps.pseudodemo_label_contract_checker")


_SYNC_EXCLUDED = {
    "Path",
    "REPO_ROOT",
    "import_module",
    "publish_module_alias",
    "sys",
    "_impl",
    "_SYNC_EXCLUDED",
    "_sync_wrapper_overrides",
    "build_parser",
    "main",
}


publish_module_alias(globals(), module_name=__name__, impl=_impl)


DEFAULT_CONTRACT_PATH = _impl.DEFAULT_CONTRACT_PATH

SPEC_START_MARKER = _impl.SPEC_START_MARKER
SPEC_END_MARKER = _impl.SPEC_END_MARKER
PASS_SENTINEL = _impl.PASS_SENTINEL
FAIL_SENTINEL = _impl.FAIL_SENTINEL


def _sync_wrapper_overrides() -> None:
    for name, value in globals().items():
        if name.startswith("__") or name in _SYNC_EXCLUDED:
            continue
        setattr(_impl, name, value)


_sync_wrapper_overrides()


def build_parser(*args, **kwargs):
    _sync_wrapper_overrides()
    return _impl.build_parser(*args, **kwargs)


def main(*args, **kwargs):
    _sync_wrapper_overrides()
    return _impl.main(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
