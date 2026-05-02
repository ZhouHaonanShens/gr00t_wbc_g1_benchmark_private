#!/usr/bin/env python3

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.demo_utils.alias_wrapper import publish_module_alias


_impl = import_module("work.demo_utils.apps.official_rollout_apple_to_plate_smoke")


_SYNC_EXCLUDED = {
    "Path",
    "REPO_ROOT",
    "import_module",
    "publish_module_alias",
    "sys",
    "_impl",
    "_SYNC_EXCLUDED",
    "_sync_wrapper_overrides",
    "_build_parser",
    "main",
}


publish_module_alias(globals(), module_name=__name__, impl=_impl)


ENV_NAME = _impl.ENV_NAME

MODEL_PATH = _impl.MODEL_PATH
EMBODIMENT_TAG = _impl.EMBODIMENT_TAG

SERVER_PYTHON = _impl.SERVER_PYTHON

SERVER_HOST = _impl.SERVER_HOST
SERVER_PORT = _impl.SERVER_PORT

N_EPISODES = _impl.N_EPISODES
N_ENVS = _impl.N_ENVS
MAX_EPISODE_STEPS = _impl.MAX_EPISODE_STEPS
N_ACTION_STEPS = _impl.N_ACTION_STEPS

MUJOCO_GL = _impl.MUJOCO_GL

RUNTIME_LOGS_REL = _impl.RUNTIME_LOGS_REL
VIDEO_ARCHIVE_DIR = _impl.VIDEO_ARCHIVE_DIR

SERVER_READY_TIMEOUT_S = _impl.SERVER_READY_TIMEOUT_S
SERVER_PING_TIMEOUT_MS = _impl.SERVER_PING_TIMEOUT_MS
SERVER_PING_INTERVAL_S = _impl.SERVER_PING_INTERVAL_S

TOTAL_TIMEOUT_S = _impl.TOTAL_TIMEOUT_S


def _sync_wrapper_overrides() -> None:
    for name, value in globals().items():
        if name.startswith("__") or name in _SYNC_EXCLUDED:
            continue
        setattr(_impl, name, value)


_sync_wrapper_overrides()


def _build_parser():
    _sync_wrapper_overrides()
    return _impl._build_parser()


def main(*args, **kwargs):
    _sync_wrapper_overrides()
    return _impl.main(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
