#!/usr/bin/env python3

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.demo_utils.alias_wrapper import publish_module_alias


_impl = import_module("work.demo_utils.apps.official_rollout_apple_to_plate_onscreen")


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

SERVER_HOST = _impl.SERVER_HOST
SERVER_PORT = _impl.SERVER_PORT

N_EPISODES = _impl.N_EPISODES
MAX_EPISODE_STEPS = _impl.MAX_EPISODE_STEPS

N_ACTION_STEPS = _impl.N_ACTION_STEPS

ONSCREEN = _impl.ONSCREEN
OFFSCREEN = _impl.OFFSCREEN
MUJOCO_GL = _impl.MUJOCO_GL

RENDER_CAMERA = _impl.RENDER_CAMERA

RENDERER = _impl.RENDERER
HARD_RESET = _impl.HARD_RESET

STEPS_PER_RENDER = _impl.STEPS_PER_RENDER

VIDEO_ARCHIVE_DIR = _impl.VIDEO_ARCHIVE_DIR
RUNTIME_LOGS_REL = _impl.RUNTIME_LOGS_REL
LOG_BASENAME = _impl.LOG_BASENAME
VARIANT_LOG_BASENAME = _impl.VARIANT_LOG_BASENAME

ENV_VARIANTS = _impl.ENV_VARIANTS

ENV_VARIANT = _impl.ENV_VARIANT
APPLE_X = _impl.APPLE_X
APPLE_Y = _impl.APPLE_Y
VARIANT_DEBUG = _impl.VARIANT_DEBUG

TASK_PROMPT_OVERRIDE = _impl.TASK_PROMPT_OVERRIDE

SERVER_READY_TIMEOUT_S = _impl.SERVER_READY_TIMEOUT_S
SERVER_PING_TIMEOUT_MS = _impl.SERVER_PING_TIMEOUT_MS
SERVER_PING_INTERVAL_S = _impl.SERVER_PING_INTERVAL_S

TOTAL_TIMEOUT_S = _impl.TOTAL_TIMEOUT_S

SAVE_COMPOSITE_VIDEO = _impl.SAVE_COMPOSITE_VIDEO
SAVE_EGO_VIDEO = _impl.SAVE_EGO_VIDEO
SAVE_TPP_VIDEO = _impl.SAVE_TPP_VIDEO
SAVE_FREE_VIDEO = _impl.SAVE_FREE_VIDEO

SHOW_EGO_WINDOW = _impl.SHOW_EGO_WINDOW
SHOW_TPP_WINDOW = _impl.SHOW_TPP_WINDOW


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
