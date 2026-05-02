"""Compatibility facade for the OpenPI LIBERO eval protocol.

The canonical implementation lives in ``work/openpi/eval/protocols/environment.py``.
This facade is intentionally loaded by file path in legacy tests/contracts, so it
loads the canonical module by path as well.  That avoids importing
``work.openpi.eval`` package side effects that pull optional runtime dependencies
unrelated to the protocol schema.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

_ENVIRONMENT_MODULE_NAME = "openpi_libero_eval_protocol_environment"
_ENVIRONMENT_MODULE_PATH = Path(__file__).resolve().parent / "protocols" / "environment.py"


def _load_environment_module() -> ModuleType:
    cached = sys.modules.get(_ENVIRONMENT_MODULE_NAME)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        _ENVIRONMENT_MODULE_NAME,
        _ENVIRONMENT_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load OpenPI eval protocol from {_ENVIRONMENT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ENVIRONMENT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_environment = _load_environment_module()

DEFAULT_ARTIFACT_TOPIC = _environment.DEFAULT_ARTIFACT_TOPIC
EXPECTED_ACTION_HORIZON = _environment.EXPECTED_ACTION_HORIZON
EXPECTED_DISCRETE_STATE_INPUT = _environment.EXPECTED_DISCRETE_STATE_INPUT
EXPECTED_EXTRA_DELTA_TRANSFORM = _environment.EXPECTED_EXTRA_DELTA_TRANSFORM
EXPECTED_NUM_STEPS_WAIT = _environment.EXPECTED_NUM_STEPS_WAIT
EXPECTED_REPLAN_STEPS = _environment.EXPECTED_REPLAN_STEPS
EXPECTED_SCHEMA_VERSION = _environment.EXPECTED_SCHEMA_VERSION
EXPECTED_SUITE = _environment.EXPECTED_SUITE
LiberoEvalProtocol = _environment.LiberoEvalProtocol
build_libero_eval_artifact_paths = _environment.build_libero_eval_artifact_paths
build_libero_eval_protocol = _environment.build_libero_eval_protocol
validate_libero_eval_protocol = _environment.validate_libero_eval_protocol

__all__ = [
    "DEFAULT_ARTIFACT_TOPIC",
    "EXPECTED_ACTION_HORIZON",
    "EXPECTED_DISCRETE_STATE_INPUT",
    "EXPECTED_EXTRA_DELTA_TRANSFORM",
    "EXPECTED_NUM_STEPS_WAIT",
    "EXPECTED_REPLAN_STEPS",
    "EXPECTED_SCHEMA_VERSION",
    "EXPECTED_SUITE",
    "LiberoEvalProtocol",
    "build_libero_eval_artifact_paths",
    "build_libero_eval_protocol",
    "validate_libero_eval_protocol",
]
