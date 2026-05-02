# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportMissingParameterType=false

from __future__ import annotations

import dataclasses
import difflib

from . import _upstream_openpi_recap_training_config as _upstream_config
from ._upstream_openpi_recap_training_config import *  # noqa: F401,F403
import tyro

from openpi.recap_overlay.config import (
    PI05_LIBERO_RECAP_CONFIG_NAME,
    build_recap_policy_metadata,
)


PI05_LIBERO_RECAP_POLICY_METADATA = build_recap_policy_metadata()


def _build_pi05_libero_recap_config() -> TrainConfig:
    base_config = _upstream_config.get_config("pi05_libero")
    base_policy_metadata = dict(base_config.policy_metadata or {})
    base_policy_metadata.update(PI05_LIBERO_RECAP_POLICY_METADATA)
    return dataclasses.replace(
        base_config,
        name=PI05_LIBERO_RECAP_CONFIG_NAME,
        policy_metadata=base_policy_metadata,
    )


PI05_LIBERO_RECAP_CONFIG = _build_pi05_libero_recap_config()
_CONFIGS = list(_upstream_config._CONFIGS) + [PI05_LIBERO_RECAP_CONFIG]
if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique after recap overlay injection.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli(
        {key: (key, value) for key, value in _CONFIGS_DICT.items()}
    )


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
