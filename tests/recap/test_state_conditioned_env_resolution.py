from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from agent.run import state_conditioned_env_resolution


def _fake_gym(*env_ids: str) -> object:
    return SimpleNamespace(
        envs=SimpleNamespace(registry={env_id: object() for env_id in env_ids})
    )


def test_resolve_apple_to_plate_g1_accepts_exact_registered_id() -> None:
    gym_module = _fake_gym(
        state_conditioned_env_resolution.DEFAULT_APPLE_TO_PLATE_G1_REQUESTED_ENV_NAME
    )

    resolution = state_conditioned_env_resolution.resolve_apple_to_plate_g1_env_name(
        gym_module
    )

    assert resolution["logical_task"] == "apple_to_plate_g1"
    assert resolution["requested_env_name"] == (
        state_conditioned_env_resolution.DEFAULT_APPLE_TO_PLATE_G1_REQUESTED_ENV_NAME
    )
    assert resolution["resolved_env_name"] == (
        state_conditioned_env_resolution.DEFAULT_APPLE_TO_PLATE_G1_REQUESTED_ENV_NAME
    )
    assert resolution["alias_applied"] is False


def test_resolve_apple_to_plate_g1_uses_explicit_g1_sim_alias() -> None:
    alias_env_name = state_conditioned_env_resolution.APPLE_TO_PLATE_G1_ENV_ALIASES[
        state_conditioned_env_resolution.DEFAULT_APPLE_TO_PLATE_G1_REQUESTED_ENV_NAME
    ][0]
    gym_module = _fake_gym(alias_env_name)

    resolution = state_conditioned_env_resolution.resolve_apple_to_plate_g1_env_name(
        gym_module
    )

    assert resolution["resolved_env_name"] == alias_env_name
    assert resolution["alias_applied"] is True
    assert resolution["alias_candidates"] == [
        state_conditioned_env_resolution.DEFAULT_APPLE_TO_PLATE_G1_REQUESTED_ENV_NAME,
        alias_env_name,
    ]


def test_resolve_apple_to_plate_g1_reports_machine_readable_unavailable_error() -> None:
    close_match = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_g1_fixed_base_gear_wbc"
    gym_module = _fake_gym(
        close_match,
        "gr00tlocomanip_g1_sim/LMPnPBottleToPlateDC_g1_sim_gear_wbc",
    )

    with pytest.raises(
        state_conditioned_env_resolution.StateConditionedEnvResolutionError
    ) as exc_info:
        state_conditioned_env_resolution.resolve_apple_to_plate_g1_env_name(gym_module)

    payload = exc_info.value.to_machine_payload()
    assert payload["code"] == "state_conditioned_env_unavailable"
    assert payload["logical_task"] == "apple_to_plate_g1"
    assert payload["requested_env_name"] == (
        state_conditioned_env_resolution.DEFAULT_APPLE_TO_PLATE_G1_REQUESTED_ENV_NAME
    )
    assert close_match in payload["available_close_matches"]
    assert payload["registered_env_prefix"] == "gr00tlocomanip_g1_sim/"
