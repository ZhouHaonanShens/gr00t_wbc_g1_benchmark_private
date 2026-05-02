from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_eval_module():
    module_path = REPO_ROOT / "work" / "recap" / "scripts" / "3D_recap_eval.py"
    spec = importlib.util.spec_from_file_location("recap_3d_eval", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_env_registration_import_promotes_registry_from_zero(
    monkeypatch,
) -> None:
    module = _load_eval_module()
    alias_env_name = module.state_conditioned_env_resolution.APPLE_TO_PLATE_G1_ENV_ALIASES[
        module.state_conditioned_env_resolution.DEFAULT_APPLE_TO_PLATE_G1_REQUESTED_ENV_NAME
    ][0]
    registry: dict[str, object] = {}
    gym_module = types.SimpleNamespace(envs=types.SimpleNamespace(registry=registry))
    real_import_module = module.importlib.import_module
    import_calls: list[str] = []

    def _fake_import_module(name: str):
        if name == "gr00t_wbc.control.envs.robocasa.sync_env":
            import_calls.append(name)
            registry[alias_env_name] = object()
            return types.SimpleNamespace(__file__="/tmp/fake_sync_env.py")
        return real_import_module(name)

    monkeypatch.setattr(module.importlib, "import_module", _fake_import_module)

    info = module._ensure_explicit_g1_env_registration(gym_module)

    assert import_calls == ["gr00t_wbc.control.envs.robocasa.sync_env"]
    assert info["registered_env_count_before_import"] == 0
    assert info["registered_env_count_before_resolution"] == 1
    assert info["registered_env_ids_sample"] == [alias_env_name]


def test_explicit_env_registration_import_failure_is_not_silent(
    monkeypatch,
) -> None:
    module = _load_eval_module()
    gym_module = types.SimpleNamespace(envs=types.SimpleNamespace(registry={}))
    real_import_module = module.importlib.import_module

    def _fake_import_module(name: str):
        if name == "gr00t_wbc.control.envs.robocasa.sync_env":
            raise ModuleNotFoundError("simulated sync_env import failure")
        return real_import_module(name)

    monkeypatch.setattr(module.importlib, "import_module", _fake_import_module)

    with pytest.raises(RuntimeError, match="explicit G1 env registration failed"):
        module._ensure_explicit_g1_env_registration(gym_module)


def test_g1_default_execution_steps_stay_at_20_not_policy_horizon_30() -> None:
    module = _load_eval_module()
    requested_env_name = module.state_conditioned_env_resolution.DEFAULT_APPLE_TO_PLATE_G1_REQUESTED_ENV_NAME
    alias_env_name = (
        module.state_conditioned_env_resolution.APPLE_TO_PLATE_G1_ENV_ALIASES[
            requested_env_name
        ][0]
    )

    assert module.DEFAULT_G1_EXECUTION_N_ACTION_STEPS == 20
    assert module._default_n_action_steps_for_env(requested_env_name) == 20
    assert module._default_n_action_steps_for_env(alias_env_name) == 20
    assert module._default_n_action_steps_for_env("some_other_env") is None


def test_public_episode_result_promotes_authoritative_fields() -> None:
    module = _load_eval_module()

    result = module._build_public_episode_result(
        {
            "episode_index": 3,
            "seed": 20260423,
            "success": True,
            "episode_elapsed_seconds": 12.5,
            "done": True,
            "terminated": True,
            "truncated": False,
            "outer_steps": 72,
            "failure_reason": None,
            "final_snapshot": {"ignored": True},
        }
    )

    assert result == {
        "episode_index": 3,
        "seed": 20260423,
        "success": True,
        "episode_elapsed_seconds": 12.5,
        "done": True,
        "terminated": True,
        "truncated": False,
        "outer_steps": 72,
        "failure_reason": None,
    }
