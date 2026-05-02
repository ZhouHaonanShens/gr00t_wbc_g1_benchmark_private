from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Callable, TypeAlias, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "work/openpi/eval/protocol.py"
SPEC = importlib.util.spec_from_file_location(
    "openpi_libero_eval_protocol", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load eval protocol module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["openpi_libero_eval_protocol"] = MODULE
SPEC.loader.exec_module(MODULE)

BuildLiberoEvalProtocol: TypeAlias = Callable[..., object]
ValidateLiberoEvalProtocol: TypeAlias = Callable[[object], object]
BuildLiberoArtifactPaths: TypeAlias = Callable[..., dict[str, Path]]

build_libero_eval_protocol = cast(
    BuildLiberoEvalProtocol,
    getattr(MODULE, "build_libero_eval_protocol"),
)
validate_libero_eval_protocol = cast(
    ValidateLiberoEvalProtocol,
    getattr(MODULE, "validate_libero_eval_protocol"),
)
build_libero_eval_artifact_paths = cast(
    BuildLiberoArtifactPaths,
    getattr(MODULE, "build_libero_eval_artifact_paths"),
)


def test_libero_eval_protocol_happy_path_matches_task4_smoke_tier() -> None:
    protocol = build_libero_eval_protocol()
    validated = validate_libero_eval_protocol(protocol)
    paths = build_libero_eval_artifact_paths()

    assert getattr(validated, "schema_version") == "openpi_libero_eval_protocol_v1"
    assert getattr(validated, "suite") == "libero_spatial"
    assert getattr(validated, "task_ids") == (0,)
    assert getattr(validated, "seed_manifest") == (7,)
    assert getattr(validated, "num_trials_per_task") == 1
    assert getattr(validated, "evaluation_tier") == "smoke"
    assert getattr(validated, "action_horizon") == 10
    assert getattr(validated, "discrete_state_input") is False
    assert getattr(validated, "extra_delta_transform") is False
    assert getattr(validated, "replan_steps") == 5
    assert getattr(validated, "num_steps_wait") == 10
    assert (
        paths["summary_json"]
        .as_posix()
        .endswith("agent/artifacts/openpi_libero_native/summary.json")
    )


def test_libero_eval_protocol_accepts_frozen_comparison_tier() -> None:
    protocol = build_libero_eval_protocol(
        task_ids=(0, 1),
        seed_manifest=(7, 17),
        num_trials_per_task=2,
    )
    validated = validate_libero_eval_protocol(protocol)
    paths = build_libero_eval_artifact_paths(
        "comparison_run", topic="openpi_libero_eval"
    )

    assert getattr(validated, "evaluation_tier") == "comparison"
    assert getattr(validated, "task_ids") == (0, 1)
    assert getattr(validated, "seed_manifest") == (7, 17)
    assert (
        paths["summary_json"]
        .as_posix()
        .endswith("agent/artifacts/openpi_libero_eval/comparison_run/summary.json")
    )


def test_libero_eval_protocol_rejects_legacy_g1_env_id() -> None:
    payload = {
        "schema_version": "openpi_libero_eval_protocol_v1",
        "suite": "libero_spatial",
        "task_ids": [0],
        "seed_manifest": [7],
        "num_trials_per_task": 1,
        "action_horizon": 10,
        "discrete_state_input": False,
        "extra_delta_transform": False,
        "replan_steps": 5,
        "num_steps_wait": 10,
        "env_id": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
    }

    try:
        _ = validate_libero_eval_protocol(payload)
    except ValueError as exc:
        assert "env_id" in str(exc)
        assert "suite/task_ids/seed_manifest" in str(exc)
    else:
        raise AssertionError("expected legacy G1 env id to fail")


def test_libero_eval_protocol_rejects_policy_horizon_30() -> None:
    payload = {
        "schema_version": "openpi_libero_eval_protocol_v1",
        "suite": "libero_spatial",
        "task_ids": [0],
        "seed_manifest": [7],
        "num_trials_per_task": 1,
        "action_horizon": 10,
        "discrete_state_input": False,
        "extra_delta_transform": False,
        "replan_steps": 5,
        "num_steps_wait": 10,
        "policy_horizon": 30,
    }

    try:
        _ = validate_libero_eval_protocol(payload)
    except ValueError as exc:
        assert "policy_horizon" in str(exc)
        assert "suite/task_ids/seed_manifest" in str(exc)
    else:
        raise AssertionError("expected legacy Phase1 policy_horizon to fail")
