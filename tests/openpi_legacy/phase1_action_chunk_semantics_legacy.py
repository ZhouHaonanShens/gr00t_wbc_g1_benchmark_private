from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Callable, TypeAlias, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "work/openpi/eval/protocol.py"
SPEC = importlib.util.spec_from_file_location("openpi_eval_protocol", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load eval protocol module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["openpi_eval_protocol"] = MODULE
SPEC.loader.exec_module(MODULE)

BuildEvalProtocol: TypeAlias = Callable[..., object]
ValidateEvalProtocol: TypeAlias = Callable[[object], object]
BuildArtifactPaths: TypeAlias = Callable[[str], dict[str, Path]]

build_phase1_eval_protocol = cast(
    BuildEvalProtocol, getattr(MODULE, "build_phase1_eval_protocol")
)
validate_phase1_eval_protocol = cast(
    ValidateEvalProtocol, getattr(MODULE, "validate_phase1_eval_protocol")
)
build_phase1_eval_artifact_paths = cast(
    BuildArtifactPaths, getattr(MODULE, "build_phase1_eval_artifact_paths")
)


def test_phase1_eval_protocol_happy_path() -> None:
    protocol = build_phase1_eval_protocol()
    validated = validate_phase1_eval_protocol(protocol)
    paths = build_phase1_eval_artifact_paths("demo_run")

    assert (
        getattr(validated, "env_id")
        == "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
    )
    assert getattr(validated, "logical_task") == "apple_to_plate_g1"
    assert getattr(validated, "n_episodes") == 10
    assert getattr(validated, "policy_horizon") == 30
    assert getattr(validated, "executed_action_steps") == 20
    assert (
        paths["summary_json"]
        .as_posix()
        .endswith("agent/artifacts/openpi_phase1/demo_run/summary.json")
    )


def test_phase1_eval_protocol_rejects_horizon_mismatch() -> None:
    try:
        _ = build_phase1_eval_protocol(policy_horizon=20)
    except ValueError as exc:
        assert "policy_horizon" in str(exc)
        assert "30" in str(exc)
    else:
        raise AssertionError("expected policy_horizon mismatch to fail")


def test_phase1_eval_protocol_rejects_executed_action_step_mismatch() -> None:
    try:
        _ = build_phase1_eval_protocol(executed_action_steps=30)
    except ValueError as exc:
        assert "executed_action_steps" in str(exc)
        assert "20" in str(exc)
    else:
        raise AssertionError("expected executed_action_steps mismatch to fail")


def test_phase1_eval_protocol_rejects_seed_manifest_drift() -> None:
    try:
        _ = build_phase1_eval_protocol(seed_values=(0,))
    except ValueError as exc:
        assert "seed manifest" in str(exc)
        assert "20000" in str(exc)
    else:
        raise AssertionError("expected seed manifest drift to fail")
