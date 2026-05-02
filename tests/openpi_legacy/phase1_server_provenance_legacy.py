from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Callable, TypeAlias, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "work/openpi/serve/provenance.py"
SPEC = importlib.util.spec_from_file_location("openpi_server_provenance", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load server provenance module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["openpi_server_provenance"] = MODULE
SPEC.loader.exec_module(MODULE)

BuildHealthPayload: TypeAlias = Callable[..., dict[str, object]]
ValidateHealthPayload: TypeAlias = Callable[[dict[str, object]], object]

build_phase1_health_payload = cast(
    BuildHealthPayload, getattr(MODULE, "build_phase1_health_payload")
)
validate_phase1_health_payload = cast(
    ValidateHealthPayload, getattr(MODULE, "validate_phase1_health_payload")
)


def _prompt_provenance() -> dict[str, str]:
    return {
        "prompt_route": "recap_conditioned_prompt_token_v1",
        "conditioning_mode": "prompt_text_only",
    }


def _norm_provenance() -> dict[str, str]:
    return {
        "norm_stats_source": "dataset_meta_stats",
        "norm_stats_path": "agent/artifacts/lerobot_datasets/demo/meta/stats.json",
        "asset_id": "task_local_recomputed",
    }


def test_phase1_server_provenance_happy_path() -> None:
    payload = build_phase1_health_payload(
        prompt_provenance=_prompt_provenance(),
        norm_provenance=_norm_provenance(),
    )
    validated = validate_phase1_health_payload(payload)

    assert payload["model_family"] == "openpi"
    assert payload["model_anchor"] == "pi05_droid"
    assert payload["policy_horizon"] == 30
    assert payload["executed_action_steps"] == 20
    assert getattr(validated, "prompt_route") == "recap_conditioned_prompt_token_v1"
    assert getattr(validated, "norm_stats_source") == "dataset_meta_stats"


def test_phase1_server_provenance_rejects_baseline_fallback() -> None:
    payload = build_phase1_health_payload(
        prompt_provenance=_prompt_provenance(),
        norm_provenance=_norm_provenance(),
    )
    payload["model_family"] = "gr00t"

    try:
        _ = validate_phase1_health_payload(payload)
    except ValueError as exc:
        assert "model_family" in str(exc)
        assert "openpi" in str(exc)
    else:
        raise AssertionError("expected baseline fallback to fail")


def test_phase1_server_provenance_rejects_horizon_mismatch() -> None:
    payload = build_phase1_health_payload(
        prompt_provenance=_prompt_provenance(),
        norm_provenance=_norm_provenance(),
    )
    payload["policy_horizon"] = 20

    try:
        _ = validate_phase1_health_payload(payload)
    except ValueError as exc:
        assert "policy_horizon" in str(exc)
        assert "30" in str(exc)
    else:
        raise AssertionError("expected horizon mismatch to fail")


def test_phase1_server_provenance_requires_norm_fields() -> None:
    payload = build_phase1_health_payload(
        prompt_provenance=_prompt_provenance(),
        norm_provenance=_norm_provenance(),
    )
    payload["asset_id"] = ""

    try:
        _ = validate_phase1_health_payload(payload)
    except ValueError as exc:
        assert "asset_id" in str(exc)
    else:
        raise AssertionError("expected missing norm provenance fields to fail")
