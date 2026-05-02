from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Callable, TypeAlias, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "work/openpi/serve/provenance.py"
SPEC = importlib.util.spec_from_file_location(
    "openpi_libero_server_provenance", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load server provenance module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["openpi_libero_server_provenance"] = MODULE
SPEC.loader.exec_module(MODULE)

BuildLiberoServerProvenance: TypeAlias = Callable[..., dict[str, object]]
ValidateLiberoServerProvenance: TypeAlias = Callable[[dict[str, object]], object]

build_libero_server_provenance_payload = cast(
    BuildLiberoServerProvenance,
    getattr(MODULE, "build_libero_server_provenance_payload"),
)
validate_libero_server_provenance = cast(
    ValidateLiberoServerProvenance,
    getattr(MODULE, "validate_libero_server_provenance"),
)


def _prompt_provenance() -> dict[str, str]:
    return {
        "prompt_route": "libero_prompt_text_only_v1",
        "conditioning_mode": "prompt_text_only",
    }


def _norm_provenance() -> dict[str, str]:
    return {
        "norm_stats_source": "dataset_meta_stats",
        "norm_stats_path": "agent/artifacts/openpi_libero_native/norm/stats.json",
        "asset_id": "pi05_libero_stock_baseline",
    }


def test_libero_server_provenance_happy_path_records_stock_constants_and_manifest() -> (
    None
):
    payload = build_libero_server_provenance_payload(
        prompt_provenance=_prompt_provenance(),
        norm_provenance=_norm_provenance(),
    )
    validated = validate_libero_server_provenance(payload)

    assert payload["schema_version"] == "openpi_libero_stock_checkpoint_v1"
    assert payload["model_anchor"] == "pi05_libero"
    assert payload["config_name"] == "pi05_libero"
    assert payload["action_horizon"] == 10
    assert payload["discrete_state_input"] is False
    assert payload["extra_delta_transform"] is False
    assert payload["replan_steps"] == 5
    assert payload["num_steps_wait"] == 10
    assert payload["suite"] == "libero_spatial"
    assert payload["task_ids"] == [0]
    assert payload["seed_manifest"] == [7]
    assert payload["num_trials_per_task"] == 1
    assert payload["evaluation_tier"] == "smoke"
    assert payload["action_semantics"] == {
        "extra_delta_transform": False,
        "replan_steps": 5,
        "num_steps_wait": 10,
    }

    assert getattr(validated, "model_anchor") == "pi05_libero"
    assert getattr(validated, "task_ids") == (0,)
    assert getattr(validated, "seed_manifest") == (7,)
    assert getattr(validated, "evaluation_tier") == "smoke"


def test_libero_server_provenance_accepts_comparison_manifest() -> None:
    payload = build_libero_server_provenance_payload(
        prompt_provenance=_prompt_provenance(),
        norm_provenance=_norm_provenance(),
        task_ids=(0, 1),
        seed_manifest=(7, 17),
        num_trials_per_task=2,
    )
    validated = validate_libero_server_provenance(payload)

    assert payload["evaluation_tier"] == "comparison"
    assert getattr(validated, "task_ids") == (0, 1)
    assert getattr(validated, "seed_manifest") == (7, 17)
    assert getattr(validated, "evaluation_tier") == "comparison"


def test_libero_server_provenance_rejects_legacy_g1_fields() -> None:
    payload = build_libero_server_provenance_payload(
        prompt_provenance=_prompt_provenance(),
        norm_provenance=_norm_provenance(),
    )
    payload["env_id"] = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"

    try:
        _ = validate_libero_server_provenance(payload)
    except ValueError as exc:
        assert "env_id" in str(exc)
        assert "LIBERO stock provenance" in str(exc)
    else:
        raise AssertionError("expected legacy G1 env id to fail")


def test_libero_server_provenance_rejects_old_phase1_horizon_constant() -> None:
    payload = build_libero_server_provenance_payload(
        prompt_provenance=_prompt_provenance(),
        norm_provenance=_norm_provenance(),
    )
    payload["action_horizon"] = 30

    try:
        _ = validate_libero_server_provenance(payload)
    except ValueError as exc:
        assert "action_horizon" in str(exc)
        assert "10" in str(exc)
    else:
        raise AssertionError("expected old Phase1 horizon constant to fail")
