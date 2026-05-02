from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import cast

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.critic_bridge import (
    AUDIT_CONCLUSION_ADAPTER_REQUIRED,
    BRIDGE_SCHEMA_VERSION,
    CriticBridgeContractError,
    build_bridge_contract,
    build_provenance_handle,
    validate_bridge_request,
    validate_bridge_response,
)


def test_bridge_contract_freezes_adapter_required_boundary() -> None:
    contract = cast(Mapping[str, object], build_bridge_contract())
    repo_boundary = cast(Mapping[str, object], contract["repo_presence_vs_active_path"])
    input_contract = cast(Mapping[str, object], contract["input_contract"])
    task_metadata = cast(Mapping[str, object], input_contract["task_metadata"])
    semantics = cast(Mapping[str, object], task_metadata["semantics"])

    assert contract["schema_version"] == BRIDGE_SCHEMA_VERSION
    assert contract["audit_conclusion"] == AUDIT_CONCLUSION_ADAPTER_REQUIRED
    assert repo_boundary["critic_repo_surface_present"] is True
    assert repo_boundary["openpi_train_path_actively_consumes_critic"] is False
    assert repo_boundary["openpi_rollout_path_actively_consumes_critic"] is False
    assert "adapter is still required" in str(repo_boundary["interpretation"])
    assert task_metadata["required_keys"] == ["step_index", "episode_length"]
    assert (
        semantics["timestep_norm"]
        == "optional cached value; if provided it must agree with step_index / max(episode_length - 1, 1)"
    )


def test_validate_bridge_request_requires_non_vague_payload_keys() -> None:
    with pytest.raises(
        CriticBridgeContractError, match="observation.image is required"
    ):
        _ = validate_bridge_request(
            {
                "observation": {"state": [0.0, 1.0]},
                "language": {"prompt_raw": "put the bowl on the plate"},
                "task_metadata": {"step_index": 0, "episode_length": 3},
            }
        )

    with pytest.raises(
        CriticBridgeContractError, match="must agree with step_index/episode_length"
    ):
        _ = validate_bridge_request(
            {
                "observation": {"image": "frame-0", "state": [0.0, 1.0]},
                "language": {"prompt_raw": "put the bowl on the plate"},
                "task_metadata": {
                    "step_index": 1,
                    "episode_length": 3,
                    "timestep_norm": 0.1,
                },
            }
        )


def test_validate_bridge_response_requires_distribution_and_provenance_handle() -> None:
    handle = build_provenance_handle(
        critic_dir=REPO_ROOT / "agent" / "artifacts" / "critics" / "fixture"
    )
    response = validate_bridge_response(
        {
            "value_distribution": {
                "bin_centers": [-1.0, 0.0, 1.0],
                "bin_logits": [0.1, 0.3, 0.6],
                "bin_probs": [0.2, 0.3, 0.5],
            },
            "decoded_value": {
                "value_V_raw": 0.3,
                "value_scale": "raw_return",
            },
            "provenance_handle": handle,
        }
    )

    decoded_value = cast(Mapping[str, object], response["decoded_value"])
    provenance_handle = cast(Mapping[str, object], response["provenance_handle"])
    required_files = cast(Mapping[str, object], provenance_handle["required_files"])

    assert decoded_value["value_scale"] == "raw_return"
    assert provenance_handle["frame_policy"] == "current_step_index"
    assert str(required_files["provenance"]).endswith("provenance.json")

    with pytest.raises(CriticBridgeContractError, match="must sum to 1.0"):
        _ = validate_bridge_response(
            {
                "value_distribution": {
                    "bin_centers": [-1.0, 0.0, 1.0],
                    "bin_logits": [0.1, 0.3, 0.6],
                    "bin_probs": [0.2, 0.3, 0.4],
                },
                "decoded_value": {
                    "value_V_raw": 0.2,
                    "value_scale": "raw_return",
                },
                "provenance_handle": handle,
            }
        )
