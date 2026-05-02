from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable, cast

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import text_indicator
from work.recap.scripts import state_conditioned_build_training_set
from work.recap.scripts import state_conditioned_contract_gate


def _base_row(*, prompt_raw: str, indicator_I: int) -> dict[str, object]:
    phase = "TRANSPORT"
    mode = "RECOVERY"
    return {
        "sample_id": "sample_001",
        "source_bucket": "canonical_bucket_A",
        "source_kind": "sidecar_transition",
        "source_episode_id": "episode_001",
        "source_t": 0,
        "source_snapshot_id": None,
        "pseudodemo.source_snapshot_family": None,
        "pseudodemo.source_bucket_key": None,
        "source_anchor_episode_id": "episode_001",
        "source_sample_key": "episode_001:0",
        "reset_boundary": "no_cross_episode",
        "label_data.domain": "training_flat_artifact_only",
        "label_data.version": "state_conditioned_training_flat_artifacts_v2",
        "label_data.m2_backfill_source": "source_dataset_m2_labels",
        "label_data.m2_backfill_version": "recap_m2_label_fields_v1",
        "pseudodemo.teacher_policy_id": None,
        "pseudodemo.teacher_target": None,
        "pseudodemo.teacher_target_truthfulness": None,
        "pseudodemo.label_kind": None,
        "pseudodemo.dataset_version": None,
        "budget_group": "shared_state_conditioned_equal_data_training_budget_v1",
        "repeat_index": 0,
        "recovery_oversample_factor": 3,
        "return_G": 1.0,
        "value_V": 0.5,
        "advantage_A": 0.5,
        "epsilon_l": 0.1,
        "indicator_I": indicator_I,
        "carrier_text_v1": state_conditioned_build_training_set.build_carrier_text_v1(
            prompt_raw=prompt_raw,
            indicator_value=indicator_I,
        ),
        "history_k": 8,
        "history_stride": 1,
        "history_valid_mask": [True] * 8,
        "history_t_std_indices": list(range(8)),
        "history_t_raw_indices": list(range(8)),
        "history_timestamp_s": [float(index) * 0.05 for index in range(8)],
        "deployable.previous_action_history": [[float(index)] for index in range(8)],
        "deployable.proprio_history": [[float(index), 1.0] for index in range(8)],
        "deployable.short_visual_history_refs": [
            f"video://episode_001/{index}" for index in range(8)
        ],
        "canonical_phase": phase,
        "canonical_mode": mode,
        "canonical_policy_condition_text": (
            state_conditioned_contract_gate.build_policy_condition_text(phase, mode)
        ),
    }


def test_build_view_rows_emit_carrier_text_v1_as_shared_authority() -> None:
    prompt_raw = "pick up the apple and place it on the plate"
    base_row = _base_row(prompt_raw=prompt_raw, indicator_I=1)

    c0_row = state_conditioned_build_training_set._build_view_rows(
        [base_row],
        training_view=state_conditioned_build_training_set.VIEW_C0,
    )[0]
    c1_row = state_conditioned_build_training_set._build_view_rows(
        [base_row],
        training_view=state_conditioned_build_training_set.VIEW_C1,
    )[0]
    expected_carrier = text_indicator.build_canonical_text_indicator(
        prompt_raw,
        text_indicator.TEXT_INDICATOR_POSITIVE,
    )

    assert c0_row["carrier_text_v1"] == expected_carrier
    assert c1_row["carrier_text_v1"] == expected_carrier
    assert c0_row["policy_condition_text"] == (
        state_conditioned_build_training_set.build_null_policy_condition_text()
    )
    assert c1_row["policy_condition_text"] == (
        state_conditioned_contract_gate.build_policy_condition_text(
            "TRANSPORT",
            "RECOVERY",
        )
    )
    assert list(c1_row).index("carrier_text_v1") < list(c1_row).index(
        "policy_condition.phase"
    )
    assert (
        "carrier_text_v1"
        not in state_conditioned_build_training_set.CONDITIONING_FIELD_NAMES
    )


def test_contract_gate_freezes_carrier_text_v1_as_mainline_authority() -> None:
    freeze = state_conditioned_contract_gate.build_state_conditioned_freeze()
    mainline_training_text = cast(dict[str, Any], freeze["mainline_training_text"])
    candidate: dict[str, Any] = dict(
        state_conditioned_contract_gate.build_reference_contract_example()
    )
    validated = state_conditioned_contract_gate.validate_contract_candidate(candidate)
    validated_mainline = cast(dict[str, Any], validated["mainline_training_text"])
    validated_policy_text = cast(dict[str, Any], validated["policy_text"])

    assert mainline_training_text["field"] == "carrier_text_v1"
    assert mainline_training_text["schema_version"] == (
        text_indicator.RECAP_TEXT_INDICATOR_SCHEMA_VERSION
    )
    assert mainline_training_text["policy_metadata_fields"] == [
        "policy_condition.phase",
        "policy_condition.mode",
        "policy_condition_text",
    ]
    assert validated_mainline["carrier_field"] == "carrier_text_v1"
    assert (
        validated_policy_text["policy_condition_text"]
        == candidate["policy_condition_text"]
    )


def _mutate_carrier_with_policy_text(candidate: dict[str, Any]) -> None:
    candidate["carrier_text_v1"] = str(candidate["policy_condition_text"])


def _mutate_prompt_conditioned_override(candidate: dict[str, Any]) -> None:
    candidate["prompt_conditioned"] = (
        "legacy conditioned text that must not override authority"
    )


def _mutate_advantage_passthrough(candidate: dict[str, Any]) -> None:
    candidate["advantage_input"] = 0.25


def _mutate_dual_task_text(candidate: dict[str, Any]) -> None:
    candidate["dual_task_text"] = True


@pytest.mark.parametrize(
    ("mutator", "error_match"),
    [
        (
            _mutate_carrier_with_policy_text,
            "canonical prompt_raw \\+ indicator_I text-indicator carrier",
        ),
        (
            _mutate_prompt_conditioned_override,
            "prompt_conditioned must not override carrier_text_v1 authority",
        ),
        (
            _mutate_advantage_passthrough,
            "advantage_input must remain out of the mainline carrier_text_v1 authority channel",
        ),
        (
            _mutate_dual_task_text,
            "dual_task_text authority must remain disabled",
        ),
    ],
)
def test_contract_gate_rejects_legacy_or_mixed_authority_leakage(
    mutator: Callable[[dict[str, Any]], None],
    error_match: str,
) -> None:
    candidate: dict[str, Any] = dict(
        state_conditioned_contract_gate.build_reference_contract_example()
    )
    mutator(candidate)

    with pytest.raises(ValueError, match=error_match):
        _ = state_conditioned_contract_gate.validate_contract_candidate(candidate)
