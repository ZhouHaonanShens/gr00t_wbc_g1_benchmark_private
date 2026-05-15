from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import get_args

import pytest

from work.recap.r7_recipe_diff import contract


EXPECTED_COMPONENTS = (
    ("C1_dual_loss", "Dual conditional/unconditional loss objective", "OpenPI fidelity Q6", "enable_dual_loss"),
    ("C2_indicator_dropout", "Stochastic indicator omission/dropout at training time", "OpenPI fidelity Q7", "indicator_dropout_p"),
    ("C3_learned_value", "Learned value-function / critic in training loop", "OpenPI fidelity Q2", "enable_learned_value_head"),
    ("C4_advantage_embedding_active", "advantage_input numeric sidecar consumed by action head", "OpenPI fidelity Q3", "action_head_advantage_input"),
    ("C5_carrier_text_v1_grad_path", "carrier_text_v1 tokens get gradient signal through dual-loss path", "OpenPI fidelity Q8 / GR00T R6.1 verdict", "dual_loss_uses_carrier_text"),
)
EXPECTED_DIFF_ACTIONS = {
    "C1_dual_loss": "ADD_LOSS_TERM",
    "C2_indicator_dropout": "ADD_DATASET_AUG",
    "C3_learned_value": "ADD_LOSS_TERM",
    "C4_advantage_embedding_active": "ENABLE_FLAG",
    "C5_carrier_text_v1_grad_path": "ADD_CLI_ARG",
}


def test_literal_closures_are_exact() -> None:
    assert get_args(contract.ComponentState) == ("IMPLEMENTED", "PARTIAL", "ABSENT")
    assert get_args(contract.DiffAction) == (
        "ENABLE_FLAG",
        "ADD_CLI_ARG",
        "ADD_LOSS_TERM",
        "ADD_DATASET_AUG",
        "NO_OP",
    )


def test_fidelity_components_are_frozen_ssot() -> None:
    observed = tuple(
        (
            item.component_id,
            item.title,
            item.paper_section_cite,
            item.training_arg_name,
        )
        for item in contract.FIDELITY_COMPONENTS
    )
    assert observed == EXPECTED_COMPONENTS


def test_dataclass_fields_are_exact_and_frozen() -> None:
    assert tuple(field.name for field in fields(contract.FidelityComponent)) == (
        "component_id",
        "title",
        "paper_section_cite",
        "training_arg_name",
    )
    assert tuple(field.name for field in fields(contract.RecipeDelta)) == (
        "component",
        "current_state",
        "paper_prescribed_state",
        "diff_action",
        "cli_arg_addition",
        "config_path_diff",
        "evidence_files",
        "rationale",
    )
    component = contract.FIDELITY_COMPONENTS[0]
    with pytest.raises(FrozenInstanceError):
        component.title = "mutated"  # type: ignore[misc]


def test_expected_diff_actions_are_representable() -> None:
    allowed = set(get_args(contract.DiffAction))
    assert set(EXPECTED_DIFF_ACTIONS.values()) <= allowed
