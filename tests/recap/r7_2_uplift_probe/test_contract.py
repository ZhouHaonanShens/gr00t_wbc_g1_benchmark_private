from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import get_args

import pytest

from work.recap.r7_1_recipe_plumbing.flags import RecipeFlags
from work.recap.r7_2_uplift_probe.contract import (
    FINAL_VERDICTS,
    FinalVerdict,
    R7BudgetExceeded,
    R7UpliftError,
    StepwiseCounterfactual,
    TrialReport,
    TrialRequest,
    preset_to_recipe_flags,
)

TOKEN = "a" * 64


def _request(**overrides) -> TrialRequest:
    values = dict(
        trial_id="trial-1",
        base_ckpt_abs_path="/abs/ckpt",
        recipe_flags=preset_to_recipe_flags("full_C1_C2_C5"),
        recipe_preset="full_C1_C2_C5",
        output_root="out",
        leader_approval_token=TOKEN,
        gpu_id=1,
    )
    values.update(overrides)
    return TrialRequest(**values)


def test_final_verdict_has_exact_four_values() -> None:
    assert get_args(FinalVerdict) == (
        "INDICATOR_SENSITIVE_AT_STEP_N",
        "INDICATOR_INVARIANT_AT_MAX_STEPS",
        "TRAINING_FAILED",
        "BUDGET_EXCEEDED",
    )
    assert FINAL_VERDICTS == get_args(FinalVerdict)


def test_recipe_presets_map_to_r7_1_recipe_flags() -> None:
    full = preset_to_recipe_flags("full_C1_C2_C5")
    no_dropout = preset_to_recipe_flags("subset_C1_C5_no_dropout")
    c1_only = preset_to_recipe_flags("subset_C1_only")
    assert isinstance(full, RecipeFlags)
    assert full.indicator_dropout_p == 0.15 and full.dual_loss_uses_carrier_text is True
    assert no_dropout.indicator_dropout_p == 0.0 and no_dropout.carrier_text_field == "carrier_text_v1"
    assert c1_only.dual_loss_uses_carrier_text is False


def test_trial_request_is_frozen_and_has_explicit_fields() -> None:
    request = _request()
    assert [field.name for field in fields(request)] == [
        "trial_id",
        "base_ckpt_abs_path",
        "recipe_flags",
        "recipe_preset",
        "output_root",
        "leader_approval_token",
        "gpu_id",
        "max_steps",
        "probe_interval",
        "lora_rank",
        "lora_alpha",
        "lora_target_modules_top_k_layers",
        "budget_gpu_minutes",
        "seed",
    ]
    with pytest.raises(FrozenInstanceError):
        request.gpu_id = 2  # type: ignore[misc]


@pytest.mark.parametrize("kwargs", [
    {"trial_id": "trial-4"},
    {"leader_approval_token": "bad"},
    {"gpu_id": 0},
    {"max_steps": 2001},
    {"probe_interval": 100},
    {"lora_rank": 17},
])
def test_trial_request_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(R7UpliftError):
        _request(**kwargs)


def test_budget_exceeded_is_specific_error() -> None:
    with pytest.raises(R7BudgetExceeded):
        _request(budget_gpu_minutes=241)


def test_stepwise_counterfactual_validates_literal_and_finite_diff() -> None:
    item = StepwiseCounterfactual(200, "adapter", "cf.json", "INDICATOR_INVARIANT", True, 0.0)
    assert item.step == 200
    with pytest.raises(R7UpliftError):
        StepwiseCounterfactual(201, "adapter", "cf.json", "INDICATOR_INVARIANT", True, 0.0)
    with pytest.raises(R7UpliftError):
        StepwiseCounterfactual(200, "adapter", "cf.json", "OTHER", True, 0.0)  # type: ignore[arg-type]


def test_trial_report_validates_final_verdict() -> None:
    request = _request()
    report = TrialReport(request, "TRAINING_FAILED", 0, (), 0.0, None)
    assert report.final_verdict == "TRAINING_FAILED"
    with pytest.raises(R7UpliftError):
        TrialReport(request, "OTHER", 0, (), 0.0, None)  # type: ignore[arg-type]
