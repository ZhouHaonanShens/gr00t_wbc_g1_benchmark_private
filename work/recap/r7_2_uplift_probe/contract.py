from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Literal, get_args

from work.recap.r7_1_recipe_plumbing.flags import RecipeFlags

RecipePreset = Literal["full_C1_C2_C5", "subset_C1_C5_no_dropout", "subset_C1_only"]
FinalVerdict = Literal["INDICATOR_SENSITIVE_AT_STEP_N", "INDICATOR_INVARIANT_AT_MAX_STEPS", "TRAINING_FAILED", "BUDGET_EXCEEDED"]
StepCounterfactualVerdict = Literal["INDICATOR_SENSITIVE", "INDICATOR_INVARIANT"]
TRIAL_IDS = ("trial-1", "trial-2", "trial-3")
RECIPE_PRESETS = get_args(RecipePreset)
FINAL_VERDICTS = get_args(FinalVerdict)
_HEX_64 = re.compile(r"^[0-9a-fA-F]{64}$")


class R7UpliftError(RuntimeError):
    pass
class R7BudgetExceeded(R7UpliftError):
    pass
class R7AdapterTooLargeError(R7UpliftError):
    pass
class R7TrainingFailedError(R7UpliftError):
    pass

@dataclass(frozen=True)
class TrialRequest:
    trial_id: str
    base_ckpt_abs_path: str
    recipe_flags: RecipeFlags
    recipe_preset: RecipePreset
    output_root: str
    leader_approval_token: str
    gpu_id: int
    max_steps: int = 2000
    probe_interval: int = 200
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules_top_k_layers: int = 4
    budget_gpu_minutes: int = 240
    seed: int = 20000

    def __post_init__(self) -> None:
        if self.trial_id not in TRIAL_IDS or self.recipe_preset not in RECIPE_PRESETS:
            raise R7UpliftError("invalid trial_id or recipe_preset")
        if not isinstance(self.recipe_flags, RecipeFlags) or not _is_hex_64(self.leader_approval_token):
            raise R7UpliftError("recipe_flags must be RecipeFlags and token must be 64 hex")
        if int(self.gpu_id) not in {1, 2} or int(self.max_steps) <= 0 or int(self.max_steps) > 2000:
            raise R7UpliftError("gpu_id/max_steps out of range")
        if int(self.probe_interval) != 200 or int(self.lora_rank) <= 0 or int(self.lora_rank) > 16:
            raise R7UpliftError("probe_interval must be 200 and lora_rank in [1,16]")
        if int(self.lora_alpha) <= 0 or int(self.lora_target_modules_top_k_layers) != 4:
            raise R7UpliftError("lora_alpha must be positive and top_k_layers must be 4")
        if int(self.budget_gpu_minutes) <= 0 or int(self.budget_gpu_minutes) > 240:
            raise R7BudgetExceeded("budget_gpu_minutes must be in [1,240]")

@dataclass(frozen=True)
class StepwiseCounterfactual:
    step: int
    adapter_path: str
    counterfactual_path: str
    counterfactual_verdict: StepCounterfactualVerdict
    condition_sha_equal: bool
    first_5_actions_l2_diff_max: float

    def __post_init__(self) -> None:
        if int(self.step) <= 0 or int(self.step) % 200 != 0:
            raise R7UpliftError("step must be a positive multiple of 200")
        if self.counterfactual_verdict not in get_args(StepCounterfactualVerdict):
            raise R7UpliftError("invalid counterfactual verdict")
        if not math.isfinite(float(self.first_5_actions_l2_diff_max)):
            raise R7UpliftError("diff max must be finite")

@dataclass(frozen=True)
class TrialReport:
    request: TrialRequest
    final_verdict: FinalVerdict
    final_step: int
    counterfactual_evolution: tuple[StepwiseCounterfactual, ...]
    total_gpu_seconds: float
    training_loss_final: float | None

    def __post_init__(self) -> None:
        if self.final_verdict not in FINAL_VERDICTS:
            raise R7UpliftError("invalid final verdict")
        if int(self.final_step) < 0 or int(self.final_step) > int(self.request.max_steps):
            raise R7UpliftError("final_step out of bounds")
        if float(self.total_gpu_seconds) < 0.0:
            raise R7UpliftError("gpu seconds must be non-negative")

def preset_to_recipe_flags(recipe_preset: RecipePreset) -> RecipeFlags:
    if recipe_preset == "full_C1_C2_C5":
        return RecipeFlags(True, 0.5, 0.15, 0, True, "carrier_text_v1")
    if recipe_preset == "subset_C1_C5_no_dropout":
        return RecipeFlags(True, 0.5, 0.0, 0, True, "carrier_text_v1")
    if recipe_preset == "subset_C1_only":
        return RecipeFlags(True, 0.5, 0.0, 0, False, "prompt_raw")
    raise R7UpliftError(f"unsupported recipe_preset: {recipe_preset!r}")

def planned_steps(max_steps: int = 2000, probe_interval: int = 200) -> tuple[int, ...]:
    interval = int(probe_interval)
    upper_bound = int(max_steps)
    if interval != 200 or upper_bound > 2000:
        raise R7UpliftError("invalid planned step bounds")
    steps = tuple(range(200, 2001, interval))
    return steps

def _is_hex_64(value: str) -> bool:
    if not isinstance(value, str):
        return False
    match = _HEX_64.fullmatch(value)
    is_valid = bool(match)
    return is_valid
