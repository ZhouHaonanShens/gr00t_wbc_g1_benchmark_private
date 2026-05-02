from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.data_transforms import (  # noqa: E402
    RepairedStagePromptTransform,
    build_stage_prompt_bundle,
)
from work.openpi.recap.train_config import (  # noqa: E402
    DEFAULT_REPAIRED_STAGE_SAVE_INTERVAL,
    DEFAULT_REPAIRED_STAGE_NUM_TRAIN_STEPS,
    RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL,
    RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS,
    REPAIRED_STAGE_VALUES,
    resolve_repaired_stage_config,
)


def test_repaired_stage_names_are_exactly_frozen() -> None:
    assert REPAIRED_STAGE_VALUES == (
        "sft_fixed_positive",
        "recap_informative",
        "shuffled_indicator",
        "omit_control",
    )


def test_stage_prompt_semantics_match_task6_authority() -> None:
    fixed_config = resolve_repaired_stage_config("sft_fixed_positive")
    omit_config = resolve_repaired_stage_config("omit_control")
    informative_config = resolve_repaired_stage_config("recap_informative")

    fixed_bundle = build_stage_prompt_bundle(
        stage_config=fixed_config,
        label_row={
            "prompt_raw": "put the bowl on the plate",
            "recap_m2.indicator_I": 0,
        },
    )
    informative_positive = build_stage_prompt_bundle(
        stage_config=informative_config,
        label_row={
            "prompt_raw": "put the bowl on the plate",
            "recap_m2.indicator_I": 1,
        },
    )
    informative_negative = build_stage_prompt_bundle(
        stage_config=informative_config,
        label_row={
            "prompt_raw": "put the bowl on the plate",
            "recap_m2.indicator_I": 0,
        },
    )
    omit_bundle = build_stage_prompt_bundle(
        stage_config=omit_config,
        label_row={
            "prompt_raw": "put the bowl on the plate",
            "recap_m2.indicator_I": 1,
        },
    )

    assert fixed_bundle.prompt_text.endswith("Advantage: positive")
    assert informative_positive.prompt_text.endswith("Advantage: positive")
    assert informative_negative.prompt_text.endswith("Advantage: negative")
    assert omit_bundle.prompt_text == "put the bowl on the plate"
    assert omit_bundle.prompt_text != fixed_bundle.prompt_text
    assert omit_config.indicator_mode_train == "omit"
    assert fixed_config.indicator_mode_train == "fixed_positive"


def test_shuffled_indicator_uses_deterministic_sample_key_behavior() -> None:
    config = resolve_repaired_stage_config("shuffled_indicator")
    transform = RepairedStagePromptTransform(
        stage=config.stage,
        consumer_mode=config.consumer_mode,
        fixed_indicator_mode=config.fixed_indicator_mode,
    )
    prompt_a = str(
        transform(
            {
                "prompt_raw": "put the bowl on the plate",
                "recap_m2.indicator_I": 1,
                "episode_index": 5,
                "observation/state": [0.1, 0.2, 0.3],
            }
        )["prompt"].item()
    )
    prompt_b = str(
        transform(
            {
                "prompt_raw": "put the bowl on the plate",
                "recap_m2.indicator_I": 0,
                "episode_index": 5,
                "observation/state": [0.1, 0.2, 0.3],
            }
        )["prompt"].item()
    )

    assert prompt_a == prompt_b
    assert prompt_a.startswith("put the bowl on the plate")
    assert prompt_a == "put the bowl on the plate" or prompt_a.endswith(
        ("Advantage: positive", "Advantage: negative")
    )


def test_recap_informative_uses_stage_scoped_non_one_step_default_budget() -> None:
    informative_config = resolve_repaired_stage_config("recap_informative")
    shuffled_config = resolve_repaired_stage_config("shuffled_indicator")

    assert RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS == 24
    assert (
        informative_config.default_num_train_steps
        == RECAP_INFORMATIVE_DEFAULT_NUM_TRAIN_STEPS
    )
    assert informative_config.default_num_train_steps > 18
    assert (
        informative_config.default_save_interval
        == RECAP_INFORMATIVE_DEFAULT_SAVE_INTERVAL
    )
    assert (
        informative_config.default_save_interval
        == informative_config.default_num_train_steps
    )
    assert (
        shuffled_config.default_num_train_steps
        == DEFAULT_REPAIRED_STAGE_NUM_TRAIN_STEPS
    )
    assert shuffled_config.default_save_interval == DEFAULT_REPAIRED_STAGE_SAVE_INTERVAL
