from __future__ import annotations

import argparse
from dataclasses import fields
from typing import get_args

import pytest

from work.recap.r7_1_recipe_plumbing.flags import CarrierTextField, RecipeFlags, build_argparse_group
from work.recap.r7_1_recipe_plumbing.cli import split_recipe_args


def _parse_recipe_flags(args: list[str]) -> RecipeFlags:
    parser = argparse.ArgumentParser()
    build_argparse_group(parser)
    namespace = parser.parse_args(args)
    return RecipeFlags.from_argparse(namespace)


def test_default_recipe_flags_are_all_off() -> None:
    flags = RecipeFlags.default()
    assert flags == RecipeFlags(False, 0.0, 0.0, 0, False, "prompt_raw")
    assert flags.is_default()


def test_recipe_flags_has_exactly_six_fields() -> None:
    field_names = [field.name for field in fields(RecipeFlags)]
    assert field_names == [
        "enable_dual_loss",
        "dual_loss_alpha",
        "indicator_dropout_p",
        "indicator_dropout_seed",
        "dual_loss_uses_carrier_text",
        "carrier_text_field",
    ]


def test_carrier_text_field_literal_is_locked() -> None:
    assert get_args(CarrierTextField) == ("prompt_raw", "carrier_text_v1")
    assert RecipeFlags(carrier_text_field="carrier_text_v1").carrier_text_field == "carrier_text_v1"


def test_from_argparse_round_trips_all_flags() -> None:
    flags = _parse_recipe_flags(
        [
            "--enable-dual-loss",
            "--dual-loss-alpha",
            "0.5",
            "--indicator-dropout-p=0.15",
            "--indicator-dropout-seed",
            "7",
            "--dual-loss-uses-carrier-text",
            "--carrier-text-field",
            "carrier_text_v1",
        ]
    )
    assert flags == RecipeFlags(True, 0.5, 0.15, 7, True, "carrier_text_v1")


def test_single_flag_keeps_other_defaults() -> None:
    flags = _parse_recipe_flags(["--indicator-dropout-p", "0.25"])
    assert flags.indicator_dropout_p == 0.25
    assert flags.enable_dual_loss is False
    assert flags.carrier_text_field == "prompt_raw"


def test_invalid_carrier_text_field_rejected() -> None:
    with pytest.raises(ValueError):
        RecipeFlags(carrier_text_field="prompt_conditioned")


def test_invalid_probability_and_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        RecipeFlags(indicator_dropout_p=1.5)
    with pytest.raises(ValueError):
        RecipeFlags(dual_loss_alpha=-0.1)


def test_seed_type_and_bool_type_rejected() -> None:
    with pytest.raises(TypeError):
        RecipeFlags(indicator_dropout_seed=True)
    with pytest.raises(TypeError):
        RecipeFlags(enable_dual_loss=1)


def test_split_recipe_args_strips_only_recipe_flags() -> None:
    flags, remaining, explicit = split_recipe_args(
        [
            "--dataset-path",
            "data",
            "--enable-dual-loss",
            "--dual-loss-alpha=0.5",
            "--max-steps",
            "1",
        ]
    )
    assert flags.enable_dual_loss is True
    assert remaining == ["--dataset-path", "data", "--max-steps", "1"]
    assert explicit == ["--enable-dual-loss", "--dual-loss-alpha", "0.5"]


def test_split_recipe_args_drops_explicit_default_equivalent() -> None:
    flags, remaining, explicit = split_recipe_args(
        ["--dual-loss-alpha", "0.0", "--carrier-text-field=prompt_raw", "--num-gpus", "1"]
    )
    assert flags.is_default()
    assert remaining == ["--num-gpus", "1"]
    assert explicit == []
