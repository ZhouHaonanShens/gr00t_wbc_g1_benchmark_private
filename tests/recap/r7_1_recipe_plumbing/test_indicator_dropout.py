from __future__ import annotations

import random

import pytest

from work.recap.r7_1_recipe_plumbing.indicator_dropout import apply_indicator_dropout, make_rng


def test_p_zero_returns_original_value() -> None:
    value = {"indicator": "positive"}
    assert apply_indicator_dropout(value, p=0.0, rng=make_rng(0)) is value


def test_p_one_always_returns_none() -> None:
    assert apply_indicator_dropout("positive", p=1.0, rng=make_rng(0)) is None


def test_same_seed_replays_same_sequence() -> None:
    left_rng = make_rng(11)
    right_rng = make_rng(11)
    left = [apply_indicator_dropout("x", p=0.5, rng=left_rng) for _ in range(8)]
    right = [apply_indicator_dropout("x", p=0.5, rng=right_rng) for _ in range(8)]
    assert left == right


def test_different_seed_changes_sequence() -> None:
    left_rng = make_rng(1)
    right_rng = make_rng(2)
    left = [apply_indicator_dropout("x", p=0.5, rng=left_rng) for _ in range(12)]
    right = [apply_indicator_dropout("x", p=0.5, rng=right_rng) for _ in range(12)]
    assert left != right


def test_probability_above_one_rejected() -> None:
    with pytest.raises(ValueError):
        apply_indicator_dropout("x", p=1.01, rng=make_rng(0))


def test_probability_below_zero_rejected() -> None:
    with pytest.raises(ValueError):
        apply_indicator_dropout("x", p=-0.01, rng=make_rng(0))


def test_seed_type_and_rng_type_rejected() -> None:
    with pytest.raises(TypeError):
        make_rng(True)
    with pytest.raises(TypeError):
        apply_indicator_dropout("x", p=0.1, rng=random)
