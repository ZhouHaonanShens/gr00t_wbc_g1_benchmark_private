from __future__ import annotations

import pytest

from work.recap.r1_repro.protocol import P0B_PROTOCOL, T81_B0_VARIANT_CKPT_ROOT
from work.recap.r1_repro.protocol_swap import (
    UnknownAxis,
    UnresolvedProtocolField,
    enumerate_axes,
    swap_single_axis,
)


def test_swap_single_axis_changes_one_field() -> None:
    swapped = swap_single_axis(
        P0B_PROTOCOL, "max_episode_steps", 720, name="episode_length_swap"
    )

    assert swapped.max_episode_steps == 720
    assert P0B_PROTOCOL.max_episode_steps == 1440
    assert enumerate_axes(P0B_PROTOCOL, swapped) == [
        ("max_episode_steps", 1440, 720)
    ]


def test_swap_unknown_axis_raises() -> None:
    with pytest.raises(UnknownAxis):
        swap_single_axis(P0B_PROTOCOL, "not_a_field", 1, name="bad_axis")


def test_swap_none_value_on_required_field_raises_unresolved() -> None:
    with pytest.raises(UnresolvedProtocolField):
        swap_single_axis(P0B_PROTOCOL, "ckpt_root", None, name="missing_ckpt")


def test_enumerate_axes_returns_only_diffs() -> None:
    swapped = swap_single_axis(
        P0B_PROTOCOL, "ckpt_root", T81_B0_VARIANT_CKPT_ROOT, name="ckpt_root_swap"
    )

    assert enumerate_axes(P0B_PROTOCOL, swapped) == [
        ("ckpt_root", P0B_PROTOCOL.ckpt_root, T81_B0_VARIANT_CKPT_ROOT)
    ]
