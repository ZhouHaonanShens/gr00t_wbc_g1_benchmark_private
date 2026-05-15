from __future__ import annotations

import ast
from pathlib import Path

from work.recap.r7_1_recipe_plumbing.dual_loss_wiring import build_dual_loss_kwargs
from work.recap.r7_1_recipe_plumbing.flags import RecipeFlags


def test_off_returns_empty_kwargs() -> None:
    assert build_dual_loss_kwargs(RecipeFlags.default()) == {}


def test_on_returns_complete_kwargs() -> None:
    flags = RecipeFlags(
        enable_dual_loss=True,
        dual_loss_alpha=0.5,
        dual_loss_uses_carrier_text=True,
        carrier_text_field="carrier_text_v1",
    )
    assert build_dual_loss_kwargs(flags) == {
        "alpha": 0.5,
        "uses_carrier_text": True,
        "carrier_text_field": "carrier_text_v1",
    }


def test_alpha_zero_still_returns_when_explicitly_enabled() -> None:
    kwargs = build_dual_loss_kwargs(RecipeFlags(enable_dual_loss=True, dual_loss_alpha=0.0))
    assert kwargs["alpha"] == 0.0
    assert kwargs["uses_carrier_text"] is False


def test_field_mapping_is_one_to_one() -> None:
    flags = RecipeFlags(
        enable_dual_loss=True,
        dual_loss_alpha=0.25,
        dual_loss_uses_carrier_text=True,
        carrier_text_field="prompt_raw",
    )
    kwargs = build_dual_loss_kwargs(flags)
    assert kwargs["alpha"] == flags.dual_loss_alpha
    assert kwargs["carrier_text_field"] == flags.carrier_text_field


def test_dual_loss_wiring_does_not_import_runtime_modules() -> None:
    tree = ast.parse(Path("work/recap/r7_1_recipe_plumbing/dual_loss_wiring.py").read_text())
    imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]
    assert "torch" not in imports
