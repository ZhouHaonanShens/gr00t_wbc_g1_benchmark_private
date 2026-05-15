from __future__ import annotations

from typing import Any

from work.recap.r7_1_recipe_plumbing.flags import RecipeFlags


def build_dual_loss_kwargs(flags: RecipeFlags) -> dict[str, Any]:
    if not isinstance(flags, RecipeFlags):
        raise TypeError(f"flags must be RecipeFlags, got {type(flags).__name__}")
    if not flags.enable_dual_loss:
        return {}
    kwargs: dict[str, Any] = {
        "alpha": float(flags.dual_loss_alpha),
        "uses_carrier_text": bool(flags.dual_loss_uses_carrier_text),
        "carrier_text_field": flags.carrier_text_field,
    }
    return kwargs
