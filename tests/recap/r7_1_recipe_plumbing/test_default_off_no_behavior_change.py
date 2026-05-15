from __future__ import annotations

from types import SimpleNamespace

from work.recap import model as recap_model
from work.recap.r7_1_recipe_plumbing.flags import RecipeFlags


def test_model_recipe_helper_omitted_vs_default_are_equivalent() -> None:
    omitted_config = SimpleNamespace()
    default_config = SimpleNamespace(r7_1_recipe_flags=RecipeFlags.default())
    assert recap_model.resolve_r7_1_recipe_flags(omitted_config) is None
    assert recap_model.resolve_r7_1_recipe_flags(default_config) is None


def test_model_dual_loss_metadata_default_off_is_empty() -> None:
    omitted_config = SimpleNamespace()
    default_config = SimpleNamespace(r7_1_recipe_flags=RecipeFlags.default())
    assert recap_model.build_r7_1_dual_loss_metadata(omitted_config) == {}
    assert recap_model.build_r7_1_dual_loss_metadata(default_config) == {}


def test_model_dual_loss_metadata_only_appears_when_enabled() -> None:
    enabled = RecipeFlags(
        enable_dual_loss=True,
        dual_loss_alpha=0.5,
        dual_loss_uses_carrier_text=True,
        carrier_text_field="carrier_text_v1",
    )
    metadata = recap_model.build_r7_1_dual_loss_metadata(SimpleNamespace(r7_1_recipe_flags=enabled))
    assert metadata == {"alpha": 0.5, "uses_carrier_text": True, "carrier_text_field": "carrier_text_v1"}
