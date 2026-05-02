from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.real_variant_export import VariantPromptTransform  # noqa: E402
from work.openpi.recap.runtime_prompt import (  # noqa: E402
    build_runtime_prompt_bundle,
    build_training_prompt_bundle,
    resolve_runtime_indicator_config,
)


def test_runtime_positive_prompt_matches_training_positive_surface() -> None:
    training = build_training_prompt_bundle(
        {
            "prompt_raw": "put the bowl on the plate",
            "recap_m2.indicator_I": 1,
        },
        consumer_mode="informative_adv",
        fixed_indicator_mode=None,
    )
    runtime = build_runtime_prompt_bundle(
        "put the bowl on the plate",
        config=resolve_runtime_indicator_config(
            requested_indicator_mode="positive",
            variant="recap_only_relabel8d_v2",
        ),
    )

    assert runtime.prompt_text == training.prompt_text
    assert runtime.prompt_text_surface == training.prompt_text_surface
    assert runtime.authoritative_carrier_text == training.authoritative_carrier_text
    assert runtime.authoritative_carrier_matches_prompt_text is True
    assert (
        training.prompt_provenance["authoritative_carrier_field"] == "carrier_text_v1"
    )


def test_runtime_omit_and_positive_do_not_collapse_for_same_prompt_raw() -> None:
    omit_runtime = build_runtime_prompt_bundle(
        "put the bowl on the plate",
        config=resolve_runtime_indicator_config(
            requested_indicator_mode="omit",
            variant="recap_only_relabel8d_v2",
        ),
    )
    positive_runtime = build_runtime_prompt_bundle(
        "put the bowl on the plate",
        config=resolve_runtime_indicator_config(
            requested_indicator_mode="positive",
            variant="recap_only_relabel8d_v2",
        ),
    )
    training_transform = VariantPromptTransform(
        consumer_mode="informative_adv",
        fixed_indicator_mode=None,
    )
    positive_training = str(
        training_transform(
            {
                "prompt_raw": "put the bowl on the plate",
                "recap_m2.indicator_I": 1,
            }
        )["prompt"].item()
    )

    assert omit_runtime.prompt_text == "put the bowl on the plate"
    assert positive_runtime.prompt_text == positive_training
    assert omit_runtime.prompt_text != positive_runtime.prompt_text
    assert positive_runtime.authoritative_carrier_text == positive_training
