from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.runtime_prompt import (  # noqa: E402
    build_runtime_prompt_bundle,
    resolve_runtime_indicator_config,
)


def test_cfg_runtime_for_recap_variant_injects_positive_indicator_by_default() -> None:
    config = resolve_runtime_indicator_config(
        requested_indicator_mode="cfg",
        variant="recap_only_relabel8d_v2",
        checkpoint_provenance={
            "variant_derivation": {"consumer_mode": "informative_adv"}
        },
    )
    bundle = build_runtime_prompt_bundle(
        "put the bowl on the plate",
        config=config,
    )

    assert config.indicator_mode == "positive"
    assert config.indicator_source == "cfg.consumer_mode.informative_adv"
    assert bundle.prompt_text == "put the bowl on the plate\nAdvantage: positive"
    assert bundle.prompt_text_surface == "canonical_text_indicator"


def test_runtime_explicit_omit_stays_distinct_from_positive_surface() -> None:
    omit = build_runtime_prompt_bundle(
        "put the bowl on the plate",
        config=resolve_runtime_indicator_config(
            requested_indicator_mode="omit",
            variant="recap_only_relabel8d_v2",
        ),
    )
    positive = build_runtime_prompt_bundle(
        "put the bowl on the plate",
        config=resolve_runtime_indicator_config(
            requested_indicator_mode="positive",
            variant="recap_only_relabel8d_v2",
        ),
    )

    assert omit.prompt_text == "put the bowl on the plate"
    assert omit.prompt_text_surface == "prompt_raw_only"
    assert positive.prompt_text.endswith("Advantage: positive")
    assert positive.prompt_text_surface == "canonical_text_indicator"
    assert omit.prompt_text != positive.prompt_text


def test_cfg_fixedadv_runtime_resolves_to_omit_without_pretending_critic_use() -> None:
    config = resolve_runtime_indicator_config(
        requested_indicator_mode="cfg",
        variant="fixedadv_relabel8d_control_v1",
        checkpoint_provenance={
            "variant_derivation": {
                "consumer_mode": "fixedadv_constant",
                "fixed_indicator_mode": "omit",
            }
        },
    )
    bundle = build_runtime_prompt_bundle(
        "put the bowl on the plate",
        config=config,
    )

    assert config.indicator_mode == "omit"
    assert config.indicator_source == "cfg.fixed_indicator_mode"
    assert bundle.prompt_text == "put the bowl on the plate"
    assert bundle.critic_checkpoint_ref == "adapter_required"
