from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.runtime_prompt import (  # noqa: E402
    build_runtime_prompt_bundle,
    build_training_prompt_bundle,
    resolve_runtime_indicator_config,
)
from work.recap import text_indicator  # noqa: E402


def test_openpi_mainline_train_runtime_share_authoritative_carrier_text_v1() -> None:
    prompt_raw = "put the bowl on the plate"
    carrier_text_v1 = text_indicator.build_authoritative_carrier_text_v1(
        prompt_raw,
        text_indicator.TEXT_INDICATOR_POSITIVE,
    )

    training = build_training_prompt_bundle(
        {
            "prompt_raw": prompt_raw,
            "recap_m2.indicator_I": 1,
            "carrier_text_v1": carrier_text_v1,
        },
        consumer_mode="informative_adv",
        fixed_indicator_mode=None,
    )
    runtime = build_runtime_prompt_bundle(
        prompt_raw,
        config=resolve_runtime_indicator_config(
            requested_indicator_mode="positive",
            variant="recap_only_relabel8d_v2",
        ),
    )

    assert training.prompt_text == carrier_text_v1
    assert runtime.prompt_text == carrier_text_v1
    assert training.authoritative_carrier_text == carrier_text_v1
    assert runtime.authoritative_carrier_text == carrier_text_v1
    assert training.authoritative_carrier_source == "carrier_text_v1"
    assert runtime.authoritative_carrier_source == "prompt_raw+indicator_mode"
    assert (
        training.prompt_provenance["authoritative_carrier_field"]
        == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    )
    assert (
        runtime.prompt_provenance["authoritative_carrier_schema_version"]
        == text_indicator.RECAP_TEXT_INDICATOR_SCHEMA_VERSION
    )


def test_openpi_training_bundle_rejects_mismatched_carrier_text_v1() -> None:
    with pytest.raises(
        ValueError,
        match=r"carrier_text_v1 must match the canonical prompt_raw \+ indicator_I text-indicator carrier",
    ):
        _ = build_training_prompt_bundle(
            {
                "prompt_raw": "put the bowl on the plate",
                "recap_m2.indicator_I": 1,
                "carrier_text_v1": "put the bowl on the plate\nAdvantage: negative",
            },
            consumer_mode="informative_adv",
            fixed_indicator_mode=None,
        )
