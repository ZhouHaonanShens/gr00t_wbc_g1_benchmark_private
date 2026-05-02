from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.runtime_prompt import (  # noqa: E402
    build_runtime_prompt_bundle,
    build_runtime_prompt_from_inputs,
    resolve_runtime_indicator_config,
)
from work.recap import text_indicator  # noqa: E402


def test_runtime_prompt_from_inputs_uses_recap_authority_contract() -> None:
    prompt_raw = "put the bowl on the plate"
    expected = text_indicator.build_authoritative_carrier_text_v1(
        prompt_raw,
        text_indicator.TEXT_INDICATOR_NEGATIVE,
    )

    prompt_text = build_runtime_prompt_from_inputs(
        prompt_raw,
        indicator_mode="negative",
    )

    assert prompt_text == expected


def test_runtime_prompt_bundle_matches_runtime_prompt_from_inputs() -> None:
    prompt_raw = "put the bowl on the plate"
    bundle = build_runtime_prompt_bundle(
        prompt_raw,
        config=resolve_runtime_indicator_config(
            requested_indicator_mode="omit",
            variant="recap_only_relabel8d_v2",
        ),
    )

    assert bundle.prompt_text == build_runtime_prompt_from_inputs(
        prompt_raw,
        indicator_mode="omit",
    )
    assert bundle.authoritative_carrier_text == bundle.prompt_text
    assert bundle.authoritative_carrier_matches_prompt_text is True
