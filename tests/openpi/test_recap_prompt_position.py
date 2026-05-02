from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.runtime_prompt import (  # noqa: E402
    build_runtime_prompt_bundle,
    build_training_prompt_bundle,
    resolve_runtime_indicator_config,
)


def test_prompt_carrier_uses_canonical_newline_after_task_text() -> None:
    training = build_training_prompt_bundle(
        {
            "prompt_raw": "put the bowl on the plate",
            "recap_m2.indicator_I": 1,
            "prompt_conditioned": "put the bowl on the plate\nAdvantage: positive",
        },
        consumer_mode="informative",
        fixed_indicator_mode=None,
    )
    runtime = build_runtime_prompt_bundle(
        "put the bowl on the plate",
        config=resolve_runtime_indicator_config(
            requested_indicator_mode="positive",
            variant="recap_only_relabel8d_v2",
        ),
    )

    assert training.prompt_text == "put the bowl on the plate\nAdvantage: positive"
    assert runtime.prompt_text == training.prompt_text
    assert not training.prompt_text.startswith("advantage positive ")
    assert training.prompt_text.splitlines() == [
        "put the bowl on the plate",
        "Advantage: positive",
    ]
