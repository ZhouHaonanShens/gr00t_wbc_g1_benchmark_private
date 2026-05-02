from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import text_indicator  # noqa: E402


def test_build_recap_text_indicator_v1_record_freezes_authoritative_carrier() -> None:
    record = text_indicator.build_recap_text_indicator_v1_record(
        "put the bowl on the plate",
        "positive",
    )

    assert (
        record["schema_version"] == text_indicator.RECAP_TEXT_INDICATOR_SCHEMA_VERSION
    )
    assert (
        record["authority_name"] == text_indicator.RECAP_TEXT_INDICATOR_AUTHORITY_NAME
    )
    assert record["carrier_field"] == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    assert record[text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD] == (
        "put the bowl on the plate\nAdvantage: positive"
    )
    assert record["source_prompt_field"] == "prompt_raw"
    assert record["prompt_conditioned_role"] == "non_authority_sidecar_only"
    assert record["policy_condition_text_role"] == (
        "separate_state_conditioned_lane_not_authority"
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"advantage_input": 0.25}, "numeric advantage passthrough"),
        ({"dual_task_text": True}, "dual_task_text authority"),
        ({"policy_condition_text": "joint pose bucket text"}, "policy_condition_text"),
        (
            {"prompt_conditioned": "advantage positive put the bowl on the plate"},
            "non-canonical prompt_conditioned authority",
        ),
    ],
)
def test_build_recap_text_indicator_v1_record_rejects_mixed_lane_contamination(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = text_indicator.build_recap_text_indicator_v1_record(
            "put the bowl on the plate",
            "positive",
            **kwargs,
        )
