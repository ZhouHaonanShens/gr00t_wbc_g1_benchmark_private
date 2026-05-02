from __future__ import annotations

import inspect
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import labeler, text_indicator
from work.recap.lerobot_export import dataset_export, video_export
from work.recap.scripts import state_conditioned_contract_gate


def test_mainline_training_text_field_is_fixed_to_carrier_text_v1() -> None:
    prompt_raw = "pick up the apple and place it on the plate"
    labels = labeler.finalize_m2_prelabels(
        [
            {
                "schema_version": "recap-v0",
                "code_version": "test",
                "iter_tag": "iter_001",
                "episode_id": "episode_001",
                "t": 0,
                "return_G": 1.0,
                "value_V": 0.2,
                "advantage_A": 0.8,
                "is_correction": False,
                "prompt_raw": prompt_raw,
            }
        ],
        epsilon_l=0.0,
    )
    expected_carrier_text = text_indicator.build_canonical_text_indicator(
        prompt_raw,
        text_indicator.TEXT_INDICATOR_POSITIVE,
    )

    assert (
        inspect.signature(dataset_export.export_recap_to_lerobot_v2)
        .parameters["task_text_field"]
        .default
        == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    )
    assert (
        inspect.signature(video_export.export_recap_to_lerobot_v2_with_video)
        .parameters["task_text_field"]
        .default
        == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    )
    assert (
        state_conditioned_contract_gate.MAINLINE_TRAINING_TEXT_FIELD
        == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    )
    assert labels[0]["carrier_text_v1"] == expected_carrier_text
