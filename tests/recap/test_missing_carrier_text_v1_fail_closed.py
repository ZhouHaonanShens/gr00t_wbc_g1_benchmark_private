from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.label_writer import validate_label_record


def test_missing_carrier_text_v1_causes_fail_closed_label_validation() -> None:
    with pytest.raises(ValueError, match=r"carrier_text_v1"):
        validate_label_record(
            {
                "schema_version": "recap-v0",
                "code_version": "test",
                "iter_tag": "iter_001",
                "episode_id": "episode_001",
                "t": 0,
                "return_G": 1.0,
                "value_V": 0.5,
                "advantage_A": 0.5,
                "epsilon_l": 0.0,
                "indicator_I": 1,
                "is_correction": False,
                "prompt_raw": "pick up the apple and place it on the plate",
                "prompt_conditioned": "advantage positive pick up the apple and place it on the plate",
            }
        )
