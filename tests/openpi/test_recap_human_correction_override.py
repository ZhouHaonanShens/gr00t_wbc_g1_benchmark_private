from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.advantage import build_advantage_generation_plan  # noqa: E402


def test_human_correction_rows_force_positive_before_indicator_emission() -> None:
    plan = build_advantage_generation_plan(
        prompts_by_episode={0: "task a", 1: "task a"},
        returns_by_episode={0: [-1.0], 1: [0.0]},
        values_by_episode={0: [0.0], 1: [-1.0]},
        corrections_by_episode={0: [True], 1: [False]},
        critic_metadata={
            "value_source": "critic",
            "critic_dir": "/tmp/critic/best",
            "critic_checkpoint_ref": "/tmp/critic/best",
        },
    )

    corrected = cast(dict[int, list[dict[str, object]]], plan["records_by_episode"])[0][
        0
    ]

    assert float(cast(Any, corrected["advantage_A"])) < float(
        cast(Any, corrected["epsilon_l"])
    )
    assert corrected["indicator_I"] == 1
    assert corrected["human_correction_override_applied"] is True
    assert str(corrected["prompt_conditioned"]).endswith("Advantage: positive")
