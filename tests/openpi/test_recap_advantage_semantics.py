from __future__ import annotations

from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.advantage import build_advantage_generation_plan  # noqa: E402
from work.openpi.recap.thresholds import build_epsilon_source  # noqa: E402


def test_advantage_plan_uses_critic_values_for_a_equals_g_minus_v() -> None:
    plan = build_advantage_generation_plan(
        prompts_by_episode={0: "task a", 1: "task b"},
        returns_by_episode={0: [-2.0, -1.0, 0.0], 1: [-1.0, 0.0]},
        values_by_episode={0: [-2.5, -0.5, 0.2], 1: [-0.2, -0.8]},
        critic_metadata={
            "value_source": "critic",
            "critic_dir": "/tmp/critic/best",
            "critic_checkpoint_ref": "/tmp/critic/best",
        },
    )

    records_by_episode = cast(
        dict[int, list[dict[str, object]]], plan["records_by_episode"]
    )
    first = records_by_episode[0][0]
    second = records_by_episode[0][1]
    contract = cast(dict[str, object], plan["advantage_contract"])

    assert first["return_G"] == -2.0
    assert first["value_V"] == -2.5
    assert first["advantage_A"] == 0.5
    assert second["advantage_A"] == -0.5
    assert contract["value_source"] == "critic"
    assert contract["indicator_formula"] == "I = 1[A > epsilon_l(task)]"
    assert plan["epsilon_source"] == build_epsilon_source()
