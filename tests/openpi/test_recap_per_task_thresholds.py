from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.thresholds import estimate_per_task_epsilons  # noqa: E402


def test_thresholds_are_estimated_per_task_not_globally() -> None:
    estimates = estimate_per_task_epsilons(
        [
            {"prompt_raw": "task a", "advantage_A": 0.0},
            {"prompt_raw": "task a", "advantage_A": 0.5},
            {"prompt_raw": "task a", "advantage_A": 1.0},
            {"prompt_raw": "task b", "advantage_A": -1.0},
            {"prompt_raw": "task b", "advantage_A": -0.5},
            {"prompt_raw": "task b", "advantage_A": -0.1},
        ]
    )

    assert set(estimates) == {"task a", "task b"}
    assert estimates["task a"].epsilon_l > 0.0
    assert estimates["task b"].epsilon_l < 0.0
    assert estimates["task a"].epsilon_l != estimates["task b"].epsilon_l
