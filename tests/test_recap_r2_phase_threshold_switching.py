from __future__ import annotations

from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.advantage import build_advantage_generation_plan  # noqa: E402
from work.openpi.recap.thresholds import estimate_per_task_epsilons  # noqa: E402
from work.recap.labeler import compute_m2_epsilon_l  # noqa: E402
from work.recap.phase_thresholds import (  # noqa: E402
    FINE_TUNING_EPSILON_QUANTILE,
    PRETRAINING_EPSILON_QUANTILE,
    resolve_phase_threshold_policy,
)


def _prelabel(advantage: float, index: int) -> dict[str, object]:
    return {
        "schema_version": "recap-v0",
        "code_version": "test",
        "iter_tag": "iter3-r2-test",
        "episode_id": "episode-a",
        "t": int(index),
        "return_G": float(advantage),
        "value_V": 0.0,
        "advantage_A": float(advantage),
        "is_correction": False,
        "prompt_raw": "task a",
    }


def test_phase_policy_maps_pretraining_and_fine_tuning_quantiles() -> None:
    pretraining = resolve_phase_threshold_policy("pretraining")
    fine_tuning = resolve_phase_threshold_policy("fine-tuning")

    assert pretraining.epsilon_quantile == PRETRAINING_EPSILON_QUANTILE
    assert pretraining.target_positive_fraction == 0.3
    assert fine_tuning.epsilon_quantile == FINE_TUNING_EPSILON_QUANTILE
    assert fine_tuning.target_positive_fraction == 0.4


def test_fine_tuning_threshold_is_lower_than_pretraining_for_same_task() -> None:
    records = [{"prompt_raw": "task a", "advantage_A": float(value)} for value in range(5)]

    pretraining = estimate_per_task_epsilons(records, quantile=PRETRAINING_EPSILON_QUANTILE)
    fine_tuning = estimate_per_task_epsilons(records, quantile=FINE_TUNING_EPSILON_QUANTILE)

    assert fine_tuning["task a"].epsilon_l < pretraining["task a"].epsilon_l


def test_openpi_advantage_plan_records_threshold_phase_metadata() -> None:
    plan = build_advantage_generation_plan(
        prompts_by_episode={0: "task a"},
        returns_by_episode={0: [-1.0, 0.0, 1.0, 2.0, 3.0]},
        values_by_episode={0: [0.0, 0.0, 0.0, 0.0, 0.0]},
        critic_metadata={
            "value_source": "critic",
            "critic_dir": "/tmp/critic/best",
            "critic_checkpoint_ref": "/tmp/critic/best",
        },
        threshold_phase="fine_tuning",
    )

    contract = cast(dict[str, object], plan["advantage_contract"])
    policy = cast(dict[str, object], contract["epsilon_threshold_policy"])
    records_by_episode = cast(
        dict[int, list[dict[str, object]]],
        plan["records_by_episode"],
    )

    assert plan["epsilon_threshold_phase"] == "fine_tuning"
    assert plan["epsilon_quantile"] == FINE_TUNING_EPSILON_QUANTILE
    assert contract["epsilon_threshold_phase"] == "fine_tuning"
    assert policy["target_positive_fraction"] == 0.4
    assert records_by_episode[0][0]["epsilon_l"] == 1.4


def test_gr00t_labeler_uses_phase_quantile_when_explicit_quantile_absent() -> None:
    prelabels = [_prelabel(float(value), value) for value in range(5)]

    pretraining = compute_m2_epsilon_l(prelabels, threshold_phase="pretraining")
    fine_tuning = compute_m2_epsilon_l(prelabels, threshold_phase="fine_tuning")

    assert pretraining == 2.8
    assert fine_tuning == 2.4
    assert fine_tuning < pretraining
