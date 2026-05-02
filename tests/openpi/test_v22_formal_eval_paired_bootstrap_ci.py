from __future__ import annotations

import random

import pytest


TASKS = ("all_tasks_round_robin_episode_index_modulo_10",)


def _reference_ci(
    baseline: list[int],
    treatment: list[int],
    *,
    episode_indices: list[int],
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    groups: dict[int, list[float]] = {}
    deltas: list[float] = []
    for index, base_value, treatment_value in zip(
        episode_indices,
        baseline,
        treatment,
        strict=True,
    ):
        delta = float(treatment_value) - float(base_value)
        groups.setdefault(index % 10, []).append(delta)
        deltas.append(delta)
    keys = sorted(groups)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        count = 0
        for _key_index in keys:
            sampled_key = rng.choice(keys)
            values = groups[sampled_key]
            total += sum(values)
            count += len(values)
        estimates.append(total / count)
    estimates.sort()
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    observed = sum(deltas) / len(deltas)
    return observed, lower, upper


def test_paired_bootstrap_ci_is_seed_deterministic() -> None:
    from work.openpi.eval.v22_formal_eval_contracts import paired_bootstrap_ci

    baseline = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2
    treatment = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1] * 2
    episode_indices = list(range(len(baseline)))

    first = paired_bootstrap_ci(
        baseline,
        treatment,
        episode_indices=episode_indices,
        tasks=TASKS,
        n_resamples=10000,
        seed=20260427,
    )
    second = paired_bootstrap_ci(
        baseline,
        treatment,
        episode_indices=episode_indices,
        tasks=TASKS,
        n_resamples=10000,
        seed=20260427,
    )
    observed, lower, upper = _reference_ci(
        baseline,
        treatment,
        episode_indices=episode_indices,
        n_resamples=10000,
        seed=20260427,
    )

    assert first == second
    assert first["n_resamples"] == 10000
    assert first["seed"] == 20260427
    assert first["paired_by"] == "episode_index_modulo_10"
    assert first["pairing_key_count"] == 10
    assert first["observed_delta"] == pytest.approx(observed)
    assert first["paired_bootstrap_ci_lower_upper"] == pytest.approx([lower, upper])


def test_paired_bootstrap_ci_requires_large_resample_count() -> None:
    from work.openpi.eval.v22_formal_eval_contracts import paired_bootstrap_ci

    with pytest.raises(ValueError, match="at_least_10000"):
        paired_bootstrap_ci([0, 1], [1, 1], n_resamples=9999)

