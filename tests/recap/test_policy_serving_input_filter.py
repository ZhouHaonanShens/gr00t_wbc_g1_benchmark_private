from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import collector
from work.recap import policy as recap_policy
from work.recap.scripts import gr00t_public_anchor_eval
from work.recap.scripts import gr00t_same_checkpoint_triplet_eval
from work.recap.scripts import state_conditioned_phase0_smoke


def _base_observation() -> dict[str, object]:
    return {
        "annotation.human.task_description": "pick up the apple",
        "state.q": np.asarray([1.0, 2.0], dtype=np.float64),
        "video.rgb": np.zeros((2, 2, 3), dtype=np.uint8),
    }


def _observation_with_sidechannels() -> dict[str, object]:
    obs = _base_observation()
    obs.update(
        {
            "policy_condition_text": "[PolicyCondition-v1]\nPHASE=TRANSPORT\nMODE=RECOVERY",
            "policy_condition.phase": "TRANSPORT",
            "policy_condition.mode": "RECOVERY",
            "prompt_conditioned": "pick up the apple\nAdvantage: positive",
            "privileged.apple_visible": True,
            "debug.trace_id": "dbg-1",
            "telemetry.latency_ms": 8.0,
            "rtc.tick": 17,
            "info.server": "shadow",
            "analysis_only": {"semantic_state": "debug-only"},
            "debug_probe": {"surface": "probe"},
            "runtime_trace": {"trace_role": "debug_probe"},
        }
    )
    return obs


def _assert_same_authoritative_surface(
    left: Mapping[str, object], right: Mapping[str, object]
) -> None:
    assert sorted(left.keys()) == sorted(right.keys())
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if hasattr(left_value, "shape") or hasattr(right_value, "shape"):
            np.testing.assert_array_equal(
                np.asarray(left_value), np.asarray(right_value)
            )
            continue
        assert left_value == right_value


def test_filter_canonical_serving_observation_strips_non_authoritative_sidechannels() -> (
    None
):
    filtered = recap_policy.filter_canonical_serving_observation(
        _observation_with_sidechannels(),
        field_name="observation",
    )

    assert sorted(filtered.keys()) == [
        "annotation.human.task_description",
        "state.q",
        "video.rgb",
    ]
    assert recap_policy.find_non_authoritative_serving_field_paths(
        _observation_with_sidechannels(),
        field_name="observation",
    ) == (
        "analysis_only",
        "debug.trace_id",
        "debug_probe",
        "info.server",
        "policy_condition.mode",
        "policy_condition.phase",
        "policy_condition_text",
        "privileged.apple_visible",
        "prompt_conditioned",
        "rtc.tick",
        "runtime_trace",
        "telemetry.latency_ms",
    )


def test_phase0_and_public_anchor_keep_authoritative_input_invariant_when_sidecars_exist() -> (
    None
):
    base = _base_observation()
    with_sidecars = _observation_with_sidechannels()

    phase0_base = state_conditioned_phase0_smoke._batch_observation_for_policy(base)
    phase0_sidecars = state_conditioned_phase0_smoke._batch_observation_for_policy(
        with_sidecars
    )
    public_base = gr00t_public_anchor_eval._normalize_policy_observation(base)
    public_sidecars = gr00t_public_anchor_eval._normalize_policy_observation(
        with_sidecars
    )

    _assert_same_authoritative_surface(phase0_base, phase0_sidecars)
    _assert_same_authoritative_surface(public_base, public_sidecars)


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_action(self, obs: Mapping[str, object]) -> dict[str, object]:
        self.calls.append(dict(obs))
        return {"action.right_arm": np.zeros((1, 1), dtype=np.float32)}


class _OneStepEnv:
    def __init__(self, obs: Mapping[str, object]) -> None:
        self._obs = dict(obs)

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[str, object], dict[str, object]]:
        del seed
        return dict(self._obs), {}

    def step(
        self, action: Mapping[str, object]
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        del action
        return (
            dict(self._obs),
            0.0,
            True,
            False,
            {
                "rewards": np.asarray([[0.0]], dtype=np.float32),
                "dones": np.asarray([[True]]),
                "success": False,
            },
        )


def test_collector_rejects_policy_prompt_prefix_rewrite() -> None:
    with pytest.raises(ValueError, match="policy_prompt_prefix is forbidden"):
        collector.collect_episode(
            env=_OneStepEnv(_base_observation()),
            client=_RecordingClient(),
            iter_tag="iter0",
            episode_id="ep0",
            env_name="env",
            model_path="model",
            embodiment_tag="UNITREE_G1",
            server_host="127.0.0.1",
            server_port=5555,
            seed=0,
            max_policy_steps=1,
            policy_prompt_prefix="rewrite me",
        )


def test_collector_strips_sidechannels_before_client_get_action() -> None:
    client = _RecordingClient()
    _episode, _transitions, _arrays = collector.collect_episode(
        env=_OneStepEnv(_observation_with_sidechannels()),
        client=client,
        iter_tag="iter0",
        episode_id="ep0",
        env_name="env",
        model_path="model",
        embodiment_tag="UNITREE_G1",
        server_host="127.0.0.1",
        server_port=5555,
        seed=0,
        max_policy_steps=1,
    )

    assert len(client.calls) == 1
    sent_obs = client.calls[0]
    assert "annotation.human.task_description" in sent_obs
    assert "policy_condition_text" not in sent_obs
    assert "prompt_conditioned" not in sent_obs
    assert "debug.trace_id" not in sent_obs
    assert "telemetry.latency_ms" not in sent_obs
    assert "runtime_trace" not in sent_obs


def test_same_checkpoint_triplet_normalization_strips_probe_sidecars() -> None:
    normalized = gr00t_same_checkpoint_triplet_eval._normalize_policy_observation(
        _observation_with_sidechannels()
    )

    assert sorted(normalized.keys()) == [
        "annotation.human.task_description",
        "state.q",
        "video.rgb",
    ]
    assert np.asarray(normalized["video.rgb"]).shape == (1, 2, 2, 3)
    assert "debug_probe" not in normalized
    assert "runtime_trace" not in normalized


def test_same_checkpoint_triplet_policy_server_batching() -> None:
    normalized = gr00t_same_checkpoint_triplet_eval._normalize_policy_observation(
        _observation_with_sidechannels()
    )
    batched = gr00t_same_checkpoint_triplet_eval._batch_policy_server_observation(
        normalized
    )

    assert np.asarray(batched["video.rgb"]).shape == (1, 1, 2, 2, 3)
    assert np.asarray(batched["state.q"]).shape == (1, 1, 2)
    assert batched["annotation.human.task_description"] == ["pick up the apple"]
