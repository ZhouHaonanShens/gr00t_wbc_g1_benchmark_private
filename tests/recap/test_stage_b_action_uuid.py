from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.stage_b.action_uuid import (  # noqa: E402
    build_action_identity,
    build_chain_action_uuid,
    build_contrast_group_uuid,
    stable_content_hash,
    validate_trace_alignment,
)


def test_chain_action_uuid_is_deterministic_and_seeded_by_step() -> None:
    fields = {
        "episode_id": "ep-20000-positive",
        "step_id": 1,
        "seed": 20000,
        "policy_call_index": 1,
        "obs_hash": "sha256:obs",
        "checkpoint_id": "checkpoint-6600",
        "indicator_mode": "positive",
    }

    assert build_chain_action_uuid(**fields) == build_chain_action_uuid(**fields)
    assert build_chain_action_uuid(**fields) != build_chain_action_uuid(
        **{**fields, "step_id": 2}
    )


def test_action_content_hash_is_stable_for_mapping_order_and_sequences() -> None:
    left = {"q": [1.0, 2.0], "tau": (0.0, float("nan"))}
    right = {"tau": [0.0, float("nan")], "q": [1.0, 2.0]}

    assert stable_content_hash(left) == stable_content_hash(right)


def test_action_identity_is_sidecar_without_action_mutation() -> None:
    action = {"q": [0.1, 0.2], "tau": [0.0, 0.0]}
    before = dict(action)

    identity = build_action_identity(
        action_payload=action,
        episode_id="ep-20000",
        step_id=1,
        seed=20000,
        policy_call_index=1,
        obs_hash="sha256:obs",
        checkpoint_id="checkpoint-6600",
        indicator_mode="omit",
        stage_name="policy_output",
    )

    assert action == before
    sidecar = identity.to_jsonable()
    assert sidecar["chain_action_uuid"]
    assert sidecar["action_content_hash"].startswith("sha256:")
    assert sidecar["stage_name"] == "policy_output"


def test_validate_trace_alignment_requires_same_uuid_and_hash() -> None:
    identity = build_action_identity(
        action_payload={"q": [1, 2, 3]},
        episode_id="ep-1",
        step_id=1,
        seed=20000,
        policy_call_index=1,
        obs_hash="sha256:obs",
    ).to_jsonable()

    assert (
        validate_trace_alignment(
            [
                {**identity, "stage_name": "policy_output"},
                {**identity, "stage_name": "controller_input"},
                {**identity, "stage_name": "env_applied_action"},
            ]
        )["status"]
        == "PASS"
    )
    assert (
        validate_trace_alignment(
            [
                identity,
                {**identity, "action_content_hash": stable_content_hash({"q": [9]})},
            ]
        )["status"]
        == "FAIL"
    )


def test_contrast_group_uuid_keeps_modes_paired_on_same_observation() -> None:
    common = {
        "episode_id": "ep-20000",
        "seed": 20000,
        "obs_hash": "sha256:obs",
        "contrast_axis": "indicator_mode",
        "checkpoint_pair_id": "ref-vs-g3",
    }

    assert build_contrast_group_uuid(**common) == build_contrast_group_uuid(**common)
    assert build_contrast_group_uuid(**common) != build_contrast_group_uuid(
        **{**common, "obs_hash": "sha256:other"}
    )

