from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.transitions import transition_schema
from work.recap.transitions import transition_sidecar


def _record(
    episode_id: str,
    t: int,
    *,
    phase: str,
    semantic_state: str,
    apple_in_hand: bool,
    plate_distance: float,
    drop_during_transport: bool = False,
) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "t": int(t),
        "policy_condition": {
            "phase": phase,
            "mode": "NOMINAL",
        },
        "analysis_only": {
            "semantic_state": semantic_state,
            "drop_during_transport": drop_during_transport,
        },
        "privileged": {
            "apple_in_hand": apple_in_hand,
            "apple_to_plate_rel_pose": [plate_distance, 0.0, 0.0],
        },
    }


def test_transition_label_vocab_is_frozen_exactly() -> None:
    assert transition_schema.TRANSITION_LABEL_VOCAB == (
        "approach",
        "grasp",
        "stable_grasp",
        "transport",
        "near_plate",
        "place",
        "release",
        "drop",
    )


def test_build_transition_sidecar_rows_uses_existing_phase_semantic_normalization() -> (
    None
):
    rows = transition_sidecar.build_transition_sidecar_rows(
        records=[
            _record(
                "episode_001",
                0,
                phase="APPROACH",
                semantic_state="APPLE_VISIBLE_APPROACH",
                apple_in_hand=False,
                plate_distance=0.40,
            ),
            _record(
                "episode_001",
                1,
                phase="GRASP",
                semantic_state="GRASPING",
                apple_in_hand=False,
                plate_distance=0.30,
            ),
            _record(
                "episode_001",
                2,
                phase="VERIFY_HOLD",
                semantic_state="VERIFYING_HOLD",
                apple_in_hand=True,
                plate_distance=0.25,
            ),
            _record(
                "episode_001",
                3,
                phase="TRANSPORT",
                semantic_state="TRANSPORTING",
                apple_in_hand=True,
                plate_distance=0.30,
            ),
            _record(
                "episode_001",
                4,
                phase="TRANSPORT",
                semantic_state="TRANSPORTING",
                apple_in_hand=True,
                plate_distance=0.08,
            ),
            _record(
                "episode_001",
                5,
                phase="PLACE",
                semantic_state="PLACING",
                apple_in_hand=True,
                plate_distance=0.05,
            ),
            _record(
                "episode_001",
                6,
                phase="PLACE",
                semantic_state="PLACING",
                apple_in_hand=False,
                plate_distance=0.04,
            ),
            _record(
                "episode_002",
                0,
                phase="TRANSPORT",
                semantic_state="TRANSPORTING",
                apple_in_hand=True,
                plate_distance=0.30,
                drop_during_transport=True,
            ),
        ],
        runtime_trace_by_episode={
            "episode_001": {
                "controller_output_available": False,
                "controller_output_unavailable_reason": "controller_output missing in live seam",
            }
        },
        execution_audit_by_episode={
            "episode_001": {
                "schema_version": "g1_execution_surface_audit_v1",
                "artifact_kind": "g1_execution_surface_audit",
                "verdict": "postprocess",
            }
        },
    )

    assert [row["transition_label"] for row in rows] == [
        "approach",
        "grasp",
        "stable_grasp",
        "transport",
        "near_plate",
        "place",
        "release",
        "drop",
    ]
    assert rows[0]["normalized_stage"] == "APPROACH"
    assert rows[2]["normalized_stage"] == "VERIFY_HOLD"
    assert rows[0]["controller_output_available"] is False
    assert rows[0]["terminal_stage_used"] == "controller_input"
    assert rows[0]["execution_surface_verdict"] == "postprocess"
    assert rows[0]["authority_boundary"] == "optional_runtime_context_only"
    assert rows[0]["mainline_authority"] is False
    assert rows[0]["diagnostic_only"] is True


def test_build_transition_sidecar_context_returns_not_available_when_rows_absent() -> (
    None
):
    context = transition_sidecar.build_transition_sidecar_context(None)

    assert context["status"] == "not_available"
    assert context["reason_code"] == "not_available"
    assert context["steps_labeled"] == 0
    assert context["episodes_covered"] == 0
    assert context["label_counts"] == {
        "approach": 0,
        "grasp": 0,
        "stable_grasp": 0,
        "transport": 0,
        "near_plate": 0,
        "place": 0,
        "release": 0,
        "drop": 0,
    }


def test_invalid_transition_label_invalidates_sidecar_without_partial_accept() -> None:
    valid_row = transition_sidecar.build_transition_sidecar_row(
        record=_record(
            "episode_001",
            0,
            phase="APPROACH",
            semantic_state="APPLE_VISIBLE_APPROACH",
            apple_in_hand=False,
            plate_distance=0.40,
        )
    )
    invalid_row = dict(valid_row)
    invalid_row["t"] = 1
    invalid_row["transition_label"] = "planner_override"

    context = transition_sidecar.build_transition_sidecar_context(
        [valid_row, invalid_row],
        path="agent/artifacts/apple_recap_flux_graft/rtc_transition_sidecar.jsonl",
        expected_join_keys=[("episode_001", 0), ("episode_001", 1)],
    )

    assert context["status"] == "invalid"
    assert context["reason_code"].startswith("invalid:")
    assert context["path"] == (
        "agent/artifacts/apple_recap_flux_graft/rtc_transition_sidecar.jsonl"
    )
    assert context["steps_labeled"] == 0
    assert context["episodes_covered"] == 0
    assert all(count == 0 for count in context["label_counts"].values())


def test_join_mismatch_invalidates_sidecar_only() -> None:
    rows = transition_sidecar.build_transition_sidecar_rows(
        records=[
            _record(
                "episode_001",
                0,
                phase="APPROACH",
                semantic_state="APPLE_VISIBLE_APPROACH",
                apple_in_hand=False,
                plate_distance=0.40,
            )
        ]
    )

    context = transition_sidecar.build_transition_sidecar_context(
        rows,
        expected_join_keys=[("episode_001", 0), ("episode_001", 1)],
    )

    assert context["status"] == "invalid"
    assert "join mismatch" in context["reason_code"]


def test_validate_transition_summary_round_trips_available_context() -> None:
    rows = transition_sidecar.build_transition_sidecar_rows(
        records=[
            _record(
                "episode_001",
                0,
                phase="APPROACH",
                semantic_state="APPLE_VISIBLE_APPROACH",
                apple_in_hand=False,
                plate_distance=0.40,
            ),
            _record(
                "episode_001",
                1,
                phase="GRASP",
                semantic_state="GRASPING",
                apple_in_hand=False,
                plate_distance=0.30,
            ),
        ]
    )

    context = transition_sidecar.build_transition_sidecar_context(
        rows,
        path="agent/artifacts/apple_recap_flux_graft/rtc_transition_sidecar.jsonl",
        expected_join_keys=[("episode_001", 0), ("episode_001", 1)],
    )
    validated = transition_sidecar.validate_transition_context_summary(context)

    assert validated["status"] == "available"
    assert validated["reason_code"] == "ok"
    assert validated["path"] == (
        "agent/artifacts/apple_recap_flux_graft/rtc_transition_sidecar.jsonl"
    )
    assert validated["schema_version"] == "rtc_transition_sidecar_summary_v1"
    assert validated["label_vocab_version"] == "rtc_transition_label_vocab_v1"
    assert validated["episodes_covered"] == 1
    assert validated["steps_labeled"] == 2
    assert validated["label_counts"]["approach"] == 1
    assert validated["label_counts"]["grasp"] == 1
    assert validated["authority_boundary"] == "optional_runtime_context_only"
    assert validated["mainline_authority"] is False
