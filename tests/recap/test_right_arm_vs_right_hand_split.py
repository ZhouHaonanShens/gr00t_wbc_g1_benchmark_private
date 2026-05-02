from __future__ import annotations

import json
import importlib
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


interface_localization_right_hand_split = importlib.import_module(
    "work.recap.scripts.interface_localization_right_hand_split"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        interface_localization_right_hand_split.main(["--help"])
    assert exc_info.value.code == 0


def test_main_writes_split_audit_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "interface_localization_sprint"

    exit_code = interface_localization_right_hand_split.main(
        ["--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    payload = _read_json(
        output_dir / interface_localization_right_hand_split.SPLIT_AUDIT_JSON_NAME
    )

    assert (
        payload
        == interface_localization_right_hand_split.build_right_arm_vs_right_hand_split_audit(
            REPO_ROOT,
            output_dir=output_dir,
        )
    )
    assert (
        payload["schema_version"]
        == interface_localization_right_hand_split.SPLIT_AUDIT_SCHEMA_VERSION
    )
    assert (
        payload["artifact_kind"]
        == interface_localization_right_hand_split.SPLIT_AUDIT_ARTIFACT_KIND
    )
    assert payload["backpointer"]["writer_script"] == (
        "work/recap/scripts/interface_localization_right_hand_split.py"
    )


def test_split_audit_keeps_top_level_buckets_and_ownership_binding() -> None:
    payload = interface_localization_right_hand_split.build_right_arm_vs_right_hand_split_audit(
        REPO_ROOT
    )

    assert payload["summary"]["ownership_binding_by_bucket"] == {
        "body_wrist_upper_limb_chain": "right_arm",
        "dex3_finger_hand_path": "right_hand",
    }
    assert "body_wrist_upper_limb_chain" in payload
    assert "dex3_finger_hand_path" in payload

    body_bucket = payload["body_wrist_upper_limb_chain"]
    dex_bucket = payload["dex3_finger_hand_path"]

    assert body_bucket["source_evidence"]["action_key"] == "right_arm"
    assert body_bucket["source_evidence"]["reference_state_key"] == "right_arm"
    assert body_bucket["source_evidence"]["boundary_focus_joints"] == [
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    assert all(
        joint.startswith("right_wrist_")
        for joint in body_bucket["source_evidence"]["boundary_focus_joints"]
    )
    assert body_bucket["upstream_source_route_state"]["status"] == "survived"
    assert body_bucket["audit_summary"]["watch_bucket_classification"] == (
        "live_difference_persists"
    )

    assert dex_bucket["source_evidence"]["action_key"] == "right_hand"
    assert dex_bucket["source_evidence"]["reference_state_key"] is None
    assert dex_bucket["source_evidence"]["boundary_focus_joints"] == [
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
    ]
    assert all(
        joint.startswith("right_hand_")
        for joint in dex_bucket["source_evidence"]["boundary_focus_joints"]
    )
    assert all(
        "wrist" not in joint
        for joint in dex_bucket["source_evidence"]["boundary_focus_joints"]
    )


def test_split_audit_separates_source_telemetry_and_blocker_without_overclaiming_dex3() -> (
    None
):
    payload = interface_localization_right_hand_split.build_right_arm_vs_right_hand_split_audit(
        REPO_ROOT
    )
    dex_bucket = payload["dex3_finger_hand_path"]

    assert set(dex_bucket.keys()) >= {
        "source_evidence",
        "telemetry_evidence",
        "upstream_source_route_state",
        "audit_summary",
    }
    assert (
        dex_bucket["telemetry_evidence"]["action_chain_watch_bucket"][
            "model_insensitive"
        ]
        is True
    )
    assert (
        dex_bucket["telemetry_evidence"]["action_chain_watch_bucket"][
            "difference_disappeared_at"
        ]
        == "model"
    )
    trace_surfaces = dex_bucket["telemetry_evidence"]["repo_local_trace_surfaces"]
    assert trace_surfaces["blocked_field_names"] == []
    assert trace_surfaces["server_live_blocked_field_names"] == []
    assert {"q_error", "q_measured", "q_target"} <= set(
        trace_surfaces["observed_field_names"]
    )
    assert {"q_error", "q_measured", "q_target"} <= set(
        trace_surfaces["synthetic_field_names"]
    )
    assert dex_bucket["telemetry_evidence"]["task18_shared_finding"] == {
        "finding_code": "shared_model_insensitive_groups",
        "summary": "两条线共同出现 model_insensitive_groups=['right_hand']，说明至少有部分 action group 仍对 prompt/branch 条件不敏感。",
    }

    assert dex_bucket["upstream_source_route_state"]["status"] == (
        "blocked_missing_upstream"
    )
    assert (
        dex_bucket["upstream_source_route_state"]["repo_local_telemetry_retained"]
        is True
    )
    assert "Dex3" in dex_bucket["upstream_source_route_state"]["summary"]
    assert payload["summary"]["blocked_upstream_source_route_buckets"] == [
        "dex3_finger_hand_path"
    ]
    assert payload["summary"]["model_insensitive_buckets"] == ["dex3_finger_hand_path"]
