from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_action_chain_telemetry


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_action_chain_telemetry.main(["--help"])
    assert exc_info.value.code == 0


@pytest.mark.parametrize(
    ("branch", "expected_name", "expected_public_anchor_comparable"),
    [
        ("UNITREE_G1", "action_chain_telemetry_unitree_g1.json", True),
        ("NEW_EMBODIMENT", "action_chain_telemetry_new_embodiment.json", False),
    ],
)
def test_main_writes_branch_telemetry_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    branch: str,
    expected_name: str,
    expected_public_anchor_comparable: bool,
) -> None:
    output_path = tmp_path / expected_name

    exit_code = gr00t_action_chain_telemetry.main(
        ["--branch", branch, "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    written = _read_json(output_path)

    assert exit_code == 0
    assert captured.err == ""
    assert payload == written
    assert (
        payload["schema_version"] == gr00t_action_chain_telemetry.REPORT_SCHEMA_VERSION
    )
    assert payload["artifact_kind"] == gr00t_action_chain_telemetry.REPORT_ARTIFACT_KIND
    assert payload["branch"] == branch
    assert payload["public_anchor_comparable"] is expected_public_anchor_comparable
    assert payload["output_path"].endswith(expected_name)
    assert payload["action_order"] == [
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "waist",
        "base_height_command",
        "navigate_command",
    ]
    assert sorted(payload["per_group_stats"].keys()) == [
        "base_height_command",
        "left_arm",
        "left_hand",
        "navigate_command",
        "right_arm",
        "right_hand",
        "waist",
    ]
    assert set(payload.keys()) >= {
        "raw_action",
        "decoded_action",
        "absolute_action",
        "controller_input",
        "per_group_stats",
        "clip_rate",
        "saturation_rate",
        "zero_motion_flags",
    }
    left_arm = payload["per_group_stats"]["left_arm"]
    assert left_arm["difference_metrics"]["raw_action_l2"] > 0.0
    assert left_arm["difference_metrics"]["absolute_action_l2"] > 0.0
    assert left_arm["difference_metrics"]["controller_input_l2"] == 0.0
    assert (
        left_arm["difference_metrics"]["difference_disappeared_at"]
        == "controller_input"
    )
    assert (
        left_arm["difference_metrics"]["controller_absorbed_upstream_difference"]
        is True
    )
    assert payload["controller_absorbed_upstream_difference"] is True
    assert "left_arm" in payload["controller_absorbed_groups"]


def test_default_output_path_matches_branch_contract() -> None:
    assert (
        gr00t_action_chain_telemetry.default_output_path_for_branch("UNITREE_G1").name
        == "action_chain_telemetry_unitree_g1.json"
    )
    assert (
        gr00t_action_chain_telemetry.default_output_path_for_branch(
            "NEW_EMBODIMENT"
        ).name
        == "action_chain_telemetry_new_embodiment.json"
    )


def test_build_report_distinguishes_controller_absorption_from_model_insensitivity() -> (
    None
):
    report = gr00t_action_chain_telemetry.build_telemetry_report("UNITREE_G1")
    left_arm = report["per_group_stats"]["left_arm"]
    right_hand = report["per_group_stats"]["right_hand"]

    assert left_arm["difference_metrics"]["raw_action_l2"] > 0.0
    assert left_arm["difference_metrics"]["absolute_action_l2"] > 0.0
    assert left_arm["difference_metrics"]["controller_input_l2"] == 0.0
    assert (
        left_arm["difference_metrics"]["controller_absorbed_upstream_difference"]
        is True
    )
    assert left_arm["difference_metrics"]["model_insensitive"] is False

    assert right_hand["difference_metrics"]["raw_action_l2"] == 0.0
    assert right_hand["difference_metrics"]["model_insensitive"] is True
    assert (
        right_hand["difference_metrics"]["controller_absorbed_upstream_difference"]
        is False
    )
    assert "right_hand" in report["zero_motion_flags"]["all_zero_in_both_groups"]
