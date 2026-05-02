from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_condition_flip_probe


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = gr00t_condition_flip_probe.main(["--help"])
    assert exc_info.value.code == 0


@pytest.mark.parametrize(
    ("branch", "expected_name", "expected_scene_prefix"),
    [
        ("UNITREE_G1", "condition_flip_scorecard_unitree_g1.json", "unitree_g1::"),
        (
            "NEW_EMBODIMENT",
            "condition_flip_scorecard_new_embodiment.json",
            "new_embodiment::",
        ),
    ],
)
def test_main_writes_branch_isolated_condition_flip_scorecard(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    branch: str,
    expected_name: str,
    expected_scene_prefix: str,
) -> None:
    output_path = tmp_path / expected_name

    exit_code = gr00t_condition_flip_probe.main(
        ["--branch", branch, "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    written = _read_json(output_path)

    assert exit_code == 0
    assert captured.err == ""
    assert payload == written
    assert payload["schema_version"] == gr00t_condition_flip_probe.REPORT_SCHEMA_VERSION
    assert payload["artifact_kind"] == gr00t_condition_flip_probe.REPORT_ARTIFACT_KIND
    assert payload["branch"] == branch
    assert payload["output_path"].endswith(expected_name)
    assert payload["failure_note_path"] is None
    assert payload["paired_scene_id"].startswith(expected_scene_prefix)
    assert payload["same_scene_locked"] is True
    assert payload["same_observation_locked"] is True
    assert payload["pass_fail_gate"] == "PASS"
    assert payload["response_ratio"]["min_ratio_across_semantic_flips"] > 0.0
    assert payload["response_ratio"]["passing_variants"]
    assert {row["variant_id"] for row in payload["prompt_variants"]} == {
        "original",
        "blank",
        "target_swapped",
        "contradictory",
    }
    assert set(payload["per_group_deltas"].keys()) == {
        "blank",
        "target_swapped",
        "contradictory",
    }
    blank_nav = payload["per_group_deltas"]["blank"]["navigate_command"]
    swapped_arm = payload["per_group_deltas"]["target_swapped"]["right_arm"]
    assert blank_nav["controller_input"]["mean_abs"] > 0.0
    assert swapped_arm["controller_input"]["l2"] > 0.0
    assert (
        payload["trajectory_divergence"]["contradictory"]["controller_input"][
            "mean_abs"
        ]
        > 0.0
    )
    assert (
        payload["focus_key_deltas"]["target_swapped"]["action.right_arm"][
            "controller_input"
        ]["max_abs"]
        > 0.0
    )


def test_default_output_path_matches_branch_contract() -> None:
    assert (
        gr00t_condition_flip_probe.default_output_path_for_branch("UNITREE_G1").name
        == "condition_flip_scorecard_unitree_g1.json"
    )
    assert (
        gr00t_condition_flip_probe.default_output_path_for_branch("NEW_EMBODIMENT").name
        == "condition_flip_scorecard_new_embodiment.json"
    )


def test_identical_semantic_variants_fail_gate_and_write_failure_note(
    tmp_path: Path,
) -> None:
    branch = "UNITREE_G1"
    default_suite = gr00t_condition_flip_probe.build_default_raw_action_suite(branch)
    original = default_suite["original"]
    insensitive_suite = {
        variant: {key: value.copy() for key, value in original.items()}
        for variant in ["original", "blank", "target_swapped", "contradictory"]
    }
    output_path = tmp_path / "condition_flip_scorecard_unitree_g1.json"

    report = gr00t_condition_flip_probe.build_condition_flip_scorecard(
        branch,
        output_path=output_path,
        raw_action_suite=insensitive_suite,
    )
    _ = gr00t_condition_flip_probe.write_scorecard_artifacts(
        report, output_path=output_path
    )

    failure_note_path = output_path.with_name(
        gr00t_condition_flip_probe.FAILURE_NOTE_MARKDOWN_NAME_BY_BRANCH[branch]
    )
    written = _read_json(output_path)

    assert report["pass_fail_gate"] == "FAIL"
    assert written["pass_fail_gate"] == "FAIL"
    assert report["gate_details"]["reason_code"] == "semantic_variants_near_identical"
    assert report["response_ratio"]["min_ratio_across_semantic_flips"] == 0.0
    assert failure_note_path.is_file()
    assert "semantic_variants_near_identical" in failure_note_path.read_text(
        encoding="utf-8"
    )


def test_resolve_paired_scene_id_prefers_task10_reachable_scene_pool() -> None:
    assert (
        gr00t_condition_flip_probe.resolve_paired_scene_id("UNITREE_G1")
        == "unitree_g1::S_drop"
    )
    assert (
        gr00t_condition_flip_probe.resolve_paired_scene_id("NEW_EMBODIMENT")
        == "new_embodiment::S_drop"
    )
