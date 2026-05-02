from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_controller_audit_new_embodiment


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_modality_config() -> Path:
    return REPO_ROOT / "work" / "configs" / "new_embodiment" / "modality_config.json"


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_controller_audit_new_embodiment.main(["--help"])
    assert exc_info.value.code == 0


def test_main_materializes_new_embodiment_branch_manifest_and_audit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    branch_manifest_path = tmp_path / "branch_manifest.json"
    output_path = tmp_path / "controller_audit_new_embodiment.json"

    exit_code = gr00t_controller_audit_new_embodiment.main(
        [
            "--modality-config-path",
            str(_repo_modality_config()),
            "--branch-manifest-path",
            str(branch_manifest_path),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    written = _read_json(output_path)
    manifest = _read_json(branch_manifest_path)

    assert exit_code == 0
    assert captured.err == ""
    assert payload == written
    assert (
        payload["schema_version"]
        == gr00t_controller_audit_new_embodiment.REPORT_SCHEMA_VERSION
    )
    assert (
        payload["artifact_kind"]
        == gr00t_controller_audit_new_embodiment.REPORT_ARTIFACT_KIND
    )
    assert payload["branch_tag"] == "NEW_EMBODIMENT"
    assert payload["public_anchor_comparable"] is False
    assert payload["unitree_equivalence_reference"] == "informational_only"
    assert payload["formal_branch_eligibility"] == "ALLOW"
    assert payload["reason_code"] == "OK"
    assert payload["state_order_expected"] == [
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    ]
    assert payload["action_order_expected"] == [
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "waist",
        "base_height_command",
        "navigate_command",
    ]
    assert payload["relative_action_policy"]["relative_action_keys"] == [
        "left_arm",
        "right_arm",
    ]
    assert payload["relative_action_policy"]["absolute_action_keys"] == [
        "left_hand",
        "right_hand",
        "waist",
        "base_height_command",
        "navigate_command",
    ]
    assert payload["normalization_source"]["cross_branch_reuse_allowed"] is False
    assert payload["controller_provenance"]["public_benchmark_equivalent"] is False
    assert payload["branch_manifest_created"] is True
    assert (
        manifest["schema_version"]
        == gr00t_controller_audit_new_embodiment.BRANCH_MANIFEST_SCHEMA_VERSION
    )
    assert (
        manifest["artifact_kind"]
        == gr00t_controller_audit_new_embodiment.BRANCH_MANIFEST_ARTIFACT_KIND
    )
    assert (
        manifest["modality_config_fingerprint_sha256"]
        == payload["modality_config_fingerprint_sha256"]
    )
    assert manifest["public_anchor_comparable"] is False
    assert manifest["formal_branch_eligibility"] == "ALLOW"
    assert not output_path.with_name(
        gr00t_controller_audit_new_embodiment.FAILURE_NOTE_MARKDOWN_NAME
    ).exists()


def test_main_blocks_when_manifest_is_missing_normalization_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    modality_contract = gr00t_controller_audit_new_embodiment.load_modality_contract(
        _repo_modality_config()
    )
    branch_manifest_path = tmp_path / "branch_manifest_missing_normalization.json"
    output_path = tmp_path / "controller_audit_new_embodiment.json"
    manifest = gr00t_controller_audit_new_embodiment.build_branch_manifest_payload(
        modality_config_path=_repo_modality_config(),
        modality_contract=modality_contract,
    )
    del manifest["normalization_source"]
    branch_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    exit_code = gr00t_controller_audit_new_embodiment.main(
        [
            "--modality-config-path",
            str(_repo_modality_config()),
            "--branch-manifest-path",
            str(branch_manifest_path),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    failure_note_path = output_path.with_name(
        gr00t_controller_audit_new_embodiment.FAILURE_NOTE_MARKDOWN_NAME
    )

    assert exit_code == 0
    assert captured.err == ""
    assert payload["formal_branch_eligibility"] == "BLOCK"
    assert payload["reason_code"] == "missing_branch_manifest.normalization_source"
    assert "normalization_source" in payload["mismatch_fields"]
    assert failure_note_path.exists()
    assert (
        "missing_branch_manifest.normalization_source"
        in failure_note_path.read_text(encoding="utf-8")
    )


def test_main_blocks_when_manifest_is_missing_modality_config_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    modality_contract = gr00t_controller_audit_new_embodiment.load_modality_contract(
        _repo_modality_config()
    )
    branch_manifest_path = tmp_path / "branch_manifest_missing_config_path.json"
    output_path = tmp_path / "controller_audit_new_embodiment.json"
    manifest = gr00t_controller_audit_new_embodiment.build_branch_manifest_payload(
        modality_config_path=_repo_modality_config(),
        modality_contract=modality_contract,
    )
    del manifest["modality_config_path"]
    branch_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    exit_code = gr00t_controller_audit_new_embodiment.main(
        [
            "--modality-config-path",
            str(_repo_modality_config()),
            "--branch-manifest-path",
            str(branch_manifest_path),
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["formal_branch_eligibility"] == "BLOCK"
    assert payload["reason_code"] == "missing_branch_manifest.modality_config_path"
    assert "modality_config_path" in payload["mismatch_fields"]
