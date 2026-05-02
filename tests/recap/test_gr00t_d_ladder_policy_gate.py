from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_d_ladder_policy_gate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _registry(output_dir: Path) -> dict[str, Any]:
    return _read_json(
        output_dir / gr00t_d_ladder_policy_gate.DATASET_SOURCE_REGISTRY_JSON_NAME
    )


def _gate(output_dir: Path, branch: str) -> dict[str, Any]:
    return _read_json(
        output_dir / gr00t_d_ladder_policy_gate.GATE_JSON_NAME_BY_BRANCH[branch]
    )


def _registry_entries_by_id(
    registry_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(item["dataset_id"]): dict(item)
        for item in registry_payload.get("datasets", [])
    }


def _admission_inputs(
    rung: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_ids = [
        spec.dataset_id
        for spec in gr00t_d_ladder_policy_gate._included_dataset_specs(rung)
    ]
    dataset_fingerprints: dict[str, Any] = {}
    dataset_provenance: dict[str, Any] = {}
    normalization_records: dict[str, Any] = {}
    for index, dataset_id in enumerate(dataset_ids):
        controller_family = gr00t_d_ladder_policy_gate._dataset_spec_by_id(
            dataset_id
        ).controller_family
        dataset_fingerprints[dataset_id] = {
            "dataset_fingerprint": f"fingerprint-{index}",
            "fingerprint_source": "pytest_fixture",
        }
        dataset_provenance[dataset_id] = {
            "source_registry_id": dataset_id,
            "local_dataset_path": f"/tmp/{dataset_id.replace('/', '_')}",
            "controller_family": controller_family,
            "provenance_complete": True,
            "guardrails_satisfied": True,
        }
        normalization_records[dataset_id] = {
            "stats_fingerprint": f"stats-{index}",
            "hidden_stats_fingerprint": f"stats-{index}",
            "normalization_owner": f"owner-{index}",
            "explicit_stats_policy": "branch_scoped_stats",
            "cross_dataset_reuse_declared": False,
            "regenerated_for_branch": True,
        }
    return dataset_fingerprints, dataset_provenance, normalization_records


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_d_ladder_policy_gate.main(["--help"])
    assert exc_info.value.code == 0


def test_unitree_cli_materializes_gate_and_registry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "artifacts"

    exit_code = gr00t_d_ladder_policy_gate.main(
        ["--branch", "UNITREE_G1", "--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    registry_payload = _registry(output_dir)
    gate_payload = _gate(output_dir, "UNITREE_G1")
    entries = _registry_entries_by_id(registry_payload)
    d4_entry = entries["LightwheelAI/Lightwheel-Tasks-G1-Controller"]
    d4_rung = next(
        item for item in gate_payload["dataset_rungs"] if item["rung"] == "D4"
    )

    assert exit_code == 0
    assert captured.err == ""
    assert Path(payload["dataset_source_registry_path"]) == (
        output_dir / gr00t_d_ladder_policy_gate.DATASET_SOURCE_REGISTRY_JSON_NAME
    )
    assert Path(payload["d_ladder_policy_gate_path"]) == (
        output_dir / gr00t_d_ladder_policy_gate.GATE_JSON_NAME_BY_BRANCH["UNITREE_G1"]
    )
    assert (
        registry_payload["artifact_kind"]
        == gr00t_d_ladder_policy_gate.REGISTRY_ARTIFACT_KIND
    )
    assert (
        gate_payload["artifact_kind"] == gr00t_d_ladder_policy_gate.REPORT_ARTIFACT_KIND
    )
    assert gate_payload["branch_scope"] == "official_public_anchor_line"
    assert gate_payload["branch_only_rungs"] == ["D4"]
    assert d4_entry["branch_only"] is True
    assert d4_entry["branch_eligibility"]["UNITREE_G1"]["status"] == "BLOCK"
    assert (
        d4_entry["branch_eligibility"]["UNITREE_G1"]["redirect_branch"]
        == "NEW_EMBODIMENT"
    )
    assert d4_rung["admission_status"] == "BLOCK"
    assert (
        "branch_only_dataset:LightwheelAI/Lightwheel-Tasks-G1-Controller"
        in d4_rung["admission_reason_codes"]
    )


def test_new_embodiment_cli_materializes_branch_only_d4_as_internal_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "artifacts"

    exit_code = gr00t_d_ladder_policy_gate.main(
        ["--branch", "NEW_EMBODIMENT", "--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    gate_payload = _gate(output_dir, "NEW_EMBODIMENT")
    d4_rung = next(
        item for item in gate_payload["dataset_rungs"] if item["rung"] == "D4"
    )

    assert exit_code == 0
    assert captured.err == ""
    assert payload["branch_scope"] == "branch_internal_only"
    assert payload["public_anchor_comparable"] is False
    assert gate_payload["branch_only_rungs"] == ["D4"]
    assert d4_rung["admission_status"] == "REQUIRES_ADMISSION_EVIDENCE"
    assert (
        d4_rung["mixing_eligibility"]["LightwheelAI/Lightwheel-Tasks-G1-Controller"][
            "status"
        ]
        == "ALLOW_WITH_GUARDRAILS"
    )


def test_admission_blocks_hidden_normalization_drift() -> None:
    dataset_fingerprints, dataset_provenance, normalization_records = _admission_inputs(
        "D3"
    )
    normalization_records["LightwheelAI/Lightwheel-Tasks-G1-WBC"] = {
        **normalization_records["LightwheelAI/Lightwheel-Tasks-G1-WBC"],
        "hidden_stats_fingerprint": "drifted-hidden-stats",
    }

    report = gr00t_d_ladder_policy_gate.build_admission_report(
        branch="NEW_EMBODIMENT",
        rung="D3",
        dataset_fingerprints=dataset_fingerprints,
        dataset_provenance=dataset_provenance,
        normalization_records=normalization_records,
    )

    assert report["admission_status"] == "BLOCK"
    assert (
        "hidden_normalization_drift:LightwheelAI/Lightwheel-Tasks-G1-WBC"
        in report["reason_codes"]
    )
    assert report["checks"]["normalization_guardrails_pass"] is False


def test_admission_blocks_missing_dataset_provenance() -> None:
    dataset_fingerprints, dataset_provenance, normalization_records = _admission_inputs(
        "D2"
    )
    del dataset_provenance["nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1"]

    report = gr00t_d_ladder_policy_gate.build_admission_report(
        branch="NEW_EMBODIMENT",
        rung="D2",
        dataset_fingerprints=dataset_fingerprints,
        dataset_provenance=dataset_provenance,
        normalization_records=normalization_records,
    )

    assert report["admission_status"] == "BLOCK"
    assert (
        "missing_dataset_provenance:nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1"
        in report["reason_codes"]
    )
    assert report["checks"]["dataset_provenance_complete"] is False


def test_d4_branch_only_dataset_cannot_enter_unitree_trunk() -> None:
    dataset_fingerprints, dataset_provenance, normalization_records = _admission_inputs(
        "D4"
    )

    report = gr00t_d_ladder_policy_gate.build_admission_report(
        branch="UNITREE_G1",
        rung="D4",
        dataset_fingerprints=dataset_fingerprints,
        dataset_provenance=dataset_provenance,
        normalization_records=normalization_records,
    )

    assert report["admission_status"] == "BLOCK"
    assert (
        "branch_only_dataset:LightwheelAI/Lightwheel-Tasks-G1-Controller"
        in report["reason_codes"]
    )
    assert (
        "redirect_to_new_embodiment:LightwheelAI/Lightwheel-Tasks-G1-Controller"
        in report["reason_codes"]
    )


def test_d4_can_pass_on_new_embodiment_with_explicit_evidence() -> None:
    dataset_fingerprints, dataset_provenance, normalization_records = _admission_inputs(
        "D4"
    )

    report = gr00t_d_ladder_policy_gate.build_admission_report(
        branch="NEW_EMBODIMENT",
        rung="D4",
        dataset_fingerprints=dataset_fingerprints,
        dataset_provenance=dataset_provenance,
        normalization_records=normalization_records,
    )

    assert report["admission_status"] == "PASS"
    assert report["checks"]["dataset_fingerprint_complete"] is True
    assert report["checks"]["dataset_provenance_complete"] is True
    assert report["checks"]["normalization_guardrails_pass"] is True
