from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import cast

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import gr00t_baseline_freeze_matrix
from work.recap.scripts import gr00t_checkpoint_provenance_gate
from work.recap.scripts import gr00t_dual_branch_scorecard
from work.recap.scripts import gr00t_ladder_policy_gate
from work.recap.scripts import gr00t_public_anchor_eval
from work.recap.scripts import state_conditioned_oracle_eval


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "source"
    output = tmp_path / "output"

    public_anchor_formal = _write_json(
        source / "public_anchor_formal.json",
        {
            "schema_version": gr00t_public_anchor_eval.FORMAL_SCHEMA_VERSION,
            "artifact_kind": gr00t_public_anchor_eval.FORMAL_ARTIFACT_KIND,
            "success_rate": 0.5,
            "success_count": 5,
            "systemic_break_flags": [],
            "report_signature_sha256": "public-anchor-formal-signature",
        },
    )
    public_anchor_sanity_gate = _write_json(
        source / "public_anchor_sanity_gate.json",
        {
            "schema_version": gr00t_public_anchor_eval.SANITY_GATE_SCHEMA_VERSION,
            "artifact_kind": gr00t_public_anchor_eval.SANITY_GATE_ARTIFACT_KIND,
            "continue_to_audit": True,
            "public_anchor_comparable": True,
            "report_signature_sha256": "public-anchor-gate-signature",
        },
    )
    checkpoint_provenance_report = _write_json(
        source / "checkpoint_provenance_report.json",
        {
            "schema_version": gr00t_checkpoint_provenance_gate.REPORT_SCHEMA_VERSION,
            "artifact_kind": gr00t_checkpoint_provenance_gate.REPORT_ARTIFACT_KIND,
            "formal_eligibility": "ALLOW",
            "reason_code": "ok",
            "report_signature_sha256": "checkpoint-provenance-signature",
        },
    )
    dual_branch_scorecard = _write_json(
        source / "dual_branch_scorecard.json",
        {
            "schema_version": gr00t_dual_branch_scorecard.REPORT_SCHEMA_VERSION,
            "artifact_kind": gr00t_dual_branch_scorecard.REPORT_ARTIFACT_KIND,
            "branch_order": ["unitree_g1", "new_embodiment"],
            "official_comparable_line": "unitree_g1",
            "internal_only_comparable_line": "new_embodiment",
            "branches": [
                {
                    "branch_key": "unitree_g1",
                    "official_comparable_line": True,
                    "internal_only_comparable_line": False,
                },
                {
                    "branch_key": "new_embodiment",
                    "official_comparable_line": False,
                    "internal_only_comparable_line": True,
                },
            ],
            "report_signature_sha256": "dual-branch-signature",
        },
    )
    p_gate = _write_json(
        source / "p_ladder_policy_gate_unitree_g1.json",
        {
            "schema_version": gr00t_ladder_policy_gate.REPORT_SCHEMA_VERSION,
            "artifact_kind": gr00t_ladder_policy_gate.REPORT_ARTIFACT_KIND,
            "ladder_axis": "P",
            "branch_key": "unitree_g1",
            "change_policy": "PARAMETER_ONLY_WHITELIST",
            "report_signature_sha256": "p-gate-signature",
        },
    )
    d_gate = _write_json(
        source / "d_ladder_policy_gate_unitree_g1.json",
        {
            "schema_version": gr00t_ladder_policy_gate.REPORT_SCHEMA_VERSION,
            "artifact_kind": gr00t_ladder_policy_gate.REPORT_ARTIFACT_KIND,
            "ladder_axis": "D",
            "branch_key": "unitree_g1",
            "change_policy": "DATA_ONLY_WHITELIST_WITH_EXPLICIT_NORMALIZATION_DIFFS",
            "report_signature_sha256": "d-gate-signature",
        },
    )
    run_metadata_c1 = _write_json(
        source / "run_metadata_C1_phase_mode.json",
        {
            "schema_version": "state_conditioned_training_run_v1",
            "artifact_kind": "state_conditioned_training_run_metadata",
            "variant_key": "c1",
        },
    )
    legacy_c1_scorecard = _write_json(
        source / "oracle_conditioned_dev_scorecard.json",
        {
            "schema_version": state_conditioned_oracle_eval.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_oracle_conditioned_dev_scorecard",
            "line_order": list(state_conditioned_oracle_eval.LINE_ORDER),
            "line_labels": dict(state_conditioned_oracle_eval.LINE_LABELS),
            "training_run_metadata": {
                "c0": str(source / "run_metadata_C0_equal_data_control.json"),
                "c1": str(run_metadata_c1),
            },
            "lines": [
                {
                    "line_key": "baseline",
                    "line_label": state_conditioned_oracle_eval.LINE_LABELS["baseline"],
                    "model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
                    "oracle_phase_mode_supplied": False,
                },
                {
                    "line_key": "c0",
                    "line_label": state_conditioned_oracle_eval.LINE_LABELS["c0"],
                    "model_path": "/tmp/checkpoint_C0_equal_data_control/checkpoint-100",
                    "oracle_phase_mode_supplied": False,
                },
                {
                    "line_key": "c1",
                    "line_label": state_conditioned_oracle_eval.LINE_LABELS["c1"],
                    "model_path": "/tmp/checkpoint_C1_phase_mode/checkpoint-100",
                    "oracle_phase_mode_supplied": True,
                },
            ],
            "report_signature_sha256": "legacy-c1-scorecard-signature",
        },
    )
    legacy_result_split = _write_json(
        source / "result_split_decision.json",
        {
            "schema_version": state_conditioned_oracle_eval.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_result_split_decision",
            "next_step": "condition_interface_analysis",
            "branch_reason": "C1 backpointer is legacy-only and not part of the active G1 mainline authority.",
            "ab_case": "D",
            "report_signature_sha256": "legacy-c1-result-split-signature",
        },
    )
    return {
        "output": output / "baseline_freeze_matrix.json",
        "public_anchor_formal": public_anchor_formal,
        "public_anchor_sanity_gate": public_anchor_sanity_gate,
        "checkpoint_provenance_report": checkpoint_provenance_report,
        "dual_branch_scorecard": dual_branch_scorecard,
        "p_gate": p_gate,
        "d_gate": d_gate,
        "legacy_c1_scorecard": legacy_c1_scorecard,
        "legacy_result_split": legacy_result_split,
    }


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = gr00t_baseline_freeze_matrix.main(["--help"])
    assert exc_info.value.code == 0


def test_materialize_baseline_freeze_matrix_uses_namespaced_ids_and_backpointers(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)

    payload = gr00t_baseline_freeze_matrix.materialize_baseline_freeze_matrix(
        output=paths["output"],
        public_anchor_formal=paths["public_anchor_formal"],
        public_anchor_sanity_gate=paths["public_anchor_sanity_gate"],
        checkpoint_provenance_report=paths["checkpoint_provenance_report"],
        dual_branch_scorecard_json=paths["dual_branch_scorecard"],
        p_ladder_policy_gate_unitree=paths["p_gate"],
        d_ladder_policy_gate_unitree=paths["d_gate"],
        legacy_c1_scorecard=paths["legacy_c1_scorecard"],
        legacy_c1_result_split_decision=paths["legacy_result_split"],
    )
    written = _read_json(paths["output"])

    assert written == payload
    assert payload["artifact_kind"] == gr00t_baseline_freeze_matrix.REPORT_ARTIFACT_KIND
    assert (
        payload["schema_version"] == gr00t_baseline_freeze_matrix.REPORT_SCHEMA_VERSION
    )
    assert payload["official_comparable_line"] == "unitree_g1"
    assert payload["internal_only_comparable_line"] == "new_embodiment"
    assert payload["baseline_id_order"] == [
        gr00t_baseline_freeze_matrix.B0_BASELINE_ID,
        gr00t_baseline_freeze_matrix.B1_BASELINE_ID,
    ]
    assert payload["display_rows"] == [
        {
            "display_label": gr00t_baseline_freeze_matrix.DISPLAY_LABEL_B0,
            "baseline_id": gr00t_baseline_freeze_matrix.B0_BASELINE_ID,
        },
        {
            "display_label": gr00t_baseline_freeze_matrix.DISPLAY_LABEL_B1,
            "baseline_id": gr00t_baseline_freeze_matrix.B1_BASELINE_ID,
        },
    ]
    assert payload["machine_id_policy"]["display_labels_are_not_machine_ids"] is True
    assert payload["machine_id_policy"]["disallowed_machine_ids"] == ["B0", "B1"]
    assert "B0" not in payload["baselines"]
    assert "B1" not in payload["baselines"]

    b0 = payload["baselines"][gr00t_baseline_freeze_matrix.B0_BASELINE_ID]
    assert b0["display_label"] == "B0"
    assert b0["branch_key"] == "unitree_g1"
    assert b0["mainline_authority"] is True
    assert b0["parameter_baseline_rung"] == "P0"
    assert b0["data_baseline_rung"] == "D0"
    assert {row["artifact_id"] for row in b0["source_artifacts"]} == {
        "public_anchor_formal",
        "public_anchor_sanity_gate",
        "checkpoint_provenance_report",
        "dual_branch_scorecard",
        "p_ladder_policy_gate_unitree_g1",
        "d_ladder_policy_gate_unitree_g1",
    }

    b1 = payload["baselines"][gr00t_baseline_freeze_matrix.B1_BASELINE_ID]
    assert b1["display_label"] == "B1"
    assert b1["mainline_authority"] is False
    assert b1["legacy_backpointer_only"] is True
    assert b1["official_comparable_line"] is False
    assert b1["internal_only_comparable_line"] is False
    assert b1["promotion_to_active_mainline_authority_forbidden"] is True
    assert b1["legacy_line_backpointer"]["legacy_family"] == "oldworld_c1"
    assert b1["legacy_line_backpointer"]["line_key"] == "c1"
    assert b1["legacy_line_backpointer"]["line_index"] == 2
    assert (
        b1["legacy_line_backpointer"]["line_label"]
        == (state_conditioned_oracle_eval.LINE_LABELS["c1"])
    )
    assert b1["legacy_line_backpointer"]["training_run_metadata_path"] == str(
        paths["legacy_c1_scorecard"].parent / "run_metadata_C1_phase_mode.json"
    )
    assert {row["artifact_id"] for row in b1["source_artifacts"]} == {
        "legacy_c1_scorecard",
        "legacy_c1_result_split_decision",
    }


def test_materialize_baseline_freeze_matrix_is_no_overwrite_and_read_only(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    source_digests = {
        key: _sha256_file(path) for key, path in paths.items() if key != "output"
    }

    first_payload = gr00t_baseline_freeze_matrix.materialize_baseline_freeze_matrix(
        output=paths["output"],
        public_anchor_formal=paths["public_anchor_formal"],
        public_anchor_sanity_gate=paths["public_anchor_sanity_gate"],
        checkpoint_provenance_report=paths["checkpoint_provenance_report"],
        dual_branch_scorecard_json=paths["dual_branch_scorecard"],
        p_ladder_policy_gate_unitree=paths["p_gate"],
        d_ladder_policy_gate_unitree=paths["d_gate"],
        legacy_c1_scorecard=paths["legacy_c1_scorecard"],
        legacy_c1_result_split_decision=paths["legacy_result_split"],
    )
    first_output_digest = _sha256_file(paths["output"])

    with pytest.raises(ValueError, match="no-overwrite"):
        _ = gr00t_baseline_freeze_matrix.materialize_baseline_freeze_matrix(
            output=paths["output"],
            public_anchor_formal=paths["public_anchor_formal"],
            public_anchor_sanity_gate=paths["public_anchor_sanity_gate"],
            checkpoint_provenance_report=paths["checkpoint_provenance_report"],
            dual_branch_scorecard_json=paths["dual_branch_scorecard"],
            p_ladder_policy_gate_unitree=paths["p_gate"],
            d_ladder_policy_gate_unitree=paths["d_gate"],
            legacy_c1_scorecard=paths["legacy_c1_scorecard"],
            legacy_c1_result_split_decision=paths["legacy_result_split"],
        )

    assert _sha256_file(paths["output"]) == first_output_digest
    assert _read_json(paths["output"]) == first_payload
    assert {
        key: _sha256_file(path) for key, path in paths.items() if key != "output"
    } == source_digests


def test_legacy_c1_backpointer_cannot_masquerade_as_mainline_authority(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    tampered = _read_json(paths["legacy_c1_scorecard"])
    lines = list(cast(list[object], tampered["lines"]))
    c1_line = dict(cast(Mapping[str, object], lines[2]))
    c1_line["official_comparable_line"] = True
    lines[2] = c1_line
    tampered["lines"] = lines
    _ = _write_json(paths["legacy_c1_scorecard"], tampered)

    with pytest.raises(ValueError, match="must not masquerade"):
        _ = gr00t_baseline_freeze_matrix.materialize_baseline_freeze_matrix(
            output=paths["output"],
            public_anchor_formal=paths["public_anchor_formal"],
            public_anchor_sanity_gate=paths["public_anchor_sanity_gate"],
            checkpoint_provenance_report=paths["checkpoint_provenance_report"],
            dual_branch_scorecard_json=paths["dual_branch_scorecard"],
            p_ladder_policy_gate_unitree=paths["p_gate"],
            d_ladder_policy_gate_unitree=paths["d_gate"],
            legacy_c1_scorecard=paths["legacy_c1_scorecard"],
            legacy_c1_result_split_decision=paths["legacy_result_split"],
        )
