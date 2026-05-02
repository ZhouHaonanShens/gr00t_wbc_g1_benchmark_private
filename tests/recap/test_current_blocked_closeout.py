from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import build_current_blocked_closeout


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_blocked_closeout_artifacts_match_canonical_blocked_surface() -> None:
    expected = build_current_blocked_closeout.build_current_blocked_closeout(
        repo_root=REPO_ROOT
    )
    uplift_verdict_path = (
        REPO_ROOT / "agent/artifacts/apple_recap_exec/uplift_verdict.json"
    )
    uplift_verdict_md_path = (
        REPO_ROOT / "agent/artifacts/apple_recap_exec/uplift_verdict.md"
    )
    block_reason_path = (
        REPO_ROOT
        / "agent/artifacts/apple_recap_exec/block_reasons/carrier_export_authority_violation.json"
    )

    uplift_verdict = _read_json(uplift_verdict_path)
    block_reason = _read_json(block_reason_path)
    uplift_verdict_md = uplift_verdict_md_path.read_text(encoding="utf-8")

    assert uplift_verdict == expected["uplift_verdict"]
    assert block_reason == expected["block_reason"]
    assert uplift_verdict_md == expected["uplift_verdict_markdown"]

    assert uplift_verdict["freshness"] == block_reason["freshness"]
    assert uplift_verdict["execution_sha"] == block_reason["execution_sha"]
    assert uplift_verdict["status"] == block_reason["status"] == "BLOCK"
    assert (
        uplift_verdict["terminal_state"] == block_reason["terminal_state"] == "blocked"
    )
    assert (
        uplift_verdict["authority_level"]
        == block_reason["authority_level"]
        == "blocked_closeout"
    )
    assert uplift_verdict["gating_eligible"] is False
    assert block_reason["gating_eligible"] is False
    assert uplift_verdict["requires_successor_execution"] is True
    assert uplift_verdict["current_execution_reopen_forbidden"] is True
    assert uplift_verdict["execution_verdict"] == "inconclusive_rerun_needed"
    assert uplift_verdict["theory_verdict"] == "not yet proven"
    assert uplift_verdict["block_stage"] == "T10_formal_carrier_parity"
    assert uplift_verdict["block_reason"] == "carrier_export_authority_violation"
    assert uplift_verdict["authority_violation_count"] == 61246
    assert uplift_verdict["full_scan_row_count"] == 61246
    assert uplift_verdict["missing_field"] == "carrier_text_v1"
    assert uplift_verdict["source_fields_present"] == ["prompt_raw", "indicator_I"]
    assert (
        block_reason["issue"]["field_path"] == "field_presence_summary.carrier_text_v1"
    )
    assert block_reason["issue"]["code"] == "missing_required_carrier_export"

    assert (
        uplift_verdict["freshness"]["execution_sha"]
        == "29d7396b51d5f3db1204f59df2e376ebd7e64ef9"
    )
    assert uplift_verdict["freshness"]["manifest_hash"] == (
        "3de2e772d69955993ae7acd2528e5046b4fb764228aa8d60e1e78e773553e401"
    )
    assert uplift_verdict["freshness"]["checkpoint_id"] == "phase_a_tooling_frozen"
    assert uplift_verdict["freshness"]["seed_bundle_id"] == "20000:20009"
    assert uplift_verdict["freshness"]["timestamp"] == "2026-04-12T04:27:21+00:00"

    assert [item["artifact_id"] for item in uplift_verdict["source_artifacts"]] == [
        "carrier_parity_report",
        "task_11_carrier_parity_evidence",
        "execution_freeze_contract",
        "b0_e1_e2_run_ledger",
    ]
    assert uplift_verdict["non_authoritative_inputs"][0]["artifact_id"] == (
        "draft_final_verdict_pack"
    )
    assert uplift_verdict["non_authoritative_inputs"][0]["authority_role"] == (
        "non_authoritative_draft_input"
    )
    assert uplift_verdict["non_authoritative_inputs"][0]["relative_path"] == (
        "agent/artifacts/apple_recap_exec/final_report/final_verdict_pack.json"
    )
    assert uplift_verdict["superseded_inputs"][0]["relative_path"] == (
        "agent/artifacts/apple_recap_exec/final_report/final_verdict_pack.json"
    )
    assert (
        "must not exercise mainline authority"
        in uplift_verdict["non_authoritative_inputs"][0]["reason"]
    )
    assert "不得被解释为“当前 execution 仍可继续 pending”" in uplift_verdict_md
