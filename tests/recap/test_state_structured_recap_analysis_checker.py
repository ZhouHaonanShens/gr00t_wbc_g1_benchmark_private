from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import state_structured_recap_analysis_checker


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        state_structured_recap_analysis_checker.main(["--help"])
    assert exc_info.value.code == 0


def test_load_contract_spec_validates_expected_shape() -> None:
    _markdown, _raw_spec, spec = (
        state_structured_recap_analysis_checker.load_contract_spec(
            state_structured_recap_analysis_checker.DEFAULT_CONTRACT_PATH
        )
    )

    assert spec["schema_version"] == "state_structured_recap_analysis_only_v1"
    assert spec["container_path"] == "analysis_only.state_structured_recap"
    assert spec["leaf_field_order"] == [
        "edge_id",
        "candidate_edge_mask",
        "recovery_family",
        "semantic_commit",
        "failed_edge_to_recovery_edge",
    ]


def test_run_leak_negative_checks_rejects_all_fields() -> None:
    _markdown, _raw_spec, spec = (
        state_structured_recap_analysis_checker.load_contract_spec(
            state_structured_recap_analysis_checker.DEFAULT_CONTRACT_PATH
        )
    )

    result = state_structured_recap_analysis_checker.run_leak_negative_checks(spec)

    assert result["status"] == "PASS"
    assert result["field_count"] == 5
    assert len(result["deployable_contract_gate"]) == 5
    assert len(result["deployable_export_gate"]) == 5
    assert len(result["train_payload_gate"]) == 5
    assert result["train_container_probe"]["status"] == "REJECTED"
    assert (
        "analysis-only field leaked into train payload"
        in result["train_container_probe"]["message"]
    )
    assert all(
        "REJECTED" == row["status"] for row in result["deployable_contract_gate"]
    )
    assert all("REJECTED" == row["status"] for row in result["deployable_export_gate"])
    assert all("REJECTED" == row["status"] for row in result["train_payload_gate"])


def test_train_payload_validator_rejects_analysis_only_field() -> None:
    _markdown, _raw_spec, spec = (
        state_structured_recap_analysis_checker.load_contract_spec(
            state_structured_recap_analysis_checker.DEFAULT_CONTRACT_PATH
        )
    )

    with pytest.raises(
        ValueError, match="analysis-only field leaked into train payload"
    ):
        state_structured_recap_analysis_checker.validate_train_payload_field_names(
            ["schema_version", "edge_id"],
            spec=spec,
        )


def test_main_prints_pass_sentinel(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = state_structured_recap_analysis_checker.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert state_structured_recap_analysis_checker.PASS_SENTINEL in captured.out
    payload = json.loads(captured.out.splitlines()[0])
    assert payload["status"] == "PASS"
    assert payload["leak_negative"]["status"] == "PASS"
