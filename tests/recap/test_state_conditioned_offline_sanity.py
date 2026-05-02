from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import state_conditioned_offline_sanity


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        state_conditioned_offline_sanity.main(["--help"])
    assert exc_info.value.code == 0


def test_materialize_offline_sanity_happy_path_writes_machine_readable_report(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "sanity"

    result = state_conditioned_offline_sanity.materialize_offline_sanity(output_dir)

    report_path = output_dir / state_conditioned_offline_sanity.REPORT_JSON_NAME
    report = _read_json(report_path)
    assert result == report
    assert result["status"] == "PASS"
    assert result["failure"] is None
    assert result["output_dir"] == str(output_dir.resolve())
    assert result["report_path"] == str(report_path)
    assert result["summary"] == {"passed_check_count": 5, "total_check_count": 5}
    assert list(result["checks"].keys()) == list(
        state_conditioned_offline_sanity.CHECK_ORDER
    )
    assert all(check["status"] == "PASS" for check in result["checks"].values())
    assert result["checks"]["sidecar_round_trip"]["phase_values"] == [
        "APPROACH",
        "GRASP",
        "PLACE",
        "SEARCH",
        "TRANSPORT",
        "VERIFY_HOLD",
    ]
    assert result["checks"]["sidecar_round_trip"]["mode_values"] == [
        "NOMINAL",
        "RECOVERY",
    ]
    assert result["checks"]["history_window_padding_reset_boundary"][
        "first_step_valid_counts"
    ] == {
        "offline_episode_000": 1,
        "offline_episode_001": 1,
    }
    assert result["checks"]["phase_mode_parsing"]["mixed_case_inputs_verified"] is True
    assert (
        result["checks"]["label_round_trip"]["scale_rule"]
        == "sign_aware_quantile_by_sign_v1"
    )
    assert (
        result["checks"]["exporter_round_trip"]["advantage_input_column"]
        == "recap_m2.advantage_input"
    )


@pytest.mark.parametrize(
    ("mismatch_mode", "expected_stage", "error_fragment"),
    [
        ("sidecar", "sidecar_round_trip", "sidecar join mismatch"),
        (
            "history",
            "history_window_padding_reset_boundary",
            "history_t_std_indices mismatch",
        ),
        ("exporter", "exporter_round_trip", "Advantage input mismatch"),
    ],
)
def test_round_trip_mismatch_modes_fail_machine_readably(
    tmp_path: Path,
    mismatch_mode: str,
    expected_stage: str,
    error_fragment: str,
) -> None:
    output_dir = tmp_path / mismatch_mode

    result = state_conditioned_offline_sanity.materialize_offline_sanity(
        output_dir,
        mismatch_mode=mismatch_mode,
    )

    report = _read_json(output_dir / state_conditioned_offline_sanity.REPORT_JSON_NAME)
    assert result == report
    assert result["status"] == "FAIL"
    assert result["failure"]["stage"] == expected_stage
    assert error_fragment in result["failure"]["message"]
    assert result["checks"][expected_stage]["status"] == "FAIL"
    assert error_fragment in result["checks"][expected_stage]["error"]


def test_main_failure_path_has_clean_stderr_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure_payload = {
        "schema_version": state_conditioned_offline_sanity.SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_offline_sanity_report",
        "status": "FAIL",
        "output_dir": str((tmp_path / "sanity").resolve()),
        "repo_root": str(REPO_ROOT),
        "iter_tag": state_conditioned_offline_sanity.ITER_TAG,
        "failure": {
            "stage": "sidecar_round_trip",
            "type": "ValueError",
            "message": "sidecar join mismatch: missing=[('offline_episode_001', 2)] extra=[]",
        },
        "checks": {
            name: {
                "name": name,
                "passed": None,
                "status": "NOT_RUN",
            }
            for name in state_conditioned_offline_sanity.CHECK_ORDER
        },
        "summary": {"passed_check_count": 0, "total_check_count": 5},
        "report_path": str(
            (
                tmp_path / "sanity" / state_conditioned_offline_sanity.REPORT_JSON_NAME
            ).resolve()
        ),
    }

    monkeypatch.setattr(
        state_conditioned_offline_sanity,
        "materialize_offline_sanity",
        lambda output_dir: dict(failure_payload),
    )

    exit_code = state_conditioned_offline_sanity.main(
        ["--output-dir", str(tmp_path / "sanity")]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "sidecar join mismatch" in captured.err
    assert "Traceback" not in captured.err
    stdout_payload = json.loads(captured.out)
    assert stdout_payload["status"] == "FAIL"
    assert stdout_payload["failure"]["stage"] == "sidecar_round_trip"
