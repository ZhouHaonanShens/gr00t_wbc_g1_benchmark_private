from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import state_conditioned_contract_gate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate() -> dict[str, Any]:
    return copy.deepcopy(
        state_conditioned_contract_gate.build_reference_contract_example()
    )


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        state_conditioned_contract_gate.main(["--help"])
    assert exc_info.value.code == 0


def test_main_writes_schema_complete_freeze_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "freeze"

    exit_code = state_conditioned_contract_gate.main(["--output-dir", str(output_dir)])

    assert exit_code == 0
    freeze = _read_json(output_dir / state_conditioned_contract_gate.FREEZE_JSON_NAME)
    report = _read_json(
        output_dir / state_conditioned_contract_gate.CONTRACT_GATE_REPORT_JSON_NAME
    )
    fsm = _read_json(
        output_dir / state_conditioned_contract_gate.PHASE_MODE_FSM_JSON_NAME
    )

    assert freeze["baseline_dataset"] == {
        "kind": "dataset_dir",
        "value": str(
            Path(
                state_conditioned_contract_gate.state_conditioned_bucket_a_import.DEFAULT_SOURCE
            ).resolve()
        ),
    }
    assert freeze["stable_base_checkpoint"] == {
        "kind": "model_path",
        "value": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
    }
    assert freeze["history_contract"]["history_k"] == 8
    assert freeze["history_contract"]["history_stride"] == 1
    assert freeze["deployable_history_allowlist"] == list(
        state_conditioned_contract_gate.DEPLOYABLE_HISTORY_ALLOWLIST
    )
    assert freeze["analysis_only_fields"] == list(
        state_conditioned_contract_gate.ANALYSIS_ONLY_FIELDS
    )
    assert freeze["policy_text_allowlist"] == ["phase", "mode"]
    assert "history." not in freeze["deployable_denylist"]["prefixes"]

    assert [row["name"] for row in fsm["phases"]] == list(
        state_conditioned_contract_gate.PHASE_VOCAB
    )
    assert [row["name"] for row in fsm["modes"]] == list(
        state_conditioned_contract_gate.MODE_VOCAB
    )
    assert fsm["allowed_phase_transitions"]["SEARCH"] == ["SEARCH", "APPROACH"]
    assert fsm["reset_clear_semantics"]["reset_boundary"] == "no_cross_episode"
    assert fsm["reset_clear_semantics"]["history_valid_mask_required"] is True

    assert report["contract_gate"] == {
        "name": "ContractGate",
        "passed": True,
        "status": "PASS",
    }
    assert report["checks"]["freeze_payload"]["passed"] is True
    assert report["checks"]["phase_mode_fsm"]["passed"] is True
    assert report["checks"]["reference_contract_example"]["passed"] is True


def test_main_rejects_non_directory_output_path_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_output_path = tmp_path / "freeze.json"
    bad_output_path.write_text("{}\n", encoding="utf-8")

    exit_code = state_conditioned_contract_gate.main(
        ["--output-dir", str(bad_output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "output-dir must be a directory path" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("field_name", "bad_value", "error_fragment"),
    [
        (
            "policy_condition.phase",
            "TRANSPORTING",
            "policy_condition.phase must be one of",
        ),
        (
            "policy_condition.mode",
            "teacher",
            "policy_condition.mode must be one of",
        ),
    ],
)
def test_validate_contract_candidate_rejects_illegal_phase_or_mode(
    field_name: str,
    bad_value: object,
    error_fragment: str,
) -> None:
    candidate = _candidate()
    candidate["deployable_observation"][field_name] = bad_value

    with pytest.raises(ValueError, match=error_fragment):
        state_conditioned_contract_gate.validate_contract_candidate(candidate)


def test_validate_contract_candidate_requires_history_valid_mask() -> None:
    candidate = _candidate()
    del candidate["deployable_observation"]["history_valid_mask"]

    with pytest.raises(ValueError, match="missing frozen fields: history_valid_mask"):
        state_conditioned_contract_gate.validate_contract_candidate(candidate)


def test_validate_contract_candidate_rejects_cross_episode_history() -> None:
    candidate = _candidate()
    candidate["history_context"]["history_episode_ids"][-1] = "episode_002"

    with pytest.raises(ValueError, match="cross-episode history is forbidden"):
        state_conditioned_contract_gate.validate_contract_candidate(candidate)


def test_validate_contract_candidate_rejects_future_timestamp_history() -> None:
    candidate = _candidate()
    candidate["deployable_observation"]["history_t_std_indices"][-1] = 8

    with pytest.raises(ValueError, match="future timestamp history is forbidden"):
        state_conditioned_contract_gate.validate_contract_candidate(candidate)


@pytest.mark.parametrize(
    ("field_name", "bad_value", "error_fragment"),
    [
        (
            "privileged.apple_pose_world",
            [1.0, 0.0, 0.0],
            "analysis-only field leaked into deployable observation",
        ),
        (
            "semantic_state",
            "VERIFYING_HOLD",
            "analysis-only field leaked into deployable observation",
        ),
        (
            "teacher.fallback_action",
            [0.1, 0.2],
            "forbidden prefix leaked into deployable observation",
        ),
        (
            "oracle.next_subgoal",
            "recover_grasp",
            "forbidden prefix leaked into deployable observation",
        ),
    ],
)
def test_validate_contract_candidate_rejects_deployable_observation_leakage(
    field_name: str,
    bad_value: object,
    error_fragment: str,
) -> None:
    candidate = _candidate()
    candidate["deployable_observation"][field_name] = bad_value

    with pytest.raises(ValueError, match=error_fragment):
        state_conditioned_contract_gate.validate_contract_candidate(candidate)


@pytest.mark.parametrize(
    "extra_text_line",
    [
        "SEMANTIC_STATE=VERIFYING_HOLD",
        "TEACHER_TRIGGER_REASON=oracle_fallback",
        "ORACLE_NEXT_SUBGOAL=recover_grasp",
    ],
)
def test_validate_contract_candidate_rejects_policy_text_leakage(
    extra_text_line: str,
) -> None:
    candidate = _candidate()
    candidate["policy_condition_text"] = (
        str(candidate["policy_condition_text"]) + "\n" + extra_text_line
    )

    with pytest.raises(
        ValueError,
        match=r"canonical \[phase,mode\]-only template",
    ):
        state_conditioned_contract_gate.validate_contract_candidate(candidate)
