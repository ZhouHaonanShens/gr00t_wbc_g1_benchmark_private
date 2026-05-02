from __future__ import annotations

import json
from pathlib import Path

import pytest


POLICY_PATH = Path("work/openpi/eval/configs/v22_early_stop_policy_default.json")


def test_early_stop_policy_schema_version_matches() -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "v22_calibration_early_stop_policy_v1"
    assert payload["allowed_inputs"] == ["A", "optional_B"]
    assert payload["forbidden_inputs"] == ["C", "X"]
    assert payload["forbidden_early_stop_reason_word_tokens"] == ["C", "X"]


def test_early_stop_policy_allowed_inputs_subset_of_a_and_optional_b() -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        load_early_stop_policy,
    )

    policy = load_early_stop_policy(POLICY_PATH)

    assert set(policy["allowed_inputs"]) <= {"A", "optional_B"}
    assert policy["allowed_early_stop_reasons"] == [
        "A_success_rate_clearly_above_headroom_max",
        "A_success_rate_clearly_below_headroom_min",
        "trace_completeness_too_low",
        "timeout_rate_too_high",
    ]


@pytest.mark.parametrize(
    "reason",
    [
        "selection_using_C_aggregate",
        "selection_using_X_aggregate",
        "headroom_from_C",
        "X_timeout_comparison",
    ],
)
def test_early_stop_reason_referencing_c_or_x_is_rejected(reason: str) -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        validate_early_stop_reason,
    )

    with pytest.raises(ValueError, match="C|X"):
        validate_early_stop_reason(reason)


@pytest.mark.parametrize(
    "reason",
    [
        "A_success_rate_clearly_above_headroom_max",
        "trace_completeness_too_low",
        "timeout_rate_too_high",
        "client_timeout_excess",
    ],
)
def test_early_stop_reason_allows_benign_words_with_c_or_x(reason: str) -> None:
    from work.openpi.pipelines.recap.blind_calibration_runtime import (
        validate_early_stop_reason,
    )

    validate_early_stop_reason(reason)


def test_per_cell_timeout_distinct_from_client_and_overall_timeouts() -> None:
    from work.openpi.eval import v22_blind_calibration_runner as runner

    config = runner.config_from_args(
        runner.build_parser().parse_args(
            [
                "--input-contract",
                "agent/artifacts/stage1_v22_calibration_iter7_20260426T_nextZ/coordinator/w6_iter7_input_contract.json",
                "--input-contract-sha256",
                "3b19b6ac911c5a124972d58138b3398cc6d4585e30eea5fb42094c78fd191a1b",
                "--output-dir",
                "agent/artifacts/stage1_v22_runner_surface_iter7_5_20260426T_nextZ/openpi/v22_blind_calibration_dry_run",
                "--runtime-log-dir",
                "agent/runtime_logs/stage1_v22_runner_surface_iter7_5_20260426T_nextZ/w3_dry_run",
                "--mode",
                "dry-run",
                "--no-c-results",
                "--no-x-results",
                "--no-sudo",
                "--per-cell-timeout-sec",
                "1800",
                "--early-stop-policy",
                "work/openpi/eval/configs/v22_early_stop_policy_default.json",
            ]
        )
    )

    assert config.per_cell_timeout_sec == 1800
    assert hasattr(config, "per_cell_timeout_sec")
    assert not hasattr(config, "client_timeout_sec")
    assert not hasattr(config, "overall_timeout_sec")
