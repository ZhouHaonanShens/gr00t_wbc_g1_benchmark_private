from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COORDINATOR = (
    REPO_ROOT / "agent/artifacts/stage1_v22_blind_calibration_iter8_20260426T_nextZ/coordinator"
)
INPUT_CONTRACT = COORDINATOR / "w6_iter8_input_contract.json"
EARLY_STOP_POLICY = REPO_ROOT / "work/openpi/eval/configs/v22_early_stop_policy_default.json"


def _contract_sha() -> str:
    return INPUT_CONTRACT.with_name(f"{INPUT_CONTRACT.name}.sha256").read_text(
        encoding="utf-8"
    ).strip()


def _runner_cmd(output_dir: Path, runtime_log_dir: Path) -> list[str]:
    return [
        sys.executable,
        "work/openpi/eval/v22_blind_calibration_runner.py",
        "--input-contract",
        str(INPUT_CONTRACT),
        "--input-contract-sha256",
        _contract_sha(),
        "--output-dir",
        str(output_dir),
        "--runtime-log-dir",
        str(runtime_log_dir),
        "--mode",
        "calibrate",
        "--max-cells",
        "2",
        "--resume",
        "--skip-completed",
        "--calibration-variants",
        "A",
        "--optional-control-variants",
        "B",
        "--episodes-per-cell-A",
        "2",
        "--episodes-per-cell-B",
        "2",
        "--b-scan-policy",
        "headroom_eligible_only",
        "--per-cell-timeout-sec",
        "1800",
        "--early-stop-policy",
        str(EARLY_STOP_POLICY),
        "--no-c-results",
        "--no-x-results",
        "--no-sudo",
        "--gpu2-memory-threshold-mib",
        "500",
    ]


def _env(signal_dir: Path | None = None, *, sleep_s: str = "0") -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["V22_BLIND_CALIBRATION_TEST_GPU2_MEMORY_MIB"] = "0"
    env["V22_BLIND_CALIBRATION_TEST_SUITE_PROBE_PASS"] = "1"
    env["V22_BLIND_CALIBRATION_TEST_STUB_POLICY"] = "deterministic_resume"
    env["V22_BLIND_CALIBRATION_TEST_FIXED_UTC"] = "2026-04-27T00:00:00Z"
    env["V22_BLIND_CALIBRATION_TEST_EPISODE_SLEEP_SEC"] = sleep_s
    if signal_dir is not None:
        env["V22_BLIND_CALIBRATION_TEST_SIGNAL_DIR"] = str(signal_dir)
    return env


def _wait_for_second_cell_marker(signal_dir: Path) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if list(signal_dir.glob("1_*.episode_0.started")):
            return
        time.sleep(0.05)
    raise AssertionError("runner did not enter cell 2 before timeout")


def _run_to_completion(output_dir: Path, runtime_log_dir: Path) -> None:
    completed = subprocess.run(
        _runner_cmd(output_dir, runtime_log_dir),
        cwd=REPO_ROOT,
        env=_env(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_deterministic_resume_byte_equal_decision_and_headroom(tmp_path: Path) -> None:
    interrupted_out = tmp_path / "interrupted"
    interrupted_logs = tmp_path / "interrupted_logs"
    clean_out = tmp_path / "clean"
    clean_logs = tmp_path / "clean_logs"
    signal_dir = tmp_path / "signals"

    process = subprocess.Popen(
        _runner_cmd(interrupted_out, interrupted_logs),
        cwd=REPO_ROOT,
        env=_env(signal_dir, sleep_s="0.2"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_for_second_cell_marker(signal_dir)
    process.terminate()
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=10)
    assert process.returncode != 0

    _run_to_completion(interrupted_out, interrupted_logs)
    _run_to_completion(clean_out, clean_logs)

    for filename in (
        "desaturation_selection_decision.json",
        "headroom_eligibility_pre_b_scan.json",
    ):
        assert (interrupted_out / filename).read_bytes() == (clean_out / filename).read_bytes()

    summary = json.loads(
        (
            interrupted_out
            / "cells"
            / "libero_spatial_expanded__budget_0_10"
            / "stock_A"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    rows = [
        json.loads(line)
        for line in (
            interrupted_out
            / "cells"
            / "libero_spatial_expanded__budget_0_10"
            / "stock_A"
            / "per_episode_trace.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["synthetic_policy"] is True
    assert summary["stock_A_success_rate_source"] == "synthetic_test_stub_not_real_policy"
    assert summary["stock_A_success_rate_source"] != "real_policy"
    assert {row["policy_output_source"] for row in rows} == {"synthetic_test_stub"}

