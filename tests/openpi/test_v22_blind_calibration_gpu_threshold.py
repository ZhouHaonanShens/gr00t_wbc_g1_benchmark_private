from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COORDINATOR = (
    REPO_ROOT / "agent/artifacts/stage1_v22_blind_calibration_iter8_20260426T_nextZ/coordinator"
)
INPUT_CONTRACT = COORDINATOR / "w6_iter8_input_contract.json"
EARLY_STOP_POLICY = REPO_ROOT / "work/openpi/eval/configs/v22_early_stop_policy_default.json"


def test_calibrate_blocks_when_gpu2_memory_exceeds_contract_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from work.openpi.eval import v22_blind_calibration_runner as runner

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(runner, "_read_gpu2_memory_used_mib", lambda: (501, None))
    monkeypatch.setattr(
        runner,
        "_suite_probe_import_resolution",
        lambda: {
            "targets": ["openpi.policies.libero_policy", "libero.libero.benchmark"],
            "status": "PASS",
            "failures": [],
        },
    )

    config = runner.config_from_args(
        runner.build_parser().parse_args(
            [
                "--input-contract",
                str(INPUT_CONTRACT),
                "--input-contract-sha256",
                INPUT_CONTRACT.with_name(f"{INPUT_CONTRACT.name}.sha256").read_text(
                    encoding="utf-8"
                ).strip(),
                "--output-dir",
                str(tmp_path / "out"),
                "--runtime-log-dir",
                str(tmp_path / "logs"),
                "--mode",
                "calibrate",
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
                "--gpu2-memory-threshold-mib",
                "500",
                "--early-stop-policy",
                str(EARLY_STOP_POLICY),
                "--no-c-results",
                "--no-x-results",
                "--no-sudo",
            ]
        )
    )

    precondition, _contract = runner.validate_preconditions(config)

    assert precondition["status"] == "BLOCK"
    assert precondition["gpu2_memory"]["threshold_mib"] == 500
    assert precondition["gpu2_memory"]["observed_mib"] == 501
    assert "BLOCK_GPU2_MEMORY_NOT_IDLE" in precondition["blocking_reasons"]

