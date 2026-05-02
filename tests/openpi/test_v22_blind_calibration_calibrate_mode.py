from __future__ import annotations

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


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "--input-contract",
        str(INPUT_CONTRACT),
        "--input-contract-sha256",
        _contract_sha(),
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
        "--early-stop-policy",
        str(EARLY_STOP_POLICY),
        "--no-c-results",
        "--no-x-results",
        "--no-sudo",
    ]


def test_old_iter7_calibrate_guard_string_is_removed() -> None:
    source = (REPO_ROOT / "work/openpi/eval/v22_blind_calibration_runner.py").read_text(
        encoding="utf-8"
    )

    assert "calibrate_mode_guarded_off_for_iter7_5" not in source


def test_calibrate_precondition_can_pass_without_old_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from work.openpi.eval import v22_blind_calibration_runner as runner

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(runner, "_read_gpu2_memory_used_mib", lambda: (0, None))
    monkeypatch.setattr(
        runner,
        "_suite_probe_import_resolution",
        lambda: {
            "targets": ["openpi.policies.libero_policy", "libero.libero.benchmark"],
            "status": "PASS",
            "failures": [],
        },
    )

    config = runner.config_from_args(runner.build_parser().parse_args(_base_args(tmp_path)))
    precondition, _contract = runner.validate_preconditions(config)

    assert precondition["status"] == "PASS"
    assert "calibrate_mode_guarded_off_for_iter7_5" not in precondition["blocking_reasons"]


def test_calibrate_precondition_emits_block_code_for_c_x_leakage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from work.openpi.eval import v22_blind_calibration_runner as runner

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(runner, "_read_gpu2_memory_used_mib", lambda: (0, None))
    monkeypatch.setattr(
        runner,
        "_suite_probe_import_resolution",
        lambda: {
            "targets": ["openpi.policies.libero_policy", "libero.libero.benchmark"],
            "status": "PASS",
            "failures": [],
        },
    )
    args = _base_args(tmp_path)
    args[args.index("B")] = "C"

    config = runner.config_from_args(runner.build_parser().parse_args(args))
    precondition, _contract = runner.validate_preconditions(config)

    assert precondition["status"] == "BLOCK"
    assert "BLOCK_C_X_LEAKAGE" in precondition["blocking_reasons"]

