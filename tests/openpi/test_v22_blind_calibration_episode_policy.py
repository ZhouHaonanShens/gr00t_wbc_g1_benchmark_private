from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COORDINATOR = (
    REPO_ROOT / "agent/artifacts/stage1_v22_blind_calibration_iter8_20260426T_nextZ/coordinator"
)
INPUT_CONTRACT = COORDINATOR / "w6_iter8_input_contract.json"
EARLY_STOP_POLICY = REPO_ROOT / "work/openpi/eval/configs/v22_early_stop_policy_default.json"


def test_episode_policy_flags_thread_into_config_and_precondition(tmp_path: Path) -> None:
    from work.openpi.eval import v22_blind_calibration_runner as runner
    from work.openpi.eval.v22_calibration_contracts import coerce_episode_policy

    args = runner.build_parser().parse_args(
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
            "dry-run",
            "--calibration-variants",
            "A",
            "--optional-control-variants",
            "B",
            "--episodes-per-cell-A",
            "7",
            "--episodes-per-cell-B",
            "5",
            "--episodes-per-cell-smoke",
            "3",
            "--b-scan-policy",
            "all_cells",
            "--early-stop-policy",
            str(EARLY_STOP_POLICY),
            "--no-c-results",
            "--no-x-results",
            "--no-sudo",
        ]
    )

    config = runner.config_from_args(args)
    policy = coerce_episode_policy(config)
    precondition, _contract = runner.validate_preconditions(config)

    assert policy.episodes_per_cell_A == 7
    assert policy.episodes_per_cell_B == 5
    assert policy.episodes_per_cell_smoke == 3
    assert policy.b_scan_policy == "all_cells"
    assert precondition["episode_policy"] == {
        "episodes_per_cell_A": 7,
        "episodes_per_cell_B": 5,
        "episodes_per_cell_smoke": 3,
        "b_scan_policy": "all_cells",
    }

