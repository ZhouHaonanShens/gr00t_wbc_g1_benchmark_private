from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HASH_LOCK = (
    REPO_ROOT
    / "agent/artifacts/stage1_v22_blind_calibration_iter8_20260426T_nextZ/openpi/v22_preregistration/v22_preregistration_hash_lock.json"
)


def _precondition_reasons(tmp_path: Path, *extra_args: str) -> list[str]:
    from work.openpi.eval import v22_formal_eval_runner as runner

    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--prereg-hash-lock",
            str(HASH_LOCK),
            "--output-dir",
            str(tmp_path / "out"),
            "--runtime-log-dir",
            str(tmp_path / "runtime"),
            "--mode",
            "dry-run",
            "--no-sudo",
            "--cuda-visible-devices",
            "2",
            *extra_args,
        ]
    )
    config = runner.config_from_args(args)
    precondition, _lock, _manifest = runner.validate_preconditions(config)
    return list(precondition["blocking_reasons"])  # type: ignore[arg-type]


def test_rejects_n_per_variant_mutation(tmp_path: Path) -> None:
    reasons = _precondition_reasons(
        tmp_path,
        "--n-per-variant",
        "96",
        "--episodes-per-variant",
        "96",
    )

    assert "BLOCK_PROTOCOL_N_PER_VARIANT_MUTATION" in reasons


def test_rejects_selected_protocol_mutations(tmp_path: Path) -> None:
    reasons = _precondition_reasons(
        tmp_path,
        "--suite",
        "libero_goal",
        "--budget",
        "0.6",
        "--step-cap",
        "111",
    )

    assert "BLOCK_PROTOCOL_SUITE_MUTATION" in reasons
    assert "BLOCK_PROTOCOL_BUDGET_MUTATION" in reasons
    assert "BLOCK_PROTOCOL_STEP_CAP_MUTATION" in reasons

