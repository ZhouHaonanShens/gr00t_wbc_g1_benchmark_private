from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HASH_LOCK = (
    REPO_ROOT
    / "agent/artifacts/stage1_v22_blind_calibration_iter8_20260426T_nextZ/openpi/v22_preregistration/v22_preregistration_hash_lock.json"
)
REQUIRED_FLAGS = (
    "--prereg-hash-lock",
    "--variant-authority-manifest",
    "--output-dir",
    "--runtime-log-dir",
    "--n-per-variant",
    "--variants",
    "--resume",
    "--skip-completed",
    "--cuda-visible-devices",
    "--no-sudo",
    "--mode",
    "--episodes-per-variant",
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_surface_exposes_required_flags() -> None:
    from work.openpi.eval import v22_formal_eval_runner as runner

    parser = runner.build_parser()
    help_text = parser.format_help()

    for flag in REQUIRED_FLAGS:
        assert flag in help_text

    mode_actions = [
        action for action in parser._actions if "--mode" in action.option_strings
    ]
    assert len(mode_actions) == 1
    assert set(mode_actions[0].choices) == {"dry-run", "smoke", "long-run"}


def test_dry_run_emits_contract_artifacts(tmp_path: Path) -> None:
    from work.openpi.eval import v22_formal_eval_runner as runner

    output_dir = tmp_path / "dry_run"
    runtime_dir = tmp_path / "runtime"
    exit_code = runner.main(
        [
            "--prereg-hash-lock",
            str(HASH_LOCK),
            "--output-dir",
            str(output_dir),
            "--runtime-log-dir",
            str(runtime_dir),
            "--mode",
            "dry-run",
            "--no-sudo",
            "--cuda-visible-devices",
            "2",
        ]
    )

    assert exit_code == 0
    precondition = _load_json(output_dir / "precondition_check.json")
    plan = _load_json(output_dir / "formal_eval_plan.json")
    requirements = _load_json(output_dir / "variant_manifest_requirements.json")

    assert precondition["schema_version"] == "v22_formal_eval_precondition_v1"
    assert precondition["paired_bootstrap_ci_helper_present"] is True
    assert "BLOCK_VARIANT_AUTHORITY_MANIFEST_MISSING" in precondition["blocking_reasons"]
    assert plan["schema_version"] == "v22_formal_eval_plan_v1"
    assert plan["variants"] == ["A", "B", "C", "X"]
    assert plan["n_per_variant"] == 192
    assert requirements["schema_version"] == "v22_variant_manifest_requirements_v1"
    assert requirements["formal_eval_allowed_required"] is True

