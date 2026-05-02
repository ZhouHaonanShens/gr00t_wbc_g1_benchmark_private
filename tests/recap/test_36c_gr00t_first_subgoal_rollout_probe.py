from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = REPO_ROOT / "work/recap/scripts/36c_gr00t_first_subgoal_rollout_probe.py"
    spec = importlib.util.spec_from_file_location(
        "gr00t_first_subgoal_rollout_probe_36c_for_tests",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_cli_flag_parsing(tmp_path: Path) -> None:
    args = MODULE.build_arg_parser().parse_args(
        [
            "--conditioned-checkpoint",
            str(tmp_path / "checkpoint-200"),
            "--continuation-run-root",
            str(tmp_path / "continuation"),
            "--baseline-probe",
            str(tmp_path / "baseline_first_subgoal_probe_v1.json"),
            "--seed",
            "20260421",
            "--output-json",
            str(tmp_path / "first_subgoal_probe.json"),
            "--telemetry-dir",
            str(tmp_path / "telemetry"),
            "--runtime-log-dir",
            str(tmp_path / "runtime_logs"),
            "--no-sudo",
        ]
    )

    assert args.conditioned_checkpoint == tmp_path / "checkpoint-200"
    assert args.continuation_run_root == tmp_path / "continuation"
    assert args.baseline_probe == tmp_path / "baseline_first_subgoal_probe_v1.json"
    assert args.seed == 20260421
    assert args.output_json == tmp_path / "first_subgoal_probe.json"
    assert args.telemetry_dir == tmp_path / "telemetry"
    assert args.runtime_log_dir == tmp_path / "runtime_logs"
    assert args.no_sudo is True
    assert args.n_episodes == 30
    assert args.dry_run_smoke is False


def test_dry_run_smoke_emits_smoke_gate(tmp_path: Path) -> None:
    output_json = tmp_path / "probe" / "first_subgoal_probe.json"
    exit_code = MODULE.main(
        [
            "--conditioned-checkpoint",
            str(tmp_path / "checkpoint-200"),
            "--baseline-probe",
            str(tmp_path / "baseline_first_subgoal_probe_v1.json"),
            "--seed",
            "20260421",
            "--output-json",
            str(output_json),
            "--telemetry-dir",
            str(tmp_path / "telemetry"),
            "--runtime-log-dir",
            str(tmp_path / "runtime_logs"),
            "--dry-run-smoke",
        ]
    )

    smoke_gate = output_json.parent / "smoke_gate.json"
    payload = json.loads(smoke_gate.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(MODULE.DRY_RUN_TELEMETRY_SENTINEL).hexdigest()

    assert exit_code == 0
    assert payload == {
        "schema_version": "smoke_gate_v1",
        "status": "PASS",
        "telemetry_sha256": expected_hash,
    }
    assert not output_json.exists()
