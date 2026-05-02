from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_wrapper_module():
    path = REPO_ROOT / "work/recap/scripts/4X_recap_r2r4_closed_finetune.py"
    spec = importlib.util.spec_from_file_location(
        "gr00t_r2r4_closed_finetune_wrapper", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_feature_flags() -> list[str]:
    return [
        "--enable-r2-phase-threshold-switching",
        "--enable-r4-alpha-dual-loss",
        "--emit-loss-decomposition",
        "--emit-threshold-switch-trace",
        "--emit-alpha-dual-trace",
        "--emit-gradient-path-attestation",
        "--emit-dynamic-delta-audit",
        "--emit-checkpoint-sha256sums",
    ]


def _happy_contract_args(module, contract_output_dir: Path) -> list[str]:
    return [
        "--contract-output-dir",
        str(contract_output_dir),
        "--emit-contract-only",
        "--no-sudo",
        *_required_feature_flags(),
    ]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_flag_surface_and_direct_import_boundary() -> None:
    module = _load_wrapper_module()

    help_text = module._build_parser().format_help()
    for flag in [
        "--warm-start-checkpoint",
        "--output-dir",
        "--runtime-log-dir",
        "--max-steps",
        "--save-total-limit",
        "--no-sudo",
        "--cuda-visible-devices",
        *_required_feature_flags(),
    ]:
        assert flag in help_text

    source = inspect.getsource(module)
    assert "RecapFinetuneReproScriptApp" in source
    assert "import work.recap.finetune_full" not in source
    assert "from work.recap import finetune_full" not in source


def test_contract_only_emits_g1_contract_and_preflight(
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_wrapper_module()
    contract_dir = tmp_path / "training_runner_contract"

    exit_code = module.main(_happy_contract_args(module, contract_dir))

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    contract = _read_json(contract_dir / module.CONTRACT_JSON_NAME)
    preflight = _read_json(contract_dir / module.PREFLIGHT_JSON_NAME)

    assert exit_code == 0
    assert summary["artifact_result"]["preflight_status"] == "PASS"
    assert contract["schema_version"] == "gr00t_r2r4_training_runner_contract_v1"
    assert contract["common_warm_start"] == module.COMMON_WARM_START_REL.as_posix()
    assert contract["do_not_retry_same_checkpoint"] is True
    assert contract["required_features"] == {
        "loss_decomposition_required": True,
        "threshold_switch_trace_required": True,
        "alpha_dual_loss_trace_required": True,
        "dynamic_delta_audit_required": True,
        "sha256sums_required": True,
        "gradient_path_attestation_required": True,
    }
    assert contract["checkpoint_save_policy"] == {
        "save_total_limit": 1,
        "save_strategy": "steps",
        "save_steps": 1100,
    }
    assert contract["g2_protocol_checkpoint_save_policy"] == {
        "save_total_limit": 2,
        "save_strategy": "steps",
        "save_steps": 1100,
    }
    throughput_probe = contract["throughput_probe_spec"]
    assert throughput_probe["schema_version"] == "gr00t_g2_throughput_probe_spec_v1"
    assert throughput_probe["required_before_g2_full_launch"] is True
    assert throughput_probe["probe_steps"] == 50
    assert throughput_probe["cuda_visible_devices"] == "1"
    assert throughput_probe["no_sudo"] is True
    assert throughput_probe["checkpoint_save_policy"] == {
        "save_total_limit": 1,
        "save_strategy": "steps",
        "save_steps": 50,
    }
    assert throughput_probe["gate"]["metric"] == "estimated_steps_per_hour"
    assert throughput_probe["gate"]["minimum_steps_per_hour"] == round(2200 / 14, 6)
    assert throughput_probe["gate"]["failure_status"] == "BLOCK_INFRA_G2_THROUGHPUT"
    assert "--max-steps" in throughput_probe["command"]
    assert "50" in throughput_probe["command"]
    assert preflight["schema_version"] == "gr00t_training_preflight_v1"
    assert preflight["status"] == "PASS"
    assert preflight["checks"]["warm_start_matches_common_contract"] is True
    assert preflight["checks"]["cuda_visible_devices_is_gpu1"] is True
    assert preflight["checks"]["no_sudo"] is True
    assert preflight["checks"]["all_required_feature_flags_enabled"] is True
    assert preflight["checks"]["throughput_probe_spec_present"] is True
    assert preflight["checks"]["throughput_probe_steps_is_50"] is True
    assert preflight["checks"]["throughput_probe_gate_matches_14h_soft_cap"] is True
    assert preflight["throughput_probe_spec"] == throughput_probe


def test_preflight_blocks_unapproved_warm_start(
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_wrapper_module()
    bad_checkpoint = tmp_path / "checkpoint-20"
    bad_checkpoint.mkdir()
    (bad_checkpoint / "config.json").write_text("{}", encoding="utf-8")
    contract_dir = tmp_path / "blocked_contract"

    exit_code = module.main(
        [
            *_happy_contract_args(module, contract_dir),
            "--warm-start-checkpoint",
            str(bad_checkpoint),
        ]
    )

    capsys.readouterr()
    preflight = _read_json(contract_dir / module.PREFLIGHT_JSON_NAME)
    assert exit_code == 1
    assert preflight["status"] == "BLOCK"
    assert preflight["checks"]["warm_start_matches_common_contract"] is False
    assert preflight["checks"]["warm_start_exists"] is True


def test_delegate_argv_uses_recap_repro_app_surface() -> None:
    module = _load_wrapper_module()
    config = {
        "dataset_path": Path("/tmp/dataset"),
        "output_dir": Path("/tmp/output"),
        "max_steps": 2200,
        "save_steps": 1100,
        "save_total_limit": 2,
        "python": "",
    }

    argv = module._build_delegated_argv(config, dry_run=True)

    assert argv[0] == "34_recap_finetune_repro.py"
    assert "--dry-run" in argv
    assert "--dataset-path" in argv
    assert "--output-dir" in argv
    assert "--save-total-limit" in argv
    assert "2" in argv
