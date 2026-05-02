from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from configs.apple_recap.flux import train_smoke_cpu
from work.recap.scripts import gr00t_recap_training_smoke


def _build_args(*cli_args: str):
    return gr00t_recap_training_smoke.build_parser().parse_args(list(cli_args))


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gr00t_recap_training_smoke.main(["--help"])
    assert exc_info.value.code == 0


def test_cpu_config_freezes_tiny_dry_run_diagnostic_lane() -> None:
    config: dict[str, Any] = train_smoke_cpu.build_config()

    assert config["smoke_mode"] == "cpu"
    assert config["execution"]["dry_run"] is True
    assert config["execution"]["max_steps"] == 1
    assert config["execution"]["num_gpus"] == 0
    assert config["execution"]["save_total_limit"] == 1
    assert config["trainable_surface"]["preferred"] == "head_only"
    assert config["trainable_surface"]["head_only"]["tune_top_llm_layers"] == 0
    assert config["trainable_surface"]["head_only"]["tune_vlln"] is True


def test_cpu_materialization_emits_non_authoritative_dry_run_summary(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    args = _build_args(
        "--dataset-path",
        str(dataset_dir),
        "--output-dir",
        str(tmp_path / "output"),
        "--runtime-log-dir",
        str(tmp_path / "runtime_logs"),
        "--summary-json",
        str(tmp_path / "summary.json"),
    )
    config_module, profile = gr00t_recap_training_smoke.load_training_smoke_config(
        smoke_mode=str(args.smoke_mode),
        config_module=str(args.config_module),
    )

    payload = gr00t_recap_training_smoke.materialize_flux_training_smoke(
        args=args,
        profile=profile,
        config_module=config_module,
        repo_root=REPO_ROOT,
    )

    assert payload["wrapper_status"] == "ok"
    assert payload["smoke_mode"] == "cpu"
    assert payload["dry_run"] is True
    assert payload["trainable_surface"] == "head_only_dry_run"
    assert payload["requested_trainable_surface"] == "head_only"
    assert payload["diagnostic_only"] is True
    assert payload["mainline_authority"] is False
    assert payload["main_verdict_eligible"] is False
    assert payload["external_reference_only"] is True
    assert payload["gate_semantics"] == "diagnostic_only_non_release_gate"
    assert payload["max_steps"] == 1
    assert payload["num_gpus"] == 0
    assert payload["save_total_limit"] == 1
    assert payload["selected_checkpoint_path"] is None
    assert payload["selected_checkpoint_asset_path"] is None
    assert payload["delegate_cmd"] is not None
