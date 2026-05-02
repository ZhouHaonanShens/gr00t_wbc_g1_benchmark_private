from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any
from collections.abc import Sequence

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import state_conditioned_build_training_set
from work.recap.scripts import state_conditioned_train
from tests.recap import test_state_conditioned_sft_labels as sft_label_fixtures


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_flag_value(cmd: Sequence[str], flag: str) -> str:
    for index, token in enumerate(cmd):
        if token == flag:
            return str(cmd[index + 1])
        if token.startswith(flag + "="):
            return str(token.split("=", 1)[1])
    raise AssertionError(f"missing flag {flag!r} in delegated command: {cmd!r}")


def _build_training_set_root(tmp_path: Path) -> Path:
    bucket_dir, dev_dir, collection_dir, harvest_dir = (
        sft_label_fixtures._build_full_fixture(tmp_path)
    )
    output_dir = tmp_path / "training_set"
    state_conditioned_build_training_set.materialize_state_conditioned_training_set(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        collection_dir=collection_dir,
        harvest_dir=harvest_dir,
        output_dir=output_dir,
    )
    return output_dir


def _build_args(*cli_args: str) -> tuple[Any, list[str]]:
    parser = state_conditioned_train.build_parser()
    return parser.parse_known_args(list(cli_args))


def _fake_runner_factory(
    *,
    drift_variant: str | None = None,
    drift_field: str | None = None,
    drift_value: Any = None,
    multi_checkpoint_variant: str | None = None,
):
    def _runner(cmd: Sequence[str], cwd: Path, summary_path: Path) -> dict[str, Any]:
        del cwd
        dataset_path = Path(_parse_flag_value(cmd, "--dataset-path")).resolve()
        output_dir = Path(_parse_flag_value(cmd, "--output-dir")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        variant = (
            "c0" if "checkpoint_C0_equal_data_control" in str(output_dir) else "c1"
        )
        checkpoint_path: Path | None = None
        if "--dry-run" not in cmd:
            checkpoint_path = (
                output_dir / f"checkpoint-{_parse_flag_value(cmd, '--save-steps')}"
            )
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            (checkpoint_path / "model.safetensors").write_text(
                f"weights-for-{variant}",
                encoding="utf-8",
            )
            if multi_checkpoint_variant == variant:
                extra = output_dir / "checkpoint-1"
                extra.mkdir(parents=True, exist_ok=True)
                (extra / "model.safetensors").write_text("extra", encoding="utf-8")

        summary: dict[str, Any] = {
            "wrapper_status": "ok",
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
            "selected_checkpoint_path": None
            if checkpoint_path is None
            else str(checkpoint_path),
            "effective_config": {
                "base_model": _parse_flag_value(cmd, "--base-model"),
                "base_model_revision": _parse_flag_value(cmd, "--base-model-revision")
                if any(token == "--base-model-revision" for token in cmd)
                else "",
                "embodiment_tag": _parse_flag_value(cmd, "--embodiment-tag"),
                "max_steps": int(_parse_flag_value(cmd, "--max-steps")),
                "save_steps": int(_parse_flag_value(cmd, "--save-steps")),
                "save_total_limit": int(_parse_flag_value(cmd, "--save-total-limit")),
                "global_batch_size": int(_parse_flag_value(cmd, "--global-batch-size")),
                "gradient_accumulation_steps": int(
                    _parse_flag_value(cmd, "--gradient-accumulation-steps")
                ),
                "dataloader_num_workers": int(
                    _parse_flag_value(cmd, "--dataloader-num-workers")
                ),
                "learning_rate": float(_parse_flag_value(cmd, "--learning-rate")),
                "num_gpus": int(_parse_flag_value(cmd, "--num-gpus")),
                "tune_projector": "--tune-projector" in cmd,
                "tune_diffusion_model": "--tune-diffusion-model" in cmd,
                "use_wandb": "--use-wandb" in cmd,
            },
        }
        if drift_variant == variant and drift_field is not None:
            if drift_field == "dataset_path":
                summary["dataset_path"] = str(drift_value)
            else:
                effective_config = dict(summary["effective_config"])
                effective_config[str(drift_field)] = drift_value
                summary["effective_config"] = effective_config
        _write_json(summary_path, summary)
        return summary

    return _runner


def _build_happy_path_metadata_pair(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training_root = _build_training_set_root(tmp_path / "fixtures")
    output_root = tmp_path / "training_runs"
    args, forwarded = _build_args(
        "--training-set-root",
        str(training_root),
        "--output-dir",
        str(output_root),
    )
    state_conditioned_train.materialize_state_conditioned_training(
        args=args,
        forwarded=forwarded,
        kernel_runner=_fake_runner_factory(),
    )
    c0_metadata = _read_json(
        output_root / state_conditioned_train.RUN_METADATA_BASENAME_BY_VARIANT["c0"]
    )
    c1_metadata = _read_json(
        output_root / state_conditioned_train.RUN_METADATA_BASENAME_BY_VARIANT["c1"]
    )
    return c0_metadata, c1_metadata


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        state_conditioned_train.main(["--help"])
    assert exc_info.value.code == 0


def test_happy_path_runs_c0_c1_and_enforces_whitelist_only_differences(
    tmp_path: Path,
) -> None:
    training_root = _build_training_set_root(tmp_path / "fixtures")
    lerobot_dataset_path = Path(
        _read_json(
            training_root
            / state_conditioned_build_training_set.STATE_CONDITIONED_SFT_STATS_JSON_NAME
        )["lerobot_dataset_path"]
    ).resolve()
    output_root = tmp_path / "training_runs"
    args, forwarded = _build_args(
        "--training-set-root",
        str(training_root),
        "--output-dir",
        str(output_root),
    )

    result = state_conditioned_train.materialize_state_conditioned_training(
        args=args,
        forwarded=forwarded,
        kernel_runner=_fake_runner_factory(),
    )

    diff_payload = _read_json(Path(result["diff_whitelist_path"]))
    c0_metadata = _read_json(
        output_root / state_conditioned_train.RUN_METADATA_BASENAME_BY_VARIANT["c0"]
    )
    c1_metadata = _read_json(
        output_root / state_conditioned_train.RUN_METADATA_BASENAME_BY_VARIANT["c1"]
    )

    assert result["baseline_trained"] is False
    assert result["diff_whitelist_status"] == "PASS"
    assert diff_payload["status"] == "PASS"
    assert set(diff_payload["observed_difference_paths"]) == {
        "conditioning_enabled",
        "null_phase_mode_token_enabled",
        "output_dir",
        "checkpoint_rule.selected_checkpoint_path",
    }
    assert c0_metadata["comparable_run_spec"]["conditioning_enabled"] is False
    assert c0_metadata["comparable_run_spec"]["null_phase_mode_token_enabled"] is True
    assert c1_metadata["comparable_run_spec"]["conditioning_enabled"] is True
    assert c1_metadata["comparable_run_spec"]["null_phase_mode_token_enabled"] is False
    assert (
        c0_metadata["comparable_run_spec"]["training_budget"]["tune_diffusion_model"]
        is False
    )
    assert (
        c1_metadata["comparable_run_spec"]["training_budget"]["tune_diffusion_model"]
        is False
    )
    assert (
        Path(c0_metadata["comparable_run_spec"]["dataset_path"]).resolve()
        == lerobot_dataset_path
    )
    assert (
        Path(c1_metadata["comparable_run_spec"]["dataset_path"]).resolve()
        == lerobot_dataset_path
    )
    assert (
        c0_metadata["comparable_run_spec"]["source_data"][
            "equal_data_fairness_audit_path"
        ]
        == c1_metadata["comparable_run_spec"]["source_data"][
            "equal_data_fairness_audit_path"
        ]
    )
    assert (
        output_root / "checkpoint_C0_equal_data_control" / "checkpoint-100"
    ).is_dir()
    assert (output_root / "checkpoint_C1_phase_mode" / "checkpoint-100").is_dir()


def test_tune_diffusion_model_defaults_off_but_allows_explicit_override() -> None:
    default_args, _ = _build_args()
    assert default_args.tune_diffusion_model is False

    explicit_on_args, _ = _build_args("--tune-diffusion-model")
    assert explicit_on_args.tune_diffusion_model is True

    explicit_off_args, _ = _build_args("--no-tune-diffusion-model")
    assert explicit_off_args.tune_diffusion_model is False


def test_training_contract_uses_lerobot_dataset_path_instead_of_flat_root(
    tmp_path: Path,
) -> None:
    training_root = _build_training_set_root(tmp_path / "fixtures")
    contract = state_conditioned_train._load_training_set_contract(training_root)
    stats = _read_json(
        training_root
        / state_conditioned_build_training_set.STATE_CONDITIONED_SFT_STATS_JSON_NAME
    )

    assert Path(contract["training_set_root"]).resolve() == training_root.resolve()
    assert (
        Path(contract["dataset_path"]).resolve()
        == Path(stats["lerobot_dataset_path"]).resolve()
    )
    assert Path(contract["dataset_path"]).resolve() != training_root.resolve()


def test_training_contract_rejects_missing_lerobot_meta_info(tmp_path: Path) -> None:
    training_root = _build_training_set_root(tmp_path / "fixtures")
    stats_path = (
        training_root
        / state_conditioned_build_training_set.STATE_CONDITIONED_SFT_STATS_JSON_NAME
    )
    stats = _read_json(stats_path)
    info_path = Path(stats["lerobot_dataset_path"]) / "meta" / "info.json"
    info_path.unlink()

    with pytest.raises(ValueError, match=r"lerobot_dataset_path/meta/info.json"):
        state_conditioned_train._load_training_set_contract(training_root)


def test_budget_drift_failure(tmp_path: Path) -> None:
    c0_metadata, c1_metadata = _build_happy_path_metadata_pair(tmp_path)
    drifted = copy.deepcopy(c1_metadata)
    drifted["comparable_run_spec"]["training_budget"]["max_steps"] += 1

    with pytest.raises(ValueError, match=r"training_budget.max_steps"):
        state_conditioned_train.validate_diff_whitelist_or_raise(c0_metadata, drifted)


def test_source_data_drift_failure(tmp_path: Path) -> None:
    c0_metadata, c1_metadata = _build_happy_path_metadata_pair(tmp_path)
    drifted = copy.deepcopy(c1_metadata)
    drifted["comparable_run_spec"]["dataset_fingerprint"] = "fingerprint-drift"

    with pytest.raises(ValueError, match=r"dataset fingerprint drifted"):
        state_conditioned_train.validate_diff_whitelist_or_raise(c0_metadata, drifted)


def test_sampling_seed_drift_failure(tmp_path: Path) -> None:
    c0_metadata, c1_metadata = _build_happy_path_metadata_pair(tmp_path)
    drifted = copy.deepcopy(c1_metadata)
    drifted["comparable_run_spec"]["sampling"]["seed"] += 1

    with pytest.raises(ValueError, match=r"sampling.seed"):
        state_conditioned_train.validate_diff_whitelist_or_raise(c0_metadata, drifted)


def test_checkpoint_retention_drift_failure(tmp_path: Path) -> None:
    training_root = _build_training_set_root(tmp_path / "fixtures")
    output_root = tmp_path / "training_runs"
    args, forwarded = _build_args(
        "--training-set-root",
        str(training_root),
        "--output-dir",
        str(output_root),
    )

    with pytest.raises(ValueError, match=r"expected exactly one retained checkpoint"):
        state_conditioned_train.materialize_state_conditioned_training(
            args=args,
            forwarded=forwarded,
            kernel_runner=_fake_runner_factory(multi_checkpoint_variant="c1"),
        )
