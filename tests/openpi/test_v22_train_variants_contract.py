from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_lock_payload() -> dict[str, object]:
    return {
        "schema_version": "v22_preregistration_hash_lock_v1",
        "run_id": "stage1_v22_blind_calibration_iter8_20260426T_nextZ",
        "n_per_variant": 192,
        "selected_using_c_results": False,
        "selected_using_x_results": False,
        "iter5_r2_sha256": "0f4a1b74152a8e4f88c7259b0033e0f262257a1222e1a6fb5436f084547e7e69",
        "iter5_r4_sha256": "c08da923e96c6d2d6f1f6b2522219eee7f3ab5b6851f880e7502b3fadd7af965",
        "variants": ["A", "B", "C", "X"],
        "selected_protocol": {
            "suite": "libero_spatial",
            "budget": 0.5,
            "cell_id": "libero_spatial__budget_0_50",
            "step_cap": 110,
            "max_steps_full": 220,
            "tasks": ["all_tasks_round_robin_episode_index_modulo_10"],
        },
    }


def _base_args(
    tmp_path: Path,
    *,
    variant: str = "B",
    variant_id: str = "control_no_recap_shuffled_adversarial_relabel",
    cuda_visible_devices: str = "1",
) -> list[str]:
    hash_lock = tmp_path / "hash_lock.json"
    warm_start = tmp_path / "warm_start"
    _write_json(hash_lock, _hash_lock_payload())
    warm_start.mkdir()
    return [
        "--variant",
        variant,
        "--variant-id",
        variant_id,
        "--prereg-hash-lock",
        str(hash_lock),
        "--output-dir",
        str(tmp_path / "out"),
        "--runtime-log-dir",
        str(tmp_path / "logs"),
        "--warm-start-checkpoint",
        str(warm_start),
        "--emit-loss-decomposition",
        "--emit-threshold-trace",
        "--emit-gradient-attestation",
        "--emit-control-signal-absence-attestation",
        "--emit-sha256sums",
        "--no-sudo",
        "--cuda-visible-devices",
        cuda_visible_devices,
    ]


def test_cli_surface_exposes_required_flags() -> None:
    from work.openpi.pipelines.recap import v22_train_variants as trainer

    help_text = trainer.build_parser().format_help()

    required_flags = (
        "--variant",
        "--variant-id",
        "--prereg-hash-lock",
        "--output-dir",
        "--runtime-log-dir",
        "--warm-start-checkpoint",
        "--enable-r2-phase-threshold-switching",
        "--enable-r4-alpha-dual-loss",
        "--emit-loss-decomposition",
        "--emit-threshold-trace",
        "--emit-alpha-dual-trace",
        "--emit-gradient-attestation",
        "--emit-shuffle-manifest",
        "--emit-deterministic-shuffle-provenance",
        "--emit-control-signal-absence-attestation",
        "--emit-sha256sums",
        "--no-sudo",
        "--cuda-visible-devices",
    )
    for flag in required_flags:
        assert flag in help_text


@pytest.mark.parametrize("variant", ["A", "D"])
def test_parser_rejects_non_training_variants(tmp_path: Path, variant: str) -> None:
    from work.openpi.pipelines.recap import v22_train_variants as trainer

    with pytest.raises(SystemExit):
        trainer.build_parser().parse_args(_base_args(tmp_path, variant=variant))


def test_main_emits_contract_artifacts(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap import v22_train_variants as trainer

    rc = trainer.main(_base_args(tmp_path))

    output_dir = tmp_path / "out"
    assert rc == 0
    assert json.loads((output_dir / "precondition_check.json").read_text())[
        "schema_version"
    ] == "v22_variant_train_precondition_v1"
    assert json.loads((output_dir / "training_run_manifest.json").read_text())[
        "schema_version"
    ] == "v22_variant_train_manifest_v1"
    assert json.loads((output_dir / "checkpoint_provenance.json").read_text())[
        "schema_version"
    ] == "v22_variant_checkpoint_provenance_v1"
    assert (output_dir / "loss_decomposition.jsonl").is_file()
    assert (output_dir / "threshold_switch_trace.jsonl").is_file()
    assert json.loads((output_dir / "gradient_attestation.json").read_text())[
        "schema_version"
    ] == "v22_variant_gradient_attestation_v1"
    assert json.loads(
        (output_dir / "control_signal_absence_attestation.json").read_text()
    )["schema_version"] == "v22_variant_control_signal_absence_v1"
    assert (output_dir / "checkpoint" / "checkpoint.json").is_file()
    assert (output_dir / "SHA256SUMS").is_file()
    assert (tmp_path / "logs" / "v22_train_variants.log").is_file()


def test_c_variant_accepts_gpu2_dispatch_override(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap import v22_train_variants as trainer

    args = _base_args(
        tmp_path,
        variant="C",
        variant_id="main_recap_method",
        cuda_visible_devices="2",
    )
    args.remove("--emit-control-signal-absence-attestation")
    args.extend(
        [
            "--enable-r2-phase-threshold-switching",
            "--enable-r4-alpha-dual-loss",
            "--emit-alpha-dual-trace",
        ]
    )

    rc = trainer.main(args)

    output_dir = tmp_path / "out"
    precondition = json.loads((output_dir / "precondition_check.json").read_text())
    manifest = json.loads((output_dir / "training_run_manifest.json").read_text())
    alpha_summary = json.loads(
        (output_dir / "alpha_term_contribution_summary.json").read_text()
    )
    assert rc == 0
    assert precondition["status"] == "PASS"
    assert manifest["cuda_visible_devices"] == "2"
    assert alpha_summary["alpha_term_contribution_nonzero_or_explained"] is True


def test_b_variant_still_rejects_gpu2_dispatch(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap import v22_train_variants as trainer

    rc = trainer.main(_base_args(tmp_path, cuda_visible_devices="2"))

    precondition = json.loads(
        (tmp_path / "out" / "precondition_check.json").read_text()
    )
    assert rc == 4
    assert precondition["status"] == "BLOCK"
    assert "BLOCK_CUDA_VISIBLE_DEVICES_BOUNDARY" in precondition["blocking_reasons"]
