from __future__ import annotations

import json
from pathlib import Path


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


def _anchor_payload() -> dict[str, object]:
    return {
        "schema_version": "v22_canonical_training_anchor_v1",
        "run_id": "stage1_v22_full_training_eval_iter9_20260426T_nextZ",
        "source_config_name": "pi0_libero",
        "anchor_values": {
            "num_train_steps": 30000,
            "batch_size": 32,
            "seed": 42,
            "num_workers": 2,
            "log_interval_canonical": 100,
            "log_interval_v22_real_training_override": 1,
            "save_interval": 1000,
            "keep_period": 5000,
        },
    }


def test_stub_output_requires_archive_sibling_before_real_training(
    monkeypatch, tmp_path: Path
) -> None:
    from work.openpi.pipelines.recap import v22_train_variants_real as trainer

    def fail_export(_request):
        raise AssertionError("export must not run until prereg stub is archived")

    monkeypatch.setattr(trainer, "run_real_variant_training_export", fail_export)
    hash_lock = tmp_path / "hash_lock.json"
    anchor = tmp_path / "canonical_training_anchor.json"
    dataset = tmp_path / "dataset"
    warm_start = tmp_path / "warm_start"
    output_dir = tmp_path / "B"
    _write_json(hash_lock, _hash_lock_payload())
    _write_json(anchor, _anchor_payload())
    dataset.mkdir()
    warm_start.mkdir()
    output_dir.mkdir()
    (output_dir / "loss_decomposition.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "v22_variant_loss_decomposition_v1",
                "run_id": "stage1_v22_full_training_eval_iter9_20260426T_nextZ",
                "variant": "B",
                "step": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = trainer.main(
        [
            "--variant",
            "B",
            "--variant-id",
            "control_no_recap_shuffled_adversarial_relabel",
            "--prereg-hash-lock",
            str(hash_lock),
            "--canonical-anchor",
            str(anchor),
            "--output-dir",
            str(output_dir),
            "--runtime-log-dir",
            str(tmp_path / "logs"),
            "--warm-start-checkpoint",
            str(warm_start),
            "--dataset-dir",
            str(dataset),
            "--num-train-steps",
            "2",
            "--batch-size",
            "1",
            "--emit-loss-decomposition",
            "--emit-threshold-trace",
            "--emit-gradient-attestation",
            "--emit-control-signal-absence-attestation",
            "--no-sudo",
            "--cuda-visible-devices",
            "1",
        ]
    )

    precondition = json.loads((output_dir / "precondition_check.json").read_text())
    assert rc == 4
    assert precondition["status"] == "BLOCK"
    assert "BLOCK_STUB_ARCHIVE_REQUIRED" in precondition["blocking_reasons"]
    assert precondition["archive_precondition"]["stub_present"] is True
    assert precondition["archive_precondition"]["archive_sibling_exists"] is False

