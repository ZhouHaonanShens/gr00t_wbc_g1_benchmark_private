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


def _base_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    hash_lock = tmp_path / "hash_lock.json"
    anchor = tmp_path / "canonical_training_anchor.json"
    dataset = tmp_path / "dataset"
    warm_start = tmp_path / "warm_start"
    _write_json(anchor, _anchor_payload())
    dataset.mkdir()
    warm_start.mkdir()
    return hash_lock, anchor, dataset, warm_start


def _args(
    tmp_path: Path,
    *,
    hash_lock: Path,
    anchor: Path,
    dataset: Path,
    warm_start: Path | str,
) -> list[str]:
    return [
        "--variant",
        "B",
        "--variant-id",
        "control_no_recap_shuffled_adversarial_relabel",
        "--prereg-hash-lock",
        str(hash_lock),
        "--canonical-anchor",
        str(anchor),
        "--output-dir",
        str(tmp_path / "B"),
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


def test_hash_lock_mutation_blocks_before_export(monkeypatch, tmp_path: Path) -> None:
    from work.openpi.pipelines.recap import v22_train_variants_real as trainer

    def fail_export(_request):
        raise AssertionError("export must not run when hash-lock validation blocks")

    monkeypatch.setattr(trainer, "run_real_variant_training_export", fail_export)
    hash_lock, anchor, dataset, warm_start = _base_paths(tmp_path)
    payload = _hash_lock_payload()
    payload["n_per_variant"] = 96
    _write_json(hash_lock, payload)

    rc = trainer.main(
        _args(
            tmp_path,
            hash_lock=hash_lock,
            anchor=anchor,
            dataset=dataset,
            warm_start=warm_start,
        )
    )

    precondition = json.loads((tmp_path / "B" / "precondition_check.json").read_text())
    assert rc == 4
    assert precondition["status"] == "BLOCK"
    assert "BLOCK_HASH_LOCK_N_PER_VARIANT" in precondition["blocking_reasons"]
    assert "BLOCK_HASH_LOCK_N_PER_VARIANT_MISMATCH" in precondition["blocking_reasons"]


def test_media_warm_start_blocks_before_export(monkeypatch, tmp_path: Path) -> None:
    from work.openpi.pipelines.recap import v22_train_variants_real as trainer

    def fail_export(_request):
        raise AssertionError("export must not run for legacy /media warm-start")

    monkeypatch.setattr(trainer, "run_real_variant_training_export", fail_export)
    hash_lock, anchor, dataset, _warm_start = _base_paths(tmp_path)
    _write_json(hash_lock, _hash_lock_payload())

    rc = trainer.main(
        _args(
            tmp_path,
            hash_lock=hash_lock,
            anchor=anchor,
            dataset=dataset,
            warm_start="/media/howard/Data/old_checkpoint",
        )
    )

    precondition = json.loads((tmp_path / "B" / "precondition_check.json").read_text())
    assert rc == 4
    assert precondition["status"] == "BLOCK"
    assert "BLOCK_WARM_START_LEGACY_MEDIA_ROOT" in precondition["blocking_reasons"]

