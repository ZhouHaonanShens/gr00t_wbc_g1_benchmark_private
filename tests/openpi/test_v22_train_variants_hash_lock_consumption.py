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


def test_hash_lock_load_is_byte_bound_and_mutation_rejected(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap.v22_training_contracts import (
        load_v22_hash_lock,
        validate_hash_lock,
    )

    hash_lock_path = tmp_path / "hash_lock.json"
    payload = _hash_lock_payload()
    _write_json(hash_lock_path, payload)
    original = load_v22_hash_lock(hash_lock_path)

    assert validate_hash_lock(original) == ()
    assert original.sha256
    assert original.selected_protocol.n_per_variant == 192

    payload["n_per_variant"] = 96
    _write_json(hash_lock_path, payload)
    mutated = load_v22_hash_lock(hash_lock_path)

    assert mutated.sha256 != original.sha256
    assert "BLOCK_HASH_LOCK_N_PER_VARIANT" in validate_hash_lock(mutated)


def test_warm_start_media_path_blocks_before_training(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap import v22_train_variants as trainer

    hash_lock = tmp_path / "hash_lock.json"
    _write_json(hash_lock, _hash_lock_payload())

    rc = trainer.main(
        [
            "--variant",
            "B",
            "--variant-id",
            "control_no_recap_shuffled_adversarial_relabel",
            "--prereg-hash-lock",
            str(hash_lock),
            "--output-dir",
            str(tmp_path / "out"),
            "--runtime-log-dir",
            str(tmp_path / "logs"),
            "--warm-start-checkpoint",
            "/media/howard/Data/old_checkpoint",
            "--emit-control-signal-absence-attestation",
            "--no-sudo",
            "--cuda-visible-devices",
            "1",
        ]
    )

    precondition = json.loads((tmp_path / "out" / "precondition_check.json").read_text())
    assert rc == 4
    assert precondition["status"] == "BLOCK"
    assert "BLOCK_WARM_START_LEGACY_MEDIA_ROOT" in precondition["blocking_reasons"]
