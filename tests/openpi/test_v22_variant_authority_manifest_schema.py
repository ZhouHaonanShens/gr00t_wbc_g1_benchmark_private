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


def _variant_row(
    tmp_path: Path,
    *,
    variant_id: str,
    role: str,
    trained_after: bool | str,
    loss: bool,
    threshold: bool,
    alpha: bool,
    shuffle: bool,
    control_absence: bool,
) -> dict[str, object]:
    from work.openpi.pipelines.recap.v22_training_contracts import sha256_path

    root = tmp_path / variant_id
    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    _write_json(checkpoint / "checkpoint.json", {"variant_id": variant_id})
    train_manifest = root / "train_manifest.json"
    _write_json(train_manifest, {"variant_id": variant_id})
    (root / "SHA256SUMS").write_text("fixture\n", encoding="utf-8")
    return {
        "variant_id": variant_id,
        "role": role,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "train_manifest_path": str(train_manifest),
        "train_manifest_sha256": sha256_path(train_manifest),
        "loss_decomposition_present": loss,
        "threshold_switch_trace_present": threshold,
        "alpha_dual_loss_trace_present": alpha,
        "shuffle_manifest_present": shuffle,
        "control_signal_absence_attestation_present": control_absence,
        "trained_after_r2_r4_closed": trained_after,
        "no_legacy_media_root": True,
        "sha256sums_present": True,
    }


def test_variant_authority_manifest_round_trip_and_gate_pass(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap.v22_training_contracts import (
        build_variant_authority_manifest,
        evaluate_variant_authority_manifest,
        load_v22_hash_lock,
        verify_manifest_self_hash,
        write_variant_authority_manifest,
    )

    hash_lock_path = tmp_path / "hash_lock.json"
    no_leakage = tmp_path / "no_c_x_leakage_attestation.json"
    _write_json(hash_lock_path, _hash_lock_payload())
    _write_json(no_leakage, {"status": "PASS"})
    lock = load_v22_hash_lock(hash_lock_path)
    rows = [
        _variant_row(
            tmp_path,
            variant_id="A_stock_pi0_libero",
            role="stock",
            trained_after="not_applicable",
            loss=False,
            threshold=False,
            alpha=False,
            shuffle=False,
            control_absence=False,
        ),
        _variant_row(
            tmp_path,
            variant_id="control_no_recap_shuffled_adversarial_relabel",
            role="B_control",
            trained_after=True,
            loss=True,
            threshold=True,
            alpha=False,
            shuffle=False,
            control_absence=True,
        ),
        _variant_row(
            tmp_path,
            variant_id="main_recap_method",
            role="C_recap",
            trained_after=True,
            loss=True,
            threshold=True,
            alpha=True,
            shuffle=False,
            control_absence=False,
        ),
        _variant_row(
            tmp_path,
            variant_id="recap_variant_shuffle_diag",
            role="X_shuffle_diag",
            trained_after=True,
            loss=True,
            threshold=True,
            alpha=True,
            shuffle=True,
            control_absence=False,
        ),
    ]

    manifest = build_variant_authority_manifest(
        hash_lock=lock,
        no_c_x_leakage_attestation_path=no_leakage,
        variants=rows,
    )
    output = tmp_path / "variant_authority_manifest.json"
    write_variant_authority_manifest(output, manifest)
    round_tripped = json.loads(output.read_text(encoding="utf-8"))

    assert round_tripped["schema_version"] == "variant_authority_manifest_v22_v1"
    assert round_tripped["formal_eval_allowed"] is True
    assert evaluate_variant_authority_manifest(round_tripped).formal_eval_allowed is True
    assert verify_manifest_self_hash(round_tripped)


def test_variant_authority_manifest_gate_reports_predicate_failure(tmp_path: Path) -> None:
    from work.openpi.pipelines.recap.v22_training_contracts import (
        build_variant_authority_manifest,
        evaluate_variant_authority_manifest,
        load_v22_hash_lock,
    )

    hash_lock_path = tmp_path / "hash_lock.json"
    no_leakage = tmp_path / "no_c_x_leakage_attestation.json"
    _write_json(hash_lock_path, _hash_lock_payload())
    _write_json(no_leakage, {"status": "PASS"})
    lock = load_v22_hash_lock(hash_lock_path)
    rows = [
        _variant_row(
            tmp_path,
            variant_id="A_stock_pi0_libero",
            role="stock",
            trained_after="not_applicable",
            loss=False,
            threshold=False,
            alpha=False,
            shuffle=False,
            control_absence=False,
        ),
        _variant_row(
            tmp_path,
            variant_id="control_no_recap_shuffled_adversarial_relabel",
            role="B_control",
            trained_after=True,
            loss=True,
            threshold=True,
            alpha=False,
            shuffle=False,
            control_absence=True,
        ),
        _variant_row(
            tmp_path,
            variant_id="main_recap_method",
            role="C_recap",
            trained_after=True,
            loss=True,
            threshold=True,
            alpha=False,
            shuffle=False,
            control_absence=False,
        ),
        _variant_row(
            tmp_path,
            variant_id="recap_variant_shuffle_diag",
            role="X_shuffle_diag",
            trained_after=True,
            loss=True,
            threshold=True,
            alpha=True,
            shuffle=True,
            control_absence=False,
        ),
    ]

    manifest = build_variant_authority_manifest(
        hash_lock=lock,
        no_c_x_leakage_attestation_path=no_leakage,
        variants=rows,
    )
    evaluation = evaluate_variant_authority_manifest(manifest)

    assert manifest["formal_eval_allowed"] is False
    assert evaluation.formal_eval_allowed is False
    assert "c_trace_predicates_missing" in evaluation.reasons
