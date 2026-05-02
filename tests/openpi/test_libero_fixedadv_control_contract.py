from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import work.openpi.pipelines.recap.variant_training as variant_train_script
from work.openpi.checkpoint import resolve_servable_checkpoint_ref
from work.openpi.recap.real_variant_export import (
    RealVariantExportBlockedError,
    RealVariantExportBundle,
)
from work.openpi.pipelines.recap.variant_training import main
from tests.openpi.carrier_text_v1_fixture import (  # noqa: E402
    carrier_text_v1_handoff_metadata,
)


DOC = REPO_ROOT / "agent/exchange/openpi_libero_fixedadv_control_contract.md"


def _mapping(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise TypeError(f"expected dict, got {type(raw).__name__}")
    return {str(key): value for key, value in raw.items()}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_minimal_recap_dataset(root: Path) -> tuple[Path, str, str]:
    dataset_dir = root / "physical_intelligence_libero_official_8d_recap_relabels_v1"
    _write_json(
        dataset_dir / "meta" / "info.json",
        {
            "schema_version": "openpi_libero_official_8d_recap_relabels_v1",
            "route_id": "official_native_8d_recap_relabels_v1",
            **carrier_text_v1_handoff_metadata(),
            "source_dataset_name": "physical_intelligence_libero_official_8d",
            "source_dataset_dir": str(
                (root / "physical_intelligence_libero_official_8d").resolve()
            ),
            "features": {
                "observation.images.ego_view": {
                    "dtype": "image",
                    "shape": [256, 256, 3],
                },
                "observation.state": {"dtype": "float32", "shape": [8]},
                "action": {"dtype": "float32", "shape": [7]},
                "annotation.human.task_description": {
                    "dtype": "int64",
                    "shape": [1],
                },
            },
            "recap_advantage_input_contract": {
                "contract_version": "full_recap_continuous_adv_v2"
            },
        },
    )
    _write_json(
        dataset_dir / "meta" / "modality.json",
        {
            "video": {
                "observation.images.ego_view": {
                    "original_key": "observation.images.ego_view"
                }
            },
            "state": {"observation.state": {}},
            "action": {"action": {}},
            "annotation": {"annotation.human.task_description": {}},
        },
    )
    _write_jsonl(
        dataset_dir / "meta" / "tasks.jsonl",
        [{"task": "put the bowl on the plate", "task_index": 0}],
    )
    fingerprint = hashlib.sha256(b"fixedadv-fixture-dataset").hexdigest()
    episode_universe_hash = hashlib.sha256(b"fixedadv-fixture-episodes").hexdigest()
    _write_json(
        dataset_dir / "meta" / "dataset_fingerprint.json",
        {
            "schema_version": "openpi_libero_relabel_dataset_fingerprint_v1",
            "route_id": "official_native_8d_recap_relabels_v1",
            "fingerprint_sha256": fingerprint,
        },
    )
    _ = (dataset_dir / "meta" / "episode_universe_hash.txt").write_text(
        episode_universe_hash + "\n",
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "action": [[0.1] * 7, [0.2] * 7],
            "episode_index": [0, 0],
            "observation.state": [[0.0] * 8, [1.0] * 8],
            "recap_m2.advantage_A": [0.5, -0.5],
            "recap_m2.advantage_input": [0.25, -0.25],
            "recap_m2.indicator_I": [1, 0],
            "recap_m2.prompt_conditioned": [
                "advantage positive put the bowl on the plate",
                "advantage negative put the bowl on the plate",
            ],
            "recap_m2.prompt_raw": [
                "put the bowl on the plate",
                "put the bowl on the plate",
            ],
            "recap_m2.return_G": [0.0, -1.0],
            "recap_m2.value_V": [-0.5, -0.5],
        }
    )
    parquet_path = dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    return dataset_dir, fingerprint, episode_universe_hash


def _write_minimal_real_export(root: Path) -> Path:
    checkpoint_dir = root / "real_variant_export"
    _ = (checkpoint_dir / "params" / "_METADATA").parent.mkdir(
        parents=True, exist_ok=True
    )
    _ = (checkpoint_dir / "params" / "_METADATA").write_text(
        '{"tree":"fixture"}\n', encoding="utf-8"
    )
    _ = (checkpoint_dir / "params" / "manifest.ocdbt").write_text(
        "fixture-ocdbt-manifest\n", encoding="utf-8"
    )
    _write_json(
        checkpoint_dir
        / "assets"
        / "physical-intelligence"
        / "libero"
        / "norm_stats.json",
        {
            "state": {
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
                "q01": [0.0] * 8,
                "q99": [1.0] * 8,
            },
            "actions": {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "q01": [0.0] * 7,
                "q99": [1.0] * 7,
            },
        },
    )
    _write_json(
        checkpoint_dir / "export_manifest.json",
        {
            "schema_version": "openpi_real_variant_export_v1",
            "source_checkpoint_dir": str(checkpoint_dir),
        },
    )
    return checkpoint_dir


def test_fixedadv_contract_freezes_variant_scope_and_authority() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "openpi LIBERO fixedadv matched-control contract",
        "variant=fixedadv_relabel8d_control_v1",
        "consumer_mode=fixedadv_constant",
        "B 不新建第二份 control dataset，而是直接复用 `physical_intelligence_libero_official_8d_recap_relabels_v1`。",
        "same relabeled official/native 8D dataset as recap_only_relabel8d_v2",
        "same prompt/IO shape/eval authority as recap_only_relabel8d_v2",
        "eval_authority=fresh_rollout_v2",
    ]
    for item in required:
        assert item in text, f"missing fixedadv scope item: {item}"


def test_fixedadv_contract_freezes_parity_with_recap_only() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "dataset_name=physical_intelligence_libero_official_8d_recap_relabels_v1",
        "dataset_route_id=official_native_8d_recap_relabels_v1",
        "base_checkpoint_id=pi05_libero_anchor",
        "train_budget_id=libero_cmp_budget_v2",
        "prompt_route=recap_conditioned_prompt_token_v1",
        "conditioning_mode=prompt_text_only",
        "source_prompt_field=prompt_raw",
        "observation.state.shape=[8]",
        "action.shape=[7]",
        "B/C 共用同一 relabeled official/native 8D dataset。",
        "B/C 共用同一 prompt/IO shape。",
        "B/C 共用同一 eval authority。",
        "B/C 共用同一 base checkpoint 与 train budget。",
    ]
    for item in required:
        assert item in text, f"missing fixedadv parity item: {item}"


def test_fixedadv_contract_freezes_omit_text_carrier_neutralization() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        'fixed_indicator_mode="omit"',
        "training prompt = prompt_raw only",
        "text carrier neutralized to prompt_raw only",
        '文本 carrier 通过 `fixed_indicator_mode="omit"` neutralize。',
        "训练 prompt 退化为 `prompt_raw` only。",
        "numeric_advantage_mode=constant_zero",
        "advantage provenance constant = 0.0",
    ]
    for item in required:
        assert item in text, f"missing fixedadv carrier item: {item}"


def test_fixedadv_contract_forbids_per_sample_advantage_consumption() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "must not consume per-sample recap_m2.indicator_I",
        "must not consume per-sample prompt_conditioned",
        "must not consume per-sample advantage_input",
        "must not build a second control dataset",
        "不得继续消费逐样本 `recap_m2.indicator_I`。",
        "不得继续消费逐样本 `prompt_conditioned`。",
        "不得继续消费逐样本 `advantage_input`。",
        "不得把 `prompt_conditioned` 升级成 fixedadv 训练输入。",
        "不得把 `recap_m2.advantage_input` 扩张成新的 live prompt 或 live inference API。",
        "no second control dataset",
        "no per-sample indicator consumption",
        "no prompt_conditioned training input",
        "no per-sample advantage_input consumption",
        "no checkpoint warm-start from recap_only_relabel8d_v2",
        "no alternate eval authority",
    ]
    for item in required:
        assert item in text, f"missing fixedadv forbid item: {item}"


def test_fixedadv_train_entry_records_v2_reuse_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir, dataset_fingerprint, episode_universe_hash = (
        _write_minimal_recap_dataset(tmp_path)
    )
    source_checkpoint_dir = _write_minimal_real_export(tmp_path / "source")
    monkeypatch.setattr(
        variant_train_script,
        "run_real_variant_training_export",
        lambda request: RealVariantExportBundle(
            export_dir=source_checkpoint_dir,
            runtime_log_path=request.runtime_dir / "fake_real_variant_training.log",
        ),
    )
    gate_eval_manifest = tmp_path / "eval_manifest_rollout_lite_v2.json"
    _write_json(
        gate_eval_manifest,
        {
            "schema_version": "openpi_libero_rollout_eval_manifest_v2",
            "eval_authority": "fresh_rollout_v2",
            "manifest_name": "rollout_lite_v2",
            "task_suite_name": "libero_spatial",
            "task_ids": [0, 1],
            "seed_manifest": [7, 17, 27, 37],
            "num_trials_per_task": 4,
        },
    )
    output_dir = tmp_path / "fixedadv_relabel8d_control_v1"

    rc = main(
        [
            "--variant",
            "fixedadv_control",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--task-suite-name",
            "libero_spatial",
            "--task-ids",
            "0,1",
            "--seeds",
            "7,17,27,37",
            "--num-trials-per-task",
            "4",
            "--gate-eval-manifest",
            str(gate_eval_manifest),
        ]
    )

    assert rc == 0
    gate_eval_manifest_hash = hashlib.sha256(
        gate_eval_manifest.read_bytes()
    ).hexdigest()
    train_manifest = _mapping(
        json.loads((output_dir / "train_manifest.json").read_text(encoding="utf-8"))
    )
    checkpoint_provenance = _mapping(
        json.loads(
            (output_dir / "checkpoint_provenance.json").read_text(encoding="utf-8")
        )
    )

    for payload in (train_manifest, checkpoint_provenance):
        assert payload["variant"] == "fixedadv_control"
        assert payload["variant_name"] == "fixedadv_relabel8d_control_v1"
        assert payload["dataset_route_id"] == "official_native_8d_recap_relabels_v1"
        assert payload["dataset_fingerprint"] == dataset_fingerprint
        assert payload["episode_universe_hash"] == episode_universe_hash
        assert payload["base_checkpoint_id"] == "pi05_libero_anchor"
        assert payload["train_budget_id"] == "libero_cmp_budget_v2"
        assert payload["consumer_mode"] == "fixedadv_constant"
        assert payload["gate_eval_manifest_hash"] == gate_eval_manifest_hash
        assert payload["reuse_existing_checkpoint"] is False
        assert payload["reuse_verdict"] == "materialize_new_checkpoint"

    training_route = _mapping(train_manifest["training_route"])
    variant_derivation = _mapping(checkpoint_provenance["variant_derivation"])
    assert training_route["consumer_mode"] == "fixedadv_constant"
    assert training_route["fixed_indicator_mode"] == "omit"
    assert training_route["per_sample_indicator_consumption"] is False
    assert variant_derivation["consumer_mode"] == "fixedadv_constant"
    assert variant_derivation["fixed_indicator_mode"] == "omit"
    assert variant_derivation["per_sample_indicator_consumption"] is False
    assert (
        checkpoint_provenance["checkpoint_source"]
        == "repo_local_openpi_fixedadv_relabel8d_control_v1"
    )
    assert not hasattr(variant_train_script, "_resolve_stock_libero_checkpoint_dir")
    best_dir = output_dir / "best"
    assert (best_dir / "params" / "_METADATA").is_file()
    assert (best_dir / "params" / "manifest.ocdbt").read_text(encoding="utf-8") == (
        "fixture-ocdbt-manifest\n"
    )
    assert json.loads(
        (
            best_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
        ).read_text(encoding="utf-8")
    ) == json.loads(
        (
            source_checkpoint_dir
            / "assets"
            / "physical-intelligence"
            / "libero"
            / "norm_stats.json"
        ).read_text(encoding="utf-8")
    )
    serve_checkpoint_ref, serve_checkpoint_mode = resolve_servable_checkpoint_ref(
        checkpoint_ref=str(best_dir),
        variant="fixedadv_relabel8d_control_v1",
        stock_variants=frozenset({"stock", "stock_libero_ref_v1"}),
    )
    assert serve_checkpoint_ref == str(best_dir)
    assert serve_checkpoint_mode == "local_orbax_checkpoint"


def test_fixedadv_train_entry_blocks_instead_of_faking_stock_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir, _, _ = _write_minimal_recap_dataset(tmp_path)
    output_dir = tmp_path / "fixedadv_relabel8d_control_v1"
    gate_eval_manifest = tmp_path / "eval_manifest_rollout_lite_v2.json"
    _write_json(
        gate_eval_manifest,
        {
            "schema_version": "openpi_libero_rollout_eval_manifest_v2",
            "eval_authority": "fresh_rollout_v2",
            "manifest_name": "rollout_lite_v2",
            "task_suite_name": "libero_spatial",
            "task_ids": [0, 1],
            "seed_manifest": [7, 17, 27, 37],
            "num_trials_per_task": 4,
        },
    )

    def _blocked(request: object) -> RealVariantExportBundle:
        raise RealVariantExportBlockedError(
            "fixture blocked real export",
            payload={
                "status": "blocked",
                "blocker_code": "fixture_real_export_blocked",
                "reason": "unit-test forced blocked export",
                "variant": "fixedadv_control",
                "variant_name": "fixedadv_relabel8d_control_v1",
                "runtime_dir": str(output_dir),
                "dataset_dir": str(dataset_dir),
            },
        )

    monkeypatch.setattr(
        variant_train_script, "run_real_variant_training_export", _blocked
    )

    rc = main(
        [
            "--variant",
            "fixedadv_control",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--task-suite-name",
            "libero_spatial",
            "--task-ids",
            "0,1",
            "--seeds",
            "7,17,27,37",
            "--num-trials-per-task",
            "4",
            "--gate-eval-manifest",
            str(gate_eval_manifest),
        ]
    )

    assert rc == 2
    blocker_report = _mapping(
        json.loads((output_dir / "blocker_report.json").read_text(encoding="utf-8"))
    )
    assert blocker_report["status"] == "blocked"
    assert blocker_report["blocker_code"] == "fixture_real_export_blocked"
    assert not (output_dir / "best" / "params" / "_METADATA").exists()


def test_fixedadv_train_entry_rejects_reuse_on_gate_manifest_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir, _, _ = _write_minimal_recap_dataset(tmp_path)
    source_checkpoint_dir = _write_minimal_real_export(tmp_path / "source")
    monkeypatch.setattr(
        variant_train_script,
        "run_real_variant_training_export",
        lambda request: RealVariantExportBundle(
            export_dir=source_checkpoint_dir,
            runtime_log_path=request.runtime_dir / "fake_real_variant_training.log",
        ),
    )
    gate_eval_manifest_a = tmp_path / "eval_manifest_a.json"
    gate_eval_manifest_b = tmp_path / "eval_manifest_b.json"
    _write_json(
        gate_eval_manifest_a,
        {
            "schema_version": "openpi_libero_rollout_eval_manifest_v2",
            "eval_authority": "fresh_rollout_v2",
            "manifest_name": "rollout_lite_v2",
            "task_suite_name": "libero_spatial",
            "task_ids": [0, 1],
            "seed_manifest": [7, 17, 27, 37],
            "num_trials_per_task": 4,
        },
    )
    _write_json(
        gate_eval_manifest_b,
        {
            "schema_version": "openpi_libero_rollout_eval_manifest_v2",
            "eval_authority": "fresh_rollout_v2",
            "manifest_name": "rollout_lite_v2",
            "task_suite_name": "libero_spatial",
            "task_ids": [0, 1],
            "seed_manifest": [7, 17, 27, 37],
            "num_trials_per_task": 4,
            "note": "hash drift fixture",
        },
    )
    output_dir = tmp_path / "fixedadv_relabel8d_control_v1"

    rc = main(
        [
            "--variant",
            "fixedadv_control",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--task-suite-name",
            "libero_spatial",
            "--task-ids",
            "0,1",
            "--seeds",
            "7,17,27,37",
            "--num-trials-per-task",
            "4",
            "--gate-eval-manifest",
            str(gate_eval_manifest_a),
        ]
    )
    assert rc == 0

    with pytest.raises(ValueError, match="gate_eval_manifest_hash mismatch"):
        _ = main(
            [
                "--variant",
                "fixedadv_control",
                "--dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(output_dir),
                "--task-suite-name",
                "libero_spatial",
                "--task-ids",
                "0,1",
                "--seeds",
                "7,17,27,37",
                "--num-trials-per-task",
                "4",
                "--gate-eval-manifest",
                str(gate_eval_manifest_b),
                "--reuse-existing-checkpoint",
            ]
        )
