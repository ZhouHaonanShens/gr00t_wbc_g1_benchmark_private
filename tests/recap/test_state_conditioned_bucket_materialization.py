from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import state_conditioned_bucket_a_import
from work.recap.scripts import state_conditioned_collect_buckets
from work.recap.scripts import state_conditioned_dev_manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _build_bucket_a_fixture(tmp_path: Path, *, ready: bool) -> Path:
    bucket_dir = tmp_path / "bucket_a"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        bucket_dir / state_conditioned_bucket_a_import.GATE_A_READY_JSON_NAME,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "bucket_key": state_conditioned_bucket_a_import.BUCKET_KEY,
            "ready": bool(ready),
            "required_distinct_accepted_episode_count": 24,
            "accepted_episode_count": 24,
            "distinct_accepted_episode_count": 24,
        },
    )
    _write_json(
        bucket_dir / state_conditioned_bucket_a_import.MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_bucket_a_import.SCHEMA_VERSION,
            "bucket_key": state_conditioned_bucket_a_import.BUCKET_KEY,
            "required_distinct_episode_count": 24,
            "episodes": [
                {
                    "episode_id": f"fresh_accept_{index:03d}",
                    "accepted": True,
                    "debug_only": False,
                    "fresh_nominal_recollection": True,
                    "reused_existing_live_dataset": False,
                    "seed": 700 + index,
                    "source_dataset_dir": str(
                        tmp_path / f"canonical_dataset_{index:03d}"
                    ),
                }
                for index in range(24)
            ],
        },
    )
    return bucket_dir


def _build_dev_fixture(tmp_path: Path) -> Path:
    dev_dir = tmp_path / "devbench"
    dev_dir.mkdir(parents=True, exist_ok=True)
    paired_seed_values = list(state_conditioned_dev_manifest.DEFAULT_PAIRED_SEEDS)
    _write_json(
        dev_dir / state_conditioned_dev_manifest.FIXED_STRATA_DEFINITION_JSON_NAME,
        {
            "schema_version": state_conditioned_dev_manifest.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_dev_fixed_strata_definition",
            "paired_seed_values": paired_seed_values,
            "paired_seed_count": len(paired_seed_values),
            "strata": [
                {
                    **dict(row),
                    "paired_episode_count": len(paired_seed_values),
                }
                for row in state_conditioned_dev_manifest.DEFAULT_STRATA_DEFINITIONS
            ],
        },
    )
    _write_json(
        dev_dir / state_conditioned_dev_manifest.BASELINE_MANIFEST_JSON_NAME,
        {
            "schema_version": state_conditioned_dev_manifest.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_dev_baseline_manifest",
            "baseline_policy": {
                "kind": "original_baseline",
                "model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
            },
            "counts": {
                "entries": 32,
                "paired_seed_count": len(paired_seed_values),
                "per_stratum": dict(
                    state_conditioned_dev_manifest.EXPECTED_STRATA_COUNTS
                ),
            },
            "entries": [],
        },
    )
    _write_json(
        dev_dir / state_conditioned_dev_manifest.BASELINE_DEV_SCORECARD_JSON_NAME,
        {
            "schema_version": state_conditioned_dev_manifest.SCHEMA_VERSION,
            "artifact_kind": "state_conditioned_dev_baseline_scorecard",
            "baseline_invocation": {
                "runner": "fake_runner",
                "model_path": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
            },
            "counts": {
                "requested_entries": 32,
                "per_stratum": dict(
                    state_conditioned_dev_manifest.EXPECTED_STRATA_COUNTS
                ),
            },
        },
    )
    return dev_dir


def _make_collection_runner(
    tmp_path: Path,
    *,
    contamination_for_bucket_b: bool = False,
) -> Any:
    counter = {"value": 0}

    def _runner(
        *,
        output_dir: Path,
        bucket_key: str,
        plan_index: int,
        plan_entry: dict[str, Any],
    ) -> dict[str, Any]:
        del output_dir
        counter["value"] += 1
        dataset_dir = tmp_path / "datasets" / f"{bucket_key}_{plan_index:03d}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        episode_id = f"{bucket_key.lower()}_episode_{plan_index:03d}"
        episode_record: dict[str, Any] = {
            "episode_id": episode_id,
            "seed": int(plan_entry["seed"]),
            "success_episode": True,
            "env_name": "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc",
            "prompt_raw": "pick up the apple and place it on the plate",
            "prompt_conditioned": "pick up the apple and place it on the plate",
        }
        if contamination_for_bucket_b and bucket_key == "bucket_B" and plan_index == 0:
            episode_record["failure_injection_kind"] = "drop_during_transport"
        _write_jsonl(dataset_dir / "episodes.jsonl", [episode_record])
        _write_jsonl(
            dataset_dir / "transitions.jsonl",
            [{"episode_id": episode_id, "t": 0, "success_step": False}],
        )
        return {
            "iter_tag": f"fixture_{bucket_key}_{plan_index:03d}",
            "dataset_dir": str(dataset_dir),
            "episodes_path": str(dataset_dir / "episodes.jsonl"),
            "transitions_path": str(dataset_dir / "transitions.jsonl"),
            "episode_order": [episode_id],
            "episodes_by_id": {episode_id: dict(episode_record)},
            "materialized_episode_count": 1,
            "collected_episode_count": 1,
            "collection_command": ["fixture"],
            "runtime_log_path": str(
                tmp_path / "runtime_logs" / f"{bucket_key}_{plan_index:03d}.log"
            ),
            "materialization_mode": "fixture_materialization",
            "reused_existing_live_dataset": False,
        }

    return _runner


def test_happy_path_materializes_bucket_b16_and_bucket_c24_with_metadata(
    tmp_path: Path,
) -> None:
    bucket_dir = _build_bucket_a_fixture(tmp_path, ready=True)
    dev_dir = _build_dev_fixture(tmp_path)
    output_dir = tmp_path / "collection"

    result = state_conditioned_collect_buckets.materialize_state_conditioned_buckets(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        output_dir=output_dir,
        collection_runner=_make_collection_runner(tmp_path),
    )

    bucket_b_manifest = _read_json(
        output_dir / state_conditioned_collect_buckets.BUCKET_B_MANIFEST_JSON_NAME
    )
    bucket_c_manifest = _read_json(
        output_dir / state_conditioned_collect_buckets.BUCKET_C_MANIFEST_JSON_NAME
    )
    summary = _read_json(
        output_dir
        / state_conditioned_collect_buckets.BUCKET_COLLECTION_SUMMARY_JSON_NAME
    )

    assert Path(result["bucket_B_manifest_path"]).is_file()
    assert Path(result["bucket_C_manifest_path"]).is_file()
    assert Path(result["bucket_collection_summary_path"]).is_file()

    assert bucket_b_manifest["counts"]["episodes"] == 16
    assert bucket_c_manifest["counts"]["episodes"] == 24
    assert bucket_c_manifest["counts"]["per_failure_family"] == {
        "drop_during_transport": 8,
        "failed_grasp_occluded": 8,
        "failed_grasp_visible": 8,
    }
    assert summary["counts"] == {
        "bucket_B": 16,
        "bucket_C": 24,
        "bucket_C_per_failure_family": {
            "drop_during_transport": 8,
            "failed_grasp_occluded": 8,
            "failed_grasp_visible": 8,
        },
    }

    for entry in bucket_b_manifest["episodes"]:
        episode_record = _read_jsonl(Path(entry["episodes_path"]))[0]
        sidecar_path = Path(entry["dataset_dir"]) / "state_conditioned_sidecar.jsonl"
        sidecar_rows = _read_jsonl(sidecar_path)
        assert episode_record["experiment_split"] == "devtrain"
        assert episode_record["stable_base_checkpoint_kind"] == "model_path"
        assert (
            episode_record["stable_base_checkpoint_value"]
            == "nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
        )
        assert sidecar_path.is_file()
        assert len(sidecar_rows) == 1
        assert sidecar_rows[0]["anchor_mujoco_state_ref"] == (
            f"mujoco://{entry['episode_id']}/0"
        )
        assert len(sidecar_rows[0]["prehistory_window"]) == 8
        assert len(sidecar_rows[0]["history_valid_mask"]) == 8
        assert "privileged.apple_visible" in sidecar_rows[0]
        assert isinstance(episode_record["provenance"], dict)
        for field in state_conditioned_collect_buckets.INJECTION_METADATA_FIELDS:
            assert field not in episode_record
            assert field not in entry

    for entry in bucket_c_manifest["episodes"]:
        episode_record = _read_jsonl(Path(entry["episodes_path"]))[0]
        sidecar_path = Path(entry["dataset_dir"]) / "state_conditioned_sidecar.jsonl"
        sidecar_rows = _read_jsonl(sidecar_path)
        assert episode_record["experiment_split"] == "devtrain"
        assert episode_record["stable_base_checkpoint_kind"] == "model_path"
        assert isinstance(episode_record["provenance"], dict)
        assert sidecar_path.is_file()
        assert len(sidecar_rows) == 1
        assert sidecar_rows[0]["policy_condition.mode"] == "RECOVERY"
        assert sidecar_rows[0]["anchor_mujoco_state_ref"] == (
            f"mujoco://{entry['episode_id']}/0"
        )
        assert (
            entry["failure_injection_kind"]
            in state_conditioned_collect_buckets.REQUIRED_FAILURE_FAMILIES
        )
        assert isinstance(entry["failure_injection_seed"], int)
        assert isinstance(entry["failure_injection_trigger_t"], int)
        assert (
            episode_record["failure_injection_kind"] == entry["failure_injection_kind"]
        )
        assert (
            episode_record["failure_injection_seed"] == entry["failure_injection_seed"]
        )
        assert (
            episode_record["failure_injection_trigger_t"]
            == entry["failure_injection_trigger_t"]
        )


def test_backfill_state_conditioned_sidecars_from_manifest_writes_missing_files(
    tmp_path: Path,
) -> None:
    bucket_dir = _build_bucket_a_fixture(tmp_path, ready=True)
    dev_dir = _build_dev_fixture(tmp_path)
    output_dir = tmp_path / "collection"

    state_conditioned_collect_buckets.materialize_state_conditioned_buckets(
        bucket_dir=bucket_dir,
        dev_dir=dev_dir,
        output_dir=output_dir,
        collection_runner=_make_collection_runner(tmp_path),
    )

    bucket_b_manifest_path = (
        output_dir / state_conditioned_collect_buckets.BUCKET_B_MANIFEST_JSON_NAME
    )
    bucket_b_manifest = _read_json(bucket_b_manifest_path)
    first_dataset_dir = Path(bucket_b_manifest["episodes"][0]["dataset_dir"])
    sidecar_path = first_dataset_dir / "state_conditioned_sidecar.jsonl"
    sidecar_path.unlink()

    result = state_conditioned_collect_buckets.backfill_state_conditioned_sidecars_from_manifest(
        bucket_b_manifest_path
    )

    assert sidecar_path.is_file()
    assert result["dataset_count"] == 16
    restored_rows = _read_jsonl(sidecar_path)
    assert restored_rows[0]["anchor_mujoco_state_ref"] == (
        f"mujoco://{bucket_b_manifest['episodes'][0]['episode_id']}/0"
    )


def test_bucket_b_injection_forbidden(tmp_path: Path) -> None:
    bucket_dir = _build_bucket_a_fixture(tmp_path, ready=True)
    dev_dir = _build_dev_fixture(tmp_path)

    with pytest.raises(
        ValueError, match="Bucket B must not contain injection metadata"
    ):
        state_conditioned_collect_buckets.materialize_state_conditioned_buckets(
            bucket_dir=bucket_dir,
            dev_dir=dev_dir,
            output_dir=tmp_path / "collection_contaminated",
            collection_runner=_make_collection_runner(
                tmp_path,
                contamination_for_bucket_b=True,
            ),
        )


def test_bucket_c_missing_injection_metadata_fails(tmp_path: Path) -> None:
    bucket_dir = _build_bucket_a_fixture(tmp_path, ready=True)
    dev_dir = _build_dev_fixture(tmp_path)
    bucket_plans = state_conditioned_collect_buckets.build_default_bucket_plans(
        paired_seed_values=state_conditioned_dev_manifest.DEFAULT_PAIRED_SEEDS,
        bucket_b_target=16,
        bucket_c_target=24,
    )
    broken_bucket_c = [dict(entry) for entry in bucket_plans["bucket_C"]]
    broken_bucket_c[0].pop("failure_injection_trigger_t")

    with pytest.raises(
        ValueError,
        match="Bucket C injected episode is missing injection metadata",
    ):
        state_conditioned_collect_buckets.materialize_state_conditioned_buckets(
            bucket_dir=bucket_dir,
            dev_dir=dev_dir,
            output_dir=tmp_path / "collection_missing_metadata",
            collection_runner=_make_collection_runner(tmp_path),
            bucket_plans={
                "bucket_B": bucket_plans["bucket_B"],
                "bucket_C": broken_bucket_c,
            },
        )


def test_gate_a_not_ready_fails_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bucket_dir = _build_bucket_a_fixture(tmp_path, ready=False)
    dev_dir = _build_dev_fixture(tmp_path)

    exit_code = state_conditioned_collect_buckets.main(
        [
            "--bucket-dir",
            str(bucket_dir),
            "--dev-dir",
            str(dev_dir),
            "--output-dir",
            str(tmp_path / "collection_cli"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "bucket_A_gate_a_ready.json.ready == true" in captured.err
    assert "Traceback" not in captured.err
