from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, cast

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from agent.run import build_flux_dataset_probe as agent_build_flux_dataset_probe
from work.openpi.norm.policy import PHASE1_NORM_POLICY, PHASE1_NORM_SOURCE
from work.openpi.recap.prompt_builder import CONDITIONING_MODE, PHASE1_PROMPT_ROUTE
from work.recap.datasets import flux_grouped_dataset
from work.recap.datasets import flux_parquet_dataset
from work.recap.lerobot_export import dataset_export
from work.recap.scripts import build_flux_dataset_probe
from work.recap.scripts.gr00t_eval_contract_gate import DEFAULT_CAMERA_CONFIG


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping_section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload[key]
    assert isinstance(value, dict)
    return dict(cast(dict[str, Any], value))


def _make_dataset(
    tmp_path: Path,
    *,
    include_stats: bool = True,
    include_prompt_source_field: bool = True,
) -> Path:
    dataset_dir = tmp_path / "flux_dataset"
    info_payload = {
        "schema_version": "openpi_libero_recap_dataset_v1",
        "route_id": "official_native_8d_recap_relabels_v1",
        "fps": 10,
        "total_episodes": 1,
        "total_frames": 2,
        "total_tasks": 1,
        "task_text_field": dataset_export.EXPORTER_MAINLINE_TASK_TEXT_FIELD,
        "carrier_route": dataset_export.EXPORTER_CARRIER_ROUTE,
        "carrier_schema_version": dataset_export.EXPORTER_CARRIER_SCHEMA_VERSION,
        "prompt_route": dataset_export.EXPORTER_PROMPT_ROUTE,
        "conditioning_mode": dataset_export.EXPORTER_CONDITIONING_MODE,
        "recap_export.dual_task_text": False,
        "features": {
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": [64, 64, 3],
            },
            "observation.images.wrist_view": {
                "dtype": "video",
                "shape": [64, 64, 3],
            },
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [7]},
            "annotation.human.task_description": {"dtype": "string", "shape": [1]},
            "annotation.human.action.task_description": {
                "dtype": "string",
                "shape": [1],
            },
            "recap_m2.prompt_raw": {"dtype": "string", "shape": [1]},
            "recap_m2.prompt_conditioned": {"dtype": "string", "shape": [1]},
            "recap_m2.indicator_I": {"dtype": "int64", "shape": [1]},
            "recap_m2.advantage_input": {"dtype": "float32", "shape": [1]},
            "recap_m2.advantage_A": {"dtype": "float32", "shape": [1]},
            "recap_m2.return_G": {"dtype": "float32", "shape": [1]},
            "recap_m2.value_V": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
        },
    }
    if include_prompt_source_field:
        info_payload["prompt_source_field"] = (
            dataset_export.EXPORTER_PROMPT_SOURCE_FIELD
        )
    _write_json(dataset_dir / "meta" / "info.json", info_payload)
    _write_json(
        dataset_dir / "meta" / "modality.json",
        {
            "video": {
                "ego_view": {"original_key": "observation.images.ego_view"},
                "wrist_view": {"original_key": "observation.images.wrist_view"},
            },
            "state": {
                "libero_state": {
                    "start": 0,
                    "end": 8,
                    "original_key": "observation.state",
                }
            },
            "action": {
                "libero_action": {
                    "start": 0,
                    "end": 7,
                    "original_key": "action",
                }
            },
        },
    )
    if include_stats:
        _write_json(
            dataset_dir / "meta" / "stats.json",
            {
                "observation.images.ego_view": {"mean": 0.0},
                "observation.state": {"mean": [0.0] * 8},
                "action": {"mean": [0.0] * 7},
            },
        )
    _write_jsonl(
        dataset_dir / "meta" / "tasks.jsonl",
        [{"task": "put the bowl on the plate", "task_index": 0}],
    )
    _write_jsonl(
        dataset_dir / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "tasks": ["put the bowl on the plate"], "length": 2}],
    )
    _write_jsonl(
        dataset_dir / "meta" / "episodes_stats.jsonl",
        [{"episode_index": 0, "frame_count": 2, "task": "put the bowl on the plate"}],
    )
    _write_json(
        dataset_dir / "meta" / "dataset_fingerprint.json",
        {
            "schema_version": "openpi_libero_relabel_dataset_fingerprint_v1",
            "route_id": "official_native_8d_recap_relabels_v1",
            "fingerprint_sha256": "fixture_dataset_fingerprint_sha256",
        },
    )
    (dataset_dir / "meta" / "episode_universe_hash.txt").write_text(
        "fixture_episode_universe_hash_sha256\n",
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
                "put the bowl on the plate\nAdvantage: positive",
                "put the bowl on the plate\nAdvantage: negative",
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
    frame.to_parquet(parquet_path, engine="pyarrow", index=False)
    return dataset_dir


def _blocker_codes(bundle_payload: dict[str, Any]) -> set[str]:
    return {str(item["code"]) for item in bundle_payload["blocking_reasons"]}


def test_cli_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_flux_dataset_probe.main(["--help"])
    assert exc_info.value.code == 0
    with pytest.raises(SystemExit) as exc_info:
        agent_build_flux_dataset_probe.main(["--help"])
    assert exc_info.value.code == 0


def test_inventory_bundle_is_provenance_complete_for_valid_dataset(
    tmp_path: Path,
) -> None:
    dataset_dir = _make_dataset(tmp_path)

    bundle = flux_grouped_dataset.build_flux_dataset_inventory_bundle(dataset_dir)
    payload = flux_grouped_dataset.inventory_bundle_to_dict(bundle)
    dataset_source = _mapping_section(payload, "dataset_source")
    schema_compatibility = _mapping_section(payload, "schema_compatibility")
    prompt_source = _mapping_section(payload, "prompt_source")
    task_description_source = _mapping_section(payload, "task_description_source")
    camera_inventory = _mapping_section(payload, "camera_inventory")
    action_state_normalization_source = _mapping_section(
        payload,
        "action_state_normalization_source",
    )
    grouped_stats = _mapping_section(payload, "grouped_stats")
    binding_join_contract = _mapping_section(payload, "binding_join_contract")

    assert payload["schema_version"] == flux_grouped_dataset.SCHEMA_VERSION
    assert payload["artifact_kind"] == flux_grouped_dataset.ARTIFACT_KIND
    assert payload["verdict"] == flux_grouped_dataset.VERDICT_COMPLETE
    assert payload["blocking_reasons"] == []
    assert dataset_source["dataset_dir"] == str(dataset_dir.resolve())
    assert payload["dataset_fingerprint"] == "fixture_dataset_fingerprint_sha256"
    assert payload["stats_fingerprint"]
    assert schema_compatibility["status"] == "compatible"
    assert schema_compatibility["source_prompt_feature_key"] == (
        "annotation.human.task_description"
    )
    assert prompt_source == {
        "prompt_source_field": dataset_export.EXPORTER_PROMPT_SOURCE_FIELD,
        "prompt_route": PHASE1_PROMPT_ROUTE,
        "conditioning_mode": CONDITIONING_MODE,
        "provenance_complete": True,
    }
    assert task_description_source["task_text_field"] == (
        dataset_export.EXPORTER_MAINLINE_TASK_TEXT_FIELD
    )
    assert camera_inventory["view_count"] == 2
    assert camera_inventory["expected_eval_camera_config"] == DEFAULT_CAMERA_CONFIG
    assert action_state_normalization_source["norm_stats_policy"] == (
        PHASE1_NORM_POLICY
    )
    assert action_state_normalization_source["norm_stats_source"] == (
        PHASE1_NORM_SOURCE
    )
    assert grouped_stats["task_row_count"] == 1
    assert grouped_stats["episode_row_count"] == 1
    assert grouped_stats["episode_stats_row_count"] == 1
    assert binding_join_contract["dataset_fingerprint"] == (
        "fixture_dataset_fingerprint_sha256"
    )
    assert binding_join_contract["prompt_source"] == "prompt_raw"
    assert binding_join_contract["norm_stats_source"] == PHASE1_NORM_SOURCE
    assert binding_join_contract["action_state_norm_source"] == (PHASE1_NORM_POLICY)
    assert binding_join_contract["expected_embodiment_tag"] == "UNITREE_G1"
    assert binding_join_contract["expected_action_space_signature"]


@pytest.mark.parametrize(
    ("include_stats", "include_prompt_source_field", "expected_code"),
    [
        (False, True, "missing_stats_fingerprint"),
        (True, False, "missing_prompt_provenance"),
    ],
)
def test_inventory_bundle_fails_closed_when_required_provenance_is_missing(
    tmp_path: Path,
    include_stats: bool,
    include_prompt_source_field: bool,
    expected_code: str,
) -> None:
    dataset_dir = _make_dataset(
        tmp_path,
        include_stats=include_stats,
        include_prompt_source_field=include_prompt_source_field,
    )

    bundle = flux_grouped_dataset.build_flux_dataset_inventory_bundle(dataset_dir)
    payload = flux_grouped_dataset.inventory_bundle_to_dict(bundle)

    assert payload["verdict"] == flux_grouped_dataset.VERDICT_MISSING
    assert expected_code in _blocker_codes(payload)


def test_probe_script_writes_inventory_and_evidence(
    tmp_path: Path,
    capsys,
) -> None:
    dataset_dir = _make_dataset(tmp_path)
    output_dir = tmp_path / "probe"
    evidence_json = tmp_path / "task-7-dataset-inventory-bundle.json"

    exit_code = build_flux_dataset_probe.main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--evidence-json",
            str(evidence_json),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    inventory = _read_json(
        output_dir / build_flux_dataset_probe.DATASET_INVENTORY_BUNDLE_JSON_NAME
    )
    evidence = _read_json(evidence_json)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["status"] == "PASS"
    assert payload["inventory_verdict"] == flux_grouped_dataset.VERDICT_COMPLETE
    assert inventory["verdict"] == flux_grouped_dataset.VERDICT_COMPLETE
    assert evidence["inventory_verdict"] == flux_grouped_dataset.VERDICT_COMPLETE
    assert evidence["backpointer"]["dataset_inventory_bundle_json"] == str(
        output_dir / build_flux_dataset_probe.DATASET_INVENTORY_BUNDLE_JSON_NAME
    )


def test_flux_parquet_dataset_adapter_exposes_dataset_side_sources(
    tmp_path: Path,
) -> None:
    dataset_dir = _make_dataset(tmp_path)

    adapter = flux_parquet_dataset.build_flux_parquet_dataset_adapter(dataset_dir)

    assert adapter.dataset_name == "flux_dataset"
    assert adapter.dataset_fingerprint == "fixture_dataset_fingerprint_sha256"
    assert adapter.stats_fingerprint is not None
    assert adapter.schema_compatibility["status"] == "compatible"
    assert adapter.prompt_source["prompt_source_field"] == "prompt_raw"
    assert adapter.task_description_source["carrier_route"] == "carrier_text_v1"
    assert adapter.action_state_normalization_source["norm_stats_policy"] == (
        PHASE1_NORM_POLICY
    )
