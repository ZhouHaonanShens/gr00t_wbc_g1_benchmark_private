from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.data import contract_mapping
from work.openpi.recap import prompt_builder
from work.recap.lerobot_export import dataset_export


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _make_dataset_dir(tmp_path: Path, *, dual_task_text: bool = False) -> Path:
    dataset_dir = tmp_path / "dataset"
    info: dict[str, object] = {
        "features": {
            "action": {"dtype": "float32", "shape": [32], "names": None},
            "observation.state": {
                "dtype": "float32",
                "shape": [45],
                "names": None,
            },
            "annotation.human.task_description": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        },
        "task_text_field": dataset_export.EXPORTER_MAINLINE_TASK_TEXT_FIELD,
        "carrier_route": dataset_export.EXPORTER_CARRIER_ROUTE,
        "carrier_schema_version": dataset_export.EXPORTER_CARRIER_SCHEMA_VERSION,
        "prompt_source_field": dataset_export.EXPORTER_PROMPT_SOURCE_FIELD,
        "prompt_route": prompt_builder.PHASE1_PROMPT_ROUTE,
        "conditioning_mode": prompt_builder.CONDITIONING_MODE,
        "recap_export.dual_task_text": bool(dual_task_text),
    }
    if dual_task_text:
        info["task_text_mode"] = "mix50"

    modality = {"video": {"ego_view": {"original_key": "observation.images.ego_view"}}}
    _write_json(dataset_dir / "meta/info.json", info)
    _write_json(dataset_dir / "meta/modality.json", modality)
    return dataset_dir


def test_phase1_contract_mapping_happy_path(tmp_path: Path) -> None:
    dataset_dir = _make_dataset_dir(tmp_path, dual_task_text=False)
    spec = contract_mapping.build_phase1_dataset_mapping_spec(dataset_dir)

    assert spec.dataset_handoff_kind == "lerobot_v2_with_video_single_prompt"
    assert spec.source_image_feature_key == "observation.images.ego_view"
    assert spec.source_state_feature_key == "observation.state"
    assert spec.source_action_feature_key == "action"
    assert spec.source_prompt_feature_key == "annotation.human.task_description"
    assert spec.openpi_primary_image_key == "observation/image"
    assert spec.openpi_wrist_image_key == "observation/wrist_image"
    assert spec.openpi_state_key == "observation/state"
    assert spec.openpi_prompt_key == "prompt"
    assert spec.image_bridge_mode == "duplicate_ego_view_for_wrist_image"
    assert spec.state_dim == 45
    assert spec.action_dim == 32


def test_phase1_contract_mapping_rejects_missing_mainline_carrier_authority(
    tmp_path: Path,
) -> None:
    dataset_dir = _make_dataset_dir(tmp_path, dual_task_text=False)
    info_path = dataset_dir / "meta/info.json"
    info = cast(dict[str, object], json.loads(info_path.read_text(encoding="utf-8")))
    _ = info.pop("task_text_field")
    _write_json(info_path, info)

    try:
        _ = contract_mapping.build_phase1_dataset_mapping_spec(dataset_dir)
    except ValueError as exc:
        assert "task_text_field=carrier_text_v1" in str(exc)
    else:
        raise AssertionError(
            "expected missing mainline carrier authority to be rejected"
        )


def test_phase1_contract_mapping_rejects_missing_prompt_route_authority(
    tmp_path: Path,
) -> None:
    dataset_dir = _make_dataset_dir(tmp_path, dual_task_text=False)
    info_path = dataset_dir / "meta/info.json"
    info = cast(dict[str, object], json.loads(info_path.read_text(encoding="utf-8")))
    _ = info.pop("prompt_route")
    _write_json(info_path, info)

    try:
        _ = contract_mapping.build_phase1_dataset_mapping_spec(dataset_dir)
    except ValueError as exc:
        assert "prompt_route" in str(exc)
    else:
        raise AssertionError("expected missing prompt_route authority to be rejected")


def test_phase1_contract_mapping_uses_prompt_fallback_when_alias_missing(
    tmp_path: Path,
) -> None:
    dataset_dir = _make_dataset_dir(tmp_path, dual_task_text=False)
    info_path = dataset_dir / "meta/info.json"
    info = cast(dict[str, object], json.loads(info_path.read_text(encoding="utf-8")))
    features = cast(dict[str, object], info["features"])
    _ = features.pop("annotation.human.task_description")
    features["annotation.human.action.task_description"] = {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    }
    _write_json(info_path, info)

    spec = contract_mapping.build_phase1_dataset_mapping_spec(dataset_dir)
    assert spec.source_prompt_feature_key == "annotation.human.action.task_description"


def test_phase1_contract_mapping_rejects_ambiguous_dual_task_text(
    tmp_path: Path,
) -> None:
    dataset_dir = _make_dataset_dir(tmp_path, dual_task_text=True)
    try:
        _ = contract_mapping.build_phase1_dataset_mapping_spec(dataset_dir)
    except ValueError as exc:
        assert "dual_task_text" in str(exc)
        assert "mix50" in str(exc)
    else:
        raise AssertionError("expected dual_task_text dataset to be rejected")


def test_phase1_contract_mapping_rejects_missing_video_key(tmp_path: Path) -> None:
    dataset_dir = _make_dataset_dir(tmp_path, dual_task_text=False)
    _write_json(dataset_dir / "meta/modality.json", {"video": {}})

    try:
        _ = contract_mapping.build_phase1_dataset_mapping_spec(dataset_dir)
    except ValueError as exc:
        assert "observation.images.ego_view" in str(exc)
    else:
        raise AssertionError("expected missing video mapping to be rejected")
