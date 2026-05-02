from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap import prompt_builder
from work.openpi.recap import runtime_prompt
from work.recap import policy as recap_policy
from work.recap import text_indicator
from work.recap.lerobot_export import dataset_export


Float32Array = npt.NDArray[np.float32]


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            _ = handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            _ = handle.write("\n")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return dict(payload)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise TypeError(
                        f"expected JSON object line at {path}, got {type(payload).__name__}"
                    )
                records.append(dict(payload))
    return records


def _build_npz_payload() -> dict[str, Float32Array]:
    payload: dict[str, Float32Array] = {}
    for key in dataset_export.STATE_KEY_ORDER_LOCK:
        payload[key] = np.asarray([[[[1.0]]]], dtype=np.float32)
    for key in dataset_export.ACTION_KEY_ORDER_LOCK:
        payload[key] = np.asarray([[[[2.0]]]], dtype=np.float32)
    return payload


def _build_source_dataset(
    dataset_dir: Path,
    *,
    episode_id: str,
    label_row: dict[str, object],
) -> None:
    arrays_dir = dataset_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    npz_path = arrays_dir / f"{episode_id}.npz"
    np.savez(npz_path, **_build_npz_payload())
    _write_jsonl(
        dataset_dir / "episodes.jsonl",
        [
            {
                "episode_id": episode_id,
                "prompt_raw": label_row.get("prompt_raw"),
                "prompt_conditioned": label_row.get("prompt_conditioned"),
                "npz_path": str(Path("arrays") / npz_path.name),
                "n_action_steps_config": 1,
                "n_policy_steps": 1,
            }
        ],
    )
    _write_jsonl(
        dataset_dir / "transitions.jsonl",
        [
            {
                "episode_id": episode_id,
                "t": 0,
                "T_action": 1,
                "n_action_steps_config": 1,
                "n_action_steps_executed": 1,
                "inner_rewards": [0.0],
                "inner_dones": [False],
            }
        ],
    )
    _write_jsonl(dataset_dir / "m2_labels" / "labels.jsonl", [dict(label_row)])


def test_exporter_defaults_to_carrier_text_v1_and_emits_mainline_provenance(
    tmp_path: Path,
) -> None:
    prompt_raw = "pick up the apple and place it on the plate"
    carrier_text = text_indicator.build_canonical_text_indicator(
        prompt_raw,
        text_indicator.TEXT_INDICATOR_POSITIVE,
    )
    input_dir = tmp_path / "recap_source"
    _build_source_dataset(
        input_dir,
        episode_id="ep_mainline",
        label_row={
            "episode_id": "ep_mainline",
            "t": 0,
            "prompt_raw": prompt_raw,
            "prompt_conditioned": prompt_raw,
            "carrier_text_v1": carrier_text,
            "indicator_I": 1,
            "indicator_mode": text_indicator.TEXT_INDICATOR_POSITIVE,
            "indicator_source": "label.recap_m2.indicator_source",
        },
    )

    result = dataset_export.export_recap_to_lerobot_v2(
        iter_tag="exporter_mainline",
        repo_root=tmp_path,
        input_recap_dataset_dir=input_dir,
        include_m2_label_columns=False,
    )

    info = _read_json(
        result.output_dataset_dir / "meta" / dataset_export.META_INFO_JSON
    )
    tasks = _read_jsonl(
        result.output_dataset_dir / "meta" / dataset_export.META_TASKS_JSONL
    )

    assert (
        str(
            inspect.signature(dataset_export.export_recap_to_lerobot_v2)
            .parameters["task_text_field"]
            .default
        )
        == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    )
    assert info["task_text_field"] == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    assert (
        info["carrier_schema_version"]
        == text_indicator.RECAP_TEXT_INDICATOR_SCHEMA_VERSION
    )
    assert info["carrier_route"] == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    assert (
        info["prompt_source_field"]
        == text_indicator.RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD
    )
    assert info["prompt_route"] == prompt_builder.PHASE1_PROMPT_ROUTE
    assert info["conditioning_mode"] == prompt_builder.CONDITIONING_MODE
    assert info["indicator_mode_field"] == "indicator_mode"
    assert info["indicator_source_field"] == "indicator_source"
    assert info["indicator_mode_source_field"] == "recap_m2.indicator_I"
    assert info["indicator_mode"] == text_indicator.TEXT_INDICATOR_POSITIVE
    assert info["indicator_mode_values"] == [text_indicator.TEXT_INDICATOR_POSITIVE]
    assert info["indicator_source"] == "label.recap_m2.indicator_source"
    assert info["indicator_source_values"] == ["label.recap_m2.indicator_source"]
    assert tasks == [{"task_index": 0, "task": carrier_text}]


def test_exporter_fails_closed_when_mainline_carrier_text_v1_is_missing(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "recap_source_missing_carrier"
    _build_source_dataset(
        input_dir,
        episode_id="ep_missing_carrier",
        label_row={
            "episode_id": "ep_missing_carrier",
            "t": 0,
            "prompt_raw": "pick up the apple and place it on the plate",
            "prompt_conditioned": "legacy conditioned text should not be used",
            "indicator_I": 1,
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "Missing authoritative carrier_text_v1 for mainline single-text export; "
            "exporter will not fall back to prompt_conditioned or prompt_raw"
        ),
    ):
        _ = dataset_export.export_recap_to_lerobot_v2(
            iter_tag="exporter_missing_carrier",
            repo_root=tmp_path,
            input_recap_dataset_dir=input_dir,
            include_m2_label_columns=False,
        )


def test_exporter_explicit_legacy_prompt_conditioned_path_remains_available(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "recap_source_legacy"
    legacy_text = "legacy conditioned text"
    _build_source_dataset(
        input_dir,
        episode_id="ep_legacy",
        label_row={
            "episode_id": "ep_legacy",
            "t": 0,
            "prompt_raw": "pick up the apple and place it on the plate",
            "prompt_conditioned": legacy_text,
            "indicator_I": 0,
        },
    )

    result = dataset_export.export_recap_to_lerobot_v2(
        iter_tag="exporter_legacy_explicit",
        repo_root=tmp_path,
        input_recap_dataset_dir=input_dir,
        task_text_field="prompt_conditioned",
        include_m2_label_columns=False,
    )

    info = _read_json(
        result.output_dataset_dir / "meta" / dataset_export.META_INFO_JSON
    )
    tasks = _read_jsonl(
        result.output_dataset_dir / "meta" / dataset_export.META_TASKS_JSONL
    )

    assert info["task_text_field"] == "prompt_conditioned"
    assert "carrier_schema_version" not in info
    assert tasks == [{"task_index": 0, "task": legacy_text}]


def test_runtime_mainline_route_defaults_to_text_indicator_policy() -> None:
    runtime_spec = recap_policy.build_runtime_policy_spec(
        indicator_mode=text_indicator.TEXT_INDICATOR_POSITIVE
    )

    assert recap_policy.resolve_runtime_policy_route() == (
        text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    )
    assert (
        recap_policy.resolve_runtime_policy_class(
            indicator_mode=text_indicator.TEXT_INDICATOR_NEGATIVE
        )
        is recap_policy.TextIndicatorGr00tPolicy
    )
    assert runtime_spec["route"] == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    assert (
        runtime_spec["carrier_route"]
        == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    )
    assert runtime_spec["carrier_schema_version"] == (
        text_indicator.RECAP_TEXT_INDICATOR_SCHEMA_VERSION
    )
    assert runtime_spec["prompt_source_field"] == (
        text_indicator.RECAP_TEXT_INDICATOR_SOURCE_PROMPT_FIELD
    )
    assert runtime_spec["indicator_source"] == "indicator_mode"
    assert runtime_spec["policy_class_name"] == "TextIndicatorGr00tPolicy"
    assert runtime_spec["mainline_authority"] is True
    assert runtime_spec["diagnostic_only"] is False
    assert runtime_spec["runtime_indicator_mode_required"] is True

    with pytest.raises(ValueError, match=r"options\['indicator_mode'\]|indicator_mode"):
        _ = recap_policy.resolve_runtime_policy_class()


def test_runtime_mainline_modes_match_runtime_prompt_authority_and_keep_numeric_adv_diagnostic() -> (
    None
):
    runtime_prompt_modes = {
        str(mode)
        for mode in runtime_prompt.RUNTIME_INDICATOR_CLI_MODES
        if str(mode) != runtime_prompt.RUNTIME_INDICATOR_CFG
    }

    assert set(recap_policy.MAINLINE_RUNTIME_INDICATOR_MODES) == runtime_prompt_modes

    for indicator_mode in recap_policy.MAINLINE_RUNTIME_INDICATOR_MODES:
        runtime_spec = recap_policy.build_runtime_policy_spec(
            indicator_mode=indicator_mode
        )
        assert runtime_spec["indicator_mode"] == indicator_mode

    diagnostic_spec = recap_policy.build_runtime_policy_spec(
        route=recap_policy.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE
    )
    assert diagnostic_spec["route"] == recap_policy.DIAGNOSTIC_NUMERIC_ADV_RUNTIME_ROUTE
    assert diagnostic_spec["policy_class_name"] == (
        recap_policy.DIAGNOSTIC_NUMERIC_ADV_POLICY_CLASS_NAME
    )
    assert diagnostic_spec["mainline_authority"] is False
    assert diagnostic_spec["diagnostic_only"] is True
    assert diagnostic_spec["runtime_indicator_mode_required"] is False
