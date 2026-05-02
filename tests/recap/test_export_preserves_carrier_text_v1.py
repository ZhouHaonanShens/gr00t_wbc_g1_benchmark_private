from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import text_indicator
from work.recap.lerobot_export import dataset_export
import pytest


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
            if not stripped:
                continue
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


def test_export_preserves_carrier_text_v1_when_present_and_canonical(
    tmp_path: Path,
) -> None:
    prompt_raw = "pick up the apple and place it on the plate"
    carrier_text = text_indicator.build_canonical_text_indicator(
        prompt_raw,
        text_indicator.TEXT_INDICATOR_NEGATIVE,
    )
    input_dir = tmp_path / "recap_source"
    _build_source_dataset(
        input_dir,
        episode_id="ep_mainline",
        label_row={
            "schema_version": "recap-v0",
            "code_version": "test",
            "iter_tag": "iter_001",
            "episode_id": "ep_mainline",
            "t": 0,
            "return_G": 0.0,
            "value_V": 0.2,
            "advantage_A": -0.2,
            "epsilon_l": 0.0,
            "indicator_I": 0,
            "is_correction": False,
            "prompt_raw": prompt_raw,
            "prompt_conditioned": "advantage negative pick up the apple and place it on the plate",
            "carrier_text_v1": carrier_text,
        },
    )

    result = dataset_export.export_recap_to_lerobot_v2(
        iter_tag="export_preserves_carrier",
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
    episodes = _read_jsonl(
        result.output_dataset_dir / "meta" / dataset_export.META_EPISODES_JSONL
    )

    assert info["task_text_field"] == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    assert info["carrier_route"] == text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD
    assert tasks == [{"task_index": 0, "task": carrier_text}]
    assert len(episodes) == 1
    assert episodes[0]["episode_index"] == 0
    assert episodes[0]["tasks"] == [carrier_text]
    assert episodes[0]["length"] == 1


def test_export_rejects_mismatched_carrier_text_v1_even_when_present(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "recap_source_bad_carrier"
    _build_source_dataset(
        input_dir,
        episode_id="ep_bad_mainline",
        label_row={
            "schema_version": "recap-v0",
            "code_version": "test",
            "iter_tag": "iter_001",
            "episode_id": "ep_bad_mainline",
            "t": 0,
            "return_G": 0.0,
            "value_V": 0.2,
            "advantage_A": 0.2,
            "epsilon_l": 0.0,
            "indicator_I": 1,
            "is_correction": False,
            "prompt_raw": "pick up the apple and place it on the plate",
            "prompt_conditioned": "advantage positive pick up the apple and place it on the plate",
            "carrier_text_v1": "pick up the apple and place it on the plate\nAdvantage: negative",
        },
    )

    with pytest.raises(
        ValueError,
        match=r"carrier_text_v1 must match the canonical prompt_raw \+ indicator_I text-indicator carrier",
    ):
        _ = dataset_export.export_recap_to_lerobot_v2(
            iter_tag="export_bad_carrier",
            repo_root=tmp_path,
            input_recap_dataset_dir=input_dir,
            include_m2_label_columns=False,
        )
