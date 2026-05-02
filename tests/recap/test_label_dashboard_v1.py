from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import text_indicator
from work.recap import label_policy
from work.recap.lerobot_export import video_export as lerobot_video_export
from work.recap.scripts import gr00t_label_dashboard
from work.recap.state_conditioned import build_training_set as training_set_impl


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


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "\n".join(
        json.dumps(record, ensure_ascii=True, separators=(",", ":"))
        for record in records
    )
    path.write_text(serialized + "\n", encoding="utf-8")
    return path


def _label_row(
    *,
    prompt: str,
    indicator_mode: str,
    phase: str,
    mode: str,
    epsilon_l: float,
    repeat_index: int = 0,
    recovery_oversample_factor: int = 3,
    training_view: str = "C1",
    sample_id: str | None = None,
) -> dict[str, object]:
    carrier_text = text_indicator.build_canonical_text_indicator(prompt, indicator_mode)
    indicator_value = (
        1 if indicator_mode == text_indicator.TEXT_INDICATOR_POSITIVE else 0
    )
    suffix = sample_id or f"{phase.lower()}_{mode.lower()}_{repeat_index}"
    return {
        "sample_id": f"sample::{suffix}",
        "source_sample_key": f"source::{suffix}",
        "training_view": training_view,
        "carrier_text_v1": carrier_text,
        "policy_condition.phase": phase,
        "policy_condition.mode": mode,
        "indicator_I": indicator_value,
        "epsilon_l": float(epsilon_l),
        "repeat_index": int(repeat_index),
        "recovery_oversample_factor": int(recovery_oversample_factor),
    }


def _stats_payload(row_count: int) -> dict[str, object]:
    return {
        "schema_version": training_set_impl.SCHEMA_VERSION,
        "artifact_kind": "state_conditioned_sft_stats",
        "counts": {"unified_base_row_count": int(row_count)},
    }


def test_state_conditioned_materialization_uses_carrier_text_v1_for_dashboard_tasks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    carrier_text = text_indicator.build_canonical_text_indicator(
        "pick up the apple and place it on the plate",
        text_indicator.TEXT_INDICATOR_POSITIVE,
    )
    policy_condition_text = "[PolicyCondition-v1]\nPHASE=TRANSPORT\nMODE=RECOVERY"
    base_rows = [
        {
            "sample_id": "sample_001",
            "source_bucket": "canonical_bucket_A",
            "source_episode_id": "episode_001",
            "source_t": 0,
            "carrier_text_v1": carrier_text,
            "canonical_policy_condition_text": policy_condition_text,
        }
    ]

    monkeypatch.setattr(
        training_set_impl,
        "_build_source_episode_dataset_index",
        lambda prerequisites: {},
    )
    monkeypatch.setattr(
        training_set_impl,
        "_build_formal_pseudodemo_snapshot_index",
        lambda prerequisites: ({}, {}),
    )

    def _fake_source_step_export_spec(raw_row, **kwargs):
        del kwargs
        state_dims = {
            key: 1 for key in training_set_impl.lerobot_v2_export.STATE_KEY_ORDER_LOCK
        }
        action_dims = {
            key: 1 for key in training_set_impl.lerobot_v2_export.ACTION_KEY_ORDER_LOCK
        }
        state_vector = np.asarray(
            list(range(len(state_dims))),
            dtype=np.float32,
        )
        action_vector = np.asarray(
            [list(range(len(action_dims)))],
            dtype=np.float32,
        )
        return {
            "source_dataset_episode_id": "episode_001",
            "source_video_dir_archived": "/tmp/fake_videos",
            "source_t": 0,
            "source_n_policy_steps": 1,
            "task_text": str(raw_row["carrier_text_v1"]),
            "state_vector": state_vector,
            "action_vector": action_vector,
            "state_dims": state_dims,
            "action_dims": action_dims,
        }

    monkeypatch.setattr(
        training_set_impl,
        "_source_step_export_spec",
        _fake_source_step_export_spec,
    )
    monkeypatch.setattr(
        training_set_impl,
        "_build_episode_video_export_spec",
        lambda **kwargs: {"episode_index": int(kwargs["episode_index"])},
    )

    def _fake_attach_videos_to_existing_lerobot_dataset(**kwargs):
        output_dataset_dir = Path(str(kwargs["output_dataset_dir"]))
        meta_dir = output_dataset_dir / "meta"
        video_map_path = meta_dir / "video_map.json"
        video_map_path.write_text(
            json.dumps({"total_videos": len(kwargs["episode_video_specs"])}),
            encoding="utf-8",
        )
        return lerobot_video_export.LeRobotV2ExportWithVideoResult(
            output_dataset_dir=output_dataset_dir,
            total_episodes=len(kwargs["episode_video_specs"]),
            total_videos=len(kwargs["episode_video_specs"]),
            video_path_template=lerobot_video_export.DEFAULT_VIDEO_PATH_TEMPLATE,
            image_key=str(kwargs["image_key"]),
            original_key=str(kwargs["original_key"]),
            video_map_path=video_map_path,
        )

    monkeypatch.setattr(
        training_set_impl.lerobot_v2_export_with_video,
        "attach_videos_to_existing_lerobot_dataset",
        _fake_attach_videos_to_existing_lerobot_dataset,
    )

    result = training_set_impl.materialize_lerobot_training_dataset(
        output_dir=tmp_path / "training_dataset",
        base_rows=base_rows,
        prerequisites={},
    )

    dataset_root = Path(str(result["dataset_root"]))
    tasks = _read_jsonl(
        dataset_root / "meta" / training_set_impl.lerobot_v2_export.META_TASKS_JSONL
    )
    episodes = _read_jsonl(
        dataset_root / "meta" / training_set_impl.lerobot_v2_export.META_EPISODES_JSONL
    )
    info = _read_json(
        dataset_root / "meta" / training_set_impl.lerobot_v2_export.META_INFO_JSON
    )

    assert tasks == [{"task_index": 0, "task": carrier_text}]
    assert episodes[0]["tasks"] == [carrier_text]
    assert episodes[0]["tasks"] != [policy_condition_text]
    assert info["task_text_field"] == "carrier_text_v1"
    assert info["carrier_route"] == "carrier_text_v1"
    assert (
        info["carrier_schema_version"]
        == text_indicator.RECAP_TEXT_INDICATOR_SCHEMA_VERSION
    )


def test_state_conditioned_materialization_rejects_missing_carrier_text_v1_even_with_policy_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        training_set_impl,
        "_build_source_episode_dataset_index",
        lambda prerequisites: {},
    )
    monkeypatch.setattr(
        training_set_impl,
        "_build_formal_pseudodemo_snapshot_index",
        lambda prerequisites: ({}, {}),
    )
    with pytest.raises((TypeError, ValueError), match="carrier_text_v1"):
        training_set_impl.materialize_lerobot_training_dataset(
            output_dir=tmp_path / "training_dataset_missing_carrier",
            base_rows=[
                {
                    "sample_id": "sample_001",
                    "source_bucket": "canonical_bucket_A",
                    "source_episode_id": "episode_001",
                    "source_t": 0,
                    "canonical_policy_condition_text": "[PolicyCondition-v1]\\nPHASE=TRANSPORT\\nMODE=RECOVERY",
                }
            ],
            prerequisites={},
        )


def test_label_dashboard_groups_tasks_by_indicator_neutral_carrier_surface() -> None:
    prompt = "pick up the apple and place it on the plate"
    label_rows = [
        _label_row(
            prompt=prompt,
            indicator_mode=text_indicator.TEXT_INDICATOR_POSITIVE,
            phase="TRANSPORT",
            mode="RECOVERY",
            epsilon_l=0.15,
        ),
        _label_row(
            prompt=prompt,
            indicator_mode=text_indicator.TEXT_INDICATOR_NEGATIVE,
            phase="APPROACH",
            mode="NOMINAL",
            epsilon_l=0.30,
        ),
    ]

    dashboard = label_policy.build_label_dashboard(
        label_rows=label_rows,
        stats=_stats_payload(len(label_rows)),
    )

    per_task = dashboard["summaries"]["per_task"]
    assert len(per_task) == 1
    assert per_task[0]["task"] == prompt
    assert sorted(per_task[0]["carrier_text_variants"]) == sorted(
        [str(row["carrier_text_v1"]) for row in label_rows]
    )
    assert per_task[0]["positive_count"] == 1
    assert per_task[0]["negative_count"] == 1
    assert (
        dashboard["task_grouping_authority"]["task_surface_field"] == "carrier_text_v1"
    )
    assert (
        dashboard["task_grouping_authority"]["task_grouping_key"]
        == "carrier_text_v1_without_indicator_suffix"
    )


def test_label_policy_detects_all_positive_and_all_negative_indicator_collapse() -> (
    None
):
    positive_rows = [
        _label_row(
            prompt="task one",
            indicator_mode=text_indicator.TEXT_INDICATOR_POSITIVE,
            phase="TRANSPORT",
            mode="RECOVERY",
            epsilon_l=0.10,
        ),
        _label_row(
            prompt="task two",
            indicator_mode=text_indicator.TEXT_INDICATOR_POSITIVE,
            phase="PLACE",
            mode="RECOVERY",
            epsilon_l=0.20,
        ),
    ]
    negative_rows = [
        _label_row(
            prompt="task one",
            indicator_mode=text_indicator.TEXT_INDICATOR_NEGATIVE,
            phase="TRANSPORT",
            mode="NOMINAL",
            epsilon_l=0.10,
        ),
        _label_row(
            prompt="task two",
            indicator_mode=text_indicator.TEXT_INDICATOR_NEGATIVE,
            phase="PLACE",
            mode="NOMINAL",
            epsilon_l=0.20,
        ),
    ]

    positive_policy = label_policy.build_label_policy(
        label_rows=positive_rows,
        stats=_stats_payload(len(positive_rows)),
    )
    negative_policy = label_policy.build_label_policy(
        label_rows=negative_rows,
        stats=_stats_payload(len(negative_rows)),
    )

    assert positive_policy["collapse_checks"]["all_positive_indicator"] is True
    assert positive_policy["collapse_checks"]["all_negative_indicator"] is False
    assert negative_policy["collapse_checks"]["all_positive_indicator"] is False
    assert negative_policy["collapse_checks"]["all_negative_indicator"] is True


def test_gr00t_label_dashboard_script_writes_dashboard_and_policy_sidecars(
    tmp_path: Path,
) -> None:
    prompt = "pick up the apple and place it on the plate"
    label_rows = [
        _label_row(
            prompt=prompt,
            indicator_mode=text_indicator.TEXT_INDICATOR_POSITIVE,
            phase="TRANSPORT",
            mode="RECOVERY",
            epsilon_l=0.15,
            repeat_index=0,
            sample_id="shared_recovery_0",
        ),
        _label_row(
            prompt=prompt,
            indicator_mode=text_indicator.TEXT_INDICATOR_POSITIVE,
            phase="TRANSPORT",
            mode="RECOVERY",
            epsilon_l=0.15,
            repeat_index=1,
            sample_id="shared_recovery_1",
        ),
        _label_row(
            prompt=prompt,
            indicator_mode=text_indicator.TEXT_INDICATOR_NEGATIVE,
            phase="APPROACH",
            mode="NOMINAL",
            epsilon_l=0.30,
            repeat_index=0,
            sample_id="approach_nominal_0",
        ),
    ]
    labels_path = _write_jsonl(
        tmp_path / "state_conditioned_sft_labels.jsonl", label_rows
    )
    stats_path = _write_json(
        tmp_path / "state_conditioned_sft_stats.json",
        _stats_payload(len(label_rows)),
    )
    output_dir = tmp_path / "dashboard_out"

    rc = gr00t_label_dashboard.main(
        [
            "--labels-jsonl",
            str(labels_path),
            "--stats-json",
            str(stats_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    dashboard = _read_json(output_dir / label_policy.LABEL_DASHBOARD_JSON_NAME)
    policy = _read_json(output_dir / label_policy.LABEL_POLICY_JSON_NAME)
    positive_duplication_policy = policy["positive_duplication_policy"]
    assert isinstance(positive_duplication_policy, Mapping)
    target_condition = positive_duplication_policy["target_condition"]
    assert isinstance(target_condition, Mapping)
    assert dashboard["artifact_kind"] == label_policy.LABEL_DASHBOARD_ARTIFACT_KIND
    assert policy["artifact_kind"] == label_policy.LABEL_POLICY_ARTIFACT_KIND
    assert positive_duplication_policy["enabled"] is True
    assert positive_duplication_policy["factor"] == 3
    assert target_condition["indicator_I"] == 1
    assert target_condition["mode_values"] == ["RECOVERY"]
