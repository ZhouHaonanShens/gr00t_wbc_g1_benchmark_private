from __future__ import annotations

import json
from pathlib import Path
import sys
from collections.abc import Sequence
from typing import cast

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap import dataset_aggregation  # noqa: E402
import work.openpi.pipelines.recap.collect as collect_script  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_demo_source(root: Path) -> Path:
    demo_dir = root / "physical_intelligence_libero_official_8d"
    _write_json(
        demo_dir / "meta" / "info.json",
        {
            "schema_version": "official_native_8d_fixture_v1",
            "codebase_version": "v2.0",
            "robot_type": "panda",
            "total_episodes": 2,
            "total_frames": 4,
            "total_tasks": 2,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 10,
            "splits": {"train": "0:2"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {
                "image": {"dtype": "binary", "shape": [1]},
                "wrist_image": {"dtype": "binary", "shape": [1]},
                "state": {"dtype": "float32", "shape": [8]},
                "actions": {"dtype": "float32", "shape": [7]},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
            },
        },
    )
    _write_json(
        demo_dir / "meta" / "stats.json",
        {
            "image": {},
            "wrist_image": {},
            "state": {},
            "actions": {},
            "timestamp": {},
            "frame_index": {},
            "episode_index": {},
            "index": {},
            "task_index": {},
        },
    )
    _write_jsonl(
        demo_dir / "meta" / "tasks.jsonl",
        [
            {"task": "put the bowl on the plate", "task_index": 0},
            {"task": "open the drawer", "task_index": 1},
        ],
    )
    _write_jsonl(
        demo_dir / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["put the bowl on the plate"], "length": 2},
            {"episode_index": 1, "tasks": ["open the drawer"], "length": 2},
        ],
    )
    demo_rows = [
        (
            0,
            0,
            [0.0] * 8,
            [0.1] * 7,
            b"ego-task0-frame0",
            b"wrist-task0-frame0",
            0,
        ),
        (
            0,
            1,
            [1.0] * 8,
            [0.2] * 7,
            b"ego-task0-frame1",
            b"wrist-task0-frame1",
            1,
        ),
        (
            1,
            0,
            [2.0] * 8,
            [0.3] * 7,
            b"ego-task1-frame0",
            b"wrist-task1-frame0",
            2,
        ),
        (
            1,
            1,
            [3.0] * 8,
            [0.4] * 7,
            b"ego-task1-frame1",
            b"wrist-task1-frame1",
            3,
        ),
    ]
    for (
        episode_index,
        task_index,
        state,
        action,
        image,
        wrist_image,
        global_index,
    ) in demo_rows:
        parquet_path = (
            demo_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frame_index = 0 if global_index % 2 == 0 else 1
        frame = pd.DataFrame(
            {
                "image": [image],
                "wrist_image": [wrist_image],
                "state": [state],
                "actions": [action],
                "timestamp": [float(frame_index)],
                "frame_index": [frame_index],
                "episode_index": [episode_index],
                "index": [global_index],
                "task_index": [task_index],
            }
        )
        if parquet_path.exists():
            existing = pd.read_parquet(parquet_path)
            frame = pd.concat([existing, frame], ignore_index=True)
        frame.to_parquet(parquet_path, engine="pyarrow", index=False)
    return demo_dir


def write_policy_checkpoint(root: Path, *, critic_checkpoint_ref: str) -> Path:
    checkpoint_dir = root / "policy" / "best"
    _write_json(
        checkpoint_dir / "train_manifest.json",
        {
            "stage": "sft_fixed_positive",
            "critic_checkpoint_ref": critic_checkpoint_ref,
            "training_route": {
                "critic_checkpoint_ref": critic_checkpoint_ref,
                "source_dataset_dir": str((root / "source_dataset").resolve()),
                "prepared_dataset_dir": str((root / "prepared_dataset").resolve()),
            },
        },
    )
    _write_json(
        checkpoint_dir / "checkpoint_provenance.json",
        {
            "stage": "sft_fixed_positive",
            "critic_checkpoint_ref": critic_checkpoint_ref,
            "variant_derivation": {
                "critic_checkpoint_ref": critic_checkpoint_ref,
                "source_dataset_dir": str((root / "source_dataset").resolve()),
                "prepared_dataset_dir": str((root / "prepared_dataset").resolve()),
            },
        },
    )
    return checkpoint_dir


def _trace_row(
    *, task_id: int, seed: int, trial_idx: int, success: bool
) -> dict[str, object]:
    return {
        "variant": "fixedadv_relabel8d_control_v1",
        "task_id": task_id,
        "seed": seed,
        "trial_idx": trial_idx,
        "success": success,
        "first_success_step": 18 if success else None,
        "executed_steps": 40,
        "max_steps_resolved": 80,
        "success_within_50pct_budget": success,
        "success_within_75pct_budget": success,
        "timeout_flag": not success,
        "deviation_notes": [] if success else ["timeout"],
    }


def patch_rollout_eval(
    *,
    monkeypatch: pytest.MonkeyPatch,
    critic_checkpoint_ref: str,
    success_pattern: tuple[bool, ...] = (True, False, True),
) -> None:
    def _fake_rollout_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        output_dir = Path(argv[argv.index("--output-dir") + 1]).resolve()
        indicator_mode = argv[argv.index("--indicator-mode") + 1]
        output_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = output_dir / "_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            _trace_row(
                task_id=index % 2, seed=7000 + index, trial_idx=0, success=success
            )
            for index, success in enumerate(success_pattern)
        ]
        _write_jsonl(output_dir / "per_episode_trace.jsonl", rows)
        _write_json(
            staging_dir / "rollout_input_summary.json",
            {
                "runtime_prompting": {
                    "indicator_mode_requested": indicator_mode,
                    "indicator_mode": indicator_mode
                    if indicator_mode != "cfg"
                    else "positive",
                    "indicator_source": "cli.indicator_mode",
                    "prompt_text_surface": "canonical_text_indicator",
                    "prompt_text": "put the bowl on the plate\nAdvantage: positive",
                    "critic_checkpoint_ref": critic_checkpoint_ref,
                }
            },
        )
        return 0

    monkeypatch.setattr(
        collect_script.libero_rollout_eval_v21, "main", _fake_rollout_main
    )


def run_collection_fixture(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    success_pattern: tuple[bool, ...] = (True, False, True),
) -> tuple[Path, Path, Path, str]:
    demo_dir = write_demo_source(tmp_path)
    critic_checkpoint_ref = str((tmp_path / "critic" / "best").resolve())
    policy_checkpoint = write_policy_checkpoint(
        tmp_path,
        critic_checkpoint_ref=critic_checkpoint_ref,
    )
    patch_rollout_eval(
        monkeypatch=monkeypatch,
        critic_checkpoint_ref=critic_checkpoint_ref,
        success_pattern=success_pattern,
    )
    output_dir = tmp_path / "collect_output"
    bundle = collect_script.run_collection(
        collect_script.CollectConfig(
            policy_checkpoint=policy_checkpoint,
            critic_checkpoint=None,
            indicator_mode="positive",
            task_suite_name="libero_spatial",
            task_ids=(0, 1),
            episodes=len(success_pattern),
            output_dir=output_dir,
            demo_dir=demo_dir,
        )
    )
    assert bundle.output_dir == output_dir.resolve()
    return demo_dir, policy_checkpoint, output_dir, critic_checkpoint_ref


def test_collect_script_materializes_task9_collection_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo_dir, policy_checkpoint, output_dir, critic_checkpoint_ref = (
        run_collection_fixture(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    )

    manifest = dataset_aggregation.read_json(
        output_dir / dataset_aggregation.COLLECTION_MANIFEST_NAME
    )
    rows = dataset_aggregation.read_jsonl(
        output_dir / dataset_aggregation.AUTONOMOUS_TRIALS_NAME
    )
    correction_rows = dataset_aggregation.read_jsonl(
        output_dir / dataset_aggregation.CORRECTION_SEGMENTS_NAME
    )
    canonical_source = cast(dict[str, object], manifest["canonical_demo_source"])

    assert manifest["route_id"] == dataset_aggregation.COLLECTION_ROUTE_ID
    assert manifest["policy_checkpoint_ref"] == str(policy_checkpoint)
    assert manifest["critic_checkpoint_ref"] == critic_checkpoint_ref
    assert manifest["policy_stage"] == "sft_fixed_positive"
    assert manifest["episodes_materialized"] == 3
    assert manifest["task_ids"] == [0, 1]
    assert manifest["indicator_mode"] == "positive"
    assert manifest["prompt_text_surface"] == "canonical_text_indicator"
    assert canonical_source["status"] == "ready"
    assert canonical_source["dataset_dir"] == str(demo_dir)

    assert [row["label"] for row in rows] == ["success", "failure", "success"]
    assert [row["indicator_I"] for row in rows] == [1, 0, 1]
    assert all(row["is_correction"] is False for row in rows)
    assert all(row["critic_checkpoint_ref"] == critic_checkpoint_ref for row in rows)
    assert correction_rows == []
