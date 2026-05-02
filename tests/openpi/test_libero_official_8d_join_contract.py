from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.sources.libero_official.join import (
    BLOCKED_EXIT_CODE,
    ROUTE_ID,
    assess_join,
    main,
)


DOC = REPO_ROOT / "agent/exchange/openpi_libero_official_8d_join_contract.md"
OFFICIAL_DIR = (
    REPO_ROOT
    / "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d"
)
RECAP_DIR = REPO_ROOT / "agent/artifacts/lerobot_datasets/recap_reward_approved_v1"


def _mapping(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"expected mapping, got {type(raw).__name__}")
    return cast(Mapping[str, object], raw)


def _mapping_list(raw: object) -> Sequence[Mapping[str, object]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise TypeError(f"expected sequence of mappings, got {type(raw).__name__}")
    values: list[Mapping[str, object]] = []
    for item in raw:
        values.append(_mapping(item))
    return tuple(values)


def _string_list(raw: object) -> Sequence[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise TypeError(f"expected sequence of strings, got {type(raw).__name__}")
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise TypeError(f"expected string item, got {type(item).__name__}")
        values.append(item)
    return tuple(values)


def test_join_contract_doc_freezes_blocker_first_scope_and_rejected_weak_keys() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "openpi LIBERO official 8D recap label join 合同",
        "route_id=safe_recap_label_join_onto_official_8d_v1",
        "blocker-first / fail-fast",
        "cross-dataset task identity",
        "cross-dataset frame identity",
        "verified episode/frame crosswalk",
        "task-text universes not proven compatible",
        "weak-key-only join is rejected",
        "episode_index` / `index` / `timestamp` 只能作为 rejected weak keys",
        "当前仓内真实输入应输出 `blocked`",
    ]
    for item in required:
        assert item in text, f"missing join contract item: {item}"


def test_assess_join_reports_current_repo_local_blockers() -> None:
    report = assess_join(
        official_dataset_dir=OFFICIAL_DIR,
        recap_label_dataset_dir=RECAP_DIR,
        output_dir=REPO_ROOT
        / "agent/artifacts/lerobot_datasets/test_join_contract_tmp",
    )

    assert report["route_id"] == ROUTE_ID
    assert report["final_status"] == "blocked"
    assert report["materialization_allowed"] is False
    blocker_codes = {item["code"] for item in _mapping_list(report["hard_blockers"])}
    assert blocker_codes == {
        "missing_cross_dataset_task_identity",
        "missing_verified_episode_frame_crosswalk",
        "missing_cross_dataset_frame_identity",
        "task_text_universe_not_proven_compatible",
        "weak_key_only_join_rejected",
    }
    rejected_keys = {
        item["key"] for item in _mapping_list(report["join_keys_rejected"])
    }
    assert {"episode_index", "index", "timestamp"}.issubset(rejected_keys)
    source_summary = _mapping(report["source_summary"])
    official_summary = _mapping(source_summary["official"])
    recap_summary = _mapping(source_summary["recap"])
    assert official_summary["declared_total_episodes"] == 1693
    assert official_summary["observed_episode_files"] == 1693
    assert recap_summary["declared_total_episodes"] == 200


def test_materializer_writes_machine_checkable_blocker_reports_for_current_inputs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "physical_intelligence_libero_official_8d_recap_join_v1"

    rc = main(
        [
            "--official-dataset-dir",
            str(OFFICIAL_DIR),
            "--recap-label-dataset-dir",
            str(RECAP_DIR),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == BLOCKED_EXIT_CODE
    join_report = _mapping(
        json.loads((output_dir / "join_report.json").read_text(encoding="utf-8"))
    )
    unmatched = _mapping(
        json.loads(
            (output_dir / "unmatched_conflicts.json").read_text(encoding="utf-8")
        )
    )
    assert join_report["route_id"] == ROUTE_ID
    assert join_report["final_status"] == "blocked"
    assert join_report["dedicated_blocker_exit_code"] == BLOCKED_EXIT_CODE
    matched_counts = _mapping(join_report["matched_counts"])
    unmatched_counts = _mapping(join_report["unmatched_counts"])
    assert matched_counts["matched_frame_count"] == 0
    assert unmatched_counts["unmatched_official_frame_count"] == 273465
    assert unmatched["route_id"] == ROUTE_ID
    assert unmatched["final_status"] == "blocked"
    assert set(_string_list(unmatched["hard_blocker_codes"])) == {
        "missing_cross_dataset_task_identity",
        "missing_verified_episode_frame_crosswalk",
        "missing_cross_dataset_frame_identity",
        "task_text_universe_not_proven_compatible",
        "weak_key_only_join_rejected",
    }
    assert {item["key"] for item in _mapping_list(unmatched["rejected_join_keys"])} >= {
        "episode_index",
        "index",
        "timestamp",
    }
