from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DOC = REPO_ROOT / "agent/exchange/openpi_libero_relabel_stats_gate.md"
DATASET_ROOT = (
    REPO_ROOT
    / "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1"
)
META_DIR = DATASET_ROOT / "meta"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"expected JSON object at {path}"
    return cast(dict[str, object], payload)


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict), f"expected dict, got {type(value).__name__}"
    return cast(dict[str, object], value)


def _sequence(value: object) -> list[object]:
    assert isinstance(value, list), f"expected list, got {type(value).__name__}"
    return cast(list[object], value)


def _as_int(value: object) -> int:
    assert not isinstance(value, bool)
    return int(cast(int | str | float, value))


def _as_float(value: object) -> float:
    assert not isinstance(value, bool)
    return float(cast(int | str | float, value))


def _assert_required_items(text: str, required: list[str], *, context: str) -> None:
    for item in required:
        assert item in text, f"missing {context}: {item}"


def test_relabel_stats_gate_doc_freezes_required_artifacts_and_shapes() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "openpi LIBERO relabel stats / sample-universe gate 合同",
        "route_id=official_native_8d_recap_relabels_v1",
        "meta/relabel_stats_report.json",
        "meta/dataset_fingerprint.json",
        "meta/episode_universe_hash.txt",
        "meta/split_filter_hash.txt",
        "observation.state.shape=[8]",
        "action.shape=[7]",
        "state.shape=[8]",
        "action.shape=[7]",
        "state_shape=[8]",
        "action_shape=[7]",
    ]
    _assert_required_items(text, required, context="artifact/shape contract item")


def test_relabel_stats_gate_doc_freezes_required_recap_columns() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "required_recap_columns",
        "recap_m2.t",
        "recap_m2.return_G",
        "recap_m2.value_V",
        "recap_m2.advantage_A",
        "recap_m2.advantage_input",
        "recap_m2.epsilon_l",
        "recap_m2.indicator_I",
        "recap_m2.prompt_raw",
        "recap_m2.prompt_conditioned",
        "关键列分布无法生成，必须 `fail-fast`",
    ]
    _assert_required_items(text, required, context="recap column gate item")


def test_relabel_stats_gate_doc_freezes_drift_to_rerun_semantics() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "sample universe drift",
        "fingerprint drift",
        "split filter drift",
        "rerun_required=true",
        "full_rerun_required=true",
        "sample universe drift -> final_status=rerun_required",
        "split filter drift -> final_status=rerun_required",
        "fingerprint drift -> final_status=full_rerun_required",
        "不得静默复用旧 `relabel_stats_report.json`",
        "不得静默复用旧统计结果",
    ]
    _assert_required_items(text, required, context="drift/rerun gate item")


def test_relabel_stats_gate_doc_freezes_fail_fast_and_blocked_semantics() -> None:
    text = DOC.read_text(encoding="utf-8")
    required = [
        "fail-fast / blocker / rerun 规则",
        "machine-checkable blocker",
        "shape drift -> final_status=blocked",
        "missing recap_m2.* columns -> final_status=blocked",
        "dataset_fingerprint 无法计算或无法比对，必须 `fail-fast`",
        "episode_universe_hash 无法计算或无法比对，必须 `fail-fast`",
        "split_filter_hash 无法计算或无法比对，必须 `fail-fast`",
        "本合同只冻结 stats gate surface，不实现 `summarize_libero_relabel_stats.py`",
    ]
    _assert_required_items(text, required, context="fail-fast/blocker gate item")


def test_emitted_dataset_fingerprint_contains_plan_required_fields() -> None:
    fingerprint = _read_json(META_DIR / "dataset_fingerprint.json")
    report = _read_json(META_DIR / "relabel_stats_report.json")
    episode_universe_hash = (
        (META_DIR / "episode_universe_hash.txt").read_text(encoding="utf-8").strip()
    )
    split_filter_hash = (
        (META_DIR / "split_filter_hash.txt").read_text(encoding="utf-8").strip()
    )

    required_fields = [
        "route_id",
        "schema_version",
        "source_dataset_name",
        "state_dim",
        "action_dim",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "episodes_hash",
        "tasks_hash",
        "parquet_inventory_hash",
        "recap_advantage_input_contract",
        "fingerprint_sha256",
    ]
    for field in required_fields:
        assert field in fingerprint, f"missing fingerprint field: {field}"

    assert fingerprint["route_id"] == "official_native_8d_recap_relabels_v1"
    assert (
        fingerprint["source_dataset_name"] == "physical_intelligence_libero_official_8d"
    )
    assert fingerprint["state_dim"] == 8
    assert fingerprint["action_dim"] == 7
    assert fingerprint["total_episodes"] == 1693
    assert fingerprint["total_frames"] == 273465
    assert fingerprint["total_tasks"] == 40
    assert fingerprint["episodes_hash"] == episode_universe_hash
    assert fingerprint["episode_universe_hash"] == episode_universe_hash
    assert fingerprint["split_filter_hash"] == split_filter_hash
    advantage_contract = _mapping(fingerprint["recap_advantage_input_contract"])
    assert advantage_contract["contract_version"] == ("full_recap_continuous_adv_v2")
    for hash_field in [
        "episodes_hash",
        "tasks_hash",
        "parquet_inventory_hash",
        "fingerprint_sha256",
    ]:
        value = fingerprint[hash_field]
        assert isinstance(value, str) and len(value) == 64

    assert report["episodes_hash"] == fingerprint["episodes_hash"]
    assert report["tasks_hash"] == fingerprint["tasks_hash"]
    assert report["parquet_inventory_hash"] == fingerprint["parquet_inventory_hash"]


def test_emitted_relabel_stats_report_contains_plan_required_summary_surfaces() -> None:
    report = _read_json(META_DIR / "relabel_stats_report.json")

    for field in [
        "per_task_rollup",
        "recap_column_non_null_rates",
        "positive_rate",
        "indicator_distribution_summary",
        "task_distribution_summary",
        "prompt_distribution_summary",
    ]:
        assert field in report, f"missing report field: {field}"

    per_task_rollup = [_mapping(row) for row in _sequence(report["per_task_rollup"])]
    assert len(per_task_rollup) == _as_int(report["total_tasks"]) == 40
    assert sum(_as_int(row["episode_count"]) for row in per_task_rollup) == _as_int(
        report["total_episodes"]
    )
    assert sum(_as_int(row["frame_count"]) for row in per_task_rollup) == _as_int(
        report["total_frames"]
    )
    assert all(
        {
            "task_index",
            "task",
            "episode_count",
            "frame_count",
            "positive_rate",
        }.issubset(row)
        for row in per_task_rollup
    )

    non_null_rates = _mapping(report["recap_column_non_null_rates"])
    required_recap_columns = [
        "recap_m2.t",
        "recap_m2.return_G",
        "recap_m2.value_V",
        "recap_m2.advantage_A",
        "recap_m2.advantage_input",
        "recap_m2.epsilon_l",
        "recap_m2.indicator_I",
        "recap_m2.prompt_raw",
        "recap_m2.prompt_conditioned",
    ]
    assert set(required_recap_columns).issubset(non_null_rates)
    for column in required_recap_columns:
        payload = _mapping(non_null_rates[column])
        assert _as_int(payload["total_count"]) == _as_int(report["total_frames"])
        assert 0.0 <= _as_float(payload["non_null_rate"]) <= 1.0
        assert _as_int(payload["non_null_count"]) + _as_int(
            payload["null_count"]
        ) == _as_int(report["total_frames"])

    assert 0.0 <= _as_float(report["positive_rate"]) <= 1.0
    indicator_distribution = _mapping(report["indicator_distribution_summary"])
    assert _as_int(indicator_distribution["positive_count"]) == _as_int(
        report["indicator_positive_count"]
    )
    assert _as_int(indicator_distribution["negative_count"]) == _as_int(
        report["indicator_negative_count"]
    )
    assert 0.0 <= _as_float(indicator_distribution["entropy_bits"]) <= 1.0

    task_distribution = _mapping(report["task_distribution_summary"])
    assert _as_int(task_distribution["task_count"]) == _as_int(report["total_tasks"])
    assert _as_float(task_distribution["episode_entropy_bits"]) >= 0.0
    assert _as_float(task_distribution["frame_entropy_bits"]) >= 0.0

    prompt_distribution = _mapping(report["prompt_distribution_summary"])
    assert _as_int(prompt_distribution["prompt_raw_unique_count"]) == _as_int(
        report["prompt_raw_unique_count"]
    )
    assert _as_int(prompt_distribution["prompt_conditioned_unique_count"]) == _as_int(
        report["prompt_conditioned_unique_count"]
    )
    assert _as_float(prompt_distribution["prompt_raw_entropy_bits"]) >= 0.0
    assert _as_float(prompt_distribution["prompt_conditioned_entropy_bits"]) >= 0.0
