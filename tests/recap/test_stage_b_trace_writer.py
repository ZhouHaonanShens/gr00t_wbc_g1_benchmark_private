from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from work.recap.stage_b.action_uuid import (  # noqa: E402
    make_action_content_hash,
    make_chain_action_uuid,
    make_contrast_group_uuid,
)
from work.recap.stage_b.array_summary import summarize_array  # noqa: E402
from work.recap.stage_b.schema import TRACE_VERSION, build_seam_trace_schema  # noqa: E402
from work.recap.stage_b.seam_trace_writer import (  # noqa: E402
    SeamTraceWriter,
    run_self_test,
)


def test_chain_uuid_is_stable_and_content_independent() -> None:
    kwargs = {
        "trace_version": TRACE_VERSION,
        "episode_id": "ep0",
        "step_id": 1,
        "seed": 20000,
        "policy_call_index": 0,
        "obs_hash": "obs_hash",
    }

    first = make_chain_action_uuid(**kwargs)
    second = make_chain_action_uuid(**kwargs)

    assert first == second
    assert make_action_content_hash([1.0, 2.0]) != make_action_content_hash([1.0, 3.0])


def test_contrast_group_uuid_excludes_indicator_mode_and_action_content() -> None:
    base = make_contrast_group_uuid(
        trace_version=TRACE_VERSION,
        seed=20000,
        obs_hash="obs_hash",
        frozen_controller_state_hash="controller_hash",
        probe_name="same_obs_triplet",
    )
    repeated = make_contrast_group_uuid(
        trace_version=TRACE_VERSION,
        seed=20000,
        obs_hash="obs_hash",
        frozen_controller_state_hash="controller_hash",
        probe_name="same_obs_triplet",
    )
    changed_controller = make_contrast_group_uuid(
        trace_version=TRACE_VERSION,
        seed=20000,
        obs_hash="obs_hash",
        frozen_controller_state_hash="different_controller_hash",
        probe_name="same_obs_triplet",
    )

    assert base == repeated
    assert base != changed_controller


def test_array_summary_reports_hash_and_nan_inf_counts() -> None:
    summary = summarize_array(np.array([1.0, np.nan, np.inf], dtype=np.float32))

    assert summary["shape"] == [3]
    assert summary["dtype"] == "float32"
    assert summary["nan_count"] == 1
    assert summary["inf_count"] == 1
    assert len(summary["sha256"]) == 64


def test_trace_writer_writes_jsonl_npz_and_does_not_mutate_input(tmp_path: Path) -> None:
    writer = SeamTraceWriter(tmp_path)
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    before = array.copy()
    chain_uuid = make_chain_action_uuid(
        trace_version=TRACE_VERSION,
        episode_id="ep0",
        step_id=0,
        seed=20000,
        policy_call_index=0,
        obs_hash="obs_hash",
    )
    contrast_uuid = make_contrast_group_uuid(
        trace_version=TRACE_VERSION,
        seed=20000,
        obs_hash="obs_hash",
        frozen_controller_state_hash="controller_hash",
        probe_name="same_obs_triplet",
    )

    event = writer.record_array_event(
        stage="policy",
        name="decoded_action",
        episode_id="ep0",
        step_id=0,
        chain_action_uuid=chain_uuid,
        contrast_group_uuid=contrast_uuid,
        seed=20000,
        indicator_mode="positive",
        obs_hash="obs_hash",
        array=array,
    )
    writer.record_array_event(
        stage="controller",
        name="controller_output",
        episode_id="ep0",
        step_id=0,
        chain_action_uuid=chain_uuid,
        missing_stage_reason="true_torque_unobservable",
    )
    flush = writer.flush()

    assert event is not None
    assert np.array_equal(array, before)
    assert flush["events_written"] == 2
    assert flush["arrays_written"] == 1

    lines = [json.loads(line) for line in (tmp_path / "seam_trace.jsonl").read_text().splitlines()]
    assert lines[0]["array_summary"]["shape"] == [2, 2]
    assert lines[0]["array_ref"]["array_key"]
    assert lines[1]["missing_stage_reason"] == "true_torque_unobservable"

    arrays = np.load(tmp_path / lines[0]["array_ref"]["npz_path"])
    np.testing.assert_array_equal(arrays[lines[0]["array_ref"]["array_key"]], array)


def test_disabled_writer_is_noop(tmp_path: Path) -> None:
    writer = SeamTraceWriter(tmp_path, enabled=False)

    event = writer.record_array_event(
        stage="policy",
        name="decoded_action",
        episode_id="ep0",
        step_id=0,
        chain_action_uuid="disabled",
        array=[1, 2, 3],
    )
    flush = writer.flush()

    assert event is None
    assert flush == {"enabled": False, "events_written": 0, "arrays_written": 0}
    assert not any(tmp_path.iterdir())


def test_schema_and_self_test_outputs(tmp_path: Path) -> None:
    schema = build_seam_trace_schema()
    assert schema["properties"]["trace_version"]["const"] == TRACE_VERSION

    report = run_self_test(tmp_path / "self_test")

    assert report["self_test"] == "PASS"
    assert Path(report["jsonl_path"]).exists()
    assert Path(report["schema_path"]).exists()
    assert (tmp_path / "self_test" / "self_test_report.json").exists()
