from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import pytest
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.stage_b import p1_phase1_aggregator as agg  # noqa: E402


P0A = "P0a_post_recap_nenvs_1"
P0B = "P0b_base_reference_nenvs_1"
SCHEMA_PATH = (
    REPO_ROOT / "work/recap/stage_b/schemas/p1_episode_record_v1.schema.json"
)

REQUIRED_STOP_CODES = {
    "STOP_POST_UNEXPECTED_RECOVERY",
    "STOP_BASE_INFRA_DRIFT",
    "STOP_BASE_OOR_LOW",
    "STOP_BASE_OOR_HIGH",
    "STOP_SERVER_FAIL",
    "STOP_NAN",
    "STOP_INF",
    "STOP_MUJOCO_CRASH",
    "STOP_TIMEOUT",
    "STOP_VRAM_HEADROOM",
    "STOP_DRYRUN_BLOCKER",
    "STOP_HUNG",
    "STOP_SCHEMA_DRIFT",
    "STOP_PHASE0_LOG_DRIFT",
}


def _call(names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
    for name in names:
        func = getattr(agg, name, None)
        if callable(func):
            return func(*args, **kwargs)
    joined = ", ".join(names)
    raise AssertionError(f"p1_phase1_aggregator must expose one of: {joined}")


def _stop_code(result: Any) -> str | None:
    if result is None or result is False:
        return None
    if isinstance(result, str):
        return result
    if isinstance(result, tuple) and result:
        return _stop_code(result[0])
    if isinstance(result, dict):
        return result.get("stop_code") or result.get("code") or result.get("reason")
    return getattr(result, "stop_code", None)


def _classify_stop(snapshot: dict[str, Any]) -> str | None:
    if callable(getattr(agg, "evaluate_stop_table", None)):
        result = agg.evaluate_stop_table(
            cell=snapshot["cell"],
            completed_episodes=int(snapshot.get("completed_episodes", 0)),
            success_count=int(snapshot.get("success_count", 0)),
            status=str(snapshot.get("status", "PASS")),
            server_failed=bool(snapshot.get("server_failed", False)),
            mujoco_crash=bool(snapshot.get("mujoco_crash", False)),
            nan_count=int(snapshot.get("nan_count", 0)),
            inf_count=int(snapshot.get("inf_count", 0)),
            wall_clock_s=21601.0 if snapshot.get("timeout") else None,
            timeout_s=21600.0,
            peak_vram_mib=snapshot.get(
                "peak_vram_mib", snapshot.get("gpu_memory_used_mib")
            ),
            total_vram_mib=snapshot.get(
                "total_vram_mib", snapshot.get("gpu_memory_total_mib")
            ),
            gpu_headroom_floor_mib=int(
                snapshot.get("gpu_headroom_floor_mib", 4096)
            ),
            dryrun_blocker=bool(
                snapshot.get("dryrun_blocker", snapshot.get("dryrun_failed", False))
            ),
            schema_drift=bool(snapshot.get("schema_drift", False)),
            phase0_log_drift=bool(snapshot.get("phase0_log_drift", False)),
        )
        return _stop_code(result)
    result = _call(
        ("classify_stop", "classify_stop_condition", "evaluate_stop_condition"),
        snapshot,
    )
    return _stop_code(result)


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _schema_errors(record: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(_schema())
    return [error.message for error in validator.iter_errors(record)]


def _record(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema_version": "p1_episode_record_v1",
        "cell": P0B,
        "seed": 20000,
        "indicator_mode": "positive",
        "outer_steps": 36,
        "success": False,
        "terminated": False,
        "truncated": True,
        "max_apple_lift_z": 0.55,
        "final_apple_height_z": 0.30,
        "failure_reason": "outer_step_budget_exhausted",
        "failure_stage_guess": "outer_step_budget",
    }
    payload.update(overrides)
    return payload


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _build_episode_record(
    *,
    cell: str,
    seed: int,
    indicator_mode: str,
    runner_episode: dict[str, Any],
    steps_path: Path,
) -> dict[str, Any]:
    if callable(getattr(agg, "derive_episode_record", None)):
        return agg.derive_episode_record(
            cell=cell,
            seed=seed,
            indicator_mode=indicator_mode,
            runner_episode=runner_episode,
            steps_path=steps_path,
        )
    if callable(getattr(agg, "build_episode_record", None)):
        return agg.build_episode_record(
            cell=cell,
            runner_episode_record=runner_episode,
            steps_path=steps_path,
            process_exited=True,
            schema_path=SCHEMA_PATH,
        )
    raise AssertionError("p1_phase1_aggregator must expose build_episode_record")


def _streaming_watchdog_stop_code(samples: list[dict[str, Any]]) -> str | None:
    evaluator_names = (
        "evaluate_streaming_watchdog",
        "evaluate_p0b_first5_watchdog",
        "streaming_watchdog_stop_code",
    )
    if any(callable(getattr(agg, name, None)) for name in evaluator_names):
        return _stop_code(_call(evaluator_names, samples))
    if not hasattr(agg, "P0bFirstFiveWatchdog"):
        raise AssertionError("p1_phase1_aggregator must expose P0bFirstFiveWatchdog")

    watchdog = agg.P0bFirstFiveWatchdog(quiescent_ticks_required=2, first_seed_count=5)
    decision = None
    for sample in samples:
        positive = sample["summary"]["mode_summaries"]["positive"]
        streaming_sample = agg.StreamingSample(
            episodes_completed=int(positive["episodes_completed"]),
            success_count=int(positive["success_count"]),
            mtime_ns=int(sample["summary_mtime_ns"]),
        )
        decision = watchdog.tick(streaming_sample)
    return _stop_code(decision)


def _hung_watchdog_stop_code(samples: list[dict[str, Any]]) -> str | None:
    evaluator_names = (
        "evaluate_hung_child_watchdog",
        "hung_child_watchdog_stop_code",
        "evaluate_hung_watchdog",
    )
    if any(callable(getattr(agg, name, None)) for name in evaluator_names):
        return _stop_code(_call(evaluator_names, samples))
    if not hasattr(agg, "HungChildWatchdog"):
        raise AssertionError("p1_phase1_aggregator must expose HungChildWatchdog")

    watchdog = agg.HungChildWatchdog(required_stagnant_samples=4, cell=P0A)
    decision = None
    for sample in samples:
        decision = watchdog.tick(
            vram_used_mib=int(sample["peak_vram_mib"]),
            episode_end_count=int(sample["episode_end_count"]),
            episode_jsonl_line_count=int(sample["episodes_jsonl_line_count"]),
        )
    return _stop_code(decision)


def _archive_phase0_log(source: Path, archive: Path, sha_path: Path) -> None:
    if callable(getattr(agg, "pin_phase0_log_archive", None)):
        agg.pin_phase0_log_archive(
            source_path=source,
            archive_path=archive,
            sha_path=sha_path,
        )
        return
    if callable(getattr(agg, "archive_phase0_log_with_sha_pin", None)):
        agg.archive_phase0_log_with_sha_pin(
            source_log=source,
            archive_log=archive,
            sha256_path=sha_path,
        )
        return
    raise AssertionError("p1_phase1_aggregator must expose phase0 SHA pin helper")


def _verify_phase0_log_archive(
    archive: Path, sha_path: Path, source: Path
) -> str | None:
    if callable(getattr(agg, "verify_phase0_log_archive", None)):
        return _stop_code(
            agg.verify_phase0_log_archive(archive_path=archive, sha_path=sha_path)
        )
    try:
        agg.archive_phase0_log_with_sha_pin(
            source_log=source,
            archive_log=archive,
            sha256_path=sha_path,
        )
    except Exception as exc:
        return _stop_code(exc) or ("STOP_PHASE0_LOG_DRIFT" if "SHA" in str(exc) else None)
    return None


def test_stop_code_table_covers_phase1_contract() -> None:
    assert REQUIRED_STOP_CODES <= set(getattr(agg, "STOP_CODES", ()))


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (
            {
                "cell": P0A,
                "check_scope": "final_summary",
                "completed_episodes": 30,
                "success_count": 26,
            },
            "STOP_POST_UNEXPECTED_RECOVERY",
        ),
        (
            {
                "cell": P0A,
                "check_scope": "final_summary",
                "completed_episodes": 30,
                "success_count": 27,
            },
            "STOP_POST_UNEXPECTED_RECOVERY",
        ),
        (
            {
                "cell": P0A,
                "check_scope": "final_summary",
                "completed_episodes": 30,
                "success_count": 28,
            },
            "STOP_POST_UNEXPECTED_RECOVERY",
        ),
        (
            {
                "cell": P0B,
                "check_scope": "final_summary",
                "completed_episodes": 30,
                "success_count": 0,
            },
            "STOP_BASE_OOR_LOW",
        ),
        (
            {
                "cell": P0B,
                "check_scope": "final_summary",
                "completed_episodes": 30,
                "success_count": 1,
            },
            "STOP_BASE_OOR_LOW",
        ),
        (
            {
                "cell": P0B,
                "check_scope": "final_summary",
                "completed_episodes": 30,
                "success_count": 11,
            },
            "STOP_BASE_OOR_LOW",
        ),
        (
            {
                "cell": P0B,
                "check_scope": "final_summary",
                "completed_episodes": 30,
                "success_count": 12,
            },
            None,
        ),
        (
            {
                "cell": P0B,
                "check_scope": "final_summary",
                "completed_episodes": 30,
                "success_count": 22,
            },
            None,
        ),
        (
            {
                "cell": P0B,
                "check_scope": "final_summary",
                "completed_episodes": 30,
                "success_count": 23,
            },
            "STOP_BASE_OOR_HIGH",
        ),
        ({"cell": P0A, "nan_count": 1}, "STOP_NAN"),
        ({"cell": P0B, "inf_count": 1}, "STOP_INF"),
        ({"cell": P0A, "server_failed": True}, "STOP_SERVER_FAIL"),
        ({"cell": P0B, "mujoco_crash": True}, "STOP_MUJOCO_CRASH"),
        ({"cell": P0A, "timeout": True}, "STOP_TIMEOUT"),
        (
            {
                "cell": P0B,
                "gpu_memory_used_mib": 94209,
                "gpu_memory_total_mib": 98304,
                "gpu_headroom_floor_mib": 4096,
            },
            "STOP_VRAM_HEADROOM",
        ),
        ({"cell": P0B, "dryrun_failed": True}, "STOP_DRYRUN_BLOCKER"),
        ({"cell": P0B, "schema_drift": True}, "STOP_SCHEMA_DRIFT"),
        ({"cell": P0A, "phase0_log_drift": True}, "STOP_PHASE0_LOG_DRIFT"),
    ],
)
def test_stop_threshold_logic_covers_leader_rows(
    snapshot: dict[str, Any], expected: str | None
) -> None:
    assert _classify_stop(snapshot) == expected


def test_schema_accepts_three_hand_curated_episode_fixtures() -> None:
    Draft202012Validator.check_schema(_schema())
    fixtures = [
        _record(
            cell=P0A,
            success=True,
            terminated=True,
            truncated=False,
            failure_reason=None,
            failure_stage_guess=None,
        ),
        _record(
            cell=P0B,
            success=False,
            terminated=False,
            truncated=True,
            failure_reason="outer_step_budget_exhausted",
            failure_stage_guess="outer_step_budget",
        ),
        _record(
            cell=P0A,
            success=False,
            terminated=False,
            truncated=False,
            failure_reason="mujoco_crash",
            failure_stage_guess=None,
        ),
    ]

    assert all(_schema_errors(fixture) == [] for fixture in fixtures)


@pytest.mark.parametrize(
    "mutated",
    [
        pytest.param(
            lambda: {
                key: value
                for key, value in _record().items()
                if key != "max_apple_lift_z"
            },
            id="missing-required-key",
        ),
        pytest.param(lambda: {**_record(), "debug": "forbidden"}, id="extra-key"),
        pytest.param(lambda: {**_record(), "seed": "20000"}, id="type-mismatch"),
        pytest.param(
            lambda: {
                **_record(),
                "success": False,
                "failure_reason": None,
                "failure_stage_guess": "timeout",
            },
            id="bad-null-failure-reason",
        ),
    ],
)
def test_schema_rejects_missing_extra_type_and_bad_null_cases(
    mutated: Any,
) -> None:
    assert _schema_errors(mutated())


def test_derives_peak_lift_from_steps_not_final_snapshot(tmp_path: Path) -> None:
    steps_path = tmp_path / "telemetry" / "positive" / "steps.jsonl"
    _write_jsonl(
        steps_path,
        [
            {"seed": 19999, "indicator_mode": "positive", "apple_height_z": 9.99},
            {"seed": 20000, "indicator_mode": "omit", "apple_height_z": 9.99},
            {"seed": 20000, "indicator_mode": "positive", "apple_height_z": 0.31},
            {"seed": 20000, "indicator_mode": "positive", "apple_height_z": 0.55},
            {"seed": 20000, "indicator_mode": "positive", "apple_height_z": 0.30},
        ],
    )
    runner_episode = {
        "seed": 20000,
        "indicator_mode": "positive",
        "outer_steps": 36,
        "success": False,
        "terminated": False,
        "truncated": True,
        "failure_reason": "outer_step_budget_exhausted",
        "failure_stage_guess": "outer_step_budget",
        "final_snapshot": {"apple_height_z": 0.30},
    }

    record = _build_episode_record(
        cell=P0B,
        seed=20000,
        indicator_mode="positive",
        runner_episode=runner_episode,
        steps_path=steps_path,
    )

    assert math.isclose(record["max_apple_lift_z"], 0.55)
    assert math.isclose(record["final_apple_height_z"], 0.30)
    assert _schema_errors(record) == []


def test_derivation_zero_matching_steps_is_schema_drift(tmp_path: Path) -> None:
    steps_path = tmp_path / "telemetry" / "positive" / "steps.jsonl"
    _write_jsonl(
        steps_path,
        [{"seed": 20001, "indicator_mode": "positive", "apple_height_z": 0.55}],
    )
    runner_episode = {
        "seed": 20000,
        "indicator_mode": "positive",
        "outer_steps": 1,
        "success": False,
        "terminated": False,
        "truncated": True,
        "failure_reason": "outer_step_budget_exhausted",
        "failure_stage_guess": "outer_step_budget",
        "final_snapshot": {"apple_height_z": 0.30},
    }

    with pytest.raises(Exception) as excinfo:
        _build_episode_record(
            cell=P0B,
            seed=20000,
            indicator_mode="positive",
            runner_episode=runner_episode,
            steps_path=steps_path,
        )
    assert "STOP_SCHEMA_DRIFT" in str(excinfo.value) or (
        getattr(excinfo.value, "stop_code", None) == "STOP_SCHEMA_DRIFT"
    )


def test_streaming_watchdog_requires_completed_five_and_two_mtime_ticks() -> None:
    stable_zero_success = {
        "summary": {
            "mode_summaries": {
                "positive": {"episodes_completed": 5, "success_count": 0}
            }
        },
        "summary_mtime_ns": 200,
    }
    samples = [
        {
            "summary": {
                "mode_summaries": {
                    "positive": {"episodes_completed": 4, "success_count": 0}
                }
            },
            "summary_mtime_ns": 100,
        },
        {**copy.deepcopy(stable_zero_success), "summary_mtime_ns": 200},
        copy.deepcopy(stable_zero_success),
        copy.deepcopy(stable_zero_success),
    ]

    assert _streaming_watchdog_stop_code(samples[:1]) is None
    assert _streaming_watchdog_stop_code(samples[:2]) is None
    assert _streaming_watchdog_stop_code(samples[:3]) == "STOP_BASE_INFRA_DRIFT"
    assert _streaming_watchdog_stop_code(samples[:4]) == "STOP_BASE_INFRA_DRIFT"

    one_success = copy.deepcopy(samples)
    for sample in one_success:
        if sample["summary"]["mode_summaries"]["positive"]["episodes_completed"] >= 5:
            sample["summary"]["mode_summaries"]["positive"]["success_count"] = 1
    assert _streaming_watchdog_stop_code(one_success) is None


def test_hung_child_watchdog_triple_and_fires_at_tick_four_not_earlier() -> None:
    stagnant_tick = {
        "peak_vram_mib": 7310,
        "episode_end_count": 2,
        "episodes_jsonl_line_count": 2,
    }
    samples = [copy.deepcopy(stagnant_tick) for _ in range(4)]

    assert _hung_watchdog_stop_code(samples[:1]) is None
    assert _hung_watchdog_stop_code(samples[:2]) is None
    assert _hung_watchdog_stop_code(samples[:3]) is None
    assert _hung_watchdog_stop_code(samples[:4]) == "STOP_HUNG"

    progress_samples = [copy.deepcopy(stagnant_tick) for _ in range(4)]
    progress_samples[-1]["episodes_jsonl_line_count"] = 3
    assert _hung_watchdog_stop_code(progress_samples) is None


def test_phase0_archive_sha_pin_detects_corruption(tmp_path: Path) -> None:
    source = tmp_path / "g3_formal_server.log"
    archive = tmp_path / "g3_formal_server.phase0_archived.log"
    sha_path = tmp_path / "g3_formal_server.phase0_archived.sha256"
    source.write_text("phase0 server log\n", encoding="utf-8")

    _archive_phase0_log(source, archive, sha_path)

    assert archive.read_text(encoding="utf-8") == "phase0 server log\n"
    expected_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert expected_hash in sha_path.read_text(encoding="utf-8")
    assert (
        _stop_code(
            _verify_phase0_log_archive(archive, sha_path, source)
        )
        is None
    )

    archive.write_text("phase0 server log\ncorrupt\n", encoding="utf-8")
    assert _verify_phase0_log_archive(archive, sha_path, source) == "STOP_PHASE0_LOG_DRIFT"
