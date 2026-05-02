#!/usr/bin/env python3
"""Stage B P0 Phase 1 downstream aggregation and STOP enforcement.

This module is intentionally downstream of ``gr00t_g3_formal_eval.py``.  It
reads runner telemetry, derives Phase 1 records, validates the public record
contract, and writes new aggregation/STOP artifacts.  It never edits or wraps
the runner internals.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION_EPISODE = "p1_episode_record_v1"
SCHEMA_VERSION_STOP = "p1_stop_record_v1"
SCHEMA_VERSION_RUNTIME_SUMMARY = "p1_phase1_runtime_summary_v1"

CELL_P0A = "P0a_post_recap_nenvs_1"
CELL_P0B = "P0b_base_reference_nenvs_1"
ALLOWED_CELLS = (CELL_P0A, CELL_P0B)
ALLOWED_MODES = ("positive", "omit", "negative")

STOP_POST_UNEXPECTED_RECOVERY = "STOP_POST_UNEXPECTED_RECOVERY"
STOP_BASE_INFRA_DRIFT = "STOP_BASE_INFRA_DRIFT"
STOP_BASE_OOR_LOW = "STOP_BASE_OOR_LOW"
STOP_BASE_OOR_HIGH = "STOP_BASE_OOR_HIGH"
STOP_SERVER_FAIL = "STOP_SERVER_FAIL"
STOP_NAN = "STOP_NAN"
STOP_INF = "STOP_INF"
STOP_MUJOCO_CRASH = "STOP_MUJOCO_CRASH"
STOP_TIMEOUT = "STOP_TIMEOUT"
STOP_VRAM_HEADROOM = "STOP_VRAM_HEADROOM"
STOP_DRYRUN_BLOCKER = "STOP_DRYRUN_BLOCKER"
STOP_HUNG = "STOP_HUNG"
STOP_SCHEMA_DRIFT = "STOP_SCHEMA_DRIFT"
STOP_PHASE0_LOG_DRIFT = "STOP_PHASE0_LOG_DRIFT"

STOP_CODES = (
    STOP_POST_UNEXPECTED_RECOVERY,
    STOP_BASE_INFRA_DRIFT,
    STOP_BASE_OOR_LOW,
    STOP_BASE_OOR_HIGH,
    STOP_SERVER_FAIL,
    STOP_NAN,
    STOP_INF,
    STOP_MUJOCO_CRASH,
    STOP_TIMEOUT,
    STOP_VRAM_HEADROOM,
    STOP_DRYRUN_BLOCKER,
    STOP_HUNG,
    STOP_SCHEMA_DRIFT,
    STOP_PHASE0_LOG_DRIFT,
)

EPISODE_KEYS = (
    "schema_version",
    "cell",
    "seed",
    "indicator_mode",
    "outer_steps",
    "success",
    "terminated",
    "truncated",
    "max_apple_lift_z",
    "final_apple_height_z",
    "failure_reason",
    "failure_stage_guess",
)

FAILURE_REASONS_ALLOW_NULL_STAGE = frozenset({"mujoco_crash", "server_disconnect"})
ALLOWED_FAILURE_REASONS = frozenset(
    {
        "truncated_without_success",
        "terminated_without_success",
        "done_without_success",
        "outer_step_budget_exhausted",
        "episode_incomplete_without_success",
        "mujoco_crash",
        "server_disconnect",
    }
)
FAILURE_REASON_ALIASES = {
    "outer_step_limit": "outer_step_budget_exhausted",
    "outer_step_budget": "outer_step_budget_exhausted",
    "timeout": "outer_step_budget_exhausted",
}
DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parent / "schemas" / "p1_episode_record_v1.schema.json"
)


class AggregatorError(RuntimeError):
    """Base exception for Phase 1 aggregator failures."""


class StopCondition(AggregatorError):
    """Raised when aggregation detects a leader-defined STOP condition."""

    def __init__(self, stop_code: str, message: str, *, evidence_paths: Sequence[Path | str] = ()):
        if stop_code not in STOP_CODES:
            raise ValueError(f"unknown stop code: {stop_code}")
        super().__init__(message)
        self.stop_code = stop_code
        self.evidence_paths = tuple(str(path) for path in evidence_paths)


@dataclasses.dataclass(frozen=True)
class StopDecision:
    stop_code: str
    message: str
    cell: str
    completed_episodes: int
    success_count: int
    evidence_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stop_code not in STOP_CODES:
            raise ValueError(f"unknown stop code: {self.stop_code}")
        if self.cell not in ALLOWED_CELLS:
            raise ValueError(f"unknown Phase 1 cell: {self.cell}")

    def to_stop_record(self, *, triggered_at_utc: str | None = None) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION_STOP,
            "cell": self.cell,
            "triggered_at_utc": triggered_at_utc or utc_now(),
            "stop_code": self.stop_code,
            "completed_episodes": int(self.completed_episodes),
            "success_count": int(self.success_count),
            "evidence_paths": list(self.evidence_paths),
            "no_retry": True,
            "leader_action_required": True,
        }


@dataclasses.dataclass(frozen=True)
class CellAggregation:
    cell: str
    output_dir: str
    records: tuple[dict[str, Any], ...]
    status: str
    success_count: int
    completed_episodes: int
    mode_split: dict[str, dict[str, Any]]
    lift_z_distribution: dict[str, Any]
    failure_stage_distribution: dict[str, int]
    stop_decision: StopDecision | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cell": self.cell,
            "output_dir": self.output_dir,
            "status": self.status,
            "completed_episodes": int(self.completed_episodes),
            "success_count": int(self.success_count),
            "success_rate": (
                float(self.success_count / self.completed_episodes)
                if self.completed_episodes
                else 0.0
            ),
            "mode_split": self.mode_split,
            "lift_z_distribution": self.lift_z_distribution,
            "failure_stage_distribution": self.failure_stage_distribution,
            "episode_records": list(self.records),
        }
        if self.stop_decision is not None:
            payload["stop_decision"] = self.stop_decision.to_stop_record()
        return payload


@dataclasses.dataclass(frozen=True)
class StreamingSample:
    episodes_completed: int
    success_count: int
    mtime_ns: int
    evidence_paths: tuple[str, ...] = ()


class P0bFirstFiveWatchdog:
    """Streaming P0b first-five watchdog with mtime quiescence fencing."""

    def __init__(self, *, quiescent_ticks_required: int = 2, first_seed_count: int = 5):
        if quiescent_ticks_required < 2:
            raise ValueError("quiescent_ticks_required must be >= 2")
        if first_seed_count < 1:
            raise ValueError("first_seed_count must be >= 1")
        self.quiescent_ticks_required = int(quiescent_ticks_required)
        self.first_seed_count = int(first_seed_count)
        self._last_mtime_ns: int | None = None
        self._same_mtime_ticks = 0

    def tick(self, sample: StreamingSample) -> StopDecision | None:
        if self._last_mtime_ns == int(sample.mtime_ns):
            self._same_mtime_ticks += 1
        else:
            self._last_mtime_ns = int(sample.mtime_ns)
            self._same_mtime_ticks = 1

        if int(sample.episodes_completed) < self.first_seed_count:
            return None
        if self._same_mtime_ticks < self.quiescent_ticks_required:
            return None
        if int(sample.success_count) != 0:
            return None
        return StopDecision(
            stop_code=STOP_BASE_INFRA_DRIFT,
            message="P0b completed first five seeds with zero successes after mtime fence.",
            cell=CELL_P0B,
            completed_episodes=int(sample.episodes_completed),
            success_count=int(sample.success_count),
            evidence_paths=tuple(sample.evidence_paths),
        )

    def tick_summary_path(self, summary_path: Path) -> StopDecision | None:
        summary = read_json(summary_path)
        positive = _mode_summary(summary, "positive")
        stat = summary_path.stat()
        return self.tick(
            StreamingSample(
                episodes_completed=as_int(
                    positive.get("episodes_completed", positive.get("episodes", 0)),
                    field_name="mode_summaries.positive.episodes_completed",
                ),
                success_count=as_int(
                    positive.get("success_count", 0),
                    field_name="mode_summaries.positive.success_count",
                ),
                mtime_ns=int(stat.st_mtime_ns),
                evidence_paths=(str(summary_path),),
            )
        )


class HungChildWatchdog:
    """Triple-AND hung-child detector over N consecutive stagnant samples."""

    def __init__(self, *, required_stagnant_samples: int = 4, cell: str = CELL_P0A):
        if required_stagnant_samples < 2:
            raise ValueError("required_stagnant_samples must be >= 2")
        if cell not in ALLOWED_CELLS:
            raise ValueError(f"unknown Phase 1 cell: {cell}")
        self.required_stagnant_samples = int(required_stagnant_samples)
        self.cell = cell
        self._window: list[tuple[int, int, int]] = []

    def tick(
        self,
        *,
        vram_used_mib: int,
        episode_end_count: int,
        episode_jsonl_line_count: int,
        evidence_paths: Sequence[Path | str] = (),
    ) -> StopDecision | None:
        current = (
            int(vram_used_mib),
            int(episode_end_count),
            int(episode_jsonl_line_count),
        )
        if self._window and current[2] < self._window[-1][2]:
            self._window.clear()
        self._window.append(current)
        self._window = self._window[-self.required_stagnant_samples :]
        if len(self._window) < self.required_stagnant_samples:
            return None
        if len(set(self._window)) != 1:
            return None
        return StopDecision(
            stop_code=STOP_HUNG,
            message="VRAM, EPISODE_END heartbeat count, and episodes.jsonl line count are stagnant.",
            cell=self.cell,
            completed_episodes=int(episode_jsonl_line_count),
            success_count=0,
            evidence_paths=tuple(str(path) for path in evidence_paths),
        )


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AggregatorError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StopCondition(STOP_SCHEMA_DRIFT, f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise StopCondition(STOP_SCHEMA_DRIFT, f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    tmp.replace(path)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise StopCondition(
                        STOP_SCHEMA_DRIFT,
                        f"invalid JSONL record at {path}:{line_number}",
                        evidence_paths=(path,),
                    ) from exc
                if not isinstance(payload, dict):
                    raise StopCondition(
                        STOP_SCHEMA_DRIFT,
                        f"expected JSON object at {path}:{line_number}",
                        evidence_paths=(path,),
                    )
                yield payload
    except FileNotFoundError as exc:
        raise StopCondition(
            STOP_SCHEMA_DRIFT, f"missing JSONL telemetry: {path}", evidence_paths=(path,)
        ) from exc


def as_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StopCondition(STOP_SCHEMA_DRIFT, f"{field_name} must be an int")
    return int(value)


def as_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise StopCondition(STOP_SCHEMA_DRIFT, f"{field_name} must be a bool")
    return bool(value)


def optional_finite_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return finite_float(value, field_name=field_name)


def finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StopCondition(STOP_SCHEMA_DRIFT, f"{field_name} must be a finite float")
    number = float(value)
    if math.isnan(number):
        raise StopCondition(STOP_NAN, f"{field_name} is NaN")
    if math.isinf(number):
        raise StopCondition(STOP_INF, f"{field_name} is infinite")
    return number


def launcher_has_episode_end(log_path: Path | None, *, seed: int, indicator_mode: str) -> bool:
    if log_path is None or not log_path.is_file():
        return False
    seed_token = f"seed={int(seed)}"
    mode_token = f"indicator_mode={indicator_mode}"
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        return any(
            "[EPISODE_END]" in line and seed_token in line and mode_token in line
            for line in handle
        )


def count_episode_end_lines(log_path: Path | None) -> int:
    if log_path is None or not log_path.is_file():
        return 0
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if "[EPISODE_END]" in line)


def summary_fence_completed(
    summary_path: Path | None, *, indicator_mode: str, episode_index: int
) -> bool:
    if summary_path is None or not summary_path.is_file():
        return False
    summary = read_json(summary_path)
    mode_summary = _mode_summary(summary, indicator_mode)
    raw_completed = mode_summary.get("episodes_completed", mode_summary.get("episodes", 0))
    return as_int(
        raw_completed, field_name=f"mode_summaries.{indicator_mode}.episodes_completed"
    ) >= int(episode_index)


def read_after_write_fence_met(
    *,
    seed: int,
    indicator_mode: str,
    episode_index: int,
    launcher_log_path: Path | None = None,
    summary_path: Path | None = None,
    process_exited: bool = False,
) -> bool:
    return bool(
        process_exited
        or launcher_has_episode_end(launcher_log_path, seed=seed, indicator_mode=indicator_mode)
        or summary_fence_completed(
            summary_path, indicator_mode=indicator_mode, episode_index=episode_index
        )
    )


def derive_max_apple_lift_z(
    steps_path: Path,
    *,
    seed: int,
    indicator_mode: str,
    episode_index: int = 1,
    launcher_log_path: Path | None = None,
    summary_path: Path | None = None,
    process_exited: bool = False,
    require_fence: bool = True,
) -> float:
    if require_fence and not read_after_write_fence_met(
        seed=seed,
        indicator_mode=indicator_mode,
        episode_index=episode_index,
        launcher_log_path=launcher_log_path,
        summary_path=summary_path,
        process_exited=process_exited,
    ):
        raise AggregatorError(
            f"read-after-write fence not met for seed={seed} mode={indicator_mode}"
        )

    heights: list[float] = []
    for record in iter_jsonl(steps_path):
        if int(record.get("seed", -1)) != int(seed):
            continue
        if str(record.get("indicator_mode")) != str(indicator_mode):
            continue
        if int(record.get("episode_index", episode_index)) != int(episode_index):
            continue
        if "apple_height_z" not in record:
            raise StopCondition(
                STOP_SCHEMA_DRIFT,
                "steps.jsonl record is missing apple_height_z",
                evidence_paths=(steps_path,),
            )
        heights.append(finite_float(record["apple_height_z"], field_name="apple_height_z"))

    if not heights:
        raise StopCondition(
            STOP_SCHEMA_DRIFT,
            f"zero matching step records for seed={seed} mode={indicator_mode}",
            evidence_paths=(steps_path,),
        )
    return max(heights)


def build_episode_record(
    *,
    cell: str,
    runner_episode_record: Mapping[str, Any],
    steps_path: Path,
    launcher_log_path: Path | None = None,
    summary_path: Path | None = None,
    process_exited: bool = False,
    schema_path: Path | None = None,
) -> dict[str, Any]:
    seed = as_int(runner_episode_record.get("seed"), field_name="episode.seed")
    mode = str(runner_episode_record.get("indicator_mode", ""))
    episode_index = as_int(
        runner_episode_record.get("episode_index", 1), field_name="episode.episode_index"
    )
    final_snapshot = runner_episode_record.get("final_snapshot", {})
    if not isinstance(final_snapshot, Mapping):
        raise StopCondition(STOP_SCHEMA_DRIFT, "episode.final_snapshot must be an object")

    success = as_bool(runner_episode_record.get("success"), field_name="episode.success")
    failure_reason = normalize_failure_reason(runner_episode_record.get("failure_reason"))
    fsg_raw = runner_episode_record.get("failure_stage_guess")
    if isinstance(fsg_raw, Mapping):
        label = fsg_raw.get("label")
        failure_stage_guess = label if isinstance(label, str) and label.strip() else None
    else:
        failure_stage_guess = fsg_raw
    if success:
        failure_reason = None
        failure_stage_guess = None

    record = {
        "schema_version": SCHEMA_VERSION_EPISODE,
        "cell": str(cell),
        "seed": seed,
        "indicator_mode": mode,
        "outer_steps": as_int(
            runner_episode_record.get("outer_steps"), field_name="episode.outer_steps"
        ),
        "success": success,
        "terminated": as_bool(
            runner_episode_record.get("terminated"), field_name="episode.terminated"
        ),
        "truncated": as_bool(
            runner_episode_record.get("truncated"), field_name="episode.truncated"
        ),
        "max_apple_lift_z": derive_max_apple_lift_z(
            steps_path,
            seed=seed,
            indicator_mode=mode,
            episode_index=episode_index,
            launcher_log_path=launcher_log_path,
            summary_path=summary_path,
            process_exited=process_exited,
        ),
        "final_apple_height_z": optional_finite_float(
            final_snapshot.get("apple_height_z"), field_name="final_snapshot.apple_height_z"
        ),
        "failure_reason": failure_reason,
        "failure_stage_guess": failure_stage_guess,
    }
    validate_episode_record(record, schema_path=schema_path)
    return record


def validate_episode_record(record: Mapping[str, Any], *, schema_path: Path | None = None) -> None:
    actual_keys = tuple(record.keys())
    if actual_keys != EPISODE_KEYS:
        missing = [key for key in EPISODE_KEYS if key not in record]
        extra = [key for key in actual_keys if key not in EPISODE_KEYS]
        raise StopCondition(
            STOP_SCHEMA_DRIFT,
            f"episode record keys drifted: missing={missing} extra={extra}",
        )
    if schema_path is not None and schema_path.exists():
        _assert_schema_is_strict(schema_path)

    if record["schema_version"] != SCHEMA_VERSION_EPISODE:
        raise StopCondition(STOP_SCHEMA_DRIFT, "episode schema_version drifted")
    if record["cell"] not in ALLOWED_CELLS:
        raise StopCondition(STOP_SCHEMA_DRIFT, f"unknown cell: {record['cell']}")
    if record["indicator_mode"] not in ALLOWED_MODES:
        raise StopCondition(
            STOP_SCHEMA_DRIFT, f"unknown indicator_mode: {record['indicator_mode']}"
        )
    as_int(record["seed"], field_name="seed")
    as_int(record["outer_steps"], field_name="outer_steps")
    success = as_bool(record["success"], field_name="success")
    as_bool(record["terminated"], field_name="terminated")
    as_bool(record["truncated"], field_name="truncated")
    finite_float(record["max_apple_lift_z"], field_name="max_apple_lift_z")
    optional_finite_float(record["final_apple_height_z"], field_name="final_apple_height_z")

    failure_reason = record["failure_reason"]
    failure_stage_guess = record["failure_stage_guess"]
    if success:
        if failure_reason is not None or failure_stage_guess is not None:
            raise StopCondition(
                STOP_SCHEMA_DRIFT,
                "successful episode must not carry failure_reason/failure_stage_guess",
            )
        return
    if not isinstance(failure_reason, str) or not failure_reason.strip():
        raise StopCondition(STOP_SCHEMA_DRIFT, "failed episode must carry failure_reason")
    if failure_reason not in ALLOWED_FAILURE_REASONS:
        raise StopCondition(
            STOP_SCHEMA_DRIFT,
            f"failed episode carries unknown failure_reason: {failure_reason}",
        )
    if failure_stage_guess is None and failure_reason in FAILURE_REASONS_ALLOW_NULL_STAGE:
        return
    if not isinstance(failure_stage_guess, str) or not failure_stage_guess.strip():
        raise StopCondition(STOP_SCHEMA_DRIFT, "failed episode must carry failure_stage_guess")


def _assert_schema_is_strict(schema_path: Path) -> None:
    schema = read_json(schema_path)
    if schema.get("additionalProperties") is not False:
        raise StopCondition(
            STOP_SCHEMA_DRIFT,
            f"schema must reject extra keys: {schema_path}",
            evidence_paths=(schema_path,),
        )
    required = schema.get("required")
    if sorted(required or []) != sorted(EPISODE_KEYS):
        raise StopCondition(
            STOP_SCHEMA_DRIFT,
            f"schema required keys drifted: {schema_path}",
            evidence_paths=(schema_path,),
        )


def aggregate_cell(
    *,
    cell: str,
    output_dir: Path,
    summary_path: Path | None = None,
    launcher_log_path: Path | None = None,
    schema_path: Path | None = None,
    process_exited: bool = True,
) -> CellAggregation:
    if cell not in ALLOWED_CELLS:
        raise ValueError(f"unknown Phase 1 cell: {cell}")
    output_dir = output_dir.resolve()
    summary_path = summary_path or output_dir / "formal_eval_summary.json"
    summary = read_json(summary_path)
    records: list[dict[str, Any]] = []
    mode_split: dict[str, dict[str, Any]] = {}

    for mode, mode_summary in sorted(_mode_summaries(summary).items()):
        if mode not in ALLOWED_MODES:
            continue
        if not isinstance(mode_summary, Mapping):
            raise StopCondition(STOP_SCHEMA_DRIFT, f"mode summary is not an object: {mode}")
        steps_path = telemetry_path_from_mode_summary(
            output_dir, mode, mode_summary, kind="step"
        )
        runner_records = _runner_episode_records(output_dir, mode, mode_summary)
        mode_records: list[dict[str, Any]] = []
        for runner_record in runner_records:
            record = build_episode_record(
                cell=cell,
                runner_episode_record=runner_record,
                steps_path=steps_path,
                launcher_log_path=launcher_log_path,
                summary_path=summary_path,
                process_exited=process_exited,
                schema_path=schema_path,
            )
            records.append(record)
            mode_records.append(record)
        mode_success_count = sum(1 for record in mode_records if record["success"])
        mode_split[mode] = {
            "episodes": len(mode_records),
            "success_count": int(mode_success_count),
            "success_rate": float(mode_success_count / len(mode_records))
            if mode_records
            else 0.0,
        }

    completed = len(records)
    success_count = sum(1 for record in records if record["success"])
    stop_decision = evaluate_stop_table(
        cell=cell,
        completed_episodes=completed,
        success_count=success_count,
        status=str(summary.get("status", "UNKNOWN")),
        evidence_paths=(summary_path,),
    )
    return CellAggregation(
        cell=cell,
        output_dir=str(output_dir),
        records=tuple(records),
        status="STOP" if stop_decision is not None else str(summary.get("status", "UNKNOWN")),
        success_count=int(success_count),
        completed_episodes=int(completed),
        mode_split=mode_split,
        lift_z_distribution=lift_z_distribution(records),
        failure_stage_distribution=failure_stage_distribution(records),
        stop_decision=stop_decision,
    )


def telemetry_path_from_mode_summary(
    output_dir: Path, mode: str, mode_summary: Mapping[str, Any], *, kind: str
) -> Path:
    key = "step_telemetry_jsonl" if kind == "step" else "episode_telemetry_jsonl"
    raw = mode_summary.get(key)
    if isinstance(raw, str) and raw.strip():
        candidate = Path(raw)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if candidate.exists():
            return candidate.resolve()
        repo_candidate = Path.cwd() / candidate
        if repo_candidate.exists():
            return repo_candidate.resolve()
        output_candidate = output_dir / candidate
        if output_candidate.exists():
            return output_candidate.resolve()
    filename = "steps.jsonl" if kind == "step" else "episodes.jsonl"
    return output_dir / "telemetry" / mode / filename


def evaluate_stop_table(
    *,
    cell: str,
    completed_episodes: int,
    success_count: int,
    status: str = "PASS",
    server_failed: bool = False,
    mujoco_crash: bool = False,
    nan_count: int = 0,
    inf_count: int = 0,
    wall_clock_s: float | None = None,
    timeout_s: float = 21600.0,
    peak_vram_mib: int | None = None,
    total_vram_mib: int | None = None,
    gpu_headroom_floor_mib: int = 4096,
    dryrun_blocker: bool = False,
    schema_drift: bool = False,
    phase0_log_drift: bool = False,
    evidence_paths: Sequence[Path | str] = (),
) -> StopDecision | None:
    if cell not in ALLOWED_CELLS:
        raise ValueError(f"unknown Phase 1 cell: {cell}")
    completed = int(completed_episodes)
    successes = int(success_count)
    evidence = tuple(str(path) for path in evidence_paths)

    ordered_checks = (
        (phase0_log_drift, STOP_PHASE0_LOG_DRIFT, "Phase 0 archived log SHA drifted."),
        (schema_drift, STOP_SCHEMA_DRIFT, "Phase 1 record/schema contract drifted."),
        (dryrun_blocker, STOP_DRYRUN_BLOCKER, "Phase 1 dry-run gate failed."),
        (server_failed or str(status).upper() == "SERVER_FAIL", STOP_SERVER_FAIL, "Server failed."),
        (mujoco_crash, STOP_MUJOCO_CRASH, "MuJoCo crash detected."),
        (int(nan_count) > 0, STOP_NAN, "NaN detected in runtime output."),
        (int(inf_count) > 0, STOP_INF, "Inf detected in runtime output."),
        (
            wall_clock_s is not None and float(wall_clock_s) > float(timeout_s),
            STOP_TIMEOUT,
            "Cell exceeded wall-clock timeout.",
        ),
        (
            peak_vram_mib is not None
            and total_vram_mib is not None
            and int(peak_vram_mib) > int(total_vram_mib) - int(gpu_headroom_floor_mib),
            STOP_VRAM_HEADROOM,
            "GPU VRAM headroom floor was breached.",
        ),
    )
    for triggered, stop_code, message in ordered_checks:
        if triggered:
            return StopDecision(stop_code, message, cell, completed, successes, evidence)

    if cell == CELL_P0A and successes >= 9:
        return StopDecision(
            STOP_POST_UNEXPECTED_RECOVERY,
            "P0a post-RECAP success_count met unexpected recovery threshold.",
            cell,
            completed,
            successes,
            evidence,
        )
    if cell == CELL_P0B and completed >= 30 and successes < 12:
        return StopDecision(
            STOP_BASE_OOR_LOW,
            "P0b base success_count is below NVIDIA literature band count floor.",
            cell,
            completed,
            successes,
            evidence,
        )
    if cell == CELL_P0B and completed >= 30 and successes > 22:
        return StopDecision(
            STOP_BASE_OOR_HIGH,
            "P0b base success_count is above NVIDIA literature band count ceiling.",
            cell,
            completed,
            successes,
            evidence,
        )
    return None


def classify_stop(snapshot: Mapping[str, Any]) -> StopDecision | None:
    """Classify a synthetic or live STOP snapshot.

    This compatibility surface is intentionally small so tests and the launcher
    can exercise the same STOP table without constructing internal objects.
    """

    cell = str(snapshot.get("cell", CELL_P0A))
    completed = int(snapshot.get("completed_episodes", 0))
    successes = int(snapshot.get("success_count", 0))
    if str(snapshot.get("check_scope", "")) == "p0b_first5_watchdog":
        if (
            cell == CELL_P0B
            and completed >= 5
            and successes == 0
            and int(snapshot.get("mtime_quiescent_ticks", 0)) >= 2
        ):
            return StopDecision(
                STOP_BASE_INFRA_DRIFT,
                "P0b completed first five seeds with zero successes after mtime fence.",
                cell,
                completed,
                successes,
                _snapshot_evidence(snapshot),
            )
        return None

    return evaluate_stop_table(
        cell=cell,
        completed_episodes=completed,
        success_count=successes,
        status=str(snapshot.get("status", "PASS")),
        server_failed=bool(snapshot.get("server_failed", False)),
        mujoco_crash=bool(snapshot.get("mujoco_crash", False)),
        nan_count=int(snapshot.get("nan_count", 0)),
        inf_count=int(snapshot.get("inf_count", 0)),
        wall_clock_s=(
            float(snapshot["wall_clock_s"])
            if "wall_clock_s" in snapshot
            else (float("inf") if snapshot.get("timeout") else None)
        ),
        timeout_s=float(snapshot.get("timeout_s", 21600.0)),
        peak_vram_mib=(
            int(snapshot["gpu_memory_used_mib"])
            if "gpu_memory_used_mib" in snapshot
            else (
                int(snapshot["peak_vram_mib"]) if "peak_vram_mib" in snapshot else None
            )
        ),
        total_vram_mib=(
            int(snapshot["gpu_memory_total_mib"])
            if "gpu_memory_total_mib" in snapshot
            else (
                int(snapshot["total_vram_mib"]) if "total_vram_mib" in snapshot else None
            )
        ),
        gpu_headroom_floor_mib=int(snapshot.get("gpu_headroom_floor_mib", 4096)),
        dryrun_blocker=bool(snapshot.get("dryrun_failed", snapshot.get("dryrun_blocker", False))),
        schema_drift=bool(snapshot.get("schema_drift", False)),
        phase0_log_drift=bool(snapshot.get("phase0_log_drift", False)),
        evidence_paths=_snapshot_evidence(snapshot),
    )


def classify_stop_condition(snapshot: Mapping[str, Any]) -> StopDecision | None:
    return classify_stop(snapshot)


def evaluate_stop_condition(snapshot: Mapping[str, Any]) -> StopDecision | None:
    return classify_stop(snapshot)


def derive_episode_record(
    *,
    cell: str,
    seed: int | None = None,
    indicator_mode: str | None = None,
    runner_episode: Mapping[str, Any] | None = None,
    runner_episode_record: Mapping[str, Any] | None = None,
    steps_path: Path,
    launcher_log_path: Path | None = None,
    summary_path: Path | None = None,
    process_exited: bool = True,
    schema_path: Path | None = None,
) -> dict[str, Any]:
    payload = dict(runner_episode if runner_episode is not None else runner_episode_record or {})
    if seed is not None:
        payload["seed"] = int(seed)
    if indicator_mode is not None:
        payload["indicator_mode"] = str(indicator_mode)
    payload.setdefault("episode_index", 1)
    return build_episode_record(
        cell=cell,
        runner_episode_record=payload,
        steps_path=steps_path,
        launcher_log_path=launcher_log_path,
        summary_path=summary_path,
        process_exited=process_exited,
        schema_path=schema_path,
    )


def evaluate_streaming_watchdog(samples: Sequence[Mapping[str, Any]]) -> StopDecision | None:
    watchdog = P0bFirstFiveWatchdog()
    decision: StopDecision | None = None
    for sample in samples:
        summary = _as_mapping(sample.get("summary", {}), field_name="sample.summary")
        positive = _mode_summary(summary, "positive")
        decision = watchdog.tick(
            StreamingSample(
                episodes_completed=int(positive.get("episodes_completed", 0)),
                success_count=int(positive.get("success_count", 0)),
                mtime_ns=int(sample.get("summary_mtime_ns", 0)),
                evidence_paths=tuple(str(p) for p in sample.get("evidence_paths", ())),
            )
        )
    return decision


def evaluate_p0b_first5_watchdog(samples: Sequence[Mapping[str, Any]]) -> StopDecision | None:
    return evaluate_streaming_watchdog(samples)


def streaming_watchdog_stop_code(samples: Sequence[Mapping[str, Any]]) -> StopDecision | None:
    return evaluate_streaming_watchdog(samples)


def evaluate_hung_child_watchdog(samples: Sequence[Mapping[str, Any]]) -> StopDecision | None:
    watchdog = HungChildWatchdog(cell=str(samples[-1].get("cell", CELL_P0A)) if samples else CELL_P0A)
    decision: StopDecision | None = None
    for sample in samples:
        decision = watchdog.tick(
            vram_used_mib=int(sample.get("peak_vram_mib", sample.get("vram_used_mib", 0))),
            episode_end_count=int(sample.get("episode_end_count", 0)),
            episode_jsonl_line_count=int(
                sample.get("episodes_jsonl_line_count", sample.get("episode_jsonl_line_count", 0))
            ),
            evidence_paths=tuple(str(p) for p in sample.get("evidence_paths", ())),
        )
    return decision


def hung_child_watchdog_stop_code(samples: Sequence[Mapping[str, Any]]) -> StopDecision | None:
    return evaluate_hung_child_watchdog(samples)


def evaluate_hung_watchdog(samples: Sequence[Mapping[str, Any]]) -> StopDecision | None:
    return evaluate_hung_child_watchdog(samples)


def archive_phase0_log_with_sha_pin(
    *, source_log: Path, archive_log: Path, sha256_path: Path
) -> dict[str, Any]:
    if not source_log.is_file() and not archive_log.is_file():
        raise StopCondition(
            STOP_PHASE0_LOG_DRIFT,
            f"Phase 0 source log missing before archive: {source_log}",
            evidence_paths=(source_log,),
        )
    archive_log.parent.mkdir(parents=True, exist_ok=True)
    sha256_path.parent.mkdir(parents=True, exist_ok=True)

    if not archive_log.exists():
        shutil.copy2(source_log, archive_log)

    current_sha = sha256_file(archive_log)
    if sha256_path.exists():
        pinned_sha = sha256_path.read_text(encoding="utf-8").split()[0].strip()
        if pinned_sha != current_sha:
            raise StopCondition(
                STOP_PHASE0_LOG_DRIFT,
                "Phase 0 archived log SHA does not match existing pin.",
                evidence_paths=(archive_log, sha256_path),
            )
    else:
        sha256_path.write_text(f"{current_sha}  {archive_log.name}\n", encoding="utf-8")

    return {
        "source_log": str(source_log),
        "archive_log": str(archive_log),
        "sha256_path": str(sha256_path),
        "sha256": current_sha,
    }


def pin_phase0_log_archive(
    *, source_path: Path, archive_path: Path, sha_path: Path
) -> dict[str, Any]:
    return archive_phase0_log_with_sha_pin(
        source_log=source_path, archive_log=archive_path, sha256_path=sha_path
    )


def archive_phase0_log_with_sha(
    *, source_path: Path, archive_path: Path, sha_path: Path
) -> dict[str, Any]:
    return pin_phase0_log_archive(
        source_path=source_path, archive_path=archive_path, sha_path=sha_path
    )


def verify_phase0_log_archive(
    *, archive_path: Path, sha_path: Path
) -> StopDecision | None:
    if not archive_path.is_file() or not sha_path.is_file():
        return StopDecision(
            STOP_PHASE0_LOG_DRIFT,
            "Phase 0 archive or SHA pin is missing.",
            CELL_P0A,
            0,
            0,
            (str(archive_path), str(sha_path)),
        )
    pinned_sha = sha_path.read_text(encoding="utf-8").split()[0].strip()
    current_sha = sha256_file(archive_path)
    if pinned_sha == current_sha:
        return None
    return StopDecision(
        STOP_PHASE0_LOG_DRIFT,
        "Phase 0 archive SHA does not match pin.",
        CELL_P0A,
        0,
        0,
        (str(archive_path), str(sha_path)),
    )


def detect_phase0_log_drift(*, archive_path: Path, sha_path: Path) -> StopDecision | None:
    return verify_phase0_log_archive(archive_path=archive_path, sha_path=sha_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_stop_record(path: Path, decision: StopDecision) -> dict[str, Any]:
    record = decision.to_stop_record()
    write_json(path, record)
    return record


def build_runtime_summary(
    *,
    cells: Sequence[CellAggregation],
    phase0_archived_log: str | None = None,
    incident_record: str | None = None,
    gate_state_before: str | None = None,
    gate_state_after: str | None = None,
    stop_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION_RUNTIME_SUMMARY,
        "created_at_utc": utc_now(),
        "cells": {cell.cell: cell.to_json() for cell in cells},
        "stop_records": list(stop_records),
        "phase0_archived_log": phase0_archived_log,
        "incident_record": incident_record,
        "gate_state_before": gate_state_before,
        "gate_state_after": gate_state_after,
    }


def lift_z_distribution(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [finite_float(record["max_apple_lift_z"], field_name="max_apple_lift_z") for record in records]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "values": []}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "values": values,
    }


def failure_stage_distribution(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = "success" if record.get("success") is True else str(record.get("failure_stage_guess"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def normalize_failure_reason(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return FAILURE_REASON_ALIASES.get(text, text)


def _snapshot_evidence(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    raw = snapshot.get("evidence_paths", ())
    if isinstance(raw, (str, Path)):
        return (str(raw),)
    return tuple(str(path) for path in raw)


def _mode_summaries(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = summary.get("mode_summaries")
    if not isinstance(raw, Mapping):
        raise StopCondition(STOP_SCHEMA_DRIFT, "summary.mode_summaries must be an object")
    return raw


def _mode_summary(summary: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    raw = _mode_summaries(summary).get(mode)
    if not isinstance(raw, Mapping):
        raise StopCondition(STOP_SCHEMA_DRIFT, f"summary.mode_summaries.{mode} missing")
    return raw


def _runner_episode_records(
    output_dir: Path, mode: str, mode_summary: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    raw = mode_summary.get("episode_results")
    if isinstance(raw, list):
        return [_as_mapping(record, field_name=f"{mode}.episode_results") for record in raw]
    episode_path = telemetry_path_from_mode_summary(output_dir, mode, mode_summary, kind="episode")
    return list(iter_jsonl(episode_path))


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StopCondition(STOP_SCHEMA_DRIFT, f"{field_name} must contain JSON objects")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="p1_phase1_aggregator.py",
        description="Aggregate Phase 1 GR00T formal-eval telemetry into strict P1 records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-cell")
    validate.add_argument("--cell", required=True, choices=ALLOWED_CELLS)
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--summary-path", default="")
    validate.add_argument("--launcher-log", default="")
    validate.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    validate.add_argument("--records-jsonl", default="")

    pin = sub.add_parser("sha-pin")
    pin.add_argument("--source-log", required=True)
    pin.add_argument("--archive-log", required=True)
    pin.add_argument("--sha256-path", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "validate-cell":
            output_dir = Path(args.output_dir)
            schema_path = Path(args.schema_path) if str(args.schema_path).strip() else None
            summary_path = Path(args.summary_path) if str(args.summary_path).strip() else None
            launcher_log = Path(args.launcher_log) if str(args.launcher_log).strip() else None
            aggregation = aggregate_cell(
                cell=args.cell,
                output_dir=output_dir,
                summary_path=summary_path,
                launcher_log_path=launcher_log,
                schema_path=schema_path,
                process_exited=True,
            )
            if str(args.records_jsonl).strip():
                write_jsonl(Path(args.records_jsonl), aggregation.records)
            print(json.dumps(aggregation.to_json(), indent=2, sort_keys=True))
            return 0 if aggregation.stop_decision is None else 2
        if args.command == "sha-pin":
            payload = archive_phase0_log_with_sha_pin(
                source_log=Path(args.source_log),
                archive_log=Path(args.archive_log),
                sha256_path=Path(args.sha256_path),
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
    except StopCondition as exc:
        print(
            json.dumps(
                {
                    "status": "STOP",
                    "stop_code": exc.stop_code,
                    "message": str(exc),
                    "evidence_paths": list(exc.evidence_paths),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            json.dumps({"status": "ERROR", "message": f"{type(exc).__name__}: {exc}"}),
            file=sys.stderr,
        )
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
