"""Side-effect-free JSONL/NPZ seam trace writer for Stage B diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time
from typing import Any, Sequence

from .action_uuid import make_action_content_hash
from .array_summary import copy_array_for_trace, summarize_array
from .schema import TRACE_VERSION, write_schema

_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.=-]+")


def _safe_token(value: str | int) -> str:
    return _SAFE_TOKEN_RE.sub("_", str(value)).strip("_") or "unknown"


def make_array_key(
    *,
    episode_id: str | int,
    step_id: str | int,
    stage: str,
    name: str,
) -> str:
    """Return the NPZ key required to recover an event's raw array."""

    return "__".join(
        [
            f"episode_{_safe_token(episode_id)}",
            f"step_{_safe_token(step_id)}",
            f"stage_{_safe_token(stage)}",
            f"name_{_safe_token(name)}",
        ]
    )


@dataclass
class SeamTraceWriter:
    """Append-only Stage B trace writer.

    Events are buffered in memory and flushed after caller-selected safe points.
    Raw arrays are copied into the buffer before summary/hash computation so the
    writer does not mutate policy, controller, or env payloads.
    """

    output_dir: str | Path
    trace_version: str = TRACE_VERSION
    enabled: bool = True
    strict: bool = False
    compress_npz: bool = True
    jsonl_name: str = "seam_trace.jsonl"
    _pending_events: list[dict[str, Any]] = field(default_factory=list, init=False)
    _pending_arrays: dict[str, Any] = field(default_factory=dict, init=False)
    _flush_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            write_schema(self.output_dir / "seam_trace_schema_v1.json")

    @property
    def jsonl_path(self) -> Path:
        return Path(self.output_dir) / self.jsonl_name

    def record_array_event(
        self,
        *,
        stage: str,
        name: str,
        episode_id: str | int,
        step_id: str | int,
        chain_action_uuid: str,
        array: Any | None = None,
        contrast_group_uuid: str | None = None,
        seed: int | str | None = None,
        indicator_mode: str | None = None,
        obs_hash: str | None = None,
        prompt_text_hash: str | None = None,
        sim_time: float | None = None,
        wall_time_ns: int | None = None,
        diagnostics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        missing_stage_reason: str | None = None,
    ) -> dict[str, Any] | None:
        """Buffer one event and optional raw array for later append-only flush."""

        if not self.enabled:
            return None
        try:
            event = self._build_event(
                stage=stage,
                name=name,
                episode_id=episode_id,
                step_id=step_id,
                chain_action_uuid=chain_action_uuid,
                array=array,
                contrast_group_uuid=contrast_group_uuid,
                seed=seed,
                indicator_mode=indicator_mode,
                obs_hash=obs_hash,
                prompt_text_hash=prompt_text_hash,
                sim_time=sim_time,
                wall_time_ns=wall_time_ns,
                diagnostics=diagnostics,
                metadata=metadata,
                missing_stage_reason=missing_stage_reason,
            )
            self._pending_events.append(event)
            return event
        except Exception as exc:  # pragma: no cover - strict branch tested via API shape
            if self.strict:
                raise
            return {
                "trace_version": self.trace_version,
                "trace_error": type(exc).__name__,
                "trace_error_message": str(exc),
            }

    def _build_event(
        self,
        *,
        stage: str,
        name: str,
        episode_id: str | int,
        step_id: str | int,
        chain_action_uuid: str,
        array: Any | None,
        contrast_group_uuid: str | None,
        seed: int | str | None,
        indicator_mode: str | None,
        obs_hash: str | None,
        prompt_text_hash: str | None,
        sim_time: float | None,
        wall_time_ns: int | None,
        diagnostics: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        missing_stage_reason: str | None,
    ) -> dict[str, Any]:
        array_summary = None
        array_ref = None
        action_content_hash = None
        if array is not None:
            array_copy = copy_array_for_trace(array)
            array_key = make_array_key(
                episode_id=episode_id,
                step_id=step_id,
                stage=stage,
                name=name,
            )
            self._pending_arrays[array_key] = array_copy
            npz_name = f"arrays_{self._flush_index:06d}.npz"
            array_ref = {"npz_path": npz_name, "array_key": array_key}
            array_summary = summarize_array(array_copy)
            action_content_hash = make_action_content_hash(array_copy)

        return {
            "trace_version": self.trace_version,
            "episode_id": episode_id,
            "step_id": step_id,
            "stage": stage,
            "name": name,
            "chain_action_uuid": chain_action_uuid,
            "contrast_group_uuid": contrast_group_uuid,
            "action_content_hash": action_content_hash,
            "wall_time_ns": int(wall_time_ns if wall_time_ns is not None else time.time_ns()),
            "sim_time": sim_time,
            "seed": seed,
            "indicator_mode": indicator_mode,
            "obs_hash": obs_hash,
            "prompt_text_hash": prompt_text_hash,
            "array_summary": array_summary,
            "array_ref": array_ref,
            "missing_stage_reason": missing_stage_reason,
            "diagnostics": diagnostics or {},
            "metadata": metadata or {},
        }

    def flush(self) -> dict[str, Any]:
        """Write pending events/arrays and clear the in-memory buffer."""

        if not self.enabled:
            return {"enabled": False, "events_written": 0, "arrays_written": 0}
        if not self._pending_events and not self._pending_arrays:
            return {"enabled": True, "events_written": 0, "arrays_written": 0}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = None
        arrays_written = len(self._pending_arrays)
        if self._pending_arrays:
            npz_path = self.output_dir / f"arrays_{self._flush_index:06d}.npz"
            np = __import__("numpy")
            if self.compress_npz:
                np.savez_compressed(npz_path, **self._pending_arrays)
            else:
                np.savez(npz_path, **self._pending_arrays)

        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            for event in self._pending_events:
                stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

        events_written = len(self._pending_events)
        self._pending_events.clear()
        self._pending_arrays.clear()
        self._flush_index += 1
        return {
            "enabled": True,
            "events_written": events_written,
            "arrays_written": arrays_written,
            "jsonl_path": str(self.jsonl_path),
            "npz_path": str(npz_path) if npz_path is not None else None,
        }


def run_self_test(output_dir: str | Path) -> dict[str, Any]:
    """Exercise writer, IDs, schema, missing-stage events, and no-mutation guard."""

    import numpy as np

    from .action_uuid import make_chain_action_uuid, make_contrast_group_uuid

    output_path = Path(output_dir)
    writer = SeamTraceWriter(output_path)
    array = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]], dtype=np.float32)
    before = array.copy()
    chain_uuid = make_chain_action_uuid(
        trace_version=TRACE_VERSION,
        episode_id="self_test_ep",
        step_id=0,
        seed=20000,
        policy_call_index=0,
        obs_hash="obs_self_test_hash",
    )
    contrast_uuid = make_contrast_group_uuid(
        trace_version=TRACE_VERSION,
        seed=20000,
        obs_hash="obs_self_test_hash",
        frozen_controller_state_hash="controller_reset_hash",
        probe_name="writer_self_test",
    )
    writer.record_array_event(
        stage="policy",
        name="decoded_action",
        episode_id="self_test_ep",
        step_id=0,
        chain_action_uuid=chain_uuid,
        contrast_group_uuid=contrast_uuid,
        seed=20000,
        indicator_mode="omit",
        obs_hash="obs_self_test_hash",
        array=array,
        diagnostics={"nan_count_expected": 1},
    )
    writer.record_array_event(
        stage="controller",
        name="controller_output",
        episode_id="self_test_ep",
        step_id=0,
        chain_action_uuid=chain_uuid,
        contrast_group_uuid=contrast_uuid,
        seed=20000,
        indicator_mode="omit",
        missing_stage_reason="self_test_missing_true_torque_proxy_only",
    )
    flush_result = writer.flush()
    unchanged = bool(np.array_equal(array, before, equal_nan=True))
    report = {
        "self_test": "PASS" if unchanged else "FAIL",
        "input_array_unchanged": unchanged,
        "flush_result": flush_result,
        "schema_path": str(output_path / "seam_trace_schema_v1.json"),
        "jsonl_path": str(writer.jsonl_path),
        "chain_action_uuid": chain_uuid,
        "contrast_group_uuid": contrast_uuid,
    }
    report_path = output_path / "self_test_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run writer self-test.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        report = run_self_test(args.output_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["self_test"] == "PASS" else 1
    parser.error("--self-test is required for this diagnostic CLI")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
