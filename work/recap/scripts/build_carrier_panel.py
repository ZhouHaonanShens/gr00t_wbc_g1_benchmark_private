from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


DEFAULT_OUTPUT_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/carrier_panel_manifest.json"
)
REPORT_SCHEMA_VERSION = "carrier_panel_manifest_v1"
REPORT_ARTIFACT_KIND = "carrier_panel_manifest"
WRITER_SCRIPT = "work/recap/scripts/build_carrier_panel.py"
TRACE_FIXTURE_JSON_NAME = "triplet_runtime_traces.json"
EPISODES_JSONL_NAME = "episodes.jsonl"
STEPS_JSONL_NAME = "steps.jsonl"
SELECTION_ALGORITHM = "deterministic_carrier_panel_v1"
PANEL_POINT_SPECS: tuple[dict[str, object], ...] = (
    {
        "panel_slot_name": "success_t10",
        "episode_outcome": "success",
        "timestep_fraction": 0.10,
    },
    {
        "panel_slot_name": "success_t80",
        "episode_outcome": "success",
        "timestep_fraction": 0.80,
    },
    {
        "panel_slot_name": "failure_t10",
        "episode_outcome": "failure",
        "timestep_fraction": 0.10,
    },
    {
        "panel_slot_name": "failure_t50",
        "episode_outcome": "failure",
        "timestep_fraction": 0.50,
    },
    {
        "panel_slot_name": "failure_t90",
        "episode_outcome": "failure",
        "timestep_fraction": 0.90,
    },
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_carrier_panel.py",
        description=(
            "Build the deterministic five-point runtime carrier panel from episode/step telemetry and per-sample triplet runtime traces."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing episodes.jsonl, steps.jsonl, and triplet_runtime_traces.json."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output carrier panel manifest JSON path.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(f"out must be a file path, got directory: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object in {path}, got {type(payload).__name__}")
    return dict(payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"expected JSON object in {path}:{line_number}, got {type(payload).__name__}"
            )
        records.append(dict(payload))
    return records


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: object) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _resolve_dataset_dir(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"dataset-dir does not exist: {resolved}")
    return resolved


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a float-compatible number")
    return float(value)


def _panel_sample_key(episode_index: int, outer_step: int) -> str:
    return f"episode_{int(episode_index):04d}::outer_step_{int(outer_step):04d}"


def _select_episode(
    episode_records: Sequence[Mapping[str, object]],
    *,
    success: bool,
) -> dict[str, Any]:
    matches = [
        dict(record)
        for record in episode_records
        if _as_bool(record.get("success"), field_name="episode.success") is success
    ]
    if not matches:
        label = "success" if success else "failure"
        raise ValueError(f"dataset must contain at least one {label} episode")
    matches.sort(
        key=lambda record: _as_int(
            record.get("episode_index"), field_name="episode_index"
        )
    )
    return matches[0]


def _steps_for_episode(
    step_records: Sequence[Mapping[str, object]],
    *,
    episode_index: int,
) -> list[dict[str, Any]]:
    matches = [
        dict(record)
        for record in step_records
        if _as_int(record.get("episode_index"), field_name="step.episode_index")
        == int(episode_index)
    ]
    if not matches:
        raise ValueError(f"no step telemetry found for episode_index={episode_index}")
    matches.sort(
        key=lambda record: _as_int(record.get("outer_step"), field_name="outer_step")
    )
    return matches


def _select_fractional_step(
    step_records: Sequence[Mapping[str, object]],
    *,
    timestep_fraction: float,
) -> dict[str, Any]:
    if not step_records:
        raise ValueError("cannot select a panel point from an empty episode")
    last_index = len(step_records) - 1
    selected_index = int(math.floor(float(last_index) * float(timestep_fraction)))
    selected_index = max(0, min(last_index, selected_index))
    return dict(step_records[selected_index])


def _load_triplet_trace_index(
    dataset_dir: Path,
) -> dict[tuple[int, int], dict[str, Any]]:
    payload = _read_json(dataset_dir / TRACE_FIXTURE_JSON_NAME)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise TypeError(
            f"{TRACE_FIXTURE_JSON_NAME}.samples must be a list, got {type(samples).__name__}"
        )
    index: dict[tuple[int, int], dict[str, Any]] = {}
    for item in samples:
        if not isinstance(item, Mapping):
            raise TypeError(
                f"{TRACE_FIXTURE_JSON_NAME}.samples entries must be objects, got {type(item).__name__}"
            )
        episode_index = _as_int(item.get("episode_index"), field_name="episode_index")
        outer_step = _as_int(item.get("outer_step"), field_name="outer_step")
        triplet_summary = item.get("triplet_summary")
        runtime_trace = item.get("runtime_trace")
        if runtime_trace is None and isinstance(triplet_summary, Mapping):
            runtime_trace = triplet_summary.get("runtime_trace")
        if not isinstance(runtime_trace, Mapping):
            raise TypeError(
                "triplet trace fixture entries must include runtime_trace or triplet_summary.runtime_trace"
            )
        index[(episode_index, outer_step)] = {
            "runtime_trace": dict(runtime_trace),
            "triplet_summary": dict(triplet_summary)
            if isinstance(triplet_summary, Mapping)
            else None,
            "triplet_summary_path": item.get("triplet_summary_path"),
        }
    return index


def _annotation_payload(
    *,
    episode_record: Mapping[str, object],
    step_record: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key in (
        "failure_reason",
        "failure_stage_guess",
        "n_success_steps",
        "step_telemetry_records",
    ):
        if key in episode_record:
            payload[key] = episode_record[key]
    for key in (
        "success_step",
        "episode_success_so_far",
        "intermediate_signals",
    ):
        if key in step_record:
            payload[key] = step_record[key]
    return payload


def build_carrier_panel(dataset_dir: Path) -> dict[str, Any]:
    resolved_dir = _resolve_dataset_dir(dataset_dir)
    episode_records = _read_jsonl(resolved_dir / EPISODES_JSONL_NAME)
    step_records = _read_jsonl(resolved_dir / STEPS_JSONL_NAME)
    trace_index = _load_triplet_trace_index(resolved_dir)

    success_episode = _select_episode(episode_records, success=True)
    failure_episode = _select_episode(episode_records, success=False)
    samples: list[dict[str, object]] = []
    for panel_slot_index, spec in enumerate(PANEL_POINT_SPECS, start=1):
        episode_outcome = str(spec["episode_outcome"])
        timestep_fraction = _as_float(
            spec["timestep_fraction"],
            field_name="panel_point_specs.timestep_fraction",
        )
        episode_record = (
            success_episode if episode_outcome == "success" else failure_episode
        )
        episode_index = _as_int(
            episode_record.get("episode_index"), field_name="episode_index"
        )
        episode_steps = _steps_for_episode(step_records, episode_index=episode_index)
        step_record = _select_fractional_step(
            episode_steps,
            timestep_fraction=timestep_fraction,
        )
        outer_step = _as_int(step_record.get("outer_step"), field_name="outer_step")
        trace_entry = trace_index.get((episode_index, outer_step))
        if trace_entry is None:
            raise ValueError(
                "missing triplet runtime trace fixture for "
                + _panel_sample_key(episode_index, outer_step)
            )
        triplet_summary = trace_entry.get("triplet_summary")
        triplet_summary_payload = (
            cast(Mapping[str, object], triplet_summary)
            if isinstance(triplet_summary, Mapping)
            else None
        )
        samples.append(
            {
                "sample_id": _panel_sample_key(episode_index, outer_step),
                "panel_slot_index": int(panel_slot_index),
                "panel_slot_name": str(spec["panel_slot_name"]),
                "episode_outcome": episode_outcome,
                "timestep_fraction": float(timestep_fraction),
                "selection_rule": f"{episode_outcome} episode {int(timestep_fraction * 100)}% timestep",
                "episode_index": int(episode_index),
                "outer_step": int(outer_step),
                "episode_outer_steps": int(len(episode_steps)),
                "annotation": _annotation_payload(
                    episode_record=episode_record,
                    step_record=step_record,
                ),
                "episode_record": dict(episode_record),
                "step_record": dict(step_record),
                "debug_probe": {
                    "trace_role": str(
                        cast(Mapping[str, object], trace_entry["runtime_trace"]).get(
                            "trace_role", "debug_probe"
                        )
                    ),
                    "main_gate_eligible": bool(
                        cast(Mapping[str, object], trace_entry["runtime_trace"]).get(
                            "main_gate_eligible", False
                        )
                    ),
                    "runtime_trace": dict(
                        cast(Mapping[str, object], trace_entry["runtime_trace"])
                    ),
                    "triplet_summary_path": trace_entry.get("triplet_summary_path"),
                    "triplet_report_signature_sha256": triplet_summary_payload.get(
                        "report_signature_sha256"
                    )
                    if triplet_summary_payload is not None
                    else None,
                },
            }
        )
    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "selection_algorithm": SELECTION_ALGORITHM,
        "dataset_dir": str(resolved_dir),
        "required_dataset_files": [
            EPISODES_JSONL_NAME,
            STEPS_JSONL_NAME,
            TRACE_FIXTURE_JSON_NAME,
        ],
        "panel_sample_target": len(PANEL_POINT_SPECS),
        "panel_sample_count": len(samples),
        "panel_point_specs": [dict(spec) for spec in PANEL_POINT_SPECS],
        "samples": samples,
        "backpointer": {
            "writer_script": WRITER_SCRIPT,
            "trace_fixture_json": str(
                (resolved_dir / TRACE_FIXTURE_JSON_NAME).resolve()
            ),
        },
    }
    payload["report_signature_sha256"] = _sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "report_signature_sha256"
        }
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        panel = build_carrier_panel(Path(args.dataset_dir))
        output_path = _validate_output_path(Path(args.out))
        panel["output_path"] = str(output_path)
        panel["report_signature_sha256"] = _sha256(
            {
                key: value
                for key, value in panel.items()
                if key != "report_signature_sha256"
            }
        )
        _ = _write_json(output_path, panel)
        print(json.dumps(panel, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
