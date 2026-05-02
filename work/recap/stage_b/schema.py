"""Stage B seam trace schema v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRACE_VERSION = "stage_b_seam_trace_v1"
SCHEMA_VERSION = "stage_b_seam_trace_schema_v1"

REQUIRED_EVENT_FIELDS: tuple[str, ...] = (
    "trace_version",
    "episode_id",
    "step_id",
    "stage",
    "name",
    "chain_action_uuid",
    "wall_time_ns",
)


def build_seam_trace_schema() -> dict[str, Any]:
    """Build the JSON schema frozen for Stage B JSONL event metadata."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_VERSION,
        "title": "Stage B seam trace event metadata",
        "type": "object",
        "additionalProperties": True,
        "required": list(REQUIRED_EVENT_FIELDS),
        "properties": {
            "trace_version": {"const": TRACE_VERSION},
            "episode_id": {"type": ["string", "integer"]},
            "step_id": {"type": ["string", "integer"]},
            "stage": {"type": "string"},
            "name": {"type": "string"},
            "chain_action_uuid": {"type": "string", "format": "uuid"},
            "contrast_group_uuid": {"type": ["string", "null"], "format": "uuid"},
            "action_content_hash": {"type": ["string", "null"]},
            "wall_time_ns": {"type": "integer"},
            "sim_time": {"type": ["number", "null"]},
            "seed": {"type": ["integer", "string", "null"]},
            "indicator_mode": {"type": ["string", "null"]},
            "obs_hash": {"type": ["string", "null"]},
            "prompt_text_hash": {"type": ["string", "null"]},
            "array_summary": {"type": ["object", "null"]},
            "array_ref": {"type": ["object", "null"]},
            "missing_stage_reason": {"type": ["string", "null"]},
            "diagnostics": {"type": "object"},
        },
    }


def write_schema(path: str | Path) -> Path:
    """Write schema JSON and return the path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_seam_trace_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
