from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, cast


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    sort_keys: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "\n".join(
        json.dumps(json_ready(dict(row)), ensure_ascii=False, sort_keys=sort_keys)
        for row in rows
    )
    if serialized:
        serialized = f"{serialized}\n"
    _ = path.write_text(serialized, encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload at {path} must be an object")
    mapping_payload = cast(Mapping[object, object], payload)
    return {str(key): value for key, value in mapping_payload.items()}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL row {line_number} at {path} must be an object")
        mapping_payload = cast(Mapping[object, object], payload)
        rows.append({str(key): value for key, value in mapping_payload.items()})
    return rows


def load_rollout_eval_v2_authority_bundle(
    authority_dir: str | Path,
    *,
    summary_name: str = "summary.json",
    eval_manifest_name: str = "eval_manifest.json",
    per_episode_name: str = "per_episode_rollouts.jsonl",
    video_index_name: str = "video_index.json",
    bootstrap_name: str = "bootstrap_ci.json",
    paired_delta_name: str | None = None,
    deviation_notes_name: str = "deviation_notes.md",
) -> dict[str, object]:
    authority_dir_path = Path(authority_dir).resolve()
    summary = read_json(authority_dir_path / summary_name)
    resolved_paired_delta_name = paired_delta_name
    if resolved_paired_delta_name is None:
        baseline_variant = str(summary.get("baseline_variant", "")).strip()
        if not baseline_variant:
            raise ValueError(
                "v2 authority summary missing baseline_variant required to resolve paired delta"
            )
        resolved_paired_delta_name = f"paired_delta_vs_{baseline_variant}.json"
    return {
        "authority_dir": str(authority_dir_path),
        "summary": summary,
        "eval_manifest": read_json(authority_dir_path / eval_manifest_name),
        "per_episode_rollouts": read_jsonl(authority_dir_path / per_episode_name),
        "video_index": read_json(authority_dir_path / video_index_name),
        "bootstrap_ci": read_json(authority_dir_path / bootstrap_name),
        "paired_delta": read_json(authority_dir_path / resolved_paired_delta_name),
        "deviation_notes": (authority_dir_path / deviation_notes_name).read_text(
            encoding="utf-8"
        ),
    }


def load_rollout_eval_v21_authority_bundle(
    authority_dir: str | Path,
    *,
    summary_name: str = "summary.json",
    eval_manifest_name: str = "eval_manifest.json",
    trace_name: str = "per_episode_trace.jsonl",
    metric_ladder_name: str = "metric_ladder_summary.json",
    bootstrap_name: str = "bootstrap_ci.json",
    pairwise_delta_name: str = "pairwise_delta.json",
    deviation_notes_name: str = "deviation_notes.md",
) -> dict[str, object]:
    authority_dir_path = Path(authority_dir).resolve()
    return {
        "authority_dir": str(authority_dir_path),
        "summary": read_json(authority_dir_path / summary_name),
        "eval_manifest": read_json(authority_dir_path / eval_manifest_name),
        "per_episode_trace": read_jsonl(authority_dir_path / trace_name),
        "metric_ladder_summary": read_json(authority_dir_path / metric_ladder_name),
        "bootstrap_ci": read_json(authority_dir_path / bootstrap_name),
        "pairwise_delta": read_json(authority_dir_path / pairwise_delta_name),
        "deviation_notes": (authority_dir_path / deviation_notes_name).read_text(
            encoding="utf-8"
        ),
    }


def load_v2_authority_bundle(
    authority_dir: str | Path,
    *,
    summary_name: str,
    eval_manifest_name: str,
    per_episode_name: str,
    video_index_name: str,
    bootstrap_name: str,
    paired_delta_name: str,
    deviation_notes_name: str,
) -> dict[str, object]:
    return load_rollout_eval_v2_authority_bundle(
        authority_dir,
        summary_name=summary_name,
        eval_manifest_name=eval_manifest_name,
        per_episode_name=per_episode_name,
        video_index_name=video_index_name,
        bootstrap_name=bootstrap_name,
        paired_delta_name=paired_delta_name,
        deviation_notes_name=deviation_notes_name,
    )


def load_v21_authority_bundle(
    authority_dir: str | Path,
    *,
    summary_name: str,
    eval_manifest_name: str,
    trace_name: str,
    metric_ladder_name: str,
    bootstrap_name: str,
    pairwise_delta_name: str,
    deviation_notes_name: str,
) -> dict[str, object]:
    return load_rollout_eval_v21_authority_bundle(
        authority_dir,
        summary_name=summary_name,
        eval_manifest_name=eval_manifest_name,
        trace_name=trace_name,
        metric_ladder_name=metric_ladder_name,
        bootstrap_name=bootstrap_name,
        pairwise_delta_name=pairwise_delta_name,
        deviation_notes_name=deviation_notes_name,
    )
