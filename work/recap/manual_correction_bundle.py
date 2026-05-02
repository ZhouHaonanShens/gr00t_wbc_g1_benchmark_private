from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from work.recap.dataset_reader import read_m1_dataset
from work.recap.formal_branch_resolution import load_stage3_iteration_contract


SPEC_SCHEMA_VERSION = "recap_stage3_manual_correction_spec_v1"
SEGMENT_SCHEMA_VERSION = "recap_stage3_manual_correction_segment_v1"
REPORT_SCHEMA_VERSION = "recap_stage3_manual_correction_bundle_report_v1"
DEFAULT_BUNDLE_ITER_TAG_SUFFIX = "_manual_bundle"

FORBIDDEN_CORRECTION_SPEC_FIELDS = frozenset(
    {
        "corrected_actions",
        "corrected_rewards",
        "corrected_npz_payload",
        "force_success_episode",
        "force_success_step",
        "inline_transitions",
        "inline_episode",
        "inline_npz",
    }
)

REQUIRED_CORRECTION_SPEC_KEYS = (
    "nominal_episode_id",
    "nominal_t_start",
    "nominal_t_end",
    "human_note",
    "corrected_source_dataset_dir",
    "corrected_episode_id",
)

OPTIONAL_CORRECTION_SPEC_KEYS = (
    "correction_id",
    "prompt_raw",
)


def _resolve_path(base_dir: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"expected JSON object in {path}, got {type(payload).__name__}"
        )
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    tmp_path.replace(path)
    return path


def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an int")
    return int(value)


def _dataset_index(
    dataset_dir: Path,
) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]
]:
    dataset = read_m1_dataset(dataset_dir, check_npz_keys=True)
    episodes_raw = dataset.get("episodes")
    transitions_raw = dataset.get("transitions_by_episode")
    if not isinstance(episodes_raw, list):
        raise ValueError(
            f"invalid episodes payload from read_m1_dataset: {dataset_dir}"
        )
    if not isinstance(transitions_raw, dict):
        raise ValueError(
            f"invalid transitions_by_episode payload from read_m1_dataset: {dataset_dir}"
        )
    episodes = [dict(row) for row in episodes_raw]
    episode_by_id = {str(row["episode_id"]): dict(row) for row in episodes}
    transitions_by_episode: dict[str, list[dict[str, Any]]] = {}
    for episode_id, rows in transitions_raw.items():
        if not isinstance(rows, list):
            raise ValueError(
                f"invalid transitions list from read_m1_dataset: {dataset_dir} episode_id={episode_id}"
            )
        transitions_by_episode[str(episode_id)] = [dict(row) for row in rows]
    return episodes, episode_by_id, transitions_by_episode


def _episode_last_t(
    transitions_by_episode: Mapping[str, list[dict[str, Any]]], episode_id: str
) -> int:
    rows = transitions_by_episode.get(episode_id)
    if not rows:
        raise ValueError(f"episode_id={episode_id} has no transitions")
    return int(dict(rows[-1]).get("t", 0))


def _bundle_iter_tag(formal_iter_tag: str) -> str:
    return f"{formal_iter_tag}{DEFAULT_BUNDLE_ITER_TAG_SUFFIX}"


def scaffold_manual_correction_spec(
    repo_root: Path,
    *,
    spec_path: Path,
    manifest_path: Path | None = None,
    nominal_dataset_dir: Path | None = None,
) -> dict[str, Any]:
    contract = load_stage3_iteration_contract(
        Path(repo_root), manifest_path=manifest_path
    )
    nominal_dir = (
        _resolve_path(Path(repo_root), nominal_dataset_dir)
        if nominal_dataset_dir is not None
        else contract.nominal_dataset_dir
    )
    _, episode_by_id, transitions_by_episode = _dataset_index(nominal_dir)
    corrections: list[dict[str, Any]] = []
    for episode_id, episode in sorted(episode_by_id.items()):
        if bool(episode.get("success_episode", False)):
            continue
        corrections.append(
            {
                "correction_id": f"candidate_{len(corrections) + 1:03d}",
                "nominal_episode_id": episode_id,
                "nominal_t_start": 0,
                "nominal_t_end": _episode_last_t(transitions_by_episode, episode_id),
                "prompt_raw": episode.get("prompt_raw"),
                "human_note": "",
                "corrected_source_dataset_dir": "",
                "corrected_episode_id": "",
            }
        )

    spec = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "formal_iter_tag": contract.formal_iter_tag,
        "nominal_dataset_dir": str(nominal_dir),
        "bundle_iter_tag": _bundle_iter_tag(contract.formal_iter_tag),
        "bundle_output_dir": str(contract.external_manual_correction_bundle_dir),
        "corrections": corrections,
    }
    _write_json(_resolve_path(Path(repo_root), spec_path), spec)
    return spec


def _normalize_spec(
    repo_root: Path,
    *,
    spec_path: Path,
    manifest_path: Path | None = None,
) -> tuple[dict[str, Any], Path, Path, str]:
    contract = load_stage3_iteration_contract(
        Path(repo_root), manifest_path=manifest_path
    )
    payload = _read_json(_resolve_path(Path(repo_root), spec_path))
    schema_version = _require_str(
        payload.get("schema_version"), field_name="schema_version"
    )
    if schema_version != SPEC_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manual correction spec schema_version={schema_version!r}; expected {SPEC_SCHEMA_VERSION!r}"
        )

    formal_iter_tag = _require_str(
        payload.get("formal_iter_tag"), field_name="formal_iter_tag"
    )
    if formal_iter_tag != contract.formal_iter_tag:
        raise ValueError(
            f"manual correction spec formal_iter_tag={formal_iter_tag!r} does not match stage3 contract {contract.formal_iter_tag!r}"
        )

    nominal_dataset_dir = _resolve_path(
        Path(repo_root),
        _require_str(
            payload.get("nominal_dataset_dir"), field_name="nominal_dataset_dir"
        ),
    )
    if nominal_dataset_dir != contract.nominal_dataset_dir:
        raise ValueError(
            "manual correction spec must reference the frozen stage3 nominal dataset dir "
            + f"{contract.nominal_dataset_dir}, got {nominal_dataset_dir}"
        )

    bundle_iter_tag = _require_str(
        payload.get("bundle_iter_tag") or _bundle_iter_tag(contract.formal_iter_tag),
        field_name="bundle_iter_tag",
    )
    corrections = payload.get("corrections")
    if not isinstance(corrections, list) or not corrections:
        raise ValueError(
            "manual correction spec must contain a non-empty corrections list"
        )
    return (
        payload,
        nominal_dataset_dir,
        contract.external_manual_correction_bundle_dir,
        bundle_iter_tag,
    )


def validate_manual_correction_spec(
    repo_root: Path,
    *,
    spec_path: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    payload, nominal_dataset_dir, _, bundle_iter_tag = _normalize_spec(
        repo_root,
        spec_path=spec_path,
        manifest_path=manifest_path,
    )
    _, nominal_episode_by_id, nominal_transitions_by_episode = _dataset_index(
        nominal_dataset_dir
    )

    validated: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(payload["corrections"], start=1):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"corrections[{index}] must be an object")
        entry = dict(raw_entry)
        unknown = (
            set(entry.keys())
            - set(REQUIRED_CORRECTION_SPEC_KEYS)
            - set(OPTIONAL_CORRECTION_SPEC_KEYS)
        )
        forbidden = set(entry.keys()) & FORBIDDEN_CORRECTION_SPEC_FIELDS
        if forbidden:
            forbidden_str = ", ".join(sorted(forbidden))
            raise ValueError(
                f"corrections[{index}] uses forbidden synthetic fields: {forbidden_str}"
            )
        if unknown:
            unknown_str = ", ".join(sorted(unknown))
            raise ValueError(
                f"corrections[{index}] contains unknown fields: {unknown_str}"
            )

        nominal_episode_id = _require_str(
            entry.get("nominal_episode_id"),
            field_name=f"corrections[{index}].nominal_episode_id",
        )
        if nominal_episode_id not in nominal_episode_by_id:
            raise ValueError(
                f"corrections[{index}] references unknown nominal_episode_id={nominal_episode_id!r}"
            )
        nominal_episode = nominal_episode_by_id[nominal_episode_id]
        if bool(nominal_episode.get("success_episode", False)):
            raise ValueError(
                f"corrections[{index}] nominal_episode_id={nominal_episode_id!r} is already successful; manual correction bundle expects failed nominal episodes"
            )

        nominal_t_start = _require_int(
            entry.get("nominal_t_start"),
            field_name=f"corrections[{index}].nominal_t_start",
        )
        nominal_t_end = _require_int(
            entry.get("nominal_t_end"), field_name=f"corrections[{index}].nominal_t_end"
        )
        if nominal_t_start < 0 or nominal_t_end < nominal_t_start:
            raise ValueError(
                f"corrections[{index}] has invalid nominal t range [{nominal_t_start}, {nominal_t_end}]"
            )
        max_nominal_t = _episode_last_t(
            nominal_transitions_by_episode, nominal_episode_id
        )
        if nominal_t_end > max_nominal_t:
            raise ValueError(
                f"corrections[{index}] nominal_t_end={nominal_t_end} exceeds max nominal t={max_nominal_t} for episode_id={nominal_episode_id!r}"
            )

        human_note = _require_str(
            entry.get("human_note"), field_name=f"corrections[{index}].human_note"
        )
        corrected_source_dataset_dir = _resolve_path(
            Path(repo_root),
            _require_str(
                entry.get("corrected_source_dataset_dir"),
                field_name=f"corrections[{index}].corrected_source_dataset_dir",
            ),
        )
        if corrected_source_dataset_dir == nominal_dataset_dir:
            raise ValueError(
                f"corrections[{index}] corrected_source_dataset_dir must not equal the nominal dataset dir; refusing to relabel nominal failures as manual corrections"
            )
        corrected_episode_id = _require_str(
            entry.get("corrected_episode_id"),
            field_name=f"corrections[{index}].corrected_episode_id",
        )

        _, corrected_episode_by_id, corrected_transitions_by_episode = _dataset_index(
            corrected_source_dataset_dir
        )
        if corrected_episode_id not in corrected_episode_by_id:
            raise ValueError(
                f"corrections[{index}] references unknown corrected_episode_id={corrected_episode_id!r} in {corrected_source_dataset_dir}"
            )
        corrected_episode = corrected_episode_by_id[corrected_episode_id]
        if not bool(corrected_episode.get("success_episode", False)):
            raise ValueError(
                f"corrections[{index}] corrected_episode_id={corrected_episode_id!r} is not marked success_episode=true in {corrected_source_dataset_dir}"
            )
        corrected_prompt_raw = corrected_episode.get("prompt_raw")
        nominal_prompt_raw = nominal_episode.get("prompt_raw")
        if (
            isinstance(corrected_prompt_raw, str)
            and corrected_prompt_raw
            and isinstance(nominal_prompt_raw, str)
            and nominal_prompt_raw
            and corrected_prompt_raw != nominal_prompt_raw
        ):
            raise ValueError(
                f"corrections[{index}] prompt mismatch between nominal episode {nominal_episode_id!r} and corrected source {corrected_episode_id!r}"
            )
        if not corrected_transitions_by_episode.get(corrected_episode_id):
            raise ValueError(
                f"corrections[{index}] corrected_episode_id={corrected_episode_id!r} has no transitions"
            )

        validated.append(
            {
                "correction_id": _require_str(
                    entry.get("correction_id") or f"correction_{index:03d}",
                    field_name=f"corrections[{index}].correction_id",
                ),
                "nominal_episode_id": nominal_episode_id,
                "nominal_t_start": nominal_t_start,
                "nominal_t_end": nominal_t_end,
                "prompt_raw": nominal_prompt_raw,
                "human_note": human_note,
                "corrected_source_dataset_dir": str(corrected_source_dataset_dir),
                "corrected_episode_id": corrected_episode_id,
            }
        )

    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "formal_iter_tag": payload["formal_iter_tag"],
        "nominal_dataset_dir": str(nominal_dataset_dir),
        "bundle_iter_tag": bundle_iter_tag,
        "corrections": validated,
    }


def _copy_npz(
    source_dataset_dir: Path,
    source_episode: Mapping[str, Any],
    *,
    output_arrays_dir: Path,
    output_episode_id: str,
) -> str:
    npz_path_val = source_episode.get("npz_path")
    source_npz = None
    if isinstance(npz_path_val, str) and npz_path_val:
        candidate = Path(npz_path_val)
        source_npz = (
            candidate if candidate.is_absolute() else (source_dataset_dir / candidate)
        )
    else:
        source_npz = (
            source_dataset_dir / "arrays" / f"{source_episode['episode_id']}.npz"
        )
    if source_npz is None or not source_npz.exists():
        raise ValueError(
            f"missing corrected source npz for episode_id={source_episode['episode_id']!r}"
        )
    output_arrays_dir.mkdir(parents=True, exist_ok=True)
    destination = output_arrays_dir / f"{output_episode_id}.npz"
    shutil.copy2(source_npz, destination)
    return str(Path("arrays") / destination.name)


def build_manual_correction_bundle(
    repo_root: Path,
    *,
    spec_path: Path,
    bundle_dir: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    validated = validate_manual_correction_spec(
        repo_root,
        spec_path=spec_path,
        manifest_path=manifest_path,
    )
    bundle_path = _resolve_path(Path(repo_root), bundle_dir)
    if bundle_path.exists():
        shutil.rmtree(bundle_path)
    bundle_path.mkdir(parents=True, exist_ok=True)
    arrays_dir = bundle_path / "arrays"
    episodes_out: list[dict[str, Any]] = []
    transitions_out: list[dict[str, Any]] = []
    correction_segments: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    bundle_iter_tag = str(validated["bundle_iter_tag"])
    corrections = validated["corrections"]
    if not isinstance(corrections, list):
        raise ValueError("validated corrections payload is invalid")

    for index, correction in enumerate(corrections, start=1):
        correction_entry = dict(correction)
        corrected_source_dataset_dir = Path(
            str(correction_entry["corrected_source_dataset_dir"])
        )
        _, corrected_episode_by_id, corrected_transitions_by_episode = _dataset_index(
            corrected_source_dataset_dir
        )
        corrected_episode_id = str(correction_entry["corrected_episode_id"])
        source_episode = dict(corrected_episode_by_id[corrected_episode_id])
        source_transitions = [
            dict(row) for row in corrected_transitions_by_episode[corrected_episode_id]
        ]
        output_episode_id = f"{bundle_iter_tag}_corr{index:03d}_{corrected_episode_id}"
        output_npz_rel = _copy_npz(
            corrected_source_dataset_dir,
            source_episode,
            output_arrays_dir=arrays_dir,
            output_episode_id=output_episode_id,
        )
        episode_out = dict(source_episode)
        episode_out["iter_tag"] = bundle_iter_tag
        episode_out["episode_id"] = output_episode_id
        episode_out["npz_path"] = output_npz_rel
        episodes_out.append(episode_out)

        for transition in source_transitions:
            transition_out = dict(transition)
            transition_out["iter_tag"] = bundle_iter_tag
            transition_out["episode_id"] = output_episode_id
            transition_out["npz_path"] = output_npz_rel
            transition_out["is_correction"] = True
            transitions_out.append(transition_out)

        correction_id = str(correction_entry["correction_id"])
        correction_segments.append(
            {
                "schema_version": SEGMENT_SCHEMA_VERSION,
                "correction_id": correction_id,
                "nominal_episode_id": str(correction_entry["nominal_episode_id"]),
                "nominal_t_start": int(correction_entry["nominal_t_start"]),
                "nominal_t_end": int(correction_entry["nominal_t_end"]),
                "human_note": str(correction_entry["human_note"]),
                "corrected_source_dataset_dir": str(corrected_source_dataset_dir),
                "corrected_episode_id": corrected_episode_id,
                "output_episode_id": output_episode_id,
                "prompt_raw": correction_entry.get("prompt_raw"),
                "is_correction": True,
            }
        )
        provenance_rows.append(
            {
                "correction_id": correction_id,
                "nominal_episode_id": str(correction_entry["nominal_episode_id"]),
                "corrected_source_dataset_dir": str(corrected_source_dataset_dir),
                "corrected_episode_id": corrected_episode_id,
                "output_episode_id": output_episode_id,
            }
        )

    _write_jsonl(bundle_path / "episodes.jsonl", episodes_out)
    _write_jsonl(bundle_path / "transitions.jsonl", transitions_out)
    _write_jsonl(bundle_path / "correction_segments.jsonl", correction_segments)
    _write_json(
        bundle_path / "manual_correction_bundle_report.json",
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "formal_iter_tag": validated["formal_iter_tag"],
            "bundle_iter_tag": bundle_iter_tag,
            "bundle_dir": str(bundle_path),
            "correction_count": len(correction_segments),
            "provenance": provenance_rows,
        },
    )
    validated_bundle = read_m1_dataset(bundle_path, check_npz_keys=True)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "bundle_dir": str(bundle_path),
        "bundle_iter_tag": bundle_iter_tag,
        "episode_count": _require_int(
            validated_bundle["n_episodes"], field_name="validated_bundle.n_episodes"
        ),
        "transition_count": _require_int(
            validated_bundle["n_transitions"],
            field_name="validated_bundle.n_transitions",
        ),
        "correction_segments_path": str(bundle_path / "correction_segments.jsonl"),
        "report_path": str(bundle_path / "manual_correction_bundle_report.json"),
    }


__all__ = [
    "DEFAULT_BUNDLE_ITER_TAG_SUFFIX",
    "FORBIDDEN_CORRECTION_SPEC_FIELDS",
    "REPORT_SCHEMA_VERSION",
    "SEGMENT_SCHEMA_VERSION",
    "SPEC_SCHEMA_VERSION",
    "build_manual_correction_bundle",
    "scaffold_manual_correction_spec",
    "validate_manual_correction_spec",
]
