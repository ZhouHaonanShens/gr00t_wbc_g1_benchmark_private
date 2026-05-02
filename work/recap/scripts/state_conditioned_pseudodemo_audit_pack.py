#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import random
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_HARVEST_DIR = Path("agent/artifacts/state_conditioned_materialization/harvest")
DEFAULT_TRAINING_DIR = Path(
    "agent/artifacts/state_conditioned_materialization/training"
)
DEFAULT_AUDIT_DIR = Path("agent/artifacts/state_conditioned_materialization/audit")

DEFAULT_SNAPSHOT_CANDIDATES = DEFAULT_HARVEST_DIR / "snapshot_candidates.jsonl"
DEFAULT_PSEUDODEMO_MANIFEST = (
    DEFAULT_HARVEST_DIR / "local_recovery_pseudodemo_manifest.json"
)
DEFAULT_LABELS = DEFAULT_TRAINING_DIR / "state_conditioned_sft_labels.jsonl"

DEFAULT_OUTPUT = DEFAULT_AUDIT_DIR / "pseudodemo_label_audit_pack.md"
DEFAULT_TAXONOMY_OUTPUT = DEFAULT_AUDIT_DIR / "pseudodemo_label_taxonomy.md"

AUDIT_SCHEMA_VERSION = "pseudodemo_label_audit_pack_v1"
TAXONOMY_SCHEMA_VERSION = "pseudodemo_label_taxonomy_v1"
FORMAL_PSEUDODEMO_SOURCE_BUCKET = "formal_pseudodemo"
COUNT_MIN = 50
COUNT_MAX = 100
REQUIRED_VIEWS: tuple[str, str] = ("C0", "C1")

NULL_PHASE_TOKEN = "__NULL_PHASE__"
NULL_MODE_TOKEN = "__NULL_MODE__"

TAXONOMY_CLASS_ORDER: tuple[str, ...] = ("valid", "ambiguous", "invalid")
TAXONOMY_SUBTYPES: tuple[dict[str, str], ...] = (
    {
        "decision_class": "valid",
        "taxonomy_subtype": "valid.phase_mode_supported_by_snapshot",
        "summary": "样本的 C1 phase/mode 与 source snapshot 的 phase/mode 一致。",
        "required_evidence": "audit_pack:<audit_ref>, snapshot:<source_snapshot_id>, label:<sample_id>",
    },
    {
        "decision_class": "valid",
        "taxonomy_subtype": "valid.teacher_provenance_consistent",
        "summary": "teacher provenance 字段与 manifest、producer_by_family 互相对齐。",
        "required_evidence": "audit_pack:<audit_ref>, manifest:<source_episode_id>, manifest_producer:<family>",
    },
    {
        "decision_class": "ambiguous",
        "taxonomy_subtype": "ambiguous.phase_boundary_unclear",
        "summary": "快照接近 phase 切换边界，现有证据不足以稳定判定 phase。",
        "required_evidence": "audit_pack:<audit_ref>, snapshot:<source_snapshot_id>, history:<source_snapshot_id>",
    },
    {
        "decision_class": "ambiguous",
        "taxonomy_subtype": "ambiguous.history_signal_insufficient",
        "summary": "history_valid_mask 或短视觉引用不足，人工无法可靠复核标签。",
        "required_evidence": "audit_pack:<audit_ref>, snapshot:<source_snapshot_id>, history:<source_snapshot_id>",
    },
    {
        "decision_class": "ambiguous",
        "taxonomy_subtype": "ambiguous.teacher_target_underexplained",
        "summary": "teacher target 触发条件存在合理疑问，但还不足以直接判 invalid。",
        "required_evidence": "audit_pack:<audit_ref>, manifest:<source_episode_id>, label:<sample_id>",
    },
    {
        "decision_class": "invalid",
        "taxonomy_subtype": "invalid.missing_sample_reference",
        "summary": "taxonomy claim 没有给出任何 sample_refs，按规则直接判无效。",
        "required_evidence": "taxonomy_claim:<claim_id>",
    },
    {
        "decision_class": "invalid",
        "taxonomy_subtype": "invalid.sample_ref_not_in_registry",
        "summary": "claim 引用的 audit_ref 不在本次 audit pack registry 中。",
        "required_evidence": "taxonomy_claim:<claim_id>, audit_registry:<audit_ref>",
    },
    {
        "decision_class": "invalid",
        "taxonomy_subtype": "invalid.snapshot_manifest_mismatch",
        "summary": "source_snapshot_id、anchor_t 或 anchor_episode_id 在 manifest 与 snapshot 间不一致。",
        "required_evidence": "audit_pack:<audit_ref>, snapshot:<source_snapshot_id>, manifest:<source_episode_id>",
    },
    {
        "decision_class": "invalid",
        "taxonomy_subtype": "invalid.phase_mode_contract_mismatch",
        "summary": "C1 label phase/mode 与 source snapshot phase/mode 冲突。",
        "required_evidence": "audit_pack:<audit_ref>, snapshot:<source_snapshot_id>, label:<sample_id>",
    },
    {
        "decision_class": "invalid",
        "taxonomy_subtype": "invalid.history_window_contract_mismatch",
        "summary": "history contract 字段之间互相冲突，导致标签上下文不可用。",
        "required_evidence": "audit_pack:<audit_ref>, snapshot:<source_snapshot_id>, history:<source_snapshot_id>",
    },
    {
        "decision_class": "invalid",
        "taxonomy_subtype": "invalid.teacher_provenance_mismatch",
        "summary": "teacher_trigger_* 或 producer / teacher_version 与 manifest 记录冲突。",
        "required_evidence": "audit_pack:<audit_ref>, manifest:<source_episode_id>, label:<sample_id>",
    },
)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic human-readable audit pack and taxonomy skeleton "
            "for state-conditioned formal pseudodemo labels."
        )
    )
    parser.add_argument(
        "--snapshot-candidates",
        type=Path,
        default=DEFAULT_SNAPSHOT_CANDIDATES,
        help="Task 8 snapshot_candidates.jsonl input.",
    )
    parser.add_argument(
        "--pseudodemo-manifest",
        type=Path,
        default=DEFAULT_PSEUDODEMO_MANIFEST,
        help="Task 9 local_recovery_pseudodemo_manifest.json input.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Task 10 state_conditioned_sft_labels.jsonl input.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=COUNT_MIN,
        help=f"Deterministic audit sample count, must be within [{COUNT_MIN}, {COUNT_MAX}].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic sampling seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Markdown path for the audit pack.",
    )
    parser.add_argument(
        "--taxonomy-output",
        type=Path,
        default=DEFAULT_TAXONOMY_OUTPUT,
        help="Markdown path for the taxonomy skeleton.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _validate_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"missing required {arg_name}: {resolved}")
    return resolved


def _validate_output_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(f"{arg_name} must be a file path, got directory: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    return state_conditioned_bucket_a_import._read_json(path)


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    return state_conditioned_bucket_a_import._read_jsonl_dicts(path)


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    return path


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _as_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


def _as_list(
    value: object,
    *,
    field_name: str,
    expected_len: int | None = None,
) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    items = list(value)
    if expected_len is not None and len(items) != int(expected_len):
        raise ValueError(
            f"{field_name} must have length {expected_len}, got {len(items)}"
        )
    return items


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _format_number(value: object | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _format_scalar(value: object | None) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _short_value(value: object | None, *, limit: int = 120) -> str:
    text = _format_scalar(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _mask_summary(mask: Sequence[object]) -> str:
    tokens = ["T" if item is True else "F" for item in mask]
    valid_count = sum(1 for item in mask if item is True)
    return f"{''.join(tokens)} , valid={valid_count}/{len(mask)}"


def _non_null_items(values: Sequence[object]) -> list[object]:
    return [value for value in values if value is not None]


def _history_window_summary(record: Mapping[str, Any]) -> list[str]:
    snapshot_record = dict(_as_mapping(record.get("snapshot"), field_name="snapshot"))
    c1 = dict(_as_mapping(record.get("c1"), field_name="c1"))
    history_valid_mask = _as_list(
        c1.get("history_valid_mask"),
        field_name="c1.history_valid_mask",
    )
    history_t_std_indices = _as_list(
        c1.get("history_t_std_indices"),
        field_name="c1.history_t_std_indices",
    )
    history_timestamp_s = _as_list(
        c1.get("history_timestamp_s"),
        field_name="c1.history_timestamp_s",
    )
    short_visual_refs = _as_list(
        c1.get("deployable.short_visual_history_refs"),
        field_name="c1.deployable.short_visual_history_refs",
    )
    previous_action_history = _as_list(
        c1.get("deployable.previous_action_history"),
        field_name="c1.deployable.previous_action_history",
    )
    proprio_history = _as_list(
        c1.get("deployable.proprio_history"),
        field_name="c1.deployable.proprio_history",
    )
    non_null_visual_refs = _non_null_items(short_visual_refs)
    latest_visual_ref = non_null_visual_refs[-1] if non_null_visual_refs else None
    return [
        (
            "`history_k/history_stride`: "
            + f"{_as_int(snapshot_record.get('history_k'), field_name='snapshot.history_k')}"
            + " / "
            + f"{_as_int(snapshot_record.get('history_stride'), field_name='snapshot.history_stride')}"
        ),
        f"`history_valid_mask`: {_mask_summary(history_valid_mask)}",
        (
            "`history_t_std_indices`: "
            + ", ".join(_format_scalar(item) for item in history_t_std_indices)
        ),
        (
            "`history_timestamp_s`: "
            + ", ".join(_format_number(item) for item in history_timestamp_s)
        ),
        (
            "`deployable.short_visual_history_refs`: "
            + f"non_null={len(non_null_visual_refs)}/{len(short_visual_refs)}, latest={_short_value(latest_visual_ref)}"
        ),
        (
            "`deployable.previous_action_history`: "
            + f"non_null={len(_non_null_items(previous_action_history))}/{len(previous_action_history)}"
        ),
        (
            "`deployable.proprio_history`: "
            + f"non_null={len(_non_null_items(proprio_history))}/{len(proprio_history)}"
        ),
    ]


def _load_snapshot_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for raw_row in _read_jsonl_dicts(path):
        row = dict(_as_mapping(raw_row, field_name="snapshot_candidates[]"))
        snapshot_id = _as_non_empty_string(
            row.get("snapshot_id"), field_name="snapshot_id"
        )
        if snapshot_id in index:
            raise ValueError(
                f"duplicate snapshot_id in snapshot candidates: {snapshot_id}"
            )
        index[snapshot_id] = row
    if not index:
        raise ValueError("snapshot candidate context is empty")
    return index


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = dict(_as_mapping(_read_json(path), field_name="pseudodemo_manifest"))
    pseudodemo_by_episode_id: dict[str, dict[str, Any]] = {}
    for raw_record in list(manifest.get("pseudodemos", [])):
        record = dict(
            _as_mapping(raw_record, field_name="pseudodemo_manifest.pseudodemos[]")
        )
        episode_id = _as_non_empty_string(
            record.get("episode_id"), field_name="episode_id"
        )
        if episode_id in pseudodemo_by_episode_id:
            raise ValueError(
                f"duplicate pseudodemo episode_id in manifest: {episode_id}"
            )
        pseudodemo_by_episode_id[episode_id] = record
    if not pseudodemo_by_episode_id:
        raise ValueError("pseudodemo manifest is empty")
    return manifest, pseudodemo_by_episode_id


def _assert_shared_value(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    field_name: str,
) -> Any:
    left_value = left.get(field_name)
    right_value = right.get(field_name)
    if left_value != right_value:
        raise ValueError(
            f"paired label rows disagree on {field_name}: {left_value!r} != {right_value!r}"
        )
    return left_value


def _load_formal_label_pairs(path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_row in _read_jsonl_dicts(path):
        row = dict(_as_mapping(raw_row, field_name="state_conditioned_sft_labels[]"))
        source_bucket = _as_non_empty_string(
            row.get("source_bucket"), field_name="source_bucket"
        )
        if source_bucket != FORMAL_PSEUDODEMO_SOURCE_BUCKET:
            continue
        sample_id = _as_non_empty_string(row.get("sample_id"), field_name="sample_id")
        training_view = _as_non_empty_string(
            row.get("training_view"), field_name="training_view"
        )
        if training_view not in REQUIRED_VIEWS:
            raise ValueError(
                f"formal pseudodemo sample {sample_id!r} has unsupported training_view={training_view!r}"
            )
        sample_views = grouped.setdefault(sample_id, {})
        if training_view in sample_views:
            raise ValueError(
                f"duplicate formal pseudodemo row for sample_id={sample_id!r}, training_view={training_view!r}"
            )
        sample_views[training_view] = row

    if not grouped:
        raise ValueError("formal_pseudodemo label rows are empty")

    result: list[dict[str, Any]] = []
    for sample_id in sorted(grouped):
        sample_views = grouped[sample_id]
        missing_views = [view for view in REQUIRED_VIEWS if view not in sample_views]
        if missing_views:
            raise ValueError(
                f"formal pseudodemo sample {sample_id!r} is missing views: {missing_views}"
            )
        c0 = sample_views["C0"]
        c1 = sample_views["C1"]
        source_episode_id = _as_non_empty_string(
            _assert_shared_value(c0, c1, field_name="source_episode_id"),
            field_name="source_episode_id",
        )
        source_snapshot_id = _as_non_empty_string(
            _assert_shared_value(c0, c1, field_name="source_snapshot_id"),
            field_name="source_snapshot_id",
        )
        source_sample_key = _as_non_empty_string(
            _assert_shared_value(c0, c1, field_name="source_sample_key"),
            field_name="source_sample_key",
        )
        repeat_index = _as_int(
            _assert_shared_value(c0, c1, field_name="repeat_index"),
            field_name="repeat_index",
        )
        source_t = _as_int(
            _assert_shared_value(c0, c1, field_name="source_t"), field_name="source_t"
        )
        result.append(
            {
                "sample_id": sample_id,
                "source_episode_id": source_episode_id,
                "source_snapshot_id": source_snapshot_id,
                "source_sample_key": source_sample_key,
                "repeat_index": repeat_index,
                "source_t": source_t,
                "c0": c0,
                "c1": c1,
            }
        )
    return result


def _join_inputs(
    *,
    label_pairs: Sequence[Mapping[str, Any]],
    snapshot_index: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    pseudodemo_by_episode_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    producer_by_family = dict(
        _as_mapping(
            manifest.get("producer_by_family", {}), field_name="producer_by_family"
        )
    )
    joined: list[dict[str, Any]] = []
    for pair in label_pairs:
        source_episode_id = _as_non_empty_string(
            pair.get("source_episode_id"), field_name="source_episode_id"
        )
        source_snapshot_id = _as_non_empty_string(
            pair.get("source_snapshot_id"), field_name="source_snapshot_id"
        )
        manifest_record = dict(
            _as_mapping(
                pseudodemo_by_episode_id.get(source_episode_id),
                field_name=f"pseudodemo_manifest[{source_episode_id}]",
            )
        )
        snapshot_record = dict(
            _as_mapping(
                snapshot_index.get(source_snapshot_id),
                field_name=f"snapshot_index[{source_snapshot_id}]",
            )
        )
        manifest_snapshot_id = _as_non_empty_string(
            manifest_record.get("source_snapshot_id"),
            field_name="manifest.source_snapshot_id",
        )
        if manifest_snapshot_id != source_snapshot_id:
            raise ValueError(
                f"manifest/source snapshot mismatch for {source_episode_id!r}: {manifest_snapshot_id!r} != {source_snapshot_id!r}"
            )
        if _as_non_empty_string(
            manifest_record.get("anchor_episode_id"),
            field_name="manifest.anchor_episode_id",
        ) != _as_non_empty_string(
            snapshot_record.get("anchor_episode_id"),
            field_name="snapshot.anchor_episode_id",
        ):
            raise ValueError(
                f"anchor_episode_id mismatch for {source_episode_id!r} and {source_snapshot_id!r}"
            )
        if _as_int(
            manifest_record.get("anchor_t"), field_name="manifest.anchor_t"
        ) != _as_int(snapshot_record.get("anchor_t"), field_name="snapshot.anchor_t"):
            raise ValueError(
                f"anchor_t mismatch for {source_episode_id!r} and {source_snapshot_id!r}"
            )
        family = _as_non_empty_string(
            manifest_record.get("source_snapshot_family"),
            field_name="manifest.source_snapshot_family",
        )
        producer = _as_non_empty_string(
            manifest_record.get("producer"), field_name="manifest.producer"
        )
        expected_producer = producer_by_family.get(family)
        if expected_producer is not None and str(expected_producer) != producer:
            raise ValueError(
                f"producer_by_family mismatch for family={family!r}: {expected_producer!r} != {producer!r}"
            )
        joined.append(
            {
                "sample_id": _as_non_empty_string(
                    pair.get("sample_id"), field_name="sample_id"
                ),
                "source_episode_id": source_episode_id,
                "source_snapshot_id": source_snapshot_id,
                "source_sample_key": _as_non_empty_string(
                    pair.get("source_sample_key"), field_name="source_sample_key"
                ),
                "repeat_index": _as_int(
                    pair.get("repeat_index"), field_name="repeat_index"
                ),
                "source_t": _as_int(pair.get("source_t"), field_name="source_t"),
                "family": family,
                "producer": producer,
                "c0": dict(_as_mapping(pair.get("c0"), field_name="c0")),
                "c1": dict(_as_mapping(pair.get("c1"), field_name="c1")),
                "manifest": manifest_record,
                "snapshot": snapshot_record,
            }
        )
    if not joined:
        raise ValueError("formal pseudodemo label pairs are empty after input join")
    return joined


def _select_samples(
    records: Sequence[Mapping[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if count < COUNT_MIN or count > COUNT_MAX:
        raise ValueError(
            f"--count must be within [{COUNT_MIN}, {COUNT_MAX}], got {count}"
        )
    ordered_records = [
        dict(record)
        for record in sorted(
            records,
            key=lambda item: (
                str(item.get("family", "")),
                str(item.get("source_snapshot_id", "")),
                str(item.get("source_episode_id", "")),
                int(item.get("repeat_index", 0)),
                str(item.get("sample_id", "")),
            ),
        )
    ]
    if count > len(ordered_records):
        raise ValueError(
            f"requested count={count} exceeds available formal pseudodemo sample pairs={len(ordered_records)}"
        )
    selected = random.Random(seed).sample(ordered_records, count)
    result: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        mutable = dict(record)
        mutable["audit_ref"] = f"audit-sample-{index:03d}"
        result.append(mutable)
    return result


def _sample_index_table(records: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| audit_ref | family | sample_id | source_snapshot_id | repeat_index | C1 phase | C1 mode |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        c1 = dict(_as_mapping(record.get("c1"), field_name="c1"))
        lines.append(
            "| "
            + f"{record['audit_ref']} | {record['family']} | {_short_value(record['sample_id'], limit=72)} | "
            + f"{_short_value(record['source_snapshot_id'], limit=56)} | {record['repeat_index']} | "
            + f"{_format_scalar(c1.get('policy_condition.phase'))} | {_format_scalar(c1.get('policy_condition.mode'))} |"
        )
    return lines


def _metadata_lines(
    *,
    snapshot_candidates_path: Path,
    pseudodemo_manifest_path: Path,
    labels_path: Path,
    count: int,
    seed: int,
    available_count: int,
    selected_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    family_counts = Counter(
        str(record.get("family", "unknown")) for record in selected_records
    )
    return [
        f"- `schema_version`: `{AUDIT_SCHEMA_VERSION}`",
        f"- `generated_at_utc`: `{datetime.now(timezone.utc).isoformat()}`",
        f"- `snapshot_candidates`: `{_relative_path(snapshot_candidates_path)}`",
        f"- `pseudodemo_manifest`: `{_relative_path(pseudodemo_manifest_path)}`",
        f"- `labels`: `{_relative_path(labels_path)}`",
        f"- `requested_count`: `{count}`",
        f"- `sample_seed`: `{seed}`",
        f"- `available_formal_pseudodemo_sample_pairs`: `{available_count}`",
        f"- `selected_family_counts`: `{dict(sorted(family_counts.items()))}`",
        "- 说明: 本文件只读引用既有 artifacts, 用于后续人工审计, 不回写训练数据。",
    ]


def _snapshot_moment_lines(record: Mapping[str, Any]) -> list[str]:
    snapshot = dict(_as_mapping(record.get("snapshot"), field_name="snapshot"))
    return [
        f"`source_snapshot_id`: `{record['source_snapshot_id']}`",
        f"`anchor_episode_id`: `{_as_non_empty_string(snapshot.get('anchor_episode_id'), field_name='snapshot.anchor_episode_id')}`",
        f"`anchor_t`: `{_as_int(snapshot.get('anchor_t'), field_name='snapshot.anchor_t')}`",
        f"`anchor_mujoco_state_ref`: `{_format_scalar(snapshot.get('anchor_mujoco_state_ref'))}`",
        f"`anchor_xy_distance`: `{_format_number(snapshot.get('anchor_xy_distance'))}`",
    ]


def _family_source_lines(record: Mapping[str, Any]) -> list[str]:
    manifest = dict(_as_mapping(record.get("manifest"), field_name="manifest"))
    return [
        f"`source_snapshot_family`: `{record['family']}`",
        f"`producer`: `{record['producer']}`",
        f"`teacher_version`: `{_format_scalar(manifest.get('teacher_version'))}`",
        f"`source_bucket`: `{_format_scalar(record['c1'].get('source_bucket'))}`",
        f"`source_kind`: `{_format_scalar(record['c1'].get('source_kind'))}`",
        f"`source_episode_id`: `{record['source_episode_id']}`",
        f"`source_sample_key`: `{record['source_sample_key']}`",
    ]


def _phase_mode_lines(record: Mapping[str, Any]) -> list[str]:
    snapshot = dict(_as_mapping(record.get("snapshot"), field_name="snapshot"))
    c0 = dict(_as_mapping(record.get("c0"), field_name="c0"))
    c1 = dict(_as_mapping(record.get("c1"), field_name="c1"))
    return [
        (
            "`snapshot.policy_condition.phase/mode`: `"
            + f"{_format_scalar(snapshot.get('policy_condition.phase'))} / {_format_scalar(snapshot.get('policy_condition.mode'))}`"
        ),
        (
            "`C0.policy_condition.phase/mode`: `"
            + f"{_format_scalar(c0.get('policy_condition.phase'))} / {_format_scalar(c0.get('policy_condition.mode'))}`"
        ),
        (
            "`C1.policy_condition.phase/mode`: `"
            + f"{_format_scalar(c1.get('policy_condition.phase'))} / {_format_scalar(c1.get('policy_condition.mode'))}`"
        ),
        f"`snapshot.policy_condition_text`: `{_short_value(snapshot.get('policy_condition_text'), limit=120)}`",
        f"`C1.policy_condition_text`: `{_short_value(c1.get('policy_condition_text'), limit=120)}`",
    ]


def _teacher_target_lines(record: Mapping[str, Any]) -> list[str]:
    manifest = dict(_as_mapping(record.get("manifest"), field_name="manifest"))
    return [
        f"`teacher_trigger_reason`: `{_format_scalar(manifest.get('teacher_trigger_reason'))}`",
        f"`teacher_trigger_success_rate`: `{_format_number(manifest.get('teacher_trigger_success_rate'))}`",
        f"`teacher_trigger_threshold`: `{_format_number(manifest.get('teacher_trigger_threshold'))}`",
        (
            "`failure_prefix_source_t_range`: `"
            + f"{_format_scalar(manifest.get('failure_prefix_source_t_range'))}`, "
            + f"step_count={_format_number(manifest.get('failure_prefix_step_count'))}"
        ),
        (
            "`recovery_suffix_source_t_range`: `"
            + f"{_format_scalar(manifest.get('recovery_suffix_source_t_range'))}`, "
            + f"step_count={_format_number(manifest.get('recovery_suffix_step_count'))}"
        ),
    ]


def _pseudodemo_label_lines(record: Mapping[str, Any]) -> list[str]:
    c0 = dict(_as_mapping(record.get("c0"), field_name="c0"))
    c1 = dict(_as_mapping(record.get("c1"), field_name="c1"))
    return [
        f"`sample_id`: `{record['sample_id']}`",
        f"`repeat_index`: `{record['repeat_index']}`",
        f"`source_t`: `{record['source_t']}`",
        (
            "`C0 label`: `"
            + f"{_format_scalar(c0.get('policy_condition.phase'))} / {_format_scalar(c0.get('policy_condition.mode'))}`"
        ),
        (
            "`C1 label`: `"
            + f"{_format_scalar(c1.get('policy_condition.phase'))} / {_format_scalar(c1.get('policy_condition.mode'))}`"
        ),
        (
            "`label_decision_anchor`: `"
            + f"{record['audit_ref']} | {record['sample_id']} | {record['source_snapshot_id']}`"
        ),
    ]


def _evidence_reference_lines(record: Mapping[str, Any]) -> list[str]:
    return [
        f"`audit_ref`: `{record['audit_ref']}`",
        f"`audit_pack_ref`: `audit_pack:{record['audit_ref']}`",
        f"`snapshot_ref`: `snapshot:{record['source_snapshot_id']}`",
        f"`manifest_ref`: `manifest:{record['source_episode_id']}`",
        f"`label_ref`: `label:{record['sample_id']}`",
    ]


def _append_block(lines: list[str], title: str, items: Sequence[str]) -> None:
    lines.append(title)
    for item in items:
        lines.append(f"- {item}")
    lines.append("")


def build_audit_pack_markdown(
    *,
    snapshot_candidates_path: Path,
    pseudodemo_manifest_path: Path,
    labels_path: Path,
    count: int,
    seed: int,
    available_count: int,
    records: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# state_conditioned pseudodemo 标签抽样审计包",
        "",
        "本文件用于后续人工抽样复核。采样只读引用现有 bridge artifacts，不修改原始 labels、manifest 或 snapshot 数据。",
        "",
        "## 生成元信息",
        *_metadata_lines(
            snapshot_candidates_path=snapshot_candidates_path,
            pseudodemo_manifest_path=pseudodemo_manifest_path,
            labels_path=labels_path,
            count=count,
            seed=seed,
            available_count=available_count,
            selected_records=records,
        ),
        "",
        "## 样本索引",
        *_sample_index_table(records),
        "",
    ]

    for record in records:
        lines.append(f"## {record['audit_ref']}")
        lines.append("")
        _append_block(lines, "### 1. 快照时刻", _snapshot_moment_lines(record))
        _append_block(
            lines,
            "### 2. 历史窗口摘要",
            _history_window_summary(record),
        )
        _append_block(
            lines, "### 3. family 与 source provenance", _family_source_lines(record)
        )
        _append_block(lines, "### 4. phase 与 mode", _phase_mode_lines(record))
        _append_block(
            lines, "### 5. teacher target 摘要", _teacher_target_lines(record)
        )
        _append_block(
            lines, "### 6. pseudodemo label 摘要", _pseudodemo_label_lines(record)
        )
        _append_block(lines, "### 7. 证据引用", _evidence_reference_lines(record))
    return "\n".join(lines).rstrip() + "\n"


def _taxonomy_registry_table(records: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| audit_ref | sample_id | source_snapshot_id | family | repeat_index | C1 phase | C1 mode |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        c1 = dict(_as_mapping(record.get("c1"), field_name="c1"))
        lines.append(
            "| "
            + f"{record['audit_ref']} | {_short_value(record['sample_id'], limit=72)} | {_short_value(record['source_snapshot_id'], limit=56)} | "
            + f"{record['family']} | {record['repeat_index']} | {_format_scalar(c1.get('policy_condition.phase'))} | {_format_scalar(c1.get('policy_condition.mode'))} |"
        )
    return lines


def _taxonomy_claim_template(subtype: Mapping[str, Any]) -> list[str]:
    taxonomy_subtype = _as_non_empty_string(
        subtype.get("taxonomy_subtype"), field_name="taxonomy_subtype"
    )
    decision_class = _as_non_empty_string(
        subtype.get("decision_class"), field_name="decision_class"
    )
    sample_refs_stub = "[]"
    evidence_refs_stub = "[]"
    if taxonomy_subtype != "invalid.missing_sample_reference":
        sample_refs_stub = "[audit-sample-001]"
        evidence_refs_stub = (
            "["
            + _as_non_empty_string(
                subtype.get("required_evidence"),
                field_name="required_evidence",
            )
            + "]"
        )
    return [
        "```text",
        f"claim_id: TBD_{taxonomy_subtype.replace('.', '_').upper()}_001",
        f"decision_class: {decision_class}",
        f"taxonomy_subtype: {taxonomy_subtype}",
        f"sample_refs: {sample_refs_stub}",
        f"evidence_refs: {evidence_refs_stub}",
        "status: pending",
        "reviewer: TBD",
        "review_notes: ",
        "```",
    ]


def build_taxonomy_markdown(
    *,
    audit_pack_path: Path,
    records: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# pseudodemo 标签 taxonomy 骨架",
        "",
        "本文件是后续人工 taxonomy 标注骨架。格式固定，便于后续 QA 或 checker 直接读取。",
        "",
        "## 元信息",
        f"- `taxonomy_schema_version`: `{TAXONOMY_SCHEMA_VERSION}`",
        f"- `generated_at_utc`: `{datetime.now(timezone.utc).isoformat()}`",
        f"- `audit_pack_path`: `{_relative_path(audit_pack_path)}`",
        f"- `audit_sample_count`: `{len(records)}`",
        f"- `allowed_decision_classes`: `{list(TAXONOMY_CLASS_ORDER)}`",
        f"- `subtype_count`: `{len(TAXONOMY_SUBTYPES)}`",
        "",
        "## 审计样本引用表",
        *_taxonomy_registry_table(records),
        "",
        "## QA 最小校验路径",
        "- 每条 claim 必须保留固定字段: `claim_id`, `decision_class`, `taxonomy_subtype`, `sample_refs`, `evidence_refs`, `status`, `reviewer`, `review_notes`。",
        "- `sample_refs` 不能为空。若为空，该 claim 只能归到 `invalid.missing_sample_reference`，否则 QA 直接失败。",
        "- `sample_refs` 中的每个值都必须出现在上面的 `audit_ref` registry 里。不存在的引用只能归到 `invalid.sample_ref_not_in_registry`。",
        "- `evidence_refs` 至少应包含一条 `audit_pack:*` 引用，以及一条 `snapshot:*`、`manifest:*` 或 `label:*` 引用。",
        "- `decision_class` 只能是 `valid`、`ambiguous`、`invalid`。`taxonomy_subtype` 必须与其 class 前缀一致。",
        "- 后续 task 8 若要填充 claim，只能追加 claim 内容，不能改本文件的 section 名和字段名。",
        "",
    ]

    for decision_class in TAXONOMY_CLASS_ORDER:
        lines.append(f"## {decision_class}")
        lines.append("")
        for subtype in TAXONOMY_SUBTYPES:
            if subtype["decision_class"] != decision_class:
                continue
            lines.append(f"### {subtype['taxonomy_subtype']}")
            lines.append("")
            lines.append(f"- 定义: {subtype['summary']}")
            lines.append(f"- 必需 evidence_refs: `{subtype['required_evidence']}`")
            lines.append("- claim 模板:")
            lines.extend(_taxonomy_claim_template(subtype))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(args: argparse.Namespace) -> tuple[Path, Path, int]:
    snapshot_candidates_path = _validate_existing_file(
        args.snapshot_candidates,
        arg_name="snapshot candidates JSONL",
    )
    pseudodemo_manifest_path = _validate_existing_file(
        args.pseudodemo_manifest,
        arg_name="pseudodemo manifest JSON",
    )
    labels_path = _validate_existing_file(
        args.labels, arg_name="state-conditioned labels JSONL"
    )
    output_path = _validate_output_file(args.output, arg_name="audit output")
    taxonomy_output_path = _validate_output_file(
        args.taxonomy_output,
        arg_name="taxonomy output",
    )

    snapshot_index = _load_snapshot_index(snapshot_candidates_path)
    manifest, pseudodemo_by_episode_id = _load_manifest(pseudodemo_manifest_path)
    label_pairs = _load_formal_label_pairs(labels_path)
    joined_records = _join_inputs(
        label_pairs=label_pairs,
        snapshot_index=snapshot_index,
        manifest=manifest,
        pseudodemo_by_episode_id=pseudodemo_by_episode_id,
    )
    selected_records = _select_samples(
        joined_records,
        count=_as_int(args.count, field_name="count"),
        seed=_as_int(args.seed, field_name="seed"),
    )

    audit_pack = build_audit_pack_markdown(
        snapshot_candidates_path=snapshot_candidates_path,
        pseudodemo_manifest_path=pseudodemo_manifest_path,
        labels_path=labels_path,
        count=int(args.count),
        seed=int(args.seed),
        available_count=len(joined_records),
        records=selected_records,
    )
    taxonomy = build_taxonomy_markdown(
        audit_pack_path=output_path,
        records=selected_records,
    )

    _write_text(output_path, audit_pack)
    _write_text(taxonomy_output_path, taxonomy)
    return output_path, taxonomy_output_path, len(selected_records)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        output_path, taxonomy_output_path, written_count = run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {_exception_message(exc)}", file=sys.stderr)
        return 1
    print(f"wrote audit pack: {_relative_path(output_path)}")
    print(f"wrote taxonomy skeleton: {_relative_path(taxonomy_output_path)}")
    print(f"selected audit samples: {written_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
