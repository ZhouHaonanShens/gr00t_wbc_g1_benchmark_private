#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


sys.dont_write_bytecode = True


STAGE_LABEL_MATERIALIZATION = "label_materialization"
STAGE_STATE_CONDITIONED_BUILD = "state_conditioned_build"
STAGE_LEROBOT_EXPORT = "lerobot_export"
STAGE_POST_EXPORT_SCHEMA_PROJECTION = "post_export_schema_projection"
STAGE_READER_SCHEMA_MISMATCH = "reader_schema_mismatch"
STAGE_READER_ALIAS_MISMATCH = "reader_alias_mismatch"

STAGE_ORDER: tuple[str, ...] = (
    STAGE_LABEL_MATERIALIZATION,
    STAGE_STATE_CONDITIONED_BUILD,
    STAGE_LEROBOT_EXPORT,
    STAGE_POST_EXPORT_SCHEMA_PROJECTION,
    STAGE_READER_SCHEMA_MISMATCH,
    STAGE_READER_ALIAS_MISMATCH,
)

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_BLOCKED = "blocked_by_upstream_failure"
STATUS_INSUFFICIENT = "insufficient_evidence"

AUDIT_STATUS_COMPLETE = "complete"
AUDIT_STATUS_FAIL_CLOSED = "fail_closed"

SCHEMA_VERSION = "carrier_lineage_audit_v1"
TRACE_SCHEMA_VERSION = "carrier_lineage_trace_v1"
AUDIT_ARTIFACT_KIND = "carrier_lineage_audit"

AUDIT_JSON_NAME = "carrier_lineage_audit.json"
AUDIT_MD_NAME = "carrier_lineage_audit.md"
MISSING_EXAMPLES_CSV_NAME = "carrier_missing_examples.csv"
TRACE_DIRNAME = "carrier_lineage_traces"

DEFAULT_OUTPUT_DIR = Path("agent/artifacts/apple_recap_exec")
DEFAULT_LABELS = Path(
    "agent/artifacts/recap_datasets/fullsize_relabel_v1/m2_labels/labels.jsonl"
)
DEFAULT_RECAP_DATASET_DIR = Path("agent/artifacts/recap_datasets/fullsize_relabel_v1")
DEFAULT_EXPORT_DATASET_DIR = Path(
    "agent/artifacts/lerobot_datasets/fullsize_relabel_v1"
)
DEFAULT_STATE_CONDITIONED_LABELS = Path(
    "agent/artifacts/state_conditioned_materialization/training/state_conditioned_sft_labels.jsonl"
)
DEFAULT_EXECUTION_FREEZE = Path(
    "agent/artifacts/apple_recap_exec/execution_freeze_contract.json"
)
DEFAULT_CARRIER_PARITY_REPORT = Path(
    "agent/artifacts/apple_recap_exec/carrier_parity_report.json"
)
DEFAULT_UPLIFT_VERDICT = Path("agent/artifacts/apple_recap_exec/uplift_verdict.json")

DEFAULT_TRACE_COUNT = 5
DEFAULT_MISSING_EXAMPLE_LIMIT = 24

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.data.contract_mapping import PHASE1_PROMPT_FEATURE_FALLBACK_KEY
from work.openpi.data.contract_mapping import PHASE1_PROMPT_FEATURE_PRIMARY_KEY
from work.openpi.data.contract_mapping import build_phase1_dataset_mapping_spec
from work.recap import text_indicator
from work.recap.lerobot_export import dataset_export as lerobot_dataset_export
from work.recap.scripts.state_conditioned_common import (
    exception_message as _exc_message,
)
from work.recap.scripts.state_conditioned_common import read_json as _read_json
from work.recap.scripts.state_conditioned_common import read_jsonl_dicts as _read_jsonl
from work.recap.scripts.state_conditioned_common import (
    validate_existing_dir as _validate_existing_dir,
)
from work.recap.scripts.state_conditioned_common import (
    validate_existing_file as _validate_existing_file,
)
from work.recap.state_conditioned import build_training_set as state_conditioned_build


@dataclass(frozen=True)
class CanonicalRow:
    row_key: str
    line_number: int
    episode_id: str
    t: int
    prompt_raw: str
    indicator_I: int
    indicator_mode: str
    expected_carrier_text_v1: str
    carrier_text_v1: str | None
    source_artifact_id: str


@dataclass(frozen=True)
class StageCoverage:
    stage_name: str
    pass_row_keys: frozenset[str] = frozenset()
    fail_row_keys: frozenset[str] = frozenset()
    unresolved_row_keys: frozenset[str] = frozenset()
    reasons: tuple[str, ...] = ()
    source_artifact_ids: tuple[str, ...] = ()
    blocked_by_stage: str | None = None
    explicit_status: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the authoritative carrier lineage for the current blocked "
            "AppleToPlate execution and either lock a unique first failing stage "
            "or fail closed deterministically."
        )
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Frozen mainline labels artifact (authority input).",
    )
    parser.add_argument(
        "--recap-dataset-dir",
        type=Path,
        default=DEFAULT_RECAP_DATASET_DIR,
        help="RECAP dataset directory containing episodes/transitions/source refs.",
    )
    parser.add_argument(
        "--export-dataset-dir",
        type=Path,
        default=DEFAULT_EXPORT_DATASET_DIR,
        help="Exported LeRobot dataset directory for the same iter_tag.",
    )
    parser.add_argument(
        "--state-conditioned-labels",
        type=Path,
        default=DEFAULT_STATE_CONDITIONED_LABELS,
        help="Optional state-conditioned training labels artifact for same-cohort pass evidence.",
    )
    parser.add_argument(
        "--execution-freeze-contract",
        type=Path,
        default=DEFAULT_EXECUTION_FREEZE,
        help="Current blocked execution freeze contract.",
    )
    parser.add_argument(
        "--carrier-parity-report",
        type=Path,
        default=DEFAULT_CARRIER_PARITY_REPORT,
        help="Existing carrier parity report for current blocked execution.",
    )
    parser.add_argument(
        "--uplift-verdict",
        type=Path,
        default=DEFAULT_UPLIFT_VERDICT,
        help="Blocked closeout verdict for current execution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for audit JSON/Markdown/CSV/trace outputs.",
    )
    parser.add_argument(
        "--trace-count",
        type=int,
        default=DEFAULT_TRACE_COUNT,
        help="How many row lineage traces to emit (minimum 5 recommended).",
    )
    parser.add_argument(
        "--missing-example-limit",
        type=int,
        default=DEFAULT_MISSING_EXAMPLE_LIMIT,
        help="How many missing-carrier examples to write to CSV.",
    )
    return parser


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(dict(payload)), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(path)
    return path


def _write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
    tmp.replace(path)
    return path


def _write_csv_atomic(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)
    return path


def _as_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text else None


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {type(value).__name__}")
    return int(value)


def _ordered_digest(row_keys: Iterable[str]) -> str:
    return _sha256_bytes(
        "\n".join(sorted(str(item) for item in row_keys)).encode("utf-8")
    )


def stable_row_lineage_key(
    *,
    source_artifact_id: str,
    resolved_path: str | Path,
    content_sha256: str,
    row_locator: str,
    episode_id: str,
    t: int,
    manifest_context: Mapping[str, object] | None = None,
) -> str:
    payload: dict[str, object] = {
        "source_artifact_id": str(source_artifact_id),
        "resolved_path": str(Path(resolved_path).resolve()),
        "content_sha256": str(content_sha256),
        "row_locator": str(row_locator),
        "episode_id": str(episode_id),
        "t": int(t),
    }
    if manifest_context:
        payload["manifest_context"] = {
            str(key): manifest_context[key] for key in sorted(manifest_context)
        }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _coverageset(items: Iterable[str]) -> frozenset[str]:
    return frozenset(str(item) for item in items)


def build_stage_coverage(
    *,
    stage_name: str,
    canonical_row_keys: Iterable[str],
    pass_row_keys: Iterable[str] = (),
    fail_row_keys: Iterable[str] = (),
    unresolved_row_keys: Iterable[str] = (),
    reasons: Sequence[str] = (),
    source_artifact_ids: Sequence[str] = (),
    blocked_by_stage: str | None = None,
    explicit_status: str | None = None,
) -> StageCoverage:
    canonical_set = _coverageset(canonical_row_keys)
    pass_set = _coverageset(pass_row_keys)
    fail_set = _coverageset(fail_row_keys)
    unresolved_set = _coverageset(unresolved_row_keys)
    covered = pass_set | fail_set | unresolved_set
    if not covered.issubset(canonical_set):
        extra = sorted(covered - canonical_set)
        raise ValueError(
            f"stage={stage_name} contains row keys outside canonical cohort: {extra[:3]}"
        )
    return StageCoverage(
        stage_name=stage_name,
        pass_row_keys=pass_set,
        fail_row_keys=fail_set,
        unresolved_row_keys=unresolved_set,
        reasons=tuple(str(item) for item in reasons),
        source_artifact_ids=tuple(str(item) for item in source_artifact_ids),
        blocked_by_stage=blocked_by_stage,
        explicit_status=explicit_status,
    )


def summarize_stage_coverage(
    coverage: StageCoverage,
    *,
    canonical_row_keys: Iterable[str],
) -> dict[str, Any]:
    canonical_set = _coverageset(canonical_row_keys)
    pass_set = set(coverage.pass_row_keys)
    fail_set = set(coverage.fail_row_keys)
    unresolved_set = set(coverage.unresolved_row_keys)
    uncovered_set = set(canonical_set) - pass_set - fail_set - unresolved_set
    unresolved_total = unresolved_set | uncovered_set

    if coverage.explicit_status is not None:
        status = coverage.explicit_status
    elif coverage.blocked_by_stage is not None:
        status = STATUS_BLOCKED
    elif fail_set:
        status = STATUS_FAIL
    elif pass_set == set(canonical_set) and not unresolved_total:
        status = STATUS_PASS
    else:
        status = STATUS_INSUFFICIENT

    return {
        "stage_name": coverage.stage_name,
        "stage_index": int(STAGE_ORDER.index(coverage.stage_name)),
        "status": status,
        "blocked_by_stage": coverage.blocked_by_stage,
        "reason_codes": list(coverage.reasons),
        "source_artifact_ids": list(coverage.source_artifact_ids),
        "canonical_row_count": int(len(canonical_set)),
        "canonical_row_key_digest": _ordered_digest(canonical_set),
        "pass_row_count": int(len(pass_set)),
        "pass_row_key_digest": _ordered_digest(pass_set),
        "fail_row_count": int(len(fail_set)),
        "fail_row_key_digest": _ordered_digest(fail_set),
        "unresolved_row_count": int(len(unresolved_total)),
        "unresolved_row_key_digest": _ordered_digest(unresolved_total),
        "sample_pass_row_keys": sorted(pass_set)[:5],
        "sample_fail_row_keys": sorted(fail_set)[:5],
        "sample_unresolved_row_keys": sorted(unresolved_total)[:5],
        "all_rows_pass": bool(
            pass_set == set(canonical_set) and not fail_set and not unresolved_total
        ),
        "all_rows_fail": bool(
            fail_set == set(canonical_set) and not pass_set and not unresolved_total
        ),
    }


def resolve_audit_outcome(
    stage_summaries: Sequence[Mapping[str, Any]],
    *,
    canonical_row_keys: Iterable[str],
    provenance_blockers: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    canonical_set = _coverageset(canonical_row_keys)
    if not canonical_set:
        return {
            "audit_status": AUDIT_STATUS_FAIL_CLOSED,
            "first_failing_stage": None,
            "reason_codes": ["empty_canonical_cohort"],
        }

    for stage_summary in stage_summaries:
        stage_name = str(stage_summary["stage_name"])
        status = str(stage_summary["status"])
        all_rows_fail = bool(stage_summary.get("all_rows_fail", False))
        if not all_rows_fail:
            continue
        earlier_summaries = [
            item
            for item in stage_summaries
            if int(item["stage_index"]) < int(stage_summary["stage_index"])
        ]
        if all(bool(item.get("all_rows_pass", False)) for item in earlier_summaries):
            reason_codes = [
                f"unique_first_failure:{stage_name}",
                f"stage_status:{status}",
            ]
            if provenance_blockers:
                reason_codes.append("lineage_binding_blocked")
            return {
                "audit_status": AUDIT_STATUS_COMPLETE,
                "first_failing_stage": stage_name,
                "reason_codes": reason_codes,
            }
    fail_closed_reason_codes = ["no_unique_stage_covers_full_canonical_cohort"]
    if provenance_blockers:
        fail_closed_reason_codes.append("lineage_binding_blocked")
    return {
        "audit_status": AUDIT_STATUS_FAIL_CLOSED,
        "first_failing_stage": None,
        "reason_codes": fail_closed_reason_codes,
    }


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".json":
        return None
    return dict(_read_json(path))


def _read_jsonl_with_line_numbers(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object in {path}:{line_number}, got {type(obj).__name__}"
                )
            rows.append((line_number, dict(obj)))
    return rows


def _artifact_entry(
    *,
    artifact_id: str,
    authority_role: str,
    path: Path,
    artifact_kind: str | None = None,
    must_exist: bool = True,
    read_only: bool = True,
) -> dict[str, Any]:
    resolved = path.resolve()
    payload = _load_json_if_exists(resolved)
    entry: dict[str, Any] = {
        "artifact_id": artifact_id,
        "artifact_kind": artifact_kind
        if artifact_kind is not None
        else (payload or {}).get("artifact_kind"),
        "authority_role": authority_role,
        "content_sha256": _sha256_file(resolved),
        "must_exist": bool(must_exist),
        "path_kind": "file",
        "read_only": bool(read_only),
        "relative_path": _repo_relative(resolved),
        "resolved_path": str(resolved),
    }
    if payload is not None:
        if payload.get("schema_version") is not None:
            entry["schema_version"] = payload.get("schema_version")
        if payload.get("report_signature_sha256") is not None:
            entry["report_signature_sha256"] = payload.get("report_signature_sha256")
    return entry


def _label_rows(
    *,
    labels_path: Path,
    freshness_context: Mapping[str, object],
    label_artifact_digest: str,
) -> tuple[list[CanonicalRow], dict[tuple[str, int], CanonicalRow]]:
    rows_with_line_numbers = _read_jsonl_with_line_numbers(labels_path)
    rows: list[CanonicalRow] = []
    by_episode_t: dict[tuple[str, int], CanonicalRow] = {}
    for line_number, row in rows_with_line_numbers:
        episode_id = _as_non_empty_string(
            row.get("episode_id"), field_name="episode_id"
        )
        t_value = _as_int(row.get("t"), field_name="t")
        prompt_raw = text_indicator.require_prompt_raw(
            row.get("prompt_raw"),
            field_name="prompt_raw",
        )
        indicator_mode = text_indicator.indicator_mode_from_indicator_value(
            row.get("indicator_I"),
            field_name="indicator_I",
        )
        expected_carrier = text_indicator.build_canonical_text_indicator(
            prompt_raw,
            indicator_mode,
        )
        carrier_text_v1 = _optional_string(
            row.get(text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD)
        )
        row_key = stable_row_lineage_key(
            source_artifact_id="frozen_mainline_labels",
            resolved_path=labels_path,
            content_sha256=label_artifact_digest,
            row_locator=f"line:{line_number}",
            episode_id=episode_id,
            t=t_value,
            manifest_context=freshness_context,
        )
        canonical_row = CanonicalRow(
            row_key=row_key,
            line_number=int(line_number),
            episode_id=episode_id,
            t=int(t_value),
            prompt_raw=prompt_raw,
            indicator_I=int(row["indicator_I"]),
            indicator_mode=indicator_mode,
            expected_carrier_text_v1=expected_carrier,
            carrier_text_v1=carrier_text_v1,
            source_artifact_id="frozen_mainline_labels",
        )
        pair_key = (episode_id, int(t_value))
        if pair_key in by_episode_t:
            raise ValueError(
                f"Duplicate label mapping for episode_id={episode_id} t={t_value}"
            )
        by_episode_t[pair_key] = canonical_row
        rows.append(canonical_row)
    if not rows:
        raise ValueError(f"labels file is empty: {labels_path}")
    return rows, by_episode_t


def _build_freshness_context(
    execution_freeze_contract: Mapping[str, Any],
) -> dict[str, object]:
    freshness_raw = execution_freeze_contract.get("freshness")
    freshness = dict(freshness_raw) if isinstance(freshness_raw, Mapping) else {}
    return {
        "execution_sha": execution_freeze_contract.get("execution_sha"),
        "checkpoint_id": freshness.get("checkpoint_id"),
        "manifest_hash": freshness.get("manifest_hash"),
        "seed_bundle_id": freshness.get("seed_bundle_id"),
    }


def _validate_recap_lineage_bindings(
    *,
    recap_dataset_dir: Path,
    source_dataset_ref_path: Path,
    source_dataset_ref: Mapping[str, Any],
    episodes_path: Path,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    output_dataset_dir_text = _optional_string(
        source_dataset_ref.get("output_dataset_dir")
    )
    expected_recap_dataset_dir = str(recap_dataset_dir.resolve())
    if output_dataset_dir_text is None:
        issues.append(
            {
                "code": "missing_output_dataset_backpointer",
                "artifact_id": "source_dataset_ref",
                "message": "source_dataset_ref.json does not declare output_dataset_dir",
            }
        )
    elif Path(output_dataset_dir_text).as_posix() != expected_recap_dataset_dir:
        issues.append(
            {
                "code": "output_dataset_backpointer_mismatch",
                "artifact_id": "source_dataset_ref",
                "message": (
                    "source_dataset_ref.output_dataset_dir does not match the live recap dataset dir"
                ),
                "expected": expected_recap_dataset_dir,
                "observed": output_dataset_dir_text,
            }
        )

    arrays_backpointer = _optional_string(source_dataset_ref.get("arrays_path"))
    live_arrays_path = recap_dataset_dir / "arrays"
    if arrays_backpointer is None:
        issues.append(
            {
                "code": "missing_arrays_backpointer",
                "artifact_id": "source_dataset_ref",
                "message": "source_dataset_ref.json does not declare arrays_path",
            }
        )
    else:
        if Path(arrays_backpointer).as_posix() != str(live_arrays_path.resolve()):
            issues.append(
                {
                    "code": "arrays_backpointer_mismatch",
                    "artifact_id": "source_dataset_ref",
                    "message": (
                        "source_dataset_ref.arrays_path does not bind the current recap dataset arrays path"
                    ),
                    "expected": str(live_arrays_path.resolve()),
                    "observed": arrays_backpointer,
                }
            )

    if live_arrays_path.is_symlink() and not live_arrays_path.exists():
        issues.append(
            {
                "code": "broken_arrays_symlink",
                "artifact_id": "recap_dataset_arrays",
                "message": "recap dataset arrays symlink is broken in the live workspace",
                "observed": str(live_arrays_path),
                "resolved_target": str(live_arrays_path.resolve()),
            }
        )

    source_dataset_dir_text = _optional_string(
        source_dataset_ref.get("source_dataset_dir")
    )
    if source_dataset_dir_text is None:
        issues.append(
            {
                "code": "missing_source_dataset_dir_backpointer",
                "artifact_id": "source_dataset_ref",
                "message": "source_dataset_ref.json does not declare source_dataset_dir",
            }
        )
    else:
        source_dataset_dir_path = Path(source_dataset_dir_text).expanduser()
        if not source_dataset_dir_path.exists():
            issues.append(
                {
                    "code": "source_dataset_dir_missing",
                    "artifact_id": "source_dataset_ref",
                    "message": "source_dataset_ref.source_dataset_dir does not exist in the live workspace",
                    "observed": source_dataset_dir_text,
                }
            )

    episodes = _read_jsonl(episodes_path)
    for index, episode in enumerate(episodes[:5], start=1):
        npz_path_text = _optional_string(episode.get("npz_path"))
        if npz_path_text is None:
            issues.append(
                {
                    "code": "missing_episode_npz_path",
                    "artifact_id": "recap_dataset_episodes",
                    "message": f"episodes.jsonl record #{index} is missing npz_path",
                }
            )
            break
        candidate = recap_dataset_dir / npz_path_text
        if not candidate.exists():
            issues.append(
                {
                    "code": "episode_npz_missing",
                    "artifact_id": "recap_dataset_episodes",
                    "message": (
                        "episodes.jsonl references npz files that do not exist under the live recap dataset dir"
                    ),
                    "observed": str(candidate),
                }
            )
            break
    return issues


def _evaluate_label_materialization(
    canonical_rows: Sequence[CanonicalRow],
) -> StageCoverage:
    pass_row_keys: list[str] = []
    fail_row_keys: list[str] = []
    for row in canonical_rows:
        if row.carrier_text_v1 == row.expected_carrier_text_v1:
            pass_row_keys.append(row.row_key)
        else:
            fail_row_keys.append(row.row_key)
    reasons = ["carrier_text_v1_missing_or_noncanonical_in_frozen_labels"]
    return build_stage_coverage(
        stage_name=STAGE_LABEL_MATERIALIZATION,
        canonical_row_keys=(row.row_key for row in canonical_rows),
        pass_row_keys=pass_row_keys,
        fail_row_keys=fail_row_keys,
        reasons=reasons,
        source_artifact_ids=["frozen_mainline_labels"],
    )


def _load_state_conditioned_index(
    state_conditioned_labels_path: Path,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    index: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(state_conditioned_labels_path):
        source_episode_id = _optional_string(
            row.get("source_anchor_episode_id")
        ) or _optional_string(row.get("source_episode_id"))
        source_t = row.get("source_t")
        if (
            source_episode_id is None
            or isinstance(source_t, bool)
            or not isinstance(source_t, int)
        ):
            continue
        index[(source_episode_id, int(source_t))].append(dict(row))
    return index


def _evaluate_state_conditioned_build(
    *,
    canonical_rows: Sequence[CanonicalRow],
    state_conditioned_labels_path: Path | None,
) -> StageCoverage:
    canonical_keys = [row.row_key for row in canonical_rows]
    if (
        state_conditioned_labels_path is None
        or not state_conditioned_labels_path.is_file()
    ):
        return build_stage_coverage(
            stage_name=STAGE_STATE_CONDITIONED_BUILD,
            canonical_row_keys=canonical_keys,
            unresolved_row_keys=canonical_keys,
            reasons=["state_conditioned_labels_missing"],
            source_artifact_ids=[],
        )

    state_index = _load_state_conditioned_index(state_conditioned_labels_path)
    pass_row_keys: list[str] = []
    fail_row_keys: list[str] = []
    unresolved_row_keys: list[str] = []
    for row in canonical_rows:
        matches = state_index.get((row.episode_id, row.t), [])
        if not matches:
            unresolved_row_keys.append(row.row_key)
            continue
        carriers = {
            _optional_string(
                match.get(text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD)
            )
            for match in matches
        }
        normalized_carriers = {item for item in carriers if item is not None}
        if not normalized_carriers:
            fail_row_keys.append(row.row_key)
            continue
        if normalized_carriers == {row.expected_carrier_text_v1}:
            pass_row_keys.append(row.row_key)
        else:
            fail_row_keys.append(row.row_key)
    return build_stage_coverage(
        stage_name=STAGE_STATE_CONDITIONED_BUILD,
        canonical_row_keys=canonical_keys,
        pass_row_keys=pass_row_keys,
        fail_row_keys=fail_row_keys,
        unresolved_row_keys=unresolved_row_keys,
        reasons=[
            "same_cohort_state_conditioned_mapping_via_source_anchor_episode_id_source_t"
        ],
        source_artifact_ids=["state_conditioned_labels"],
    )


def _import_pandas() -> Any:
    try:
        return importlib.import_module("pandas")
    except Exception as exc:
        raise RuntimeError(
            f"audit_carrier_lineage requires pandas at runtime: {exc}"
        ) from exc


def _export_episode_parquet_path(
    export_dataset_dir: Path,
    *,
    episode_index: int,
    chunks_size: int,
    data_path_template: str,
) -> Path:
    return export_dataset_dir / data_path_template.format(
        episode_chunk=int(episode_index) // int(chunks_size),
        episode_index=int(episode_index),
    )


def _load_tasks_by_index(tasks_path: Path) -> dict[int, str]:
    tasks_by_index: dict[int, str] = {}
    for row in _read_jsonl(tasks_path):
        task_index = _as_int(row.get("task_index"), field_name="task_index")
        task_text = _as_non_empty_string(row.get("task"), field_name="task")
        tasks_by_index[int(task_index)] = task_text
    return tasks_by_index


def _actual_export_columns(
    export_dataset_dir: Path, *, chunks_size: int, data_path_template: str
) -> set[str]:
    pd = _import_pandas()
    sample_path = _export_episode_parquet_path(
        export_dataset_dir,
        episode_index=0,
        chunks_size=chunks_size,
        data_path_template=data_path_template,
    )
    frame = pd.read_parquet(sample_path)
    return {str(column) for column in frame.columns}


def _load_export_groups(
    export_dataset_dir: Path,
) -> dict[str, Any]:
    info_path = export_dataset_dir / "meta" / lerobot_dataset_export.META_INFO_JSON
    tasks_path = export_dataset_dir / "meta" / lerobot_dataset_export.META_TASKS_JSONL
    episodes_path = (
        export_dataset_dir / "meta" / lerobot_dataset_export.META_EPISODES_JSONL
    )

    info = _read_json(info_path)
    tasks_by_index = _load_tasks_by_index(tasks_path)
    episodes_meta = _read_jsonl(episodes_path)
    data_path_template = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    )
    chunks_size = int(info.get("chunks_size", 1000))
    pd = _import_pandas()

    groups: dict[tuple[str, int], dict[str, Any]] = {}
    parquet_columns_union: set[str] = set()
    for episode_meta in episodes_meta:
        episode_index = _as_int(
            episode_meta.get("episode_index"), field_name="episode_index"
        )
        recap_episode_id = _as_non_empty_string(
            episode_meta.get("recap.episode_id"),
            field_name="recap.episode_id",
        )
        parquet_path = _export_episode_parquet_path(
            export_dataset_dir,
            episode_index=episode_index,
            chunks_size=chunks_size,
            data_path_template=data_path_template,
        )
        frame = pd.read_parquet(parquet_path)
        frame_columns = {str(column) for column in frame.columns}
        parquet_columns_union.update(frame_columns)
        if "recap_m2.t" not in frame_columns:
            raise ValueError(f"export parquet missing recap_m2.t: {parquet_path}")
        for t_value, group in frame.groupby("recap_m2.t", sort=False):
            pair_key = (recap_episode_id, int(t_value))
            if pair_key in groups:
                raise ValueError(
                    f"duplicate export row group for recap.episode_id={recap_episode_id} t={t_value}"
                )
            fallback_task_values: list[str] = []
            if PHASE1_PROMPT_FEATURE_FALLBACK_KEY in group.columns:
                fallback_indices = sorted(
                    {
                        int(value)
                        for value in group[PHASE1_PROMPT_FEATURE_FALLBACK_KEY].tolist()
                    }
                )
                fallback_task_values = [
                    tasks_by_index[int(index)]
                    for index in fallback_indices
                    if int(index) in tasks_by_index
                ]
            primary_task_values: list[str] = []
            if PHASE1_PROMPT_FEATURE_PRIMARY_KEY in group.columns:
                primary_indices = sorted(
                    {
                        int(value)
                        for value in group[PHASE1_PROMPT_FEATURE_PRIMARY_KEY].tolist()
                    }
                )
                primary_task_values = [
                    tasks_by_index[int(index)]
                    for index in primary_indices
                    if int(index) in tasks_by_index
                ]
            prompt_raw_values = sorted(
                {
                    str(value)
                    for value in group.get("recap_m2.prompt_raw", [])
                    if isinstance(value, str) and value
                }
            )
            prompt_conditioned_values = sorted(
                {
                    str(value)
                    for value in group.get("recap_m2.prompt_conditioned", [])
                    if isinstance(value, str) and value
                }
            )
            indicator_values = sorted(
                {
                    int(value)
                    for value in group.get("recap_m2.indicator_I", [])
                    if not isinstance(value, bool)
                }
            )
            groups[pair_key] = {
                "episode_index": int(episode_index),
                "parquet_path": str(parquet_path),
                "frame_count": int(len(group)),
                "primary_prompt_column_present": PHASE1_PROMPT_FEATURE_PRIMARY_KEY
                in frame_columns,
                "fallback_prompt_column_present": PHASE1_PROMPT_FEATURE_FALLBACK_KEY
                in frame_columns,
                "primary_task_text_values": primary_task_values,
                "fallback_task_text_values": fallback_task_values,
                "prompt_raw_values": prompt_raw_values,
                "prompt_conditioned_values": prompt_conditioned_values,
                "indicator_values": indicator_values,
            }
    return {
        "info": info,
        "tasks_by_index": tasks_by_index,
        "groups": groups,
        "parquet_columns_union": sorted(parquet_columns_union),
        "chunks_size": int(chunks_size),
        "data_path_template": data_path_template,
    }


def _evaluate_lerobot_export(
    *,
    canonical_rows: Sequence[CanonicalRow],
    export_bundle: Mapping[str, Any],
) -> StageCoverage:
    groups = dict(export_bundle["groups"])
    pass_row_keys: list[str] = []
    fail_row_keys: list[str] = []
    unresolved_row_keys: list[str] = []
    for row in canonical_rows:
        group = groups.get((row.episode_id, row.t))
        if group is None:
            fail_row_keys.append(row.row_key)
            continue
        prompt_raw_values = list(group.get("prompt_raw_values", []))
        indicator_values = list(group.get("indicator_values", []))
        if prompt_raw_values != [row.prompt_raw]:
            fail_row_keys.append(row.row_key)
            continue
        if indicator_values != [row.indicator_I]:
            fail_row_keys.append(row.row_key)
            continue
        if int(group.get("frame_count", 0)) <= 0:
            unresolved_row_keys.append(row.row_key)
            continue
        pass_row_keys.append(row.row_key)
    return build_stage_coverage(
        stage_name=STAGE_LEROBOT_EXPORT,
        canonical_row_keys=(row.row_key for row in canonical_rows),
        pass_row_keys=pass_row_keys,
        fail_row_keys=fail_row_keys,
        unresolved_row_keys=unresolved_row_keys,
        reasons=["export_groups_bound_via_recap_episode_id_and_recap_m2_t"],
        source_artifact_ids=[
            "export_info",
            "export_tasks",
            "export_episodes_meta",
        ],
    )


def _evaluate_post_export_schema_projection(
    *,
    canonical_rows: Sequence[CanonicalRow],
    export_bundle: Mapping[str, Any],
) -> StageCoverage:
    info = dict(export_bundle["info"])
    groups = dict(export_bundle["groups"])
    dual_task_text = bool(info.get("recap_export.dual_task_text", False))
    task_text_mode = _optional_string(info.get("task_text_mode"))
    pass_row_keys: list[str] = []
    fail_row_keys: list[str] = []
    unresolved_row_keys: list[str] = []
    for row in canonical_rows:
        group = groups.get((row.episode_id, row.t))
        if group is None:
            unresolved_row_keys.append(row.row_key)
            continue
        task_text_values = list(group.get("primary_task_text_values", [])) or list(
            group.get("fallback_task_text_values", [])
        )
        if (
            task_text_values == [row.expected_carrier_text_v1]
            and not dual_task_text
            and task_text_mode != "mix50"
        ):
            pass_row_keys.append(row.row_key)
            continue
        fail_row_keys.append(row.row_key)
    return build_stage_coverage(
        stage_name=STAGE_POST_EXPORT_SCHEMA_PROJECTION,
        canonical_row_keys=(row.row_key for row in canonical_rows),
        pass_row_keys=pass_row_keys,
        fail_row_keys=fail_row_keys,
        unresolved_row_keys=unresolved_row_keys,
        reasons=[
            "exported_task_surface_does_not_equal_canonical_carrier_text_v1",
            f"task_text_mode:{task_text_mode or 'unset'}",
            f"dual_task_text:{str(dual_task_text).lower()}",
        ],
        source_artifact_ids=["export_info", "export_tasks", "export_episodes_meta"],
    )


def _evaluate_reader_schema_mismatch(
    *,
    canonical_rows: Sequence[CanonicalRow],
    export_dataset_dir: Path,
) -> StageCoverage:
    canonical_keys = [row.row_key for row in canonical_rows]
    try:
        _ = build_phase1_dataset_mapping_spec(export_dataset_dir)
    except Exception as exc:
        return build_stage_coverage(
            stage_name=STAGE_READER_SCHEMA_MISMATCH,
            canonical_row_keys=canonical_keys,
            fail_row_keys=canonical_keys,
            reasons=[f"contract_mapping_rejected_dataset:{_exc_message(exc)}"],
            source_artifact_ids=["export_info", "export_modality"],
        )
    return build_stage_coverage(
        stage_name=STAGE_READER_SCHEMA_MISMATCH,
        canonical_row_keys=canonical_keys,
        pass_row_keys=canonical_keys,
        reasons=["phase1_dataset_mapping_spec_succeeded"],
        source_artifact_ids=["export_info", "export_modality"],
    )


def _evaluate_reader_alias_mismatch(
    *,
    canonical_rows: Sequence[CanonicalRow],
    export_dataset_dir: Path,
    export_bundle: Mapping[str, Any],
    reader_schema_coverage: StageCoverage,
) -> StageCoverage:
    canonical_keys = [row.row_key for row in canonical_rows]
    if reader_schema_coverage.fail_row_keys:
        return build_stage_coverage(
            stage_name=STAGE_READER_ALIAS_MISMATCH,
            canonical_row_keys=canonical_keys,
            unresolved_row_keys=canonical_keys,
            reasons=["reader_schema_mismatch_prevents_alias_resolution"],
            source_artifact_ids=[
                "export_info",
                "export_modality",
                "export_first_parquet",
            ],
            blocked_by_stage=STAGE_READER_SCHEMA_MISMATCH,
        )

    mapping_spec = build_phase1_dataset_mapping_spec(export_dataset_dir)
    selected_prompt_key = str(mapping_spec.source_prompt_feature_key)
    actual_columns = set(str(item) for item in export_bundle["parquet_columns_union"])
    if selected_prompt_key in actual_columns:
        return build_stage_coverage(
            stage_name=STAGE_READER_ALIAS_MISMATCH,
            canonical_row_keys=canonical_keys,
            pass_row_keys=canonical_keys,
            reasons=[f"selected_prompt_feature_present:{selected_prompt_key}"],
            source_artifact_ids=[
                "export_info",
                "export_modality",
                "export_first_parquet",
            ],
        )

    fallback_present = PHASE1_PROMPT_FEATURE_FALLBACK_KEY in actual_columns
    if fallback_present:
        return build_stage_coverage(
            stage_name=STAGE_READER_ALIAS_MISMATCH,
            canonical_row_keys=canonical_keys,
            fail_row_keys=canonical_keys,
            reasons=[
                f"selected_prompt_feature_missing:{selected_prompt_key}",
                f"fallback_prompt_feature_present:{PHASE1_PROMPT_FEATURE_FALLBACK_KEY}",
            ],
            source_artifact_ids=[
                "export_info",
                "export_modality",
                "export_first_parquet",
            ],
        )
    return build_stage_coverage(
        stage_name=STAGE_READER_ALIAS_MISMATCH,
        canonical_row_keys=canonical_keys,
        unresolved_row_keys=canonical_keys,
        reasons=[f"selected_prompt_feature_missing:{selected_prompt_key}"],
        source_artifact_ids=["export_info", "export_modality", "export_first_parquet"],
    )


def _select_trace_rows(
    canonical_rows: Sequence[CanonicalRow], *, limit: int
) -> list[CanonicalRow]:
    if limit <= 0:
        raise ValueError(f"trace_count must be positive, got {limit!r}")
    negatives = [
        row
        for row in canonical_rows
        if row.indicator_mode == text_indicator.TEXT_INDICATOR_NEGATIVE
    ]
    positives = [
        row
        for row in canonical_rows
        if row.indicator_mode == text_indicator.TEXT_INDICATOR_POSITIVE
    ]
    selected: list[CanonicalRow] = []
    if negatives and positives:
        neg_target = max(1, limit // 2)
        pos_target = max(1, limit - neg_target)
        selected.extend(negatives[:neg_target])
        selected.extend(positives[:pos_target])
    else:
        selected.extend(canonical_rows[:limit])
    seen: set[str] = set()
    deduped: list[CanonicalRow] = []
    for row in selected + list(canonical_rows):
        if row.row_key in seen:
            continue
        seen.add(row.row_key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def _row_trace_stage_entry(
    *,
    row: CanonicalRow,
    stage_name: str,
    export_bundle: Mapping[str, Any] | None,
    export_dataset_dir: Path,
    state_conditioned_labels_path: Path | None,
    stage_summaries_by_name: Mapping[str, Mapping[str, Any]],
    reader_schema_coverage: StageCoverage,
) -> dict[str, Any]:
    stage_summary = dict(stage_summaries_by_name[stage_name])
    status = str(stage_summary["status"])
    entry: dict[str, Any] = {
        "status": status,
        "blocked_by_stage": stage_summary.get("blocked_by_stage"),
        "reason_codes": list(stage_summary.get("reason_codes", [])),
    }
    if stage_name == STAGE_LABEL_MATERIALIZATION:
        entry["evidence"] = {
            "line_number": int(row.line_number),
            "carrier_text_v1": row.carrier_text_v1,
            "expected_carrier_text_v1": row.expected_carrier_text_v1,
        }
        return entry
    if stage_name == STAGE_STATE_CONDITIONED_BUILD:
        entry["evidence"] = {
            "state_conditioned_labels_path": (
                str(state_conditioned_labels_path.resolve())
                if state_conditioned_labels_path is not None
                and state_conditioned_labels_path.exists()
                else None
            ),
            "join_key": {
                "source_anchor_episode_id": row.episode_id,
                "source_t": int(row.t),
            },
        }
        return entry
    if export_bundle is None:
        entry["evidence"] = {"available": False}
        return entry
    export_group = dict(export_bundle["groups"].get((row.episode_id, row.t), {}))
    if stage_name == STAGE_LEROBOT_EXPORT:
        entry["evidence"] = export_group
        return entry
    if stage_name == STAGE_POST_EXPORT_SCHEMA_PROJECTION:
        entry["evidence"] = {
            "task_text_mode": export_bundle["info"].get("task_text_mode"),
            "dual_task_text": export_bundle["info"].get("recap_export.dual_task_text"),
            "task_text_values": export_group.get("primary_task_text_values")
            or export_group.get("fallback_task_text_values"),
            "expected_carrier_text_v1": row.expected_carrier_text_v1,
        }
        return entry
    if stage_name == STAGE_READER_SCHEMA_MISMATCH:
        entry["evidence"] = {
            "features_prompt_keys": sorted(
                key
                for key in export_bundle["info"].get("features", {})
                if key
                in {
                    PHASE1_PROMPT_FEATURE_PRIMARY_KEY,
                    PHASE1_PROMPT_FEATURE_FALLBACK_KEY,
                }
            ),
            "parquet_columns_union": list(export_bundle["parquet_columns_union"]),
        }
        return entry
    if stage_name == STAGE_READER_ALIAS_MISMATCH:
        if reader_schema_coverage.fail_row_keys:
            entry["evidence"] = {
                "available": False,
                "blocked_by_stage": STAGE_READER_SCHEMA_MISMATCH,
            }
            return entry
        mapping_spec = build_phase1_dataset_mapping_spec(export_dataset_dir)
        entry["evidence"] = {
            "selected_prompt_feature_key": mapping_spec.source_prompt_feature_key,
            "primary_present": PHASE1_PROMPT_FEATURE_PRIMARY_KEY
            in export_bundle["parquet_columns_union"],
            "fallback_present": PHASE1_PROMPT_FEATURE_FALLBACK_KEY
            in export_bundle["parquet_columns_union"],
        }
        return entry
    return entry


def _missing_examples_rows(
    *,
    canonical_rows: Sequence[CanonicalRow],
    export_bundle: Mapping[str, Any] | None,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for canonical_row in canonical_rows:
        if canonical_row.carrier_text_v1 == canonical_row.expected_carrier_text_v1:
            continue
        export_group = (
            dict(
                export_bundle["groups"].get(
                    (canonical_row.episode_id, canonical_row.t), {}
                )
            )
            if export_bundle is not None
            else {}
        )
        rows.append(
            {
                "row_key": canonical_row.row_key,
                "line_number": int(canonical_row.line_number),
                "episode_id": canonical_row.episode_id,
                "t": int(canonical_row.t),
                "indicator_mode": canonical_row.indicator_mode,
                "indicator_I": int(canonical_row.indicator_I),
                "prompt_raw": canonical_row.prompt_raw,
                "carrier_text_v1": canonical_row.carrier_text_v1 or "",
                "expected_carrier_text_v1": canonical_row.expected_carrier_text_v1,
                "export_task_text_values": " | ".join(
                    export_group.get("primary_task_text_values", [])
                    or export_group.get("fallback_task_text_values", [])
                ),
                "export_frame_count": int(export_group.get("frame_count", 0)),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _build_markdown(
    *,
    payload: Mapping[str, Any],
    trace_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines: list[str] = [
        "# carrier lineage 审计",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- first_failing_stage: `{payload['first_failing_stage']}`",
        f"- canonical_row_count: {payload['canonical_cohort']['row_count']}",
        f"- provenance_blocker_count: {payload['provenance']['blocker_count']}",
        "",
        "## 最终症状（与 first failure 分离）",
        "",
        f"- frozen labels path: `{payload['inputs']['frozen_labels_path']}`",
        f"- rows missing `carrier_text_v1`: {payload['final_symptom']['missing_carrier_row_count']}",
        f"- rows with `prompt_raw + indicator_I`: {payload['final_symptom']['rows_with_prompt_and_indicator_count']}",
        "",
        "## Stage verdicts",
        "",
        "| stage | status | pass_rows | fail_rows | unresolved_rows | reasons |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for stage in payload["stage_verdicts"]:
        reasons = "<br>".join(str(item) for item in stage.get("reason_codes", []))
        lines.append(
            "| "
            + f"{stage['stage_name']} | {stage['status']} | {stage['pass_row_count']} | {stage['fail_row_count']} | {stage['unresolved_row_count']} | {reasons} |"
        )

    lines.extend(["", "## Provenance blockers", ""])
    blockers = payload["provenance"].get("blockers", [])
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{blocker['code']}`: {blocker['message']}")
    else:
        lines.append("- 无")

    lines.extend(["", "## Row trace samples", ""])
    lines.append(
        "| trace_id | indicator_mode | episode_id | t | label_stage | projection_stage | |"
    )
    lines.append("| --- | --- | --- | ---: | --- | --- | --- |")
    for trace in trace_rows:
        stages = trace["stages"]
        lines.append(
            "| "
            + f"{trace['trace_id']} | {trace['canonical_row']['indicator_mode']} | {trace['canonical_row']['episode_id']} | {trace['canonical_row']['t']} | {stages[STAGE_LABEL_MATERIALIZATION]['status']} | {stages[STAGE_POST_EXPORT_SCHEMA_PROJECTION]['status']} | {trace['trace_path']} |"
        )
    return "\n".join(lines)


def materialize_carrier_lineage_audit(
    *,
    labels_path: Path,
    recap_dataset_dir: Path,
    export_dataset_dir: Path,
    state_conditioned_labels_path: Path | None,
    execution_freeze_contract_path: Path,
    carrier_parity_report_path: Path,
    uplift_verdict_path: Path,
    output_dir: Path,
    trace_count: int,
    missing_example_limit: int,
) -> dict[str, Any]:
    labels_path = _validate_existing_file(labels_path, arg_name="labels")
    recap_dataset_dir = _validate_existing_dir(
        recap_dataset_dir, arg_name="recap_dataset_dir"
    )
    export_dataset_dir = _validate_existing_dir(
        export_dataset_dir, arg_name="export_dataset_dir"
    )
    execution_freeze_contract_path = _validate_existing_file(
        execution_freeze_contract_path,
        arg_name="execution_freeze_contract",
    )
    carrier_parity_report_path = _validate_existing_file(
        carrier_parity_report_path,
        arg_name="carrier_parity_report",
    )
    uplift_verdict_path = _validate_existing_file(
        uplift_verdict_path,
        arg_name="uplift_verdict",
    )
    state_conditioned_labels_path = (
        state_conditioned_labels_path.resolve()
        if state_conditioned_labels_path is not None
        and state_conditioned_labels_path.exists()
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / TRACE_DIRNAME
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    execution_freeze_contract = _read_json(execution_freeze_contract_path)
    carrier_parity_report = _read_json(carrier_parity_report_path)
    uplift_verdict = _read_json(uplift_verdict_path)
    freshness_context = _build_freshness_context(execution_freeze_contract)

    source_artifacts = [
        _artifact_entry(
            artifact_id="execution_freeze_contract",
            authority_role="execution_freeze_authority",
            path=execution_freeze_contract_path,
        ),
        _artifact_entry(
            artifact_id="carrier_parity_report",
            authority_role="formal_block_report",
            path=carrier_parity_report_path,
        ),
        _artifact_entry(
            artifact_id="uplift_verdict",
            authority_role="blocked_closeout_authority",
            path=uplift_verdict_path,
        ),
        _artifact_entry(
            artifact_id="frozen_mainline_labels",
            authority_role="frozen_mainline_dataset",
            path=labels_path,
            artifact_kind="recap_m2_labels",
        ),
    ]

    recap_episodes_path = _validate_existing_file(
        recap_dataset_dir / "episodes.jsonl", arg_name="recap_dataset_episodes"
    )
    source_dataset_ref_path = _validate_existing_file(
        recap_dataset_dir / "source_dataset_ref.json", arg_name="source_dataset_ref"
    )
    continuous_contract_path = _validate_existing_file(
        recap_dataset_dir / "continuous_advantage_contract.json",
        arg_name="continuous_advantage_contract",
    )
    export_info_path = _validate_existing_file(
        export_dataset_dir / "meta" / "info.json", arg_name="export_info"
    )
    export_modality_path = _validate_existing_file(
        export_dataset_dir / "meta" / "modality.json", arg_name="export_modality"
    )
    export_tasks_path = _validate_existing_file(
        export_dataset_dir / "meta" / "tasks.jsonl", arg_name="export_tasks"
    )
    export_episodes_meta_path = _validate_existing_file(
        export_dataset_dir / "meta" / "episodes.jsonl",
        arg_name="export_episodes_meta",
    )
    export_first_parquet_path = _validate_existing_file(
        export_dataset_dir / "data" / "chunk-000" / "episode_000000.parquet",
        arg_name="export_first_parquet",
    )

    source_artifacts.extend(
        [
            _artifact_entry(
                artifact_id="recap_dataset_episodes",
                authority_role="recap_dataset_episode_index",
                path=recap_episodes_path,
            ),
            _artifact_entry(
                artifact_id="source_dataset_ref",
                authority_role="recap_source_backpointer",
                path=source_dataset_ref_path,
            ),
            _artifact_entry(
                artifact_id="continuous_advantage_contract",
                authority_role="advantage_contract",
                path=continuous_contract_path,
            ),
            _artifact_entry(
                artifact_id="export_info",
                authority_role="export_meta_info",
                path=export_info_path,
            ),
            _artifact_entry(
                artifact_id="export_modality",
                authority_role="export_modality_contract",
                path=export_modality_path,
            ),
            _artifact_entry(
                artifact_id="export_tasks",
                authority_role="export_task_index",
                path=export_tasks_path,
            ),
            _artifact_entry(
                artifact_id="export_episodes_meta",
                authority_role="export_episode_backpointer",
                path=export_episodes_meta_path,
            ),
            _artifact_entry(
                artifact_id="export_first_parquet",
                authority_role="export_row_surface_sample",
                path=export_first_parquet_path,
            ),
        ]
    )
    if (
        state_conditioned_labels_path is not None
        and state_conditioned_labels_path.is_file()
    ):
        source_artifacts.append(
            _artifact_entry(
                artifact_id="state_conditioned_labels",
                authority_role="state_conditioned_reference_lane",
                path=state_conditioned_labels_path,
            )
        )

    source_dataset_ref = _read_json(source_dataset_ref_path)
    provenance_blockers = _validate_recap_lineage_bindings(
        recap_dataset_dir=recap_dataset_dir,
        source_dataset_ref_path=source_dataset_ref_path,
        source_dataset_ref=source_dataset_ref,
        episodes_path=recap_episodes_path,
    )

    label_artifact_digest = _sha256_file(labels_path)
    canonical_rows, canonical_by_episode_t = _label_rows(
        labels_path=labels_path,
        freshness_context=freshness_context,
        label_artifact_digest=label_artifact_digest,
    )
    del canonical_by_episode_t

    export_bundle = _load_export_groups(export_dataset_dir)
    label_materialization = _evaluate_label_materialization(canonical_rows)
    state_conditioned_coverage = _evaluate_state_conditioned_build(
        canonical_rows=canonical_rows,
        state_conditioned_labels_path=state_conditioned_labels_path,
    )
    export_coverage = _evaluate_lerobot_export(
        canonical_rows=canonical_rows,
        export_bundle=export_bundle,
    )
    projection_coverage = _evaluate_post_export_schema_projection(
        canonical_rows=canonical_rows,
        export_bundle=export_bundle,
    )
    reader_schema_coverage = _evaluate_reader_schema_mismatch(
        canonical_rows=canonical_rows,
        export_dataset_dir=export_dataset_dir,
    )
    reader_alias_coverage = _evaluate_reader_alias_mismatch(
        canonical_rows=canonical_rows,
        export_dataset_dir=export_dataset_dir,
        export_bundle=export_bundle,
        reader_schema_coverage=reader_schema_coverage,
    )

    stage_coverages = [
        label_materialization,
        state_conditioned_coverage,
        export_coverage,
        projection_coverage,
        reader_schema_coverage,
        reader_alias_coverage,
    ]
    stage_summaries = [
        summarize_stage_coverage(
            coverage,
            canonical_row_keys=(row.row_key for row in canonical_rows),
        )
        for coverage in stage_coverages
    ]
    audit_outcome = resolve_audit_outcome(
        stage_summaries,
        canonical_row_keys=(row.row_key for row in canonical_rows),
        provenance_blockers=provenance_blockers,
    )
    stage_summaries_by_name = {
        str(summary["stage_name"]): summary for summary in stage_summaries
    }

    trace_rows_payload: list[dict[str, Any]] = []
    for trace_index, canonical_row in enumerate(
        _select_trace_rows(canonical_rows, limit=trace_count),
        start=1,
    ):
        trace_id = f"row_{trace_index:04d}"
        trace_payload = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_id": trace_id,
            "row_key": canonical_row.row_key,
            "canonical_row": {
                "line_number": int(canonical_row.line_number),
                "episode_id": canonical_row.episode_id,
                "t": int(canonical_row.t),
                "indicator_mode": canonical_row.indicator_mode,
                "indicator_I": int(canonical_row.indicator_I),
                "prompt_raw": canonical_row.prompt_raw,
                "carrier_text_v1": canonical_row.carrier_text_v1,
                "expected_carrier_text_v1": canonical_row.expected_carrier_text_v1,
            },
            "stages": {
                stage_name: _row_trace_stage_entry(
                    row=canonical_row,
                    stage_name=stage_name,
                    export_bundle=export_bundle,
                    export_dataset_dir=export_dataset_dir,
                    state_conditioned_labels_path=state_conditioned_labels_path,
                    stage_summaries_by_name=stage_summaries_by_name,
                    reader_schema_coverage=reader_schema_coverage,
                )
                for stage_name in STAGE_ORDER
            },
            "source_artifact_ids": [
                "frozen_mainline_labels",
                "export_info",
                "export_tasks",
                "export_episodes_meta",
            ],
            "generated_at": _utc_now(),
        }
        trace_path = trace_dir / f"{trace_id}.json"
        _write_json_atomic(trace_path, trace_payload)
        trace_rows_payload.append(
            {
                "trace_id": trace_id,
                "canonical_row": trace_payload["canonical_row"],
                "stages": trace_payload["stages"],
                "trace_path": _repo_relative(trace_path),
            }
        )

    missing_examples = _missing_examples_rows(
        canonical_rows=canonical_rows,
        export_bundle=export_bundle,
        limit=missing_example_limit,
    )

    final_symptom = {
        "frozen_labels_path": str(labels_path),
        "missing_carrier_row_count": int(
            sum(1 for row in canonical_rows if row.carrier_text_v1 is None)
        ),
        "rows_with_prompt_and_indicator_count": int(len(canonical_rows)),
        "expected_carrier_field": text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD,
        "must_not_be_used_as_first_failure_alone": True,
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": AUDIT_ARTIFACT_KIND,
        "generated_at": _utc_now(),
        "audit_status": audit_outcome["audit_status"],
        "first_failing_stage": audit_outcome["first_failing_stage"],
        "reason_codes": list(audit_outcome["reason_codes"]),
        "stage_order": list(STAGE_ORDER),
        "inputs": {
            "frozen_labels_path": str(labels_path),
            "recap_dataset_dir": str(recap_dataset_dir),
            "export_dataset_dir": str(export_dataset_dir),
            "state_conditioned_labels_path": (
                str(state_conditioned_labels_path)
                if state_conditioned_labels_path is not None
                else None
            ),
            "execution_freeze_contract_path": str(execution_freeze_contract_path),
            "carrier_parity_report_path": str(carrier_parity_report_path),
            "uplift_verdict_path": str(uplift_verdict_path),
        },
        "canonical_cohort": {
            "row_count": int(len(canonical_rows)),
            "row_key_digest": _ordered_digest(row.row_key for row in canonical_rows),
            "source_artifact_id": "frozen_mainline_labels",
        },
        "freshness": dict(execution_freeze_contract.get("freshness", {})),
        "final_symptom": final_symptom,
        "provenance": {
            "status": "blocked" if provenance_blockers else "passed",
            "blocker_count": int(len(provenance_blockers)),
            "blockers": provenance_blockers,
        },
        "stage_verdicts": stage_summaries,
        "source_artifacts": source_artifacts,
        "upstream_context": {
            "carrier_parity_report": {
                "authority_violation_count": carrier_parity_report.get(
                    "authority_violation_count"
                ),
                "full_scan_row_count": carrier_parity_report.get("full_scan_row_count"),
            },
            "uplift_verdict": {
                "status": uplift_verdict.get("status"),
                "block_reason": uplift_verdict.get("block_reason"),
                "terminal_state": uplift_verdict.get("terminal_state"),
            },
        },
        "artifacts": {
            "audit_json": _repo_relative(output_dir / AUDIT_JSON_NAME),
            "audit_markdown": _repo_relative(output_dir / AUDIT_MD_NAME),
            "missing_examples_csv": _repo_relative(
                output_dir / MISSING_EXAMPLES_CSV_NAME
            ),
            "trace_dir": _repo_relative(trace_dir),
            "trace_count": int(len(trace_rows_payload)),
        },
    }

    audit_json_path = output_dir / AUDIT_JSON_NAME
    audit_md_path = output_dir / AUDIT_MD_NAME
    missing_examples_path = output_dir / MISSING_EXAMPLES_CSV_NAME
    _write_json_atomic(audit_json_path, payload)
    _write_text_atomic(
        audit_md_path,
        _build_markdown(payload=payload, trace_rows=trace_rows_payload),
    )
    _write_csv_atomic(
        missing_examples_path,
        fieldnames=(
            "row_key",
            "line_number",
            "episode_id",
            "t",
            "indicator_mode",
            "indicator_I",
            "prompt_raw",
            "carrier_text_v1",
            "expected_carrier_text_v1",
            "export_task_text_values",
            "export_frame_count",
        ),
        rows=missing_examples,
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        materialize_carrier_lineage_audit(
            labels_path=args.labels,
            recap_dataset_dir=args.recap_dataset_dir,
            export_dataset_dir=args.export_dataset_dir,
            state_conditioned_labels_path=args.state_conditioned_labels,
            execution_freeze_contract_path=args.execution_freeze_contract,
            carrier_parity_report_path=args.carrier_parity_report,
            uplift_verdict_path=args.uplift_verdict,
            output_dir=args.output_dir,
            trace_count=int(args.trace_count),
            missing_example_limit=int(args.missing_example_limit),
        )
    except Exception as exc:
        print(_exc_message(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
