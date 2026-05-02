from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

DEFAULT_METADATA = Path("agent/artifacts/example_run_metadata.json")
DEFAULT_OUTPUT_DIR = Path("agent/artifacts/gr00t_checkpoint_provenance")

CHECKPOINT_PROVENANCE_REPORT_JSON_NAME = "checkpoint_provenance_report.json"
FAILURE_NOTE_MARKDOWN_NAME = "checkpoint_provenance_failure_note.md"

REPORT_SCHEMA_VERSION = "gr00t_checkpoint_provenance_report_v1"
REPORT_ARTIFACT_KIND = "gr00t_checkpoint_provenance_report"
CHECKPOINT_PROVENANCE_GATE_NAME = "GR00TCheckpointProvenanceGate"
WRONG_CHECKPOINT_REASON_CODE = "wrong_checkpoint_or_missing_finetune_artifact"
OK_REASON_CODE = "ok"

SELECTED_CHECKPOINT_CONTRACT_PATH = (
    "comparable_run_spec.checkpoint_rule.selected_checkpoint_path"
)
BASE_MODEL_CANDIDATE_PATHS: tuple[str, ...] = (
    "comparable_run_spec.stable_base.base_model",
    "stable_base.base_model",
    "base_model_path",
)
EVAL_USES_FINETUNED_CANDIDATE_PATHS: tuple[str, ...] = (
    "evaluation_binding.eval_uses_finetuned",
    "eval_uses_finetuned",
)
SERVER_LOAD_PATH_CANDIDATE_PATHS: tuple[str, ...] = (
    "evaluation_binding.server_load_path",
    "server_load_path",
    "server_model_path",
    "model_path",
    "eval_policy_path",
)
SERVER_LOAD_MODE_CANDIDATE_PATHS: tuple[str, ...] = (
    "evaluation_binding.server_load_mode",
    "server_load_mode",
)
OOM_TEXT_CANDIDATE_PATHS: tuple[str, ...] = (
    "finetune_failure_reason",
    "training_failure_reason",
    "failure_reason",
    "delegate_summary.error",
)

MISSING = object()


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap import state_conditioned_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_checkpoint_provenance_gate.py",
        description=(
            "Validate that a formal finetuned evaluation is bound to exactly one loadable "
            "selected checkpoint rather than silently falling back to the base model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=(
            "Run metadata JSON carrying comparable_run_spec.checkpoint_rule.selected_checkpoint_path "
            "and finetuned-eval binding fields."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory that receives checkpoint_provenance_report.json and, on BLOCK, a "
            "machine-readable failure note markdown."
        ),
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _deep_get(payload: Mapping[str, Any], field_path: str) -> object:
    current: object = payload
    for key in field_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return MISSING
        current = current[key]
    return current


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"metadata path does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return dict(_as_mapping(payload, field_name="metadata"))


def _resolve_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


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
    return _sha256_bytes(path.read_bytes())


def _stable_signature(payload: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def _first_string(
    payload: Mapping[str, Any],
    *,
    candidate_paths: Sequence[str],
) -> tuple[str | None, str | None]:
    for field_path in candidate_paths:
        value = _deep_get(payload, field_path)
        if value is MISSING or value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip(), str(field_path)
    return None, None


def _normalize_bool(value: object) -> bool | None:
    if value is None or value is MISSING:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _first_bool(
    payload: Mapping[str, Any],
    *,
    candidate_paths: Sequence[str],
) -> tuple[bool | None, str | None]:
    for field_path in candidate_paths:
        value = _deep_get(payload, field_path)
        normalized = _normalize_bool(value)
        if normalized is not None:
            return normalized, str(field_path)
    return None, None


def _is_oom_contamination(metadata: Mapping[str, Any]) -> bool:
    failure_texts: list[str] = []
    for field_path in OOM_TEXT_CANDIDATE_PATHS:
        value = _deep_get(metadata, field_path)
        if isinstance(value, str) and value.strip():
            failure_texts.append(value.strip().lower())
    combined = "\n".join(failure_texts)
    if "outofmemory" in combined or "out of memory" in combined or "oom" in combined:
        return True
    finetune_rc = _deep_get(metadata, "finetune_returncode")
    if isinstance(finetune_rc, int) and finetune_rc != 0 and combined:
        return True
    delegate_rc = _deep_get(metadata, "delegate_summary.upstream_returncode")
    if isinstance(delegate_rc, int) and delegate_rc != 0 and combined:
        return True
    return False


def _inspect_checkpoint_path(
    repo_root: Path,
    raw_path: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "raw_path": raw_path,
        "resolved_input_path": None,
        "normalized_checkpoint_path": None,
        "exists": False,
        "selected_checkpoint_asset_path": None,
        "selected_checkpoint_asset_name": None,
        "loadable": False,
        "error": None,
    }
    if raw_path is None or not str(raw_path).strip():
        result["error"] = "missing selected checkpoint path"
        return result

    resolved_input = _resolve_path(repo_root, raw_path)
    result["resolved_input_path"] = str(resolved_input)
    if resolved_input.is_dir():
        checkpoint_dir = resolved_input
    elif resolved_input.is_file():
        checkpoint_dir = resolved_input.parent
    else:
        result["error"] = f"checkpoint path does not exist: {resolved_input}"
        return result

    result["exists"] = True
    result["normalized_checkpoint_path"] = str(checkpoint_dir)
    selected_asset = state_conditioned_train._selected_checkpoint_asset(checkpoint_dir)
    if selected_asset is None:
        result["error"] = (
            "checkpoint directory does not contain a retained checkpoint asset"
        )
        return result

    result["selected_checkpoint_asset_path"] = str(selected_asset.resolve())
    result["selected_checkpoint_asset_name"] = selected_asset.name
    result["loadable"] = True
    return result


def _normalize_server_load_binding(
    *,
    repo_root: Path,
    declared_server_load_path: str | None,
    base_model_path: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "server_load_path": None,
        "server_load_path_raw": declared_server_load_path,
        "server_load_path_is_local_checkpoint": False,
        "server_load_path_loadable": False,
        "server_load_path_error": None,
    }
    if declared_server_load_path is None:
        return result
    if base_model_path and str(declared_server_load_path) == str(base_model_path):
        result["server_load_path"] = str(declared_server_load_path)
        return result

    inspected = _inspect_checkpoint_path(repo_root, declared_server_load_path)
    result["server_load_path"] = (
        inspected["normalized_checkpoint_path"]
        or inspected["resolved_input_path"]
        or declared_server_load_path
    )
    result["server_load_path_is_local_checkpoint"] = bool(
        inspected["normalized_checkpoint_path"] is not None
    )
    result["server_load_path_loadable"] = bool(inspected["loadable"])
    result["server_load_path_error"] = inspected["error"]
    return result


def _build_checksum_or_signature(
    *,
    report_core: Mapping[str, Any],
    selected_checkpoint_asset_path: str | None,
) -> str:
    if selected_checkpoint_asset_path:
        return f"sha256:{_sha256_file(Path(selected_checkpoint_asset_path))}"
    return f"signature:{_stable_signature(report_core)}"


def build_checkpoint_provenance_report(
    *,
    metadata: Mapping[str, Any],
    metadata_path: Path,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    selected_raw = _deep_get(metadata, SELECTED_CHECKPOINT_CONTRACT_PATH)
    selected_checkpoint_raw = (
        None if selected_raw in (MISSING, None) else str(selected_raw).strip() or None
    )
    selected_checkpoint = _inspect_checkpoint_path(repo_root, selected_checkpoint_raw)

    base_model_path, base_model_source = _first_string(
        metadata,
        candidate_paths=BASE_MODEL_CANDIDATE_PATHS,
    )
    eval_uses_finetuned, eval_uses_finetuned_source = _first_bool(
        metadata,
        candidate_paths=EVAL_USES_FINETUNED_CANDIDATE_PATHS,
    )
    declared_server_load_path, declared_server_load_path_source = _first_string(
        metadata,
        candidate_paths=SERVER_LOAD_PATH_CANDIDATE_PATHS,
    )
    server_load_mode, server_load_mode_source = _first_string(
        metadata,
        candidate_paths=SERVER_LOAD_MODE_CANDIDATE_PATHS,
    )

    normalized_server_binding = _normalize_server_load_binding(
        repo_root=repo_root,
        declared_server_load_path=declared_server_load_path,
        base_model_path=base_model_path,
    )
    server_load_path = (
        str(normalized_server_binding["server_load_path"])
        if normalized_server_binding["server_load_path"] is not None
        else selected_checkpoint["normalized_checkpoint_path"]
        if selected_checkpoint["normalized_checkpoint_path"] is not None
        else None
    )
    server_load_path_source = (
        declared_server_load_path_source
        if declared_server_load_path_source is not None
        else SELECTED_CHECKPOINT_CONTRACT_PATH
        if server_load_path is not None
        else None
    )

    selected_checkpoint_path = selected_checkpoint["normalized_checkpoint_path"]
    server_load_path_matches_selected = bool(
        server_load_path is not None
        and selected_checkpoint_path is not None
        and str(server_load_path) == str(selected_checkpoint_path)
    )
    explicit_server_load_declared = declared_server_load_path is not None
    explicit_wrong_checkpoint = bool(
        explicit_server_load_declared
        and selected_checkpoint_path is not None
        and not server_load_path_matches_selected
    )
    explicit_base_fallback = bool(
        base_model_path
        and server_load_path is not None
        and str(server_load_path) == str(base_model_path)
    )
    explicit_server_load_not_loadable = bool(
        explicit_server_load_declared
        and not explicit_base_fallback
        and normalized_server_binding["server_load_path_is_local_checkpoint"]
        and not normalized_server_binding["server_load_path_loadable"]
    )
    eval_implies_base_fallback = bool(eval_uses_finetuned is False)
    is_base_fallback = bool(explicit_base_fallback or eval_implies_base_fallback)

    gate_reasons: list[str] = []
    if selected_checkpoint_raw is None:
        gate_reasons.append("selected_checkpoint_path_missing")
    if eval_uses_finetuned is not True:
        gate_reasons.append(f"eval_uses_finetuned={eval_uses_finetuned!r}")
    if selected_checkpoint["exists"] is False and selected_checkpoint_raw is not None:
        gate_reasons.append("selected_checkpoint_not_found")
    if selected_checkpoint_raw is not None and not bool(
        selected_checkpoint["loadable"]
    ):
        gate_reasons.append(
            "selected_checkpoint_not_loadable: " + str(selected_checkpoint["error"])
        )
    if explicit_base_fallback:
        gate_reasons.append("server_load_path_resolves_to_base_model")
    if explicit_wrong_checkpoint:
        gate_reasons.append("server_load_path_mismatches_selected_checkpoint")
    if explicit_server_load_not_loadable:
        gate_reasons.append(
            "server_load_path_not_loadable: "
            + str(normalized_server_binding["server_load_path_error"])
        )
    if server_load_mode is not None and str(server_load_mode) != "model_path":
        gate_reasons.append(f"server_load_mode={server_load_mode!r}")

    if selected_checkpoint_raw is None:
        loadability_status = "BLOCKED_SELECTED_CHECKPOINT_MISSING"
    elif selected_checkpoint["exists"] is False:
        loadability_status = "BLOCKED_SELECTED_CHECKPOINT_NOT_FOUND"
    elif selected_checkpoint["loadable"] is not True:
        loadability_status = "BLOCKED_SELECTED_CHECKPOINT_ASSET_MISSING"
    elif eval_uses_finetuned is not True:
        loadability_status = "BLOCKED_EVAL_NOT_FINETUNED"
    elif explicit_base_fallback:
        loadability_status = "BLOCKED_BASE_FALLBACK"
    elif explicit_wrong_checkpoint:
        loadability_status = "BLOCKED_SERVER_LOAD_PATH_MISMATCH"
    elif explicit_server_load_not_loadable:
        loadability_status = "BLOCKED_SERVER_LOAD_PATH_NOT_LOADABLE"
    elif server_load_mode is not None and str(server_load_mode) != "model_path":
        loadability_status = "BLOCKED_SERVER_LOAD_MODE_NOT_MODEL_PATH"
    else:
        loadability_status = "LOADABLE_CHECKPOINT_CONFIRMED"

    formal_eligibility = "ALLOW" if not gate_reasons else "BLOCK"
    historical_oom_contamination_pattern = bool(
        _is_oom_contamination(metadata)
        and selected_checkpoint_raw is None
        and eval_uses_finetuned is False
    )
    report_core = {
        "selected_checkpoint_path": selected_checkpoint_path,
        "base_model_path": base_model_path,
        "server_load_path": server_load_path,
        "is_base_fallback": bool(is_base_fallback),
        "loadability_status": loadability_status,
        "formal_eligibility": formal_eligibility,
        "reason_code": OK_REASON_CODE
        if not gate_reasons
        else WRONG_CHECKPOINT_REASON_CODE,
        "gate_reasons": list(gate_reasons),
    }
    checksum_or_signature = _build_checksum_or_signature(
        report_core=report_core,
        selected_checkpoint_asset_path=selected_checkpoint[
            "selected_checkpoint_asset_path"
        ],
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "gate_name": CHECKPOINT_PROVENANCE_GATE_NAME,
        "metadata_path": str(metadata_path),
        "artifact_path": str(output_dir / CHECKPOINT_PROVENANCE_REPORT_JSON_NAME),
        "status": "PASS" if formal_eligibility == "ALLOW" else "FAIL",
        "reason_code": report_core["reason_code"],
        "formal_eligibility": formal_eligibility,
        "selected_checkpoint_path": selected_checkpoint_path,
        "base_model_path": base_model_path,
        "server_load_path": server_load_path,
        "is_base_fallback": bool(is_base_fallback),
        "loadability_status": loadability_status,
        "checksum_or_signature": checksum_or_signature,
        "gate_reasons": list(gate_reasons),
        "selected_checkpoint_metadata": {
            "contract_path": SELECTED_CHECKPOINT_CONTRACT_PATH,
            "raw_selected_checkpoint_path": selected_checkpoint_raw,
            "selected_checkpoint_exists": bool(selected_checkpoint["exists"]),
            "selected_checkpoint_asset_path": selected_checkpoint[
                "selected_checkpoint_asset_path"
            ],
            "selected_checkpoint_asset_name": selected_checkpoint[
                "selected_checkpoint_asset_name"
            ],
            "selected_checkpoint_loadable": bool(selected_checkpoint["loadable"]),
            "selected_checkpoint_error": selected_checkpoint["error"],
        },
        "server_binding": {
            "server_load_path_source": server_load_path_source,
            "server_load_mode": server_load_mode or "model_path",
            "server_load_mode_source": server_load_mode_source,
            "declared_server_load_path": declared_server_load_path,
            "declared_server_load_path_source": declared_server_load_path_source,
            "declared_server_load_path_error": normalized_server_binding[
                "server_load_path_error"
            ],
            "declared_server_load_path_loadable": normalized_server_binding[
                "server_load_path_loadable"
            ],
            "server_load_path_matches_selected_checkpoint": bool(
                server_load_path_matches_selected
            ),
            "server_uses_model_path_branch": bool(
                (server_load_mode or "model_path") == "model_path"
                and server_load_path is not None
            ),
        },
        "base_model_binding": {
            "base_model_path_source": base_model_source,
            "eval_uses_finetuned": eval_uses_finetuned,
            "eval_uses_finetuned_source": eval_uses_finetuned_source,
        },
        "historical_regressions": {
            "historical_oom_contamination_pattern": historical_oom_contamination_pattern,
            "wrong_checkpoint_or_missing_finetune_artifact": bool(
                formal_eligibility == "BLOCK"
            ),
        },
        "failure_note_path": None,
    }
    return report


def _build_failure_note(report: Mapping[str, object]) -> str:
    reasons = report.get("gate_reasons")
    rendered_reasons = (
        [str(item) for item in reasons]
        if isinstance(reasons, list)
        else ["no gate reasons captured"]
    )
    lines = [
        "# GR00T checkpoint provenance gate 失败说明",
        "",
        f"- formal_eligibility: `{report.get('formal_eligibility', 'BLOCK')}`",
        f"- reason_code: `{report.get('reason_code', WRONG_CHECKPOINT_REASON_CODE)}`",
        f"- selected_checkpoint_path: `{report.get('selected_checkpoint_path')}`",
        f"- server_load_path: `{report.get('server_load_path')}`",
        f"- base_model_path: `{report.get('base_model_path')}`",
        f"- loadability_status: `{report.get('loadability_status')}`",
        f"- is_base_fallback: `{report.get('is_base_fallback')}`",
        "- gate_reasons:",
    ]
    lines.extend(f"  - `{reason}`" for reason in rendered_reasons)
    lines.extend(
        [
            "",
            "该评估目标未能证明 server 将加载被选中的 finetuned checkpoint，因此不得作为 public anchor / P rung / D rung 的正式证据。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_failure_note(path: Path, report: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_build_failure_note(report), encoding="utf-8")
    tmp.replace(path)
    return path


def _build_internal_error_report(
    *,
    metadata_path: Path,
    output_dir: Path,
    message: str,
) -> dict[str, Any]:
    report_core = {
        "selected_checkpoint_path": None,
        "base_model_path": None,
        "server_load_path": None,
        "is_base_fallback": False,
        "loadability_status": "BLOCKED_METADATA_INVALID",
        "formal_eligibility": "BLOCK",
        "reason_code": WRONG_CHECKPOINT_REASON_CODE,
        "gate_reasons": [str(message)],
    }
    checksum_or_signature = _build_checksum_or_signature(
        report_core=report_core,
        selected_checkpoint_asset_path=None,
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "gate_name": CHECKPOINT_PROVENANCE_GATE_NAME,
        "metadata_path": str(metadata_path),
        "artifact_path": str(output_dir / CHECKPOINT_PROVENANCE_REPORT_JSON_NAME),
        "status": "FAIL",
        "reason_code": WRONG_CHECKPOINT_REASON_CODE,
        "formal_eligibility": "BLOCK",
        "selected_checkpoint_path": None,
        "base_model_path": None,
        "server_load_path": None,
        "is_base_fallback": False,
        "loadability_status": "BLOCKED_METADATA_INVALID",
        "checksum_or_signature": checksum_or_signature,
        "gate_reasons": [str(message)],
        "selected_checkpoint_metadata": {
            "contract_path": SELECTED_CHECKPOINT_CONTRACT_PATH,
            "raw_selected_checkpoint_path": None,
            "selected_checkpoint_exists": False,
            "selected_checkpoint_asset_path": None,
            "selected_checkpoint_asset_name": None,
            "selected_checkpoint_loadable": False,
            "selected_checkpoint_error": str(message),
        },
        "server_binding": {
            "server_load_path_source": None,
            "server_load_mode": "model_path",
            "server_load_mode_source": None,
            "declared_server_load_path": None,
            "declared_server_load_path_source": None,
            "server_load_path_matches_selected_checkpoint": False,
            "server_uses_model_path_branch": False,
        },
        "base_model_binding": {
            "base_model_path_source": None,
            "eval_uses_finetuned": None,
            "eval_uses_finetuned_source": None,
        },
        "historical_regressions": {
            "historical_oom_contamination_pattern": False,
            "wrong_checkpoint_or_missing_finetune_artifact": True,
        },
        "failure_note_path": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = REPO_ROOT
    output_dir = _validate_output_dir(_resolve_path(repo_root, args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = _resolve_path(repo_root, args.metadata)
    try:
        metadata = _read_json(metadata_path)
        report = build_checkpoint_provenance_report(
            metadata=metadata,
            metadata_path=metadata_path,
            repo_root=repo_root,
            output_dir=output_dir,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report = _build_internal_error_report(
            metadata_path=metadata_path,
            output_dir=output_dir,
            message=_exception_message(exc),
        )

    report_path = _write_json(
        output_dir / CHECKPOINT_PROVENANCE_REPORT_JSON_NAME,
        report,
    )
    report["artifact_path"] = str(report_path)
    if str(report.get("formal_eligibility")) == "BLOCK":
        failure_note_path = _write_failure_note(
            output_dir / FAILURE_NOTE_MARKDOWN_NAME,
            report,
        )
        report["failure_note_path"] = str(failure_note_path)
        report_path = _write_json(report_path, report)
        report["artifact_path"] = str(report_path)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if str(report.get("formal_eligibility")) == "ALLOW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
