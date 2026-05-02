#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


DEFAULT_DATASET_INVENTORY_JSON = Path(
    "agent/artifacts/flux_dataset_probe/dataset_inventory_bundle.json"
)
DEFAULT_AUTHORITATIVE_SCREENING_JSON = Path(
    "agent/artifacts/apple_recap_flux_graft/authoritative_screening/"
    "authoritative_screening_summary.json"
)
DEFAULT_PREFLIGHT_REPORT_JSON = Path(
    "agent/artifacts/gr00t_wbc_preflight/preflight_report.json"
)
DEFAULT_TRAIN_SMOKE_SUMMARY_JSON = Path(
    "agent/artifacts/apple_recap_flux_graft/train_smoke_gpu_lora/"
    "train_smoke_summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "agent/artifacts/apple_recap_flux_graft/training_promotion_gate"
)

TRAINING_PROMOTION_GATE_JSON_NAME = "training_promotion_gate.json"
SCHEMA_VERSION = "gr00t_training_promotion_gate_v1"
ARTIFACT_KIND = "gr00t_training_promotion_gate"
GATE_NAME = "GR00TTrainingPromotionGate"

DATASET_INVENTORY_SCHEMA_VERSION = "flux_dataset_inventory_bundle_v1"
DATASET_INVENTORY_ARTIFACT_KIND = "flux_dataset_inventory_bundle"
DATASET_INVENTORY_COMPLETE = "inventory-complete"

SCREENING_SCHEMA_VERSION = "flux_authoritative_screening_summary_v1"
SCREENING_ARTIFACT_KIND = "flux_authoritative_screening_summary"
SCREENING_REASON_CODES = {
    "ok",
    "inventory_blocked",
    "model_binding_blocked",
    "binding_join_blocked",
}

PREFLIGHT_SCHEMA_VERSION = "g1_gr00t_wbc_preflight_gate_v1"

SMOKE_ARTIFACT_KIND = "flux_gr00t_training_smoke_summary"
SMOKE_GATE_SEMANTICS = "diagnostic_only_non_release_gate"

PROVENANCE_SCHEMA_VERSION = "gr00t_checkpoint_provenance_report_v1"
PROVENANCE_ARTIFACT_KIND = "gr00t_checkpoint_provenance_report"

PROMOTION_STATUS_PASS = "PASS"
PROMOTION_STATUS_BLOCK = "BLOCK"

CHECK_STATUS_PASS = "PASS"
CHECK_STATUS_BLOCK = "BLOCK"
CHECK_STATUS_NOT_REACHED = "NOT_REACHED"
CHECK_STATUS_INVALID_INPUT = "INVALID_INPUT"

CHECK_DEBUG = "debug_fake_data_config_pass"
CHECK_INFERENCE = "inference_surface_contract_pass"
CHECK_SERVEABLE = "serveable_checkpoint_inventory_pass"
CHECK_PREFLIGHT = "preflight_pass"
CHECK_SMOKE = "non_uplift_smoke_pass"

CHECK_ORDER: tuple[str, ...] = (
    CHECK_DEBUG,
    CHECK_INFERENCE,
    CHECK_SERVEABLE,
    CHECK_PREFLIGHT,
    CHECK_SMOKE,
)

REASON_CODE_OK = "ok"
REASON_CODE_DEBUG_BLOCKED = "debug_config_blocked"
REASON_CODE_INFERENCE_BLOCKED = "inference_surface_blocked"
REASON_CODE_SERVEABLE_BLOCKED = "serveable_inventory_blocked"
REASON_CODE_PREFLIGHT_BLOCKED = "preflight_blocked"
REASON_CODE_SMOKE_BLOCKED = "smoke_blocked"
REASON_CODE_INPUT_INVALID = "gate_input_invalid"

CHECK_REASON_CODE = {
    CHECK_DEBUG: REASON_CODE_DEBUG_BLOCKED,
    CHECK_INFERENCE: REASON_CODE_INFERENCE_BLOCKED,
    CHECK_SERVEABLE: REASON_CODE_SERVEABLE_BLOCKED,
    CHECK_PREFLIGHT: REASON_CODE_PREFLIGHT_BLOCKED,
    CHECK_SMOKE: REASON_CODE_SMOKE_BLOCKED,
}


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_training_promotion_gate.py",
        description=(
            "Aggregate existing dataset/screening/preflight/smoke artifacts into a "
            "single machine-readable planning gate for whether the next real "
            "training stage may be planned. This gate is diagnostic-only and never "
            "starts training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-inventory-json",
        type=Path,
        default=DEFAULT_DATASET_INVENTORY_JSON,
        help="Task 7 dataset inventory bundle consumed as a read-only source artifact.",
    )
    parser.add_argument(
        "--authoritative-screening-json",
        type=Path,
        default=DEFAULT_AUTHORITATIVE_SCREENING_JSON,
        help="Task 8 authoritative screening summary consumed as the joined dataset+inference contract.",
    )
    parser.add_argument(
        "--preflight-report-json",
        type=Path,
        default=DEFAULT_PREFLIGHT_REPORT_JSON,
        help="Machine-readable GR00T G1 WBC preflight report JSON.",
    )
    parser.add_argument(
        "--train-smoke-summary-json",
        type=Path,
        default=DEFAULT_TRAIN_SMOKE_SUMMARY_JSON,
        help="Task 13 diagnostic train-smoke summary JSON.",
    )
    parser.add_argument(
        "--checkpoint-provenance-json",
        type=str,
        default="",
        help="Optional checkpoint provenance report JSON; when present it must also pass before rung 3 goes green.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives training_promotion_gate.json.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _resolve_repo_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _repo_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)
    return path


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return dict(cast(Mapping[str, Any], payload))


def _non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field_name} must be a non-empty string")
    return value.strip()


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_issue_list(value: object, *, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    issues: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_name}[{index}] must be an object")
        issues.append(dict(cast(Mapping[str, Any], item)))
    return issues


def _as_optional_issue_list(value: object, *, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    return _as_issue_list(value, field_name=field_name)


def _as_string_list(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    normalized: list[str] = []
    for index, item in enumerate(value):
        normalized.append(_non_empty_string(item, field_name=f"{field_name}[{index}]"))
    return normalized


def _issue(
    code: str,
    field_path: str,
    message: str,
    *,
    source_artifact: str,
    expected: object | None = None,
    observed: object | None = None,
    detail: object | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": str(code),
        "field_path": str(field_path),
        "message": str(message),
        "source_artifact": str(source_artifact),
    }
    if expected is not None:
        payload["expected"] = expected
    if observed is not None:
        payload["observed"] = observed
    if detail is not None:
        payload["detail"] = detail
    return payload


def _invalid_artifact_result(
    *,
    artifact_id: str,
    path: Path | None,
    repo_root: Path,
    optional: bool,
    provided: bool,
    issues: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_artifact: dict[str, Any] = {
        "artifact_id": str(artifact_id),
        "optional": bool(optional),
        "provided": bool(provided),
        "exists": bool(path is not None and path.exists()),
        "valid": False,
    }
    if path is not None:
        source_artifact["path"] = str(path)
        source_artifact["relative_path"] = _repo_relative_path(repo_root, path)
    if isinstance(payload, Mapping):
        if "schema_version" in payload:
            source_artifact["schema_version"] = payload.get("schema_version")
        if "artifact_kind" in payload:
            source_artifact["artifact_kind"] = payload.get("artifact_kind")
    return {
        "payload": dict(payload) if isinstance(payload, Mapping) else None,
        "issues": [dict(cast(Mapping[str, Any], item)) for item in issues],
        "source_artifact": source_artifact,
        "path": path,
        "provided": bool(provided),
        "valid": False,
    }


def _valid_artifact_result(
    *,
    artifact_id: str,
    path: Path | None,
    repo_root: Path,
    optional: bool,
    provided: bool,
    payload: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_artifact: dict[str, Any] = {
        "artifact_id": str(artifact_id),
        "optional": bool(optional),
        "provided": bool(provided),
        "exists": bool(path is not None and path.exists()),
        "valid": True,
    }
    if path is not None:
        source_artifact["path"] = str(path)
        source_artifact["relative_path"] = _repo_relative_path(repo_root, path)
    if isinstance(payload, Mapping):
        if "schema_version" in payload:
            source_artifact["schema_version"] = payload.get("schema_version")
        if "artifact_kind" in payload:
            source_artifact["artifact_kind"] = payload.get("artifact_kind")
    if isinstance(summary, Mapping):
        source_artifact["summary"] = dict(summary)
    return {
        "payload": dict(payload) if isinstance(payload, Mapping) else None,
        "issues": [],
        "source_artifact": source_artifact,
        "path": path,
        "provided": bool(provided),
        "valid": True,
    }


def _load_dataset_inventory(
    *,
    path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    artifact_id = "dataset_inventory"
    try:
        payload = _read_json_object(path)
    except Exception as exc:
        return _invalid_artifact_result(
            artifact_id=artifact_id,
            path=path,
            repo_root=repo_root,
            optional=False,
            provided=True,
            issues=[
                _issue(
                    "invalid_required_artifact",
                    artifact_id,
                    "failed to load dataset inventory artifact",
                    source_artifact=artifact_id,
                    observed=_exception_message(exc),
                )
            ],
        )

    try:
        if payload.get("schema_version") != DATASET_INVENTORY_SCHEMA_VERSION:
            raise ValueError("dataset inventory schema_version mismatch")
        if payload.get("artifact_kind") != DATASET_INVENTORY_ARTIFACT_KIND:
            raise ValueError("dataset inventory artifact_kind mismatch")
        verdict = _non_empty_string(payload.get("verdict"), field_name="verdict")
        blocking_reasons = _as_issue_list(
            payload.get("blocking_reasons", []),
            field_name="blocking_reasons",
        )
    except Exception as exc:
        return _invalid_artifact_result(
            artifact_id=artifact_id,
            path=path,
            repo_root=repo_root,
            optional=False,
            provided=True,
            issues=[
                _issue(
                    "invalid_required_artifact",
                    artifact_id,
                    "dataset inventory artifact does not match the expected Task 7 surface",
                    source_artifact=artifact_id,
                    observed=_exception_message(exc),
                )
            ],
            payload=payload,
        )

    return _valid_artifact_result(
        artifact_id=artifact_id,
        path=path,
        repo_root=repo_root,
        optional=False,
        provided=True,
        payload=payload,
        summary={
            "verdict": verdict,
            "blocking_reason_count": len(blocking_reasons),
        },
    )


def _load_authoritative_screening(
    *,
    path: Path,
    repo_root: Path,
    expected_inventory_path: Path,
) -> dict[str, Any]:
    artifact_id = "authoritative_screening"
    try:
        payload = _read_json_object(path)
    except Exception as exc:
        return _invalid_artifact_result(
            artifact_id=artifact_id,
            path=path,
            repo_root=repo_root,
            optional=False,
            provided=True,
            issues=[
                _issue(
                    "invalid_required_artifact",
                    artifact_id,
                    "failed to load authoritative screening artifact",
                    source_artifact=artifact_id,
                    observed=_exception_message(exc),
                )
            ],
        )

    try:
        if payload.get("schema_version") != SCREENING_SCHEMA_VERSION:
            raise ValueError("authoritative screening schema_version mismatch")
        if payload.get("artifact_kind") != SCREENING_ARTIFACT_KIND:
            raise ValueError("authoritative screening artifact_kind mismatch")
        formal_eligibility = _non_empty_string(
            payload.get("formal_eligibility"),
            field_name="formal_eligibility",
        )
        if formal_eligibility not in {"ALLOW", "BLOCK"}:
            raise ValueError("authoritative screening formal_eligibility mismatch")
        reason_code = _non_empty_string(
            payload.get("reason_code"),
            field_name="reason_code",
        )
        if reason_code not in SCREENING_REASON_CODES:
            raise ValueError("authoritative screening reason_code mismatch")
        _ = _as_optional_issue_list(payload.get("issues"), field_name="issues")
        inventory_json = _resolve_repo_path(
            repo_root,
            _non_empty_string(
                payload.get("dataset_inventory_json"),
                field_name="dataset_inventory_json",
            ),
        )
        if inventory_json != expected_inventory_path:
            raise ValueError(
                "authoritative screening must reference the provided dataset inventory artifact"
            )
        inventory_verdict = _non_empty_string(
            payload.get("inventory_verdict"),
            field_name="inventory_verdict",
        )
        servable = _as_bool(payload.get("servable"), field_name="servable")
    except Exception as exc:
        return _invalid_artifact_result(
            artifact_id=artifact_id,
            path=path,
            repo_root=repo_root,
            optional=False,
            provided=True,
            issues=[
                _issue(
                    "invalid_required_artifact",
                    artifact_id,
                    "authoritative screening artifact does not match the expected Task 8 surface",
                    source_artifact=artifact_id,
                    observed=_exception_message(exc),
                )
            ],
            payload=payload,
        )

    return _valid_artifact_result(
        artifact_id=artifact_id,
        path=path,
        repo_root=repo_root,
        optional=False,
        provided=True,
        payload=payload,
        summary={
            "formal_eligibility": formal_eligibility,
            "reason_code": reason_code,
            "inventory_verdict": inventory_verdict,
            "servable": servable,
        },
    )


def _load_preflight_report(
    *,
    path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    artifact_id = "preflight_report"
    try:
        payload = _read_json_object(path)
    except Exception as exc:
        return _invalid_artifact_result(
            artifact_id=artifact_id,
            path=path,
            repo_root=repo_root,
            optional=False,
            provided=True,
            issues=[
                _issue(
                    "invalid_required_artifact",
                    artifact_id,
                    "failed to load preflight report artifact",
                    source_artifact=artifact_id,
                    observed=_exception_message(exc),
                )
            ],
        )

    try:
        if payload.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
            raise ValueError("preflight report schema_version mismatch")
        status = _non_empty_string(payload.get("status"), field_name="status")
        if status not in {"PASS", "FAIL"}:
            raise ValueError("preflight report status mismatch")
        reason_code = _non_empty_string(
            payload.get("reason_code"),
            field_name="reason_code",
        )
    except Exception as exc:
        return _invalid_artifact_result(
            artifact_id=artifact_id,
            path=path,
            repo_root=repo_root,
            optional=False,
            provided=True,
            issues=[
                _issue(
                    "invalid_required_artifact",
                    artifact_id,
                    "preflight report artifact does not match the expected surface",
                    source_artifact=artifact_id,
                    observed=_exception_message(exc),
                )
            ],
            payload=payload,
        )

    return _valid_artifact_result(
        artifact_id=artifact_id,
        path=path,
        repo_root=repo_root,
        optional=False,
        provided=True,
        payload=payload,
        summary={
            "status": status,
            "reason_code": reason_code,
        },
    )


def _load_train_smoke_summary(
    *,
    path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    artifact_id = "train_smoke_summary"
    try:
        payload = _read_json_object(path)
    except Exception as exc:
        return _invalid_artifact_result(
            artifact_id=artifact_id,
            path=path,
            repo_root=repo_root,
            optional=False,
            provided=True,
            issues=[
                _issue(
                    "invalid_required_artifact",
                    artifact_id,
                    "failed to load train smoke summary artifact",
                    source_artifact=artifact_id,
                    observed=_exception_message(exc),
                )
            ],
        )

    try:
        if payload.get("artifact_kind") != SMOKE_ARTIFACT_KIND:
            raise ValueError("train smoke artifact_kind mismatch")
        wrapper_status = _non_empty_string(
            payload.get("wrapper_status"),
            field_name="wrapper_status",
        )
        if wrapper_status not in {"ok", "blocked"}:
            raise ValueError("train smoke wrapper_status mismatch")
        if (
            _as_bool(payload.get("diagnostic_only"), field_name="diagnostic_only")
            is not True
        ):
            raise ValueError("train smoke diagnostic_only mismatch")
        if (
            _as_bool(
                payload.get("mainline_authority"),
                field_name="mainline_authority",
            )
            is not False
        ):
            raise ValueError("train smoke mainline_authority mismatch")
        if (
            _as_bool(
                payload.get("main_verdict_eligible"),
                field_name="main_verdict_eligible",
            )
            is not False
        ):
            raise ValueError("train smoke main_verdict_eligible mismatch")
        if (
            _as_bool(payload.get("release_gate"), field_name="release_gate")
            is not False
        ):
            raise ValueError("train smoke release_gate mismatch")
        if (
            _as_bool(
                payload.get("external_reference_only"),
                field_name="external_reference_only",
            )
            is not True
        ):
            raise ValueError("train smoke external_reference_only mismatch")
        gate_semantics = _non_empty_string(
            payload.get("gate_semantics"),
            field_name="gate_semantics",
        )
        if gate_semantics != SMOKE_GATE_SEMANTICS:
            raise ValueError("train smoke gate_semantics mismatch")
    except Exception as exc:
        return _invalid_artifact_result(
            artifact_id=artifact_id,
            path=path,
            repo_root=repo_root,
            optional=False,
            provided=True,
            issues=[
                _issue(
                    "invalid_required_artifact",
                    artifact_id,
                    "train smoke summary artifact does not match the expected Task 13 surface",
                    source_artifact=artifact_id,
                    observed=_exception_message(exc),
                )
            ],
            payload=payload,
        )

    return _valid_artifact_result(
        artifact_id=artifact_id,
        path=path,
        repo_root=repo_root,
        optional=False,
        provided=True,
        payload=payload,
        summary={
            "wrapper_status": wrapper_status,
            "trainable_surface": payload.get("trainable_surface"),
            "selected_checkpoint_path": payload.get("selected_checkpoint_path"),
        },
    )


def _load_checkpoint_provenance(
    *,
    path: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    artifact_id = "checkpoint_provenance"
    if path is None:
        return _valid_artifact_result(
            artifact_id=artifact_id,
            path=None,
            repo_root=repo_root,
            optional=True,
            provided=False,
            payload=None,
            summary={"provided": False},
        )

    try:
        payload = _read_json_object(path)
    except Exception as exc:
        return _invalid_artifact_result(
            artifact_id=artifact_id,
            path=path,
            repo_root=repo_root,
            optional=True,
            provided=True,
            issues=[
                _issue(
                    "invalid_optional_artifact",
                    artifact_id,
                    "failed to load checkpoint provenance artifact",
                    source_artifact=artifact_id,
                    observed=_exception_message(exc),
                )
            ],
        )

    try:
        if payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
            raise ValueError("checkpoint provenance schema_version mismatch")
        if payload.get("artifact_kind") != PROVENANCE_ARTIFACT_KIND:
            raise ValueError("checkpoint provenance artifact_kind mismatch")
        formal_eligibility = _non_empty_string(
            payload.get("formal_eligibility"),
            field_name="formal_eligibility",
        )
        if formal_eligibility not in {"ALLOW", "BLOCK"}:
            raise ValueError("checkpoint provenance formal_eligibility mismatch")
        reason_code = _non_empty_string(
            payload.get("reason_code"),
            field_name="reason_code",
        )
        gate_reasons = _as_string_list(
            payload.get("gate_reasons", []),
            field_name="gate_reasons",
        )
    except Exception as exc:
        return _invalid_artifact_result(
            artifact_id=artifact_id,
            path=path,
            repo_root=repo_root,
            optional=True,
            provided=True,
            issues=[
                _issue(
                    "invalid_optional_artifact",
                    artifact_id,
                    "checkpoint provenance artifact does not match the expected surface",
                    source_artifact=artifact_id,
                    observed=_exception_message(exc),
                )
            ],
            payload=payload,
        )

    return _valid_artifact_result(
        artifact_id=artifact_id,
        path=path,
        repo_root=repo_root,
        optional=True,
        provided=True,
        payload=payload,
        summary={
            "formal_eligibility": formal_eligibility,
            "reason_code": reason_code,
            "gate_reason_count": len(gate_reasons),
        },
    )


def _check_payload(
    *,
    check_name: str,
    status: str,
    source_artifact_ids: Sequence[str],
    issues: Sequence[Mapping[str, Any]],
    blocked_by: Sequence[str] = (),
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": str(status),
        "passed": bool(status == CHECK_STATUS_PASS),
        "source_artifact_ids": [str(item) for item in source_artifact_ids],
        "issues": [dict(cast(Mapping[str, Any], item)) for item in issues],
    }
    if blocked_by:
        payload["blocked_by"] = [str(item) for item in blocked_by]
    if details is not None:
        payload["details"] = dict(details)
    return {str(check_name): payload}


def _screening_reason_issues(
    screening_payload: Mapping[str, Any],
    *,
    expected_reason_code: str,
    fallback_message: str,
) -> list[dict[str, Any]]:
    if screening_payload.get("reason_code") == expected_reason_code:
        issues = screening_payload.get("issues")
        if isinstance(issues, list):
            return [
                dict(cast(Mapping[str, Any], item))
                for item in issues
                if isinstance(item, Mapping)
            ]
    return [
        _issue(
            expected_reason_code,
            "authoritative_screening.reason_code",
            fallback_message,
            source_artifact="authoritative_screening",
            observed=screening_payload.get("reason_code"),
        )
    ]


def _preflight_block_issues(
    preflight_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    failure = preflight_payload.get("failure")
    failure_mapping = (
        dict(cast(Mapping[str, Any], failure)) if isinstance(failure, Mapping) else {}
    )
    detail = failure_mapping.get("detail")
    blockers = (
        list(failure_mapping.get("blockers", []))
        if isinstance(failure_mapping.get("blockers"), list)
        else []
    )
    return [
        _issue(
            str(preflight_payload.get("reason_code") or "blocked_preflight"),
            "preflight_report.reason_code",
            str(failure_mapping.get("message") or "preflight gate blocked"),
            source_artifact="preflight_report",
            observed=preflight_payload.get("status"),
            detail={
                "blockers": blockers,
                "detail": detail,
            },
        )
    ]


def _smoke_block_issues(smoke_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _issue(
            str(smoke_payload.get("wrapper_status") or "blocked"),
            "train_smoke_summary.wrapper_status",
            str(
                smoke_payload.get("error")
                or "diagnostic train smoke did not report wrapper_status=ok"
            ),
            source_artifact="train_smoke_summary",
            observed=smoke_payload.get("wrapper_status"),
            detail={
                "trainable_surface": smoke_payload.get("trainable_surface"),
                "selected_checkpoint_path": smoke_payload.get(
                    "selected_checkpoint_path"
                ),
            },
        )
    ]


def _inventory_block_issues(
    *,
    inventory_payload: Mapping[str, Any],
    screening_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    screening_issues = _screening_reason_issues(
        screening_payload,
        expected_reason_code="inventory_blocked",
        fallback_message="authoritative screening reported inventory blockage",
    )
    if screening_payload.get("reason_code") == "inventory_blocked" and screening_issues:
        return screening_issues
    blocking_reasons = inventory_payload.get("blocking_reasons")
    if isinstance(blocking_reasons, list) and blocking_reasons:
        return [
            dict(cast(Mapping[str, Any], item))
            for item in blocking_reasons
            if isinstance(item, Mapping)
        ]
    return [
        _issue(
            "inventory_verdict_blocked",
            "dataset_inventory.verdict",
            "dataset inventory must be inventory-complete before planning the next training stage",
            source_artifact="dataset_inventory",
            expected=DATASET_INVENTORY_COMPLETE,
            observed=inventory_payload.get("verdict"),
        )
    ]


def _checkpoint_provenance_block_issues(
    provenance_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _issue(
            str(
                provenance_payload.get("reason_code") or "checkpoint_provenance_blocked"
            ),
            "checkpoint_provenance.formal_eligibility",
            "checkpoint provenance gate blocked next-stage training planning",
            source_artifact="checkpoint_provenance",
            observed=provenance_payload.get("formal_eligibility"),
            detail={
                "gate_reasons": list(provenance_payload.get("gate_reasons", [])),
                "loadability_status": provenance_payload.get("loadability_status"),
                "selected_checkpoint_path": provenance_payload.get(
                    "selected_checkpoint_path"
                ),
            },
        )
    ]


def _inference_surface_block_issues(
    screening_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if screening_payload.get("reason_code") == "model_binding_blocked":
        issues = screening_payload.get("issues")
        if isinstance(issues, list) and issues:
            return [
                dict(cast(Mapping[str, Any], item))
                for item in issues
                if isinstance(item, Mapping)
            ]
    return [
        _issue(
            "inference_surface_contract_blocked",
            "authoritative_screening.materialized_model_ref",
            "authoritative screening did not prove an inference-only materialized surface",
            source_artifact="authoritative_screening",
            detail={
                "inference_model_materialized": screening_payload.get(
                    "inference_model_materialized"
                ),
                "train_model_materialized": screening_payload.get(
                    "train_model_materialized"
                ),
                "materialized_model_ref": screening_payload.get(
                    "materialized_model_ref"
                ),
                "inference_model_ref": screening_payload.get("inference_model_ref"),
            },
        )
    ]


def _build_checks(
    *,
    dataset_inventory: Mapping[str, Any],
    authoritative_screening: Mapping[str, Any],
    preflight_report: Mapping[str, Any],
    train_smoke_summary: Mapping[str, Any],
    checkpoint_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    inventory_payload = cast(Mapping[str, Any], dataset_inventory["payload"])
    screening_payload = cast(Mapping[str, Any], authoritative_screening["payload"])
    preflight_payload = cast(Mapping[str, Any], preflight_report["payload"])
    smoke_payload = cast(Mapping[str, Any], train_smoke_summary["payload"])
    provenance_payload = cast(
        Mapping[str, Any] | None,
        checkpoint_provenance.get("payload"),
    )

    checks: dict[str, Any] = {}

    debug_pass = screening_payload.get("reason_code") != "binding_join_blocked"
    checks.update(
        _check_payload(
            check_name=CHECK_DEBUG,
            status=CHECK_STATUS_PASS if debug_pass else CHECK_STATUS_BLOCK,
            source_artifact_ids=("authoritative_screening", "dataset_inventory"),
            issues=(
                []
                if debug_pass
                else _screening_reason_issues(
                    screening_payload,
                    expected_reason_code="binding_join_blocked",
                    fallback_message=(
                        "authoritative screening reported a dataset/config join mismatch"
                    ),
                )
            ),
            details={
                "screening_reason_code": screening_payload.get("reason_code"),
                "binding_join_evaluation": screening_payload.get(
                    "binding_join_evaluation"
                ),
            },
        )
    )

    inference_pass = bool(
        screening_payload.get("inference_model_materialized") is True
        and screening_payload.get("train_model_materialized") is False
        and isinstance(screening_payload.get("inference_model_ref"), Mapping)
        and screening_payload.get("materialized_model_ref")
        == screening_payload.get("inference_model_ref")
        and screening_payload.get("reason_code") != "model_binding_blocked"
    )
    checks.update(
        _check_payload(
            check_name=CHECK_INFERENCE,
            status=CHECK_STATUS_PASS if inference_pass else CHECK_STATUS_BLOCK,
            source_artifact_ids=("authoritative_screening",),
            issues=[]
            if inference_pass
            else _inference_surface_block_issues(screening_payload),
            details={
                "screening_reason_code": screening_payload.get("reason_code"),
                "inference_model_materialized": screening_payload.get(
                    "inference_model_materialized"
                ),
                "train_model_materialized": screening_payload.get(
                    "train_model_materialized"
                ),
                "materialized_model_ref": screening_payload.get(
                    "materialized_model_ref"
                ),
                "inference_model_ref": screening_payload.get("inference_model_ref"),
            },
        )
    )

    if not debug_pass or not inference_pass:
        checks.update(
            _check_payload(
                check_name=CHECK_SERVEABLE,
                status=CHECK_STATUS_NOT_REACHED,
                source_artifact_ids=(
                    "dataset_inventory",
                    "authoritative_screening",
                    "checkpoint_provenance",
                ),
                issues=[],
                blocked_by=[
                    name
                    for name, passed in (
                        (CHECK_DEBUG, debug_pass),
                        (CHECK_INFERENCE, inference_pass),
                    )
                    if not passed
                ],
                details={
                    "inventory_verdict": inventory_payload.get("verdict"),
                    "screening_inventory_verdict": screening_payload.get(
                        "inventory_verdict"
                    ),
                    "servable": screening_payload.get("servable"),
                    "checkpoint_provenance_present": bool(
                        checkpoint_provenance.get("provided", False)
                    ),
                },
            )
        )
        serveable_pass = False
    else:
        provenance_required_pass = True
        provenance_issues: list[dict[str, Any]] = []
        if bool(checkpoint_provenance.get("provided", False)):
            provenance_required_pass = bool(
                provenance_payload is not None
                and provenance_payload.get("formal_eligibility") == "ALLOW"
            )
            if not provenance_required_pass and provenance_payload is not None:
                provenance_issues = _checkpoint_provenance_block_issues(
                    provenance_payload
                )

        serveable_pass = bool(
            inventory_payload.get("verdict") == DATASET_INVENTORY_COMPLETE
            and screening_payload.get("inventory_verdict") == DATASET_INVENTORY_COMPLETE
            and screening_payload.get("servable") is True
            and provenance_required_pass
        )
        serveable_issues = []
        if not serveable_pass:
            if provenance_issues:
                serveable_issues = provenance_issues
            else:
                serveable_issues = _inventory_block_issues(
                    inventory_payload=inventory_payload,
                    screening_payload=screening_payload,
                )
        checks.update(
            _check_payload(
                check_name=CHECK_SERVEABLE,
                status=CHECK_STATUS_PASS if serveable_pass else CHECK_STATUS_BLOCK,
                source_artifact_ids=(
                    "dataset_inventory",
                    "authoritative_screening",
                    "checkpoint_provenance",
                ),
                issues=serveable_issues,
                details={
                    "inventory_verdict": inventory_payload.get("verdict"),
                    "screening_inventory_verdict": screening_payload.get(
                        "inventory_verdict"
                    ),
                    "servable": screening_payload.get("servable"),
                    "checkpoint_provenance_present": bool(
                        checkpoint_provenance.get("provided", False)
                    ),
                    "checkpoint_provenance_eligible": (
                        None
                        if provenance_payload is None
                        else provenance_payload.get("formal_eligibility")
                    ),
                },
            )
        )

    if not checks[CHECK_SERVEABLE]["passed"]:
        checks.update(
            _check_payload(
                check_name=CHECK_PREFLIGHT,
                status=CHECK_STATUS_NOT_REACHED,
                source_artifact_ids=("preflight_report",),
                issues=[],
                blocked_by=[CHECK_SERVEABLE],
                details={
                    "status": preflight_payload.get("status"),
                    "reason_code": preflight_payload.get("reason_code"),
                },
            )
        )
        preflight_pass = False
    else:
        preflight_pass = bool(
            preflight_payload.get("status") == "PASS"
            and preflight_payload.get("reason_code") == "ok"
        )
        checks.update(
            _check_payload(
                check_name=CHECK_PREFLIGHT,
                status=CHECK_STATUS_PASS if preflight_pass else CHECK_STATUS_BLOCK,
                source_artifact_ids=("preflight_report",),
                issues=[]
                if preflight_pass
                else _preflight_block_issues(preflight_payload),
                details={
                    "status": preflight_payload.get("status"),
                    "reason_code": preflight_payload.get("reason_code"),
                    "live_checks_attempted": preflight_payload.get(
                        "live_checks_attempted"
                    ),
                },
            )
        )

    if not checks[CHECK_PREFLIGHT]["passed"]:
        checks.update(
            _check_payload(
                check_name=CHECK_SMOKE,
                status=CHECK_STATUS_NOT_REACHED,
                source_artifact_ids=("train_smoke_summary",),
                issues=[],
                blocked_by=[CHECK_PREFLIGHT],
                details={
                    "wrapper_status": smoke_payload.get("wrapper_status"),
                    "trainable_surface": smoke_payload.get("trainable_surface"),
                },
            )
        )
    else:
        smoke_pass = smoke_payload.get("wrapper_status") == "ok"
        checks.update(
            _check_payload(
                check_name=CHECK_SMOKE,
                status=CHECK_STATUS_PASS if smoke_pass else CHECK_STATUS_BLOCK,
                source_artifact_ids=("train_smoke_summary",),
                issues=[] if smoke_pass else _smoke_block_issues(smoke_payload),
                details={
                    "wrapper_status": smoke_payload.get("wrapper_status"),
                    "trainable_surface": smoke_payload.get("trainable_surface"),
                    "selected_checkpoint_path": smoke_payload.get(
                        "selected_checkpoint_path"
                    ),
                    "gate_semantics": smoke_payload.get("gate_semantics"),
                },
            )
        )

    return checks


def _invalid_input_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for check_name, artifact_ids in (
        (CHECK_DEBUG, ("authoritative_screening", "dataset_inventory")),
        (CHECK_INFERENCE, ("authoritative_screening",)),
        (
            CHECK_SERVEABLE,
            ("dataset_inventory", "authoritative_screening", "checkpoint_provenance"),
        ),
        (CHECK_PREFLIGHT, ("preflight_report",)),
        (CHECK_SMOKE, ("train_smoke_summary",)),
    ):
        checks.update(
            _check_payload(
                check_name=check_name,
                status=CHECK_STATUS_INVALID_INPUT,
                source_artifact_ids=artifact_ids,
                issues=[],
            )
        )
    return checks


def _first_failure_from_checks(
    checks: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    for check_name in CHECK_ORDER:
        check_payload = checks.get(check_name)
        if not isinstance(check_payload, Mapping):
            continue
        if check_payload.get("status") == CHECK_STATUS_BLOCK:
            issues = check_payload.get("issues")
            normalized_issues = (
                [
                    dict(cast(Mapping[str, Any], item))
                    for item in issues
                    if isinstance(item, Mapping)
                ]
                if isinstance(issues, list)
                else []
            )
            return CHECK_REASON_CODE[check_name], normalized_issues
    return REASON_CODE_OK, []


def materialize_training_promotion_gate(
    *,
    dataset_inventory_json: Path | str = DEFAULT_DATASET_INVENTORY_JSON,
    authoritative_screening_json: Path | str = DEFAULT_AUTHORITATIVE_SCREENING_JSON,
    preflight_report_json: Path | str = DEFAULT_PREFLIGHT_REPORT_JSON,
    train_smoke_summary_json: Path | str = DEFAULT_TRAIN_SMOKE_SUMMARY_JSON,
    checkpoint_provenance_json: Path | str | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    resolved_output_dir = _resolve_repo_path(repo_root, output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_inventory_json = _resolve_repo_path(repo_root, dataset_inventory_json)
    resolved_screening_json = _resolve_repo_path(
        repo_root, authoritative_screening_json
    )
    resolved_preflight_json = _resolve_repo_path(repo_root, preflight_report_json)
    resolved_smoke_json = _resolve_repo_path(repo_root, train_smoke_summary_json)
    resolved_provenance_json: Path | None = None
    if checkpoint_provenance_json is not None:
        raw_provenance = str(checkpoint_provenance_json).strip()
        if raw_provenance:
            resolved_provenance_json = _resolve_repo_path(repo_root, raw_provenance)

    dataset_inventory = _load_dataset_inventory(
        path=resolved_inventory_json,
        repo_root=repo_root,
    )
    authoritative_screening = _load_authoritative_screening(
        path=resolved_screening_json,
        repo_root=repo_root,
        expected_inventory_path=resolved_inventory_json,
    )
    preflight_report = _load_preflight_report(
        path=resolved_preflight_json,
        repo_root=repo_root,
    )
    train_smoke_summary = _load_train_smoke_summary(
        path=resolved_smoke_json,
        repo_root=repo_root,
    )
    checkpoint_provenance = _load_checkpoint_provenance(
        path=resolved_provenance_json,
        repo_root=repo_root,
    )

    source_artifacts = [
        dataset_inventory["source_artifact"],
        authoritative_screening["source_artifact"],
        preflight_report["source_artifact"],
        train_smoke_summary["source_artifact"],
    ]
    if (
        bool(checkpoint_provenance.get("provided", False))
        or resolved_provenance_json is None
    ):
        source_artifacts.append(checkpoint_provenance["source_artifact"])

    invalid_issues = [
        *cast(list[dict[str, Any]], dataset_inventory["issues"]),
        *cast(list[dict[str, Any]], authoritative_screening["issues"]),
        *cast(list[dict[str, Any]], preflight_report["issues"]),
        *cast(list[dict[str, Any]], train_smoke_summary["issues"]),
        *cast(list[dict[str, Any]], checkpoint_provenance["issues"]),
    ]

    if invalid_issues:
        checks = _invalid_input_checks()
        reason_code = REASON_CODE_INPUT_INVALID
        issues = invalid_issues
        failure_reasons = [REASON_CODE_INPUT_INVALID]
        allow_plan_next_training_stage = False
    else:
        checks = _build_checks(
            dataset_inventory=dataset_inventory,
            authoritative_screening=authoritative_screening,
            preflight_report=preflight_report,
            train_smoke_summary=train_smoke_summary,
            checkpoint_provenance=checkpoint_provenance,
        )
        reason_code, issues = _first_failure_from_checks(checks)
        failure_reasons = [] if reason_code == REASON_CODE_OK else [reason_code]
        allow_plan_next_training_stage = bool(reason_code == REASON_CODE_OK)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "gate_name": GATE_NAME,
        "output_dir": str(resolved_output_dir),
        "artifact_path": str(resolved_output_dir / TRAINING_PROMOTION_GATE_JSON_NAME),
        "promotion_allowed": allow_plan_next_training_stage,
        "allow_plan_next_training_stage": allow_plan_next_training_stage,
        "promotion_status": (
            PROMOTION_STATUS_PASS
            if allow_plan_next_training_stage
            else PROMOTION_STATUS_BLOCK
        ),
        "reason_code": reason_code,
        "failure_reasons": failure_reasons,
        "checks": checks,
        "issues": issues,
        "source_artifacts": source_artifacts,
        "diagnostic_only": True,
        "mainline_authority": False,
        "main_verdict_eligible": False,
        "external_reference_only": True,
        "release_gate": False,
        "gate_semantics": "plan_next_training_stage_only",
    }
    _write_json(resolved_output_dir / TRAINING_PROMOTION_GATE_JSON_NAME, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        payload = materialize_training_promotion_gate(
            dataset_inventory_json=args.dataset_inventory_json,
            authoritative_screening_json=args.authoritative_screening_json,
            preflight_report_json=args.preflight_report_json,
            train_smoke_summary_json=args.train_smoke_summary_json,
            checkpoint_provenance_json=args.checkpoint_provenance_json,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if bool(payload.get("allow_plan_next_training_stage")) else 1


__all__ = [
    "ARTIFACT_KIND",
    "CHECK_DEBUG",
    "CHECK_INFERENCE",
    "CHECK_ORDER",
    "CHECK_PREFLIGHT",
    "CHECK_SERVEABLE",
    "CHECK_SMOKE",
    "DEFAULT_AUTHORITATIVE_SCREENING_JSON",
    "DEFAULT_DATASET_INVENTORY_JSON",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PREFLIGHT_REPORT_JSON",
    "DEFAULT_TRAIN_SMOKE_SUMMARY_JSON",
    "GATE_NAME",
    "SCHEMA_VERSION",
    "TRAINING_PROMOTION_GATE_JSON_NAME",
    "build_parser",
    "main",
    "materialize_training_promotion_gate",
]


if __name__ == "__main__":
    raise SystemExit(main())
