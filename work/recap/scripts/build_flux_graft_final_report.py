#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.datasets import flux_grouped_dataset
from work.recap.scripts import apple_recap_execution_contract
from work.recap.scripts import gr00t_screening_authoritative


DEFAULT_SCREENING_JSON = (
    gr00t_screening_authoritative.DEFAULT_OUTPUT_DIR
    / gr00t_screening_authoritative.AUTHORITATIVE_SCREENING_JSON_NAME
)
DEFAULT_DATASET_INVENTORY_JSON = (
    gr00t_screening_authoritative.DEFAULT_DATASET_INVENTORY_JSON
)
DEFAULT_OUT_MD = Path("agent/exchange/Flux_Graft_RECAP_final_report.md")
DEFAULT_OUT_JSON = Path(
    "agent/artifacts/apple_recap_flux_graft/final_report/final_verdict_pack.json"
)
DEFAULT_TRIAGE_JSON = (
    gr00t_screening_authoritative.DEFAULT_LIVE_MODEL_TRIAGE_OUTPUT_DIR
    / gr00t_screening_authoritative.LIVE_MODEL_TRIAGE_JSON_NAME
)

REPORT_SCHEMA_VERSION = "flux_graft_final_report_builder_v2"
REPORT_ARTIFACT_KIND = "flux_graft_final_report_builder"
SCREENING_CANONICAL_ROOTS: tuple[str, ...] = (
    "agent/artifacts/apple_recap_flux_graft/authoritative_screening/",
)
DATASET_INVENTORY_CANONICAL_ROOTS: tuple[str, ...] = (
    "agent/artifacts/flux_dataset_probe/",
)
TRIAGE_CANONICAL_ROOTS: tuple[str, ...] = (
    "agent/artifacts/apple_recap_flux_graft/live_model_triage/",
)
FINAL_REPORT_JSON_CANONICAL_ROOTS: tuple[str, ...] = (
    "agent/artifacts/apple_recap_flux_graft/final_report/",
)
FINAL_REPORT_MD_CANONICAL_ROOTS: tuple[str, ...] = ("agent/exchange/",)

GLOBAL_VERDICT_BLOCKED = "BLOCKED"
GLOBAL_VERDICT_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
GLOBAL_VERDICT_AUTHORITATIVE_POSITIVE = "AUTHORITATIVE_POSITIVE"

TRIAGE_INPUT_STATUS_EXPLICIT = "explicitly_provided"
TRIAGE_INPUT_STATUS_AUTO_DISCOVERED = "auto_discovered"
TRIAGE_INPUT_STATUS_NOT_PROVIDED = "not_provided"

OPTIONAL_CONTEXT_STATUS_AVAILABLE = "available"
OPTIONAL_CONTEXT_STATUS_NOT_PROVIDED = "not_provided"

CLAIM_BOUNDARY_FORBIDDEN_IMPLICATIONS: tuple[str, ...] = (
    "Do not claim OpenPI parity.",
    "Do not claim RECAP parity.",
    "Do not claim paper-faithful parity.",
    "Do not claim full online-RL parity.",
    "Do not claim uplift is proven or confirmed beyond the current frozen authoritative protocol.",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_flux_graft_final_report.py",
        description=(
            "Validate the Flux dataset inventory + authoritative screening pair and emit "
            "a single-file Markdown report plus a machine-readable JSON pack."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--screening-json",
        type=Path,
        default=DEFAULT_SCREENING_JSON,
        help="Authoritative screening JSON produced by gr00t_screening_authoritative.py.",
    )
    parser.add_argument(
        "--dataset-inventory-json",
        type=Path,
        default=DEFAULT_DATASET_INVENTORY_JSON,
        help="Task 7 dataset_inventory_bundle.json consumed by authoritative screening.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_OUT_MD,
        help="Output path for the human-readable Markdown report.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help="Output path for the machine-readable final report pack JSON.",
    )
    parser.add_argument(
        "--triage-json",
        type=Path,
        default=None,
        help=(
            "Optional live-model triage JSON produced by "
            "materialize_live_model_triage(). When present, the report summarizes it "
            "without re-implementing the triage logic."
        ),
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_repo_path(repo_root: Path, raw: Path | str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _repo_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _invalid_input(message: str) -> ValueError:
    return ValueError(f"invalid_input: {message}")


def _stale_artifact(message: str) -> ValueError:
    return ValueError(f"stale_artifact: {message}")


def _resolve_authoritative_path(
    *,
    repo_root: Path,
    raw: Path | str,
    field_name: str,
    canonical_roots: Sequence[str],
) -> Path:
    return apple_recap_execution_contract.resolve_repo_contained_path(
        repo_root,
        raw,
        field_name=field_name,
        canonical_roots=canonical_roots,
        reject_noncanonical_parts=True,
    )


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    try:
        return _non_empty_string(value, field_name=field_name)
    except (TypeError, ValueError) as exc:
        raise _invalid_input(_exception_message(exc)) from exc


def _require_sha256_string(value: object, *, field_name: str) -> str:
    digest = _require_non_empty_string(value, field_name=field_name)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise _invalid_input(f"{field_name} must be a lowercase sha256 hex digest")
    return digest


def _validate_required_authority_ref(
    ref: Mapping[str, Any],
    *,
    artifact_id: str,
    ref_path: str,
    repo_root: Path,
    canonical_roots: Sequence[str],
) -> dict[str, Any]:
    relative_path = _require_non_empty_string(
        ref.get("relative_path"),
        field_name=f"{ref_path}.relative_path",
    )
    declared_sha = _require_sha256_string(
        ref.get("content_sha256"),
        field_name=f"{ref_path}.content_sha256",
    )
    resolved_path = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=relative_path,
        field_name=f"{ref_path}.relative_path",
        canonical_roots=canonical_roots,
    )
    try:
        actual_ref = apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id=str(ref.get("artifact_id") or artifact_id),
            authority_role=str(ref.get("authority_role") or "upstream"),
            relative_path=resolved_path,
            reject_noncanonical_parts=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        message = _exception_message(exc)
        if message.startswith("noncanonical_root_contamination:"):
            raise ValueError(message) from exc
        raise _invalid_input(message) from exc
    normalized_relative_path = str(actual_ref["relative_path"])
    if declared_sha != actual_ref["content_sha256"]:
        raise _stale_artifact(
            f"{artifact_id} authority ref digest mismatch for {normalized_relative_path}"
        )
    for optional_field in (
        "artifact_kind",
        "schema_version",
        "report_signature_sha256",
    ):
        if optional_field in ref and ref.get(optional_field) != actual_ref.get(
            optional_field
        ):
            raise _stale_artifact(
                f"{artifact_id} authority ref {optional_field} mismatch for {normalized_relative_path}"
            )
    normalized_ref = dict(ref)
    normalized_ref["relative_path"] = normalized_relative_path
    normalized_ref["content_sha256"] = declared_sha
    return normalized_ref


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _read_json(path: Path, *, artifact_id: str) -> dict[str, Any]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _invalid_input(
            f"{artifact_id} must point to a readable JSON artifact: {_exception_message(exc)}"
        ) from exc
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise _invalid_input(
            f"{artifact_id} must contain valid JSON: {_exception_message(exc)}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise _invalid_input(
            f"{artifact_id} must contain a JSON object, got {type(payload).__name__}"
        )
    return dict(cast(Mapping[str, Any], payload))


def _non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _signature_for_payload(payload: Mapping[str, Any]) -> str:
    signature_basis = {
        str(key): value
        for key, value in dict(payload).items()
        if key != "report_signature_sha256"
    }
    return apple_recap_execution_contract._sha256_payload(signature_basis)


def _validate_report_signature(payload: Mapping[str, Any], *, artifact_id: str) -> str:
    declared = _require_non_empty_string(
        payload.get("report_signature_sha256"),
        field_name=f"{artifact_id}.report_signature_sha256",
    )
    expected = _signature_for_payload(payload)
    if declared != expected:
        raise _stale_artifact(
            f"{artifact_id} report_signature_sha256 mismatch: expected {expected!r}, got {declared!r}"
        )
    return declared


def _collect_authority_refs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    direct = payload.get("authority_ref")
    if isinstance(direct, Mapping):
        refs.append(dict(cast(Mapping[str, Any], direct)))
    for field_name in (
        "source_artifacts",
        "authority_refs",
        "read_only_authority_refs",
    ):
        raw = payload.get(field_name)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, Mapping):
                refs.append(dict(cast(Mapping[str, Any], item)))
    return refs


def _validate_issue_list(
    payload: Mapping[str, Any], *, artifact_id: str
) -> list[dict[str, Any]]:
    raw_issues = payload.get("issues")
    if not isinstance(raw_issues, list):
        raise _invalid_input(f"{artifact_id}.issues must be a list")
    issues: list[dict[str, Any]] = []
    for index, item in enumerate(raw_issues):
        if not isinstance(item, Mapping):
            raise _invalid_input(f"{artifact_id}.issues[{index}] must be an object")
        issue = dict(cast(Mapping[str, Any], item))
        for key in ("code", "field_path", "message"):
            _ = _require_non_empty_string(
                issue.get(key), field_name=f"{artifact_id}.issues[{index}].{key}"
            )
        issues.append(issue)
    return issues


def _validate_dataset_inventory_artifact(
    *,
    path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    payload = _read_json(path, artifact_id="dataset_inventory_bundle")
    if payload.get("schema_version") != flux_grouped_dataset.SCHEMA_VERSION:
        raise _invalid_input(
            "dataset_inventory_bundle.schema_version mismatch: expected "
            + repr(flux_grouped_dataset.SCHEMA_VERSION)
        )
    if payload.get("artifact_kind") != flux_grouped_dataset.ARTIFACT_KIND:
        raise _invalid_input(
            "dataset_inventory_bundle.artifact_kind mismatch: expected "
            + repr(flux_grouped_dataset.ARTIFACT_KIND)
        )
    inventory_entry = {
        "artifact_id": "dataset_inventory_bundle",
        "path": _repo_relative_path(repo_root, path),
        "file_sha256": apple_recap_execution_contract._sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "artifact_kind": payload.get("artifact_kind"),
        "authority_ref": apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id="dataset_inventory_bundle",
            authority_role="flux_final_report_input",
            relative_path=path,
            reject_noncanonical_parts=True,
        ),
    }
    return {"payload": payload, "inventory_entry": inventory_entry}


def _validate_screening_artifact(
    *,
    path: Path,
    repo_root: Path,
    dataset_inventory_entry: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_json(path, artifact_id="authoritative_screening")
    if (
        payload.get("schema_version")
        != gr00t_screening_authoritative.SCREENING_SCHEMA_VERSION
    ):
        raise _invalid_input(
            "authoritative_screening.schema_version mismatch: expected "
            + repr(gr00t_screening_authoritative.SCREENING_SCHEMA_VERSION)
        )
    if (
        payload.get("artifact_kind")
        != gr00t_screening_authoritative.SCREENING_ARTIFACT_KIND
    ):
        raise _invalid_input(
            "authoritative_screening.artifact_kind mismatch: expected "
            + repr(gr00t_screening_authoritative.SCREENING_ARTIFACT_KIND)
        )
    report_signature = _validate_report_signature(
        payload,
        artifact_id="authoritative_screening",
    )
    screening_mode = _require_non_empty_string(
        payload.get("screening_mode"),
        field_name="authoritative_screening.screening_mode",
    )
    if screening_mode != "authoritative":
        raise _invalid_input(
            "authoritative_screening.screening_mode mismatch: expected 'authoritative'"
        )
    formal_eligibility = _require_non_empty_string(
        payload.get("formal_eligibility"),
        field_name="authoritative_screening.formal_eligibility",
    )
    reason_code = _require_non_empty_string(
        payload.get("reason_code"),
        field_name="authoritative_screening.reason_code",
    )
    if formal_eligibility not in {"ALLOW", "BLOCK"}:
        raise _invalid_input(
            "authoritative_screening.formal_eligibility must be 'ALLOW' or 'BLOCK'"
        )
    if reason_code not in {
        "ok",
        "inventory_blocked",
        "model_binding_blocked",
        "binding_join_blocked",
    }:
        raise _invalid_input("authoritative_screening.reason_code is not recognized")
    issues = _validate_issue_list(payload, artifact_id="authoritative_screening")
    refs = _collect_authority_refs(payload)
    if not refs:
        raise _invalid_input(
            "authoritative_screening is missing required authority refs"
        )
    expected_inventory_ref = cast(
        Mapping[str, Any],
        dataset_inventory_entry["authority_ref"],
    )
    matched_inventory_ref = None
    for index, ref in enumerate(refs):
        validated_ref = _validate_required_authority_ref(
            ref,
            artifact_id="authoritative_screening",
            ref_path=f"authoritative_screening.authority_refs[{index}]",
            repo_root=repo_root,
            canonical_roots=DATASET_INVENTORY_CANONICAL_ROOTS,
        )
        if validated_ref.get("content_sha256") != expected_inventory_ref.get(
            "content_sha256"
        ):
            continue
        if validated_ref.get("relative_path") != expected_inventory_ref.get(
            "relative_path"
        ):
            continue
        matched_inventory_ref = validated_ref
        break
    if matched_inventory_ref is None:
        raise _stale_artifact(
            "authoritative_screening must reference the provided dataset inventory artifact via source_artifacts"
        )

    inventory_entry = {
        "artifact_id": "authoritative_screening",
        "path": _repo_relative_path(repo_root, path),
        "file_sha256": apple_recap_execution_contract._sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "artifact_kind": payload.get("artifact_kind"),
        "report_signature_sha256": report_signature,
        "formal_eligibility": formal_eligibility,
        "reason_code": reason_code,
        "issues": issues,
        "servable": bool(payload.get("servable") is True),
        "authority_ref": apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id="authoritative_screening",
            authority_role="flux_final_report_input",
            relative_path=path,
            reject_noncanonical_parts=True,
        ),
    }
    return {"payload": payload, "inventory_entry": inventory_entry}


def _validate_live_model_triage_artifact(
    *,
    path: Path,
    repo_root: Path,
    dataset_inventory_entry: Mapping[str, Any],
    screening_entry: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_json(path, artifact_id="live_model_triage")
    try:
        validated_payload = (
            gr00t_screening_authoritative.validate_live_model_triage_payload(payload)
        )
    except ValueError as exc:
        raise _invalid_input(_exception_message(exc)) from exc
    triage_inventory_json = validated_payload.get("dataset_inventory_json")
    if triage_inventory_json is None:
        raise _invalid_input("live_model_triage.dataset_inventory_json is required")
    resolved_inventory = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=_require_non_empty_string(
            triage_inventory_json,
            field_name="live_model_triage.dataset_inventory_json",
        ),
        field_name="live_model_triage.dataset_inventory_json",
        canonical_roots=DATASET_INVENTORY_CANONICAL_ROOTS,
    )
    expected_inventory = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=str(dataset_inventory_entry["path"]),
        field_name="dataset_inventory_bundle.path",
        canonical_roots=DATASET_INVENTORY_CANONICAL_ROOTS,
    )
    if resolved_inventory != expected_inventory:
        raise _stale_artifact(
            "live_model_triage must reference the provided dataset inventory artifact"
        )
    triage_screening_json = validated_payload.get("authoritative_screening_json")
    if triage_screening_json is None:
        raise _invalid_input(
            "live_model_triage.authoritative_screening_json is required"
        )
    resolved_screening = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=_require_non_empty_string(
            triage_screening_json,
            field_name="live_model_triage.authoritative_screening_json",
        ),
        field_name="live_model_triage.authoritative_screening_json",
        canonical_roots=SCREENING_CANONICAL_ROOTS,
    )
    expected_screening = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=str(screening_entry["path"]),
        field_name="authoritative_screening.path",
        canonical_roots=SCREENING_CANONICAL_ROOTS,
    )
    if resolved_screening != expected_screening:
        raise _stale_artifact(
            "live_model_triage must reference the provided authoritative screening artifact"
        )
    triage_result = validated_payload.get("triage_result")
    inventory_entry = {
        "artifact_id": "live_model_triage",
        "path": _repo_relative_path(repo_root, path),
        "file_sha256": apple_recap_execution_contract._sha256_file(path),
        "schema_version": validated_payload.get("schema_version"),
        "artifact_kind": validated_payload.get("artifact_kind"),
        "report_signature_sha256": validated_payload.get("report_signature_sha256"),
        "triage_status": validated_payload.get("triage_status"),
        "triage_result": triage_result,
        "reason_code": validated_payload.get("reason_code"),
        "authority_ref": apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id="live_model_triage",
            authority_role="flux_final_report_input",
            relative_path=path,
            reject_noncanonical_parts=True,
        ),
    }
    return {"payload": validated_payload, "inventory_entry": inventory_entry}


def _resolve_triage_input(
    *,
    triage_json: Path | str | None,
    repo_root: Path,
) -> tuple[Path | None, str]:
    if triage_json is not None and str(triage_json).strip():
        return (
            _resolve_authoritative_path(
                repo_root=repo_root,
                raw=triage_json,
                field_name="triage_json",
                canonical_roots=TRIAGE_CANONICAL_ROOTS,
            ),
            TRIAGE_INPUT_STATUS_EXPLICIT,
        )
    default_path = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=DEFAULT_TRIAGE_JSON,
        field_name="DEFAULT_TRIAGE_JSON",
        canonical_roots=TRIAGE_CANONICAL_ROOTS,
    )
    if default_path.exists() and default_path.is_file():
        return default_path, TRIAGE_INPUT_STATUS_AUTO_DISCOVERED
    return None, TRIAGE_INPUT_STATUS_NOT_PROVIDED


def _build_authoritative_plane(screening_payload: Mapping[str, Any]) -> dict[str, Any]:
    formal_eligibility = str(screening_payload.get("formal_eligibility") or "").strip()
    reason_code = str(screening_payload.get("reason_code") or "").strip()
    blocked = formal_eligibility != "ALLOW"
    conclusion = (
        "Current frozen authoritative screening is blocked. This preserves the active blocker only and does not imply method weakness, parity failure, or uplift failure."
        if blocked
        else "Current frozen authoritative screening passed the dataset/inference binding and servable checks. This is a narrow protocol-local positive and not a parity or uplift claim."
    )
    return {
        "status": GLOBAL_VERDICT_BLOCKED if blocked else "READY",
        "screening_mode": screening_payload.get("screening_mode"),
        "formal_eligibility": formal_eligibility,
        "reason_code": reason_code,
        "inventory_verdict": screening_payload.get("inventory_verdict"),
        "stats_fingerprint": screening_payload.get("stats_fingerprint"),
        "prompt_source": screening_payload.get("prompt_source"),
        "action_space_compatibility": screening_payload.get(
            "action_space_compatibility", {}
        ),
        "inference_model_ref": screening_payload.get("inference_model_ref"),
        "servable": bool(screening_payload.get("servable") is True),
        "conclusion": conclusion,
    }


def _triage_plane_status(triage_status: str | None) -> str:
    if triage_status == gr00t_screening_authoritative.TRIAGE_STATUS_READY:
        return GLOBAL_VERDICT_AUTHORITATIVE_POSITIVE
    if triage_status in {
        gr00t_screening_authoritative.TRIAGE_STATUS_BLOCKED,
        gr00t_screening_authoritative.TRIAGE_STATUS_STALE,
    }:
        return GLOBAL_VERDICT_BLOCKED
    return GLOBAL_VERDICT_INSUFFICIENT_EVIDENCE


def _build_triage_plane_without_payload(*, input_status: str) -> dict[str, Any]:
    return {
        "input_status": input_status,
        "status": GLOBAL_VERDICT_INSUFFICIENT_EVIDENCE,
        "triage_status": None,
        "triage_result": None,
        "reason_code": "triage_not_provided",
        "reason_basis": {"trigger": "triage_not_provided"},
        "conclusion": "Canonical live-model triage was not provided, so the final report cannot upgrade screening into an authoritative triage conclusion.",
    }


def _build_triage_plane_from_payload(
    *,
    triage_payload: Mapping[str, Any],
    input_status: str,
) -> dict[str, Any]:
    triage_status = str(triage_payload.get("triage_status") or "").strip() or None
    triage_result = triage_payload.get("triage_result")
    reason_code = str(triage_payload.get("reason_code") or "insufficient_evidence")
    status = _triage_plane_status(triage_status)
    if status == GLOBAL_VERDICT_BLOCKED:
        conclusion = "Live-model triage is blocked by the preserved blocker surface. This does not imply parity failure, paper mismatch, or method weakness."
    elif status == GLOBAL_VERDICT_INSUFFICIENT_EVIDENCE:
        conclusion = "Live-model triage remains inconclusive under the current frozen protocol because the available evidence is insufficient."
    else:
        conclusion = "Live-model triage produced a machine-readable authoritative conclusion inside the current frozen protocol. This is still narrower than any parity or uplift claim."
    return {
        "input_status": input_status,
        "status": status,
        "triage_status": triage_status,
        "triage_result": triage_result,
        "reason_code": reason_code,
        "reason_basis": triage_payload.get("reason_basis") or {},
        "conclusion": conclusion,
    }


def _build_optional_diagnostic_context(
    triage_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(triage_payload, Mapping):
        return {
            "status": OPTIONAL_CONTEXT_STATUS_NOT_PROVIDED,
            "comparison_verdict": None,
            "comparison_basis": None,
            "authority_boundary": "context_only_diagnostic",
        }
    diagnostic_reference = triage_payload.get("diagnostic_reference")
    if not isinstance(diagnostic_reference, Mapping):
        return {
            "status": OPTIONAL_CONTEXT_STATUS_NOT_PROVIDED,
            "comparison_verdict": None,
            "comparison_basis": None,
            "authority_boundary": "context_only_diagnostic",
        }
    return {
        "status": OPTIONAL_CONTEXT_STATUS_AVAILABLE,
        "comparison_verdict": diagnostic_reference.get("comparison_verdict"),
        "comparison_basis": diagnostic_reference.get("comparison_basis"),
        "authority_boundary": "context_only_diagnostic",
        "note": "Diagnostic comparison is reference-only context and never upgrades the final report global verdict.",
    }


def _build_optional_promotion_context(
    triage_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(triage_payload, Mapping):
        return {
            "status": OPTIONAL_CONTEXT_STATUS_NOT_PROVIDED,
            "promotion_allowed": None,
            "reason_code": None,
            "authority_boundary": "planning_only_non_release",
        }
    promotion_gate = triage_payload.get("promotion_gate")
    if not isinstance(promotion_gate, Mapping):
        return {
            "status": OPTIONAL_CONTEXT_STATUS_NOT_PROVIDED,
            "promotion_allowed": None,
            "reason_code": None,
            "authority_boundary": "planning_only_non_release",
        }
    return {
        "status": OPTIONAL_CONTEXT_STATUS_AVAILABLE,
        "promotion_allowed": promotion_gate.get("promotion_allowed"),
        "reason_code": promotion_gate.get("reason_code"),
        "authority_boundary": "planning_only_non_release",
        "note": "Training promotion is planning-only and non-release; it is not uplift proof and does not silently upgrade the report verdict.",
    }


def _build_optional_rtc_context(
    triage_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(triage_payload, Mapping):
        return {
            "status": OPTIONAL_CONTEXT_STATUS_NOT_PROVIDED,
            "row_id": None,
            "verdict": None,
            "reason_code": None,
            "path": None,
            "authority_boundary": "optional_runtime_context_only",
        }
    authoritative_rows = triage_payload.get("authoritative_rows")
    if not isinstance(authoritative_rows, Mapping):
        return {
            "status": OPTIONAL_CONTEXT_STATUS_NOT_PROVIDED,
            "row_id": None,
            "verdict": None,
            "reason_code": None,
            "path": None,
            "authority_boundary": "optional_runtime_context_only",
        }
    for row_id in gr00t_screening_authoritative.AUTHORITATIVE_TRIAGE_ROW_LABELS:
        row_payload = authoritative_rows.get(row_id)
        if not isinstance(row_payload, Mapping):
            continue
        execution_audit = row_payload.get("execution_audit")
        if not isinstance(execution_audit, Mapping):
            continue
        if all(
            execution_audit.get(field_name) in (None, "")
            for field_name in ("path", "verdict", "reason_code")
        ):
            continue
        return {
            "status": OPTIONAL_CONTEXT_STATUS_AVAILABLE,
            "row_id": row_id,
            "verdict": execution_audit.get("verdict"),
            "reason_code": execution_audit.get("reason_code"),
            "path": execution_audit.get("path"),
            "authority_boundary": "optional_runtime_context_only",
            "note": "RTC / execution-audit evidence is optional context only and never serves as the authoritative release or parity verdict.",
        }
    return {
        "status": OPTIONAL_CONTEXT_STATUS_NOT_PROVIDED,
        "row_id": None,
        "verdict": None,
        "reason_code": None,
        "path": None,
        "authority_boundary": "optional_runtime_context_only",
    }


def _build_non_authoritative_context(
    triage_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "diagnostic_context": _build_optional_diagnostic_context(triage_payload),
        "promotion_context": _build_optional_promotion_context(triage_payload),
        "rtc_context": _build_optional_rtc_context(triage_payload),
    }


def _build_claim_boundary() -> dict[str, Any]:
    return {
        "status": "enforced",
        "authoritative_scope": "Only the current frozen authoritative screening plane plus the current frozen live-model triage plane are report-authoritative.",
        "positive_scope": "AUTHORITATIVE_POSITIVE only means the frozen protocol produced a machine-readable authoritative conclusion. It is not parity or uplift proof.",
        "forbidden_implications": list(CLAIM_BOUNDARY_FORBIDDEN_IMPLICATIONS),
        "markdown_notice": "This report must not be used to imply OpenPI/RECAP/paper-faithful/full-online-RL parity, or to claim uplift beyond the current frozen authoritative protocol.",
    }


def _build_global_verdict(
    *,
    authoritative_plane: Mapping[str, Any],
    triage_plane: Mapping[str, Any],
) -> tuple[str, str]:
    if authoritative_plane.get("formal_eligibility") != "ALLOW":
        return (
            GLOBAL_VERDICT_BLOCKED,
            str(authoritative_plane.get("reason_code") or "authoritative_blocked"),
        )
    triage_status = str(triage_plane.get("status") or "").strip()
    if triage_status == GLOBAL_VERDICT_BLOCKED:
        return (
            GLOBAL_VERDICT_BLOCKED,
            str(triage_plane.get("reason_code") or "triage_blocked"),
        )
    if triage_status == GLOBAL_VERDICT_AUTHORITATIVE_POSITIVE:
        return (
            GLOBAL_VERDICT_AUTHORITATIVE_POSITIVE,
            str(
                triage_plane.get("triage_result")
                or triage_plane.get("reason_code")
                or "authoritative_positive"
            ),
        )
    return (
        GLOBAL_VERDICT_INSUFFICIENT_EVIDENCE,
        str(triage_plane.get("reason_code") or "insufficient_evidence"),
    )


def build_report_markdown(payload: Mapping[str, Any]) -> str:
    summary = cast(Mapping[str, Any], payload["summary"])
    artifacts = cast(Mapping[str, Mapping[str, Any]], payload["artifacts"])
    authoritative_plane = cast(Mapping[str, Any], payload["authoritative_plane"])
    triage_plane = cast(Mapping[str, Any], payload["triage_plane"])
    non_authoritative_context = cast(
        Mapping[str, Any], payload["non_authoritative_context"]
    )
    diagnostic_context = cast(
        Mapping[str, Any], non_authoritative_context["diagnostic_context"]
    )
    promotion_context = cast(
        Mapping[str, Any], non_authoritative_context["promotion_context"]
    )
    rtc_context = cast(Mapping[str, Any], non_authoritative_context["rtc_context"])
    claim_boundary = cast(Mapping[str, Any], payload["claim_boundary"])
    lines = [
        "# Flux Graft authoritative final report builder",
        "",
        "> 说明：此报告只消费 frozen authoritative screening 与 optional frozen live-model triage，",
        "> 并把 diagnostic / promotion / RTC 明确降级为 non-authoritative context。",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- global_verdict: `{payload['global_verdict']}`",
        f"- formal_eligibility: `{payload['formal_eligibility']}`",
        f"- reason_code: `{payload['reason_code']}`",
        f"- dataset_inventory_json: `{payload['dataset_inventory_json']}`",
        f"- screening_json: `{payload['screening_json']}`",
        f"- triage_json: `{payload['triage_json']}`",
        "",
        "## Global verdict",
        "",
        f"- global_verdict: `{payload['global_verdict']}`",
        f"- reason_code: `{payload['reason_code']}`",
        f"- failure_reasons: `{payload['failure_reasons']}`",
        f"- summary.live_model_triage_status: `{summary['live_model_triage']['triage_status']}`",
        f"- summary.live_model_triage_result: `{summary['live_model_triage']['triage_result']}`",
        "",
        "## Authoritative plane",
        "",
        f"- status: `{authoritative_plane['status']}`",
        f"- formal_eligibility: `{authoritative_plane['formal_eligibility']}`",
        f"- reason_code: `{authoritative_plane['reason_code']}`",
        f"- inventory_verdict: `{authoritative_plane['inventory_verdict']}`",
        f"- servable: `{authoritative_plane['servable']}`",
        f"- stats_fingerprint: `{authoritative_plane['stats_fingerprint']}`",
        f"- prompt_source: `{authoritative_plane['prompt_source']}`",
        f"- action_space_compatibility: `{cast(Mapping[str, Any], authoritative_plane['action_space_compatibility']).get('status') if isinstance(authoritative_plane.get('action_space_compatibility'), Mapping) else None}`",
        f"- inference_model_ref.surface_role: `{authoritative_plane['inference_model_ref'].get('surface_role') if isinstance(authoritative_plane['inference_model_ref'], Mapping) else None}`",
        f"- conclusion: {authoritative_plane['conclusion']}",
        "",
        "## Triage plane",
        "",
        f"- input_status: `{triage_plane['input_status']}`",
        f"- status: `{triage_plane['status']}`",
        f"- triage_status: `{triage_plane['triage_status']}`",
        f"- triage_result: `{triage_plane['triage_result']}`",
        f"- reason_code: `{triage_plane['reason_code']}`",
        f"- conclusion: {triage_plane['conclusion']}",
        "",
        "## Non-authoritative context",
        "",
        f"- diagnostic_context.status: `{diagnostic_context['status']}`",
        f"- diagnostic_context.comparison_verdict: `{diagnostic_context['comparison_verdict']}`",
        f"- promotion_context.status: `{promotion_context['status']}`",
        f"- promotion_context.promotion_allowed: `{promotion_context['promotion_allowed']}`",
        f"- promotion_context.reason_code: `{promotion_context['reason_code']}`",
        f"- rtc_context.status: `{rtc_context['status']}`",
        f"- rtc_context.row_id: `{rtc_context['row_id']}`",
        f"- rtc_context.verdict: `{rtc_context['verdict']}`",
        f"- rtc_context.reason_code: `{rtc_context['reason_code']}`",
        "",
        "## Claim boundary",
        "",
        f"- status: `{claim_boundary['status']}`",
        f"- authoritative_scope: {claim_boundary['authoritative_scope']}",
        f"- positive_scope: {claim_boundary['positive_scope']}",
        "- forbidden_implications:",
    ]
    for item in cast(Sequence[str], claim_boundary["forbidden_implications"]):
        lines.append(f"  - {item}")
    lines.extend(
        [
            "",
            "## Validated artifact inventory",
            "",
            "| artifact_id | artifact_kind | schema_version | path |",
            "| --- | --- | --- | --- |",
        ]
    )
    for artifact_id in sorted(artifacts):
        entry = artifacts[artifact_id]
        lines.append(
            "| "
            + artifact_id
            + " | "
            + str(entry.get("artifact_kind", ""))
            + " | "
            + str(entry.get("schema_version", ""))
            + " | `"
            + str(entry.get("path", ""))
            + "` |"
        )
    return "\n".join(lines) + "\n"


def build_final_report_pack(
    *,
    screening_json: Path | str = DEFAULT_SCREENING_JSON,
    dataset_inventory_json: Path | str = DEFAULT_DATASET_INVENTORY_JSON,
    triage_json: Path | str | None = None,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_inventory_json = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=dataset_inventory_json,
        field_name="dataset_inventory_json",
        canonical_roots=DATASET_INVENTORY_CANONICAL_ROOTS,
    )
    resolved_screening_json = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=screening_json,
        field_name="screening_json",
        canonical_roots=SCREENING_CANONICAL_ROOTS,
    )
    inventory_result = _validate_dataset_inventory_artifact(
        path=resolved_inventory_json,
        repo_root=repo_root,
    )
    screening_result = _validate_screening_artifact(
        path=resolved_screening_json,
        repo_root=repo_root,
        dataset_inventory_entry=cast(
            Mapping[str, Any], inventory_result["inventory_entry"]
        ),
    )
    screening_payload = cast(dict[str, Any], screening_result["payload"])
    triage_result = None
    resolved_triage_json, triage_input_status = _resolve_triage_input(
        triage_json=triage_json,
        repo_root=repo_root,
    )
    if resolved_triage_json is not None:
        triage_result = _validate_live_model_triage_artifact(
            path=resolved_triage_json,
            repo_root=repo_root,
            dataset_inventory_entry=cast(
                Mapping[str, Any], inventory_result["inventory_entry"]
            ),
            screening_entry=cast(
                Mapping[str, Any], screening_result["inventory_entry"]
            ),
        )
    authoritative_plane = _build_authoritative_plane(screening_payload)
    triage_payload = (
        None
        if triage_result is None
        else cast(Mapping[str, Any], triage_result["payload"])
    )
    triage_plane = (
        _build_triage_plane_without_payload(input_status=triage_input_status)
        if triage_payload is None
        else _build_triage_plane_from_payload(
            triage_payload=triage_payload,
            input_status=triage_input_status,
        )
    )
    non_authoritative_context = _build_non_authoritative_context(triage_payload)
    claim_boundary = _build_claim_boundary()
    global_verdict, global_reason_code = _build_global_verdict(
        authoritative_plane=authoritative_plane,
        triage_plane=triage_plane,
    )
    summary = {
        "global_verdict": global_verdict,
        "inventory_verdict": screening_payload.get("inventory_verdict"),
        "stats_fingerprint": screening_payload.get("stats_fingerprint"),
        "prompt_source": screening_payload.get("prompt_source"),
        "action_space_compatibility": screening_payload.get(
            "action_space_compatibility", {}
        ),
        "inference_model_ref": screening_payload.get("inference_model_ref"),
        "servable": bool(screening_payload.get("servable") is True),
        "live_model_triage": {
            "input_status": triage_plane.get("input_status"),
            "status": triage_plane.get("status"),
            "triage_status": triage_plane.get("triage_status"),
            "triage_result": triage_plane.get("triage_result"),
            "reason_code": triage_plane.get("reason_code"),
        },
    }
    formal_eligibility = str(screening_payload.get("formal_eligibility"))
    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "generated_at": generated_at or _now_iso(),
        "global_verdict": global_verdict,
        "formal_eligibility": formal_eligibility,
        "reason_code": global_reason_code,
        "failure_reasons": []
        if global_verdict == GLOBAL_VERDICT_AUTHORITATIVE_POSITIVE
        else [global_reason_code],
        "dataset_inventory_json": _repo_relative_path(
            repo_root, resolved_inventory_json
        ),
        "screening_json": _repo_relative_path(repo_root, resolved_screening_json),
        "triage_json": None
        if resolved_triage_json is None
        else _repo_relative_path(repo_root, resolved_triage_json),
        "artifacts": {
            "authoritative_screening": screening_result["inventory_entry"],
            "dataset_inventory_bundle": inventory_result["inventory_entry"],
        },
        "summary": summary,
        "authoritative_plane": authoritative_plane,
        "triage_plane": triage_plane,
        "non_authoritative_context": non_authoritative_context,
        "claim_boundary": claim_boundary,
        "core": {
            "global_verdict": global_verdict,
            "reason_code": global_reason_code,
            "formal_eligibility": formal_eligibility,
            "servable": summary["servable"],
        },
    }
    if triage_result is not None:
        cast(dict[str, Any], payload["artifacts"])["live_model_triage"] = triage_result[
            "inventory_entry"
        ]
    payload["core_digest"] = apple_recap_execution_contract.core_digest(
        cast(Mapping[str, Any], payload["core"])
    )
    payload["report_signature_sha256"] = _signature_for_payload(payload)
    return payload


def materialize_flux_graft_final_report(
    *,
    screening_json: Path | str = DEFAULT_SCREENING_JSON,
    dataset_inventory_json: Path | str = DEFAULT_DATASET_INVENTORY_JSON,
    triage_json: Path | str | None = None,
    out_md: Path | str = DEFAULT_OUT_MD,
    out_json: Path | str = DEFAULT_OUT_JSON,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    payload = build_final_report_pack(
        screening_json=screening_json,
        dataset_inventory_json=dataset_inventory_json,
        triage_json=triage_json,
        repo_root=repo_root,
        generated_at=generated_at,
    )
    resolved_out_md = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=out_md,
        field_name="out_md",
        canonical_roots=FINAL_REPORT_MD_CANONICAL_ROOTS,
    )
    resolved_out_json = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=out_json,
        field_name="out_json",
        canonical_roots=FINAL_REPORT_JSON_CANONICAL_ROOTS,
    )
    markdown = build_report_markdown(payload)
    _write_text(resolved_out_md, markdown)
    payload["report_artifacts"] = {
        "markdown": _repo_relative_path(repo_root, resolved_out_md),
        "json": _repo_relative_path(repo_root, resolved_out_json),
    }
    payload["report_signature_sha256"] = _signature_for_payload(payload)
    _write_json(resolved_out_json, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = materialize_flux_graft_final_report(
            screening_json=args.screening_json,
            dataset_inventory_json=args.dataset_inventory_json,
            triage_json=args.triage_json,
            out_md=args.out_md,
            out_json=args.out_json,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if payload.get("formal_eligibility") == "ALLOW" else 1


__all__ = [
    "DEFAULT_DATASET_INVENTORY_JSON",
    "DEFAULT_OUT_JSON",
    "DEFAULT_OUT_MD",
    "DEFAULT_SCREENING_JSON",
    "DEFAULT_TRIAGE_JSON",
    "REPORT_ARTIFACT_KIND",
    "REPORT_SCHEMA_VERSION",
    "build_final_report_pack",
    "build_parser",
    "build_report_markdown",
    "main",
    "materialize_flux_graft_final_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
