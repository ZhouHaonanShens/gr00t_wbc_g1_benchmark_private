#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import drop_events
from work.recap.scripts import apple_recap_execution_contract
from work.recap.scripts import build_readonly_refs
from work.recap.scripts import build_uplift_schemas
from work.recap.scripts import critic_build_episode_traces
from work.recap.scripts import critic_build_sample_pack
from work.recap.scripts import critic_scorecard_all_splits
from work.recap.scripts import gr00t_action_absorption_audit
from work.recap.scripts import gr00t_carrier_panel_gate
from work.recap.scripts import inspect_mainline_carrier
from work.recap.scripts import relabel_counterfactual_rewards
from work.recap.scripts import state_conditioned_bucket_a_import


DEFAULT_EXECUTION_ROOT = Path("agent/artifacts/apple_recap_exec")
DEFAULT_OUT_MD = Path("agent/exchange/AppleToPlate_RECAP_final_report.md")
DEFAULT_OUT_JSON = DEFAULT_EXECUTION_ROOT / "final_report" / "final_verdict_pack.json"

REPORT_SCHEMA_VERSION = "apple_recap_final_report_builder_v1"
REPORT_ARTIFACT_KIND = "apple_recap_final_report_builder"
QUESTION_IDS: tuple[str, ...] = ("Q1", "Q2", "Q3")
FREEZE_CONTEXT_FIELDS: tuple[str, ...] = (
    "execution_sha",
    "manifest_hash",
    "checkpoint_id",
    "seed_bundle_id",
    "timestamp",
)
EXECUTION_ROOT_CANONICAL_ROOTS: tuple[str, ...] = (
    apple_recap_execution_contract.DEFAULT_EXECUTION_ROOT_SCOPE,
)
FINAL_REPORT_JSON_CANONICAL_ROOTS: tuple[str, ...] = (
    "agent/artifacts/apple_recap_exec/final_report/",
)
FINAL_REPORT_MD_CANONICAL_ROOTS: tuple[str, ...] = ("agent/exchange/",)


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    relative_paths: tuple[str, ...]
    is_json: bool = True
    expected_schema_versions: tuple[str, ...] = ()
    expected_artifact_kinds: tuple[str, ...] = ()
    require_report_signature: bool = False
    require_core_digest: bool = False
    require_freshness: bool = False
    require_authority_refs: bool = False
    require_formal_eligibility: bool = False
    require_failure_reasons: bool = False


REQUIRED_ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        artifact_id="execution_contract",
        relative_paths=(
            apple_recap_execution_contract.EXECUTION_CONTRACT_JSON_NAME,
            "execution_freeze_contract.json",
        ),
        expected_schema_versions=(
            apple_recap_execution_contract.SCHEMA_VERSION,
            apple_recap_execution_contract.FINAL_SCHEMA_VERSION,
        ),
        expected_artifact_kinds=(
            apple_recap_execution_contract.ARTIFACT_KIND,
            apple_recap_execution_contract.FINAL_ARTIFACT_KIND,
        ),
        require_report_signature=True,
        require_core_digest=True,
        require_freshness=True,
        require_authority_refs=True,
    ),
    ArtifactSpec(
        artifact_id="baseline_refs_manifest",
        relative_paths=(
            f"phase_a_tooling_draft/{build_readonly_refs.DEFAULT_OUTPUT.name}",
        ),
        expected_schema_versions=(build_readonly_refs.SCHEMA_VERSION,),
        expected_artifact_kinds=(build_readonly_refs.ARTIFACT_KIND,),
        require_report_signature=True,
        require_core_digest=True,
        require_freshness=True,
        require_authority_refs=True,
    ),
    ArtifactSpec(
        artifact_id="experiment_matrix_frozen",
        relative_paths=(
            f"phase_a_tooling_draft/{build_uplift_schemas.FROZEN_MATRIX_JSON_NAME}",
        ),
        expected_schema_versions=(build_uplift_schemas.FROZEN_MATRIX_SCHEMA_VERSION,),
        expected_artifact_kinds=(build_uplift_schemas.FROZEN_MATRIX_ARTIFACT_KIND,),
        require_report_signature=True,
        require_core_digest=True,
        require_freshness=True,
        require_authority_refs=True,
    ),
    ArtifactSpec(
        artifact_id="uplift_verdict_schema",
        relative_paths=(
            f"phase_a_tooling_draft/{build_uplift_schemas.UPLIFT_VERDICT_SCHEMA_JSON_NAME}",
        ),
        expected_schema_versions=(build_uplift_schemas.UPLIFT_VERDICT_SCHEMA_VERSION,),
        expected_artifact_kinds=(build_uplift_schemas.UPLIFT_VERDICT_ARTIFACT_KIND,),
        require_report_signature=True,
        require_core_digest=True,
        require_freshness=True,
    ),
    ArtifactSpec(
        artifact_id="carrier_parity_report",
        relative_paths=(
            f"phase_a_tooling_draft/{inspect_mainline_carrier.PARITY_REPORT_JSON_NAME}",
        ),
        expected_schema_versions=(inspect_mainline_carrier.SCHEMA_VERSION,),
        require_freshness=True,
    ),
    ArtifactSpec(
        artifact_id="carrier_inspection_markdown",
        relative_paths=(
            f"phase_a_tooling_draft/{inspect_mainline_carrier.INSPECTION_MD_NAME}",
        ),
        is_json=False,
    ),
    ArtifactSpec(
        artifact_id="carrier_panel_gate",
        relative_paths=("phase_a_tooling_draft/carrier_panel_gate.json",),
        expected_schema_versions=(gr00t_carrier_panel_gate.REPORT_SCHEMA_VERSION,),
        expected_artifact_kinds=(gr00t_carrier_panel_gate.REPORT_ARTIFACT_KIND,),
        require_report_signature=True,
        require_freshness=True,
    ),
    ArtifactSpec(
        artifact_id="action_absorption_root_cause",
        relative_paths=(
            f"phase_a_tooling_draft/{gr00t_action_absorption_audit.ACTION_ABSORPTION_ROOT_CAUSE_JSON_NAME}",
        ),
        expected_schema_versions=(gr00t_action_absorption_audit.REPORT_SCHEMA_VERSION,),
        expected_artifact_kinds=(gr00t_action_absorption_audit.REPORT_ARTIFACT_KIND,),
        require_report_signature=True,
        require_freshness=True,
    ),
    ArtifactSpec(
        artifact_id="critic_scorecard_all_splits",
        relative_paths=("phase_a_tooling_draft/critic_scorecard_all_splits.json",),
        expected_schema_versions=(critic_scorecard_all_splits.SCHEMA_VERSION,),
        expected_artifact_kinds=(critic_scorecard_all_splits.ARTIFACT_KIND,),
        require_freshness=True,
    ),
    ArtifactSpec(
        artifact_id="critic_sample_pack",
        relative_paths=("phase_a_tooling_draft/critic_sample_pack.json",),
        expected_schema_versions=(critic_build_sample_pack.SCHEMA_VERSION,),
        expected_artifact_kinds=(critic_build_sample_pack.ARTIFACT_KIND,),
        require_freshness=True,
    ),
    ArtifactSpec(
        artifact_id="critic_episode_traces",
        relative_paths=("phase_a_tooling_draft/critic_episode_traces.json",),
        expected_schema_versions=(critic_build_episode_traces.SCHEMA_VERSION,),
        expected_artifact_kinds=(critic_build_episode_traces.ARTIFACT_KIND,),
        require_freshness=True,
    ),
    ArtifactSpec(
        artifact_id="run_ledger_csv",
        relative_paths=(
            f"phase_a_tooling_draft/{build_uplift_schemas.LEDGER_CSV_NAME}",
        ),
        is_json=False,
    ),
    ArtifactSpec(
        artifact_id="reward_recommendation",
        relative_paths=(
            f"reward/{relabel_counterfactual_rewards.REWARD_RECOMMENDATION_JSON_NAME}",
        ),
        expected_schema_versions=(drop_events.REWARD_RECOMMENDATION_SCHEMA_VERSION,),
        expected_artifact_kinds=(drop_events.REWARD_RECOMMENDATION_ARTIFACT_KIND,),
        require_freshness=True,
        require_formal_eligibility=True,
        require_failure_reasons=True,
    ),
    ArtifactSpec(
        artifact_id="counterfactual_reward_summary",
        relative_paths=(
            f"reward/{relabel_counterfactual_rewards.COUNTERFACTUAL_SUMMARY_JSON_NAME}",
        ),
        expected_schema_versions=(
            drop_events.COUNTERFACTUAL_REWARD_SUMMARY_SCHEMA_VERSION,
        ),
        expected_artifact_kinds=(
            drop_events.COUNTERFACTUAL_REWARD_SUMMARY_ARTIFACT_KIND,
        ),
        require_freshness=True,
    ),
    ArtifactSpec(
        artifact_id="reward_counterfactual_report_markdown",
        relative_paths=(
            f"reward/{relabel_counterfactual_rewards.REWARD_COUNTERFACTUAL_REPORT_MD_NAME}",
        ),
        is_json=False,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_apple_recap_final_report.py",
        description=(
            "Validate the AppleToPlate tooling-phase artifact pack and emit a single-file "
            "Markdown report skeleton plus a machine-readable JSON pack."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--execution-root",
        type=Path,
        default=DEFAULT_EXECUTION_ROOT,
        help="Execution-root directory that contains the full artifact collection.",
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


def _validate_existing_dir(path: Path | str, *, repo_root: Path, arg_name: str) -> Path:
    resolved = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=path,
        field_name=arg_name,
        canonical_roots=EXECUTION_ROOT_CANONICAL_ROOTS,
    )
    if not resolved.exists() or not resolved.is_dir():
        raise _invalid_input(f"{arg_name} directory does not exist: {resolved}")
    return resolved


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return state_conditioned_bucket_a_import._write_json(path, payload)


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
    return dict(payload)


def _non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _parse_timestamp(value: object, *, field_name: str) -> str:
    normalized = _require_non_empty_string(value, field_name=field_name)
    try:
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _invalid_input(f"{field_name} must be an ISO-8601 timestamp") from exc
    return normalized


def _extract_context_candidates(
    payload: Mapping[str, Any], *, field_name: str
) -> list[str]:
    candidates: list[str] = []
    if field_name in payload and payload.get(field_name) is not None:
        candidates.append(
            _require_non_empty_string(payload.get(field_name), field_name=field_name)
        )
    for container_name in ("freshness", "provenance", "freeze_context"):
        container = payload.get(container_name)
        if not isinstance(container, Mapping) or field_name not in container:
            continue
        nested_value = container.get(field_name)
        nested_field = f"{container_name}.{field_name}"
        candidates.append(
            _require_non_empty_string(nested_value, field_name=nested_field)
        )
    deduped: list[str] = []
    for item in candidates:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _extract_freeze_context(
    payload: Mapping[str, Any], *, artifact_id: str
) -> dict[str, str]:
    context: dict[str, str] = {}
    for field_name in FREEZE_CONTEXT_FIELDS:
        candidates = _extract_context_candidates(payload, field_name=field_name)
        if not candidates:
            raise _invalid_input(
                f"{artifact_id} is missing required freshness field: {field_name}"
            )
        if len(candidates) > 1:
            raise _invalid_input(
                f"{artifact_id} has conflicting freshness values for {field_name}: {candidates}"
            )
        value = candidates[0]
        context[field_name] = (
            _parse_timestamp(value, field_name=f"{artifact_id}.{field_name}")
            if field_name == "timestamp"
            else value
        )
    return context


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
            f"{artifact_id} is off-freeze: report_signature_sha256 mismatch"
        )
    return declared


def _validate_core_digest(payload: Mapping[str, Any], *, artifact_id: str) -> str:
    core = payload.get("core")
    if not isinstance(core, Mapping):
        raise _invalid_input(f"{artifact_id}.core must be an object")
    declared = _require_non_empty_string(
        payload.get("core_digest"),
        field_name=f"{artifact_id}.core_digest",
    )
    expected = apple_recap_execution_contract.core_digest(cast(Mapping[str, Any], core))
    if declared != expected:
        raise _stale_artifact(f"{artifact_id} is off-freeze: core_digest mismatch")
    return declared


def _validate_exact_value(
    payload: Mapping[str, Any],
    *,
    field_name: str,
    allowed_values: Sequence[str],
    artifact_id: str,
) -> str | None:
    if not allowed_values:
        return None
    value = _require_non_empty_string(
        payload.get(field_name),
        field_name=f"{artifact_id}.{field_name}",
    )
    if value not in set(allowed_values):
        raise _invalid_input(
            f"{artifact_id}.{field_name} mismatch: expected one of {list(allowed_values)!r}, got {value!r}"
        )
    return value


def _collect_authority_refs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    direct = payload.get("authority_ref")
    if isinstance(direct, Mapping):
        refs.append(dict(direct))
    for field_name in (
        "authority_refs",
        "source_artifacts",
        "read_only_authority_refs",
    ):
        raw = payload.get(field_name)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, Mapping):
                refs.append(dict(item))
    return refs


def _validate_authority_refs(
    payload: Mapping[str, Any],
    *,
    artifact_id: str,
    repo_root: Path,
    required: bool,
) -> list[dict[str, Any]]:
    refs = _collect_authority_refs(payload)
    if required and not refs:
        raise _invalid_input(f"{artifact_id} is missing required authority refs")
    validated: list[dict[str, Any]] = []
    for index, ref in enumerate(refs):
        ref_path = f"{artifact_id}.authority_refs[{index}]"
        artifact_ref_id = str(ref.get("artifact_id") or f"{artifact_id}_source_{index}")
        authority_role = str(ref.get("authority_role") or "upstream")
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
            canonical_roots=apple_recap_execution_contract.READ_ONLY_AUTHORITY_REF_SCOPES,
        )
        try:
            actual_ref = apple_recap_execution_contract.build_read_only_authority_ref(
                repo_root=repo_root,
                artifact_id=artifact_ref_id,
                authority_role=authority_role,
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
                f"{artifact_id} is off-freeze: authority ref digest mismatch for {normalized_relative_path}"
            )
        for optional_field in (
            "artifact_kind",
            "schema_version",
            "report_signature_sha256",
        ):
            if optional_field not in ref:
                continue
            declared_value = ref.get(optional_field)
            actual_value = actual_ref.get(optional_field)
            if declared_value != actual_value:
                raise _stale_artifact(
                    f"{artifact_id} is off-freeze: authority ref {optional_field} mismatch for {normalized_relative_path}"
                )
        validated_ref = dict(ref)
        validated_ref["relative_path"] = normalized_relative_path
        validated_ref["content_sha256"] = declared_sha
        validated.append(validated_ref)
    return validated


def _resolve_artifact_path(execution_root: Path, spec: ArtifactSpec) -> Path:
    for relative_path in spec.relative_paths:
        candidate = (execution_root / relative_path).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
    attempted = ", ".join(
        str((execution_root / item).resolve()) for item in spec.relative_paths
    )
    raise _invalid_input(
        f"missing required artifact {spec.artifact_id}: tried {attempted}"
    )


def _validate_csv_artifact(path: Path, *, artifact_id: str) -> None:
    if artifact_id != "run_ledger_csv":
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if header != list(build_uplift_schemas.LEDGER_COLUMNS):
        raise _invalid_input(
            "run_ledger_csv must preserve the frozen B0_E1_E2_run_ledger.csv header"
        )


def _validate_json_artifact(
    *,
    path: Path,
    spec: ArtifactSpec,
    repo_root: Path,
    canonical_context: Mapping[str, str] | None,
) -> dict[str, Any]:
    payload = _read_json(path, artifact_id=spec.artifact_id)
    _ = _validate_exact_value(
        payload,
        field_name="schema_version",
        allowed_values=spec.expected_schema_versions,
        artifact_id=spec.artifact_id,
    )
    _ = _validate_exact_value(
        payload,
        field_name="artifact_kind",
        allowed_values=spec.expected_artifact_kinds,
        artifact_id=spec.artifact_id,
    )

    if spec.artifact_id == "execution_contract":
        validation = (
            apple_recap_execution_contract.validate_execution_freeze_contract_draft(
                payload,
                repo_root=repo_root,
            )
        )
        if validation["formal_eligibility"] != "ALLOW":
            issues = validation.get("issues", [])
            first_issue = issues[0] if isinstance(issues, list) and issues else None
            if isinstance(first_issue, Mapping):
                reason_code = str(first_issue.get("code") or "invalid_input").strip()
                if reason_code == "noncanonical_root_contamination":
                    raise ValueError(
                        "noncanonical_root_contamination: execution_contract blocked by fail-closed validator: "
                        + str(first_issue.get("message", "unknown issue"))
                    )
                raise ValueError(
                    "invalid_input: execution_contract blocked by fail-closed validator: "
                    + str(first_issue.get("message", "unknown issue"))
                )
            raise _invalid_input("execution_contract blocked by fail-closed validator")

    report_signature = None
    if spec.require_report_signature and spec.artifact_id != "execution_contract":
        report_signature = _validate_report_signature(
            payload, artifact_id=spec.artifact_id
        )
    elif isinstance(payload.get("report_signature_sha256"), str):
        report_signature = str(payload.get("report_signature_sha256"))

    core_digest = None
    if spec.require_core_digest:
        core_digest = _validate_core_digest(payload, artifact_id=spec.artifact_id)
    elif isinstance(payload.get("core_digest"), str):
        core_digest = str(payload.get("core_digest"))

    authority_refs = _validate_authority_refs(
        payload,
        artifact_id=spec.artifact_id,
        repo_root=repo_root,
        required=spec.require_authority_refs,
    )

    freshness: dict[str, str] | None = None
    if spec.require_freshness:
        freshness = _extract_freeze_context(payload, artifact_id=spec.artifact_id)
        if canonical_context is not None:
            mismatches = [
                key
                for key, expected_value in canonical_context.items()
                if freshness.get(key) != expected_value
            ]
            if mismatches:
                raise ValueError(
                    f"stale_artifact: stale artifact {spec.artifact_id}: freeze context mismatch in {', '.join(mismatches)}"
                )

    formal_eligibility = payload.get("formal_eligibility")
    if spec.require_formal_eligibility:
        formal_eligibility = _require_non_empty_string(
            formal_eligibility,
            field_name=f"{spec.artifact_id}.formal_eligibility",
        )

    failure_reasons = payload.get("failure_reasons")
    if spec.require_failure_reasons:
        if not isinstance(failure_reasons, list) or any(
            not isinstance(item, str) or not item.strip() for item in failure_reasons
        ):
            raise _invalid_input(
                f"{spec.artifact_id}.failure_reasons must be a string list"
            )

    inventory_entry: dict[str, Any] = {
        "artifact_id": spec.artifact_id,
        "path": _repo_relative_path(repo_root, path),
        "file_sha256": apple_recap_execution_contract._sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "artifact_kind": payload.get("artifact_kind"),
        "report_signature_sha256": report_signature,
        "core_digest": core_digest,
        "formal_eligibility": formal_eligibility,
        "failure_reasons": list(failure_reasons)
        if isinstance(failure_reasons, list)
        else [],
        "freshness": freshness,
        "authority_ref": apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id=spec.artifact_id,
            authority_role="final_report_input",
            relative_path=path,
            reject_noncanonical_parts=True,
        ),
        "validated_authority_refs": authority_refs,
    }
    return {
        "payload": payload,
        "inventory_entry": inventory_entry,
    }


def _validate_artifacts(
    *,
    execution_root: Path,
    repo_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    artifacts: dict[str, dict[str, Any]] = {}

    execution_spec = REQUIRED_ARTIFACT_SPECS[0]
    execution_path = _resolve_artifact_path(execution_root, execution_spec)
    execution_result = _validate_json_artifact(
        path=execution_path,
        spec=execution_spec,
        repo_root=repo_root,
        canonical_context=None,
    )
    canonical_context = cast(
        dict[str, str],
        execution_result["inventory_entry"]["freshness"],
    )
    artifacts[execution_spec.artifact_id] = execution_result["inventory_entry"]

    for spec in REQUIRED_ARTIFACT_SPECS[1:]:
        path = _resolve_artifact_path(execution_root, spec)
        if spec.is_json:
            result = _validate_json_artifact(
                path=path,
                spec=spec,
                repo_root=repo_root,
                canonical_context=canonical_context if spec.require_freshness else None,
            )
            artifacts[spec.artifact_id] = result["inventory_entry"]
            continue
        _validate_csv_artifact(path, artifact_id=spec.artifact_id)
        artifacts[spec.artifact_id] = {
            "artifact_id": spec.artifact_id,
            "path": _repo_relative_path(repo_root, path),
            "file_sha256": apple_recap_execution_contract._sha256_file(path),
            "authority_ref": apple_recap_execution_contract.build_read_only_authority_ref(
                repo_root=repo_root,
                artifact_id=spec.artifact_id,
                authority_role="final_report_input",
                relative_path=path,
                reject_noncanonical_parts=True,
            ),
        }
    return artifacts, canonical_context


def _question_payloads() -> dict[str, dict[str, Any]]:
    return {
        "Q1": {
            "question": "RECAP 在 AppleToPlate 上是否已成立 uplift",
            "status": "tooling_placeholder",
            "reviewer_verdict": None,
            "tool_generated_summary": (
                "builder 只确认 uplift evidence surface 已齐备并且 provenance/freshness 一致，"
                "不在 tooling 阶段写正式 uplift 结论。"
            ),
            "referenced_artifact_ids": [
                "experiment_matrix_frozen",
                "run_ledger_csv",
                "carrier_panel_gate",
                "action_absorption_root_cause",
            ],
        },
        "Q2": {
            "question": "critic 学到了什么、没学到什么",
            "status": "tooling_placeholder",
            "reviewer_verdict": None,
            "tool_generated_summary": (
                "builder 只确认 critic scorecard、sample pack、episode traces 已齐备，"
                "正式 reviewer 结论留给 T17。"
            ),
            "referenced_artifact_ids": [
                "critic_scorecard_all_splits",
                "critic_sample_pack",
                "critic_episode_traces",
            ],
        },
        "Q3": {
            "question": "drop-aware reward 第一版是否应进主线，建议哪个版本",
            "status": "tooling_placeholder",
            "reviewer_verdict": None,
            "tool_generated_summary": (
                "builder 只确认 reward recommendation surface 与 counterfactual summary 已齐备，"
                "不会提前写入 mainline/reviewer 级结论。"
            ),
            "referenced_artifact_ids": [
                "reward_recommendation",
                "counterfactual_reward_summary",
                "reward_counterfactual_report_markdown",
            ],
        },
    }


def build_report_markdown(payload: Mapping[str, Any]) -> str:
    freeze_context = cast(Mapping[str, str], payload["freeze_context"])
    artifacts = cast(Mapping[str, Mapping[str, Any]], payload["artifacts"])
    questions = cast(Mapping[str, Mapping[str, Any]], payload["questions"])

    lines = [
        "# AppleToPlate RECAP final report builder skeleton",
        "",
        "> 说明：这是 T8 tooling-phase 生成的占位报告。它只确认输入 artifact 集合完整、freshness/provenance 一致，",
        "> 不在这里发布 reviewer-grade 最终结论。",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- execution_root: `{payload['execution_root']}`",
        f"- formal_eligibility: `{payload['formal_eligibility']}`",
        f"- execution_sha: `{freeze_context['execution_sha']}`",
        f"- manifest_hash: `{freeze_context['manifest_hash']}`",
        f"- checkpoint_id: `{freeze_context['checkpoint_id']}`",
        f"- seed_bundle_id: `{freeze_context['seed_bundle_id']}`",
        f"- timestamp: `{freeze_context['timestamp']}`",
        "",
        "## Validated artifact inventory",
        "",
        "| artifact_id | artifact_kind | schema_version | path |",
        "| --- | --- | --- | --- |",
    ]
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

    for question_id in QUESTION_IDS:
        question_payload = questions[question_id]
        referenced = ", ".join(
            f"`{artifact_id}`"
            for artifact_id in cast(
                Sequence[str],
                question_payload.get("referenced_artifact_ids", []),
            )
        )
        lines.extend(
            [
                "",
                f"## {question_id}. {question_payload['question']}",
                "",
                f"- status: `{question_payload['status']}`",
                f"- reviewer_verdict: `{question_payload['reviewer_verdict']}`",
                f"- tool_generated_summary: {question_payload['tool_generated_summary']}",
                f"- referenced_artifacts: {referenced}",
                "",
                "### Reviewer answer slot",
                "",
                "- [T17 pending] 这里保留给后续 reviewer-grade 正式答案。",
            ]
        )

    lines.extend(
        [
            "",
            "## Fail-closed policy",
            "",
            "- 缺失 required artifact，BLOCK。",
            "- freshness field 缺失或与 execution contract 不一致，BLOCK。",
            "- signature/digest/authority-ref backpointer 不匹配，BLOCK。",
        ]
    )
    return "\n".join(lines)


def build_final_report_pack(
    *,
    execution_root: Path | str = DEFAULT_EXECUTION_ROOT,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_execution_root = _validate_existing_dir(
        execution_root,
        repo_root=repo_root,
        arg_name="execution-root",
    )
    artifacts, freeze_context = _validate_artifacts(
        execution_root=resolved_execution_root,
        repo_root=repo_root,
    )
    questions = _question_payloads()
    core = {"commit": freeze_context["execution_sha"]}
    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "generated_at": generated_at or _now_iso(),
        "tooling_phase_only": True,
        "formal_eligibility": "ALLOW",
        "failure_reasons": [],
        "execution_root": _repo_relative_path(repo_root, resolved_execution_root),
        "freeze_context": dict(freeze_context),
        "required_artifact_ids": [spec.artifact_id for spec in REQUIRED_ARTIFACT_SPECS],
        "artifacts": artifacts,
        "questions": questions,
        "core": core,
        "core_digest": apple_recap_execution_contract.core_digest(core),
    }
    payload["report_signature_sha256"] = _signature_for_payload(payload)
    return payload


def materialize_apple_recap_final_report(
    *,
    execution_root: Path | str = DEFAULT_EXECUTION_ROOT,
    out_md: Path | str = DEFAULT_OUT_MD,
    out_json: Path | str = DEFAULT_OUT_JSON,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    payload = build_final_report_pack(
        execution_root=execution_root,
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
        payload = materialize_apple_recap_final_report(
            execution_root=args.execution_root,
            out_md=args.out_md,
            out_json=args.out_json,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_EXECUTION_ROOT",
    "DEFAULT_OUT_JSON",
    "DEFAULT_OUT_MD",
    "QUESTION_IDS",
    "REPORT_ARTIFACT_KIND",
    "REPORT_SCHEMA_VERSION",
    "REQUIRED_ARTIFACT_SPECS",
    "build_final_report_pack",
    "build_parser",
    "build_report_markdown",
    "main",
    "materialize_apple_recap_final_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
