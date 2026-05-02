#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from importlib import import_module
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


DEFAULT_CONFIG_MODULE = "configs.apple_recap.flux.gr00t_g1_flux_recap_base"
DEFAULT_OUTPUT_DIR = Path(
    "agent/artifacts/apple_recap_flux_graft/authoritative_screening"
)
DEFAULT_DATASET_INVENTORY_JSON = Path(
    "agent/artifacts/flux_dataset_probe/dataset_inventory_bundle.json"
)
AUTHORITATIVE_SCREENING_JSON_NAME = "authoritative_screening_summary.json"

SCREENING_SCHEMA_VERSION = "flux_authoritative_screening_summary_v1"
SCREENING_ARTIFACT_KIND = "flux_authoritative_screening_summary"

AUTHORITATIVE_STATE_DIM = 8
AUTHORITATIVE_ACTION_DIM = 7

JOIN_KEY_FIELDS: tuple[str, ...] = (
    "prompt_source",
    "norm_stats_source",
    "action_state_norm_source",
    "expected_embodiment_tag",
    "expected_action_space_signature",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.norm.policy import PHASE1_NORM_POLICY
from work.openpi.norm.policy import PHASE1_NORM_SOURCE
from work.recap.datasets import flux_grouped_dataset
from work.recap.models import flux_recap_vla
from work.recap.run_manifest import PROMPT_SOURCE_FIELD
from work.recap.scripts import analyze_exact_probe_failure
from work.recap.scripts import apple_recap_execution_contract
from work.recap.scripts import audit_g1_execution_surface
from work.recap.scripts import build_uplift_schemas
from work.recap.scripts.gr00t_eval_contract_gate import DEFAULT_ABSOLUTE_ACTION_KEYS
from work.recap.scripts.gr00t_eval_contract_gate import (
    DEFAULT_ACTION_REPRESENTATION_BY_KEY,
)
from work.recap.scripts.gr00t_eval_contract_gate import DEFAULT_N_ACTION_STEPS
from work.recap.scripts.gr00t_eval_contract_gate import DEFAULT_POLICY_HORIZON_EXPECTED
from work.recap.scripts.gr00t_eval_contract_gate import DEFAULT_RELATIVE_ACTION_KEYS
from work.recap.scripts.gr00t_eval_contract_gate import build_eval_contract_freeze
from work.recap.scripts import gr00t_screening_probe_bypass_diagnostic
from work.recap.scripts import gr00t_training_promotion_gate


DEFAULT_LIVE_MODEL_TRIAGE_OUTPUT_DIR = Path(
    "agent/artifacts/apple_recap_flux_graft/live_model_triage"
)
LIVE_MODEL_TRIAGE_JSON_NAME = "triage_result.json"
LIVE_MODEL_TRIAGE_SCHEMA_VERSION = "flux_live_model_triage_v1"
LIVE_MODEL_TRIAGE_ARTIFACT_KIND = "flux_live_model_triage"

TRIAGE_STATUS_READY = "READY"
TRIAGE_STATUS_BLOCKED = "BLOCKED"
TRIAGE_STATUS_STALE = "STALE"
TRIAGE_STATUS_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

TRIAGE_RESULT_PROBE_FAILED_BUT_B0_NONZERO = "probe_failed_but_b0_nonzero"
TRIAGE_RESULT_PROBE_FAILED_AND_B0_ZERO = "probe_failed_and_b0_zero"
TRIAGE_RESULT_PROBE_PASSED_B0_NONZERO = "probe_passed_b0_nonzero"
TRIAGE_RESULT_PROBE_PASSED_E1_SIGNAL_PRESENT = "probe_passed_e1_signal_present"
TRIAGE_RESULT_PROBE_PASSED_E1E2_FLAT = "probe_passed_e1e2_flat"

TRIAGE_RESULT_ENUM: tuple[str, ...] = (
    TRIAGE_RESULT_PROBE_FAILED_BUT_B0_NONZERO,
    TRIAGE_RESULT_PROBE_FAILED_AND_B0_ZERO,
    TRIAGE_RESULT_PROBE_PASSED_B0_NONZERO,
    TRIAGE_RESULT_PROBE_PASSED_E1_SIGNAL_PRESENT,
    TRIAGE_RESULT_PROBE_PASSED_E1E2_FLAT,
)
TRIAGE_STATUS_ENUM: tuple[str, ...] = (
    TRIAGE_STATUS_READY,
    TRIAGE_STATUS_BLOCKED,
    TRIAGE_STATUS_STALE,
    TRIAGE_STATUS_INSUFFICIENT_EVIDENCE,
)

AUTHORITATIVE_TRIAGE_ROW_LABELS: tuple[str, ...] = (
    build_uplift_schemas.LEDGER_REQUIRED_ROW_LABELS[0],
    build_uplift_schemas.LEDGER_REQUIRED_ROW_LABELS[1],
    build_uplift_schemas.LEDGER_REQUIRED_ROW_LABELS[2],
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_screening_authoritative.py",
        description=(
            "Materialize the authoritative Flux screening surface while enforcing "
            "dataset inventory provenance, train-vs-inference model split semantics, "
            "and pre-runtime binding-join gates."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-module",
        type=str,
        default=DEFAULT_CONFIG_MODULE,
        help="Python module that exposes CONFIG and/or build_config_surface().",
    )
    parser.add_argument(
        "--dataset-inventory-json",
        type=Path,
        default=DEFAULT_DATASET_INVENTORY_JSON,
        help="Task 7 dataset_inventory_bundle.json consumed before runtime materialization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives authoritative_screening_summary.json.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _signature_for_payload(payload: Mapping[str, Any]) -> str:
    signature_basis = {
        str(key): value
        for key, value in dict(payload).items()
        if key != "report_signature_sha256"
    }
    return apple_recap_execution_contract._sha256_payload(signature_basis)


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)
    return path


def _validate_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output-dir must be a directory path: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


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


def _load_config_module(module_name: str):
    normalized = str(module_name).strip()
    if not normalized:
        raise ValueError("config-module must be a non-empty import path")
    return import_module(normalized)


def _surface_from_config_payload(
    *,
    payload: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    if "train_model_ref" in payload and "inference_model_ref" in payload:
        validation = flux_recap_vla.validate_flux_recap_model_surface(
            payload,
            repo_root=repo_root,
        )
        if validation["formal_eligibility"] != "ALLOW":
            first_issue = validation["issues"][0] if validation["issues"] else None
            if isinstance(first_issue, Mapping):
                raise ValueError(
                    str(first_issue.get("message", "blocked config surface"))
                )
            raise ValueError("blocked config surface")
        return dict(validation["normalized_surface"])
    return flux_recap_vla.build_flux_recap_model_surface(
        repo_root=repo_root,
        variant_id=(
            str(payload.get("variant_id"))
            if payload.get("variant_id") is not None
            else None
        ),
        pinned_fluxvla_commit=str(payload.get("pinned_fluxvla_commit") or ""),
        train_model_spec=payload.get("train_model", {}),
        inference_model_spec=payload.get("inference_model", {}),
    )


def load_authoritative_config_surface(
    *,
    config_module: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    module = _load_config_module(config_module)
    build_surface = getattr(module, "build_config_surface", None)
    if callable(build_surface):
        surface = build_surface(repo_root=repo_root)
        if not isinstance(surface, Mapping):
            raise TypeError(
                f"{config_module}.build_config_surface() must return an object"
            )
        return dict(surface)
    raw_payload = getattr(module, "CONFIG", None)
    if not isinstance(raw_payload, Mapping):
        raise ValueError(
            f"{config_module} must expose CONFIG or build_config_surface()"
        )
    return _surface_from_config_payload(payload=raw_payload, repo_root=repo_root)


def _issue(
    code: str,
    field_path: str,
    message: str,
    *,
    expected: object | None = None,
    observed: object | None = None,
) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "code": str(code),
        "field_path": str(field_path),
        "message": str(message),
    }
    if expected is not None:
        issue["expected"] = expected
    if observed is not None:
        issue["observed"] = observed
    return issue


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return dict(cast(Mapping[str, Any], payload))


def _safe_blocking_reasons(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("blocking_reasons")
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            normalized.append(dict(cast(Mapping[str, Any], item)))
    return normalized


def _inventory_prompt_source(payload: Mapping[str, Any]) -> str | None:
    prompt_source = payload.get("prompt_source")
    if isinstance(prompt_source, Mapping):
        candidate = str(prompt_source.get("prompt_source_field") or "").strip()
        if candidate:
            return candidate
    binding_join_contract = payload.get("binding_join_contract")
    if isinstance(binding_join_contract, Mapping):
        candidate = str(binding_join_contract.get("prompt_source") or "").strip()
        if candidate:
            return candidate
    return None


def _inventory_join_contract(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = payload.get("binding_join_contract")
    if not isinstance(raw, Mapping):
        return None
    return dict(cast(Mapping[str, Any], raw))


def _build_authoritative_action_space_contract() -> dict[str, Any]:
    eval_contract = build_eval_contract_freeze()
    server_contract = cast(Mapping[str, Any], eval_contract["server_contract"])
    return {
        "expected_embodiment_tag": str(server_contract["embodiment_tag"]),
        "policy_horizon_expected": int(DEFAULT_POLICY_HORIZON_EXPECTED),
        "n_action_steps": int(DEFAULT_N_ACTION_STEPS),
        "state_dim": int(AUTHORITATIVE_STATE_DIM),
        "action_dim": int(AUTHORITATIVE_ACTION_DIM),
        "relative_action_keys": list(DEFAULT_RELATIVE_ACTION_KEYS),
        "absolute_action_keys": list(DEFAULT_ABSOLUTE_ACTION_KEYS),
        "action_representation_by_key": dict(DEFAULT_ACTION_REPRESENTATION_BY_KEY),
    }


def build_authoritative_binding_join_contract() -> dict[str, Any]:
    action_space_contract = _build_authoritative_action_space_contract()
    return {
        "prompt_source": PROMPT_SOURCE_FIELD,
        "norm_stats_source": PHASE1_NORM_SOURCE,
        "action_state_norm_source": PHASE1_NORM_POLICY,
        "expected_embodiment_tag": action_space_contract["expected_embodiment_tag"],
        "expected_action_space_signature": apple_recap_execution_contract._sha256_payload(
            action_space_contract
        ),
        "expected_action_space_contract": action_space_contract,
    }


def _inventory_gate_result(
    *,
    dataset_inventory_json: Path | str,
    repo_root: Path,
) -> dict[str, Any]:
    resolved_path = _resolve_repo_path(repo_root, dataset_inventory_json)
    issues: list[dict[str, Any]] = []
    payload: dict[str, Any] | None = None
    authority_ref: dict[str, Any] | None = None
    try:
        if not resolved_path.exists() or not resolved_path.is_file():
            raise FileNotFoundError(resolved_path)
        payload = _read_json_object(resolved_path)
    except FileNotFoundError:
        issues.append(
            _issue(
                "missing_inventory_artifact",
                "dataset_inventory_json",
                "dataset inventory artifact is required before authoritative runtime screening",
                observed=str(resolved_path),
            )
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(
            _issue(
                "invalid_inventory_artifact",
                "dataset_inventory_json",
                "dataset inventory artifact is malformed",
                observed=_exception_message(exc),
            )
        )

    if payload is not None:
        if payload.get("schema_version") != flux_grouped_dataset.SCHEMA_VERSION:
            issues.append(
                _issue(
                    "invalid_inventory_artifact",
                    "dataset_inventory_bundle.schema_version",
                    "dataset inventory schema_version does not match the Task 7 authoritative surface",
                    expected=flux_grouped_dataset.SCHEMA_VERSION,
                    observed=payload.get("schema_version"),
                )
            )
        if payload.get("artifact_kind") != flux_grouped_dataset.ARTIFACT_KIND:
            issues.append(
                _issue(
                    "invalid_inventory_artifact",
                    "dataset_inventory_bundle.artifact_kind",
                    "dataset inventory artifact_kind does not match the Task 7 authoritative surface",
                    expected=flux_grouped_dataset.ARTIFACT_KIND,
                    observed=payload.get("artifact_kind"),
                )
            )

        verdict = str(payload.get("verdict") or "").strip()
        if verdict != flux_grouped_dataset.VERDICT_COMPLETE:
            issues.append(
                _issue(
                    "inventory_verdict_blocked",
                    "dataset_inventory_bundle.verdict",
                    "dataset inventory must be inventory-complete before authoritative rerun",
                    expected=flux_grouped_dataset.VERDICT_COMPLETE,
                    observed=verdict or None,
                )
            )

        blocking_reasons = _safe_blocking_reasons(payload)
        if blocking_reasons:
            issues.append(
                _issue(
                    "inventory_blocking_reasons_present",
                    "dataset_inventory_bundle.blocking_reasons",
                    "dataset inventory declares blocking reasons and cannot enter authoritative rerun comparison",
                    observed=[reason.get("code") for reason in blocking_reasons],
                )
            )

        join_contract = _inventory_join_contract(payload)
        if join_contract is None:
            issues.append(
                _issue(
                    "missing_inventory_join_contract",
                    "dataset_inventory_bundle.binding_join_contract",
                    "dataset inventory must expose binding_join_contract before authoritative comparison",
                )
            )
        else:
            for field_name in JOIN_KEY_FIELDS:
                value = join_contract.get(field_name)
                if value in (None, ""):
                    issues.append(
                        _issue(
                            "missing_inventory_join_key",
                            f"dataset_inventory_bundle.binding_join_contract.{field_name}",
                            "dataset inventory binding_join_contract is missing a required authoritative join key",
                        )
                    )

        try:
            authority_ref = (
                apple_recap_execution_contract.build_read_only_authority_ref(
                    repo_root=repo_root,
                    artifact_id="dataset_inventory_bundle",
                    authority_role="authoritative_rerun_dataset_inventory",
                    relative_path=resolved_path,
                )
            )
        except (OSError, TypeError, ValueError):
            authority_ref = None

    return {
        "dataset_inventory_json": str(resolved_path),
        "payload": payload,
        "issues": issues,
        "authority_ref": authority_ref,
        "inventory_verdict": (
            str(payload.get("verdict"))
            if isinstance(payload, Mapping)
            else "inventory-missing"
        ),
        "dataset_fingerprint": (
            payload.get("dataset_fingerprint") if isinstance(payload, Mapping) else None
        ),
        "stats_fingerprint": (
            payload.get("stats_fingerprint") if isinstance(payload, Mapping) else None
        ),
        "prompt_source": _inventory_prompt_source(payload or {}),
        "schema_compatibility": (
            dict(cast(Mapping[str, Any], payload.get("schema_compatibility")))
            if isinstance(payload, Mapping)
            and isinstance(payload.get("schema_compatibility"), Mapping)
            else None
        ),
        "blocking_reasons": _safe_blocking_reasons(payload or {}),
        "binding_join_contract": _inventory_join_contract(payload or {}),
    }


def _model_binding_result(
    *,
    config_module: str,
    repo_root: Path,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    model_surface: dict[str, Any] | None = None
    materialization_summary: dict[str, Any] | None = None
    try:
        model_surface = load_authoritative_config_surface(
            config_module=config_module,
            repo_root=repo_root,
        )
    except Exception as exc:
        issues.append(
            _issue(
                "invalid_model_binding_surface",
                "config_module",
                "failed to load authoritative model binding surface",
                observed=_exception_message(exc),
            )
        )
    else:
        try:
            materialized = flux_recap_vla.materialize_authoritative_inference_surface(
                model_surface,
                repo_root=repo_root,
            )
        except Exception as exc:
            issues.append(
                _issue(
                    "invalid_model_binding_surface",
                    "inference_model_ref",
                    "authoritative consumer failed to materialize the inference model surface",
                    observed=_exception_message(exc),
                )
            )
        else:
            materialization_summary = dict(materialized["materialization_summary"])

    if materialization_summary is not None:
        inference_model_ref = materialization_summary.get("inference_model_ref")
        materialized_model_ref = materialization_summary.get("materialized_model_ref")
        if not isinstance(inference_model_ref, Mapping):
            issues.append(
                _issue(
                    "missing_inference_model_ref",
                    "inference_model_ref",
                    "authoritative screening must record inference_model_ref",
                )
            )
        if (
            materialization_summary.get("materialized_surface_role")
            != flux_recap_vla.INFERENCE_MODEL_ROLE
        ):
            issues.append(
                _issue(
                    "non_authoritative_materialized_surface",
                    "materialized_surface_role",
                    "authoritative consumer must materialize only the inference surface",
                    expected=flux_recap_vla.INFERENCE_MODEL_ROLE,
                    observed=materialization_summary.get("materialized_surface_role"),
                )
            )
        if (
            materialization_summary.get("materialized_registered_class")
            != flux_recap_vla.INFERENCE_MODEL_CLASS
        ):
            issues.append(
                _issue(
                    "non_authoritative_materialized_surface",
                    "materialized_registered_class",
                    "authoritative consumer must materialize the inference registered class",
                    expected=flux_recap_vla.INFERENCE_MODEL_CLASS,
                    observed=materialization_summary.get(
                        "materialized_registered_class"
                    ),
                )
            )
        if materialization_summary.get("train_model_materialized") is not False:
            issues.append(
                _issue(
                    "train_model_materialized",
                    "train_model_materialized",
                    "authoritative consumer must never materialize train_model_ref",
                    expected=False,
                    observed=materialization_summary.get("train_model_materialized"),
                )
            )
        if materialization_summary.get("inference_model_materialized") is not True:
            issues.append(
                _issue(
                    "inference_model_not_materialized",
                    "inference_model_materialized",
                    "authoritative consumer must materialize inference_model_ref",
                    expected=True,
                    observed=materialization_summary.get(
                        "inference_model_materialized"
                    ),
                )
            )
        if inference_model_ref != materialized_model_ref:
            issues.append(
                _issue(
                    "materialized_model_ref_mismatch",
                    "materialized_model_ref",
                    "authoritative consumer must materialize exactly inference_model_ref",
                    expected=inference_model_ref,
                    observed=materialized_model_ref,
                )
            )

    return {
        "issues": issues,
        "model_surface": model_surface,
        "materialization_summary": materialization_summary,
        "binding_join_contract": build_authoritative_binding_join_contract(),
    }


def _binding_join_result(
    *,
    inventory_contract: Mapping[str, Any] | None,
    model_contract: Mapping[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    mismatched_fields: list[str] = []
    for field_name in JOIN_KEY_FIELDS:
        expected_value = model_contract.get(field_name)
        observed_value = (
            inventory_contract.get(field_name) if inventory_contract else None
        )
        if observed_value != expected_value:
            mismatched_fields.append(field_name)
            issues.append(
                _issue(
                    "binding_join_mismatch",
                    f"binding_join_contract.{field_name}",
                    "dataset inventory join key does not match the authoritative inference binding surface",
                    expected=expected_value,
                    observed=observed_value,
                )
            )
    return {
        "issues": issues,
        "matched": not issues,
        "mismatched_fields": mismatched_fields,
    }


def _action_space_compatibility(
    *,
    inventory_failed: bool,
    model_failed: bool,
    inventory_contract: Mapping[str, Any] | None,
    model_contract: Mapping[str, Any],
    join_issues: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if inventory_failed or model_failed:
        status = "blocked"
    else:
        mismatch_fields = {
            str(item.get("field_path"))
            for item in join_issues
            if isinstance(item, Mapping)
        }
        status = (
            "mismatch"
            if {
                "binding_join_contract.expected_embodiment_tag",
                "binding_join_contract.expected_action_space_signature",
            }
            & mismatch_fields
            else "compatible"
        )
    return {
        "status": status,
        "expected_embodiment_tag": model_contract.get("expected_embodiment_tag"),
        "observed_embodiment_tag": (
            inventory_contract.get("expected_embodiment_tag")
            if inventory_contract
            else None
        ),
        "expected_action_space_signature": model_contract.get(
            "expected_action_space_signature"
        ),
        "observed_action_space_signature": (
            inventory_contract.get("expected_action_space_signature")
            if inventory_contract
            else None
        ),
        "expected_action_space_contract": model_contract.get(
            "expected_action_space_contract"
        ),
    }


def materialize_authoritative_screening(
    *,
    config_module: str = DEFAULT_CONFIG_MODULE,
    dataset_inventory_json: Path | str = DEFAULT_DATASET_INVENTORY_JSON,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    resolved_output_dir = _validate_output_dir(Path(output_dir))
    inventory_result = _inventory_gate_result(
        dataset_inventory_json=dataset_inventory_json,
        repo_root=repo_root,
    )
    model_result = _model_binding_result(
        config_module=config_module,
        repo_root=repo_root,
    )

    inventory_failed = bool(inventory_result["issues"])
    model_failed = bool(model_result["issues"])

    join_result: dict[str, Any] = {
        "issues": [],
        "matched": False,
        "mismatched_fields": [],
    }
    if not inventory_failed and not model_failed:
        join_result = _binding_join_result(
            inventory_contract=cast(
                Mapping[str, Any] | None,
                inventory_result["binding_join_contract"],
            ),
            model_contract=cast(
                Mapping[str, Any], model_result["binding_join_contract"]
            ),
        )

    if inventory_failed:
        formal_eligibility = "BLOCK"
        reason_code = "inventory_blocked"
        issues = list(cast(list[dict[str, Any]], inventory_result["issues"]))
    elif model_failed:
        formal_eligibility = "BLOCK"
        reason_code = "model_binding_blocked"
        issues = list(cast(list[dict[str, Any]], model_result["issues"]))
    elif join_result["issues"]:
        formal_eligibility = "BLOCK"
        reason_code = "binding_join_blocked"
        issues = list(cast(list[dict[str, Any]], join_result["issues"]))
    else:
        formal_eligibility = "ALLOW"
        reason_code = "ok"
        issues = []

    materialization_summary = cast(
        dict[str, Any] | None,
        model_result["materialization_summary"],
    )
    authoritative_binding_join_contract = dict(
        cast(Mapping[str, Any], model_result["binding_join_contract"])
    )
    inventory_binding_join_contract = cast(
        dict[str, Any] | None,
        inventory_result["binding_join_contract"],
    )

    servable = bool(
        formal_eligibility == "ALLOW"
        and materialization_summary is not None
        and materialization_summary.get("inference_model_materialized") is True
        and materialization_summary.get("train_model_materialized") is False
        and materialization_summary.get("materialized_model_ref")
        == materialization_summary.get("inference_model_ref")
    )

    payload: dict[str, Any] = {
        "schema_version": SCREENING_SCHEMA_VERSION,
        "artifact_kind": SCREENING_ARTIFACT_KIND,
        "screening_mode": "authoritative",
        "config_module": str(config_module),
        "dataset_inventory_json": str(inventory_result["dataset_inventory_json"]),
        "output_dir": str(resolved_output_dir),
        "artifact_path": str(resolved_output_dir / AUTHORITATIVE_SCREENING_JSON_NAME),
        "formal_eligibility": formal_eligibility,
        "reason_code": reason_code,
        "issues": issues,
        "inventory_verdict": inventory_result["inventory_verdict"],
        "inventory_blocking_reasons": list(
            cast(list[dict[str, Any]], inventory_result["blocking_reasons"])
        ),
        "dataset_fingerprint": inventory_result["dataset_fingerprint"],
        "stats_fingerprint": inventory_result["stats_fingerprint"],
        "prompt_source": inventory_result["prompt_source"],
        "schema_compatibility": inventory_result["schema_compatibility"],
        "inventory_binding_join_contract": inventory_binding_join_contract,
        "authoritative_binding_join_contract": authoritative_binding_join_contract,
        "binding_join_evaluation": {
            "matched": bool(join_result["matched"]),
            "mismatched_fields": list(join_result["mismatched_fields"]),
            "join_key_fields": list(JOIN_KEY_FIELDS),
        },
        "action_space_compatibility": _action_space_compatibility(
            inventory_failed=inventory_failed,
            model_failed=model_failed,
            inventory_contract=inventory_binding_join_contract,
            model_contract=authoritative_binding_join_contract,
            join_issues=cast(list[Mapping[str, Any]], join_result["issues"]),
        ),
        "servable": servable,
        "source_artifacts": (
            [inventory_result["authority_ref"]]
            if inventory_result["authority_ref"] is not None
            else []
        ),
    }
    if materialization_summary is not None:
        payload.update(materialization_summary)
    payload["schema_version"] = SCREENING_SCHEMA_VERSION
    payload["artifact_kind"] = SCREENING_ARTIFACT_KIND
    payload["report_signature_sha256"] = _signature_for_payload(payload)
    _ = _write_json(resolved_output_dir / AUTHORITATIVE_SCREENING_JSON_NAME, payload)
    return payload


def validate_authoritative_screening_summary_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(cast(Mapping[str, Any], payload))
    if normalized.get("schema_version") != SCREENING_SCHEMA_VERSION:
        raise ValueError(
            "authoritative_screening.schema_version mismatch: expected "
            + repr(SCREENING_SCHEMA_VERSION)
        )
    if normalized.get("artifact_kind") != SCREENING_ARTIFACT_KIND:
        raise ValueError(
            "authoritative_screening.artifact_kind mismatch: expected "
            + repr(SCREENING_ARTIFACT_KIND)
        )
    if normalized.get("screening_mode") != "authoritative":
        raise ValueError(
            "authoritative_screening.screening_mode mismatch: expected 'authoritative'"
        )
    formal_eligibility = str(normalized.get("formal_eligibility") or "").strip()
    if formal_eligibility not in {"ALLOW", "BLOCK"}:
        raise ValueError(
            "authoritative_screening.formal_eligibility must be 'ALLOW' or 'BLOCK'"
        )
    reason_code = str(normalized.get("reason_code") or "").strip()
    if reason_code not in {
        "ok",
        "inventory_blocked",
        "model_binding_blocked",
        "binding_join_blocked",
    }:
        raise ValueError("authoritative_screening.reason_code is not recognized")
    issues = normalized.get("issues")
    if not isinstance(issues, list) or any(
        not isinstance(item, Mapping) for item in issues
    ):
        raise ValueError("authoritative_screening.issues must be a list of objects")
    servable = normalized.get("servable")
    if not isinstance(servable, bool):
        raise ValueError("authoritative_screening.servable must be a bool")
    declared_signature = str(normalized.get("report_signature_sha256") or "").strip()
    expected_signature = _signature_for_payload(normalized)
    if declared_signature != expected_signature:
        raise ValueError(
            "authoritative_screening.report_signature_sha256 mismatch: expected "
            + repr(expected_signature)
        )
    return normalized


def _read_json_optional(
    path: Path | str | None,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any] | None, Path | None]:
    if path is None:
        return None, None
    raw = str(path).strip()
    if not raw:
        return None, None
    resolved = _resolve_repo_path(repo_root, raw)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(resolved)
    return _read_json_object(resolved), resolved


def _read_jsonl_optional(
    path: Path | str | None,
    *,
    repo_root: Path,
) -> tuple[list[dict[str, Any]] | None, Path | None]:
    if path is None:
        return None, None
    raw = str(path).strip()
    if not raw:
        return None, None
    resolved = _resolve_repo_path(repo_root, raw)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(resolved)
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        resolved.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"expected JSON object in {resolved}:{line_number}, got {type(payload).__name__}"
            )
        rows.append(dict(cast(Mapping[str, Any], payload)))
    return rows, resolved


def _safe_read_json_optional(
    path: Path | str | None,
    *,
    repo_root: Path,
    source_artifact: str,
    field_path: str,
    message: str,
    code: str = "invalid_optional_artifact",
) -> tuple[dict[str, Any] | None, Path | None, list[dict[str, Any]]]:
    try:
        payload, resolved = _read_json_optional(path, repo_root=repo_root)
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return (
            None,
            None,
            [
                _issue_with_source(
                    code,
                    field_path,
                    message,
                    source_artifact=source_artifact,
                    observed=_exception_message(exc),
                )
            ],
        )
    return payload, resolved, []


def _safe_read_jsonl_optional(
    path: Path | str | None,
    *,
    repo_root: Path,
    source_artifact: str,
    field_path: str,
    message: str,
    code: str = "invalid_optional_artifact",
) -> tuple[list[dict[str, Any]] | None, Path | None, list[dict[str, Any]]]:
    try:
        payload, resolved = _read_jsonl_optional(path, repo_root=repo_root)
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return (
            None,
            None,
            [
                _issue_with_source(
                    code,
                    field_path,
                    message,
                    source_artifact=source_artifact,
                    observed=_exception_message(exc),
                )
            ],
        )
    return payload, resolved, []


def _issue_with_source(
    code: str,
    field_path: str,
    message: str,
    *,
    source_artifact: str,
    expected: object | None = None,
    observed: object | None = None,
    detail: object | None = None,
) -> dict[str, Any]:
    payload = _issue(
        code,
        field_path,
        message,
        expected=expected,
        observed=observed,
    )
    payload["source_artifact"] = str(source_artifact)
    if detail is not None:
        payload["detail"] = detail
    return payload


def _safe_source_artifacts(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("source_artifacts")
    if not isinstance(raw, list):
        return []
    return [
        dict(cast(Mapping[str, Any], item)) for item in raw if isinstance(item, Mapping)
    ]


def _build_source_ref(
    *,
    repo_root: Path,
    path: Path | None,
    artifact_id: str,
    authority_role: str,
) -> dict[str, Any] | None:
    if path is None:
        return None
    return apple_recap_execution_contract.build_read_only_authority_ref(
        repo_root=repo_root,
        artifact_id=artifact_id,
        authority_role=authority_role,
        relative_path=path,
    )


def _relative_or_none(repo_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    return _repo_relative_path(repo_root, path)


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_optional_issue_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        dict(cast(Mapping[str, Any], item))
        for item in value
        if isinstance(item, Mapping)
    ]


def _load_exact_probe_surface(
    *,
    exact_probe_json: Path | str | None,
    repo_root: Path,
) -> dict[str, Any]:
    payload, path, load_issues = _safe_read_json_optional(
        exact_probe_json,
        repo_root=repo_root,
        source_artifact="exact_probe",
        field_path="exact_probe_json",
        message="exact probe artifact could not be loaded",
    )
    issues: list[dict[str, Any]] = []
    issues.extend(load_issues)
    if payload is None and path is None and load_issues:
        return {
            "status": "UNKNOWN",
            "reason_code": "exact_probe_invalid",
            "payload": None,
            "path": None,
            "issues": issues,
            "source_artifact": None,
        }
    if payload is None:
        return {
            "status": "MISSING",
            "reason_code": "exact_probe_missing",
            "payload": None,
            "path": None,
            "issues": [
                _issue_with_source(
                    "exact_probe_missing",
                    "exact_probe_json",
                    "exact probe artifact is required before conclusive live-model triage",
                    source_artifact="exact_probe",
                )
            ],
            "source_artifact": None,
        }
    probe_status = str(payload.get("probe_status") or "").strip().upper()
    if not probe_status:
        if isinstance(payload.get("probe_passed"), bool):
            probe_status = "PASS" if bool(payload.get("probe_passed")) else "FAIL"
        else:
            formal_eligibility = str(payload.get("formal_eligibility") or "").strip()
            status = str(payload.get("status") or "").strip().upper()
            reason_code = str(payload.get("reason_code") or "").strip()
            success_rate = _optional_float(payload.get("success_rate"))
            if (
                formal_eligibility == "BLOCK"
                or reason_code == "triplet_binding_blocked"
            ):
                probe_status = "BLOCKED"
            elif status == "PASS" or formal_eligibility == "ALLOW":
                probe_status = "PASS"
            elif status == "FAIL":
                probe_status = "FAIL"
            elif success_rate is not None:
                probe_status = "PASS" if success_rate > 0.0 else "FAIL"
            else:
                probe_status = "UNKNOWN"
    if probe_status not in {"PASS", "FAIL", "BLOCKED", "UNKNOWN"}:
        issues.append(
            _issue_with_source(
                "exact_probe_status_invalid",
                "exact_probe.probe_status",
                "exact probe surface must resolve to PASS, FAIL, BLOCKED, or UNKNOWN",
                source_artifact="exact_probe",
                observed=probe_status,
            )
        )
        probe_status = "UNKNOWN"
    return {
        "status": probe_status,
        "reason_code": str(
            payload.get("reason_code") or "exact_probe_status_unknown"
        ).strip()
        or "exact_probe_status_unknown",
        "payload": payload,
        "path": path,
        "issues": issues,
        "source_artifact": _build_source_ref(
            repo_root=repo_root,
            path=path,
            artifact_id="exact_probe",
            authority_role="live_model_triage_input",
        ),
    }


def _load_diagnostic_surface(
    *,
    diagnostic_json: Path | str | None,
    repo_root: Path,
) -> dict[str, Any]:
    payload, path, issues = _safe_read_json_optional(
        diagnostic_json,
        repo_root=repo_root,
        source_artifact="diagnostic_probe_bypass",
        field_path="diagnostic_json",
        message="diagnostic probe-bypass artifact could not be loaded",
    )
    if issues:
        return {
            "status": "INVALID_IGNORED",
            "payload": None,
            "path": None,
            "issues": issues,
            "source_artifact": None,
        }
    if payload is None:
        return {
            "status": "MISSING",
            "payload": None,
            "path": None,
            "issues": [],
            "source_artifact": None,
        }
    try:
        validated = gr00t_screening_probe_bypass_diagnostic.validate_probe_bypass_diagnostic_payload(
            payload
        )
    except ValueError as exc:
        return {
            "status": "INVALID_IGNORED",
            "payload": payload,
            "path": path,
            "issues": [
                _issue_with_source(
                    "invalid_optional_artifact",
                    "diagnostic_probe_bypass",
                    "diagnostic probe-bypass artifact was ignored because it failed validation",
                    source_artifact="diagnostic_probe_bypass",
                    observed=_exception_message(exc),
                )
            ],
            "source_artifact": _build_source_ref(
                repo_root=repo_root,
                path=path,
                artifact_id="diagnostic_probe_bypass",
                authority_role="live_model_triage_optional_input",
            ),
        }
    return {
        "status": "READY",
        "payload": validated,
        "path": path,
        "issues": [],
        "source_artifact": _build_source_ref(
            repo_root=repo_root,
            path=path,
            artifact_id="diagnostic_probe_bypass",
            authority_role="live_model_triage_optional_input",
        ),
    }


def _load_promotion_gate_surface(
    *,
    promotion_gate_json: Path | str | None,
    repo_root: Path,
) -> dict[str, Any]:
    payload, path, issues = _safe_read_json_optional(
        promotion_gate_json,
        repo_root=repo_root,
        source_artifact="training_promotion_gate",
        field_path="promotion_gate_json",
        message="training promotion gate artifact could not be loaded",
        code="gate_input_invalid",
    )
    if issues:
        return {
            "status": "INVALID",
            "payload": None,
            "path": None,
            "issues": issues,
            "source_artifact": None,
        }
    if payload is None:
        return {
            "status": "MISSING",
            "payload": None,
            "path": None,
            "issues": [
                _issue_with_source(
                    "promotion_gate_missing",
                    "promotion_gate_json",
                    "training promotion gate is required before training-related live-model triage conclusions",
                    source_artifact="training_promotion_gate",
                )
            ],
            "source_artifact": None,
        }
    issues: list[dict[str, Any]] = []
    try:
        if (
            payload.get("schema_version")
            != gr00t_training_promotion_gate.SCHEMA_VERSION
        ):
            raise ValueError("training promotion gate schema_version mismatch")
        if payload.get("artifact_kind") != gr00t_training_promotion_gate.ARTIFACT_KIND:
            raise ValueError("training promotion gate artifact_kind mismatch")
        if not isinstance(payload.get("allow_plan_next_training_stage"), bool):
            raise ValueError("allow_plan_next_training_stage must be a bool")
        if not isinstance(payload.get("promotion_allowed"), bool):
            raise ValueError("promotion_allowed must be a bool")
        if bool(payload.get("promotion_allowed")) != bool(
            payload.get("allow_plan_next_training_stage")
        ):
            raise ValueError(
                "promotion_allowed must equal allow_plan_next_training_stage"
            )
        if not str(payload.get("reason_code") or "").strip():
            raise ValueError("reason_code must be non-empty")
    except ValueError as exc:
        issues.append(
            _issue_with_source(
                "gate_input_invalid",
                "training_promotion_gate",
                "training promotion gate does not match the expected Task 14 surface",
                source_artifact="training_promotion_gate",
                observed=_exception_message(exc),
            )
        )
    status = "READY" if not issues else "INVALID"
    return {
        "status": status,
        "payload": payload,
        "path": path,
        "issues": issues,
        "source_artifact": _build_source_ref(
            repo_root=repo_root,
            path=path,
            artifact_id="training_promotion_gate",
            authority_role="live_model_triage_input",
        ),
    }


def _row_surface_metadata(*, row_label: str) -> dict[str, Any]:
    comparable_to = None if row_label == AUTHORITATIVE_TRIAGE_ROW_LABELS[0] else "B0"
    return {
        "row_id": str(row_label),
        "display_label": str(row_label),
        "screening_mode": "authoritative",
        "diagnostic_only": False,
        "mainline_authority": True,
        "main_verdict_eligible": True,
        "external_reference_only": False,
        "authority_scope": "authoritative_live_model_triage",
        "surface_route": "live_model_triage",
        "surface_kind": "authoritative_probe_row_surface",
        "comparable_to": comparable_to,
    }


def _summary_metrics(summary_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary_payload, Mapping):
        return {
            "episodes": None,
            "requested_episodes": None,
            "success_count": None,
            "success_rate": None,
            "nonzero_success": None,
        }
    success_count = _optional_int(summary_payload.get("success_count"))
    success_rate = _optional_float(summary_payload.get("success_rate"))
    nonzero_success = None
    if success_count is not None:
        nonzero_success = bool(success_count > 0)
    elif success_rate is not None:
        nonzero_success = bool(success_rate > 0.0)
    return {
        "episodes": _optional_int(summary_payload.get("episodes")),
        "requested_episodes": _optional_int(summary_payload.get("requested_episodes")),
        "success_count": success_count,
        "success_rate": success_rate,
        "nonzero_success": nonzero_success,
    }


def _load_execution_audit_surface(
    *,
    payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    normalized = dict(cast(Mapping[str, Any], payload))
    if (
        normalized.get("schema_version")
        != audit_g1_execution_surface.REPORT_SCHEMA_VERSION
    ):
        raise ValueError("execution audit schema_version mismatch")
    if (
        normalized.get("artifact_kind")
        != audit_g1_execution_surface.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("execution audit artifact_kind mismatch")
    verdict = str(normalized.get("verdict") or "").strip()
    if verdict not in audit_g1_execution_surface.VERDICT_ENUM:
        raise ValueError("execution audit verdict mismatch")
    return normalized


def _build_authoritative_row_surface(
    *,
    row_label: str,
    row_spec: Mapping[str, Any] | None,
    repo_root: Path,
) -> dict[str, Any]:
    normalized_spec = dict(cast(Mapping[str, Any], row_spec or {}))
    issues: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []

    screening_payload, screening_path, screening_load_issues = _safe_read_json_optional(
        normalized_spec.get("screening_json"),
        repo_root=repo_root,
        source_artifact=f"authoritative_screening_{str(row_label).lower()}",
        field_path=f"rows.{row_label}.screening_json",
        message="authoritative row screening artifact could not be loaded",
        code="invalid_required_artifact",
    )
    issues.extend(screening_load_issues)
    if screening_path is not None:
        screening_ref = _build_source_ref(
            repo_root=repo_root,
            path=screening_path,
            artifact_id=f"authoritative_screening_{str(row_label).lower()}",
            authority_role="live_model_triage_row_screening",
        )
        if screening_ref is not None:
            source_artifacts.append(screening_ref)
    if screening_payload is not None:
        try:
            screening_payload = validate_authoritative_screening_summary_payload(
                screening_payload
            )
        except ValueError as exc:
            issues.append(
                _issue_with_source(
                    "invalid_required_artifact",
                    f"rows.{row_label}.screening_json",
                    "authoritative row screening artifact failed validation",
                    source_artifact=f"authoritative_screening_{str(row_label).lower()}",
                    observed=_exception_message(exc),
                )
            )

    summary_payload, summary_path, summary_load_issues = _safe_read_json_optional(
        normalized_spec.get("summary_json"),
        repo_root=repo_root,
        source_artifact=f"authoritative_summary_{str(row_label).lower()}",
        field_path=f"rows.{row_label}.summary_json",
        message="authoritative row summary artifact could not be loaded",
    )
    issues.extend(summary_load_issues)
    if summary_path is not None:
        summary_ref = _build_source_ref(
            repo_root=repo_root,
            path=summary_path,
            artifact_id=f"authoritative_summary_{str(row_label).lower()}",
            authority_role="live_model_triage_row_summary",
        )
        if summary_ref is not None:
            source_artifacts.append(summary_ref)

    execution_audit_payload, execution_audit_path, execution_load_issues = (
        _safe_read_json_optional(
            normalized_spec.get("execution_audit_json"),
            repo_root=repo_root,
            source_artifact=f"execution_audit_{str(row_label).lower()}",
            field_path=f"rows.{row_label}.execution_audit_json",
            message="execution audit artifact could not be loaded",
        )
    )
    issues.extend(execution_load_issues)
    if execution_audit_path is not None:
        execution_ref = _build_source_ref(
            repo_root=repo_root,
            path=execution_audit_path,
            artifact_id=f"execution_audit_{str(row_label).lower()}",
            authority_role="live_model_triage_row_execution_audit",
        )
        if execution_ref is not None:
            source_artifacts.append(execution_ref)
    if execution_audit_payload is not None:
        try:
            execution_audit_payload = _load_execution_audit_surface(
                payload=execution_audit_payload
            )
        except ValueError as exc:
            issues.append(
                _issue_with_source(
                    "invalid_optional_artifact",
                    f"rows.{row_label}.execution_audit_json",
                    "execution audit artifact was ignored because it failed validation",
                    source_artifact=f"execution_audit_{str(row_label).lower()}",
                    observed=_exception_message(exc),
                )
            )
            execution_audit_payload = None

    episode_record, episode_path, episode_load_issues = _safe_read_json_optional(
        normalized_spec.get("episode_json"),
        repo_root=repo_root,
        source_artifact=f"episode_{str(row_label).lower()}",
        field_path=f"rows.{row_label}.episode_json",
        message="row episode artifact could not be loaded",
    )
    issues.extend(episode_load_issues)
    step_records, steps_path, steps_load_issues = _safe_read_jsonl_optional(
        normalized_spec.get("steps_jsonl"),
        repo_root=repo_root,
        source_artifact=f"steps_{str(row_label).lower()}",
        field_path=f"rows.{row_label}.steps_jsonl",
        message="row step telemetry artifact could not be loaded",
    )
    issues.extend(steps_load_issues)
    runtime_trace, runtime_path, runtime_load_issues = _safe_read_json_optional(
        normalized_spec.get("runtime_trace_json"),
        repo_root=repo_root,
        source_artifact=f"runtime_trace_{str(row_label).lower()}",
        field_path=f"rows.{row_label}.runtime_trace_json",
        message="row runtime-trace artifact could not be loaded",
    )
    issues.extend(runtime_load_issues)
    drop_episode_summary, drop_path, drop_load_issues = _safe_read_json_optional(
        normalized_spec.get("drop_episode_json"),
        repo_root=repo_root,
        source_artifact=f"drop_episode_{str(row_label).lower()}",
        field_path=f"rows.{row_label}.drop_episode_json",
        message="row drop-episode artifact could not be loaded",
    )
    issues.extend(drop_load_issues)
    triplet_gate, triplet_path, triplet_load_issues = _safe_read_json_optional(
        normalized_spec.get("triplet_gate_json"),
        repo_root=repo_root,
        source_artifact=f"triplet_gate_{str(row_label).lower()}",
        field_path=f"rows.{row_label}.triplet_gate_json",
        message="row triplet-gate artifact could not be loaded",
    )
    issues.extend(triplet_load_issues)
    action_telemetry, action_path, action_load_issues = _safe_read_json_optional(
        normalized_spec.get("action_telemetry_json"),
        repo_root=repo_root,
        source_artifact=f"action_telemetry_{str(row_label).lower()}",
        field_path=f"rows.{row_label}.action_telemetry_json",
        message="row action-telemetry artifact could not be loaded",
    )
    issues.extend(action_load_issues)
    for path, artifact_id, authority_role in (
        (
            episode_path,
            f"episode_{str(row_label).lower()}",
            "live_model_triage_row_episode",
        ),
        (steps_path, f"steps_{str(row_label).lower()}", "live_model_triage_row_steps"),
        (
            runtime_path,
            f"runtime_trace_{str(row_label).lower()}",
            "live_model_triage_row_runtime_trace",
        ),
        (
            drop_path,
            f"drop_episode_{str(row_label).lower()}",
            "live_model_triage_row_drop_summary",
        ),
        (
            triplet_path,
            f"triplet_gate_{str(row_label).lower()}",
            "live_model_triage_row_triplet_gate",
        ),
        (
            action_path,
            f"action_telemetry_{str(row_label).lower()}",
            "live_model_triage_row_action_telemetry",
        ),
    ):
        ref = _build_source_ref(
            repo_root=repo_root,
            path=path,
            artifact_id=artifact_id,
            authority_role=authority_role,
        )
        if ref is not None:
            source_artifacts.append(ref)

    row_probe_report = None
    if any(
        value is not None
        for value in (
            episode_record,
            step_records,
            runtime_trace,
            drop_episode_summary,
            triplet_gate,
            action_telemetry,
        )
    ):
        row_probe_report = (
            gr00t_screening_probe_bypass_diagnostic.build_probe_evidence_row_report(
                row_label=row_label,
                evidence={
                    "episode_record": episode_record,
                    "step_records": step_records,
                    "runtime_trace": runtime_trace,
                    "drop_episode_summary": drop_episode_summary,
                    "triplet_gate": triplet_gate,
                    "action_telemetry": action_telemetry,
                },
                metadata=_row_surface_metadata(row_label=row_label),
                comparable_to=(None if row_label == "B0" else "B0"),
            )
        )

    metrics = _summary_metrics(summary_payload)
    signal_present = bool(metrics["nonzero_success"] is True)
    if row_probe_report is not None and row_probe_report.get("row_signal") == (
        gr00t_screening_probe_bypass_diagnostic.ROW_SIGNAL_POSITIVE
    ):
        signal_present = True
    flat_signal = bool(
        not signal_present
        and metrics["nonzero_success"] is False
        and (
            summary_payload is not None
            or row_probe_report is not None
            or execution_audit_payload is not None
        )
    )
    screening_reason_code = (
        str(screening_payload.get("reason_code") or "").strip()
        if isinstance(screening_payload, Mapping)
        else None
    )
    screening_eligibility = (
        str(screening_payload.get("formal_eligibility") or "").strip()
        if isinstance(screening_payload, Mapping)
        else None
    )
    row_status = "MISSING"
    if screening_payload is not None and screening_eligibility == "ALLOW":
        if any(
            value is not None
            for value in (
                summary_payload,
                row_probe_report,
                execution_audit_payload,
            )
        ):
            row_status = "READY"
        else:
            row_status = "INSUFFICIENT_EVIDENCE"
    elif screening_payload is not None:
        row_status = "BLOCKED"
    elif normalized_spec:
        row_status = "BLOCKED"
        issues.append(
            _issue_with_source(
                "missing_row_screening",
                f"rows.{row_label}.screening_json",
                "authoritative row evidence cannot be interpreted without the matching Task 8 screening artifact",
                source_artifact=f"authoritative_screening_{str(row_label).lower()}",
            )
        )

    return {
        "row_id": str(row_label),
        "status": row_status,
        "screening": {
            "path": _relative_or_none(repo_root, screening_path),
            "formal_eligibility": screening_eligibility,
            "reason_code": screening_reason_code,
            "servable": (
                screening_payload.get("servable")
                if isinstance(screening_payload, Mapping)
                else None
            ),
            "dataset_inventory_json": (
                screening_payload.get("dataset_inventory_json")
                if isinstance(screening_payload, Mapping)
                else None
            ),
            "dataset_fingerprint": (
                screening_payload.get("dataset_fingerprint")
                if isinstance(screening_payload, Mapping)
                else None
            ),
            "stats_fingerprint": (
                screening_payload.get("stats_fingerprint")
                if isinstance(screening_payload, Mapping)
                else None
            ),
        },
        "summary": {
            "path": _relative_or_none(repo_root, summary_path),
            **metrics,
        },
        "probe_evidence": row_probe_report,
        "execution_audit": {
            "path": _relative_or_none(repo_root, execution_audit_path),
            "verdict": (
                execution_audit_payload.get("verdict")
                if isinstance(execution_audit_payload, Mapping)
                else None
            ),
            "reason_code": (
                execution_audit_payload.get("reason_code")
                if isinstance(execution_audit_payload, Mapping)
                else None
            ),
        },
        "signal_flags": {
            "success_nonzero": metrics["nonzero_success"],
            "probe_positive": bool(
                row_probe_report is not None
                and row_probe_report.get("row_signal")
                == gr00t_screening_probe_bypass_diagnostic.ROW_SIGNAL_POSITIVE
            ),
            "signal_present": signal_present,
            "flat_signal": flat_signal,
        },
        "issues": issues,
        "source_artifacts": source_artifacts,
    }


def _screening_context_mismatch_issue(
    *,
    row_label: str,
    field_name: str,
    expected: object,
    observed: object,
) -> dict[str, Any]:
    return _issue_with_source(
        "stale_screening_context_mismatch",
        f"rows.{row_label}.screening.{field_name}",
        "authoritative row screening does not match the baseline authoritative screening context",
        source_artifact=f"authoritative_screening_{str(row_label).lower()}",
        expected=expected,
        observed=observed,
    )


def _promotion_backpointer_mismatch_issues(
    *,
    promotion_payload: Mapping[str, Any],
    repo_root: Path,
    expected_inventory_path: str | None,
    expected_screening_path: str | None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    source_artifacts = _safe_source_artifacts(promotion_payload)
    expected_pairs = {
        "dataset_inventory": expected_inventory_path,
        "authoritative_screening": expected_screening_path,
    }
    observed_by_id: dict[str, str] = {}
    for item in source_artifacts:
        artifact_id = str(item.get("artifact_id") or "").strip()
        relative_path = str(item.get("relative_path") or "").strip()
        if artifact_id and relative_path:
            observed_by_id[artifact_id] = relative_path
    for artifact_id, expected_path in expected_pairs.items():
        if not expected_path:
            continue
        observed_path = observed_by_id.get(artifact_id)
        if observed_path is None:
            issues.append(
                _issue_with_source(
                    "stale_promotion_gate_backpointer_missing",
                    f"training_promotion_gate.source_artifacts[{artifact_id}]",
                    "training promotion gate is missing a source-artifact backpointer required for live triage freshness checks",
                    source_artifact="training_promotion_gate",
                    expected=expected_path,
                )
            )
            continue
        resolved_observed = _repo_relative_path(
            repo_root, _resolve_repo_path(repo_root, observed_path)
        )
        if resolved_observed != expected_path:
            issues.append(
                _issue_with_source(
                    "stale_promotion_gate_backpointer_mismatch",
                    f"training_promotion_gate.source_artifacts[{artifact_id}].relative_path",
                    "training promotion gate points at a different authoritative artifact than the current live triage inputs",
                    source_artifact="training_promotion_gate",
                    expected=expected_path,
                    observed=resolved_observed,
                )
            )
    return issues


def _conclusive_payload(
    *,
    triage_result: str,
    reason_basis: Mapping[str, Any],
) -> tuple[str, str | None, list[dict[str, Any]], dict[str, Any]]:
    return TRIAGE_STATUS_READY, triage_result, [], dict(reason_basis)


def build_live_model_triage_payload(
    *,
    exact_probe_json: Path | str | None = None,
    diagnostic_json: Path | str | None = None,
    promotion_gate_json: Path
    | str
    | None = gr00t_training_promotion_gate.DEFAULT_OUTPUT_DIR
    / gr00t_training_promotion_gate.TRAINING_PROMOTION_GATE_JSON_NAME,
    row_artifacts_by_label: Mapping[str, Mapping[str, Any] | None] | None = None,
    output_dir: Path | str = DEFAULT_LIVE_MODEL_TRIAGE_OUTPUT_DIR,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_output_dir = _validate_output_dir(Path(output_dir))
    exact_probe = _load_exact_probe_surface(
        exact_probe_json=exact_probe_json,
        repo_root=repo_root,
    )
    diagnostic = _load_diagnostic_surface(
        diagnostic_json=diagnostic_json,
        repo_root=repo_root,
    )
    promotion_gate = _load_promotion_gate_surface(
        promotion_gate_json=promotion_gate_json,
        repo_root=repo_root,
    )

    normalized_rows = {
        str(key): value for key, value in dict(row_artifacts_by_label or {}).items()
    }
    row_surfaces = {
        row_label: _build_authoritative_row_surface(
            row_label=row_label,
            row_spec=normalized_rows.get(row_label),
            repo_root=repo_root,
        )
        for row_label in AUTHORITATIVE_TRIAGE_ROW_LABELS
    }
    issues: list[dict[str, Any]] = [
        *cast(list[dict[str, Any]], exact_probe["issues"]),
        *cast(list[dict[str, Any]], diagnostic["issues"]),
        *cast(list[dict[str, Any]], promotion_gate["issues"]),
    ]
    for row_payload in row_surfaces.values():
        issues.extend(cast(list[dict[str, Any]], row_payload["issues"]))

    b0_row = row_surfaces["B0"]
    baseline_screening = cast(Mapping[str, Any], b0_row["screening"])
    expected_inventory_path = (
        _repo_relative_path(
            repo_root,
            _resolve_repo_path(
                repo_root,
                str(baseline_screening.get("dataset_inventory_json") or ""),
            ),
        )
        if baseline_screening.get("dataset_inventory_json")
        else None
    )
    expected_screening_path = cast(Mapping[str, Any], b0_row["screening"]).get("path")

    for row_label in ("E1", "E2"):
        row_screening = cast(Mapping[str, Any], row_surfaces[row_label]["screening"])
        if row_screening.get("path") is None:
            continue
        for field_name in (
            "dataset_inventory_json",
            "dataset_fingerprint",
            "stats_fingerprint",
        ):
            expected_value = baseline_screening.get(field_name)
            observed_value = row_screening.get(field_name)
            if expected_value is None or observed_value is None:
                continue
            expected_normalized = (
                _repo_relative_path(
                    repo_root, _resolve_repo_path(repo_root, str(expected_value))
                )
                if field_name == "dataset_inventory_json"
                else expected_value
            )
            observed_normalized = (
                _repo_relative_path(
                    repo_root, _resolve_repo_path(repo_root, str(observed_value))
                )
                if field_name == "dataset_inventory_json"
                else observed_value
            )
            if observed_normalized != expected_normalized:
                issues.append(
                    _screening_context_mismatch_issue(
                        row_label=row_label,
                        field_name=field_name,
                        expected=expected_normalized,
                        observed=observed_normalized,
                    )
                )

    promotion_payload = cast(Mapping[str, Any] | None, promotion_gate.get("payload"))
    if isinstance(promotion_payload, Mapping):
        issues.extend(
            _promotion_backpointer_mismatch_issues(
                promotion_payload=promotion_payload,
                repo_root=repo_root,
                expected_inventory_path=expected_inventory_path,
                expected_screening_path=(
                    str(expected_screening_path) if expected_screening_path else None
                ),
            )
        )

    triage_status = TRIAGE_STATUS_INSUFFICIENT_EVIDENCE
    triage_result: str | None = None
    reason_code = "insufficient_evidence"
    reason_basis: dict[str, Any] = {}

    stale_issues = [
        issue for issue in issues if str(issue.get("code") or "").startswith("stale_")
    ]
    if stale_issues:
        triage_status = TRIAGE_STATUS_STALE
        reason_code = str(stale_issues[0].get("code") or "stale_artifact")
        reason_basis = {"trigger": "stale_artifact_detected"}
    elif cast(str, promotion_gate["status"]) == "MISSING":
        triage_status = TRIAGE_STATUS_INSUFFICIENT_EVIDENCE
        reason_code = "promotion_gate_missing"
        reason_basis = {"trigger": "missing_promotion_gate"}
    elif cast(str, promotion_gate["status"]) != "READY":
        triage_status = TRIAGE_STATUS_BLOCKED
        reason_code = gr00t_training_promotion_gate.REASON_CODE_INPUT_INVALID
        reason_basis = {"trigger": "invalid_promotion_gate_surface"}
    else:
        promotion_payload = cast(Mapping[str, Any], promotion_gate["payload"])
        if bool(promotion_payload.get("allow_plan_next_training_stage")) is not True:
            triage_status = TRIAGE_STATUS_BLOCKED
            reason_code = str(
                promotion_payload.get("reason_code") or "promotion_blocked"
            )
            reason_basis = {"trigger": "promotion_gate_blocked"}
        elif cast(str, exact_probe["status"]) == "MISSING":
            triage_status = TRIAGE_STATUS_INSUFFICIENT_EVIDENCE
            reason_code = "exact_probe_missing"
            reason_basis = {"trigger": "missing_exact_probe"}
        elif cast(str, exact_probe["status"]) == "BLOCKED":
            triage_status = TRIAGE_STATUS_BLOCKED
            reason_code = str(
                exact_probe.get("reason_code") or "triplet_binding_blocked"
            )
            reason_basis = {"trigger": "exact_probe_blocked"}
        elif b0_row["status"] == "BLOCKED":
            triage_status = TRIAGE_STATUS_BLOCKED
            reason_code = str(
                baseline_screening.get("reason_code") or "inventory_blocked"
            )
            reason_basis = {"trigger": "b0_authoritative_screening_blocked"}
        else:
            b0_nonzero = cast(Mapping[str, Any], b0_row["summary"]).get(
                "nonzero_success"
            )
            exact_probe_status = cast(str, exact_probe["status"])
            if exact_probe_status == "FAIL":
                if b0_nonzero is True:
                    triage_status, triage_result, _, reason_basis = _conclusive_payload(
                        triage_result=TRIAGE_RESULT_PROBE_FAILED_BUT_B0_NONZERO,
                        reason_basis={"trigger": "probe_fail_b0_nonzero"},
                    )
                    reason_code = TRIAGE_RESULT_PROBE_FAILED_BUT_B0_NONZERO
                elif b0_nonzero is False:
                    triage_status, triage_result, _, reason_basis = _conclusive_payload(
                        triage_result=TRIAGE_RESULT_PROBE_FAILED_AND_B0_ZERO,
                        reason_basis={"trigger": "probe_fail_b0_zero"},
                    )
                    reason_code = TRIAGE_RESULT_PROBE_FAILED_AND_B0_ZERO
                else:
                    triage_status = TRIAGE_STATUS_INSUFFICIENT_EVIDENCE
                    reason_code = "authoritative_b0_missing"
                    reason_basis = {"trigger": "missing_b0_success_surface"}
            elif exact_probe_status == "PASS":
                if (
                    row_surfaces["E2"]["status"] != "MISSING"
                    and row_surfaces["E1"]["status"] == "MISSING"
                ):
                    triage_status = TRIAGE_STATUS_INSUFFICIENT_EVIDENCE
                    reason_code = "authoritative_e2_requires_e1"
                    reason_basis = {"trigger": "e2_without_e1"}
                else:
                    post_b0_rows = [row_surfaces["E1"], row_surfaces["E2"]]
                    ready_post_b0_rows = [
                        row
                        for row in post_b0_rows
                        if row["status"] in {"READY", "INSUFFICIENT_EVIDENCE"}
                        and (
                            cast(Mapping[str, Any], row["summary"]).get("path")
                            is not None
                            or row.get("probe_evidence") is not None
                            or cast(Mapping[str, Any], row["execution_audit"]).get(
                                "path"
                            )
                            is not None
                        )
                    ]
                    signal_rows = [
                        cast(str, row["row_id"])
                        for row in ready_post_b0_rows
                        if cast(Mapping[str, Any], row["signal_flags"]).get(
                            "signal_present"
                        )
                        is True
                    ]
                    if signal_rows:
                        triage_status, triage_result, _, reason_basis = (
                            _conclusive_payload(
                                triage_result=TRIAGE_RESULT_PROBE_PASSED_E1_SIGNAL_PRESENT,
                                reason_basis={
                                    "trigger": "authoritative_post_b0_signal_present",
                                    "signal_rows": signal_rows,
                                },
                            )
                        )
                        reason_code = TRIAGE_RESULT_PROBE_PASSED_E1_SIGNAL_PRESENT
                    elif ready_post_b0_rows and all(
                        cast(Mapping[str, Any], row["signal_flags"]).get("flat_signal")
                        is True
                        for row in ready_post_b0_rows
                    ):
                        triage_status, triage_result, _, reason_basis = (
                            _conclusive_payload(
                                triage_result=TRIAGE_RESULT_PROBE_PASSED_E1E2_FLAT,
                                reason_basis={
                                    "trigger": "available_post_b0_rows_flat",
                                    "flat_rows": [
                                        cast(str, row["row_id"])
                                        for row in ready_post_b0_rows
                                    ],
                                },
                            )
                        )
                        reason_code = TRIAGE_RESULT_PROBE_PASSED_E1E2_FLAT
                    elif b0_nonzero is True:
                        triage_status, triage_result, _, reason_basis = (
                            _conclusive_payload(
                                triage_result=TRIAGE_RESULT_PROBE_PASSED_B0_NONZERO,
                                reason_basis={"trigger": "probe_pass_b0_nonzero_only"},
                            )
                        )
                        reason_code = TRIAGE_RESULT_PROBE_PASSED_B0_NONZERO
                    else:
                        triage_status = TRIAGE_STATUS_INSUFFICIENT_EVIDENCE
                        reason_code = "authoritative_signal_missing"
                        reason_basis = {
                            "trigger": "probe_pass_without_authoritative_signal"
                        }
            else:
                triage_status = TRIAGE_STATUS_INSUFFICIENT_EVIDENCE
                reason_code = "exact_probe_unknown"
                reason_basis = {"trigger": "exact_probe_status_unknown"}

    source_artifacts: list[dict[str, Any]] = []
    for source_artifact in (
        exact_probe.get("source_artifact"),
        diagnostic.get("source_artifact"),
        promotion_gate.get("source_artifact"),
    ):
        if isinstance(source_artifact, Mapping):
            source_artifacts.append(dict(cast(Mapping[str, Any], source_artifact)))
    for row_payload in row_surfaces.values():
        source_artifacts.extend(
            cast(list[dict[str, Any]], row_payload.get("source_artifacts", []))
        )

    payload: dict[str, Any] = {
        "schema_version": LIVE_MODEL_TRIAGE_SCHEMA_VERSION,
        "artifact_kind": LIVE_MODEL_TRIAGE_ARTIFACT_KIND,
        "generated_at": generated_at or apple_recap_execution_contract._now_iso(),
        "output_dir": _repo_relative_path(repo_root, resolved_output_dir),
        "artifact_path": _repo_relative_path(
            repo_root,
            resolved_output_dir / LIVE_MODEL_TRIAGE_JSON_NAME,
        ),
        "triage_status": triage_status,
        "triage_result": triage_result,
        "allowed_triage_results": list(TRIAGE_RESULT_ENUM),
        "reason_code": reason_code,
        "issues": issues,
        "reason_basis": reason_basis,
        "diagnostic_only": False,
        "mainline_authority": True,
        "main_verdict_eligible": True,
        "external_reference_only": False,
        "authority_scope": "authoritative_live_model_triage",
        "surface_route": "live_model_triage",
        "source_artifacts": source_artifacts,
        "dataset_inventory_json": expected_inventory_path,
        "authoritative_screening_json": expected_screening_path,
        "authoritative_screening_row_paths": {
            row_label: cast(Mapping[str, Any], row_payload["screening"]).get("path")
            for row_label, row_payload in row_surfaces.items()
        },
        "promotion_gate_json": _relative_or_none(
            repo_root,
            cast(Path | None, promotion_gate.get("path")),
        ),
        "diagnostic_json": _relative_or_none(
            repo_root,
            cast(Path | None, diagnostic.get("path")),
        ),
        "exact_probe_json": _relative_or_none(
            repo_root,
            cast(Path | None, exact_probe.get("path")),
        ),
        "exact_probe": {
            "status": exact_probe.get("status"),
            "reason_code": exact_probe.get("reason_code"),
        },
        "diagnostic_reference": (
            None
            if not isinstance(diagnostic.get("payload"), Mapping)
            else {
                "status": diagnostic.get("status"),
                "comparison_verdict": cast(
                    Mapping[str, Any], diagnostic["payload"]
                ).get("comparison_verdict"),
                "comparison_basis": cast(Mapping[str, Any], diagnostic["payload"]).get(
                    "comparison_basis"
                ),
            }
        ),
        "promotion_gate": (
            None
            if not isinstance(promotion_gate.get("payload"), Mapping)
            else {
                "allow_plan_next_training_stage": cast(
                    Mapping[str, Any], promotion_gate["payload"]
                ).get("allow_plan_next_training_stage"),
                "promotion_allowed": cast(
                    Mapping[str, Any], promotion_gate["payload"]
                ).get("promotion_allowed"),
                "reason_code": cast(Mapping[str, Any], promotion_gate["payload"]).get(
                    "reason_code"
                ),
            }
        ),
        "authoritative_rows": row_surfaces,
    }
    payload["report_signature_sha256"] = _signature_for_payload(payload)
    return payload


def materialize_live_model_triage(
    *,
    exact_probe_json: Path | str | None = None,
    diagnostic_json: Path | str | None = None,
    promotion_gate_json: Path
    | str
    | None = gr00t_training_promotion_gate.DEFAULT_OUTPUT_DIR
    / gr00t_training_promotion_gate.TRAINING_PROMOTION_GATE_JSON_NAME,
    row_artifacts_by_label: Mapping[str, Mapping[str, Any] | None] | None = None,
    output_dir: Path | str = DEFAULT_LIVE_MODEL_TRIAGE_OUTPUT_DIR,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    payload = build_live_model_triage_payload(
        exact_probe_json=exact_probe_json,
        diagnostic_json=diagnostic_json,
        promotion_gate_json=promotion_gate_json,
        row_artifacts_by_label=row_artifacts_by_label,
        output_dir=output_dir,
        repo_root=repo_root,
        generated_at=generated_at,
    )
    resolved_output_dir = _validate_output_dir(Path(output_dir))
    _write_json(resolved_output_dir / LIVE_MODEL_TRIAGE_JSON_NAME, payload)
    return payload


def validate_live_model_triage_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(cast(Mapping[str, Any], payload))
    if normalized.get("schema_version") != LIVE_MODEL_TRIAGE_SCHEMA_VERSION:
        raise ValueError(
            "live_model_triage.schema_version mismatch: expected "
            + repr(LIVE_MODEL_TRIAGE_SCHEMA_VERSION)
        )
    if normalized.get("artifact_kind") != LIVE_MODEL_TRIAGE_ARTIFACT_KIND:
        raise ValueError(
            "live_model_triage.artifact_kind mismatch: expected "
            + repr(LIVE_MODEL_TRIAGE_ARTIFACT_KIND)
        )
    triage_status = str(normalized.get("triage_status") or "").strip()
    if triage_status not in TRIAGE_STATUS_ENUM:
        raise ValueError("live_model_triage.triage_status is not recognized")
    triage_result = normalized.get("triage_result")
    if (
        triage_result is not None
        and str(triage_result).strip() not in TRIAGE_RESULT_ENUM
    ):
        raise ValueError("live_model_triage.triage_result is not recognized")
    if bool(normalized.get("diagnostic_only")) is not False:
        raise ValueError("live_model_triage.diagnostic_only mismatch")
    if bool(normalized.get("mainline_authority")) is not True:
        raise ValueError("live_model_triage.mainline_authority mismatch")
    if bool(normalized.get("main_verdict_eligible")) is not True:
        raise ValueError("live_model_triage.main_verdict_eligible mismatch")
    if bool(normalized.get("external_reference_only")) is not False:
        raise ValueError("live_model_triage.external_reference_only mismatch")
    declared_signature = str(normalized.get("report_signature_sha256") or "").strip()
    expected_signature = _signature_for_payload(normalized)
    if declared_signature != expected_signature:
        raise ValueError(
            "live_model_triage.report_signature_sha256 mismatch: expected "
            + repr(expected_signature)
        )
    return normalized


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = materialize_authoritative_screening(
            config_module=args.config_module,
            dataset_inventory_json=args.dataset_inventory_json,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if payload.get("formal_eligibility") == "ALLOW" else 1


__all__ = [
    "AUTHORITATIVE_SCREENING_JSON_NAME",
    "AUTHORITATIVE_TRIAGE_ROW_LABELS",
    "DEFAULT_CONFIG_MODULE",
    "DEFAULT_DATASET_INVENTORY_JSON",
    "DEFAULT_LIVE_MODEL_TRIAGE_OUTPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "JOIN_KEY_FIELDS",
    "LIVE_MODEL_TRIAGE_ARTIFACT_KIND",
    "LIVE_MODEL_TRIAGE_JSON_NAME",
    "LIVE_MODEL_TRIAGE_SCHEMA_VERSION",
    "SCREENING_ARTIFACT_KIND",
    "SCREENING_SCHEMA_VERSION",
    "TRIAGE_RESULT_ENUM",
    "TRIAGE_STATUS_ENUM",
    "build_authoritative_binding_join_contract",
    "build_live_model_triage_payload",
    "build_parser",
    "load_authoritative_config_surface",
    "main",
    "materialize_authoritative_screening",
    "materialize_live_model_triage",
    "validate_authoritative_screening_summary_payload",
    "validate_live_model_triage_payload",
]


if __name__ == "__main__":
    raise SystemExit(main())
