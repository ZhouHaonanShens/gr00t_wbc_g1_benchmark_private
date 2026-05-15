#!/usr/bin/env python3
"""Canonical identity preflight contract for GR00T checkpoints.

This module is intentionally file-surface based: it runs before model/server/
training initialization and does not load GR00T weights.  It prevents known
processor/config drift (especially ``formalize_language``) from silently
re-entering eval/training entrypoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]

BEHAVIOR_SURFACE_FILES = (
    "config.json",
    "processor_config.json",
    "statistics.json",
    "modality.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "embodiment_id.json",
)

CORE_HASH_FILES = ("config.json", "processor_config.json", "statistics.json", "modality.json")
DEFAULT_TASK_PROMPT = "pick up the apple, walk left and place the apple on the plate."
LANGUAGE_TOKEN_SURFACE_FILES = (
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
)


class PreflightMode(str, Enum):
    STRICT_PROMOTION = "STRICT_PROMOTION"
    SURFACE_CAUSALITY_DIAGNOSTIC = "SURFACE_CAUSALITY_DIAGNOSTIC"


@dataclass(frozen=True)
class SurfaceEntry:
    status: str
    sha256: str | None
    canonical_sha256: str | None
    equal_to_canonical: bool
    reason: str


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def git_output(args: Sequence[str], *, cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(cwd), text=True).strip()
    except Exception as exc:  # pragma: no cover - defensive diagnostic payload
        return f"UNKNOWN:{exc}"


def git_status_short(path: Path) -> list[str]:
    out = git_output(["-C", str(path), "status", "--short"])
    return [line for line in out.splitlines() if line.strip()]


def load_canonical_manifest(canonical_dir: Path) -> dict[str, Any] | None:
    manifest_path = canonical_dir / "canonical_identity_manifest.json"
    if not manifest_path.exists():
        return None
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        return None
    return payload


def _get_nested(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def extract_formalize_language(bundle: Path) -> dict[str, Any]:
    config_path = bundle / "config.json"
    processor_path = bundle / "processor_config.json"
    config_value = None
    processor_value = None
    if config_path.exists():
        config = read_json(config_path)
        if isinstance(config, Mapping):
            config_value = config.get("formalize_language")
    if processor_path.exists():
        processor = read_json(processor_path)
        if isinstance(processor, Mapping):
            processor_value = _get_nested(
                processor,
                ("processor_kwargs", "formalize_language"),
            )
    config_processor_match = (
        (config_value == processor_value)
        if (config_value is not None and processor_value is not None)
        else None
    )
    return {
        "config": config_value,
        "processor_config": processor_value,
        "effective": processor_value if processor_value is not None else config_value,
        "config_processor_match": config_processor_match,
    }


def _missing_surface_files(canonical_manifest: Mapping[str, Any] | None) -> set[str]:
    raw = [] if canonical_manifest is None else canonical_manifest.get("missing_surface_files", [])
    return {str(item) for item in raw if str(item).strip()}


def surface_file_entry(
    candidate: Path,
    canonical: Path,
    file_name: str,
    *,
    canonical_manifest: Mapping[str, Any] | None,
) -> SurfaceEntry:
    candidate_file = candidate / file_name
    canonical_file = canonical / file_name
    candidate_exists = candidate_file.exists()
    canonical_exists = canonical_file.exists()
    expected_missing = file_name in _missing_surface_files(canonical_manifest)
    if candidate_exists and canonical_exists:
        candidate_hash = sha256_file(candidate_file)
        canonical_hash = sha256_file(canonical_file)
        return SurfaceEntry(
            status="PRESENT",
            sha256=candidate_hash,
            canonical_sha256=canonical_hash,
            equal_to_canonical=candidate_hash == canonical_hash,
            reason="present_in_candidate_and_canonical",
        )
    if not candidate_exists and not canonical_exists and expected_missing:
        return SurfaceEntry(
            status="MISSING_EXPECTED",
            sha256=None,
            canonical_sha256=None,
            equal_to_canonical=True,
            reason="missing_from_candidate_and_canonical_and_recorded_in_manifest",
        )
    if candidate_exists != canonical_exists:
        return SurfaceEntry(
            status="MISSING_UNEXPECTED",
            sha256=sha256_file(candidate_file) if candidate_exists else None,
            canonical_sha256=sha256_file(canonical_file) if canonical_exists else None,
            equal_to_canonical=False,
            reason="present_in_only_one_surface",
        )
    return SurfaceEntry(
        status="MISSING_UNEXPECTED",
        sha256=None,
        canonical_sha256=None,
        equal_to_canonical=False,
        reason="missing_from_both_surfaces_but_not_recorded_in_canonical_manifest",
    )


def build_surface_file_hashes(
    candidate: Path,
    canonical: Path,
    *,
    canonical_manifest: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for file_name in BEHAVIOR_SURFACE_FILES:
        entry = surface_file_entry(
            candidate,
            canonical,
            file_name,
            canonical_manifest=canonical_manifest,
        )
        out[file_name] = {
            "status": entry.status,
            "sha256": entry.sha256,
            "canonical_sha256": entry.canonical_sha256,
            "equal_to_canonical": entry.equal_to_canonical,
            "reason": entry.reason,
        }
    return out


def _hash_field(surface_hashes: Mapping[str, Mapping[str, Any]], file_name: str) -> str:
    entry = surface_hashes.get(file_name, {})
    if entry.get("status") == "PRESENT":
        return str(entry.get("sha256") or "")
    return str(entry.get("status") or "NOT_FOUND")


def processed_language(prompt: str, formalize_language: Any) -> str:
    if bool(formalize_language):
        return re.sub(r"[^\w\s]", "", prompt.lower())
    return prompt


def language_surface_projection(
    surface_hashes: Mapping[str, Mapping[str, Any]],
    *,
    canonical: bool,
) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for file_name in LANGUAGE_TOKEN_SURFACE_FILES:
        entry = surface_hashes.get(file_name, {})
        status = str(entry.get("status") or "NOT_FOUND")
        projection[file_name] = {
            "status": status,
            "sha256": (
                entry.get("canonical_sha256")
                if canonical and status == "PRESENT"
                else entry.get("sha256")
            ),
        }
    return projection


def language_token_fingerprint(
    *,
    prompt: str,
    formalize_language: Any,
    language_surface_hash: str,
) -> dict[str, Any]:
    # Preflight runs before model/server initialization.  This deterministic
    # surrogate is intentionally tied to the processor surface and formalizer; a
    # future L4/L5 gate records true model-collator token_ids_hash.
    processed = processed_language(prompt, formalize_language)
    payload = {
        "fingerprint_schema": "processed_language_plus_processor_surface_v1",
        "raw_language": prompt,
        "processed_language": processed,
        "formalize_language": bool(formalize_language),
        "language_surface_hash": language_surface_hash,
    }
    return {
        "token_ids_hash": sha256_json(payload),
        "source": "SURROGATE_PRE_MODEL_INITIALIZATION",
        "raw_language": prompt,
        "processed_language": processed,
        "note": (
            "True tokenizer token_ids_hash is verified in L4/L5; Phase 1 "
            "preflight uses a deterministic pre-model surrogate tied to "
            "formalize_language and processor/tokenizer surface hashes."
        ),
    }


def _strict_issues(
    candidate: Path,
    canonical: Path,
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not report.get("canonical_manifest_present"):
        issues.append({
            "reason_code": "FAIL_CLOSED_MISSING_MANIFEST",
            "message": "canonical_identity_manifest.json missing",
        })
    if report.get("canonical_A1_status") != "PASS":
        issues.append({
            "reason_code": "FAIL_CLOSED_CANONICAL_A1",
            "message": "canonical identity A1 is not PASS",
        })
    if report.get("canonical_A2_status") != "PASS":
        issues.append({
            "reason_code": "FAIL_CLOSED_CANONICAL_A2",
            "message": "canonical identity A2 is not PASS",
        })
    candidate_formalize = report.get("candidate_formalize_language", {}).get("effective")
    canonical_formalize = report.get("canonical_formalize_language", {}).get("effective")
    if candidate_formalize != canonical_formalize:
        issues.append({
            "reason_code": "FAIL_CLOSED_FORMALIZE_MISMATCH",
            "message": "candidate formalize_language differs from canonical/base",
        })
    surface_hashes = report.get("surface_file_hashes", {})
    for file_name in ("processor_config.json", "statistics.json"):
        entry = surface_hashes.get(file_name, {})
        if entry.get("status") == "PRESENT" and not bool(entry.get("equal_to_canonical", False)):
            reason_code = (
                "FAIL_CLOSED_PROCESSOR_HASH"
                if file_name == "processor_config.json"
                else "FAIL_CLOSED_STATS_HASH"
            )
            issues.append({
                "reason_code": reason_code,
                "file": file_name,
                "message": "candidate surface file hash differs from canonical",
            })
    if report.get("language_token_hash") != report.get("canonical_language_token_hash"):
        issues.append({
            "reason_code": "FAIL_CLOSED_TOKEN_HASH",
            "message": "language token fingerprint differs from canonical",
        })
    for file_name, entry in surface_hashes.items():
        status = entry.get("status")
        if status == "MISSING_UNEXPECTED":
            issues.append({
                "reason_code": "FAIL_CLOSED_MISSING_UNEXPECTED",
                "file": file_name,
                "message": "surface file missing unexpectedly",
            })
        if status == "PRESENT" and not bool(entry.get("equal_to_canonical", False)):
            if file_name not in {"processor_config.json", "statistics.json"}:
                issues.append({
                    "reason_code": "FAIL_CLOSED_ACTION_SURFACE",
                    "file": file_name,
                    "message": "candidate surface file hash differs from canonical",
                })
    return issues


def _diagnostic_issues(matrix_id: str | None, diagnostic_manifest: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not matrix_id:
        issues.append({
            "reason_code": "DIAGNOSTIC_MATRIX_ID_MISSING",
            "message": "diagnostic mode requires --matrix-id",
        })
    if not isinstance(diagnostic_manifest, Mapping):
        issues.append({
            "reason_code": "DIAGNOSTIC_MANIFEST_MISSING",
            "message": "diagnostic mode requires --diagnostic-manifest-json",
        })
        return issues
    if diagnostic_manifest.get("diagnostic_mode") is not True:
        issues.append({
            "reason_code": "DIAGNOSTIC_MODE_NOT_TRUE",
            "message": "diagnostic manifest diagnostic_mode must be true",
        })
    if diagnostic_manifest.get("training_allowed") is not False:
        issues.append({
            "reason_code": "DIAGNOSTIC_TRAINING_NOT_FALSE",
            "message": "diagnostic manifest training_allowed must be false",
        })
    if diagnostic_manifest.get("promotion_allowed") is not False:
        issues.append({
            "reason_code": "DIAGNOSTIC_PROMOTION_NOT_FALSE",
            "message": "diagnostic manifest promotion_allowed must be false",
        })
    if diagnostic_manifest.get("outer_verdict") != "DIAGNOSTIC_ONLY":
        issues.append({
            "reason_code": "DIAGNOSTIC_OUTER_VERDICT_NOT_DIAGNOSTIC_ONLY",
            "message": "diagnostic manifest outer_verdict must be DIAGNOSTIC_ONLY",
        })
    return issues


def build_preflight_report(
    *,
    checkpoint: Path,
    canonical: Path,
    mode: PreflightMode,
    entrypoint: str,
    entrypoint_kind: str,
    prompt: str = DEFAULT_TASK_PROMPT,
    matrix_id: str | None = None,
    diagnostic_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = resolve_path(checkpoint)
    canonical = resolve_path(canonical)
    canonical_manifest = load_canonical_manifest(canonical)
    surface_hashes = build_surface_file_hashes(checkpoint, canonical, canonical_manifest=canonical_manifest)
    candidate_formalize = extract_formalize_language(checkpoint)
    canonical_formalize = extract_formalize_language(canonical)
    candidate_language_surface_hash = sha256_json(language_surface_projection(surface_hashes, canonical=False))
    canonical_language_surface_hash = sha256_json(language_surface_projection(surface_hashes, canonical=True))
    candidate_lang = language_token_fingerprint(
        prompt=prompt,
        formalize_language=candidate_formalize.get("effective"),
        language_surface_hash=candidate_language_surface_hash,
    )
    canonical_lang = language_token_fingerprint(
        prompt=prompt,
        formalize_language=canonical_formalize.get("effective"),
        language_surface_hash=canonical_language_surface_hash,
    )
    surface_projection = {
        name: {
            "status": data["status"],
            "sha256": data["sha256"],
            "equal_to_canonical": data["equal_to_canonical"],
        }
        for name, data in sorted(surface_hashes.items())
    }
    action_surface_hash = sha256_json(surface_projection)
    canonical_a1 = None
    canonical_a2 = None
    if isinstance(canonical_manifest, Mapping):
        canonical_a1 = (canonical_manifest.get("A1_dangerous_diff") or {}).get("status")
        canonical_a2 = (canonical_manifest.get("A2_offline_l1_l3") or {}).get("status")
    repo_status = git_status_short(REPO_ROOT)
    isaac_status = git_status_short(REPO_ROOT / "submodules" / "Isaac-GR00T")
    wbc_status = git_status_short(
        REPO_ROOT
        / "submodules"
        / "Isaac-GR00T"
        / "gr00t"
        / "eval"
        / "sim"
        / "GR00T-WholeBodyControl"
        / "GR00T-WholeBodyControl_uv"
    )
    imported_submodule_dirty = bool(isaac_status or wbc_status)
    base_report: dict[str, Any] = {
        "schema_version": "gr00t_canonical_identity_preflight_v1",
        "artifact_kind": "canonical_identity_preflight_report",
        "mode": mode.value,
        "entrypoint": entrypoint,
        "entrypoint_kind": entrypoint_kind,
        "checkpoint_path": repo_rel(checkpoint),
        "canonical_identity_path": repo_rel(canonical),
        "canonical_manifest_present": canonical_manifest is not None,
        "canonical_A1_status": canonical_a1,
        "canonical_A2_status": canonical_a2,
        "candidate_formalize_language": candidate_formalize,
        "canonical_formalize_language": canonical_formalize,
        "config_hash": _hash_field(surface_hashes, "config.json"),
        "processor_config_hash": _hash_field(surface_hashes, "processor_config.json"),
        "statistics_hash": _hash_field(surface_hashes, "statistics.json"),
        "modality_hash": surface_hashes.get("modality.json"),
        "surface_file_hashes": surface_hashes,
        "action_surface_hash": action_surface_hash,
        "canonical_action_surface_hash": sha256_json({
            name: {
                "status": data["status"],
                "sha256": data["canonical_sha256"] if data["status"] == "PRESENT" else None,
                "equal_to_canonical": True,
            }
            for name, data in sorted(surface_hashes.items())
        }),
        "language": candidate_lang,
        "canonical_language": canonical_lang,
        "language_token_hash": candidate_lang["token_ids_hash"],
        "canonical_language_token_hash": canonical_lang["token_ids_hash"],
        "git": {
            "repo_head": git_output(["rev-parse", "HEAD"]),
            "repo_status_short": repo_status,
            "isaac_gr00t_status_short": isaac_status,
            "wbc_status_short": wbc_status,
            "imported_submodule_dirty": imported_submodule_dirty,
        },
        "submodule_status_recorded": True,
        "max_deployable_label": "A_PASS_DEPLOYABLE_LOCAL_DIRTY"
        if imported_submodule_dirty
        else "A_PASS_DEPLOYABLE_CANDIDATE",
    }
    if mode is PreflightMode.STRICT_PROMOTION:
        issues = _strict_issues(checkpoint, canonical, base_report)
        verdict = "PASS" if not issues else "FAIL"
        reason_code = "STRICT_PROMOTION_PASS" if not issues else issues[0]["reason_code"]
        training_allowed = False
        promotion_allowed = verdict == "PASS" and not imported_submodule_dirty
    else:
        issues = _diagnostic_issues(matrix_id, diagnostic_manifest)
        verdict = "DIAGNOSTIC_ONLY" if not issues else "FAIL"
        reason_code = "SURFACE_CAUSALITY_DIAGNOSTIC_ACCEPTED" if not issues else issues[0]["reason_code"]
        training_allowed = False
        promotion_allowed = False
    base_report.update(
        {
            "verdict": verdict,
            "reason_code": reason_code,
            "issues": issues,
            "training_allowed": training_allowed,
            "promotion_allowed": promotion_allowed,
            "matrix_id": matrix_id,
            "diagnostic_manifest": diagnostic_manifest,
        }
    )
    return base_report


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# GR00T Canonical Identity Preflight Report",
        "",
        f"- verdict: `{report.get('verdict')}`",
        f"- reason_code: `{report.get('reason_code')}`",
        f"- mode: `{report.get('mode')}`",
        f"- checkpoint: `{report.get('checkpoint_path')}`",
        f"- canonical_identity: `{report.get('canonical_identity_path')}`",
        f"- canonical A1/A2: `{report.get('canonical_A1_status')}` / `{report.get('canonical_A2_status')}`",
        f"- candidate formalize: `{(report.get('candidate_formalize_language') or {}).get('effective')}`",
        f"- canonical formalize: `{(report.get('canonical_formalize_language') or {}).get('effective')}`",
        f"- language_token_hash: `{report.get('language_token_hash')}`",
        f"- canonical_language_token_hash: `{report.get('canonical_language_token_hash')}`",
        "",
        "## Issues",
        "",
    ]
    issues = report.get("issues") or []
    if issues:
        for issue in issues:
            lines.append(f"- `{issue.get('reason_code')}` {issue.get('file', '')}: {issue.get('message')}")
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Surface file hashes",
        "",
        "| file | status | equal_to_canonical | sha256 |",
        "|---|---|---:|---|",
    ])
    for file_name, entry in sorted((report.get("surface_file_hashes") or {}).items()):
        lines.append(
            f"| `{file_name}` | `{entry.get('status')}` | "
            f"`{entry.get('equal_to_canonical')}` | `{entry.get('sha256')}` |"
        )
    return "\n".join(lines) + "\n"


def write_preflight_outputs(report: Mapping[str, Any], *, report_json: Path, report_md: Path) -> None:
    write_json(report_json, report)
    write_text(report_md, markdown_report(report))


def validate_preflight_report_for_entrypoint(report_path: Path, *, require_strict: bool = True) -> dict[str, Any]:
    payload = read_json(resolve_path(report_path))
    if not isinstance(payload, dict):
        raise ValueError("preflight report must be a JSON object")
    if payload.get("schema_version") != "gr00t_canonical_identity_preflight_v1":
        raise ValueError("preflight report schema_version mismatch")
    if require_strict and payload.get("mode") != PreflightMode.STRICT_PROMOTION.value:
        raise ValueError("entrypoint requires STRICT_PROMOTION preflight report")
    if payload.get("verdict") != "PASS":
        raise ValueError(f"preflight report is not PASS: {payload.get('reason_code')}")
    return payload


LAUNCHER_INVENTORY_ROWS = (
    (
        "work/recap/scripts/3D_recap_eval.py",
        "eval",
        "GR00T/RECAP WBC evaluation path",
        "canonical_preflight_report_required_before_eval",
    ),
    (
        "work/recap/scripts/gr00t_eval_contract_gate.py",
        "eval_gate",
        "public eval contract gate",
        "should_record_canonical_preflight_requirements",
    ),
    (
        "work/recap/scripts/gr00t_training_promotion_gate.py",
        "training_gate",
        "promotion gate for training artifacts",
        "canonical_preflight_report_required_for_checkpoint_promotion",
    ),
    (
        "work/recap/scripts/gr00t_recap_training_smoke.py",
        "training",
        "GR00T Flux training smoke lane",
        "canonical_preflight_report_required_before_training_materialization",
    ),
    (
        "work/recap/scripts/3D_recap_finetune_full.py",
        "training",
        "full finetune wrapper",
        "script_app_must_require_canonical_preflight_or_phase_gate",
    ),
    (
        "work/recap/scripts/34_recap_finetune_repro.py",
        "training",
        "finetune repro wrapper",
        "script_app_must_require_canonical_preflight_or_phase_gate",
    ),
    (
        "work/recap/scripts/state_conditioned_train.py",
        "training",
        "state-conditioned training wrapper",
        "script_app_must_require_canonical_preflight_or_phase_gate",
    ),
    (
        "work/recap/scripts/38_recap_online_loop_iterate.py",
        "online_loop",
        "RECAP online loop iteration",
        "phase_gate_required_before_any_training_step",
    ),
    (
        "work/recap/scripts/3A_recap_multi_iter_loop.py",
        "online_loop",
        "RECAP multi-iteration loop",
        "phase_gate_required_before_any_training_step",
    ),
    (
        "agent/run/gr00t_flux_train_smoke.py",
        "thin_wrapper",
        "public wrapper for training smoke",
        "must_delegate_to_work_gate_only",
    ),
    (
        "agent/run/gr00t_screening_authoritative.py",
        "thin_wrapper",
        "screening wrapper",
        "must_delegate_to_work_gate_only",
    ),
    (
        "agent/run/stage_b_p0_eval_protocol.py",
        "thin_wrapper",
        "stage B eval protocol wrapper",
        "must_delegate_to_work_gate_only",
    ),
    (
        "agent/run/official_rollout_apple_to_plate_smoke.py",
        "thin_wrapper",
        "official rollout smoke wrapper",
        "must_delegate_to_work_gate_only",
    ),
)


def launcher_inventory_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path, kind, purpose, required in LAUNCHER_INVENTORY_ROWS:
        rows.append({
            "path": path,
            "kind": kind,
            "purpose": purpose,
            "phase1_requirement": required,
            "exists": str((REPO_ROOT / path).exists()).lower(),
        })
    return rows
