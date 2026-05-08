"""Read-only verifier for the GR00T T8.2 final artifact bundle.

T8.2 is a no-training diagnostic run.  This verifier intentionally avoids model
loads and validates only the machine-readable evidence contract: source evidence
lock, forbidden-scope guard, JSON/CSV/JSONL schemas, same-seed controls, and the
single user-provided final enum list.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass, field
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shlex
from typing import Any


ALLOWED_T8_2_FINAL_DECISIONS: tuple[str, ...] = (
    "BASE_PROTOCOL_TOO_WEAK",
    "SAFE_SFT_NONCOLLAPSE_PRELIM",
    "REACH_NAV_BLOCKER",
    "POST_LIFT_PLACE_BLOCKER",
    "POST_LIFT_SPLICE_IDENTIFIED_FIX",
    "HAND_REGRESSION",
    "READY_FOR_SAFE_SFT_30",
    "GUARDED_RECAP_STILL_FORBIDDEN",
)

LEGACY_T8_FINAL_DECISIONS: frozenset[str] = frozenset(
    {
        "RUNNER_OR_CONTRACT_REGRESSION",
        "BASE_SEEDS_TOO_HARD",
        "NAV_SPLICE_IMPROVES",
        "NAV_DIRECTION_TIMING_BUG",
        "SAFE_SFT_30_READY",
        "T8_FAIL_ENTRYPOINT_REGRESSION",
        "T8_FAIL_HAND_SIGNAL_NOT_LEARNED",
        "T8_FAIL_HAND_REPAIRED_BUT_NO_LIFT",
        "T8_PASS_HAND_REPAIR_SMOKE",
        "T8_READY_FOR_30SEED_SAFE_SFT",
        "T8_NAV_BLOCKER_AFTER_HAND_FIX",
    }
)

REQUIRED_JSON_ARTIFACTS: tuple[str, ...] = (
    "evidence_lock.json",
    "command_manifest.json",
    "seed_scan_manifest.json",
    "seed_bank.json",
    "seed_bank_deficits.json",
    "control_regression_report.json",
    "stratum_effects.json",
    "post_lift_place_audit.json",
    "final_decision.json",
)

REQUIRED_CSV_ARTIFACTS: tuple[str, ...] = (
    "candidate_seed_scout.csv",
    "paired_eval_summary.csv",
    "candidate_eval_summary.csv",
    "post_lift_place_audit.csv",
)

REQUIRED_JSONL_ARTIFACTS: tuple[str, ...] = (
    "paired_eval_per_seed.jsonl",
    "lifted_episode_index.jsonl",
)

REQUIRED_TEXT_ARTIFACTS: tuple[str, ...] = ("t8_2_summary.md",)

ALLOWED_STRATA: frozenset[str] = frozenset(
    {
        "BASE_SUCCESS",
        "BASE_LIFT_NO_SUCCESS",
        "BASE_REACHED_NO_LIFT",
        "BASE_NEVER_REACHED",
        "INVALID",
    }
)

SEED_SCAN_REQUIRED_FIELDS: tuple[str, ...] = (
    "max_candidate_seeds",
    "max_wall_time",
    "max_episode_steps",
    "base_policy_path",
    "canonical_surface_path",
    "exact_command_template",
)

CANDIDATE_SEED_REQUIRED_FIELDS: tuple[str, ...] = (
    "seed",
    "base_success",
    "base_reached",
    "base_lifted",
    "base_failure_mode",
    "reached_t",
    "lifted_t",
    "success_t",
    "apple_to_plate_min_after_lift",
    "stratum",
    "selected",
    "exclusion_reason",
)

PAIRED_ROW_REQUIRED_FIELDS: tuple[str, ...] = (
    "seed",
    "stratum",
    "row_id",
    "policy_or_splice",
    "success",
    "reached",
    "lifted",
    "failure_mode",
    "reached_t",
    "lifted_t",
    "success_t",
    "apple_to_plate_min_after_lift",
    "reached_plate_proxy",
    "forbidden_scope_pass",
)

PAIRED_SUMMARY_REQUIRED_FIELDS: tuple[str, ...] = (
    "ID",
    "seeds",
    "success",
    "reached",
    "lifted",
    "lift_given_reached",
    "success_given_lifted",
    "apple_to_plate_min_after_lift",
    "reached_plate_proxy",
    "failure_modes",
)

POST_LIFT_AUDIT_REQUIRED_FIELDS: tuple[str, ...] = (
    "seed",
    "stratum",
    "row_id",
    "episode_id",
    "lifted_t",
    "apple_height_peak",
    "carried_duration_after_lift",
    "min_apple_to_plate_dist_after_lift",
    "delta_apple_to_plate_dist_after_lift",
    "moved_toward_plate_after_lift",
    "reached_plate_proxy",
    "release_or_open_proxy_after_lift",
    "hand_close_open_profile_summary",
    "arm_energy_after_lift",
    "base_nav_energy_after_lift",
    "nav_projection_to_plate",
    "chunk_boundary_jump_q99",
    "final_failure_mode",
)

POST_LIFT_ANSWER_FIELDS: tuple[str, ...] = (
    "apple_moves_toward_plate_after_lift",
    "any_policy_reaches_plate_proxy",
    "hand_release_or_hold_timing",
    "transport_driver",
    "dominant_transition_failure",
)

_FORBIDDEN_COMMAND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("guarded_recap", re.compile(r"(?:--guarded-recap\b|\bguarded[_-]?recap\b)", re.I)),
    ("recap", re.compile(r"--recap\b", re.I)),
    ("fatg", re.compile(r"(?:--fatg\b|\bfatg\b)", re.I)),
    ("advantage", re.compile(r"--advantage\b", re.I)),
    ("per_edge", re.compile(r"--per-edge\b|\bper[_-]?edge\b", re.I)),
    ("full_scope", re.compile(r"--full-scope\b|\bfull[_-]?scope\b", re.I)),
    ("training", re.compile(r"--(?:train|run-training|optimizer-step)\b", re.I)),
    ("lora_merge", re.compile(r"--merge-lora\b|--merge-lora-before-eval\b", re.I)),
    ("trainer_fallback", re.compile(r"Gr00tTrainer\(|trainer\.train\(|launch_finetune\.py|finetune_full", re.I)),
)

_MODEL_ENV_COMMAND_HINTS: tuple[str, ...] = (
    "gr00t",
    "rollout",
    "eval",
    "seed",
    "policy",
    "robosuite",
    "env",
)

_TRUE_VALUES = {"1", "true", "yes", "y", "pass", "passed"}
_FALSE_VALUES = {"0", "false", "no", "n", "fail", "failed", ""}


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    status: str
    message: str
    artifacts: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.artifacts:
            payload["artifacts"] = list(self.artifacts)
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status(name: str, passed: bool, message: str, *, artifacts: Iterable[str] = (), details: Mapping[str, Any] | None = None) -> VerificationCheck:
    return VerificationCheck(
        name=name,
        status="PASS" if passed else "FAIL",
        message=message,
        artifacts=tuple(artifacts),
        details=dict(details or {}),
    )


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return repr(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, tuple(reader.fieldnames or ())


def _read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL line {index}: {exc}") from exc
    return rows


def _boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return None


def _extract_nested_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            found.extend(_extract_nested_strings(child))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for child in value:
            found.extend(_extract_nested_strings(child))
    return found


def _extract_commands(payload: Any) -> list[str]:
    commands: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key).lower()
            if "command" in key_text or key_text in {"argv", "args"}:
                if isinstance(value, str):
                    commands.append(value)
                elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
                    if key_text in {"argv", "args"} and all(isinstance(item, (str, int, float)) for item in value):
                        commands.append(" ".join(shlex.quote(str(item)) for item in value))
                    elif all(isinstance(item, str) for item in value):
                        commands.extend(str(item) for item in value)
                    else:
                        commands.extend(_extract_commands(value))
                elif isinstance(value, Mapping):
                    commands.extend(_extract_commands(value))
            elif isinstance(value, (Mapping, list, tuple)):
                commands.extend(_extract_commands(value))
    elif isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray, str)):
        for item in payload:
            commands.extend(_extract_commands(item))
    return commands


def _path_from_artifact(root: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return root / path


def _iter_evidence_rows(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        has_artifact_shape = "path" in value and (
            "sha256" in value or "hash" in value or "exists" in value
        )
        if has_artifact_shape:
            yield value
        for child in value.values():
            yield from _iter_evidence_rows(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _iter_evidence_rows(child)


def _extract_t8_1_final_decision(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    direct_keys = (
        "t8_1_final_decision",
        "t8_1_decision",
        "source_t8_1_final_decision",
        "source_final_decision",
    )
    for key in direct_keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    for key in ("t8_1", "source_t8_1", "source", "final_decision"):
        value = payload.get(key)
        if isinstance(value, str) and value == "BASE_SEEDS_TOO_HARD":
            return value
        if isinstance(value, Mapping):
            nested = value.get("final_decision") or value.get("decision")
            if isinstance(nested, str):
                return nested
    return None


def _extract_forbidden_lock_status(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, Mapping):
        return False, "evidence_lock payload is not an object"

    candidate_keys = (
        "forbidden_branch_status",
        "forbidden_scope_status",
        "forbidden_routes_status",
        "forbidden_branch_guard",
        "forbidden_scope_guard",
    )
    for key in candidate_keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, str):
            upper = value.upper()
            if any(token in upper for token in ("FAIL", "ACCEPTED", "ALLOW", "UNSAFE")):
                return False, f"{key}={value!r}"
            if any(token in upper for token in ("PASS", "REJECT", "FORBIDDEN_ROUTES_REJECTED", "BLOCK")):
                return True, f"{key}={value!r}"
        if isinstance(value, bool):
            # A boolean field with this name is interpreted as "guard passed".
            return value, f"{key}={value!r}"
        if isinstance(value, Mapping):
            status = value.get("status") or value.get("decision")
            if isinstance(status, str):
                upper = status.upper()
                if any(token in upper for token in ("FAIL", "ACCEPTED", "ALLOW", "UNSAFE")):
                    return False, f"{key}.status={status!r}"
                if any(token in upper for token in ("PASS", "REJECT", "BLOCK")):
                    return True, f"{key}.status={status!r}"

    strings = "\n".join(_extract_nested_strings(payload)).upper()
    if "FORBIDDEN_ROUTE_ACCEPTED" in strings or "FORBIDDEN_BRANCH_ACCEPTED" in strings:
        return False, "evidence lock contains accepted forbidden route marker"
    if "FORBIDDEN_ROUTES_REJECTED" in strings or "FORBIDDEN_SCOPE_PASS" in strings:
        return True, "evidence lock contains forbidden-routes-rejected marker"
    return False, "missing explicit forbidden branch/scope status"


def _validate_artifact_presence(root: Path) -> list[VerificationCheck]:
    checks: list[VerificationCheck] = []
    required = (
        [("json", name) for name in REQUIRED_JSON_ARTIFACTS]
        + [("csv", name) for name in REQUIRED_CSV_ARTIFACTS]
        + [("jsonl", name) for name in REQUIRED_JSONL_ARTIFACTS]
        + [("text", name) for name in REQUIRED_TEXT_ARTIFACTS]
    )
    missing = [name for _kind, name in required if not (root / name).is_file()]
    present = [name for _kind, name in required if (root / name).is_file()]
    checks.append(
        _status(
            "artifact_presence",
            not missing,
            "all required T8.2 verifier artifacts are present" if not missing else "missing required artifacts",
            artifacts=present,
            details={"missing": missing, "required": [name for _kind, name in required]},
        )
    )
    return checks


def _validate_json_and_jsonl_parse(root: Path) -> list[VerificationCheck]:
    checks: list[VerificationCheck] = []
    parse_errors: list[str] = []
    parsed_json: list[str] = []
    for name in REQUIRED_JSON_ARTIFACTS:
        path = root / name
        if not path.is_file():
            continue
        try:
            _load_json(path)
            parsed_json.append(name)
        except Exception as exc:  # noqa: BLE001 - report parser evidence, do not hide detail.
            parse_errors.append(f"{name}: {exc}")
    for name in REQUIRED_JSONL_ARTIFACTS:
        path = root / name
        if not path.is_file():
            continue
        try:
            _read_jsonl(path)
            parsed_json.append(name)
        except Exception as exc:  # noqa: BLE001
            parse_errors.append(f"{name}: {exc}")
    checks.append(
        _status(
            "json_parse",
            not parse_errors,
            "JSON/JSONL artifacts parse" if not parse_errors else "JSON/JSONL parse errors",
            artifacts=parsed_json,
            details={"errors": parse_errors},
        )
    )
    return checks


def validate_evidence_lock(root: Path) -> tuple[list[VerificationCheck], Mapping[str, Any] | None]:
    path = root / "evidence_lock.json"
    if not path.is_file():
        return [_status("evidence_lock", False, "evidence_lock.json is missing")], None
    try:
        payload = _load_json(path)
    except Exception as exc:  # noqa: BLE001
        return [_status("evidence_lock", False, f"evidence_lock.json is invalid: {exc}", artifacts=("evidence_lock.json",))], None
    if not isinstance(payload, Mapping):
        return [_status("evidence_lock", False, "evidence_lock.json must contain an object", artifacts=("evidence_lock.json",))], None

    checks: list[VerificationCheck] = []
    status_value = payload.get("status")
    status_ok = status_value is None or str(status_value).upper() in {"PASS", "OK", "LOCKED"}
    t8_1_final = _extract_t8_1_final_decision(payload)
    forbidden_ok, forbidden_message = _extract_forbidden_lock_status(payload)

    rows = list(_iter_evidence_rows(payload))
    row_errors: list[str] = []
    cited_paths: list[str] = []
    for index, row in enumerate(rows):
        raw_path = row.get("path")
        artifact_path = _path_from_artifact(root, raw_path)
        row_name = str(row.get("name") or row.get("id") or index)
        exists_field = row.get("exists")
        if exists_field is not None and _boolish(exists_field) is False:
            row_errors.append(f"{row_name}: exists=false")
        if artifact_path is None:
            row_errors.append(f"{row_name}: missing path")
            continue
        cited_paths.append(str(raw_path))
        if not artifact_path.is_file() and not artifact_path.is_dir():
            row_errors.append(f"{row_name}: path not found: {artifact_path}")
            continue
        expected_hash = row.get("sha256") or row.get("hash")
        if expected_hash is not None:
            hash_text = str(expected_hash)
            if artifact_path.is_dir():
                continue
            if not re.fullmatch(r"[0-9a-fA-F]{64}", hash_text):
                row_errors.append(f"{row_name}: invalid sha256 {hash_text!r}")
            else:
                actual = sha256_file(artifact_path)
                if actual.lower() != hash_text.lower():
                    row_errors.append(f"{row_name}: sha256 mismatch")

    checks.append(
        _status(
            "evidence_lock",
            bool(status_ok and t8_1_final == "BASE_SEEDS_TOO_HARD" and forbidden_ok and rows and not row_errors),
            "source evidence lock passed"
            if status_ok and t8_1_final == "BASE_SEEDS_TOO_HARD" and forbidden_ok and rows and not row_errors
            else "source evidence lock failed",
            artifacts=("evidence_lock.json", *cited_paths),
            details={
                "status": status_value,
                "t8_1_final_decision": t8_1_final,
                "forbidden_status": forbidden_message,
                "row_count": len(rows),
                "errors": row_errors,
            },
        )
    )
    return checks, payload


def _find_forbidden_allowed_flags(value: Any, path: str = "$", errors: list[str] | None = None) -> list[str]:
    out = errors if errors is not None else []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            next_path = f"{path}.{key_text}"
            lowered = key_text.lower()
            if lowered in {
                "training_allowed",
                "optimizer_step_allowed",
                "checkpoint_update_allowed",
                "guarded_recap_allowed",
                "recap_allowed",
                "fatg_allowed",
                "per_edge_lora_allowed",
                "full_scope_update_allowed",
                "lora_merge_allowed",
                "lora_merged_before_eval",
                "forbidden_route_accepted",
            }:
                truth = _boolish(child)
                if truth is True:
                    out.append(f"{next_path}=true")
            _find_forbidden_allowed_flags(child, next_path, out)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _find_forbidden_allowed_flags(child, f"{path}[{index}]", out)
    return out


def validate_forbidden_scope(root: Path) -> list[VerificationCheck]:
    path = root / "command_manifest.json"
    if not path.is_file():
        return [_status("forbidden_scope_guard", False, "command_manifest.json is missing")]
    try:
        payload = _load_json(path)
    except Exception as exc:  # noqa: BLE001
        return [_status("forbidden_scope_guard", False, f"command_manifest.json is invalid: {exc}", artifacts=("command_manifest.json",))]

    errors: list[str] = []
    commands = _extract_commands(payload)
    for command in commands:
        for reason, pattern in _FORBIDDEN_COMMAND_PATTERNS:
            if pattern.search(command):
                # Allow explicitly negative-test/rejection commands to be documented without being treated as accepted routes.
                lowered = command.lower()
                if "negative" in lowered or "reject" in lowered or "self-test" in lowered:
                    continue
                errors.append(f"accepted forbidden command route {reason}: {command}")
        if any(hint in command.lower() for hint in _MODEL_ENV_COMMAND_HINTS):
            has_timeout = "timeout" in shlex.split(command) if command.strip() else False
            has_cuda_1 = "CUDA_VISIBLE_DEVICES=1" in command.replace(" ", "") or "CUDA_VISIBLE_DEVICES='1'" in command
            if not has_timeout or not has_cuda_1:
                errors.append(f"model/env command lacks timeout env CUDA_VISIBLE_DEVICES=1: {command}")

    errors.extend(_find_forbidden_allowed_flags(payload))

    for key in ("submodule_status_short", "git_submodule_status_after", "submodules_after"):
        value = payload.get(key) if isinstance(payload, Mapping) else None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) > 0:
            errors.append(f"{key} is not clean: {value}")
        elif isinstance(value, str) and value.strip():
            errors.append(f"{key} is not clean: {value}")

    return [
        _status(
            "forbidden_scope_guard",
            not errors,
            "no forbidden scope route accepted" if not errors else "forbidden scope guard failed",
            artifacts=("command_manifest.json",),
            details={"commands_checked": commands, "errors": errors},
        )
    ]


def _extract_final_decision(payload: Any) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return None, ["final_decision.json must contain an object"]
    value = payload.get("final_decision")
    if isinstance(value, str):
        decision = value
    elif value is None:
        decision = None
        errors.append("missing scalar final_decision")
    else:
        decision = None
        errors.append("final_decision must be exactly one scalar string")

    suspicious_multi = payload.get("final_decisions") or payload.get("decisions")
    if suspicious_multi is not None:
        errors.append("multi-decision fields are not allowed in final_decision.json")
    return decision, errors


def validate_final_decision(root: Path) -> tuple[list[VerificationCheck], str | None]:
    path = root / "final_decision.json"
    if not path.is_file():
        return [_status("final_enum", False, "final_decision.json is missing")], None
    try:
        payload = _load_json(path)
    except Exception as exc:  # noqa: BLE001
        return [_status("final_enum", False, f"final_decision.json is invalid: {exc}", artifacts=("final_decision.json",))], None
    decision, errors = _extract_final_decision(payload)
    if decision in LEGACY_T8_FINAL_DECISIONS:
        errors.append(f"legacy T8/T8.1 enum is invalid for T8.2: {decision}")
    if decision is not None and decision not in ALLOWED_T8_2_FINAL_DECISIONS:
        errors.append(f"final_decision is not in the user-provided T8.2 enum list: {decision}")

    allowed_payload = payload.get("allowed_final_decisions") if isinstance(payload, Mapping) else None
    if list(allowed_payload or []) != list(ALLOWED_T8_2_FINAL_DECISIONS):
        errors.append("allowed_final_decisions must exactly equal the user-provided T8.2 list and order")

    return [
        _status(
            "final_enum",
            not errors,
            "final decision is exactly one allowed T8.2 enum" if not errors else "final enum validation failed",
            artifacts=("final_decision.json",),
            details={
                "final_decision": decision,
                "allowed_final_decisions": list(ALLOWED_T8_2_FINAL_DECISIONS),
                "errors": errors,
            },
        )
    ], decision


def validate_seed_bank_schema(root: Path, final_decision: str | None) -> list[VerificationCheck]:
    errors: list[str] = []
    artifacts: list[str] = []

    manifest_path = root / "seed_scan_manifest.json"
    if manifest_path.is_file():
        artifacts.append("seed_scan_manifest.json")
        manifest = _load_json(manifest_path)
        if not isinstance(manifest, Mapping):
            errors.append("seed_scan_manifest.json must contain an object")
        else:
            has_seed_origin = any(key in manifest for key in ("seed_start", "seed_sources"))
            if not has_seed_origin:
                errors.append("seed_scan_manifest.json requires seed_start or seed_sources")
            for field_name in SEED_SCAN_REQUIRED_FIELDS:
                if field_name not in manifest:
                    errors.append(f"seed_scan_manifest.json missing {field_name}")
    else:
        errors.append("seed_scan_manifest.json missing")

    scout_path = root / "candidate_seed_scout.csv"
    selected_counts: dict[str, int] = {stratum: 0 for stratum in ALLOWED_STRATA}
    selected_seeds: set[str] = set()
    if scout_path.is_file():
        artifacts.append("candidate_seed_scout.csv")
        rows, headers = _read_csv(scout_path)
        missing_headers = [field for field in CANDIDATE_SEED_REQUIRED_FIELDS if field not in headers]
        if missing_headers:
            errors.append(f"candidate_seed_scout.csv missing headers: {missing_headers}")
        for index, row in enumerate(rows, start=2):
            stratum = row.get("stratum", "")
            if stratum not in ALLOWED_STRATA:
                errors.append(f"candidate_seed_scout.csv line {index}: invalid stratum {stratum!r}")
            if _boolish(row.get("selected")) is True:
                selected_counts[stratum] = selected_counts.get(stratum, 0) + 1
                selected_seeds.add(str(row.get("seed")))
            elif row.get("exclusion_reason", "") == "" and _boolish(row.get("selected")) is False:
                errors.append(f"candidate_seed_scout.csv line {index}: excluded seed requires exclusion_reason")
    else:
        errors.append("candidate_seed_scout.csv missing")

    for name in ("seed_bank.json", "seed_bank_deficits.json"):
        path = root / name
        if path.is_file():
            artifacts.append(name)
            obj = _load_json(path)
            if not isinstance(obj, Mapping):
                errors.append(f"{name} must contain an object")
        else:
            errors.append(f"{name} missing")

    base_success = selected_counts.get("BASE_SUCCESS", 0)
    if base_success < 10 and final_decision != "BASE_PROTOCOL_TOO_WEAK":
        errors.append(
            "selected BASE_SUCCESS < 10 requires final_decision=BASE_PROTOCOL_TOO_WEAK"
        )

    return [
        _status(
            "seed_bank_schema",
            not errors,
            "seed scout and seed bank schemas passed" if not errors else "seed bank schema validation failed",
            artifacts=artifacts,
            details={
                "selected_counts": selected_counts,
                "selected_seed_count": len(selected_seeds),
                "errors": errors,
            },
        )
    ]


def validate_paired_eval_schema(root: Path) -> list[VerificationCheck]:
    errors: list[str] = []
    artifacts: list[str] = []
    selected_seeds: set[str] = set()
    scout_path = root / "candidate_seed_scout.csv"
    if scout_path.is_file():
        rows, _headers = _read_csv(scout_path)
        selected_seeds = {str(row.get("seed")) for row in rows if _boolish(row.get("selected")) is True}

    per_seed_path = root / "paired_eval_per_seed.jsonl"
    rows: list[Mapping[str, Any]] = []
    if per_seed_path.is_file():
        artifacts.append("paired_eval_per_seed.jsonl")
        raw_rows = _read_jsonl(per_seed_path)
        for index, row in enumerate(raw_rows, start=1):
            if not isinstance(row, Mapping):
                errors.append(f"paired_eval_per_seed.jsonl line {index}: row must be an object")
                continue
            missing = [field for field in PAIRED_ROW_REQUIRED_FIELDS if field not in row]
            if missing:
                errors.append(f"paired_eval_per_seed.jsonl line {index}: missing {missing}")
            if row.get("stratum") not in ALLOWED_STRATA:
                errors.append(f"paired_eval_per_seed.jsonl line {index}: invalid stratum {row.get('stratum')!r}")
            if _boolish(row.get("forbidden_scope_pass")) is not True:
                errors.append(f"paired_eval_per_seed.jsonl line {index}: forbidden_scope_pass must be true")
            rows.append(row)
    else:
        errors.append("paired_eval_per_seed.jsonl missing")

    required_controls = {"B0", "B1", "B2"}
    if selected_seeds and rows:
        controls_by_seed: dict[str, set[str]] = {seed: set() for seed in selected_seeds}
        for row in rows:
            seed = str(row.get("seed"))
            row_id = str(row.get("row_id"))
            if seed in controls_by_seed and row_id in required_controls:
                controls_by_seed[seed].add(row_id)
        missing_controls = {
            seed: sorted(required_controls - row_ids)
            for seed, row_ids in controls_by_seed.items()
            if required_controls - row_ids
        }
        if missing_controls:
            errors.append(f"selected seeds missing same-seed B0/B1/B2 controls: {missing_controls}")

    summary_path = root / "paired_eval_summary.csv"
    if summary_path.is_file():
        artifacts.append("paired_eval_summary.csv")
        _rows, headers = _read_csv(summary_path)
        missing = [field for field in PAIRED_SUMMARY_REQUIRED_FIELDS if field not in headers]
        if missing:
            errors.append(f"paired_eval_summary.csv missing headers: {missing}")
    else:
        errors.append("paired_eval_summary.csv missing")

    candidate_summary_path = root / "candidate_eval_summary.csv"
    if candidate_summary_path.is_file():
        artifacts.append("candidate_eval_summary.csv")
        _rows, headers = _read_csv(candidate_summary_path)
        for field_name in ("ID", "n_seeds", "success", "reached", "lifted", "delta_vs_B0", "delta_vs_B2"):
            if field_name not in headers:
                errors.append(f"candidate_eval_summary.csv missing header: {field_name}")
    else:
        errors.append("candidate_eval_summary.csv missing")

    for name in ("control_regression_report.json", "stratum_effects.json"):
        path = root / name
        if path.is_file():
            artifacts.append(name)
            obj = _load_json(path)
            if not isinstance(obj, Mapping):
                errors.append(f"{name} must contain an object")
            elif str(obj.get("status", "PASS")).upper() not in {"PASS", "OK", "SKIPPED_QUALITATIVE"}:
                errors.append(f"{name} status is not PASS/OK/SKIPPED_QUALITATIVE: {obj.get('status')!r}")
        else:
            errors.append(f"{name} missing")

    return [
        _status(
            "paired_eval_schema",
            not errors,
            "paired eval schema and same-seed controls passed" if not errors else "paired eval schema validation failed",
            artifacts=artifacts,
            details={"selected_seed_count": len(selected_seeds), "errors": errors},
        )
    ]


def validate_post_lift_schema(root: Path) -> list[VerificationCheck]:
    errors: list[str] = []
    artifacts: list[str] = []

    audit_path = root / "post_lift_place_audit.json"
    if audit_path.is_file():
        artifacts.append("post_lift_place_audit.json")
        audit = _load_json(audit_path)
        if not isinstance(audit, Mapping):
            errors.append("post_lift_place_audit.json must contain an object")
        else:
            answers = audit.get("answers")
            if not isinstance(answers, Mapping):
                errors.append("post_lift_place_audit.json missing answers object")
            else:
                missing = [field for field in POST_LIFT_ANSWER_FIELDS if field not in answers]
                if missing:
                    errors.append(f"post_lift_place_audit.json answers missing: {missing}")
            coverage = audit.get("coverage")
            if isinstance(coverage, Mapping) and _boolish(coverage.get("complete")) is False:
                errors.append("post-lift audit coverage is incomplete")
    else:
        errors.append("post_lift_place_audit.json missing")

    csv_path = root / "post_lift_place_audit.csv"
    if csv_path.is_file():
        artifacts.append("post_lift_place_audit.csv")
        _rows, headers = _read_csv(csv_path)
        missing = [field for field in POST_LIFT_AUDIT_REQUIRED_FIELDS if field not in headers]
        if missing:
            errors.append(f"post_lift_place_audit.csv missing headers: {missing}")
    else:
        errors.append("post_lift_place_audit.csv missing")

    jsonl_path = root / "lifted_episode_index.jsonl"
    if jsonl_path.is_file():
        artifacts.append("lifted_episode_index.jsonl")
        for index, row in enumerate(_read_jsonl(jsonl_path), start=1):
            if not isinstance(row, Mapping):
                errors.append(f"lifted_episode_index.jsonl line {index}: row must be object")
    else:
        errors.append("lifted_episode_index.jsonl missing")

    return [
        _status(
            "post_lift_schema",
            not errors,
            "post-lift/place audit schema passed" if not errors else "post-lift/place schema validation failed",
            artifacts=artifacts,
            details={"errors": errors},
        )
    ]


def validate_summary_citations(root: Path, final_decision: str | None) -> list[VerificationCheck]:
    path = root / "t8_2_summary.md"
    if not path.is_file():
        return [_status("summary_citations", False, "t8_2_summary.md is missing")]
    text = path.read_text(encoding="utf-8")
    missing_json = [name for name in ("evidence_lock.json", "seed_bank.json", "final_decision.json") if name not in text]
    missing_csv = [name for name in ("candidate_seed_scout.csv", "paired_eval_summary.csv", "post_lift_place_audit.csv") if name not in text]
    final_mentions = [decision for decision in ALLOWED_T8_2_FINAL_DECISIONS if decision in text]
    errors: list[str] = []
    if missing_json:
        errors.append(f"summary missing JSON artifact citations: {missing_json}")
    if missing_csv:
        errors.append(f"summary missing CSV artifact citations: {missing_csv}")
    if final_decision is None or final_mentions != [final_decision]:
        errors.append(
            "summary must cite exactly one allowed final enum, matching final_decision.json"
        )
    if "Guarded RECAP" not in text and "GUARDED_RECAP" not in text:
        errors.append("summary must state forbidden branches that remain forbidden")
    return [
        _status(
            "summary_citations",
            not errors,
            "summary cites JSON/CSV artifacts and exactly one final enum" if not errors else "summary citation validation failed",
            artifacts=("t8_2_summary.md",),
            details={"errors": errors, "final_enum_mentions": final_mentions},
        )
    ]


def verify_t8_2_artifact_root(root: str | Path, *, output: str | Path | None = None) -> dict[str, Any]:
    artifact_root = Path(root).expanduser().resolve()
    checks: list[VerificationCheck] = []
    checks.extend(_validate_artifact_presence(artifact_root))
    checks.extend(_validate_json_and_jsonl_parse(artifact_root))
    evidence_checks, _evidence_payload = validate_evidence_lock(artifact_root)
    checks.extend(evidence_checks)
    checks.extend(validate_forbidden_scope(artifact_root))
    final_checks, final_decision = validate_final_decision(artifact_root)
    checks.extend(final_checks)
    checks.extend(validate_seed_bank_schema(artifact_root, final_decision))
    checks.extend(validate_paired_eval_schema(artifact_root))
    checks.extend(validate_post_lift_schema(artifact_root))
    checks.extend(validate_summary_citations(artifact_root, final_decision))

    status = "PASS" if all(check.status == "PASS" for check in checks) else "FAIL"
    json_artifacts = [name for name in REQUIRED_JSON_ARTIFACTS if (artifact_root / name).is_file()]
    csv_artifacts = [name for name in REQUIRED_CSV_ARTIFACTS if (artifact_root / name).is_file()]
    payload: dict[str, Any] = {
        "schema_version": "gr00t_t8_2_post_run_verification_v1",
        "status": status,
        "artifact_root": str(artifact_root),
        "generated_at_utc": utc_now(),
        "final_decision": final_decision,
        "allowed_final_decisions": list(ALLOWED_T8_2_FINAL_DECISIONS),
        "cited_json_artifacts": json_artifacts,
        "cited_csv_artifacts": csv_artifacts,
        "checks": [check.to_jsonable() for check in checks],
    }
    out_path = Path(output).expanduser().resolve() if output is not None else artifact_root / "post_run_verification.json"
    _write_json(out_path, payload)
    payload["post_run_verification_path"] = str(out_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, help="T8.2 artifact root to verify")
    parser.add_argument(
        "--output",
        help="Output post_run_verification.json path; defaults to <artifact-root>/post_run_verification.json",
    )
    parser.add_argument("--print-json", action="store_true", help="Print the verification payload")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = verify_t8_2_artifact_root(args.artifact_root, output=args.output)
    if args.print_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
