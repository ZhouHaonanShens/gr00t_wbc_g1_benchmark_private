from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]

T8_1_RUN = "gr00t_t8_1_nav_postlift_20260508_164110"
T8_SOURCE_RUN = "gr00t_t8_no_recap_hand_smoke_20260508_052919"
T8_1_ROOT_REL = Path("agent/artifacts/gr00t_t8_1_nav_postlift") / T8_1_RUN
T8_SOURCE_ROOT_REL = Path("agent/artifacts/gr00t_t8_no_recap_hand_smoke") / T8_SOURCE_RUN

DEFAULT_SEED_START = 2026052000
DEFAULT_MAX_CANDIDATE_SEEDS = 200
DEFAULT_MAX_WALL_TIME = "24h"
DEFAULT_MAX_EPISODE_STEPS = 720
DEFAULT_TIMEOUT_SECONDS = 24 * 60 * 60
MIN_BASE_SUCCESS = 10
MAX_SELECTED_PER_STRATUM = 15

ALLOWED_STRATA = (
    "BASE_SUCCESS",
    "BASE_LIFT_NO_SUCCESS",
    "BASE_REACHED_NO_LIFT",
    "BASE_NEVER_REACHED",
    "INVALID",
)
MATERIAL_STRATA = ALLOWED_STRATA[:-1]

ALLOWED_FINAL_DECISIONS = (
    "BASE_PROTOCOL_TOO_WEAK",
    "SAFE_SFT_NONCOLLAPSE_PRELIM",
    "REACH_NAV_BLOCKER",
    "POST_LIFT_PLACE_BLOCKER",
    "POST_LIFT_SPLICE_IDENTIFIED_FIX",
    "HAND_REGRESSION",
    "READY_FOR_SAFE_SFT_30",
    "GUARDED_RECAP_STILL_FORBIDDEN",
)

FORBIDDEN_ROUTE_FLAGS = (
    "--recap",
    "--guarded-recap",
    "--fatg",
    "--advantage",
    "--advantage-weighting",
    "--per-edge",
    "--full-scope",
    "--full-head",
    "--train",
    "--run-training",
    "--optimizer-step",
    "--merge-lora",
    "--merge-lora-before-eval",
    "--tune-action-decoder",
    "--tune-vlm",
    "--tune-visual",
    "--tune-llm",
    "--tune-projector",
    "--tune-state-encoder",
    "--tune-adaln",
    "--tune-timestep",
)

STATIC_FORBIDDEN_TOKENS = (
    "Gr00t" + "Trainer(",
    "trainer" + ".train(",
    "launch" + "_finetune",
    "guarded" + "_recap_train_loop",
    "fatg" + "_train_loop",
    "merge" + "_lora_before_eval(",
)

CANDIDATE_SCOUT_FIELDS = (
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
    "steps_jsonl",
)

SEED_BANK_TARGETS = {stratum: {"min": 10, "max": 15} for stratum in MATERIAL_STRATA}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: set[str] = set()
        for row in rows:
            keys.update(str(k) for k in row)
        fieldnames = sorted(keys) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=True, default=json_default)
                    if isinstance(v, (dict, list, tuple))
                    else v
                    for k, v in row.items()
                }
            )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def resolve_path(raw: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def git_output(args: Sequence[str], cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNKNOWN:{type(exc).__name__}:{exc}"


def reject_forbidden_args(argv: Sequence[str]) -> None:
    present: list[str] = []
    for arg in argv:
        flag = arg.split("=", 1)[0]
        if flag in FORBIDDEN_ROUTE_FLAGS:
            present.append(flag)
    if present:
        raise SystemExit(f"Forbidden T8.2 route flags rejected before model load: {sorted(set(present))}")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "pass"}:
        return True
    if text in {"0", "false", "f", "no", "n", "fail", "", "none", "null"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def classify_base_stratum(*, base_success: Any, base_reached: Any, base_lifted: Any, valid: bool = True) -> str:
    if not valid:
        return "INVALID"
    success = parse_bool(base_success)
    reached = parse_bool(base_reached)
    lifted = parse_bool(base_lifted)
    if success:
        return "BASE_SUCCESS"
    if lifted:
        return "BASE_LIFT_NO_SUCCESS"
    if reached:
        return "BASE_REACHED_NO_LIFT"
    return "BASE_NEVER_REACHED"


def _path_record(name: str, path: Path, *, kind: str = "file", root: Path) -> dict[str, Any]:
    exists = path.is_dir() if kind == "dir" else path.is_file()
    return {
        "name": name,
        "kind": kind,
        "path": rel(path, root),
        "exists": exists,
        "sha256": sha256_file(path) if exists and kind == "file" else None,
    }


def build_evidence_lock(*, evidence_root: Path) -> dict[str, Any]:
    paths = {
        "t8_1_final_decision": (T8_1_ROOT_REL / "final_decision.json", "file"),
        "t8_1_summary": (T8_1_ROOT_REL / "t8_1_summary.md", "file"),
        "t8_1_verified_interpretation": (T8_1_ROOT_REL / "t8_1_verified_interpretation.md", "file"),
        "t8_source_root": (T8_SOURCE_ROOT_REL, "dir"),
        "t8_source_final_decision": (T8_SOURCE_ROOT_REL / "final_decision.json", "file"),
        "s2_checkpoint_manifest": (T8_SOURCE_ROOT_REL / "cells/S2_BASE_TEACHER/checkpoint_manifest.json", "file"),
    }
    records = [
        _path_record(name, evidence_root / relative, kind=kind, root=evidence_root)
        for name, (relative, kind) in paths.items()
    ]
    missing = [row["name"] for row in records if not row["exists"]]
    t8_1_final: dict[str, Any] = {}
    t8_source_final: dict[str, Any] = {}
    s2_manifest: dict[str, Any] = {}
    errors: list[str] = []
    t8_1_final_path = evidence_root / paths["t8_1_final_decision"][0]
    t8_source_final_path = evidence_root / paths["t8_source_final_decision"][0]
    s2_manifest_path = evidence_root / paths["s2_checkpoint_manifest"][0]
    if t8_1_final_path.is_file():
        t8_1_final = load_json(t8_1_final_path)
    if t8_source_final_path.is_file():
        t8_source_final = load_json(t8_source_final_path)
    if s2_manifest_path.is_file():
        s2_manifest = load_json(s2_manifest_path)
    if t8_1_final and t8_1_final.get("final_decision") != "BASE_SEEDS_TOO_HARD":
        errors.append("t8_1_final_decision_not_BASE_SEEDS_TOO_HARD")
    if s2_manifest:
        if s2_manifest.get("checkpoint_type") != "unmerged_lora_adapter_only":
            errors.append("s2_checkpoint_not_unmerged_lora_adapter_only")
        if s2_manifest.get("lora_merged_before_eval") is not False:
            errors.append("s2_lora_merge_status_not_false")
    forbidden_branch_status = {
        "guarded_recap_allowed": bool(t8_1_final.get("guarded_recap_allowed", False))
        if t8_1_final
        else None,
        "fatg_allowed": bool(t8_1_final.get("fatg_allowed", False)) if t8_1_final else None,
        "t8_source_guarded_recap_allowed": bool(t8_source_final.get("guarded_recap_allowed", False))
        if t8_source_final
        else None,
        "t8_source_fatg_allowed": bool(t8_source_final.get("fatg_allowed", False))
        if t8_source_final
        else None,
        "forbidden_branches_remain_forbidden": True,
    }
    if forbidden_branch_status["guarded_recap_allowed"] or forbidden_branch_status["fatg_allowed"]:
        errors.append("t8_1_forbidden_branch_marked_allowed")
    if (
        forbidden_branch_status["t8_source_guarded_recap_allowed"]
        or forbidden_branch_status["t8_source_fatg_allowed"]
    ):
        errors.append("t8_source_forbidden_branch_marked_allowed")
    status = "PASS" if not missing and not errors else "FAIL"
    return {
        "schema_version": "gr00t_t8_2_evidence_lock_v1",
        "status": status,
        "generated_at_utc": utc_now(),
        "evidence_root": str(evidence_root),
        "required_inputs": records,
        "missing": missing,
        "errors": errors,
        "t8_1_final_decision": t8_1_final.get("final_decision"),
        "t8_source_final_decision": t8_source_final.get("final_decision"),
        "s2_checkpoint_type": s2_manifest.get("checkpoint_type"),
        "s2_lora_merged_before_eval": s2_manifest.get("lora_merged_before_eval"),
        "forbidden_branch_status": forbidden_branch_status,
    }


def command_has_timeout_and_gpu(command: str) -> bool:
    return "timeout" in command.split() and "CUDA_VISIBLE_DEVICES=1" in command


def build_eval_command_template(
    *,
    output_dir_placeholder: str = "<artifact_root>",
    seed_start: int = DEFAULT_SEED_START,
    max_candidate_seeds: int = DEFAULT_MAX_CANDIDATE_SEEDS,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    max_wall_time: str = DEFAULT_MAX_WALL_TIME,
) -> str:
    return (
        f"timeout {max_wall_time} env CUDA_VISIBLE_DEVICES=1 NO_ALBUMENTATIONS_UPDATE=1 "
        "python3 work/recap/scripts/gr00t_t8_2_seedbank_postlift.py bootstrap "
        f"--output-dir {output_dir_placeholder} "
        f"--seed-start {int(seed_start)} "
        f"--max-candidate-seeds {int(max_candidate_seeds)} "
        f"--max-episode-steps {int(max_episode_steps)}"
    )


def build_seed_scan_manifest(
    *,
    seed_start: int | None,
    seed_sources: Sequence[int] | None,
    max_candidate_seeds: int,
    max_wall_time: str,
    timeout_seconds: int,
    max_episode_steps: int,
    base_policy_path: str,
    canonical_surface_path: str,
    exact_command_template: str,
    scan_window_mini_adr: str | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": "gr00t_t8_2_seed_scan_manifest_v1",
        "generated_at_utc": utc_now(),
        "seed_start": seed_start,
        "seed_sources": list(seed_sources or []),
        "max_candidate_seeds": int(max_candidate_seeds),
        "max_wall_time": str(max_wall_time),
        "timeout_envelope": {"required": True, "timeout_seconds": int(timeout_seconds)},
        "max_episode_steps": int(max_episode_steps),
        "base_policy_path": str(base_policy_path),
        "canonical_surface_path": str(canonical_surface_path),
        "exact_command_template": str(exact_command_template),
        "defaults": {
            "max_candidate_seeds": DEFAULT_MAX_CANDIDATE_SEEDS,
            "max_wall_time": DEFAULT_MAX_WALL_TIME,
            "max_episode_steps": DEFAULT_MAX_EPISODE_STEPS,
        },
        "seed_bank_targets": SEED_BANK_TARGETS,
        "base_success_hard_gate": {"min_selected_base_success": MIN_BASE_SUCCESS, "failure_final_decision": "BASE_PROTOCOL_TOO_WEAK"},
        "no_training_scope": {
            "optimizer_steps": 0,
            "lora_update": False,
            "lora_merge_before_eval": False,
            "guarded_recap": False,
            "fatg": False,
            "per_edge": False,
            "full_scope": False,
        },
    }
    if scan_window_mini_adr:
        manifest["scan_window_mini_adr"] = scan_window_mini_adr
    return manifest


def validate_seed_scan_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if manifest.get("seed_start") in (None, "") and not manifest.get("seed_sources"):
        errors.append("seed_start_or_seed_sources_required")
    if int(manifest.get("max_candidate_seeds") or 0) <= 0:
        errors.append("max_candidate_seeds_positive_required")
    if not manifest.get("max_wall_time"):
        errors.append("max_wall_time_required")
    timeout = manifest.get("timeout_envelope")
    if not isinstance(timeout, Mapping) or not timeout.get("required") or int(timeout.get("timeout_seconds") or 0) <= 0:
        errors.append("timeout_envelope_required")
    if int(manifest.get("max_episode_steps") or 0) <= 0:
        errors.append("max_episode_steps_positive_required")
    if not manifest.get("base_policy_path"):
        errors.append("base_policy_path_required")
    if not manifest.get("canonical_surface_path"):
        errors.append("canonical_surface_path_required")
    command = str(manifest.get("exact_command_template") or "")
    if not command:
        errors.append("exact_command_template_required")
    elif not command_has_timeout_and_gpu(command):
        errors.append("exact_command_template_must_include_timeout_and_cuda_visible_devices_1")
    defaults_changed = (
        int(manifest.get("max_candidate_seeds") or 0) != DEFAULT_MAX_CANDIDATE_SEEDS
        or str(manifest.get("max_wall_time") or "") != DEFAULT_MAX_WALL_TIME
        or int(manifest.get("max_episode_steps") or 0) != DEFAULT_MAX_EPISODE_STEPS
    )
    if defaults_changed and not manifest.get("scan_window_mini_adr"):
        errors.append("scan_window_mini_adr_required_when_scan_window_defaults_change")
    no_training = manifest.get("no_training_scope")
    if not isinstance(no_training, Mapping):
        errors.append("no_training_scope_required")
    else:
        if int(no_training.get("optimizer_steps") or 0) != 0:
            errors.append("optimizer_steps_must_be_zero")
        for key in ("lora_update", "lora_merge_before_eval", "guarded_recap", "fatg", "per_edge", "full_scope"):
            if no_training.get(key) is not False:
                errors.append(f"no_training_scope_{key}_must_be_false")
    return {
        "schema_version": "gr00t_t8_2_seed_scan_manifest_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def scan_static_forbidden_scope(source_paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in source_paths:
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        hits = [token for token in STATIC_FORBIDDEN_TOKENS if token in text]
        rows.append(
            {
                "path": rel(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "forbidden_runtime_tokens": hits,
                "status": "PASS" if path.is_file() and not hits else "FAIL",
            }
        )
    return rows


def build_forbidden_scope_guard(*, command_templates: Sequence[str], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    command_rows = []
    for command in command_templates:
        present = [flag for flag in FORBIDDEN_ROUTE_FLAGS if flag in command]
        command_rows.append(
            {
                "command_template": command,
                "contains_timeout": "timeout" in command.split(),
                "contains_cuda_visible_devices_1": "CUDA_VISIBLE_DEVICES=1" in command,
                "forbidden_route_flags": present,
                "status": "PASS" if not present and command_has_timeout_and_gpu(command) else "FAIL",
            }
        )
    source_paths = [
        repo_root / "work/recap/safe_sft/t8_2_seedbank_postlift.py",
        repo_root / "work/recap/scripts/gr00t_t8_2_seedbank_postlift.py",
    ]
    static_rows = scan_static_forbidden_scope(source_paths)
    submodule_status = git_output(["status", "--short", "--", "submodules"], cwd=repo_root).splitlines()
    errors: list[str] = []
    if any(row["status"] != "PASS" for row in command_rows):
        errors.append("command_template_forbidden_or_missing_timeout_gpu_guard")
    if any(row["status"] != "PASS" for row in static_rows):
        errors.append("static_forbidden_runtime_token_detected")
    if submodule_status:
        errors.append("submodule_status_not_clean")
    return {
        "schema_version": "gr00t_t8_2_forbidden_scope_guard_v1",
        "status": "PASS" if not errors else "FAIL",
        "generated_at_utc": utc_now(),
        "errors": errors,
        "forbidden_route_flags": list(FORBIDDEN_ROUTE_FLAGS),
        "command_rows": command_rows,
        "static_rows": static_rows,
        "submodule_status_short": submodule_status,
        "no_training_attestation": {
            "training": False,
            "optimizer_step": False,
            "checkpoint_update": False,
            "lora_update": False,
            "lora_merge_before_eval": False,
            "guarded_recap": False,
            "recap": False,
            "fatg": False,
            "advantage_weighting": False,
            "per_edge_lora": False,
            "full_scope_finetune": False,
            "submodule_edits": bool(submodule_status),
        },
    }


def read_candidate_scout_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def normalize_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = {field: row.get(field, "") for field in CANDIDATE_SCOUT_FIELDS}
    valid = str(out.get("base_failure_mode") or "").lower() not in {"invalid", "runner_error", "schema_error"}
    if not out.get("stratum"):
        out["stratum"] = classify_base_stratum(
            base_success=out.get("base_success"),
            base_reached=out.get("base_reached"),
            base_lifted=out.get("base_lifted"),
            valid=valid,
        )
    return out


def select_seed_bank(candidate_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    normalized = [normalize_candidate_row(row) for row in candidate_rows]
    selected_by_stratum: dict[str, list[int]] = {stratum: [] for stratum in MATERIAL_STRATA}
    annotated: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in normalized:
        row = dict(row)
        stratum = str(row.get("stratum") or "")
        if stratum not in ALLOWED_STRATA:
            errors.append(f"invalid_stratum:{stratum}")
            stratum = "INVALID"
            row["stratum"] = stratum
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError):
            errors.append(f"invalid_seed:{row.get('seed')!r}")
            row["selected"] = False
            row["exclusion_reason"] = row.get("exclusion_reason") or "invalid_seed"
            annotated.append(row)
            continue
        if stratum == "INVALID":
            row["selected"] = False
            row["exclusion_reason"] = row.get("exclusion_reason") or "invalid_runner_or_schema"
        elif len(selected_by_stratum[stratum]) < MAX_SELECTED_PER_STRATUM:
            selected_by_stratum[stratum].append(seed)
            row["selected"] = True
            row["exclusion_reason"] = ""
        else:
            row["selected"] = False
            row["exclusion_reason"] = row.get("exclusion_reason") or f"stratum_cap_{MAX_SELECTED_PER_STRATUM}_reached"
        annotated.append(row)
    counts = {stratum: len(seeds) for stratum, seeds in selected_by_stratum.items()}
    deficits = {
        stratum: max(0, int(SEED_BANK_TARGETS[stratum]["min"]) - counts[stratum])
        for stratum in MATERIAL_STRATA
    }
    underfilled = [stratum for stratum, deficit in deficits.items() if deficit > 0]
    hard_gate_pass = counts["BASE_SUCCESS"] >= MIN_BASE_SUCCESS
    seed_bank = {
        "schema_version": "gr00t_t8_2_seed_bank_v1",
        "status": "PASS" if hard_gate_pass and not errors else "FAIL",
        "generated_at_utc": utc_now(),
        "selection_rule": "first_valid_seed_per_stratum_until_cap_15",
        "targets": SEED_BANK_TARGETS,
        "min_base_success_for_candidate_attribution": MIN_BASE_SUCCESS,
        "base_success_hard_gate_pass": hard_gate_pass,
        "base_success_hard_gate_failure_final_decision": "BASE_PROTOCOL_TOO_WEAK",
        "selected_seeds_by_stratum": selected_by_stratum,
        "counts_by_stratum": counts,
        "material_claims_forbidden_for_underfilled_strata": underfilled,
        "errors": errors,
    }
    deficit_payload = {
        "schema_version": "gr00t_t8_2_seed_bank_deficits_v1",
        "status": "PASS" if not underfilled and not errors else "FAIL",
        "counts_by_stratum": counts,
        "deficits_to_minimum_by_stratum": deficits,
        "underfilled_strata": underfilled,
        "base_success_selected": counts["BASE_SUCCESS"],
        "base_success_hard_gate_pass": hard_gate_pass,
        "final_decision_if_scan_window_exhausted_now": None if hard_gate_pass else "BASE_PROTOCOL_TOO_WEAK",
        "errors": errors,
    }
    return seed_bank, deficit_payload, annotated


def validate_final_decision_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    decision = payload.get("final_decision")
    errors: list[str] = []
    if decision not in ALLOWED_FINAL_DECISIONS:
        errors.append("final_decision_not_in_t8_2_allowed_enum")
    allowed = payload.get("allowed_final_decisions")
    if allowed is not None and list(allowed) != list(ALLOWED_FINAL_DECISIONS):
        errors.append("allowed_final_decisions_must_match_t8_2_user_enum_order")
    if payload.get("guarded_recap_allowed") is True or payload.get("fatg_allowed") is True:
        errors.append("guarded_recap_or_fatg_must_remain_forbidden")
    return {
        "schema_version": "gr00t_t8_2_final_decision_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "final_decision": decision,
        "allowed_final_decisions": list(ALLOWED_FINAL_DECISIONS),
        "errors": errors,
    }


def write_bootstrap_summary(out: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# GR00T T8.2 Bootstrap Readiness",
        "",
        f"- generated_at_utc: {utc_now()}",
        f"- evidence_lock: `{out / 'evidence_lock.json'}` ({payload['evidence_lock']['status']})",
        f"- forbidden_scope_guard: `{out / 'forbidden_scope_guard.json'}` ({payload['forbidden_scope_guard']['status']})",
        f"- seed_scan_manifest: `{out / 'seed_scan_manifest.json'}` ({payload['seed_scan_manifest_validation']['status']})",
        "- no expensive eval/training was run by this bootstrap command.",
        "",
        "## Seed scout handoff command",
        "",
        "```bash",
        str(payload["seed_scout_command_template"]),
        "```",
        "",
        "## Next artifact paths",
        "",
        "- `candidate_seed_scout.csv` — produced by base official/canonical seed scout.",
        "- `seed_bank.json` and `seed_bank_deficits.json` — produced by `select-seed-bank` after scout CSV exists.",
        "- `paired_eval_per_seed.jsonl` / `paired_eval_summary.csv` — produced by same-seed paired eval after BASE_SUCCESS hard gate passes.",
        "- `final_decision.json` — must use exactly one T8.2 allowed enum.",
        "",
        "## Forbidden branches",
        "",
        "Guarded RECAP, FATG, advantage weighting, per-edge LoRA, full-scope fine-tune, optimizer steps, checkpoint updates, LoRA merge, and submodule edits remain forbidden.",
    ]
    (out / "t8_2_bootstrap_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bootstrap(args: argparse.Namespace) -> int:
    out = resolve_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    evidence_root = resolve_path(args.evidence_root) if args.evidence_root else Path(os.environ.get("OMX_TEAM_LEADER_CWD", REPO_ROOT)).resolve()
    seed_scout_command = build_eval_command_template(
        output_dir_placeholder=str(out),
        seed_start=int(args.seed_start),
        max_candidate_seeds=int(args.max_candidate_seeds),
        max_episode_steps=int(args.max_episode_steps),
        max_wall_time=str(args.max_wall_time),
    )
    manifest = build_seed_scan_manifest(
        seed_start=int(args.seed_start),
        seed_sources=[],
        max_candidate_seeds=int(args.max_candidate_seeds),
        max_wall_time=str(args.max_wall_time),
        timeout_seconds=int(args.timeout_seconds),
        max_episode_steps=int(args.max_episode_steps),
        base_policy_path=str(args.base_policy_path),
        canonical_surface_path=str(args.canonical_surface_path),
        exact_command_template=seed_scout_command,
        scan_window_mini_adr=args.scan_window_mini_adr,
    )
    evidence = build_evidence_lock(evidence_root=evidence_root)
    manifest_validation = validate_seed_scan_manifest(manifest)
    command_manifest = {
        "schema_version": "gr00t_t8_2_command_manifest_v1",
        "generated_at_utc": utc_now(),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "git_status_short": git_output(["status", "--short"]).splitlines(),
        "submodule_status_short": git_output(["status", "--short", "--", "submodules"]).splitlines(),
        "model_env_command_policy": "All model/env commands must use timeout and CUDA_VISIBLE_DEVICES=1.",
        "seed_scout_command_template": seed_scout_command,
        "no_expensive_eval_ran_in_bootstrap": True,
    }
    forbidden = build_forbidden_scope_guard(command_templates=[seed_scout_command])
    empty_seed_bank, empty_deficits, _annotated = select_seed_bank([])
    payload = {
        "schema_version": "gr00t_t8_2_bootstrap_v1",
        "status": "PASS"
        if evidence["status"] == "PASS" and manifest_validation["status"] == "PASS" and forbidden["status"] == "PASS"
        else "FAIL",
        "evidence_lock": evidence,
        "forbidden_scope_guard": forbidden,
        "seed_scan_manifest_validation": manifest_validation,
        "seed_scout_command_template": seed_scout_command,
    }
    write_json(out / "evidence_lock.json", evidence)
    write_json(out / "seed_scan_manifest.json", manifest)
    write_json(out / "seed_scan_manifest.validation.json", manifest_validation)
    write_json(out / "command_manifest.json", command_manifest)
    write_json(out / "forbidden_scope_guard.json", forbidden)
    write_csv(out / "candidate_seed_scout.csv", [], fieldnames=CANDIDATE_SCOUT_FIELDS)
    write_json(out / "seed_bank.json", empty_seed_bank)
    write_json(out / "seed_bank_deficits.json", empty_deficits)
    write_json(out / "bootstrap_status.json", payload)
    write_bootstrap_summary(out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=json_default))
    return 0 if payload["status"] == "PASS" else 2


def run_select_seed_bank(args: argparse.Namespace) -> int:
    candidate_path = resolve_path(args.candidate_seed_scout)
    out = resolve_path(args.output_dir)
    rows = read_candidate_scout_csv(candidate_path)
    seed_bank, deficits, annotated = select_seed_bank(rows)
    write_json(out / "seed_bank.json", seed_bank)
    write_json(out / "seed_bank_deficits.json", deficits)
    write_csv(out / "candidate_seed_scout.annotated.csv", annotated, fieldnames=CANDIDATE_SCOUT_FIELDS)
    result = {"seed_bank": seed_bank, "seed_bank_deficits": deficits}
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
    return 0 if seed_bank["status"] == "PASS" else 2


def run_validate_final(args: argparse.Namespace) -> int:
    final_path = resolve_path(args.final_decision)
    payload = load_json(final_path)
    result = validate_final_decision_payload(payload)
    if args.output:
        write_json(resolve_path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
    return 0 if result["status"] == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GR00T T8.2 no-training seed-bank/post-lift bootstrap and schema validator",
    )
    sub = parser.add_subparsers(dest="command")
    boot = sub.add_parser("bootstrap", help="write evidence-lock, forbidden-scope, command, and seed-scan manifest artifacts")
    boot.add_argument("--output-dir", required=True)
    boot.add_argument("--evidence-root", default=os.environ.get("OMX_TEAM_LEADER_CWD", str(REPO_ROOT)))
    boot.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    boot.add_argument("--max-candidate-seeds", type=int, default=DEFAULT_MAX_CANDIDATE_SEEDS)
    boot.add_argument("--max-wall-time", default=DEFAULT_MAX_WALL_TIME)
    boot.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    boot.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    boot.add_argument("--base-policy-path", default="agent/artifacts/gr00t_recap_live/hf_patches/models--nvidia--GR00T-N1.6-G1-PnPAppleToPlate/snapshot-897d0313a190f46a2cccaeb34077752a0db4b0de/formalize_language=False")
    boot.add_argument("--canonical-surface-path", default="agent/artifacts/gr00t_a_gate_identity_closure/a_gate_identity_20260506_133517/canonical_identity")
    boot.add_argument("--scan-window-mini-adr", default=None)
    boot.set_defaults(func=run_bootstrap)

    select = sub.add_parser("select-seed-bank", help="validate candidate_seed_scout.csv and emit seed_bank artifacts")
    select.add_argument("--candidate-seed-scout", required=True)
    select.add_argument("--output-dir", required=True)
    select.set_defaults(func=run_select_seed_bank)

    final = sub.add_parser("validate-final", help="validate final_decision.json enum and forbidden-branch status")
    final.add_argument("--final-decision", required=True)
    final.add_argument("--output", default=None)
    final.set_defaults(func=run_validate_final)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    reject_forbidden_args(raw)
    parser = build_parser()
    args = parser.parse_args(raw)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
