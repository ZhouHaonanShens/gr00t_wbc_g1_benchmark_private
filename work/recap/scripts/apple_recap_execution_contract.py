from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


DEFAULT_OUTPUT_DIR = Path("agent/artifacts/apple_recap_exec")
EXECUTION_CONTRACT_JSON_NAME = "execution_freeze_contract_draft.json"
FINAL_EXECUTION_CONTRACT_JSON_NAME = "execution_freeze_contract.json"
REPO_SNAPSHOT_JSON_NAME = "repo_snapshot.json"
REPO_COMMIT_TXT_NAME = "repo_commit.txt"
REPO_STATUS_TXT_NAME = "repo_status.txt"

SCHEMA_VERSION = "apple_recap_execution_freeze_contract_draft_v1"
ARTIFACT_KIND = "apple_recap_execution_freeze_contract_draft"
FINAL_SCHEMA_VERSION = "apple_recap_execution_freeze_contract_v1"
FINAL_ARTIFACT_KIND = "apple_recap_execution_freeze_contract"
REPO_SNAPSHOT_SCHEMA_VERSION = "apple_recap_repo_snapshot_v1"
REPO_SNAPSHOT_ARTIFACT_KIND = "apple_recap_repo_snapshot"
UNSET_EXECUTION_SHA = "UNSET_UNTIL_T1B"
DEFAULT_FREEZE_CHECKPOINT_ID = "phase_a_tooling_frozen"

DEFAULT_EXECUTION_ROOT_SCOPE = "agent/artifacts/apple_recap_exec/"
SUCCESSOR_EXECUTION_ROOT_SCOPE = "agent/artifacts/apple_recap_exec_successor/"
CANONICAL_EXECUTION_ROOT_SCOPES: tuple[str, ...] = (
    DEFAULT_EXECUTION_ROOT_SCOPE,
    SUCCESSOR_EXECUTION_ROOT_SCOPE,
)
READ_ONLY_AUTHORITY_REF_SCOPES: tuple[str, ...] = (
    "agent/artifacts/",
    "agent/exchange/",
    ".sisyphus/evidence/",
)
DEFAULT_ALLOWED_POST_FREEZE_WRITE_SCOPES: tuple[str, ...] = (
    DEFAULT_EXECUTION_ROOT_SCOPE,
    "agent/runtime_logs/",
    ".sisyphus/evidence/",
    "agent/exchange/AppleToPlate_RECAP_final_report.md",
)
ALLOWED_POST_FREEZE_WRITE_SCOPES = DEFAULT_ALLOWED_POST_FREEZE_WRITE_SCOPES

FINAL_FREEZE_RUNNABLE_DIRTY_PATH_PREFIXES: tuple[str, ...] = (
    "work/",
    "tests/",
)

WORKTREE_MANIFEST_SCOPE: dict[str, object] = {
    "include": ["work/**"],
    "exclude": ["**/__pycache__/**", "**/*.pyc"],
}

CRITIC_BASELINE_AUTHORITY = "task7_real_critic_v2"
CRITIC_CANDIDATE_TRACK = "task7_real_critic_v3"

DEFAULT_HISTORICAL_REFERENCE_COMMITS: tuple[str, ...] = (
    "46b904de9a4ab9d723d0d3cfbec0ccb5635b5e1c",
    "ac765bd16dbc9428dafc2a9cbb8a7df47d4499eb",
)

DEFAULT_READ_ONLY_AUTHORITY_REF_SPECS: tuple[dict[str, str], ...] = (
    {
        "artifact_id": "public_anchor_formal",
        "authority_role": "official_public_anchor",
        "relative_path": "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/public_anchor/public_anchor_formal.json",
    },
    {
        "artifact_id": "same_checkpoint_triplet_eval",
        "authority_role": "diagnostic_triplet_eval",
        "relative_path": "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/same_checkpoint_triplet/same_checkpoint_triplet_eval.json",
    },
    {
        "artifact_id": "critic_score_rows_v1",
        "authority_role": "critic_held_out_score_rows",
        "relative_path": "agent/artifacts/vlm_critic_scorecard/score_rows_v1.csv",
    },
    {
        "artifact_id": "recap_authority_episodes",
        "authority_role": "mainline_reward_authority_dataset",
        "relative_path": "agent/artifacts/recap_datasets/recap_mainline_fresh_20260311_121500_k0/episodes.jsonl",
    },
    {
        "artifact_id": "gr00t_experiment_matrix",
        "authority_role": "experiment_matrix_backpointer",
        "relative_path": "agent/artifacts/gr00t_anchor_controller_recap/experiment_matrix/gr00t_experiment_matrix.json",
    },
    {
        "artifact_id": "reward_gate",
        "authority_role": "reward_publish_gate",
        "relative_path": "agent/artifacts/recap_temporal_critic_upgrade/reward_audit/reward_gate.json",
    },
    {
        "artifact_id": "critic_reward_audit_markdown",
        "authority_role": "single_file_audit_summary",
        "relative_path": "agent/exchange/AppleToPlate_RECAP_status_critic_reward_audit.md",
    },
)

THRESHOLD_POLICY: dict[str, object] = {
    "carrier_panel_size": 5,
    "carrier_panel_min_pass_count": 3,
    "carrier_normalized_delta_metric": "mean_abs_delta_over_contract_range",
    "carrier_normalized_delta_threshold": 0.05,
    "screen_positive_min_success_count": 6,
    "screen_negative_max_success_count": 3,
    "uplift_margin": 0.05,
    "b0_official_seed_bundle": "20000:20009",
    "b0_repro_seed_bundles": {
        "A": "20000:20009",
        "B": "21000:21009",
        "C": "22000:22009",
    },
    "extended_50ep_seed_bundle": "30000:30049",
}

POLICY_CONTRACT: dict[str, object] = {
    "freeze_phase": "phase_a_tooling_draft",
    "runnable_logic_authority_field": "execution_sha",
    "execution_sha_placeholder_allowed": True,
    "historical_reference_commits_are_read_only": True,
    "execution_sha_must_not_match_historical_reference_commits": True,
}

FINAL_POLICY_CONTRACT: dict[str, object] = {
    "freeze_phase": "phase_b_final_execution_freeze",
    "runnable_logic_authority_field": "execution_sha",
    "execution_sha_placeholder_allowed": False,
    "historical_reference_commits_are_read_only": True,
    "execution_sha_must_not_match_historical_reference_commits": True,
}

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apple_recap_execution_contract.py",
        description=(
            "Materialize the AppleToPlate execution freeze contract draft/final payloads "
            "and run fail-closed freeze integrity checks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("draft", "finalize", "check-integrity"),
        default="draft",
        help="draft: write execution_freeze_contract_draft.json; finalize: materialize final freeze artifacts; check-integrity: validate the final freeze against the live repo state.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives execution freeze artifacts.",
    )
    parser.add_argument(
        "--draft-contract-json",
        type=Path,
        default=None,
        help="Optional explicit draft contract JSON path for --mode finalize.",
    )
    parser.add_argument(
        "--final-contract-json",
        type=Path,
        default=None,
        help="Optional explicit final contract JSON path for --mode check-integrity.",
    )
    parser.add_argument(
        "--freeze-timestamp",
        type=str,
        default=None,
        help="Optional explicit freeze timestamp (ISO-8601) for --mode finalize.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _validate_output_dir(path: Path) -> Path:
    return state_conditioned_bucket_a_import.validate_output_dir(path)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_payload(payload: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_timestamp(value: str, *, field_path: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_path} must be a non-empty ISO-8601 timestamp")
    try:
        datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_path} must be an ISO-8601 timestamp") from exc
    return normalized


def core_digest(core_payload: Mapping[str, Any]) -> str:
    return _sha256_payload(dict(core_payload))


def _resolve_path(repo_root: Path, raw: Path | str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _repo_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _noncanonical_root_marker(path: str) -> str | None:
    for part in PurePosixPath(str(path).strip()).parts:
        normalized_part = str(part).strip().lower()
        if not normalized_part:
            continue
        normalized_tokens = [
            token for token in normalized_part.replace("-", "_").split("_") if token
        ]
        if normalized_part in {"reference-only", "reference_only"} or (
            "reference" in normalized_tokens and "only" in normalized_tokens
        ):
            return "reference-only"
        for marker in ("alias", "current", "diagnostic", "smoke"):
            if marker in normalized_tokens:
                return marker
    return None


def resolve_repo_contained_path(
    repo_root: Path,
    raw: Path | str,
    *,
    field_name: str,
    canonical_roots: Sequence[str] = (),
    reject_noncanonical_parts: bool = False,
) -> Path:
    resolved_path = _resolve_path(repo_root, raw)
    resolved_repo_root = repo_root.resolve()
    try:
        relative_path = str(resolved_path.relative_to(resolved_repo_root))
    except ValueError as exc:
        raise ValueError(
            f"noncanonical_root_contamination: {field_name} resolves outside repo root: {resolved_path}"
        ) from exc
    if reject_noncanonical_parts:
        marker = _noncanonical_root_marker(relative_path)
        if marker is not None:
            raise ValueError(
                f"noncanonical_root_contamination: {field_name} points at non-authoritative {marker} lane: {relative_path}"
            )
    normalized_roots = tuple(
        str(root).strip() for root in canonical_roots if str(root).strip()
    )
    if normalized_roots and not any(
        _path_within_scope(relative_path, root) for root in normalized_roots
    ):
        allowed_roots = ", ".join(repr(root) for root in normalized_roots)
        raise ValueError(
            "noncanonical_root_contamination: "
            + f"{field_name} must stay under canonical authoritative roots [{allowed_roots}], got {relative_path!r}"
        )
    return resolved_path


def _path_within_scope(path: str, scope: str) -> bool:
    normalized_path = str(path).strip()
    normalized_scope = str(scope).strip()
    if not normalized_scope:
        return False
    if normalized_scope.endswith("/"):
        return normalized_path == normalized_scope[:-1] or normalized_path.startswith(
            normalized_scope
        )
    return normalized_path == normalized_scope


def _is_allowed_post_freeze_path(path: str) -> bool:
    return any(
        _path_within_scope(path, scope)
        for scope in DEFAULT_ALLOWED_POST_FREEZE_WRITE_SCOPES
    )


def _normalize_execution_root_scope(*, repo_root: Path, output_dir: Path) -> str:
    relative_path = _repo_relative_path(repo_root, output_dir.resolve()).strip()
    if not relative_path:
        raise ValueError("execution root scope must be a non-empty repo-relative path")
    return relative_path if relative_path.endswith("/") else f"{relative_path}/"


def allowed_post_freeze_write_scopes_for_output_dir(
    *,
    repo_root: Path,
    output_dir: Path,
) -> tuple[str, ...]:
    return (
        _normalize_execution_root_scope(repo_root=repo_root, output_dir=output_dir),
        "agent/runtime_logs/",
        ".sisyphus/evidence/",
        "agent/exchange/AppleToPlate_RECAP_final_report.md",
    )


def _allowed_post_freeze_write_scopes_from_snapshot_path(
    *,
    repo_root: Path,
    repo_snapshot_relative_path: str,
) -> tuple[str, ...]:
    snapshot_path = resolve_repo_contained_path(
        repo_root,
        repo_snapshot_relative_path,
        field_name="freeze_integrity.repo_snapshot_relative_path",
        canonical_roots=CANONICAL_EXECUTION_ROOT_SCOPES,
        reject_noncanonical_parts=True,
    )
    return allowed_post_freeze_write_scopes_for_output_dir(
        repo_root=repo_root,
        output_dir=snapshot_path.parent,
    )


def _is_allowed_post_freeze_path_for_scopes(
    path: str,
    scopes: Sequence[str],
) -> bool:
    return any(_path_within_scope(path, scope) for scope in scopes)


def _is_runnable_dirty_path_for_final_freeze(path: str) -> bool:
    normalized_path = str(path).strip()
    return any(
        normalized_path.startswith(prefix)
        for prefix in FINAL_FREEZE_RUNNABLE_DIRTY_PATH_PREFIXES
    )


def _git_text(
    repo_root: Path,
    *args: str,
    allow_failure: bool = False,
    default: str = "",
) -> str:
    env = dict(os.environ)
    env["GIT_MASTER"] = "1"
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        if allow_failure:
            return str(default)
        message = proc.stderr.strip() or proc.stdout.strip() or "git command failed"
        raise ValueError(f"git {' '.join(args)} failed: {message}")
    return proc.stdout


def _status_entries_from_porcelain(status_text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for raw_line in status_text.splitlines():
        if not raw_line or raw_line.startswith("## "):
            continue
        status_code = raw_line[:2]
        raw_path = raw_line[3:].strip()
        path = raw_path.split(" -> ")[-1].strip()
        entries.append(
            {
                "status_code": status_code,
                "path": path,
                "raw_line": raw_line,
            }
        )
    return entries


def _capture_git_status(repo_root: Path) -> dict[str, object]:
    human_text = _git_text(repo_root, "status", "--short", "--branch")
    porcelain_text = _git_text(
        repo_root, "status", "--porcelain=v1", "--branch", "-uall"
    )
    branch_summary = next(
        (
            line[3:].strip()
            for line in porcelain_text.splitlines()
            if line.startswith("## ")
        ),
        "",
    )
    return {
        "human_text": human_text,
        "porcelain_text": porcelain_text,
        "branch_summary": branch_summary,
        "entries": _status_entries_from_porcelain(porcelain_text),
    }


def _file_entry(repo_root: Path, relative_path: str) -> dict[str, object]:
    resolved_path = _resolve_path(repo_root, relative_path)
    entry: dict[str, object] = {
        "relative_path": str(relative_path),
        "resolved_path": str(resolved_path),
        "exists": resolved_path.exists(),
        "path_kind": "missing",
        "content_sha256": None,
        "size_bytes": None,
    }
    if resolved_path.is_file():
        entry["path_kind"] = "file"
        entry["content_sha256"] = _sha256_file(resolved_path)
        entry["size_bytes"] = int(resolved_path.stat().st_size)
    elif resolved_path.is_dir():
        entry["path_kind"] = "directory"
    return entry


def _iter_worktree_logic_files(repo_root: Path) -> list[Path]:
    work_root = (repo_root / "work").resolve()
    if not work_root.exists():
        return []
    paths: list[Path] = []
    for path in work_root.rglob("*"):
        if not path.is_file():
            continue
        relative = _repo_relative_path(repo_root, path)
        if "/__pycache__/" in f"/{relative}/" or relative.endswith(".pyc"):
            continue
        paths.append(path.resolve())
    return sorted(paths)


def build_worktree_manifest(repo_root: Path) -> dict[str, object]:
    files = [
        {
            "relative_path": _repo_relative_path(repo_root, path),
            "resolved_path": str(path),
            "size_bytes": int(path.stat().st_size),
            "content_sha256": _sha256_file(path),
        }
        for path in _iter_worktree_logic_files(repo_root)
    ]
    manifest: dict[str, object] = {
        "scope": json.loads(json.dumps(WORKTREE_MANIFEST_SCOPE, ensure_ascii=True)),
        "file_count": len(files),
        "files": files,
    }
    manifest["toolchain_manifest_hash"] = _sha256_payload(manifest)
    return manifest


def _capture_frozen_overrides(
    repo_root: Path,
    status_entries: Sequence[Mapping[str, str]],
    *,
    allowed_write_scopes: Sequence[str],
) -> list[dict[str, object]]:
    overrides: list[dict[str, object]] = []
    for entry in status_entries:
        relative_path = str(entry.get("path", "")).strip()
        if not relative_path or _is_allowed_post_freeze_path_for_scopes(
            relative_path,
            allowed_write_scopes,
        ):
            continue
        captured = _file_entry(repo_root, relative_path)
        captured["status_code"] = str(entry.get("status_code", ""))
        captured["raw_line"] = str(entry.get("raw_line", ""))
        overrides.append(captured)
    return overrides


def _blocked_final_freeze_dirty_entries(
    status_entries: Sequence[Mapping[str, str]],
    *,
    allowed_write_scopes: Sequence[str],
) -> list[dict[str, str]]:
    blocked: list[dict[str, str]] = []
    for entry in status_entries:
        relative_path = str(entry.get("path", "")).strip()
        if not relative_path or _is_allowed_post_freeze_path_for_scopes(
            relative_path,
            allowed_write_scopes,
        ):
            continue
        if not _is_runnable_dirty_path_for_final_freeze(relative_path):
            continue
        blocked.append(
            {
                "path": relative_path,
                "status_code": str(entry.get("status_code", "")),
                "raw_line": str(entry.get("raw_line", "")),
            }
        )
    return blocked


def build_repo_snapshot(
    *,
    repo_root: Path,
    output_dir: Path,
    execution_sha: str,
    generated_at: str,
) -> tuple[dict[str, Any], str, str]:
    branch = _git_text(repo_root, "branch", "--show-current").strip()
    upstream = (
        _git_text(
            repo_root,
            "rev-parse",
            "--abbrev-ref",
            "@{upstream}",
            allow_failure=True,
            default="NO_UPSTREAM",
        ).strip()
        or "NO_UPSTREAM"
    )
    status_capture = _capture_git_status(repo_root)
    commit_text = f"{execution_sha}\n"
    status_text = str(status_capture["human_text"])
    worktree_manifest = build_worktree_manifest(repo_root)
    allowed_write_scopes = allowed_post_freeze_write_scopes_for_output_dir(
        repo_root=repo_root,
        output_dir=output_dir,
    )
    frozen_overrides = _capture_frozen_overrides(
        repo_root,
        cast(Sequence[Mapping[str, str]], status_capture["entries"]),
        allowed_write_scopes=allowed_write_scopes,
    )
    snapshot: dict[str, Any] = {
        "schema_version": REPO_SNAPSHOT_SCHEMA_VERSION,
        "artifact_kind": REPO_SNAPSHOT_ARTIFACT_KIND,
        "generated_at": generated_at,
        "repo_root": str(repo_root.resolve()),
        "execution_sha": str(execution_sha),
        "git_branch": branch,
        "git_upstream": upstream,
        "repo_commit_relative_path": _repo_relative_path(
            repo_root,
            output_dir / REPO_COMMIT_TXT_NAME,
        ),
        "repo_status_relative_path": _repo_relative_path(
            repo_root,
            output_dir / REPO_STATUS_TXT_NAME,
        ),
        "git_status_branch_summary": str(status_capture["branch_summary"]),
        "git_status_entries": json.loads(
            json.dumps(status_capture["entries"], ensure_ascii=True)
        ),
        "frozen_worktree_overrides": frozen_overrides,
        "worktree_manifest": worktree_manifest,
        "toolchain_manifest_hash": str(worktree_manifest["toolchain_manifest_hash"]),
        "allowed_post_freeze_write_scopes": list(allowed_write_scopes),
        "working_tree_clean_except_allowed": len(frozen_overrides) == 0,
    }
    snapshot["report_signature_sha256"] = _sha256_payload(
        {
            key: value
            for key, value in snapshot.items()
            if key != "report_signature_sha256"
        }
    )
    return snapshot, commit_text, status_text


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def build_read_only_authority_ref(
    *,
    repo_root: Path,
    artifact_id: str,
    authority_role: str,
    relative_path: Path | str,
    reject_noncanonical_parts: bool = False,
) -> dict[str, object]:
    resolved_path = resolve_repo_contained_path(
        repo_root,
        relative_path,
        field_name=f"read_only_authority_ref[{artifact_id}].relative_path",
        canonical_roots=READ_ONLY_AUTHORITY_REF_SCOPES,
        reject_noncanonical_parts=reject_noncanonical_parts,
    )
    if not resolved_path.is_file():
        raise ValueError(f"read-only authority ref does not exist: {resolved_path}")
    payload: dict[str, Any] | None = None
    if resolved_path.suffix.lower() == ".json":
        payload = _read_json_object(resolved_path)
    ref: dict[str, object] = {
        "artifact_id": str(artifact_id),
        "authority_role": str(authority_role),
        "relative_path": _repo_relative_path(repo_root, resolved_path),
        "resolved_path": str(resolved_path),
        "path_kind": "file",
        "must_exist": True,
        "read_only": True,
        "content_sha256": _sha256_file(resolved_path),
    }
    if payload is not None:
        ref["artifact_kind"] = payload.get("artifact_kind")
        ref["schema_version"] = payload.get("schema_version")
        ref["report_signature_sha256"] = payload.get("report_signature_sha256")
    return ref


def build_execution_freeze_contract_draft(
    *,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
    execution_sha: str = UNSET_EXECUTION_SHA,
    historical_reference_commits: Sequence[str] = DEFAULT_HISTORICAL_REFERENCE_COMMITS,
    read_only_authority_ref_specs: Sequence[
        Mapping[str, str]
    ] = DEFAULT_READ_ONLY_AUTHORITY_REF_SPECS,
) -> dict[str, Any]:
    read_only_authority_refs = [
        build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id=str(spec["artifact_id"]),
            authority_role=str(spec["authority_role"]),
            relative_path=str(spec["relative_path"]),
            reject_noncanonical_parts=True,
        )
        for spec in read_only_authority_ref_specs
    ]
    core = {"commit": str(execution_sha)}
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "generated_at": generated_at or _now_iso(),
        "execution_sha": str(execution_sha),
        "historical_reference_commits": [
            str(value) for value in historical_reference_commits
        ],
        "read_only_authority_refs": read_only_authority_refs,
        "critic_baseline_authority": CRITIC_BASELINE_AUTHORITY,
        "critic_candidate_track": CRITIC_CANDIDATE_TRACK,
        "policy": dict(POLICY_CONTRACT),
        "threshold_policy": json.loads(
            json.dumps(THRESHOLD_POLICY, ensure_ascii=True, sort_keys=True)
        ),
        "core": core,
        "core_digest": core_digest(core),
    }
    payload["report_signature_sha256"] = _signature_for_contract(payload)
    return payload


def _signature_for_contract(payload: Mapping[str, Any]) -> str:
    signature_basis = {
        str(key): value
        for key, value in dict(payload).items()
        if key != "report_signature_sha256"
    }
    return _sha256_payload(signature_basis)


def _issue(code: str, field_path: str, message: str) -> dict[str, str]:
    return {
        "code": str(code),
        "field_path": str(field_path),
        "message": str(message),
    }


def _validate_non_empty_string(
    value: object,
    *,
    field_path: str,
    issues: list[dict[str, str]],
) -> str | None:
    if not isinstance(value, str):
        issues.append(
            _issue(
                "wrong_type",
                field_path,
                f"{field_path} must be a string, got {type(value).__name__}",
            )
        )
        return None
    normalized = value.strip()
    if not normalized:
        issues.append(
            _issue("empty_string", field_path, f"{field_path} must be non-empty")
        )
        return None
    return normalized


def _validate_bool(
    value: object,
    *,
    field_path: str,
    issues: list[dict[str, str]],
) -> bool | None:
    if not isinstance(value, bool):
        issues.append(
            _issue(
                "wrong_type",
                field_path,
                f"{field_path} must be a bool, got {type(value).__name__}",
            )
        )
        return None
    return bool(value)


def _validate_int(
    value: object,
    *,
    field_path: str,
    issues: list[dict[str, str]],
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        issues.append(
            _issue(
                "wrong_type",
                field_path,
                f"{field_path} must be an int, got {type(value).__name__}",
            )
        )
        return None
    return int(value)


def _validate_number(
    value: object,
    *,
    field_path: str,
    issues: list[dict[str, str]],
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(
            _issue(
                "wrong_type",
                field_path,
                f"{field_path} must be a number, got {type(value).__name__}",
            )
        )
        return None
    return float(value)


def _validate_sha256_string(
    value: object,
    *,
    field_path: str,
    issues: list[dict[str, str]],
) -> str | None:
    normalized = _validate_non_empty_string(value, field_path=field_path, issues=issues)
    if normalized is None:
        return None
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        issues.append(
            _issue(
                "invalid_sha256",
                field_path,
                f"{field_path} must be a lowercase sha256 hex digest",
            )
        )
        return None
    return normalized


def _validate_mapping(
    value: object,
    *,
    field_path: str,
    issues: list[dict[str, str]],
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        issues.append(
            _issue(
                "wrong_type",
                field_path,
                f"{field_path} must be an object, got {type(value).__name__}",
            )
        )
        return None
    return value


def _validate_historical_reference_commits(
    value: object,
    *,
    issues: list[dict[str, str]],
) -> list[str]:
    field_path = "historical_reference_commits"
    if not isinstance(value, list):
        issues.append(
            _issue(
                "wrong_type",
                field_path,
                f"{field_path} must be a list, got {type(value).__name__}",
            )
        )
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        commit = _validate_non_empty_string(
            item,
            field_path=f"{field_path}[{index}]",
            issues=issues,
        )
        if commit is None:
            continue
        if commit in seen:
            issues.append(
                _issue(
                    "duplicate_historical_reference_commit",
                    f"{field_path}[{index}]",
                    f"duplicate historical reference commit: {commit}",
                )
            )
            continue
        seen.add(commit)
        normalized.append(commit)
    if not normalized:
        issues.append(
            _issue(
                "missing_required_field",
                field_path,
                "historical_reference_commits must contain at least one commit",
            )
        )
    return normalized


def _validate_read_only_authority_refs(
    value: object,
    *,
    repo_root: Path,
    issues: list[dict[str, str]],
) -> list[dict[str, object]]:
    field_path = "read_only_authority_refs"
    if not isinstance(value, list):
        issues.append(
            _issue(
                "wrong_type",
                field_path,
                f"{field_path} must be a list, got {type(value).__name__}",
            )
        )
        return []
    normalized: list[dict[str, object]] = []
    for index, raw_ref in enumerate(value):
        ref_path = f"{field_path}[{index}]"
        ref = _validate_mapping(raw_ref, field_path=ref_path, issues=issues)
        if ref is None:
            continue
        artifact_id = _validate_non_empty_string(
            ref.get("artifact_id"),
            field_path=f"{ref_path}.artifact_id",
            issues=issues,
        )
        authority_role = _validate_non_empty_string(
            ref.get("authority_role"),
            field_path=f"{ref_path}.authority_role",
            issues=issues,
        )
        relative_path = _validate_non_empty_string(
            ref.get("relative_path"),
            field_path=f"{ref_path}.relative_path",
            issues=issues,
        )
        resolved_path_text = _validate_non_empty_string(
            ref.get("resolved_path"),
            field_path=f"{ref_path}.resolved_path",
            issues=issues,
        )
        path_kind = _validate_non_empty_string(
            ref.get("path_kind"),
            field_path=f"{ref_path}.path_kind",
            issues=issues,
        )
        must_exist = _validate_bool(
            ref.get("must_exist"),
            field_path=f"{ref_path}.must_exist",
            issues=issues,
        )
        read_only = _validate_bool(
            ref.get("read_only"),
            field_path=f"{ref_path}.read_only",
            issues=issues,
        )
        content_sha256 = _validate_sha256_string(
            ref.get("content_sha256"),
            field_path=f"{ref_path}.content_sha256",
            issues=issues,
        )

        normalized_ref: dict[str, object] = {}
        if artifact_id is not None:
            normalized_ref["artifact_id"] = artifact_id
        if authority_role is not None:
            normalized_ref["authority_role"] = authority_role
        if relative_path is not None:
            normalized_ref["relative_path"] = relative_path
        if resolved_path_text is not None:
            normalized_ref["resolved_path"] = str(
                Path(resolved_path_text).expanduser().resolve()
            )
        if path_kind is not None:
            normalized_ref["path_kind"] = path_kind
        if must_exist is not None:
            normalized_ref["must_exist"] = must_exist
        if read_only is not None:
            normalized_ref["read_only"] = read_only
        if content_sha256 is not None:
            normalized_ref["content_sha256"] = content_sha256

        for optional_field in (
            "artifact_kind",
            "schema_version",
            "report_signature_sha256",
        ):
            if optional_field in ref:
                normalized_ref[optional_field] = ref.get(optional_field)

        if path_kind not in (None, "file"):
            issues.append(
                _issue(
                    "invalid_authority_ref_kind",
                    f"{ref_path}.path_kind",
                    "read-only authority refs must point to files",
                )
            )
        if must_exist is False:
            issues.append(
                _issue(
                    "invalid_authority_ref_policy",
                    f"{ref_path}.must_exist",
                    "read-only authority refs must be must_exist=true",
                )
            )
        if read_only is False:
            issues.append(
                _issue(
                    "invalid_authority_ref_policy",
                    f"{ref_path}.read_only",
                    "read-only authority refs must be read_only=true",
                )
            )

        resolved_relative_path: Path | None = None
        if relative_path is not None:
            try:
                resolved_relative_path = resolve_repo_contained_path(
                    repo_root,
                    relative_path,
                    field_name=f"{ref_path}.relative_path",
                    canonical_roots=READ_ONLY_AUTHORITY_REF_SCOPES,
                    reject_noncanonical_parts=True,
                )
            except ValueError as exc:
                issues.append(
                    _issue(
                        "noncanonical_root_contamination",
                        f"{ref_path}.relative_path",
                        str(exc),
                    )
                )

        resolved_path: Path | None = None
        if resolved_path_text is not None:
            try:
                resolved_path = resolve_repo_contained_path(
                    repo_root,
                    resolved_path_text,
                    field_name=f"{ref_path}.resolved_path",
                    canonical_roots=READ_ONLY_AUTHORITY_REF_SCOPES,
                    reject_noncanonical_parts=True,
                )
            except ValueError as exc:
                issues.append(
                    _issue(
                        "noncanonical_root_contamination",
                        f"{ref_path}.resolved_path",
                        str(exc),
                    )
                )
                resolved_path = None
        if resolved_relative_path is not None and resolved_path is not None:
            if resolved_relative_path != resolved_path:
                issues.append(
                    _issue(
                        "authority_ref_path_mismatch",
                        f"{ref_path}.relative_path",
                        "declared relative_path does not resolve to resolved_path",
                    )
                )
        if resolved_path is not None:
            if must_exist and not resolved_path.is_file():
                issues.append(
                    _issue(
                        "missing_authority_ref",
                        f"{ref_path}.resolved_path",
                        f"read-only authority ref does not exist: {resolved_path}",
                    )
                )
            elif resolved_path.is_file():
                declared_relative = relative_path
                expected_relative = _repo_relative_path(repo_root, resolved_path)
                if (
                    declared_relative is not None
                    and declared_relative != expected_relative
                ):
                    issues.append(
                        _issue(
                            "authority_ref_path_mismatch",
                            f"{ref_path}.relative_path",
                            "declared relative_path does not match resolved_path",
                        )
                    )
                actual_sha256 = _sha256_file(resolved_path)
                if content_sha256 is not None and actual_sha256 != content_sha256:
                    issues.append(
                        _issue(
                            "authority_ref_digest_mismatch",
                            f"{ref_path}.content_sha256",
                            "declared content_sha256 does not match the live file bytes",
                        )
                    )
                if resolved_path.suffix.lower() == ".json":
                    payload = _read_json_object(resolved_path)
                    if payload is None:
                        issues.append(
                            _issue(
                                "invalid_authority_ref_json",
                                f"{ref_path}.resolved_path",
                                f"authority ref JSON is unreadable: {resolved_path}",
                            )
                        )
                    else:
                        for optional_field in ("artifact_kind", "schema_version"):
                            declared = normalized_ref.get(optional_field)
                            observed = payload.get(optional_field)
                            if declared != observed:
                                issues.append(
                                    _issue(
                                        "authority_ref_metadata_mismatch",
                                        f"{ref_path}.{optional_field}",
                                        f"declared {optional_field} does not match the referenced JSON",
                                    )
                                )
                        declared_signature = normalized_ref.get(
                            "report_signature_sha256"
                        )
                        observed_signature = payload.get("report_signature_sha256")
                        if declared_signature != observed_signature:
                            issues.append(
                                _issue(
                                    "authority_ref_metadata_mismatch",
                                    f"{ref_path}.report_signature_sha256",
                                    "declared report_signature_sha256 does not match the referenced JSON",
                                )
                            )
        normalized.append(normalized_ref)

    if not normalized:
        issues.append(
            _issue(
                "missing_required_field",
                field_path,
                "read_only_authority_refs must contain at least one reference",
            )
        )
    return normalized


def _validate_exact_mapping(
    value: object,
    *,
    field_path: str,
    expected: Mapping[str, object],
    issues: list[dict[str, str]],
) -> dict[str, object]:
    payload = _validate_mapping(value, field_path=field_path, issues=issues)
    if payload is None:
        return {}
    normalized: dict[str, object] = {}
    for key, expected_value in expected.items():
        nested_path = f"{field_path}.{key}"
        if key not in payload:
            issues.append(
                _issue(
                    "missing_required_field",
                    nested_path,
                    f"missing required field: {nested_path}",
                )
            )
            continue
        observed_value = payload.get(key)
        if isinstance(expected_value, bool):
            normalized_value = _validate_bool(
                observed_value,
                field_path=nested_path,
                issues=issues,
            )
        elif isinstance(expected_value, int):
            normalized_value = _validate_int(
                observed_value,
                field_path=nested_path,
                issues=issues,
            )
        elif isinstance(expected_value, float):
            normalized_value = _validate_number(
                observed_value,
                field_path=nested_path,
                issues=issues,
            )
        elif isinstance(expected_value, str):
            normalized_value = _validate_non_empty_string(
                observed_value,
                field_path=nested_path,
                issues=issues,
            )
        elif isinstance(expected_value, Mapping):
            normalized_value = _validate_exact_mapping(
                observed_value,
                field_path=nested_path,
                expected=expected_value,
                issues=issues,
            )
        else:
            normalized_value = observed_value
        if normalized_value is None:
            continue
        if normalized_value != expected_value:
            issues.append(
                _issue(
                    "invalid_value",
                    nested_path,
                    f"{nested_path} must equal {expected_value!r}",
                )
            )
            continue
        normalized[key] = normalized_value
    return normalized


def validate_execution_freeze_contract_draft(
    payload: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    normalized: dict[str, Any] = {}

    schema_version = _validate_non_empty_string(
        payload.get("schema_version"),
        field_path="schema_version",
        issues=issues,
    )
    if schema_version is not None:
        if schema_version != SCHEMA_VERSION:
            issues.append(
                _issue(
                    "invalid_schema_version",
                    "schema_version",
                    f"schema_version must equal {SCHEMA_VERSION!r}",
                )
            )
        normalized["schema_version"] = schema_version

    artifact_kind = _validate_non_empty_string(
        payload.get("artifact_kind"),
        field_path="artifact_kind",
        issues=issues,
    )
    if artifact_kind is not None:
        if artifact_kind != ARTIFACT_KIND:
            issues.append(
                _issue(
                    "invalid_artifact_kind",
                    "artifact_kind",
                    f"artifact_kind must equal {ARTIFACT_KIND!r}",
                )
            )
        normalized["artifact_kind"] = artifact_kind

    generated_at = _validate_non_empty_string(
        payload.get("generated_at"),
        field_path="generated_at",
        issues=issues,
    )
    if generated_at is not None:
        normalized["generated_at"] = generated_at

    execution_sha = _validate_non_empty_string(
        payload.get("execution_sha"),
        field_path="execution_sha",
        issues=issues,
    )
    if execution_sha is not None:
        normalized["execution_sha"] = execution_sha

    historical_reference_commits = _validate_historical_reference_commits(
        payload.get("historical_reference_commits"),
        issues=issues,
    )
    normalized["historical_reference_commits"] = historical_reference_commits
    if execution_sha is not None and execution_sha in set(historical_reference_commits):
        issues.append(
            _issue(
                "historical_execution_authority_conflict",
                "execution_sha",
                "execution_sha must not equal any historical_reference_commits entry",
            )
        )

    read_only_authority_refs = _validate_read_only_authority_refs(
        payload.get("read_only_authority_refs"),
        repo_root=repo_root,
        issues=issues,
    )
    normalized["read_only_authority_refs"] = read_only_authority_refs

    critic_baseline_authority = _validate_non_empty_string(
        payload.get("critic_baseline_authority"),
        field_path="critic_baseline_authority",
        issues=issues,
    )
    if critic_baseline_authority is not None:
        normalized["critic_baseline_authority"] = critic_baseline_authority

    critic_candidate_track = _validate_non_empty_string(
        payload.get("critic_candidate_track"),
        field_path="critic_candidate_track",
        issues=issues,
    )
    if critic_candidate_track is not None:
        normalized["critic_candidate_track"] = critic_candidate_track

    normalized["policy"] = _validate_exact_mapping(
        payload.get("policy"),
        field_path="policy",
        expected=POLICY_CONTRACT,
        issues=issues,
    )
    normalized["threshold_policy"] = _validate_exact_mapping(
        payload.get("threshold_policy"),
        field_path="threshold_policy",
        expected=THRESHOLD_POLICY,
        issues=issues,
    )

    core_mapping = _validate_mapping(
        payload.get("core"), field_path="core", issues=issues
    )
    normalized_core: dict[str, Any] = {}
    if core_mapping is not None:
        core_commit = _validate_non_empty_string(
            core_mapping.get("commit"),
            field_path="core.commit",
            issues=issues,
        )
        if core_commit is not None:
            normalized_core["commit"] = core_commit
            if execution_sha is not None and core_commit != execution_sha:
                issues.append(
                    _issue(
                        "execution_sha_core_commit_mismatch",
                        "core.commit",
                        "core.commit must equal execution_sha",
                    )
                )
    normalized["core"] = normalized_core

    computed_core_digest = core_digest(normalized_core)
    normalized["core_digest"] = computed_core_digest
    declared_core_digest = payload.get("core_digest")
    if (
        declared_core_digest is not None
        and declared_core_digest != computed_core_digest
    ):
        issues.append(
            _issue(
                "core_digest_mismatch",
                "core_digest",
                "declared core_digest does not match normalized core payload",
            )
        )

    computed_signature = _signature_for_contract(normalized)
    normalized["report_signature_sha256"] = computed_signature
    declared_signature = payload.get("report_signature_sha256")
    if declared_signature is not None and declared_signature != computed_signature:
        issues.append(
            _issue(
                "report_signature_sha256_mismatch",
                "report_signature_sha256",
                "declared report_signature_sha256 does not match normalized payload",
            )
        )

    return {
        "formal_eligibility": "ALLOW" if not issues else "BLOCK",
        "issues": issues,
        "normalized_contract": normalized,
        "core_digest": computed_core_digest,
    }


def build_execution_freeze_contract_final(
    *,
    draft_payload: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
    execution_sha: str,
    generated_at: str,
    toolchain_manifest_hash: str,
) -> dict[str, Any]:
    draft_validation = validate_execution_freeze_contract_draft(
        draft_payload,
        repo_root=repo_root,
    )
    if draft_validation["formal_eligibility"] != "ALLOW":
        first_issue = (
            draft_validation["issues"][0] if draft_validation["issues"] else None
        )
        if isinstance(first_issue, Mapping):
            message = str(
                first_issue.get("message", "draft contract validation failed")
            )
        else:
            message = "draft contract validation failed"
        raise ValueError(message)
    normalized_draft = cast(Mapping[str, Any], draft_validation["normalized_contract"])
    freeze_context = {
        "execution_sha": str(execution_sha),
        "manifest_hash": str(toolchain_manifest_hash),
        "checkpoint_id": DEFAULT_FREEZE_CHECKPOINT_ID,
        "seed_bundle_id": str(THRESHOLD_POLICY["b0_official_seed_bundle"]),
        "timestamp": _parse_timestamp(generated_at, field_path="freshness.timestamp"),
    }
    core = {"commit": str(execution_sha)}
    allowed_write_scopes = allowed_post_freeze_write_scopes_for_output_dir(
        repo_root=repo_root,
        output_dir=output_dir,
    )
    payload: dict[str, Any] = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "artifact_kind": FINAL_ARTIFACT_KIND,
        "generated_at": generated_at,
        "execution_sha": str(execution_sha),
        "historical_reference_commits": list(
            normalized_draft["historical_reference_commits"]
        ),
        "read_only_authority_refs": json.loads(
            json.dumps(
                normalized_draft["read_only_authority_refs"],
                ensure_ascii=True,
                sort_keys=True,
            )
        ),
        "critic_baseline_authority": str(normalized_draft["critic_baseline_authority"]),
        "critic_candidate_track": str(normalized_draft["critic_candidate_track"]),
        "policy": {
            **FINAL_POLICY_CONTRACT,
            "allowed_post_freeze_write_scopes": list(allowed_write_scopes),
        },
        "threshold_policy": json.loads(
            json.dumps(
                normalized_draft["threshold_policy"],
                ensure_ascii=True,
                sort_keys=True,
            )
        ),
        "freshness": freeze_context,
        "freeze_integrity": {
            "mode": "fail_closed",
            "toolchain_manifest_hash": str(toolchain_manifest_hash),
            "repo_snapshot_relative_path": _repo_relative_path(
                repo_root,
                output_dir / REPO_SNAPSHOT_JSON_NAME,
            ),
            "repo_commit_relative_path": _repo_relative_path(
                repo_root,
                output_dir / REPO_COMMIT_TXT_NAME,
            ),
            "repo_status_relative_path": _repo_relative_path(
                repo_root,
                output_dir / REPO_STATUS_TXT_NAME,
            ),
            "allowed_post_freeze_write_scopes": list(allowed_write_scopes),
            "toolchain_scope": json.loads(
                json.dumps(WORKTREE_MANIFEST_SCOPE, ensure_ascii=True, sort_keys=True)
            ),
            "on_drift": "block_and_reopen_execution_sha",
        },
        "core": core,
        "core_digest": core_digest(core),
    }
    payload["report_signature_sha256"] = _signature_for_contract(payload)
    return payload


def validate_execution_freeze_contract_final(
    payload: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    normalized: dict[str, Any] = {}

    schema_version = _validate_non_empty_string(
        payload.get("schema_version"),
        field_path="schema_version",
        issues=issues,
    )
    if schema_version is not None:
        if schema_version != FINAL_SCHEMA_VERSION:
            issues.append(
                _issue(
                    "invalid_schema_version",
                    "schema_version",
                    f"schema_version must equal {FINAL_SCHEMA_VERSION!r}",
                )
            )
        normalized["schema_version"] = schema_version

    artifact_kind = _validate_non_empty_string(
        payload.get("artifact_kind"),
        field_path="artifact_kind",
        issues=issues,
    )
    if artifact_kind is not None:
        if artifact_kind != FINAL_ARTIFACT_KIND:
            issues.append(
                _issue(
                    "invalid_artifact_kind",
                    "artifact_kind",
                    f"artifact_kind must equal {FINAL_ARTIFACT_KIND!r}",
                )
            )
        normalized["artifact_kind"] = artifact_kind

    generated_at = _validate_non_empty_string(
        payload.get("generated_at"),
        field_path="generated_at",
        issues=issues,
    )
    if generated_at is not None:
        try:
            normalized["generated_at"] = _parse_timestamp(
                generated_at,
                field_path="generated_at",
            )
        except ValueError as exc:
            issues.append(_issue("invalid_timestamp", "generated_at", str(exc)))

    execution_sha = _validate_non_empty_string(
        payload.get("execution_sha"),
        field_path="execution_sha",
        issues=issues,
    )
    if execution_sha is not None:
        normalized["execution_sha"] = execution_sha
        if execution_sha == UNSET_EXECUTION_SHA:
            issues.append(
                _issue(
                    "execution_sha_unset",
                    "execution_sha",
                    "execution_sha must be a real git SHA in the final freeze contract",
                )
            )

    historical_reference_commits = _validate_historical_reference_commits(
        payload.get("historical_reference_commits"),
        issues=issues,
    )
    normalized["historical_reference_commits"] = historical_reference_commits
    if execution_sha is not None and execution_sha in set(historical_reference_commits):
        issues.append(
            _issue(
                "historical_execution_authority_conflict",
                "execution_sha",
                "execution_sha must not equal any historical_reference_commits entry",
            )
        )

    normalized["read_only_authority_refs"] = _validate_read_only_authority_refs(
        payload.get("read_only_authority_refs"),
        repo_root=repo_root,
        issues=issues,
    )

    critic_baseline_authority = _validate_non_empty_string(
        payload.get("critic_baseline_authority"),
        field_path="critic_baseline_authority",
        issues=issues,
    )
    if critic_baseline_authority is not None:
        normalized["critic_baseline_authority"] = critic_baseline_authority

    critic_candidate_track = _validate_non_empty_string(
        payload.get("critic_candidate_track"),
        field_path="critic_candidate_track",
        issues=issues,
    )
    if critic_candidate_track is not None:
        normalized["critic_candidate_track"] = critic_candidate_track

    policy = _validate_mapping(
        payload.get("policy"), field_path="policy", issues=issues
    )
    normalized_policy: dict[str, Any] = {}
    expected_allowed_scopes: tuple[str, ...] | None = None
    freeze_integrity_raw = payload.get("freeze_integrity")
    if isinstance(freeze_integrity_raw, Mapping):
        repo_snapshot_relative_path_raw = freeze_integrity_raw.get(
            "repo_snapshot_relative_path"
        )
        if (
            isinstance(repo_snapshot_relative_path_raw, str)
            and repo_snapshot_relative_path_raw.strip()
        ):
            try:
                expected_allowed_scopes = (
                    _allowed_post_freeze_write_scopes_from_snapshot_path(
                        repo_root=repo_root,
                        repo_snapshot_relative_path=repo_snapshot_relative_path_raw,
                    )
                )
            except ValueError as exc:
                issues.append(
                    _issue(
                        "invalid_value",
                        "freeze_integrity.repo_snapshot_relative_path",
                        str(exc),
                    )
                )
    if policy is not None:
        normalized_policy = _validate_exact_mapping(
            policy,
            field_path="policy",
            expected=FINAL_POLICY_CONTRACT,
            issues=issues,
        )
        allowed_scopes = policy.get("allowed_post_freeze_write_scopes")
        if not isinstance(allowed_scopes, list):
            issues.append(
                _issue(
                    "wrong_type",
                    "policy.allowed_post_freeze_write_scopes",
                    "policy.allowed_post_freeze_write_scopes must be a list",
                )
            )
        else:
            normalized_scopes: list[str] = []
            for index, raw_scope in enumerate(allowed_scopes):
                scope = _validate_non_empty_string(
                    raw_scope,
                    field_path=f"policy.allowed_post_freeze_write_scopes[{index}]",
                    issues=issues,
                )
                if scope is not None:
                    normalized_scopes.append(scope)
            expected_scopes = list(
                expected_allowed_scopes or DEFAULT_ALLOWED_POST_FREEZE_WRITE_SCOPES
            )
            if normalized_scopes != expected_scopes:
                issues.append(
                    _issue(
                        "invalid_value",
                        "policy.allowed_post_freeze_write_scopes",
                        "policy.allowed_post_freeze_write_scopes must match the freeze allowlist",
                    )
                )
            normalized_policy["allowed_post_freeze_write_scopes"] = normalized_scopes
    normalized["policy"] = normalized_policy

    normalized["threshold_policy"] = _validate_exact_mapping(
        payload.get("threshold_policy"),
        field_path="threshold_policy",
        expected=THRESHOLD_POLICY,
        issues=issues,
    )

    freshness = _validate_mapping(
        payload.get("freshness"),
        field_path="freshness",
        issues=issues,
    )
    normalized_freshness: dict[str, Any] = {}
    if freshness is not None:
        for field_name in (
            "execution_sha",
            "manifest_hash",
            "checkpoint_id",
            "seed_bundle_id",
            "timestamp",
        ):
            value = _validate_non_empty_string(
                freshness.get(field_name),
                field_path=f"freshness.{field_name}",
                issues=issues,
            )
            if value is None:
                continue
            if field_name == "timestamp":
                try:
                    value = _parse_timestamp(
                        value, field_path=f"freshness.{field_name}"
                    )
                except ValueError as exc:
                    issues.append(
                        _issue(
                            "invalid_timestamp",
                            f"freshness.{field_name}",
                            str(exc),
                        )
                    )
                    continue
            normalized_freshness[field_name] = value
        if (
            execution_sha is not None
            and normalized_freshness.get("execution_sha") != execution_sha
        ):
            issues.append(
                _issue(
                    "execution_sha_freshness_mismatch",
                    "freshness.execution_sha",
                    "freshness.execution_sha must equal execution_sha",
                )
            )
    normalized["freshness"] = normalized_freshness

    freeze_integrity = _validate_mapping(
        payload.get("freeze_integrity"),
        field_path="freeze_integrity",
        issues=issues,
    )
    normalized_integrity: dict[str, Any] = {}
    if freeze_integrity is not None:
        mode = _validate_non_empty_string(
            freeze_integrity.get("mode"),
            field_path="freeze_integrity.mode",
            issues=issues,
        )
        if mode is not None:
            if mode != "fail_closed":
                issues.append(
                    _issue(
                        "invalid_value",
                        "freeze_integrity.mode",
                        "freeze_integrity.mode must equal 'fail_closed'",
                    )
                )
            normalized_integrity["mode"] = mode
        manifest_hash = _validate_non_empty_string(
            freeze_integrity.get("toolchain_manifest_hash"),
            field_path="freeze_integrity.toolchain_manifest_hash",
            issues=issues,
        )
        if manifest_hash is not None:
            normalized_integrity["toolchain_manifest_hash"] = manifest_hash
            if normalized_freshness.get("manifest_hash") != manifest_hash:
                issues.append(
                    _issue(
                        "manifest_hash_mismatch",
                        "freeze_integrity.toolchain_manifest_hash",
                        "freeze_integrity.toolchain_manifest_hash must equal freshness.manifest_hash",
                    )
                )
            if len(manifest_hash) != 64 or any(
                ch not in "0123456789abcdef" for ch in manifest_hash
            ):
                issues.append(
                    _issue(
                        "invalid_sha256",
                        "freeze_integrity.toolchain_manifest_hash",
                        "freeze_integrity.toolchain_manifest_hash must be a lowercase sha256 hex digest",
                    )
                )
        for field_name in (
            "repo_snapshot_relative_path",
            "repo_commit_relative_path",
            "repo_status_relative_path",
            "on_drift",
        ):
            value = _validate_non_empty_string(
                freeze_integrity.get(field_name),
                field_path=f"freeze_integrity.{field_name}",
                issues=issues,
            )
            if value is not None:
                normalized_integrity[field_name] = value
        allowed_scopes = freeze_integrity.get("allowed_post_freeze_write_scopes")
        if not isinstance(allowed_scopes, list):
            issues.append(
                _issue(
                    "wrong_type",
                    "freeze_integrity.allowed_post_freeze_write_scopes",
                    "freeze_integrity.allowed_post_freeze_write_scopes must be a list",
                )
            )
        else:
            normalized_scopes = []
            for index, raw_scope in enumerate(allowed_scopes):
                scope = _validate_non_empty_string(
                    raw_scope,
                    field_path=(
                        f"freeze_integrity.allowed_post_freeze_write_scopes[{index}]"
                    ),
                    issues=issues,
                )
                if scope is not None:
                    normalized_scopes.append(scope)
            expected_scopes = list(
                expected_allowed_scopes or DEFAULT_ALLOWED_POST_FREEZE_WRITE_SCOPES
            )
            if normalized_scopes != expected_scopes:
                issues.append(
                    _issue(
                        "invalid_value",
                        "freeze_integrity.allowed_post_freeze_write_scopes",
                        "freeze_integrity.allowed_post_freeze_write_scopes must match the freeze allowlist",
                    )
                )
            normalized_integrity["allowed_post_freeze_write_scopes"] = normalized_scopes
        toolchain_scope = _validate_mapping(
            freeze_integrity.get("toolchain_scope"),
            field_path="freeze_integrity.toolchain_scope",
            issues=issues,
        )
        if toolchain_scope is not None:
            normalized_integrity["toolchain_scope"] = _validate_exact_mapping(
                toolchain_scope,
                field_path="freeze_integrity.toolchain_scope",
                expected=WORKTREE_MANIFEST_SCOPE,
                issues=issues,
            )
    normalized["freeze_integrity"] = normalized_integrity

    core_mapping = _validate_mapping(
        payload.get("core"), field_path="core", issues=issues
    )
    normalized_core: dict[str, Any] = {}
    if core_mapping is not None:
        core_commit = _validate_non_empty_string(
            core_mapping.get("commit"),
            field_path="core.commit",
            issues=issues,
        )
        if core_commit is not None:
            normalized_core["commit"] = core_commit
            if execution_sha is not None and core_commit != execution_sha:
                issues.append(
                    _issue(
                        "execution_sha_core_commit_mismatch",
                        "core.commit",
                        "core.commit must equal execution_sha",
                    )
                )
    normalized["core"] = normalized_core

    computed_core_digest = core_digest(normalized_core)
    normalized["core_digest"] = computed_core_digest
    declared_core_digest = payload.get("core_digest")
    if (
        declared_core_digest is not None
        and declared_core_digest != computed_core_digest
    ):
        issues.append(
            _issue(
                "core_digest_mismatch",
                "core_digest",
                "declared core_digest does not match normalized core payload",
            )
        )

    computed_signature = _signature_for_contract(normalized)
    normalized["report_signature_sha256"] = computed_signature
    declared_signature = payload.get("report_signature_sha256")
    if declared_signature is not None and declared_signature != computed_signature:
        issues.append(
            _issue(
                "report_signature_sha256_mismatch",
                "report_signature_sha256",
                "declared report_signature_sha256 does not match normalized payload",
            )
        )

    return {
        "formal_eligibility": "ALLOW" if not issues else "BLOCK",
        "issues": issues,
        "normalized_contract": normalized,
        "core_digest": computed_core_digest,
    }


def materialize_final_execution_freeze(
    *,
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
    draft_contract_json: Path | None = None,
    freeze_timestamp: str | None = None,
) -> dict[str, Any]:
    resolved_output_dir = _validate_output_dir(
        resolve_repo_contained_path(
            repo_root,
            output_dir,
            field_name="output_dir",
            canonical_roots=CANONICAL_EXECUTION_ROOT_SCOPES,
            reject_noncanonical_parts=True,
        )
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    draft_path = (
        resolve_repo_contained_path(
            repo_root,
            draft_contract_json,
            field_name="draft_contract_json",
            canonical_roots=CANONICAL_EXECUTION_ROOT_SCOPES,
            reject_noncanonical_parts=True,
        )
        if draft_contract_json is not None
        else (resolved_output_dir / EXECUTION_CONTRACT_JSON_NAME).resolve()
    )
    if not draft_path.exists() or not draft_path.is_file():
        raise ValueError(f"draft contract JSON does not exist: {draft_path}")
    draft_payload = json.loads(draft_path.read_text(encoding="utf-8"))
    if not isinstance(draft_payload, Mapping):
        raise TypeError(
            f"draft contract JSON must contain an object, got {type(draft_payload).__name__}"
        )
    generated_at = _parse_timestamp(
        freeze_timestamp or _now_iso(),
        field_path="freeze_timestamp",
    )
    execution_sha = _git_text(repo_root, "rev-parse", "HEAD").strip()
    if not execution_sha:
        raise ValueError("git rev-parse HEAD returned an empty execution SHA")
    status_capture = _capture_git_status(repo_root)
    blocked_dirty_entries = _blocked_final_freeze_dirty_entries(
        cast(Sequence[Mapping[str, str]], status_capture["entries"]),
        allowed_write_scopes=allowed_post_freeze_write_scopes_for_output_dir(
            repo_root=repo_root,
            output_dir=resolved_output_dir,
        ),
    )
    if blocked_dirty_entries:
        blocked_paths = ", ".join(entry["path"] for entry in blocked_dirty_entries)
        raise ValueError(
            "final freeze blocks dirty runnable authority paths under work/** or tests/**; "
            f"commit or clean them first: {blocked_paths}"
        )
    repo_snapshot, commit_text, status_text = build_repo_snapshot(
        repo_root=repo_root,
        output_dir=resolved_output_dir,
        execution_sha=execution_sha,
        generated_at=generated_at,
    )
    _write_text(resolved_output_dir / REPO_COMMIT_TXT_NAME, commit_text)
    _write_text(resolved_output_dir / REPO_STATUS_TXT_NAME, status_text)
    _write_json(resolved_output_dir / REPO_SNAPSHOT_JSON_NAME, repo_snapshot)
    payload = build_execution_freeze_contract_final(
        draft_payload=draft_payload,
        repo_root=repo_root,
        output_dir=resolved_output_dir,
        execution_sha=execution_sha,
        generated_at=generated_at,
        toolchain_manifest_hash=str(repo_snapshot["toolchain_manifest_hash"]),
    )
    validation = validate_execution_freeze_contract_final(payload, repo_root=repo_root)
    if validation["formal_eligibility"] != "ALLOW":
        first_issue = validation["issues"][0] if validation["issues"] else None
        if isinstance(first_issue, Mapping):
            message = str(
                first_issue.get("message", "final contract validation failed")
            )
        else:
            message = "final contract validation failed"
        raise ValueError(message)
    normalized_payload = dict(validation["normalized_contract"])
    _write_json(
        resolved_output_dir / FINAL_EXECUTION_CONTRACT_JSON_NAME,
        normalized_payload,
    )
    return normalized_payload


def _load_json_object_required(path: Path, *, field_path: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{field_path} must contain a JSON object")
    return dict(payload)


def check_execution_freeze_integrity(
    *,
    final_contract_json: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    contract_path = resolve_repo_contained_path(
        repo_root,
        final_contract_json,
        field_name="final_contract_json",
        canonical_roots=CANONICAL_EXECUTION_ROOT_SCOPES,
        reject_noncanonical_parts=True,
    )
    if not contract_path.exists() or not contract_path.is_file():
        raise ValueError(f"final contract JSON does not exist: {contract_path}")
    payload = _load_json_object_required(
        contract_path, field_path="final_contract_json"
    )
    contract_validation = validate_execution_freeze_contract_final(
        payload,
        repo_root=repo_root,
    )
    issues: list[dict[str, str]] = list(contract_validation["issues"])
    normalized_contract = cast(
        Mapping[str, Any], contract_validation["normalized_contract"]
    )
    freeze_integrity = cast(
        Mapping[str, Any], normalized_contract.get("freeze_integrity", {})
    )
    snapshot_relative_path = str(
        freeze_integrity.get("repo_snapshot_relative_path", "")
    ).strip()
    if not snapshot_relative_path:
        issues.append(
            _issue(
                "missing_required_field",
                "freeze_integrity.repo_snapshot_relative_path",
                "freeze_integrity.repo_snapshot_relative_path is required for integrity checks",
            )
        )
        snapshot_payload: dict[str, Any] = {}
    else:
        snapshot_path = resolve_repo_contained_path(
            repo_root,
            snapshot_relative_path,
            field_name="freeze_integrity.repo_snapshot_relative_path",
            canonical_roots=CANONICAL_EXECUTION_ROOT_SCOPES,
            reject_noncanonical_parts=True,
        )
        if not snapshot_path.exists() or not snapshot_path.is_file():
            issues.append(
                _issue(
                    "missing_repo_snapshot",
                    "freeze_integrity.repo_snapshot_relative_path",
                    f"repo snapshot JSON does not exist: {snapshot_path}",
                )
            )
            snapshot_payload = {}
        else:
            snapshot_payload = _load_json_object_required(
                snapshot_path,
                field_path="freeze_integrity.repo_snapshot_relative_path",
            )

    expected_execution_sha = str(normalized_contract.get("execution_sha", "")).strip()
    observed_execution_sha = _git_text(repo_root, "rev-parse", "HEAD").strip()
    if expected_execution_sha and observed_execution_sha != expected_execution_sha:
        issues.append(
            _issue(
                "execution_sha_drift",
                "execution_sha",
                "current HEAD differs from the frozen execution_sha; reopen execution SHA",
            )
        )

    current_manifest = build_worktree_manifest(repo_root)
    expected_manifest_hash = str(
        freeze_integrity.get("toolchain_manifest_hash")
        or normalized_contract.get("freshness", {}).get("manifest_hash")
        or ""
    ).strip()
    observed_manifest_hash = str(current_manifest["toolchain_manifest_hash"])
    if expected_manifest_hash and observed_manifest_hash != expected_manifest_hash:
        issues.append(
            _issue(
                "toolchain_manifest_hash_drift",
                "freeze_integrity.toolchain_manifest_hash",
                "work/** logic drift detected after freeze; reopen execution SHA",
            )
        )

    status_capture = _capture_git_status(repo_root)
    allowed_write_scopes = list(
        expected_allowed_scopes
        if (
            expected_allowed_scopes := tuple(
                snapshot_payload.get("allowed_post_freeze_write_scopes", [])
            )
        )
        else _allowed_post_freeze_write_scopes_from_snapshot_path(
            repo_root=repo_root,
            repo_snapshot_relative_path=snapshot_relative_path,
        )
    )
    current_disallowed_entries = {
        str(item["path"]): _file_entry(repo_root, str(item["path"]))
        | {
            "status_code": str(item["status_code"]),
            "raw_line": str(item["raw_line"]),
        }
        for item in cast(Sequence[Mapping[str, str]], status_capture["entries"])
        if not _is_allowed_post_freeze_path_for_scopes(
            str(item["path"]),
            allowed_write_scopes,
        )
    }
    frozen_override_list = cast(
        Sequence[Mapping[str, Any]],
        snapshot_payload.get("frozen_worktree_overrides", []),
    )
    frozen_overrides = {
        str(item.get("relative_path", "")).strip(): dict(item)
        for item in frozen_override_list
        if str(item.get("relative_path", "")).strip()
    }
    frozen_runnable_dirty_paths = sorted(
        relative_path
        for relative_path in frozen_overrides
        if _is_runnable_dirty_path_for_final_freeze(relative_path)
    )
    for relative_path in frozen_runnable_dirty_paths:
        issues.append(
            _issue(
                "invalid_frozen_runnable_dirty_path",
                relative_path,
                "final freeze must not record dirty runnable authority paths under work/** or tests/**",
            )
        )
    for relative_path, current_entry in current_disallowed_entries.items():
        frozen_entry = frozen_overrides.get(relative_path)
        if frozen_entry is None:
            issues.append(
                _issue(
                    "unexpected_post_freeze_dirty_path",
                    relative_path,
                    "dirty path is outside the post-freeze allowlist and was not part of the frozen snapshot",
                )
            )
            continue
        for field_name in ("status_code", "content_sha256", "path_kind"):
            if frozen_entry.get(field_name) != current_entry.get(field_name):
                issues.append(
                    _issue(
                        "frozen_dirty_path_drift",
                        relative_path,
                        f"frozen dirty path drifted in {field_name}; reopen execution SHA",
                    )
                )
                break
    for relative_path in frozen_overrides:
        if relative_path not in current_disallowed_entries:
            issues.append(
                _issue(
                    "frozen_dirty_path_missing",
                    relative_path,
                    "frozen dirty path no longer matches the recorded working-tree override set",
                )
            )

    return {
        "formal_eligibility": "ALLOW" if not issues else "BLOCK",
        "issues": issues,
        "execution_sha": expected_execution_sha,
        "observed_execution_sha": observed_execution_sha,
        "expected_toolchain_manifest_hash": expected_manifest_hash,
        "observed_toolchain_manifest_hash": observed_manifest_hash,
        "current_disallowed_dirty_paths": sorted(current_disallowed_entries),
        "frozen_runnable_dirty_paths": frozen_runnable_dirty_paths,
    }


def materialize_execution_freeze_contract_draft(
    *,
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
    execution_sha: str = UNSET_EXECUTION_SHA,
    historical_reference_commits: Sequence[str] = DEFAULT_HISTORICAL_REFERENCE_COMMITS,
    read_only_authority_ref_specs: Sequence[
        Mapping[str, str]
    ] = DEFAULT_READ_ONLY_AUTHORITY_REF_SPECS,
) -> dict[str, Any]:
    resolved_output_dir = _validate_output_dir(
        resolve_repo_contained_path(
            repo_root,
            output_dir,
            field_name="output_dir",
            canonical_roots=CANONICAL_EXECUTION_ROOT_SCOPES,
            reject_noncanonical_parts=True,
        )
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_execution_freeze_contract_draft(
        repo_root=repo_root,
        generated_at=generated_at,
        execution_sha=execution_sha,
        historical_reference_commits=historical_reference_commits,
        read_only_authority_ref_specs=read_only_authority_ref_specs,
    )
    validation = validate_execution_freeze_contract_draft(payload, repo_root=repo_root)
    if validation["formal_eligibility"] != "ALLOW":
        first_issue = validation["issues"][0] if validation["issues"] else None
        if isinstance(first_issue, Mapping):
            message = str(first_issue.get("message", "contract validation failed"))
        else:
            message = "contract validation failed"
        raise ValueError(message)
    normalized_payload = dict(validation["normalized_contract"])
    _write_json(resolved_output_dir / EXECUTION_CONTRACT_JSON_NAME, normalized_payload)
    return normalized_payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.mode == "draft":
            payload = materialize_execution_freeze_contract_draft(
                output_dir=args.output_dir
            )
            exit_code = 0
        elif args.mode == "finalize":
            payload = materialize_final_execution_freeze(
                output_dir=args.output_dir,
                draft_contract_json=args.draft_contract_json,
                freeze_timestamp=args.freeze_timestamp,
            )
            exit_code = 0
        else:
            final_contract_json = (
                args.final_contract_json
                if args.final_contract_json is not None
                else args.output_dir / FINAL_EXECUTION_CONTRACT_JSON_NAME
            )
            payload = check_execution_freeze_integrity(
                final_contract_json=final_contract_json,
            )
            exit_code = 0 if payload.get("formal_eligibility") == "ALLOW" else 1
    except SystemExit:
        raise
    except (OSError, TypeError, ValueError) as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
