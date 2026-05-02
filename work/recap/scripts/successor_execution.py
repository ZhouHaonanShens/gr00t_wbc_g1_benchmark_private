#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import label_writer, text_indicator
from work.recap.scripts import apple_recap_execution_contract
from work.recap.scripts import build_readonly_refs
from work.recap.scripts import build_uplift_schemas
from work.recap.scripts import inspect_mainline_carrier
from work.recap.scripts import state_conditioned_bucket_a_import


BLOCKED_EXECUTION_ROOT = Path("agent/artifacts/apple_recap_exec")
SUCCESSOR_EXECUTION_ROOT = Path("agent/artifacts/apple_recap_exec_successor")
BLOCKED_EXECUTION_SHA = "29d7396b51d5f3db1204f59df2e376ebd7e64ef9"
SUCCESSOR_AUTHORITY_LEVEL = "authoritative_successor"
SUCCESSOR_MODE = "authoritative_export_fix"

DEFAULT_SOURCE_DATASET_DIR = Path("agent/artifacts/recap_datasets/fullsize_relabel_v1")
DEFAULT_SOURCE_LABELS = DEFAULT_SOURCE_DATASET_DIR / "m2_labels" / "labels.jsonl"
DEFAULT_SOURCE_CRITIC_DIR = Path("agent/artifacts/critics/task7_real_critic_v2")
DEFAULT_SOURCE_CONTINUOUS_ADVANTAGE_CONTRACT = (
    DEFAULT_SOURCE_DATASET_DIR / "continuous_advantage_contract.json"
)
DEFAULT_EXPERIMENT_MATRIX_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/experiment_matrix/gr00t_experiment_matrix.json"
)
DEFAULT_BASELINE_SUITE_DIR = BLOCKED_EXECUTION_ROOT / "baseline_suite"
SUCCESSOR_ROOT_CANONICAL_ROOTS: tuple[str, ...] = (
    "agent/artifacts/apple_recap_exec_successor/",
)
BLOCKED_ROOT_CANONICAL_ROOTS: tuple[str, ...] = (
    apple_recap_execution_contract.DEFAULT_EXECUTION_ROOT_SCOPE,
)
SOURCE_DATASET_CANONICAL_ROOTS: tuple[str, ...] = ("agent/artifacts/recap_datasets/",)
CRITIC_DIR_CANONICAL_ROOTS: tuple[str, ...] = ("agent/artifacts/critics/",)
EXPERIMENT_MATRIX_CANONICAL_ROOTS: tuple[str, ...] = (
    "agent/artifacts/gr00t_anchor_controller_recap/experiment_matrix/",
)

SUCCESSOR_LABELS_DIRNAME = "authoritative_export_fix_labels"
SUCCESSOR_LABELS_MANIFEST_NAME = "materialization_manifest.json"

SUCCESSOR_AUTHORITY_RELATIVE_PATHS: tuple[str, ...] = (
    apple_recap_execution_contract.FINAL_EXECUTION_CONTRACT_JSON_NAME,
    "baseline_refs_manifest.json",
    "experiment_matrix_frozen.json",
    "B0_E1_E2_run_ledger.csv",
    "B0_repro_band.json",
    "carrier_parity_report.json",
    "carrier_sample_rows.csv",
    "carrier_inspection.md",
)

BLOCKED_AUTHORITY_RELATIVE_PATHS: tuple[str, ...] = (
    apple_recap_execution_contract.FINAL_EXECUTION_CONTRACT_JSON_NAME,
    "uplift_verdict.json",
    "baseline_refs_manifest.json",
    "experiment_matrix_frozen.json",
    "B0_repro_band.json",
    "carrier_parity_report.json",
    "carrier_sample_rows.csv",
    "carrier_inspection.md",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="successor_execution.py",
        description=(
            "Materialize the AppleToPlate successor authority lane under "
            "agent/artifacts/apple_recap_exec_successor without touching the blocked "
            "current execution root."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--successor-root", type=Path, default=SUCCESSOR_EXECUTION_ROOT)
    parser.add_argument("--blocked-root", type=Path, default=BLOCKED_EXECUTION_ROOT)
    parser.add_argument(
        "--source-dataset-dir",
        type=Path,
        default=DEFAULT_SOURCE_DATASET_DIR,
    )
    parser.add_argument(
        "--critic-dir",
        type=Path,
        default=DEFAULT_SOURCE_CRITIC_DIR,
    )
    parser.add_argument(
        "--baseline-suite-dir",
        type=Path,
        default=DEFAULT_BASELINE_SUITE_DIR,
    )
    parser.add_argument(
        "--experiment-matrix-json",
        type=Path,
        default=DEFAULT_EXPERIMENT_MATRIX_JSON,
    )
    parser.add_argument(
        "--freeze-timestamp",
        type=str,
        default=None,
        help="Optional ISO-8601 timestamp for successor freeze finalization.",
    )
    parser.add_argument(
        "--epsilon-quantile",
        type=float,
        default=0.7,
        help="Labeler epsilon quantile for authoritative export-fix rematerialization.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_repo_path(repo_root: Path, raw: Path | str) -> Path:
    return apple_recap_execution_contract._resolve_path(repo_root, raw)


def _repo_relative_path(repo_root: Path, path: Path) -> str:
    return apple_recap_execution_contract._repo_relative_path(repo_root, path)


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return dict(payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"expected JSON object line at {path}, got {type(payload).__name__}"
                )
            rows.append(dict(payload))
    return rows


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    if not rows:
        raise ValueError(f"cannot write empty CSV rows: {path}")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    tmp.replace(path)
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _unlink_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _deep_copy_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(json.dumps(dict(payload), ensure_ascii=True, sort_keys=True)),
    )


def _ensure_successor_root(
    *,
    repo_root: Path,
    successor_root: Path,
    blocked_root: Path,
) -> Path:
    resolved_successor = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=successor_root,
        field_name="successor_root",
        canonical_roots=SUCCESSOR_ROOT_CANONICAL_ROOTS,
    )
    resolved_blocked = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=blocked_root,
        field_name="blocked_root",
        canonical_roots=BLOCKED_ROOT_CANONICAL_ROOTS,
    )
    successor_text = str(resolved_successor)
    blocked_text = str(resolved_blocked)
    if successor_text == blocked_text:
        raise ValueError(
            "successor authority root must not equal the blocked current execution root"
        )
    if successor_text.startswith(blocked_text + "/"):
        raise ValueError(
            "successor authority root must not be nested under the blocked current execution root"
        )
    resolved_successor.mkdir(parents=True, exist_ok=True)
    return resolved_successor


def snapshot_authority_hashes(
    *,
    repo_root: Path,
    execution_root: Path | str,
    relative_paths: Sequence[str],
) -> dict[str, str]:
    resolved_root = _resolve_repo_path(repo_root, execution_root)
    hashes: dict[str, str] = {}
    for relative_path in relative_paths:
        artifact_path = (resolved_root / relative_path).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"missing authority artifact for hash snapshot: {artifact_path}"
            )
        hashes[str(relative_path)] = apple_recap_execution_contract._sha256_file(
            artifact_path
        )
    return hashes


def assert_hashes_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> None:
    mismatches = [
        relative_path
        for relative_path, before_hash in before.items()
        if str(after.get(relative_path, "")).strip() != str(before_hash).strip()
    ]
    if mismatches:
        raise ValueError(
            "blocked root authority artifacts changed unexpectedly: "
            + ", ".join(sorted(mismatches))
        )


def _successor_context(
    *,
    blocked_root_relative: str,
    blocked_execution_sha: str,
    execution_root_relative: str,
) -> dict[str, Any]:
    return {
        "mode": SUCCESSOR_MODE,
        "supersedes": {
            "execution_root": blocked_root_relative,
            "execution_sha": blocked_execution_sha,
        },
        "current_execution_reopen_forbidden": True,
        "requires_successor_execution": True,
        "execution_root": execution_root_relative,
    }


def _augment_successor_payload(
    payload: Mapping[str, Any],
    *,
    execution_root_relative: str,
    execution_sha: str,
    freshness: Mapping[str, Any],
    successor_context: Mapping[str, Any],
    authority_level: str = SUCCESSOR_AUTHORITY_LEVEL,
    gating_eligible: bool = True,
) -> dict[str, Any]:
    augmented = _deep_copy_payload(payload)
    augmented["execution_root"] = execution_root_relative
    augmented["execution_sha"] = execution_sha
    augmented["freshness"] = _deep_copy_payload(freshness)
    augmented["authority_level"] = authority_level
    augmented["gating_eligible"] = bool(gating_eligible)
    augmented["successor_context"] = _deep_copy_payload(successor_context)
    return augmented


def validate_successor_authority_artifact(
    payload: Mapping[str, Any],
    *,
    artifact_relative_path: str,
    expected_execution_root: str,
    expected_execution_sha: str,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    relative_path = str(artifact_relative_path).strip()
    if "phase_a_tooling_draft/" in f"/{relative_path}":
        issues.append(
            apple_recap_execution_contract._issue(
                "phase_a_authority_forbidden",
                "artifact_relative_path",
                "phase_a_tooling_draft/** must not masquerade as successor authority",
            )
        )
    execution_root = str(payload.get("execution_root", "")).strip()
    if execution_root != expected_execution_root:
        issues.append(
            apple_recap_execution_contract._issue(
                "execution_root_mismatch",
                "execution_root",
                "successor authority artifact must bind to the configured successor execution_root",
            )
        )
    execution_sha = str(payload.get("execution_sha", "")).strip()
    if execution_sha != expected_execution_sha:
        issues.append(
            apple_recap_execution_contract._issue(
                "execution_sha_mismatch",
                "execution_sha",
                "successor authority artifact must bind to the successor execution_sha",
            )
        )
    freshness = payload.get("freshness")
    if (
        not isinstance(freshness, Mapping)
        or str(freshness.get("execution_sha", "")).strip() != expected_execution_sha
    ):
        issues.append(
            apple_recap_execution_contract._issue(
                "freshness_execution_sha_mismatch",
                "freshness.execution_sha",
                "successor freshness.execution_sha must equal the successor execution_sha",
            )
        )
    if bool(payload.get("gating_eligible")) is not True:
        issues.append(
            apple_recap_execution_contract._issue(
                "successor_gating_ineligible",
                "gating_eligible",
                "authoritative successor artifacts must remain gating_eligible=true",
            )
        )
    if str(payload.get("authority_level", "")).strip() != SUCCESSOR_AUTHORITY_LEVEL:
        issues.append(
            apple_recap_execution_contract._issue(
                "authority_level_mismatch",
                "authority_level",
                "authoritative successor artifacts must declare authority_level=authoritative_successor",
            )
        )
    successor_context = payload.get("successor_context")
    if (
        not isinstance(successor_context, Mapping)
        or str(successor_context.get("mode", "")).strip() != SUCCESSOR_MODE
    ):
        issues.append(
            apple_recap_execution_contract._issue(
                "successor_mode_mismatch",
                "successor_context.mode",
                "successor authority artifacts must declare successor_context.mode=authoritative_export_fix",
            )
        )
    source_artifacts = payload.get("source_artifacts")
    if isinstance(source_artifacts, list):
        for index, source_artifact in enumerate(source_artifacts):
            if not isinstance(source_artifact, Mapping):
                continue
            source_relative_path = str(source_artifact.get("relative_path", "")).strip()
            if "phase_a_tooling_draft/" in f"/{source_relative_path}":
                issues.append(
                    apple_recap_execution_contract._issue(
                        "phase_a_authority_forbidden",
                        f"source_artifacts[{index}].relative_path",
                        "phase_a_tooling_draft/** must not appear as successor authority input",
                    )
                )
            if (
                "research_probe/" in f"/{source_relative_path}"
                or "carrier_backfill" in source_relative_path
            ):
                issues.append(
                    apple_recap_execution_contract._issue(
                        "research_backfill_authority_forbidden",
                        f"source_artifacts[{index}].relative_path",
                        "research_backfill outputs must not appear as successor authority input",
                    )
                )
    return {
        "formal_eligibility": "ALLOW" if not issues else "BLOCK",
        "issues": issues,
        "artifact_relative_path": relative_path,
        "execution_root": execution_root,
        "execution_sha": execution_sha,
    }


def _assert_successor_authority_artifact(
    payload: Mapping[str, Any],
    *,
    artifact_relative_path: str,
    expected_execution_root: str,
    expected_execution_sha: str,
) -> None:
    validation = validate_successor_authority_artifact(
        payload,
        artifact_relative_path=artifact_relative_path,
        expected_execution_root=expected_execution_root,
        expected_execution_sha=expected_execution_sha,
    )
    if validation["formal_eligibility"] != "ALLOW":
        first_issue = validation["issues"][0] if validation["issues"] else None
        if isinstance(first_issue, Mapping):
            raise ValueError(
                str(first_issue.get("message", "successor authority validation failed"))
            )
        raise ValueError("successor authority validation failed")


def materialize_authoritative_export_fix_labels(
    *,
    repo_root: Path,
    source_dataset_dir: Path | str,
    successor_root: Path | str,
    execution_root_relative: str,
    execution_sha: str,
    freshness: Mapping[str, Any],
    successor_context: Mapping[str, Any],
    critic_dir: Path | str,
    epsilon_quantile: float = 0.7,
) -> dict[str, Any]:
    resolved_source_dataset_dir = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=source_dataset_dir,
        field_name="source_dataset_dir",
        canonical_roots=SOURCE_DATASET_CANONICAL_ROOTS,
    )
    resolved_successor_root = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=successor_root,
        field_name="successor_root",
        canonical_roots=SUCCESSOR_ROOT_CANONICAL_ROOTS,
    )
    resolved_source_labels = (
        resolved_source_dataset_dir / "m2_labels" / "labels.jsonl"
    ).resolve()
    resolved_critic_dir = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=critic_dir,
        field_name="critic_dir",
        canonical_roots=CRITIC_DIR_CANONICAL_ROOTS,
    )
    labels_root = resolved_successor_root / SUCCESSOR_LABELS_DIRNAME
    labels_path = labels_root / "m2_labels" / "labels.jsonl"
    stats_path = labels_root / "m2_labels" / "stats.json"
    manifest_path = labels_root / SUCCESSOR_LABELS_MANIFEST_NAME
    for path in (labels_path, stats_path, manifest_path):
        _unlink_if_exists(path)
    labels_root.mkdir(parents=True, exist_ok=True)

    source_labels = _read_jsonl(resolved_source_labels)
    labels: list[dict[str, Any]] = []
    for row in source_labels:
        prompt_raw = text_indicator.require_prompt_raw(
            row.get("prompt_raw"),
            field_name="prompt_raw",
        )
        indicator_mode = text_indicator.indicator_mode_from_indicator_value(
            row.get("indicator_I"),
            field_name="indicator_I",
        )
        canonical_row = dict(row)
        canonical_row[text_indicator.RECAP_TEXT_INDICATOR_CARRIER_FIELD] = (
            text_indicator.build_canonical_text_indicator(prompt_raw, indicator_mode)
        )
        label_writer.validate_label_record(canonical_row)
        labels.append(canonical_row)
    stats = dict(
        label_writer.write_m2_label_outputs(
            labels_root,
            labels,
            epsilon_strategy="quantile",
        )
    )
    manifest = {
        "schema_version": "apple_recap_successor_authoritative_export_fix_labels_v1",
        "artifact_kind": "apple_recap_successor_authoritative_export_fix_labels",
        "generated_at": _now_iso(),
        "execution_root": execution_root_relative,
        "execution_sha": execution_sha,
        "freshness": _deep_copy_payload(freshness),
        "authority_level": SUCCESSOR_AUTHORITY_LEVEL,
        "gating_eligible": True,
        "successor_context": _deep_copy_payload(successor_context),
        "materialization_mode": SUCCESSOR_MODE,
        "source_dataset_dir": _repo_relative_path(
            repo_root, resolved_source_dataset_dir
        ),
        "source_labels_path": _repo_relative_path(repo_root, resolved_source_labels),
        "critic_dir": _repo_relative_path(repo_root, resolved_critic_dir),
        "epsilon_quantile": float(epsilon_quantile),
        "label_count": int(len(labels)),
        "output_artifacts": {
            "labels_jsonl": _repo_relative_path(repo_root, labels_path),
            "stats_json": _repo_relative_path(repo_root, stats_path),
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "labels_root": labels_root,
        "labels_path": labels_path,
        "stats_path": stats_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "stats": stats,
    }


def _build_successor_ledger_rows(
    *,
    execution_root_relative: str,
    execution_sha: str,
    b0_formal_success_count: int,
    b0_formal_success_rate: float,
    b0_long_success_rate: float,
) -> list[dict[str, object]]:
    return [
        {
            "row_id": "B0",
            "execution_root": execution_root_relative,
            "execution_sha": execution_sha,
            "authority_level": SUCCESSOR_AUTHORITY_LEVEL,
            "successor_mode": SUCCESSOR_MODE,
            "stage_status": "ready",
            "row_signal": "screen_flat",
            "formal_success_count": int(b0_formal_success_count),
            "formal_success_rate": f"{float(b0_formal_success_rate):.6f}",
            "long_success_rate": f"{float(b0_long_success_rate):.6f}",
            "comparable_to": "",
            "verdict": "inconclusive_rerun_needed",
        },
        {
            "row_id": "E1",
            "execution_root": execution_root_relative,
            "execution_sha": execution_sha,
            "authority_level": SUCCESSOR_AUTHORITY_LEVEL,
            "successor_mode": SUCCESSOR_MODE,
            "stage_status": "pending",
            "row_signal": "screen_inconclusive",
            "formal_success_count": 0,
            "formal_success_rate": "0.000000",
            "long_success_rate": "0.000000",
            "comparable_to": "B0",
            "verdict": "inconclusive_rerun_needed",
        },
        {
            "row_id": "E2",
            "execution_root": execution_root_relative,
            "execution_sha": execution_sha,
            "authority_level": SUCCESSOR_AUTHORITY_LEVEL,
            "successor_mode": SUCCESSOR_MODE,
            "stage_status": "pending",
            "row_signal": "screen_inconclusive",
            "formal_success_count": 0,
            "formal_success_rate": "0.000000",
            "long_success_rate": "0.000000",
            "comparable_to": "B0",
            "verdict": "inconclusive_rerun_needed",
        },
    ]


def _baseline_bundle_paths(baseline_suite_dir: Path) -> dict[str, Path]:
    return {
        "official_comparable_10ep": baseline_suite_dir
        / "official_10ep_20000_20009/public_anchor_formal.json",
        "repro_rerun_b_10ep": baseline_suite_dir
        / "repro_rerun_b_21000_21009_live/b0_bundle_eval.json",
        "repro_rerun_c_10ep": baseline_suite_dir
        / "repro_rerun_c_22000_22009_live/b0_bundle_eval.json",
        "extended_50ep": baseline_suite_dir
        / "extended_50ep_30000_30049_live/b0_bundle_eval.json",
    }


def build_successor_b0_repro_band(
    *,
    repo_root: Path,
    baseline_suite_dir: Path | str,
    baseline_refs_manifest_path: Path,
    experiment_matrix_frozen_path: Path,
    execution_root_relative: str,
    execution_sha: str,
    freshness: Mapping[str, Any],
    successor_context: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_baseline_suite_dir = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=baseline_suite_dir,
        field_name="baseline_suite_dir",
        canonical_roots=(
            f"{_repo_relative_path(repo_root, repo_root / BLOCKED_EXECUTION_ROOT / 'baseline_suite')}/",
        ),
    )
    bundle_paths = _baseline_bundle_paths(resolved_baseline_suite_dir)
    official = _read_json(bundle_paths["official_comparable_10ep"])
    repro_b = _read_json(bundle_paths["repro_rerun_b_10ep"])
    repro_c = _read_json(bundle_paths["repro_rerun_c_10ep"])
    extended = _read_json(bundle_paths["extended_50ep"])

    rerun_rates = {
        "A_official_comparable_10ep": float(official["success_rate"]),
        "B_repro_rerun_10ep": float(repro_b["success_rate"]),
        "C_repro_rerun_10ep": float(repro_c["success_rate"]),
    }
    rate_values = list(rerun_rates.values())
    payload = {
        "schema_version": "apple_recap_b0_repro_band_v1",
        "artifact_kind": "apple_recap_b0_repro_band",
        "generated_at": _now_iso(),
        "execution_root": execution_root_relative,
        "execution_sha": execution_sha,
        "freshness": _deep_copy_payload(freshness),
        "authority_level": SUCCESSOR_AUTHORITY_LEVEL,
        "gating_eligible": True,
        "successor_context": _deep_copy_payload(successor_context),
        "core": {"commit": execution_sha},
        "core_digest": apple_recap_execution_contract.core_digest(
            {"commit": execution_sha}
        ),
        "formal_anchor_rate": float(official["success_rate"]),
        "extended_50ep_rate": float(extended["success_rate"]),
        "rerun_rates": rerun_rates,
        "min": float(min(rate_values)),
        "mean": float(sum(rate_values) / float(len(rate_values))),
        "max": float(max(rate_values)),
        "seed_bundles": {
            "official_comparable_10ep": "20000:20009",
            "repro_rerun_b_10ep": "21000:21009",
            "repro_rerun_c_10ep": "22000:22009",
            "extended_50ep": "30000:30049",
        },
        "bundle_artifacts": {
            key: _repo_relative_path(repo_root, path)
            for key, path in bundle_paths.items()
        },
        "official_anchor": {
            "identity_role": "read_only_official_anchor",
            "identity_mutated": False,
            "identity_ref_relative_path": "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/public_anchor/public_anchor_formal.json",
            "formal_comparable_rerun_relative_path": _repo_relative_path(
                repo_root,
                bundle_paths["official_comparable_10ep"],
            ),
            "formal_anchor_success_count": int(official["success_count"]),
            "formal_anchor_success_rate": float(official["success_rate"]),
        },
        "repro_band_policy": {
            "purpose": "variance_explanation_only",
            "contains_three_10ep_reruns": True,
            "contains_one_50ep_extended_baseline": True,
            "not_new_official_anchor": True,
            "official_anchor_redefinition_forbidden": True,
        },
        "source_artifacts": [
            apple_recap_execution_contract.build_read_only_authority_ref(
                repo_root=repo_root,
                artifact_id="public_anchor_identity",
                authority_role="read_only_official_anchor",
                relative_path="agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/public_anchor/public_anchor_formal.json",
                reject_noncanonical_parts=True,
            ),
            apple_recap_execution_contract.build_read_only_authority_ref(
                repo_root=repo_root,
                artifact_id="official_comparable_10ep",
                authority_role="successor_b0_repro_band_source",
                relative_path=bundle_paths["official_comparable_10ep"],
                reject_noncanonical_parts=True,
            ),
            apple_recap_execution_contract.build_read_only_authority_ref(
                repo_root=repo_root,
                artifact_id="repro_rerun_b_10ep",
                authority_role="successor_b0_repro_band_source",
                relative_path=bundle_paths["repro_rerun_b_10ep"],
                reject_noncanonical_parts=True,
            ),
            apple_recap_execution_contract.build_read_only_authority_ref(
                repo_root=repo_root,
                artifact_id="repro_rerun_c_10ep",
                authority_role="successor_b0_repro_band_source",
                relative_path=bundle_paths["repro_rerun_c_10ep"],
                reject_noncanonical_parts=True,
            ),
            apple_recap_execution_contract.build_read_only_authority_ref(
                repo_root=repo_root,
                artifact_id="extended_50ep",
                authority_role="successor_b0_repro_band_source",
                relative_path=bundle_paths["extended_50ep"],
                reject_noncanonical_parts=True,
            ),
            apple_recap_execution_contract.build_read_only_authority_ref(
                repo_root=repo_root,
                artifact_id="baseline_refs_manifest",
                authority_role="successor_b0_repro_band_context",
                relative_path=baseline_refs_manifest_path,
                reject_noncanonical_parts=True,
            ),
            apple_recap_execution_contract.build_read_only_authority_ref(
                repo_root=repo_root,
                artifact_id="experiment_matrix_frozen",
                authority_role="successor_b0_repro_band_context",
                relative_path=experiment_matrix_frozen_path,
                reject_noncanonical_parts=True,
            ),
        ],
    }
    payload["report_signature_sha256"] = apple_recap_execution_contract._sha256_payload(
        {
            key: value
            for key, value in payload.items()
            if key != "report_signature_sha256"
        }
    )
    return payload


def _augment_carrier_sample_rows_csv(
    *,
    csv_path: Path,
    execution_root_relative: str,
    execution_sha: str,
) -> None:
    rows = _read_csv(csv_path)
    if not rows:
        return
    augmented_rows = []
    for row in rows:
        augmented_rows.append(
            {
                "execution_root": execution_root_relative,
                "execution_sha": execution_sha,
                "authority_level": SUCCESSOR_AUTHORITY_LEVEL,
                "successor_mode": SUCCESSOR_MODE,
                **row,
            }
        )
    _write_csv(csv_path, augmented_rows)


def _augment_carrier_markdown(
    *,
    markdown_path: Path,
    execution_root_relative: str,
    execution_sha: str,
    freshness: Mapping[str, Any],
) -> None:
    original = markdown_path.read_text(encoding="utf-8")
    prefix = "\n".join(
        [
            "# successor carrier_text_v1 parity inspection",
            "",
            f"- execution_root: `{execution_root_relative}`",
            f"- execution_sha: `{execution_sha}`",
            f"- authority_level: `{SUCCESSOR_AUTHORITY_LEVEL}`",
            f"- successor_mode: `{SUCCESSOR_MODE}`",
            f"- freshness.timestamp: `{freshness['timestamp']}`",
            "",
            "## inspection body",
            "",
        ]
    )
    _write_text(markdown_path, prefix + original)


def _source_labels_authority_ref(
    *,
    repo_root: Path,
    labels_manifest_path: Path,
) -> dict[str, object]:
    return apple_recap_execution_contract.build_read_only_authority_ref(
        repo_root=repo_root,
        artifact_id="authoritative_export_fix_labels_manifest",
        authority_role="successor_authoritative_export_fix_labels",
        relative_path=labels_manifest_path,
        reject_noncanonical_parts=True,
    )


def materialize_successor_authority(
    *,
    repo_root: Path = REPO_ROOT,
    successor_root: Path | str = SUCCESSOR_EXECUTION_ROOT,
    blocked_root: Path | str = BLOCKED_EXECUTION_ROOT,
    source_dataset_dir: Path | str = DEFAULT_SOURCE_DATASET_DIR,
    critic_dir: Path | str = DEFAULT_SOURCE_CRITIC_DIR,
    baseline_suite_dir: Path | str = DEFAULT_BASELINE_SUITE_DIR,
    experiment_matrix_json: Path | str = DEFAULT_EXPERIMENT_MATRIX_JSON,
    freeze_timestamp: str | None = None,
    epsilon_quantile: float = 0.7,
) -> dict[str, Any]:
    resolved_successor_root = _ensure_successor_root(
        repo_root=repo_root,
        successor_root=Path(successor_root),
        blocked_root=Path(blocked_root),
    )
    resolved_blocked_root = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=blocked_root,
        field_name="blocked_root",
        canonical_roots=BLOCKED_ROOT_CANONICAL_ROOTS,
    )
    resolved_source_dataset_dir = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=source_dataset_dir,
        field_name="source_dataset_dir",
        canonical_roots=SOURCE_DATASET_CANONICAL_ROOTS,
    )
    resolved_critic_dir = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=critic_dir,
        field_name="critic_dir",
        canonical_roots=CRITIC_DIR_CANONICAL_ROOTS,
    )
    resolved_experiment_matrix_json = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=experiment_matrix_json,
        field_name="experiment_matrix_json",
        canonical_roots=EXPERIMENT_MATRIX_CANONICAL_ROOTS,
    )
    resolved_baseline_suite_dir = _resolve_authoritative_path(
        repo_root=repo_root,
        raw=baseline_suite_dir,
        field_name="baseline_suite_dir",
        canonical_roots=(
            f"{_repo_relative_path(repo_root, resolved_blocked_root / 'baseline_suite')}/",
        ),
    )
    blocked_hash_before = snapshot_authority_hashes(
        repo_root=repo_root,
        execution_root=resolved_blocked_root,
        relative_paths=BLOCKED_AUTHORITY_RELATIVE_PATHS,
    )

    draft_path = (
        resolved_successor_root
        / apple_recap_execution_contract.EXECUTION_CONTRACT_JSON_NAME
    )
    final_contract_path = (
        resolved_successor_root
        / apple_recap_execution_contract.FINAL_EXECUTION_CONTRACT_JSON_NAME
    )
    for path in (draft_path, final_contract_path):
        _unlink_if_exists(path)
    apple_recap_execution_contract.materialize_execution_freeze_contract_draft(
        output_dir=resolved_successor_root,
        repo_root=repo_root,
    )
    freeze_contract = apple_recap_execution_contract.materialize_final_execution_freeze(
        output_dir=resolved_successor_root,
        repo_root=repo_root,
        freeze_timestamp=freeze_timestamp,
    )
    execution_sha = str(freeze_contract["execution_sha"])
    if execution_sha == BLOCKED_EXECUTION_SHA:
        raise ValueError(
            "successor execution SHA must differ from the blocked execution SHA"
        )
    execution_root_relative = _repo_relative_path(repo_root, resolved_successor_root)
    blocked_root_relative = _repo_relative_path(repo_root, resolved_blocked_root)
    successor_context = _successor_context(
        blocked_root_relative=blocked_root_relative,
        blocked_execution_sha=BLOCKED_EXECUTION_SHA,
        execution_root_relative=execution_root_relative,
    )
    augmented_freeze_contract = _augment_successor_payload(
        freeze_contract,
        execution_root_relative=execution_root_relative,
        execution_sha=execution_sha,
        freshness=cast(Mapping[str, Any], freeze_contract["freshness"]),
        successor_context=successor_context,
    )
    _write_json(final_contract_path, augmented_freeze_contract)
    _assert_successor_authority_artifact(
        augmented_freeze_contract,
        artifact_relative_path=apple_recap_execution_contract.FINAL_EXECUTION_CONTRACT_JSON_NAME,
        expected_execution_root=execution_root_relative,
        expected_execution_sha=execution_sha,
    )
    integrity = apple_recap_execution_contract.check_execution_freeze_integrity(
        final_contract_json=final_contract_path,
        repo_root=repo_root,
    )
    if integrity["formal_eligibility"] != "ALLOW":
        first_issue = integrity["issues"][0] if integrity["issues"] else None
        if isinstance(first_issue, Mapping):
            raise ValueError(
                str(first_issue.get("message", "successor freeze integrity failed"))
            )
        raise ValueError("successor freeze integrity failed")

    freshness = cast(dict[str, Any], augmented_freeze_contract["freshness"])
    baseline_refs_path = resolved_successor_root / "baseline_refs_manifest.json"
    _unlink_if_exists(baseline_refs_path)
    baseline_refs_payload = build_readonly_refs.materialize_baseline_refs_manifest(
        output=baseline_refs_path,
        repo_root=repo_root,
        execution_sha=execution_sha,
    )
    baseline_refs_payload = _augment_successor_payload(
        baseline_refs_payload,
        execution_root_relative=execution_root_relative,
        execution_sha=execution_sha,
        freshness=freshness,
        successor_context=successor_context,
    )
    _write_json(baseline_refs_path, baseline_refs_payload)
    _assert_successor_authority_artifact(
        baseline_refs_payload,
        artifact_relative_path="baseline_refs_manifest.json",
        expected_execution_root=execution_root_relative,
        expected_execution_sha=execution_sha,
    )

    for name in (
        build_uplift_schemas.FROZEN_MATRIX_JSON_NAME,
        build_uplift_schemas.LEDGER_CSV_NAME,
        build_uplift_schemas.UPLIFT_VERDICT_SCHEMA_JSON_NAME,
    ):
        _unlink_if_exists(resolved_successor_root / name)
    uplift_outputs = build_uplift_schemas.materialize_uplift_schemas(
        output_dir=resolved_successor_root,
        experiment_matrix_json=resolved_experiment_matrix_json,
        repo_root=repo_root,
        execution_sha=execution_sha,
    )
    experiment_matrix_frozen_path = Path(
        str(uplift_outputs["experiment_matrix_frozen"])
    )
    experiment_matrix_payload = _augment_successor_payload(
        _read_json(experiment_matrix_frozen_path),
        execution_root_relative=execution_root_relative,
        execution_sha=execution_sha,
        freshness=freshness,
        successor_context=successor_context,
    )
    _write_json(experiment_matrix_frozen_path, experiment_matrix_payload)
    _assert_successor_authority_artifact(
        experiment_matrix_payload,
        artifact_relative_path="experiment_matrix_frozen.json",
        expected_execution_root=execution_root_relative,
        expected_execution_sha=execution_sha,
    )

    b0_repro_band_payload = build_successor_b0_repro_band(
        repo_root=repo_root,
        baseline_suite_dir=resolved_baseline_suite_dir,
        baseline_refs_manifest_path=baseline_refs_path,
        experiment_matrix_frozen_path=experiment_matrix_frozen_path,
        execution_root_relative=execution_root_relative,
        execution_sha=execution_sha,
        freshness=freshness,
        successor_context=successor_context,
    )
    b0_repro_band_path = resolved_successor_root / "B0_repro_band.json"
    _write_json(b0_repro_band_path, b0_repro_band_payload)
    _assert_successor_authority_artifact(
        b0_repro_band_payload,
        artifact_relative_path="B0_repro_band.json",
        expected_execution_root=execution_root_relative,
        expected_execution_sha=execution_sha,
    )

    official_bundle = _read_json(
        _baseline_bundle_paths(resolved_baseline_suite_dir)["official_comparable_10ep"]
    )
    extended_bundle = _read_json(
        _baseline_bundle_paths(resolved_baseline_suite_dir)["extended_50ep"]
    )
    ledger_rows = _build_successor_ledger_rows(
        execution_root_relative=execution_root_relative,
        execution_sha=execution_sha,
        b0_formal_success_count=int(official_bundle["success_count"]),
        b0_formal_success_rate=float(official_bundle["success_rate"]),
        b0_long_success_rate=float(extended_bundle["success_rate"]),
    )
    ledger_path = resolved_successor_root / "B0_E1_E2_run_ledger.csv"
    _write_csv(ledger_path, ledger_rows)

    labels_surface = materialize_authoritative_export_fix_labels(
        repo_root=repo_root,
        source_dataset_dir=resolved_source_dataset_dir,
        successor_root=resolved_successor_root,
        execution_root_relative=execution_root_relative,
        execution_sha=execution_sha,
        freshness=freshness,
        successor_context=successor_context,
        critic_dir=resolved_critic_dir,
        epsilon_quantile=float(epsilon_quantile),
    )
    labels_manifest_path = Path(str(labels_surface["manifest_path"]))

    report = inspect_mainline_carrier.run_inspection(
        labels_path=Path(str(labels_surface["labels_path"])),
        output_dir=resolved_successor_root,
        fail_on_authority_violation=True,
    )
    carrier_report_path = (
        resolved_successor_root / inspect_mainline_carrier.PARITY_REPORT_JSON_NAME
    )
    carrier_report_payload = _augment_successor_payload(
        report,
        execution_root_relative=execution_root_relative,
        execution_sha=execution_sha,
        freshness=freshness,
        successor_context=successor_context,
    )
    carrier_report_payload["source_artifacts"] = [
        apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id="successor_execution_freeze_contract",
            authority_role="successor_execution_freeze_authority",
            relative_path=final_contract_path,
        ),
        _source_labels_authority_ref(
            repo_root=repo_root,
            labels_manifest_path=labels_manifest_path,
        ),
    ]
    _write_json(carrier_report_path, carrier_report_payload)
    _assert_successor_authority_artifact(
        carrier_report_payload,
        artifact_relative_path=inspect_mainline_carrier.PARITY_REPORT_JSON_NAME,
        expected_execution_root=execution_root_relative,
        expected_execution_sha=execution_sha,
    )

    _augment_carrier_sample_rows_csv(
        csv_path=resolved_successor_root
        / inspect_mainline_carrier.SAMPLE_ROWS_CSV_NAME,
        execution_root_relative=execution_root_relative,
        execution_sha=execution_sha,
    )
    _augment_carrier_markdown(
        markdown_path=resolved_successor_root
        / inspect_mainline_carrier.INSPECTION_MD_NAME,
        execution_root_relative=execution_root_relative,
        execution_sha=execution_sha,
        freshness=freshness,
    )

    blocked_hash_after = snapshot_authority_hashes(
        repo_root=repo_root,
        execution_root=resolved_blocked_root,
        relative_paths=BLOCKED_AUTHORITY_RELATIVE_PATHS,
    )
    assert_hashes_unchanged(blocked_hash_before, blocked_hash_after)

    return {
        "successor_root": str(resolved_successor_root),
        "execution_sha": execution_sha,
        "freeze_contract_path": str(final_contract_path),
        "baseline_refs_manifest_path": str(baseline_refs_path),
        "experiment_matrix_frozen_path": str(experiment_matrix_frozen_path),
        "run_ledger_path": str(ledger_path),
        "b0_repro_band_path": str(b0_repro_band_path),
        "carrier_parity_report_path": str(carrier_report_path),
        "carrier_sample_rows_path": str(
            resolved_successor_root / inspect_mainline_carrier.SAMPLE_ROWS_CSV_NAME
        ),
        "carrier_inspection_markdown_path": str(
            resolved_successor_root / inspect_mainline_carrier.INSPECTION_MD_NAME
        ),
        "labels_manifest_path": str(labels_manifest_path),
        "blocked_root_hash_before": dict(blocked_hash_before),
        "blocked_root_hash_after": dict(blocked_hash_after),
        "successor_context": successor_context,
        "integrity": integrity,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = materialize_successor_authority(
            repo_root=REPO_ROOT,
            successor_root=args.successor_root,
            blocked_root=args.blocked_root,
            source_dataset_dir=args.source_dataset_dir,
            critic_dir=args.critic_dir,
            baseline_suite_dir=args.baseline_suite_dir,
            experiment_matrix_json=args.experiment_matrix_json,
            freeze_timestamp=args.freeze_timestamp,
            epsilon_quantile=float(args.epsilon_quantile),
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BLOCKED_AUTHORITY_RELATIVE_PATHS",
    "BLOCKED_EXECUTION_ROOT",
    "BLOCKED_EXECUTION_SHA",
    "SUCCESSOR_AUTHORITY_LEVEL",
    "SUCCESSOR_EXECUTION_ROOT",
    "SUCCESSOR_MODE",
    "SUCCESSOR_AUTHORITY_RELATIVE_PATHS",
    "assert_hashes_unchanged",
    "build_successor_b0_repro_band",
    "materialize_authoritative_export_fix_labels",
    "materialize_successor_authority",
    "snapshot_authority_hashes",
    "validate_successor_authority_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
