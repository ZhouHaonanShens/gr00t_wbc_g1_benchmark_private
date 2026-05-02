#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast


JsonObject = dict[str, object]


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

DEFAULT_TELEOP_G1_ROOT_REL = (
    "agent/artifacts/public_datasets/PhysicalAI-Robotics-GR00T-Teleop-G1"
)
DEFAULT_XEMB_ROOT_REL = (
    "agent/artifacts/public_datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
)
DEFAULT_OUTPUT_JSON_REL = "agent/artifacts/vlm_critic_manifests/public_warmstart.json"
APPROVED_XEMB_SUBSET = "unitree_g1.LMPnPAppleToPlateDC"


PASS_SENTINEL = "PUBLIC_WARMSTART_MANIFEST_OK"
FAIL_SENTINEL = "PUBLIC_WARMSTART_MANIFEST_FAIL"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(repo_root: Path, raw_path: str | None, *, default_rel: str) -> Path:
    value = str(raw_path or default_rel)
    p = Path(value)
    return p if p.is_absolute() else (repo_root / p)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        _ = f.write("\n")
    _ = tmp_path.replace(path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _path_snapshot(path: Path) -> JsonObject:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "is_dir": bool(path.is_dir()) if exists else False,
        "is_file": bool(path.is_file()) if exists else False,
    }


def _approved_subset_root(base_root: Path, approved_subset: str) -> Path:
    if base_root.name == approved_subset:
        return base_root
    return base_root / approved_subset


def _availability_status(*, exists: bool, is_dir: bool) -> str:
    if not exists:
        return "not_downloaded"
    if not is_dir:
        return "path_exists_not_directory"
    return "local_root_present_unverified"


def _xemb_availability_status(
    *, root_exists: bool, root_is_dir: bool, subset_exists: bool, subset_is_dir: bool
) -> str:
    if not root_exists:
        return "not_downloaded"
    if not root_is_dir:
        return "path_exists_not_directory"
    if not subset_exists:
        return "approved_subset_not_downloaded"
    if not subset_is_dir:
        return "approved_subset_exists_not_directory"
    return "approved_subset_present_unverified"


def _teleop_source(teleop_root: Path) -> JsonObject:
    root = _path_snapshot(teleop_root)
    return {
        "source_name": "teleop_g1",
        "dataset_id": "nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1",
        "dataset_card_url": "https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1",
        "dataset_card_readme_url": "https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1/raw/main/README.md",
        "local_root": cast(str, root["path"]),
        "local_root_exists": cast(bool, root["exists"]),
        "local_root_is_dir": cast(bool, root["is_dir"]),
        "availability_status": _availability_status(
            exists=cast(bool, root["exists"]),
            is_dir=cast(bool, root["is_dir"]),
        ),
        "license": {
            "dataset_license_id": "cc-by-4.0",
            "dataset_license_name": "CC BY 4.0",
            "license_scope_note": "Dataset license only; do not conflate with NVIDIA model license terms.",
        },
        "usage_scope": "warmstart_only",
        "formal_gate_excluded": True,
        "intended_use": "public initialization / representation bootstrap for G1 critic warm-start only",
        "task_text_field": None,
        "language_field": None,
        "task_text_notes": "The public dataset card lists language instructions but does not name a serialized text field.",
        "task_text_examples": [
            "Pick the apple from the table and place it into the basket.",
            "Pick the pear from the table and place it into the basket.",
            "Pick the grapes from the table and place them into the basket.",
            "Pick the starfruit from the table and place it into the basket.",
        ],
        "state_dim": 43,
        "action_dim": 43,
        "state_action_dims_known": True,
        "state_action_dims_note": "Verified from the public README and Hugging Face dataset viewer summary.",
        "video_availability": "documented_remote_card",
        "video_modality_notes": "RGB video, 640x480, 20fps; dataset card also lists MP4 and HDF5 files.",
        "trajectory_count": 1000,
        "source_specific_filter_scope_note": "Approved scope is the Teleop-G1 public dataset only; no extra subset filter is applied inside this dataset for Task 3.",
        "filter_scope": {
            "approved_scope": "entire_dataset",
            "selected_subset_count": 1,
            "selected_subset_names": ["teleop_g1_full_public_dataset"],
            "excluded_dataset_families": [
                "DROID",
                "BridgeData",
                "OXE",
                "RoboCasa",
                "CALVIN",
                "archive_sources",
            ],
        },
        "provenance_refs": [
            "https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1",
            "https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1/raw/main/README.md",
        ],
    }


def _xemb_source(xemb_root: Path) -> JsonObject:
    root = _path_snapshot(xemb_root)
    approved_subset_root = _approved_subset_root(xemb_root, APPROVED_XEMB_SUBSET)
    subset = _path_snapshot(approved_subset_root)
    return {
        "source_name": "xemb_apple_to_plate",
        "dataset_id": "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
        "dataset_card_url": "https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
        "dataset_card_readme_url": "https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/raw/main/README.md",
        "local_root": cast(str, root["path"]),
        "local_root_exists": cast(bool, root["exists"]),
        "local_root_is_dir": cast(bool, root["is_dir"]),
        "approved_subset": APPROVED_XEMB_SUBSET,
        "approved_subset_local_root": cast(str, subset["path"]),
        "approved_subset_local_root_exists": cast(bool, subset["exists"]),
        "approved_subset_local_root_is_dir": cast(bool, subset["is_dir"]),
        "availability_status": _xemb_availability_status(
            root_exists=cast(bool, root["exists"]),
            root_is_dir=cast(bool, root["is_dir"]),
            subset_exists=cast(bool, subset["exists"]),
            subset_is_dir=cast(bool, subset["is_dir"]),
        ),
        "license": {
            "dataset_license_id": "cc-by-4.0",
            "dataset_license_name": "CC BY 4.0",
            "license_scope_note": "Dataset license only; do not conflate with NVIDIA model license terms.",
        },
        "usage_scope": "warmstart_only",
        "formal_gate_excluded": True,
        "intended_use": "public initialization / representation bootstrap only; approved scope is the Unitree G1 AppleToPlate public subset",
        "task_text_field": None,
        "language_field": None,
        "task_text_notes": "Task 3 only verifies the approved subset identity from public docs; no serialized language/task-text field is claimed without a local subset root.",
        "state_dim": None,
        "action_dim": None,
        "state_action_dims_known": False,
        "state_action_dims_note": "The public X-Embodiment-Sim card confirms the approved subset and trajectory count, but does not publish per-subset state/action dimensions on the card text used in Task 3.",
        "video_availability": "unknown_without_local_subset",
        "video_modality_notes": "The public card describes a trajectory dataset for GR00T post-training but does not enumerate per-subset video/schema details in the card text consumed here; local data is not present for verification.",
        "trajectory_count": 102,
        "source_specific_filter_scope_note": "Only `unitree_g1.LMPnPAppleToPlateDC` is approved for Task 3; all other X-Embodiment-Sim subsets remain excluded.",
        "filter_scope": {
            "approved_scope": "single_subset_only",
            "selected_subset_count": 1,
            "selected_subset_names": [APPROVED_XEMB_SUBSET],
            "selected_subset_trajectory_count": 102,
            "excluded_other_xemb_subsets": True,
            "excluded_dataset_families": [
                "DROID",
                "BridgeData",
                "OXE",
                "RoboCasa",
                "CALVIN",
                "archive_sources",
            ],
        },
        "provenance_refs": [
            "https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
            "https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/raw/main/README.md",
            "agent/tutorials/10_pnp_apple_to_plate_training_space.md",
        ],
    }


def _build_manifest(repo_root: Path, teleop_root: Path, xemb_root: Path) -> JsonObject:
    sources = [_teleop_source(teleop_root), _xemb_source(xemb_root)]
    return {
        "schema_version": "vlm_critic_public_warmstart_manifest_v1",
        "builder_entrypoint": "work/recap/scripts/41c_vlm_critic_public_warmstart_manifest.py",
        "generated_at_utc": _utc_now_iso(),
        "repo_root": str(repo_root),
        "usage_scope": "warmstart_only",
        "formal_gate_excluded": True,
        "formal_gate_exclusion_reason": "public_data_formal_gate_forbidden: public warm-start data is initialization-only and excluded from the formal Isaac held-out gate.",
        "formal_gate_dataset_scope": "isaac_only",
        "contract_ref": "agent/artifacts/vlm_critic_manifests/critic_contract_v1.json",
        "contract_scope_note": "Matches Task 1 contract: public warm-start scope is initialization_only while formal gate dataset scope remains isaac_only.",
        "license_scope_note": "This manifest records dataset licenses/provenance only; it does not restate or merge NVIDIA model license terms.",
        "manifest_stability_note": "Task 3 is allowed to pass without downloaded local roots; absence must be recorded honestly via local_root_exists=false and availability_status=not_downloaded or equivalent subset-missing statuses.",
        "approved_public_sources": ["teleop_g1", "xemb_apple_to_plate"],
        "approved_public_source_count": 2,
        "excluded_source_families": [
            "DROID",
            "BridgeData",
            "OXE",
            "RoboCasa",
            "CALVIN",
            "archive_sources",
        ],
        "sources": sources,
    }


def _validate_manifest(manifest: JsonObject) -> list[str]:
    violations: list[str] = []

    if manifest.get("usage_scope") != "warmstart_only":
        violations.append("usage_scope must be 'warmstart_only'")
    if manifest.get("formal_gate_excluded") is not True:
        violations.append("formal_gate_excluded must be true")
    approved_sources = manifest.get("approved_public_sources")
    if approved_sources != ["teleop_g1", "xemb_apple_to_plate"]:
        violations.append(
            f"approved_public_sources must be ['teleop_g1', 'xemb_apple_to_plate'], got {approved_sources!r}"
        )
    sources_raw = manifest.get("sources")
    if not isinstance(sources_raw, list):
        violations.append("sources must be a list")
        return violations

    if len(sources_raw) != 2:
        violations.append(
            f"sources must contain exactly 2 entries, got {len(sources_raw)}"
        )
        return violations

    expected_names = ["teleop_g1", "xemb_apple_to_plate"]
    actual_names = [
        item.get("source_name") if isinstance(item, dict) else None
        for item in sources_raw
    ]
    if actual_names != expected_names:
        violations.append(
            f"sources must be ordered as {expected_names!r}, got {actual_names!r}"
        )

    required_keys = {
        "dataset_id",
        "local_root",
        "local_root_exists",
        "availability_status",
        "license",
        "intended_use",
        "video_modality_notes",
        "source_specific_filter_scope_note",
    }
    for idx, item in enumerate(sources_raw):
        if not isinstance(item, dict):
            violations.append(f"sources[{idx}] must be an object")
            continue
        for key in sorted(required_keys):
            if key not in item:
                violations.append(f"sources[{idx}] missing required key: {key}")
        if item.get("formal_gate_excluded") is not True:
            violations.append(f"sources[{idx}] formal_gate_excluded must be true")
        if item.get("usage_scope") != "warmstart_only":
            violations.append(f"sources[{idx}] usage_scope must be 'warmstart_only'")

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="41c_vlm_critic_public_warmstart_manifest.py",
        description="Build the Task 3 public warm-start manifest for approved G1 public sources only.",
    )
    _ = parser.add_argument(
        "--teleop-g1-root", type=str, default=DEFAULT_TELEOP_G1_ROOT_REL
    )
    _ = parser.add_argument("--xemb-root", type=str, default=DEFAULT_XEMB_ROOT_REL)
    _ = parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON_REL)
    _ = parser.add_argument(
        "--formal-gate-include-public",
        action="store_true",
        help="Forbidden blocker flag for negative QA: public warm-start data cannot enter formal gate.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    output_json = _resolve_path(
        repo_root, args.output_json, default_rel=DEFAULT_OUTPUT_JSON_REL
    )

    try:
        if bool(args.formal_gate_include_public):
            raise RuntimeError(
                "public_data_formal_gate_forbidden: public warm-start sources are initialization-only and cannot be included in the formal Isaac gate"
            )

        teleop_root = _resolve_path(
            repo_root, args.teleop_g1_root, default_rel=DEFAULT_TELEOP_G1_ROOT_REL
        )
        xemb_root = _resolve_path(
            repo_root, args.xemb_root, default_rel=DEFAULT_XEMB_ROOT_REL
        )
        manifest = _build_manifest(repo_root, teleop_root, xemb_root)
        violations = _validate_manifest(manifest)
        result: JsonObject = {
            "manifest": manifest,
            "pass": len(violations) == 0,
            "sentinel": PASS_SENTINEL if not violations else FAIL_SENTINEL,
            "violations": violations,
        }
        if violations:
            raise RuntimeError("; ".join(violations))
        _write_json(output_json, result)
        print(f"[INFO] wrote_json: {output_json}")
        print(f"SENTINEL:{PASS_SENTINEL}")
        return 0
    except Exception as exc:
        error_payload: JsonObject = {
            "error": f"{type(exc).__name__}: {exc}",
            "pass": False,
            "sentinel": FAIL_SENTINEL,
        }
        _write_json(output_json, error_payload)
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"[INFO] wrote_json: {output_json}", file=sys.stderr)
        print(f"SENTINEL:{FAIL_SENTINEL}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
