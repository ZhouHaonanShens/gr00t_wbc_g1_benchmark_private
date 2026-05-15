from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from work.recap.r1_repro.protocol import sha256_file


class VariantAuditError(RuntimeError):
    """Base class for variant audit errors."""


class VariantAmbiguous(VariantAuditError):
    """Raised when discovery finds more than one plausible variant directory."""


class VariantNotFound(VariantAuditError):
    """Raised when discovery finds no plausible variant directory."""


MISSING_VALUE = "__R1_REPRO_MISSING__"

DiffTuple = tuple[str, Any, Any]
RiskTuple = tuple[str, str, Any, Any]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _join_path(parent: str, key: str) -> str:
    return key if not parent else f"{parent}.{key}"


def _diff_json(lhs: Any, rhs: Any, prefix: str = "") -> list[DiffTuple]:
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        diffs: list[DiffTuple] = []
        for key in sorted(set(lhs) | set(rhs)):
            key_path = _join_path(prefix, str(key))
            if key not in lhs:
                diffs.append((key_path, MISSING_VALUE, rhs[key]))
            elif key not in rhs:
                diffs.append((key_path, lhs[key], MISSING_VALUE))
            else:
                diffs.extend(_diff_json(lhs[key], rhs[key], key_path))
        return diffs
    if lhs != rhs:
        return [(prefix, lhs, rhs)]
    return []


def audit_variant(
    variant_root: Path, raw_hf_root: Path, files: list[str]
) -> dict[str, list[DiffTuple]]:
    """Return per-file JSON diffs as variant-value vs raw-HF-value tuples."""

    audit: dict[str, list[DiffTuple]] = {}
    for file_name in files:
        variant_payload = _read_json(variant_root / file_name)
        raw_payload = _read_json(raw_hf_root / file_name)
        audit[file_name] = _diff_json(variant_payload, raw_payload)
    return audit


def _value_text(*values: Any) -> str:
    return " ".join(str(value) for value in values).lower()


def _risk_for_entry(key_path: str, lhs: Any, rhs: Any) -> str:
    key = key_path.lower()
    top_level = key.split(".", 1)[0]
    value_text = _value_text(lhs, rhs)

    high_prefixes = (
        "language_model.",
        "processor.tokenizer.",
        "processor.image_processor.",
        "processor.video_processor.",
        "action_head.",
        "action_decoder.",
        "embodiment.",
        "modality_configs.",
    )
    high_exact = {"n_action_steps", "action_dim", "action_horizon"}
    if (
        key in high_exact
        or any(key.startswith(prefix) for prefix in high_prefixes)
        or "formalize_language" in value_text
        or "unconditional" in value_text
    ):
        return "HIGH"

    low_exact = {"_commit_hash", "_name_or_path"}
    if (
        key in low_exact
        or top_level in low_exact
        or "telemetry" in key
        or "comment" in key
    ):
        return "LOW"

    if top_level == "transformers_version":
        return "MEDIUM"
    return "MEDIUM"


def classify_risk(diff: dict[str, list[DiffTuple]]) -> dict[str, list[RiskTuple]]:
    risk: dict[str, list[RiskTuple]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for file_name, entries in diff.items():
        for key_path, lhs, rhs in entries:
            bucket = _risk_for_entry(key_path, lhs, rhs)
            risk[bucket].append((file_name, key_path, lhs, rhs))
    return risk


def inventory_symlinks(variant_root: Path, raw_hf_root: Path) -> dict[str, Any]:
    symlinks: list[dict[str, Any]] = []
    non_symlink_overrides: list[dict[str, Any]] = []

    for path in sorted(variant_root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = str(path.relative_to(variant_root))
        if path.is_symlink():
            target_text = path.readlink()
            resolved_target = path.resolve(strict=True)
            symlinks.append(
                {
                    "name": relative,
                    "target": str(target_text),
                    "resolved_target": str(resolved_target),
                    "target_sha256": (
                        sha256_file(resolved_target)
                        if resolved_target.is_file()
                        else None
                    ),
                    "target_within_raw_hf_root": raw_hf_root.resolve()
                    in resolved_target.resolve().parents
                    or resolved_target.resolve() == raw_hf_root.resolve(),
                }
            )
        elif path.is_file():
            non_symlink_overrides.append(
                {"name": relative, "sha256": sha256_file(path)}
            )

    return {
        "variant_root": str(variant_root),
        "raw_hf_root": str(raw_hf_root),
        "symlinks": symlinks,
        "non_symlink_overrides": non_symlink_overrides,
    }


def confirm_variant_uniqueness(candidate_globs: list[str]) -> Path:
    matches: list[Path] = []
    for pattern in candidate_globs:
        expanded_pattern = str(Path(pattern).expanduser())
        for match in glob.glob(expanded_pattern, recursive=True):
            path = Path(match).expanduser()
            if path.is_dir():
                matches.append(path.resolve())

    unique_matches = sorted(set(matches))
    if not unique_matches:
        raise VariantNotFound("no variant directory matched candidate globs")
    if len(unique_matches) > 1:
        joined = ", ".join(str(path) for path in unique_matches)
        raise VariantAmbiguous(f"multiple variant directories matched: {joined}")
    return unique_matches[0]


__all__ = [
    "MISSING_VALUE",
    "VariantAmbiguous",
    "VariantAuditError",
    "VariantNotFound",
    "audit_variant",
    "classify_risk",
    "confirm_variant_uniqueness",
    "inventory_symlinks",
]
