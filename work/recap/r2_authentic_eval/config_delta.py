"""R2.0.5 config-delta classification and acknowledgment gate."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Literal, Sequence

from work.recap.r1_repro.protocol import RAW_HF_SNAPSHOT_ROOT

ConfigDeltaClassification = Literal[
    "ONLY_FORMALIZE_LANGUAGE",
    "ADDITIONAL_FIELDS_DIFFER",
]

ONLY_FORMALIZE_LANGUAGE: ConfigDeltaClassification = "ONLY_FORMALIZE_LANGUAGE"
ADDITIONAL_FIELDS_DIFFER: ConfigDeltaClassification = "ADDITIONAL_FIELDS_DIFFER"

CONFIG_FILES: tuple[str, ...] = ("config.json", "processor_config.json")
FORMALIZE_LANGUAGE_PATHS: frozenset[str] = frozenset(
    {
        "config.json:formalize_language",
        "processor_config.json:formalize_language",
        "processor_config.json:processor_kwargs.formalize_language",
    }
)
ATTENTION_FILENAME = "whitelist_audit.attention.md"
USER_ATTENTION_FILENAME = "r2_0_5_user_attention.md"
ACKNOWLEDGMENT_FILENAME = "whitelist_audit.acknowledged.md"
INVENTORY_FILENAME = "config_delta_inventory.json"

_MISSING = object()


class ConfigDeltaError(RuntimeError):
    """Base class for config-delta failures."""


class AcknowledgmentMissingError(ConfigDeltaError):
    """Raised when an attention file has no valid human acknowledgment."""


def classify_config_delta(
    source_ckpt: Path,
    target_ckpt: Path,
    *,
    allowed_paths: set[str],
) -> ConfigDeltaClassification:
    """Classify the config delta between source and target checkpoint roots."""
    outside_paths = _outside_paths_for_roots(
        Path(source_ckpt),
        Path(target_ckpt),
        allowed_paths=allowed_paths,
    )
    return ONLY_FORMALIZE_LANGUAGE if not outside_paths else ADDITIONAL_FIELDS_DIFFER


def audit_one_ckpt(
    ckpt_root: Path,
    *,
    allowed_paths: set[str],
    target_ckpt: Path | None = None,
) -> dict[str, Any]:
    """Return one config-delta audit row for a checkpoint root."""
    source_root = _as_ckpt_root(ckpt_root)
    target_root = Path(target_ckpt) if target_ckpt is not None else RAW_HF_SNAPSHOT_ROOT
    deltas = _diff_roots(source_root, target_root, allowed_paths=allowed_paths)
    outside_paths = [d["path"] for d in deltas if not d["allowed"]]
    source_arch = _architectures(source_root / "config.json")
    target_arch = _architectures(target_root / "config.json")
    architectures_mismatch = source_arch != target_arch
    classification: ConfigDeltaClassification = (
        ONLY_FORMALIZE_LANGUAGE if not outside_paths else ADDITIONAL_FIELDS_DIFFER
    )
    return {
        "ckpt_root": str(source_root),
        "target_ckpt": str(target_root),
        "classification": classification,
        "outside_paths": outside_paths,
        "architectures_source": list(source_arch),
        "architectures_target": list(target_arch),
        "architectures_mismatch": architectures_mismatch,
        "deltas": deltas,
    }


def audit_inventory(
    ckpt_roots: Sequence[Path],
    *,
    allowed_paths: set[str],
    target_ckpt: Path | None = None,
    dossier_dir: Path | None = None,
) -> dict[str, Any]:
    """Aggregate config-delta audit rows and optionally write dossier artifacts."""
    rows = [
        audit_one_ckpt(ckpt_root, allowed_paths=allowed_paths, target_ckpt=target_ckpt)
        for ckpt_root in ckpt_roots
    ]
    additional_rows = [
        row for row in rows if row["classification"] == ADDITIONAL_FIELDS_DIFFER
    ]
    inventory: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": _utc_slug(),
        "allowed_paths": sorted(allowed_paths),
        "row_count": len(rows),
        "summary": {
            ONLY_FORMALIZE_LANGUAGE: sum(
                1 for row in rows if row["classification"] == ONLY_FORMALIZE_LANGUAGE
            ),
            ADDITIONAL_FIELDS_DIFFER: len(additional_rows),
            "architectures_mismatch_count": sum(
                1 for row in rows if row["architectures_mismatch"]
            ),
        },
        "rows": rows,
        "attention": None,
    }
    if dossier_dir is not None:
        out_dir = Path(dossier_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if additional_rows:
            attention_md = out_dir / ATTENTION_FILENAME
            user_attention_md = out_dir / USER_ATTENTION_FILENAME
            attention_text = _render_attention(additional_rows)
            _write_text_if_changed(attention_md, attention_text)
            _write_text_if_changed(user_attention_md, attention_text)
            ack_md = out_dir / ACKNOWLEDGMENT_FILENAME
            inventory["attention"] = {
                "status": "pending",
                "attention_md": str(attention_md),
                "user_attention_md": str(user_attention_md),
                "acknowledgment_md": str(ack_md),
            }
        (out_dir / INVENTORY_FILENAME).write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return inventory


def require_acknowledgment(
    attention_md: Path,
    ack_md: Path,
    *,
    min_margin_s: float = 1.0,
) -> None:
    """Require an ack file whose mtime is strictly newer than attention by margin."""
    attention_path = Path(attention_md)
    ack_path = Path(ack_md)
    if not attention_path.exists():
        return
    if not ack_path.exists():
        raise AcknowledgmentMissingError(f"Missing acknowledgment file: {ack_path}")
    attention_mtime = attention_path.stat().st_mtime
    ack_mtime = ack_path.stat().st_mtime
    if ack_mtime <= attention_mtime + min_margin_s:
        raise AcknowledgmentMissingError(
            "Acknowledgment mtime is not newer than attention by "
            f">{min_margin_s:.3f}s: attention={attention_mtime:.6f} ack={ack_mtime:.6f}"
        )


def _as_ckpt_root(value: Any) -> Path:
    if hasattr(value, "abs_path"):
        return Path(value.abs_path)
    return Path(value)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ConfigDeltaError(f"Expected object JSON at {path}")
    return payload


def _write_text_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def _architectures(config_path: Path) -> tuple[str, ...]:
    raw_value = _load_json(config_path).get("architectures", ())
    if isinstance(raw_value, list):
        return tuple(str(v) for v in raw_value)
    if isinstance(raw_value, tuple):
        return tuple(str(v) for v in raw_value)
    if raw_value:
        return (str(raw_value),)
    return ()


def _diff_roots(
    source_root: Path,
    target_root: Path,
    *,
    allowed_paths: set[str],
) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for filename in CONFIG_FILES:
        source_flat = _flatten(_load_json(source_root / filename))
        target_flat = _flatten(_load_json(target_root / filename))
        for dotted_path in sorted(set(source_flat) | set(target_flat)):
            source_present = dotted_path in source_flat
            target_present = dotted_path in target_flat
            source_value = source_flat.get(dotted_path, _MISSING)
            target_value = target_flat.get(dotted_path, _MISSING)
            if source_value == target_value:
                continue
            path = f"{filename}:{dotted_path}"
            deltas.append(
                {
                    "path": path,
                    "file": filename,
                    "dotted_path": dotted_path,
                    "kind": _delta_kind(source_present, target_present),
                    "source_value": None if source_value is _MISSING else source_value,
                    "target_value": None if target_value is _MISSING else target_value,
                    "allowed": _path_allowed(path, dotted_path, allowed_paths),
                }
            )
    return deltas


def _outside_paths_for_roots(
    source_root: Path,
    target_root: Path,
    *,
    allowed_paths: set[str],
) -> tuple[str, ...]:
    return tuple(
        delta["path"]
        for delta in _diff_roots(source_root, target_root, allowed_paths=allowed_paths)
        if not delta["allowed"]
    )


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value[key], path))
        return out
    return {prefix: value}


def _delta_kind(source_present: bool, target_present: bool) -> str:
    if source_present and target_present:
        return "changed"
    if source_present:
        return "added"
    return "removed"


def _path_allowed(path: str, dotted_path: str, allowed_paths: set[str]) -> bool:
    return path in allowed_paths or dotted_path in allowed_paths


def _render_attention(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# R2.0.5 user attention required",
        "",
        "Config-delta audit found ADDITIONAL_FIELDS_DIFFER rows.",
        "Do not proceed to Phase E without explicit user acknowledgment.",
        "",
    ]
    for row in rows:
        lines.append(f"## {row['ckpt_root']}")
        lines.append("")
        lines.append(f"- classification: {row['classification']}")
        lines.append(f"- architectures_mismatch: {row['architectures_mismatch']}")
        lines.append(f"- architectures_source: {row['architectures_source']}")
        lines.append(f"- architectures_target: {row['architectures_target']}")
        lines.append("- outside_paths:")
        for path in row["outside_paths"]:
            lines.append(f"  - `{path}`")
        lines.append("")
    return "\n".join(lines)


def _utc_slug() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


__all__ = [
    "ACKNOWLEDGMENT_FILENAME",
    "ADDITIONAL_FIELDS_DIFFER",
    "ATTENTION_FILENAME",
    "AcknowledgmentMissingError",
    "CONFIG_FILES",
    "ConfigDeltaClassification",
    "ConfigDeltaError",
    "FORMALIZE_LANGUAGE_PATHS",
    "INVENTORY_FILENAME",
    "ONLY_FORMALIZE_LANGUAGE",
    "USER_ATTENTION_FILENAME",
    "audit_inventory",
    "audit_one_ckpt",
    "classify_config_delta",
    "require_acknowledgment",
]
