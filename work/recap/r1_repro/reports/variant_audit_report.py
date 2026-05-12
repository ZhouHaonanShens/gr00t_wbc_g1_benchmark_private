from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def render_variant_audit(
    audit: dict[str, Any],
    risk: dict[str, Any],
    symlinks: dict[str, Any],
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"audit": audit, "risk": risk, "symlinks": symlinks}
    _write_json(out_dir / "r1_2_audit_report.json", payload)
    _write_json(out_dir / "symlink_inventory.json", symlinks)

    lines = [
        "# R1.2 Variant Audit Report",
        "",
        f"- verdict: {audit.get('verdict', 'UNKNOWN')}",
        "",
        "## Risk Keys",
        "",
        "| risk | count |",
        "|---|---:|",
    ]
    for level in ("HIGH", "MEDIUM", "LOW"):
        lines.append(f"| {level} | {len(risk.get(level, []))} |")
    lines.extend(["", "## Diff Files", ""])
    for file_name, diffs in audit.items():
        if file_name in {"schema_version", "generated_at_utc", "status", "verdict"}:
            continue
        lines.append(f"- `{file_name}`: {len(diffs) if isinstance(diffs, list) else 'n/a'}")
    lines.extend(["", "## Symlink Inventory", ""])
    lines.append(f"- entries: {len(symlinks) if isinstance(symlinks, dict) else 0}")
    (out_dir / "r1_2_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
