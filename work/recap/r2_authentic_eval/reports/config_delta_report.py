"""Markdown renderer for the R2.0.5 Config-Delta Audit subsection."""
from __future__ import annotations

from typing import Any


def render_config_delta_subsection(
    inventory: dict[str, Any],
    *,
    anchor: str = "config-delta-audit",
) -> str:
    """Render the closure-report Config-Delta Audit subsection."""
    rows = list(inventory.get("rows", ()))
    summary = dict(inventory.get("summary", {}))
    allowed_paths = ", ".join(f"`{p}`" for p in inventory.get("allowed_paths", ()))
    out = [
        f"## Config-Delta Audit {{#{anchor}}}",
        "",
        f"- records_audited: {inventory.get('row_count', len(rows))}",
        f"- ONLY_FORMALIZE_LANGUAGE: {summary.get('ONLY_FORMALIZE_LANGUAGE', 0)}",
        f"- ADDITIONAL_FIELDS_DIFFER: {summary.get('ADDITIONAL_FIELDS_DIFFER', 0)}",
        f"- architectures_mismatch_count: {summary.get('architectures_mismatch_count', 0)}",
        "",
        "| ckpt | classification | allowed_paths | architectures | outside_paths | acknowledgment |",
        "|---|---|---|---|---|---|",
    ]
    attention = inventory.get("attention") or {}
    ack_status = str(attention.get("status", "not_required"))
    for row in rows:
        ckpt = str(row.get("ckpt_root", ""))
        classification = str(row.get("classification", ""))
        arch = "mismatch" if row.get("architectures_mismatch") else "match"
        outside_paths = ", ".join(f"`{p}`" for p in row.get("outside_paths", ()))
        out.append(
            f"| `{ckpt}` | `{classification}` | {allowed_paths or '-'} | {arch} | "
            f"{outside_paths or '-'} | {ack_status} |"
        )
    out.append("")
    if attention:
        out.extend(
            [
                f"- attention_md: `{attention.get('attention_md', '')}`",
                f"- user_attention_md: `{attention.get('user_attention_md', '')}`",
                f"- acknowledgment_md: `{attention.get('acknowledgment_md', '')}`",
                "",
            ]
        )
    return "\n".join(out)


__all__ = ["render_config_delta_subsection"]
