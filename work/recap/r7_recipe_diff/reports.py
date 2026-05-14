from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from work.recap.r7_recipe_diff.contract import RecipeDelta

SCHEMA_VERSION = "r7_training_recipe_diff_v1"
TOP_LEVEL_KEYS = ("schema_version", "base_cell", "generated_at_utc", "source_openpi_report", "deltas", "composable_cli_args")
DELTA_KEYS = ("component", "current_state", "paper_prescribed_state", "diff_action", "cli_arg_addition", "config_path_diff", "evidence_files", "rationale")
COMPONENT_KEYS = ("component_id", "title", "paper_section_cite", "training_arg_name")


def composable_cli_args(deltas: Sequence[RecipeDelta]) -> str:
    return " ".join(arg for delta in deltas for arg in delta.cli_arg_addition)


def _component_payload(delta: RecipeDelta) -> dict[str, str]:
    component = delta.component
    return {"component_id": component.component_id, "title": component.title, "paper_section_cite": component.paper_section_cite, "training_arg_name": component.training_arg_name}


def _delta_payload(delta: RecipeDelta) -> dict[str, object]:
    return {
        "component": _component_payload(delta),
        "current_state": delta.current_state,
        "paper_prescribed_state": delta.paper_prescribed_state,
        "diff_action": delta.diff_action,
        "cli_arg_addition": list(delta.cli_arg_addition),
        "config_path_diff": [[path, value] for path, value in delta.config_path_diff],
        "evidence_files": list(delta.evidence_files),
        "rationale": delta.rationale,
    }


def recipe_diff_payload(*, base_cell: str, generated_at_utc: str, source_openpi_report: str, deltas: Sequence[RecipeDelta]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "base_cell": str(base_cell),
        "generated_at_utc": str(generated_at_utc),
        "source_openpi_report": str(source_openpi_report),
        "deltas": [_delta_payload(delta) for delta in deltas],
        "composable_cli_args": composable_cli_args(deltas),
    }


def write_recipe_diff_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _summary_row(item: dict[str, object]) -> str:
    component = item["component"]
    assert isinstance(component, dict)
    cli_args = " ".join(str(arg) for arg in item["cli_arg_addition"])
    return f"| {component['component_id']} | {item['current_state']} | {item['paper_prescribed_state']} | {item['diff_action']} | `{cli_args}` |"


def _detail_lines(item: dict[str, object]) -> list[str]:
    component = item["component"]
    assert isinstance(component, dict)
    cli_args = " ".join(str(arg) for arg in item["cli_arg_addition"])
    return [
        f"### {component['component_id']} — {component['title']}",
        f"- cite: `{component['paper_section_cite']}`",
        f"- training_arg_name: `{component['training_arg_name']}`",
        f"- state: `{item['current_state']}` → `{item['paper_prescribed_state']}`",
        f"- diff_action: `{item['diff_action']}`",
        f"- required future args: `{cli_args}`",
        f"- config_path_diff: `{item['config_path_diff']}`",
        f"- evidence_files: `{item['evidence_files']}`",
        f"- rationale: {item['rationale']}",
        "",
    ]


def render_recipe_diff_markdown(payload: dict[str, object]) -> str:
    deltas = payload["deltas"]
    assert isinstance(deltas, list)
    lines = [
        "# R7_RECIPE_DIFF_REPORT", "", f"- schema_version: `{payload['schema_version']}`",
        f"- base_cell: `{payload['base_cell']}`", f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- source_openpi_report: `{payload['source_openpi_report']}`", "", "## Fidelity component summary", "",
        "| component_id | current | paper-prescribed | diff_action | cli_addition |", "|---|---|---|---|---|",
    ]
    lines.extend(_summary_row(item) for item in deltas if isinstance(item, dict))
    lines.extend(["", "## Per-component rationale", ""])
    for item in deltas:
        assert isinstance(item, dict)
        lines.extend(_detail_lines(item))
    lines.extend(["## Composable training recipe", "", "These are required future args for R7.1; R7.0 does not execute them and the current launcher may not accept them yet.", "", f"```bash\n{payload['composable_cli_args']}\n```", ""])
    return "\n".join(lines)
