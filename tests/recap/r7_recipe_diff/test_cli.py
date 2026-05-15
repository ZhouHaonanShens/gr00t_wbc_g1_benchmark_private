from __future__ import annotations

import json
from pathlib import Path

from work.recap.r7_recipe_diff import cli


def _single_run_dir(root: Path) -> Path:
    runs = sorted(root.glob("*_run"))
    assert len(runs) == 1
    return runs[0]


def test_audit_cli_writes_json_and_report(tmp_path: Path) -> None:
    assert cli.main(["audit", "--base-cell", "A.2", "--output-root", str(tmp_path)]) == 0
    run_dir = _single_run_dir(tmp_path)
    payload = json.loads((run_dir / "training_recipe_diff.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "r7_training_recipe_diff_v1"
    assert payload["base_cell"] == "A.2"
    assert len(payload["deltas"]) == 5
    report = (run_dir / "R7_RECIPE_DIFF_REPORT.md").read_text(encoding="utf-8")
    assert "## Composable training recipe" in report


def test_audit_cli_rejects_a1_and_unknown_cell(tmp_path: Path) -> None:
    assert cli.main(["audit", "--base-cell", "A.1", "--output-root", str(tmp_path)]) == 2
    assert cli.main(["audit", "--base-cell", "A.9", "--output-root", str(tmp_path)]) == 2
    assert list(tmp_path.glob("*_run")) == []


def test_parser_help_mentions_audit() -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "audit" in help_text
    assert "python -m work.recap.r7_recipe_diff" in help_text
