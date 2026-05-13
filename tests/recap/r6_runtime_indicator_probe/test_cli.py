from __future__ import annotations

import json
from pathlib import Path

from work.recap.r6_runtime_indicator_probe import cli


def test_trace_cell_writes_only_cell_outputs(tmp_path: Path) -> None:
    assert cli.main(["trace", "--cell", "A.2", "--output-root", str(tmp_path)]) == 0
    graph = json.loads((tmp_path / "A.2" / "wiring_graph.json").read_text(encoding="utf-8"))
    assert graph["cell_id"] == "A.2"
    assert graph["static_verdict"] == "WIRED"
    assert (tmp_path / "A.2" / "cell_probe_report.md").is_file()
    assert not (tmp_path / "FIX_R2_A1_LOAD_05_R6_WIRING_REPORT.md").exists()


def test_trace_all_writes_four_cells_summary_and_matrix(tmp_path: Path) -> None:
    assert cli.main(["trace", "--all", "--output-root", str(tmp_path)]) == 0
    for cell in ("A.2", "A.3", "A.4", "A.5"):
        assert (tmp_path / cell / "wiring_graph.json").is_file()
    report = (tmp_path / "FIX_R2_A1_LOAD_05_R6_WIRING_REPORT.md").read_text(encoding="utf-8")
    assert "R6.1_SKIPPED_NO_AMBIGUOUS_STATIC_CELLS" in report
    assert "1960de2cc4497587ed5df3195d0c3d63984fe10449a58feb9f7dbd4f413b3fe4" in report
    matrix = json.loads((tmp_path / "r6_wiring_matrix.json").read_text(encoding="utf-8"))
    assert len(matrix["cells"]) == 4


def test_trace_rejects_a1_and_unknown_cells(tmp_path: Path) -> None:
    assert cli.main(["trace", "--cell", "A.1", "--output-root", str(tmp_path)]) == 2
    assert cli.main(["trace", "--cell", "A.9", "--output-root", str(tmp_path)]) == 2


def test_probe_rejects_missing_token_gpu0_gpu3_and_no_all_flag(tmp_path: Path) -> None:
    parser = cli.build_parser()
    assert cli.main(["probe", "--cell", "A.2", "--leader-approval-token", "x", "--output-root", str(tmp_path)]) == 2
    good = "a" * 64
    assert cli.main(["probe", "--cell", "A.2", "--leader-approval-token", good, "--gpu", "0", "--output-root", str(tmp_path)]) == 2
    assert cli.main(["probe", "--cell", "A.2", "--leader-approval-token", good, "--gpu", "3", "--output-root", str(tmp_path)]) == 2
    assert "--all" not in parser.format_help()
