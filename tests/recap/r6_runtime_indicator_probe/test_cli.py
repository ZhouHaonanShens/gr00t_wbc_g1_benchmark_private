from __future__ import annotations

import json
from pathlib import Path

from work.recap.r6_runtime_indicator_probe import cli
from work.recap.r6_runtime_indicator_probe.contract import ProbeCounterfactual, RuntimeTrace


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
    assert matrix["cells"][0]["final"] == "WIRED_STATIC_UNCONFIRMED_RUNTIME"


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


def test_forced_probe_requires_token_a2_and_gpu1_before_subprocess(tmp_path: Path) -> None:
    good = "a" * 64
    assert cli.main(["probe", "--forced", "--cell", "A.2", "--leader-approval-token", "bad", "--gpu", "1", "--output-root", str(tmp_path)]) == 2
    for cell in ("A.1", "A.3", "A.4", "A.5", "A.9"):
        assert cli.main(["probe", "--forced", "--cell", cell, "--leader-approval-token", good, "--gpu", "1", "--output-root", str(tmp_path)]) == 2
    for gpu in ("0", "2", "3"):
        assert cli.main(["probe", "--forced", "--cell", "A.2", "--leader-approval-token", good, "--gpu", gpu, "--output-root", str(tmp_path)]) == 2


def test_forced_probe_calls_runtime_with_forced_counterfactual_and_writes_allowlist(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run_runtime_probe(cell, budget, token, *, forced=False, counterfactual=True):
        calls.append((cell, budget.gpu_id, token, forced, counterfactual))
        trace = RuntimeTrace("A.2", 20000, "Advantage: positive" * 100, "p" * 64, "a" * 64, (1, 2, 3, 4, 5), True, "INDICATOR_PRESENT")
        cf = ProbeCounterfactual("A.2", 20000, "a" * 64, "b" * 64, False, (0, 0, 0.01, 0, 0), "INDICATOR_SENSITIVE")
        return trace, cf

    monkeypatch.setattr("work.recap.r6_runtime_indicator_probe.runtime_probe.run_runtime_probe", fake_run_runtime_probe)
    good = "f" * 64
    assert cli.main(["probe", "--forced", "--cell", "A.2", "--leader-approval-token", good, "--gpu", "1", "--output-root", str(tmp_path)]) == 0
    assert calls == [("A.2", 1, good, True, True)]
    files = sorted(p.name for p in (tmp_path / "A.2").iterdir())
    assert files == [
        "FIX_R2_A1_LOAD_06_R6_RUNTIME_PROBE_REPORT.md",
        "counterfactual.json",
        "prompt_at_tokenizer_step0.txt",
        "runtime_trace.json",
    ]
    report = (tmp_path / "A.2" / "FIX_R2_A1_LOAD_06_R6_RUNTIME_PROBE_REPORT.md").read_text(encoding="utf-8")
    assert "branch=data_or_criterion_semantic; estimated_fix_horizon=days_to_weeks" in report
    assert len((tmp_path / "A.2" / "prompt_at_tokenizer_step0.txt").read_text(encoding="utf-8")) == 500
