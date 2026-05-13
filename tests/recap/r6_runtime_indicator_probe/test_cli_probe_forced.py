from __future__ import annotations

from pathlib import Path

from work.recap.r6_runtime_indicator_probe import cli
from work.recap.r6_runtime_indicator_probe.contract import ProbeCounterfactual, RuntimeTrace


def test_probe_forced_validation_and_runtime_flags(tmp_path: Path, monkeypatch) -> None:
    token = "c" * 64
    seen = []

    def fake_run_runtime_probe(cell, budget, leader_approval_token, *, forced=False, counterfactual=True):
        seen.append((cell, budget.gpu_id, leader_approval_token, forced, counterfactual))
        trace = RuntimeTrace("A.2", 20000, "Advantage: positive", "p" * 64, "a" * 64, (0, 0, 0, 0, 0), True, "INDICATOR_PRESENT")
        cf = ProbeCounterfactual("A.2", 20000, "a" * 64, "a" * 64, True, (0, 0, 0, 0, 0), "INDICATOR_INVARIANT")
        return trace, cf

    monkeypatch.setattr("work.recap.r6_runtime_indicator_probe.runtime_probe.run_runtime_probe", fake_run_runtime_probe)
    assert cli.main(["probe", "--forced", "--cell", "A.2", "--leader-approval-token", token, "--gpu", "1", "--output-root", str(tmp_path)]) == 0
    assert seen == [("A.2", 1, token, True, True)]
    assert cli.main(["probe", "--forced", "--cell", "A.2", "--leader-approval-token", token, "--gpu", "2", "--output-root", str(tmp_path)]) == 2
    assert cli.main(["probe", "--forced", "--cell", "A.3", "--leader-approval-token", token, "--gpu", "1", "--output-root", str(tmp_path)]) == 2
