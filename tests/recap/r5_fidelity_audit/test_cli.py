from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True)
class _FakeResult:
    question: str
    analyzer_name: str
    repo_presence: str
    active_path_consumption: str
    evidence_files: tuple[str, ...]
    evidence_artifacts: tuple[str, ...]
    conclusion: str
    confidence: str


class _FakeR5AuditError(Exception):
    pass


def _install_fake_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    questions = tuple(f"Q{i}" for i in range(1, 10))

    contract = types.ModuleType("work.recap.r5_fidelity_audit.contract")
    contract.FIDELITY_QUESTIONS = questions
    contract.FULL_REPORT_FILENAME = "gr00t_recap_fidelity_fact_report_v1.md"
    contract.R5AuditError = _FakeR5AuditError

    analyzers = types.ModuleType("work.recap.r5_fidelity_audit.analyzers")

    def audit_question(question: str) -> _FakeResult:
        if question not in questions:
            raise _FakeR5AuditError(f"unsupported question {question}")
        return _FakeResult(
            question=question,
            analyzer_name=f"analyze_{question.lower()}",
            repo_presence="IMPLEMENTED",
            active_path_consumption="ABSENT" if question == "Q2" else "IMPLEMENTED",
            evidence_files=(f"work/example/{question}.py",),
            evidence_artifacts=(f"agent/artifacts/{question}.json",),
            conclusion=f"{question} static fixture conclusion",
            confidence="HIGH",
        )

    analyzers.audit_question = audit_question

    verdicts = types.ModuleType("work.recap.r5_fidelity_audit.verdicts")

    def overall_fidelity_label(results: tuple[Any, ...]) -> str:
        return "DETACHED_RUNTIME_PATH" if any(r.active_path_consumption == "ABSENT" for r in results) else "FULL_FIDELITY"

    verdicts.overall_fidelity_label = overall_fidelity_label

    for name, module in {
        contract.__name__: contract,
        analyzers.__name__: analyzers,
        verdicts.__name__: verdicts,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("work.recap.r5_fidelity_audit.cli", None)


def _load_cli(monkeypatch: pytest.MonkeyPatch) -> Any:
    _install_fake_dependencies(monkeypatch)
    return importlib.import_module("work.recap.r5_fidelity_audit.cli")


def test_cli_all_writes_nine_question_outputs_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli(monkeypatch)

    assert cli.main(["audit", "--all", "--output-root", str(tmp_path)]) == 0

    for question in tuple(f"Q{i}" for i in range(1, 10)):
        q_dir = tmp_path / question
        assert (q_dir / "fidelity_question_manifest.json").is_file()
        assert (q_dir / "fidelity_question_report.md").is_file()
        manifest = json.loads((q_dir / "fidelity_question_manifest.json").read_text(encoding="utf-8"))
        assert manifest["question"] == question
        assert manifest["repo_presence"] == "IMPLEMENTED"

    matrix = json.loads((tmp_path / "gr00t_recap_fidelity_matrix.json").read_text(encoding="utf-8"))
    assert matrix["overall_label"] == "DETACHED_RUNTIME_PATH"
    assert len(matrix["questions"]) == 9
    summary = (tmp_path / "gr00t_recap_fidelity_fact_report_v1.md").read_text(encoding="utf-8")
    assert "repo_presence" in summary
    assert "active_path_consumption" in summary
    assert summary.count("| Q") == 9


def test_cli_question_writes_only_requested_question(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli(monkeypatch)

    assert cli.main(["audit", "--question", "q1", "--output-root", str(tmp_path)]) == 0

    assert (tmp_path / "Q1" / "fidelity_question_manifest.json").is_file()
    assert not (tmp_path / "Q2").exists()
    assert not (tmp_path / "gr00t_recap_fidelity_matrix.json").exists()


def test_cli_rejects_unknown_and_a1_without_success_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_cli(monkeypatch)

    assert cli.main(["audit", "--question", "Q99", "--output-root", str(tmp_path)]) == 2
    assert cli.main(["audit", "--question", "A.1", "--output-root", str(tmp_path)]) == 2
    assert not any(tmp_path.iterdir())


def test_main_module_is_thin_dispatch() -> None:
    text = Path("work/recap/r5_fidelity_audit/__main__.py").read_text(encoding="utf-8").strip().splitlines()

    assert len(text) <= 3
    assert any("from work.recap.r5_fidelity_audit.cli import main" in line for line in text)
    assert text[-1] == "raise SystemExit(main())"
