from __future__ import annotations

import json
from pathlib import Path

import pytest

from work.recap.r3_contract_parity import cli, collectors
from work.recap.r3_contract_parity.contract import PARITY_AXES, PASS, _MISSING, ParityAxisResult, ParityCellReport


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    search, r2 = tmp_path / "search", tmp_path / "r2"
    monkeypatch.setattr(collectors, "DEFAULT_SEARCH_ROOT", search)
    for cell, rel in collectors._CELL_CKPT_RELATIVE.items():
        ckpt = search / rel
        ckpt.mkdir(parents=True)
        (ckpt / "config.json").write_text(json.dumps({"architectures": ["Gr00tN1d6"], "formalize_language": False}), encoding="utf-8")
        (ckpt / "processor_config.json").write_text("{}", encoding="utf-8")
        (ckpt / "statistics.json").write_text("{}", encoding="utf-8")
        r2_dir = r2 / collectors._CELL_R2_DIR[cell]
        r2_dir.mkdir(parents=True)
        req = {"abs_path": str(ckpt.resolve()), "config_json_sha256": collectors._sha256_file(ckpt / "config.json"), "processor_config_json_sha256": collectors._sha256_file(ckpt / "processor_config.json"), "statistics_json_sha256": collectors._sha256_file(ckpt / "statistics.json"), "training_algo": "Gr00tN1d6", "formalize_language": False}
        (r2_dir / "cell_result.json").write_text(json.dumps({"request": {"checkpoint": req}, "formal_eval_summary_json": {"checkpoint": str(ckpt.resolve())}}), encoding="utf-8")
    return r2, tmp_path / "out"


def test_cli_all_writes_four_cells_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r2, out = _fixture(tmp_path, monkeypatch)
    assert cli.main(["audit", "--all", "--output-root", str(out), "--r2-run-root", str(r2)]) == 0
    for cell in ("A.2", "A.3", "A.4", "A.5"):
        assert (out / cell / "cell_parity_manifest.json").exists()
        assert (out / cell / "cell_parity_report.md").exists()
    assert "runtime_invocations = []" in (out / "r3_parity_summary.md").read_text(encoding="utf-8")


def test_cli_rejects_a1_and_unknown_without_success_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r2, out = _fixture(tmp_path, monkeypatch)
    assert cli.main(["audit", "--cell", "A.1", "--output-root", str(out), "--r2-run-root", str(r2)]) == 2
    assert not (out / "A.1").exists()
    assert cli.main(["audit", "--cell", "Z.9", "--output-root", str(out), "--r2-run-root", str(r2)]) == 2


def test_cell_manifest_serializes_missing_sentinel() -> None:
    axis = ParityAxisResult(PARITY_AXES[0], _MISSING, "eval", PASS, "none", "probe")
    data = cli._cell_dict(ParityCellReport("A.2", "ckpt", PASS, (axis,), ()))
    assert data["axes"][0]["train_value"] == "__MISSING__"
