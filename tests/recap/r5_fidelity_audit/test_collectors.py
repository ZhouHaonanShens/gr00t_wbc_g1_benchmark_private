from __future__ import annotations

import json
from pathlib import Path

import pytest

from work.recap.r2_authentic_eval import exclusion
from work.recap.r3_contract_parity import collectors as r3_collectors
from work.recap.r5_fidelity_audit import collectors
from work.recap.r5_fidelity_audit.contract import R5AuditError


def test_load_repo_file_text_missing_returns_none(tmp_path: Path) -> None:
    assert collectors.load_repo_file_text("missing.py", tmp_path) is None
    path = tmp_path / "present.py"
    path.write_text("from __future__ import annotations\nVALUE = 1\n", encoding="utf-8")
    assert collectors.load_repo_file_text("present.py", tmp_path) == path.read_text(encoding="utf-8")


def test_grep_symbol_in_files_reports_line_numbers(tmp_path: Path) -> None:
    src = tmp_path / "work" / "recap"
    src.mkdir(parents=True)
    file_path = src / "sample.py"
    file_path.write_text(
        "alpha = 1\nadvantage_embedding = 'x'\n# ADVANTAGE_EMBEDDING again\n",
        encoding="utf-8",
    )
    hits = collectors.grep_symbol_in_files(
        ("advantage_embedding",),
        ("work/recap/sample.py",),
        tmp_path,
        case_sensitive=False,
    )
    assert [hit.line_number for hit in hits["advantage_embedding"]] == [2, 3]
    assert hits["advantage_embedding"][0].file_path == "work/recap/sample.py"


def test_find_dataset_meta_collects_known_static_files(tmp_path: Path) -> None:
    recap = tmp_path / "work" / "recap"
    recap.mkdir(parents=True)
    (recap / "advantage.py").write_text('ADVANTAGE_INPUT_COLUMN = "recap_m2.advantage_input"\n', encoding="utf-8")
    (recap / "label_writer.py").write_text('FIELD = "indicator_I"\n', encoding="utf-8")
    meta = collectors.find_dataset_meta(tmp_path)
    assert "work/recap/advantage.py" in meta["source_paths"]
    assert "work/recap/label_writer.py" in meta["source_paths"]
    assert meta["symbol_hits"]["advantage_input"]
    assert meta["symbol_hits"]["indicator_I"]


def test_phase_a_literals_parse_q1_to_q7(tmp_path: Path) -> None:
    doc = tmp_path / "phase_a.md"
    doc.write_text("\n".join(f"- Q{i}: literal evidence" for i in range(1, 8)), encoding="utf-8")
    parsed = collectors.load_phase_a_literals(tmp_path, paths=("phase_a.md",))
    for qid in tuple(f"Q{i}" for i in range(1, 8)):
        assert parsed["question_literals"][qid][0]["file_path"] == "phase_a.md"
    assert parsed["question_literals"]["Q8"] == ()
    assert parsed["artifact_presence"] == {"phase_a.md": True}


def _write_ckpt(root: Path, cell_id: str = "A.2") -> Path:
    ckpt = root / r3_collectors._CELL_CKPT_RELATIVE[cell_id]
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text(json.dumps({"training_algo": "RECAP"}), encoding="utf-8")
    (ckpt / "processor_config.json").write_text(json.dumps({"processor": True}), encoding="utf-8")
    (ckpt / "statistics.json").write_text(json.dumps({"unitree_g1": {}}), encoding="utf-8")
    return ckpt


def test_load_ckpt_experiment_cfg_uses_known_r2_artifact_shape(tmp_path: Path) -> None:
    ckpt_root = tmp_path / "ckpts"
    r2_root = tmp_path / "r2"
    ckpt = _write_ckpt(ckpt_root)
    r2_dir = r2_root / r3_collectors._CELL_R2_DIR["A.2"]
    r2_dir.mkdir(parents=True)
    (r2_dir / "cell_result.json").write_text(json.dumps({"request": {"checkpoint": {"abs_path": str(ckpt)}}}), encoding="utf-8")
    cfg = collectors.load_ckpt_experiment_cfg(
        tmp_path,
        ckpt_search_root=ckpt_root,
        r2_run_roots=(r2_root,),
    )
    assert tuple(cfg) == exclusion.EVIDENCE_GRADE_CELL_IDS
    assert cfg["A.2"]["config"]["training_algo"] == "RECAP"
    assert cfg["A.2"]["r2_cell_result"]["request"]["checkpoint"]["abs_path"] == str(ckpt)
    assert cfg["A.3"]["source_paths"] == ()
    assert cfg["A.3"]["missing_paths"]


def test_load_ckpt_experiment_cfg_rejects_invalid_json(tmp_path: Path) -> None:
    ckpt_root = tmp_path / "ckpts"
    ckpt = _write_ckpt(ckpt_root)
    (ckpt / "config.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(R5AuditError):
        collectors.load_ckpt_experiment_cfg(
            tmp_path,
            ckpt_search_root=ckpt_root,
            r2_run_roots=(tmp_path / "r2",),
            cell_ids=("A.2",),
        )
