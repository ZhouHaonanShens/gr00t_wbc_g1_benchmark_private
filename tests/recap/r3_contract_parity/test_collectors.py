from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from work.recap.r3_contract_parity import collectors
from work.recap.r3_contract_parity.contract import R3AuditError, _MISSING


def _write_ckpt(root: Path, cell: str = "A.2") -> Path:
    ckpt = root / collectors._CELL_CKPT_RELATIVE[cell]
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text(json.dumps({"architectures": ["Gr00tN1d6"], "formalize_language": False}), encoding="utf-8")
    (ckpt / "processor_config.json").write_text('{"processor": true}', encoding="utf-8")
    (ckpt / "statistics.json").write_text(json.dumps({"unitree_g1": {"action": {"right_hand": {"q99": [1, 2]}}}}), encoding="utf-8")
    return ckpt


def test_resolve_rejects_a1_and_unknown(tmp_path: Path) -> None:
    with pytest.raises(R3AuditError):
        collectors.resolve_cell_ckpt("A.1", tmp_path)
    with pytest.raises(R3AuditError):
        collectors.resolve_cell_ckpt("Z.9", tmp_path)


def test_train_snapshot_sha_missing_and_eval_extraction(tmp_path: Path) -> None:
    ckpt = _write_ckpt(tmp_path)
    assert collectors.resolve_cell_ckpt("A.2", tmp_path) == ckpt.resolve()
    snap = collectors.load_train_snapshot(ckpt)
    cfg_bytes = (ckpt / "config.json").read_bytes()
    assert snap["checkpoint"]["config_json_sha256"] == hashlib.sha256(cfg_bytes).hexdigest()
    assert snap["checkpoint"]["training_algo"] == "Gr00tN1d6"
    assert collectors.load_train_snapshot(tmp_path / "missing")["checkpoint"]["config_json_sha256"] is _MISSING
    r2_dir = tmp_path / "r2" / collectors._CELL_R2_DIR["A.2"]
    r2_dir.mkdir(parents=True)
    (r2_dir / "cell_result.json").write_text(json.dumps({"request": {"checkpoint": {"abs_path": str(ckpt), "training_algo": "Gr00tN1d6"}}, "formal_eval_summary_json": {"checkpoint": str(ckpt)}}), encoding="utf-8")
    eval_snap = collectors.load_eval_snapshot("A.2", tmp_path / "r2")
    assert eval_snap["eval"]["checkpoint"] == str(ckpt)
    assert eval_snap["request"]["checkpoint"]["training_algo"] == "Gr00tN1d6"
