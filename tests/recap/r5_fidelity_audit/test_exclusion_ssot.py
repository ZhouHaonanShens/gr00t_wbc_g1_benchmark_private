from __future__ import annotations

from pathlib import Path

import pytest

from work.recap.r2_authentic_eval import exclusion
from work.recap.r5_fidelity_audit import collectors
from work.recap.r5_fidelity_audit.contract import R5AuditError


def test_q9_cell_inventory_defaults_to_r2_exclusion_ssot(tmp_path: Path) -> None:
    cfg = collectors.load_ckpt_experiment_cfg(tmp_path, ckpt_search_root=tmp_path / "missing", r2_run_roots=())
    assert tuple(cfg) == exclusion.EVIDENCE_GRADE_CELL_IDS
    assert exclusion.EXCLUDED_CELL_ID not in cfg


def test_excluded_or_unknown_cells_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(R5AuditError):
        collectors.load_ckpt_experiment_cfg(tmp_path, cell_ids=(exclusion.EXCLUDED_CELL_ID,))
    with pytest.raises(R5AuditError):
        collectors.load_ckpt_experiment_cfg(tmp_path, cell_ids=("Z.9",))


def test_collectors_source_reuses_ssot_without_a1_path_marker() -> None:
    source = Path(collectors.__file__).read_text(encoding="utf-8")
    assert "exclusion.EVIDENCE_GRADE_CELL_IDS" in source
    assert "g2_full_training" not in source
