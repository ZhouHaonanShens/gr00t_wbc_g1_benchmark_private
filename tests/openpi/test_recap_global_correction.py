from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_CORRECTION_DOC = (
    REPO_ROOT / "agent/exchange/openpi_recap_global_correction_v1.md"
)
GAP_MATRIX_DOC = REPO_ROOT / "agent/exchange/openpi_recap_paper_gap_matrix_v2.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_global_correction_doc_freezes_core_recap_wording() -> None:
    text = _read(GLOBAL_CORRECTION_DOC)
    required = [
        "OpenPI RECAP 全局纠偏与术语冻结 v1",
        "repo presence",
        "active-path consumption",
        "`B0 = omit-control`",
        "当前 `C` 路径负结果不是 clean RECAP verdict",
        "当前 repo 没有本地 learned critic 训练入口",
        "对应现有 paired summary / provenance 里的 B control",
        "ACTIVE-PATH-BLOCKER",
        "VERDICT-BLOCKER",
    ]
    for item in required:
        assert item in text, f"missing global correction wording: {item}"


def test_gap_matrix_v2_freezes_repo_presence_and_verdict_boundaries() -> None:
    text = _read(GAP_MATRIX_DOC)
    required = [
        "OpenPI RECAP paper gap matrix v2",
        "clean paper-faithful RECAP",
        "`B0 = omit-control`",
        "repo presence",
        "active-path consumption",
        "当前 `C` 路径负结果只能写成 inherited negative result",
        "BLOCKER-IF-DRIFTING",
        "FROZEN-DEVIATION",
    ]
    for item in required:
        assert item in text, f"missing gap-matrix wording: {item}"
