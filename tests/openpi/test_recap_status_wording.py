from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = {
    "global_correction": REPO_ROOT
    / "agent/exchange/openpi_recap_global_correction_v1.md",
    "fact_report": REPO_ROOT / "agent/exchange/openpi_recap_fidelity_fact_report_v1.md",
    "limitations": REPO_ROOT / "agent/exchange/openpi_recap_fidelity_limitations_v1.md",
    "gap_matrix": REPO_ROOT / "agent/exchange/openpi_recap_paper_gap_matrix_v2.md",
}


def _read(name: str) -> str:
    return DOCS[name].read_text(encoding="utf-8")


def test_fact_report_and_limitations_keep_frozen_status_wording() -> None:
    fact_text = _read("fact_report")
    limitations_text = _read("limitations")

    fact_required = [
        "repo presence",
        "active-path consumption",
        "`B0 = omit-control`",
        "active-path absent, repo presence exists for static relabel fields",
        "当前 `C` 路径负结果不是 clean RECAP verdict",
    ]
    for item in fact_required:
        assert item in fact_text, f"missing fact-report wording: {item}"

    limitations_required = [
        "repo presence",
        "active-path consumption",
        "`B0 = omit-control`",
        "不能写成 clean RECAP verdict",
    ]
    for item in limitations_required:
        assert item in limitations_text, f"missing limitations wording: {item}"


def test_scoped_docs_exclude_legacy_and_overclaim_wording() -> None:
    forbidden = [
        "critic globally absent",
        "SFT baseline",
        "当前 `C` 结果已经给出了 clean RECAP verdict",
        "当前 `C` 路径负结果是 clean RECAP verdict",
    ]
    for name in DOCS:
        text = _read(name)
        for item in forbidden:
            assert item not in text, f"forbidden wording remained in {name}: {item}"
