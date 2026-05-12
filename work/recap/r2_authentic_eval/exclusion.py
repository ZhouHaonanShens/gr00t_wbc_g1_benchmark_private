"""A.1 exclusion SSOT for R2 evidence-grade statistics.

Decision source: .reference/records/recap_a1_exclusion_record.md §S5/§S8.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, NamedTuple

EXCLUDED_CELL_ID: str = "A.1"
EXCLUDED_PATH_MARKER: str = "g2_full_training"
EVIDENCE_GRADE_CELL_IDS: tuple[str, ...] = ("A.2", "A.3", "A.4", "A.5")
EVIDENCE_GRADE_N_CELLS: int = len(EVIDENCE_GRADE_CELL_IDS)
A1_EXCLUSION_RECORD_PATH: str = ".reference/records/recap_a1_exclusion_record.md"


class CitePolicy(NamedTuple):
    record_path: str
    sections: tuple[str, str]
    decision: str


CITE_POLICY = CitePolicy(
    record_path=A1_EXCLUSION_RECORD_PATH,
    sections=("§S5", "§S8"),
    decision="exclude A.1 from evidence-grade statistical counts",
)


def is_excluded_cell(cell: Mapping[str, Any]) -> bool:
    """Return whether a raw R2 cell is excluded from evidence-grade counts."""
    for key in ("cell_id", "id", "label", "ckpt_label"):
        if cell.get(key) == EXCLUDED_CELL_ID:
            return True

    text_parts = [str(cell.get(key) or "") for key in ("ckpt_abs_path", "abs_path", "path")]
    request = cell.get("request")
    if isinstance(request, Mapping):
        checkpoint = request.get("checkpoint")
        if isinstance(checkpoint, Mapping):
            text_parts.extend(
                str(checkpoint.get(key) or "")
                for key in ("abs_path", "training_run_dir")
            )
    checkpoint = cell.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        text_parts.extend(
            str(checkpoint.get(key) or "") for key in ("abs_path", "training_run_dir")
        )
    return any(EXCLUDED_PATH_MARKER in part for part in text_parts)


def filter_evidence_grade(cells: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Preserve order while dropping cells excluded from evidence-grade counts."""
    return [cell for cell in cells if not is_excluded_cell(cell)]
