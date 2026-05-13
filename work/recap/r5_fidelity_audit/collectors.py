from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from work.recap.r2_authentic_eval import exclusion
from work.recap.r3_contract_parity import collectors as _r3_collectors
from work.recap.r5_fidelity_audit.contract import FIDELITY_QUESTIONS, R5AuditError

DEFAULT_DATASET_META_FILES: tuple[str, ...] = (
    "work/recap/advantage.py", "work/recap/phase_thresholds.py", "work/recap/text_indicator.py",
    "work/recap/labeler.py", "work/recap/label_writer.py", "work/recap/dataset.py", "work/recap/run_manifest.py",
)
DEFAULT_PHASE_LITERAL_FILES: tuple[str, ...] = (
    "agent/logs/phase_a_report_lookup_20260512T021512Z.md",
    "agent/exchange/openpi_recap_fidelity_fact_report_v1.md",
    "agent/exchange/openpi_recap_paper_gap_matrix_v2.md",
    "agent/exchange/openpi_recap_paper_contract_v1.md",
)
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "advantage_embedding", "advantage_input", "indicator_I", "indicator_mode",
    "carrier_text_v1", "combine_alpha_dual_loss", "EVIDENCE_GRADE_CELL_IDS",
)
_QID_RE = re.compile(r"\bQ[1-9]\b")


@dataclass(frozen=True)
class SymbolHit:
    file_path: str
    line_number: int
    line_text: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseLiteralHit:
    file_path: str
    line_number: int
    text: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _resolve_path(repo_root: Path | str, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(repo_root) / candidate


def _unique(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(path) for path in paths))


def default_evidence_files() -> tuple[str, ...]:
    return _unique(path for q in FIDELITY_QUESTIONS for path in q.evidence_files)


def default_evidence_artifacts() -> tuple[str, ...]:
    return _unique(path for q in FIDELITY_QUESTIONS for path in q.evidence_artifacts)


def load_repo_file_text(path: Path | str, repo_root: Path | str = Path(".")) -> str | None:
    resolved = _resolve_path(repo_root, path)
    if not resolved.is_file():
        return None
    try:
        return resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise R5AuditError(f"non UTF-8 repo file: {resolved}") from exc


def _read_json_dict(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise R5AuditError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise R5AuditError(f"expected JSON object: {path}")
    return data


def grep_symbol_in_files(
    symbols: str | Sequence[str],
    files: Sequence[Path | str],
    repo_root: Path | str = Path("."),
    *,
    case_sensitive: bool = True,
) -> dict[str, tuple[SymbolHit, ...]]:
    needles = (symbols,) if isinstance(symbols, str) else tuple(str(s) for s in symbols)
    hits: dict[str, list[SymbolHit]] = {needle: [] for needle in needles}
    normalized = {needle: needle if case_sensitive else needle.lower() for needle in needles}
    for file_path in files:
        text = load_repo_file_text(file_path, repo_root)
        if text is None:
            continue
        for line_number, line_text in enumerate(text.splitlines(), start=1):
            haystack = line_text if case_sensitive else line_text.lower()
            for needle, normalized_needle in normalized.items():
                if normalized_needle in haystack:
                    hits[needle].append(SymbolHit(str(file_path), line_number, line_text.strip()))
    return {needle: tuple(items) for needle, items in hits.items()}


def artifact_presence(
    paths: Sequence[Path | str] | None = None,
    repo_root: Path | str = Path("."),
) -> dict[str, bool]:
    candidates = tuple(paths) if paths is not None else default_evidence_artifacts()
    return {str(path): _resolve_path(repo_root, path).exists() for path in candidates}


def find_dataset_meta(repo_root: Path | str = Path(".")) -> dict[str, object]:
    repo_texts = {p: text for p in DEFAULT_DATASET_META_FILES if (text := load_repo_file_text(p, repo_root)) is not None}
    return {
        "candidate_files": DEFAULT_DATASET_META_FILES,
        "source_paths": tuple(repo_texts),
        "missing_paths": tuple(p for p in DEFAULT_DATASET_META_FILES if p not in repo_texts),
        "symbol_hits": _hits_as_dict(grep_symbol_in_files(DEFAULT_SYMBOLS, DEFAULT_DATASET_META_FILES, repo_root)),
    }


def _hits_as_dict(hits: dict[str, tuple[SymbolHit, ...]]) -> dict[str, tuple[dict[str, object], ...]]:
    return {symbol: tuple(hit.as_dict() for hit in items) for symbol, items in hits.items()}


def _default_r2_roots(repo_root: Path | str) -> tuple[Path, ...]:
    return tuple(root if root.is_absolute() else _resolve_path(repo_root, root) for root in _r3_collectors.DEFAULT_R2_RUN_ROOTS)


def _validate_cell_ids(cell_ids: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(cell_ids) if cell_ids is not None else tuple(exclusion.EVIDENCE_GRADE_CELL_IDS)
    for cell_id in selected:
        if exclusion.is_excluded_cell({"cell_id": cell_id}):
            raise R5AuditError(f"excluded R2 cell is not evidence-grade: {cell_id}")
        if cell_id not in exclusion.EVIDENCE_GRADE_CELL_IDS:
            raise R5AuditError(f"unknown evidence-grade R2 cell: {cell_id}")
    return selected


def _first_existing_r2_result(cell_id: str, roots: Sequence[Path]) -> Path | None:
    for root in roots:
        candidate = root / _r3_collectors._CELL_R2_DIR[cell_id] / "cell_result.json"
        if candidate.is_file():
            return candidate
    return None


def load_ckpt_experiment_cfg(
    repo_root: Path | str = Path("."),
    *,
    ckpt_search_root: Path | str | None = None,
    r2_run_roots: Sequence[Path | str] | None = None,
    cell_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, object]]:
    ckpt_root = Path(ckpt_search_root) if ckpt_search_root is not None else _resolve_path(repo_root, "agent/artifacts/gr00t_recap_live")
    r2_roots = tuple(Path(root) for root in r2_run_roots) if r2_run_roots is not None else _default_r2_roots(repo_root)
    out: dict[str, dict[str, object]] = {}
    for cell_id in _validate_cell_ids(cell_ids):
        ckpt = ckpt_root / _r3_collectors._CELL_CKPT_RELATIVE[cell_id]
        paths = (ckpt / "config.json", ckpt / "processor_config.json", ckpt / "statistics.json")
        r2_path = _first_existing_r2_result(cell_id, r2_roots)
        source_paths = tuple(str(path) for path in (*paths, r2_path) if path is not None and path.is_file())
        missing_paths = tuple(str(path) for path in paths if not path.is_file()) + (() if r2_path is not None else ("cell_result.json",))
        out[cell_id] = {
            "cell_id": cell_id,
            "checkpoint_path": str(ckpt),
            "source_paths": source_paths,
            "missing_paths": missing_paths,
            "config": _read_json_dict(paths[0]) or {},
            "processor_config": _read_json_dict(paths[1]) or {},
            "statistics": _read_json_dict(paths[2]) or {},
            "r2_cell_result": _read_json_dict(r2_path) or {},
        }
    return out


def load_phase_a_literals(
    repo_root: Path | str = Path("."),
    paths: Sequence[Path | str] | None = None,
) -> dict[str, object]:
    candidates = tuple(paths) if paths is not None else DEFAULT_PHASE_LITERAL_FILES
    question_literals: dict[str, list[PhaseLiteralHit]] = {q.qid: [] for q in FIDELITY_QUESTIONS}
    source_paths: list[str] = []
    missing_paths: list[str] = []
    for rel_path in candidates:
        text = load_repo_file_text(rel_path, repo_root)
        if text is None:
            missing_paths.append(str(rel_path)); continue
        source_paths.append(str(rel_path))
        for line_number, line_text in enumerate(text.splitlines(), start=1):
            for qid in tuple(dict.fromkeys(_QID_RE.findall(line_text))):
                question_literals[qid].append(PhaseLiteralHit(str(rel_path), line_number, line_text.strip()))
    return {
        "source_paths": tuple(source_paths),
        "missing_paths": tuple(missing_paths),
        "question_literals": {qid: tuple(hit.as_dict() for hit in hits) for qid, hits in question_literals.items()},
        "artifact_presence": artifact_presence(candidates, repo_root),
    }


def collect_static_sources(repo_root: Path | str = Path(".")) -> dict[str, object]:
    files = default_evidence_files()
    hits = grep_symbol_in_files(DEFAULT_SYMBOLS, files, repo_root)
    return {
        "repo_texts": {p: text for p in files if (text := load_repo_file_text(p, repo_root)) is not None},
        "symbol_hits": _hits_as_dict(hits),
        "dataset_meta": find_dataset_meta(repo_root),
        "ckpt_cfg_by_cell": load_ckpt_experiment_cfg(repo_root),
        "phase_a_literals": load_phase_a_literals(repo_root),
        "artifact_presence": artifact_presence(repo_root=repo_root),
    }


__all__ = (
    "DEFAULT_DATASET_META_FILES", "DEFAULT_PHASE_LITERAL_FILES", "DEFAULT_SYMBOLS", "PhaseLiteralHit",
    "SymbolHit", "artifact_presence", "collect_static_sources", "default_evidence_artifacts",
    "default_evidence_files", "find_dataset_meta", "grep_symbol_in_files", "load_ckpt_experiment_cfg",
    "load_phase_a_literals", "load_repo_file_text",
)
