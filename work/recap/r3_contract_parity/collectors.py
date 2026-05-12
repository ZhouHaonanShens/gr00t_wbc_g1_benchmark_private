from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from work.recap.r2_authentic_eval import exclusion
from work.recap.r3_contract_parity.contract import R3AuditError, _MISSING

DEFAULT_SEARCH_ROOT = Path("agent/artifacts/gr00t_recap_live").resolve()
DEFAULT_R2_RUN_ROOTS = (
    Path("agent/artifacts/recap_substrate_recovery/r2/20260511T032354Z_phaseE_unique/r2_1_authentic_eval/20260511T032406Z"),
)
_CELL_CKPT_RELATIVE = {
    "A.2": "single_gpu_v2_full_update/stage1_gr00t_r2r4_closed_candidate_iter9_20260426T_nextZ/gr00t/g2_main_v2_full_training/checkpoint-2200",
    "A.3": "single_gpu_v2_full_update/stage1_gr00t_r2r4_closed_candidate_iter9_20260426T_nextZ/gr00t/g3_conditioned_continuation_after_sanity_20260430_131809/checkpoint-4400",
    "A.4": "single_gpu_v2_full_update/stage1_gr00t_r2r4_closed_candidate_iter9_20260426T_nextZ/gr00t/g3_conditioned_continuation_6600_after_surfacefix_20260430_181210/checkpoint-6600",
    "A.5": "single_gpu_v2_full_update/stage1_gr00t_r2r4_closed_candidate_iter9_20260426T_nextZ/gr00t/g3_resume_after_demo_full_training_20260430_20260430_114803/checkpoint-2200",
}
_CELL_R2_DIR = {
    "A.2": "g2_main_v2_full_training__checkpoint-2200",
    "A.3": "g3_conditioned_continuation_after_sanity_20260430_131809__checkpoint-4400",
    "A.4": "g3_conditioned_continuation_6600_after_surfacefix_20260430_181210__checkpoint-6600",
    "A.5": "g3_resume_after_demo_full_training_20260430_20260430_114803__checkpoint-2200",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise R3AuditError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise R3AuditError(f"expected JSON object: {path}")
    return data


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return _MISSING


def _checkpoint_steps(path: Path) -> int | object:
    name = path.name
    if name.startswith("checkpoint-") and name.removeprefix("checkpoint-").isdigit():
        return int(name.removeprefix("checkpoint-"))
    return _MISSING


def _single_arch(config: Mapping[str, Any]) -> Any:
    arch = config.get("architectures")
    if isinstance(arch, list) and arch:
        return arch[0]
    return _first_present(config.get("model_type"), config.get("model_name"))


def _file_sha(path: Path) -> str | object:
    return _sha256_file(path) if path.exists() else _MISSING


def resolve_cell_ckpt(cell_id: str, search_root: Path | None = None) -> Path:
    if exclusion.is_excluded_cell({"cell_id": cell_id}):
        raise R3AuditError(f"{cell_id} is excluded by R2 evidence-grade SSOT")
    if cell_id not in exclusion.EVIDENCE_GRADE_CELL_IDS or cell_id not in _CELL_CKPT_RELATIVE:
        raise R3AuditError(f"unknown R3 evidence-grade cell: {cell_id}")
    root = search_root or DEFAULT_SEARCH_ROOT
    return (root / _CELL_CKPT_RELATIVE[cell_id]).resolve()


def load_train_snapshot(ckpt_abs_path: Path) -> dict[str, Any]:
    ckpt = ckpt_abs_path.resolve()
    config_path = ckpt / "config.json"
    proc_path = ckpt / "processor_config.json"
    stats_path = ckpt / "statistics.json"
    config = _read_json(config_path)
    stats = _read_json(stats_path)
    return {
        "checkpoint": {
            "abs_path": str(ckpt),
            "config_json_sha256": _file_sha(config_path),
            "processor_config_json_sha256": _file_sha(proc_path),
            "statistics_json_sha256": _file_sha(stats_path),
            "training_algo": _single_arch(config),
            "formalize_language": _first_present(config.get("formalize_language")),
            "n_train_steps": _checkpoint_steps(ckpt),
            "statistics_q99_right_hand": _first_present(
                (((stats.get("unitree_g1") or {}).get("action") or {}).get("right_hand") or {}).get("q99")
            ),
        },
        "source_paths": [str(p) for p in (config_path, proc_path, stats_path) if p.exists()],
    }


def _find_eval_manifest(cell_id: str, r2_run_root: Path | None) -> Path | None:
    roots = (r2_run_root,) if r2_run_root is not None else DEFAULT_R2_RUN_ROOTS
    for root in roots:
        candidate = Path(root) / _CELL_R2_DIR[cell_id] / "cell_result.json"
        if candidate.exists():
            return candidate
    return None


def load_eval_snapshot(cell_id: str, r2_run_root: Path | None = None) -> dict[str, Any]:
    if exclusion.is_excluded_cell({"cell_id": cell_id}) or cell_id not in exclusion.EVIDENCE_GRADE_CELL_IDS:
        raise R3AuditError(f"invalid R3 eval cell: {cell_id}")
    manifest = _find_eval_manifest(cell_id, r2_run_root)
    if manifest is None:
        return {"eval": {"checkpoint": _MISSING}, "request": {"checkpoint": {}}, "source_paths": []}
    data = _read_json(manifest)
    request = data.get("request") if isinstance(data.get("request"), dict) else {}
    ckpt = request.get("checkpoint") if isinstance(request.get("checkpoint"), dict) else {}
    formal = data.get("formal_eval_summary_json") if isinstance(data.get("formal_eval_summary_json"), dict) else {}
    raw = data.get("raw_repro_result") if isinstance(data.get("raw_repro_result"), dict) else {}
    protocol = raw.get("protocol") if isinstance(raw.get("protocol"), dict) else {}
    checkpoint = _first_present(formal.get("checkpoint"), protocol.get("ckpt_root"), ckpt.get("abs_path"))
    return {"eval": {"checkpoint": checkpoint}, "request": {"checkpoint": ckpt}, "source_paths": [str(manifest)]}
