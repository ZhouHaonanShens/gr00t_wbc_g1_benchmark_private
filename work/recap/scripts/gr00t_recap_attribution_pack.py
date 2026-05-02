#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


# =====================
# USER Config (edit)
# =====================

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT_DIR = Path("agent/artifacts/gr00t_anchor_controller_recap/final_wave")
DEFAULT_TASK18_EVIDENCE_PATH = Path(".sisyphus/evidence/task-18-attribution-pack.json")

DEFAULT_TASK1_EVIDENCE = Path(".sisyphus/evidence/task-1-eval-contract.json")
DEFAULT_TASK12_EVIDENCE = Path(".sisyphus/evidence/task-12-ladder-policy-gate.json")
DEFAULT_TASK14_EVIDENCE = Path(
    ".sisyphus/evidence/task-14-p-ladder-new-embodiment.json"
)
DEFAULT_TASK15_EVIDENCE = Path(".sisyphus/evidence/task-15-d-ladder-policy-gate.json")
DEFAULT_TASK16_EVIDENCE = Path(".sisyphus/evidence/task-16-d-ladder-unitree-g1.json")
DEFAULT_TASK17_EVIDENCE = Path(
    ".sisyphus/evidence/task-17-d-ladder-new-embodiment.json"
)
DEFAULT_CHECKPOINT_PROVENANCE_REPORT = Path(
    "agent/artifacts/gr00t_checkpoint_provenance/checkpoint_provenance_report.json"
)
DEFAULT_DUAL_BRANCH_SCORECARD_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/dual_branch_scorecard.json"
)

DEFAULT_P_LADDER_POLICY_GATE_UNITREE = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/ladder_policy/"
    "p_ladder_policy_gate_unitree_g1.json"
)
DEFAULT_P_LADDER_POLICY_GATE_NEW_EMBODIMENT = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/ladder_policy/"
    "p_ladder_policy_gate_new_embodiment.json"
)
DEFAULT_D_LADDER_POLICY_GATE_UNITREE = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/ladder_policy/"
    "d_ladder_policy_gate_unitree_g1.json"
)
DEFAULT_D_LADDER_POLICY_GATE_NEW_EMBODIMENT = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/ladder_policy/"
    "d_ladder_policy_gate_new_embodiment.json"
)
DEFAULT_D_LADDER_ADMISSION_GATE_UNITREE = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/d_ladder_policy_gate_unitree_g1.json"
)
DEFAULT_D_LADDER_ADMISSION_GATE_NEW_EMBODIMENT = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/"
    "d_ladder_policy_gate_new_embodiment.json"
)
DEFAULT_DATASET_SOURCE_REGISTRY = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/dataset_source_registry.json"
)

DEFAULT_UNITREE_P_ROOTS: tuple[Path, ...] = (
    Path("agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/p"),
)
DEFAULT_NEW_EMBODIMENT_P_ROOTS: tuple[Path, ...] = (
    Path("agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/p"),
    Path("agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/p_smoke_check"),
)
DEFAULT_UNITREE_D_ROOTS: tuple[Path, ...] = (
    Path("agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/d"),
)
DEFAULT_NEW_EMBODIMENT_D_ROOTS: tuple[Path, ...] = (
    Path("agent/artifacts/gr00t_anchor_controller_recap/new_embodiment/d"),
)

FINAL_ATTRIBUTION_JSON_NAME = "final_attribution_matrix.json"
BRANCH_COMPARISON_JSON_NAME = "branch_comparison_pack.json"
WAVE_FREEZE_JSON_NAME = "wave_freeze_manifest.json"

TASK18_EVIDENCE_SCHEMA_VERSION = "sisyphus_task_evidence_v1"
TASK18_EVIDENCE_ARTIFACT_KIND = "task_18_attribution_pack_evidence"
FINAL_ATTRIBUTION_SCHEMA_VERSION = "gr00t_recap_final_attribution_matrix_v1"
FINAL_ATTRIBUTION_ARTIFACT_KIND = "gr00t_recap_final_attribution_matrix"
BRANCH_COMPARISON_SCHEMA_VERSION = "gr00t_recap_branch_comparison_pack_v1"
BRANCH_COMPARISON_ARTIFACT_KIND = "gr00t_recap_branch_comparison_pack"
WAVE_FREEZE_SCHEMA_VERSION = "gr00t_recap_wave_freeze_manifest_v1"
WAVE_FREEZE_ARTIFACT_KIND = "gr00t_recap_wave_freeze_manifest"

BRANCH_UNITREE_G1 = "UNITREE_G1"
BRANCH_NEW_EMBODIMENT = "NEW_EMBODIMENT"

BRANCH_KEY_UNITREE_G1 = "unitree_g1"
BRANCH_KEY_NEW_EMBODIMENT = "new_embodiment"

AXIS_P = "P"
AXIS_D = "D"

P_RUNG_ORDER: tuple[str, ...] = ("P0", "P1", "P2", "P3")
D_RUNG_ORDER: tuple[str, ...] = ("D0", "D1", "D2", "D3", "D4")
D_EFFECTIVE_RUNG_ORDER: tuple[str, ...] = ("D1", "D2", "D3")

HYPOTHESIS_STRENGTH_ORDER = {
    "strong": 4,
    "moderate": 3,
    "weak": 2,
    "not_supported": 1,
    "insufficient_evidence": 0,
}
METRIC_EPS = 1e-9


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import


@dataclass(frozen=True)
class BranchSpec:
    branch_key: str
    embodiment_tag: str
    branch_scope: str
    public_anchor_comparable: bool
    official_comparable_line: bool
    internal_only_comparable_line: bool
    p_roots: tuple[Path, ...]
    d_roots: tuple[Path, ...]


BRANCH_SPECS: dict[str, BranchSpec] = {
    BRANCH_KEY_UNITREE_G1: BranchSpec(
        branch_key=BRANCH_KEY_UNITREE_G1,
        embodiment_tag=BRANCH_UNITREE_G1,
        branch_scope="official_public_anchor_line",
        public_anchor_comparable=True,
        official_comparable_line=True,
        internal_only_comparable_line=False,
        p_roots=DEFAULT_UNITREE_P_ROOTS,
        d_roots=DEFAULT_UNITREE_D_ROOTS,
    ),
    BRANCH_KEY_NEW_EMBODIMENT: BranchSpec(
        branch_key=BRANCH_KEY_NEW_EMBODIMENT,
        embodiment_tag=BRANCH_NEW_EMBODIMENT,
        branch_scope="branch_internal_only",
        public_anchor_comparable=False,
        official_comparable_line=False,
        internal_only_comparable_line=True,
        p_roots=DEFAULT_NEW_EMBODIMENT_P_ROOTS,
        d_roots=DEFAULT_NEW_EMBODIMENT_D_ROOTS,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_recap_attribution_pack.py",
        description=(
            "Aggregate formal tasks 1-17 artifacts into the final attribution matrix, "
            "branch comparison pack, and wave freeze manifest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    _ = parser.add_argument(
        "--task1-evidence", type=Path, default=DEFAULT_TASK1_EVIDENCE
    )
    _ = parser.add_argument(
        "--task12-evidence", type=Path, default=DEFAULT_TASK12_EVIDENCE
    )
    _ = parser.add_argument(
        "--task14-evidence", type=Path, default=DEFAULT_TASK14_EVIDENCE
    )
    _ = parser.add_argument(
        "--task15-evidence", type=Path, default=DEFAULT_TASK15_EVIDENCE
    )
    _ = parser.add_argument(
        "--task16-evidence", type=Path, default=DEFAULT_TASK16_EVIDENCE
    )
    _ = parser.add_argument(
        "--task17-evidence", type=Path, default=DEFAULT_TASK17_EVIDENCE
    )
    _ = parser.add_argument(
        "--checkpoint-provenance-report",
        type=Path,
        default=DEFAULT_CHECKPOINT_PROVENANCE_REPORT,
    )
    _ = parser.add_argument(
        "--dual-branch-scorecard-json",
        type=Path,
        default=DEFAULT_DUAL_BRANCH_SCORECARD_JSON,
    )
    _ = parser.add_argument(
        "--p-ladder-policy-gate-unitree",
        type=Path,
        default=DEFAULT_P_LADDER_POLICY_GATE_UNITREE,
    )
    _ = parser.add_argument(
        "--p-ladder-policy-gate-new-embodiment",
        type=Path,
        default=DEFAULT_P_LADDER_POLICY_GATE_NEW_EMBODIMENT,
    )
    _ = parser.add_argument(
        "--d-ladder-policy-gate-unitree",
        type=Path,
        default=DEFAULT_D_LADDER_POLICY_GATE_UNITREE,
    )
    _ = parser.add_argument(
        "--d-ladder-policy-gate-new-embodiment",
        type=Path,
        default=DEFAULT_D_LADDER_POLICY_GATE_NEW_EMBODIMENT,
    )
    _ = parser.add_argument(
        "--d-ladder-admission-gate-unitree",
        type=Path,
        default=DEFAULT_D_LADDER_ADMISSION_GATE_UNITREE,
    )
    _ = parser.add_argument(
        "--d-ladder-admission-gate-new-embodiment",
        type=Path,
        default=DEFAULT_D_LADDER_ADMISSION_GATE_NEW_EMBODIMENT,
    )
    _ = parser.add_argument(
        "--dataset-source-registry-json",
        type=Path,
        default=DEFAULT_DATASET_SOURCE_REGISTRY,
    )
    _ = parser.add_argument(
        "--task18-evidence-path",
        type=Path,
        default=DEFAULT_TASK18_EVIDENCE_PATH,
    )
    _ = parser.add_argument(
        "--unitree-p-root",
        dest="unitree_p_roots",
        type=Path,
        action="append",
        default=None,
    )
    _ = parser.add_argument(
        "--new-embodiment-p-root",
        dest="new_embodiment_p_roots",
        type=Path,
        action="append",
        default=None,
    )
    _ = parser.add_argument(
        "--unitree-d-root",
        dest="unitree_d_roots",
        type=Path,
        action="append",
        default=None,
    )
    _ = parser.add_argument(
        "--new-embodiment-d-root",
        dest="new_embodiment_d_roots",
        type=Path,
        action="append",
        default=None,
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _resolve_path(path: Path | str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    return raw.resolve()


def _validate_existing_file(path: Path | str, *, arg_name: str) -> Path:
    resolved = _resolve_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{arg_name} does not exist: {resolved}")
    return resolved


def _validate_output_dir(path: Path | str) -> Path:
    resolved = _resolve_path(path)
    return state_conditioned_bucket_a_import.validate_output_dir(resolved)


def _prepare_output_file(path: Path | str) -> Path:
    resolved = _resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path | str, *, arg_name: str) -> dict[str, Any]:
    resolved = _validate_existing_file(path, arg_name=arg_name)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{arg_name} must contain a JSON object")
    return cast(dict[str, Any], dict(payload))


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel_repo(path: Path | str | None) -> str | None:
    if path is None:
        return None
    resolved = _resolve_path(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return _rel_repo(value) or str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return cast(Mapping[str, Any], value)


def _as_list(value: object, *, field_name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)


def _as_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(value).__name__}")
    return bool(value)


def _as_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _as_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric, got {type(value).__name__}")
    return float(value)


def _round_float(value: float, *, digits: int = 8) -> float:
    return float(round(float(value), digits))


def _sorted_unique(items: Iterable[str]) -> list[str]:
    return sorted({str(item) for item in items})


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _artifact_path_from_branch_entry(
    branch_entry: Mapping[str, Any],
    key: str,
    *,
    field_name: str,
) -> Path:
    payload = _as_mapping(branch_entry.get(key, {}), field_name=field_name)
    raw_path = _as_str(
        payload.get("artifact_path"), field_name=f"{field_name}.artifact_path"
    )
    return _validate_existing_file(raw_path, arg_name=f"{field_name}.artifact_path")


def _select_branch_entry(
    dual_branch_scorecard: Mapping[str, Any],
    *,
    branch_key: str,
) -> Mapping[str, Any]:
    branches = _as_list(
        dual_branch_scorecard.get("branches", []), field_name="branches"
    )
    for index, raw in enumerate(branches):
        entry = _as_mapping(raw, field_name=f"branches[{index}]")
        if str(entry.get("branch_key")) == branch_key:
            return entry
    raise ValueError(f"dual-branch scorecard missing branch_key={branch_key}")


def _candidate_roots(
    user_roots: Sequence[Path] | None, defaults: Sequence[Path]
) -> tuple[Path, ...]:
    if user_roots:
        return tuple(user_roots)
    return tuple(defaults)


def _scorecard_is_formal(
    payload: Mapping[str, Any],
    *,
    spec: BranchSpec,
    axis: str,
    rung: str,
) -> tuple[bool, str | None]:
    if str(payload.get("axis")) != axis:
        return False, "axis_mismatch"
    if str(payload.get("branch_key")) != spec.branch_key:
        return False, "branch_key_mismatch"
    if str(payload.get("branch")) != spec.embodiment_tag:
        return False, "branch_tag_mismatch"
    if str(payload.get("branch_scope")) != spec.branch_scope:
        return False, "branch_scope_mismatch"
    if str(payload.get("rung")) != rung:
        return False, "rung_mismatch"
    if payload.get("public_anchor_comparable") is not spec.public_anchor_comparable:
        return False, "public_anchor_comparable_mismatch"
    if not isinstance(payload.get("frozen_formal_protocol"), Mapping):
        return False, "missing_frozen_formal_protocol"
    if not isinstance(payload.get("source_artifacts"), Mapping):
        return False, "missing_source_artifacts"
    if not isinstance(payload.get("comparability"), Mapping):
        return False, "missing_comparability"
    return True, None


def _discover_ladder_rung(
    *,
    spec: BranchSpec,
    axis: str,
    rung: str,
    roots: Sequence[Path],
) -> dict[str, Any]:
    candidate_paths: list[Path] = []
    seen: set[Path] = set()
    excluded_candidates: list[dict[str, str]] = []
    resolved_roots = tuple(_resolve_path(root) for root in roots)
    root_priority = {root: index for index, root in enumerate(resolved_roots)}

    for resolved_root in resolved_roots:
        exact = resolved_root / rung / "scorecard.json"
        if exact not in seen:
            seen.add(exact)
            candidate_paths.append(exact)
        if resolved_root.exists() and resolved_root.is_dir():
            for discovered in sorted(resolved_root.rglob("scorecard.json")):
                if discovered.parent.name != rung:
                    continue
                resolved = discovered.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidate_paths.append(resolved)

    valid_candidates: list[dict[str, Any]] = []
    for candidate in candidate_paths:
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            payload = _read_json(
                candidate, arg_name=f"{spec.branch_key}.{axis}.{rung}.scorecard"
            )
        except Exception as exc:
            excluded_candidates.append(
                {
                    "path": _rel_repo(candidate) or str(candidate),
                    "reason": f"invalid_json:{_exception_message(exc)}",
                }
            )
            continue
        is_formal, reason = _scorecard_is_formal(
            payload,
            spec=spec,
            axis=axis,
            rung=rung,
        )
        if not is_formal:
            excluded_candidates.append(
                {
                    "path": _rel_repo(candidate) or str(candidate),
                    "reason": str(reason),
                }
            )
            continue
        valid_candidates.append(
            {
                "path": candidate.resolve(),
                "payload": payload,
                "root_priority": min(
                    (
                        priority
                        for root_path, priority in root_priority.items()
                        if candidate.resolve().is_relative_to(root_path)
                    ),
                    default=len(resolved_roots),
                ),
                "discovery_mode": (
                    "exact"
                    if candidate.parent.name == rung
                    and candidate.parent.parent in set(resolved_roots)
                    else "fallback_glob"
                ),
            }
        )

    exact_matches = [
        item
        for item in valid_candidates
        if any(
            item["path"] == (_resolve_path(root) / rung / "scorecard.json")
            for root in roots
        )
    ]
    selected_candidates = exact_matches if exact_matches else valid_candidates
    if len(selected_candidates) > 1:
        selected_candidates = sorted(
            selected_candidates,
            key=lambda item: (
                int(cast(int, item.get("root_priority", len(resolved_roots)))),
                _rel_repo(cast(Path, item["path"])) or str(item["path"]),
            ),
        )
        best_priority = int(
            cast(int, selected_candidates[0].get("root_priority", len(resolved_roots)))
        )
        preferred_candidates = [
            item
            for item in selected_candidates
            if int(cast(int, item.get("root_priority", len(resolved_roots))))
            == best_priority
        ]
        if len(preferred_candidates) == 1:
            selected_candidates = preferred_candidates
        else:
            selected_candidates = preferred_candidates

    if len(selected_candidates) > 1:
        candidate_list = ", ".join(
            sorted(
                _rel_repo(cast(Path, item["path"])) or str(item["path"])
                for item in selected_candidates
            )
        )
        raise ValueError(
            f"ambiguous formal scorecards for {spec.branch_key} {axis}{rung}: {candidate_list}"
        )
    if not selected_candidates:
        return {
            "status": "MISSING",
            "scorecard_path": None,
            "manifest_path": None,
            "scorecard": None,
            "manifest": None,
            "excluded_candidates": excluded_candidates,
            "discovery_mode": "missing",
        }

    selected = selected_candidates[0]
    scorecard_path = cast(Path, selected["path"])
    manifest_path = scorecard_path.with_name("manifest.json")
    if not manifest_path.exists() or not manifest_path.is_file():
        raise ValueError(
            f"missing manifest.json paired with scorecard: {scorecard_path}"
        )
    manifest = _read_json(
        manifest_path, arg_name=f"{spec.branch_key}.{axis}.{rung}.manifest"
    )
    if str(manifest.get("rung")) != rung:
        raise ValueError(f"manifest rung mismatch for {manifest_path}")
    if str(manifest.get("axis")) != axis:
        raise ValueError(f"manifest axis mismatch for {manifest_path}")
    if str(manifest.get("branch_key")) != spec.branch_key:
        raise ValueError(f"manifest branch_key mismatch for {manifest_path}")
    return {
        "status": "FOUND",
        "scorecard_path": scorecard_path,
        "manifest_path": manifest_path,
        "scorecard": cast(dict[str, Any], selected["payload"]),
        "manifest": manifest,
        "excluded_candidates": excluded_candidates,
        "discovery_mode": str(selected["discovery_mode"]),
    }


def _discover_branch_ladder(
    *,
    spec: BranchSpec,
    axis: str,
    roots: Sequence[Path],
) -> dict[str, Any]:
    rung_order = P_RUNG_ORDER if axis == AXIS_P else D_RUNG_ORDER
    discoveries: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, str]] = []
    for rung in rung_order:
        discovery = _discover_ladder_rung(
            spec=spec,
            axis=axis,
            rung=rung,
            roots=roots,
        )
        discoveries[rung] = discovery
        excluded.extend(cast(list[dict[str, str]], discovery["excluded_candidates"]))
    return {
        "axis": axis,
        "roots": [_rel_repo(root) or str(_resolve_path(root)) for root in roots],
        "discoveries": discoveries,
        "excluded_candidates": excluded,
        "available_rungs": [
            rung
            for rung, item in discoveries.items()
            if str(item.get("status")) == "FOUND"
        ],
        "missing_rungs": [
            rung
            for rung, item in discoveries.items()
            if str(item.get("status")) != "FOUND"
        ],
    }


def _baseline_metric(
    payload: Mapping[str, Any],
    key: str,
) -> float | None:
    baseline = _as_mapping(
        payload.get("baseline_metrics", {}), field_name="baseline_metrics"
    )
    value = baseline.get(key)
    if value is None:
        return None
    return _as_float(value, field_name=f"baseline_metrics.{key}")


def _is_effective_p_rung(scorecard: Mapping[str, Any]) -> bool:
    if str(scorecard.get("status")) != "PASS":
        return False
    report = _as_mapping(
        scorecard.get("positive_slope_report", {}),
        field_name="positive_slope_report",
    )
    if bool(report.get("positive_slope_detected", False)):
        return True
    qualifying = _as_list(
        report.get("qualifying_metric_names", []),
        field_name="positive_slope_report.qualifying_metric_names",
    )
    return bool(qualifying)


def _is_effective_d_rung(scorecard: Mapping[str, Any]) -> bool:
    if str(scorecard.get("status")) != "PASS":
        return False
    execution_disposition = str(scorecard.get("execution_disposition", ""))
    if execution_disposition.startswith("BLOCK"):
        return False
    success_rate = _as_float(
        scorecard.get("success_rate", 0.0), field_name="success_rate"
    )
    baseline_success = _baseline_metric(scorecard, "success_rate")
    if baseline_success is not None and success_rate > baseline_success + METRIC_EPS:
        return True
    condition_ratio = _as_float(
        scorecard.get("condition_flip_response_ratio", 0.0),
        field_name="condition_flip_response_ratio",
    )
    baseline_condition = _baseline_metric(scorecard, "condition_flip_response_ratio")
    if (
        baseline_condition is not None
        and condition_ratio > baseline_condition + METRIC_EPS
    ):
        return True
    teacher_gap_value = scorecard.get("teacher_gap")
    if teacher_gap_value is None:
        teacher_gap_value = scorecard.get("teacher_student_gap")
    baseline_teacher_gap = _baseline_metric(scorecard, "teacher_gap")
    if baseline_teacher_gap is None:
        baseline_teacher_gap = _baseline_metric(scorecard, "teacher_student_gap")
    if teacher_gap_value is not None and baseline_teacher_gap is not None:
        teacher_gap = _as_float(teacher_gap_value, field_name="teacher_gap")
        if teacher_gap < baseline_teacher_gap - METRIC_EPS:
            return True
    return False


def _build_parameter_summary(
    discoveries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    effective_rungs: list[str] = []
    blocked_rungs: list[str] = []
    checked_rungs: list[str] = []
    per_rung: dict[str, dict[str, Any]] = {}
    for rung in P_RUNG_ORDER:
        discovery = _as_mapping(discoveries.get(rung, {}), field_name=f"p.{rung}")
        if str(discovery.get("status")) != "FOUND":
            continue
        scorecard = _as_mapping(
            discovery.get("scorecard", {}), field_name=f"p.{rung}.scorecard"
        )
        checked_rungs.append(rung)
        if str(scorecard.get("status")) == "BLOCK":
            blocked_rungs.append(rung)
        effective = rung in {"P1", "P2"} and _is_effective_p_rung(scorecard)
        if effective:
            effective_rungs.append(rung)
        positive_slope_report = _as_mapping(
            scorecard.get("positive_slope_report", {}),
            field_name=f"p.{rung}.positive_slope_report",
        )
        per_rung[rung] = {
            "status": str(scorecard.get("status")),
            "promotion_status": str(scorecard.get("promotion_status")),
            "effective": effective,
            "qualifying_metric_names": [
                str(item)
                for item in _as_list(
                    positive_slope_report.get("qualifying_metric_names", []),
                    field_name=f"p.{rung}.positive_slope_report.qualifying_metric_names",
                )
            ],
            "blocking_reasons": [
                str(item)
                for item in _as_list(
                    scorecard.get("blocking_reasons", []),
                    field_name=f"p.{rung}.blocking_reasons",
                )
            ],
            "scorecard_path": _rel_repo(cast(Path, discovery.get("scorecard_path"))),
            "manifest_path": _rel_repo(cast(Path, discovery.get("manifest_path"))),
        }
    assessed_rungs = [rung for rung in ("P1", "P2") if rung in checked_rungs]
    return {
        "assessed_rungs": assessed_rungs,
        "assessment_complete": set(assessed_rungs) == {"P1", "P2"},
        "available_rungs": checked_rungs,
        "missing_rungs": [rung for rung in P_RUNG_ORDER if rung not in checked_rungs],
        "effective_rungs": effective_rungs,
        "blocked_rungs": blocked_rungs,
        "per_rung": per_rung,
    }


def _build_data_summary(
    discoveries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    effective_rungs: list[str] = []
    blocked_rungs: list[str] = []
    assessed_rungs: list[str] = []
    per_rung: dict[str, dict[str, Any]] = {}
    for rung in D_RUNG_ORDER:
        discovery = _as_mapping(discoveries.get(rung, {}), field_name=f"d.{rung}")
        if str(discovery.get("status")) != "FOUND":
            continue
        scorecard = _as_mapping(
            discovery.get("scorecard", {}), field_name=f"d.{rung}.scorecard"
        )
        if rung in D_EFFECTIVE_RUNG_ORDER:
            assessed_rungs.append(rung)
        if str(scorecard.get("status")) == "BLOCK":
            blocked_rungs.append(rung)
        effective = rung in D_EFFECTIVE_RUNG_ORDER and _is_effective_d_rung(scorecard)
        if effective:
            effective_rungs.append(rung)
        per_rung[rung] = {
            "status": str(scorecard.get("status")),
            "promotion_status": str(scorecard.get("promotion_status")),
            "effective": effective,
            "execution_disposition": scorecard.get("execution_disposition"),
            "branch_local_only_rung": bool(
                scorecard.get("branch_local_only_rung", False)
            ),
            "blocking_reasons": [
                str(item)
                for item in _as_list(
                    scorecard.get("blocking_reasons", []),
                    field_name=f"d.{rung}.blocking_reasons",
                )
            ],
            "scorecard_path": _rel_repo(cast(Path, discovery.get("scorecard_path"))),
            "manifest_path": _rel_repo(cast(Path, discovery.get("manifest_path"))),
        }
    return {
        "assessed_rungs": assessed_rungs,
        "assessment_complete": set(assessed_rungs) == set(D_EFFECTIVE_RUNG_ORDER),
        "available_rungs": [
            rung
            for rung in D_RUNG_ORDER
            if str(
                _as_mapping(discoveries.get(rung, {}), field_name=f"d.{rung}").get(
                    "status"
                )
            )
            == "FOUND"
        ],
        "missing_rungs": [
            rung
            for rung in D_RUNG_ORDER
            if str(
                _as_mapping(discoveries.get(rung, {}), field_name=f"d.{rung}").get(
                    "status"
                )
            )
            != "FOUND"
        ],
        "effective_rungs": effective_rungs,
        "blocked_rungs": blocked_rungs,
        "per_rung": per_rung,
    }


def _hypothesis_payload(
    *,
    strength: str,
    summary: str,
    reason_codes: Sequence[str],
    supporting_artifacts: Sequence[str],
    evidence_scope: str,
) -> dict[str, Any]:
    return {
        "strength": strength,
        "summary": summary,
        "reason_codes": _sorted_unique(reason_codes),
        "supporting_artifacts": _sorted_unique(supporting_artifacts),
        "evidence_scope": evidence_scope,
    }


def _healthy_replay(reachability_payload: Mapping[str, Any]) -> bool:
    replay_upper_bound = _as_mapping(
        reachability_payload.get("replay_upper_bound", {}),
        field_name="teacher_reachability.replay_upper_bound",
    )
    success_count = replay_upper_bound.get("success_count")
    if success_count is None:
        return False
    return _as_float(success_count, field_name="replay_upper_bound.success_count") > 0.0


def _build_branch_hypotheses(
    *,
    spec: BranchSpec,
    branch_entry: Mapping[str, Any],
    public_anchor_payload: Mapping[str, Any] | None,
    controller_payload: Mapping[str, Any],
    action_payload: Mapping[str, Any],
    reachability_payload: Mapping[str, Any],
    parameter_summary: Mapping[str, Any],
    data_summary: Mapping[str, Any],
    checkpoint_provenance: Mapping[str, Any],
    checkpoint_provenance_path: Path,
) -> dict[str, Any]:
    stack_reason_codes: list[str] = []
    stack_artifacts: list[str] = []
    controller_reason_codes: list[str] = []
    controller_artifacts: list[str] = []
    parameter_reason_codes: list[str] = []
    parameter_artifacts: list[str] = []
    data_reason_codes: list[str] = []
    data_artifacts: list[str] = []
    recap_reason_codes: list[str] = []
    recap_artifacts: list[str] = []

    prerequisite_status = _as_mapping(
        branch_entry.get("prerequisite_status", {}),
        field_name="branch_entry.prerequisite_status",
    )
    failed_checks = {
        str(item)
        for item in _as_list(
            prerequisite_status.get("failed_checks", []),
            field_name="branch_entry.prerequisite_status.failed_checks",
        )
    }

    checkpoint_status = str(checkpoint_provenance.get("formal_eligibility", ""))
    loadability_status = str(checkpoint_provenance.get("loadability_status", ""))
    if (
        checkpoint_status != "ALLOW"
        or loadability_status != "LOADABLE_CHECKPOINT_CONFIRMED"
    ):
        stack_reason_codes.append("checkpoint_provenance_not_formally_loadable")
        stack_artifacts.append(
            _rel_repo(checkpoint_provenance_path) or str(checkpoint_provenance_path)
        )

    if public_anchor_payload is not None:
        success_count = _as_float(
            public_anchor_payload.get("success_count", 0),
            field_name="public_anchor.success_count",
        )
        systemic_break_flags = [
            str(item)
            for item in _as_list(
                public_anchor_payload.get("systemic_break_flags", []),
                field_name="public_anchor.systemic_break_flags",
            )
        ]
        if success_count <= 0.0:
            stack_reason_codes.append("public_anchor_zero_or_abnormally_low")
            stack_artifacts.append(
                _rel_repo(
                    _artifact_path_from_branch_entry(
                        branch_entry,
                        "public_anchor_status",
                        field_name="public_anchor_status",
                    )
                )
                or "public_anchor"
            )
        if systemic_break_flags:
            stack_reason_codes.append("public_anchor_systemic_break_flags_present")
            stack_artifacts.append(
                _rel_repo(
                    _artifact_path_from_branch_entry(
                        branch_entry,
                        "public_anchor_status",
                        field_name="public_anchor_status",
                    )
                )
                or "public_anchor"
            )

    if "public_anchor" in failed_checks:
        stack_reason_codes.append("public_anchor_prerequisite_failed")
        stack_artifacts.append(
            _rel_repo(
                _artifact_path_from_branch_entry(
                    branch_entry,
                    "public_anchor_status",
                    field_name="public_anchor_status",
                )
            )
            or "public_anchor"
        )

    if not _healthy_replay(reachability_payload):
        stack_reason_codes.append("replay_upper_bound_low_or_missing")
        stack_artifacts.append(
            _rel_repo(
                _artifact_path_from_branch_entry(
                    branch_entry,
                    "teacher_reachability",
                    field_name="teacher_reachability",
                )
            )
            or "teacher_reachability"
        )

    controller_equivalent = controller_payload.get("equivalent_to_official_unitree_g1")
    if spec.branch_key == BRANCH_KEY_UNITREE_G1 and controller_equivalent is not True:
        controller_reason_codes.append("controller_audit_failed_official_equivalence")
        controller_artifacts.append(
            _rel_repo(
                _artifact_path_from_branch_entry(
                    branch_entry,
                    "controller_equivalence",
                    field_name="controller_equivalence",
                )
            )
            or "controller_audit"
        )
    if spec.branch_key == BRANCH_KEY_NEW_EMBODIMENT:
        if str(controller_payload.get("formal_branch_eligibility", "ALLOW")) != "ALLOW":
            controller_reason_codes.append("new_embodiment_branch_contract_failed")
            controller_artifacts.append(
                _rel_repo(
                    _artifact_path_from_branch_entry(
                        branch_entry,
                        "controller_equivalence",
                        field_name="controller_equivalence",
                    )
                )
                or "controller_audit"
            )

    mismatch_fields = [
        str(item)
        for item in _as_list(
            controller_payload.get("mismatch_fields", []),
            field_name="controller_payload.mismatch_fields",
        )
    ]
    if mismatch_fields:
        controller_reason_codes.append("controller_audit_mismatch_fields_present")
        controller_artifacts.append(
            _rel_repo(
                _artifact_path_from_branch_entry(
                    branch_entry,
                    "controller_equivalence",
                    field_name="controller_equivalence",
                )
            )
            or "controller_audit"
        )

    controller_absorbed_groups = [
        str(item)
        for item in _as_list(
            action_payload.get("controller_absorbed_groups", []),
            field_name="action_payload.controller_absorbed_groups",
        )
    ]
    model_insensitive_groups = [
        str(item)
        for item in _as_list(
            action_payload.get("model_insensitive_groups", []),
            field_name="action_payload.model_insensitive_groups",
        )
    ]
    zero_motion_groups = [
        str(item)
        for item in _as_list(
            _as_mapping(
                action_payload.get("zero_motion_flags", {}),
                field_name="action_payload.zero_motion_flags",
            ).get("all_zero_in_both_groups", []),
            field_name="action_payload.zero_motion_flags.all_zero_in_both_groups",
        )
    ]
    if controller_absorbed_groups:
        controller_reason_codes.append("controller_absorbed_upstream_differences")
        controller_artifacts.append(
            _rel_repo(
                _artifact_path_from_branch_entry(
                    branch_entry, "action_telemetry", field_name="action_telemetry"
                )
            )
            or "action_telemetry"
        )
    if model_insensitive_groups:
        controller_reason_codes.append("model_insensitive_action_groups_present")
        controller_artifacts.append(
            _rel_repo(
                _artifact_path_from_branch_entry(
                    branch_entry, "action_telemetry", field_name="action_telemetry"
                )
            )
            or "action_telemetry"
        )
    if zero_motion_groups:
        controller_reason_codes.append("zero_motion_action_groups_present")
        controller_artifacts.append(
            _rel_repo(
                _artifact_path_from_branch_entry(
                    branch_entry, "action_telemetry", field_name="action_telemetry"
                )
            )
            or "action_telemetry"
        )

    p_effective_rungs = [
        str(item)
        for item in _as_list(
            parameter_summary.get("effective_rungs", []),
            field_name="parameter_summary.effective_rungs",
        )
    ]
    if p_effective_rungs:
        parameter_reason_codes.append("p1_or_p2_effective")
        parameter_reason_codes.extend(
            [f"effective_parameter_rung:{rung}" for rung in p_effective_rungs]
        )
        for rung in p_effective_rungs:
            per_rung = _as_mapping(
                _as_mapping(
                    parameter_summary.get("per_rung", {}),
                    field_name="parameter_summary.per_rung",
                ).get(rung, {}),
                field_name=f"parameter_summary.per_rung.{rung}",
            )
            scorecard_path = per_rung.get("scorecard_path")
            if scorecard_path:
                parameter_artifacts.append(str(scorecard_path))
    elif bool(parameter_summary.get("assessment_complete", False)):
        parameter_reason_codes.append("p1_p2_assessed_but_unmoved")
    else:
        parameter_reason_codes.append("p_ladder_assessment_partial")

    d_effective_rungs = [
        str(item)
        for item in _as_list(
            data_summary.get("effective_rungs", []),
            field_name="data_summary.effective_rungs",
        )
    ]
    if d_effective_rungs:
        data_reason_codes.append("d1_to_d3_effective")
        data_reason_codes.extend(
            [f"effective_data_rung:{rung}" for rung in d_effective_rungs]
        )
        for rung in d_effective_rungs:
            per_rung = _as_mapping(
                _as_mapping(
                    data_summary.get("per_rung", {}), field_name="data_summary.per_rung"
                ).get(rung, {}),
                field_name=f"data_summary.per_rung.{rung}",
            )
            scorecard_path = per_rung.get("scorecard_path")
            if scorecard_path:
                data_artifacts.append(str(scorecard_path))
    elif bool(data_summary.get("assessment_complete", False)):
        data_reason_codes.append("d1_to_d3_assessed_but_unmoved")
    else:
        data_reason_codes.append("d_ladder_assessment_partial")

    teacher_reachable = bool(
        reachability_payload.get("allow_formal_ladders", False)
    ) and bool(
        _as_list(
            reachability_payload.get("teacher_reachable_scene_ids", []),
            field_name="teacher_reachability.teacher_reachable_scene_ids",
        )
    )
    if teacher_reachable and _healthy_replay(reachability_payload):
        if not p_effective_rungs and not d_effective_rungs:
            recap_reason_codes.append("teacher_reachable_and_replay_healthy")
            recap_artifacts.append(
                _rel_repo(
                    _artifact_path_from_branch_entry(
                        branch_entry,
                        "teacher_reachability",
                        field_name="teacher_reachability",
                    )
                )
                or "teacher_reachability"
            )
            if bool(parameter_summary.get("assessment_complete", False)) and bool(
                data_summary.get("assessment_complete", False)
            ):
                recap_reason_codes.append("parameter_and_data_ladders_unmoved")
            else:
                recap_reason_codes.append(
                    "parameter_or_data_ladders_partially_materialized"
                )
        else:
            recap_reason_codes.append(
                "teacher_reachable_but_parameter_or_data_signal_present"
            )
    else:
        recap_reason_codes.append(
            "teacher_reachability_or_replay_not_yet_strong_enough"
        )

    if stack_reason_codes:
        stack_strength = "strong"
    elif (
        checkpoint_status == "ALLOW"
        and loadability_status == "LOADABLE_CHECKPOINT_CONFIRMED"
    ):
        stack_strength = "not_supported"
        stack_reason_codes.append(
            "public_anchor_controller_and_replay_do_not_point_first_to_stack"
        )
    else:
        stack_strength = "insufficient_evidence"

    if (
        "controller_audit_failed_official_equivalence" in controller_reason_codes
        or "new_embodiment_branch_contract_failed" in controller_reason_codes
    ):
        controller_strength = "strong"
    elif controller_absorbed_groups or mismatch_fields:
        controller_strength = "moderate"
    elif controller_reason_codes:
        controller_strength = "weak"
    else:
        controller_strength = "not_supported"
        controller_reason_codes.append("controller_audit_passed_without_material_drift")

    if p_effective_rungs:
        parameter_strength = "strong"
    elif bool(parameter_summary.get("assessment_complete", False)):
        parameter_strength = "not_supported"
    else:
        parameter_strength = "insufficient_evidence"

    if d_effective_rungs:
        data_strength = "strong"
    elif bool(data_summary.get("assessment_complete", False)):
        data_strength = "not_supported"
    else:
        data_strength = "insufficient_evidence"

    if (
        teacher_reachable
        and _healthy_replay(reachability_payload)
        and not p_effective_rungs
        and not d_effective_rungs
        and bool(parameter_summary.get("assessment_complete", False))
        and bool(data_summary.get("assessment_complete", False))
    ):
        recap_strength = "strong"
    elif (
        teacher_reachable
        and _healthy_replay(reachability_payload)
        and not p_effective_rungs
        and not d_effective_rungs
    ):
        recap_strength = "moderate"
    elif teacher_reachable:
        recap_strength = "weak"
    else:
        recap_strength = "insufficient_evidence"

    stack_hypothesis = _hypothesis_payload(
        strength=stack_strength,
        summary=(
            "公开锚点/回放/可加载性首先指向 stack 或 env/controller wiring 问题。"
            if stack_strength == "strong"
            else "现有 formal 证据没有把主因首先压到 stack 层。"
        ),
        reason_codes=stack_reason_codes,
        supporting_artifacts=stack_artifacts,
        evidence_scope="official_public_anchor_only"
        if spec.public_anchor_comparable
        else "internal_branch_only",
    )
    controller_hypothesis = _hypothesis_payload(
        strength=controller_strength,
        summary=(
            "controller audit 或 action-chain telemetry 显示 controller 语义/吞差异现象，值得优先复查。"
            if controller_strength in {"strong", "moderate"}
            else "controller formal 审计目前没有成为第一主因。"
        ),
        reason_codes=controller_reason_codes,
        supporting_artifacts=controller_artifacts,
        evidence_scope="branch_local_structural_diagnostics",
    )
    parameter_scope_hypothesis = _hypothesis_payload(
        strength=parameter_strength,
        summary=(
            "P1/P2 出现正向信号，更像参数更新范围/表征可塑性限制。"
            if parameter_strength == "strong"
            else "当前 P1/P2 未给出足够正向信号。"
        ),
        reason_codes=parameter_reason_codes,
        supporting_artifacts=parameter_artifacts,
        evidence_scope="branch_local_parameter_ladder",
    )
    data_distribution_hypothesis = _hypothesis_payload(
        strength=data_strength,
        summary=(
            "D1-D3 出现正向信号，更像数据分布过窄。"
            if data_strength == "strong"
            else "当前 D1-D3 未给出足够正向信号。"
        ),
        reason_codes=data_reason_codes,
        supporting_artifacts=data_artifacts,
        evidence_scope="branch_local_data_ladder",
    )
    recap_interface_hypothesis = _hypothesis_payload(
        strength=recap_strength,
        summary=(
            "teacher reachable 且 replay 健康，但参数/数据都不动，更像 RECAP 注入点或 action 解释问题。"
            if recap_strength in {"strong", "moderate"}
            else "现有证据不足以把主因首先压到 RECAP interface。"
        ),
        reason_codes=recap_reason_codes,
        supporting_artifacts=recap_artifacts,
        evidence_scope="branch_local_teacher_reachability_plus_ladders",
    )

    ordered_candidates = [
        ("stack", stack_hypothesis, "fix_stack_or_replay_health_before_next_wave"),
        (
            "controller",
            controller_hypothesis,
            "audit_controller_semantics_and_action_chain_before_next_wave",
        ),
        (
            "parameter",
            parameter_scope_hypothesis,
            "run_confirmatory_parameter_scope_followup",
        ),
        (
            "data",
            data_distribution_hypothesis,
            "run_confirmatory_data_distribution_followup",
        ),
        (
            "recap_interface",
            recap_interface_hypothesis,
            "audit_recap_injection_action_target_and_relative_action_interpretation",
        ),
    ]
    highest = max(
        ordered_candidates,
        key=lambda item: HYPOTHESIS_STRENGTH_ORDER[str(item[1]["strength"])],
    )
    top_strength = HYPOTHESIS_STRENGTH_ORDER[str(highest[1]["strength"])]
    if top_strength <= HYPOTHESIS_STRENGTH_ORDER["not_supported"]:
        recommended_next_step = str(
            branch_entry.get(
                "recommended_next_step", "collect_missing_formal_rungs_before_new_wave"
            )
        )
    else:
        recommended_next_step = highest[2]

    return {
        "stack_hypothesis": stack_hypothesis,
        "controller_hypothesis": controller_hypothesis,
        "parameter_scope_hypothesis": parameter_scope_hypothesis,
        "data_distribution_hypothesis": data_distribution_hypothesis,
        "recap_interface_hypothesis": recap_interface_hypothesis,
        "recommended_next_step": recommended_next_step,
    }


def _branch_structural_summary(
    *,
    branch_entry: Mapping[str, Any],
    parameter_summary: Mapping[str, Any],
    data_summary: Mapping[str, Any],
    hypotheses: Mapping[str, Any],
    discoveries: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "public_anchor_comparable": bool(
            branch_entry.get("public_anchor_comparable", False)
        ),
        "official_comparable_line": bool(
            branch_entry.get("official_comparable_line", False)
        ),
        "internal_only_comparable_line": bool(
            branch_entry.get("internal_only_comparable_line", False)
        ),
        "diagnostic_summary": dict(
            _as_mapping(
                branch_entry.get("diagnostic_summary", {}),
                field_name="branch_entry.diagnostic_summary",
            )
        ),
        "prerequisite_status": dict(
            _as_mapping(
                branch_entry.get("prerequisite_status", {}),
                field_name="branch_entry.prerequisite_status",
            )
        ),
        "parameter_ladder": dict(parameter_summary),
        "data_ladder": dict(data_summary),
        "selected_hypothesis_strengths": {
            "stack_hypothesis": _as_mapping(
                hypotheses.get("stack_hypothesis", {}), field_name="stack_hypothesis"
            ).get("strength"),
            "controller_hypothesis": _as_mapping(
                hypotheses.get("controller_hypothesis", {}),
                field_name="controller_hypothesis",
            ).get("strength"),
            "parameter_scope_hypothesis": _as_mapping(
                hypotheses.get("parameter_scope_hypothesis", {}),
                field_name="parameter_scope_hypothesis",
            ).get("strength"),
            "data_distribution_hypothesis": _as_mapping(
                hypotheses.get("data_distribution_hypothesis", {}),
                field_name="data_distribution_hypothesis",
            ).get("strength"),
            "recap_interface_hypothesis": _as_mapping(
                hypotheses.get("recap_interface_hypothesis", {}),
                field_name="recap_interface_hypothesis",
            ).get("strength"),
        },
        "discovery": dict(discoveries),
    }


def _branch_conclusion_row(
    *,
    spec: BranchSpec,
    hypotheses: Mapping[str, Any],
    branch_entry: Mapping[str, Any],
    dual_branch_scorecard_path: Path,
) -> dict[str, Any]:
    stack_strength = _as_mapping(
        hypotheses.get("stack_hypothesis", {}), field_name="stack_hypothesis"
    ).get("strength")
    parameter_strength = _as_mapping(
        hypotheses.get("parameter_scope_hypothesis", {}),
        field_name="parameter_scope_hypothesis",
    ).get("strength")
    data_strength = _as_mapping(
        hypotheses.get("data_distribution_hypothesis", {}),
        field_name="data_distribution_hypothesis",
    ).get("strength")
    recap_strength = _as_mapping(
        hypotheses.get("recap_interface_hypothesis", {}),
        field_name="recap_interface_hypothesis",
    ).get("strength")
    controller_strength = _as_mapping(
        hypotheses.get("controller_hypothesis", {}),
        field_name="controller_hypothesis",
    ).get("strength")
    return {
        "branch_key": spec.branch_key,
        "branch": spec.embodiment_tag,
        "comparability_label": (
            "official_public_comparable"
            if spec.official_comparable_line
            else "internal_structural_only"
        ),
        "summary": (
            f"{spec.embodiment_tag}：stack={stack_strength}, controller={controller_strength}, "
            f"parameter={parameter_strength}, data={data_strength}, recap_interface={recap_strength}。"
        ),
        "recommended_next_step": str(hypotheses.get("recommended_next_step")),
        "evidence_anchor": {
            "dual_branch_scorecard": _rel_repo(dual_branch_scorecard_path),
            "branch_recommended_next_step": str(
                branch_entry.get("recommended_next_step")
            ),
        },
    }


def _build_comparison_axes(
    branches: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {str(branch["branch_key"]): branch for branch in branches}
    unitree = lookup[BRANCH_KEY_UNITREE_G1]
    new_emb = lookup[BRANCH_KEY_NEW_EMBODIMENT]
    unitree_diag = _as_mapping(
        unitree.get("diagnostic_summary", {}), field_name="unitree.diagnostic_summary"
    )
    new_diag = _as_mapping(
        new_emb.get("diagnostic_summary", {}), field_name="new_emb.diagnostic_summary"
    )
    return [
        {
            "metric": "public_anchor_success_rate",
            "comparability": "official_public_anchor_only",
            "unitree_g1": unitree_diag.get("public_anchor_success_rate"),
            "new_embodiment": None,
            "note": "公开 benchmark 数值只对 UNITREE_G1 官方线成立。",
        },
        {
            "metric": "condition_flip_min_response_ratio",
            "comparability": "internal_structural_only",
            "unitree_g1": unitree_diag.get("condition_flip_min_response_ratio"),
            "new_embodiment": new_diag.get("condition_flip_min_response_ratio"),
            "note": "可用于结构性诊断对照，但不能把 NEW_EMBODIMENT 直接映射到公开 benchmark 阈值。",
        },
        {
            "metric": "teacher_reachable_scene_count",
            "comparability": "internal_structural_only",
            "unitree_g1": unitree_diag.get("teacher_reachable_scene_count"),
            "new_embodiment": new_diag.get("teacher_reachable_scene_count"),
            "note": "两条线都可比较 teacher/replay reachability 结构，但不形成单一总排名。",
        },
        {
            "metric": "teacher_student_branch_match_rate",
            "comparability": "internal_structural_only",
            "unitree_g1": unitree_diag.get("teacher_student_branch_match_rate"),
            "new_embodiment": new_diag.get("teacher_student_branch_match_rate"),
            "note": "只能解释 branch-local 条件结构，不等同公开 success benchmark。",
        },
        {
            "metric": "parameter_scope_hypothesis.strength",
            "comparability": "branch_local_attribution_only",
            "unitree_g1": _as_mapping(
                unitree.get("parameter_scope_hypothesis", {}),
                field_name="unitree.parameter_scope_hypothesis",
            ).get("strength"),
            "new_embodiment": _as_mapping(
                new_emb.get("parameter_scope_hypothesis", {}),
                field_name="new_emb.parameter_scope_hypothesis",
            ).get("strength"),
            "note": "这是 branch-local attribution，不应折叠成跨 branch 的单数值排名。",
        },
        {
            "metric": "data_distribution_hypothesis.strength",
            "comparability": "branch_local_attribution_only",
            "unitree_g1": _as_mapping(
                unitree.get("data_distribution_hypothesis", {}),
                field_name="unitree.data_distribution_hypothesis",
            ).get("strength"),
            "new_embodiment": _as_mapping(
                new_emb.get("data_distribution_hypothesis", {}),
                field_name="new_emb.data_distribution_hypothesis",
            ).get("strength"),
            "note": "用于判断各自 branch 下一步是否优先走数据轴。",
        },
    ]


def _build_shared_structural_findings(
    branches: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    teacher_reachable_all = True
    condition_sensitive_all = True
    common_controller_absorbed: set[str] | None = None
    common_model_insensitive: set[str] | None = None
    for branch in branches:
        diag = _as_mapping(
            branch.get("diagnostic_summary", {}), field_name="branch.diagnostic_summary"
        )
        reachable_count = int(diag.get("teacher_reachable_scene_count", 0) or 0)
        condition_ratio = float(
            diag.get("condition_flip_min_response_ratio", 0.0) or 0.0
        )
        teacher_reachable_all = teacher_reachable_all and reachable_count > 0
        condition_sensitive_all = condition_sensitive_all and condition_ratio > 0.0
        absorbed = {
            str(item)
            for item in _as_list(
                diag.get("action_telemetry_controller_absorbed_groups", []),
                field_name="diagnostic_summary.action_telemetry_controller_absorbed_groups",
            )
        }
        insensitive = {
            str(item)
            for item in _as_list(
                diag.get("action_telemetry_model_insensitive_groups", []),
                field_name="diagnostic_summary.action_telemetry_model_insensitive_groups",
            )
        }
        common_controller_absorbed = (
            absorbed
            if common_controller_absorbed is None
            else common_controller_absorbed & absorbed
        )
        common_model_insensitive = (
            insensitive
            if common_model_insensitive is None
            else common_model_insensitive & insensitive
        )
    if teacher_reachable_all:
        findings.append(
            {
                "finding_code": "teacher_reachable_on_both_branches",
                "summary": "两条线都存在 teacher/replay 可达 scene pool，因此最终归因不能优先停留在“teacher 根本到不了”这一层。",
            }
        )
    if condition_sensitive_all:
        findings.append(
            {
                "finding_code": "condition_flip_nonzero_on_both_branches",
                "summary": "两条线的 semantic condition-flip 都有非零响应，因此“完全不吃条件文本”不是当前最强解释。",
            }
        )
    if common_controller_absorbed:
        findings.append(
            {
                "finding_code": "shared_controller_absorbed_groups",
                "summary": f"两条线共同出现 controller_absorbed_groups={sorted(common_controller_absorbed)}，controller 语义仍需作为 watchlist 保留。",
            }
        )
    if common_model_insensitive:
        findings.append(
            {
                "finding_code": "shared_model_insensitive_groups",
                "summary": f"两条线共同出现 model_insensitive_groups={sorted(common_model_insensitive)}，说明至少有部分 action group 仍对 prompt/branch 条件不敏感。",
            }
        )
    return findings


def _build_inventory_entry(
    *,
    artifact_id: str,
    task_code: str,
    path: Path,
) -> dict[str, Any]:
    payload = _read_json(path, arg_name=artifact_id)
    selected_payload = {
        "artifact_kind": payload.get("artifact_kind"),
        "schema_version": payload.get("schema_version"),
        "branch": payload.get("branch"),
        "branch_key": payload.get("branch_key"),
        "axis": payload.get("axis"),
        "rung": payload.get("rung"),
        "status": payload.get("status"),
        "promotion_status": payload.get("promotion_status"),
        "report_signature_sha256": payload.get("report_signature_sha256"),
    }
    return {
        "artifact_id": artifact_id,
        "task_code": task_code,
        "path": _rel_repo(path),
        "artifact_kind": payload.get("artifact_kind"),
        "schema_version": payload.get("schema_version"),
        "file_sha256": _sha256_file(path),
        "config_or_schema_digest": _sha256_payload(selected_payload),
        "report_signature_sha256": payload.get("report_signature_sha256"),
    }


def _optional_inventory_entry(
    *, artifact_id: str, task_code: str, path: Path | str
) -> dict[str, Any] | None:
    resolved = _resolve_path(path)
    if not resolved.exists() or not resolved.is_file():
        return None
    return _build_inventory_entry(
        artifact_id=artifact_id, task_code=task_code, path=resolved
    )


def _build_task18_evidence_payload(
    *,
    generated_at: str,
    output_dir: Path,
    evidence_path: Path,
    final_output_entries: Mapping[str, Mapping[str, Any]],
    supporting_artifacts: Sequence[Mapping[str, Any]],
    branch_rows: Sequence[Mapping[str, Any]],
    officially_comparable_conclusions: Sequence[Mapping[str, Any]],
    internally_comparable_conclusions: Sequence[Mapping[str, Any]],
    recommended_next_step_by_branch: Mapping[str, str],
    shared_structural_findings: Sequence[Mapping[str, Any]],
    branch_artifact_coverage: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    data_ladder_limits: list[dict[str, Any]] = []
    for row in branch_rows:
        branch_key = str(row.get("branch_key"))
        data_summary = _as_mapping(
            row.get("data_ladder_summary", {}),
            field_name=f"{branch_key}.data_ladder_summary",
        )
        if bool(data_summary.get("assessment_complete", False)):
            continue
        data_ladder_limits.append(
            {
                "branch_key": branch_key,
                "limit_code": "data_ladder_assessment_partial",
                "available_rungs": list(
                    _as_list(
                        data_summary.get("available_rungs", []),
                        field_name=f"{branch_key}.data_ladder_summary.available_rungs",
                    )
                ),
                "missing_rungs": list(
                    _as_list(
                        data_summary.get("missing_rungs", []),
                        field_name=f"{branch_key}.data_ladder_summary.missing_rungs",
                    )
                ),
                "effective_rungs": list(
                    _as_list(
                        data_summary.get("effective_rungs", []),
                        field_name=f"{branch_key}.data_ladder_summary.effective_rungs",
                    )
                ),
                "current_hypothesis_strength": _as_mapping(
                    row.get("data_distribution_hypothesis", {}),
                    field_name=f"{branch_key}.data_distribution_hypothesis",
                ).get("strength"),
            }
        )

    return {
        "schema_version": TASK18_EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": TASK18_EVIDENCE_ARTIFACT_KIND,
        "task_code": "T18",
        "status": "PASS",
        "generated_at": generated_at,
        "evidence_path": _rel_repo(evidence_path),
        "output_dir": _rel_repo(output_dir),
        "comparability_contract": {
            "official_comparable_line": BRANCH_KEY_UNITREE_G1,
            "internal_only_comparable_line": BRANCH_KEY_NEW_EMBODIMENT,
            "public_anchor_projection_forbidden_to": [BRANCH_KEY_NEW_EMBODIMENT],
            "cross_branch_single_ranking_forbidden": True,
        },
        "generated_outputs": {
            key: dict(value) for key, value in final_output_entries.items()
        },
        "final_conclusions": {
            "officially_comparable_conclusions": [
                dict(item) for item in officially_comparable_conclusions
            ],
            "internally_comparable_conclusions": [
                dict(item) for item in internally_comparable_conclusions
            ],
            "recommended_next_step_by_branch": dict(recommended_next_step_by_branch),
        },
        "shared_structural_findings": [
            dict(item) for item in shared_structural_findings
        ],
        "coverage_summary": {
            "branch_artifact_coverage": [
                dict(item) for item in branch_artifact_coverage
            ],
            "data_ladder_limits": data_ladder_limits,
            "caution_codes": [
                "do_not_overstate_data_ladder_conclusions_when_d1_to_d3_are_partial"
            ]
            if data_ladder_limits
            else [],
        },
        "key_supporting_artifacts": [dict(item) for item in supporting_artifacts],
    }


def materialize_attribution_pack(
    *,
    output_dir: Path,
    task18_evidence_path: Path,
    task1_evidence: Path,
    task12_evidence: Path,
    task14_evidence: Path,
    task15_evidence: Path,
    task16_evidence: Path,
    task17_evidence: Path,
    checkpoint_provenance_report: Path,
    dual_branch_scorecard_json: Path,
    p_ladder_policy_gate_unitree: Path,
    p_ladder_policy_gate_new_embodiment: Path,
    d_ladder_policy_gate_unitree: Path,
    d_ladder_policy_gate_new_embodiment: Path,
    d_ladder_admission_gate_unitree: Path,
    d_ladder_admission_gate_new_embodiment: Path,
    dataset_source_registry_json: Path,
    unitree_p_roots: Sequence[Path],
    new_embodiment_p_roots: Sequence[Path],
    unitree_d_roots: Sequence[Path],
    new_embodiment_d_roots: Sequence[Path],
) -> dict[str, Path]:
    resolved_output_dir = _validate_output_dir(output_dir)
    resolved_task18_evidence_path = _prepare_output_file(task18_evidence_path)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    task1_payload = _read_json(task1_evidence, arg_name="task1_evidence")
    checkpoint_payload = _read_json(
        checkpoint_provenance_report,
        arg_name="checkpoint_provenance_report",
    )
    dual_branch_payload = _read_json(
        dual_branch_scorecard_json,
        arg_name="dual_branch_scorecard_json",
    )
    _ = _read_json(
        p_ladder_policy_gate_unitree, arg_name="p_ladder_policy_gate_unitree"
    )
    _ = _read_json(
        p_ladder_policy_gate_new_embodiment,
        arg_name="p_ladder_policy_gate_new_embodiment",
    )
    _ = _read_json(
        d_ladder_policy_gate_unitree, arg_name="d_ladder_policy_gate_unitree"
    )
    _ = _read_json(
        d_ladder_policy_gate_new_embodiment,
        arg_name="d_ladder_policy_gate_new_embodiment",
    )
    _ = _read_json(
        d_ladder_admission_gate_unitree, arg_name="d_ladder_admission_gate_unitree"
    )
    _ = _read_json(
        d_ladder_admission_gate_new_embodiment,
        arg_name="d_ladder_admission_gate_new_embodiment",
    )
    _ = _read_json(
        dataset_source_registry_json, arg_name="dataset_source_registry_json"
    )

    discovered_branch_rows: list[dict[str, Any]] = []
    excluded_artifacts: list[dict[str, str]] = []
    discovery_notes: list[dict[str, Any]] = []

    branch_root_overrides = {
        BRANCH_KEY_UNITREE_G1: {
            AXIS_P: unitree_p_roots,
            AXIS_D: unitree_d_roots,
        },
        BRANCH_KEY_NEW_EMBODIMENT: {
            AXIS_P: new_embodiment_p_roots,
            AXIS_D: new_embodiment_d_roots,
        },
    }

    for branch_key in (BRANCH_KEY_UNITREE_G1, BRANCH_KEY_NEW_EMBODIMENT):
        spec = BRANCH_SPECS[branch_key]
        branch_entry = _select_branch_entry(dual_branch_payload, branch_key=branch_key)

        public_anchor_payload: Mapping[str, Any] | None = None
        if spec.public_anchor_comparable:
            public_anchor_path = _artifact_path_from_branch_entry(
                branch_entry,
                "public_anchor_status",
                field_name=f"{branch_key}.public_anchor_status",
            )
            public_anchor_payload = _read_json(
                public_anchor_path, arg_name=f"{branch_key}.public_anchor"
            )
        controller_payload = _read_json(
            _artifact_path_from_branch_entry(
                branch_entry,
                "controller_equivalence",
                field_name=f"{branch_key}.controller_equivalence",
            ),
            arg_name=f"{branch_key}.controller_equivalence",
        )
        action_payload = _read_json(
            _artifact_path_from_branch_entry(
                branch_entry,
                "action_telemetry",
                field_name=f"{branch_key}.action_telemetry",
            ),
            arg_name=f"{branch_key}.action_telemetry",
        )
        reachability_payload = _read_json(
            _artifact_path_from_branch_entry(
                branch_entry,
                "teacher_reachability",
                field_name=f"{branch_key}.teacher_reachability",
            ),
            arg_name=f"{branch_key}.teacher_reachability",
        )

        p_discovery = _discover_branch_ladder(
            spec=spec,
            axis=AXIS_P,
            roots=branch_root_overrides[branch_key][AXIS_P],
        )
        d_discovery = _discover_branch_ladder(
            spec=spec,
            axis=AXIS_D,
            roots=branch_root_overrides[branch_key][AXIS_D],
        )
        excluded_artifacts.extend(
            cast(list[dict[str, str]], p_discovery["excluded_candidates"])
        )
        excluded_artifacts.extend(
            cast(list[dict[str, str]], d_discovery["excluded_candidates"])
        )
        discovery_notes.append(
            {
                "branch_key": branch_key,
                "p_available_rungs": list(p_discovery["available_rungs"]),
                "p_missing_rungs": list(p_discovery["missing_rungs"]),
                "d_available_rungs": list(d_discovery["available_rungs"]),
                "d_missing_rungs": list(d_discovery["missing_rungs"]),
            }
        )

        parameter_summary = _build_parameter_summary(
            cast(Mapping[str, Mapping[str, Any]], p_discovery["discoveries"])
        )
        data_summary = _build_data_summary(
            cast(Mapping[str, Mapping[str, Any]], d_discovery["discoveries"])
        )
        hypotheses = _build_branch_hypotheses(
            spec=spec,
            branch_entry=branch_entry,
            public_anchor_payload=public_anchor_payload,
            controller_payload=controller_payload,
            action_payload=action_payload,
            reachability_payload=reachability_payload,
            parameter_summary=parameter_summary,
            data_summary=data_summary,
            checkpoint_provenance=checkpoint_payload,
            checkpoint_provenance_path=_resolve_path(checkpoint_provenance_report),
        )

        discovered_branch_rows.append(
            {
                "branch": spec.embodiment_tag,
                "branch_key": spec.branch_key,
                "branch_scope": spec.branch_scope,
                "public_anchor_comparable": spec.public_anchor_comparable,
                "official_comparable_line": spec.official_comparable_line,
                "internal_only_comparable_line": spec.internal_only_comparable_line,
                "diagnostic_summary": dict(
                    _as_mapping(
                        branch_entry.get("diagnostic_summary", {}),
                        field_name="diagnostic_summary",
                    )
                ),
                "prerequisite_status": dict(
                    _as_mapping(
                        branch_entry.get("prerequisite_status", {}),
                        field_name="prerequisite_status",
                    )
                ),
                "parameter_ladder_summary": parameter_summary,
                "data_ladder_summary": data_summary,
                "stack_hypothesis": hypotheses["stack_hypothesis"],
                "controller_hypothesis": hypotheses["controller_hypothesis"],
                "parameter_scope_hypothesis": hypotheses["parameter_scope_hypothesis"],
                "data_distribution_hypothesis": hypotheses[
                    "data_distribution_hypothesis"
                ],
                "recap_interface_hypothesis": hypotheses["recap_interface_hypothesis"],
                "recommended_next_step": hypotheses["recommended_next_step"],
                "selected_formal_artifacts": _branch_structural_summary(
                    branch_entry=branch_entry,
                    parameter_summary=parameter_summary,
                    data_summary=data_summary,
                    hypotheses=hypotheses,
                    discoveries={"p": p_discovery, "d": d_discovery},
                ),
            }
        )

    officially_comparable_conclusions = [
        _branch_conclusion_row(
            spec=BRANCH_SPECS[row["branch_key"]],
            hypotheses=row,
            branch_entry=_select_branch_entry(
                dual_branch_payload, branch_key=str(row["branch_key"])
            ),
            dual_branch_scorecard_path=_resolve_path(dual_branch_scorecard_json),
        )
        for row in discovered_branch_rows
        if bool(row["official_comparable_line"])
    ]
    internally_comparable_conclusions = [
        _branch_conclusion_row(
            spec=BRANCH_SPECS[row["branch_key"]],
            hypotheses=row,
            branch_entry=_select_branch_entry(
                dual_branch_payload, branch_key=str(row["branch_key"])
            ),
            dual_branch_scorecard_path=_resolve_path(dual_branch_scorecard_json),
        )
        for row in discovered_branch_rows
        if bool(row["internal_only_comparable_line"])
    ]

    generated_at = _now_iso()
    final_attribution_payload = {
        "schema_version": FINAL_ATTRIBUTION_SCHEMA_VERSION,
        "artifact_kind": FINAL_ATTRIBUTION_ARTIFACT_KIND,
        "generated_at": generated_at,
        "branch_order": [BRANCH_KEY_UNITREE_G1, BRANCH_KEY_NEW_EMBODIMENT],
        "officially_comparable_conclusions": officially_comparable_conclusions,
        "internally_comparable_conclusions": internally_comparable_conclusions,
        "branches": discovered_branch_rows,
        "excluded_artifacts": excluded_artifacts,
        "discovery_notes": discovery_notes,
        "source_artifacts": {
            "task1_evidence": _rel_repo(task1_evidence),
            "checkpoint_provenance_report": _rel_repo(checkpoint_provenance_report),
            "dual_branch_scorecard": _rel_repo(dual_branch_scorecard_json),
            "p_ladder_policy_gate_unitree": _rel_repo(p_ladder_policy_gate_unitree),
            "p_ladder_policy_gate_new_embodiment": _rel_repo(
                p_ladder_policy_gate_new_embodiment
            ),
            "d_ladder_policy_gate_unitree": _rel_repo(d_ladder_policy_gate_unitree),
            "d_ladder_policy_gate_new_embodiment": _rel_repo(
                d_ladder_policy_gate_new_embodiment
            ),
            "d_ladder_admission_gate_unitree": _rel_repo(
                d_ladder_admission_gate_unitree
            ),
            "d_ladder_admission_gate_new_embodiment": _rel_repo(
                d_ladder_admission_gate_new_embodiment
            ),
            "dataset_source_registry": _rel_repo(dataset_source_registry_json),
        },
    }
    final_attribution_path = resolved_output_dir / FINAL_ATTRIBUTION_JSON_NAME
    _write_json(
        final_attribution_path,
        cast(Mapping[str, object], _json_ready(final_attribution_payload)),
    )

    branch_comparison_payload = {
        "schema_version": BRANCH_COMPARISON_SCHEMA_VERSION,
        "artifact_kind": BRANCH_COMPARISON_ARTIFACT_KIND,
        "generated_at": generated_at,
        "branch_order": [BRANCH_KEY_UNITREE_G1, BRANCH_KEY_NEW_EMBODIMENT],
        "official_comparable_line": BRANCH_KEY_UNITREE_G1,
        "internal_only_comparable_line": BRANCH_KEY_NEW_EMBODIMENT,
        "officially_comparable_conclusions": officially_comparable_conclusions,
        "internally_comparable_conclusions": internally_comparable_conclusions,
        "comparison_axes": _build_comparison_axes(discovered_branch_rows),
        "shared_structural_findings": _build_shared_structural_findings(
            discovered_branch_rows
        ),
        "non_comparable_numeric_boundaries": [
            {
                "boundary_code": "do_not_project_public_anchor_to_new_embodiment",
                "summary": "NEW_EMBODIMENT 只能做 internal-only structural comparison，不能直接套用 UNITREE_G1 public anchor 数值口径。",
            },
            {
                "boundary_code": "do_not_collapse_both_branches_into_one_ranking",
                "summary": "branch comparison pack 只并列呈现，不输出跨 branch 单一总分或排名。",
            },
        ],
        "recommended_next_step_by_branch": {
            str(row["branch_key"]): str(row["recommended_next_step"])
            for row in discovered_branch_rows
        },
        "excluded_artifacts": excluded_artifacts,
    }
    branch_comparison_path = resolved_output_dir / BRANCH_COMPARISON_JSON_NAME
    _write_json(
        branch_comparison_path,
        cast(Mapping[str, object], _json_ready(branch_comparison_payload)),
    )

    inventory_entries = [
        _build_inventory_entry(
            artifact_id="task1_eval_contract_evidence",
            task_code="T1",
            path=_resolve_path(task1_evidence),
        ),
        _build_inventory_entry(
            artifact_id="task3_checkpoint_provenance_report",
            task_code="T3",
            path=_resolve_path(checkpoint_provenance_report),
        ),
        _build_inventory_entry(
            artifact_id="task11_dual_branch_scorecard",
            task_code="T11",
            path=_resolve_path(dual_branch_scorecard_json),
        ),
        _build_inventory_entry(
            artifact_id="task12_p_ladder_policy_gate_unitree",
            task_code="T12",
            path=_resolve_path(p_ladder_policy_gate_unitree),
        ),
        _build_inventory_entry(
            artifact_id="task12_p_ladder_policy_gate_new_embodiment",
            task_code="T12",
            path=_resolve_path(p_ladder_policy_gate_new_embodiment),
        ),
        _build_inventory_entry(
            artifact_id="task12_d_ladder_policy_gate_unitree",
            task_code="T12",
            path=_resolve_path(d_ladder_policy_gate_unitree),
        ),
        _build_inventory_entry(
            artifact_id="task12_d_ladder_policy_gate_new_embodiment",
            task_code="T12",
            path=_resolve_path(d_ladder_policy_gate_new_embodiment),
        ),
        _build_inventory_entry(
            artifact_id="task15_d_ladder_admission_gate_unitree",
            task_code="T15",
            path=_resolve_path(d_ladder_admission_gate_unitree),
        ),
        _build_inventory_entry(
            artifact_id="task15_d_ladder_admission_gate_new_embodiment",
            task_code="T15",
            path=_resolve_path(d_ladder_admission_gate_new_embodiment),
        ),
        _build_inventory_entry(
            artifact_id="task15_dataset_source_registry",
            task_code="T15",
            path=_resolve_path(dataset_source_registry_json),
        ),
        _build_inventory_entry(
            artifact_id="task18_final_attribution_matrix",
            task_code="T18",
            path=final_attribution_path,
        ),
        _build_inventory_entry(
            artifact_id="task18_branch_comparison_pack",
            task_code="T18",
            path=branch_comparison_path,
        ),
    ]

    for row in discovered_branch_rows:
        branch_key = str(row["branch_key"])
        for axis_key, prefix, task_code in (
            (AXIS_P, "task13_14", "T13_T14"),
            (AXIS_D, "task16_17", "T16_T17"),
        ):
            discoveries = _as_mapping(
                _as_mapping(
                    row.get("selected_formal_artifacts", {}),
                    field_name="selected_formal_artifacts",
                ).get("discovery", {}),
                field_name="selected_formal_artifacts.discovery",
            )
            axis_discovery = _as_mapping(
                discoveries.get(axis_key.lower(), {}),
                field_name=f"discovery.{axis_key.lower()}",
            )
            by_rung = _as_mapping(
                axis_discovery.get("discoveries", {}),
                field_name=f"discovery.{axis_key.lower()}.discoveries",
            )
            for rung, discovery in by_rung.items():
                item = _as_mapping(
                    discovery, field_name=f"discovery.{axis_key.lower()}.{rung}"
                )
                if str(item.get("status")) != "FOUND":
                    continue
                inventory_entries.append(
                    _build_inventory_entry(
                        artifact_id=f"{prefix}_{branch_key}_{rung}_scorecard",
                        task_code=task_code,
                        path=cast(Path, item["scorecard_path"]),
                    )
                )
                inventory_entries.append(
                    _build_inventory_entry(
                        artifact_id=f"{prefix}_{branch_key}_{rung}_manifest",
                        task_code=task_code,
                        path=cast(Path, item["manifest_path"]),
                    )
                )

    wave_freeze_payload = {
        "schema_version": WAVE_FREEZE_SCHEMA_VERSION,
        "artifact_kind": WAVE_FREEZE_ARTIFACT_KIND,
        "generated_at": generated_at,
        "wave_label": "gr00t_anchor_controller_recap_final_wave",
        "branch_order": [BRANCH_KEY_UNITREE_G1, BRANCH_KEY_NEW_EMBODIMENT],
        "inventory": inventory_entries,
        "inventory_count": len(inventory_entries),
        "excluded_artifacts": excluded_artifacts,
        "branch_artifact_coverage": discovery_notes,
        "contract_snapshot": dict(
            _as_mapping(
                task1_payload.get("public_anchor_snapshot", {}),
                field_name="task1.public_anchor_snapshot",
            )
        ),
        "freeze_summary": {
            "officially_comparable_conclusions": officially_comparable_conclusions,
            "internally_comparable_conclusions": internally_comparable_conclusions,
            "recommended_next_step_by_branch": {
                str(row["branch_key"]): str(row["recommended_next_step"])
                for row in discovered_branch_rows
            },
        },
    }
    wave_freeze_path = resolved_output_dir / WAVE_FREEZE_JSON_NAME
    _write_json(
        wave_freeze_path,
        cast(Mapping[str, object], _json_ready(wave_freeze_payload)),
    )

    final_output_entries = {
        "final_attribution_matrix": _build_inventory_entry(
            artifact_id="task18_final_attribution_matrix",
            task_code="T18",
            path=final_attribution_path,
        ),
        "branch_comparison_pack": _build_inventory_entry(
            artifact_id="task18_branch_comparison_pack",
            task_code="T18",
            path=branch_comparison_path,
        ),
        "wave_freeze_manifest": _build_inventory_entry(
            artifact_id="task18_wave_freeze_manifest",
            task_code="T18",
            path=wave_freeze_path,
        ),
    }
    key_supporting_artifacts = [
        _build_inventory_entry(
            artifact_id="task1_eval_contract_evidence",
            task_code="T1",
            path=_resolve_path(task1_evidence),
        ),
        _build_inventory_entry(
            artifact_id="task3_checkpoint_provenance_report",
            task_code="T3",
            path=_resolve_path(checkpoint_provenance_report),
        ),
        _build_inventory_entry(
            artifact_id="task11_dual_branch_scorecard",
            task_code="T11",
            path=_resolve_path(dual_branch_scorecard_json),
        ),
    ]
    for optional_entry in (
        _optional_inventory_entry(
            artifact_id="task12_ladder_policy_gate_evidence",
            task_code="T12",
            path=task12_evidence,
        ),
        _optional_inventory_entry(
            artifact_id="task14_p_ladder_new_embodiment_evidence",
            task_code="T14",
            path=task14_evidence,
        ),
        _optional_inventory_entry(
            artifact_id="task15_d_ladder_policy_gate_evidence",
            task_code="T15",
            path=task15_evidence,
        ),
        _optional_inventory_entry(
            artifact_id="task16_d_ladder_unitree_g1_evidence",
            task_code="T16",
            path=task16_evidence,
        ),
        _optional_inventory_entry(
            artifact_id="task17_d_ladder_new_embodiment_evidence",
            task_code="T17",
            path=task17_evidence,
        ),
    ):
        if optional_entry is not None:
            key_supporting_artifacts.append(optional_entry)
    recommended_next_step_by_branch = {
        str(row["branch_key"]): str(row["recommended_next_step"])
        for row in discovered_branch_rows
    }
    shared_structural_findings = _build_shared_structural_findings(
        discovered_branch_rows
    )
    task18_evidence_payload = _build_task18_evidence_payload(
        generated_at=generated_at,
        output_dir=resolved_output_dir,
        evidence_path=resolved_task18_evidence_path,
        final_output_entries=final_output_entries,
        supporting_artifacts=key_supporting_artifacts,
        branch_rows=discovered_branch_rows,
        officially_comparable_conclusions=officially_comparable_conclusions,
        internally_comparable_conclusions=internally_comparable_conclusions,
        recommended_next_step_by_branch=recommended_next_step_by_branch,
        shared_structural_findings=shared_structural_findings,
        branch_artifact_coverage=discovery_notes,
    )
    _write_json(
        resolved_task18_evidence_path,
        cast(Mapping[str, object], _json_ready(task18_evidence_payload)),
    )

    return {
        "final_attribution_matrix": final_attribution_path,
        "branch_comparison_pack": branch_comparison_path,
        "wave_freeze_manifest": wave_freeze_path,
        "task18_evidence": resolved_task18_evidence_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = materialize_attribution_pack(
            output_dir=args.output_dir,
            task18_evidence_path=args.task18_evidence_path,
            task1_evidence=args.task1_evidence,
            task12_evidence=args.task12_evidence,
            task14_evidence=args.task14_evidence,
            task15_evidence=args.task15_evidence,
            task16_evidence=args.task16_evidence,
            task17_evidence=args.task17_evidence,
            checkpoint_provenance_report=args.checkpoint_provenance_report,
            dual_branch_scorecard_json=args.dual_branch_scorecard_json,
            p_ladder_policy_gate_unitree=args.p_ladder_policy_gate_unitree,
            p_ladder_policy_gate_new_embodiment=args.p_ladder_policy_gate_new_embodiment,
            d_ladder_policy_gate_unitree=args.d_ladder_policy_gate_unitree,
            d_ladder_policy_gate_new_embodiment=args.d_ladder_policy_gate_new_embodiment,
            d_ladder_admission_gate_unitree=args.d_ladder_admission_gate_unitree,
            d_ladder_admission_gate_new_embodiment=args.d_ladder_admission_gate_new_embodiment,
            dataset_source_registry_json=args.dataset_source_registry_json,
            unitree_p_roots=_candidate_roots(
                args.unitree_p_roots, DEFAULT_UNITREE_P_ROOTS
            ),
            new_embodiment_p_roots=_candidate_roots(
                args.new_embodiment_p_roots,
                DEFAULT_NEW_EMBODIMENT_P_ROOTS,
            ),
            unitree_d_roots=_candidate_roots(
                args.unitree_d_roots, DEFAULT_UNITREE_D_ROOTS
            ),
            new_embodiment_d_roots=_candidate_roots(
                args.new_embodiment_d_roots,
                DEFAULT_NEW_EMBODIMENT_D_ROOTS,
            ),
        )
    except Exception as exc:
        parser.exit(1, f"ERROR: {_exception_message(exc)}\n")

    for label, path in paths.items():
        print(f"{label}={_rel_repo(path) or str(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
