#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[3]


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap.scripts import gr00t_checkpoint_provenance_gate
from work.recap.scripts import gr00t_dual_branch_scorecard
from work.recap.scripts import gr00t_ladder_policy_gate
from work.recap.scripts import gr00t_public_anchor_eval
from work.recap.scripts import gr00t_recap_attribution_pack
from work.recap.scripts import state_conditioned_oracle_eval

DEFAULT_OUTPUT = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/baseline_freeze/baseline_freeze_matrix.json"
)
DEFAULT_PUBLIC_ANCHOR_FORMAL = (
    gr00t_public_anchor_eval.DEFAULT_OUTPUT_DIR
    / gr00t_public_anchor_eval.FORMAL_JSON_NAME
)
DEFAULT_PUBLIC_ANCHOR_SANITY_GATE = (
    gr00t_public_anchor_eval.DEFAULT_OUTPUT_DIR
    / gr00t_public_anchor_eval.SANITY_GATE_JSON_NAME
)
DEFAULT_CHECKPOINT_PROVENANCE_REPORT = (
    gr00t_recap_attribution_pack.DEFAULT_CHECKPOINT_PROVENANCE_REPORT
)
DEFAULT_DUAL_BRANCH_SCORECARD_JSON = (
    gr00t_recap_attribution_pack.DEFAULT_DUAL_BRANCH_SCORECARD_JSON
)
DEFAULT_P_LADDER_POLICY_GATE_UNITREE = (
    gr00t_recap_attribution_pack.DEFAULT_P_LADDER_POLICY_GATE_UNITREE
)
DEFAULT_D_LADDER_POLICY_GATE_UNITREE = (
    gr00t_recap_attribution_pack.DEFAULT_D_LADDER_POLICY_GATE_UNITREE
)
DEFAULT_LEGACY_C1_SCORECARD = Path(
    "agent/artifacts/state_conditioned_materialization/eval/oracle_conditioned_dev_scorecard.json"
)
DEFAULT_LEGACY_C1_RESULT_SPLIT_DECISION = Path(
    "agent/artifacts/state_conditioned_materialization/eval/result_split_decision.json"
)

REPORT_SCHEMA_VERSION = "gr00t_baseline_freeze_matrix_v1"
REPORT_ARTIFACT_KIND = "gr00t_baseline_freeze_matrix"

B0_BASELINE_ID = "g1_b0_public_anchor"
B1_BASELINE_ID = "g1_b1_oldworld_c1"
DISPLAY_LABEL_B0 = "B0"
DISPLAY_LABEL_B1 = "B1"

TOP_LEVEL_OFFICIAL_COMPARABLE_LINE = "unitree_g1"
TOP_LEVEL_INTERNAL_ONLY_COMPARABLE_LINE = "new_embodiment"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_baseline_freeze_matrix.py",
        description=(
            "Freeze repo-local B0/B1 display aliases into a read-only baseline layer that "
            "backpoints to existing G1 public-anchor authority artifacts and the legacy "
            "old-world C1 line without rewriting source JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _ = parser.add_argument(
        "--public-anchor-formal",
        type=Path,
        default=DEFAULT_PUBLIC_ANCHOR_FORMAL,
    )
    _ = parser.add_argument(
        "--public-anchor-sanity-gate",
        type=Path,
        default=DEFAULT_PUBLIC_ANCHOR_SANITY_GATE,
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
        "--d-ladder-policy-gate-unitree",
        type=Path,
        default=DEFAULT_D_LADDER_POLICY_GATE_UNITREE,
    )
    _ = parser.add_argument(
        "--legacy-c1-scorecard",
        type=Path,
        default=DEFAULT_LEGACY_C1_SCORECARD,
    )
    _ = parser.add_argument(
        "--legacy-c1-result-split-decision",
        type=Path,
        default=DEFAULT_LEGACY_C1_RESULT_SPLIT_DECISION,
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


def _validate_output_path(path: Path | str) -> Path:
    resolved = _resolve_path(path)
    if resolved.exists():
        raise ValueError(
            f"baseline freeze output already exists (no-overwrite): {resolved}"
        )
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


def _rel_repo(path: Path | str) -> str:
    resolved = _resolve_path(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return cast(Mapping[str, Any], value)


def _as_list(value: object, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)


def _as_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _json_backpointer(
    *,
    artifact_id: str,
    relation: str,
    owner_script: str,
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "relation": relation,
        "owner_script": owner_script,
        "path": _rel_repo(path),
        "resolved_path": str(path),
        "artifact_kind": payload.get("artifact_kind"),
        "schema_version": payload.get("schema_version"),
        "report_signature_sha256": payload.get("report_signature_sha256"),
        "sha256": _sha256_file(path),
    }


def _validate_public_anchor_formal(payload: Mapping[str, Any]) -> None:
    if payload.get("artifact_kind") != gr00t_public_anchor_eval.FORMAL_ARTIFACT_KIND:
        raise ValueError("public-anchor formal artifact_kind mismatch")
    if payload.get("schema_version") != gr00t_public_anchor_eval.FORMAL_SCHEMA_VERSION:
        raise ValueError("public-anchor formal schema_version mismatch")


def _validate_public_anchor_sanity_gate(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("artifact_kind")
        != gr00t_public_anchor_eval.SANITY_GATE_ARTIFACT_KIND
    ):
        raise ValueError("public-anchor sanity gate artifact_kind mismatch")
    if (
        payload.get("schema_version")
        != gr00t_public_anchor_eval.SANITY_GATE_SCHEMA_VERSION
    ):
        raise ValueError("public-anchor sanity gate schema_version mismatch")


def _validate_checkpoint_provenance(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("artifact_kind")
        != gr00t_checkpoint_provenance_gate.REPORT_ARTIFACT_KIND
    ):
        raise ValueError("checkpoint provenance artifact_kind mismatch")
    if (
        payload.get("schema_version")
        != gr00t_checkpoint_provenance_gate.REPORT_SCHEMA_VERSION
    ):
        raise ValueError("checkpoint provenance schema_version mismatch")


def _validate_dual_branch_scorecard(payload: Mapping[str, Any]) -> None:
    if payload.get("artifact_kind") != gr00t_dual_branch_scorecard.REPORT_ARTIFACT_KIND:
        raise ValueError("dual-branch scorecard artifact_kind mismatch")
    if (
        payload.get("schema_version")
        != gr00t_dual_branch_scorecard.REPORT_SCHEMA_VERSION
    ):
        raise ValueError("dual-branch scorecard schema_version mismatch")
    if payload.get("official_comparable_line") != TOP_LEVEL_OFFICIAL_COMPARABLE_LINE:
        raise ValueError("dual-branch scorecard official_comparable_line drifted")
    if (
        payload.get("internal_only_comparable_line")
        != TOP_LEVEL_INTERNAL_ONLY_COMPARABLE_LINE
    ):
        raise ValueError("dual-branch scorecard internal_only_comparable_line drifted")


def _validate_ladder_policy_gate(payload: Mapping[str, Any], *, axis: str) -> None:
    if payload.get("artifact_kind") != gr00t_ladder_policy_gate.REPORT_ARTIFACT_KIND:
        raise ValueError(f"{axis}-ladder policy gate artifact_kind mismatch")
    if payload.get("schema_version") != gr00t_ladder_policy_gate.REPORT_SCHEMA_VERSION:
        raise ValueError(f"{axis}-ladder policy gate schema_version mismatch")
    if payload.get("ladder_axis") != axis:
        raise ValueError(f"{axis}-ladder policy gate ladder_axis mismatch")
    if payload.get("branch_key") != TOP_LEVEL_OFFICIAL_COMPARABLE_LINE:
        raise ValueError(f"{axis}-ladder policy gate branch_key drifted")


def _load_legacy_c1_backpointer(
    *,
    legacy_c1_scorecard: Path,
    legacy_c1_result_split_decision: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    int,
]:
    scorecard = _read_json(
        legacy_c1_scorecard,
        arg_name="legacy-c1-scorecard",
    )
    if (
        scorecard.get("artifact_kind")
        != "state_conditioned_oracle_conditioned_dev_scorecard"
    ):
        raise ValueError("legacy C1 scorecard artifact_kind mismatch")
    if scorecard.get("schema_version") != state_conditioned_oracle_eval.SCHEMA_VERSION:
        raise ValueError("legacy C1 scorecard schema_version mismatch")
    if bool(scorecard.get("official_comparable_line", False)) or bool(
        scorecard.get("internal_only_comparable_line", False)
    ):
        raise ValueError(
            "legacy C1 scorecard must not declare active comparable-line authority"
        )

    line_order = [
        _as_str(item, field_name="legacy-c1-scorecard.line_order[]")
        for item in _as_list(scorecard.get("line_order", []), field_name="line_order")
    ]
    if line_order != list(state_conditioned_oracle_eval.LINE_ORDER):
        raise ValueError("legacy C1 scorecard line_order drifted")

    lines = _as_list(scorecard.get("lines", []), field_name="lines")
    c1_index = -1
    c1_line: Mapping[str, Any] | None = None
    for index, raw_line in enumerate(lines):
        line = _as_mapping(raw_line, field_name=f"lines[{index}]")
        if line.get("line_key") == state_conditioned_oracle_eval.LINE_C1:
            c1_index = index
            c1_line = line
            break
    if c1_line is None:
        raise ValueError("legacy C1 scorecard is missing the C1 line entry")
    if bool(c1_line.get("official_comparable_line", False)) or bool(
        c1_line.get("internal_only_comparable_line", False)
    ):
        raise ValueError(
            "legacy C1 line must not masquerade as an active comparable line"
        )

    result_split = _read_json(
        legacy_c1_result_split_decision,
        arg_name="legacy-c1-result-split-decision",
    )
    if result_split.get("artifact_kind") != "state_conditioned_result_split_decision":
        raise ValueError("legacy C1 result split artifact_kind mismatch")

    return scorecard, c1_line, result_split, c1_index


def materialize_baseline_freeze_matrix(
    *,
    output: Path,
    public_anchor_formal: Path,
    public_anchor_sanity_gate: Path,
    checkpoint_provenance_report: Path,
    dual_branch_scorecard_json: Path,
    p_ladder_policy_gate_unitree: Path,
    d_ladder_policy_gate_unitree: Path,
    legacy_c1_scorecard: Path,
    legacy_c1_result_split_decision: Path,
) -> dict[str, Any]:
    resolved_output = _validate_output_path(output)
    resolved_public_anchor_formal = _validate_existing_file(
        public_anchor_formal, arg_name="public-anchor-formal"
    )
    resolved_public_anchor_sanity_gate = _validate_existing_file(
        public_anchor_sanity_gate, arg_name="public-anchor-sanity-gate"
    )
    resolved_checkpoint_provenance = _validate_existing_file(
        checkpoint_provenance_report, arg_name="checkpoint-provenance-report"
    )
    resolved_dual_branch_scorecard = _validate_existing_file(
        dual_branch_scorecard_json, arg_name="dual-branch-scorecard-json"
    )
    resolved_p_gate = _validate_existing_file(
        p_ladder_policy_gate_unitree, arg_name="p-ladder-policy-gate-unitree"
    )
    resolved_d_gate = _validate_existing_file(
        d_ladder_policy_gate_unitree, arg_name="d-ladder-policy-gate-unitree"
    )
    resolved_legacy_c1_scorecard = _validate_existing_file(
        legacy_c1_scorecard, arg_name="legacy-c1-scorecard"
    )
    resolved_legacy_c1_result_split = _validate_existing_file(
        legacy_c1_result_split_decision,
        arg_name="legacy-c1-result-split-decision",
    )

    public_anchor_formal_payload = _read_json(
        resolved_public_anchor_formal, arg_name="public-anchor-formal"
    )
    public_anchor_gate_payload = _read_json(
        resolved_public_anchor_sanity_gate, arg_name="public-anchor-sanity-gate"
    )
    checkpoint_provenance_payload = _read_json(
        resolved_checkpoint_provenance,
        arg_name="checkpoint-provenance-report",
    )
    dual_branch_payload = _read_json(
        resolved_dual_branch_scorecard,
        arg_name="dual-branch-scorecard-json",
    )
    p_gate_payload = _read_json(
        resolved_p_gate, arg_name="p-ladder-policy-gate-unitree"
    )
    d_gate_payload = _read_json(
        resolved_d_gate, arg_name="d-ladder-policy-gate-unitree"
    )

    _validate_public_anchor_formal(public_anchor_formal_payload)
    _validate_public_anchor_sanity_gate(public_anchor_gate_payload)
    _validate_checkpoint_provenance(checkpoint_provenance_payload)
    _validate_dual_branch_scorecard(dual_branch_payload)
    _validate_ladder_policy_gate(p_gate_payload, axis="P")
    _validate_ladder_policy_gate(d_gate_payload, axis="D")

    legacy_scorecard_payload, legacy_c1_line, legacy_result_split_payload, c1_index = (
        _load_legacy_c1_backpointer(
            legacy_c1_scorecard=resolved_legacy_c1_scorecard,
            legacy_c1_result_split_decision=resolved_legacy_c1_result_split,
        )
    )

    b0_entry = {
        "baseline_id": B0_BASELINE_ID,
        "display_label": DISPLAY_LABEL_B0,
        "branch_key": TOP_LEVEL_OFFICIAL_COMPARABLE_LINE,
        "authority_scope": "current_unitree_g1_public_anchor_authority",
        "mainline_authority": True,
        "legacy_backpointer_only": False,
        "official_comparable_line": True,
        "internal_only_comparable_line": False,
        "public_anchor_comparable": True,
        "parameter_baseline_rung": "P0",
        "data_baseline_rung": "D0",
        "prerequisite_gate_policy": {
            "checkpoint_provenance_required": True,
            "dual_branch_scorecard_required": True,
            "unitree_g1_p_ladder_policy_gate_required": True,
            "unitree_g1_d_ladder_policy_gate_required": True,
        },
        "source_artifacts": [
            _json_backpointer(
                artifact_id="public_anchor_formal",
                relation="current_public_anchor_formal",
                owner_script="work/recap/scripts/gr00t_public_anchor_eval.py",
                path=resolved_public_anchor_formal,
                payload=public_anchor_formal_payload,
            ),
            _json_backpointer(
                artifact_id="public_anchor_sanity_gate",
                relation="current_public_anchor_sanity_gate",
                owner_script="work/recap/scripts/gr00t_public_anchor_eval.py",
                path=resolved_public_anchor_sanity_gate,
                payload=public_anchor_gate_payload,
            ),
            _json_backpointer(
                artifact_id="checkpoint_provenance_report",
                relation="g1_prerequisite_gate",
                owner_script="work/recap/scripts/gr00t_checkpoint_provenance_gate.py",
                path=resolved_checkpoint_provenance,
                payload=checkpoint_provenance_payload,
            ),
            _json_backpointer(
                artifact_id="dual_branch_scorecard",
                relation="g1_prerequisite_gate",
                owner_script="work/recap/scripts/gr00t_dual_branch_scorecard.py",
                path=resolved_dual_branch_scorecard,
                payload=dual_branch_payload,
            ),
            _json_backpointer(
                artifact_id="p_ladder_policy_gate_unitree_g1",
                relation="g1_prerequisite_gate",
                owner_script="work/recap/scripts/gr00t_ladder_policy_gate.py",
                path=resolved_p_gate,
                payload=p_gate_payload,
            ),
            _json_backpointer(
                artifact_id="d_ladder_policy_gate_unitree_g1",
                relation="g1_prerequisite_gate",
                owner_script="work/recap/scripts/gr00t_ladder_policy_gate.py",
                path=resolved_d_gate,
                payload=d_gate_payload,
            ),
        ],
        "summary": {
            "public_anchor_success_rate": public_anchor_formal_payload.get(
                "success_rate"
            ),
            "public_anchor_continue_to_audit": public_anchor_gate_payload.get(
                "continue_to_audit"
            ),
            "public_anchor_systemic_break_flags": list(
                cast(
                    list[object],
                    public_anchor_formal_payload.get("systemic_break_flags", []),
                )
            ),
            "checkpoint_formal_eligibility": checkpoint_provenance_payload.get(
                "formal_eligibility"
            ),
            "checkpoint_reason_code": checkpoint_provenance_payload.get("reason_code"),
            "official_comparable_line": dual_branch_payload.get(
                "official_comparable_line"
            ),
            "internal_only_comparable_line": dual_branch_payload.get(
                "internal_only_comparable_line"
            ),
            "p_ladder_change_policy": p_gate_payload.get("change_policy"),
            "d_ladder_change_policy": d_gate_payload.get("change_policy"),
        },
    }

    training_run_metadata = _as_mapping(
        legacy_scorecard_payload.get("training_run_metadata", {}),
        field_name="legacy-c1-scorecard.training_run_metadata",
    )
    b1_entry = {
        "baseline_id": B1_BASELINE_ID,
        "display_label": DISPLAY_LABEL_B1,
        "authority_scope": "legacy_oldworld_c1_negative_control_backpointer_only",
        "mainline_authority": False,
        "legacy_backpointer_only": True,
        "official_comparable_line": False,
        "internal_only_comparable_line": False,
        "public_anchor_comparable": False,
        "promotion_to_active_mainline_authority_forbidden": True,
        "mainline_masquerade_blockers": [
            "must_not_set_official_comparable_line",
            "must_not_set_internal_only_comparable_line",
            "must_not_replace_unitree_g1_public_anchor_authority",
        ],
        "source_artifacts": [
            _json_backpointer(
                artifact_id="legacy_c1_scorecard",
                relation="legacy_negative_control_backpointer",
                owner_script="work/recap/scripts/state_conditioned_oracle_eval.py",
                path=resolved_legacy_c1_scorecard,
                payload=legacy_scorecard_payload,
            ),
            _json_backpointer(
                artifact_id="legacy_c1_result_split_decision",
                relation="legacy_negative_control_context",
                owner_script="work/recap/scripts/state_conditioned_oracle_eval.py",
                path=resolved_legacy_c1_result_split,
                payload=legacy_result_split_payload,
            ),
        ],
        "legacy_line_backpointer": {
            "legacy_family": "oldworld_c1",
            "line_key": state_conditioned_oracle_eval.LINE_C1,
            "line_index": c1_index,
            "line_label": legacy_c1_line.get("line_label"),
            "line_entry_digest_sha256": _sha256_payload(legacy_c1_line),
            "training_run_metadata_path": training_run_metadata.get(
                state_conditioned_oracle_eval.LINE_C1
            ),
            "model_path": legacy_c1_line.get("model_path"),
            "oracle_phase_mode_supplied": bool(
                legacy_c1_line.get("oracle_phase_mode_supplied", False)
            ),
            "result_split_next_step": legacy_result_split_payload.get("next_step"),
            "result_split_branch_reason": legacy_result_split_payload.get(
                "branch_reason"
            ),
            "result_split_ab_case": legacy_result_split_payload.get("ab_case"),
        },
        "summary": {
            "legacy_scorecard_line_order": list(
                cast(list[object], legacy_scorecard_payload.get("line_order", []))
            ),
            "legacy_scorecard_line_labels": dict(
                cast(
                    Mapping[str, object],
                    legacy_scorecard_payload.get("line_labels", {}),
                )
            ),
            "legacy_line_label": legacy_c1_line.get("line_label"),
            "legacy_negative_control_reason": (
                "Legacy old-world C1 is frozen here only as a provenance-rich backpointer; "
                "it is not an active G1 mainline authority line."
            ),
        },
    }

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "generated_at": _now_iso(),
        "official_comparable_line": TOP_LEVEL_OFFICIAL_COMPARABLE_LINE,
        "internal_only_comparable_line": TOP_LEVEL_INTERNAL_ONLY_COMPARABLE_LINE,
        "machine_id_policy": {
            "display_labels_are_not_machine_ids": True,
            "namespaced_machine_ids_required": True,
            "disallowed_machine_ids": [DISPLAY_LABEL_B0, DISPLAY_LABEL_B1],
        },
        "display_rows": [
            {"display_label": DISPLAY_LABEL_B0, "baseline_id": B0_BASELINE_ID},
            {"display_label": DISPLAY_LABEL_B1, "baseline_id": B1_BASELINE_ID},
        ],
        "baseline_id_order": [B0_BASELINE_ID, B1_BASELINE_ID],
        "freeze_policy": {
            "append_only": True,
            "no_overwrite": True,
            "source_artifacts_read_only": True,
            "source_artifacts_mutated": False,
        },
        "baselines": {
            B0_BASELINE_ID: b0_entry,
            B1_BASELINE_ID: b1_entry,
        },
    }
    payload["report_signature_sha256"] = _sha256_payload(payload)
    _ = _write_json(resolved_output, cast(Mapping[str, object], payload))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = materialize_baseline_freeze_matrix(
            output=args.output,
            public_anchor_formal=args.public_anchor_formal,
            public_anchor_sanity_gate=args.public_anchor_sanity_gate,
            checkpoint_provenance_report=args.checkpoint_provenance_report,
            dual_branch_scorecard_json=args.dual_branch_scorecard_json,
            p_ladder_policy_gate_unitree=args.p_ladder_policy_gate_unitree,
            d_ladder_policy_gate_unitree=args.d_ladder_policy_gate_unitree,
            legacy_c1_scorecard=args.legacy_c1_scorecard,
            legacy_c1_result_split_decision=args.legacy_c1_result_split_decision,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "B0_BASELINE_ID",
    "B1_BASELINE_ID",
    "DISPLAY_LABEL_B0",
    "DISPLAY_LABEL_B1",
    "REPORT_ARTIFACT_KIND",
    "REPORT_SCHEMA_VERSION",
    "TOP_LEVEL_INTERNAL_ONLY_COMPARABLE_LINE",
    "TOP_LEVEL_OFFICIAL_COMPARABLE_LINE",
    "build_parser",
    "main",
    "materialize_baseline_freeze_matrix",
]


if __name__ == "__main__":
    raise SystemExit(main())
