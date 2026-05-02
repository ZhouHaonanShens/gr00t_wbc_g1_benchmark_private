#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts.state_conditioned_common import write_json
from work.recap.stage3_contract_precondition_gate import ADV_SERVER_REQUIRED
from work.recap.stage3_contract_precondition_gate import BASELINE_DEFAULT_ADV_INIT
from work.recap.stage3_contract_precondition_gate import (
    FAILURE_STATUS_INCONCLUSIVE_CONTRACT_MISMATCH,
)
from work.recap.stage3_contract_precondition_gate import (
    _inspect_checkpoint_weight_map_features,
)
from work.recap.stage3_contract_precondition_gate import _select_prelim_eval_surface


DEFAULT_CHECKPOINT = Path(
    "agent/artifacts/stage3_t10_advantage_1gpu/formal_run/checkpoint-200"
)
DEFAULT_OUTPUT_JSON = Path(
    "agent/artifacts/recap_min_loop/single_gpu_v1/t10_conditioned_surface_gate.json"
)
SCHEMA_VERSION = "stage3_conditioned_surface_gate_v1"
ARTIFACT_KIND = "stage3_conditioned_surface_gate"


def _resolve_path(raw: str | Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _repo_relative(path: str | Path) -> str:
    resolved = _resolve_path(path)
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _surface_mode_name(mode: Any) -> str:
    raw = str(mode or "").strip()
    if raw == ADV_SERVER_REQUIRED:
        return "ADV_SERVER_REQUIRED"
    if raw == BASELINE_DEFAULT_ADV_INIT:
        return "BASELINE_DEFAULT_ADV_INIT"
    return raw.upper()


def build_gate_payload(*, checkpoint: Path, output_json: Path) -> dict[str, Any]:
    features = _inspect_checkpoint_weight_map_features(
        repo_root=REPO_ROOT,
        manifest_payload={},
        checkpoint_path=checkpoint,
        checkpoint_source_field="conditioned_checkpoint",
    )
    prelim_eval_surface, local_gate_pass, failure_reason_codes = (
        _select_prelim_eval_surface(features=features)
    )
    internal_surface_mode = str(prelim_eval_surface.get("mode") or "")
    allow_baseline_default_advantage_embedding_init = bool(
        prelim_eval_surface.get("allow_baseline_default_advantage_embedding_init")
    )
    has_advantage_embedding_weight = bool(
        features.get("has_advantage_embedding_weight")
    )
    has_advantage_embedding_bias = bool(features.get("has_advantage_embedding_bias"))
    surface_mode = _surface_mode_name(internal_surface_mode)
    conditioned_surface_valid = bool(
        local_gate_pass
        and has_advantage_embedding_weight
        and has_advantage_embedding_bias
        and surface_mode == "ADV_SERVER_REQUIRED"
        and not allow_baseline_default_advantage_embedding_init
    )

    deduped_failure_reason_codes: list[str] = []
    for code in list(failure_reason_codes):
        normalized = str(code).strip()
        if normalized and normalized not in deduped_failure_reason_codes:
            deduped_failure_reason_codes.append(normalized)
    if surface_mode != "ADV_SERVER_REQUIRED":
        deduped_failure_reason_codes.append("surface_mode_not_adv_server_required")
    if allow_baseline_default_advantage_embedding_init:
        deduped_failure_reason_codes.append(
            "baseline_default_advantage_embedding_init_not_allowed"
        )
    if not has_advantage_embedding_weight:
        deduped_failure_reason_codes.append("advantage_embedding_weight_missing")
    if not has_advantage_embedding_bias:
        deduped_failure_reason_codes.append("advantage_embedding_bias_missing")

    if conditioned_surface_valid:
        deduped_failure_reason_codes = []

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "checkpoint_path": _repo_relative(checkpoint),
        "output_json_path": _repo_relative(output_json),
        "checkpoint_surface_classification": "conditioned_requires_adv_server",
        "is_baseline_like_surface": False,
        "checkpoint_weight_map_features": features,
        "surface_selection_reason": str(
            prelim_eval_surface.get("selection_reason") or ""
        ),
        "surface_mode": surface_mode,
        "surface_mode_internal": internal_surface_mode,
        "has_advantage_embedding_weight": has_advantage_embedding_weight,
        "has_advantage_embedding_bias": has_advantage_embedding_bias,
        "allow_baseline_default_advantage_embedding_init": (
            allow_baseline_default_advantage_embedding_init
        ),
        "pass": conditioned_surface_valid,
        "status": "continue"
        if conditioned_surface_valid
        else FAILURE_STATUS_INCONCLUSIVE_CONTRACT_MISMATCH,
        "failure_reason_codes": deduped_failure_reason_codes,
        "exit_code": 0 if conditioned_surface_valid else 1,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="30h_stage3_conditioned_surface_gate.py",
        description=(
            "Inspect the posttrain conditioned checkpoint surface and persist a machine-readable gate "
            "that proves the checkpoint requires the advantage-aware server path."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Conditioned checkpoint-200 directory or concrete weight asset.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Where to atomically write the conditioned surface gate artifact.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = _resolve_path(args.checkpoint)
    output_json = _resolve_path(args.output_json)
    payload = build_gate_payload(checkpoint=checkpoint, output_json=output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    _ = write_json(output_json, payload)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return int(payload.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
