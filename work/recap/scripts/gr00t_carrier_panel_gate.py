from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any, cast


sys.dont_write_bytecode = True


DEFAULT_OUTPUT_JSON = Path(
    "agent/artifacts/gr00t_anchor_controller_recap/unitree_g1/carrier_panel_gate.json"
)
DEFAULT_NORMALIZED_THRESHOLD = 0.05
REPORT_SCHEMA_VERSION = "carrier_panel_gate_v1"
REPORT_ARTIFACT_KIND = "carrier_panel_gate"
WRITER_SCRIPT = "work/recap/scripts/gr00t_carrier_panel_gate.py"
NORMALIZED_DELTA_METRIC_NAME = "mean_abs_delta_over_contract_range"

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t_carrier_panel_gate.py",
        description=(
            "Score the deterministic five-point carrier panel and decide whether runtime carrier consumption is weak or robust."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--panel-json",
        type=Path,
        required=True,
        help="Carrier panel manifest JSON produced by build_carrier_panel.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output carrier panel gate JSON path.",
    )
    parser.add_argument(
        "--normalized-threshold",
        type=float,
        default=DEFAULT_NORMALIZED_THRESHOLD,
        help=(
            "Controller-level threshold applied to mean_abs_delta_over_contract_range."
        ),
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def _validate_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and resolved.is_dir():
        raise ValueError(f"out must be a file path, got directory: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object in {path}, got {type(payload).__name__}")
    return dict(payload)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: object) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _resolve_runtime_trace(sample: Mapping[str, object]) -> Mapping[str, object]:
    debug_probe = sample.get("debug_probe")
    if isinstance(debug_probe, Mapping):
        runtime_trace = debug_probe.get("runtime_trace")
        if isinstance(runtime_trace, Mapping):
            return cast(Mapping[str, object], runtime_trace)
    runtime_trace = sample.get("runtime_trace")
    if isinstance(runtime_trace, Mapping):
        return cast(Mapping[str, object], runtime_trace)
    raise TypeError("panel sample is missing debug_probe.runtime_trace")


def build_panel_sample_summary(
    sample: Mapping[str, object],
    *,
    normalized_threshold: float,
) -> dict[str, object]:
    runtime_trace = _resolve_runtime_trace(sample)
    stage_max = cast(
        Mapping[str, object],
        runtime_trace.get("stage_max_mean_abs_delta_over_contract_range", {}),
    )
    prompt_surface = cast(Mapping[str, object], runtime_trace.get("prompt_surface", {}))
    token_surface = cast(Mapping[str, object], runtime_trace.get("token_surface", {}))
    upstream = cast(Mapping[str, object], runtime_trace.get("upstream_distinction", {}))
    decoded_action = _float_or_none(stage_max.get("decoded_action"))
    absolute_action = _float_or_none(stage_max.get("absolute_action"))
    controller_input = _float_or_none(stage_max.get("controller_input"))
    controller_output = _float_or_none(stage_max.get("controller_output"))
    prompt_or_token_distinct = bool(
        upstream.get(
            "prompt_or_token_distinct",
            bool(prompt_surface.get("any_pair_distinct", False))
            or bool(token_surface.get("any_pair_distinct", False)),
        )
    )
    raw_or_decoded_distinct = bool(upstream.get("raw_or_decoded_distinct", False))
    controller_candidates = [
        value for value in (controller_input, controller_output) if value is not None
    ]
    controller_level_delta = (
        max(controller_candidates) if controller_candidates else None
    )
    controller_stage_used = None
    if controller_level_delta is not None:
        controller_stage_used = (
            "controller_output"
            if controller_output is not None
            and controller_input is not None
            and controller_output >= controller_input
            else "controller_output"
            if controller_output is not None and controller_input is None
            else "controller_input"
        )
    panel_pass = bool(
        controller_level_delta is not None
        and float(controller_level_delta) >= float(normalized_threshold)
    )
    if not prompt_or_token_distinct or not raw_or_decoded_distinct:
        status = "not_consumed"
    elif panel_pass:
        status = "consumed_and_survived"
    elif (
        bool(upstream.get("absolute_distinct", False))
        or bool(upstream.get("raw_action_distinct", False))
        or (decoded_action is not None and decoded_action > 0.0)
    ):
        status = "consumed_but_absorbed_downstream"
    else:
        status = "not_consumed"
    return {
        "sample_id": str(sample.get("sample_id", "")),
        "panel_slot_index": sample.get("panel_slot_index"),
        "panel_slot_name": sample.get("panel_slot_name"),
        "episode_index": sample.get("episode_index"),
        "outer_step": sample.get("outer_step"),
        "episode_outcome": sample.get("episode_outcome"),
        "status": status,
        "panel_pass": bool(panel_pass),
        "prompt_or_token_distinct": bool(prompt_or_token_distinct),
        "raw_or_decoded_distinct": bool(raw_or_decoded_distinct),
        "decoded_action_" + NORMALIZED_DELTA_METRIC_NAME: decoded_action,
        "absolute_action_" + NORMALIZED_DELTA_METRIC_NAME: absolute_action,
        "controller_input_" + NORMALIZED_DELTA_METRIC_NAME: controller_input,
        "controller_output_" + NORMALIZED_DELTA_METRIC_NAME: controller_output,
        "controller_level_" + NORMALIZED_DELTA_METRIC_NAME: controller_level_delta,
        "controller_stage_used": controller_stage_used,
        "controller_output_available": bool(
            runtime_trace.get("controller_output_available", False)
        ),
        "controller_output_unavailable_reason": runtime_trace.get(
            "controller_output_unavailable_reason"
        ),
    }


def build_carrier_panel_gate(
    panel_payload: Mapping[str, object],
    *,
    normalized_threshold: float = DEFAULT_NORMALIZED_THRESHOLD,
) -> dict[str, object]:
    samples_raw = panel_payload.get("samples")
    if not isinstance(samples_raw, list):
        raise TypeError("panel.samples must be a list")
    per_sample_summary = [
        build_panel_sample_summary(
            cast(Mapping[str, object], sample),
            normalized_threshold=normalized_threshold,
        )
        for sample in samples_raw
    ]
    panel_pass_count = int(
        sum(1 for sample in per_sample_summary if bool(sample["panel_pass"]))
    )
    panel_sample_count = int(len(per_sample_summary))
    if panel_pass_count > 0:
        status = "consumed_and_survived"
    elif any(
        sample["status"] == "consumed_but_absorbed_downstream"
        for sample in per_sample_summary
    ):
        status = "consumed_but_absorbed_downstream"
    else:
        status = "not_consumed"
    gate_strength = (
        "robust" if panel_sample_count >= 5 and panel_pass_count >= 3 else "weak"
    )
    if gate_strength == "robust":
        gate_strength_reason = "at least 3/5 deterministic panel samples cleared the controller-level normalized threshold"
    elif panel_sample_count < 5:
        gate_strength_reason = "panel has fewer than five deterministic samples, so it remains a debug probe only"
    elif panel_pass_count in {1, 2}:
        gate_strength_reason = "only 1-2 deterministic panel samples cleared the controller-level normalized threshold"
    else:
        gate_strength_reason = (
            "no deterministic panel reached the robust 3/5 controller-level threshold"
        )
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "status": status,
        "gate_strength": gate_strength,
        "gate_strength_reason": gate_strength_reason,
        "panel_pass_count": int(panel_pass_count),
        "panel_sample_count": int(panel_sample_count),
        "normalized_threshold": float(normalized_threshold),
        "normalization_metric": NORMALIZED_DELTA_METRIC_NAME,
        "mainline_unlock": bool(
            status == "consumed_and_survived" and gate_strength == "robust"
        ),
        "controller_output_available": bool(
            any(
                bool(sample["controller_output_available"])
                for sample in per_sample_summary
            )
        ),
        "per_sample_summary": per_sample_summary,
        "backpointer": {
            "writer_script": WRITER_SCRIPT,
            "panel_report_signature_sha256": panel_payload.get(
                "report_signature_sha256"
            ),
        },
    }
    payload["report_signature_sha256"] = _sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "report_signature_sha256"
        }
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        panel_path = Path(args.panel_json).expanduser()
        if not panel_path.is_absolute():
            panel_path = (REPO_ROOT / panel_path).resolve()
        gate = build_carrier_panel_gate(
            _read_json(panel_path),
            normalized_threshold=float(args.normalized_threshold),
        )
        output_path = _validate_output_path(Path(args.out))
        gate["output_path"] = str(output_path)
        gate["backpointer"] = {
            **cast(Mapping[str, object], gate["backpointer"]),
            "panel_json": str(panel_path),
        }
        gate["report_signature_sha256"] = _sha256(
            {
                key: value
                for key, value in gate.items()
                if key != "report_signature_sha256"
            }
        )
        _ = _write_json(output_path, gate)
        print(json.dumps(gate, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
