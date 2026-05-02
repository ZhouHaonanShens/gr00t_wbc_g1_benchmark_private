from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


DEFAULT_OUTPUT_DIR = Path("agent/artifacts/apple_recap_exec/reward")
DROP_DETECTOR_EVAL_JSON_NAME = "drop_detector_eval.json"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import drop_events
from work.recap.scripts import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_drop_detector_against_diagnostic_pool.py",
        description=(
            "Audit drop-sidecar detector behavior against a diagnostic pool without "
            "promoting reward-aware mainline reruns by default."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sidecar-jsonl",
        type=Path,
        required=True,
        help="Row-level drop sidecar JSONL produced by build_drop_sidecar.py.",
    )
    parser.add_argument(
        "--diagnostic-pool",
        type=Path,
        required=True,
        help="JSONL or JSON file describing expected drop_during_transport labels by episode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives drop_detector_eval.json.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _validate_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"missing required {arg_name}: {resolved}")
    return resolved


def _read_diagnostic_pool(path: Path) -> list[dict[str, Any]]:
    resolved = _validate_existing_file(path, arg_name="diagnostic-pool")
    if resolved.suffix.lower() == ".jsonl":
        return state_conditioned_bucket_a_import._read_jsonl_dicts(resolved)
    payload = state_conditioned_bucket_a_import._read_json(resolved)
    rows = payload.get("rows", payload.get("episodes", payload.get("diagnostic_pool")))
    if not isinstance(rows, list):
        raise ValueError(
            f"diagnostic-pool JSON must contain list-valued rows/episodes/diagnostic_pool, got {type(rows).__name__}"
        )
    return [
        dict(
            state_conditioned_bucket_a_import._as_mapping(
                row, field_name="diagnostic_pool[]"
            )
        )
        for row in rows
    ]


def materialize_drop_detector_eval(
    sidecar_jsonl: Path,
    diagnostic_pool_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    resolved_sidecar_path = _validate_existing_file(
        sidecar_jsonl, arg_name="sidecar-jsonl"
    )
    resolved_output_dir = state_conditioned_bucket_a_import.validate_output_dir(
        output_dir
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_rows = state_conditioned_bucket_a_import._read_jsonl_dicts(
        resolved_sidecar_path
    )
    diagnostic_pool_rows = _read_diagnostic_pool(diagnostic_pool_path)
    eval_payload = drop_events.evaluate_drop_detector_against_diagnostic_pool(
        sidecar_rows=sidecar_rows,
        diagnostic_pool_rows=diagnostic_pool_rows,
    )
    eval_payload["sidecar_jsonl"] = str(resolved_sidecar_path)
    eval_payload["diagnostic_pool_path"] = str(
        diagnostic_pool_path.expanduser().resolve()
    )
    eval_path = state_conditioned_bucket_a_import._write_json(
        resolved_output_dir / DROP_DETECTOR_EVAL_JSON_NAME,
        eval_payload,
    )
    _ = eval_path
    return dict(eval_payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = materialize_drop_detector_eval(
            args.sidecar_jsonl,
            args.diagnostic_pool,
            args.output_dir,
        )
    except Exception as exc:
        payload = {
            "status": "FAIL",
            "error": {
                "type": exc.__class__.__name__,
                "message": _exception_message(exc),
            },
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if str(payload.get("status")) == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
