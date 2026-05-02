from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


DEFAULT_OUTPUT_DIR = Path("agent/artifacts/apple_recap_exec/reward")
DROP_SIDECAR_JSONL_NAME = "drop_sidecar.jsonl"
DROP_SIDECAR_SUMMARY_JSON_NAME = "drop_sidecar_summary.json"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import drop_events
from work.recap.scripts import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_drop_sidecar.py",
        description=(
            "Materialize a row-level drop sidecar from authority episodes/transitions "
            "without mutating the mainline reward dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory containing episodes.jsonl and transitions.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives drop_sidecar.jsonl and drop_sidecar_summary.json.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _validate_existing_dir(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"{arg_name} directory does not exist: {resolved}")
    return resolved


def materialize_drop_sidecar(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    resolved_dataset_dir = _validate_existing_dir(dataset_dir, arg_name="dataset-dir")
    resolved_output_dir = state_conditioned_bucket_a_import.validate_output_dir(
        output_dir
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = resolved_dataset_dir / "episodes.jsonl"
    transitions_path = resolved_dataset_dir / "transitions.jsonl"
    episodes = state_conditioned_bucket_a_import._read_jsonl_dicts(episodes_path)
    transitions = state_conditioned_bucket_a_import._read_jsonl_dicts(transitions_path)
    sidecar_rows, summary = drop_events.build_drop_sidecar_rows(
        episodes=episodes,
        transitions=transitions,
    )
    sidecar_path = state_conditioned_bucket_a_import._write_jsonl(
        resolved_output_dir / DROP_SIDECAR_JSONL_NAME,
        sidecar_rows,
    )
    summary_payload = dict(summary)
    summary_payload["dataset_dir"] = str(resolved_dataset_dir)
    summary_payload["drop_sidecar_path"] = str(sidecar_path)
    summary_path = state_conditioned_bucket_a_import._write_json(
        resolved_output_dir / DROP_SIDECAR_SUMMARY_JSON_NAME,
        summary_payload,
    )
    return {
        "status": "PASS",
        "dataset_dir": str(resolved_dataset_dir),
        "drop_sidecar_path": str(sidecar_path),
        "summary_path": str(summary_path),
        "summary": summary_payload,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = materialize_drop_sidecar(args.dataset_dir, args.output_dir)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
