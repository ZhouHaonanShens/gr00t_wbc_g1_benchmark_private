from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


DEFAULT_OUTPUT_DIR = Path("agent/artifacts/apple_recap_exec/reward")
COUNTERFACTUAL_REWARDS_JSONL_NAME = "counterfactual_rewards.jsonl"
COUNTERFACTUAL_SUMMARY_JSON_NAME = "counterfactual_rewards_summary.json"
REWARD_RECOMMENDATION_JSON_NAME = "reward_recommendation.json"
REWARD_COUNTERFACTUAL_REPORT_MD_NAME = "reward_counterfactual_report.md"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import drop_events
from work.recap.scripts import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relabel_counterfactual_rewards.py",
        description=(
            "Produce offline-only counterfactual reward artifacts plus a fail-closed "
            "reward recommendation surface."
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
        "--sidecar-jsonl",
        type=Path,
        required=True,
        help="Row-level drop sidecar JSONL produced by build_drop_sidecar.py.",
    )
    parser.add_argument(
        "--detector-eval-json",
        type=Path,
        default=None,
        help="Optional detector audit JSON. Missing input keeps the recommendation fail-closed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory that receives counterfactual_rewards.jsonl, summary JSON, "
            "reward_recommendation.json, and reward_counterfactual_report.md."
        ),
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


def _validate_existing_file(path: Path, *, arg_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"missing required {arg_name}: {resolved}")
    return resolved


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def materialize_counterfactual_rewards(
    dataset_dir: Path,
    sidecar_jsonl: Path,
    output_dir: Path,
    *,
    detector_eval_json: Path | None = None,
) -> dict[str, Any]:
    resolved_dataset_dir = _validate_existing_dir(dataset_dir, arg_name="dataset-dir")
    resolved_sidecar_path = _validate_existing_file(
        sidecar_jsonl, arg_name="sidecar-jsonl"
    )
    resolved_output_dir = state_conditioned_bucket_a_import.validate_output_dir(
        output_dir
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    episodes = state_conditioned_bucket_a_import._read_jsonl_dicts(
        resolved_dataset_dir / "episodes.jsonl"
    )
    transitions = state_conditioned_bucket_a_import._read_jsonl_dicts(
        resolved_dataset_dir / "transitions.jsonl"
    )
    sidecar_rows = state_conditioned_bucket_a_import._read_jsonl_dicts(
        resolved_sidecar_path
    )
    counterfactual_rows, counterfactual_summary = (
        drop_events.relabel_counterfactual_rewards(
            episodes=episodes,
            transitions=transitions,
            sidecar_rows=sidecar_rows,
        )
    )
    sidecar_summary = drop_events.summarize_drop_sidecar(
        sidecar_rows,
        expected_transition_keys=[
            (str(row["episode_id"]), int(row["t"])) for row in transitions
        ],
    )
    detector_eval_payload: dict[str, Any] | None = None
    resolved_detector_eval_path: Path | None = None
    if detector_eval_json is not None:
        resolved_detector_eval_path = _validate_existing_file(
            detector_eval_json,
            arg_name="detector-eval-json",
        )
        detector_eval_payload = state_conditioned_bucket_a_import._read_json(
            resolved_detector_eval_path
        )

    rows_path = state_conditioned_bucket_a_import._write_jsonl(
        resolved_output_dir / COUNTERFACTUAL_REWARDS_JSONL_NAME,
        counterfactual_rows,
    )
    summary_payload = dict(counterfactual_summary)
    summary_payload["dataset_dir"] = str(resolved_dataset_dir)
    summary_payload["sidecar_jsonl"] = str(resolved_sidecar_path)
    summary_path = state_conditioned_bucket_a_import._write_json(
        resolved_output_dir / COUNTERFACTUAL_SUMMARY_JSON_NAME,
        summary_payload,
    )

    report_path = resolved_output_dir / REWARD_COUNTERFACTUAL_REPORT_MD_NAME
    recommendation_path = resolved_output_dir / REWARD_RECOMMENDATION_JSON_NAME
    recommendation_payload = drop_events.build_reward_recommendation(
        sidecar_summary=sidecar_summary,
        counterfactual_summary=summary_payload,
        detector_eval=detector_eval_payload,
        evidence_paths={
            "drop_sidecar_jsonl": str(resolved_sidecar_path),
            "drop_sidecar_summary_json": str(
                resolved_sidecar_path.with_name("drop_sidecar_summary.json")
            ),
            "drop_detector_eval_json": (
                str(resolved_detector_eval_path)
                if resolved_detector_eval_path is not None
                else None
            ),
            "counterfactual_rows_jsonl": str(rows_path),
            "counterfactual_summary_json": str(summary_path),
            "reward_counterfactual_report_md": str(report_path),
        },
    )
    report_text = drop_events.render_reward_counterfactual_report(
        sidecar_summary=sidecar_summary,
        detector_eval=detector_eval_payload,
        counterfactual_summary=summary_payload,
        recommendation=recommendation_payload,
    )
    report_path = _write_text(report_path, report_text)
    recommendation_payload["evidence_paths"] = dict(
        recommendation_payload["evidence_paths"]
    )
    recommendation_payload["evidence_paths"]["reward_counterfactual_report_md"] = str(
        report_path
    )
    recommendation_path = state_conditioned_bucket_a_import._write_json(
        recommendation_path,
        recommendation_payload,
    )
    _ = rows_path
    _ = summary_path
    _ = recommendation_path
    return dict(recommendation_payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = materialize_counterfactual_rewards(
            args.dataset_dir,
            args.sidecar_jsonl,
            args.output_dir,
            detector_eval_json=args.detector_eval_json,
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
    return 0 if str(payload.get("formal_eligibility")) == "ALLOW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
