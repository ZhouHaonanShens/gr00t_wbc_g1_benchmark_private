from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import label_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit additive GR00T label dashboard and label-policy sidecars from "
            "existing immutable state-conditioned label rows and stats."
        )
    )
    parser.add_argument(
        "--labels-jsonl",
        type=Path,
        required=True,
        help="Path to state_conditioned_sft_labels.jsonl.",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        required=True,
        help="Path to state_conditioned_sft_stats.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that receives gr00t_label_dashboard.json and gr00t_label_policy.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = label_policy.write_label_dashboard_sidecars(
            labels_jsonl_path=args.labels_jsonl,
            stats_json_path=args.stats_json,
            output_dir=args.output_dir,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
