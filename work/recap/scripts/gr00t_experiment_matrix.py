from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import experiment_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a thin B0/B1/E1-E4 experiment-matrix layer that reuses "
            "existing baseline-freeze, run-manifest, scope, label-policy, and "
            "same-checkpoint triplet surfaces without rewriting their schemas."
        )
    )
    parser.add_argument(
        "--baseline-freeze-json",
        type=Path,
        required=True,
        help="Path to the existing gr00t_baseline_freeze_matrix.json artifact.",
    )
    parser.add_argument(
        "--experiment-spec-json",
        type=Path,
        required=True,
        help=(
            "JSON file with an experiment_rows list. Each row must provide display_label, "
            "run_manifest_path, and triplet_summary_path; row_id/compare_to_row_id are optional."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=experiment_matrix.DEFAULT_OUTPUT,
        help="Output path for gr00t_experiment_matrix.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = experiment_matrix.write_experiment_matrix_from_paths(
            baseline_freeze_json=args.baseline_freeze_json,
            experiment_spec_json=args.experiment_spec_json,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
