#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.openpi.recap.dataset_aggregation import (  # noqa: E402
    MergedDatasetBundle,
    load_collection_bundle,
    materialize_merged_dataset,
)
from work.openpi.sources.libero_official.validate import (  # noqa: E402
    BLOCKED_EXIT_CODE,
    DEFAULT_DATASET_DIR,
    build_source_prereq_report,
)


@dataclass(frozen=True)
class MergeConfig:
    demo_dir: Path
    autonomous_dir: Path
    output_dir: Path
    correction_dir: Path | None


class MergeWorkflow:
    def __init__(self, config: MergeConfig) -> None:
        self.config: MergeConfig = config

    def run(self) -> MergedDatasetBundle:
        canonical_source_report = self._validate_canonical_source()
        collection_bundle = load_collection_bundle(self.config.autonomous_dir)
        return materialize_merged_dataset(
            demo_dir=self.config.demo_dir,
            canonical_source_report=canonical_source_report,
            collection_bundle=collection_bundle,
            output_dir=self.config.output_dir,
            correction_dir=self.config.correction_dir,
        )

    def _validate_canonical_source(self) -> dict[str, object]:
        report = build_source_prereq_report(self.config.demo_dir)
        if report.get("status") != "ready":
            raise RuntimeError(f"canonical demo source blocked: {report}")
        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="libero_recap_merge_data.py",
        description=(
            "Merge canonical official/native 8D demos with autonomous task-9 collections and optional correction segments."
        ),
    )
    _ = parser.add_argument("--demo-dir", default=str(DEFAULT_DATASET_DIR))
    _ = parser.add_argument("--autonomous-dir", required=True)
    _ = parser.add_argument("--output-dir", required=True)
    _ = parser.add_argument("--correction-dir", default=None)
    return parser


def _build_config(args: argparse.Namespace) -> MergeConfig:
    return MergeConfig(
        demo_dir=Path(cast(str, args.demo_dir)).expanduser().resolve(),
        autonomous_dir=Path(cast(str, args.autonomous_dir)).expanduser().resolve(),
        output_dir=Path(cast(str, args.output_dir)).expanduser().resolve(),
        correction_dir=(
            Path(cast(str, args.correction_dir)).expanduser().resolve()
            if isinstance(args.correction_dir, str) and str(args.correction_dir).strip()
            else None
        ),
    )


def run_merge(config: MergeConfig) -> MergedDatasetBundle:
    return MergeWorkflow(config).run()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _ = run_merge(_build_config(args))
    except RuntimeError as exc:
        if str(exc).startswith("canonical demo source blocked:"):
            return BLOCKED_EXIT_CODE
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
