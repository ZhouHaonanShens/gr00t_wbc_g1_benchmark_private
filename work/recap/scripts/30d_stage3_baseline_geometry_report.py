#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path


SCHEMA_VERSION = "stage3_training_geometry_report_v1"
OLD_2GPU_REFERENCE = {
    "num_gpus": 2,
    "global_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "per_device_batch_size": 2,
    "effective_update_batch": 4,
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _json_number(value: Fraction) -> int | float:
    if value.denominator == 1:
        return value.numerator
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="30d_stage3_baseline_geometry_report.py",
        description=(
            "Emit the formal stage3 training geometry report for single_gpu_v1. "
            "This proves update-batch geometry alignment only and does not claim behavior equivalence."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--num-gpus", type=_positive_int, required=True)
    _ = parser.add_argument("--global-batch-size", type=_positive_int, required=True)
    _ = parser.add_argument(
        "--gradient-accumulation-steps",
        type=_positive_int,
        required=True,
    )
    _ = parser.add_argument(
        "--dataset-transition-count",
        type=_positive_int,
        required=True,
    )
    _ = parser.add_argument("--max-steps", type=_non_negative_int, required=True)
    _ = parser.add_argument("--output-json", type=Path, required=True)
    return parser


def build_geometry_report(
    *,
    num_gpus: int,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    dataset_transition_count: int,
    max_steps: int,
) -> dict[str, object]:
    divisor = num_gpus * gradient_accumulation_steps
    if divisor == 0:
        raise ValueError(
            "num_gpus * gradient_accumulation_steps must be non-zero to compute per_device_batch_size"
        )
    if global_batch_size % divisor != 0:
        raise ValueError(
            "global_batch_size must be evenly divisible by num_gpus * gradient_accumulation_steps"
        )

    per_device_batch_size = global_batch_size // divisor
    effective_update_batch = (
        per_device_batch_size * num_gpus * gradient_accumulation_steps
    )
    if effective_update_batch <= 0:
        raise ValueError("effective_update_batch must be positive")

    optimizer_steps_per_epoch = Fraction(
        dataset_transition_count,
        effective_update_batch,
    )
    if optimizer_steps_per_epoch == 0:
        raise ValueError("optimizer_steps_per_epoch must be non-zero")
    estimated_epochs = Fraction(max_steps, 1) / optimizer_steps_per_epoch

    return {
        "schema_version": SCHEMA_VERSION,
        "num_gpus": num_gpus,
        "global_batch_size": global_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "per_device_batch_size": per_device_batch_size,
        "effective_update_batch": effective_update_batch,
        "dataset_transition_count": dataset_transition_count,
        "optimizer_steps_per_epoch": _json_number(optimizer_steps_per_epoch),
        "max_steps": max_steps,
        "estimated_epochs": _json_number(estimated_epochs),
        "effective_update_batch_matches_old_2gpu_formal": (
            effective_update_batch == OLD_2GPU_REFERENCE["effective_update_batch"]
        ),
        "old_2gpu_reference": OLD_2GPU_REFERENCE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = build_geometry_report(
            num_gpus=args.num_gpus,
            global_batch_size=args.global_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dataset_transition_count=args.dataset_transition_count,
            max_steps=args.max_steps,
        )
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
