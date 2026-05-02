from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


DEFAULT_ARTIFACT_DIR = "agent/artifacts"
DEFAULT_OUTPUT_SUBDIR = "flux_dataset_probe"
DEFAULT_EVIDENCE_JSON = ".sisyphus/evidence/task-7-dataset-inventory-bundle.json"
DEFAULT_DATASET_DIR = "agent/artifacts/lerobot_datasets/physical_intelligence_libero_official_8d_recap_relabels_v1"

DATASET_INVENTORY_BUNDLE_JSON_NAME = "dataset_inventory_bundle.json"

EVIDENCE_SCHEMA_VERSION = "flux_dataset_probe_evidence_v1"
EVIDENCE_ARTIFACT_KIND = "flux_dataset_probe_evidence"


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import state_conditioned_bucket_a_import
from work.recap.datasets.flux_grouped_dataset import build_flux_dataset_inventory_bundle
from work.recap.datasets.flux_grouped_dataset import inventory_bundle_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_flux_dataset_probe.py",
        description=(
            "Materialize a provenance-bound Flux dataset inventory bundle with dataset-side "
            "blockers and a future-facing binding_join_contract."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="Dataset root to inspect for Flux/LeRobot-style meta, stats, and parquet provenance.",
    )
    _ = parser.add_argument(
        "--artifact-dir",
        type=str,
        default=DEFAULT_ARTIFACT_DIR,
        help=(
            "Artifact root. When --output-dir is empty, dataset inventory JSON is written to "
            "<artifact-dir>/flux_dataset_probe/."
        ),
    )
    _ = parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit output directory for generated inventory JSON.",
    )
    _ = parser.add_argument(
        "--evidence-json",
        type=str,
        default=DEFAULT_EVIDENCE_JSON,
        help="Evidence JSON written after dataset inventory generation succeeds.",
    )
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _relpath(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_output_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    raw_output_dir = str(args.output_dir).strip()
    if raw_output_dir:
        return state_conditioned_bucket_a_import.validate_output_dir(
            _resolve_path(repo_root, raw_output_dir)
        )
    artifact_dir = _resolve_path(repo_root, str(args.artifact_dir))
    return state_conditioned_bucket_a_import.validate_output_dir(
        artifact_dir / DEFAULT_OUTPUT_SUBDIR
    )


def resolve_evidence_json(repo_root: Path, args: argparse.Namespace) -> Path:
    return _resolve_path(repo_root, str(args.evidence_json))


def resolve_dataset_dir(repo_root: Path, args: argparse.Namespace) -> Path:
    return _resolve_path(repo_root, str(args.dataset_dir))


def write_artifacts(
    *,
    output_dir: Path,
    evidence_json: Path,
    dataset_dir: Path,
    inventory_payload: Mapping[str, Any],
) -> dict[str, str]:
    inventory_path = state_conditioned_bucket_a_import._write_json(
        output_dir / DATASET_INVENTORY_BUNDLE_JSON_NAME,
        dict(inventory_payload),
    )
    evidence_payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": EVIDENCE_ARTIFACT_KIND,
        "dataset_dir": str(dataset_dir),
        "inventory_verdict": str(inventory_payload.get("verdict")),
        "blocking_reason_count": len(
            list(inventory_payload.get("blocking_reasons", []))
        ),
        "backpointer": {
            "dataset_inventory_bundle_json": str(inventory_path),
        },
        "validation": {
            "help_command": "python3 work/recap/scripts/build_flux_dataset_probe.py --help",
            "wrapper_help_command": "python3 agent/run/build_flux_dataset_probe.py --help",
            "test_command": (
                "python3 -m pytest tests/recap/test_flux_dataset_inventory_binding.py -q"
            ),
        },
    }
    evidence_path = state_conditioned_bucket_a_import._write_json(
        evidence_json,
        evidence_payload,
    )
    return {
        "dataset_inventory_bundle_json": str(inventory_path),
        "evidence_json": str(evidence_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        dataset_dir = resolve_dataset_dir(REPO_ROOT, args)
        output_dir = resolve_output_dir(REPO_ROOT, args)
        evidence_json = resolve_evidence_json(REPO_ROOT, args)
        inventory_bundle = build_flux_dataset_inventory_bundle(dataset_dir)
        inventory_payload = inventory_bundle_to_dict(inventory_bundle)
        written_paths = write_artifacts(
            output_dir=output_dir,
            evidence_json=evidence_json,
            dataset_dir=dataset_dir,
            inventory_payload=inventory_payload,
        )
        print(
            _canonical_json_text(
                {
                    "status": "PASS",
                    "dataset_dir": str(dataset_dir),
                    "output_dir": str(output_dir),
                    "inventory_verdict": inventory_payload["verdict"],
                    **written_paths,
                }
            ),
            end="",
        )
        return 0
    except Exception as exc:
        print(_exception_message(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
