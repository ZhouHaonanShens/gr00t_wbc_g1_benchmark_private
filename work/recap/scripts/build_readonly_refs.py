#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True


DEFAULT_OUTPUT = Path(
    "agent/artifacts/apple_recap_exec/phase_a_tooling_draft/baseline_refs_manifest.json"
)

SCHEMA_VERSION = "apple_recap_baseline_refs_manifest_v1"
ARTIFACT_KIND = "apple_recap_baseline_refs_manifest"

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap.scripts import apple_recap_execution_contract
from work.recap.scripts import state_conditioned_bucket_a_import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_readonly_refs.py",
        description=(
            "Materialize the Phase-A baseline readonly ref manifest with fail-closed "
            "authority-file validation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


def _resolve_path(path: Path | str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    return raw.resolve()


def _validate_output_path(path: Path | str) -> Path:
    resolved = _resolve_path(path)
    if resolved.exists():
        raise ValueError(
            f"baseline refs manifest output already exists (no-overwrite): {resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _signature_for_payload(payload: Mapping[str, Any]) -> str:
    signature_basis = {
        str(key): value
        for key, value in dict(payload).items()
        if key != "report_signature_sha256"
    }
    return apple_recap_execution_contract._sha256_payload(signature_basis)


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    return state_conditioned_bucket_a_import._write_json(path, payload)


def build_baseline_refs_manifest(
    *,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
    execution_sha: str = apple_recap_execution_contract.UNSET_EXECUTION_SHA,
    read_only_authority_ref_specs: Sequence[
        Mapping[str, str]
    ] = apple_recap_execution_contract.DEFAULT_READ_ONLY_AUTHORITY_REF_SPECS,
) -> dict[str, Any]:
    read_only_authority_refs = [
        apple_recap_execution_contract.build_read_only_authority_ref(
            repo_root=repo_root,
            artifact_id=str(spec["artifact_id"]),
            authority_role=str(spec["authority_role"]),
            relative_path=str(spec["relative_path"]),
        )
        for spec in read_only_authority_ref_specs
    ]
    artifact_id_order = [
        str(ref["artifact_id"])
        for ref in read_only_authority_refs
        if "artifact_id" in ref
    ]
    authority_roles = sorted(
        {
            str(ref["authority_role"])
            for ref in read_only_authority_refs
            if "authority_role" in ref
        }
    )
    core = {"commit": str(execution_sha)}
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "generated_at": generated_at or _now_iso(),
        "execution_sha": str(execution_sha),
        "core": core,
        "core_digest": apple_recap_execution_contract.core_digest(core),
        "read_only_authority_refs": read_only_authority_refs,
        "artifact_id_order": artifact_id_order,
        "authority_roles": authority_roles,
        "freeze_policy": {
            "append_only": True,
            "no_overwrite": True,
            "source_artifacts_read_only": True,
            "source_artifacts_mutated": False,
            "missing_required_authority_ref_behavior": "fail_closed",
        },
    }
    payload["report_signature_sha256"] = _signature_for_payload(payload)
    return payload


def materialize_baseline_refs_manifest(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    repo_root: Path = REPO_ROOT,
    generated_at: str | None = None,
    execution_sha: str = apple_recap_execution_contract.UNSET_EXECUTION_SHA,
    read_only_authority_ref_specs: Sequence[
        Mapping[str, str]
    ] = apple_recap_execution_contract.DEFAULT_READ_ONLY_AUTHORITY_REF_SPECS,
) -> dict[str, Any]:
    resolved_output = _validate_output_path(output)
    payload = build_baseline_refs_manifest(
        repo_root=repo_root,
        generated_at=generated_at,
        execution_sha=execution_sha,
        read_only_authority_ref_specs=read_only_authority_ref_specs,
    )
    _ = _write_json(resolved_output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = materialize_baseline_refs_manifest(output=args.output)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {_exception_message(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ARTIFACT_KIND",
    "DEFAULT_OUTPUT",
    "SCHEMA_VERSION",
    "build_baseline_refs_manifest",
    "build_parser",
    "main",
    "materialize_baseline_refs_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
