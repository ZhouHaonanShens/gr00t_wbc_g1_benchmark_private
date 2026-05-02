from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from work.recap import critic_promotion


DEFAULT_OFFLINE_AUDIT_JSON = (
    REPO_ROOT / "agent/artifacts/vlm_critic_relabel/relabel_quality_audit_v1.json"
)
DEFAULT_DOWNSTREAM_GATE_JSON = (
    REPO_ROOT / "agent/artifacts/vlm_critic_relabel/downstream_gate.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "agent/artifacts/vlm_critic_relabel/critic_upgrade_gate.json"
)


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object in {path}, got {type(payload).__name__}"
        )
    return dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit an additive critic-promotion sidecar that keeps the critic review-only "
            "unless Gates A-F, offline audit, and downstream evidence are all green."
        )
    )
    parser.add_argument(
        "--offline-audit-json",
        type=Path,
        default=DEFAULT_OFFLINE_AUDIT_JSON,
        help="Path to the 45b offline audit JSON. Missing files fail closed to review-only.",
    )
    parser.add_argument(
        "--downstream-gate-json",
        type=Path,
        default=DEFAULT_DOWNSTREAM_GATE_JSON,
        help="Path to the 45e downstream gate JSON. Missing files fail closed to review-only.",
    )
    parser.add_argument(
        "--gates-a-f-json",
        type=Path,
        default=None,
        help="Path to a machine-readable Gates A-F bundle. Missing files fail closed to review-only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the additive critic-promotion sidecar.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        offline_audit_payload = _read_optional_json(args.offline_audit_json)
        downstream_gate_payload = _read_optional_json(args.downstream_gate_json)
        gates_a_f_payload = _read_optional_json(args.gates_a_f_json)
        payload = critic_promotion.build_critic_promotion_verdict(
            offline_audit_payload=offline_audit_payload,
            downstream_gate_payload=downstream_gate_payload,
            gates_a_f_bundle=gates_a_f_payload,
            evidence_paths={
                "offline_audit_json": args.offline_audit_json,
                "downstream_gate_json": args.downstream_gate_json,
                "gates_a_f_json": args.gates_a_f_json,
            },
        )
        critic_promotion.write_critic_promotion_payload(args.output, payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
