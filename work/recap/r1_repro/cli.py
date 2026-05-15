from __future__ import annotations

import argparse
from dataclasses import replace
import datetime as dt
from pathlib import Path

from .gates import Verdict
from .gates import gate_r1_0_baseline_reproduction
from .repro_runner import ARTIFACT_ROOT
from .repro_runner import R1BaselineMarkerStale
from .repro_runner import R1BaselineNotPassed
from .repro_runner import _git_diff_clean_outside_artifact_dir
from .repro_runner import _persist_baseline_pass_marker
from .repro_runner import run_protocol
from .repro_runner import validate_baseline_pass_marker


def _utc_slug() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_out(cell: str) -> Path:
    return ARTIFACT_ROOT / cell / _utc_slug()


def _csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _protocol_for_cell(cell: str):
    from .protocol import P0B_PROTOCOL, T81_B0_PROTOCOL, T81_B0_VARIANT_CKPT_ROOT

    if cell == "r1_1_A":
        return replace(P0B_PROTOCOL, max_episode_steps=720)
    if cell == "r1_1_B":
        return replace(P0B_PROTOCOL, seed_base=2026051000)
    if cell == "r1_1_C":
        return replace(
            P0B_PROTOCOL,
            driver_script=T81_B0_PROTOCOL.driver_script,
            driver_sha256=T81_B0_PROTOCOL.driver_sha256,
            ckpt_root=T81_B0_VARIANT_CKPT_ROOT,
        )
    if cell == "r1_1_E":
        return replace(P0B_PROTOCOL, ckpt_root=T81_B0_VARIANT_CKPT_ROOT)
    raise ValueError(f"unsupported GPU ablation cell: {cell}")


def _run_variant_audit(args: argparse.Namespace) -> int:
    from .ckpt_variant_audit import audit_variant
    from .ckpt_variant_audit import classify_risk
    from .ckpt_variant_audit import inventory_symlinks
    from .gates import gate_r1_2_variant_audit
    from .reports.variant_audit_report import render_variant_audit

    out_dir = Path(args.out)
    audit = audit_variant(Path(args.variant_root), Path(args.raw_hf_snapshot), _csv_list(args.files))
    risk = classify_risk(audit)
    symlinks = inventory_symlinks(Path(args.variant_root), Path(args.raw_hf_snapshot))
    verdict = gate_r1_2_variant_audit(audit, risk, Path(args.variant_root))
    render_variant_audit(
        {**audit, "verdict": verdict.value},
        risk,
        symlinks,
        out_dir,
    )
    return 0 if verdict in {Verdict.PASS_CLEAN, Verdict.PASS_WITH_RISK} else 2


def _run_baseline(args: argparse.Namespace) -> int:
    from .protocol import P0B_PROTOCOL
    from .reports.repro_cell_report import render_repro_cell

    out_dir = Path(args.out) if args.out else _default_out("r1_0")
    result = run_protocol(P0B_PROTOCOL, out_dir)
    verdict = gate_r1_0_baseline_reproduction(
        result.success_count,
        result.formal_eval_summary_status,
        len(result.per_episode),
        _git_diff_clean_outside_artifact_dir(out_dir),
    )
    render_repro_cell(result, verdict, None, out_dir)
    if verdict == Verdict.PASS:
        config_path = Path(P0B_PROTOCOL.ckpt_root) / "config.json"
        import hashlib

        _persist_baseline_pass_marker(result, hashlib.sha256(config_path.read_bytes()).hexdigest())
        (out_dir / "r1_0_pass_marker.txt").write_text("PASS\n", encoding="utf-8")
        return 0
    (out_dir / "r1_0_fail_dossier.md").write_text(
        f"# R1.0 Fail Dossier\n\n- verdict: {verdict.value}\n",
        encoding="utf-8",
    )
    return 2


def _run_ablation(args: argparse.Namespace) -> int:
    from .protocol import P0B_PROTOCOL
    from .reports.repro_cell_report import render_repro_cell

    validate_baseline_pass_marker(P0B_PROTOCOL)
    if args.cell == "r1_1_D":
        raise NotImplementedError(
            "r1_1_D post-classifier augmentation is intentionally non-GPU and "
            "requires a concrete baseline-cell artifact path."
        )
    protocol = _protocol_for_cell(args.cell)
    out_dir = Path(args.out) if args.out else _default_out(args.cell)
    result = run_protocol(protocol, out_dir)
    verdict = (
        Verdict.PASS
        if result.formal_eval_summary_status == "PASS" and len(result.per_episode) == 30
        else Verdict.FAIL_INCOMPLETE
    )
    render_repro_cell(
        result,
        verdict,
        float(args.baseline_rate) if args.baseline_rate is not None else None,
        out_dir,
    )
    return 0 if verdict == Verdict.PASS else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R1 substrate-recovery reproduction harness")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("variant-audit")
    audit.add_argument("--raw-hf-snapshot", required=True)
    audit.add_argument("--variant-root", required=True)
    audit.add_argument("--files", default="config.json,processor_config.json")
    audit.add_argument("--out", required=True)
    audit.set_defaults(func=_run_variant_audit)

    baseline = sub.add_parser("baseline-repro")
    baseline.add_argument("--out", default="")
    baseline.set_defaults(func=_run_baseline)

    ablation = sub.add_parser("ablation")
    ablation.add_argument("--cell", required=True, choices=["r1_1_A", "r1_1_B", "r1_1_C", "r1_1_D", "r1_1_E"])
    ablation.add_argument("--baseline-cell", default="")
    ablation.add_argument("--baseline-rate", type=float, default=None)
    ablation.add_argument("--out", default="")
    ablation.set_defaults(func=_run_ablation)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
