"""R2 CLI subcommand implementations (R2.0/R2.1/R2.2/summarise pipeline).

Split from ``cli.py`` to keep the CLI module under its 200-LOC budget. Pure
business logic + filesystem writes live here; argparse plumbing lives in
``cli.py``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from work.recap.r1_repro.protocol import P0B_PROTOCOL, RAW_HF_SNAPSHOT_ROOT
from work.recap.r1_repro.repro_runner import ARTIFACT_ROOT, validate_baseline_pass_marker
from work.recap.r2_authentic_eval.ckpt_config_swap import (
    CkptByteIdentityViolation,
    CkptRawHfMissingArtifact,
    CkptSourceMissingArtifact,
    CkptSrcMutatedDuringSwap,
    SWAP_ROOT_DEFAULT,
    materialise_swap_ckpt,
)
from work.recap.r2_authentic_eval.config_delta import (
    ADDITIONAL_FIELDS_DIFFER,
    ACKNOWLEDGMENT_FILENAME,
    ATTENTION_FILENAME,
    FORMALIZE_LANGUAGE_PATHS,
    INVENTORY_FILENAME,
    AcknowledgmentMissingError,
    audit_inventory,
    require_acknowledgment,
)
from work.recap.r2_authentic_eval.delta_stats import (
    R2_BASELINE_N_DEFAULT,
    R2_BASELINE_SUCC_DEFAULT,
    R2_TRIGGER_THRESHOLD,
    family_wise_error_rate_at_baseline,
    newcombe_half_width_at_baseline,
    per_cell_below_trigger_probability,
)
from work.recap.r2_authentic_eval.eval_runner import (
    AuthenticEvalRequest,
    R2CellResult,
    _capture_failure_context,
    run_r2_1_cell,
)
from work.recap.r2_authentic_eval.inventory import (
    R2_VALID_CELL_COUNT_EXPECTED,
    discover_recap_ckpts,
    filter_valid,
    pick_representative,
)
from work.recap.r2_authentic_eval.reports import (
    cell_report,
    inventory_report,
)
from work.recap.r2_authentic_eval.reports.config_delta_report import (
    render_config_delta_subsection,
)


_LOG = logging.getLogger("r2_authentic_eval")
R2_INVENTORY_OUT = ARTIFACT_ROOT / "r2_0_inventory"
R2_CONFIG_DELTA_OUT = ARTIFACT_ROOT / "r2_0_5_planning"


def _utc_slug() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_statistical_regime(
    *,
    baseline_marker: dict[str, Any],
    n_valid_cells: int,
) -> dict[str, Any]:
    """Populate the runtime statistical_regime dict (V3-FIX-1 + V4-FIX-1 SSOT).

    Reads ``baseline_succ`` / ``baseline_total`` from the marker payload at the
    call site (Critic 5.1 closure: never from a module-level mutable global);
    propagates ``n_valid_cells`` from the live R2.0 inventory count.
    """
    succ = int(baseline_marker.get("success_count", R2_BASELINE_SUCC_DEFAULT))
    total = int(baseline_marker.get("episode_count", R2_BASELINE_N_DEFAULT))
    return {
        "baseline_succ": succ,
        "baseline_total": total,
        "n_valid_cells": int(n_valid_cells),
        "per_cell_p_below_trigger": per_cell_below_trigger_probability(succ, total),
        "family_wise_at_baseline": family_wise_error_rate_at_baseline(
            succ, total, n_cells=int(n_valid_cells)
        ),
        "newcombe_half_width_at_baseline": newcombe_half_width_at_baseline(succ, total),
        "regime_label": "broad-net pilot signal",
    }


def run_inventory(args: argparse.Namespace) -> int:
    """R2.0 inventory subcommand — disk-only, no GPU."""
    search_root = Path(args.search_root).resolve()
    out_dir = Path(args.out) if args.out else (R2_INVENTORY_OUT / _utc_slug())
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = discover_recap_ckpts(search_root)
    valid_cells = filter_valid(inventory)
    n_valid = len(valid_cells)
    if n_valid != R2_VALID_CELL_COUNT_EXPECTED:
        _LOG.warning(
            "R2.0 inventory found %d valid cells; plan-time expectation is %d "
            "(R2_VALID_CELL_COUNT_EXPECTED). Closure narrative will use the "
            "runtime count (SSOT discipline per V4-FIX-1); informational only.",
            n_valid, R2_VALID_CELL_COUNT_EXPECTED,
        )

    md = inventory_report.render(inventory)
    (out_dir / "inventory_report.md").write_text(md, encoding="utf-8")

    done_payload = {
        "search_root": str(search_root),
        "total_recap_count": sum(1 for c in inventory if c.label == "RECAP"),
        "valid_recap_count": n_valid,
        "valid_cell_count_expected": R2_VALID_CELL_COUNT_EXPECTED,
        "candidates_total": len(inventory),
        "classification": "RECAP (negative-token rule)",
        "captured_at_utc": _utc_slug(),
    }
    (out_dir / "r2_0_done.json").write_text(
        json.dumps(done_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(str(out_dir))
    return 0


def run_config_delta_audit(args: argparse.Namespace) -> int:
    """R2.0.5 config-delta audit gate; returns 78 when user attention is required."""
    search_root = Path(args.search_root).resolve()
    out_dir = Path(args.out) if args.out else (R2_CONFIG_DELTA_OUT / _utc_slug())
    inventory = discover_recap_ckpts(search_root)
    valid_cells = filter_valid(inventory)
    config_delta_inventory = audit_inventory(
        [ck.abs_path for ck in valid_cells],
        allowed_paths=set(FORMALIZE_LANGUAGE_PATHS),
        dossier_dir=out_dir,
    )
    if config_delta_inventory["summary"][ADDITIONAL_FIELDS_DIFFER]:
        attention_md = out_dir / ATTENTION_FILENAME
        ack_md = out_dir / ACKNOWLEDGMENT_FILENAME
        try:
            require_acknowledgment(attention_md, ack_md)
        except AcknowledgmentMissingError as exc:
            _LOG.error("R2.0.5 user attention required: %s", exc)
            (out_dir / "config_delta_report.md").write_text(
                render_config_delta_subsection(config_delta_inventory),
                encoding="utf-8",
            )
            print(str(out_dir / "r2_0_5_user_attention.md"))
            return 78
        ack_mtime = dt.datetime.fromtimestamp(
            ack_md.stat().st_mtime, dt.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        if config_delta_inventory["attention"] is not None:
            config_delta_inventory["attention"]["status"] = f"acknowledged@{ack_mtime}"
        (out_dir / INVENTORY_FILENAME).write_text(
            json.dumps(config_delta_inventory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (out_dir / "config_delta_report.md").write_text(
        render_config_delta_subsection(config_delta_inventory),
        encoding="utf-8",
    )
    print(str(out_dir))
    return 0


def r2_2_decompose_dry_run(args: argparse.Namespace) -> int:
    """Step 5.5: dry-run swap with TemporaryDirectory + diagnostic-preserve-on-failure."""
    inventory = discover_recap_ckpts(Path(args.search_root).resolve())
    rep = pick_representative(filter_valid(inventory))
    raw_hf = Path(args.raw_hf_snapshot).resolve() if args.raw_hf_snapshot else RAW_HF_SNAPSHOT_ROOT
    preserve_root = ARTIFACT_ROOT / "r2_2_decomposition" / "_failed_dry_runs"

    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="r2_2_dry_swap_") as tmp_dir_str:
        try:
            materialise_swap_ckpt(rep, raw_hf, Path(tmp_dir_str))
            elapsed = time.monotonic() - start
            if elapsed > 900.0:
                _LOG.warning(
                    "Dry-run swap took %.1fs — NTFS3 is 3-10x slower than ext4 "
                    "due to FUSE-driver overhead; expected, not a hang.", elapsed,
                )
            print(f"OK dry-run swap elapsed={elapsed:.1f}s")
            return 0
        except (
            CkptByteIdentityViolation,
            CkptSrcMutatedDuringSwap,
            CkptSourceMissingArtifact,
            CkptRawHfMissingArtifact,
        ) as exc:
            slug = f"{_utc_slug()}_pid{os.getpid()}"
            preserve_path = preserve_root / slug
            preserve_path.mkdir(parents=True, exist_ok=True)
            ctx = _capture_failure_context(
                source_ckpt_root=rep.abs_path,
                raw_hf_snapshot_root=raw_hf,
                swap_root=Path(tmp_dir_str),
                preserve_path=preserve_path,
                audit_failure_class=type(exc).__name__,
            )
            (preserve_path / "_failure_context.json").write_text(
                json.dumps(ctx, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (preserve_path / "audit_failure.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
            print(f"FAIL preserved={preserve_path}")
            raise


def run_evaluate_all(args: argparse.Namespace) -> int:
    """R2.1 GPU sweep across every valid cell (refuses on missing/stale baseline marker)."""
    validate_baseline_pass_marker(P0B_PROTOCOL)
    search_root = Path(args.search_root).resolve()
    out_dir = Path(args.out) if args.out else (ARTIFACT_ROOT / "r2" / _utc_slug())
    run_dir = out_dir / "r2_1_authentic_eval" / _utc_slug()
    run_dir.mkdir(parents=True, exist_ok=True)

    inventory = discover_recap_ckpts(search_root)
    cells: list[R2CellResult] = []
    for ck in filter_valid(inventory):
        cell_out = run_dir / f"{ck.training_run_dir.name}__{ck.abs_path.name}"
        cell_out.mkdir(parents=True, exist_ok=True)
        req = AuthenticEvalRequest(
            checkpoint=ck, search_root=search_root, strict_config=args.strict_config
        )
        cell = run_r2_1_cell(req, cell_out)
        cells.append(cell)
        (cell_out / "cell_report.md").write_text(cell_report.render(cell), encoding="utf-8")
    print(str(run_dir))
    return 0


def run_config_swap(args: argparse.Namespace) -> int:
    """Materialise R2.2 config-swap on the representative ckpt."""
    inventory = discover_recap_ckpts(Path(args.search_root).resolve())
    rep = pick_representative(filter_valid(inventory))
    swap_root = Path(args.swap_root) if args.swap_root else SWAP_ROOT_DEFAULT
    raw_hf = Path(args.raw_hf_snapshot).resolve() if args.raw_hf_snapshot else RAW_HF_SNAPSHOT_ROOT
    res = materialise_swap_ckpt(rep, raw_hf, swap_root)
    print(str(res.swap_dir))
    return 0


def run_r2_summarise(args: argparse.Namespace) -> int:
    """Render summary_table.json from an existing run-dir (per IND-V5)."""
    run_dir = Path(args.run_dir).resolve()
    cell_jsons = sorted(run_dir.rglob("cell_result.json"))
    if not cell_jsons:
        _LOG.error("no cell_result.json under %s", run_dir)
        return 2

    raw_cells: list[dict[str, Any]] = [
        json.loads(p.read_text(encoding="utf-8")) for p in cell_jsons
    ]
    baseline_marker = validate_baseline_pass_marker(P0B_PROTOCOL)
    stat = build_statistical_regime(baseline_marker=baseline_marker, n_valid_cells=len(raw_cells))
    summary_table = {
        "r2_summary_table_schema_version": "1.0.0",
        **stat,
        "trigger_threshold": float(R2_TRIGGER_THRESHOLD),
        "raw_cells": raw_cells,
    }
    (run_dir / "summary_table.json").write_text(
        json.dumps(summary_table, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(str(run_dir / "summary_table.json"))
    return 0


def run_pipeline(args: argparse.Namespace) -> int:
    """Full sequential pipeline: inventory + evaluate-all."""
    rc = run_inventory(args)
    if rc != 0:
        return rc
    rc = run_config_delta_audit(args)
    if rc != 0:
        return rc
    return run_evaluate_all(args)
