"""R2 Authentic Evaluation CLI (argparse plumbing only).

Subcommands:
  inventory                    R2.0 disk-only dry-run inventory (no GPU).
  config-delta-audit           R2.0.5 config-delta audit gate (no GPU).
  evaluate-all                 R2.1 GPU sweep (refuses on missing baseline marker).
  config-swap                  Materialise R2.2 config-swap on representative ckpt.
  r2-2-decompose-dry-run       Step 5.5 dry-run swap audit (NO GPU).
  r2-summarise                 Render summary from existing run-dir (per IND-V5).
  r2-run                       Full sequential pipeline (inventory + evaluate-all).

Heavy logic is in ``_workflow.py``; this module is the entry-point/dispatcher.
"""
from __future__ import annotations

import argparse
import logging
import sys

from work.recap.r2_authentic_eval import _workflow as wf
from work.recap.r2_authentic_eval.inventory import DEFAULT_SEARCH_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R2 Authentic Evaluation harness (plan r2_authentic_eval_plan_v4)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory", help="R2.0 disk-only inventory (no GPU).")
    inv.add_argument("--search-root", default=str(DEFAULT_SEARCH_ROOT))
    inv.add_argument("--out", default="")
    inv.set_defaults(func=wf.run_inventory)

    cda = sub.add_parser(
        "config-delta-audit",
        help="R2.0.5 config-delta audit gate (no GPU).",
    )
    cda.add_argument("--search-root", default=str(DEFAULT_SEARCH_ROOT))
    cda.add_argument("--out", default="")
    cda.set_defaults(func=wf.run_config_delta_audit)

    eva = sub.add_parser("evaluate-all", help="R2.1 GPU sweep on every valid cell.")
    eva.add_argument("--search-root", default=str(DEFAULT_SEARCH_ROOT))
    eva.add_argument("--out", default="")
    eva.add_argument("--strict-config", action="store_true", default=False)
    eva.add_argument("--skip-existing-cells", action="store_true", default=False)
    eva.set_defaults(func=wf.run_evaluate_all)

    cs = sub.add_parser("config-swap", help="Materialise R2.2 config swap (no GPU).")
    cs.add_argument("--search-root", default=str(DEFAULT_SEARCH_ROOT))
    cs.add_argument("--swap-root", default="")
    cs.add_argument("--raw-hf-snapshot", default="")
    cs.set_defaults(func=wf.run_config_swap)

    dry = sub.add_parser(
        "r2-2-decompose-dry-run",
        help="Step 5.5: dry-run swap audit on real disk (NO GPU).",
    )
    dry.add_argument("--search-root", default=str(DEFAULT_SEARCH_ROOT))
    dry.add_argument("--raw-hf-snapshot", default="")
    dry.set_defaults(func=wf.r2_2_decompose_dry_run)

    summ = sub.add_parser("r2-summarise", help="Render summary from a run-dir.")
    summ.add_argument("--run-dir", required=True)
    summ.set_defaults(func=wf.run_r2_summarise)

    full = sub.add_parser("r2-run", help="Full pipeline: inventory + evaluate-all.")
    full.add_argument("--search-root", default=str(DEFAULT_SEARCH_ROOT))
    full.add_argument("--out", default="")
    full.add_argument("--strict-config", action="store_true", default=False)
    full.add_argument("--skip-existing-cells", action="store_true", default=False)
    full.set_defaults(func=wf.run_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
