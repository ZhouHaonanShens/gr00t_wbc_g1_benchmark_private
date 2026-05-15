#!/usr/bin/env python3
"""Build a scratch canonical GR00T identity bundle.

The builder never mutates input bundles.  It creates an output bundle whose
behavior-relevant surface comes from official base while model weights/index
come from the supplied identity bundle.  It performs an A1 surface diff and,
when invoked as a CLI, runs the same offline L1-L3 equivalence check used by the
A-gate harness unless an external A2 summary is supplied.
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_STAGE3_DATASET = REPO_ROOT / "agent/artifacts/lerobot_datasets/recap_stage3_iter_002"
DEFAULT_TASK_PROMPT = "pick up the apple, walk left and place the apple on the plate."

WEIGHT_PATTERNS = ("model-*.safetensors", "model.safetensors")
WEIGHT_INDEX = "model.safetensors.index.json"
BEHAVIOR_SURFACE_FILES = (
    "config.json",
    "processor_config.json",
    "statistics.json",
    "modality.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "embodiment_id.json",
)
BEHAVIOR_SURFACE_DIRS = ("experiment_cfg",)
DANGEROUS_IGNORE_PATHS = {
    "model.safetensors.index.json",  # sourced from identity weights by design
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def copy_file(src: Path, dst: Path, *, link_weights: bool = False) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    mode = "copy"
    if link_weights:
        os.symlink(src.resolve(), dst)
        mode = "symlink"
    else:
        shutil.copy2(src, dst)
    return {
        "source": rel(src),
        "target": rel(dst),
        "mode": mode,
        "source_sha256": sha256(src),
        "target_sha256": sha256(dst.resolve() if dst.is_symlink() else dst),
        "size_bytes": src.stat().st_size,
    }


def copy_tree_files(src_dir: Path, dst_dir: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    if not src_dir.exists():
        return copied
    for src in sorted(p for p in src_dir.rglob("*") if p.is_file()):
        dst = dst_dir / src.relative_to(src_dir)
        copied.append(copy_file(src, dst))
    return copied


def iter_weight_files(bundle: Path) -> list[Path]:
    out: list[Path] = []
    for pattern in WEIGHT_PATTERNS:
        out.extend(sorted(bundle.glob(pattern)))
    return out


def diff_json(base_file: Path, candidate_file: Path, path: str = "$") -> list[dict[str, Any]]:
    def rec(a: Any, b: Any, p: str) -> list[dict[str, Any]]:
        if type(a) is not type(b):
            return [{"json_path": p, "base": a, "candidate": b, "reason": "type_mismatch"}]
        if isinstance(a, dict):
            rows: list[dict[str, Any]] = []
            for k in sorted(set(a) | set(b)):
                if k not in a:
                    rows.append({"json_path": f"{p}.{k}", "base": "<MISSING>", "candidate": b[k], "reason": "missing_base"})
                elif k not in b:
                    rows.append({"json_path": f"{p}.{k}", "base": a[k], "candidate": "<MISSING>", "reason": "missing_candidate"})
                else:
                    rows.extend(rec(a[k], b[k], f"{p}.{k}"))
            return rows
        if isinstance(a, list):
            rows = []
            if len(a) != len(b):
                rows.append({"json_path": f"{p}.len", "base": len(a), "candidate": len(b), "reason": "list_length"})
            for i, (x, y) in enumerate(zip(a, b)):
                rows.extend(rec(x, y, f"{p}[{i}]"))
            return rows
        if a != b:
            return [{"json_path": p, "base": a, "candidate": b, "reason": "value"}]
        return []
    return rec(read_json(base_file), read_json(candidate_file), path)


def run_a1_diff(base: Path, candidate: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in BEHAVIOR_SURFACE_FILES:
        if name in DANGEROUS_IGNORE_PATHS:
            continue
        bf = base / name
        cf = candidate / name
        if not bf.exists() and not cf.exists():
            continue
        if bf.exists() != cf.exists():
            rows.append({"file": name, "json_path": "$", "base": bf.exists(), "candidate": cf.exists(), "dangerous": True, "reason": "existence_mismatch"})
            continue
        if name.endswith(".json"):
            for d in diff_json(bf, cf):
                rows.append({"file": name, **d, "dangerous": True})
        else:
            equal = filecmp.cmp(bf, cf, shallow=False)
            if not equal:
                rows.append({"file": name, "json_path": "$", "base": sha256(bf), "candidate": sha256(cf), "dangerous": True, "reason": "hash_mismatch"})
    for dirname in BEHAVIOR_SURFACE_DIRS:
        bdir = base / dirname
        cdir = candidate / dirname
        if not bdir.exists() and not cdir.exists():
            continue
        base_files = sorted(str(p.relative_to(bdir)) for p in bdir.rglob("*") if p.is_file()) if bdir.exists() else []
        cand_files = sorted(str(p.relative_to(cdir)) for p in cdir.rglob("*") if p.is_file()) if cdir.exists() else []
        for rel_name in sorted(set(base_files) | set(cand_files)):
            bf = bdir / rel_name
            cf = cdir / rel_name
            if not bf.exists() or not cf.exists() or sha256(bf) != sha256(cf):
                rows.append({"file": f"{dirname}/{rel_name}", "json_path": "$", "base": sha256(bf) if bf.exists() else "<MISSING>", "candidate": sha256(cf) if cf.exists() else "<MISSING>", "dangerous": True, "reason": "surface_tree_mismatch"})
    return {"status": "PASS" if not rows else "FAIL", "dangerous_diff_rows": rows}


def build_canonical_identity(base: Path, identity_weights: Path, out: Path, *, link_weights: bool, external_a2: Path | None = None) -> dict[str, Any]:
    base = base.resolve()
    identity_weights = identity_weights.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    missing_surface: list[str] = []
    for name in BEHAVIOR_SURFACE_FILES:
        src = base / name
        if src.exists():
            copied.append(copy_file(src, out / name))
        else:
            missing_surface.append(name)
    for dirname in BEHAVIOR_SURFACE_DIRS:
        copied.extend(copy_tree_files(base / dirname, out / dirname))
    weight_files = iter_weight_files(identity_weights)
    if not weight_files:
        raise FileNotFoundError(f"no model weights found in {identity_weights}")
    for src in weight_files:
        copied.append(copy_file(src, out / src.name, link_weights=link_weights))
    index_src = identity_weights / WEIGHT_INDEX
    if index_src.exists():
        copied.append(copy_file(index_src, out / WEIGHT_INDEX))
    else:
        raise FileNotFoundError(index_src)
    a1 = run_a1_diff(base, out)
    a2: dict[str, Any]
    if external_a2 and external_a2.exists():
        a2 = json.loads(external_a2.read_text(encoding="utf-8"))
    else:
        a2 = {"status": "UNKNOWN", "reason": "A2 offline L1-L3 not supplied to builder CLI; external harness must run or pass --external-a2-summary"}
    canonical = a1["status"] == "PASS" and a2.get("status") == "PASS"
    manifest = {
        "schema_version": "gr00t_canonical_identity_manifest_v1",
        "canonical": canonical,
        "base": rel(base),
        "identity_weights": rel(identity_weights),
        "out": rel(out),
        "link_weights": link_weights,
        "copied_files": copied,
        "missing_surface_files": missing_surface,
        "A1_dangerous_diff": a1,
        "A2_offline_l1_l3": a2,
    }
    write_json(out / "canonical_identity_manifest.json", manifest)
    lines = ["# Dangerous Diff Report", "", f"A1 status: `{a1['status']}`", ""]
    if a1["dangerous_diff_rows"]:
        lines += ["| file | json_path | base | candidate | reason |", "|---|---|---|---|---|"]
        for row in a1["dangerous_diff_rows"]:
            lines.append(f"| `{row['file']}` | `{row['json_path']}` | `{row['base']}` | `{row['candidate']}` | `{row['reason']}` |")
    else:
        lines.append("No dangerous diff vs base surface.")
    (out / "dangerous_diff_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Placeholder required by spec; full harness overwrites/augments when it runs A2.
    if not (out / "offline_equivalence_summary.csv").exists():
        (out / "offline_equivalence_summary.csv").write_text("status,reason\nUNKNOWN,A2 not supplied to standalone builder CLI\n", encoding="utf-8")
    write_json(out / "builder_command_manifest.json", {"argv_builder": " ".join(os.sys.argv), "manifest": rel(out / "canonical_identity_manifest.json")})
    return manifest


def run_cli_a2(
    *,
    base: Path,
    canonical: Path,
    out: Path,
    dataset: Path,
    prompt: str,
    seed: int,
    tolerance: float,
    max_stage3_obs: int,
) -> dict[str, Any]:
    """Run builder-local A2 without importing heavy GR00T code at module import time."""

    from work.recap.scripts.gr00t_a_gate_identity_closure import (
        collect_observation_bank,
        layer_rows,
        predict_bundle,
        write_csv,
    )

    if not dataset.exists():
        return {
            "status": "FAIL",
            "reason": f"dataset not found for automatic A2: {rel(dataset)}",
            "summary_csv": rel(out / "offline_equivalence_summary.csv"),
        }
    a2_dir = out / "builder_a2"
    a2_dir.mkdir(parents=True, exist_ok=True)
    loader, obs_specs = collect_observation_bank(
        dataset,
        a2_dir,
        episode_count=10,
        chunk_stride=30,
        max_stage3=max_stage3_obs,
    )
    base_denorm, base_norm, _, _ = predict_bundle(
        "base",
        base,
        loader,
        obs_specs,
        prompt,
        seed=seed,
        audit_count=0,
    )
    can_denorm, can_norm, _, _ = predict_bundle(
        "canonical_identity",
        canonical,
        loader,
        obs_specs,
        prompt,
        seed=seed,
        audit_count=0,
    )
    rows: list[dict[str, Any]] = []
    rows.extend(layer_rows(base_norm, can_norm, "base vs canonical_identity", "L1_pred_normalized", tolerance))
    rows.extend(layer_rows(base_denorm, can_denorm, "base vs canonical_identity", "L2_denorm_absolute", tolerance))
    rows.extend(layer_rows(base_denorm, can_denorm, "base vs canonical_identity", "L3_postprocessed_relative_to_absolute_alias", tolerance))
    write_csv(out / "offline_equivalence_summary.csv", rows)
    return {
        "status": "PASS" if all(r.get("pass?") for r in rows) else "FAIL",
        "summary_csv": rel(out / "offline_equivalence_summary.csv"),
        "observation_bank_status": read_json(a2_dir / "observation_bank_manifest.json")["status"],
        "dataset": rel(dataset),
        "seed": seed,
        "tolerance": tolerance,
        "max_stage3_obs": max_stage3_obs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--identity-weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--link-weights", action="store_true", help="Symlink model shards instead of copying; manifest records symlink handling.")
    ap.add_argument("--external-a2-summary")
    ap.add_argument("--dataset", default=str(DEFAULT_STAGE3_DATASET), help="LeRobot Stage3 dataset used for automatic A2 offline L1-L3.")
    ap.add_argument("--prompt", default=DEFAULT_TASK_PROMPT)
    ap.add_argument("--seed", type=int, default=20260506)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--max-stage3-obs", type=int, default=20)
    args = ap.parse_args()
    base = Path(args.base)
    out = Path(args.out)
    manifest = build_canonical_identity(base, Path(args.identity_weights), out, link_weights=args.link_weights, external_a2=Path(args.external_a2_summary) if args.external_a2_summary else None)
    if not args.external_a2_summary:
        a2 = run_cli_a2(
            base=base,
            canonical=out,
            out=out,
            dataset=Path(args.dataset),
            prompt=args.prompt,
            seed=args.seed,
            tolerance=args.tolerance,
            max_stage3_obs=args.max_stage3_obs,
        )
        manifest["A2_offline_l1_l3"] = a2
        manifest["canonical"] = manifest["A1_dangerous_diff"]["status"] == "PASS" and a2.get("status") == "PASS"
        write_json(out / "canonical_identity_manifest.json", manifest)
    print(json.dumps({"status": "PASS", "canonical": manifest["canonical"], "manifest": rel(Path(args.out) / "canonical_identity_manifest.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
