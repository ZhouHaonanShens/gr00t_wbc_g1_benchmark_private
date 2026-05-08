#!/usr/bin/env python3
"""Execute GR00T A-gate identity root-cause diagnostics.

Read-only for original checkpoints; all edits are scratch bundles under output root.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ISAAC_GR00T_ROOT = REPO_ROOT / "submodules" / "Isaac-GR00T"
WBC_ROOT = ISAAC_GR00T_ROOT / "external_dependencies" / "GR00T-WholeBodyControl"
for p in (REPO_ROOT, ISAAC_GR00T_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from work.recap.scripts.build_gr00t_canonical_identity import (  # noqa: E402
    BEHAVIOR_SURFACE_DIRS,
    BEHAVIOR_SURFACE_FILES,
    WEIGHT_INDEX,
    build_canonical_identity,
    run_a1_diff,
)
from work.recap.scripts.gr00t_safe_adaptation_action_audit import (  # noqa: E402
    MODALITIES,
    _extract_policy_observation,
    _load_policy,
    _load_stage3_loader,
    _split_normalized_action,
)

MATRIX_IDS = (
    "P0_base_current",
    "P0_identity_current",
    "P1_identity_weights_base_surface",
    "P2_base_weights_identity_surface",
    "P3_identity_formalize_false",
    "P4_base_formalize_true",
)
FIXED_DECISIONS = {
    "A_PASS_FULL",
    "A_PASS_LOCAL_ONLY",
    "A_FAIL_FORMALIZE_CAUSAL",
    "A_FAIL_SURFACE_CAUSAL_BUT_NOT_FORMALIZE_ONLY",
    "A_FAIL_WEIGHTS_OR_LOADER",
    "A_FAIL_UNKNOWN",
}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def resolve(p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:
        return f"UNKNOWN:{e}"


def git_status_short(path: Path) -> list[str]:
    try:
        out = subprocess.check_output(["git", "-C", str(path), "status", "--short"], text=True)
        return [line for line in out.splitlines() if line.strip()]
    except Exception as e:
        return [f"UNKNOWN:{e}"]


def git_diff_name_status(path: Path) -> list[str]:
    try:
        out = subprocess.check_output(["git", "-C", str(path), "diff", "--name-status"], text=True)
        return [line for line in out.splitlines() if line.strip()]
    except Exception as e:
        return [f"UNKNOWN:{e}"]


def load_paths(manifest: Path) -> tuple[Path, Path]:
    data = read_json(manifest)
    return resolve(data["checkpoints"]["base"]), resolve(data["checkpoints"]["identity"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=True) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def copy_surface(src: Path, dst: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for name in BEHAVIOR_SURFACE_FILES:
        sp = src / name
        if sp.exists():
            dp = dst / name
            dp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, dp)
            copied.append({"source": rel(sp), "target": rel(dp), "sha256": sha256(sp)})
    for dirname in BEHAVIOR_SURFACE_DIRS:
        sd = src / dirname
        if sd.exists():
            for sp in sorted(p for p in sd.rglob("*") if p.is_file()):
                dp = dst / dirname / sp.relative_to(sd)
                dp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sp, dp)
                copied.append({"source": rel(sp), "target": rel(dp), "sha256": sha256(sp)})
    return copied


def link_weights(src: Path, dst: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    weights = sorted(src.glob("model-*.safetensors")) + sorted(src.glob("model.safetensors"))
    if not weights:
        raise FileNotFoundError(f"no weights in {src}")
    for sp in weights + [src / WEIGHT_INDEX]:
        if not sp.exists():
            raise FileNotFoundError(sp)
        dp = dst / sp.name
        if dp.exists() or dp.is_symlink():
            dp.unlink()
        os.symlink(sp.resolve(), dp)
        copied.append({"source": rel(sp), "target": rel(dp), "mode": "symlink", "sha256": sha256(sp)})
    return copied


def edit_formalize(bundle: Path, value: bool) -> list[dict[str, Any]]:
    changes = []
    cfg = bundle / "config.json"
    data = read_json(cfg)
    before = data.get("formalize_language")
    data["formalize_language"] = value
    write_json(cfg, data)
    changes.append({"file": "config.json", "path": "$.formalize_language", "before": before, "after": value})
    pcfg = bundle / "processor_config.json"
    pdata = read_json(pcfg)
    before = pdata.setdefault("processor_kwargs", {}).get("formalize_language")
    pdata["processor_kwargs"]["formalize_language"] = value
    write_json(pcfg, pdata)
    changes.append({"file": "processor_config.json", "path": "$.processor_kwargs.formalize_language", "before": before, "after": value})
    return changes


def build_matrix(base: Path, identity: Path, scratch: Path) -> dict[str, Any]:
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    specs = {
        "P0_base_current": (base, base, None),
        "P0_identity_current": (identity, identity, None),
        "P1_identity_weights_base_surface": (identity, base, None),
        "P2_base_weights_identity_surface": (base, identity, None),
        "P3_identity_formalize_false": (identity, identity, False),
        "P4_base_formalize_true": (base, base, True),
    }
    bundles: dict[str, Any] = {}
    for bid, (weights_src, surface_src, formalize) in specs.items():
        out = scratch / bid
        out.mkdir()
        copied = copy_surface(surface_src, out)
        copied += link_weights(weights_src, out)
        edits = edit_formalize(out, formalize) if formalize is not None else []
        a1_vs_base = run_a1_diff(base, out)
        man = {
            "id": bid,
            "path": rel(out),
            "weights_source": rel(weights_src),
            "surface_source": rel(surface_src),
            "specific_edit": edits or "none",
            "copied_files": copied,
            "A1_vs_base_surface": a1_vs_base,
        }
        write_json(out / "scratch_bundle_manifest.json", man)
        bundles[bid] = man
    write_json(scratch / "scratch_bundle_matrix.json", {"schema_version": "a_gate_scratch_matrix_v1", "bundles": bundles})
    return bundles


def collect_observation_bank(dataset: Path, out_dir: Path, *, episode_count: int = 10, chunk_stride: int = 30, max_stage3: int = 20) -> tuple[Any, list[dict[str, Any]]]:
    loader = _load_stage3_loader(dataset)
    obs: list[dict[str, Any]] = []
    unavailable = [
        {"source": "formal_eval_initial", "status": "NOT_AVAILABLE", "searched_paths": ["previous formal eval summaries do not persist raw observations"]},
        {"source": "base_success_grasp_lift", "status": "NOT_AVAILABLE", "searched_paths": ["previous temporal audit stores summaries, not raw observation tensors"]},
    ]
    for ep in range(min(episode_count, len(loader))):
        traj = loader[ep]
        for step in range(0, len(traj), chunk_stride):
            obs.append({"obs_id": len(obs), "source": "stage3_dataset", "episode_index": ep, "step_index": step, "trajectory_len": len(traj)})
            if len(obs) >= max_stage3:
                break
        if len(obs) >= max_stage3:
            break
    status = "PASS" if len(obs) == 50 and not unavailable else "DEGRADED"
    man = {
        "schema_version": "a_gate_observation_bank_v1",
        "status": status,
        "target_count": 50,
        "available_count": len(obs),
        "target_sources": {"formal_eval_initial": 20, "stage3_dataset": 20, "base_success_grasp_lift": 10},
        "unavailable_sources": unavailable,
        "observations": obs,
        "dataset": rel(dataset),
    }
    write_json(out_dir / "observation_bank_manifest.json", man)
    return loader, obs


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.astype(np.float64).reshape(-1)
    bv = b.astype(np.float64).reshape(-1)
    den = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if den == 0:
        return 1.0 if np.allclose(av, bv) else 0.0
    return float(np.dot(av, bv) / den)


def make_policy_inputs(policy: Any, loader: Any, spec: dict[str, Any], prompt: str):
    import torch
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import _rec_to_dtype

    traj = loader[int(spec["episode_index"])]
    obs = _extract_policy_observation(loader=loader, trajectory=traj, step_index=int(spec["step_index"]), plain_prompt=prompt)
    unbatched = policy._unbatch_observation(obs)
    processed = []
    states = []
    for item in unbatched:
        vla = policy._to_vla_step_data(item)
        states.append(vla.states)
        processed.append(policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": vla}]))
    collated = policy.collate_fn(processed)
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
    batched_states = {k: np.stack([s[k] for s in states], axis=0) for k in policy.modality_configs["state"].modality_keys}
    return obs, collated, batched_states


def tensor_hash(t: Any) -> str | None:
    try:
        arr = t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        return hash_array(arr)
    except Exception:
        return None


def first_tensor_by_key(mapping: Any, needle: str) -> Any | None:
    needle = needle.lower()
    if hasattr(mapping, "items"):
        for k, v in mapping.items():
            if needle in str(k).lower():
                return v
            found = first_tensor_by_key(v, needle)
            if found is not None:
                return found
    if isinstance(mapping, (list, tuple)):
        for item in mapping:
            found = first_tensor_by_key(item, needle)
            if found is not None:
                return found
    return None


def predict_bundle(bundle_id: str, bundle_path: Path, loader: Any, obs_specs: list[dict[str, Any]], prompt: str, *, seed: int, audit_count: int) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], list[dict[str, Any]], dict[str, Any]]:
    import torch
    from gr00t.policy.gr00t_policy import _rec_to_dtype

    policy = _load_policy(bundle_path)
    policy.model.eval()
    denorm: dict[str, list[np.ndarray]] = {m: [] for m in MODALITIES}
    norm: dict[str, list[np.ndarray]] = {m: [] for m in MODALITIES}
    per_obs: list[dict[str, Any]] = []
    lang_rows: list[dict[str, Any]] = []
    try:
        for spec in obs_specs:
            set_seeds(seed + int(spec["obs_id"]))
            _, collated, batched_states = make_policy_inputs(policy, loader, spec, prompt)
            token_inputs = None
            if hasattr(collated, "get") and collated.get("vlm_content", None) is not None:
                vlm_content = collated["vlm_content"]
                vlm_list = vlm_content if isinstance(vlm_content, list) else [vlm_content]
                token_inputs = policy.model.collator([{"vlm_content": item} for item in vlm_list])["inputs"]
            with torch.inference_mode():
                action_outputs = policy.model.get_action(**collated)
            normalized = action_outputs["action_pred"].float().cpu().numpy()
            split_norm = _split_normalized_action(policy, normalized)
            decoded = policy.processor.decode_action(normalized, policy.embodiment_tag, batched_states)
            for m in MODALITIES:
                norm[m].append(np.asarray(split_norm[m][0], dtype=np.float32))
                denorm[m].append(np.asarray(decoded[m][0], dtype=np.float32))
            per_obs.append({"bundle": bundle_id, "obs_id": int(spec["obs_id"]), "policy_horizon": int(normalized.shape[1])})
            if len(lang_rows) < audit_count:
                ids = first_tensor_by_key(token_inputs or collated, "input_ids")
                am = first_tensor_by_key(token_inputs or collated, "attention_mask")
                emb = action_outputs["backbone_features"].detach().float().cpu().numpy()
                formalize = bool(getattr(policy.processor, "formalize_language", False))
                processed_language = re.sub(r"[^\w\s]", "", prompt.lower()) if formalize else prompt
                row = {
                    "bundle": bundle_id,
                    "obs_id": int(spec["obs_id"]),
                    "raw_language": prompt,
                    "processed_language": processed_language,
                    "formalize_language": formalize,
                    "token_count": int(ids.numel()) if ids is not None and hasattr(ids, "numel") else None,
                    "token_ids_hash": tensor_hash(ids),
                    "attention_mask_hash": tensor_hash(am),
                    "text_embedding_hash": hash_array(emb),
                    "text_embedding_shape": list(emb.shape),
                    "text_embedding_norm": float(np.linalg.norm(emb.astype(np.float64))),
                    "text_embedding_array": emb,  # removed before JSONL write
                }
                lang_rows.append(row)
    finally:
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    lang_payload = {"rows": lang_rows}
    return denorm, norm, per_obs, lang_payload


def layer_rows(base_vals: dict[str, list[np.ndarray]], cand_vals: dict[str, list[np.ndarray]], pair: str, layer: str, tol: float) -> list[dict[str, Any]]:
    rows = []
    for m in MODALITIES:
        a = np.stack(base_vals[m], axis=0)
        b = np.stack(cand_vals[m], axis=0)
        d = np.abs(a - b)
        rows.append({
            "pair": pair,
            "layer": layer,
            "modality": m,
            "max_abs_diff": float(np.max(d)),
            "mean_abs_diff": float(np.mean(d)),
            "q99_abs_diff": float(np.quantile(d.reshape(-1), 0.99)),
            "cosine": cosine(a, b),
            "pass?": bool(np.max(d) <= tol),
        })
    return rows


def baseline_phase(base: Path, identity: Path, out: Path) -> None:
    rows = []
    files = list(BEHAVIOR_SURFACE_FILES) + [WEIGHT_INDEX]
    for name in files:
        bf = base / name
        inf = identity / name
        rows.append({
            "file": name,
            "base_hash": sha256(bf) if bf.exists() else "NOT_FOUND",
            "identity_hash": sha256(inf) if inf.exists() else "NOT_FOUND",
            "equal?": bool(bf.exists() and inf.exists() and sha256(bf) == sha256(inf)),
        })
    write_csv(out / "bundle_hash_table.csv", rows)
    env = {
        "official_base_path": rel(base),
        "current_identity_path": rel(identity),
        "git_commit": git_commit(REPO_ROOT),
        "Isaac_GR00T_commit": git_commit(ISAAC_GR00T_ROOT),
        "WBC_commit": git_commit(WBC_ROOT),
        "python": sys.executable,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "submodule_readonly_contract": "This diagnostic harness records submodule dirtiness but does not modify submodules.",
        "git_status_short": git_status_short(REPO_ROOT),
        "Isaac_GR00T_status_short": git_status_short(ISAAC_GR00T_ROOT),
        "Isaac_GR00T_diff_name_status": git_diff_name_status(ISAAC_GR00T_ROOT),
        "WBC_status_short": git_status_short(WBC_ROOT),
    }
    write_json(out / "baseline_manifest.json", env)


def dangerous_diff_phase(base: Path, identity: Path, out: Path) -> None:
    a1 = run_a1_diff(base, identity)
    rows = []
    for row in a1["dangerous_diff_rows"]:
        rows.append({"file": row["file"], "json_path": row.get("json_path", "$"), "base": row.get("base"), "identity": row.get("candidate"), "dangerous?": True, "reason": row.get("reason")})
    # Ensure known required fields are represented even if equal via a machine checklist.
    checklist = [
        "formalize_language", "processor_class", "model_type", "embodiment_tag", "action_horizon", "n_action_steps", "denoising_steps", "max_state_dim", "max_action_dim", "action/state dim", "action representation", "relative/absolute transform", "statistics path/source", "modality order", "camera preprocessing", "language template", "tokenizer special tokens", "dtype / torch_dtype"
    ]
    write_csv(out / "dangerous_diff_table.csv", rows)
    write_json(out / "dangerous_diff_machine.json", {"A1_current_identity": a1, "required_field_checklist": checklist})
    lines = ["# Dangerous Diff Report", "", f"A1 current identity status: `{a1['status']}`", "", "| file | json_path | base | identity | dangerous? | reason |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['file']}` | `{r['json_path']}` | `{r['base']}` | `{r['identity']}` | `{r['dangerous?']}` | `{r['reason']}` |")
    (out / "dangerous_diff_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "loader_resolution_trace.json", {"status": "STATIC_ONLY", "note": "Native Gr00tPolicy load paths are represented by scratch bundle manifests and per-bundle policy load logs."})


def run_all(args: argparse.Namespace) -> int:
    manifest = resolve(args.manifest)
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base, identity = load_paths(manifest)
    prompt = args.prompt
    seed = int(args.seed)
    tol = float(args.tolerance)
    baseline_phase(base, identity, out / "phase0_baseline")
    dangerous_diff_phase(base, identity, out / "phase1_dangerous_diff")
    scratch = out / "scratch_bundles"
    bundles = build_matrix(base, identity, scratch)
    dataset = resolve(args.dataset)
    loader, obs_specs = collect_observation_bank(dataset, out / "phase4_offline_l1_l3", max_stage3=args.max_stage3_obs)
    bundle_paths = {bid: resolve(man["path"]) for bid, man in bundles.items()}
    denorm_by: dict[str, dict[str, list[np.ndarray]]] = {}
    norm_by: dict[str, dict[str, list[np.ndarray]]] = {}
    lang_rows_all: list[dict[str, Any]] = []
    base_embedding_by_obs: dict[int, np.ndarray] = {}
    per_obs_rows = []
    for bid in MATRIX_IDS:
        print(f"[BUNDLE_START] {bid} {bundle_paths[bid]}", flush=True)
        den, nor, records, lang = predict_bundle(bid, bundle_paths[bid], loader, obs_specs, prompt, seed=seed, audit_count=args.audit_obs)
        denorm_by[bid] = den
        norm_by[bid] = nor
        per_obs_rows.extend(records)
        if bid == "P0_base_current":
            for row in lang["rows"]:
                base_embedding_by_obs[int(row["obs_id"])] = row["text_embedding_array"]
        for row in lang["rows"]:
            emb = row.pop("text_embedding_array")
            base_emb = base_embedding_by_obs.get(int(row["obs_id"]))
            if base_emb is not None and base_emb.shape == emb.shape:
                row["text_embedding_cosine_vs_base"] = cosine(base_emb, emb)
                row["text_embedding_max_abs_vs_base"] = float(np.max(np.abs(base_emb - emb)))
                row["text_embedding_shape_matches_base"] = True
            else:
                row["text_embedding_cosine_vs_base"] = None
                row["text_embedding_max_abs_vs_base"] = None
                row["text_embedding_shape_matches_base"] = bool(base_emb is not None and base_emb.shape == emb.shape)
                row["base_text_embedding_shape"] = list(base_emb.shape) if base_emb is not None else None
            lang_rows_all.append(row)
        print(f"[BUNDLE_DONE] {bid}", flush=True)
    # Write language audit.
    write_json(out / "phase3_language" / "language_processor_vectors.json", {"rows": lang_rows_all})
    write_csv(out / "phase3_language" / "language_embedding_delta_summary.csv", lang_rows_all)
    answers = {
        "formalize_language_changes_processed_language": any(r["processed_language"] != r["raw_language"] for r in lang_rows_all if r["bundle"] in ("P0_identity_current", "P4_base_formalize_true")),
        "formalize_language_changes_token_ids": None,
        "P3_restores_base_token_ids": None,
        "P4_manufactures_identity_like_token_ids": None,
        "next_suspect_if_token_ids_match": "stats/denorm, vision preprocessing, embodiment/action dims, weights/index, or loader path per Phase1 dangerous diff",
    }
    def first_hash(bundle: str, key: str) -> str | None:
        for r in lang_rows_all:
            if r["bundle"] == bundle and r["obs_id"] == 0:
                return r.get(key)
        return None
    base_tok = first_hash("P0_base_current", "token_ids_hash")
    ident_tok = first_hash("P0_identity_current", "token_ids_hash")
    p3_tok = first_hash("P3_identity_formalize_false", "token_ids_hash")
    p4_tok = first_hash("P4_base_formalize_true", "token_ids_hash")
    answers["formalize_language_changes_token_ids"] = base_tok != ident_tok
    answers["P3_restores_base_token_ids"] = p3_tok == base_tok
    answers["P4_manufactures_identity_like_token_ids"] = p4_tok == ident_tok
    (out / "phase3_language" / "language_processor_audit.md").write_text("\n".join(["# Language Processor Audit", "", json.dumps(answers, indent=2, ensure_ascii=False)]) + "\n", encoding="utf-8")
    # Action equivalence.
    rows: list[dict[str, Any]] = []
    base_id = "P0_base_current"
    for cand in ["P0_identity_current", "P1_identity_weights_base_surface", "P2_base_weights_identity_surface", "P3_identity_formalize_false", "P4_base_formalize_true"]:
        rows.extend(layer_rows(norm_by[base_id], norm_by[cand], f"base vs {cand}", "L1_pred_normalized", tol))
        rows.extend(layer_rows(denorm_by[base_id], denorm_by[cand], f"base vs {cand}", "L2_denorm_absolute", tol))
        # decode_action includes relative->absolute under this seam.
        rows.extend(layer_rows(denorm_by[base_id], denorm_by[cand], f"base vs {cand}", "L3_postprocessed_relative_to_absolute_alias", tol))
    write_csv(out / "phase4_offline_l1_l3" / "offline_equivalence_summary.csv", rows)
    with (out / "phase4_offline_l1_l3" / "offline_action_equivalence.jsonl").open("w", encoding="utf-8") as f:
        for r in per_obs_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    # Conclusions.
    def pair_pass(cand: str) -> bool:
        return all(r["pass?"] for r in rows if r["pair"] == f"base vs {cand}")
    p1_pass = pair_pass("P1_identity_weights_base_surface")
    p3_pass = pair_pass("P3_identity_formalize_false")
    p4_like_identity = None
    # P4 reproduces identity if P4 vs P0_identity passes.
    p4_identity_rows = []
    p4_identity_rows.extend(layer_rows(norm_by["P0_identity_current"], norm_by["P4_base_formalize_true"], "identity vs P4_base_formalize_true", "L1_pred_normalized", tol))
    p4_identity_rows.extend(layer_rows(denorm_by["P0_identity_current"], denorm_by["P4_base_formalize_true"], "identity vs P4_base_formalize_true", "L2_denorm_absolute", tol))
    p4_like_identity = all(r["pass?"] for r in p4_identity_rows)
    write_csv(out / "phase4_offline_l1_l3" / "p4_vs_identity_summary.csv", p4_identity_rows)
    first_fail = next((r for r in rows if not r["pass?"]), None)
    (out / "phase4_offline_l1_l3" / "layer_first_mismatch.md").write_text(
        "# Layer First Mismatch\n\n" + (json.dumps(first_fail, indent=2, ensure_ascii=False) if first_fail else "All required base comparisons pass.\n"), encoding="utf-8")
    # Build canonical identity and patch A2 summary.
    canonical_out = out / "canonical_identity"
    if canonical_out.exists():
        shutil.rmtree(canonical_out)
    can_manifest = build_canonical_identity(base, identity, canonical_out, link_weights=True)
    # Predict canonical for A2.
    den, nor, records, _lang = predict_bundle("canonical_identity", canonical_out, loader, obs_specs, prompt, seed=seed, audit_count=0)
    can_rows = []
    can_rows.extend(layer_rows(norm_by[base_id], nor, "base vs canonical_identity", "L1_pred_normalized", tol))
    can_rows.extend(layer_rows(denorm_by[base_id], den, "base vs canonical_identity", "L2_denorm_absolute", tol))
    can_rows.extend(layer_rows(denorm_by[base_id], den, "base vs canonical_identity", "L3_postprocessed_relative_to_absolute_alias", tol))
    can_pass = all(r["pass?"] for r in can_rows)
    write_csv(canonical_out / "offline_equivalence_summary.csv", can_rows)
    can_manifest["A2_offline_l1_l3"] = {"status": "PASS" if can_pass else "FAIL", "summary_csv": rel(canonical_out / "offline_equivalence_summary.csv"), "observation_bank_status": read_json(out / "phase4_offline_l1_l3" / "observation_bank_manifest.json")["status"]}
    can_manifest["canonical"] = can_manifest["A1_dangerous_diff"]["status"] == "PASS" and can_pass
    write_json(canonical_out / "canonical_identity_manifest.json", can_manifest)
    # L4/L5 gated output: unavailable unless explicitly requested/implemented.
    l45_dir = out / "phase5_l4_l5"
    l45_dir.mkdir(parents=True, exist_ok=True)
    write_json(l45_dir / "l4_l5_status.json", {"status": "UNAVAILABLE", "reason": "No non-invasive server-wrapper-controller dump seam executed in this run; local A2 evidence collected only.", "precondition_l1_l3_pass": can_pass})
    (l45_dir / "l4_l5_unavailable_reason.md").write_text("# L4/L5 unavailable\n\nNo read-only server-wrapper-controller dump was executed in this run; final positive decision can only be LOCAL_ONLY if A1/A2 pass.\n", encoding="utf-8")
    write_csv(l45_dir / "l4_l5_equivalence_summary.csv", [{"pair": "base vs fixed_identity", "layer": "L4_server_wrapper_controller", "modality": "navigate_command", "max_abs_diff": "", "mean_abs_diff": "", "pass?": "UNAVAILABLE"}, {"pair": "base vs fixed_identity", "layer": "L4_server_wrapper_controller", "modality": "right_hand", "max_abs_diff": "", "mean_abs_diff": "", "pass?": "UNAVAILABLE"}, {"pair": "base vs fixed_identity", "layer": "L5_env_applied_proxy", "modality": "navigate_command", "max_abs_diff": "", "mean_abs_diff": "", "pass?": "UNAVAILABLE"}, {"pair": "base vs fixed_identity", "layer": "L5_env_applied_proxy", "modality": "right_hand", "max_abs_diff": "", "mean_abs_diff": "", "pass?": "UNAVAILABLE"}])
    # Final decision.
    if p1_pass and p3_pass and p4_like_identity:
        if can_manifest["canonical"]:
            final = "A_PASS_LOCAL_ONLY"  # L4/L5 unavailable by design in this run.
        else:
            final = "A_FAIL_FORMALIZE_CAUSAL"
    elif not p1_pass:
        final = "A_FAIL_WEIGHTS_OR_LOADER"
    elif p1_pass and not p3_pass:
        final = "A_FAIL_SURFACE_CAUSAL_BUT_NOT_FORMALIZE_ONLY"
    else:
        final = "A_FAIL_UNKNOWN"
    final_json = {
        "final_decision": final,
        "p1_identity_weights_base_surface_pass": p1_pass,
        "p3_identity_formalize_false_restores_base": p3_pass,
        "p4_base_formalize_true_reproduces_identity": p4_like_identity,
        "canonical_identity": can_manifest,
        "l4_l5_status": "UNAVAILABLE",
        "observation_bank_status": can_manifest["A2_offline_l1_l3"]["observation_bank_status"],
    }
    write_json(out / "a_gate_final_decision.json", final_json)
    write_json(out / "a_gate_evidence_index.json", {"artifacts": [rel(p) for p in sorted(out.rglob("*")) if p.is_file()]})
    report = [
        "# A-gate Final Decision", "", f"Final decision: `{final}`", "",
        f"- P1 identity weights + base surface pass: `{p1_pass}`",
        f"- P3 identity formalize false restores base: `{p3_pass}`",
        f"- P4 base formalize true reproduces identity: `{p4_like_identity}`",
        f"- canonical builder A1: `{can_manifest['A1_dangerous_diff']['status']}`",
        f"- canonical builder A2: `{can_manifest['A2_offline_l1_l3']['status']}`",
        f"- L4/L5: `UNAVAILABLE`",
        f"- observation bank: `{can_manifest['A2_offline_l1_l3']['observation_bank_status']}`",
        "", "No training/LoRA/Safe-SFT/RECAP/FATG was run.", "",
    ]
    (out / "a_gate_final_decision.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "PASS", "final_decision": final, "output_dir": rel(out)}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260506)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--max-stage3-obs", type=int, default=20)
    ap.add_argument("--audit-obs", type=int, default=5)
    ap.add_argument("--prompt", default="pick up the apple, walk left and place the apple on the plate.")
    return run_all(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
