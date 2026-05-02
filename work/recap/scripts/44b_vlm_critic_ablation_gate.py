#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# =====================
# USER Config (edit)
# =====================

DEFAULT_OUTPUT_JSON_REL = (
    "agent/artifacts/vlm_critic_offline_gate/task7_ablation_gate.json"
)
PASS_SENTINEL = "ABLATION_GATE_OK"
FAIL_SENTINEL = "ABLATION_GATE_FAIL"
REINTEGRATE_ALLOWED = "REINTEGRATE_ALLOWED"
REINTEGRATE_BLOCKED = "REINTEGRATE_BLOCKED"
PROMPT_MARGIN_THRESHOLD = 0.03


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _repo_root()


def _resolve_path(raw_path: str | None, *, default_rel: str) -> Path:
    value = str(raw_path or default_rel).strip()
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        _ = f.write("\n")
    _ = tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}")
    return cast(dict[str, object], obj)


def _load_task44_module() -> Any:
    module_path = REPO_ROOT / "agent" / "run" / "44_vlm_critic_offline_gate.py"
    spec = importlib.util.spec_from_file_location(
        "task44_offline_gate", str(module_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _emit_result(
    *, sentinel: str, output_json: Path, payload: Mapping[str, object]
) -> None:
    _write_json(output_json, payload)
    print(f"[INFO] wrote_json: {output_json}")
    verdict_text = payload.get("reintegrate_status")
    if isinstance(verdict_text, str) and verdict_text.strip():
        print(verdict_text)
    print(f"SENTINEL:{sentinel}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="44b_vlm_critic_ablation_gate.py",
        description="Assess Task 7 ablation gate, especially whether full-input beats prompt-only on the same held-out split.",
    )
    _ = parser.add_argument("--offline-gate-json", type=str, default="")
    _ = parser.add_argument("--critic-dir", type=str, default="")
    _ = parser.add_argument("--baseline-critic-dir", type=str, default="")
    _ = parser.add_argument("--test-manifest", type=str, default="")
    _ = parser.add_argument("--dataset-path", type=str, default="")
    _ = parser.add_argument("--episodes-jsonl", type=str, default="")
    _ = parser.add_argument("--prompt-only-manifest", type=str, default="")
    _ = parser.add_argument("--vision-only-manifest", type=str, default="")
    _ = parser.add_argument("--full-input-manifest", type=str, default="")
    _ = parser.add_argument("--early-max-t", type=int, default=5)
    _ = parser.add_argument("--qwen-batch-size", type=int, default=4)
    _ = parser.add_argument("--qwen-device", type=str, default="auto")
    _ = parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON_REL)
    return parser


def _load_or_run_offline_gate(args: argparse.Namespace) -> dict[str, object]:
    offline_gate_raw = str(getattr(args, "offline_gate_json", "") or "").strip()
    if offline_gate_raw:
        return _read_json(_resolve_path(offline_gate_raw, default_rel=""))
    module = _load_task44_module()
    return cast(
        dict[str, object],
        module.generate_offline_gate_result(
            critic_dir=_resolve_path(args.critic_dir, default_rel=""),
            baseline_critic_dir=_resolve_path(args.baseline_critic_dir, default_rel=""),
            test_manifest_path=_resolve_path(args.test_manifest, default_rel=""),
            dataset_path=_resolve_path(
                args.dataset_path, default_rel=module.DEFAULT_DATASET_PATH
            ),
            episodes_jsonl_path=_resolve_path(
                args.episodes_jsonl, default_rel=module.DEFAULT_EPISODES_JSONL
            ),
            prompt_only_manifest_path=_resolve_path(
                args.prompt_only_manifest, default_rel=""
            )
            if str(args.prompt_only_manifest).strip()
            else None,
            vision_only_manifest_path=_resolve_path(
                args.vision_only_manifest, default_rel=""
            )
            if str(args.vision_only_manifest).strip()
            else None,
            full_input_manifest_path=_resolve_path(
                args.full_input_manifest, default_rel=""
            )
            if str(args.full_input_manifest).strip()
            else None,
            early_max_t=int(args.early_max_t),
            qwen_batch_size=int(args.qwen_batch_size),
            qwen_device=str(args.qwen_device),
        ),
    )


def main() -> int:
    args = build_parser().parse_args()
    output_json = _resolve_path(args.output_json, default_rel=DEFAULT_OUTPUT_JSON_REL)
    try:
        offline = _load_or_run_offline_gate(args)
        ablation_summary = cast(dict[str, object], offline.get("ablation_summary", {}))
        prompt_auc = ablation_summary.get("prompt_only_auc")
        vision_auc = ablation_summary.get("vision_only_auc")
        full_auc = ablation_summary.get("full_input_auc")
        full_minus_prompt = ablation_summary.get("full_minus_prompt_auc")
        full_minus_vision = ablation_summary.get("full_minus_vision_auc")
        prompt_margin_ok = isinstance(full_minus_prompt, (int, float)) and float(
            full_minus_prompt
        ) >= float(PROMPT_MARGIN_THRESHOLD)
        reintegrate_status = (
            REINTEGRATE_ALLOWED if prompt_margin_ok else REINTEGRATE_BLOCKED
        )
        result = {
            "schema_version": "vlm_critic_ablation_gate_v1",
            "task": "task7_vlm_critic_ablation_gate",
            "offline_gate_json": str(args.offline_gate_json or ""),
            "full_input_auc": full_auc,
            "prompt_only_auc": prompt_auc,
            "vision_only_auc": vision_auc,
            "full_minus_prompt_auc": full_minus_prompt,
            "full_minus_vision_auc": full_minus_vision,
            "full_input_beats_prompt_only": bool(prompt_margin_ok),
            "prompt_shortcut_risk": not bool(prompt_margin_ok),
            "threshold_prompt_margin": float(PROMPT_MARGIN_THRESHOLD),
            "reintegrate_status": reintegrate_status,
            "reintegrate_verdict": "ALLOW" if prompt_margin_ok else "BLOCK",
            "source_offline_gate_verdict": offline.get("reintegrate_verdict"),
            "source_offline_gate_status": offline.get("reintegrate_status"),
        }
        _emit_result(sentinel=PASS_SENTINEL, output_json=output_json, payload=result)
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "vlm_critic_ablation_gate_v1",
            "task": "task7_vlm_critic_ablation_gate",
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "reintegrate_status": REINTEGRATE_BLOCKED,
        }
        _emit_result(sentinel=FAIL_SENTINEL, output_json=output_json, payload=failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
