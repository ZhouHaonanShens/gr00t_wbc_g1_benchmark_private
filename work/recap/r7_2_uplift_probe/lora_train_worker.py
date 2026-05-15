from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from work.recap.r7_2_uplift_probe.contract import R7AdapterTooLargeError, R7TrainingFailedError

ADAPTER_SIZE_LIMIT_MB = 200.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m work.recap.r7_2_uplift_probe.lora_train_worker")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dry-step-count", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if int(args.dry_step_count) > 0:
        return _run_dry_steps(int(args.dry_step_count), output_root)
    return _run_real_worker(Path(args.request_json), output_root)


def build_lora_config(rank: int, alpha: int, target_modules: list[str]) -> Any:
    from peft import LoraConfig

    if int(rank) <= 0 or int(rank) > 16:
        raise R7TrainingFailedError("lora rank must be in [1, 16]")
    if not target_modules:
        raise R7TrainingFailedError("lora target module list is empty")
    return LoraConfig(r=int(rank), lora_alpha=int(alpha), target_modules=target_modules)


def enumerate_lora_targets(model: Any, top_k_layers: int) -> list[str]:
    layer_entries: list[tuple[int, str]] = []
    action_entries: list[str] = []
    for name, _module in model.named_modules():
        layer_index = _extract_layer_index(name)
        if layer_index is not None and _is_attention_projection(name):
            layer_entries.append((layer_index, name))
        if name.startswith("action_head") and (name.endswith("proj") or "projector" in name):
            action_entries.append(name)
    if not layer_entries:
        raise R7TrainingFailedError("no language tower q/k/v/o projection modules found")
    keep_layers = sorted({layer for layer, _ in layer_entries})[-int(top_k_layers):]
    targets = [name for layer, name in layer_entries if layer in keep_layers]
    targets.extend(action_entries)
    if not targets:
        raise R7TrainingFailedError("no LoRA targets selected")
    return sorted(dict.fromkeys(targets))


def save_adapter_at_step(model: Any, step: int, output_root: Path) -> Path:
    adapter_dir = output_root / f"adapter_step_{int(step):04d}"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    size_mb = adapter_size_mb(adapter_dir)
    if size_mb > ADAPTER_SIZE_LIMIT_MB:
        raise R7AdapterTooLargeError(f"adapter {adapter_dir} is {size_mb:.3f} MB")
    return adapter_dir


def adapter_size_mb(path: Path) -> float:
    total_bytes = 0
    for child in path.rglob("*"):
        if child.is_file():
            total_bytes += child.stat().st_size
    return total_bytes / (1024.0 * 1024.0)


def loss_is_nan(value: float) -> bool:
    numeric_value = float(value)
    if math.isfinite(numeric_value):
        return False
    return True


def apply_lora_adapter_to_policy(base_policy: Any, adapter_dir: str | None) -> Any:
    if not adapter_dir:
        return base_policy
    adapter_path = Path(adapter_dir)
    if not adapter_path.is_dir():
        raise R7TrainingFailedError(f"missing lora adapter dir: {adapter_dir}")
    from peft import PeftModel

    base_policy.model = PeftModel.from_pretrained(base_policy.model, str(adapter_path))
    return base_policy


def _run_real_worker(request_json: Path, output_root: Path) -> int:
    request = json.loads(request_json.read_text(encoding="utf-8"))
    _write_worker_preflight(request, output_root)
    _emit({"event": "done", "reason": "entrypoint_unresolved"})
    return 4


def _write_worker_preflight(request: dict[str, Any], output_root: Path) -> None:
    payload = {
        "schema": "r7_2_worker_preflight_v1",
        "base_ckpt_abs_path": request.get("base_ckpt_abs_path"),
        "recipe_preset": request.get("recipe_preset"),
        "failure_reason": "entrypoint_unresolved",
    }
    (output_root / "worker_preflight.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_dry_steps(count: int, output_root: Path) -> int:
    for step in range(1, int(count) + 1):
        _emit({"event": "step", "step": step, "loss": 1.0 / step})
    _emit({"event": "done", "reason": "dry_step_count"})
    (output_root / "dry_worker_done.json").write_text(json.dumps({"dry_steps": int(count)}) + "\n")
    return 0


def _emit(payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    if not serialized:
        raise R7TrainingFailedError("empty worker event")
    print(serialized, flush=True)


def _extract_layer_index(name: str) -> int | None:
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "layers" and parts[index + 1].isdigit():
            return int(parts[index + 1])
    return None


def _is_attention_projection(name: str) -> bool:
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")
    has_projection_suffix = any(name.endswith(suffix) for suffix in suffixes)
    has_language_prefix = "language" in name or ".model.layers." in name
    if has_projection_suffix and has_language_prefix:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
