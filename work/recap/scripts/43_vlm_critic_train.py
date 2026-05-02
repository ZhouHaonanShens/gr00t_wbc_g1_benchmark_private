#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


DEFAULT_TRAIN_MANIFEST = "agent/artifacts/vlm_critic_manifests/task2_checker_fixture_with_video_train_build_rc_check.full_input.json"
DEFAULT_VAL_MANIFEST = "agent/artifacts/vlm_critic_manifests/task2_checker_fixture_with_video_train_build_rc_check.vision_only.json"
DEFAULT_PUBLIC_WARMSTART_MANIFEST = (
    "agent/artifacts/vlm_critic_manifests/public_warmstart.json"
)
DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_BATCH_SIZE = 1
DEFAULT_WARMSTART_EPOCHS = 1
DEFAULT_FORMAL_EPOCHS = 1
DEFAULT_HEAD_LR = 1e-4
DEFAULT_LORA_LR = 5e-5
DEFAULT_SEED = 7
DEFAULT_TOP_N_LORA_BLOCKS = 4
DEFAULT_PROMPT_TEXT_MODE = "manifest"
DEFAULT_USE_T_NORM = True
DEFAULT_MAX_WARMSTART_SAMPLES = 0
PASS_SENTINEL = "TRAIN_OK"
BLOCKED_SENTINEL = "TRAIN_BLOCKED"
FAIL_SENTINEL = "TRAIN_FAIL"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_path(raw_path: str | None, *, default_rel: str) -> Path:
    value = str(raw_path or default_rel)
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_optional_path(raw_path: str | None) -> Path | None:
    value = str(raw_path or "").strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


class _LineLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, level: str, message: str) -> None:
        line = f"[{level}] {message}"
        print(line)
        with self.path.open("a", encoding="utf-8") as f:
            _ = f.write(line + "\n")

    def info(self, message: str) -> None:
        self.write("INFO", message)

    def error(self, message: str) -> None:
        self.write("ERROR", message)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        _ = f.write("\n")
    _ = tmp_path.replace(path)


def _inspect_model_cache(model_id: str) -> dict[str, object]:
    cache_name = "models--" + str(model_id).replace("/", "--")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
    return {
        "model_id": str(model_id),
        "cache_dir": str(cache_dir),
        "cache_dir_exists": cache_dir.exists(),
        "cache_dir_is_dir": cache_dir.is_dir(),
    }


def _probe_hf_preflight(model_id: str) -> dict[str, object]:
    transforms = importlib.import_module("transformers")
    processor_probe: dict[str, object]
    config_probe: dict[str, object]
    auto_processor = getattr(transforms, "AutoProcessor")
    auto_config = getattr(transforms, "AutoConfig")
    try:
        processor = auto_processor.from_pretrained(model_id, trust_remote_code=True)
        processor_probe = {
            "ok": True,
            "processor_class": type(processor).__name__,
        }
    except Exception as exc:
        processor_probe = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        cfg = auto_config.from_pretrained(model_id, trust_remote_code=True)
        config_probe = {
            "ok": True,
            "config_class": type(cfg).__name__,
            "model_type": str(getattr(cfg, "model_type", "unknown")),
        }
    except Exception as exc:
        config_probe = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "processor_probe": processor_probe,
        "config_probe": config_probe,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="43_vlm_critic_train.py",
        description="Task 6 VLM critic trainer for Qwen/Qwen3-VL-2B-Instruct.",
    )
    _ = parser.add_argument(
        "--train-manifest", type=str, default=DEFAULT_TRAIN_MANIFEST
    )
    _ = parser.add_argument("--val-manifest", type=str, default=DEFAULT_VAL_MANIFEST)
    _ = parser.add_argument(
        "--public-warmstart-manifest",
        type=str,
        default=DEFAULT_PUBLIC_WARMSTART_MANIFEST,
    )
    _ = parser.add_argument("--critic-tag", type=str, required=True)
    _ = parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    _ = parser.add_argument("--device", type=str, default="auto")
    _ = parser.add_argument("--batch-size", type=int, default=int(DEFAULT_BATCH_SIZE))
    _ = parser.add_argument(
        "--warmstart-epochs", type=int, default=int(DEFAULT_WARMSTART_EPOCHS)
    )
    _ = parser.add_argument(
        "--formal-epochs", type=int, default=int(DEFAULT_FORMAL_EPOCHS)
    )
    _ = parser.add_argument("--lr-head", type=float, default=float(DEFAULT_HEAD_LR))
    _ = parser.add_argument("--lr-lora", type=float, default=float(DEFAULT_LORA_LR))
    _ = parser.add_argument("--seed", type=int, default=int(DEFAULT_SEED))
    _ = parser.add_argument(
        "--top-n-lora-blocks", type=int, default=int(DEFAULT_TOP_N_LORA_BLOCKS)
    )
    _ = parser.add_argument("--attn-implementation", type=str, default="")
    _ = parser.add_argument(
        "--prompt-text-mode",
        type=str,
        choices=("manifest", "constant_query_only"),
        default=DEFAULT_PROMPT_TEXT_MODE,
    )
    _ = parser.add_argument(
        "--disable-proprio",
        action="store_true",
        help="Disable proprio during warm-start and formal train so train/eval parity can stay zero-proprio.",
    )
    _ = parser.add_argument(
        "--remediation-diagnosis-json",
        type=str,
        default="",
        help="Optional diagnosis JSON recorded into provenance.input_remediation.",
    )
    _ = parser.add_argument("--max-warmstart-samples", type=int, default=0)
    _ = parser.add_argument("--max-train-samples", type=int, default=0)
    _ = parser.add_argument("--max-val-samples", type=int, default=0)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    runtime_dir = (
        REPO_ROOT / "agent" / "runtime_logs" / "vlm_critic_train" / str(args.critic_tag)
    ).resolve()
    critic_dir = (
        REPO_ROOT / "agent" / "artifacts" / "critics" / str(args.critic_tag)
    ).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    critic_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "train.log"
    logger = _LineLogger(log_path)

    logger.info(f"ts_utc={_iso_now()}")
    logger.info(f"repo_root={REPO_ROOT}")
    logger.info(f"critic_tag={args.critic_tag}")

    modeling_mod = importlib.import_module("work.recap.critic_vlm.modeling")
    train_mod = importlib.import_module("work.recap.critic_vlm.train")
    support = getattr(modeling_mod, "inspect_qwen3_vl_environment")()
    cache_probe = _inspect_model_cache(str(args.base_model))
    hf_probe = _probe_hf_preflight(str(args.base_model))
    remediation_diagnosis_json = _resolve_optional_path(args.remediation_diagnosis_json)
    run_payload: dict[str, object] = {
        "ts_utc": _iso_now(),
        "critic_tag": str(args.critic_tag),
        "base_model": str(args.base_model),
        "train_manifest": str(
            _resolve_path(args.train_manifest, default_rel=DEFAULT_TRAIN_MANIFEST)
        ),
        "val_manifest": str(
            _resolve_path(args.val_manifest, default_rel=DEFAULT_VAL_MANIFEST)
        ),
        "public_warmstart_manifest": str(
            _resolve_path(
                args.public_warmstart_manifest,
                default_rel=DEFAULT_PUBLIC_WARMSTART_MANIFEST,
            )
        ),
        "prompt_text_mode": str(args.prompt_text_mode),
        "use_proprio": not bool(args.disable_proprio),
        "use_t_norm": bool(DEFAULT_USE_T_NORM),
        "remediation_diagnosis_json": (
            str(remediation_diagnosis_json)
            if remediation_diagnosis_json is not None
            else None
        ),
        "environment": support.to_json(),
        "cache_probe": cache_probe,
        "hf_probe": hf_probe,
        "runtime_log": str(log_path),
        "critic_dir": str(critic_dir),
    }
    blocker_json_path = critic_dir / "blocker.json"
    run_json_path = runtime_dir / "train_run.json"

    logger.info(
        "environment_support="
        + json.dumps(support.to_json(), sort_keys=True, ensure_ascii=True)
    )
    logger.info(
        "cache_probe=" + json.dumps(cache_probe, sort_keys=True, ensure_ascii=True)
    )
    logger.info("hf_probe=" + json.dumps(hf_probe, sort_keys=True, ensure_ascii=True))

    if support.blocker:
        blocked = dict(run_payload)
        blocked["pass"] = False
        blocked["sentinel"] = BLOCKED_SENTINEL
        blocked["blocker_kind"] = "environment"
        blocked["error"] = str(support.blocker)
        _write_json(blocker_json_path, blocked)
        _write_json(run_json_path, blocked)
        logger.error(str(support.blocker))
        logger.error(f"wrote_blocker_json={blocker_json_path}")
        print(BLOCKED_SENTINEL)
        return 1

    try:
        train_cfg = getattr(train_mod, "TrainConfig")(
            train_manifest=_resolve_path(
                args.train_manifest, default_rel=DEFAULT_TRAIN_MANIFEST
            ),
            val_manifest=_resolve_path(
                args.val_manifest, default_rel=DEFAULT_VAL_MANIFEST
            ),
            public_warmstart_manifest=_resolve_path(
                args.public_warmstart_manifest,
                default_rel=DEFAULT_PUBLIC_WARMSTART_MANIFEST,
            ),
            critic_tag=str(args.critic_tag),
            base_model=str(args.base_model),
            device=str(args.device),
            batch_size=int(args.batch_size),
            warmstart_epochs=int(args.warmstart_epochs),
            formal_epochs=int(args.formal_epochs),
            lr_head=float(args.lr_head),
            lr_lora=float(args.lr_lora),
            seed=int(args.seed),
            top_n_lora_blocks=int(args.top_n_lora_blocks),
            attn_implementation=str(args.attn_implementation or "").strip() or None,
            prompt_text_mode=str(args.prompt_text_mode),
            use_proprio=not bool(args.disable_proprio),
            use_t_norm=bool(DEFAULT_USE_T_NORM),
            remediation_diagnosis_json=remediation_diagnosis_json,
            max_warmstart_samples=int(args.max_warmstart_samples)
            if int(args.max_warmstart_samples) > 0
            else None,
            max_train_samples=int(args.max_train_samples)
            if int(args.max_train_samples) > 0
            else None,
            max_val_samples=int(args.max_val_samples)
            if int(args.max_val_samples) > 0
            else None,
        )
        result = getattr(train_mod, "run_vlm_critic_training")(
            repo_root=REPO_ROOT, config=train_cfg
        )
        success = dict(run_payload)
        success["pass"] = True
        success["sentinel"] = PASS_SENTINEL
        success["metrics"] = result.metrics
        success["provenance"] = result.provenance
        _write_json(blocker_json_path, success)
        _write_json(run_json_path, success)
        logger.info(f"critic_dir={result.critic_dir}")
        logger.info(f"wrote_success_state={blocker_json_path}")
        logger.info("training_finished=true")
        print(PASS_SENTINEL)
        return 0
    except Exception as exc:
        failure = dict(run_payload)
        failure["pass"] = False
        failure["sentinel"] = FAIL_SENTINEL
        failure["blocker_kind"] = "runtime"
        failure["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(blocker_json_path, failure)
        _write_json(run_json_path, failure)
        logger.error(
            failure["error"]
            if isinstance(failure["error"], str)
            else str(failure["error"])
        )
        logger.error(f"wrote_blocker_json={blocker_json_path}")
        print(FAIL_SENTINEL)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
