#!/usr/bin/env python3
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import socket
import sys
import uuid
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
_ = os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "agent").is_dir() and (parent / "work").is_dir():
            return parent
    return Path.cwd().resolve()


def _add_import_roots(repo_root: Path) -> None:
    submodule_root = repo_root / "submodules" / "Isaac-GR00T"
    for p in (repo_root, submodule_root):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="3D_recap_run_adv_server.py",
        description=(
            "Run a GR00T policy server using AdvantageAwareGr00tPolicy "
            "(supports options['advantage'] for RECAP inference)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help=(
            "Model path or HF repo id. If --base-model-path is set, this is still used as the default "
            "--adv-embedding-from source when it points to a local checkpoint directory."
        ),
    )
    p.add_argument(
        "--base-model-path",
        type=str,
        default="",
        help=(
            "If set, the server loads the base model weights from this path, and (optionally) "
            "overlays advantage_embedding weights from --adv-embedding-from. This prevents a finetune "
            "checkpoint from unintentionally regressing the base policy during sim eval."
        ),
    )
    p.add_argument(
        "--adv-embedding-from",
        type=str,
        default="",
        help=(
            "Local checkpoint directory containing model.safetensors.index.json, used only as the source "
            "of action_head.advantage_embedding.{weight,bias}. If empty, defaults to --model-path when it is a local dir."
        ),
    )
    p.add_argument(
        "--overlay-from",
        type=str,
        default="",
        help=(
            "Optional local checkpoint directory or .safetensors file used as an additional overlay source. "
            "When set, keys matching --overlay-include-regex (and not matching --overlay-exclude-regex) will be "
            "copied into the in-memory model weights after advantage head swap/patching."
        ),
    )
    p.add_argument(
        "--overlay-include-regex",
        type=str,
        default="",
        help=(
            "Regex selecting which weight keys to overlay from --overlay-from. If empty, no overlay is applied. "
            "Example: '^action_head\\..*'"
        ),
    )
    p.add_argument(
        "--overlay-exclude-regex",
        type=str,
        default=r"$^",
        help="Regex to exclude keys from overlay (applied after include-regex).",
    )
    p.add_argument("--embodiment-tag", type=str, default="UNITREE_G1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument(
        "--stats-from-model-path",
        type=str,
        default="",
        help=(
            "If set, load processor statistics from this model path and override the "
            "checkpoint processor statistics (useful to avoid zero-std stats from "
            "state-only datasets during sim eval)."
        ),
    )
    p.add_argument(
        "--require-advantage-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, require the checkpoint to contain action_head.advantage_embedding weights and "
            "swap in GR00TRecapActionHead. If false, serve the model without advantage conditioning "
            "(still forces eager attention via AdvantageAwareGr00tPolicy)."
        ),
    )
    p.add_argument(
        "--allow-baseline-default-advantage-embedding-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow a default-initialized GR00TRecapActionHead.advantage_embedding only for the explicit "
            "unconditional baseline path when local baseline weights do not ship action_head.advantage_embedding.{weight,bias}."
        ),
    )
    p.add_argument(
        "--advantage-injection-rule",
        type=str,
        choices=["sign_consistent"],
        default="sign_consistent",
        help=(
            "Serving-side numeric advantage injection rule. sign_consistent passes options['advantage'] "
            "through unchanged. Mainline contract forbids legacy_negate."
        ),
    )
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--use-sim-policy-wrapper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap policy with Gr00tSimPolicyWrapper (recommended for sim eval).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = _repo_root()
    _add_import_roots(repo_root)

    from work.recap.transformers_compat import (
        install_transformers_image_processor_fast_compat,
    )

    install_transformers_image_processor_fast_compat()

    host_raw = str(args.host)
    host_for_probe = "127.0.0.1" if host_raw.strip() in {"0.0.0.0", "*"} else host_raw
    port_i = int(args.port)
    try:
        with socket.create_connection((host_for_probe, port_i), timeout=0.2):
            print(
                "[ERROR] port already in use (tcp connect succeeded):",
                f"host={host_for_probe}",
                f"port={port_i}",
            )
            print(
                "[ERROR] refusing to start to avoid evaluating against a different running server"
            )
            return 2
    except OSError:
        pass

    import json

    import torch
    from safetensors.torch import load_file

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    from work.recap.advantage import (
        ADVANTAGE_CONTRACT_VERSION,
        MAINLINE_TASK_TEXT_FIELD,
    )
    from work.recap.model import GR00TRecapActionHead
    from work.recap.policy import AdvantageAwareGr00tPolicy

    emb = EmbodimentTag[str(args.embodiment_tag)]
    model_path_raw = str(args.model_path).strip()
    model_path = str(Path(model_path_raw).expanduser())
    base_model_path_raw = str(getattr(args, "base_model_path", "") or "").strip()
    adv_from_raw = str(getattr(args, "adv_embedding_from", "") or "").strip()
    overlay_from_raw = str(getattr(args, "overlay_from", "") or "").strip()
    overlay_include_raw = str(getattr(args, "overlay_include_regex", "") or "").strip()
    overlay_exclude_raw = (
        str(getattr(args, "overlay_exclude_regex", "") or "").strip() or r"$^"
    )
    policy_model_path_raw = base_model_path_raw or model_path_raw

    server_uuid = uuid.uuid4().hex
    server_start_ts = _dt.datetime.now().isoformat(timespec="seconds")

    print("Starting RECAP advantage-aware GR00T server...")
    print(f"  Embodiment tag: {emb}")
    print(f"  Model path: {model_path_raw}")
    if base_model_path_raw:
        print(f"  Base model path: {base_model_path_raw}")
    if adv_from_raw:
        print(f"  Advantage embedding from: {adv_from_raw}")
    print(f"  Device: {args.device}")
    print(f"  Host: {args.host}")
    print(f"  Port: {int(args.port)}")
    print(f"  Strict: {bool(args.strict)}")
    print(f"  Use sim wrapper: {bool(args.use_sim_policy_wrapper)}")
    print(f"  Advantage injection rule: {str(args.advantage_injection_rule)}")
    provenance = {
        "policy_model_path": policy_model_path_raw,
        "base_model_path": base_model_path_raw or None,
        "adv_embedding_from": adv_from_raw or None,
        "advantage_injection_rule": str(args.advantage_injection_rule),
        "advantage_contract_version": str(ADVANTAGE_CONTRACT_VERSION),
        "advantage_none_semantics": "unconditional_baseline",
        "advantage_zero_semantics": "explicit_numeric_neutral_conditioning",
        "advantage_positive_semantics": "positive_numeric_conditioning",
        "task_text_field": str(MAINLINE_TASK_TEXT_FIELD),
        "legacy_negate_enabled": False,
        "require_advantage_embedding": bool(args.require_advantage_embedding),
        "allow_baseline_default_advantage_embedding_init": bool(
            args.allow_baseline_default_advantage_embedding_init
        ),
        "baseline_default_advantage_embedding_init": False,
        "baseline_default_advantage_embedding_init_reason": None,
        "stats_from_model_path": str(args.stats_from_model_path).strip() or None,
        "overlay_from": overlay_from_raw or None,
        "overlay_include_regex": overlay_include_raw or None,
        "overlay_exclude_regex": overlay_exclude_raw or None,
    }
    server_info = {
        "server_uuid": str(server_uuid),
        "pid": int(os.getpid()),
        "start_ts": str(server_start_ts),
        "provenance": provenance,
    }
    print("  Server UUID:", str(server_uuid))
    print("  Provenance:")
    print(json.dumps(provenance, ensure_ascii=True, sort_keys=True))

    policy = AdvantageAwareGr00tPolicy(
        embodiment_tag=emb,
        model_path=policy_model_path_raw,
        device=str(args.device),
        strict=bool(args.strict),
        advantage_injection_rule=str(args.advantage_injection_rule),
    )

    def _local_processor_config_has_empty_video_keys(raw_path: str) -> bool:
        cfg_path = Path(str(raw_path).strip()).expanduser() / "processor_config.json"
        if not cfg_path.is_file():
            return False
        try:
            cfg_obj = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        kwargs = cfg_obj.get("processor_kwargs")
        if not isinstance(kwargs, dict):
            return False
        modality_cfgs = kwargs.get("modality_configs")
        if not isinstance(modality_cfgs, dict):
            return False
        emb_cfg = modality_cfgs.get(str(emb.value))
        if not isinstance(emb_cfg, dict):
            return False
        video_cfg = emb_cfg.get("video")
        if not isinstance(video_cfg, dict):
            return False
        video_keys = list(video_cfg.get("modality_keys") or [])
        return len(video_keys) == 0

    if str(args.stats_from_model_path).strip():
        stats_src = str(args.stats_from_model_path).strip()
        if bool(args.strict) and _local_processor_config_has_empty_video_keys(
            stats_src
        ):
            raise ValueError(
                "Refusing to override processor statistics from a no-video checkpoint under strict mainline serving: "
                f"{stats_src}. Use official/base model stats or a with-video finetune checkpoint."
            )
        try:
            from transformers.models.auto.processing_auto import AutoProcessor

            _ = __import__("gr00t.model")
            base_proc = AutoProcessor.from_pretrained(stats_src, trust_remote_code=True)
            stats = getattr(
                getattr(base_proc, "state_action_processor", None), "statistics", None
            )
            if isinstance(stats, dict):
                emb_key = str(emb.value)
                if emb_key in stats:
                    policy.processor.set_statistics(
                        {emb_key: stats[emb_key]}, override=True
                    )
                else:
                    policy.processor.set_statistics(stats, override=True)
                print(
                    "[INFO] overridden_processor_statistics_from:",
                    stats_src,
                )
        except Exception as e:
            print(
                "[WARN] failed to override processor statistics:",
                type(e).__name__,
                str(e),
            )

    cfg = getattr(policy.model, "config", None)
    proc = getattr(policy, "processor", None)
    if cfg is not None and proc is not None:
        for name in (
            "max_state_dim",
            "max_action_dim",
            "max_action_horizon",
            "apply_sincos_state_encoding",
            "use_relative_action",
        ):
            if hasattr(cfg, name) and hasattr(proc, name):
                before = getattr(proc, name)
                after = getattr(cfg, name)
                if before != after:
                    setattr(proc, name, after)
                    print(f"[INFO] patched_processor.{name}: {before} -> {after}")

    try:
        proc_modality = policy.processor.get_modality_configs()[emb.value]
        if len(getattr(proc_modality["video"], "modality_keys", []) or []) == 0:
            proc_modality["video"].modality_keys = ["ego_view"]
            proc_modality["video"].delta_indices = [0]
        proc_modality["language"].modality_keys = ["annotation.human.task_description"]
        proc_modality["language"].delta_indices = [0]
        proc_modality["state"].delta_indices = [0]
        policy.modality_configs = proc_modality
        print(
            "[INFO] modality_keys.video:",
            list(proc_modality["video"].modality_keys),
        )
        print(
            "[INFO] modality_keys.language:",
            list(proc_modality["language"].modality_keys),
        )
    except Exception as e:
        print("[WARN] failed to patch modality configs:", type(e).__name__, str(e))

    adv_swap_applied = False
    baseline_default_advantage_embedding_init_used = False
    baseline_default_advantage_embedding_reason: str | None = None
    allow_baseline_default_advantage_embedding_init = bool(
        args.allow_baseline_default_advantage_embedding_init
    )
    if allow_baseline_default_advantage_embedding_init:
        if not bool(args.require_advantage_embedding):
            raise ValueError(
                "--allow-baseline-default-advantage-embedding-init requires --require-advantage-embedding"
            )
        if base_model_path_raw or adv_from_raw or overlay_from_raw:
            raise ValueError(
                "--allow-baseline-default-advantage-embedding-init is baseline-only and cannot be combined with "
                "--base-model-path, --adv-embedding-from, or --overlay-from"
            )

    adv_src_dir_raw = adv_from_raw or model_path_raw
    adv_src_path = Path(str(adv_src_dir_raw)).expanduser()
    index_path = adv_src_path / "model.safetensors.index.json"
    print("[INFO] adv_embedding_source_raw:", adv_src_dir_raw)
    print("[INFO] adv_embedding_source_is_file:", bool(adv_src_path.is_file()))
    print("[INFO] adv_embedding_source_is_dir:", bool(adv_src_path.is_dir()))
    print("[INFO] adv_embedding_index_path:", str(index_path))
    w_key = "action_head.advantage_embedding.weight"
    b_key = "action_head.advantage_embedding.bias"

    def _swap_in_recap_action_head(
        *,
        adv_w: Any | None,
        adv_b: Any | None,
        use_default_advantage_embedding: bool = False,
    ) -> None:
        recap_ah = GR00TRecapActionHead(config=policy.model.config)
        if adv_w is not None and adv_b is not None:
            print("[INFO] ckpt_adv_weight_shape:", tuple(adv_w.shape))
            out_features = int(recap_ah.advantage_embedding.weight.shape[0])
            ckpt_out_features = int(adv_w.shape[0])
            if out_features != ckpt_out_features:
                msg = (
                    "advantage_embedding output dim mismatch: "
                    f"model_config_dim={out_features} ckpt_dim={ckpt_out_features}"
                )
                if bool(args.strict):
                    raise ValueError(msg)
                print("[WARN]", msg)
                print(
                    "[WARN] adjusting advantage_embedding dim to match checkpoint (strict=False)"
                )
                recap_ah.advantage_embedding = torch.nn.Linear(1, ckpt_out_features)
        elif not use_default_advantage_embedding:
            raise ValueError(
                "_swap_in_recap_action_head requires checkpoint weights or explicit default-init mode"
            )

        missing, unexpected = recap_ah.load_state_dict(
            policy.model.action_head.state_dict(), strict=False
        )
        print("[INFO] recap_action_head_load missing_keys:", list(missing))
        print("[INFO] recap_action_head_load unexpected_keys:", list(unexpected))

        if adv_w is not None and adv_b is not None:
            with torch.no_grad():
                recap_ah.advantage_embedding.weight.copy_(
                    adv_w.to(dtype=recap_ah.advantage_embedding.weight.dtype)
                )
                recap_ah.advantage_embedding.bias.copy_(
                    adv_b.to(dtype=recap_ah.advantage_embedding.bias.dtype)
                )
        else:
            print("[INFO] baseline_default_advantage_embedding_init: True")
            print(
                "[INFO] baseline_default_advantage_embedding_shape:",
                tuple(recap_ah.advantage_embedding.weight.shape),
            )

        policy.model.action_head = recap_ah
        policy.model.action_head.to(device=policy.model.device, dtype=torch.bfloat16)
        policy.model.action_head.eval()

    def _try_load_adv_embedding_from_overlay_source() -> tuple[Any | None, Any | None]:
        if adv_from_raw or not overlay_from_raw or not overlay_include_raw:
            return None, None
        try:
            include_re = re.compile(overlay_include_raw)
        except re.error as exc:
            raise ValueError(
                f"Invalid --overlay-include-regex: {overlay_include_raw!r} ({exc})"
            ) from exc
        try:
            exclude_re = re.compile(overlay_exclude_raw)
        except re.error as exc:
            raise ValueError(
                f"Invalid --overlay-exclude-regex: {overlay_exclude_raw!r} ({exc})"
            ) from exc

        if not include_re.match(w_key) or not include_re.match(b_key):
            print(
                "[INFO] overlay_advantage_source_skipped: include regex does not cover advantage_embedding keys"
            )
            return None, None
        if exclude_re.match(w_key) or exclude_re.match(b_key):
            print(
                "[INFO] overlay_advantage_source_skipped: exclude regex filters advantage_embedding keys"
            )
            return None, None

        overlay_src_path = Path(str(overlay_from_raw)).expanduser()
        overlay_index_path = overlay_src_path / "model.safetensors.index.json"
        print("[INFO] overlay_advantage_source_raw:", str(overlay_from_raw))
        print(
            "[INFO] overlay_advantage_source_is_file:",
            bool(overlay_src_path.is_file()),
        )
        print(
            "[INFO] overlay_advantage_source_is_dir:",
            bool(overlay_src_path.is_dir()),
        )
        print(
            "[INFO] overlay_advantage_index_path:",
            str(overlay_index_path),
        )

        if overlay_src_path.is_file() and overlay_src_path.suffix == ".safetensors":
            sd = load_file(str(overlay_src_path), device="cpu")
            if w_key in sd and b_key in sd:
                print(
                    "[INFO] loaded_adv_embedding_from_overlay_safetensors:",
                    str(overlay_src_path),
                )
                return sd[w_key], sd[b_key]
            return None, None

        if overlay_src_path.is_dir() and overlay_index_path.is_file():
            idx = json.loads(overlay_index_path.read_text(encoding="utf-8"))
            wm = idx.get("weight_map", {})
            if w_key not in wm or b_key not in wm:
                return None, None
            w_path = overlay_src_path / wm[w_key]
            b_path = overlay_src_path / wm[b_key]
            sd_w = load_file(str(w_path), device="cpu")
            sd_b = (
                sd_w
                if str(w_path) == str(b_path)
                else load_file(str(b_path), device="cpu")
            )
            print(
                "[INFO] loaded_adv_embedding_from_overlay_ckpt_dir:",
                str(overlay_src_path),
            )
            return sd_w[w_key], sd_b[b_key]

        return None, None

    adv_w = None
    adv_b = None
    use_default_advantage_embedding = False
    if adv_src_path.is_file() and adv_src_path.suffix == ".safetensors":
        sd = load_file(str(adv_src_path), device="cpu")
        if w_key in sd and b_key in sd:
            adv_w = sd[w_key]
            adv_b = sd[b_key]
            print("[INFO] loaded_adv_embedding_from_safetensors:", str(adv_src_path))
        elif bool(args.require_advantage_embedding):
            adv_w, adv_b = _try_load_adv_embedding_from_overlay_source()
            if adv_w is None or adv_b is None:
                if allow_baseline_default_advantage_embedding_init:
                    use_default_advantage_embedding = True
                    baseline_default_advantage_embedding_reason = (
                        "missing_from_safetensors_file"
                    )
                    print(
                        "[WARN] baseline-safe fallback: adv embedding file is missing required keys; "
                        "installing default-initialized GR00TRecapActionHead.advantage_embedding for unconditional baseline"
                    )
                else:
                    raise KeyError(
                        f"Adv embedding file is missing keys; expected {w_key!r} and {b_key!r}"
                    )
        else:
            print(
                "[WARN] adv embedding file missing required keys; serving without advantage conditioning"
            )
    elif adv_src_path.is_dir() and index_path.is_file():
        ckpt_dir = adv_src_path
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        wm = idx.get("weight_map", {})
        has_adv = bool(w_key in wm and b_key in wm)
        if has_adv:
            w_path = ckpt_dir / wm[w_key]
            b_path = ckpt_dir / wm[b_key]
            sd_w = load_file(str(w_path), device="cpu")
            sd_b = (
                sd_w
                if str(w_path) == str(b_path)
                else load_file(str(b_path), device="cpu")
            )
            adv_w = sd_w[w_key]
            adv_b = sd_b[b_key]
            print("[INFO] loaded_adv_embedding_from_ckpt_dir:", str(ckpt_dir))
        elif bool(args.require_advantage_embedding):
            adv_w, adv_b = _try_load_adv_embedding_from_overlay_source()
            if adv_w is None or adv_b is None:
                if allow_baseline_default_advantage_embedding_init:
                    use_default_advantage_embedding = True
                    baseline_default_advantage_embedding_reason = (
                        "missing_from_checkpoint_weight_map"
                    )
                    print(
                        "[WARN] baseline-safe fallback: checkpoint has no advantage_embedding weights; "
                        "installing default-initialized GR00TRecapActionHead.advantage_embedding for unconditional baseline"
                    )
                else:
                    raise KeyError(
                        f"Checkpoint does not contain advantage_embedding weights; expected {w_key!r} and {b_key!r} in weight_map"
                    )
        else:
            print(
                "[WARN] checkpoint has no advantage_embedding weights; serving without advantage conditioning"
            )

    if adv_w is not None and adv_b is not None:
        _swap_in_recap_action_head(adv_w=adv_w, adv_b=adv_b)
        adv_swap_applied = True
    elif use_default_advantage_embedding:
        _swap_in_recap_action_head(
            adv_w=None,
            adv_b=None,
            use_default_advantage_embedding=True,
        )
        adv_swap_applied = True
        baseline_default_advantage_embedding_init_used = True
        provenance["baseline_default_advantage_embedding_init"] = True
        provenance["baseline_default_advantage_embedding_init_reason"] = (
            baseline_default_advantage_embedding_reason
        )
    elif bool(args.require_advantage_embedding):
        adv_w, adv_b = _try_load_adv_embedding_from_overlay_source()
        if adv_w is not None and adv_b is not None:
            print("[INFO] retrying_advantage_swap_with_overlay_source: True")
            _swap_in_recap_action_head(adv_w=adv_w, adv_b=adv_b)
            adv_swap_applied = True
        else:
            hint = ""
            if base_model_path_raw:
                hint = (
                    " Tip: you set --base-model-path, so you likely want "
                    "--adv-embedding-from /path/to/finetune/checkpoint-dir"
                )
            raise FileNotFoundError(
                "Missing local safetensors index; when --require-advantage-embedding is enabled, "
                + "--adv-embedding-from (or --model-path) must be a local checkpoint directory containing model.safetensors.index.json or a safetensors file containing advantage_embedding keys"
                + hint
            )
    else:
        print(
            "[INFO] non-local model path (or missing safetensors index); serving without advantage conditioning"
        )

    def _apply_overlay_from_local_checkpoint(*, overlay_src: Path) -> bool:
        if not overlay_src.exists():
            raise FileNotFoundError(f"overlay source does not exist: {overlay_src}")
        if not overlay_include_raw:
            return False

        try:
            include_re = re.compile(overlay_include_raw)
        except re.error as exc:
            raise ValueError(
                f"Invalid --overlay-include-regex: {overlay_include_raw!r} ({exc})"
            ) from exc
        try:
            exclude_re = re.compile(overlay_exclude_raw)
        except re.error as exc:
            raise ValueError(
                f"Invalid --overlay-exclude-regex: {overlay_exclude_raw!r} ({exc})"
            ) from exc

        target_sd = policy.model.state_dict()
        overlay_sd: dict[str, Any] = {}
        missing_in_target: list[str] = []
        shape_mismatch: list[str] = []

        def _maybe_add(k: str, v: Any) -> None:
            if not include_re.match(k):
                return
            if exclude_re.match(k):
                return
            if k not in target_sd:
                missing_in_target.append(k)
                return
            tgt = target_sd[k]
            if (
                hasattr(tgt, "shape")
                and hasattr(v, "shape")
                and tuple(tgt.shape) != tuple(v.shape)
            ):
                shape_mismatch.append(k)
                if bool(args.strict):
                    raise ValueError(
                        f"overlay shape mismatch for key {k!r}: target={tuple(tgt.shape)} src={tuple(v.shape)}"
                    )
                return
            try:
                overlay_sd[k] = v.to(device=tgt.device, dtype=tgt.dtype)
            except Exception:
                overlay_sd[k] = v

        if overlay_src.is_file() and overlay_src.suffix == ".safetensors":
            sd = load_file(str(overlay_src), device="cpu")
            for k, v in sd.items():
                _maybe_add(str(k), v)
        elif overlay_src.is_dir():
            index_path2 = overlay_src / "model.safetensors.index.json"
            if not index_path2.is_file():
                raise FileNotFoundError(
                    f"overlay source dir missing model.safetensors.index.json: {overlay_src}"
                )
            idx = json.loads(index_path2.read_text(encoding="utf-8"))
            wm = idx.get("weight_map", {})
            if not isinstance(wm, dict):
                raise TypeError(f"Invalid overlay weight_map type: {type(wm).__name__}")
            file_to_keys: dict[str, list[str]] = {}
            for k, rel in wm.items():
                if not isinstance(k, str) or not isinstance(rel, str):
                    continue
                if not include_re.match(k) or exclude_re.match(k):
                    continue
                file_to_keys.setdefault(rel, []).append(k)
            for rel, keys in sorted(file_to_keys.items()):
                shard_path = (overlay_src / rel).resolve()
                sd = load_file(str(shard_path), device="cpu")
                for k in keys:
                    if k in sd:
                        _maybe_add(k, sd[k])
        else:
            raise ValueError(f"Invalid overlay source: {overlay_src}")

        if len(overlay_sd) == 0:
            print(
                "[INFO] overlay_applied: False (no matching keys)",
                f"include={overlay_include_raw!r}",
                f"exclude={overlay_exclude_raw!r}",
            )
            if missing_in_target:
                print(
                    "[INFO] overlay_missing_in_target_count:",
                    int(len(missing_in_target)),
                )
            if shape_mismatch:
                print(
                    "[INFO] overlay_shape_mismatch_count:",
                    int(len(shape_mismatch)),
                )
            return False

        missing_keys, unexpected_keys = policy.model.load_state_dict(
            overlay_sd, strict=False
        )
        print(
            "[INFO] overlay_applied: True",
            "keys=",
            int(len(overlay_sd)),
            "missing_keys=",
            int(len(missing_keys)),
            "unexpected_keys=",
            int(len(unexpected_keys)),
            "missing_in_target=",
            int(len(missing_in_target)),
            "shape_mismatch=",
            int(len(shape_mismatch)),
        )
        return True

    overlay_applied = False
    if overlay_from_raw and overlay_include_raw:
        overlay_src_path = Path(str(overlay_from_raw)).expanduser()
        print("[INFO] overlay_from:", str(overlay_src_path))
        print("[INFO] overlay_include_regex:", overlay_include_raw)
        print("[INFO] overlay_exclude_regex:", overlay_exclude_raw)
        overlay_applied = _apply_overlay_from_local_checkpoint(
            overlay_src=overlay_src_path
        )

    policy.model.eval()
    print(
        "[INFO] advantage_swap_applied:",
        bool(adv_swap_applied),
    )
    if baseline_default_advantage_embedding_init_used:
        print(
            "[INFO] baseline_default_advantage_embedding_init_reason:",
            baseline_default_advantage_embedding_reason,
        )
    if overlay_from_raw and overlay_include_raw:
        print("[INFO] overlay_applied:", bool(overlay_applied))
    if bool(args.use_sim_policy_wrapper):
        policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(policy=policy, host=str(args.host), port=int(args.port))

    def _handle_ping() -> dict[str, Any]:
        return {"status": "ok", "message": "Server is running", **server_info}

    server.register_endpoint("ping", _handle_ping, requires_input=False)
    server.register_endpoint(
        "get_server_info", lambda: dict(server_info), requires_input=False
    )
    server.register_endpoint(
        "get_provenance", lambda: dict(provenance), requires_input=False
    )
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
