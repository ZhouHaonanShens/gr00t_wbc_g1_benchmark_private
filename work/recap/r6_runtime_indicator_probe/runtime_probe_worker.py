from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHONPATH_ROOTS = (
    REPO_ROOT,
    REPO_ROOT / "submodules/Isaac-GR00T",
    REPO_ROOT / "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl",
    REPO_ROOT / "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobosuite",
    REPO_ROOT / "submodules/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobocasa",
    REPO_ROOT / "submodules/Isaac-GR00T/external_dependencies/robocasa",
)
DEFAULT_PROMPT_RAW = "pick up the apple, walk left and place the apple on the plate."
DEFAULT_ENV_NAME = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"


def _prepare_import_env() -> None:
    os.environ.setdefault("GR00T_SKIP_WBC_REEXEC", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    for root in reversed(PYTHONPATH_ROOTS):
        text = str(root)
        if text not in sys.path:
            sys.path.insert(0, text)


def _json_sha(payload: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _first_five_action_l2(action: dict[str, Any]) -> tuple[float, float, float, float, float]:
    import numpy as np

    arrays = []
    for key in sorted(action):
        value = np.asarray(action[key])
        if value.ndim == 3:
            value = value[0]
        elif value.ndim == 2:
            pass
        else:
            value = value.reshape(1, -1)
        arrays.append(value.astype(np.float64, copy=False))
    horizon = min([a.shape[0] for a in arrays] + [5])
    norms = [float(np.linalg.norm(np.concatenate([a[i].reshape(-1) for a in arrays]))) for i in range(horizon)]
    while len(norms) < 5:
        norms.append(0.0)
    return tuple(norms[:5])  # type: ignore[return-value]


def _run_once(cell_id: str, seed: int, max_steps: int, indicator_mode: str) -> dict[str, Any]:
    _prepare_import_env()
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from work.recap import policy as recap_policy
    from work.recap.r3_contract_parity.collectors import resolve_cell_ckpt
    from work.recap.r6_runtime_indicator_probe import instrumentation
    from work.recap.scripts import gr00t_g3_formal_eval as formal_eval

    checkpoint = resolve_cell_ckpt(cell_id)
    base_policy = recap_policy.TextIndicatorGr00tPolicy(
        EmbodimentTag.UNITREE_G1,
        str(checkpoint),
        device="cuda",
        strict=True,
        attn_implementation="eager",
    )
    token_snapshot = instrumentation.attach_tokenizer_hook(base_policy)
    action_snapshot = instrumentation.attach_action_head_input_hook(base_policy.model)
    policy = Gr00tSimPolicyWrapper(base_policy, strict=True)
    helper3d = formal_eval._load_helper3d()
    env, _metadata = formal_eval._prepare_env(
        helper3d,
        policy.get_modality_config(),
        DEFAULT_ENV_NAME,
        int(max_steps),
        formal_eval.DEFAULT_N_ACTION_STEPS,
    )
    try:
        obs, _info = env.reset(seed=int(seed))
        for language_key in ("task", "annotation.human.coarse_action"):
            if language_key in obs:
                obs[language_key] = (DEFAULT_PROMPT_RAW,)
        options = {"seed": int(seed), "indicator_mode": str(indicator_mode)}
        action, action_info = policy.get_action(obs, options=options)
        token_payload = json.loads(token_snapshot())
        action_payload = json.loads(action_snapshot())
        prompt_text = str(token_payload.get("prompt_text_at_tokenizer", ""))
        condition_sha = str(action_payload.get("action_head_conditioning_sha256") or _json_sha(action))
        return {
            "cell_id": str(cell_id),
            "episode_seed": int(seed),
            "indicator_mode": str(indicator_mode),
            "prompt_text_at_tokenizer": prompt_text,
            "prompt_tokens_sha256": str(token_payload.get("prompt_tokens_sha256") or _json_sha(prompt_text)),
            "action_head_conditioning_sha256": condition_sha,
            "first_5_actions_l2": list(_first_five_action_l2(dict(action))),
            "indicator_substring_present": ("Advantage:" in prompt_text or "advantage" in prompt_text.lower()),
            "action_info": action_info,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m work.recap.r6_runtime_indicator_probe.runtime_probe_worker")
    parser.add_argument("--cell", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force-indicator-mode", required=True, choices=("positive", "negative"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = _run_once(str(args.cell).strip().upper(), int(args.seed), int(args.max_steps), str(args.force_indicator_mode))
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
