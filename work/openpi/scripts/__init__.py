from __future__ import annotations

import importlib


_SCRIPT_ALIAS_MAP = {
    "libero_rollout_eval_v21": "work.openpi.eval.workflows.rollout_support",
    "libero_recap_collect": "work.openpi.pipelines.recap.collect",
    "libero_recap_iteration": "work.openpi.pipelines.recap.iteration",
    "libero_recap_loop": "work.openpi.pipelines.recap.loop",
    "libero_recap_merge_data": "work.openpi.pipelines.recap.merge",
    "libero_recap_train": "work.openpi.pipelines.recap.policy_training",
    "train_recap_critic": "work.openpi.pipelines.recap.critic_training",
    "libero_variant_train": "work.openpi.pipelines.recap.variant_training",
}


def __getattr__(name: str):
    target = _SCRIPT_ALIAS_MAP.get(name)
    if target is None:
        raise AttributeError(name)
    return importlib.import_module(target)
