# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportMissingParameterType=false

from __future__ import annotations

from . import _upstream_openpi_recap_policy_config as _upstream_policy_config
from ._upstream_openpi_recap_policy_config import *  # noqa: F401,F403

from openpi.recap_overlay.config import build_recap_policy_metadata


def create_trained_policy(
    train_config,
    checkpoint_dir,
    *,
    repack_transforms=None,
    sample_kwargs=None,
    default_prompt=None,
    norm_stats=None,
    pytorch_device=None,
):
    policy = _upstream_policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        repack_transforms=repack_transforms,
        sample_kwargs=sample_kwargs,
        default_prompt=default_prompt,
        norm_stats=norm_stats,
        pytorch_device=pytorch_device,
    )
    if bool((train_config.policy_metadata or {}).get("recap_enabled", False)):
        merged_metadata = dict(policy.metadata)
        merged_metadata.update(build_recap_policy_metadata())
        policy._metadata = merged_metadata
    return policy
