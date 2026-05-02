from __future__ import annotations

import importlib
import math
import os
import random
from typing import Any, cast

from .contracts import RANKING_LOSS_WEIGHT


def load_modeling_module() -> Any:
    return importlib.import_module("work.recap.critic_vlm.modeling")


def runtime_import_torch() -> Any:
    import torch

    return torch


def runtime_import_torch_utils() -> tuple[Any, Any, Any]:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset

    return torch, F, (DataLoader, Dataset)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    torch = runtime_import_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def select_device(requested: str) -> str:
    torch = runtime_import_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_optimizer(*, model: Any, lr: float) -> Any:
    torch = runtime_import_torch()
    params = [
        param
        for param in model.parameters()
        if bool(getattr(param, "requires_grad", False))
    ]
    return torch.optim.AdamW(params, lr=float(lr))


def set_lora_trainable(model: Any, *, enabled: bool) -> None:
    for name, param in model.named_parameters():
        if ".lora_" not in str(name):
            continue
        param.requires_grad = bool(enabled)


def pairwise_ranking_loss(
    *,
    predicted_value: Any,
    target_return: Any,
    episode_index: Any,
) -> Any:
    torch, F, _ = runtime_import_torch_utils()
    device = predicted_value.device
    total = torch.tensor(0.0, dtype=predicted_value.dtype, device=device)
    count = 0
    unique_eps = torch.unique(episode_index)
    for ep in unique_eps.tolist():
        mask = episode_index == int(ep)
        if int(mask.sum().item()) < 2:
            continue
        pred_ep = predicted_value[mask]
        tgt_ep = target_return[mask]
        diff_target = tgt_ep.reshape(-1, 1) - tgt_ep.reshape(1, -1)
        sign = torch.sign(diff_target)
        valid = sign != 0
        if int(valid.sum().item()) == 0:
            continue
        diff_pred = pred_ep.reshape(-1, 1) - pred_ep.reshape(1, -1)
        losses = F.softplus(-sign[valid] * diff_pred[valid])
        total = total + losses.mean()
        count += 1
    if count == 0:
        return torch.tensor(0.0, dtype=predicted_value.dtype, device=device)
    return total / float(count)


def epoch_metrics(
    *,
    critic: Any,
    data_loader: Any,
    criterion: Any,
    device: str,
    training: bool,
    optimizer: Any | None,
) -> tuple[float, float, float]:
    torch = runtime_import_torch()
    total_loss = 0.0
    total_ce = 0.0
    total_rank = 0.0
    total_count = 0
    if training:
        critic.train()
    else:
        critic.eval()
    for batch in data_loader:
        model_inputs = move_batch_to_device(
            cast(dict[str, Any], batch["model_inputs"]), device
        )
        proprio = batch["proprio"].to(device)
        t_norm = batch["t_norm"].to(device)
        target_bin = batch["target_bin"].to(device)
        target_return = batch["target_return"].to(device)
        episode_index = batch["episode_index"].to(device)
        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = critic(model_inputs=model_inputs, proprio=proprio, t_norm=t_norm)
            ce_loss = criterion(output["logits"], target_bin)
            rank_loss = pairwise_ranking_loss(
                predicted_value=output["value_V_raw"],
                target_return=target_return,
                episode_index=episode_index,
            )
            loss = ce_loss + float(RANKING_LOSS_WEIGHT) * rank_loss
            if training and optimizer is not None:
                loss.backward()
                optimizer.step()
        batch_size = int(target_bin.shape[0])
        total_loss += float(loss.detach().item()) * float(batch_size)
        total_ce += float(ce_loss.detach().item()) * float(batch_size)
        total_rank += float(rank_loss.detach().item()) * float(batch_size)
        total_count += batch_size
    denom = float(max(1, total_count))
    return total_loss / denom, total_ce / denom, total_rank / denom


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed
