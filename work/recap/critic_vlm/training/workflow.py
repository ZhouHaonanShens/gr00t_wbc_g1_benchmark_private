from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common import JsonObject
from ..targets import resolve_effective_bin_centers
from .artifacts import TrainingArtifactWriter
from .contracts import (
    FUSION_HIDDEN_DIM,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    PROPRIO_DIM,
    PROPRIO_HIDDEN_DIM,
    T_HIDDEN_DIM,
    TrainConfig,
    TrainResult,
)
from .data import CriticTrainingDataService, LoadedManifestSet, TrainingDataBundle
from .runtime import (
    build_optimizer,
    epoch_metrics,
    load_modeling_module,
    runtime_import_torch,
    select_device,
    set_lora_trainable,
    set_seed,
)


@dataclass(frozen=True)
class TrainingHistoryBundle:
    warmstart_history: list[JsonObject]
    formal_history: list[JsonObject]
    best_state: dict[str, Any] | None
    best_val_loss: float


@dataclass
class CriticModelFactory:
    config: TrainConfig

    def build(self, *, bin_centers: list[float], device: str) -> tuple[Any, Any]:
        modeling_mod = load_modeling_module()
        processor = modeling_mod.load_qwen3_vl_processor(self.config.base_model)
        backbone = modeling_mod.load_qwen3_vl_backbone(
            base_model=self.config.base_model,
            torch_dtype=modeling_mod.resolve_torch_dtype(device),
            attn_implementation=self.config.attn_implementation,
        )
        hidden_size = modeling_mod.resolve_hidden_size(backbone)
        critic = modeling_mod.Qwen3VLLateFusionCritic(
            backbone=backbone,
            hidden_size=hidden_size,
            bin_centers=bin_centers,
            proprio_dim=PROPRIO_DIM,
            proprio_hidden_dim=PROPRIO_HIDDEN_DIM,
            t_hidden_dim=T_HIDDEN_DIM,
            fusion_hidden_dim=FUSION_HIDDEN_DIM,
            use_proprio=bool(self.config.use_proprio),
            use_t_norm=bool(self.config.use_t_norm),
        )
        critic.freeze_backbone()
        critic.unfreeze_trainable_modules()
        critic.backbone = modeling_mod.apply_top_block_lora(
            critic.backbone,
            top_n=int(self.config.top_n_lora_blocks),
            lora_rank=int(LORA_RANK),
            lora_alpha=int(LORA_ALPHA),
            lora_dropout=float(LORA_DROPOUT),
        )
        critic.unfreeze_trainable_modules()
        torch = runtime_import_torch()
        critic = critic.to(device)
        critic.keep_trainable_path_fp32()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return processor, critic


@dataclass
class EpochRunner:
    critic: Any
    device: str

    def execute(
        self, *, data_bundle: TrainingDataBundle, config: TrainConfig
    ) -> TrainingHistoryBundle:
        torch = runtime_import_torch()
        criterion = torch.nn.CrossEntropyLoss()
        best_state: dict[str, Any] | None = None
        best_val_loss = float("inf")
        warmstart_history: list[JsonObject] = []
        if data_bundle.public_warmstart_loader is not None:
            set_lora_trainable(self.critic, enabled=False)
            head_optimizer = build_optimizer(model=self.critic, lr=config.lr_head)
            for epoch in range(int(max(0, config.warmstart_epochs))):
                train_loss, train_ce, train_rank = epoch_metrics(
                    critic=self.critic,
                    data_loader=data_bundle.public_warmstart_loader,
                    criterion=criterion,
                    device=self.device,
                    training=True,
                    optimizer=head_optimizer,
                )
                warmstart_history.append(
                    {
                        "epoch": int(epoch + 1),
                        "train_loss": float(train_loss),
                        "ce_loss": float(train_ce),
                        "ranking_loss": float(train_rank),
                        "source": "localized_public_warmstart",
                        "sample_count": int(len(data_bundle.public_warmstart_samples)),
                    }
                )
            set_lora_trainable(self.critic, enabled=True)

        modeling_mod = load_modeling_module()
        formal_optimizer = build_optimizer(model=self.critic, lr=config.lr_lora)
        formal_history: list[JsonObject] = []
        for epoch in range(int(max(1, config.formal_epochs))):
            train_loss, train_ce, train_rank = epoch_metrics(
                critic=self.critic,
                data_loader=data_bundle.train_loader,
                criterion=criterion,
                device=self.device,
                training=True,
                optimizer=formal_optimizer,
            )
            val_loss, val_ce, val_rank = epoch_metrics(
                critic=self.critic,
                data_loader=data_bundle.val_loader,
                criterion=criterion,
                device=self.device,
                training=False,
                optimizer=None,
            )
            formal_history.append(
                {
                    "epoch": int(epoch + 1),
                    "train_loss": float(train_loss),
                    "train_ce_loss": float(train_ce),
                    "train_ranking_loss": float(train_rank),
                    "val_loss": float(val_loss),
                    "val_ce_loss": float(val_ce),
                    "val_ranking_loss": float(val_rank),
                }
            )
            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_state = modeling_mod.select_trainable_state_dict(self.critic)
        return TrainingHistoryBundle(
            warmstart_history=warmstart_history,
            formal_history=formal_history,
            best_state=best_state,
            best_val_loss=best_val_loss,
        )


def _maybe_restore_best_state(critic: Any, best_state: dict[str, Any] | None) -> None:
    if best_state is None:
        return
    load_modeling_module().load_partial_state_dict(critic, best_state)


@dataclass
class VlmCriticTrainingWorkflow:
    repo_root: Path
    config: TrainConfig
    data_service: CriticTrainingDataService = field(init=False)
    model_factory: CriticModelFactory = field(init=False)

    def __post_init__(self) -> None:
        self.data_service = CriticTrainingDataService(self.config)
        self.model_factory = CriticModelFactory(self.config)

    def _validate_public_surface_contract(self) -> None:
        if self.config.base_model != "Qwen/Qwen3-VL-2B-Instruct":
            raise ValueError(
                "base_model_fixed_mismatch: Task 6 requires base_model='Qwen/Qwen3-VL-2B-Instruct'"
            )
        if str(self.config.prompt_text_mode) not in {"manifest", "constant_query_only"}:
            raise ValueError(
                "prompt_text_mode_invalid: expected 'manifest' or 'constant_query_only', got "
                + repr(self.config.prompt_text_mode)
            )

    def _resolve_environment_support(self) -> Any:
        modeling_mod = load_modeling_module()
        support = modeling_mod.inspect_qwen3_vl_environment()
        if support.blocker:
            try:
                _ = modeling_mod.load_qwen3_vl_processor(self.config.base_model)
            except Exception:
                pass
            raise RuntimeError(support.blocker)
        return support

    def _effective_side_channels(self) -> list[str]:
        effective_side_channels = ["t_norm"] if bool(self.config.use_t_norm) else []
        if bool(self.config.use_proprio):
            effective_side_channels = ["proprio", *effective_side_channels]
        return effective_side_channels

    def execute(self) -> TrainResult:
        self._validate_public_surface_contract()
        support = self._resolve_environment_support()
        manifests = self.data_service.load_manifests()
        effective_bin_centers = resolve_effective_bin_centers(
            configured_bin_centers=self.config.bin_centers,
            manifest_bin_centers=manifests.train_manifest.bin_centers,
        )
        device = select_device(self.config.device)
        set_seed(self.config.seed)
        processor, critic = self.model_factory.build(
            bin_centers=effective_bin_centers,
            device=device,
        )
        data_bundle = self.data_service.build_data_bundle(
            processor=processor,
            manifests=manifests,
        )
        history = EpochRunner(critic=critic, device=device).execute(
            data_bundle=data_bundle,
            config=self.config,
        )
        _maybe_restore_best_state(critic, history.best_state)
        artifact_writer = TrainingArtifactWriter(
            repo_root=self.repo_root,
            config=self.config,
            train_manifest=data_bundle.train_manifest,
            val_manifest=data_bundle.val_manifest,
            support=support,
            device=device,
            warmstart_plan=data_bundle.warmstart_plan,
            effective_side_channels=self._effective_side_channels(),
            bin_centers=effective_bin_centers,
        )
        return artifact_writer.write(
            processor=processor,
            critic=critic,
            train_sample_count=data_bundle.train_sample_count,
            val_sample_count=data_bundle.val_sample_count,
            warmstart_history=history.warmstart_history,
            formal_history=history.formal_history,
            best_val_loss=history.best_val_loss,
        )


def run_vlm_critic_training(*, repo_root: Path, config: TrainConfig) -> TrainResult:
    return VlmCriticTrainingWorkflow(repo_root=repo_root, config=config).execute()
