# work/recap/critic_vlm/

本目录承接 VLM critic 的共享实现与训练实现层。

本目录的实现与阅读约定，以 `agent/contract/script_workflow_layering_contract.md`（通用编程实现与可读性合同）为总则；当前具体布局如下：

- `train.py`：保留的稳定 public import surface，仅继续暴露 `TrainConfig`、`TrainResult`、`WarmstartPlan`、`PublicWarmstartSample` 与 `run_vlm_critic_training`
- `inference.py`：保留的稳定 public import surface，仅继续暴露 `run_critic_inference` 与 `run_artifact_smoke`
- `training/workflow.py`：真实训练主流程，定义 `VlmCriticTrainingWorkflow`、模型装配与 epoch orchestration
- `training/data.py`：warmstart 样本发现、formal manifest 校验、dataset/dataloader 构建
- `training/artifacts.py`：critic artifact、metrics、provenance、processor/model 落盘
- `training/runtime.py`：torch/model runtime import helper、seed/device/optimizer/epoch helper
- `training/contracts.py`：训练 public/internal dataclass 与共享常量
- `inference_runtime/workflow.py`：真实推理主流程，定义 `CriticInferenceWorkflow` 与 `ArtifactSmokeWorkflow`
- `inference_runtime/synthetic_backend.py`：synthetic checker backend 推理实现
- `inference_runtime/qwen_backend.py`：Qwen late-fusion backend 推理实现
- `inference_runtime/common.py`：推理侧 processor contract 校验、softmax、video/frame/device helper

保留 `train.py` 的原因：

1. `work/recap/scripts/43_vlm_critic_train.py` 仍直接依赖 `work.recap.critic_vlm.train`
2. 当前仓库 contract 明确允许保留薄兼容层，但真实实现必须下沉到 reusable workflow/service 层

保留 `inference.py` 的原因：

1. `work/recap/critic_vlm/__init__.py` 仍对外导出 `run_critic_inference` 与 `run_artifact_smoke`
2. `work/recap/scripts/43b_vlm_critic_artifact_smoke.py`、`work/recap/scripts/44_vlm_critic_offline_gate.py` 继续依赖该 public surface

推荐阅读顺序：

1. `work/recap/scripts/43_vlm_critic_train.py`：看公开 CLI 如何组装 `TrainConfig`
2. `work/recap/critic_vlm/train.py`：看保留的兼容 import surface
3. `work/recap/critic_vlm/inference.py`：看保留的推理兼容 import surface
4. `work/recap/critic_vlm/training/workflow.py`：看真实训练阶段链路
5. `work/recap/critic_vlm/inference_runtime/workflow.py`：看真实推理阶段链路
6. `work/recap/critic_vlm/training/data.py` / `training/artifacts.py` 与 `inference_runtime/*.py`：看具体 service/helper
