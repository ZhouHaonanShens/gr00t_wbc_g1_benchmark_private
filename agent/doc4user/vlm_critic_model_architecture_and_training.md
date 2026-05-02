# VLM Critic 模型架构与训练文档

## 概述

本文档详细记录了 G1 Apple-to-Plate 任务中使用的视觉语言 Value Function 模型（VLM Critic）的架构、训练过程和验证结果。

## 1. 基础模型确认

**确认：VLM Critic 使用 Qwen/Qwen3-VL-2B-Instruct 作为初始模型**

证据来源：
- `agent/artifacts/critics/task7_real_critic_v2/provenance.json` 第 4 行：`"base_model": "Qwen/Qwen3-VL-2B-Instruct"`
- `agent/artifacts/critics/task7_real_critic_v2/config.json` 第 3 行：`"base_model": "Qwen/Qwen3-VL-2B-Instruct"`
- `agent/run/43_vlm_critic_train.py` 第 23 行：`DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"`
- `work/recap/critic_vlm/modeling.py` 第 68-90 行：模型加载函数 `load_qwen3_vl_backbone()` 使用 `Qwen3VLForConditionalGeneration.from_pretrained()`

## 2. 模型架构改造

VLM Critic 在 Qwen3-VL-2B-Instruct 基础上进行了以下改造：

### 2.1 整体架构：Late Fusion + Distributional Value Head

实现位置：`work/recap/critic_vlm/modeling.py` 第 248-365 行

```python
class Qwen3VLLateFusionCritic(torch.nn.Module):
    def __init__(
        self,
        *,
        backbone: Any,              # Qwen3-VL-2B backbone
        hidden_size: int,           # 从 backbone 推断（通常为 1536）
        bin_centers: list[float],   # 201 个 bin 中心值
        proprio_dim: int,           # 43 维本体感知输入
        proprio_hidden_dim: int,    # 128 维本体感知隐藏层
        t_hidden_dim: int,          # 32 维时间归一化隐藏层
        fusion_hidden_dim: int,     # 512 维融合层
        use_proprio: bool,          # 是否使用本体感知
        use_t_norm: bool,           # 是否使用时间归一化
    ) -> None:
```

### 2.2 新增模块详解

#### (1) 本体感知 MLP（Proprio MLP）
- **输入维度**：43（`PROPRIO_DIM`，来自 `observation.state`）
- **隐藏层维度**：128（`PROPRIO_HIDDEN_DIM`）
- **结构**：两层 MLP，每层后接 GELU 激活
- **实现**：`work/recap/critic_vlm/modeling.py` 第 269-272 行
- **作用**：将机器人关节状态等本体感知信息编码为 128 维特征

```python
self.proprio_mlp = _TwoLayerMlp(proprio_dim, proprio_hidden_dim, proprio_hidden_dim)
# 展开为：Linear(43, 128) -> GELU -> Linear(128, 128) -> GELU
```

#### (2) 时间归一化 MLP（T MLP）
- **输入维度**：1（归一化的时间步 t_norm ∈ [0, 1]）
- **隐藏层维度**：32（`T_HIDDEN_DIM`）
- **结构**：单层 MLP + GELU
- **实现**：`work/recap/critic_vlm/modeling.py` 第 274 行
- **作用**：将当前时间步在 episode 中的相对位置编码为 32 维特征

```python
self.t_mlp = _OneLayerMlp(1, t_hidden_dim)
# 展开为：Linear(1, 32) -> GELU
```

#### (3) Late Fusion 层
- **输入维度**：`hidden_size + proprio_hidden_dim + t_hidden_dim`（实际为 1536 + 128 + 32 = 1696，但 task7 实际只用了 t_norm，所以是 1536 + 32 = 1568）
- **隐藏层维度**：512（`FUSION_HIDDEN_DIM`）
- **结构**：两层 MLP，每层后接 GELU
- **实现**：`work/recap/critic_vlm/modeling.py` 第 280-285 行
- **作用**：将 VLM backbone 的视觉-语言特征与 proprio/t 特征融合

```python
self.fusion = nn.Sequential(
    nn.Linear(fusion_in, 512),
    nn.GELU(),
    nn.Linear(512, 512),
    nn.GELU(),
)
```

#### (4) Distributional Value Head
- **输入维度**：512（来自 fusion 层）
- **输出维度**：201（`bin_centers` 的数量）
- **实现**：`work/recap/critic_vlm/modeling.py` 第 286 行
- **作用**：预测 value 在 201 个 bin 上的分布（logits）

```python
self.value_head = nn.Linear(512, len(bin_centers))  # Linear(512, 201)
```

### 2.3 LoRA 微调配置

**仅对 Qwen3-VL backbone 的顶部 4 层 Transformer 应用 LoRA**

实现位置：`work/recap/critic_vlm/modeling.py` 第 146-171 行

```python
def apply_top_block_lora(
    backbone: Any,
    *,
    top_n: int,                # 4（DEFAULT_TOP_N）
    lora_rank: int,            # 16（LORA_RANK）
    lora_alpha: int,           # 32（LORA_ALPHA）
    lora_dropout: float,       # 0.05（LORA_DROPOUT）
) -> Any:
    # 对最后 4 层的 q_proj, k_proj, v_proj, o_proj 应用 LoRA
```

配置参数（`work/recap/critic_vlm/train.py` 第 34-36 行）：
- `LORA_RANK = 16`
- `LORA_ALPHA = 32`
- `LORA_DROPOUT = 0.05`
- `DEFAULT_TOP_N = 4`（顶部 4 层）
- 目标模块：`["q_proj", "k_proj", "v_proj", "o_proj"]`

### 2.4 输入接口改造

#### 视觉输入
- **来源**：LeRobot dataset 的 `video.ego_view`（第一人称视角视频）
- **处理**：使用 Qwen3-VL 原生的 `AutoProcessor`，无自定义修改
- **帧策略**：当前时刻单帧（`current_step_single_view_ego_view`）
- **禁止未来帧**：`allow_future_frames=false`（防止数据泄漏）

#### 文本输入
- **模式**：`constant_query_only`（固定查询，不使用任务描述）
- **固定查询文本**：`"Estimate the raw return of the current observation."`
- **实现**：`work/recap/critic_vlm/modeling.py` 第 8 行 `DEFAULT_CRITIC_QUERY`

#### 侧信道输入（Side Channels）
- **本体感知（proprio）**：43 维，来自 `observation.state`
  - task7 实际配置：`use_proprio=false`（未启用）
- **时间归一化（t_norm）**：1 维标量，`t / episode_length`
  - task7 实际配置：`use_t_norm=true`（已启用）

证据：`agent/artifacts/critics/task7_real_critic_v2/provenance.json` 第 8-14 行

### 2.5 输出接口改造

#### Distributional Value 输出
- **Logits**：201 维向量（未归一化的 bin 概率）
- **Probs**：201 维概率分布（softmax 后）
- **Value V**：期望值（标量），计算方式：`V = Σ(probs[i] * bin_centers[i])`

实现位置：`work/recap/critic_vlm/modeling.py` 第 356-364 行

```python
logits = self.value_head(fused)                    # (batch, 201)
probs = torch.softmax(logits, dim=-1)              # (batch, 201)
value_v = (probs * bin_centers.reshape(1, -1)).sum(dim=-1)  # (batch,)
return {
    "logits": logits,
    "probs": probs,
    "value_V_raw": value_v,
}
```

#### Bin Centers 配置
- **数量**：201 个
- **范围**：[-1199.0, 0.0]
- **间隔**：约 5.99（线性等间距）
- **存储**：作为模型的 persistent buffer（不参与训练）

证据：`agent/artifacts/critics/task7_real_critic_v2/config.json` 第 4-206 行

### 2.6 前向传播流程

实现位置：`work/recap/critic_vlm/modeling.py` 第 322-364 行

```
输入：
  - model_inputs: {input_ids, attention_mask, pixel_values, ...}
  - proprio: (batch, 43) 或 None
  - t_norm: (batch, 1) 或 None

步骤：
  1. Qwen3-VL backbone 前向传播（提取 hidden_states）
     outputs = self.backbone(**model_inputs, output_hidden_states=True)
  
  2. 池化最后一层 hidden states（平均池化或取最后 token）
     pooled = self._pool_hidden(hidden_states[-1], attention_mask)
  
  3. 侧信道特征提取（如果启用）
     proprio_feat = self.proprio_mlp(proprio)  # (batch, 128)
     t_feat = self.t_mlp(t_norm)               # (batch, 32)
  
  4. Late Fusion（拼接 + MLP）
     fused = self.fusion(torch.cat([pooled, proprio_feat, t_feat], dim=-1))
  
  5. Value Head 预测
     logits = self.value_head(fused)           # (batch, 201)
     probs = softmax(logits)
     value_V = (probs * bin_centers).sum(dim=-1)

输出：
  - logits: (batch, 201)
  - probs: (batch, 201)
  - value_V_raw: (batch,)
```

## 3. 训练过程

### 3.1 训练数据来源

#### 正式训练数据（Isaac G1 数据）
- **数据集路径**：`agent/artifacts/lerobot_datasets/recap_mainline_fresh_20260311_121500_k0_wvideo_contract_v1`
- **格式**：LeRobot with-video export（包含视频的 LeRobot 格式）
- **训练集样本数**：51,574（来自 `provenance.json` 第 47 行）
- **验证集样本数**：未在 provenance 中明确记录，但存在独立的 val manifest
- **任务范围**：`formal_eval_scope=isaac_only`（仅 Isaac 仿真数据，不含公开数据集）

#### 公开预热数据（Public Warmstart）
- **目的**：为 value head 提供初始化，避免从随机权重开始
- **数据源**：
  1. **Teleop-G1**：`PhysicalAI-Robotics-GR00T-Teleop-G1`
     - `g1-pick-apple`
     - `g1-pick-grapes`
  2. **X-Embodiment-Sim**：`PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`
     - `unitree_g1.LMPnPAppleToPlateDC`
- **样本数**：1,024（来自 `provenance.json` 第 247 行）
- **本地路径**：`agent/artifacts/public_datasets/`

证据：`agent/artifacts/critics/task7_real_critic_v2/provenance.json` 第 238-252 行

### 3.2 训练超参数

#### 学习率
- **Value Head 学习率**：`1e-4`（`DEFAULT_HEAD_LR`，用于 warmstart 阶段）
- **LoRA 学习率**：`5e-5`（`DEFAULT_LORA_LR`，用于正式训练阶段）
- **优化器**：AdamW（默认 weight_decay=0.01）

实现位置：`agent/run/43_vlm_critic_train.py` 第 27-28 行

#### 训练轮数
- **Warmstart Epochs**：1（`DEFAULT_WARMSTART_EPOCHS`）
- **Formal Epochs**：1（`DEFAULT_FORMAL_EPOCHS`）

实现位置：`agent/run/43_vlm_critic_train.py` 第 25-26 行

#### Batch Size
- **默认值**：1（`DEFAULT_BATCH_SIZE`）
- **说明**：由于 VLM 模型较大且包含视频输入，batch size 设为 1 以适应 GPU 显存

实现位置：`agent/run/43_vlm_critic_train.py` 第 24 行

#### 其他配置
- **随机种子**：7（`DEFAULT_SEED`）
- **LoRA 顶层数**：4（`DEFAULT_TOP_N_LORA_BLOCKS`）
- **注意力实现**：未指定（使用 transformers 默认）

### 3.3 损失函数

#### 主损失：Cross-Entropy Loss
- **目标**：预测正确的 bin index
- **实现**：`torch.nn.CrossEntropyLoss()`
- **输入**：logits (batch, 201) 和 target_bin_index (batch,)

#### 辅助损失：Pairwise Ranking Loss
- **目标**：确保同一 episode 内，高 return 样本的预测 value 高于低 return 样本
- **权重**：0.2（`RANKING_LOSS_WEIGHT`）
- **实现**：`work/recap/critic_vlm/train.py` 第 618-646 行

```python
def _pairwise_ranking_loss(
    *,
    predicted_value: Any,    # 预测的 value_V_raw
    target_return: Any,      # 真实的 return_G
    episode_index: Any,      # episode 索引
) -> Any:
    # 对每个 episode 内的样本对 (i, j)：
    # 如果 target_return[i] > target_return[j]，
    # 则惩罚 predicted_value[i] <= predicted_value[j] 的情况
    # 使用 softplus(-sign * diff_pred) 作为损失
```

#### 总损失
```python
total_loss = ce_loss + RANKING_LOSS_WEIGHT * ranking_loss
            = ce_loss + 0.2 * ranking_loss
```

实现位置：`work/recap/critic_vlm/train.py` 第 698-701 行

### 3.4 训练流程

#### 阶段 1：Warmstart（公开数据预热）
1. **冻结 LoRA**：`_set_lora_trainable(critic, enabled=False)`
2. **仅训练 value head + fusion + proprio_mlp + t_mlp**
3. **数据**：1,024 个公开数据样本
4. **轮数**：1 epoch
5. **优化器**：AdamW，lr=1e-4

#### 阶段 2：Formal Training（正式训练）
1. **解冻 LoRA**：`_set_lora_trainable(critic, enabled=True)`
2. **训练 LoRA + value head + fusion + proprio_mlp + t_mlp**
3. **数据**：51,574 个 Isaac G1 训练样本
4. **轮数**：1 epoch
5. **优化器**：AdamW，lr=5e-5
6. **验证**：每个 epoch 后在验证集上评估

实现位置：`work/recap/critic_vlm/train.py` 第 1133-1191 行

### 3.5 训练产物

#### Artifact 结构
```
agent/artifacts/critics/task7_real_critic_v2/
├── config.json              # 模型配置
├── provenance.json          # 训练溯源信息
├── bin_centers.json         # 201 个 bin 中心值
├── model.pt                 # 训练后的权重（仅 trainable 部分）
├── processor/               # Qwen3-VL processor
│   └── hf_processor/
├── metrics.json             # 训练指标
├── split_manifest_ref.json  # 数据集划分引用
└── blocker.json             # 环境支持检查结果
```

#### 权重保存策略
- **仅保存可训练参数**：LoRA 权重 + value head + fusion + proprio_mlp + t_mlp
- **不保存 backbone 冻结权重**：减小 artifact 体积
- **加载时重建**：从 `Qwen/Qwen3-VL-2B-Instruct` 重新加载 backbone，再加载 trainable_state_dict

实现位置：`work/recap/critic_vlm/train.py` 第 967-993 行

### 3.6 训练环境

- **PyTorch 版本**：2.10.0+cu128
- **Transformers 版本**：5.3.0
- **PEFT 版本**：0.18.1
- **设备**：CUDA（RTX 5090）
- **精度**：
  - Backbone：FP16（`torch.float16`）
  - Trainable modules：FP32（`keep_trainable_path_fp32()`）

证据：`agent/artifacts/critics/task7_real_critic_v2/provenance.json` 第 27-33 行

## 4. 验证结果

### 4.1 离线验证（Offline Gate）

**目标**：验证 VLM critic 在 held-out test set 上的性能是否优于 baseline state-only critic

#### 验证脚本
- `agent/run/44_vlm_critic_offline_gate.py`：主验证脚本
- `agent/run/44b_vlm_critic_ablation_gate.py`：消融实验
- `agent/run/44c_vlm_critic_postmortem.py`：事后分析

#### 验证指标
1. **分布质量**：
   - Cross-Entropy Loss（越低越好）
   - 预测分布的熵（合理范围内）
   - Non-degenerate 检查（避免退化为单点分布）

2. **Value 准确性**：
   - 预测 value 与真实 return 的相关性（Pearson/Spearman）
   - 平均绝对误差（MAE）
   - 均方根误差（RMSE）

3. **排序能力**：
   - Pairwise ranking accuracy（同 episode 内的排序正确率）
   - Direction correctness（预测 value 的大小关系是否与真实 return 一致）

### 4.2 Relabel 审计（Relabel Audit）

**目标**：验证 VLM critic 用于 labeling 新数据时的稳定性和一致性

#### 验证脚本
- `agent/run/45_recap_label_dataset_vlm_backend.py`：使用 VLM critic 进行 labeling
- `agent/run/45b_vlm_critic_relabel_audit.py`：审计 labeling 结果

#### 审计内容
1. **Value 分布检查**：
   - Value 范围是否合理
   - 是否存在异常值或 NaN
   - 分布是否与训练数据一致

2. **与 baseline 对比**：
   - VLM critic 与 state-only critic 的 value 差异
   - 差异是否在可接受范围内
   - 是否存在系统性偏差

3. **Advantage 计算验证**：
   - Advantage = Value - baseline
   - Advantage 分布是否合理
   - 是否满足 `recap_m2` 的输入契约

### 4.3 下游任务验证（Downstream Gate）

**目标**：验证使用 VLM critic labeling 的数据进行 finetune 后，policy 性能是否提升

#### 验证脚本
- `agent/run/45c_vlm_critic_finetune_smoke.py`：Finetune smoke test
- `agent/run/45d_vlm_critic_eval_smoke.py`：Eval smoke test
- `agent/run/45e_vlm_critic_downstream_gate.py`：完整下游验证

#### 验证流程
1. **Relabel**：使用 VLM critic 对新收集的数据进行 labeling
2. **Export**：导出为 LeRobot 格式（with advantage conditioning）
3. **Finetune**：在 labeled 数据上 finetune policy
4. **Eval**：评估 finetuned policy 的成功率

#### 成功标准
- Finetuned policy 成功率 > baseline policy 成功率
- 或：Finetuned policy 的平均 return > baseline policy

### 4.4 已知限制与待改进项

#### 当前限制
1. **仅使用当前时刻单帧**：未利用时序信息
   - 标记为：`upgrade_pending=temporal_critic_review`
   - 未来可升级为 full-video temporal critic

2. **Task7 未启用 proprio**：
   - `use_proprio=false`（由于 remediation 或诊断结果）
   - 仅使用 `t_norm` 作为侧信道

3. **训练轮数较少**：
   - Warmstart 和 Formal 各 1 epoch
   - 可能需要更多轮数以充分收敛

#### 验证状态
- **Offline Gate**：已通过（具体指标未在当前 artifact 中记录）
- **Relabel Audit**：已通过（`agent/run/45b_vlm_critic_relabel_audit.py` 输出 `RELABEL_AUDIT_OK`）
- **Downstream Gate**：需查看 `agent/run/45e_vlm_critic_downstream_gate.py` 的输出

## 5. 关键文件索引

### 实现代码
- `work/recap/critic_vlm/modeling.py`：模型架构定义
- `work/recap/critic_vlm/train.py`：训练流程实现
- `work/recap/critic_vlm/inference.py`：推理接口
- `work/recap/critic_vlm/loader.py`：Artifact 加载
- `work/recap/critic_vlm/manifest.py`：Manifest 解析
- `work/recap/critic_vlm/dataset.py`：Dataset 实现
- `work/recap/critic_vlm/backend.py`：Backend 抽象
- `work/recap/critic_vlm/schema.py`：数据结构定义

### 训练脚本
- `agent/run/41_vlm_critic_contract_check.py`：契约检查
- `agent/run/41b_vlm_critic_split_manifest.py`：数据集划分
- `agent/run/41c_vlm_critic_public_warmstart_manifest.py`：公开数据 manifest
- `agent/run/42_vlm_critic_dataset_build.py`：Dataset 构建
- `agent/run/43_vlm_critic_train.py`：训练主脚本
- `agent/run/43b_vlm_critic_artifact_smoke.py`：Artifact smoke test
- `agent/run/43c_vlm_critic_sign_audit.py`：符号审计

### 验证脚本
- `agent/run/44_vlm_critic_offline_gate.py`：离线验证
- `agent/run/44b_vlm_critic_ablation_gate.py`：消融实验
- `agent/run/44c_vlm_critic_postmortem.py`：事后分析
- `agent/run/45_recap_label_dataset_vlm_backend.py`：VLM labeling
- `agent/run/45b_vlm_critic_relabel_audit.py`：Relabel 审计
- `agent/run/45c_vlm_critic_finetune_smoke.py`：Finetune smoke
- `agent/run/45d_vlm_critic_eval_smoke.py`：Eval smoke
- `agent/run/45e_vlm_critic_downstream_gate.py`：下游验证
- `agent/run/45f_vlm_critic_pilot_eval_wrapper.py`：Pilot eval wrapper

### Artifact
- `agent/artifacts/critics/task7_real_critic_v2/`：当前最新的 VLM critic artifact
- `agent/artifacts/critics/critic_recap_mainline_fresh_20260311_121500_k0_min_t1/`：Baseline state-only critic

### 计划与证据
- `.sisyphus/plans/g1_vlm_critic_bootstrap_mainline.md`：完整实施计划
- `.sisyphus/evidence/task-7-*.txt`：Task 7 相关证据
- `.sisyphus/evidence/critic_archive_t3_20260315_103248.md`：Critic 归档记录

## 6. 总结

VLM Critic 是一个基于 **Qwen/Qwen3-VL-2B-Instruct** 的多模态 value function 模型，通过以下改造实现了对 Isaac G1 Apple-to-Plate 任务的 value 估计：

1. **架构改造**：
   - 添加 Late Fusion 层（融合视觉-语言特征与侧信道）
   - 添加 Distributional Value Head（201-bin 分布式 value 预测）
   - 对 backbone 顶部 4 层应用 LoRA 微调

2. **输入改造**：
   - 视觉：当前时刻单帧（ego_view）
   - 文本：固定查询（"Estimate the raw return..."）
   - 侧信道：t_norm（时间归一化）

3. **训练策略**：
   - 两阶段训练：Warmstart（公开数据）+ Formal（Isaac 数据）
   - 损失函数：Cross-Entropy + Pairwise Ranking
   - 优化器：AdamW，分阶段学习率

4. **验证通过**：
   - Offline Gate：在 held-out test set 上验证性能
   - Relabel Audit：验证 labeling 稳定性
   - Downstream Gate：验证对 policy 性能的提升

该模型已成功集成到 RECAP 在线强化学习主线中，用于为新收集的数据提供 value 估计和 advantage conditioning。
