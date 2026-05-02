# AGENTS.md — GR00T-WholeBodyControl（Unitree G1）本地复现协作指南

本仓库的权威协作入口，分三层：**L0** 任何 agent 行动前必读；**L1** 提交 PR / 结构变动前必读；**L2** 历史与按需查阅。

## L0 — 任何 agent 行动前必读（环境状态变量 + 不可降级红线）

- **项目身份**：复现 GR00T-WholeBodyControl 的 Unitree G1 loco-manipulation MuJoCo benchmark；新增 RECAP-style online RL 学习回环（详见 L1）。
- **协作语言**：与用户/文档面交流必须中文；代码、变量、内部推理可英文。
- **canonical project root**：`/home/howard/Projects/gr00t_wbc_g1_benchmark`（NVMe 工作区）。
- **HDD live root**：`/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_live/agent/`。
- **重要事实**：`agent/runtime_logs` 与 `agent/artifacts` **是 symlink，不是 NVMe 真实目录**——所有写入透明落 HDD，不要按 NVMe 容量规划。
- **archives root**（冷存储，仅大体积非代码数据）：`/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_archives/`。
- **硬件**：4× NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition（每卡 ~97887 MiB）。精确数值以 `nvidia-smi` 为准。
- **GPU 默认策略**：优先只用 **GPU1 / GPU2**；单卡优先 GPU1；不主动用 GPU0/3，除非用户明确要求 4 卡复现。
- **agent 不能直接 `sudo`**：必须生成 `agent/run/<topic>_sudo.sh` 交用户手动执行（流程见 L1）。
- **`/tmp` 不是长期存储**：写到 /tmp 的视频/大文件必须当次复制到 `agent/artifacts/`。
- **`submodules/**` 视为只读**；必要修改前先在协作日志说明。
- **长任务必须带 `timeout` 保险丝**；关键依赖失败必须当步停下。
- **L0 漂移自检**：`bash agent/run/check_l0_drift.sh`（探针实际值 vs L0 声明值，仅警告不阻塞）。

## L1 — 提交 PR / 结构变动前必读（工作流与协议）

### 工作流（复现优先）
1. 创建协作日志：`agent/logs/<topic>_YYYYMMDD_HHMMSS.md`
2. 跑最小 smoke，stdout/stderr tee 到 `agent/runtime_logs/`
3. 产物保存到 `agent/artifacts/`，日志中记录路径
4. 已验证事实写回 `agent/exchange/`
5. 复用实现/业务逻辑沉到 `work/**`；`agent/run/*` 仅留薄封装

### 目录结构
- `agent/contract/`：工程合同与代码实现标准（唯一权威）
- `agent/exchange/`：I/O 契约与已验证事实（唯一权威）
- `agent/prompts/`：阶段化 prompts（索引：`agent/prompts/INDEX.md`）
- `agent/run/`：保留为 public CLI / thin wrapper 边界
- `agent/logs/`：Markdown 协作日志（每次行动必写）
- `agent/runtime_logs/` / `agent/artifacts/`：canonical entrypoint，symlink 到 HDD live root
- `submodules/`：第三方只读；`work/`：自研业务实现默认落点

### 代码交付约定（摘要；权威以 `agent/contract/script_workflow_layering_contract.md` 为准）
- 业务实现/共享 helper/canonical script logic 默认进 `work/**`；`agent/run/*` 仅作 thin wrapper（参数解析、USER Config、环境探测、调用 `work/**`）
- 通用复用抽到 `work/demo_utils/`，禁止在多个 `agent/run/*` 间复制粘贴
- 反模式、依赖方向、方法拆分触发器、DoD 全部以合同为单一权威；自检清单：`agent/contract/general_programming_readability_checklist.md`

### 大文件治理（摘要；权威同上）
- 保留策略：checkpoints 最近 2 个 finetune × 1 ckpt（`save_total_limit=1`）；videos 最近 3 个 topic；datasets 仅 1 份在用。范围：`agent/artifacts/{checkpoints,videos,lerobot_datasets,recap_datasets}/`
- 任何 move/copy/delete 必须双证据：协作日志 + `.sisyphus/evidence/*.md`（源/目标路径、命令、体积变化、校验）。大体积非代码数据归 archives root；代码不进 archives

### sudo gate（agent 永不直接 sudo）
1. agent 生成 `agent/run/<topic>_sudo.sh`（`set -euxo pipefail`、tee 到 `agent/runtime_logs/<topic>/`、显式 `sudo ...`）
2. 协作日志记录：为何需 sudo、脚本路径、预期修改点、手动 DoD
3. agent **暂停**，提示用户 `bash agent/run/<topic>_sudo.sh`
4. 用户执行后回传日志路径
5. agent 以日志为证据继续

### 永远卡死点（仍现役）
- **#2 flash-attn**：GR00T-N1.6 (G1) 远程 ckpt 强约束 `flash_attention_2`；缺失会触发 `Qwen3 must use flash_attention_2 but got sdpa` 断言。`transformers.utils.is_flash_attn_2_available()` 必须为 True；torch 变更后必须重装/重编译 `flash-attn`。
- **#4 不许吞错继续**：关键依赖（torch CUDA 变体 / flash-attn / nvcc）失败必须当步停下，先修复再继续。
- **#5 长等待/轮询必须带硬超时**：用 `timeout`；注意 `timeout ... env VAR=... cmd` 写法避免把 env 当可执行文件。参考 `agent/run/11b_sm120_run_once.sh`。

### 启动前检查清单
- [ ] 无重复进程 / 重复日志；磁盘 ≥50 GB；GPU 显存 ≥8 GB；支持断点续跑（进度文件原子更新 + 文件锁或等价并行协调）
- [ ] GPU 限制 GPU1/2（单卡优先 GPU1）；用 GPU0/3 已在协作日志写明原因；已记录启动前 `nvidia-smi --query-gpu=...` / `--query-compute-apps=...` 快照

### 权威契约入口
- `agent/exchange/gr00t_policy_io.md`：GR00T policy server I/O（observation/action key、shape、单位、horizon、相对/绝对规则）
- `agent/exchange/wbc_env_io.md`：WBC/MuJoCo 环境 I/O（Gym env 名、action space、关节组、单位、控制频率）
- `agent/exchange/repo_placement_contract.md`：仓库落点契约
- `agent/exchange/agent_run_wrapper_allowlist.md`：`agent/run/*` 包装层白名单
- `agent/exchange/strict_run_entrypoint_matrix.md`：严格运行入口矩阵
- Prompts 索引：`agent/prompts/INDEX.md`（注意：该索引内部仍含 `agent/archive/prompts/**` 历史链接，待另案修复，不影响其作为索引入口的作用）

## L2 — 历史与按需查阅

### 路径迁移历史（一律视为历史引用，当场未重新验证不得当作 live root）

| 旧路径 / 旧根 | 类别 |
|---|---|
| `/media/howard/Data/Projects/gr00t_wbc_g1_benchmark` | 旧 live workspace |
| `/media/howard/Data/DataFromUbuntu/gr00t_wbc_g1_benchmark_archives/` | 旧错误归档 root |
| `/media/howard/HDD/DataFromUbuntu/gr00t_wbc_g1_benchmark_archives/` | 上一版 cold-storage root |
| `agent/runtime_logs_nvme_legacy_*` / `agent/artifacts_nvme_legacy_*` | NVMe→HDD 切换后的 NVMe 过渡备份，不再作新输出落点 |

**Legacy NVMe 副本现状（截至 2026-05-02）**：`agent/runtime_logs_nvme_legacy_20260428_101953` (~511 GB) 与 `agent/artifacts_nvme_legacy_20260428_101953` (~835 GB) 仍在本地 NVMe；ongoing 归档到 archives root，未全量删除。

### 永远卡死点（历史归档）
- **#1 RTX 5090 / `sm_120` PyTorch wheel 不兼容**：仅在重放旧 5090 bring-up 证据链时适用；当前 RTX PRO 6000 工作站默认不命中。处理：切到含 `sm_120` 的 PyTorch CUDA 变体并用 `torch.cuda.get_arch_list()` 探针确认。
- **#3 uv venv 断链（2026-04-20 一次性事故）**：`.venv/bin/python` 是 broken symlink，指向缺失的 uv base interpreter；遇到时先修复 uv interpreter 再做任何依赖安装。

### 已知问题
目前无积压。

### 关键证据日志（指针）
- 工作站迁移：`agent/logs/rtx_pro_6000_root_migration_20260420_222840.md`、`.sisyphus/evidence/task-rtx-pro-6000-root-migration-20260420_222840.md`
- 冷存储 root 切换：`agent/logs/cold_storage_root_switch_20260427_235719.md`、`.sisyphus/evidence/cold_storage_root_switch_20260427_235719.md`
- HDD live 输出切换：`agent/logs/hdd_live_output_switch_20260428_101953.md`、`.sisyphus/evidence/hdd_live_output_switch_20260428_101953.md`

### sudo gate 完整流程（reference，超出 L1 摘要的细节）
- 默认假设 agent 无 sudo / 不可交互输入密码；不应反复尝试绕过（自行 runfile、写入奇怪路径等）。
- 脚本要求：`set -euxo pipefail`、打印关键版本与环境变量、所有输出 tee 到 `agent/runtime_logs/<topic>/`、显式 `sudo ...`。
- 协作日志必写：为何需 sudo、脚本路径、预期修改点（装哪些包 / 写哪些目录）、手动 DoD（用哪些命令验证成功）。
- 用户执行 `bash agent/run/<topic>_sudo.sh` 后将 `agent/runtime_logs/<topic>/*.log` 路径回传给 agent，agent 才以日志为证据推进。
