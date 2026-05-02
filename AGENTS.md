# AGENTS.md — GR00T-WholeBodyControl（Unitree G1）本地复现协作指南

本工程目标：在本机复现 **GR00T-WholeBodyControl** 的 **Unitree G1 loco-manipulation benchmark**（MuJoCo 路线），并把“可复现的命令、版本、I/O 契约、产物证据”固化到本仓库。
新增目标（Learning Track）：
- 在不破坏“复现优先”主线的前提下，探索并落地一种 **在线自我改进（online RL loop）** 的训练回环（优先参考 Physical Intelligence 的 RECAP / advantage conditioning 思路），为后续把 GR00T 从“纯离线训练/离线微调”扩展到“可迭代部署-改进”提供工程骨架与证据链。
语言约定（硬性）：
- 代码、变量名、内部推理过程：可以使用英文。
- 与用户交流、以及需要让用户/协作者直接阅读的文档：必须使用中文（本仓库的 `*.md` 默认都按此要求编写）。

## 仓库根目录迁移说明
- 当前整个 repo/worktree 的 **canonical project root** 已更新为：`/home/howard/Projects/gr00t_wbc_g1_benchmark`。
- 2026-04-20 里程碑：当前工作站环境已从旧的 **RTX 5090 / Blackwell bring-up 机器** 正式切换到新的大型工作站；现场探针当前读到的是 **4× `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`（每卡约 97887 MiB）**。后续凡涉及精确 GPU 型号/显存，一律以实际 `nvidia-smi` / torch probe 为准。
- 当前 live workspace **不再**位于 `/media/howard/Data/Projects/gr00t_wbc_g1_benchmark`；旧 `/media/howard/Data/...` root 只应视为历史引用，除非当场重新验证，否则不得再把它当作 live root。
- 当前**live 输出根目录**（唯一）现定为：`/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_live/agent/`。后续默认读写的 `runtime_logs/` 与 `artifacts/` 应直接落到这个 HDD live root；repo 内的 `agent/runtime_logs` 与 `agent/artifacts` 是 canonical entrypoints，但允许通过 symlink / 等效映射回链到该 HDD live root。
- 冷存储 / 大体积归档根目录（唯一）现定为：`/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_archives/`。凡是“暂时不用、但仍需保留证据链”的大体积数据（checkpoint、videos、lerobot/recap datasets、导出包、其它非代码产物），默认归档到这里；**代码和脚本不进入该冷存储根目录**。
- 历史日志、prompts、evidence 中若出现旧的 `/media/howard/Data/Projects/gr00t_wbc_g1_benchmark` 绝对路径，除非当场重新验证，否则一律视为历史引用，不是当前 live root。
- 历史日志、evidence、manifest 中若出现旧的 `/media/howard/Data/DataFromUbuntu/gr00t_wbc_g1_benchmark_archives/`，也一律视为**历史错误归档路径**；除非当场重新验证，否则不得再把它当作 live cold-storage root。
- 历史日志、evidence、manifest 中若出现旧的 `/media/howard/HDD/DataFromUbuntu/gr00t_wbc_g1_benchmark_archives/`，也一律视为**上一版 cold-storage root** 的历史引用；除非当场重新验证，否则不得再把它当作当前 live cold-storage root。
- 历史日志、evidence、manifest 中若出现 `agent/runtime_logs_nvme_legacy_*` / `agent/artifacts_nvme_legacy_*`，一律视为 **HDD live-root 切换后的 NVMe 过渡备份**；除非当场明确在做历史核对或分批迁移，否则不得继续把这些 legacy roots 当作新输出落点。
- 相关里程碑与根因证据：`agent/logs/rtx_pro_6000_root_migration_20260420_222840.md`、`.sisyphus/evidence/task-rtx-pro-6000-root-migration-20260420_222840.md`、`agent/logs/cold_storage_root_switch_20260427_235719.md`、`.sisyphus/evidence/cold_storage_root_switch_20260427_235719.md`、`agent/logs/hdd_live_output_switch_20260428_101953.md`、`.sisyphus/evidence/hdd_live_output_switch_20260428_101953.md`。

## 为什么是 MuJoCo
上游 `NVIDIA/Isaac-GR00T` 对 `UNITREE_G1` 的公开 benchmark 明确走的是 **MuJoCo + Whole-Body Control**。本仓库只复现这条公开路线，避免混入其它仿真栈。

## 目录结构
- `agent/`：协作与交付工作区（可复现交接的唯一入口）
- `submodules/`：第三方源码（如 `Isaac-GR00T`）
- `work/`：本项目自定义 glue / 配置 / 脚本
`agent/` 细分：
- `agent/contract/`：工程合同与代码实现标准（唯一权威）
- `agent/exchange/`：I/O 契约与已验证事实（唯一权威）
- `agent/prompts/`：无记忆 agent 的阶段化 prompts（stateless）
- `agent/run/`：一键脚本（从项目根目录执行）
- `agent/logs/`：Markdown-only 协作日志（每次行动必须写）
- `agent/runtime_logs/`：程序 stdout/stderr 日志入口（setup/server/client/eval 等）；它是 **repo 内 canonical entrypoint**，但当前默认应通过 symlink / 等效映射写到 HDD live root 下的 `runtime_logs/`。
- `agent/artifacts/`：产物入口（视频、checkpoint、导出文件、图表、压缩包）；它是 **repo 内 canonical entrypoint**，但当前默认应通过 symlink / 等效映射写到 HDD live root 下的 `artifacts/`。

## 工作流（复现优先）
- [ ] 创建协作日志：`agent/logs/<topic>_YYYYMMDD_HHMMSS.md`
- [ ] 先跑最小 smoke，并将 stdout/stderr tee 到 `agent/runtime_logs/`（该 canonical 路径当前默认应回链到 HDD live root）
- [ ] 保存产物到 `agent/artifacts/`（该 canonical 路径当前默认应回链到 HDD live root），并在日志里记录路径
- [ ] 把已验证事实写回 `agent/exchange/`
- [ ] 把复用实现、业务逻辑与 canonical script logic 固化到 `work/**`，只把对外运行入口/薄封装保留在 `agent/run/*.sh` 或 `agent/run/*.py`

## 交付代码约定（所有执行 agent 必须遵守）
本仓库不仅要“跑通”，还要把流程沉淀成可复现、可维护的脚本与可复用代码。任何新增/修改代码都必须遵守以下约定。

### 0) 通用编程实现与可读性合同（强制）
- 所有 agent 在新增/修改 live code、调整目录落点、保留兼容层、设计 workflow/service、拆分巨型方法或新增 public CLI 前，必须先阅读并遵守：`agent/contract/script_workflow_layering_contract.md`
- 该合同直接约束 **所有 live code** 的实现标准、可读性要求、角色边界、依赖方向、复用规则、README 导航规则与迁移 DoD；不得把它误读成“只约束脚本分层”的旧合同。
- 本节只保留仓库级摘要；对于 `scripts/`、`script_apps/`、`agent/run/`、`work/**` 的具体边界、层间依赖方向、反模式、方法拆分触发器与 DoD，一律以该合同为单一权威；若本文件与合同冲突，以合同为准。
- 如需快速自检，可参考：`agent/contract/general_programming_readability_checklist.md`；若与正式合同冲突，以正式合同为准。

### 1) 落点与脚本交付形态（默认）
- 新增业务实现默认进入：`work/**`
- `agent/run/` 是保留的 **public CLI / thin wrapper** 边界，不再是默认实现层
- 交付优先级：
  - 业务实现、可复用 helper、canonical script logic：放在 `work/**`（如 `work/demo_utils/**`、`work/recap/**` 或其它按职责命名的子目录）
  - 对外运行入口、兼容旧路径的一键封装：放在 `agent/run/*.py` / `agent/run/*.sh`
- 若保留 `agent/run/<script>.py` 作为公开入口，单文件运行方式仍可为：`python agent/run/<script>.py`
- `agent/run/*` 必须保持薄封装：参数解析、USER Config、环境探测、调用 `work/**` 的真实实现；不要把新增业务逻辑直接堆进这里
- 默认不自动退出（便于 demo）：
  - 但必须提供 `--exit-after-s` / `--exit-after-episodes` 等保险丝用于 agent 自测与 CI

### 2) 代码复用与维护（禁止复制粘贴）
新增业务实现、共享 helper 与可复用脚本逻辑默认放在 `work/**`。其中，`agent/run/*.py` 中重复出现的通用逻辑必须抽到：`work/demo_utils/`，包括但不限于：
- repo_root 定位、目录创建（runtime logs / artifacts）
- stdout/stderr tee 到日志文件
- policy server 生命周期（spawn/ping/kill/端口检测）
- env registry 列举
- 视频目录归档（例如从 `/tmp/sim_eval_videos_*` 复制到 `agent/artifacts/videos/`）
- Ctrl+C / SIGTERM 清理逻辑
规则：
- 新增业务实现不得默认落到 `agent/run/`；应优先写入 `work/**`
- 新增 public CLI 若需要保留在 `agent/run/`，应做成 thin wrapper，并 `import work.demo_utils.*` 或其它 `work/**` 模块
- demo_utils 模块顶层只 import 标准库；第三方依赖（gymnasium/mujoco/numpy/robosuite 等）必须 runtime import
- 对 demo_utils 的新增功能要：
  - 写清楚模块职责（建议补充到 `work/demo_utils/README.md`）
  - 提供最小 smoke（例如 `python -c "import work.demo_utils"`）并把输出落盘到 `agent/runtime_logs/`

### 3) 只读边界
- `submodules/**` 视为第三方只读：不随手修 bug、不打补丁；若必须修改，必须先在协作日志说明理由与风险，并把修改最小化。
- I/O 契约与已验证事实以 `agent/exchange/*` 为权威：只写已验证事实，TBD 必须带验证计划。
- 工程合同、代码落层规则与 future code 默认准入标准以 `agent/contract/*` 为权威。

### 4) 证据化与归档（不可省略）
- 任何可视化 demo 必须产出：
  - runtime logs：`agent/runtime_logs/<topic>/`
  - 视频归档：`agent/artifacts/videos/`
- 不允许把关键证据留在 `/tmp`：必须复制归档并在协作日志写明路径

### 5) 大文件治理与外置盘归档（强制）
本仓库的首要目标是可复现，不是长期存储。当前 live workspace 位于 `/home/howard/Projects/gr00t_wbc_g1_benchmark`（本地 NVMe 工作区），但大体积 live 输出默认应直接写入 HDD live root `/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_live/agent/`（通过 repo 内 `agent/runtime_logs` / `agent/artifacts` canonical entrypoints 回链）；任何会快速膨胀的产物都必须遵守以下规则，避免本机 NVMe 被 checkpoint 和视频等撑爆。
强制规则（必须同时满足）：
- 外置盘归档根目录（唯一）：`/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_archives/`
- HDD live 输出根目录（唯一）：`/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_live/agent/`
- 后续默认 live 写路径：
  - `agent/runtime_logs/` → HDD live root 下的 `runtime_logs/`
  - `agent/artifacts/` → HDD live root 下的 `artifacts/`
- `agent/runtime_logs` / `agent/artifacts` 可以是 symlink / 等效映射；future code 应把它们视为 canonical entrypoints，而不是强假设它们必须是 NVMe 上的真实目录 inode。
- 该根目录只用于**大体积非代码数据**的冷存储；代码、脚本、patch、配置源文件、README/契约文档都应继续留在 repo/worktree，而不是搬进冷存储。
- 大文件目录范围（仓库内，默认视为“必须治理”）：
  - `agent/artifacts/checkpoints/`
  - `agent/artifacts/videos/`
  - `agent/artifacts/lerobot_datasets/`
  - `agent/artifacts/recap_datasets/`（如存在）
- 默认本机保留策略（超过就归档到外置盘，然后删除本机副本）：
  - checkpoints：仅保留最近 2 次 finetune 的运行目录，每次 finetune 目录内只保留 1 个 checkpoint
  - videos：仅保留最近 3 个 topic 的视频目录，其余全部归档
  - datasets（lerobot/recap）：仅保留 1 份“当前正在使用”的版本，其余全部归档
- Checkpoint 约束（写进所有训练脚本/配置，必须落地执行）：
  - 必须配置 `save_total_limit=1`（或等价机制），保证每次 finetune 最终只留下 1 个 checkpoint（best 或 last 二选一）
  - 发现历史遗留的多 checkpoint 堆积时，先归档，再做 prune，最后删除本机多余 checkpoint
- `/tmp` 规则（长期禁用）：任何写到 `/tmp` 的视频或其它大文件都只能作为临时中转，必须在当次行动结束前复制到 `agent/artifacts/` 并按本节策略归档，禁止把 `/tmp` 当作长期存储
- 归档与删除的证据要求（缺一不可）：
  - 任何 move/copy/delete 都必须记录到 `agent/logs/*.md`
  - 同一次清理还必须新增一份证据到 `.sisyphus/evidence/*.md`，写清楚源路径、目标路径、执行命令、体积变化、校验方式
清理操作 DoD（完成定义，全部勾完才算完成一次清理）：
- [ ] 在 `agent/logs/*.md` 写明本次清理原因、涉及目录、预期释放空间
- [ ] 把数据复制到外置盘归档根目录下的对应子目录（建议按 `<topic>_YYYYMMDD_HHMMSS` 命名，避免覆盖）
- [ ] 校验复制成功（至少对比 `du -sh` 与文件数量，必要时补充 checksum 记录）
- [ ] 在 `.sisyphus/evidence/*.md` 记录源和目标路径、复制校验结果、删除前后 `du -sh` 的体积变化
- [ ] 删除本机已归档副本，只保留本节规定的“最近 N”

## 模型/硬件“永远卡死点”（强约束，先看这里）
有些问题不是“再试几次就好”，而是**配置不满足就必然失败**。遇到下列情况，必须先修复环境/依赖，再继续跑 prompt：
1) **[历史 bring-up 专项] RTX 5090 / Blackwell（`sm_120`）+ PyTorch wheel 不含 `sm_120`**
- 现象：运行时 warning 提示 “`sm_120 is not compatible`”，随后报 `CUDA error: no kernel image is available for execution on the device`。
- 结论：这属于 **旧 5090 / Blackwell 工作站** 的 bring-up 问题，不应自动假设会在当前 `RTX PRO 6000 Blackwell Max-Q` 工作站上复现；是否仍处于 `sm_120` 风险域，必须以当前 live probe 为准。
- 处理：仅当你明确在重放旧 5090 证据链时，才需要切换到包含 `sm_120` 的 PyTorch CUDA 变体（通常是 `cu128` 或更高 / nightly），并用探针确认 `torch.cuda.get_arch_list()` 包含 `sm_120`。
- 参考执行：`agent/archive/prompts/11b_blackwell_sm120_pytorch.md`
2) **GR00T-N1.6（G1）远程 checkpoint 依赖 FlashAttention2（`flash-attn`）**
- 现象：server 启动或推理时报 `AssertionError: Qwen3 must use flash_attention_2 but got sdpa`。
- 结论：这是模型/remote code 的强约束；`flash-attn` 不可用会直接触发断言退出。
- 处理：必须让 `transformers.utils.is_flash_attn_2_available()` 返回 `True`，且 `flash_attn_2_cuda` 可 import；若 torch 变更，`flash-attn` 必须重装/重编译（否则常见 ABI/symbol 错误）。
- 参考执行：`agent/archive/prompts/11a_smoke_eval_fix.md`
3) **repo `.venv` / WBC `.venv` 断链到缺失的 uv base interpreter**
- 现象：目录里仍能看到 `.venv/bin/python`、`.venv/bin/python3`，但它们实际是 broken symlink，指向缺失的 `/home/howard/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10`。
- 结论：这是 2026-04-20 root/workstation 迁移后已经实锤出现过的 **环境层 blocker**；不是业务逻辑 bug，也不是“缺少激活 venv”这么简单。
- 处理：先修复 uv base interpreter / venv 入口完整性，再继续任何 repo `.venv`、WBC `.venv`、stage3 delegate runtime、OpenPI runtime 的依赖安装与 smoke。
- 证据：`agent/logs/rtx_pro_6000_root_migration_20260420_222840.md`、`.sisyphus/evidence/task-rtx-pro-6000-root-migration-20260420_222840.md`。
4) **脚本“吞错继续跑”会让你后面永远撞墙**
- 典型：安装 `flash-attn` 失败仍继续跑，后续只会以更隐蔽的方式崩（例如 SDPA fallback -> 断言）。
- 规则：关键依赖（torch CUDA 变体 / flash-attn / nvcc）失败必须在当步停下，先修复，再继续。
5) **长等待/轮询必须带硬超时（避免永远挂死）**
- 典型：等待 server ready/ping、rollout 没有 timeout，CI/远程执行会永远卡住。
- 规则：所有“等待/长任务”都必须使用 `timeout` 保险丝；且注意 `timeout ... env VAR=... cmd` 的写法，避免 `timeout` 把环境变量当成可执行文件。
- 参考实现：`agent/run/11b_sm120_run_once.sh`

## 硬件资源检查与并行加速（重要）
- 开跑前先检查 CPU、内存、GPU 显存、磁盘空间，并确认没有重复进程或重复日志在抢资源。
- 并行只在任务彼此独立时开启；有全局依赖时先并行产出分片，再串行汇总；无全局依赖时也要有文件锁保护。
- 长任务必须支持断点续跑，进度文件用原子更新，启动时跳过已完成的 episodes。

### GPU 选择默认策略（全局约束）
- 默认优先只使用 **GPU1 / GPU2**；若单卡即可满足任务，则**优先只使用 GPU1**。
- 除非用户**明确要求**复现 4 卡启动面、或明确要求覆盖 GPU0 / GPU3 的特定场景，否则**不要主动使用 GPU0 / GPU3**。
- 如需偏离本默认策略（例如必须复现 `num_gpus=4` 的真实启动面），必须先在协作日志或证据中写明：
  - 为什么不能只用 GPU1 / GPU2；
  - 当次使用了哪些 GPU（例如 `CUDA_VISIBLE_DEVICES` / `num_gpus`）；
  - 启动前的 `nvidia-smi` 占用快照与理由。
- 进行任何新的 GPU 任务前，默认先记录一次 `nvidia-smi --query-gpu=...` 与 `--query-compute-apps=...`，优先选择空闲且不与现有任务冲突的 GPU。

### 启动前检查清单
- [ ] 无重复进程或重复日志写入
- [ ] 支持断点续跑，进度文件原子更新
- [ ] 有文件锁或等价并行协调
- [ ] 磁盘空间充足（≥50GB）
- [ ] GPU 显存充足（单进程≥8GB）
- [ ] 并行度合理（建议 2-4 进程/GPU）
- [ ] 若不是明确的 4 卡/指定卡复现实验，本次 GPU 选择已优先限制在 GPU1 / GPU2，且单卡优先 GPU1
- [ ] 若使用了 GPU0 / GPU3，已在协作日志或证据中写明原因、目标 GPU 集合与启动前占用快照

## 已知问题（待后续排查）
详见 `agent/archive/AGENTS_archived_sections.md` 的“已知问题”归档段落。

## 权限与 sudo Gate（重要）
在实际复现过程中，少量步骤可能需要管理员权限（例如：`apt-get install ...`、安装系统级 CUDA Toolkit、安装 EGL/GLU 依赖、写入 `/usr/local` 等）。
当遇到“必须 sudo/必须 root 才能继续”的步骤时，协作规范如下：
1) **不要让 agent 直接执行 sudo**
- 默认假设执行 agent 没有 sudo 权限或不可交互输入密码。
- agent 不应反复尝试绕过（例如改用 runfile、自行下载解压到奇怪路径）来“曲线救国”，除非这是明确的、可复现且更安全的方案。
2) **必须生成可复现的一键脚本，交给用户手动执行**
- 脚本位置：`agent/run/<topic>_sudo.sh`
- 脚本要求：
  - `set -euxo pipefail`
  - 打印关键版本信息与环境变量
  - 所有输出必须 `tee` 到 `agent/runtime_logs/<topic>/` 下的日志文件
  - 脚本内显式使用 `sudo ...`（必要时提示会要求输入密码）
- 协作日志里必须记录：
  - 为什么需要 sudo
  - 脚本路径
  - 预期修改点（安装哪些包、写入哪些目录）
  - 手动执行的 DoD（用哪些命令验证成功）
3) **暂停并等待用户执行**
- agent 在生成脚本后必须暂停，提示用户：
  - 在项目根目录运行：`bash agent/run/<topic>_sudo.sh`
  - 执行完成后把日志路径（`agent/runtime_logs/<topic>/*.log`）回传给 agent
4) **用户执行完成后，agent 才恢复后续流程**
- agent 以日志为证据继续推进下一步。

## 权威契约（必须维护）
本项目建议至少维护两份契约：
- `agent/exchange/gr00t_policy_io.md`：GR00T policy server 的输入/输出（observation/action 的 key、shape、单位、horizon、relative/absolute 规则）
- `agent/exchange/wbc_env_io.md`：WholeBodyControl/MuJoCo 环境接口（Gym env 名称、action space、关节组、单位、控制频率）

## Prompts 体系
详见 `agent/archive/AGENTS_archived_sections.md` 的“Prompts 体系”归档段落。
