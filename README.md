# GR00T WBC G1 Benchmark — 代码评审快速入口

本 README **不是**面向普通用户的营销/安装文档；它的目标是让新的评审者或执行者在进入本仓库后，能在最短时间内建立**正确的审查上下文**。

## 1. 仓库目的（先建立正确问题定义）

- 主目标：在本机复现 **GR00T-WholeBodyControl / Unitree G1 loco-manipulation benchmark** 的 **MuJoCo + Whole-Body Control** 路线。
- 扩展目标：在不破坏复现主线的前提下，探索 **RECAP / online RL loop / advantage conditioning** 相关工程骨架。
- canonical root：`/home/howard/Projects/gr00t_wbc_g1_benchmark`
- 2026-04-20 里程碑：当前工作站已从旧的 `RTX 5090 / Blackwell` bring-up 机器切换到新的大型工作站；现场 probe 当前读到的是 **4× `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`（每卡约 97887 MiB）**，仓内精确型号以 live probe 为准。
- 旧的 `/media/howard/Data/Projects/gr00t_wbc_g1_benchmark` 现在默认视为**历史 root**，不要继续把它当作 live workspace。
- 冷存储根目录：`/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_archives/`（仅存放大体积非代码归档数据）
- 历史文档中若出现 `/media/howard/Data/Projects/gr00t_wbc_g1_benchmark`，默认视为历史引用，**不要**把它当作 live root。
- 历史文档/日志中若出现 `/media/howard/Data/DataFromUbuntu/gr00t_wbc_g1_benchmark_archives/`，默认视为旧的错误 SSD 归档路径，**不要**继续使用。
- 历史文档/日志中若出现 `/media/howard/HDD/DataFromUbuntu/gr00t_wbc_g1_benchmark_archives/`，默认视为上一版 cold-storage root 的历史引用，**不要**继续把它当作当前 live 归档目标。

## 2. 先读这些文件（按优先级）

1. `AGENTS.md`
   - 仓库级最高优先级约束：目录职责、证据落盘、只读边界、硬件/依赖 blocker、sudo gate。
2. `agent/README.md`
   - `agent/` 目录职责；日志、runtime logs、artifacts、exchange 的边界。
3. `work/README.md`
   - `work/` 是本仓库自定义 glue/config/script 层，不是第三方源码区。
4. `submodules/README.md`
   - `submodules/**` 默认第三方只读，评审时不要随手修改。
5. `agent/prompts/INDEX.md`
   - 活跃 prompt 入口图；需要理解当前工作流时先看这里。
6. `agent/exchange/gr00t_policy_io.md`
   - policy server 输入/输出契约，审查 obs/action 变更前必须先看。
7. `agent/exchange/wbc_env_io.md`
   - WBC env 名称、action space、timebase 等权威契约。
8. `agent/prompts/02_git_repo_versioning.md`
   - 提交、ignore、clean repo、文档同步的卫生规则。

## 3. 目录地图（审查时先分层，而不是一把抓）

### `agent/`

- `agent/exchange/`：权威契约与已验证事实。
- `agent/prompts/`：stateless agent prompts。
- `agent/run/`：项目级一键运行入口与保留的 public CLI / thin wrapper 边界；不是新增业务实现的默认落点。
- `agent/logs/`：Markdown-only 协作日志；是**应追踪**内容。
- `agent/runtime_logs/`：程序 stdout/stderr；是**运行证据**，默认不进版本库。
- `agent/artifacts/`：视频/checkpoint/export/plots；默认不进版本库。

### `work/`

- 本项目自定义 glue / helper / data processing / exporter / tooling，也是新增业务实现的默认承载层（`work/**`）。
- 若 `agent/run/*.py` 出现可复用逻辑，应优先抽到 `work/demo_utils/`；RECAP / online RL loop 等真实实现优先放在 `work/recap/` 或其它 `work/**` 子目录。

### `submodules/`

- 第三方上游源码区（例如 `Isaac-GR00T`）。
- 默认只读；除非有明确理由和证据链，否则不要把变更打进 `submodules/**`。

## 4. Critical constraints（这些规则优先级最高）

### 4.1 证据优先

- 任何运行/验证必须把 stdout/stderr 落到 `agent/runtime_logs/`。
- 任何视频/checkpoint/export 必须归档到 `agent/artifacts/`。
- 任何“已验证事实”应写回 `agent/exchange/`，不要只留在日志里。

### 4.2 只读边界

- `submodules/**` 默认只读。
- 本仓库自己的新增业务实现默认放在 `work/**`；`agent/run/**` 只保留 public CLI / thin wrapper 等公开运行边界。

### 4.3 常见致命 blocker（不是业务 bug）

- **[历史 bring-up 专项] RTX 5090 / Blackwell (`sm_120`) 与 torch wheel 架构不匹配**
  - 典型症状：`no kernel image is available for execution on the device`
- **GR00T-N1.6 G1 远程 checkpoint 需要 FlashAttention2 / `flash-attn`**
  - 典型症状：`Qwen3 must use flash_attention_2 but got sdpa`
- **root/workstation 迁移后，repo `.venv` / WBC `.venv` 可能断链到缺失的 uv base interpreter**
  - 典型症状：`.venv/bin/python` 看起来存在，但 `file`/执行会显示它是 broken symbolic link
- **等待 server ready / ping / rollout 必须带 timeout**

细节统一以 `AGENTS.md` 为准。

### 4.4 sudo gate

- 需要 sudo 的步骤不要由 agent 直接硬跑。
- 正确做法：生成 `agent/run/<topic>_sudo.sh`，让用户手动执行，再回收日志继续。

## 5. Code review 热点（先看入口，不要直接钻模型内部）

| 审查主题 | 优先入口 |
| --- | --- |
| server 如何启动 / sim wrapper 如何挂接 | `submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py` |
| rollout 如何构造 env / wrapper | `submodules/Isaac-GR00T/gr00t/eval/rollout_policy.py` |
| UNITREE_G1 obs/action/horizon/relative rules | `submodules/Isaac-GR00T/gr00t/configs/data/embodiment_configs.py` |
| policy 输入校验 / strict 报错 | `submodules/Isaac-GR00T/gr00t/policy/gr00t_policy.py` |
| finetune CLI / training config | `submodules/Isaac-GR00T/gr00t/configs/finetune_config.py` / `submodules/Isaac-GR00T/gr00t/experiment/launch_finetune.py` |
| 模型结构 / action head / diffusion | `submodules/Isaac-GR00T/gr00t/model/gr00t_n1d6/` |
| LeRobot 数据读取 / stats | `submodules/Isaac-GR00T/gr00t/data/` |
| WBC action space / timebase | `agent/exchange/wbc_env_io.md` |
| policy server I/O | `agent/exchange/gr00t_policy_io.md` |
| demo / script 工程化复用 | `work/demo_utils/README.md` + `work/demo_utils/*.py` |
| RECAP / critic / exporter 自定义逻辑 | `work/recap/scripts/` + `work/recap/`（共享模块）；对外保留的 shell 入口仅见 `agent/run/*.sh` |

## 6. 当前仓库最重要的“不要猜”契约

### 6.1 Policy I/O

- 先看：`agent/exchange/gr00t_policy_io.md`
- 审查 obs/action 相关 PR 时，先核对：
  - key 集合是否一致
  - shape / dtype 是否一致
  - horizon / relative-vs-absolute 语义是否一致

### 6.2 WBC env I/O

- 先看：`agent/exchange/wbc_env_io.md`
- 审查 env/action/timebase 相关 PR 时，先核对：
  - env id
  - action keys 与 dims
  - 50Hz / 200Hz timebase

## 7. 代码审查时推荐的最小导航路线

1. 先看 `agent/logs/` 是否已经有本轮改动的协作日志。
2. 再看 `agent/exchange/` 是否已有权威契约可约束讨论。
3. 先看 `agent/run/**` 是否仍有保留的 public CLI；若对应 Python wrapper 已清理，则直接看 `work/recap/scripts/**` 或 `work/demo_utils/apps/**`。
4. 若入口脚本里出现重复通用逻辑，去 `work/demo_utils/**` 或 `work/recap/**` 看真实实现/复用层。
5. 只有在确认入口/契约都不够时，才下钻到 `submodules/**`。

## 8. 最小验证命令（给维护者/评审者的，不是给终端新手的）

### 8.1 静态检查

```bash
python3 -m pytest --version
python3 -m pytest tests/recap/test_45d_vlm_critic_eval_smoke.py
```

### 8.2 运行入口优先级

- 优先从 `agent/run/*.py` / `agent/run/*.sh` 进入，但把它们视为 public CLI / wrapper；新增实现默认应在 `work/**`
- 不要优先照抄上游双终端命令
- 长命令必须带 `timeout`

### 8.3 证据落盘原则

- stdout/stderr → `agent/runtime_logs/<topic>/`
- 视频 / checkpoint / exports → `agent/artifacts/`
- 事实结论 → `agent/exchange/`
- 行动摘要 → `agent/logs/*.md`

## 9. Git / review hygiene（避免把仓库搞脏）

- 根 `.gitignore` 是 ignore 边界权威来源。
- `agent/runtime_logs/`、`agent/artifacts/`、`.venv/`、`.sisyphus/`、`submodules/*` 默认不应进入版本库。
- `agent/prompts/**`、`agent/exchange/**`、`agent/run/**`、`work/**`、`tests/**`、`agent/logs/*.md` 默认是应追踪内容。
- 提交前先看：
  - `git status --short`
  - `git diff --stat`
  - `git diff --staged --stat`
- 详细规则看：`agent/prompts/02_git_repo_versioning.md`

## 10. 如果 PR / diff 动到了这些区域，优先警惕什么

| 区域 | 优先警惕 |
| --- | --- |
| `agent/run/*.py` | 是否越界承载新增业务实现；是否复制粘贴了 demo_utils 逻辑；是否缺 timeout / USER Config / 证据落盘 |
| `work/demo_utils/*` | 顶层是否误引入第三方依赖；是否改变既有日志/目录行为 |
| `work/recap/*` | 是否破坏 exporter/contract 对齐；是否混入 analysis-only 字段到 deployable/train payload |
| `agent/exchange/*` | 是否写入未经验证的事实；是否缺验证计划 |
| `submodules/**` | 是否违反只读边界 |
| `.gitignore` | 是否误伤应追踪路径 |

## 11. 推荐继续阅读顺序

1. `AGENTS.md`
2. `agent/README.md`
3. `agent/prompts/INDEX.md`
4. `agent/exchange/gr00t_policy_io.md`
5. `agent/exchange/wbc_env_io.md`
6. `work/demo_utils/README.md`
7. `agent/archive/tutorials/05_code_navigation_playbook.md`
8. `agent/prompts/02_git_repo_versioning.md`

## 12. 一句话总结

审查这个仓库时，**先看契约和入口，再看自定义 glue，最后才碰上游源码**；任何运行相关改动都要同时检查 **证据落盘、timeout、只读边界、git cleanliness**。
