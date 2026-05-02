# agent/ 协作工作区

本目录是“可复现交接”的协作工作区。约定：代码与内部推理可用英文；凡是需要让用户/协作者直接阅读的文档，一律使用中文。

硬性规则：
- `agent/logs/` 只允许放 `*.md`（协作日志）。
- 程序 stdout/stderr、安装日志、评测日志等一律放到 `agent/runtime_logs/`。
- 产物（视频、checkpoint 归档、导出文件、图表等）一律放到 `agent/artifacts/`。
- “权威事实/已验证交换信息”写到 `agent/exchange/`；“仓库级工程/代码标准合同”写到 `agent/contract/`；不要长期散落在日志里。

子目录：
- `agent/exchange/`：I/O 契约与已验证事实（唯一权威）。
- `agent/contract/`：工程合同与代码实现标准（唯一权威）。
- `agent/prompts/`：无记忆（stateless）agent 的阶段化 prompts。
- `agent/tutorials/`：面向学习者的教程文档（结构化解释仓库原理/代码路径；允许“解释性内容”，但避免把未经验证的事实写进 `agent/exchange/`）。
- `agent/run/`：从项目根目录可一键执行的 public CLI / thin wrapper 封装；它是保留的运行边界，不是新增业务实现的默认承载层。
- `agent/logs/`：每次行动的 Markdown 协作日志（命令摘要 + 关键输出摘录 + 结论 + 产物路径）。
- `agent/runtime_logs/`：程序运行日志（按子系统分目录）。
- `agent/artifacts/`：产物归档（videos/checkpoints/exports/plots/zips）。

补充约定：
- 新增业务实现、共享 helper、canonical script logic 默认放在 `work/**`。
- `agent/run/*` 若继续存在，应优先充当公开入口、兼容旧路径的薄封装，真实实现应下沉到 `work/demo_utils/**`、`work/recap/**` 或其它 `work/**` 子目录。
- `agent/exchange/**` 继续作为 I/O 契约、已验证事实与迁移真相的权威位置。
- `agent/contract/**` 继续作为工程合同、代码落层规则与 future code 默认准入标准的权威位置。
- 当前 repo/worktree 位于 `/home/howard/Projects/gr00t_wbc_g1_benchmark`（本地 NVMe 工作区）；`agent/artifacts/` 是 live / nearline 产物区，不是长期冷存储。
- 暂时不用但需要保留证据链的大体积**非代码**数据，统一冷存储到 `/media/howard/DATA/Projects/gr00t_wbc_g1_benchmark_archives/`。
