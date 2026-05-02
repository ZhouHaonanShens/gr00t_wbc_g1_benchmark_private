# submodules/（第三方依赖源码）

本项目会把第三方依赖放到 `submodules/`，以便：
- 版本固定（commit/tag 可记录在日志与契约里）
- 复现时路径稳定

本仓库的 git 策略（重要）：
- `submodules/**` 默认视为“第三方只读工作区”，通常不纳入本仓库的 git 跟踪（避免误提交大体积第三方源码）。
- 第三方版本 pin 请写入 `agent/exchange/*`（例如 `agent/exchange/gr00t_policy_io.md` / `agent/exchange/wbc_env_io.md`），并在 `agent/logs/**` 给出证据路径。
- 获取第三方源码的可复现流程以 `agent/prompts/10_setup_repo_and_env.md`、保留的 `agent/run/**` public CLI / thin wrapper 入口，以及 `work/**` 中的真实实现为准。

预计会包含：
- `submodules/Isaac-GR00T/`：上游 NVIDIA Isaac-GR00T。

注意：
- 第三方仓库可能使用 Git LFS（例如 G1 网格）。务必在安装/拉取阶段记录是否执行了 `git lfs pull`。
- 任何“必须能复现”的命令，最终都应把真实实现固化到 `work/**`，并只在需要公开入口或兼容旧路径时保留 `agent/run/` 薄封装；已验证事实与迁移真相继续写入 `agent/exchange/`。
