# work/（本项目自定义代码与配置）

本目录用于存放本项目自己的 glue 代码、配置快照、辅助脚本与新增业务实现；按照当前 live placement contract，`work/**` 是默认实现层。

约定：
- 尽量不把第三方大仓库直接改在 `submodules/` 里；需要修改时优先以 patch/说明记录在 `work/` 与 `agent/logs/`。
- 用户/协作者可读文档使用中文；代码与注释可使用英文。
- 新增业务实现、共享 helper、canonical script logic 默认写在 `work/**`，而不是 `agent/run/**`。
- 若需要保留对外运行入口，应让 `agent/run/*` 只承担 public CLI / thin wrapper 角色，真实实现下沉到 `work/demo_utils/**`、`work/recap/**` 或其它 `work/**` 子目录。

建议结构：
- `work/demo_utils/`：可复用的脚本工程化能力（路径、日志 tee、server 生命周期、视频归档等共享逻辑）。
- `work/recap/`：RECAP / online RL loop / exporter / critic 等本项目自定义实现；其中共享库模块保留在包根/子包，脚本型入口实现优先放在 `work/recap/scripts/`。
- `work/scripts/`：本项目脚本（如：一键拉起 server/client、日志归档、环境自检）。
- `work/configs/`：实验与评测配置快照（参数、环境变量、版本信息）。
- `work/notes/`：杂项记录（调研、对齐表、已知坑）。
