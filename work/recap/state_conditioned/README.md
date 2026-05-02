# work/recap/state_conditioned/

本目录承接 state-conditioned 主线的**真实实现层**。

本目录的实现与阅读约定，以 `agent/contract/script_workflow_layering_contract.md`（通用编程实现与可读性合同）为总则；下面只补充 state-conditioned 主线当前的真实实现层布局。

目标：

- 让 `work/recap/script_apps/state_conditioned_*_app.py` 保持薄 public surface
- 让真实业务流程、workflow/service、阶段装配逻辑落在可复用模块里
- 让 `work/recap/scripts/state_conditioned_*` 继续作为稳定公开入口，而不是业务堆积点

当前模块：

- `collect_buckets.py`：T7 Bucket B / C 状态条件化采集 workflow
- `snapshot_harvest.py`：T8/T9 snapshot feasibility 与 formal pseudo-demo harvest workflow
- `build_training_set.py`：T10 训练集构建、审计与 LeRobot training dataset materialization
- `training.py`：T11 C0 / C1 训练变体调度、run metadata 与 diff whitelist

说明：

- 这里已经承接了主要的 workflow/service 逻辑。
- 但为了保持当前 public surface 和测试兼容，这些模块里仍保留少量 `build_parser()` / `main()` / `*ScriptApp` 相关入口，不应把它们误读成“完全没有脚本 surface”。
- 这些残留 surface 属于过渡兼容面，不是 future code 应复制的默认模板。

推荐阅读顺序：

1. `work/recap/scripts/state_conditioned_*.py`：看公开入口
2. `work/recap/script_apps/state_conditioned_*_app.py`：看薄 surface / 兼容层
3. 本目录对应模块：看真实 workflow/service
4. `work/recap/*.py` 与 `work/recap/scripts/*common*.py`：看共享 kernel 与 helper
