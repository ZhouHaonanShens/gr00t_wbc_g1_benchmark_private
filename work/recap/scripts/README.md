# work/recap/scripts/

本目录承载 `work/recap/` 体系下的**脚本型入口实现**：包括 collect / label / export / finetune / eval / VLM critic / multi-iteration orchestration 等可执行模块与 shell 编排脚本。

本目录的实现与阅读约定，以 `agent/contract/script_workflow_layering_contract.md`（通用编程实现与可读性合同）为总则；下面只补充本目录当前的分层与入口布局。

约定：

- `work/recap/scripts/*.py` / `*.sh`：public script / thin wrapper 边界。
- `work/recap/script_apps/*.py`：脚本对应的 class-based public surface / 兼容层；不是新业务实现的默认落点。
- `work/recap/` 包根：保留共享库模块与子包；不再镜像脚本型同名文件。
- `agent/run/*.sh`：仅保留少量仍需公开保留的 shell public CLI；对应的 Python wrapper 已从 live tree 删除。

当前迁移边界：

- 已迁入本目录的主干：RECAP 主流水线（31/32/33/34/34b/38/39/3A/3D）、VLM critic（41-46d）、state-conditioned / interface-localization / gr00t gates-probes-scorecards，以及相关 shell public CLI（34/36/37/38/3C/3E）。
- `work/recap/` 根目录中仍保留的非脚本模块（如 `collector.py`、`dataset_reader.py`、`labeler.py`）属于共享实现层；其中一部分历史大模块会继续收敛到子包，并只在根目录保留兼容 shim。

迁移目标：

1. 让 `work/recap/` 包根回到“库模块 + 子包”的角色。
2. 让脚本公开入口集中到 `work/recap/scripts/`，减少根目录噪声。
3. 让 `agent/run/` 与 `work/recap/` 的边界清晰：前者只剩极少量必要入口，后者承载实际实现。

## 当前 class-based 入口约定

当前活跃的 RECAP / state-conditioned 主线，已经切成两层：

1. `work/recap/scripts/*.py`
   - 只保留 public script surface
   - 负责 repo root 注入、兼容旧导入面、实例化 `ScriptApp`
2. `work/recap/script_apps/*.py`
   - 承接脚本对应的 class-based public surface / 兼容层
   - 对于 state-conditioned 主线，真实实现已进一步下沉到 `work/recap/state_conditioned/*.py`

这意味着：

- 如果你在看“脚本怎么被调用”，先看 `work/recap/scripts/*.py`
- 如果你在看“脚本入口 class 和兼容 surface 在哪里”，看 `work/recap/script_apps/*.py`
- 如果你在看 state-conditioned 主线的真实 workflow/service，在 `work/recap/state_conditioned/*.py`
- 如果你在看“共享业务 kernel 在哪里”，继续看 `work/recap/*.py` 包根与子包；对于 LeRobot export 主线，优先看 `work/recap/lerobot_export/`

补充说明：

- 对已经把真实实现下沉到 `work/recap/*.py` 或 `work/recap/<subpkg>/*.py` 的脚本来说，`script_apps/` 不再是额外的“实现层”；它若继续存在，主要只是提供一个**可 import 的浅 surface / 兼容名**。
- 尤其像 `3A_recap_multi_iter_loop.py` 这类文件名本身不是理想的 import surface 时，`work/recap/script_apps/recap_multi_iter_loop_app.py` 目前更多是在承担“可 import 的薄入口”角色，而不是承载真实 orchestrator。
- 因此 future code 默认不必机械复制 `scripts/ + script_apps/ + real workflow` 三层；若 `scripts/*.py` 已足够充当公开入口，且没有额外 import-compat 需求，可以直接采用“薄 script surface + 真实实现层”两层结构。

兼容性说明：

- `3A_recap_multi_iter_loop.py` / `38_recap_online_loop_iterate.py` / `39_recap_export_lerobot_v2_with_video.py` 为了兼容现有测试中的 wrapper-level monkeypatch，保留了少量 wrapper→app 的 helper 同步桥；该兼容面默认以调用 wrapper `main()` 为边界。
- `state_conditioned_snapshot_harvest.py` 是一个特例：它在 import 时会别名到对应的 `script_apps` 实现层，以保持其历史上大量内部 helper monkeypatch 的兼容性。

## 当前建议优先阅读的活跃流水线

### 经典 RECAP 主流水线

如果你想看 collect → label → export → train / eval 的经典 RECAP 主链，优先看：

- `31_recap_collect_rollouts.py`：采集 / 评测 rollouts
- `32_recap_label_dataset.py`：对 M1 结果做 M2 label
- `39_recap_export_lerobot_v2_with_video.py`：当前更常用的导出入口（带视频）
- `33_recap_export_lerobot_v2_dataset.py`：较早的基础导出入口
- `3D_recap_finetune_full.py` / `34_recap_finetune_repro.py`：训练包装层
- `3A_recap_multi_iter_loop.py`：当前更完整的多轮 online loop 编排入口
- `38_recap_online_loop_iterate.py`：较早的单轮 online loop 编排入口

对应的 class-based public surface 在：

- `work/recap/script_apps/recap_collect_rollouts_app.py`
- `work/recap/script_apps/recap_label_dataset_app.py`
- `work/recap/script_apps/recap_export_lerobot_with_video_app.py`
- `work/recap/script_apps/recap_finetune_full_app.py`
- `work/recap/script_apps/recap_finetune_repro_app.py`
- `work/recap/script_apps/recap_multi_iter_loop_app.py`
- `work/recap/script_apps/recap_online_loop_iterate_app.py`

其中 3A 主线当前的真实实现已进一步下沉到：

- `work/recap/multi_iter_loop.py`：承接 `RecapMultiIterLoopWorkflow` 与多轮 orchestrator 的真实实现

这一轮继续下沉后的经典 RECAP 真实实现还包括：

- `work/recap/collect_rollouts.py`：承接 `RecapCollectRolloutsWorkflow` 的真实实现
- `work/recap/label_dataset.py`：承接 `RecapLabelDatasetWorkflow` 的真实实现
- `work/recap/lerobot_export/workflow.py`：承接 `LeRobotVideoExportWorkflow` 的真实实现
- `work/recap/lerobot_export/dataset_export.py`：承接 `export_recap_to_lerobot_v2` 与 LeRobot v2 schema/export contract
- `work/recap/lerobot_export/video_export.py`：承接视频附着与 `export_recap_to_lerobot_v2_with_video`
- `work/recap/finetune_full.py`：承接 `RecapFinetuneFullWorkflow` 的真实实现

其中已经落地的真实 workflow 例子包括：

- `RecapCollectRolloutsWorkflow`
- `RecapLabelDatasetWorkflow`
- `LeRobotVideoExportWorkflow`
- `RecapFinetuneReproWorkflow`
- `RecapFinetuneFullWorkflow`
- `RecapMultiIterLoopWorkflow`
- `RecapOnlineLoopIterateWorkflow`

对应的 shell 包装主要是：

- `3C_launch_multi_iter.sh`：`3A` 的薄启动器
- `3E_quick_validation.sh`：`3A` 的 quick validation 包装
- `38_recap_smoke_end_to_end.sh`：端到端 smoke wrapper

### Iter5.5 authority ingest

- `iter5p5_w0p5_ingest_authorization.py`：iter5.5 W0p5 启动授权 ingest 入口；真实实现位于 `work/recap/iter5p5_authority.py`，负责把 `.omc/iter5p5_launch_authorization.md` 与 iter5 carry-forward 证据转成 coordinator/verifier 机器可读产物。

### state-conditioned 主流水线

如果你想看 state 相关的数据收集、构建训练集与训练主线，优先看：

- `state_conditioned_collect_buckets.py`：T7，补齐 Bucket B / C 的状态条件化采集
- `state_conditioned_snapshot_harvest.py`：T8/T9，做 snapshot feasibility 与 pseudo-demo harvest
- `state_conditioned_build_training_set.py`：T10，把 Bucket A / B / pseudo-demo 汇总成 state-conditioned 训练集
- `state_conditioned_train.py`：T11，基于 T10 训练集启动 C0 / C1 两个训练变体

对应的 class-based 入口在：

- `work/recap/script_apps/state_conditioned_collect_buckets_app.py`
- `work/recap/script_apps/state_conditioned_snapshot_harvest_app.py`
- `work/recap/script_apps/state_conditioned_build_training_set_app.py`
- `work/recap/script_apps/state_conditioned_train_app.py`

对应的真实实现模块现在在：

- `work/recap/state_conditioned/collect_buckets.py`
- `work/recap/state_conditioned/snapshot_harvest.py`
- `work/recap/state_conditioned/build_training_set.py`
- `work/recap/state_conditioned/training.py`

其中已经落地的真实 workflow / service 例子包括：

- `StateConditionedBucketCollectionWorkflow`
- `SnapshotHarvestWorkflow`
- `StateConditionedTrainingSetWorkflow`
- `StateConditionedTrainingWorkflow`
- `TrainingSetContractLoader`
- `VariantTrainingRunner`
- `DiffWhitelistManager`
- `TrainingSetPrerequisiteLoader`
- `TrainingRowBuilder`
- `TrainingArtifactWriter`
- `BucketPreconditionLoader`
- `BucketPlanBuilder`
- `BucketManifestBuilder`

### 这一轮整理后的共用 helper / workflow

为了降低入口脚本的样板噪声，本目录现在额外有两类共用 helper：

- `recap_orchestration_common.py`：给 `3A/38` 这类 RECAP orchestrator 复用 repo/path/json/git/checkpoint/port helper
- `state_conditioned_common.py`：给 state-conditioned 脚本复用路径校验、JSON/JSONL I/O 和错误消息 helper

与此同时，state-conditioned 主线的真实业务流程已经进一步下沉到 `work/recap/state_conditioned/*.py`，而 `script_apps` 保持为薄 surface。

当前完成度上：

- `state_conditioned_*` 主线已经形成更明确的 workflow/service 分层；
- classic RECAP 主线已经完成 workflow 命名化与主流程收拢，但部分模块仍主要是较大的 `*Workflow.run()`，后续还可以继续细拆 stage/service。
- 其中 `31/32/39/3D` 的真实实现现已从 `script_apps/` 退出并下沉到 `work/recap/*.py`；对应 `script_apps/*_app.py` 主要只保留 alias / import surface 角色。
- 其中 `31_recap_collect_rollouts.py` 与 `32_recap_label_dataset.py` 的 core workflow 现已进入“第二轮可读性治理”：`run()` 顶层主要保留阶段骨架，细节继续下沉到私有 stage 方法；后续若再拆，会优先抽出更独立的 service，而不是把逻辑搬回 surface。

这样阅读主脚本时，可以更快聚焦到 collect / label / train 业务流程本身，而不是重复的运行时样板。
