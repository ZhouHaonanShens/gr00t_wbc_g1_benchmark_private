# work/recap/script_apps/

本目录承接 `work/recap/scripts/*.py` 对应的 **class-based 脚本实现入口**。对 state-conditioned 主线而言，这里现在主要是兼容 surface / alias shim，而不是重业务实现层。

本目录的实现与阅读约定，以 `agent/contract/script_workflow_layering_contract.md`（通用编程实现与可读性合同）为总则；下面只补充当前 app surface / 兼容面的具体布局。

目标：

- 让 `work/recap/scripts/*.py` 只保留 thin wrapper / public CLI 边界
- 让脚本级高层流程通过类方法暴露
- 继续复用 `work/recap/*.py` 下已有的共享 kernel，而不是在 wrapper 里重复实现

当前约定：

- 一个活跃脚本，对应一个 `*_app.py`
- `*_app.py` 中至少提供一个 `*ScriptApp` 类；对于较新的重构路径，真实 workflow/service 可以进一步下沉到 `work/recap` 下的非 script_app 模块
- wrapper 默认通过 `ScriptApp().run()` 或同级 `materialize_*()` 方法调用实现
- 裸 `main()` / `materialize_*()` 只应作为 public function surface，真实控制流尽量下沉到 workflow/service 类

这里描述的是当前兼容与演进规则，不是要求 future code 必须继续复制 `*_app.py + ScriptApp.run()` 这种形状；未来新增代码仍以合同中的层级职责为准。

进一步说：当真实实现已经下沉到 `work/recap/*.py` 或对应子包后，`script_apps/` 若继续保留，通常只应该承担两件事：

1. 提供一个**可 import 的浅 surface**（尤其当 `scripts/*.py` 文件名不适合直接作为稳定 import 路径时）
2. 保留历史 monkeypatch / alias / public import 兼容面

如果一个脚本没有这两类额外需求，future code 默认不必再人为插入 `script_apps/` 这一层。

兼容性补充：

- 大多数脚本 wrapper 通过 `__getattr__` 公开旧属性导入面。
- 少数仍被旧测试直接 monkeypatch 的 wrapper（目前主要是 `3A/38/39`）保留了定点 helper 同步桥。
- `state_conditioned_snapshot_harvest.py` 由于历史 monkeypatch 面最重，当前采用 import-time alias 到 app module 的方式保持兼容。

当前活跃入口示例：

- `RecapMultiIterLoopScriptApp`
- `RecapOnlineLoopIterateScriptApp`
- `RecapCollectRolloutsScriptApp`
- `RecapLabelDatasetScriptApp`
- `RecapExportLeRobotWithVideoScriptApp`
- `StateConditionedCollectBucketsScriptApp`
- `StateConditionedSnapshotHarvestScriptApp`
- `StateConditionedBuildTrainingSetScriptApp`
- `StateConditionedTrainScriptApp`

当前已落地的 workflow / service 示例：

- `RecapMultiIterLoopWorkflow`
- `RecapOnlineLoopIterateWorkflow`
- `RecapCollectRolloutsWorkflow`
- `RecapLabelDatasetWorkflow`
- `LeRobotVideoExportWorkflow`
- `RecapFinetuneReproWorkflow`
- `RecapFinetuneFullWorkflow`
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

说明：

- `recap_multi_iter_loop_app.py` 当前主要保留 parser、helper compatibility surface 与 `main()` composition；3A 的真实 workflow 已进一步下沉到 `work/recap/multi_iter_loop.py`。
- `recap_collect_rollouts_app.py` / `recap_label_dataset_app.py` / `recap_export_lerobot_with_video_app.py` / `recap_finetune_full_app.py` 当前已收缩为 alias shim；对应真实实现分别在 `work/recap/collect_rollouts.py`、`work/recap/label_dataset.py`、`work/recap/lerobot_export/workflow.py`、`work/recap/finetune_full.py`。
- `state_conditioned_*` 主线中的 workflow/service 分层目前最完整，已经把 precondition/load/build/persist/diff 之类的职责拆开。
- 对应的真实实现现在进一步下沉到了 `work/recap/state_conditioned/` 子包，`script_apps/state_conditioned_*_app.py` 主要保留兼容 surface。
- classic RECAP 主线目前也已经不再是单纯 `ScriptApp.run()->main()` 的壳；其中 `31/32/39/3D` 已进一步收敛到“app shim + canonical package 实现”，而 `3A/38` 等 orchestrator 仍保留更厚的 app/composition surface，后续可继续向更细的 stage/service 分层推进。

阅读顺序建议：

1. 先看 `work/recap/scripts/*.py`，理解公开入口
2. 再看本目录的 `*ScriptApp` 与兼容 surface
3. 如果是 state-conditioned 主线，再看 `work/recap/state_conditioned/*.py` 中的真实 workflow/service
4. 最后看 `work/recap/*.py` 包根共享模块，理解真正的业务 kernel
