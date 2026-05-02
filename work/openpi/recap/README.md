# work/openpi/recap/

## 做什么

- 承载 RECAP 主线的共享实现：dataset 聚合、runtime prompt、stage checkpoint 物化、indicator/control gate、summary 与训练配置等。
- 为 `pipelines/recap/` 提供可复用的业务实现层，而不是把细节散落在入口脚本里。

## 不做什么

- 不充当 public CLI。
- 不替代 `work/openpi/pipelines/recap/` 的 workflow 编排职责。

## 在 baseline -> RECAP best 主链里的位置

- 这是 RECAP 主链的业务内核层。collect / iteration 等 pipeline 在这里拿到 dataset、checkpoint、prompt、gate、summary 的具体实现。

## 推荐阅读顺序

1. `scenarios.py`
2. `dataset_aggregation.py`
3. `checkpoint.py`
4. `checkpoint_provenance.py`
5. `runtime_prompt.py`
6. `protocol.py`
7. `control_gate.py`
8. `train_config.py`
9. `data_transforms.py`
10. `summary.py`

## 关键入口、关键 scenario、关键类

- 关键 scenario：`scenarios.py`
- 关键实现：`dataset_aggregation.py`、`checkpoint.py`、`real_variant_export.py`
- 典型调用方：`work/openpi/pipelines/recap/*.py`

## internal-only / compatibility-only

- 本包里的模块大多是 pipeline 内部实现层；真正的主线 workflow 入口在 `work/openpi/pipelines/recap/`。
