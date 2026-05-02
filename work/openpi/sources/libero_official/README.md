# work/openpi/sources/libero_official/

## 做什么

- 处理 official/native 8D LIBERO source 的校验、relabel 与 join。
- 为 RECAP collect/merge 主线提供受控、可验证的 canonical source 输入。

## 不做什么

- 不运行 rollout。
- 不做最终指标汇总。

## 在 baseline -> RECAP best 主链里的位置

- 它位于 RECAP collect / merge 的输入侧：在把 canonical demo 与 autonomous trials 合并前，先在这里确认官方 source 的可用性和衍生产物。

## 推荐阅读顺序

1. `validate.py`
2. `relabels.py`
3. `join.py`

## 关键入口、关键 scenario、关键类

- 关键入口：`validate.py`
- 典型调用方：`work/openpi/pipelines/recap/collect.py`、`work/openpi/pipelines/recap/merge.py`

## internal-only / compatibility-only

- 这是 source 处理层，不是对外入口；其输出通常被 pipeline 或 dataset aggregation 间接消费。
