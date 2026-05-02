# work/openpi/overlays/openpi_recap/

## 做什么

- 承载 OpenPI RECAP overlay 的 materialization 与上游桥接逻辑。
- 让仓库内的 RECAP 适配层能以 overlay 形式接到上游 OpenPI 结构，而不把仓库特有代码直接写进第三方源码。

## 不做什么

- 不作为 canonical eval / iteration 的入口。
- 不负责 rollout metrics 与 gate verdict 聚合。

## 在 baseline -> RECAP best 主链里的位置

- 这是桥接层而不是主线入口；当需要把 RECAP 适配结果投影到 overlay/upstream 结构时，会经过这里。

## 推荐阅读顺序

1. `materialize.py`
2. `src/openpi/recap_overlay/`

## 关键入口、关键 scenario、关键类

- 关键入口：`materialize.py`
- 关键子包：`src/openpi/recap_overlay/`

## internal-only / compatibility-only

- 本包偏 bridge/overlay，不建议当作业务主线默认阅读路径；默认主线仍从 `work/openpi/scripts/openpi.py` 进入。
