# work/openpi/metrics/

## 做什么

- 提供 bootstrap、metric ladder、pairwise delta、gate 相关指标构建逻辑。
- 为 rollout eval 与 repaired gate verdict 提供稳定、机器可读的汇总输出。

## 不做什么

- 不启动 runtime。
- 不解析 checkpoint。
- 不承载 RECAP iteration orchestration。

## 在 baseline -> RECAP best 主链里的位置

- 位于 rollout trace 之后、summary/blocker verdict 之前，是主链里把 episode 结果聚合成 headline/gate 结论的指标层。

## 推荐阅读顺序

1. `ladder.py`
2. `pairwise.py`
3. `bootstrap.py`
4. `gating.py`

## 关键入口、关键 scenario、关键类

- 关键类：`MetricLadderBuilder`、`PairwiseDeltaBuilder`
- 关键底层实现：`gating.py`

## internal-only / compatibility-only

- `__init__.py` 是导出面；实际 builder 定义点在 `ladder.py`、`pairwise.py`。
