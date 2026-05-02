# work/openpi/norm/

## 做什么

- 承载 norm stats 与 policy 侧的 glue 逻辑。
- 当前主文件是 `policy.py`，用于把 norm 相关约束接到 OpenPI 适配层。

## 不做什么

- 不负责训练流程。
- 不负责 rollout 运行时桥接。

## 在 baseline -> RECAP best 主链里的位置

- 位于模型绑定与数据消费之间，是较底层的配置/对齐辅助层。

## 推荐阅读顺序

1. `policy.py`

## 关键入口、关键 scenario、关键类

- 关键入口：`policy.py`
- 本包没有独立 scenario。

## internal-only / compatibility-only

- 默认作为内部 glue 模块被其它包引用，不是 public entry。
