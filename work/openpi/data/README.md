# work/openpi/data/

## 做什么

- 提供 OpenPI 侧的数据 contract 映射与字段对齐辅助。
- 当前主文件是 `contract_mapping.py`，用于把仓库内的结构化数据 surface 对齐到 OpenPI 适配层可消费的形状。

## 不做什么

- 不负责 checkpoint 解析。
- 不负责 runtime bridge、rollout 执行或指标聚合。

## 在 baseline -> RECAP best 主链里的位置

- 这是偏底层的 contract helper，供上层 data / dataloader / recap 相关流程复用。

## 推荐阅读顺序

1. `contract_mapping.py`

## 关键入口、关键 scenario、关键类

- 关键入口：`contract_mapping.py`
- 本包没有独立 scenario。

## internal-only / compatibility-only

- 当前没有公开 CLI；默认作为 `work/openpi/**` 的内部复用层使用。
