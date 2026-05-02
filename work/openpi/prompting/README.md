# work/openpi/prompting/

## 做什么

- 集中维护 prompt/token 路由常量与主线可复用的 prompting surface。
- 当前主文件是 `routes.py`。

## 不做什么

- 不直接运行 runtime episode。
- 不负责 recap dataset 物化与训练导出。

## 在 baseline -> RECAP best 主链里的位置

- 它为 model/runtime/recap 提供统一的 prompt route 标识，是主线 contract 的底层组成部分。

## 推荐阅读顺序

1. `routes.py`

## 关键入口、关键 scenario、关键类

- 关键入口：`routes.py`
- 本包没有独立 scenario。

## internal-only / compatibility-only

- 本包是内部 contract 层；推荐通过上游 workflow 或 builder 间接消费，而不是在入口层直接拼字符串常量。
