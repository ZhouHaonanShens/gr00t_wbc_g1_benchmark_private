# work/openpi/dataloader/

## 做什么

- 提供 JSON / JSONL / authority bundle / rollout source 的薄 I/O 读取与写入能力。
- 保持上层 workflow 使用统一的数据落盘与读取 contract。

## 不做什么

- 不解释业务语义。
- 不负责 runtime 验证或 checkpoint 解析。

## 在 baseline -> RECAP best 主链里的位置

- 这是主链的 I/O surface：eval、runtime、recap 都通过这里的 helper/loader 读写结构化产物。

## 推荐阅读顺序

1. `json_store.py`
2. `rollout_source.py`
3. `authority_bundle.py`

## 关键入口、关键 scenario、关键类

- 关键类：`RolloutSourceLoader`
- 关键入口：`json_store.py`

## internal-only / compatibility-only

- `RolloutSourceLoader` 是薄 loader，不承担高层业务校验；语义验证应放在上层 workflow 或 protocol。
