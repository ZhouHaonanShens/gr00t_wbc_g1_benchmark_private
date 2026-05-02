# work/openpi/

本目录是当前仓库 **唯一允许承载 OpenPI 适配与主线 workflow 的位置**。本 README 不再重复展开每个包的细节，而是作为总索引，把读者导向各个包级 README。

## 总体职责

- 承载 OpenPI 相关的 live adapter code、canonical eval、runtime bridge、RECAP pipeline 与配套 contract helper。
- 保持上游 `submodules/openpi/` 只读，把仓库特有逻辑固化在 `work/openpi/**`。
- 让读者可以按 `入口 -> scenario -> app/workflow -> 共享实现` 的顺序理解主线，而不是从历史脚本或杂散 helper 逆向。

## 不属于这里的内容

- 不把对用户可读的权威事实长期留在这里；事实与契约请写回 `agent/exchange/**`。
- 不把 public CLI 以外的真实业务实现堆进 `agent/run/**`。
- 不把已退出主线的历史文件重新纳入 live import graph。

## 主线阅读顺序

1. [`scripts/README.md`](scripts/README.md)
2. `scripts/openpi.py`
3. [`eval/README.md`](eval/README.md)
4. `eval/scenarios.py`
5. `eval/app.py`
6. [`pipelines/recap/README.md`](pipelines/recap/README.md)
7. [`recap/README.md`](recap/README.md)
8. [`checkpoint/README.md`](checkpoint/README.md)
9. [`model/README.md`](model/README.md)
10. [`runtime/README.md`](runtime/README.md)
11. [`dataloader/README.md`](dataloader/README.md)
12. [`metrics/README.md`](metrics/README.md)

## 包级 README 索引

- [`data/README.md`](data/README.md)
- [`norm/README.md`](norm/README.md)
- [`prompting/README.md`](prompting/README.md)
- [`serve/README.md`](serve/README.md)
- [`checkpoint/README.md`](checkpoint/README.md)
- [`model/README.md`](model/README.md)
- [`dataloader/README.md`](dataloader/README.md)
- [`runtime/README.md`](runtime/README.md)
- [`metrics/README.md`](metrics/README.md)
- [`eval/README.md`](eval/README.md)
- [`recap/README.md`](recap/README.md)
- [`state_tokens/README.md`](state_tokens/README.md)
- [`sources/libero_official/README.md`](sources/libero_official/README.md)
- [`overlays/openpi_recap/README.md`](overlays/openpi_recap/README.md)
- [`pipelines/recap/README.md`](pipelines/recap/README.md)
- [`scripts/README.md`](scripts/README.md)

## 仍然重要的边界约束

- 上游只读依赖预期路径：`submodules/openpi/`
- 本目录是仓库内唯一 OpenPI adapter package root
- `agent/run/**` 不承载 OpenPI 业务逻辑
- `pipelines/recap/*.py` 默认只做 facade；真实 collect / iteration workflow 在 `*_workflow.py`

## 历史材料

- 已退出主线的历史 OpenPI 脚本、版本化 eval surface、报告/图表类入口，统一归档到 `agent/archive/openpi/`
- 历史文件不再是 live import graph 的一部分；需要追溯旧行为时再去 archive 看
