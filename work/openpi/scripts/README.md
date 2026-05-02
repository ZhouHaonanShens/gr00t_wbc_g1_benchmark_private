# work/openpi/scripts/

## 做什么

- 保留 OpenPI 主线的公开脚本入口。
- 当前推荐入口是 `openpi.py`，通过显式 `ACTIVE_WORKFLOW` / `ACTIVE_SCENARIO` 选择默认主线路径。

## 不做什么

- 不作为真实业务实现层。
- 不在这里堆大量 workflow、checkpoint 或 runtime 细节。

## 在 baseline -> RECAP best 主链里的位置

- 这是主线入口层：读者从这里进入，再跳转到 scenario、app、workflow 与 pipeline。

## 推荐阅读顺序

1. `openpi.py`

## 关键入口、关键 scenario、关键类

- 关键入口：`openpi.py`
- 推荐默认路径：`Scenario -> App -> Workflow`，而不是先逆向一个字符串 CLI router。

## internal-only / compatibility-only

- `scripts/` 只保留公开入口与薄封装；真实实现应继续下沉到 `work/openpi/**` 其它包。
- 若必须保留旧的 `work.openpi.scripts.<legacy_name>` 导入路径，应优先提供**独立薄 shim 文件**，并在文件头明确写出真实实现位置；不要把兼容逻辑堆进 `__init__.py` 做 eager import。
