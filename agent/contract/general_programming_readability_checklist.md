# 通用编程可读性合同：Agent 执行清单

这是 `agent/contract/script_workflow_layering_contract.md` 的执行摘要版。  
用途：给 agent 或 code review 当作快速检查表使用。

---

## 1. 写代码前先判断角色

新代码必须先判断自己属于哪一类：

- Public Surface：入口、CLI、API handler、脚本包装
- Compatibility / Composition：兼容层、facade、composition root
- Workflow / Service：真实业务流程、可复用业务职责
- Shared Kernel / Helper：低层复用能力
- Data Contract / Domain Model：配置、请求、结果、实体、值对象

规则：

- 默认业务逻辑不进 surface / compatibility
- 默认真实实现进 workflow / service
- 默认共享能力进 kernel
- 稳定边界数据必须形成 data contract

---

## 2. 顶层 workflow 的黄金形态

顶层 `run()` / `execute()` / `process()` 只做：

1. 表达阶段顺序
2. 调用阶段方法或服务
3. 做少量顶层分支
4. 做边界级错误转换
5. 汇总结果

顶层 workflow 不应直接展开大体量实现细节。

合格示例：

```python
def run(self) -> Result:
    plan = self._build_plan()
    batch = self._load_inputs(plan)
    normalized = self._normalize(batch)
    output = self._write_outputs(normalized)
    return self._build_result(output)
```

---

## 3. 方法必须立刻拆分的信号

出现以下任一情况，立即拆分：

- 一个方法里已经有多个语义阶段
- 需要注释分段才能读
- 同时做 orchestration 和实现细节
- 同时做计算和 I/O
- 同时做正常路径和复杂异常恢复
- 同时处理多种模式分支
- 读者需要滚很多屏才能理解
- 无法用一句话概括方法职责
- 一部分逻辑已在别处重复出现
- 方法名必须用多个动词串起来才说得清

---

## 4. 类必须立刻拆分的信号

出现以下任一情况，类应继续拆：

- 类名无法覆盖大多数方法
- 类内部有多个不相关子职责
- 类同时做 workflow、loader、writer、validator
- 类主要价值只是给旧长函数包了个壳
- 类持有很多互不相关字段
- 改这个类时很难预测影响范围

---

## 5. 明确禁止

以下直接视为坏味道：

- 薄 surface + 巨型 `run()`
- `Workflow.run()` 只是旧 `main()` 改名
- `App.run()` 唯一职责是 `return main()`
- compatibility 层偷偷藏主流程
- 用注释分段的大方法长期存在
- `Helper / Util / Common / Manager` 成为默认容器
- 新代码复制旧兼容层形状
- 用巨型 dict / kwargs 传所有边界
- 多个布尔参数控制互斥模式
- 低层 helper 反向依赖入口层

---

## 6. 数据边界规则

- 稳定输入输出必须有明确类型
- 强关联参数要收束为对象
- 复杂返回值不要用魔法 tuple
- 不要让共享可变 dict 在多阶段漂移
- 配置、运行中间态、最终结果应尽量分开

---

## 7. 命名规则

优先使用：

- `Workflow`
- `UseCase`
- `Service`
- `Loader`
- `Runner`
- `Writer`
- `Builder`
- `Parser`
- `Validator`
- `Result`
- `Config`

谨慎或避免默认使用：

- `Manager`
- `Helper`
- `Util`
- `Common`
- `Misc`

---

## 8. 注释规则

注释只解释：

- 为什么这样设计
- 不变量
- 边界条件
- 兼容原因
- 协议限制

不要用注释替代结构拆分。  
如果注释在给代码分阶段，阶段就应该变成方法或服务。

---

## 9. 测试规则

- surface：smoke test
- compatibility：兼容测试
- workflow：流程测试
- service / kernel：单元测试
- data contract：schema / serialization / validation test

目标：

- 核心逻辑可单测
- workflow 可通过少量集成测试验证编排
- 不要只能从最顶层做大而脆的测试

---

## 10. README 规则

README 必须明确：

- 入口在哪
- 真实实现在哪
- 兼容层为什么存在
- 新代码默认该落哪类模块

不能只写入口，不写真实实现位置。

---

## 11. 提交前 10 问

提交前必须能回答：

1. 这段代码属于哪个角色层？
2. 为什么它不该在 surface？
3. workflow 顶层是否只表达阶段？
4. 是否还存在隐藏的大方法？
5. 类名 / 方法名是否诚实？
6. 是否有共享逻辑没下沉？
7. 数据边界是否显式？
8. 副作用是否尽量靠边界？
9. README 是否能带人找到真实实现？
10. 现在的代码是否比之前更容易局部理解？

---

## 12. 一句话总纲

**类不是终点；阶段清晰、职责诚实、方法可局部理解，才是终点。**
