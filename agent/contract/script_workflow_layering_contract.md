# 通用编程实现与可读性合同

本文件冻结当前与未来代码实现的**单一权威规范**。

它不绑定特定项目、目录、语言或框架；只约束一件事：

> **所有 live code 都必须以“结构诚实、职责清晰、导航可预测、局部易理解”的方式编写。**

本合同既约束：
- 新增代码如何落层、如何组织、如何命名、如何复用
- 存量巨型模块、巨型类、巨型方法如何治理
- public surface / compatibility surface / composition root / workflow / service / kernel 的边界
- tests、README、review、agent 执行与迁移完成标准

后续所有 agent、人工协作者、代码评审者、以及未来新增代码，都必须直接以本合同为准；不得再从旧习惯、历史日志、口头约定、零散 README、prompt 残片或“过去一直这么写”推断实现标准。

---

## 1. 适用范围

本合同适用于：

1. 当前 live tree 中所有手写业务代码
2. 未来新增的所有业务代码、共享逻辑、入口封装、工作流实现、兼容层与工具层
3. 跨语言实现，包括但不限于：
   - Python
   - TypeScript / JavaScript
   - Go
   - Java / Kotlin
   - Rust
   - Shell / Bash
4. 与代码组织直接相关的辅助材料：
   - README
   - 架构说明
   - 协作日志
   - 测试设计
   - 评审标准
   - agent 执行规范

说明：

- 本合同约束的是**角色边界与可读性原则**，不是固定目录模板。
- 每个仓库可以自行把这些角色映射到不同目录，但**目录不是本合同本体**；真正冻结的是职责边界与依赖方向。
- 当仓库 README、历史文档、临时 prompt、旧风格与本合同冲突时，以本合同为准。

---

## 2. 核心目标

本合同冻结以下目标：

1. **public surface 必须薄。**
2. **兼容层与真实实现必须分离。**
3. **主流程必须可分阶段阅读，而不是藏在巨型函数或巨型方法里。**
4. **类不是为了“看起来面向对象”；类必须明确状态、边界或阶段职责。**
5. **复用必须通过真实抽象完成，而不是复制粘贴。**
6. **任何代码单元都必须尽量让读者快速回答：**
   - 它是什么角色？
   - 它做什么？
   - 它依赖谁？
   - 它为什么在这里？
7. **旧兼容面可以保留，但不得继续成为新业务逻辑默认落点。**
8. **future code 默认一次性落在正确职责层级，而不是“先糊进去，未来再拆”。**
9. **可读性优先于伪抽象。**
10. **“表面分层”不算分层；只有真实职责下沉才算分层。**

---

## 3. 最高原则：结构诚实

所有实现必须遵守以下结构诚实原则：

### 3.1 名字必须诚实

- 模块名应真实表达其职责
- 类名应真实表达其职责
- 方法名应真实表达其职责
- 如果一个名字必须依赖 “and / then / with / process_everything / handle_all / utils” 才能成立，通常说明职责已经混杂

### 3.2 位置必须诚实

- 入口就放入口逻辑
- 兼容层就放兼容逻辑
- orchestration 就放 orchestration
- 复用实现就放复用实现
- helper 就做 helper，不得成为隐藏总实现层

### 3.3 边界必须诚实

- public surface 不得伪装成实现层
- compatibility surface 不得伪装成 workflow 层
- workflow 不得伪装成一个换名的 `main()`
- service 不得伪装成“把几千行实现放进类里”
- helper 不得伪装成没有架构责任的垃圾桶

### 3.4 注释不能替代拆分

- 如果一个方法必须靠注释写出“第一步 / 第二步 / 第三步 / 收尾”才能勉强阅读，说明这些阶段应该成为独立方法或独立服务
- 如果一个类必须靠长段说明告诉读者“虽然叫 X，但里面其实做了 Y、Z、W”，说明类边界已经不诚实

---

## 4. 通用角色模型（路径无关）

本合同不冻结目录，但冻结以下**角色层**。

### 4.1 Role A — Public Surface

典型形态：

- CLI 入口
- HTTP / RPC handler surface
- job entrypoint
- script entrypoint
- public import/export surface
- shell wrapper
- framework adapter

职责：

- 接收外部输入
- 做最薄的 bootstrap
- 调用 composition root / app surface / workflow
- 维持外部兼容路径
- 把用户输入翻译为内部调用

允许：

- 参数解析
- repo/root/env bootstrap
- 最小必要的 boundary validation
- 调用下层实现
- 错误码或响应码转换
- 少量入口级日志

禁止：

- 真正业务实现
- 大体量数据处理
- 巨型 validator
- 巨型 artifact 组装
- 主流程 orchestration 落在 surface
- 新共享 helper 默认落在这里

判定标准：

- 读者打开后，应一眼看出“这是入口，不是实现层”。

### 4.2 Role B — Compatibility / Composition Surface

典型形态：

- retained public wrapper
- 兼容旧 import path 的 facade
- composition root
- app surface
- framework glue layer

职责：

- 兼容旧调用方式
- 组装 workflow / service / loader / writer
- 提供薄别名、薄 facade、薄组合入口
- 保持旧 tests / monkeypatch / public import 的兼容性

允许：

- 少量明确的组装逻辑
- 最小必要的 monkeypatch bridge / alias bridge
- 参数对象构造
- 配置解析到 workflow/service

禁止：

- 成为默认重业务逻辑落点
- 承载长期主流程实现
- 藏大量 helper、contract 处理、数据整形和主控制流
- 用 `ScriptApp.run()->main()`、`App.execute()->legacy_main()` 这种形式伪装分层

默认规则：

- compatibility / composition 不是 future business logic 的默认落点。

### 4.3 Role C — Workflow / Service Implementation

典型形态：

- Workflow
- UseCase
- Service
- Loader
- Runner
- Builder
- Writer
- Coordinator
- Domain application module

职责：

- 承接真实主流程
- 表达阶段边界
- 组合 kernel/helper 完成业务链路
- 为多个入口、测试、流程复用提供稳定实现面

要求：

- workflow 必须表达阶段，而不是“把 `main()` 改名成 `run()`”
- service 必须围绕单一可复用职责
- builder / writer / loader / runner 必须名副其实
- 如果保留 `main()` / `build_parser()` / `App.run()` 兼容面，它们只是 surface，不是新的实现默认模式

### 4.4 Role D — Shared Kernel / Helper

典型形态：

- 纯函数
- 计算 kernel
- I/O helper
- validation helper
- parser
- serializer
- format converter
- small reusable utility with clear purpose

职责：

- 提供低层复用能力
- 封装稳定的小粒度能力
- 尽量无高层控制流

禁止：

- 变成新的“大杂烩实现桶”
- 混入特定入口的高层 orchestration
- 反向依赖 surface/composition

### 4.5 Role E — Data Contract / Domain Model

这是旧合同未单独强调、但在通用规范中必须补上的角色。

典型形态：

- Config object
- Params object
- Request / Response object
- Manifest / Record / Result object
- Domain entity
- Value object
- Error type

职责：

- 明确跨层边界的数据形状
- 让方法、服务、工作流之间通过显式结构交互
- 替代“巨型 dict / 巨型 kwargs / 神秘 tuple / 魔法列表”

要求：

- 稳定结构的数据必须有稳定类型
- 多个强关联参数必须收束为对象，而不是长期散落函数签名
- 领域概念必须通过命名类型表达，而不是用原始 map/dict 到处漂

---

## 5. 依赖方向（硬性）

### 5.1 允许方向

1. Public Surface 可以依赖 Compatibility / Composition、Workflow / Service、Kernel、Data Contract
2. Compatibility / Composition 可以依赖 Workflow / Service、Kernel、Data Contract
3. Workflow / Service 可以依赖同层实现、Kernel、Data Contract
4. Kernel 可以依赖 Data Contract 中的低层类型，但不得依赖高层入口
5. Data Contract 不得依赖具体入口实现

### 5.2 禁止方向

1. Kernel 不得反向依赖 Public Surface 或 Compatibility / Composition
2. Workflow / Service 默认不得反向依赖 Public Surface
3. Workflow / Service 默认不得为了兼容偷依赖 Compatibility / Composition
4. Data Contract 不得夹带入口框架语义
5. README / 文档不得把读者引到 surface 误以为那是真实实现层

### 5.3 例外规则

出现历史兼容例外时：

- 必须显式记录原因
- 必须显式记录计划
- 例外只能是薄桥接，不得成为 future template

---

## 6. 代码单元的职责边界

本合同不只约束“层”，也约束**文件、类、方法、代码块**。

### 6.1 文件级边界

一个文件应满足：

1. 有一个主角色
2. 有一个主导航点
3. 新读者能在短时间内理解“为什么这些东西放在一起”

禁止：

- 一个文件既做 CLI，又做 workflow，又做 helper，又做 writer
- 一个文件里堆多个互不紧密相关的大类
- 一个文件承担多个不同变化原因

判定标准：

- 如果你必须先解释很久“虽然都放在这个文件，但其实……”——通常就该拆

### 6.2 类级边界

一个类应满足：

1. 表达一个清晰的状态边界、职责边界或阶段协调边界
2. public method surface 小而清楚
3. 实例字段代表稳定上下文，而不是临时变量仓库
4. 类名可以覆盖大多数方法的真实职责

禁止：

- 仅把旧函数搬进类里
- 类只是给一堆函数套命名空间
- 类依赖太多 collaborators，却没有真实抽象收益
- 类内部方法围绕多个不相关子职责展开
- 类的主要收益只是“现在有 self 了”

### 6.3 方法 / 函数级边界

一个方法 / 函数应满足：

1. 能用一句话准确描述职责
2. 调用者能从名字预测大致行为
3. 读者不需要先通读全文件才知道它做什么
4. 输入输出边界清楚
5. 控制流深度、局部状态量、分支复杂度均保持局部可理解

禁止：

- 同时承担 orchestration、计算、I/O、日志、错误翻译、格式转换多个维度
- 靠长注释人工分段
- 把多个阶段硬塞在一个方法里
- 通过共享可变状态把逻辑隐藏在远处

### 6.4 代码块级边界

即使在单个方法内部，也必须保证：

- 每个代码块有单一目的
- 条件分支表达明确语义
- 嵌套层数低
- guard clause 优先于深层缩进
- 临时变量名表达意图，不做字母谜语

---

## 7. 真正解决“长类里只有一个巨型方法”的规则

这是本合同的重点补强。

### 7.1 核心结论

**“薄 surface + 使用 class”并不自动产生可读性。**

如果一个类内部仍然只有一个几千行的 `run()` / `execute()` / `process()` / `build()` / `handle()`，
那么它本质上仍然是**巨型函数**，只是从模块级移动到了类级。

这类实现**不符合本合同**。

### 7.2 Workflow 方法的允许形态

`run()` / `execute()` / `process()` 这类顶层 workflow 方法，只允许承担以下角色：

1. 定义阶段顺序
2. 调用命名良好的阶段方法
3. 做少量顶层分支调度
4. 做边界级错误转换
5. 做阶段级日志
6. 汇总最终结果

换言之，顶层 workflow 方法应让读者能够快速看见：

- 先做什么
- 再做什么
- 哪些分支会走不同路径
- 结果如何收束

### 7.3 Workflow 顶层方法的禁止形态

以下情况均视为不合格：

1. `run()` 内直接展开几百上千行实现细节
2. `run()` 同时负责参数清洗、数据加载、校验、转换、计算、外部调用、写盘、汇总、报告
3. `run()` 内出现大量注释形如：
   - `# prepare`
   - `# step 1`
   - `# collect`
   - `# label`
   - `# export`
   - `# finalize`
   但这些阶段没有被抽为独立方法或服务
4. `run()` 成为整个类的唯一真实方法，其他方法只是薄 wrapper
5. `run()` 里塞满模式分支、嵌套循环、错误处理、日志、状态写入，导致必须通读全部细节才能知道主流程

### 7.4 方法强制拆分触发器（硬性）

一个方法出现以下任一情况，必须拆分：

1. 可以明确识别出两个及以上语义阶段
2. 需要用注释或空行人为分区来帮助阅读
3. 同时处理 orchestration 与具体业务细节
4. 同时进行数据准备、业务执行、结果收尾
5. 同时做纯计算与 I/O 副作用
6. 同时做正常路径与复杂异常恢复
7. 同时处理多种运行模式，而这些模式具备可命名差异
8. 同时依赖过多局部变量，读者必须在脑中长期缓存上下文
9. 出现明显的“滚屏阅读”问题：读者需要跨多个屏幕才能理解方法整体
10. 读者无法用一句话概括方法职责
11. 方法名必须靠多个动词串接才能准确描述
12. 方法中的一部分逻辑已经在别处出现或即将复用

### 7.5 方法拆分后的目标形态

拆分后通常应形成以下形态之一：

#### 形态 A：顶层 workflow + 多个阶段方法

```python
class ImportWorkflow:
    def run(self) -> ImportResult:
        plan = self._build_plan()
        raw_data = self._load_inputs(plan)
        validated_data = self._validate_inputs(raw_data)
        transformed = self._transform(validated_data)
        persisted = self._persist(transformed)
        return self._build_result(persisted)
```

要求：

- 顶层方法主要表达阶段
- 每个阶段方法职责单一
- 复杂阶段再继续下沉到 service

#### 形态 B：workflow + service

```python
class ImportWorkflow:
    def run(self) -> ImportResult:
        plan = self._planner.build(...)
        batch = self._loader.load(plan)
        normalized = self._normalizer.normalize(batch)
        output = self._writer.write(normalized)
        return output
```

要求：

- workflow 表达 orchestration
- service 承担复用职责
- service 不得只是换名字包旧巨函数

#### 形态 C：strategy / mode handler

如果不同模式差异已经足够大，不应在一个方法中堆满 if/else，而应改成：

- strategy object
- mode-specific handler
- dedicated workflow branch

### 7.6 “一屏原则”与“导航优先”

本合同不把固定行数作为唯一铁律，但冻结两个审查原则：

#### 一屏原则

- 读者默认应能在**一屏或接近一屏**的范围内，看清一个方法的主职责
- 如果一个方法的语义必须跨多个屏幕拼接，通常已经过大

#### 导航优先

- 顶层 workflow 应优先服务“导航”
- 细节应下沉到被命名的方法或服务
- 读者首先应该看见流程骨架，而不是首先被细节淹没

### 7.7 注释分段即拆分信号

出现如下模式时，默认说明应该拆分：

```python
def run(self):
    # validate inputs
    ...
    # load records
    ...
    # convert records
    ...
    # write outputs
    ...
    # summarize
    ...
```

若这些注释表达的是**真实阶段名**，则这些阶段应成为：

- 独立方法
- 或独立 service
- 或独立 workflow 子步骤

注释只能解释**为什么**，不能长期替代结构。

---

## 8. Orchestration 设计规则

### 8.1 顶层只排流程，不做细活

任何 orchestration 单元都必须优先承担：

- 阶段编排
- 依赖组织
- 模式调度
- 结果收束

默认不应承担：

- 复杂数据整形细节
- 复杂校验细节
- 外部接口协议细节
- 大体量文本 / artifact 拼装
- 大段逐元素处理逻辑

### 8.2 阶段必须可命名

一个 workflow 能否成立，取决于其阶段是否可命名。

若你无法给阶段起出稳定、诚实的名字，通常说明：

- 职责边界还不清楚
- 抽象粒度不合适
- 代码仍在按实现顺序堆叠，而不是按语义阶段组织

### 8.3 阶段之间必须通过显式数据交接

禁止：

- 通过随处写入 `self.xxx` 让数据在多个阶段隐式漂移
- 通过共享可变 dict 在多个阶段间偷偷积累状态
- 通过“先调用 A 才能让 B 正常”但接口上完全看不出来的 temporal coupling

要求：

- 阶段输入输出要么显式返回
- 要么通过明确命名的数据对象传递
- 若必须持有 workflow context，也应控制为少量、稳定、可预测字段

### 8.4 复杂分支应外提

当 workflow 中某个模式分支已经形成独立链路时，应优先：

- 提取为 strategy
- 提取为 mode-specific service
- 提取为子 workflow

禁止长期在顶层塞满：

- 多重模式判断
- 多层 if/elif
- 模式特定变量拼接
- 模式特定错误恢复逻辑

---

## 9. 类设计强制标准

### 9.1 何时必须用类

出现以下任一情况，应优先考虑类：

1. 存在明确状态边界
2. 存在多阶段 workflow
3. 存在多个相关操作共享稳定依赖
4. 存在一个完整可复用职责，需要自己的协作者和配置
5. 存在生命周期语义（prepare / execute / finalize）

### 9.2 何时不应强行用类

出现以下情况，保持函数更好：

1. 纯函数
2. 无状态小工具
3. 单一转换逻辑
4. 轻量 parser / formatter
5. 抽成类只会多出 `self`，不会带来更清晰边界

### 9.3 类的字段纪律

类字段必须尽量满足以下规则：

1. 字段代表稳定依赖或稳定上下文
2. 字段数量保持克制
3. 临时变量不要升级为实例字段
4. 阶段中间结果只有在跨多个方法稳定共享时才允许放到实例状态
5. 默认优先显式参数和返回值，而不是把一切都塞进 `self`

### 9.4 类拆分触发器

出现以下情况，类应继续拆分：

1. 类名无法诚实覆盖大多数方法
2. 类的 private method 数量很多，但分属不同子职责
3. 类持有大量互不紧密相关的字段
4. 类同时承担 workflow、loader、writer、validator、formatter
5. 类的单元测试必须覆盖太多互不相关场景
6. 评审者很难判断“改这个类会影响什么”

### 9.5 禁止把类当“长函数容器”

以下设计明确禁止：

```python
class GiantProcessor:
    def run(self):
        # 2000 lines here
        ...
```

若类的主要价值只是“把巨型实现移入类中”，则不符合本合同。

---

## 10. 方法 / 函数设计强制标准

### 10.1 单一语义职责

每个方法 / 函数必须有单一语义中心。
这里的“单一”不是“只能有一行”，而是：

- 能被一句话概括
- 有一个主要变化原因
- 调用者能基于名称预测其作用

### 10.2 优先使用 guard clause

优先：

- 提前返回
- 提前拒绝非法输入
- 提前隔离异常路径

避免：

- 多层嵌套才进入真实逻辑
- 正常路径被包在层层 if 内

### 10.3 控制流层级克制

默认目标：

- 尽量低嵌套
- 尽量少的跨区状态共享
- 尽量把“对每个元素如何处理”抽成独立函数 / 方法
- 尽量把“不同模式如何处理”抽成策略而不是堆分支

### 10.4 参数设计规则

禁止：

- 超长参数列表长期存在
- 多个布尔参数控制模式
- 不透明的 `**kwargs` 贯穿关键业务路径
- 把稳定 schema 长期放在匿名 dict 中漂移

要求：

- 强关联参数收束为对象
- 模式用 enum / strategy / dedicated config 表达
- 关键输入输出必须有明确命名

### 10.5 返回值设计规则

禁止：

- 用魔法 tuple 表达复杂结果
- 返回结构没有命名语义
- 某些字段偶尔有、偶尔没有，但没有显式类型表达

要求：

- 简单结果返回简单值
- 复杂结果返回命名对象
- 错误、警告、统计、产物等需要稳定结构时，应形成明确 result type

### 10.6 方法中的循环规则

当循环体开始膨胀时，应优先提取：

- 单元素处理函数
- 单批次处理服务
- 可复用转换器

禁止：

- 在一个大循环里同时做校验、转换、统计、写盘、错误恢复、日志拼装

### 10.7 布尔旗标爆炸禁止

如果一个方法已经需要多个布尔参数来控制模式差异，通常说明应改为：

- 专门 config object
- strategy
- 拆分方法
- 拆分 workflow

---

## 11. 数据契约与状态管理

### 11.1 边界数据必须显式

跨层边界的输入输出必须尽量显式表达：

- 类型
- 字段
- 默认值
- 可空性
- 约束

### 11.2 稳定结构必须类型化

以下情况应优先使用显式对象，而不是原始 dict：

1. 配置项稳定存在
2. 返回结果有多个语义字段
3. 阶段间交接的数据结构稳定
4. 测试需要针对字段做断言
5. 文档需要解释字段语义

### 11.3 状态必须最小化

要求：

- 副作用尽量靠边界
- 状态尽量局部
- 共享状态尽量少
- 可变状态尽量收敛在清晰的 owner 中

禁止：

- 任意方法都可改写共享状态
- 通过对象字段在远距离传播隐式依赖
- “先调用 A 再调用 B 才会把某字段填好”的隐式协议

### 11.4 配置与运行态分离

- 配置对象表达静态参数
- 运行结果对象表达阶段产物
- 不要把配置、运行中间态、最终结果全部揉进一个万能对象

---

## 12. 复用与抽象规则

### 12.1 复用必须发生在正确层

- 入口层的重复，不应通过复制入口逻辑解决，而应下沉到 workflow/service
- 工作流间共享逻辑，应进入 service/kernel
- 数据格式、I/O、校验等共享能力，应进入 helper/kernel

### 12.2 第二次重复就是抽象信号

- 同样逻辑出现第二次时，应主动评估抽象
- 但抽象必须围绕真实语义，不得为了消灭重复而发明模糊基类

### 12.3 抽象必须保留可读性

禁止：

- 为复用创建过度通用、名字空泛的抽象
- 通过复杂继承树隐藏行为
- 用“Manager / Util / Common / Helper”掩盖职责不清

优先：

- 组合优于继承
- 显式协作者优于神秘全局
- 清晰 service 优于万能基类

### 12.4 允许重复优于坏抽象

如果抽象会显著损害可读性，则少量重复优于引入坏抽象。
但一旦模式稳定，应回到清晰抽象。

---

## 13. 命名合同

推荐命名语义：

- `*Workflow`：完整阶段链路 / orchestration
- `*UseCase`：明确业务动作
- `*Service`：单一可复用业务职责
- `*Loader`：装载输入、配置、记录、工件
- `*Runner`：驱动外部执行或批量执行
- `*Writer`：落盘或输出工件
- `*Builder`：构造计划、命令、payload、rows、report
- `*Parser`：解析输入格式
- `*Validator`：验证契约
- `*Normalizer` / `*Formatter` / `*Converter`：转换数据表示
- `*Result` / `*Config` / `*Request` / `*Response`：数据契约对象

禁止命名：

- `Manager`
- `Helper`
- `Util`
- `Common`
- `Misc`
- `Processor`（若实际职责仍然含混）
- `Engine`（若不是清楚的执行内核）
- 名字抽象到读者无法预测职责

例外：

- 若某领域内这些词具有明确行业含义，可保留，但必须确保职责诚实

---

## 14. 注释、文档与 README 规则

### 14.1 注释只解释“为什么”，不解释显而易见的“是什么”

应写：

- 设计原因
- 不变量
- 边界条件
- 历史兼容原因
- 外部协议约束
- 性能 / 正确性 trade-off

不应写：

- 代码已经清楚表达的动作描述
- 用注释替代结构拆分
- 用大段注释掩盖坏命名

### 14.2 模块级文档要求

复杂模块应在顶部说明：

- 该模块的角色
- 真实实现边界
- 与其它模块的关系
- 为什么它存在

### 14.3 README 必须引导到真实实现层

README 必须让读者清楚知道：

- 对外入口在哪
- 真实 workflow / service 在哪
- 兼容层为什么还存在
- 新代码默认应该落在哪类模块

禁止：

- README 只列入口，不告诉读者真实实现位置
- README 继续把读者引到已经退化为兼容层的目录 / 文件

### 14.4 兼容层必须说明存在理由

若保留兼容层，README 或协作日志必须明确写出：

1. 为什么还保留
2. 哪些调用依赖它
3. 真实实现在哪里
4. 是否有未来收缩计划

---

## 15. 测试设计合同

### 15.1 测试必须映射职责层

- Public Surface：smoke / CLI / API contract test
- Compatibility Surface：兼容行为测试
- Workflow：集成式业务流程测试
- Service / Kernel：单元测试
- Data Contract：序列化 / 校验 / schema test

### 15.2 测试应尽量围绕稳定契约

优先测试：

- 输入输出契约
- 阶段行为
- 错误边界
- 兼容面保证
- 关键不变量

避免：

- 对内部行级实现细节过度绑定
- 因重构内部结构就大面积失效的脆弱测试

### 15.3 巨型方法通常意味着测试边界不好

如果一个方法大到只能靠高层集成测试覆盖，通常说明内部职责未拆清。
良好架构应允许：

- kernel/service 被直接单测
- workflow 被少量高层测试验证编排
- surface 被薄 smoke 测试验证入口

### 15.4 新增或迁移代码的最低测试要求

至少应覆盖：

1. 主成功路径
2. 关键失败路径
3. 关键兼容路径（若存在）
4. 新抽出的 service/kernel 的直接测试

---

## 16. Review 与 Agent 执行标准

### 16.1 任何代码变更前，必须先回答五个问题

1. 这段新代码的角色是什么？
2. 它为什么不该放在 surface / 兼容层？
3. 它的主职责能否被一个诚实名字表达？
4. 它是否会成为复用能力？
5. 读者以后会去哪里找它？

### 16.2 Agent 实施步骤（硬性）

所有 agent 在写代码时，默认按以下步骤执行：

1. 先判定角色：surface / compatibility / workflow / service / kernel / data contract
2. 先设计边界：输入、输出、依赖、状态
3. 先命名主类 / 主模块 / 主方法
4. 先搭骨架：让 workflow 先表现阶段顺序
5. 再填阶段细节
6. 一旦某方法开始跨越多个语义阶段，立即停下并拆分
7. 一旦共享逻辑出现第二次，立即评估下沉复用
8. 提交前完成可读性审查、测试审查、README 审查

### 16.3 Review 必查问题

评审者至少检查：

1. 打开入口后，能否迅速定位真实实现？
2. workflow 顶层是否只表达阶段，而非埋细节？
3. 是否存在“注释分段的大方法”？
4. 类名、文件名、方法名是否诚实？
5. 是否出现兼容层偷藏主流程？
6. 是否出现 helper / util / common 垃圾桶？
7. 是否存在反向依赖？
8. 测试是否覆盖真实边界而不是假边界？
9. README 是否把读者带到真实实现位置？
10. 变更后是否比变更前更易导航、更易局部理解？

### 16.4 读者可理解性检查

若一个陌生读者无法在短时间内回答以下问题，则设计通常不合格：

- 真实入口在哪？
- 真实实现在哪？
- 主流程有哪些阶段？
- 哪些部分是复用能力？
- 哪些部分是兼容桥？
- 改哪一层最安全？

---

## 17. Shell / Script 特别规则

Shell / Bash / 小脚本必须更加克制。

允许：

- 环境变量整理
- 参数转发
- 调用真实实现
- 薄包装
- 开发便利脚本

禁止：

- 把复杂业务逻辑长期堆在 shell 中
- 在 shell 里实现复杂数据转换、复杂错误恢复、复杂控制流
- 让 shell 成为唯一真实实现位置

原则：

- shell 应尽量只做 bootstrap 与 delegation
- 复杂逻辑应转入更可测试、更可读的实现模块

---

## 18. 存量巨文件、巨类、巨方法治理规则

治理顺序固定如下：

1. 冻结 public / compatibility contract
2. 标出真实阶段边界
3. 先抽 workflow 骨架
4. 再抽 service / kernel
5. 再收缩入口和兼容层
6. 再整理 README / 协作日志
7. 最后回归测试与静态检查

### 18.1 处理巨型方法时的推荐步骤

以一个 2000 行 `run()` 为例，正确做法不是：

- 直接把它搬到新文件
- 或加一层 `Workflow` 名字
- 或只拆出几个无意义 `_step1/_step2`

正确做法应是：

1. 先识别真实阶段
2. 给阶段起诚实名字
3. 明确每个阶段的输入输出
4. 把纯计算、校验、I/O、格式转换分别下沉
5. 让顶层只留下阶段调度
6. 把模式差异抽到 strategy / dedicated handler
7. 用结果对象、配置对象替代漂移状态

### 18.2 迁移中允许保留的过渡层

允许存在：

- 薄 facade
- 薄 alias
- 薄 adapter
- 薄 legacy wrapper

但必须满足：

1. 很薄
2. 原因明确
3. 不继续堆新逻辑
4. 文档中可定位真实实现

---

## 19. 明确禁止的反模式

以下模式在 future code 中禁止，在存量治理中应优先清理：

1. 薄 surface 外观之下，真实实现仍藏在单个巨型方法中
2. `Workflow.run()` 只是旧 `main()` 改名，没有阶段拆分
3. `ScriptApp.run()` / `App.execute()` 唯一职责只是 `return main()`
4. compatibility layer 藏大量 helper、数据整形、contract 处理、artifact 拼装
5. 把巨文件整体搬家，再包一层 class / facade，就宣称完成分层
6. 一个类同时做 workflow、validator、loader、writer、formatter
7. 一个方法同时做 orchestration、业务计算、I/O、错误翻译和日志
8. 用注释分段的大方法长期存在
9. `Manager / Helper / Util / Common` 成为默认容器
10. kernel / helper 反向 import surface / compatibility
11. 新代码继续复制历史兼容层形状，而不是遵守当前合同
12. 把 if/elif 模式分支越堆越大，不抽 strategy
13. 用共享可变状态在远距离传递数据
14. 用巨型 dict / kwargs 包裹一切边界
15. 用多个布尔参数控制互斥模式
16. README 指向旧入口，却不指向真实实现
17. 测试只能从最顶层验证，因为内部根本无法单独理解和测试
18. 为了“类化”而类化，但没有产生任何导航收益或边界收益

---

## 20. 完成定义（DoD）

一次符合本合同的新增实现、重构或治理，至少满足：

- [ ] public surface 稳定或兼容策略明确
- [ ] 真实业务逻辑不落在 surface / compatibility 默认层
- [ ] workflow 顶层能够清楚表达阶段顺序
- [ ] 不存在以单个巨型方法承载整个业务链路的伪分层实现
- [ ] service / kernel / data contract 边界清楚
- [ ] 共享逻辑已下沉到可复用位置
- [ ] README 能正确引导读者找到真实实现层
- [ ] 兼容层存在理由已被记录
- [ ] 相关测试覆盖主成功路径、关键失败路径、关键兼容路径
- [ ] 相关静态检查 / 诊断无 error
- [ ] 变更后的代码比变更前更易导航、更易局部理解、更易复用

---

## 21. 给 Agent 的极简执行口令

当 agent 不确定如何落代码时，默认执行以下规则：

1. **先判定角色，再写代码。**
2. **先写 workflow 骨架，再填阶段细节。**
3. **顶层方法只排阶段，不埋实现。**
4. **一旦出现注释分段的大方法，立即拆。**
5. **一旦出现第二次重复，立即评估抽象。**
6. **稳定结构必须类型化。**
7. **副作用靠边界，核心逻辑可单测。**
8. **README 必须把人带到真实实现。**
9. **兼容可以保留，但必须薄。**
10. **类不是终点；可读的阶段边界才是终点。**

---

## 22. 机器可读冻结块

<!-- GENERAL_PROGRAMMING_READABILITY_CONTRACT_SPEC_START -->
```json
{
  "contract_key": "general_programming_readability_contract",
  "schema_version": "general_programming_readability_contract_v1",
  "doc_language": "zh-CN",
  "single_source_of_truth": "agent/contract/script_workflow_layering_contract.md",
  "applies_to": [
    "current_live_code",
    "future_new_code",
    "all_languages",
    "all_projects"
  ],
  "goals": [
    "thin_public_surface",
    "separate_compatibility_from_real_implementation",
    "human_readable_structure",
    "staged_workflow_visibility",
    "reuse_via_real_abstraction",
    "predictable_navigation",
    "honest_naming_and_honest_placement"
  ],
  "roles": {
    "public_surface": {
      "allowed": [
        "cli_entry",
        "api_entry",
        "bootstrap",
        "thin_boundary_validation",
        "call_lower_layers",
        "compatibility_path_retention"
      ],
      "forbidden": [
        "default_business_logic_home",
        "giant_orchestrator",
        "heavy_artifact_assembly",
        "helper_dump_zone"
      ]
    },
    "compatibility_composition_surface": {
      "allowed": [
        "facade",
        "composition_root",
        "legacy_wrapper",
        "alias_bridge",
        "minimal_monkeypatch_bridge"
      ],
      "forbidden": [
        "default_heavy_business_logic_home",
        "monolithic_pipeline_home",
        "hidden_helper_dump_zone"
      ]
    },
    "workflow_service_implementation": {
      "required": [
        "real_business_flow",
        "staged_orchestration",
        "service_extraction",
        "reusable_implementation"
      ],
      "forbidden": [
        "fake_workflow_that_only_renames_main",
        "single_huge_run_method"
      ]
    },
    "shared_kernel_helper": {
      "allowed": [
        "pure_functions",
        "io_helpers",
        "validation_helpers",
        "small_reusable_kernels"
      ],
      "forbidden": [
        "high_level_orchestration",
        "reverse_dependency_on_surface_layers",
        "becoming_misc_bucket"
      ]
    },
    "data_contract_domain_model": {
      "required": [
        "explicit_boundary_types",
        "stable_input_output_shapes",
        "named_results_and_configs"
      ],
      "forbidden": [
        "giant_untyped_dicts_as_primary_contract",
        "magic_tuples_for_complex_results"
      ]
    }
  },
  "dependency_direction": {
    "public_surface": [
      "may_depend_on_compatibility_composition",
      "may_depend_on_workflow_service",
      "may_depend_on_shared_kernel",
      "may_depend_on_data_contract"
    ],
    "compatibility_composition_surface": [
      "may_depend_on_workflow_service",
      "may_depend_on_shared_kernel",
      "may_depend_on_data_contract"
    ],
    "workflow_service_implementation": [
      "may_depend_on_same_layer",
      "may_depend_on_shared_kernel",
      "may_depend_on_data_contract",
      "must_not_depend_on_public_surface",
      "must_not_depend_on_compatibility_composition_by_default"
    ],
    "shared_kernel_helper": [
      "may_depend_on_low_level_data_contract",
      "must_not_depend_on_public_surface",
      "must_not_depend_on_compatibility_composition"
    ]
  },
  "method_rules": {
    "top_level_workflow_method_must": [
      "express_stage_order",
      "delegate_stage_details",
      "keep_navigation_visible"
    ],
    "hard_split_triggers": [
      "multiple_semantic_stages",
      "comment_based_manual_sectioning",
      "mix_of_orchestration_and_details",
      "mix_of_compute_and_io",
      "mode_branch_explosion",
      "scrolling_required_to_understand_whole_method",
      "cannot_describe_responsibility_in_one_sentence"
    ],
    "forbidden_patterns": [
      "single_huge_run_method",
      "comment_sectioned_megamethod",
      "workflow_method_containing_full_pipeline_details"
    ]
  },
  "class_rules": {
    "class_required_when": [
      "state_boundary_exists",
      "staged_workflow_exists",
      "stable_collaborators_exist",
      "lifecycle_exists"
    ],
    "prefer_function_when": [
      "pure_function",
      "stateless_small_transform",
      "class_adds_only_self_without_boundary_benefit"
    ],
    "forbidden_patterns": [
      "class_as_namespace_only",
      "class_as_long_function_container",
      "class_name_not_honest_about_real_responsibilities"
    ]
  },
  "naming_rules": {
    "preferred_suffixes": [
      "Workflow",
      "UseCase",
      "Service",
      "Loader",
      "Runner",
      "Writer",
      "Builder",
      "Parser",
      "Validator",
      "Result",
      "Config"
    ],
    "discouraged_names": [
      "Manager",
      "Helper",
      "Util",
      "Common",
      "Misc"
    ]
  },
  "review_gates": [
    "can_reader_find_real_implementation_quickly",
    "can_reader_see_workflow_stages_quickly",
    "no_hidden_megamethods",
    "readme_points_to_real_implementation",
    "tests_match_real_boundaries"
  ],
  "migration_dod": [
    "public_surface_stable_or_documented",
    "real_logic_moved_out_of_surface_layers",
    "workflow_top_level_reads_as_stage_skeleton",
    "no_fake_layering_via_single_huge_method",
    "shared_logic_extracted",
    "readme_updated",
    "tests_updated",
    "diagnostics_clean"
  ]
}
```
<!-- GENERAL_PROGRAMMING_READABILITY_CONTRACT_SPEC_END -->

---

## 23. 当前冻结结论

1. 以后所有新增代码，都必须先判断自己属于哪个角色层，再决定落点。
2. 以后所有新增 workflow，都必须先把阶段骨架写清楚，再填实现细节。
3. 以后所有新增 class，都必须回答“这个类到底明确了什么边界”；若答不上来，通常不该建类。
4. 以后所有巨型方法，都不得再以“已经 class 化”为理由继续保留。
5. 以后所有兼容层，都只能做兼容；不得再作为真实实现默认落点。
6. 以后所有 README，都必须把读者引到真实实现，而不是旧入口。
7. 以后所有 agent，都必须把“方法级可读性”视为与“目录级分层”同等重要的硬性要求。

---

## 24. 最简总结

本合同真正冻结的不是“要不要用类”，也不是“目录叫什么”，而是以下四件事：

1. **入口必须薄。**
2. **实现必须诚实。**
3. **workflow 必须能看见阶段。**
4. **任何大方法都不能靠 class 外壳伪装成好结构。**

只要一个实现仍然需要读者穿越一个巨型 `run()` / `process()` / `build()` 才能理解真实流程，那么它就还没有达到本合同要求。
