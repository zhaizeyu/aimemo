# AIMemo 架构设计总览（截至当前版本）

> 本文档汇总当前代码仓库的系统架构、模块边界、数据流、存储结构与测试策略。

## 1. 系统目标与边界

AIMemo 当前包含两大能力面：

1. **逻辑脑 / 记忆脑（AIMemo Core）**
   - 多层级记忆管理（core / working / short_term / long_term）
   - 多类型记忆（episodic / semantic / procedural / reflection）
   - 检索、衰减、整合、反思、事实层与归档
2. **情绪脑（Emotion Brain）**
   - 快变量情绪状态
   - 慢变量关系状态
   - 事件级信号分析与可解释策略更新
   - 仅将稳定模式沉淀回长期记忆

两者通过 Runtime 编排协作，保持 schema 与职责解耦。

---

## 2. 目录与模块分层

```text
src/aimemo/
  api/                    # FastAPI 路由层
  app.py                  # 应用初始化与生命周期
  core/                   # 配置、模型、LLM、Embedding 抽象
  engine/                 # MemoryEngine（逻辑脑编排）
  storage/                # MemoryStore（SQLite + vector index）

  emotion/                # Emotion Brain（独立子系统）
    models.py             # EmotionState/RelationshipState/EmotionEvent/EmotionContext
    analyzer.py           # 文本信号抽取（规则优先）
    policies.py           # 状态更新策略、回归基线、沉淀门控
    store.py              # 情绪独立存储（emotion_state/relationship_state/emotion_events）
    engine.py             # 情绪编排入口

  runtime/                # 运行时联动编排
    assistant_runtime.py  # 一轮对话中的跨系统编排
```

---

## 3. 逻辑脑（MemoryEngine）架构

### 3.1 核心职责
- 写入记忆（含 embedding）
- 多信号检索（相关度 + 时间 + importance）
- working memory 管理与 session 绑定
- consolidation（晋升、衰减、合并、反思）
- semantic contradiction 检测与 supersede
- 归档审计

### 3.2 存储结构（MemoryStore）
- `memories`：主记忆表
- `memory_archive`：归档表
- 进程内 numpy 向量索引用于近似向量搜索

---

## 4. 情绪脑（Emotion Brain）架构

### 4.1 设计原则
- 与 AIMemo 主 schema 解耦
- 快变量与慢变量分离
- 规则策略驱动为主，分类能力可插拔
- 状态变化可追踪、可测试

### 4.2 核心模型
- `EmotionState`（快变量）：mood/energy/stress/guardedness/support_mode/style_mode
- `RelationshipState`（慢变量）：familiarity/trust/affection/safety/closeness
- `EmotionEvent`：一轮用户输入提取的信号
- `EmotionContext`：面向 LLM 的摘要上下文

### 4.3 独立存储
- `emotion_state`：当前快变量真相（upsert）
- `relationship_state`：当前关系真相（upsert）
- `emotion_events`：事件审计日志（append-only）

### 4.4 状态更新策略
- Analyzer 仅做信号抽取（support/conflict/praise/urgency 等）
- Policies 执行数值更新与模式决策
- `decay_to_baseline` 实现状态恢复
- `should_materialize_pattern` + cooldown 控制长期沉淀

---

## 5. Runtime 编排（AssistantRuntime）

单轮流程：

1. 读取用户输入
2. 调用 AIMemo retrieve 获取记忆上下文
3. 调用 EmotionEngine 处理事件并更新状态
4. 构建 unified context（memory + emotion）
5. 调主模型生成回复
6. 写入 working memory（user/assistant turn）
7. 判定并执行长期模式沉淀
8. 写入最小 episodic trace

---

## 6. 数据流（高层）

```text
User text
  └─> Runtime
      ├─> MemoryEngine.retrieve()
      ├─> EmotionEngine.process_user_event()
      ├─> ResponseGenerator.generate(context)
      ├─> MemoryEngine.add_working_memory()
      ├─> EmotionEngine.materialize_patterns_to_memory()
      └─> MemoryEngine.add_memory(episodic trace)
```

---

## 7. 测试策略

### 7.1 已覆盖方向
- 记忆引擎核心行为（已有 tests/test_engine.py, tests/test_api.py）
- 情绪脑关键行为（tests/test_emotion_engine.py）：
  - comforting/advising 模式切换
  - praise/conflict 对状态影响
  - familiarity 累积
  - 快慢变量变化速率差异
  - 基线回归
  - 稳定模式沉淀门控

### 7.2 后续建议
- Runtime 端到端集成测试（mock responder + mock memory engine）
- 情绪策略参数化测试（多语种、边界输入、抗噪）

---

## 8. 可扩展点

1. Analyzer 接入 LLM 分类器（保持策略层决定权）
2. 情绪策略参数配置化（按 agent profile 加载）
3. 长期模式沉淀细分为 semantic/reflection 模板
4. Runtime 增加 post-turn 异步任务队列（materialization / summarization）

---

## 9. 关键非目标（当前版本）

- 不做表演脑/VTuber 演出系统
- 不做重型心理学全量建模
- 不在 `MemoryRecord` 内塞大量情绪字段
- 不将瞬时波动全部落入长期记忆

