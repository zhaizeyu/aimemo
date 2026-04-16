# Emotion Brain 详细设计（MVP v1）

## 1. 问题定义

AIMemo 已覆盖“我知道什么/记得什么”，但陪伴助手还需要“我现在如何回应更合适”。
Emotion Brain 负责：
- 当前状态（快变量）
- 用户关系（慢变量）
- 输入事件的情绪信号提取
- 将状态压缩为可直接用于回答的 context

## 2. 模块职责

- `models.py`：定义状态与上下文契约
- `analyzer.py`：规则优先信号抽取
- `policies.py`：状态转移与沉淀策略
- `store.py`：独立持久化与审计日志
- `engine.py`：统一调用入口

## 3. 快变量/慢变量

### 快变量（每轮可明显变化）
- mood_valence
- energy
- arousal
- stress
- loneliness
- comfort_drive
- guardedness
- support_mode
- style_mode

### 慢变量（需持续互动累积）
- familiarity
- trust
- affection
- dependence
- safety
- emotional_closeness
- interaction_count

## 4. 规则策略要点

1. 冲突优先级最高：触发 boundary/direct
2. support/vulnerability 触发 comforting/warm
3. advice intent 触发 advising/calm
4. praise 触发 celebrating/warm
5. 其余进入 neutral/calm

## 5. 长期沉淀策略

只在满足下列条件时写入 AIMemo：
- 最近事件数达到稳定窗口（例如 ≥6）
- support_need 或关系指标呈稳定趋势
- 24h cooldown 已过

沉淀内容采用 `reflection`，并带 `source=emotion_brain` 元数据，方便检索与回溯。

## 6. 与 AIMemo 的边界

- Emotion Brain 不修改 memories 主表 schema
- 不侵入 MemoryEngine 内部逻辑
- 通过 Runtime / Engine API 协作

## 7. 后续演进

- 引入配置驱动策略（不同 agent persona）
- 引入 topic-level emotional triggers
- 引入情绪状态可视化诊断 API

