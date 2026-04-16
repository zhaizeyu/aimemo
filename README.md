# AIMemo

SOTA AI 智能体记忆模块 — 基于 FastAPI 的多层级、关联式记忆存取管理系统，集成 LiteLLM 大模型能力。

## 核心特性

| 特性 | 说明 |
|------|------|
| **四级记忆层级** | core（钉选）→ working（工作）→ short_term（短期）→ long_term（长期） |
| **四种记忆类型** | episodic（情景）、semantic（语义/事实）、procedural（程序性）、reflection（反思） |
| **语义事实层** | semantic 记忆自动提取 SPO 三元组（主语-谓语-宾语），支持事实投影查询 |
| **矛盾检测** | 新 semantic 记忆自动与已有事实比对，矛盾时归档旧版、合并最新事实 |
| **Core 记忆** | 钉选的核心记忆，始终参与每次检索，不衰减、不淘汰 |
| **Working Memory** | 会话绑定的工作记忆，容量限制（Miller's Law ≤7），溢出自动沉淀为 episodic |
| **LLM 智能分析** | DeepSeek-V3.2 自动判断 memory_type、importance、tags、SPO 三元组 |
| **LLM 反思生成** | 累积足够记忆后自动生成高阶反思（参考 Generative Agents） |
| **按类型合并策略** | episodic 聚类摘要、semantic 矛盾替换、procedural 版本更新、reflection 不合并 |
| **多模态记忆** | 通过 gemini-3-flash-preview 将图片转为文字描述后存储 |
| **记忆衰减** | Ebbinghaus 遗忘曲线，访问可重置衰减（spacing effect） |
| **完整归档** | 所有删除操作自动归档完整快照，支持溯源（reason + successor_id） |
| **多 Agent 隔离** | 通过 `agent_id` 支持多个智能体独立的记忆空间 |
| **优雅降级** | 无 API Key 时自动退回内置哈希向量化和简单字符串操作 |
| **Benchmark 评测** | 8 项自动化评测：事实更新正确率、冲突精度、检索命中率、时效性等 |
| **Emotion Brain 接入** | 生命周期内初始化 EmotionStore/EmotionEngine/AssistantRuntime，可通过 HTTP 调试 |

## 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 配置 LiteLLM 网关（可选，不配置则使用内置向量化）
export LITELLM_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-litellm-gateway/v1"

# 启动开发服务器
uvicorn aimemo.app:app --reload --port 8000

# 打开 API 文档
open http://localhost:8000/docs
```

## 模型配置

通过 LiteLLM 网关访问 OpenAI 兼容接口：

| 模型 | 用途 | 环境变量 |
|------|------|----------|
| `text-embedding-3-small` | 向量化（1536 维） | `AIMEMO_EMBEDDING_MODEL` |
| `DeepSeek-V3.2` | 反思生成、记忆合并、智能分析、矛盾检测 | `AIMEMO_CHAT_MODEL` |
| `gemini-3-flash-preview` | 图片→文字（多模态） | `AIMEMO_VISION_MODEL` |

## 记忆架构

### 层级（Tier）

| 层级 | 说明 | 衰减 | 删除 |
|------|------|------|------|
| `core` | 钉选记忆，始终参与检索上下文 | 不衰减 | 不自动删除 |
| `working` | 会话绑定，容量 ≤7，溢出沉淀 | 不衰减 | 不自动删除 |
| `short_term` | 近期记忆，可晋升/合并/淘汰 | 衰减 | importance<0.05 时删除 |
| `long_term` | 重要记忆，持久保存 | 衰减但不删除 | 不自动删除 |

### 类型（Type）

| 类型 | 说明 | 合并策略 |
|------|------|---------|
| `episodic` | 事件、对话、经历 | 聚类 + LLM 摘要 |
| `semantic` | 事实、知识、偏好（含 SPO 三元组） | 矛盾检测替换（不参与批量合并） |
| `procedural` | 操作步骤、规则、流程 | 版本更新（保新归档旧） |
| `reflection` | LLM 生成的高阶洞察 | 不合并（derived, non-authoritative） |

### 事实层（Semantic Fact Layer）

semantic 记忆自动提取结构化三元组：

```json
{
  "content": "张三的主力编程语言是Python",
  "metadata": {
    "subject": "张三",
    "predicate": "主力编程语言",
    "object_value": "Python",
    "update_type": "new_fact"
  }
}
```

`update_type` 取值：`new_fact` / `temporal_update` / `contradiction` / `refinement` / `duplicate`

当检测到矛盾时，旧事实自动归档（`reason=superseded`），新事实内容由 LLM 合并最新真相。

## API 端点

### 记忆 CRUD

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/memories` | 创建记忆（手动指定所有字段） |
| POST | `/api/v1/memories/smart` | **智能创建**：只需 content，LLM 自动分析其余字段 |
| POST | `/api/v1/memories/core` | 创建 core 记忆（钉选，永不衰减） |
| POST | `/api/v1/memories/image` | 上传图片，vision 模型转文字后存储 |
| GET | `/api/v1/memories` | 列出记忆（支持 tier/type/importance 过滤） |
| GET | `/api/v1/memories/{id}` | 获取单条记忆 |
| PATCH | `/api/v1/memories/{id}` | 更新记忆 |
| DELETE | `/api/v1/memories/{id}` | 删除记忆（自动归档） |

### Working Memory（工作记忆）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/working` | 写入工作记忆（绑定 session_id） |
| GET | `/api/v1/working` | 获取某 session 的工作记忆（按序） |
| POST | `/api/v1/working/flush` | 将 session 的工作记忆全部沉淀为 episodic |

### 事实查询

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/facts` | 查询活跃事实（支持 subject/predicate 过滤） |

### 检索 & 操作

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/retrieve` | 语义检索（多信号融合 + core 记忆自动附带） |
| POST | `/api/v1/consolidate` | 触发记忆整合（晋升/衰减/合并/反思） |

### 归档

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/archive/{original_id}` | 查询某条记忆的归档历史 |
| GET | `/api/v1/archive` | 列出所有归档记忆（可按 reason 过滤） |

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/stats` | 记忆统计（含归档计数） |
| GET | `/api/v1/health` | 健康检查 |

### Emotion / Runtime

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/emotion/state` | 查询当前情绪状态（可按 session） |
| GET | `/api/v1/emotion/relationship` | 查询指定用户关系状态 |
| POST | `/api/v1/emotion/process` | 输入一条用户文本并更新情绪状态机 |
| POST | `/api/v1/runtime/chat` | 完整单轮编排（检索→情绪更新→回复→写回） |

## 示例

```bash
# 智能创建（推荐）— 只传 content，LLM 自动分析
curl -X POST http://localhost:8000/api/v1/memories/smart \
  -H "Content-Type: application/json" \
  -d '{"content": "张三偏好使用Python编程，技术水平为中级开发者"}'

# 创建 core 记忆
curl -X POST http://localhost:8000/api/v1/memories/core \
  -H "Content-Type: application/json" \
  -d '{"content": "始终使用中文回复用户", "tags": ["system"]}'

# 写入工作记忆
curl -X POST http://localhost:8000/api/v1/working \
  -H "Content-Type: application/json" \
  -d '{"content": "用户问：如何用FastAPI写异步接口？", "session_id": "chat-001"}'

# 查询事实（SPO 投影）
curl "http://localhost:8000/api/v1/facts?subject=张三"

# 语义检索
curl -X POST http://localhost:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "用户的编程偏好", "top_k": 5}'

# 上传图片创建多模态记忆
curl -X POST http://localhost:8000/api/v1/memories/image \
  -F "file=@photo.png" -F "tags=screenshot,meeting"

# 触发整合
curl -X POST http://localhost:8000/api/v1/consolidate

# 情绪处理
curl -X POST http://localhost:8000/api/v1/emotion/process \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","session_id":"s1","raw_text":"我今天有点难过，想聊聊"}'

# runtime 单轮对话
curl -X POST http://localhost:8000/api/v1/runtime/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u1","session_id":"s1","user_text":"你在吗？今天有点累"}'

# 查看归档历史
curl "http://localhost:8000/api/v1/archive?reason=superseded"
```

## 整合策略

`POST /consolidate` 按顺序执行：

| 步骤 | 说明 |
|------|------|
| 1. 晋升 | importance ≥ 0.6 且 access_count ≥ 2 的 short_term → long_term |
| 2. 衰减 | Ebbinghaus 曲线降低 importance，< 0.05 的 short_term 删除（core/working 免疫） |
| 3. 合并 | 按类型区分：episodic 聚类摘要、procedural 版本更新、semantic/reflection 跳过 |
| 4. 反思 | LLM 生成高阶反思记忆 |

## 开发

```bash
# 运行全部测试（不需要 API Key）
pytest tests/ -v

# 仅运行 benchmark 评测
pytest tests/test_benchmark.py -v

# Lint 检查
ruff check src/ tests/
```

### Benchmark 评测项

| 评测指标 | 测试名 |
|---------|--------|
| 事实更新正确率 | `test_fact_update_correctness` |
| 冲突处理精度（无冲突） | `test_conflict_precision_no_conflict` |
| 冲突处理精度（有冲突） | `test_conflict_precision_with_conflict` |
| 检索命中率 | `test_retrieval_hit_rate` |
| 时效性一致率 | `test_temporal_consistency` |
| 归档溯源完整性 | `test_archive_traceability` |
| 摘要信息损失率 | `test_summary_loss_rate` |
| 工作记忆容量限制 | `test_working_memory_capacity` |

## 架构

```
src/aimemo/
├── app.py                # FastAPI 应用工厂
├── api/routes.py         # HTTP 路由（20 个端点）
├── core/
│   ├── config.py         # 配置（环境变量驱动）
│   ├── models.py         # Pydantic 模型（17 个）
│   ├── embeddings.py     # 向量化（OpenAI / 内置降级）
│   └── llm.py            # LLM 客户端（聊天/分析/矛盾检测/视觉）
├── engine/
│   └── memory_engine.py  # 记忆引擎（9 大 SOTA 特性）
└── storage/
    └── sqlite.py         # SQLite + numpy 向量索引 + 归档
tests/
├── conftest.py           # 测试配置（mock LLM, builtin embeddings）
├── test_api.py           # API 端点测试
├── test_emotion_engine.py# Emotion Brain 规则与状态迁移测试
├── test_runtime.py       # Runtime 编排链路测试
├── test_engine.py        # 引擎逻辑测试
└── test_benchmark.py     # Benchmark 评测套件
```

## 配置

通过环境变量配置，前缀 `AIMEMO_`：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AIMEMO_OPENAI_API_KEY` | (空) | API Key，也读取 `LITELLM_API_KEY` |
| `AIMEMO_OPENAI_BASE_URL` | (空) | Base URL，也读取 `OPENAI_BASE_URL` |
| `AIMEMO_EMBEDDING_MODEL` | `text-embedding-3-small` | 向量化模型 |
| `AIMEMO_CHAT_MODEL` | `DeepSeek-V3.2` | 聊天/反思/分析/矛盾检测模型 |
| `AIMEMO_VISION_MODEL` | `gemini-3-flash-preview` | 图片转文字模型 |
| `AIMEMO_DB_PATH` | `~/.aimemo/memory.db` | 数据库路径 |
| `AIMEMO_EMBEDDING_PROVIDER` | `auto` | `auto` / `openai` / `builtin` |
| `AIMEMO_EMBEDDING_DIM` | `1536` | 向量维度 |
| `AIMEMO_WORKING_MEMORY_CAPACITY` | `7` | 工作记忆容量上限（Miller's Law） |
| `AIMEMO_DECAY_RATE` | `0.02` | 衰减速率 |
| `AIMEMO_CONSOLIDATION_THRESHOLD` | `0.6` | 晋升阈值 |
| `AIMEMO_REFLECTION_TRIGGER_COUNT` | `10` | 触发反思的记忆数 |

## 参考文献

- Park et al. (2023) — *Generative Agents: Interactive Simulacra of Human Behavior*（反思机制）
- Packer et al. (2023) — *MemGPT: Towards LLMs as Operating Systems*（Core memory / 分层记忆）
- Ebbinghaus (1885) — *Memory: A Contribution to Experimental Psychology*（遗忘曲线）

## License

MIT
