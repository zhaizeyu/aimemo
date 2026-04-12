# AIMemo

SOTA AI 智能体记忆模块 — 基于 FastAPI 的多层级、关联式记忆存取管理系统，集成 LiteLLM 大模型能力。

## 核心特性

| 特性 | 说明 |
|------|------|
| **多层级记忆** | working → short_term → long_term，模拟人类认知的记忆分层 |
| **多类型记忆** | episodic（情景）、semantic（语义）、procedural（程序性）、reflection（反思） |
| **语义检索** | 基于 text-embedding-3-small 向量 + 时间衰减 + 重要性的多信号融合排序 |
| **LLM 反思生成** | 使用 DeepSeek-V3.2 自动生成高阶反思记忆（参考 Generative Agents） |
| **LLM 记忆合并** | 使用 DeepSeek-V3.2 智能合并相似记忆，保留所有关键信息 |
| **多模态记忆** | 通过 gemini-3-flash-preview 将图片转为文字描述后存储 |
| **记忆衰减** | Ebbinghaus 遗忘曲线，访问可重置衰减（spacing effect） |
| **自动整合** | 近似记忆合并、低重要性记忆淘汰、短期→长期自动晋升 |
| **多 Agent 隔离** | 通过 `agent_id` 支持多个智能体独立的记忆空间 |
| **优雅降级** | 无 API Key 时自动退回内置哈希向量化和简单字符串操作 |

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
| `text-embedding-3-small` | 向量化 | `AIMEMO_EMBEDDING_MODEL` |
| `DeepSeek-V3.2` | 反思生成、记忆合并 | `AIMEMO_CHAT_MODEL` |
| `gemini-3-flash-preview` | 图片→文字（多模态） | `AIMEMO_VISION_MODEL` |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/memories` | 创建记忆 |
| POST | `/api/v1/memories/image` | 上传图片创建多模态记忆 |
| GET | `/api/v1/memories` | 列出记忆（支持过滤） |
| GET | `/api/v1/memories/{id}` | 获取单条记忆 |
| PATCH | `/api/v1/memories/{id}` | 更新记忆 |
| DELETE | `/api/v1/memories/{id}` | 删除记忆 |
| POST | `/api/v1/retrieve` | 语义检索 |
| POST | `/api/v1/consolidate` | 触发记忆整合 |
| GET | `/api/v1/stats` | 记忆统计 |
| GET | `/api/v1/health` | 健康检查 |

## 示例

```bash
# 存储文本记忆
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "用户偏好中文交流", "importance": 0.8, "tags": ["preference"]}'

# 上传图片创建多模态记忆
curl -X POST http://localhost:8000/api/v1/memories/image \
  -F "file=@photo.png" \
  -F "tags=screenshot,meeting" \
  -F "importance=0.7"

# 语义检索
curl -X POST http://localhost:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "用户的语言偏好", "top_k": 5}'
```

## 开发

```bash
# 运行测试（不需要 API Key，自动使用内置向量化）
pytest tests/ -v

# Lint 检查
ruff check src/ tests/
```

## 架构

```
src/aimemo/
├── app.py              # FastAPI 应用工厂
├── api/routes.py       # HTTP 路由（含图片上传端点）
├── core/
│   ├── config.py       # 配置（环境变量驱动）
│   ├── models.py       # Pydantic 模型
│   ├── embeddings.py   # 向量化（OpenAI / 内置降级）
│   └── llm.py          # LLM 客户端（聊天 + 多模态视觉）
├── engine/
│   └── memory_engine.py # 记忆引擎（检索/LLM整合/衰减/LLM反思）
└── storage/
    └── sqlite.py       # SQLite + numpy 向量索引
```

## 配置

通过环境变量配置，前缀 `AIMEMO_`：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AIMEMO_OPENAI_API_KEY` | (空) | API Key，也读取 `LITELLM_API_KEY` |
| `AIMEMO_OPENAI_BASE_URL` | (空) | Base URL，也读取 `OPENAI_BASE_URL` |
| `AIMEMO_EMBEDDING_MODEL` | `text-embedding-3-small` | 向量化模型 |
| `AIMEMO_CHAT_MODEL` | `DeepSeek-V3.2` | 聊天/反思模型 |
| `AIMEMO_VISION_MODEL` | `gemini-3-flash-preview` | 图片转文字模型 |
| `AIMEMO_DB_PATH` | `~/.aimemo/memory.db` | 数据库路径 |
| `AIMEMO_EMBEDDING_PROVIDER` | `auto` | `auto` / `openai` / `builtin` |
| `AIMEMO_EMBEDDING_DIM` | `1536` | 向量维度 |
| `AIMEMO_DECAY_RATE` | `0.02` | 衰减速率 |
| `AIMEMO_CONSOLIDATION_THRESHOLD` | `0.6` | 晋升阈值 |
| `AIMEMO_REFLECTION_TRIGGER_COUNT` | `10` | 触发反思的记忆数 |

## License

MIT
