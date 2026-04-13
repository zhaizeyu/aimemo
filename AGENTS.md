# AGENTS.md

## Cursor Cloud specific instructions

AIMemo is a FastAPI-based AI agent memory module with SQLite storage and in-process vector search. It uses LiteLLM-proxied OpenAI-compatible APIs for embeddings, chat, and vision.

### Services

| Service | Command | Notes |
|---------|---------|-------|
| API server | `uvicorn aimemo.app:app --reload --port 8000` | Auto-reloads on code changes. Swagger docs at `/docs`. |

### Development commands

- **Install:** `pip install -e ".[dev]"`
- **Lint:** `ruff check src/ tests/`
- **Test:** `pytest tests/ -v`
- **Dev server:** `uvicorn aimemo.app:app --reload --port 8000`

### LLM / Embedding configuration

The system uses OpenAI-compatible APIs via LiteLLM. Three models are configured:

| Model | Config key | Purpose |
|-------|-----------|---------|
| `text-embedding-3-small` | `AIMEMO_EMBEDDING_MODEL` | Vector embeddings |
| `DeepSeek-V3.2` | `AIMEMO_CHAT_MODEL` | Reflection generation, memory merging |
| `gemini-3-flash-preview` | `AIMEMO_VISION_MODEL` | Image → text (multimodal memories) |

Environment variables needed at runtime:
- `LITELLM_API_KEY` — API key for the LiteLLM gateway (read as fallback if `AIMEMO_OPENAI_API_KEY` is empty)
- `OPENAI_BASE_URL` — base URL of the LiteLLM gateway (read as fallback if `AIMEMO_OPENAI_BASE_URL` is empty)

When no API key is available, embedding automatically falls back to the builtin hash-based provider. LLM-powered features (reflection, merge) also fall back to simple string operations.

### Starting the dev server

The dev server needs `LITELLM_API_KEY` and `OPENAI_BASE_URL` exported in the shell to use real LLM/embedding models. These are injected as Cursor Secrets. When starting the server in tmux or a new shell, ensure these variables are exported:

```bash
export PATH="$HOME/.local/bin:$PATH"
uvicorn aimemo.app:app --host 0.0.0.0 --port 8000 --reload
```

### Gotchas

- The `~/.local/bin` directory must be on `PATH` for `uvicorn`, `pytest`, and `ruff` (installed by pip as user packages). Run `export PATH="$HOME/.local/bin:$PATH"` if commands are not found.
- SQLite DB is created at `~/.aimemo/memory.db` by default. The parent directory is auto-created on startup.
- `AIMEMO_EMBEDDING_PROVIDER` defaults to `auto` — it will use OpenAI embeddings if an API key is present, otherwise the builtin hash-based fallback.
- When switching between OpenAI and builtin embeddings, delete the DB first (`rm ~/.aimemo/memory.db`) because the embedding dimensions differ (1536 vs 128).
- `pytest-asyncio` is configured in `pyproject.toml` with `asyncio_mode = "auto"` so async test functions work without per-function markers.
- Tests always use `builtin` embeddings via `conftest.py` auto-use fixture, so they run without API keys.
- The `B008` ruff rule is disabled in `pyproject.toml` because FastAPI's `File(...)` / `Form(...)` defaults require function calls in parameter defaults.
