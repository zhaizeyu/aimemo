# AGENTS.md

## Cursor Cloud specific instructions

AIMemo is a FastAPI-based AI agent memory module with SQLite storage and in-process vector search.

### Services

| Service | Command | Notes |
|---------|---------|-------|
| API server | `uvicorn aimemo.app:app --reload --port 8000` | Auto-reloads on code changes. Swagger docs at `/docs`. |

### Development commands

- **Install:** `pip install -e ".[dev]"`
- **Lint:** `ruff check src/ tests/`
- **Test:** `pytest tests/ -v`
- **Dev server:** `uvicorn aimemo.app:app --reload --port 8000`

### Gotchas

- The `~/.local/bin` directory must be on `PATH` for `uvicorn`, `pytest`, and `ruff` (installed by pip as user packages). Run `export PATH="$HOME/.local/bin:$PATH"` if commands are not found.
- SQLite DB is created at `~/.aimemo/memory.db` by default. The parent directory is auto-created on startup.
- The builtin embedding provider uses deterministic hash-based projections — no external API key required. For OpenAI embeddings, set `AIMEMO_EMBEDDING_PROVIDER=openai` and `AIMEMO_OPENAI_API_KEY`.
- `pytest-asyncio` is configured in `pyproject.toml` with `asyncio_mode = "auto"` so async test functions work without per-function markers.
