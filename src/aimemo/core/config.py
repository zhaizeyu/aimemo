"""Application configuration with sensible defaults."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "AIMEMO_"}

    app_name: str = "AIMemo"
    debug: bool = False

    # Storage
    db_path: str = str(Path.home() / ".aimemo" / "memory.db")

    # LiteLLM / OpenAI-compatible API
    openai_api_key: str = ""
    openai_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "DeepSeek-V3.2"
    vision_model: str = "gemini-3-flash-preview"

    # Embedding
    embedding_dim: int = 1536
    embedding_provider: str = "auto"  # "auto" | "openai" | "builtin"

    # Memory engine
    short_term_capacity: int = 50
    working_memory_capacity: int = 7
    consolidation_threshold: float = 0.6
    decay_rate: float = 0.02
    importance_boost_on_access: float = 0.1
    max_retrieval_results: int = 20

    # Reflection
    reflection_trigger_count: int = 10
    reflection_min_memories: int = 5

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    def resolve_api_key(self) -> str:
        return self.openai_api_key or os.environ.get("LITELLM_API_KEY", "")

    def resolve_base_url(self) -> str:
        return self.openai_base_url or os.environ.get("OPENAI_BASE_URL", "")


settings = Settings()
