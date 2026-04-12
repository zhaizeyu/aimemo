"""Application configuration with sensible defaults."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "AIMEMO_"}

    app_name: str = "AIMemo"
    debug: bool = False

    # Storage
    db_path: str = str(Path.home() / ".aimemo" / "memory.db")

    # Embedding
    embedding_dim: int = 128
    embedding_provider: str = "builtin"  # "builtin" | "openai" | "custom"
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

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


settings = Settings()
