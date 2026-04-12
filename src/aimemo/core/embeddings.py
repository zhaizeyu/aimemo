"""Embedding providers.

Default provider calls the OpenAI-compatible embeddings API via LiteLLM
(model: text-embedding-3-small).  A deterministic hash-based fallback
(``builtin``) is available when no API key is configured.
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod

import numpy as np

from aimemo.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# ── OpenAI-compatible provider (LiteLLM) ────────────────────────────


class OpenAIEmbedding(EmbeddingProvider):
    """Calls text-embedding-3-small (or any model) via the OpenAI SDK."""

    def __init__(self, model: str | None = None, dimensions: int | None = None):
        from aimemo.core.llm import get_openai_client

        self._client_factory = get_openai_client
        self._model = model or settings.embedding_model
        self._dimensions = dimensions or settings.embedding_dim

    @property
    def dim(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        client = self._client_factory()
        resp = await client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimensions,
        )
        return resp.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._client_factory()
        resp = await client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        sorted_data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]


# ── Builtin hash-based fallback ─────────────────────────────────────


class BuiltinEmbedding(EmbeddingProvider):
    """Deterministic hash-based embedding — zero external dependencies."""

    def __init__(self, dimensions: int = 128):
        self._dim = dimensions

    @property
    def dim(self) -> int:
        return self._dim

    def _hash_project(self, text: str) -> list[float]:
        text = text.lower().strip()
        vec = np.zeros(self._dim, dtype=np.float64)

        tokens = text.split()
        if not tokens:
            tokens = [text] if text else [""]

        for token in tokens:
            for salt in range(max(1, self._dim // 32)):
                h = hashlib.sha256(f"{salt}:{token}".encode()).digest()
                for i, b in enumerate(h):
                    idx = (salt * 32 + i) % self._dim
                    vec[idx] += (b / 127.5) - 1.0

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def embed(self, text: str) -> list[float]:
        return self._hash_project(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_project(t) for t in texts]


# ── Utilities ────────────────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── Factory ──────────────────────────────────────────────────────────

_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        provider = settings.embedding_provider
        if provider == "auto":
            has_key = bool(settings.resolve_api_key())
            provider = "openai" if has_key else "builtin"

        if provider == "openai":
            _provider = OpenAIEmbedding()
            logger.info("Using OpenAI embedding provider (model=%s, dim=%d)",
                        settings.embedding_model, settings.embedding_dim)
        else:
            dim = min(settings.embedding_dim, 128)
            _provider = BuiltinEmbedding(dimensions=dim)
            logger.info("Using builtin hash embedding provider (dim=%d)", dim)
    return _provider


def reset_provider() -> None:
    """Reset the cached provider (useful for tests)."""
    global _provider
    _provider = None
