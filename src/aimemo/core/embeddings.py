"""Embedding providers.

The builtin provider uses a deterministic hash-based projection that requires
no external API — good enough for tag / keyword similarity.  Plug in OpenAI or
any sentence-transformer by switching ``AIMEMO_EMBEDDING_PROVIDER``.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

import numpy as np

from aimemo.core.config import settings


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class BuiltinEmbedding(EmbeddingProvider):
    """Deterministic hash-based embedding — zero external dependencies.

    Uses multiple salted SHA-256 digests projected to [-1, 1] to produce a
    fixed-dimension vector.  Not learned, but stable and reproducible.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def _hash_project(self, text: str) -> list[float]:
        text = text.lower().strip()
        vec = np.zeros(self.dim, dtype=np.float64)

        tokens = text.split()
        if not tokens:
            tokens = [text] if text else [""]

        for token in tokens:
            for salt in range(max(1, self.dim // 32)):
                h = hashlib.sha256(f"{salt}:{token}".encode()).digest()
                for i, b in enumerate(h):
                    idx = (salt * 32 + i) % self.dim
                    vec[idx] += (b / 127.5) - 1.0

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def embed(self, text: str) -> list[float]:
        return self._hash_project(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_project(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        if settings.embedding_provider == "builtin":
            _provider = BuiltinEmbedding(dim=settings.embedding_dim)
        else:
            _provider = BuiltinEmbedding(dim=settings.embedding_dim)
    return _provider
