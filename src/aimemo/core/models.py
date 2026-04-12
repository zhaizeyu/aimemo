"""Domain models — the vocabulary of the memory system.

Memory tiers follow cognitive science:
  - working   : active focus (≤7 items, Millerʼs law)
  - short_term: recent context (≤N items, configurable)
  - long_term : consolidated, persisted indefinitely
  - episodic  : event-based autobiographical memories
  - semantic  : factual / conceptual knowledge distilled from episodes
  - procedural: how-to / skill-based knowledge
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


class MemoryTier(StrEnum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    REFLECTION = "reflection"


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=_new_id)
    content: str
    memory_type: MemoryType = MemoryType.EPISODIC
    tier: MemoryTier = MemoryTier.SHORT_TERM
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=_utcnow)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    source_ids: list[str] = Field(default_factory=list)
    agent_id: str = "default"


# ── Request / Response Schemas ──────────────────────────────────────

class MemoryCreate(BaseModel):
    content: str
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    agent_id: str = "default"


class MemoryUpdate(BaseModel):
    content: str | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class MemoryQuery(BaseModel):
    query: str
    agent_id: str = "default"
    memory_types: list[MemoryType] | None = None
    tiers: list[MemoryTier] | None = None
    tags: list[str] | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)


class RetrievalResult(BaseModel):
    memory: MemoryRecord
    relevance_score: float
    recency_score: float
    importance_score: float
    combined_score: float


class MemoryStats(BaseModel):
    total_memories: int
    by_tier: dict[str, int]
    by_type: dict[str, int]
    avg_importance: float
    agent_id: str


class ConsolidationResult(BaseModel):
    promoted: int
    decayed: int
    merged: int
    reflections_generated: int
