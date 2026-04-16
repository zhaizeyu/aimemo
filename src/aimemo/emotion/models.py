"""Domain models for the Emotion Brain subsystem."""

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


class SupportMode(StrEnum):
    NEUTRAL = "neutral"
    COMFORTING = "comforting"
    ADVISING = "advising"
    CELEBRATING = "celebrating"
    BOUNDARY = "boundary"


class StyleMode(StrEnum):
    WARM = "warm"
    CALM = "calm"
    DIRECT = "direct"


class EmotionState(BaseModel):
    """Fast-changing internal state for a specific agent and optional session."""

    agent_id: str = "default"
    session_id: str | None = None
    mood_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    energy: float = Field(default=0.5, ge=0.0, le=1.0)
    arousal: float = Field(default=0.4, ge=0.0, le=1.0)
    stress: float = Field(default=0.2, ge=0.0, le=1.0)
    loneliness: float = Field(default=0.3, ge=0.0, le=1.0)
    comfort_drive: float = Field(default=0.5, ge=0.0, le=1.0)
    guardedness: float = Field(default=0.2, ge=0.0, le=1.0)
    support_mode: SupportMode = SupportMode.NEUTRAL
    style_mode: StyleMode = StyleMode.WARM
    updated_at: datetime = Field(default_factory=_utcnow)


class RelationshipState(BaseModel):
    """Slow-changing relationship profile for an agent-user pair."""

    agent_id: str = "default"
    user_id: str
    familiarity: float = Field(default=0.1, ge=0.0, le=1.0)
    trust: float = Field(default=0.5, ge=0.0, le=1.0)
    affection: float = Field(default=0.3, ge=0.0, le=1.0)
    dependence: float = Field(default=0.2, ge=0.0, le=1.0)
    safety: float = Field(default=0.6, ge=0.0, le=1.0)
    emotional_closeness: float = Field(default=0.3, ge=0.0, le=1.0)
    interaction_count: int = 0
    last_interaction_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmotionEvent(BaseModel):
    """Signals extracted from a user turn."""

    id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_utcnow)
    raw_text: str
    user_id: str
    session_id: str | None = None
    intent: str = "chat"
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    urgency: float = Field(default=0.0, ge=0.0, le=1.0)
    support_need: float = Field(default=0.0, ge=0.0, le=1.0)
    attachment_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    praise_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    conflict_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    vulnerability_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    playfulness_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class EmotionContext(BaseModel):
    """LLM-facing emotion summary, not the raw state dump."""

    agent_id: str
    user_id: str
    support_mode: SupportMode
    style_mode: StyleMode
    response_guidance: str
    boundaries: list[str] = Field(default_factory=list)
    state_summary: str
    relationship_summary: str
    salient_signals: list[str] = Field(default_factory=list)


class EmotionUpdateResult(BaseModel):
    """Result of processing one user event."""

    event: EmotionEvent
    emotion_state: EmotionState
    relationship_state: RelationshipState
    context: EmotionContext
    materialized_memory_ids: list[str] = Field(default_factory=list)
