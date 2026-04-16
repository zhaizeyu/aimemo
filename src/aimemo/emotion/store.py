"""SQLite storage for Emotion Brain states and event logs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from aimemo.emotion.models import (
    EmotionEvent,
    EmotionState,
    RelationshipState,
    StyleMode,
    SupportMode,
)

_DDL = """
CREATE TABLE IF NOT EXISTS emotion_state (
    agent_id      TEXT NOT NULL,
    session_id    TEXT,
    mood_valence  REAL NOT NULL,
    energy        REAL NOT NULL,
    arousal       REAL NOT NULL,
    stress        REAL NOT NULL,
    loneliness    REAL NOT NULL,
    comfort_drive REAL NOT NULL,
    guardedness   REAL NOT NULL,
    support_mode  TEXT NOT NULL,
    style_mode    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (agent_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_emotion_state_agent ON emotion_state(agent_id);

CREATE TABLE IF NOT EXISTS relationship_state (
    agent_id             TEXT NOT NULL,
    user_id              TEXT NOT NULL,
    familiarity          REAL NOT NULL,
    trust                REAL NOT NULL,
    affection            REAL NOT NULL,
    dependence           REAL NOT NULL,
    safety               REAL NOT NULL,
    emotional_closeness  REAL NOT NULL,
    interaction_count    INTEGER NOT NULL,
    last_interaction_at  TEXT NOT NULL,
    metadata             TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (agent_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_relationship_user ON relationship_state(user_id);

CREATE TABLE IF NOT EXISTS emotion_events (
    id                   TEXT PRIMARY KEY,
    created_at           TEXT NOT NULL,
    agent_id             TEXT NOT NULL,
    user_id              TEXT NOT NULL,
    session_id           TEXT,
    raw_text             TEXT NOT NULL,
    intent               TEXT NOT NULL,
    sentiment            REAL NOT NULL,
    urgency              REAL NOT NULL,
    support_need         REAL NOT NULL,
    attachment_signal    REAL NOT NULL,
    praise_signal        REAL NOT NULL,
    conflict_signal      REAL NOT NULL,
    vulnerability_signal REAL NOT NULL,
    playfulness_signal   REAL NOT NULL,
    confidence           REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_emotion_events_user ON emotion_events(agent_id, user_id, created_at DESC);
"""


class EmotionStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_DDL)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def get_emotion_state(self, agent_id: str, session_id: str | None = None) -> EmotionState:
        assert self._db
        cursor = await self._db.execute(
            """
            SELECT * FROM emotion_state
            WHERE agent_id = :agent_id
              AND (
                session_id = :session_id
                OR (session_id IS NULL AND :session_id IS NULL)
              )
            """,
            {"agent_id": agent_id, "session_id": session_id},
        )
        row = await cursor.fetchone()
        if row is None:
            return EmotionState(agent_id=agent_id, session_id=session_id)

        return EmotionState(
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            mood_valence=row["mood_valence"],
            energy=row["energy"],
            arousal=row["arousal"],
            stress=row["stress"],
            loneliness=row["loneliness"],
            comfort_drive=row["comfort_drive"],
            guardedness=row["guardedness"],
            support_mode=SupportMode(row["support_mode"]),
            style_mode=StyleMode(row["style_mode"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def save_emotion_state(self, state: EmotionState) -> EmotionState:
        assert self._db
        await self._db.execute(
            """
            INSERT INTO emotion_state (
                agent_id, session_id, mood_valence, energy, arousal, stress,
                loneliness, comfort_drive, guardedness, support_mode, style_mode, updated_at
            ) VALUES (
                :agent_id, :session_id, :mood_valence, :energy, :arousal, :stress,
                :loneliness, :comfort_drive, :guardedness, :support_mode, :style_mode, :updated_at
            )
            ON CONFLICT(agent_id, session_id) DO UPDATE SET
                mood_valence = excluded.mood_valence,
                energy = excluded.energy,
                arousal = excluded.arousal,
                stress = excluded.stress,
                loneliness = excluded.loneliness,
                comfort_drive = excluded.comfort_drive,
                guardedness = excluded.guardedness,
                support_mode = excluded.support_mode,
                style_mode = excluded.style_mode,
                updated_at = excluded.updated_at
            """,
            {
                "agent_id": state.agent_id,
                "session_id": state.session_id,
                "mood_valence": state.mood_valence,
                "energy": state.energy,
                "arousal": state.arousal,
                "stress": state.stress,
                "loneliness": state.loneliness,
                "comfort_drive": state.comfort_drive,
                "guardedness": state.guardedness,
                "support_mode": state.support_mode.value,
                "style_mode": state.style_mode.value,
                "updated_at": state.updated_at.isoformat(),
            },
        )
        await self._db.commit()
        return state

    async def get_relationship_state(self, agent_id: str, user_id: str) -> RelationshipState:
        assert self._db
        cursor = await self._db.execute(
            "SELECT * FROM relationship_state WHERE agent_id = :agent_id AND user_id = :user_id",
            {"agent_id": agent_id, "user_id": user_id},
        )
        row = await cursor.fetchone()
        if row is None:
            return RelationshipState(agent_id=agent_id, user_id=user_id)

        return RelationshipState(
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            familiarity=row["familiarity"],
            trust=row["trust"],
            affection=row["affection"],
            dependence=row["dependence"],
            safety=row["safety"],
            emotional_closeness=row["emotional_closeness"],
            interaction_count=row["interaction_count"],
            last_interaction_at=datetime.fromisoformat(row["last_interaction_at"]),
            metadata=json.loads(row["metadata"]),
        )

    async def save_relationship_state(self, relation: RelationshipState) -> RelationshipState:
        assert self._db
        await self._db.execute(
            """
            INSERT INTO relationship_state (
                agent_id, user_id, familiarity, trust, affection, dependence,
                safety, emotional_closeness, interaction_count, last_interaction_at, metadata
            ) VALUES (
                :agent_id, :user_id, :familiarity, :trust, :affection, :dependence,
                :safety, :emotional_closeness, :interaction_count, :last_interaction_at, :metadata
            )
            ON CONFLICT(agent_id, user_id) DO UPDATE SET
                familiarity = excluded.familiarity,
                trust = excluded.trust,
                affection = excluded.affection,
                dependence = excluded.dependence,
                safety = excluded.safety,
                emotional_closeness = excluded.emotional_closeness,
                interaction_count = excluded.interaction_count,
                last_interaction_at = excluded.last_interaction_at,
                metadata = excluded.metadata
            """,
            {
                "agent_id": relation.agent_id,
                "user_id": relation.user_id,
                "familiarity": relation.familiarity,
                "trust": relation.trust,
                "affection": relation.affection,
                "dependence": relation.dependence,
                "safety": relation.safety,
                "emotional_closeness": relation.emotional_closeness,
                "interaction_count": relation.interaction_count,
                "last_interaction_at": relation.last_interaction_at.isoformat(),
                "metadata": json.dumps(relation.metadata),
            },
        )
        await self._db.commit()
        return relation

    async def append_event(self, agent_id: str, event: EmotionEvent) -> EmotionEvent:
        assert self._db
        await self._db.execute(
            """
            INSERT INTO emotion_events (
                id, created_at, agent_id, user_id, session_id, raw_text, intent,
                sentiment, urgency, support_need, attachment_signal, praise_signal,
                conflict_signal, vulnerability_signal, playfulness_signal, confidence
            ) VALUES (
                :id, :created_at, :agent_id, :user_id, :session_id, :raw_text, :intent,
                :sentiment, :urgency, :support_need, :attachment_signal, :praise_signal,
                :conflict_signal, :vulnerability_signal, :playfulness_signal, :confidence
            )
            """,
            {
                "id": event.id,
                "created_at": event.created_at.isoformat(),
                "agent_id": agent_id,
                "user_id": event.user_id,
                "session_id": event.session_id,
                "raw_text": event.raw_text,
                "intent": event.intent,
                "sentiment": event.sentiment,
                "urgency": event.urgency,
                "support_need": event.support_need,
                "attachment_signal": event.attachment_signal,
                "praise_signal": event.praise_signal,
                "conflict_signal": event.conflict_signal,
                "vulnerability_signal": event.vulnerability_signal,
                "playfulness_signal": event.playfulness_signal,
                "confidence": event.confidence,
            },
        )
        await self._db.commit()
        return event

    async def list_recent_events(
        self,
        agent_id: str,
        user_id: str,
        limit: int = 20,
    ) -> list[EmotionEvent]:
        assert self._db
        cursor = await self._db.execute(
            """
            SELECT * FROM emotion_events
            WHERE agent_id = :agent_id AND user_id = :user_id
            ORDER BY created_at DESC
            LIMIT :limit
            """,
            {"agent_id": agent_id, "user_id": user_id, "limit": limit},
        )
        rows = await cursor.fetchall()
        results: list[EmotionEvent] = []
        for row in rows:
            results.append(
                EmotionEvent(
                    id=row["id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    raw_text=row["raw_text"],
                    user_id=row["user_id"],
                    session_id=row["session_id"],
                    intent=row["intent"],
                    sentiment=row["sentiment"],
                    urgency=row["urgency"],
                    support_need=row["support_need"],
                    attachment_signal=row["attachment_signal"],
                    praise_signal=row["praise_signal"],
                    conflict_signal=row["conflict_signal"],
                    vulnerability_signal=row["vulnerability_signal"],
                    playfulness_signal=row["playfulness_signal"],
                    confidence=row["confidence"],
                )
            )
        return results

    async def event_count_since(
        self,
        agent_id: str,
        user_id: str,
        since: datetime,
    ) -> int:
        assert self._db
        cursor = await self._db.execute(
            """
            SELECT COUNT(*) AS cnt FROM emotion_events
            WHERE agent_id = :agent_id
              AND user_id = :user_id
              AND created_at >= :since
            """,
            {
                "agent_id": agent_id,
                "user_id": user_id,
                "since": since.astimezone(UTC).isoformat(),
            },
        )
        row = await cursor.fetchone()
        return int(row["cnt"])
