"""Emotion Brain engine.

Coordinates analyzer, policies, store, and optional AIMemo materialization.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aimemo.core.models import MemoryCreate, MemoryType
from aimemo.emotion.analyzer import EmotionAnalyzer
from aimemo.emotion.models import (
    EmotionContext,
    EmotionState,
    EmotionUpdateResult,
    RelationshipState,
)
from aimemo.emotion.policies import (
    apply_emotion_event,
    apply_relationship_event,
    decay_to_baseline,
    materialization_cooldown,
    should_materialize_pattern,
)
from aimemo.emotion.store import EmotionStore


class EmotionEngine:
    def __init__(self, store: EmotionStore, analyzer: EmotionAnalyzer | None = None):
        self.store = store
        self.analyzer = analyzer or EmotionAnalyzer()

    async def process_user_event(
        self,
        *,
        agent_id: str,
        user_id: str,
        raw_text: str,
        session_id: str | None = None,
    ) -> EmotionUpdateResult:
        event = await self.analyzer.analyze(raw_text=raw_text, user_id=user_id, session_id=session_id)

        state = await self.get_state(agent_id=agent_id, session_id=session_id)
        relation = await self.get_relationship(agent_id=agent_id, user_id=user_id)

        state = apply_emotion_event(state, event)
        relation = apply_relationship_event(relation, event)

        await self.store.save_emotion_state(state)
        await self.store.save_relationship_state(relation)
        await self.store.append_event(agent_id, event)

        context = self.build_emotion_context(
            agent_id=agent_id,
            user_id=user_id,
            state=state,
            relation=relation,
            event=event,
        )
        return EmotionUpdateResult(
            event=event,
            emotion_state=state,
            relationship_state=relation,
            context=context,
        )

    async def get_state(self, *, agent_id: str, session_id: str | None = None) -> EmotionState:
        return await self.store.get_emotion_state(agent_id=agent_id, session_id=session_id)

    async def get_relationship(self, *, agent_id: str, user_id: str) -> RelationshipState:
        return await self.store.get_relationship_state(agent_id=agent_id, user_id=user_id)

    def build_emotion_context(
        self,
        *,
        agent_id: str,
        user_id: str,
        state: EmotionState,
        relation: RelationshipState,
        event,
    ) -> EmotionContext:
        signals: list[str] = []
        if event.support_need > 0.4:
            signals.append("user-needs-support")
        if event.conflict_signal > 0.4:
            signals.append("conflict-risk")
        if event.praise_signal > 0.4:
            signals.append("positive-feedback")

        guidance = {
            "comforting": "先共情，再提一条可执行的小建议。",
            "advising": "结构化回答，给步骤和取舍。",
            "celebrating": "积极回应并强化用户成就感。",
            "boundary": "保持边界，平静且明确。",
            "neutral": "自然对话，简洁清晰。",
        }[state.support_mode.value]
        generation_notes = (
            f"当前用户 {user_id} 更偏向 {state.support_mode.value} 支持模式；"
            f"关系熟悉度 {relation.familiarity:.2f}、信任度 {relation.trust:.2f}。"
            "优先回应情绪，再给最小可执行建议；避免说教和过度承诺。"
        )

        return EmotionContext(
            agent_id=agent_id,
            user_id=user_id,
            support_mode=state.support_mode,
            style_mode=state.style_mode,
            response_guidance=guidance,
            boundaries=["不夸大承诺", "避免情绪绑架"],
            state_summary=(
                f"valence={state.mood_valence:.2f}, stress={state.stress:.2f}, "
                f"guardedness={state.guardedness:.2f}, support_mode={state.support_mode.value}"
            ),
            relationship_summary=(
                f"familiarity={relation.familiarity:.2f}, trust={relation.trust:.2f}, "
                f"affection={relation.affection:.2f}, closeness={relation.emotional_closeness:.2f}"
            ),
            salient_signals=signals,
            generation_notes=generation_notes,
        )

    async def decay_or_recover_state(
        self,
        *,
        agent_id: str,
        session_id: str | None = None,
        now: datetime | None = None,
    ) -> EmotionState:
        state = await self.get_state(agent_id=agent_id, session_id=session_id)
        state = decay_to_baseline(state, now=now)
        await self.store.save_emotion_state(state)
        return state

    async def materialize_patterns_to_memory(
        self,
        *,
        agent_id: str,
        user_id: str,
        memory_engine,
    ) -> list[str]:
        relation = await self.get_relationship(agent_id=agent_id, user_id=user_id)
        recent_events = await self.store.list_recent_events(agent_id=agent_id, user_id=user_id, limit=10)
        if not should_materialize_pattern(recent_events, relation):
            return []

        last_marker = relation.metadata.get("last_materialized_at")
        if materialization_cooldown(last_marker):
            return []

        support_heavy = sum(1 for e in recent_events if e.support_need >= 0.5)
        content = (
            f"用户 {user_id} 在近期互动中较常需要情绪支持（{support_heavy}/{len(recent_events)}），"
            "回应策略应优先倾听+安抚。"
        )
        created = await memory_engine.add_memory(
            MemoryCreate(
                content=content,
                memory_type=MemoryType.REFLECTION,
                importance=0.72,
                tags=["emotion_pattern", "relationship"],
                metadata={"source": "emotion_brain", "user_id": user_id},
                agent_id=agent_id,
            )
        )
        relation.metadata["last_materialized_at"] = datetime.now(UTC).isoformat()
        await self.store.save_relationship_state(relation)
        return [created.id]

    async def recent_interaction_volume(self, *, agent_id: str, user_id: str, days: int = 7) -> int:
        since = datetime.now(UTC) - timedelta(days=days)
        return await self.store.event_count_since(agent_id=agent_id, user_id=user_id, since=since)
