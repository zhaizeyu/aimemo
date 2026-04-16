from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from aimemo.emotion.engine import EmotionEngine
from aimemo.emotion.models import SupportMode
from aimemo.emotion.store import EmotionStore


@pytest.fixture
async def emotion_engine(tmp_path):
    store = EmotionStore(db_path=str(tmp_path / "emotion.db"))
    await store.init()
    engine = EmotionEngine(store)
    yield engine
    await store.close()


@pytest.mark.asyncio
async def test_negative_input_switches_to_comforting(emotion_engine: EmotionEngine):
    result = await emotion_engine.process_user_event(
        agent_id="default",
        user_id="u1",
        session_id="s1",
        raw_text="我今天很难过，压力好大，能陪我聊聊吗",
    )
    assert result.emotion_state.support_mode == SupportMode.COMFORTING
    assert result.emotion_state.stress > 0.2


@pytest.mark.asyncio
async def test_advice_input_switches_to_advising(emotion_engine: EmotionEngine):
    result = await emotion_engine.process_user_event(
        agent_id="default",
        user_id="u1",
        session_id="s1",
        raw_text="我应该怎么办，给我一个建议和计划",
    )
    assert result.emotion_state.support_mode == SupportMode.ADVISING


@pytest.mark.asyncio
async def test_praise_increases_affection(emotion_engine: EmotionEngine):
    before = await emotion_engine.get_relationship(agent_id="default", user_id="u2")
    await emotion_engine.process_user_event(
        agent_id="default",
        user_id="u2",
        session_id="s2",
        raw_text="谢谢你，真的帮了我很多，你太棒了",
    )
    after = await emotion_engine.get_relationship(agent_id="default", user_id="u2")
    assert after.affection > before.affection


@pytest.mark.asyncio
async def test_stable_interaction_increases_familiarity(emotion_engine: EmotionEngine):
    for i in range(12):
        await emotion_engine.process_user_event(
            agent_id="default",
            user_id="u3",
            session_id="s3",
            raw_text=f"第{i}次日常聊天",
        )
    relation = await emotion_engine.get_relationship(agent_id="default", user_id="u3")
    assert relation.familiarity >= 0.25
    assert relation.interaction_count == 12


@pytest.mark.asyncio
async def test_conflict_increases_guardedness_and_stress(emotion_engine: EmotionEngine):
    result = await emotion_engine.process_user_event(
        agent_id="default",
        user_id="u4",
        session_id="s4",
        raw_text="你错了，闭嘴，我讨厌你",
    )
    assert result.emotion_state.support_mode == SupportMode.BOUNDARY
    assert result.emotion_state.guardedness > 0.2
    assert result.emotion_state.stress > 0.2


@pytest.mark.asyncio
async def test_state_recovers_towards_baseline(emotion_engine: EmotionEngine):
    await emotion_engine.process_user_event(
        agent_id="default",
        user_id="u5",
        session_id="s5",
        raw_text="紧急！我现在要崩溃了",
    )
    state = await emotion_engine.get_state(agent_id="default", session_id="s5")
    state.updated_at = datetime.now(UTC) - timedelta(hours=6)
    await emotion_engine.store.save_emotion_state(state)

    recovered = await emotion_engine.decay_or_recover_state(agent_id="default", session_id="s5")
    assert recovered.stress < state.stress


@pytest.mark.asyncio
async def test_relationship_changes_slower_than_emotion(emotion_engine: EmotionEngine):
    relation_before = await emotion_engine.get_relationship(agent_id="default", user_id="u6")
    state_before = await emotion_engine.get_state(agent_id="default", session_id="s6")

    result = await emotion_engine.process_user_event(
        agent_id="default",
        user_id="u6",
        session_id="s6",
        raw_text="我很焦虑，快帮帮我",
    )

    fast_delta = abs(result.emotion_state.stress - state_before.stress)
    slow_delta = abs(result.relationship_state.trust - relation_before.trust)
    assert fast_delta > slow_delta


@pytest.mark.asyncio
async def test_only_stable_patterns_materialize(emotion_engine: EmotionEngine):
    memory_engine = AsyncMock()
    memory_engine.add_memory = AsyncMock(return_value=type("X", (), {"id": "mem-1"})())

    ids = await emotion_engine.materialize_patterns_to_memory(
        agent_id="default", user_id="u7", memory_engine=memory_engine
    )
    assert ids == []
    memory_engine.add_memory.assert_not_called()

    for _ in range(7):
        await emotion_engine.process_user_event(
            agent_id="default",
            user_id="u7",
            session_id="s7",
            raw_text="我很难过，能听我说吗",
        )

    ids = await emotion_engine.materialize_patterns_to_memory(
        agent_id="default", user_id="u7", memory_engine=memory_engine
    )
    assert ids == ["mem-1"]
    memory_engine.add_memory.assert_awaited_once()
