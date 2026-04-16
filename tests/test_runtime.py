from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aimemo.emotion.engine import EmotionEngine
from aimemo.emotion.store import EmotionStore
from aimemo.engine.memory_engine import MemoryEngine
from aimemo.runtime.assistant_runtime import AssistantRuntime
from aimemo.storage.sqlite import MemoryStore


@pytest.mark.asyncio
async def test_runtime_turn_updates_emotion_and_working_memory(tmp_path):
    memory_store = MemoryStore(db_path=str(tmp_path / "runtime.db"))
    await memory_store.init()
    emotion_store = EmotionStore(db_path=str(tmp_path / "runtime.db"))
    await emotion_store.init()

    memory_engine = MemoryEngine(memory_store)
    emotion_engine = EmotionEngine(emotion_store)
    responder = AsyncMock()
    responder.generate = AsyncMock(return_value="我在，先慢慢说。")
    runtime = AssistantRuntime(memory_engine, emotion_engine, responder)

    result = await runtime.handle_turn(
        agent_id="default",
        user_id="u-runtime-unit",
        session_id="sess-1",
        user_text="我有点焦虑，能听我说说吗",
    )

    assert result["reply"] == "我在，先慢慢说。"
    assert result["emotion_context"].support_mode.value == "comforting"
    responder.generate.assert_awaited_once()

    working = await memory_engine.get_working_memory("sess-1", "default")
    assert len(working) == 2
    assert working[0].content.startswith("User:")
    assert working[1].content.startswith("Assistant:")

    await emotion_store.close()
    await memory_store.close()
