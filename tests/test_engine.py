"""Tests for the memory engine core logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aimemo.core.models import MemoryCreate, MemoryQuery, MemoryType
from aimemo.engine.memory_engine import MemoryEngine
from aimemo.storage.sqlite import MemoryStore


@pytest.fixture
async def engine(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "test.db"))
    await store.init()
    eng = MemoryEngine(store)
    yield eng
    await store.close()


@pytest.mark.asyncio
async def test_add_and_get(engine: MemoryEngine):
    mem = await engine.add_memory(
        MemoryCreate(content="The capital of France is Paris.", importance=0.7, tags=["geography"])
    )
    assert mem.id
    assert mem.content == "The capital of France is Paris."
    assert mem.embedding is not None

    fetched = await engine.store.get(mem.id)
    assert fetched is not None
    assert fetched.content == mem.content


@pytest.mark.asyncio
async def test_retrieve_relevant(engine: MemoryEngine):
    await engine.add_memory(MemoryCreate(content="Python is a programming language."))
    await engine.add_memory(MemoryCreate(content="The weather today is sunny."))
    await engine.add_memory(MemoryCreate(content="JavaScript runs in the browser."))

    results = await engine.retrieve(MemoryQuery(query="programming languages"))
    assert len(results) > 0
    top = results[0]
    assert top.combined_score > 0


@pytest.mark.asyncio
async def test_high_importance_goes_long_term(engine: MemoryEngine):
    mem = await engine.add_memory(
        MemoryCreate(content="Critical system rule.", importance=0.9)
    )
    assert mem.tier.value == "long_term"


@pytest.mark.asyncio
async def test_consolidation(engine: MemoryEngine):
    for i in range(5):
        await engine.add_memory(
            MemoryCreate(content=f"Memory item {i}", importance=0.5, tags=["batch"])
        )
    result = await engine.consolidate("default")
    assert result.promoted >= 0
    assert result.decayed >= 0


@pytest.mark.asyncio
async def test_delete_memory(engine: MemoryEngine):
    mem = await engine.add_memory(MemoryCreate(content="To be deleted."))
    deleted = await engine.store.delete(mem.id)
    assert deleted is True
    assert await engine.store.get(mem.id) is None


@pytest.mark.asyncio
async def test_stats(engine: MemoryEngine):
    await engine.add_memory(MemoryCreate(content="Stat test memory."))
    stats = await engine.store.count("default")
    assert stats["total_memories"] >= 1


@pytest.mark.asyncio
async def test_reflection_triggered(engine: MemoryEngine):
    from aimemo.core.config import settings

    original_trigger = settings.reflection_trigger_count
    original_min = settings.reflection_min_memories
    settings.reflection_trigger_count = 3
    settings.reflection_min_memories = 3

    for i in range(6):
        await engine.add_memory(
            MemoryCreate(content=f"Reflection trigger memory {i}", importance=0.6)
        )

    all_mems = await engine.store.list_memories(agent_id="default", limit=100)
    reflection_mems = [m for m in all_mems if m.memory_type == MemoryType.REFLECTION]
    assert len(reflection_mems) >= 1

    settings.reflection_trigger_count = original_trigger
    settings.reflection_min_memories = original_min


@pytest.mark.asyncio
async def test_image_memory(engine: MemoryEngine, mock_vision_describe):
    """Image memory uses the vision model to convert image → text."""
    mem = await engine.add_image_memory(
        image_data=b"\x89PNG\r\n\x1a\nfake_image_bytes",
        mime_type="image/png",
        agent_id="default",
        tags=["photo", "test"],
        importance=0.6,
    )
    assert mem.content == "A photograph showing a cat sitting on a windowsill."
    assert mem.metadata["source"] == "image"
    assert "photo" in mem.tags
    mock_vision_describe.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_merge_fallback(engine: MemoryEngine):
    """When LLM merge fails, the engine falls back to concatenation."""
    with patch(
        "aimemo.engine.memory_engine.MemoryEngine._llm_merge",
        new_callable=AsyncMock,
    ) as mock_merge:
        mock_merge.return_value = "Merged: content A and content B together."

        result = await engine._llm_merge("content A", "content B")
        assert "Merged" in result


@pytest.mark.asyncio
async def test_llm_reflect_fallback(engine: MemoryEngine):
    """When LLM reflection fails, the engine falls back to simple concatenation."""
    from aimemo.core.models import MemoryRecord

    memories = [
        MemoryRecord(content=f"Memory {i}", memory_type=MemoryType.EPISODIC)
        for i in range(3)
    ]

    with patch("aimemo.core.llm.chat_completion", new_callable=AsyncMock) as mock_chat:
        mock_chat.side_effect = Exception("API down")
        result = await engine._llm_reflect(memories)
        assert "Reflection:" in result
        assert "Memory 0" in result


@pytest.mark.asyncio
async def test_smart_add_memory(engine: MemoryEngine, mock_analyze_memory):
    """Smart memory creation uses LLM to auto-fill type, importance, tags."""
    from aimemo.core.models import SmartMemoryCreate

    mem = await engine.smart_add_memory(
        SmartMemoryCreate(content="用户张三喜欢使用Python编程")
    )
    assert mem.memory_type.value == "semantic"
    assert mem.importance >= 0.8
    assert "python" in mem.tags
    assert mem.embedding is not None
    mock_analyze_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_archives_memory(engine: MemoryEngine):
    """Deleting a memory archives it before removal."""
    mem = await engine.add_memory(MemoryCreate(content="Will be archived."))
    original_id = mem.id

    await engine.store.delete(original_id, reason="manual")
    assert await engine.store.get(original_id) is None

    archived = await engine.store.get_archive(original_id)
    assert len(archived) == 1
    assert archived[0].original_id == original_id
    assert archived[0].content == "Will be archived."
    assert archived[0].reason == "manual"


@pytest.mark.asyncio
async def test_merge_archives_with_successor(engine: MemoryEngine):
    """Merged memories are archived with reason='merged' and successor_id."""
    mem_a = await engine.add_memory(MemoryCreate(content="Duplicate content alpha."))
    mem_b = await engine.add_memory(MemoryCreate(content="Duplicate content alpha."))

    await engine.store.delete(mem_a.id, reason="merged", successor_id="new-merged-id")
    await engine.store.delete(mem_b.id, reason="merged", successor_id="new-merged-id")

    archive_a = await engine.store.get_archive(mem_a.id)
    archive_b = await engine.store.get_archive(mem_b.id)
    assert len(archive_a) == 1
    assert archive_a[0].reason == "merged"
    assert archive_a[0].successor_id == "new-merged-id"
    assert len(archive_b) == 1
    assert archive_b[0].successor_id == "new-merged-id"


@pytest.mark.asyncio
async def test_archive_count(engine: MemoryEngine):
    """Archive count tracks deleted memories."""
    for i in range(3):
        mem = await engine.add_memory(MemoryCreate(content=f"Archive count test {i}"))
        await engine.store.delete(mem.id, reason="decay")

    count = await engine.store.archive_count("default")
    assert count == 3


@pytest.mark.asyncio
async def test_smart_add_memory_llm_fallback(engine: MemoryEngine):
    """When LLM analysis fails, smart_add falls back to defaults."""
    with patch("aimemo.core.llm.analyze_memory", new_callable=AsyncMock) as m:
        m.side_effect = Exception("API down")
        from aimemo.core.models import SmartMemoryCreate

        mem = await engine.smart_add_memory(
            SmartMemoryCreate(content="Some fallback content")
        )
        assert mem.memory_type.value == "episodic"
        assert mem.importance == 0.5
        assert mem.embedding is not None
