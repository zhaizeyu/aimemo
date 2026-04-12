"""Tests for the memory engine core logic."""

from __future__ import annotations

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
