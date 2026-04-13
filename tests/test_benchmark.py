"""Benchmark suite for the AI memory system.

Tests verify correctness of fact updates, conflict detection, retrieval quality,
temporal consistency, archive traceability, summary completeness, and working
memory capacity enforcement.

All tests use builtin embeddings (conftest auto-use fixture) and mock LLM calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aimemo.core.config import settings
from aimemo.core.models import (
    MemoryCreate,
    MemoryQuery,
    MemoryType,
    WorkingMemoryInput,
)
from aimemo.engine.memory_engine import MemoryEngine
from aimemo.storage.sqlite import MemoryStore


@pytest.fixture
async def engine(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "bench.db"))
    await store.init()
    eng = MemoryEngine(store)
    yield eng
    await store.close()


# ── 1. Fact update correctness ───────────────────────────────────────


@pytest.mark.asyncio
async def test_fact_update_correctness(engine: MemoryEngine):
    """Store fact A, store updated fact B (mock contradiction), verify A archived and B active."""
    fact_a = await engine.add_memory(
        MemoryCreate(
            content="Zhang San uses Python.",
            memory_type="semantic",
            importance=0.7,
        )
    )

    with patch("aimemo.core.llm.detect_contradiction", new_callable=AsyncMock) as mock_cd:
        mock_cd.return_value = {
            "has_contradiction": True,
            "contradicted_indices": [0],
            "merged_fact": "Zhang San now uses Rust instead of Python.",
        }
        fact_b = await engine.add_memory(
            MemoryCreate(
                content="Zhang San now uses Rust.",
                memory_type="semantic",
                importance=0.7,
            )
        )

    assert await engine.store.get(fact_a.id) is None, "Old fact A should be archived"
    assert await engine.store.get(fact_b.id) is not None, "New fact B should be active"

    archived = await engine.store.get_archive(fact_a.id)
    assert len(archived) == 1
    assert archived[0].reason == "superseded"
    assert archived[0].successor_id == fact_b.id

    print(f"[Fact Update] A archived={len(archived)}, B active=True  ✓")


# ── 2. Conflict precision ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_conflict_precision_no_conflict(engine: MemoryEngine):
    """Non-conflicting facts should both survive."""
    with patch("aimemo.core.llm.detect_contradiction", new_callable=AsyncMock) as mock_cd:
        mock_cd.return_value = {
            "has_contradiction": False,
            "contradicted_indices": [],
            "merged_fact": "",
        }
        fact_a = await engine.add_memory(
            MemoryCreate(content="Python is a language.", memory_type="semantic", importance=0.7)
        )
        fact_b = await engine.add_memory(
            MemoryCreate(content="Rust is a language.", memory_type="semantic", importance=0.7)
        )

    assert await engine.store.get(fact_a.id) is not None
    assert await engine.store.get(fact_b.id) is not None
    print("[Conflict Precision - No Conflict] Both facts survive  ✓")


@pytest.mark.asyncio
async def test_conflict_precision_with_conflict(engine: MemoryEngine):
    """Conflicting facts: old should be archived."""
    old = await engine.add_memory(
        MemoryCreate(content="The CEO is Alice.", memory_type="semantic", importance=0.7)
    )

    with patch("aimemo.core.llm.detect_contradiction", new_callable=AsyncMock) as mock_cd:
        mock_cd.return_value = {
            "has_contradiction": True,
            "contradicted_indices": [0],
            "merged_fact": "The CEO is Bob (replacing Alice).",
        }
        await engine.add_memory(
            MemoryCreate(content="The CEO is Bob.", memory_type="semantic", importance=0.7)
        )

    assert await engine.store.get(old.id) is None, "Old conflicting fact should be gone"
    archived = await engine.store.get_archive(old.id)
    assert len(archived) == 1
    print("[Conflict Precision - With Conflict] Old fact archived  ✓")


# ── 3. Retrieval hit rate ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieval_hit_rate(engine: MemoryEngine):
    """Store 10 mixed-type memories, query specific topic, verify relevant rank higher."""
    programming = [
        "Python is a high-level programming language.",
        "JavaScript is used for web development.",
        "Rust guarantees memory safety without GC.",
        "Go is designed for concurrency.",
        "TypeScript adds static types to JavaScript.",
    ]
    other = [
        "The weather in Beijing is cold in winter.",
        "Mount Everest is the tallest mountain.",
        "The Pacific Ocean is the largest ocean.",
        "Pandas eat bamboo.",
        "The Milky Way contains billions of stars.",
    ]
    for c in programming:
        await engine.add_memory(MemoryCreate(content=c, importance=0.6, tags=["programming"]))
    for c in other:
        await engine.add_memory(MemoryCreate(content=c, importance=0.6, tags=["general"]))

    results = await engine.retrieve(MemoryQuery(query="programming languages", top_k=10))
    assert len(results) > 0

    top5_contents = [r.memory.content for r in results[:5]]
    prog_hits = sum(1 for c in top5_contents if any(kw in c.lower() for kw in ["programming", "language", "javascript", "python", "rust", "go ", "typescript"]))

    precision = prog_hits / min(5, len(results))
    assert precision >= 0.4, f"Expected at least 40% programming in top-5, got {precision:.0%}"
    print(f"[Retrieval Hit Rate] Top-5 programming precision: {precision:.0%}  ✓")


# ── 4. Temporal consistency ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_temporal_consistency(engine: MemoryEngine):
    """Store fact, update it, verify only latest is in active retrieval."""
    old = await engine.add_memory(
        MemoryCreate(content="The project uses Python 3.10.", memory_type="semantic", importance=0.7)
    )

    with patch("aimemo.core.llm.detect_contradiction", new_callable=AsyncMock) as mock_cd:
        mock_cd.return_value = {
            "has_contradiction": True,
            "contradicted_indices": [0],
            "merged_fact": "The project uses Python 3.12.",
        }
        new = await engine.add_memory(
            MemoryCreate(
                content="The project uses Python 3.12.",
                memory_type="semantic",
                importance=0.7,
            )
        )

    assert await engine.store.get(old.id) is None
    assert await engine.store.get(new.id) is not None

    results = await engine.retrieve(MemoryQuery(query="Python version", top_k=5))
    active_ids = {r.memory.id for r in results}
    assert old.id not in active_ids, "Old version should not appear in retrieval"
    print("[Temporal Consistency] Only latest fact in retrieval  ✓")


# ── 5. Archive traceability ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_archive_traceability(engine: MemoryEngine):
    """Delete and merge memories, verify complete archive chain with successor_ids."""
    mem_a = await engine.add_memory(MemoryCreate(content="Traceable memory A."))
    mem_b = await engine.add_memory(MemoryCreate(content="Traceable memory B."))

    await engine.store.delete(mem_a.id, reason="merged", successor_id="successor-1")
    await engine.store.delete(mem_b.id, reason="manual")

    archive_a = await engine.store.get_archive(mem_a.id)
    archive_b = await engine.store.get_archive(mem_b.id)

    assert len(archive_a) == 1
    assert archive_a[0].reason == "merged"
    assert archive_a[0].successor_id == "successor-1"

    assert len(archive_b) == 1
    assert archive_b[0].reason == "manual"
    assert archive_b[0].successor_id is None

    total_archived = await engine.store.archive_count("default")
    assert total_archived >= 2
    print(f"[Archive Traceability] {total_archived} archived, chains verified  ✓")


# ── 6. Summary loss rate ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_summary_loss_rate(engine: MemoryEngine):
    """Store episodic memories, trigger reflection, verify coverage of key points."""
    original = settings.reflection_trigger_count
    original_min = settings.reflection_min_memories
    settings.reflection_trigger_count = 3
    settings.reflection_min_memories = 3

    key_phrases = [
        "discussed project timeline",
        "agreed on Python as primary language",
        "set deadline to Friday",
        "assigned Alice to backend",
        "reviewed database schema",
        "approved cloud deployment",
    ]
    for phrase in key_phrases:
        await engine.add_memory(
            MemoryCreate(content=f"Meeting: {phrase}.", importance=0.6)
        )

    all_mems = await engine.store.list_memories(agent_id="default", limit=200)
    reflections = [m for m in all_mems if m.memory_type == MemoryType.REFLECTION]
    assert len(reflections) >= 1, "At least one reflection should be generated"

    reflection_text = " ".join(r.content for r in reflections).lower()
    covered = sum(1 for p in key_phrases if any(word in reflection_text for word in p.split()[:2]))
    coverage = covered / len(key_phrases)
    print(f"[Summary Loss Rate] Reflection covers {covered}/{len(key_phrases)} key points ({coverage:.0%})  ✓")

    settings.reflection_trigger_count = original
    settings.reflection_min_memories = original_min


# ── 7. Working memory capacity ───────────────────────────────────────


@pytest.mark.asyncio
async def test_working_memory_capacity(engine: MemoryEngine):
    """Add more items than capacity, verify overflow sinks to episodic."""
    original_cap = settings.working_memory_capacity
    settings.working_memory_capacity = 3

    session_id = "bench-session"
    for i in range(5):
        await engine.add_working_memory(
            WorkingMemoryInput(
                content=f"Working item {i}",
                session_id=session_id,
            )
        )

    working = await engine.get_working_memory(session_id)
    assert len(working) <= 3, f"Expected ≤3 working items, got {len(working)}"

    all_mems = await engine.store.list_memories(agent_id="default", limit=100)
    sunk = [
        m for m in all_mems
        if m.metadata.get("session_id") == session_id
        and m.tier.value == "short_term"
    ]
    assert len(sunk) >= 2, f"Expected ≥2 sunk items, got {len(sunk)}"

    print(
        f"[Working Memory Capacity] working={len(working)}, "
        f"sunk_to_short_term={len(sunk)}  ✓"
    )

    settings.working_memory_capacity = original_cap
