"""Tests for the FastAPI HTTP interface."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from aimemo.api.routes import set_services
from aimemo.app import create_app
from aimemo.emotion.engine import EmotionEngine
from aimemo.emotion.store import EmotionStore
from aimemo.engine.memory_engine import MemoryEngine
from aimemo.runtime.assistant_runtime import AssistantRuntime
from aimemo.storage.sqlite import MemoryStore


@pytest.fixture
async def client(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "api_test.db"))
    await store.init()
    engine = MemoryEngine(store)
    emotion_store = EmotionStore(db_path=str(tmp_path / "api_test.db"))
    await emotion_store.init()
    emotion_engine = EmotionEngine(emotion_store)
    responder = AsyncMock()
    responder.generate = AsyncMock(return_value="收到，我在这陪你。")
    runtime = AssistantRuntime(engine, emotion_engine, responder)
    set_services(memory_engine=engine, emotion_engine=emotion_engine, runtime=runtime)

    app = create_app()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await emotion_store.close()
    await store.close()


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_create_and_get_memory(client: AsyncClient):
    resp = await client.post(
        "/api/v1/memories",
        json={"content": "Test memory via API", "importance": 0.7, "tags": ["test"]},
    )
    assert resp.status_code == 201
    data = resp.json()
    mem_id = data["id"]
    assert data["content"] == "Test memory via API"

    resp2 = await client.get(f"/api/v1/memories/{mem_id}")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == mem_id


@pytest.mark.asyncio
async def test_list_memories(client: AsyncClient):
    await client.post("/api/v1/memories", json={"content": "List item 1"})
    await client.post("/api/v1/memories", json={"content": "List item 2"})
    resp = await client.get("/api/v1/memories")
    assert resp.status_code == 200
    assert len(resp.json()) >= 2


@pytest.mark.asyncio
async def test_update_memory(client: AsyncClient):
    resp = await client.post("/api/v1/memories", json={"content": "Original"})
    mem_id = resp.json()["id"]

    resp2 = await client.patch(
        f"/api/v1/memories/{mem_id}", json={"content": "Updated", "importance": 0.9}
    )
    assert resp2.status_code == 200
    assert resp2.json()["content"] == "Updated"
    assert resp2.json()["importance"] == 0.9


@pytest.mark.asyncio
async def test_patch_semantic_reprocesses_fact_and_contradiction(client: AsyncClient):
    old = await client.post(
        "/api/v1/memories",
        json={"content": "Alice lives in Paris.", "memory_type": "semantic"},
    )
    old_id = old.json()["id"]
    target = await client.post(
        "/api/v1/memories",
        json={"content": "Alice likes coffee.", "memory_type": "semantic"},
    )
    target_id = target.json()["id"]

    with (
        patch("aimemo.core.llm.analyze_memory", new_callable=AsyncMock) as mock_analyze,
        patch("aimemo.core.llm.detect_contradiction", new_callable=AsyncMock) as mock_detect,
    ):
        mock_analyze.return_value = {
            "memory_type": "semantic",
            "importance": 0.8,
            "tags": ["fact"],
            "subject": "Alice",
            "predicate": "lives_in",
            "object_value": "Berlin",
            "update_type": "temporal_update",
        }
        mock_detect.return_value = {
            "has_contradiction": True,
            "contradicted_indices": [0],
            "merged_fact": "",
        }
        patched = await client.patch(
            f"/api/v1/memories/{target_id}",
            json={"content": "Alice lives in Berlin."},
        )

    assert patched.status_code == 200
    data = patched.json()
    assert data["metadata"]["subject"] == "Alice"
    assert data["metadata"]["predicate"] == "lives_in"
    assert data["metadata"]["object_value"] == "Berlin"
    assert data["metadata"]["update_type"] == "contradiction"
    assert data["metadata"]["superseded_count"] == 1

    old_fetch = await client.get(f"/api/v1/memories/{old_id}")
    assert old_fetch.status_code == 404
    archive = await client.get(f"/api/v1/archive/{old_id}")
    assert archive.status_code == 200
    archived = archive.json()
    assert archived[0]["reason"] == "superseded"
    assert archived[0]["successor_id"] == target_id


@pytest.mark.asyncio
async def test_delete_memory(client: AsyncClient):
    resp = await client.post("/api/v1/memories", json={"content": "To delete"})
    mem_id = resp.json()["id"]

    resp2 = await client.delete(f"/api/v1/memories/{mem_id}")
    assert resp2.status_code == 204

    resp3 = await client.get(f"/api/v1/memories/{mem_id}")
    assert resp3.status_code == 404


@pytest.mark.asyncio
async def test_retrieve(client: AsyncClient):
    await client.post("/api/v1/memories", json={"content": "Python programming language"})
    await client.post("/api/v1/memories", json={"content": "Sunny weather forecast"})

    resp = await client.post(
        "/api/v1/retrieve", json={"query": "programming", "top_k": 5}
    )
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) > 0
    assert results[0]["combined_score"] > 0


@pytest.mark.asyncio
async def test_consolidate(client: AsyncClient):
    for i in range(3):
        await client.post("/api/v1/memories", json={"content": f"Consolidation memory {i}"})

    resp = await client.post("/api/v1/consolidate")
    assert resp.status_code == 200
    data = resp.json()
    assert "promoted" in data
    assert "decayed" in data


@pytest.mark.asyncio
async def test_stats(client: AsyncClient):
    await client.post("/api/v1/memories", json={"content": "Stats test"})
    resp = await client.get("/api/v1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_memories"] >= 1


@pytest.mark.asyncio
async def test_not_found(client: AsyncClient):
    resp = await client.get("/api/v1/memories/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_image_memory_endpoint(client: AsyncClient, mock_vision_describe):
    """Test the image upload endpoint."""
    fake_image = b"\x89PNG\r\n\x1a\nfake_image_data"
    resp = await client.post(
        "/api/v1/memories/image",
        files={"file": ("test.png", fake_image, "image/png")},
        data={"agent_id": "default", "tags": "photo,test", "importance": "0.7"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["metadata"]["source"] == "image"
    assert "photo" in data["tags"]
    mock_vision_describe.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_creates_archive(client: AsyncClient):
    """Deleting via API archives the memory."""
    resp = await client.post("/api/v1/memories", json={"content": "To archive"})
    mem_id = resp.json()["id"]

    await client.delete(f"/api/v1/memories/{mem_id}")

    resp2 = await client.get(f"/api/v1/archive/{mem_id}")
    assert resp2.status_code == 200
    data = resp2.json()
    assert len(data) == 1
    assert data[0]["original_id"] == mem_id
    assert data[0]["reason"] == "manual"


@pytest.mark.asyncio
async def test_list_archive(client: AsyncClient):
    """List archive endpoint returns archived memories."""
    for i in range(2):
        resp = await client.post(
            "/api/v1/memories", json={"content": f"Archive list test {i}"}
        )
        await client.delete(f"/api/v1/memories/{resp.json()['id']}")

    resp2 = await client.get("/api/v1/archive")
    assert resp2.status_code == 200
    assert len(resp2.json()) >= 2


@pytest.mark.asyncio
async def test_stats_includes_archive_count(client: AsyncClient):
    """Stats endpoint includes archived_count."""
    resp = await client.post("/api/v1/memories", json={"content": "Stat archive test"})
    await client.delete(f"/api/v1/memories/{resp.json()['id']}")

    resp2 = await client.get("/api/v1/stats")
    assert resp2.status_code == 200
    assert resp2.json()["archived_count"] >= 1


@pytest.mark.asyncio
async def test_create_core_memory(client: AsyncClient):
    """Core memory endpoint creates pinned memories."""
    resp = await client.post(
        "/api/v1/memories/core",
        json={"content": "System: always use Chinese", "tags": ["system"]},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["tier"] == "core"
    assert data["importance"] == 1.0
    assert data["memory_type"] == "semantic"


@pytest.mark.asyncio
async def test_core_in_retrieval(client: AsyncClient):
    """Core memories always appear in retrieval results."""
    resp = await client.post(
        "/api/v1/memories/core",
        json={"content": "Critical system constraint"},
    )
    core_id = resp.json()["id"]

    resp2 = await client.post(
        "/api/v1/retrieve", json={"query": "completely unrelated", "top_k": 5}
    )
    assert resp2.status_code == 200
    ids = [r["memory"]["id"] for r in resp2.json()]
    assert core_id in ids


@pytest.mark.asyncio
async def test_smart_create_memory(client: AsyncClient, mock_analyze_memory):
    """Smart endpoint auto-generates memory_type, importance, tags via LLM."""
    resp = await client.post(
        "/api/v1/memories/smart",
        json={"content": "用户偏好Python编程"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["memory_type"] == "semantic"
    assert data["importance"] >= 0.8
    assert "python" in data["tags"]
    mock_analyze_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_emotion_endpoints(client: AsyncClient):
    processed = await client.post(
        "/api/v1/emotion/process",
        json={
            "agent_id": "default",
            "user_id": "u-emotion",
            "session_id": "s-emotion",
            "raw_text": "我今天很难过，想和你聊聊",
        },
    )
    assert processed.status_code == 200
    assert processed.json()["context"]["support_mode"] == "comforting"

    state = await client.get("/api/v1/emotion/state", params={"session_id": "s-emotion"})
    assert state.status_code == 200
    assert state.json()["support_mode"] == "comforting"

    relation = await client.get(
        "/api/v1/emotion/relationship",
        params={"user_id": "u-emotion"},
    )
    assert relation.status_code == 200
    assert relation.json()["interaction_count"] >= 1


@pytest.mark.asyncio
async def test_runtime_chat_endpoint(client: AsyncClient):
    resp = await client.post(
        "/api/v1/runtime/chat",
        json={
            "agent_id": "default",
            "user_id": "u-runtime",
            "session_id": "s-runtime",
            "user_text": "今天压力很大，想听你说句话",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"] == "收到，我在这陪你。"
    assert data["emotion_context"]["support_mode"] == "comforting"
