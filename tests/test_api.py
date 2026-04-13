"""Tests for the FastAPI HTTP interface."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from aimemo.api.routes import set_engine
from aimemo.app import create_app
from aimemo.engine.memory_engine import MemoryEngine
from aimemo.storage.sqlite import MemoryStore


@pytest.fixture
async def client(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "api_test.db"))
    await store.init()
    engine = MemoryEngine(store)
    set_engine(engine)

    app = create_app()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

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
