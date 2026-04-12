"""FastAPI routes — the public HTTP interface of AIMemo."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from aimemo.core.models import (
    ConsolidationResult,
    MemoryCreate,
    MemoryQuery,
    MemoryRecord,
    MemoryStats,
    MemoryTier,
    MemoryType,
    MemoryUpdate,
    RetrievalResult,
)
from aimemo.engine.memory_engine import MemoryEngine

router = APIRouter()

_engine: MemoryEngine | None = None


def set_engine(engine: MemoryEngine) -> None:
    global _engine
    _engine = engine


def _get_engine() -> MemoryEngine:
    if _engine is None:
        raise RuntimeError("MemoryEngine not initialised")
    return _engine


# ── Memory CRUD ──────────────────────────────────────────────────────


@router.post("/memories", response_model=MemoryRecord, status_code=201, tags=["memories"])
async def create_memory(body: MemoryCreate):
    """Store a new memory."""
    return await _get_engine().add_memory(body)


@router.post("/memories/image", response_model=MemoryRecord, status_code=201, tags=["memories"])
async def create_image_memory(
    file: UploadFile = File(...),
    agent_id: str = Form("default"),
    tags: str = Form("image"),
    importance: float = Form(0.5),
    prompt: str = Form(""),
):
    """Upload an image and create a memory from its visual content.

    The image is sent to the vision model (gemini-3-flash-preview) which
    produces a text description that becomes the memory content.
    """
    image_data = await file.read()
    mime_type = file.content_type or "image/png"
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    return await _get_engine().add_image_memory(
        image_data=image_data,
        mime_type=mime_type,
        agent_id=agent_id,
        tags=tag_list,
        importance=importance,
        prompt=prompt,
    )


@router.get("/memories/{memory_id}", response_model=MemoryRecord, tags=["memories"])
async def get_memory(memory_id: str):
    """Retrieve a single memory by ID."""
    engine = _get_engine()
    rec = await engine.store.get(memory_id)
    if rec is None:
        raise HTTPException(404, "Memory not found")
    return rec


@router.patch("/memories/{memory_id}", response_model=MemoryRecord, tags=["memories"])
async def update_memory(memory_id: str, body: MemoryUpdate):
    """Partially update a memory."""
    engine = _get_engine()
    rec = await engine.store.get(memory_id)
    if rec is None:
        raise HTTPException(404, "Memory not found")
    if body.content is not None:
        rec.content = body.content
        rec.embedding = await engine.embedder.embed(body.content)
    if body.importance is not None:
        rec.importance = body.importance
    if body.tags is not None:
        rec.tags = body.tags
    if body.metadata is not None:
        rec.metadata = body.metadata
    return await engine.store.save(rec)


@router.delete("/memories/{memory_id}", status_code=204, tags=["memories"])
async def delete_memory(memory_id: str):
    """Delete a memory."""
    ok = await _get_engine().store.delete(memory_id)
    if not ok:
        raise HTTPException(404, "Memory not found")


@router.get("/memories", response_model=list[MemoryRecord], tags=["memories"])
async def list_memories(
    agent_id: str = "default",
    tier: MemoryTier | None = None,
    memory_type: MemoryType | None = None,
    min_importance: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List memories with optional filters."""
    return await _get_engine().store.list_memories(
        agent_id=agent_id,
        tier=tier,
        memory_type=memory_type,
        min_importance=min_importance,
        limit=limit,
        offset=offset,
    )


# ── Retrieval ────────────────────────────────────────────────────────


@router.post("/retrieve", response_model=list[RetrievalResult], tags=["retrieval"])
async def retrieve_memories(body: MemoryQuery):
    """Semantic + multi-signal retrieval."""
    return await _get_engine().retrieve(body)


# ── Operations ───────────────────────────────────────────────────────


@router.post("/consolidate", response_model=ConsolidationResult, tags=["operations"])
async def consolidate(agent_id: str = "default"):
    """Run consolidation: promote, decay, merge, reflect."""
    return await _get_engine().consolidate(agent_id)


# ── System ───────────────────────────────────────────────────────────


@router.get("/stats", response_model=MemoryStats, tags=["system"])
async def stats(agent_id: str = "default"):
    """Return memory statistics for an agent."""
    data = await _get_engine().store.count(agent_id)
    return MemoryStats(**data)


@router.get("/health", tags=["system"])
async def health():
    """Liveness probe."""
    return {"status": "ok"}
