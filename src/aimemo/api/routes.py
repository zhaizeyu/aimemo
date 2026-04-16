"""FastAPI routes — the public HTTP interface of AIMemo."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from aimemo.core.models import (
    ArchivedMemory,
    ConsolidationResult,
    CoreMemoryCreate,
    FactRecord,
    MemoryCreate,
    MemoryQuery,
    MemoryRecord,
    MemoryStats,
    MemoryTier,
    MemoryType,
    MemoryUpdate,
    RetrievalResult,
    SmartMemoryCreate,
    WorkingMemoryFlush,
    WorkingMemoryInput,
)
from aimemo.emotion.engine import EmotionEngine
from aimemo.emotion.models import EmotionProcessInput, RuntimeChatInput, RuntimeChatOutput
from aimemo.engine.memory_engine import MemoryEngine
from aimemo.runtime.assistant_runtime import AssistantRuntime

router = APIRouter()

_engine: MemoryEngine | None = None
_emotion_engine: EmotionEngine | None = None
_runtime: AssistantRuntime | None = None


def set_engine(engine: MemoryEngine) -> None:
    global _engine
    _engine = engine


def set_services(
    *,
    memory_engine: MemoryEngine | None = None,
    emotion_engine: EmotionEngine | None = None,
    runtime: AssistantRuntime | None = None,
) -> None:
    global _engine, _emotion_engine, _runtime
    _engine = memory_engine
    _emotion_engine = emotion_engine
    _runtime = runtime


def _get_engine(request: Request) -> MemoryEngine:
    if _engine is not None:
        return _engine
    engine = getattr(request.app.state, "memory_engine", None)
    if engine is None:
        raise RuntimeError("MemoryEngine not initialised")
    return engine


def _get_emotion_engine(request: Request) -> EmotionEngine:
    if _emotion_engine is not None:
        return _emotion_engine
    engine = getattr(request.app.state, "emotion_engine", None)
    if engine is None:
        raise RuntimeError("EmotionEngine not initialised")
    return engine


def _get_runtime(request: Request) -> AssistantRuntime:
    if _runtime is not None:
        return _runtime
    runtime = getattr(request.app.state, "assistant_runtime", None)
    if runtime is None:
        raise RuntimeError("AssistantRuntime not initialised")
    return runtime


# ── Memory CRUD ──────────────────────────────────────────────────────


@router.post("/memories", response_model=MemoryRecord, status_code=201, tags=["memories"])
async def create_memory(body: MemoryCreate, request: Request):
    """Store a new memory."""
    return await _get_engine(request).add_memory(body)


@router.post("/memories/smart", response_model=MemoryRecord, status_code=201, tags=["memories"])
async def smart_create_memory(body: SmartMemoryCreate, request: Request):
    """Store a memory with LLM-analyzed metadata.

    Only `content` is required. The LLM (DeepSeek-V3.2) automatically
    determines memory_type, importance, and tags from the content.
    """
    return await _get_engine(request).smart_add_memory(body)


@router.post("/memories/core", response_model=MemoryRecord, status_code=201, tags=["memories"])
async def create_core_memory(body: CoreMemoryCreate, request: Request):
    """Store a core memory — pinned, always included in every retrieval.

    Core memories never decay, never get deleted by consolidation.
    Use for: system rules, user identity, critical constraints.
    """
    return await _get_engine(request).add_core_memory(body)


@router.post("/memories/image", response_model=MemoryRecord, status_code=201, tags=["memories"])
async def create_image_memory(
    request: Request,
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

    return await _get_engine(request).add_image_memory(
        image_data=image_data,
        mime_type=mime_type,
        agent_id=agent_id,
        tags=tag_list,
        importance=importance,
        prompt=prompt,
    )


@router.get("/memories/{memory_id}", response_model=MemoryRecord, tags=["memories"])
async def get_memory(memory_id: str, request: Request):
    """Retrieve a single memory by ID."""
    engine = _get_engine(request)
    rec = await engine.store.get(memory_id)
    if rec is None:
        raise HTTPException(404, "Memory not found")
    return rec


@router.patch("/memories/{memory_id}", response_model=MemoryRecord, tags=["memories"])
async def update_memory(memory_id: str, body: MemoryUpdate, request: Request):
    """Partially update a memory."""
    engine = _get_engine(request)
    updated = await engine.update_memory(memory_id, body)
    if updated is None:
        raise HTTPException(404, "Memory not found")
    return updated


@router.delete("/memories/{memory_id}", status_code=204, tags=["memories"])
async def delete_memory(memory_id: str, request: Request):
    """Delete a memory."""
    ok = await _get_engine(request).store.delete(memory_id)
    if not ok:
        raise HTTPException(404, "Memory not found")


@router.get("/memories", response_model=list[MemoryRecord], tags=["memories"])
async def list_memories(
    request: Request,
    agent_id: str = "default",
    tier: MemoryTier | None = None,
    memory_type: MemoryType | None = None,
    min_importance: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List memories with optional filters."""
    return await _get_engine(request).store.list_memories(
        agent_id=agent_id,
        tier=tier,
        memory_type=memory_type,
        min_importance=min_importance,
        limit=limit,
        offset=offset,
    )


# ── Working Memory ────────────────────────────────────────────────────


@router.post("/working", response_model=MemoryRecord, status_code=201, tags=["working"])
async def add_working_memory(body: WorkingMemoryInput, request: Request):
    """Add an item to working memory for a session."""
    return await _get_engine(request).add_working_memory(body)


@router.get("/working", response_model=list[MemoryRecord], tags=["working"])
async def get_working_memory(
    request: Request,
    session_id: str = Query(...),
    agent_id: str = "default",
):
    """Return ordered working memory for a session."""
    return await _get_engine(request).get_working_memory(session_id, agent_id)


@router.post("/working/flush", tags=["working"])
async def flush_working_memory(body: WorkingMemoryFlush, request: Request):
    """Convert all working memories for a session to episodic/short_term."""
    count = await _get_engine(request).flush_working_memory(body)
    return {"flushed": count}


# ── Facts ─────────────────────────────────────────────────────────────


@router.get("/facts", response_model=list[FactRecord], tags=["facts"])
async def list_facts(
    request: Request,
    agent_id: str = "default",
    subject: str | None = None,
    predicate: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return active semantic memories with their SPO triples projected."""
    engine = _get_engine(request)
    records = await engine.store.list_facts(
        agent_id=agent_id,
        subject=subject,
        predicate=predicate,
        limit=limit,
        offset=offset,
    )
    results: list[FactRecord] = []
    for r in records:
        results.append(
            FactRecord(
                id=r.id,
                content=r.content,
                importance=r.importance,
                tags=r.tags,
                subject=r.metadata.get("subject"),
                predicate=r.metadata.get("predicate"),
                object_value=r.metadata.get("object_value"),
                update_type=r.metadata.get("update_type"),
                agent_id=r.agent_id,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
        )
    return results


# ── Retrieval ────────────────────────────────────────────────────────


@router.post("/retrieve", response_model=list[RetrievalResult], tags=["retrieval"])
async def retrieve_memories(body: MemoryQuery, request: Request):
    """Semantic + multi-signal retrieval."""
    return await _get_engine(request).retrieve(body)


# ── Operations ───────────────────────────────────────────────────────


@router.post("/consolidate", response_model=ConsolidationResult, tags=["operations"])
async def consolidate(request: Request, agent_id: str = "default"):
    """Run consolidation: promote, decay, merge, reflect."""
    return await _get_engine(request).consolidate(agent_id)


# ── Archive ──────────────────────────────────────────────────────────


@router.get(
    "/archive/{original_id}", response_model=list[ArchivedMemory], tags=["archive"]
)
async def get_archive(original_id: str, request: Request):
    """Get all archived versions of a memory by its original ID."""
    return await _get_engine(request).store.get_archive(original_id)


@router.get("/archive", response_model=list[ArchivedMemory], tags=["archive"])
async def list_archive(
    request: Request,
    agent_id: str = "default",
    reason: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List archived memories with optional reason filter.

    Reasons: "merged" (consolidation), "decay" (importance dropped), "manual" (user deleted).
    """
    return await _get_engine(request).store.list_archive(
        agent_id=agent_id, reason=reason, limit=limit, offset=offset
    )


# ── System ───────────────────────────────────────────────────────────


@router.get("/stats", response_model=MemoryStats, tags=["system"])
async def stats(request: Request, agent_id: str = "default"):
    """Return memory statistics for an agent."""
    data = await _get_engine(request).store.count(agent_id)
    data["archived_count"] = await _get_engine(request).store.archive_count(agent_id)
    return MemoryStats(**data)


@router.get("/health", tags=["system"])
async def health():
    """Liveness probe."""
    return {"status": "ok"}


# ── Emotion / Runtime ────────────────────────────────────────────────


@router.get("/emotion/state", tags=["emotion"])
async def get_emotion_state(
    request: Request,
    agent_id: str = "default",
    session_id: str | None = None,
):
    return await _get_emotion_engine(request).get_state(agent_id=agent_id, session_id=session_id)


@router.get("/emotion/relationship", tags=["emotion"])
async def get_relationship_state(
    request: Request,
    user_id: str = Query(...),
    agent_id: str = "default",
):
    return await _get_emotion_engine(request).get_relationship(agent_id=agent_id, user_id=user_id)


@router.post("/emotion/process", tags=["emotion"])
async def process_emotion(body: EmotionProcessInput, request: Request):
    return await _get_emotion_engine(request).process_user_event(
        agent_id=body.agent_id,
        user_id=body.user_id,
        session_id=body.session_id,
        raw_text=body.raw_text,
    )


@router.post("/runtime/chat", response_model=RuntimeChatOutput, tags=["runtime"])
async def runtime_chat(body: RuntimeChatInput, request: Request):
    result = await _get_runtime(request).handle_turn(
        agent_id=body.agent_id,
        user_id=body.user_id,
        session_id=body.session_id,
        user_text=body.user_text,
    )
    return RuntimeChatOutput(
        reply=result["reply"],
        emotion_context=result["emotion_context"],
        materialized_memory_ids=result["materialized_memory_ids"],
    )
