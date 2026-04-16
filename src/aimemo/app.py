"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from aimemo.api.routes import router, set_services
from aimemo.core.config import settings
from aimemo.emotion.engine import EmotionEngine
from aimemo.emotion.store import EmotionStore
from aimemo.engine.memory_engine import MemoryEngine
from aimemo.runtime.assistant_runtime import AssistantRuntime, LLMResponseGenerator
from aimemo.storage.sqlite import MemoryStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    memory_store = MemoryStore()
    await memory_store.init()
    memory_engine = MemoryEngine(memory_store)

    emotion_store = EmotionStore(db_path=settings.db_path)
    await emotion_store.init()
    emotion_engine = EmotionEngine(emotion_store)

    runtime = AssistantRuntime(
        memory_engine=memory_engine,
        emotion_engine=emotion_engine,
        responder=LLMResponseGenerator(),
    )
    app.state.memory_store = memory_store
    app.state.memory_engine = memory_engine
    app.state.emotion_store = emotion_store
    app.state.emotion_engine = emotion_engine
    app.state.assistant_runtime = runtime
    set_services(memory_engine=memory_engine, emotion_engine=emotion_engine, runtime=runtime)
    yield
    await emotion_store.close()
    await memory_store.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="SOTA AI Agent Memory Module — multi-tier, associative, with consolidation and reflection.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
