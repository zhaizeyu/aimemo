"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from aimemo.api.routes import router, set_engine
from aimemo.core.config import settings
from aimemo.engine.memory_engine import MemoryEngine
from aimemo.storage.sqlite import MemoryStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = MemoryStore()
    await store.init()
    engine = MemoryEngine(store)
    set_engine(engine)
    yield
    await store.close()


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
