"""Async SQLite storage backend with in-process vector index.

Schema keeps embeddings as JSON-encoded float arrays alongside rich metadata.
A companion in-memory numpy matrix provides fast approximate cosine search
without requiring a separate vector-DB dependency.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import numpy as np

from aimemo.core.config import settings
from aimemo.core.models import MemoryRecord, MemoryTier, MemoryType

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    content       TEXT NOT NULL,
    memory_type   TEXT NOT NULL,
    tier          TEXT NOT NULL,
    importance    REAL NOT NULL DEFAULT 0.5,
    tags          TEXT NOT NULL DEFAULT '[]',
    metadata      TEXT NOT NULL DEFAULT '{}',
    embedding     TEXT,
    access_count  INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    source_ids    TEXT NOT NULL DEFAULT '[]',
    agent_id      TEXT NOT NULL DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS idx_memories_agent  ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_tier   ON memories(tier);
CREATE INDEX IF NOT EXISTS idx_memories_type   ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _record_to_row(rec: MemoryRecord) -> dict:
    return {
        "id": rec.id,
        "content": rec.content,
        "memory_type": rec.memory_type.value,
        "tier": rec.tier.value,
        "importance": rec.importance,
        "tags": json.dumps(rec.tags),
        "metadata": json.dumps(rec.metadata),
        "embedding": json.dumps(rec.embedding) if rec.embedding else None,
        "access_count": rec.access_count,
        "last_accessed": rec.last_accessed.isoformat(),
        "created_at": rec.created_at.isoformat(),
        "updated_at": rec.updated_at.isoformat(),
        "source_ids": json.dumps(rec.source_ids),
        "agent_id": rec.agent_id,
    }


def _row_to_record(row: aiosqlite.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        content=row["content"],
        memory_type=MemoryType(row["memory_type"]),
        tier=MemoryTier(row["tier"]),
        importance=row["importance"],
        tags=json.loads(row["tags"]),
        metadata=json.loads(row["metadata"]),
        embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        access_count=row["access_count"],
        last_accessed=datetime.fromisoformat(row["last_accessed"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        source_ids=json.loads(row["source_ids"]),
        agent_id=row["agent_id"],
    )


class MemoryStore:
    """Async SQLite store with a co-located numpy vector index."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.db_path
        self._db: aiosqlite.Connection | None = None
        self._vec_ids: list[str] = []
        self._vec_matrix: np.ndarray | None = None

    # ── Lifecycle ────────────────────────────────────────────────────

    async def init(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_DDL)
        await self._db.commit()
        await self._rebuild_vector_index()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── Vector index ─────────────────────────────────────────────────

    async def _rebuild_vector_index(self) -> None:
        assert self._db
        cursor = await self._db.execute(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
        )
        rows = await cursor.fetchall()
        ids: list[str] = []
        vecs: list[list[float]] = []
        for r in rows:
            ids.append(r["id"])
            vecs.append(json.loads(r["embedding"]))
        self._vec_ids = ids
        if vecs:
            self._vec_matrix = np.array(vecs, dtype=np.float32)
        else:
            self._vec_matrix = None

    def _upsert_vec(self, mem_id: str, embedding: list[float]) -> None:
        vec = np.array(embedding, dtype=np.float32)
        if mem_id in self._vec_ids:
            idx = self._vec_ids.index(mem_id)
            assert self._vec_matrix is not None
            self._vec_matrix[idx] = vec
        else:
            self._vec_ids.append(mem_id)
            if self._vec_matrix is None:
                self._vec_matrix = vec.reshape(1, -1)
            else:
                self._vec_matrix = np.vstack([self._vec_matrix, vec])

    def _remove_vec(self, mem_id: str) -> None:
        if mem_id in self._vec_ids:
            idx = self._vec_ids.index(mem_id)
            self._vec_ids.pop(idx)
            if self._vec_matrix is not None:
                self._vec_matrix = np.delete(self._vec_matrix, idx, axis=0)
                if self._vec_matrix.size == 0:
                    self._vec_matrix = None

    def vector_search(
        self, query_vec: list[float], top_k: int = 10, candidate_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        if self._vec_matrix is None or not self._vec_ids:
            return []

        qv = np.array(query_vec, dtype=np.float32)
        qn = np.linalg.norm(qv)
        if qn == 0:
            return []
        qv = qv / qn

        norms = np.linalg.norm(self._vec_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = self._vec_matrix / norms
        sims = normed @ qv

        if candidate_ids is not None:
            mask = np.array([mid in candidate_ids for mid in self._vec_ids], dtype=bool)
            sims = np.where(mask, sims, -2.0)

        k = min(top_k, len(self._vec_ids))
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score > -1.5:
                results.append((self._vec_ids[idx], score))
        return results

    # ── CRUD ─────────────────────────────────────────────────────────

    async def save(self, record: MemoryRecord) -> MemoryRecord:
        assert self._db
        row = _record_to_row(record)
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row)
        updates = ", ".join(f"{k}=excluded.{k}" for k in row if k != "id")
        sql = f"INSERT INTO memories ({cols}) VALUES ({placeholders}) ON CONFLICT(id) DO UPDATE SET {updates}"
        await self._db.execute(sql, row)
        await self._db.commit()
        if record.embedding:
            self._upsert_vec(record.id, record.embedding)
        return record

    async def get(self, memory_id: str) -> MemoryRecord | None:
        assert self._db
        cursor = await self._db.execute("SELECT * FROM memories WHERE id = :id", {"id": memory_id})
        row = await cursor.fetchone()
        return _row_to_record(row) if row else None

    async def delete(self, memory_id: str) -> bool:
        assert self._db
        cursor = await self._db.execute(
            "DELETE FROM memories WHERE id = :id", {"id": memory_id}
        )
        await self._db.commit()
        self._remove_vec(memory_id)
        return cursor.rowcount > 0

    async def list_memories(
        self,
        agent_id: str = "default",
        tier: MemoryTier | None = None,
        memory_type: MemoryType | None = None,
        min_importance: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        assert self._db
        conditions = ["agent_id = :agent_id", "importance >= :min_importance"]
        params: dict = {"agent_id": agent_id, "min_importance": min_importance}
        if tier:
            conditions.append("tier = :tier")
            params["tier"] = tier.value
        if memory_type:
            conditions.append("memory_type = :memory_type")
            params["memory_type"] = memory_type.value
        where = " AND ".join(conditions)
        sql = f"SELECT * FROM memories WHERE {where} ORDER BY updated_at DESC LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_record(r) for r in rows]

    async def update_access(self, memory_id: str, importance_boost: float = 0.0) -> None:
        assert self._db
        now = _now_iso()
        await self._db.execute(
            """UPDATE memories
               SET access_count = access_count + 1,
                   last_accessed = :now,
                   importance = MIN(1.0, importance + :boost),
                   updated_at = :now
               WHERE id = :id""",
            {"id": memory_id, "now": now, "boost": importance_boost},
        )
        await self._db.commit()

    async def batch_update_importance(self, updates: dict[str, float]) -> None:
        assert self._db
        now = _now_iso()
        for mid, new_imp in updates.items():
            await self._db.execute(
                "UPDATE memories SET importance = :imp, updated_at = :now WHERE id = :id",
                {"id": mid, "imp": max(0.0, min(1.0, new_imp)), "now": now},
            )
        await self._db.commit()

    async def batch_update_tier(self, updates: dict[str, MemoryTier]) -> None:
        assert self._db
        now = _now_iso()
        for mid, new_tier in updates.items():
            await self._db.execute(
                "UPDATE memories SET tier = :tier, updated_at = :now WHERE id = :id",
                {"id": mid, "tier": new_tier.value, "now": now},
            )
        await self._db.commit()

    async def batch_delete(self, ids: list[str]) -> int:
        assert self._db
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        cursor = await self._db.execute(
            f"SELECT id FROM memories WHERE id IN ({placeholders})", ids
        )
        existing = [r["id"] for r in await cursor.fetchall()]
        if existing:
            await self._db.execute(
                f"DELETE FROM memories WHERE id IN ({','.join('?' for _ in existing)})",
                existing,
            )
            await self._db.commit()
            for mid in existing:
                self._remove_vec(mid)
        return len(existing)

    async def count(self, agent_id: str = "default") -> dict:
        assert self._db
        total_cur = await self._db.execute(
            "SELECT COUNT(*) as c FROM memories WHERE agent_id = :a", {"a": agent_id}
        )
        total = (await total_cur.fetchone())["c"]

        tier_cur = await self._db.execute(
            "SELECT tier, COUNT(*) as c FROM memories WHERE agent_id = :a GROUP BY tier",
            {"a": agent_id},
        )
        by_tier = {r["tier"]: r["c"] for r in await tier_cur.fetchall()}

        type_cur = await self._db.execute(
            "SELECT memory_type, COUNT(*) as c FROM memories WHERE agent_id = :a GROUP BY memory_type",
            {"a": agent_id},
        )
        by_type = {r["memory_type"]: r["c"] for r in await type_cur.fetchall()}

        avg_cur = await self._db.execute(
            "SELECT AVG(importance) as a FROM memories WHERE agent_id = :a", {"a": agent_id}
        )
        avg_imp = (await avg_cur.fetchone())["a"] or 0.0

        return {
            "total_memories": total,
            "by_tier": by_tier,
            "by_type": by_type,
            "avg_importance": round(avg_imp, 4),
            "agent_id": agent_id,
        }

    async def get_candidates(
        self, agent_id: str, tiers: list[MemoryTier] | None, types: list[MemoryType] | None
    ) -> set[str]:
        assert self._db
        conditions = ["agent_id = :agent_id"]
        params: dict = {"agent_id": agent_id}
        if tiers:
            tier_vals = [t.value for t in tiers]
            placeholders = ",".join(f":t{i}" for i in range(len(tier_vals)))
            conditions.append(f"tier IN ({placeholders})")
            for i, v in enumerate(tier_vals):
                params[f"t{i}"] = v
        if types:
            type_vals = [t.value for t in types]
            placeholders = ",".join(f":mt{i}" for i in range(len(type_vals)))
            conditions.append(f"memory_type IN ({placeholders})")
            for i, v in enumerate(type_vals):
                params[f"mt{i}"] = v
        where = " AND ".join(conditions)
        cursor = await self._db.execute(f"SELECT id FROM memories WHERE {where}", params)
        return {r["id"] for r in await cursor.fetchall()}
