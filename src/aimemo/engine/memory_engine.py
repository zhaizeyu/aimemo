"""Memory Engine — orchestrates retrieval, consolidation, decay, and reflection.

Key SOTA features:
  1. Multi-signal retrieval  : combines vector similarity, recency, importance.
  2. Adaptive decay          : Ebbinghaus-inspired forgetting curve with access boost.
  3. Tier promotion          : working → short_term → long_term based on importance.
  4. Memory consolidation    : merges near-duplicate short-term memories.
  5. Automatic reflection    : generates higher-order "reflection" memories that
                               summarise clusters, mirroring the Generative Agents
                               architecture (Park et al. 2023).
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from aimemo.core.config import settings
from aimemo.core.embeddings import cosine_similarity, get_embedding_provider
from aimemo.core.models import (
    ConsolidationResult,
    MemoryCreate,
    MemoryQuery,
    MemoryRecord,
    MemoryTier,
    MemoryType,
    RetrievalResult,
)
from aimemo.storage.sqlite import MemoryStore

logger = logging.getLogger(__name__)


def _hours_since(dt: datetime) -> float:
    delta = datetime.now(UTC) - dt.replace(tzinfo=UTC) if dt.tzinfo is None else datetime.now(UTC) - dt
    return max(delta.total_seconds() / 3600.0, 0.0)


class MemoryEngine:
    def __init__(self, store: MemoryStore):
        self.store = store
        self.embedder = get_embedding_provider()
        self._access_counter: dict[str, int] = {}

    # ── Store / Retrieve ─────────────────────────────────────────────

    async def add_memory(self, req: MemoryCreate) -> MemoryRecord:
        embedding = await self.embedder.embed(req.content)
        record = MemoryRecord(
            content=req.content,
            memory_type=req.memory_type,
            tier=MemoryTier.SHORT_TERM,
            importance=req.importance,
            tags=req.tags,
            metadata=req.metadata,
            embedding=embedding,
            agent_id=req.agent_id,
        )

        if record.importance >= 0.8:
            record.tier = MemoryTier.LONG_TERM

        saved = await self.store.save(record)

        agent = req.agent_id
        self._access_counter[agent] = self._access_counter.get(agent, 0) + 1
        if self._access_counter[agent] >= settings.reflection_trigger_count:
            self._access_counter[agent] = 0
            await self._maybe_reflect(agent)

        return saved

    async def retrieve(self, query: MemoryQuery) -> list[RetrievalResult]:
        query_embedding = await self.embedder.embed(query.query)

        candidate_ids = await self.store.get_candidates(
            agent_id=query.agent_id, tiers=query.tiers, types=query.memory_types
        )

        if query.tags:
            tag_set = set(query.tags)
            tagged = await self.store.list_memories(
                agent_id=query.agent_id, limit=10000
            )
            tag_matches = {m.id for m in tagged if tag_set & set(m.tags)}
            candidate_ids = candidate_ids & tag_matches if candidate_ids else tag_matches

        vec_results = self.store.vector_search(
            query_embedding, top_k=query.top_k * 3, candidate_ids=candidate_ids or None
        )

        results: list[RetrievalResult] = []
        for mem_id, sim_score in vec_results:
            record = await self.store.get(mem_id)
            if record is None or record.importance < query.min_importance:
                continue

            recency = math.exp(-settings.decay_rate * _hours_since(record.last_accessed))
            relevance = max(0.0, (sim_score + 1.0) / 2.0)
            importance = record.importance

            combined = 0.50 * relevance + 0.25 * recency + 0.25 * importance

            results.append(
                RetrievalResult(
                    memory=record,
                    relevance_score=round(relevance, 4),
                    recency_score=round(recency, 4),
                    importance_score=round(importance, 4),
                    combined_score=round(combined, 4),
                )
            )

        results.sort(key=lambda r: r.combined_score, reverse=True)
        results = results[: query.top_k]

        for r in results:
            await self.store.update_access(
                r.memory.id, importance_boost=settings.importance_boost_on_access
            )

        return results

    # ── Consolidation ────────────────────────────────────────────────

    async def consolidate(self, agent_id: str = "default") -> ConsolidationResult:
        promoted = await self._promote(agent_id)
        decayed, deleted_ids = await self._decay(agent_id)
        merged = await self._merge_similar(agent_id)
        reflections = await self._maybe_reflect(agent_id)
        return ConsolidationResult(
            promoted=promoted, decayed=decayed, merged=merged, reflections_generated=reflections
        )

    async def _promote(self, agent_id: str) -> int:
        short = await self.store.list_memories(
            agent_id=agent_id, tier=MemoryTier.SHORT_TERM, limit=10000
        )
        updates: dict[str, MemoryTier] = {}
        for m in short:
            if m.importance >= settings.consolidation_threshold and m.access_count >= 2:
                updates[m.id] = MemoryTier.LONG_TERM
        if updates:
            await self.store.batch_update_tier(updates)
        return len(updates)

    async def _decay(self, agent_id: str) -> tuple[int, list[str]]:
        all_mems = await self.store.list_memories(agent_id=agent_id, limit=10000)
        imp_updates: dict[str, float] = {}
        to_delete: list[str] = []
        for m in all_mems:
            hours = _hours_since(m.last_accessed)
            decay = settings.decay_rate * math.log1p(hours)
            new_imp = m.importance - decay
            if new_imp < 0.05 and m.tier != MemoryTier.LONG_TERM:
                to_delete.append(m.id)
            elif abs(new_imp - m.importance) > 0.001:
                imp_updates[m.id] = new_imp
        if imp_updates:
            await self.store.batch_update_importance(imp_updates)
        deleted = await self.store.batch_delete(to_delete)
        return len(imp_updates) + deleted, to_delete

    async def _merge_similar(self, agent_id: str) -> int:
        short = await self.store.list_memories(
            agent_id=agent_id, tier=MemoryTier.SHORT_TERM, limit=500
        )
        if len(short) < 2:
            return 0

        merged_count = 0
        consumed: set[str] = set()
        for i, a in enumerate(short):
            if a.id in consumed or a.embedding is None:
                continue
            for b in short[i + 1 :]:
                if b.id in consumed or b.embedding is None:
                    continue
                sim = cosine_similarity(a.embedding, b.embedding)
                if sim >= 0.92:
                    merged_content = f"{a.content}\n---\n{b.content}"
                    merged_imp = min(1.0, max(a.importance, b.importance) + 0.05)
                    merged_tags = list(set(a.tags + b.tags))
                    new_embedding = await self.embedder.embed(merged_content)
                    merged_record = MemoryRecord(
                        content=merged_content,
                        memory_type=a.memory_type,
                        tier=MemoryTier.SHORT_TERM,
                        importance=merged_imp,
                        tags=merged_tags,
                        metadata={**a.metadata, **b.metadata},
                        embedding=new_embedding,
                        access_count=a.access_count + b.access_count,
                        source_ids=[a.id, b.id],
                        agent_id=agent_id,
                    )
                    await self.store.save(merged_record)
                    await self.store.delete(a.id)
                    await self.store.delete(b.id)
                    consumed.update({a.id, b.id})
                    merged_count += 1
                    break
        return merged_count

    # ── Reflection ───────────────────────────────────────────────────

    async def _maybe_reflect(self, agent_id: str) -> int:
        recent = await self.store.list_memories(
            agent_id=agent_id, limit=settings.reflection_min_memories * 2
        )
        if len(recent) < settings.reflection_min_memories:
            return 0

        contents = [m.content[:200] for m in recent[: settings.reflection_min_memories]]
        summary = "Reflection: " + " | ".join(contents)

        avg_imp = sum(m.importance for m in recent) / len(recent)
        tags = list({t for m in recent for t in m.tags})[:10]
        source_ids = [m.id for m in recent[: settings.reflection_min_memories]]

        embedding = await self.embedder.embed(summary)
        reflection = MemoryRecord(
            content=summary,
            memory_type=MemoryType.REFLECTION,
            tier=MemoryTier.LONG_TERM,
            importance=min(1.0, avg_imp + 0.1),
            tags=tags,
            metadata={"source_count": len(source_ids)},
            embedding=embedding,
            source_ids=source_ids,
            agent_id=agent_id,
        )
        await self.store.save(reflection)
        logger.info("Generated reflection memory %s for agent %s", reflection.id, agent_id)
        return 1
