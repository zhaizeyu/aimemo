"""Runtime orchestrator joining AIMemo and Emotion Brain."""

from __future__ import annotations

from typing import Protocol

from aimemo.core.llm import chat_completion
from aimemo.core.models import MemoryCreate, MemoryQuery, MemoryType, WorkingMemoryInput
from aimemo.emotion.engine import EmotionEngine


class ResponseGenerator(Protocol):
    async def generate(self, context: dict) -> str:
        """Generate the final assistant response from a unified context."""


class LLMResponseGenerator:
    """Default responder backed by configured chat model."""

    async def generate(self, context: dict) -> str:
        prompt = (
            f"用户输入：{context['user_text']}\n\n"
            f"情绪引导：{context['emotion_context'].get('generation_notes', '')}\n"
            f"策略：support_mode={context['response_policy']['support_mode']}, "
            f"style_mode={context['response_policy']['style_mode']}\n"
            f"相关记忆：{context['retrieved_memories']}\n\n"
            "请输出一段自然、简洁、以共情为先的中文回复。"
        )
        return await chat_completion(prompt, system="你是一个温暖、可靠的陪伴助手。", temperature=0.5)


class AssistantRuntime:
    """High-level turn pipeline for companion assistant scenarios."""

    def __init__(self, memory_engine, emotion_engine: EmotionEngine, responder: ResponseGenerator):
        self.memory_engine = memory_engine
        self.emotion_engine = emotion_engine
        self.responder = responder

    async def handle_turn(
        self,
        *,
        agent_id: str,
        user_id: str,
        session_id: str,
        user_text: str,
    ) -> dict:
        # 1) memory retrieval (working + relevant + core via AIMemo retrieve)
        memories = await self.memory_engine.retrieve(
            MemoryQuery(query=user_text, agent_id=agent_id, top_k=8)
        )

        # 2) emotion update
        update = await self.emotion_engine.process_user_event(
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            raw_text=user_text,
        )

        # 3) build unified context
        context = {
            "user_text": user_text,
            "retrieved_memories": [r.memory.content for r in memories],
            "emotion_context": update.context.model_dump(),
            "response_policy": {
                "support_mode": update.context.support_mode.value,
                "style_mode": update.context.style_mode.value,
            },
        }

        # 4) model response
        reply = await self.responder.generate(context)

        # 5) post-turn writes
        await self.memory_engine.add_working_memory(
            WorkingMemoryInput(content=f"User: {user_text}", session_id=session_id, agent_id=agent_id)
        )
        await self.memory_engine.add_working_memory(
            WorkingMemoryInput(content=f"Assistant: {reply}", session_id=session_id, agent_id=agent_id)
        )

        # 6) selective long-term materialization
        memory_ids = await self.emotion_engine.materialize_patterns_to_memory(
            agent_id=agent_id,
            user_id=user_id,
            memory_engine=self.memory_engine,
        )

        # 7) minimal episodic trace
        await self.memory_engine.add_memory(
            MemoryCreate(
                content=f"与用户 {user_id} 一轮对话完成，support_mode={update.context.support_mode.value}",
                memory_type=MemoryType.EPISODIC,
                importance=0.45,
                tags=["interaction", "runtime"],
                metadata={"session_id": session_id, "user_id": user_id},
                agent_id=agent_id,
            )
        )

        return {
            "reply": reply,
            "emotion_context": update.context,
            "materialized_memory_ids": memory_ids,
        }
